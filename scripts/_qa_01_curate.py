# -*- coding: utf-8 -*-
"""
_qa_01_curate.py — `01_curate_cmrt.py` 产出物的门禁校验
=========================================================
跑法: PYTHONIOENCODING=utf-8 python scripts/_qa_01_curate.py
全程只读 data/curated/，失败项以 [FAIL] 标出并计入退出码。

为什么需要它：主表是 Paper 1 全部结论的地基。审计已经暴露过「靶变量定义错了却看不出来」
的教训，所以每一条化学/统计断言都要有一个**独立的**检查，而不是靠人眼看。
"""
from __future__ import annotations

import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
CUR = ROOT / "data" / "curated"
SPLITS = CUR / "splits"

FAILS: list[str] = []
N_PASS = 0


def check(cond: bool, label: str, detail: str = "") -> bool:
    global N_PASS
    if cond:
        N_PASS += 1
        print(f"  [ok]   {label}" + (f"  ({detail})" if detail else ""))
    else:
        FAILS.append(label)
        print(f"  [FAIL] {label}  {detail}")
    return cond


def invert_stereo(m):
    m = Chem.RWMol(m)
    for a in m.GetAtoms():
        t = a.GetChiralTag()
        if t == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
            a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
        elif t == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
            a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    return m.GetMol()


def canon(s: str) -> str | None:
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m else None


def inchi_layers(s: str):
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None, None
    i = Chem.MolToInchi(m)
    t = re.search(r"/t([^/]*)", i)
    mm = re.search(r"/m(\d)", i)
    return (t.group(1) if t else ""), (mm.group(1) if mm else "")


def main() -> int:
    print("=" * 74)
    print("_qa_01_curate — 主表门禁校验")
    print("=" * 74)
    p = pd.read_csv(CUR / "cmrt_pairs.csv")
    m = pd.read_csv(CUR / "cmrt_measurements.csv")
    c = pd.read_csv(CUR / "columns.csv")

    print("\n── 1. 表结构与内部一致性 ──")
    check(len(p) > 9000, "主表规模 > 9000 对", f"{len(p)} 对 / {len(m)} 条记录")
    check(p["pair_id"].is_unique, "pair_id 唯一")
    nn = ["pair_id", "conn", "smi_S", "smi_R", "Column", "k_S", "k_R",
          "alpha_true", "y", "EMO", "doi"]
    check(p[nn].notna().all().all(), "关键列无缺失",
          f"缺失={ {k: int(p[k].isna().sum()) for k in nn if p[k].isna().any()} }")
    check(np.allclose(p["y"], np.log(p["alpha_true"]), atol=1e-9), "y == ln(alpha_true)")
    check(np.allclose(p["alpha_true"], p["k_S"] / p["k_R"], atol=1e-9), "alpha_true == k_S/k_R")
    check((p["EMO"] == (p["y"] > 0).astype(int)).all(), "EMO == 1{y>0}")
    check(bool((p["k_S"] > 0).all() and (p["k_R"] > 0).all()), "k_S / k_R 全为正")
    check(bool(p["Column"].isin(c["Column"]).all()), "每对的柱都在 columns.csv 里")
    check(c["Column"].is_unique, "columns.csv 柱名唯一")
    # 辅助量必须与主靶同向，否则符号无法对照（曾是实际存在的 bug）
    same_sign = float((np.sign(p["dRT_over_t0"]) == np.sign(p["y"])).mean())
    check(same_sign > 0.999, "dRT_over_t0 与 y 同号（正 = S 后出峰）", f"{same_sign*100:.2f}%")
    # 并列（α≈1）必须用 flag 判定：其中有 2 行的 alpha_true 是 1+2.2e-16，**精确相等判不出来**
    ns = p["flag_no_separation"].astype(bool)
    # rt_S/rt_R 是聚合均值，同值时可能差 1e-15 的浮点噪声 → 必须带容差比较
    check(bool(np.allclose(p.loc[ns, "rt_S"], p.loc[ns, "rt_R"], rtol=0, atol=1e-6)),
          "flag_no_separation ⇒ rt_S ≈ rt_R", f"{int(ns.sum())} 对")
    check(bool((p.loc[~ns, "rt_S"] > p.loc[~ns, "rt_R"]).eq(p.loc[~ns, "y"] > 0).all()),
          "rt_S > rt_R ⟺ y > 0（排除并列）")
    check(float(p.loc[ns, "y"].abs().max()) < 1e-12, "并列行 |y| < 1e-12")
    check(float(p.loc[~ns, "y"].abs().min()) > 1e-6, "非并列行 |y| 不退化到 0")
    check(bool((p["alpha_true"] > 0).all() and np.isfinite(p["y"]).all()), "alpha_true/y 有限且为正")
    check(bool(p["t0"].gt(0).all()), "t0 全为正")
    check(bool(((p["ipr"] >= 0) & (p["ipr"] <= 1)).all()), "i-PrOH 体积分数落在 [0,1]")

    print("\n── 2. 对映体关系（几何/图论断言）──")
    sub = p.sample(min(1200, len(p)), random_state=0)
    ok_mir = sum(canon(Chem.MolToSmiles(invert_stereo(Chem.MolFromSmiles(a)))) == canon(b)
                 for a, b in zip(sub["smi_S"], sub["smi_R"]))
    check(ok_mir == len(sub), "把 smi_S 做镜像 == smi_R（抽样）", f"{ok_mir}/{len(sub)}")
    ok_conn = sum(Chem.MolToInchiKey(Chem.MolFromSmiles(a))[:14]
                  == Chem.MolToInchiKey(Chem.MolFromSmiles(b))[:14]
                  for a, b in zip(sub["smi_S"], sub["smi_R"]))
    check(ok_conn == len(sub), "smi_S 与 smi_R 连接层(InChIKey 前14位)相同", f"{ok_conn}/{len(sub)}")
    same = sum(canon(a) == canon(b) for a, b in zip(sub["smi_S"], sub["smi_R"]))
    check(same == 0, "smi_S != smi_R（不是同一个分子）", f"相同 {same} 对")

    print("\n── 3. CIP 定向（靶变量的符号语义）──")
    lens = p["cip_S"].astype(str).str.len()
    mono = p[lens == 1]
    check(len(mono) > 0.9 * len(p), "绝大多数对是单手性中心（定向 = 字面 CIP S/R）",
          f"{len(mono)}/{len(p)} = {len(mono)/len(p)*100:.1f}%")
    check(bool((mono["cip_S"] == "S").all()), "单中心对的 cip_S 全部为 'S'")
    check(bool((mono["cip_R"] == "R").all()), "单中心对的 cip_R 全部为 'R'")
    # 逐位互补：R<->S 全翻转
    comp = p["cip_S"].astype(str).str.translate(str.maketrans("RS", "SR"))
    bad_comp = int((comp != p["cip_R"].astype(str)).sum())
    check(bad_comp <= 1, "cip_S 逐位取反 == cip_R（对映体签名应互补；允许 ≤1 例环对称并列）",
          f"不互补 {bad_comp} 对")
    degen = int((p["cip_S"].astype(str).str.len() == 0).sum())
    check(degen == 0, "没有「无手性中心却被判为对映体」的退化行（定向会变成任意）", f"{degen} 行")

    # ⚠️ 反面教材（已实测否定）：**不要**用 InChI 的 /m 层当 R/S oracle。
    #    /m 是 InChI 按各分子规范编号做的归一化标记，与 R/S 无固定对应
    #    （实测同一批分子里 73.7% 是 m0=S、26.3% 是 m1=S）。用它做判据会得到假警报。
    # 有效 oracle 1：RDKit 的**两套独立 CIP 实现**（新版 rdCIPLabeler vs legacy）逐分子对比。
    uniq = pd.concat([p["smi_S"], p["smi_R"]]).unique()
    samp = pd.Series(uniq).sample(min(4000, len(uniq)), random_state=2)
    n_cmp = n_diff = 0
    diff_cip: Counter = Counter()

    def sig_of(s: str, legacy: bool) -> str:
        m = Chem.MolFromSmiles(s)
        if m is None:
            return ""
        r = list(Chem.CanonicalRankAtoms(m, breakTies=True))
        c = Chem.FindMolChiralCenters(m, includeUnassigned=True, useLegacyImplementation=legacy)
        return "".join(x for _, x in sorted((r[i], x) for i, x in c if x in ("R", "S")))

    for s in samp:
        a, b = sig_of(s, False), sig_of(s, True)
        if not a:
            continue
        n_cmp += 1
        if a != b:
            n_diff += 1
            sub = p[(p.smi_S == s) | (p.smi_R == s)]
            diff_cip["含硫(亚砜/砜)" if re.search(r"\[S[@H]|S\(=O\)|\[S\]\(", s) else "其他"] += len(sub)
    rate = 1 - n_diff / max(n_cmp, 1)
    check(rate > 0.95, "两套 CIP 实现一致率 > 95%（分歧应集中在 legacy 已知有 bug 的亚砜）",
          f"{rate*100:.2f}%（{n_diff}/{n_cmp} 分子不同，涉及主表 {sum(diff_cip.values())} 行）")
    print(f"        分歧构成: {dict(diff_cip)}  ← 新版 rdCIPLabeler 是本书采用的一方")

    # 有效 oracle 2：与外部权威（PubChem / IUPAC 参考 InChI）对账的标准品
    refs = [
        ("N[C@@H](C)C(=O)O", "S"), ("N[C@H](C)C(=O)O", "R"),                       # 丙氨酸
        ("C[C@@H](C1=CC=C(C=C1)CC(C)C)C(=O)O", "S"),                               # (S)-布洛芬
        ("C[C@H](C1=CC=C(C=C1)CC(C)C)C(=O)O", "R"),                                # (R)-布洛芬
        ("C[C@@H](C(=O)O)O", "S"), ("C[C@H](C(=O)O)O", "R"),                       # 乳酸（InChI /m 标定）
        ("OC(=O)[C@@H](N)Cc1ccccc1", "S"),                                          # L-苯丙氨酸
        ("N1CCC[C@H]1C(O)=O", "S"),                                                 # L-脯氨酸
    ]
    bad_ref = [(s, e, sig_of(s, False)) for s, e in refs if sig_of(s, False) != e]
    check(not bad_ref, "与外部权威对账的 8 个标准品 CIP 全部正确",
          f"错 {bad_ref}" if bad_ref else "8/8")

    # 有效 oracle 3：CIP 不互补的行必须已被打标（符号语义不可信）
    diag_cip = int((p["cip_S"].astype(str).str.len()
                    != p["cip_R"].astype(str).str.len()).sum())
    check(diag_cip == 0, "cip_S / cip_R 长度一致")
    flag = int(p["flag_cip_ambiguous"].sum()) if "flag_cip_ambiguous" in p else -1
    check(flag >= bad_comp, "非互补行都已打 flag_cip_ambiguous",
          f"flag={flag} vs 非互补={bad_comp}")

    print("\n── 4. 定向是否无系统偏置（EMO 应 ≈ 50%）──")
    frac = float(p["EMO"].mean())
    check(0.45 < frac < 0.55, "EMO 均值落在 45–55%（无系统偏置）", f"{frac*100:.1f}%")
    check(abs(float(p["y"].mean())) < 0.05, "y 的均值≈0", f"{p['y'].mean():+.4f}")

    # ★ 第八场补：**「conn 不是完备分子身份」的回归断言（已知偏差，非零容忍）**
    #   `conn` = InChIKey 前 14 位 = **连接层**，不含 C=C 的 E/Z。
    #   实测：`HEAMCWRVNWTZEW` 下有 2 行，只差双键构型（P05981 vs P07958），
    #   却各自与自己的镜像构成一对「对映体」⇒ 它们是**两个不同的非对映体**共用同一个 conn。
    #   影响：所有 `set_index("conn")` / `drop_duplicates("conn")` 类合并
    #   （06/06c/14/15/16/13/_baseline 等）在这 2 行上只能保留一支描述符。
    #   📌 处置（Lead 裁定，2026-09-17）：**保留主表、不改 01 的身份键**，
    #      在论文数据卡里显式声明该已知偏差 **2/10525 = 0.02%**。
    #   本断言是**回归守护**：这个「已知为 2 行 / 1 个 conn」的图景一变就 FAIL，
    #   提醒重估影响面（而不是静默接受新的身份碰撞）。
    KNOWN_CONN_COLLISION = {"HEAMCWRVNWTZEW"}
    _g = p.groupby("conn")["murcko"].nunique()
    _bad = _g[_g > 1]
    _rows = int(p["conn"].isin(_bad.index).sum())
    check(set(_bad.index) == KNOWN_CONN_COLLISION and _rows == 2,
          "conn→murcko 非唯一仅限已知偏差（1 个 conn / 2 行 = 0.02%，论文需声明）",
          f"异常 conn={sorted(_bad.index)} 涉及 {_rows} 行；期望 {sorted(KNOWN_CONN_COLLISION)} / 2 行")

    print("\n── 5. 划分完整性（防泄漏）──")


    def spans(df, key, fold_col="fold") -> int:
        """有多少个 group 跨了多个 fold（应为 0，S1 除外）"""
        g = df.groupby(key)[fold_col].nunique()
        return int((g > 1).sum())


    s1 = pd.read_csv(SPLITS / "S1_random.csv")
    s2 = pd.read_csv(SPLITS / "S2_group_dup.csv")
    s2b = pd.read_csv(SPLITS / "S2b_leave_paper.csv")
    s2c = pd.read_csv(SPLITS / "S2c_leave_doi.csv")      # ★ 第八场补：原先完全没检查 S2c
    s3 = pd.read_csv(SPLITS / "S3_leave_column.csv")
    s4 = pd.read_csv(SPLITS / "S4_murcko.csv")
    s4b = pd.read_csv(SPLITS / "S4b_ecfp_class.csv")
    s3p = pd.read_csv(SPLITS / "S3p_factor.csv")

    pj = p.reset_index(drop=True)
    for name, s in (("S1", s1), ("S2", s2), ("S2b", s2b), ("S2c", s2c),
                    ("S3", s3), ("S4", s4), ("S4b", s4b)):
        check(len(s) == len(p), f"{name} 行数 == 主表行数", f"{len(s)} vs {len(p)}")

    # ★ 第八场补（坑 32 的对齐口径）：所有划分必须靠 `row_id` 与主表对齐，
    #   而不是「碰巧长度相等」。原先门禁只比长度、再按位置贴 fold（`.values`），
    #   一旦划分文件被按别的顺序重写（或主表被重排）就检查不到 —— 故显式断言。
    for name, s in (("S1", s1), ("S2", s2), ("S2b", s2b), ("S2c", s2c),
                    ("S3", s3), ("S4", s4), ("S4b", s4b)):
        # ⚠️ detail 串必须**长度安全**：行数不等时 `np.isclose` 会广播报错，
        #    而「行数不等」正是本项要 FAIL 的情形（第八场自检时踩到）。
        _rid = s["row_id"].values
        _ok = len(_rid) == len(p) and np.array_equal(_rid, np.arange(len(p)))
        if _ok:
            _det = ""
        elif len(_rid) != len(p):
            _det = f"长度不符：{len(_rid)} vs 主表 {len(p)}"
        else:
            _det = f"首个不符位置={int(np.argmax(~np.isclose(_rid, np.arange(len(p)))))}"
        check(bool(_ok), f"{name}: row_id 严格等于主表行号 0..n-1（对齐口径）", _det)

    d = pj[["conn", "Column", "murcko", "doi"]].copy()
    # ⚠️ 长度安全护栏：若某个划分文件行数与主表不符（已在上面报 FAIL），
    #    按位置贴 fold 会 ValueError 把整个门禁打断；而且拿 NaN/-1 贴折号会让
    #    下面的「跨折」检查**空过**（NaN 被 nunique 丢弃）—— 所以这里用 len_ok
    #    把「行数一致」设成跨折检查的前置条件：长度不符时这些检查直接 FAIL，不空过。
    SPLIT_LIST = (("S1", s1), ("S2", s2), ("S2b", s2b), ("S2c", s2c),
                  ("S3", s3), ("S4", s4), ("S4b", s4b))
    len_ok = all(len(s) == len(d) for _, s in SPLIT_LIST)
    for nm, s in SPLIT_LIST:
        d[nm] = s["fold"].values if len(s) == len(d) else -1
    d["dup_group"] = d["conn"] + "|" + d["Column"].astype(str)
    check(len_ok and spans(d, "dup_group", "S2") == 0, "S2: 无 dup_group 跨折",
          f"len_ok={len_ok} 跨折 {spans(d,'dup_group','S2')}")
    check(len_ok and spans(d, "doi", "S2b") == 0, "S2b: 无文献(DOI) 跨折",
          f"len_ok={len_ok} 跨折 {spans(d,'doi','S2b')}")
    # ★ 第八场补：S2c 是论文主口径（真·留一文献），原先门禁一个字都没查
    check(len_ok and spans(d, "doi", "S2c") == 0, "S2c: 无文献(DOI) 跨折（真·留一文献）",
          f"len_ok={len_ok} 跨折 {spans(d,'doi','S2c')}")
    check(int(s2c["fold"].nunique()) == int(p["doi"].nunique()),
          "S2c: 折数 == 文献数（真 leave-one-DOI-out）",
          f"{s2c['fold'].nunique()} 折 vs {p['doi'].nunique()} 篇 DOI")
    check(int(s2c.groupby("fold")["paper"].nunique().max()) == 1,
          "S2c: 每折恰好一篇文献")
    check(len_ok and spans(d, "murcko", "S4") == 0, "S4: 无 Murcko 骨架跨折",
          f"len_ok={len_ok} 跨折 {spans(d,'murcko','S4')}")
    colmap = s3.groupby("heldout_column")["fold"].nunique()
    check(bool((colmap == 1).all()), "S3: 柱 ↔ 折一一对应（真 LOO-column）")
    check(len_ok and spans(d, "dup_group", "S1") > 0,
          "S1 故意不分组（应存在跨折 = 泄漏，用于对照）",
          f"len_ok={len_ok} 跨折 {spans(d,'dup_group','S1')}")

    print("\n── 6. S3′ 因子外推设计 ──")
    fam = c.set_index("Column")["family"].to_dict()
    tasks = sorted(s3p["task"].unique())
    check(len(tasks) == 6, "3 组匹配对 × 2 个方向 = 6 个任务", f"{len(tasks)}")
    ok_design = True
    for t in tasks:
        sub = s3p[s3p["task"] == t]
        # ⚠️ 用 isin 而不是 .loc[list]：主表被改短时 row_id 会越界，.loc 直接 KeyError
        #    把门禁打断；isin 会安全地退成空集 ⇒ 下面 good=False ⇒ FAIL（不空过）。
        tr = set(pj.loc[pj.index.isin(sub.loc[sub.role == "train", "row_id"]), "Column"])
        te = set(pj.loc[pj.index.isin(sub.loc[sub.role == "test", "row_id"]), "Column"])
        fams = {fam.get(x) for x in tr} | {fam.get(x) for x in te}
        surf_tr = {fam.get(x, "-").split("-")[0] for x in tr}
        surf_te = {fam.get(x, "-").split("-")[0] for x in te}
        good = len(surf_tr) == 1 and len(surf_te) == 1 and surf_tr != surf_te and len(fams) >= 2
        ok_design &= good
        print(f"        {t:34s} train柱={len(tr)} test柱={len(te)} "
              f"骨架 {surf_tr}→{surf_te} {'ok' if good else 'BAD'}")
    check(ok_design, "每个 S3′ 任务的 train/test 骨架不同（真外推）")
    cov = s3p["row_id"].nunique() / len(p)
    check(cov > 0.75, "S3′ 覆盖主表 > 75%", f"{cov*100:.1f}%")
    # ★ 第八场补（坑 32）：S3′ 是**长表**（同一 row_id 出现在多个 task 里），
    #   绝不能混进「按 fold 遍历」的循环 —— 必须显式断言它没有 fold 列。
    check("fold" not in s3p.columns, "S3′ 是长表、不含 fold 列（不可按 fold 遍历）",
          f"列={list(s3p.columns)}")

    print("\n── 7. 与审计/体检数字的对账 ──")
    check(abs(len(m) - 25867) <= 10, "记录级表 ≈ 25,867（原始规模）", f"{len(m)}")
    check(1000 < len(p) < 12000, "主表对数在合理区间", f"{len(p)}")
    a = np.exp(p["y"].abs())
    check(abs(float(a.median()) - 1.26) < 0.05, "无向 α 中位 ≈ 1.26（审计报 1.270）",
          f"{a.median():.4f}")
    check(abs(float(a.max()) - 32.18) < 1.0, "无向 α 最大 ≈ 32.18（审计报 32.18）",
          f"{a.max():.3f}")

    print("\n── 8. 划分的规模均衡性 ──")
    for nm, s in (("S1", s1), ("S2", s2), ("S2b", s2b), ("S4", s4), ("S4b", s4b)):
        vc = s["fold"].value_counts()
        ratio = vc.max() / vc.min()
        check(ratio < 12, f"{nm} 各折规模比 < 12×", f"{vc.min()}–{vc.max()} ({ratio:.1f}×)")

    print("\n" + "=" * 74)
    if FAILS:
        print(f"✗ {len(FAILS)} 项失败 / 共 {N_PASS + len(FAILS)} 项：")
        for f in FAILS:
            print("   -", f)
        return 1
    print(f"✓ 全部 {N_PASS} 项通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
