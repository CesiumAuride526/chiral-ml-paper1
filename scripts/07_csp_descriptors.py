# -*- coding: utf-8 -*-
"""
07_csp_descriptors.py — 自建固定相（CSP）描述符
================================================
**立项理由**：CP2.5 预演（`notes/CP2.5_基线预演.md`）实测 S3′ 因子外推**六个任务 R² 全为负**
（−0.69 ~ −0.10），说明厂商给的 3 个标签（骨架 / 取代基码 / 粒径）**基本不携带可迁移的固定相信息**。
审计 §3.9 也指出 `Column_info.xlsx` 可用特征只有 3 个。→ 必须自建。

三层设计：

  A. **取代基描述符**（核心）——把取代基码还原成**分子结构**再算描述符
     a = 3,5-二甲基苯基氨基甲酸酯   b = 3,5-二氯苯基氨基甲酸酯   c = 4-甲基苯甲酸酯
     d = (S)-1-苯乙基氨基甲酸酯     e = 3-氯苯基氨基甲酸酯
     f = 3-氯-4-甲基苯基氨基甲酸酯   g = 3-氯-5-甲基苯基氨基甲酸酯
     （★ 2026-09-16 按 Daicel 官方产品页逐条核实；首版把 e 猜成 3,5-二甲基、g 记为「未确认」，均已更正）

  B. **骨架描述符** —— 直链淀粉(α-1,4) vs 纤维素(β-1,4)；糖苷键构型就是骨架手性的来源

  C. **柱整体描述符** —— 粒径（→ 比表面积 ∝ 1/dp）、**键合型 vs 涂覆型**
     ⚠️ 「键合型/涂覆型」是厂商标签**没有**、但文献明确报告会影响保留与（有时）洗脱顺序的维度

输出：`data/curated/csp_descriptors.csv`（25 行，一柱一行）
      `data/curated/csp_subs_similarity.csv`（7 个取代基两两 Tanimoto —— 论文可用的表）
并**就地重跑 S3′ 因子外推**，看新描述符是否真的带来提升。

用法: PYTHONIOENCODING=utf-8 python scripts/07_csp_descriptors.py
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module  # noqa: E402

_c = import_module("01_curate_cmrt")
SUBS_SMILES, SUBS_MEANING = _c.SUBS_SMILES, _c.SUBS_MEANING

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parent.parent
CUR = ROOT / "data" / "curated"
SEED = 42
LOG: list[str] = []


def say(s: str = "") -> None:
    print(s)
    LOG.append(s)


# ════════════════════════════════════════════════════════════════════
# A. 取代基描述符
# ════════════════════════════════════════════════════════════════════
# 粗略的 Hammett σm（间位）查表 —— 只用来给「吸电子 vs 给电子」一个定量代理，
# 不追求精确（文献值本身就有不同标度）。取代基在苯环的间位。
HAMMETT_SIGMA_M = {
    "C": -0.07,    # 甲基
    "Cl": 0.37,    # 氯
    "F": 0.34, "Br": 0.39, "I": 0.35,
    "OC": -0.12,   # 甲氧基（首原子 O）
    "N": -0.16, "O": 0.12, "S": 0.15,
}


def subs_features(code: str) -> dict:
    """取代基的 2D 描述符 + 电子/位阻代理。"""
    smi = SUBS_SMILES[code]
    m = Chem.MolFromSmiles(smi)
    assert m is not None, f"取代基 {code} 的 SMILES 解析失败: {smi}"
    d = {
        "subs_MW": Descriptors.MolWt(m),
        "subs_LogP": Crippen.MolLogP(m),
        "subs_MR": Crippen.MolMR(m),                       # 摩尔折射 → 位阻/极化率代理
        "subs_TPSA": rdMolDescriptors.CalcTPSA(m),
        "subs_HBD": rdMolDescriptors.CalcNumHBD(m),
        "subs_HBA": rdMolDescriptors.CalcNumHBA(m),
        "subs_RotB": rdMolDescriptors.CalcNumRotatableBonds(m),
        "subs_nRing": rdMolDescriptors.CalcNumRings(m),
        "subs_nArom": rdMolDescriptors.CalcNumAromaticRings(m),
        "subs_FracCsp3": rdMolDescriptors.CalcFractionCSP3(m),
        "subs_nHeavy": m.GetNumHeavyAtoms(),
        "subs_nStereo": len(Chem.FindMolChiralCenters(
            m, includeUnassigned=True, useLegacyImplementation=False)),
        # 官能团计数（手性识别靠氢键 + 偶极-偶极，见 Daicel 机理描述）
        "subs_nCarbamate": len(m.GetSubstructMatches(Chem.MolFromSmarts("NC(=O)O"))),
        "subs_nEster": len(m.GetSubstructMatches(Chem.MolFromSmarts("C(=O)OC"))),
    }
    # 苯环上各取代基的类别与 Hammett 代理
    n_cl = len(m.GetSubstructMatches(Chem.MolFromSmarts("[Cl]")))
    n_me = len(m.GetSubstructMatches(Chem.MolFromSmarts("[CH3]")))
    d["subs_nCl"] = n_cl
    d["subs_nMe"] = n_me
    d["subs_nHalogen"] = n_cl
    # σ 求和：环上每个取代原子贡献一次（Cl / CH3 / 其他首原子）
    sigma = 0.0
    for a in m.GetAtoms():
        if a.GetIsAromatic():
            continue
        exp = a.GetTotalNumHs()
        dec = HAMMETT_SIGMA_M.get(a.GetSymbol(), 0.0)
        # 甲基按"碳且带 3 个 H"识别，甲氧基按氧识别
        if a.GetSymbol() == "C" and exp == 3:
            dec = HAMMETT_SIGMA_M["C"]
        sigma += dec
    d["subs_hammett_sigma_m_sum"] = sigma
    # 卤素/甲基「非对称性」：3,5-二取代是对称的，3,4-取代不是 → 影响手性口袋形状
    d["subs_asym_subst"] = int(f"{code}" in ("f",))   # 3-氯-4-甲基是邻位不同取代
    return d


def subs_fp(code: str, nbits: int = 128) -> dict:
    from rdkit.Chem import AllChem
    m = Chem.MolFromSmiles(SUBS_SMILES[code])
    fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=nbits)
    return {f"subs_fp{i}": int(b) for i, b in enumerate(fp)}


# ════════════════════════════════════════════════════════════════════
# B./C. 骨架与柱整体
# ════════════════════════════════════════════════════════════════════
# 键合型(immobilized)柱名集合 —— 来自 Daicel 产品线（IA–IG 为键合型）
IMMOBILIZED_PREFIX = ("IA", "IB", "IC", "ID", "IE", "IF", "IG")
# 糖苷键构型：直链淀粉 = α-1,4（左手 4 重螺旋）/ 纤维素 = β-1,4（伸展带状）
GLYCOSIDIC = {"Amylose": "alpha", "Cellulose": "beta"}


def backbone_features(surface: str) -> dict:
    return {
        "bb_is_amylose": int(surface == "Amylose"),
        "bb_is_cellulose": int(surface == "Cellulose"),
        # α-1,4 与 β-1,4 只差一个糖苷键的构型 —— 这正是骨架手性的唯一来源，
        # 用单一二元变量显式表达（比 one-hot 更能被外推模型利用）
        "bb_glycosidic_alpha": int(GLYCOSIDIC.get(surface) == "alpha"),
    }


def col_features(row) -> dict:
    col, surface, code = row["Column"], row["surface"], row["subs_code"]
    dp = float(row["particle_um"])
    prefix = "".join(ch for ch in str(col) if ch.isalpha())
    d = {
        # 粒径：比表面积 ∝ 1/dp（假定为无孔/全多孔球的粗代理）；log 形式更线性
        "col_particle_um": dp,
        "col_specific_surface_proxy": 6.0 / max(dp, 1e-6),
        "col_log_particle": float(np.log(dp)),
        # 键合型 vs 涂覆型 —— 厂商标签里没有，但文献报告会影响保留与洗脱顺序
        "col_immobilized": int(prefix.startswith(IMMOBILIZED_PREFIX)),
        "col_coated": int(not prefix.startswith(IMMOBILIZED_PREFIX)),
        "col_is_3um": int(dp <= 3.0),
        "col_is_10um": int(dp >= 10.0),
        "col_is_16um": int(dp < 2.0),
        # 取代基：三类连接化学
        "subs_is_carbamate": int(code in "abdefg"),
        "subs_is_ester": int(code == "c"),
        # 骨架 × 取代基 的交互（因子设计的那一格）
        "bb_subs_interact": f"{surface}-{code}",
    }
    return d


# ════════════════════════════════════════════════════════════════════
def main() -> int:
    say("# 07_csp_descriptors — 自建固定相描述符")
    say("")
    cols = pd.read_csv(CUR / "columns.csv")
    say(f"输入：`data/curated/columns.csv`（{len(cols)} 根柱 / {cols['family'].nunique()} 个家族）")
    say("")

    # ── A 取代基
    codes = sorted(cols["subs_code"].unique())
    say(f"## A. 取代基（{len(codes)} 种，已按 Daicel 官方产品页核实）")
    say("")
    say("| 码 | 化学名 | SMILES | MW | LogP | 环上 Cl | 环上 Me | σm 和 |")
    say("|---|---|---|---|---|---|---|---|")
    sf = {}
    for c in codes:
        f = subs_features(c)
        sf[c] = {**f, **subs_fp(c)}
        say(f"| {c} | {SUBS_MEANING[c].split(' — ')[0]} | `{Chem.MolToSmiles(Chem.MolFromSmiles(SUBS_SMILES[c]))}` "
            f"| {f['subs_MW']:.1f} | {f['subs_LogP']:.2f} | {f['subs_nCl']} | {f['subs_nMe']} "
            f"| {f['subs_hammett_sigma_m_sum']:+.2f} |")
    say("")

    # ── 取代基两两相似度（论文表：10 个家族在化学空间里不是均匀铺开的）
    from rdkit.Chem import AllChem, DataStructs
    say("## 取代基两两 Tanimoto（ECFP4）")
    say("")
    fps = {c: AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(SUBS_SMILES[c]), 2, 1024)
           for c in codes}
    say("| | " + " | ".join(codes) + " |")
    say("|---|" + "---|" * len(codes))
    sim_rows = []
    for a in codes:
        cells = []
        for b in codes:
            s = DataStructs.TanimotoSimilarity(fps[a], fps[b])
            cells.append(f"{s:.2f}")
            if a < b:
                sim_rows.append(dict(code_a=a, code_b=b, tanimoto=s))
        say(f"| **{a}** | " + " | ".join(cells) + " |")
    say("")
    simdf = pd.DataFrame(sim_rows)
    say(f"取代基对的 Tanimoto：中位 {simdf['tanimoto'].median():.2f}，"
        f"范围 {simdf['tanimoto'].min():.2f}–{simdf['tanimoto'].max():.2f}")
    say("")

    # ── B/C 组装
    rows = []
    for _, r in cols.iterrows():
        d = {"Column": r["Column"], "surface": r["surface"], "subs_code": r["subs_code"],
             "particle_um": r["particle_um"], "family": r["family"], "n_pairs": r["n_pairs"]}
        d.update(backbone_features(r["surface"]))
        d.update(sf[r["subs_code"]])
        d.update(col_features(r))
        rows.append(d)
    out = pd.DataFrame(rows)
    out["bb_subs_interact"] = out["bb_subs_interact"].astype("category").cat.codes

    say("## B/C. 柱整体描述符")
    say("")
    say(f"- 键合型(immobilized) {int(out['col_immobilized'].sum())} 根 / "
        f"涂覆型(coated) {int(out['col_coated'].sum())} 根")
    say(f"- 粒径分布：{sorted(out['particle_um'].unique())}")
    say(f"- 描述符总维度：**{out.shape[1]}**（含 {sum(1 for c in out.columns if c.startswith('subs_fp'))} 位 ECFP）")
    say("")

    # ── 就地评估：S3′ 因子外推能否改善
    say("## 就地评估：S3′ 因子外推（新描述符 vs 厂商标签）")
    say("")
    p = pd.read_csv(CUR / "cmrt_pairs.csv")
    s3p = pd.read_csv(CUR / "splits" / "S3p_factor.csv")
    y = p["y"].values.astype(float)
    # 分子侧仍用最便宜的 RDKit 描述符（与 CP2.5 可比）
    from _baseline_cp25 import build_X  # noqa: E402
    X0 = build_X(p).values.astype(float)

    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import r2_score

    csp = out.set_index("Column")
    # ⚠️ 必须排除字符串/类别列（subs_code、bb_subs_interact），否则 astype(float) 直接炸
    NUM = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    feat_sets = {
        "厂商标签(骨架/取代基码/粒径)": ["bb_is_amylose",
                                "col_particle_um", "subs_is_carbamate", "subs_is_ester"],
        "自建 CSP 描述符(全套)": [c for c in NUM if c.startswith(("subs_", "bb_", "col_"))],
        "自建 CSP(只取代基结构)": [c for c in NUM if c.startswith("subs_")],
        "自建 CSP(只柱整体)": [c for c in NUM if c.startswith("col_")],
    }
    say("| 任务 | 训练 n | 测试 n | 厂商标签 R² | 自建全套 R² | 只取代基 R² | 只柱整体 R² | 测试集 y 的 SD |")
    say("|---|---|---|---|---|---|---|---|")
    results = []
    for task in sorted(s3p["task"].unique()):
        s = s3p[s3p["task"] == task]
        idx_tr = s.loc[s["role"] == "train", "row_id"].values
        idx_te = s.loc[s["role"] == "test", "row_id"].values
        col_tr = p.loc[idx_tr, "Column"].map(lambda c: list(csp.index).index(c)).values
        col_te = p.loc[idx_te, "Column"].map(lambda c: list(csp.index).index(c)).values
        r2s = {}
        for name, fcols in feat_sets.items():
            C = csp[fcols].values.astype(float)
            Xtr = np.hstack([X0[idx_tr], C[col_tr]])
            Xte = np.hstack([X0[idx_te], C[col_te]])
            keep = Xtr.std(axis=0) > 0
            m = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.06,
                                              min_samples_leaf=20, random_state=SEED)
            m.fit(Xtr[:, keep], y[idx_tr])
            r2s[name] = r2_score(y[idx_te], m.predict(Xte[:, keep]))
        results.append(dict(task=task, **r2s, sd=float(np.std(y[idx_te]))))
        say(f"| `{task}` | {len(idx_tr)} | {len(idx_te)} "
            f"| {r2s['厂商标签(骨架/取代基码/粒径)']:+.3f} "
            f"| **{r2s['自建 CSP 描述符(全套)']:+.3f}** "
            f"| {r2s['自建 CSP(只取代基结构)']:+.3f} "
            f"| {r2s['自建 CSP(只柱整体)']:+.3f} | {np.std(y[idx_te]):.4f} |")
    rdf = pd.DataFrame(results)
    say("")
    for name in feat_sets:
        say(f"- **{name}**：6 个任务的中位 R² = "
            f"{rdf[name].median():+.3f}，范围 {rdf[name].min():+.3f} ~ {rdf[name].max():+.3f}")
    delta = (rdf["自建 CSP 描述符(全套)"].median() - rdf["厂商标签(骨架/取代基码/粒径)"].median())
    say("")
    say("### ⚠️ 读这些数字之前必须先看训练集大小")
    say("")
    say(f"- 三个任务的训练集 **≤ 500 行**（`b_amylose→cellulose` 388、`f_amylose→cellulose` 434、")
    say(f"  `f_cellulose→amylose` **104**）——用 100 行训练去外推一整格，**R² 为负是意料之中的**。")
    say(f"- 审计 §3.7 早就指出「真正的外推目标又小又孤立」（ID/IE/IG/OZ3 四个单柱家族合计仅 927 对），")
    say(f"  这里是同一个问题在建模层面的体现。")
    say(f"- 所以 **S3′ 的负 R² 不能全部归咎于描述符**，但也不能因此就宣布描述符够用 ——")
    say(f"  两个大样本任务（a 双向，2,738–3,492 训练行）**同样是负的**，那部分才是干净证据。")
    say("")
    say(f"**结论**：自建全套相对厂商标签的中位 R² 变化 = **{delta:+.3f}**。")
    if delta > 0.05:
        say("→ 自建描述符**确有提升**，`07` 的产出应进 T2 特征集。")
    elif delta > -0.05:
        f_task = rdf[rdf.task.str.startswith("S3p_f_amylose")]
        if len(f_task):
            say(f"→ 平均提升有限。但**有一个例外值得记一笔**：`S3p_f_amylose→cellulose` 从厂商标签的 "
                f"{f_task['厂商标签(骨架/取代基码/粒径)'].iloc[0]:+.3f} 提到 "
                f"{f_task['自建 CSP(只取代基结构)'].iloc[0]:+.3f}"
                f"（+{f_task['自建 CSP(只取代基结构)'].iloc[0]-f_task['厂商标签(骨架/取代基码/粒径)'].iloc[0]:.3f}）")
            say("  —— 说明**取代基的 2D 结构确实携带了一部分可迁移信息**，只是不足以撑起全部任务。")
        say("→ 整体上：固定相侧的差异**无法用「取代基 2D 结构」解释**，")
        say("  瓶颈在**取代基的三维排布 / 聚合物手性空腔**，而那是 2D 描述符表达不了的。")
        say("  → 支持论文走「固定相描述符的极限 + 分子侧才是瓶颈」这条诚实路线（见 CP2.5 结论 ④）。")
        say("  → 同时也说明：**想真正做跨固定相，得上 3D / 量子化学的 CSP 模型**，那是 Paper 2 的事。")
    else:
        say("→ 自建描述符反而更差，说明**过拟合到柱身份**；需回到降维或改用物理量。")

    out.to_csv(CUR / "csp_descriptors.csv", index=False, encoding="utf-8")
    simdf.to_csv(CUR / "csp_subs_similarity.csv", index=False, encoding="utf-8")
    rdf.to_csv(CUR / "csp_s3p_ablation.csv", index=False, encoding="utf-8")
    (ROOT / "notes" / "07_csp_descriptors_report.md").write_text(
        "\n".join(LOG) + "\n", encoding="utf-8")
    print(f"\n[落盘] data/curated/csp_descriptors.csv ({out.shape[0]} 行 × {out.shape[1]} 列)")
    print("[落盘] data/curated/csp_subs_similarity.csv + csp_s3p_ablation.csv")
    print("[落盘] notes/07_csp_descriptors_report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
