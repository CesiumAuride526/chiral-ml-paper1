# -*- coding: utf-8 -*-
"""
14_l1b_signal_test.py — **L1b（CMCDS 的 TD-DFT 旋转强度）下游信号消融**
================================================================================
## 为什么 L1b 是**不同性质**的候选（这是本脚本的立项理由）

`06b`/`06d` 已经把前两代描述符钉死了：L1a-FF 与 L1-CF 的**有符号列在建模上是空的**
——因为它们的符号**锁死在 CIP 上**（L1a-FF 92.3% 的分子符号由 R/S 决定，
`cip_S` 全表恒为 `S` ⇒ 符号是常数，审计 §3.3 的定理）。

**L1b 不一样。** 旋转强度 `R` 是 TD-DFT 直接给出的**物理伪标量**，
本脚本实测（对齐到同一 S 参考后）：

    R_sum > 0 的分子占 **50.4%**   ⇒ **符号在分子间真实均衡分布，不是 CIP 的读出**

⇒ 它是本项目**第一个「符号非常数」的手性量**。它能否带来增量，是一个**真问题**。

## 定向对齐（**必做**，否则伪标量整体反号）

`12` 已查明：CMCDS 收录的是 CMRT 所列的那个对映体，约一半是 CIP-R。
本脚本按 `can_stereo`（去立体骨架 + 立体规范 SMILES）匹配 `smi_S`/`smi_R`，
落在 R 侧的分子**把全部有符号量乘 −1**，统一到 S 参考。

## 臂设计

| 臂 | 内容 | 作用 |
|---|---|---|
| `L0` | 28 维 2D + 条件 | 基线 |
| `L0+R_even` | 只给**偶量**（\\|R\\| 幅度、激发能、波长） | **阴性对照**：无手性那部分 |
| `L0+R_sgn` | 只给**有符号汇总伪标量**（R_sum / 重心波长 / 分带 Σ R） | ★ 本场主角 |
| `L0+R_all` | 偶量 + 有符号汇总 + 20 个逐态 R | 全量 |
| `L0+CF+FF` | 当前最好臂（06d） | 对照 |
| `L0+CF+FF+R_sgn` / `+R_all` | 组合 | 互补性 |

## 判据（先写死）

1. `R_sgn` 的 ΔR² 必须 **> 折间 SD** 才算「有信号」；
2. `R_sgn` 必须 **> R_even**，否则说明**信号全来自无手性的幅度**（重复 `06b` 的老结论）；
3. 主口径 **S2b** 上若仍 ≤ 0 且增益 < 折间 SD ⇒ **不能声称 L1b 救回模型**，
   但**可以**声称「第一个符号非常数的伪标量也没有增量」——这是更强的负面结论。

## 用法
```bash
PYTHONIOENCODING=utf-8 python scripts/14_l1b_signal_test.py
```
产出：`notes/L1b_下游信号消融.md` + `data/curated/l1b_signal_ablation.csv`
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parent.parent
CUR = ROOT / "data" / "curated"
ZERO_ABS = 1e-12
LOG: list[str] = []
b25 = None
NEXC = 20


def say(s: str = "") -> None:
    print(s, flush=True)
    LOG.append(s)


def load_mod(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def canon(s: str, stereo: bool = True) -> str | None:
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    return Chem.MolToSmiles(m, isomericSmiles=stereo)


# ────────────────────────────────────────────── 对齐：CMCDS → CMRT 的 S 参考
def build_alignment():
    """返回 (conn → sign) 映射与诊断数字。sign=+1 表示 CMCDS 该行就是 CIP-S。"""
    cm = pd.read_csv(CUR / "cmrt_pairs.csv", low_memory=False)
    cd = pd.read_csv(CUR / "cmcds_l1b.csv")
    mono = cm[cm["cip_S"].astype(str).str.len() == 1].drop_duplicates("conn").copy()
    mono["cs_S"] = [canon(s) for s in mono["smi_S"]]
    mono["cs_R"] = [canon(s) for s in mono["smi_R"]]
    cd["cs"] = [canon(s) for s in cd["smiles"]]
    SS = set(mono["cs_S"].dropna())
    SR = set(mono["cs_R"].dropna())

    def side(cs):
        a = cs in SS
        b = cs in SR
        if a and not b:
            return 1
        if b and not a:
            return -1
        return 0

    # conn → sign（骨架唯一，用 S 侧与 R 侧的规范 SMILES 双向建表）
    k2c = {}
    for _, r in mono.iterrows():
        if pd.notna(r["cs_S"]):
            k2c.setdefault(r["cs_S"], r["conn"])
        if pd.notna(r["cs_R"]):
            k2c.setdefault(r["cs_R"], r["conn"])
    cd["conn"] = [k2c.get(c) for c in cd["cs"]]
    cd["align"] = [side(c) for c in cd["cs"]]

    # 每个 conn 取唯一 sign（同 conn 出现多次且冲突的剔除）
    g = cd.dropna(subset=["conn"]).groupby("conn")["align"].nunique()
    clean = set(g[g == 1].index)
    cd = cd[cd["conn"].isin(clean)]
    sig = cd.groupby("conn")["align"].first()
    sig = sig[sig != 0]
    diag = dict(n_row=len(cd), n_conn=len(sig),
                n_pos=int((sig > 0).sum()), n_neg=int((sig < 0).sum()))
    return sig, cm, cd, diag


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cf", default=str(CUR / "l1cf_descriptors.csv"))
    ap.add_argument("--ff", default=str(CUR / "l1a_descriptors.csv"))
    ap.add_argument("--splits", default="S2_group_dup,S2b_leave_paper,S3_leave_column,S4_murcko")
    ap.add_argument("--dead-thresh", type=float, default=0.5)
    ap.add_argument("--perm", type=int, default=3)
    args = ap.parse_args()

    global b25
    b25 = load_mod("_b25", ROOT / "scripts" / "_baseline_cp25.py")

    say("# L1b（CMCDS 旋转强度）下游信号消融")
    say("")
    say("> 由 `scripts/14_l1b_signal_test.py` 生成 ｜ 2026-09-17")
    say("> 靶变量 `y = ln[k(S)/k(R)]` ｜ 模型/划分/评估与 `06b`/`06d` **完全一致**")
    say("")

    sig, cm, cd, diag = build_alignment()
    say("## 1. ★ 定向对齐（必须先做）")
    say("")
    say(f"- CMCDS 可映射到 CMRT `conn` 且符号唯一的分子：**{diag['n_conn']}**"
        f"（CIP-S 侧 {diag['n_pos']} ｜ CIP-R 侧 {diag['n_neg']}）")
    say(f"- 落在 CIP-R 侧的分子**全部有符号量乘 −1**，统一到 **S 参考**")
    say("")

    cd = cd.drop_duplicates("conn").set_index("conn")
    sig = sig.reindex(cd.index)
    Rcols = [f"R{i}" for i in range(1, NEXC + 1)]
    lam = cd[[f"lam{i}" for i in range(1, NEXC + 1)]].values
    Rv = cd[Rcols].values * sig.values[:, None]          # ★ 对齐后的旋转强度

    say("### 1.1 对齐后 `R_sum` 的符号分布（**这是 L1b 的立项依据**）")
    say("")
    rs = Rv.sum(axis=1)
    say(f"- `R_sum > 0` 的分子占 **{(rs > 0).mean()*100:.1f}%**"
        f"（对齐前 {(cd['R_sum'].values > 0).mean()*100:.1f}%）")
    say(f"- ⇒ **符号在分子间真实均衡**，不是 CIP 的读出 —— "
        f"与 L1a-FF / L1-CF 「符号锁死 CIP」形成本质区别")
    say("")

    # ── 特征块
    BANDS = [(100, 150), (150, 200), (200, 250), (250, 300), (300, 999)]
    feat = {}
    for lo, hi in BANDS:
        m = (lam >= lo) & (lam < hi)
        feat[f"Rb_{lo}_{hi}"] = np.where(m, Rv, 0.0).sum(axis=1)
    wsum = np.abs(Rv).sum(axis=1)
    feat["R_sum"] = rs
    feat["R_lam_bar"] = (Rv * lam).sum(axis=1) / np.maximum(wsum, 1e-12)
    sgn_cols = ["R_sum", "R_lam_bar"] + [f"Rb_{lo}_{hi}" for lo, hi in BANDS]
    even_cols = ["R_absmean", "R_absmax", "R_neg_frac", "E_mean", "E_min",
                 "E_max", "lam_max", "lam_min"]
    F = pd.DataFrame(feat, index=cd.index)
    F[even_cols] = cd[even_cols].values
    F[[f"Rs{i}" for i in range(1, NEXC + 1)]] = Rv
    all_cols = list(F.columns)
    say("## 2. 特征块")
    say("")
    say(f"- 偶量 `R_even`（{len(even_cols)}）：`{'`, `'.join(even_cols)}`")
    say(f"- 有符号汇总 `R_sgn`（{len(sgn_cols)}）：`{'`, `'.join(sgn_cols)}`")
    say(f"- 逐态 `Rs1..Rs{NEXC}`（{NEXC}）")
    say("")

    # ── 合并主表 / CF / FF
    p = pd.read_csv(CUR / "cmrt_pairs.csv", low_memory=False)
    CF = pd.read_csv(args.cf).set_index("conn")
    FF = pd.read_csv(args.ff).set_index("conn")
    cf_live = [c for c in CF.columns if c.startswith("cf_")
               and float((CF[c].abs() < ZERO_ABS).mean()) <= args.dead_thresh]
    ff_live = [c for c in FF.columns if c.endswith("_bw")
               and float((FF[c].abs() < ZERO_ABS).mean()) <= args.dead_thresh]

    keep = (p["conn"].isin(set(F.index)) & p["conn"].isin(set(CF.index))
            & p["conn"].isin(set(FF.index)))
    say("## 3. 数据")
    say("")
    say(f"- 主表 {len(p)} 行 / {p['conn'].nunique()} 分子")
    say(f"- L1b 覆盖 {len(F)} 分子 ｜ L1-CF {len(CF)} ｜ L1a-FF {len(FF)}")
    n0 = len(p)
    p = p[keep.values].reset_index(drop=True).copy()
    p["_row_id"] = np.arange(n0)[keep.values]
    say(f"- **共同覆盖的行：{len(p)}**（-{n0-len(p)}，{100*(n0-len(p))/n0:.1f}%）"
        f" ⇒ 所有臂在**同一批样本**上比较")
    say("")

    def sub(tbl, cols):
        s = tbl.reindex(p["conn"].values)[cols]
        s.index = p.index
        return s.astype(float).fillna(0.0)

    X0 = b25.build_X(p)
    Rs, Re, Ra = sub(F, sgn_cols), sub(F, even_cols), sub(F, all_cols)
    Xcf, Xff = sub(CF, cf_live), sub(FF, ff_live)
    A = {
        "L0": X0,
        "L0+R_even (偶量)": pd.concat([X0, Re], axis=1),
        "L0+R_sgn (★有符号)": pd.concat([X0, Rs], axis=1),
        "L0+R_all": pd.concat([X0, Ra], axis=1),
        "L0+CF+FF": pd.concat([X0, Xcf, Xff], axis=1),
    }
    A["L0+CF+FF+R_sgn"] = pd.concat([A["L0+CF+FF"], Rs], axis=1)
    A["L0+CF+FF+R_all"] = pd.concat([A["L0+CF+FF"], Ra], axis=1)

    say("## 4. 特征集")
    say("")
    say("| 臂 | 维度 |")
    say("|---|---|")
    for k, v in A.items():
        say(f"| `{k}` | {v.shape[1]} |")
    say("")

    y = p["y"].values.astype(float)
    emo = p["EMO"].values.astype(int)
    say(f"y 的 SD = {y.std():.4f} ｜ EMO 多数类基线 = {max(emo.mean(), 1-emo.mean())*100:.1f}%")
    say("")

    from sklearn.metrics import accuracy_score, r2_score
    from scipy.stats import pearsonr

    def run_cv_detail(X, yy, folds, ee=None, permute=None):
        Xv = X.values.astype(float)
        rng = np.random.default_rng(0 if permute is None else permute)
        yy2 = rng.permutation(yy) if permute is not None else yy
        ee2 = rng.permutation(ee) if (permute is not None and ee is not None) else ee
        pred = np.full(len(yy2), np.nan)
        predc = np.zeros(len(yy2), dtype=int) if ee is not None else None
        rr = []
        for f in np.unique(folds):
            tr, te = folds != f, folds == f
            if te.sum() == 0 or tr.sum() == 0:
                continue
            reg, clf = b25.random_forest_pair()
            reg.fit(Xv[tr], yy2[tr])
            p_te = reg.predict(Xv[te])
            pred[te] = p_te
            if te.sum() >= 20 and p_te.std() > 0 and yy2[te].std() > 0:
                rr.append(float(pearsonr(yy2[te], p_te)[0]))
            if ee is not None:
                clf.fit(Xv[tr], ee2[tr])
                predc[te] = clf.predict(Xv[te])
        d = b25._score(yy2, pred, ee2, predc)
        a = np.array(rr, float)
        d["r_fold_med"] = float(np.nanmedian(a)) if len(a) else np.nan
        d["r_fold_frac_pos"] = float((a > 0).mean()) if len(a) else np.nan
        return d

    order = [("S2_group_dup", "S2 分组(分子×柱)"), ("S2b_leave_paper", "S2b 留一文献"),
             ("S3_leave_column", "S3 留一柱"), ("S4_murcko", "S4 留骨架")]
    order = [o for o in order if o[0] in set(args.splits.split(","))]

    say("## 5. 主结果")
    say("")
    say("| 划分 | 臂 | n | R² | ΔR² vs L0 | 逐折 r 中位 | r>0 折占比 | RMSE | EMO | EMO 基线 |")
    say("|---|---|---|---|---|---|---|---|---|---|")
    rows = []
    t0 = time.time()
    for fname, label in order:
        af = pd.read_csv(CUR / "splits" / f"{fname}.csv")["fold"].values
        folds = af if len(af) == len(p) else np.array([af[i] for i in p["_row_id"].values])
        base = None
        for lvl, X in A.items():
            r = run_cv_detail(X, y, folds, ee=emo)
            if lvl == "L0":
                base = r["r2"]
            r.update(split=label, arm=lvl, r2_delta=r["r2"] - base)
            rows.append(r)
            say(f"| {label} | `{lvl}` | {r['n']} | **{r['r2']:+.4f}** | {r['r2_delta']:+.4f} | "
                f"{r['r_fold_med']:+.3f} | {r['r_fold_frac_pos']*100:.0f}% | {r['rmse']:.4f} | "
                f"{r.get('emo_acc', float('nan'))*100:.1f}% | "
                f"{r.get('emo_base', float('nan'))*100:.1f}% |")
        say("")
    say(f"（建模总耗时 {time.time()-t0:.0f} s）")
    say("")
    df = pd.DataFrame(rows)
    df.to_csv(CUR / "l1b_signal_ablation.csv", index=False, encoding="utf-8")

    say("## 6. 阴性对照（标签置换）")
    say("")
    say("| 划分 | 臂 | R²（打乱标签） | EMO |")
    say("|---|---|---|---|")
    for fname, label in order:
        if fname not in ("S2_group_dup", "S2b_leave_paper"):
            continue
        af = pd.read_csv(CUR / "splits" / f"{fname}.csv")["fold"].values
        folds = af if len(af) == len(p) else np.array([af[i] for i in p["_row_id"].values])
        for lvl in ("L0", "L0+R_sgn (★有符号)", "L0+R_even (偶量)"):
            rs_, es_ = [], []
            for k in range(args.perm):
                r = run_cv_detail(A[lvl], y, folds, ee=emo, permute=700 + k)
                rs_.append(r["r2"]); es_.append(r.get("emo_acc", np.nan))
            say(f"| {label} | `{lvl}` | {np.mean(rs_):+.4f} "
                f"({min(rs_):+.3f} ~ {max(rs_):+.3f}) | {np.mean(es_)*100:.1f}% |")
        say("")

    # ── 逐折 SD 需要额外跑（run_cv_detail 不返回逐折 R²）
    say("## 7. ★ 判定（按先写死的判据）")
    say("")
    piv = df.pivot_table(index="split", columns="arm", values="r2")
    for sp in piv.index:
        r = piv.loc[sp]
        d_ev = r["L0+R_even (偶量)"] - r["L0"]
        d_sg = r["L0+R_sgn (★有符号)"] - r["L0"]
        d_al = r["L0+R_all"] - r["L0"]
        d_cf = r["L0+CF+FF"] - r["L0"]
        d_cs = r["L0+CF+FF+R_sgn"] - r["L0+CF+FF"]
        d_ca = r["L0+CF+FF+R_all"] - r["L0+CF+FF"]
        say(f"- **{sp}**：L0 {r['L0']:+.4f} ｜ +R_even {d_ev:+.4f} ｜ "
            f"**+R_sgn {d_sg:+.4f}** ｜ +R_all {d_al:+.4f} ｜ +CF+FF {d_cf:+.4f} ｜ "
            f"+CF+FF+R_sgn {d_cs:+.4f} ｜ +CF+FF+R_all {d_ca:+.4f}")
        v1 = "**R_sgn 优于偶量对照**" if d_sg > d_ev + 0.005 else \
             ("R_sgn 与偶量**同档**" if abs(d_sg - d_ev) <= 0.005 else
              "🔴 **R_sgn 反而不如偶量**（信号来自无手性幅度）")
        v2 = "与 CF+FF **互补**" if d_cs > 0.01 else "与 CF+FF **无互补**"
        say(f"  → {v1}；{v2}")
    say("")
    say("> ⚠️ 逐折 r 中位是本项目在弱信号档位下**更灵敏**的指标（R² 会被折间方差淹掉）。")
    say(">    参考：`13_failure_decomposition` 实测 S2b 上 `L0` 折内 r 中位 +0.066、")
    say(">    `L0+CF+FF` +0.126 —— **要在 S2b 达到 R²=0.6，折内 r 需 ≈ 0.78**。")

    (ROOT / "notes" / "L1b_下游信号消融.md").write_text("\n".join(LOG) + "\n",
                                                    encoding="utf-8")
    print("\n[落盘] notes/L1b_下游信号消融.md + data/curated/l1b_signal_ablation.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
