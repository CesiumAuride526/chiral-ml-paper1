# -*- coding: utf-8 -*-
"""
36_condition_pair_bootstrap.py — **真·化合物级聚类 bootstrap**（修正 scripts/34 的重采样口径）
================================================================================
## 为什么（R2 独立复核发现的口径错）
`scripts/21` 的 `condition_pairwise.csv` 给出 66 个条件对的**中位 r**，但**没有不确定性**。
本场先是在 `scripts/34` 里画了"compound-clustered bootstrap"——**但那时代码重采样的是
r 值本身（`rng.choice(rr, ...)`），不是化合物**，方法与标注不符（verifier 复刻其 seed=11
精确复现了图上那三个区间，证实机制）。

本脚本做**真的**聚类 bootstrap：化合物是抽样单元，每次抽 43 个分子（有放回），
在抽中的分子集合里重算每对条件的 Pearson r，再取中位数——这才对应"换一批化合物会怎样"。

## 产出
`data/curated/condition_pair_bootstrap.csv`（三类条件对的 median r + 95 % CI + n_pairs）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CUR = ROOT / "data" / "curated"
SEED = 20260918
B = 2000


def pair_r(mat: pd.DataFrame, a: str, b: str) -> float | None:
    d = mat[[a, b]].dropna()
    if len(d) < 5 or d[a].std() == 0 or d[b].std() == 0:
        return None
    return float(np.corrcoef(d[a], d[b])[0, 1])


def main() -> int:
    dg = pd.read_csv(CUR / "external_dg25_all.csv")
    g = dg[dg["conn"].notna() & dg["y"].notna()].copy()
    g["cond"] = g["column"].astype(str) + " | " + g["phase"].astype(str)
    mat = g.pivot_table(index="conn", columns="cond", values="y", aggfunc="mean")
    mols = list(mat.index)

    cp = pd.read_csv(CUR / "condition_pairwise.csv")
    groups = {
        "mobile phase only (same column)": cp[cp["same_col"] & ~cp["same_ph"]],
        "selector only (same mobile phase)": cp[~cp["same_col"] & cp["same_ph"]],
        "both changed": cp[~cp["same_col"] & ~cp["same_ph"]],
    }
    mat = mat.reindex(columns=sorted(mat.columns))

    # ── 1. 观测中位数（与 condition_pairwise.csv 对得上）─────────────────────
    rows = []
    rng = np.random.default_rng(SEED)
    draws = [rng.choice(mols, size=len(mols), replace=True) for _ in range(B)]

    print(f"化合物 {len(mols)} 个，条件 {mat.shape[1]} 个，bootstrap {B} 次（seed {SEED}）")
    for label, sub in groups.items():
        obs = [pair_r(mat, a, b) for a, b in zip(sub["a"], sub["b"])]
        obs = [x for x in obs if x is not None]
        med = float(np.median(obs))
        boot = []
        for idx in draws:
            mm = mat.loc[idx]
            rr = [pair_r(mm, a, b) for a, b in zip(sub["a"], sub["b"])]
            rr = [x for x in rr if x is not None]
            if rr:
                boot.append(float(np.median(rr)))
        boot = np.array(boot)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"  {label:<38} n_pairs={len(obs):>2}  median r={med:+.4f}  "
              f"95% CI [{lo:+.4f}, {hi:+.4f}]  (boot n={len(boot)})")
        rows.append(dict(group=label, n_pairs=len(obs), median_r=med,
                         ci_lo=float(lo), ci_hi=float(hi), boot_n=len(boot), seed=SEED))

    # ── 2. 组间差值：配对 bootstrap（同一批抽中的分子上比较两组的配对中位数）──
    print()
    print("配对差异（同一次重采样内比较两组）：")
    def boot_pairs(sub):
        out = []
        for idx in draws:
            mm = mat.loc[idx]
            rr = [pair_r(mm, a, b) for a, b in zip(sub["a"], sub["b"])]
            rr = [x for x in rr if x is not None]
            out.append(float(np.median(rr)) if rr else np.nan)
        return np.array(out)

    bp = {k: boot_pairs(v) for k, v in groups.items()}
    keys = list(groups)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            d = bp[a] - bp[b]
            d = d[~np.isnan(d)]
            lo, hi = np.percentile(d, [2.5, 97.5])
            frac = float((d <= 0).mean())
            print(f"  {a} − {b}: Δmedian = {np.median(d):+.4f} 95% CI [{lo:+.4f}, {hi:+.4f}]  "
                  f"P(Δ≤0) = {frac:.4f}  ⇒ {'可区分' if lo > 0 or hi < 0 else '不可区分'}")
            rows.append(dict(group=f"diff: {a} - {b}", n_pairs=len(d), median_r=float(np.median(d)),
                             ci_lo=float(lo), ci_hi=float(hi), boot_n=len(d), seed=SEED))

    out = pd.DataFrame(rows)
    out.insert(0, "script", "scripts/36_condition_pair_bootstrap.py")
    p = CUR / "condition_pair_bootstrap.csv"
    out.to_csv(p, index=False, encoding="utf-8")
    print(f"\n[落盘] {p.relative_to(ROOT)}  ({len(out)} 行)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
