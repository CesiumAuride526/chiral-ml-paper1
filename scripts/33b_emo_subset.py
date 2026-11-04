# -*- coding: utf-8 -*-
"""33b — EMO 上界（分开子集）+ DG25 网格 y=0 占比；配合 scripts/33 使用（定性补充）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CUR = ROOT / "data" / "curated"


def add_fit(g, cols):
    y = g["y"].to_numpy(float)
    X = pd.get_dummies(g[list(cols)].astype(str), drop_first=False).to_numpy(float)
    X = np.column_stack([np.ones(len(g)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return X @ beta


def report(g, label):
    n = len(g)
    sign = (g["y"] > 0).astype(int)
    maj = max(sign.mean(), 1 - sign.mean())
    zshare = float((g["y"].abs() < 1e-9).mean())
    pred_add = add_fit(g, ["mol", "cond"])
    acc_add = float(((pred_add > 0).astype(int) == sign).mean())
    # 留一分子：去掉分子效应，仅用条件效应 + 全局均值
    grand = g["y"].mean()
    cond_eff = g["y"] - g.groupby("mol")["y"].transform("mean")
    acc_loo = float((((grand + cond_eff) > 0).astype(int) == sign).mean())
    print(f"  {label}: n={n}  y=0 占 {100*zshare:.1f}%  多数类 {100*maj:.2f}%  "
          f"oracle 加性 {100*acc_add:.2f}%  LOO-分子 {100*acc_loo:.2f}%")
    return dict(subset=label, n=n, y_zero_share=zshare, majority=100 * maj,
                emo_oracle_additive=100 * acc_add, emo_loo_molecule=100 * acc_loo)


def main() -> int:
    dg = pd.read_csv(CUR / "external_dg25_all.csv")
    g = dg[dg["conn"].notna() & dg["y"].notna()].copy()
    g["mol"] = g["conn"]
    g["cond"] = g["column"].astype(str) + "|" + g["phase"].astype(str)
    rows = [report(g, "全部 494 格（含未分开）")]
    sep = g[g["alpha"] != 1].copy()
    rows.append(report(sep, "仅分开格 alpha!=1"))
    print("  ※『分开』= alpha != 1；未分开格的 y 恒为 0（无洗脱顺序可言）")
    out = pd.DataFrame(rows)
    p = CUR / "review_round2_emo_subset.csv"
    out.to_csv(p, index=False, encoding="utf-8")
    print(f"[落盘] {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
