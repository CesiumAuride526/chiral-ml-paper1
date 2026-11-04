# -*- coding: utf-8 -*-
"""
33c_emo_order_ceiling.py — **洗脱顺序上界**：正确口径的"标签可加规则能做多好"
================================================================================
## 为什么需要这一版
`scripts/33b` 把整张 494 格网格的符号当成标签，得到"oracle 加性 57.7 %"。
但 DG25 网格里 **56.5 % 的格子 α ≡ 1（根本没分开）⇒ y 恒为 0**，符号本身无意义；
于是那个数实际衡量的是"能否预测『没分开』"，而不是"能否预测洗脱顺序"，
也不该与领域报告的 76–96 % 直接比。

## 正确口径（本脚本）
- **分母 = 真正分开的格子**（α ≠ 1，215 格）；标签 = 洗脱顺序（y>0 与否）。
- 候选规则：
  1. 全局多数类（常数预测器）
  2. **条件主效应**（每根柱/每个条件的多数方向）——任何"只用条件"的规则能做到的最好水平
  3. **oracle 两因素加性拟合**（分子哑变量 + 条件哑变量，无交互）
  4. **留一分子**：分子效应未知，用全局均值 + 条件效应预测（诚实的加性基线）
  5. **每分子 oracle 多数类**：上界参照（等于"知道这个分子"），用来显示"分子身份本身"的信息量
- 同时报 **AUC**（阈值无关），避免"阈值 0.5"这种人为选择被质疑。

## 产出
- `data/curated/review_round2_emo_order_ceiling.csv`
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CUR = ROOT / "data" / "curated"


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Mann–Whitney AUC（正确处理并列）。"""
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    s = score[order]
    i = 0
    r = 1
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (r + (r + (j - i))) / 2.0
        r += (j - i + 1)
        i = j + 1
    n1 = label.sum()
    n0 = len(label) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[label == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> int:
    dg = pd.read_csv(CUR / "external_dg25_all.csv")
    g = dg[dg["conn"].notna() & dg["y"].notna()].copy()
    g["mol"] = g["conn"]
    g["cond"] = g["column"].astype(str) + "|" + g["phase"].astype(str)
    sep = g[g["alpha"] != 1].copy()
    sep["sign"] = (sep["y"] > 0).astype(int)
    n = len(sep)
    base = max(sep["sign"].mean(), 1 - sep["sign"].mean())

    print("=" * 78)
    print(f"洗脱顺序上界：仅 {n} 个真正分开的格子（DG25 网格中 α≠1）")
    print(f"  正类（y>0）占 {100*sep['sign'].mean():.2f}%  全局多数类 {100*base:.2f}%")
    print("=" * 78)

    rows = []
    rows.append(dict(rule="global_majority", n=n, emo_acc=100 * base, auc=50.0,
                     note="常数预测器（阈值 0.5）"))

    # 1) 条件主效应：每个条件取该条件下的多数方向（阈值 0.5 用其均值符号）
    cond_mean = sep.groupby("cond")["y"].transform("mean")
    m = (cond_mean > 0).astype(int)
    rows.append(dict(rule="condition_main_effect", n=n,
                     emo_acc=100 * float((m == sep["sign"]).mean()),
                     auc=100 * auc(cond_mean.to_numpy(), sep["sign"].to_numpy()),
                     note="只用条件身份的多数方向"))

    # 2) oracle 两因素加性拟合（分子哑变量 + 条件哑变量）
    X = pd.get_dummies(sep[["mol", "cond"]].astype(str), drop_first=False).to_numpy(float)
    X = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(X, sep["y"].to_numpy(float), rcond=None)
    pred = X @ beta
    rows.append(dict(rule="oracle_additive_two_way", n=n,
                     emo_acc=100 * float(((pred > 0).astype(int) == sep["sign"]).mean()),
                     auc=100 * auc(pred, sep["sign"].to_numpy()),
                     note="分子 + 条件哑变量，无交互 ⇒ 标签可加类的上界"))

    # 3) 留一分子：分子效应未知 ⇒ 全局均值 + 条件效应
    cond_eff = sep["y"] - sep.groupby("mol")["y"].transform("mean")
    pred_loo = sep["y"].mean() + cond_eff
    rows.append(dict(rule="leave_one_molecule_additive", n=n,
                     emo_acc=100 * float(((pred_loo > 0).astype(int) == sep["sign"]).mean()),
                     auc=100 * auc(pred_loo.to_numpy(), sep["sign"].to_numpy()),
                     note="诚实加性基线：未知分子的分子效应按全局均值"))

    # 4) 留一条件（对称检查）
    mol_eff = sep["y"] - sep.groupby("cond")["y"].transform("mean")
    pred_looc = sep["y"].mean() + mol_eff
    rows.append(dict(rule="leave_one_condition_additive", n=n,
                     emo_acc=100 * float(((pred_looc > 0).astype(int) == sep["sign"]).mean()),
                     auc=100 * auc(pred_looc.to_numpy(), sep["sign"].to_numpy()),
                     note="未知条件：条件效应按全局均值"))

    # 5) 每分子 oracle 多数类（上界参照：知道分子身份）
    molmaj = sep.groupby("mol")["sign"].transform("mean")
    mm = (molmaj > 0.5).astype(int)
    rows.append(dict(rule="per_molecule_oracle_majority", n=n,
                     emo_acc=100 * float((mm == sep["sign"]).mean()),
                     auc=100 * auc(molmaj.to_numpy(), sep["sign"].to_numpy()),
                     note="上界参照：知道该分子的多数方向"))

    # 6) 每个分子的多数类基线加权（更容易读的对照）
    from collections import Counter
    per = [max(Counter(grp["sign"]).values()) / len(grp) for _, grp in sep.groupby("mol")]
    rows.append(dict(rule="per_molecule_majority_weighted", n=n,
                     emo_acc=100 * float(np.mean(per)), auc=float("nan"),
                     note="按分子先算多数类准确率再平均"))

    out = pd.DataFrame(rows)
    for _, r in out.iterrows():
        a = f"{r['auc']:.2f}" if pd.notna(r["auc"]) else "  —  "
        print(f"  {r['rule']:<34} EMO {r['emo_acc']:6.2f}%   AUC {a:>6}   {r['note']}")

    print()
    print("  读法：'condition_main_effect' 与 'oracle_additive_two_way' 是所有『标签可加』")
    print("  规则能达到的水平；'per_molecule_oracle_majority' 是知道分子身份后的上限参照。")
    out.insert(0, "script", "scripts/33c_emo_order_ceiling.py")
    p = CUR / "review_round2_emo_order_ceiling.csv"
    out.to_csv(p, index=False, encoding="utf-8")
    print(f"\n[落盘] {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
