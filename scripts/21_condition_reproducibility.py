# -*- coding: utf-8 -*-
"""
21_condition_reproducibility.py — **无模型、无描述符的条件间可复现性**（DG25 内部）
================================================================================
## 立项理由（2026-09-17 第七场）

`scripts/19` 证明了「同实验室 r=0.185 / 换实验室 r≈0」，`scripts/17` 证明了
文献偏移不可由组成解释。但这两条都**要经过模型**，审稿人总可以说
「那是你的模型不行」。本脚本给出**绕开模型和描述符**的那一块证据。

**DG25 的设计天然适合**：De Gauquier 2025 把**同一批 42–43 个分子**
在 **6 根柱 × 2 种流动相 = 12 个条件**下**全部测了一遍**（同一实验室、同一批样品）。

⇒ 于是可以直接问一个**纯测量学**问题：

    在 A 条件下测到的 α，能告诉你多少关于 B 条件下 α 的信息？

把 12 个条件两两配对，按「是否同柱」「是否同流动相」分成 2×2 = 四类，
**不需要任何描述符、不需要任何模型**，直接算跨条件的相关系数。

**如果连「换一根柱 / 换一个流动相」都复现不出来，那么**
**「预测一个没见过的实验室」的天花板就被钉死了 —— 而且这是纯测量学的结论。**

## 口径
- 只保留**两个条件都有测量**的分子对（成对完整观测，pairwise-complete）
- 报 Pearson r、Spearman ρ，以及**符号一致率**（EMO 意义上的）
- 四类：① 同柱 × 同相（**不存在**，只有 1 次测量 ⇒ 无法估）
         ② 同柱 × 异相  ③ 异柱 × 同相  ④ 异柱 × 异相
- ⚠️ 「同柱×同相」没有重复测量 ⇒ **本数据无法估计测量噪声**，只能做**条件间**对照

## 产出
- `data/curated/condition_pairwise.csv`（逐条件对）
- `notes/条件间可复现性_无模型.md`

## 用法
```bash
PYTHONIOENCODING=utf-8 python scripts/21_condition_reproducibility.py
```
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CUR = ROOT / "data" / "curated"
LOG: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    LOG.append(s)


def main() -> int:
    from scipy.stats import pearsonr, spearmanr

    d = pd.read_csv(CUR / "external_dg25_all.csv", low_memory=False)
    say("# 条件间可复现性（无模型、无描述符）—— DG25 内部")
    say("")
    say("> 由 `scripts/21_condition_reproducibility.py` 生成 ｜ 2026-09-17")
    say("")
    say(f"DG25：**{len(d)}** 条记录 / **{d['conn'].nunique()}** 个分子 / "
        f"**{d['column'].nunique()}** 根柱 × **2** 种流动相 ⇒ "
        f"**{d.groupby(['column','phase']).ngroups}** 个条件")
    say("")
    say("> ★ **这是同一实验室、同一批分子、只换条件的测量** ⇒")
    say("> 记到的差异**不可能**是「分子不同」或「实验室不同」，只能是**条件**。")
    say("")

    # ── 条件 × 分子 的矩阵
    d["cond"] = d["column"] + " | " + d["phase"]
    M = d.pivot_table(index="conn", columns="cond", values="y", aggfunc="mean")
    say(f"矩阵：**{M.shape[0]}** 分子 × **{M.shape[1]}** 条件"
        f"（完整观测的单元格 {int(M.notna().sum().sum())}/{M.size}）")
    say("")

    # ── 逐条件对的相关系数
    rows = []
    for a, b in combinations(M.columns, 2):
        sub = M[[a, b]].dropna()
        if len(sub) < 10:
            continue
        x, y = sub[a].values, sub[b].values
        if x.std() < 1e-9 or y.std() < 1e-9:
            continue
        col_a, ph_a = a.split(" | ")
        col_b, ph_b = b.split(" | ")
        same_col = col_a == col_b
        same_ph = ph_a == ph_b
        rows.append(dict(
            a=a, b=b, n=len(sub),
            same_col=same_col, same_ph=same_ph,
            r=float(pearsonr(x, y)[0]), rho=float(spearmanr(x, y)[0]),
            sign_agree=float(np.mean(np.sign(x) == np.sign(y))),
        ))
    P = pd.DataFrame(rows)
    P.to_csv(CUR / "condition_pairwise.csv", index=False, encoding="utf-8")

    say(f"共 **{len(P)}** 个条件对（每对都有 ≥10 个共同分子）")
    say("")

    # ── 2×2 分类
    say("## 1. ★ 按「换柱 / 换流动相」分四类")
    say("")
    say("| 类别 | 说明 | 对数 | r 中位 | r 四分位 | ρ 中位 | **符号一致率** |")
    say("|---|---|---|---|---|---|---|")
    classes = [
        (True, False, "**同柱 × 异相**", "只换流动相（选择剂不变）"),
        (False, True, "**异柱 × 同相**", "只换选择剂（流动相不变）"),
        (False, False, "**异柱 × 异相**", "两者都换"),
    ]
    summ = []
    for sc, sp, tag, desc in classes:
        s = P[(P.same_col == sc) & (P.same_ph == sp)]
        if len(s) == 0:
            continue
        q1, q3 = s["r"].quantile(.25), s["r"].quantile(.75)
        row = dict(tag=tag, n=len(s), r_med=float(s["r"].median()),
                   r_q1=float(q1), r_q3=float(q3),
                   rho_med=float(s["rho"].median()),
                   sign=float(s["sign_agree"].median()))
        summ.append(row)
        say(f"| {tag} | {desc} | {len(s)} | **{row['r_med']:+.3f}** | "
            f"{row['r_q1']:+.3f} ~ {row['r_q3']:+.3f} | {row['rho_med']:+.3f} | "
            f"**{row['sign']*100:.1f}%** |")
    say("")
    say("> ⚠️ **「同柱 × 同相」不可能存在**（同一条件只测一次）⇒")
    say("> 本数据**无法估计测量噪声本身**，只能做**条件间**对照。")
    say("")

    # ── 逐柱细分（异柱 × 同相里，看是不是某几根柱特别不一致）
    say("## 2. 「异柱 × 同相」逐柱对细分")
    say("")
    s = P[(~P.same_col) & (P.same_ph)].copy()
    if len(s):
        s["pair"] = s.apply(lambda r: " ↔ ".join(sorted([r["a"].split(" | ")[0],
                                                        r["b"].split(" | ")[0]])), axis=1)
        g = s.groupby("pair").agg(n=("r", "size"), r_med=("r", "median"),
                                  sign=("sign_agree", "median"))
        for pair, row in g.sort_values("r_med").iterrows():
            say(f"- `{pair}`：r 中位 **{row['r_med']:+.3f}**，"
                f"符号一致率 {row['sign']*100:.1f}%（{int(row['n'])} 对）")
    say("")

    # ── 判定
    say("## 3. ★ 判定")
    say("")
    for row in summ:
        lvl = "✅ 可复现" if row["r_med"] > 0.5 else (
              "🟡 弱" if row["r_med"] > 0.2 else "🔴 **不可复现**")
        say(f"- {row['tag']}：r 中位 {row['r_med']:+.3f}、符号一致率 "
            f"{row['sign']*100:.1f}% ⇒ {lvl}")
    say("")
    say("> **读法（必须写清）**：")
    say("> - 「同柱 × 异相」= 选择剂不变、**只换流动相** ⇒ 它给出「**同一根柱、同一实验室、")
    say(">   换个流动相**」的可复现性上界；")
    say("> - 「异柱 × 同相」= 流动相不变、**只换选择剂** ⇒ 它给出「**换一根柱**」的可复现性；")
    say("> - 两者与「异柱 × 异相」一起，构成**条件变化幅度**的梯度。")
    say("")
    say("> ⚠️ 若符号一致率接近 50%，说明**洗脱顺序在条件间根本不可迁移** ——")
    say("> 这与 `scripts/19` 的「跨实验室 r≈0」是**同一现象在纯测量层面的呈现**，")
    say("> 但**不需要任何模型**，因此更难反驳。")
    say("")

    notes = ROOT / "notes" / "条件间可复现性_无模型.md"
    notes.write_text("\n".join(LOG) + "\n", encoding="utf-8")
    print(f"\n[落盘] {notes}")
    print(f"[落盘] data/curated/condition_pairwise.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
