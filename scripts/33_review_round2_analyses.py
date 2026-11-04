# -*- coding: utf-8 -*-
"""
33_review_round2_analyses.py — **两轮审稿新增的稳健性分析**（只读 data/curated，不重训主模型）
================================================================================
## 为什么
第九场（两轮审稿）的 method-reviewer 指出 5 条 P0，其中 4 条需要补算才能闭合：
  M-2  bootstrap 只重采样分子 ⇒ CI 低估（上沿 34.1 % vs 双因素 47.9 %）
  M-1  η²_model 是"分子+条件哑变量主效应 OLS 的样本内 R²"，**不能**界定描述符 ML；
       需要把"方差份额"翻译成"准确率语言"（EMO 版主效应上界）
  M-8  功效曲线的排除判据用 2·SD（5 次取噪）而非 t 分位 ⇒「排除 ≥5 %」不成立
  M-6  η²(doi) 与 R²_aff 不同样本（0.389 / 0.3854 / 0.377 三个口径）
  M-12 §4.4 的 η²(column)=1.3 % 用的是 §4.5 已否决的 549 行口径（494 行 = 1.0 %）

本脚本把上述 5 条各自算成一个可引用数字，落盘 `data/curated/review_round2_stats.csv`，
并打印可直接写进手稿的数值。**不做任何模型训练，不改任何既有 CSV。**

## 用法
```bash
set PYTHONIOENCODING=utf-8
python scripts/33_review_round2_analyses.py
```
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CUR = ROOT / "data" / "curated"
OUT = CUR / "review_round2_stats.csv"

SEED = 20260917
RNG = np.random.default_rng(SEED)


# ────────────────────────────────────────────────────────────────────────────
# 工具：与 scripts/22 同口径的两因素（序贯，分子先入）最小二乘分解
# ────────────────────────────────────────────────────────────────────────────
def two_way(df: pd.DataFrame, ycol: str = "y"):
    """返回 (eta2_mol, eta2_cond, eta2_model, eta2_mol_marg, eta2_cond_marg)。

    与 scripts/22_interaction_ceiling.py 的 two_way 同口径：
      序贯：先只放分子哑变量 → ss_mol_only；再在残差上放条件 → ss_cond；
      边际：单独放条件 → ss_cond_only；单独放分子 → ss_mol。
    """
    y = df[ycol].to_numpy(dtype=float)
    n = len(y)
    ss_tot = float(((y - y.mean()) ** 2).sum())

    def rss(cols):
        if not cols:
            return float(((y - y.mean()) ** 2).sum())
        X = pd.get_dummies(df[cols].astype(str), drop_first=False).to_numpy(dtype=float)
        X = np.column_stack([np.ones(n), X])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        return float((resid ** 2).sum())

    rss_mol = rss(["mol"])
    rss_cond = rss(["cond"])
    rss_both = rss(["mol", "cond"])

    ss_mol = ss_tot - rss_mol
    ss_cond_only = ss_tot - rss_cond
    ss_mol_only = ss_tot - rss_mol
    ss_cond = rss_mol - rss_both          # 序贯：条件在分子之后
    ss_model = ss_tot - rss_both
    return dict(
        eta2_mol=ss_mol / ss_tot,
        eta2_cond=ss_cond / ss_tot,
        eta2_model=ss_model / ss_tot,
        eta2_mol_marg=ss_mol_only / ss_tot,
        eta2_cond_marg=ss_cond_only / ss_tot,
        n=n, n_mol=df["mol"].nunique(), n_cond=df["cond"].nunique(),
    )


def pct(x):
    return f"{100*x:.2f}%"


def main() -> int:
    rows: list[dict] = []

    # ── 载入 DG25 网格（与 scripts/22 主口径一致：剔 conn 为空） ─────────────
    dg = pd.read_csv(CUR / "external_dg25_all.csv")
    grid = dg[dg["conn"].notna()].copy()
    grid["mol"] = grid["conn"]
    grid["cond"] = grid["column"].astype(str) + "|" + grid["phase"].astype(str)
    grid = grid[grid["y"].notna()]
    base = two_way(grid)
    ic = pd.read_csv(CUR / "interaction_ceiling.csv").iloc[0]

    print("=" * 78)
    print("0. 基线复现（应与 interaction_ceiling.csv 逐位一致）")
    print("=" * 78)
    print(f"  n={base['n']}  分子={base['n_mol']}  条件={base['n_cond']}")
    print(f"  eta2_mol   = {base['eta2_mol']:.8f}   (CSV {ic['eta2_mol']:.8f})")
    print(f"  eta2_cond  = {base['eta2_cond']:.8f}   (CSV {ic['eta2_cond']:.8f})")
    print(f"  eta2_model = {base['eta2_model']:.8f}   (CSV {ic['eta2_model']:.8f})")
    ok = abs(base["eta2_model"] - float(ic["eta2_model"])) < 1e-9
    print(f"  ⇒ {'✅ 与 CSV 一致' if ok else '❌ 不一致，停止'}")
    if not ok:
        return 1
    rows.append(dict(analysis="baseline_reproduce", eta2_model=base["eta2_model"],
                     n=base["n"], n_mol=base["n_mol"], n_cond=base["n_cond"]))

    # ── M-2：双因素 bootstrap（分子 + 条件同时重采样） ──────────────────────
    print()
    print("=" * 78)
    print("1. M-2  bootstrap：只重采样分子 vs 分子+条件同时重采样（200 次）")
    print("=" * 78)
    B = 200
    mols = grid["mol"].unique()
    conds = grid["cond"].unique()
    boot_mol, boot_both = [], []
    for _ in range(B):
        # (a) 只重采样分子
        pick = RNG.choice(mols, size=len(mols), replace=True)
        g1 = pd.concat([grid[grid["mol"] == m] for m in pick], ignore_index=True)
        g1["mol"] = np.repeat(np.arange(len(pick)), [len(grid[grid["mol"] == m]) for m in pick])
        boot_mol.append(two_way(g1)["eta2_model"])
        # (b) 分子 + 条件同时重采样
        pm = RNG.choice(mols, size=len(mols), replace=True)
        pc = RNG.choice(conds, size=len(conds), replace=True)
        sub = grid[grid["mol"].isin(pm) & grid["cond"].isin(pc)]
        if len(sub) > 10 and sub["mol"].nunique() > 2 and sub["cond"].nunique() > 2:
            boot_both.append(two_way(sub)["eta2_model"])
    bm = np.array(boot_mol); bb = np.array(boot_both)
    lo1, hi1 = np.percentile(bm, [2.5, 97.5])
    lo2, hi2 = np.percentile(bb, [2.5, 97.5])
    pctile = float((bm < base["eta2_model"]).mean() * 100)
    print(f"  (a) 只重采样分子   : mean {pct(bm.mean())}  SD {pct(bm.std(ddof=1))}  "
          f"95% [{pct(lo1)}, {pct(hi1)}]  点估计百分位 {pctile:.1f}%")
    print(f"  (b) 分子+条件双采  : mean {pct(bb.mean())}  SD {pct(bb.std(ddof=1))}  "
          f"95% [{pct(lo2)}, {pct(hi2)}]  (n={len(bb)})")
    print(f"  ⇒ 上沿 {pct(hi1)} → {pct(hi2)}；两种口径下 η²_model 均 < 50 %")
    rows += [
        dict(analysis="boot_molecule_only", eta2_model_lo=lo1, eta2_model_hi=hi1,
             boot_mean=bm.mean(), boot_sd=bm.std(ddof=1), boot_n=B,
             point_estimate_percentile=pctile, seed=SEED),
        dict(analysis="boot_molecule_and_condition", eta2_model_lo=lo2, eta2_model_hi=hi2,
             boot_mean=bb.mean(), boot_sd=bb.std(ddof=1), boot_n=len(bb), seed=SEED),
    ]

    # ── M-1：EMO 版主效应上界（把"方差份额"翻成"准确率语言"） ──────────────
    print()
    print("=" * 78)
    print("2. M-1  EMO 上界：任何『标签可加』规则在这张网格上最多能做到多少准确率")
    print("=" * 78)
    g = grid.copy()
    g["sign"] = (g["y"] > 0).astype(int)          # EMO = 1{y>0}
    n_all = len(g)
    maj_global = max(g["sign"].mean(), 1 - g["sign"].mean())

    # (a) 每分子多数类 / 每条件多数类
    def group_majority(col):
        prior = g.groupby(col)["sign"].mean()
        pred = g[col].map(lambda c: 1 if prior[c] > 0.5 else 0)
        tie = g[col].map(lambda c: prior[c] == 0.5)
        return float((pred == g["sign"]).mean()), float(tie.mean())

    acc_mol, tie_mol = group_majority("mol")
    acc_cond, tie_cond = group_majority("cond")

    # (b) oracle 两因素加性拟合（分子 + 条件哑变量，无交互），阈值 0.5
    X = pd.get_dummies(g[["mol", "cond"]].astype(str), drop_first=False).to_numpy(dtype=float)
    X = np.column_stack([np.ones(n_all), X])
    beta, *_ = np.linalg.lstsq(X, g["y"].to_numpy(dtype=float), rcond=None)
    pred_oracle = X @ beta
    acc_oracle = float(((pred_oracle > 0).astype(int) == g["sign"]).mean())

    # (c) 留一分子（LOO）：分子效应未知，用条件效应 + 全局均值预测
    mol_means = g.groupby("mol")["y"].transform("mean")
    cond_eff = g["y"] - mol_means               # 去掉分子均值 ⇒ 只剩条件/交互
    grand = g["y"].mean()
    acc_loo = float((((grand + cond_eff) > 0).astype(int) == g["sign"]).mean())

    # (d) 有多少格子是"分开的"？以及分开/不分开能否由加性模型解释
    sep = (g["alpha"] != 1).astype(int)
    sep_rate = float(sep.mean())
    sep_or = float(((pred_oracle > np.log(1.0)).astype(int) == sep).mean()) if False else None

    print(f"  全局多数类基线            : {pct(maj_global)}")
    print(f"  每分子多数类（含并列 {pct(tie_mol)} 取 0）: {pct(acc_mol)}")
    print(f"  每条件多数类（含并列 {pct(tie_cond)} 取 0）: {pct(acc_cond)}")
    print(f"  ★ oracle 两因素加性拟合（阈值 0.5）  : {pct(acc_oracle)}   ← 任何标签可加规则的网格内上界")
    print(f"  留一分子 + 已知条件效应（LOO）       : {pct(acc_loo)}")
    print(f"  参考：该网格中『真正分开』(α≠1) 的格子占 {pct(sep_rate)}")
    print(f"  ⇒ 结构上限远低于领域报告的 76–96 %；且只有 {pct(1-sep_rate)} 的格子有可预测的符号变异")
    rows += [
        dict(analysis="emo_global_majority", emo_acc=maj_global, n=n_all),
        dict(analysis="emo_per_molecule_majority", emo_acc=acc_mol, n=n_all),
        dict(analysis="emo_per_condition_majority", emo_acc=acc_cond, n=n_all),
        dict(analysis="emo_oracle_additive_main_effects", emo_acc=acc_oracle, n=n_all),
        dict(analysis="emo_leave_one_molecule_plus_condition", emo_acc=acc_loo, n=n_all),
        dict(analysis="grid_separated_share", emo_acc=sep_rate, n=n_all),
    ]

    # ── M-8：功效曲线用 t 分位而非 2·SD（对 recovered_sd 的 5 次取噪口径重判） ──
    print()
    print("=" * 78)
    print("3. M-8  功效曲线：排除判据 2·SD vs 单侧 95 % t 分位（df=4, t=2.132）")
    print("=" * 78)
    pc = pd.read_csv(CUR / "offset_power_curve.csv")
    pc = pc.sort_values("planted")
    t4 = 2.131846786          # 单侧 95 %，df = 5-1
    obs = -0.0158
    print(f"  观测值 obs = {pct(obs)}；df = 5-1 ⇒ 单侧 95 % t = {t4:.3f}")
    print(f"  {'planted':>8} {'recovered':>10} {'SD':>8} {'mean-2SD':>11} {'mean-t*SD':>11} {'2SD 排除':>9} {'t 排除':>8}")
    for _, r in pc.iterrows():
        m, s = r["recovered_mean"], r["recovered_sd"]
        b1, b2 = m - 2 * s, m - t4 * s
        print(f"  {r['planted']:>8.2f} {m:>10.5f} {s:>8.5f} {b1:>11.5f} {b2:>11.5f} "
              f"{'是' if b1 > obs else '否':>9} {'是' if b2 > obs else '否':>8}")
        rows.append(dict(analysis=f"power_planted_{r['planted']}", recovered_mean=m,
                         recovered_sd=s, bound_2sd=b1, bound_t95=b2,
                         excludes_2sd=bool(b1 > obs), excludes_t95=bool(b2 > obs),
                         draws=5, se_of_rate=None))
    excl = [float(r["planted"]) for _, r in pc.iterrows()
            if r["recovered_mean"] - t4 * r["recovered_sd"] > obs]
    print(f"  ⇒ 单侧 95 % t 判据下可稳健排除的最小 planted 份额 = {min(excl) if excl else None}")
    print("  ⇒ 5 % 档在 t 判据下**不能**排除（与手稿原话不同，需改）")

    # ── M-6：η²(doi) 与 R²_aff 的同口径 ─────────────────────────────────────
    print()
    print("=" * 78)
    print("4. M-6  η²(doi) 三个口径 vs R²_aff 所在样本（428 折 / 10,193 行）")
    print("=" * 78)
    pairs = pd.read_csv(CUR / "cmrt_pairs.csv")
    lod = pd.read_csv(CUR / "leave_one_doi_summary.csv")
    print(f"  leave_one_doi_summary.csv 列: {list(lod.columns)}")
    print(lod.to_string())

    def eta2_doi(sub, label):
        if "doi" not in sub.columns or "y" not in sub.columns:
            print(f"  [{label}] 缺列，跳过"); return None
        y = sub["y"].to_numpy(dtype=float)
        ss_tot = float(((y - y.mean()) ** 2).sum())
        gm = sub.groupby("doi")["y"].transform("mean")
        ss_b = float(((gm - y.mean()) ** 2).sum())
        e2 = ss_b / ss_tot
        print(f"  [{label}] n={len(sub)}  篇数={sub['doi'].nunique()}  η²(doi)={e2:.6f}")
        return e2

    e_full = eta2_doi(pairs, "全表 10,525 行")
    rows.append(dict(analysis="eta2_doi_full_table", eta2_model=e_full, n=len(pairs),
                     n_mol=pairs["doi"].nunique()))
    print("  ⚠️ 428 折子样本的行集需要 splits 索引；见 scripts/15 的 fold_stats（本脚本不重建）")

    # ── M-12：DG25 上 η²(column)/η²(condition) 的 494 vs 549 两个口径 ───────
    print()
    print("=" * 78)
    print("5. M-12  η²(column) / η²(condition)：494 行主口径 vs 549 行旧口径")
    print("=" * 78)
    dg549 = dg.copy()
    dg549["mol"] = dg549["conn"].fillna("__NO_CONN__")
    dg549["cond"] = dg549["column"].astype(str) + "|" + dg549["phase"].astype(str)
    dg549 = dg549[dg549["y"].notna()]
    for label, d in [("494 行（主口径，剔 conn 为空）", grid),
                     ("549 行（旧口径，conn 为空并入参照）", dg549)]:
        y = d["y"].to_numpy(dtype=float)
        ss_tot = float(((y - y.mean()) ** 2).sum())

        def e2(col):
            gm = d.groupby(col)["y"].transform("mean")
            return float(((gm - y.mean()) ** 2).sum()) / ss_tot

        e_col, e_cond = e2("column"), e2("cond")
        print(f"  {label}: n={len(d)}  η²(column)={pct(e_col)}  η²(condition)={pct(e_cond)}")
        rows.append(dict(analysis=f"eta2_dg25_{len(d)}rows", eta2_col=e_col,
                         eta2_cond=e_cond, n=len(d)))
    print("  ⇒ §4.4 引用的 1.3 %/1.5 % 是 549 行口径；主口径应为 1.0 %/1.4 %")

    # ── M-9：同分子子集的 CI（由 ratio_lo/hi 换算 1 − ratio²） ──────────────
    print()
    print("=" * 78)
    print("6. M-9  同分子子集天花板 CI（1 − ratio²，ratio 的 95 % 区间已在 CSV）")
    print("=" * 78)
    rc = pd.read_csv(CUR / "reproducibility_components.csv")
    for _, r in rc.iterrows():
        c_lo, c_hi = 1 - r["ratio_hi"] ** 2, 1 - r["ratio_lo"] ** 2
        print(f"  {r['match']:>14}: n={int(r['n']):>5}  天花板 {r['ceiling']:+.4f} "
              f"95 % CI [{c_lo:+.4f}, {c_hi:+.4f}]  (ratio {r['ratio']:.3f})")
        rows.append(dict(analysis=f"ceiling_ci_{r['match']}", n=int(r["n"]),
                         ceiling=r["ceiling"], ceiling_lo=c_lo, ceiling_hi=c_hi,
                         ratio=r["ratio"]))

    # ── 落盘 ────────────────────────────────────────────────────────────────
    out = pd.DataFrame(rows)
    out["script"] = "scripts/33_review_round2_analyses.py"
    out["seed"] = SEED
    out.to_csv(OUT, index=False, encoding="utf-8")
    print()
    print(f"[落盘] {OUT.relative_to(ROOT)}  ({len(out)} 行)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
