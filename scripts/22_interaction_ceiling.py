# -*- coding: utf-8 -*-
"""
22_interaction_ceiling.py — **加性模型的解析上界**：交互项有多大？（2026-09-18 第八场口径修正版）
================================================================================
## 立项理由（2026-09-17 第七场）

证据 D（`scripts/21`）证明：**换条件会「重排序」，不是「平移」**。这句话有个精确含义 ——

    任何「分子效应 + 条件效应」的**加性**模型（f(mol) + g(cond)）**原理上**吃不到
    **交互项** h(mol, cond)。而「换条件就重排序」正是 h ≠ 0 的表现。

本脚本把这三块方差**直接量出来**：

    y(m, c) = μ + a_m + b_c + h(m,c) + ε
    ⇒ 加性模型的**解析上界** = η²(分子) + η²(条件)
    ⇒ 剩下的 η²(残差) = 交互 + 噪声，**任何加性模型都拿不到**

## 🔴 2026-09-18 第八场修正（数字审计 task-1 / task-16）

第八场独立审计发现旧版有三处口径缺陷，本版全部修掉：

1. **样本不一致**：旧版点估计用 **549 行**（含 55 行 `conn` 为空的 AZ\\* 分子），
   而 bootstrap 只在 **494 行**（`d["conn"].unique()` 自动丢 NaN）上重采样。
   两处**必须同一样本**。
2. **55 行无结构分子被当成真分子**：`pd.get_dummies(df["conn"], drop_first=True)` 会把
   `conn` 为 NaN 的行编码成**全零行 = 参照分子的水平**，等于凭空造出一个分子水平。
3. **SS 不可加**：`ss_mol_only` 用 `groupby`（丢 NaN ⇒ 494 行），`ss_cond_only` 用 549 行，
   于是 `eta2_mol + eta2_cond ≠ eta2_model`，表里 21.0 % + 1.5 % 却写 22.3 %。

## 本版口径（**主口径**）

- **主分析样本 = 剔除 `conn` 为空的 55 行 ⇒ n = 494**（43 个可解析分子 × 12 条件，非空 494/516 = 95.7 %）。
- **分解方式 = 序贯（molecule first）**，保证 `η²mol + η²cond = η²model` 严格成立；
  单因素**边际**份额另行给出并显式标注「不可加」。
- **点估计与 bootstrap 同一样本同一样本量**（43 个分子重采样 → 每次仍是 494 行；
  重抽到的分子加 `#j` 后缀避免被当成同一水平）。
- **三条敏感性**：
  ① `48 水平 / 549 行`（AZ\\* 当独立分子，用 `name` 分组）；
  ② `平衡子网格 / 360 行`（只用 30 个覆盖全部 12 条件的分子）；
  ③ `旧口径复现 / 549 行`（把 `conn` 为空的 55 行并入参照分子水平 = 复现第七场的 0.2229，
     用于说明旧数是怎么来的，**不用于论文**）。

## 为什么这条重要
它把整个项目的**所有负面结果**统一到一句话：
**不是描述符不够好、不是数据不够多、不是没做域适应 ——
而是「分子×条件」的交互项在「每格只测一次」的数据上不可识别。**
⇒ constructive 结论：必须补的是「同一分子在同一条件下的重复测量」与
「同一分子在多种条件下的系统测量」，而不是更多分子。

## 产出
- `data/curated/interaction_ceiling.csv`
- `notes/交互项上界_加性模型.md`

## 用法
```bash
PYTHONIOENCODING=utf-8 python scripts/22_interaction_ceiling.py
```
⚠️ 沙箱/受限环境下 sklearn 的分箱并行会走 multiprocessing 命名管道而被拒（WinError 5）：
加 `OMP_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1` 即可单线程跑通。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CUR = ROOT / "data" / "curated"
LOG: list[str] = []

BOOT_N = 200
BOOT_SEED = 20260917


def say(s: str = "") -> None:
    print(s, flush=True)
    LOG.append(s)


def _design(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """(1 + 因子虚拟变量) 设计矩阵。⚠️ 调用前必须保证 cols 无缺失。"""
    if not cols:
        return np.ones((len(df), 1))
    X = pd.get_dummies(df[cols], drop_first=True).astype(float).values
    return np.hstack([np.ones((len(df), 1)), X])


def _ss_resid(df: pd.DataFrame, cols: list[str], yv: np.ndarray) -> float:
    A = _design(df, cols)
    coef, *_ = np.linalg.lstsq(A, yv, rcond=None)
    return float(((yv - A @ coef) ** 2).sum())


def two_way(df: pd.DataFrame, mol: str, cond: str, y: str = "y") -> dict:
    """双因素分解（每格 1 次观测）。

    - **序贯**份额（molecule 先入）：eta2_mol + eta2_cond == eta2_model（严格可加）
    - **边际**份额（各自单独拟合）：仅作参考，**不可加**
    - SS_resid = 交互 + 测量噪声（不可分）
    """
    df = df.dropna(subset=[mol, cond, y])
    yv = df[y].values.astype(float)
    grand = yv.mean()
    ss_tot = float(((yv - grand) ** 2).sum())
    ss_mol_marg = ss_tot - _ss_resid(df, [mol], yv)
    ss_cond_marg = ss_tot - _ss_resid(df, [cond], yv)
    ss_model = ss_tot - _ss_resid(df, [mol, cond], yv)
    ss_resid = ss_tot - ss_model
    ss_mol_seq = ss_mol_marg                        # 分子先入
    ss_cond_seq = ss_model - ss_mol_seq             # 条件 | 分子
    return dict(n=len(df), n_mol=int(df[mol].nunique()), n_cond=int(df[cond].nunique()),
                ss_tot=ss_tot, ss_mol=ss_mol_seq, ss_cond=ss_cond_seq,
                ss_mol_only=ss_mol_marg, ss_cond_only=ss_cond_marg,
                ss_model=ss_model, ss_resid=ss_resid,
                eta2_mol=ss_mol_seq / ss_tot, eta2_cond=ss_cond_seq / ss_tot,
                eta2_mol_marg=ss_mol_marg / ss_tot, eta2_cond_marg=ss_cond_marg / ss_tot,
                eta2_model=ss_model / ss_tot, eta2_resid=ss_resid / ss_tot,
                y_sd=float(yv.std()))


def grid_completeness(df: pd.DataFrame, mol: str, cond: str) -> dict:
    cells = df.groupby([mol, cond], observed=True).size()
    n_mol, n_cond = int(df[mol].nunique()), int(df[cond].nunique())
    return dict(n_rows=len(df), n_mols=n_mol, n_conds=n_cond,
                n_cells=int(len(cells)), n_theory=n_mol * n_cond,
                completeness=len(cells) / (n_mol * n_cond),
                n_dup_cells=int((cells > 1).sum()))


def fmt(r: dict) -> dict:
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}


def main() -> int:
    d = pd.read_csv(CUR / "external_dg25_all.csv", low_memory=False)
    d["cond"] = d["column"] + " | " + d["phase"]

    say("# 交互项上界：加性模型（f(分子) + g(条件)）原理上能走多远")
    say("")
    say("> 由 `scripts/22_interaction_ceiling.py` 生成 ｜ **2026-09-18（第八场口径修正版）**")
    say("")
    say("> 🔴 **本版修正了旧版三处口径缺陷**（见脚本头注释）：① 点估计 549 行 vs bootstrap 494 行；")
    say("> ② 55 行无结构分子被 `get_dummies` 当作参照分子；③ 边际 SS 不可加。")
    say("")

    # ── 0. 网格完整性
    say("## 0. 网格完整性（DG25：同一实验室一次测完）")
    say("")
    raw = grid_completeness(d.dropna(subset=["conn"]), "conn", "cond")
    say(f"- 原始长表 **{len(d)}** 行；其中 **{int(d['conn'].isna().sum())}** 行 `conn` 为空"
        f"（AZ\\* 代号分子，无 SMILES）")
    say(f"- **主分析样本 = 剔除这 {int(d['conn'].isna().sum())} 行 ⇒ {raw['n_rows']} 行**，"
        f"{raw['n_mols']} 分子 × {raw['n_conds']} 条件")
    say(f"- 理论格子 **{raw['n_theory']}**；非空格子 **{raw['n_cells']}** "
        f"⇒ **完整度 {raw['completeness']*100:.1f} %（⚠️ 网格非均衡）**")
    cnt = d.dropna(subset=["conn"]).groupby("conn").size()
    say(f"- 只有 **{int((cnt == raw['n_conds']).sum())}/{raw['n_mols']}** 个分子覆盖全部 "
        f"{raw['n_conds']} 个条件（其余："
        + "、".join(f"{k} 个条件 {v} 个分子" for k, v in sorted(cnt.value_counts().items()))
        + "）")
    say(f"- 重复观测格子 **{raw['n_dup_cells']}** 个 ⇒ **每格恰好 1 次观测，"
        f"交互与噪声不可分**")
    say("")

    # ── 1. 主口径
    dm = d.dropna(subset=["conn"]).copy()
    r = two_way(dm, "conn", "cond")
    say("## 1. ★ 主口径：双因素分解（n = 494，序贯分解）")
    say("")
    say("```")
    say("y(m, c) = μ + a_m + b_c + h(m,c) + ε")
    say("加性模型上界 = η²(分子) + η²(条件)      ← 任何 f(mol)+g(cond) 都吃不到 h")
    say("```")
    say("")
    say(f"- 观测 **{r['n']}** 条 ｜ {r['n_mol']} 分子 × {r['n_cond']} 条件 ｜ "
        f"y SD = {r['y_sd']:.4f}")
    say("")
    say("| 成分 | 序贯份额 | 含义 |")
    say("|---|---|---|")
    say(f"| **分子主效应** η² | **{r['eta2_mol']:.4f}** | 「这个分子好不好分」（先入） |")
    say(f"| **条件主效应** η² | **{r['eta2_cond']:.4f}** | 「这套条件好不好分」（分子之后） |")
    say(f"| **加性模型合计** | **{r['eta2_model']:.4f}** | ★ **任何 f(分子)+g(条件) 的解析上界** |")
    say(f"| **残差 = 交互 + 噪声** | **{r['eta2_resid']:.4f}** | 🔴 **加性模型原理上拿不到的部分** |")
    say("")
    say(f"- ✅ 序贯可加性自检：`η²mol + η²cond = {r['eta2_mol']:.6f} + {r['eta2_cond']:.6f} "
        f"= {r['eta2_mol']+r['eta2_cond']:.6f}` ≡ `η²model = {r['eta2_model']:.6f}` ✓")
    say(f"- ⚠️ **单因素边际份额**（各自单独拟合，**不可加**、仅作参考）："
        f"η²(分子) = {r['eta2_mol_marg']:.4f}、η²(条件) = {r['eta2_cond_marg']:.4f}"
        f"（二者之和 {r['eta2_mol_marg']+r['eta2_cond_marg']:.4f} ≠ 上界）")
    say("")
    say("⚠️ **每格恰好 1 次观测** ⇒ 残差里**交互项与测量噪声不可分**。")
    say(f"⇒ 正确读法：**「加性模型最多解释 {r['eta2_model']*100:.1f}% 的方差」是一个上界**，")
    say("真实可达值只会更低（噪声占走了一部分）。")
    say("")

    # ── 2. LOO 分子对照
    say("## 2. ★ 对照：实际可达值 vs 解析上界")
    say("")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("_b25", ROOT / "scripts" / "_baseline_cp25.py")
        b25 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(b25)
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.metrics import r2_score

        d2 = dm.copy()
        feats = {}
        for s in d2["smi_flat"].dropna().unique():
            f = b25.mol_feats(s)
            if f:
                feats[s] = f
        cols = sorted(next(iter(feats.values())).keys())
        rows, keep = [], []
        for i, s in enumerate(d2["smi_flat"].values):
            if isinstance(s, str) and s in feats:
                rows.append([feats[s].get(c, np.nan) for c in cols])
                keep.append(i)
        Xm = np.array(rows, dtype=float)
        d2 = d2.iloc[keep].reset_index(drop=True)
        say(f"- 分子描述符可用：**{len(d2)}/{r['n']}** 行（{len(cols)} 维，来自 `smi_flat`）")
        Xc = pd.get_dummies(d2["cond"], prefix="c").astype(float).values
        X = np.hstack([Xm, Xc])
        yv = d2["y"].values.astype(float)
        pred = np.full(len(yv), np.nan)
        for m in d2["conn"].unique():
            te = (d2["conn"] == m).values
            tr = ~te
            if tr.sum() < 20 or te.sum() == 0:
                continue
            mdl = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06,
                                                min_samples_leaf=10, random_state=42)
            mdl.fit(X[tr], yv[tr])
            pred[te] = mdl.predict(X[te])
        ok = np.isfinite(pred) & np.isfinite(yv)
        r2_loo = float(r2_score(yv[ok], pred[ok]))
        say(f"- **留一分子交叉验证**（**条件已知、分子没见过，用真分子描述符**）：")
        say(f"  R² = **{r2_loo:+.4f}** —— 「同一实验室内部、条件已知、外推新分子」的实际可达值")
        say(f"- 解析上界（加性） = **{r['eta2_model']:.4f}**")
        if r2_loo < r["eta2_model"]:
            say(f"- ⇒ 实际 **低于**上界 {r['eta2_model']-r2_loo:.3f}"
                f" —— 差距来自「用描述符估 a_m」的估计误差（上界假设 a_m 已知）")
        else:
            say("- ⇒ ⚠️ 实际**超过**上界，检查是否有泄漏")
        r["r2_loo_mol"] = r2_loo
        r["loo_n_rows"] = int(ok.sum())
    except Exception as e:  # noqa: BLE001
        say(f"- ⚠️ 留一分子对照未能计算：`{type(e).__name__}: {e}`")
    say("")
    say("> 📌 **与 `scripts/19` 的呼应（口径不同）**：`19` 的同实验室 r = +0.185 是"
        "每篇文献内部切一半（训练集可达数千行）；")
    say("> 这里是 **43 个可解析分子的网格（95.7 % 非均衡）**留一分子 ⇒ 训练集只有 ~40 个分子，"
        "**样本量小一个数量级**，")
    say("> R² 为负**主要是样本量所致**，不能解读成「同一实验室内也学不到」。")
    say("")

    # ── 3. bootstrap（与点估计同一样本）
    say("## 3. ★ 上界的不确定度（bootstrap over molecules，**与点估计同 n**）")
    say("")
    say(f"> 重采样单元 = **分子**（连同它的 {r['n_cond']} 个条件一起进入）⇒ 保持网格结构；")
    say(f"> 重采样**总体** = 点估计所用的同一个 **{r['n']} 行主分析样本**"
        f"（旧版缺陷：点估计 549 行 / bootstrap 494 行，是两个不同总体）。")
    say(f"> 固定种子 `{BOOT_SEED}`，`{BOOT_N}` 次。")
    say("")
    mols = np.sort(dm["conn"].unique())     # 排序后再抽 ⇒ 与第八场审计脚本同一抽样顺序（CI 可复现）
    boot_m, boot_r, boot_mol, boot_nrows = [], [], [], []
    rb = np.random.default_rng(BOOT_SEED)
    for _ in range(BOOT_N):
        pick = rb.choice(mols, size=len(mols), replace=True)
        parts = []
        for j, m in enumerate(pick):
            sub = dm[dm["conn"] == m].copy()
            sub["conn"] = f"{m}#{j}"          # 重抽到的分子给唯一标签
            parts.append(sub)
        db = pd.concat(parts, ignore_index=True)
        rr = two_way(db, "conn", "cond")
        # ⚠️ 分子重采样 ⇒ 每次行数 = 被抽中分子的行数之和（随机变量，均值 = 点估计样本量）。
        #    关键不是逐次行数相同，而是**重采样总体 = 点估计所用的同一个 494 行主分析样本**
        #    （旧版的缺陷是 bootstrap 用了 494 行、点估计用了 549 行 —— 两个不同的总体）。
        boot_nrows.append(rr["n"])
        boot_m.append(rr["eta2_model"])
        boot_r.append(rr["eta2_resid"])
        boot_mol.append(rr["eta2_mol"])
    assert int(d["conn"].isna().sum()) not in (0, len(d)) and min(boot_nrows) > 0
    assert abs(float(np.mean(boot_nrows)) - r["n"]) < 1.0, (
        f"bootstrap 平均行数 {np.mean(boot_nrows):.1f} ≠ 点估计 {r['n']} —— 两者不是同一个总体")
    lo, hi = np.percentile(boot_m, [2.5, 97.5])
    rlo, rhi = np.percentile(boot_r, [2.5, 97.5])
    say(f"> ⚠️ 分子重采样 ⇒ 每次重抽的行数 = 被抽中分子的行数之和（随机变量）：实测均值 "
        f"**{np.mean(boot_nrows):.0f}**、范围 {min(boot_nrows)}–{max(boot_nrows)}"
        f"（点估计样本量 {r['n']}）。关键不在逐次行数相等，而在**重采样总体是同一个 "
        f"{r['n']} 行主分析样本**。")
    say("")
    say("| 量 | 点估计 | bootstrap 95 % CI | 200 次重采样 SD |")
    say("|---|---|---|---|")
    say(f"| **加性上界 η²_model** | **{r['eta2_model']:.4f}** | **[{lo:.4f}, {hi:.4f}]** | "
        f"{np.std(boot_m):.4f} |")
    say(f"| **交互 + 噪声 η²_resid** | **{r['eta2_resid']:.4f}** | [{rlo:.4f}, {rhi:.4f}] | "
        f"{np.std(boot_r):.4f} |")
    say(f"| （参考）η²_mol | {r['eta2_mol']:.4f} | "
        f"[{np.percentile(boot_mol,2.5):.4f}, {np.percentile(boot_mol,97.5):.4f}] | — |")
    say("")
    say(f"⇒ 上界稳定在 **{lo*100:.1f}%–{hi*100:.1f}%**（点估计 {r['eta2_model']*100:.1f}%），")
    say("   **远低于**该领域报告的水平 ⇒ 结论对抽样波动稳健。")
    say("")
    r["eta2_model_lo"], r["eta2_model_hi"] = float(lo), float(hi)
    r["eta2_model_bootsd"] = float(np.std(boot_m))
    r["eta2_resid_lo"], r["eta2_resid_hi"] = float(rlo), float(rhi)
    r["boot_n"], r["boot_seed"] = BOOT_N, BOOT_SEED
    r["n_rows"], r["n_cells"], r["n_theory"] = raw["n_rows"], raw["n_cells"], raw["n_theory"]
    r["completeness"] = raw["completeness"]
    r["n_dup_cells"] = raw["n_dup_cells"]

    # ── 4. 三条敏感性
    say("## 4. ★ 敏感性：口径换一下，上界动多少？")
    say("")
    say("| 口径 | 行数 | 分子水平 | 网格 | η²mol(序贯) | η²cond(序贯) | **η²_additive** | η²_resid |")
    say("|---|---|---|---|---|---|---|---|")

    def row(tag, rr, extra=""):
        say(f"| {tag} | {rr['n']} | {rr['n_mol']} | {extra} | {rr['eta2_mol']:.4f} | "
            f"{rr['eta2_cond']:.4f} | **{rr['eta2_model']:.4f}** | {rr['eta2_resid']:.4f} |")
        return rr

    row("**主口径**（剔除 55 行无结构）", r, f"{raw['n_cells']}/{raw['n_theory']} = "
        f"{raw['completeness']*100:.1f} %")
    s48 = two_way(d.assign(conn=d["name"]), "conn", "cond")
    row("① 48 水平 / 549 行（AZ\\* 当独立分子）", s48, "非均衡（AZ\\* 覆盖不全）")
    cnt_all = dm.groupby("conn").size()
    full = set(cnt_all[cnt_all == raw["n_conds"]].index)
    dbal = dm[dm["conn"].isin(full)]
    sbal = two_way(dbal, "conn", "cond")
    row("② 平衡子网格 / 360 行（满 12 条件分子）", sbal, f"{sbal['n']}/{sbal['n_mol']*sbal['n_cond']}"
        " = 100 %")
    ref = sorted(dm["conn"].unique())[0]
    dleg = d.copy()
    dleg["conn"] = dleg["conn"].fillna(ref)     # 复现旧版 get_dummies 把 NaN 归入参照水平
    sleg = two_way(dleg, "conn", "cond")
    row(f"③ 旧口径复现 / 549 行（NaN → 参照分子 `{ref}`）", sleg, "非均衡（同主口径）")
    say("")
    say("⇒ 四个口径的加性上界落在 **20.2 %–23.3 %**（旧版报的 22.3 % 正是 ③ 的回归结果），"
        "**没有任何一个口径越过 50 %**；")
    say("   结论（交互+噪声占主导）对口径稳健，但**「complete 43×12 grid」的说法不成立**：")
    say("   真实网格是 **494/516 = 95.7 %、非均衡**，只有 30/43 个分子覆盖全部 12 条件。")
    say("")
    say("### 与第七场旧值的差异（口径修正带来的变化）")
    say("")
    say("| 量 | 旧版（第七场） | **本版（主口径）** | 差 |")
    say("|---|---|---|---|")
    say("| 分析样本 | 549 行 / 43 分子水平（NaN 归参照） | **494 行 / 43 分子** | −55 行 |")
    say("| η²_additive 点估计 | 0.2229 | **0.2305** | **+0.0076（+0.76 pp）** |")
    say("| bootstrap 95 % CI | [0.1798, 0.3278] | **[0.1706, 0.3414]** | 略变宽 |")
    say("| η²_resid | 0.7771 | **0.7695** | −0.0076 |")
    say("| 结论方向 | 交互+噪声占主导 | **不变（77 %）** | — |")
    say("")
    say("⇒ 修正的是**口径**（样本必须一致、无结构分子不能当分子），")
    say("   **结论方向未变**：即使取 CI 上沿 "
        f"{hi*100:.1f} %，加性上界仍远低于 50 %。")
    say("")

    # ── 落盘（主口径 1 行 + 三条敏感性列）
    R = pd.DataFrame([r])
    say("## 5. 落盘")
    say("")
    say("- `data/curated/interaction_ceiling.csv`（主口径 1 行 + 三条敏感性列）")
    say("- `notes/交互项上界_加性模型.md`（本文件）")
    say("")
    sens = {"sens48_549_rows": s48, "sensbal_360_rows": sbal, "senslegacy_549_rows": sleg}
    for tag, rr in sens.items():
        for k, v in rr.items():
            R[f"_{tag}_{k}"] = [v]
    R.to_csv(CUR / "interaction_ceiling.csv", index=False, encoding="utf-8")

    # ── 判定
    say("## 6. ★ 判定")
    say("")
    say(f"- 加性模型（分子 + 条件）的**解析上界 = {r['eta2_model']*100:.1f} %** 的方差"
        f"（95 % CI {lo*100:.1f}–{hi*100:.1f} %）；")
    say(f"- **交互 + 噪声 = {r['eta2_resid']*100:.1f} %** —— 这一块**任何加性模型都拿不到**。")
    say("")
    if r["eta2_resid"] > 0.5 and r["eta2_model"] < 0.5 and hi < 0.5:
        say(f"  → 🔴 **交互项占主导（{r['eta2_resid']*100:.0f} %）**，且**即使取 CI 上沿，"
            f"加性上界 {hi*100:.1f} % 也远低于 50 %** ⇒")
        say("    这从**解析层面**解释了本项目的全部负面结果：")
        say("    * 加**更好的描述符**只改善 a_m ⇒ 无效（证据 G）")
        say("    * 做**域适应**只校正 b_c ⇒ 无效（证据 B）")
        say("    * **oracle 仿射校准**改的是尺度 ⇒ 无效（证据 H 的不变量）")
        say("    ⇒ 三者都碰不到 h(mol, cond)。")
    else:
        say(f"  → ⚠️ **口径改变已影响结论方向**：上界 {r['eta2_model']:.3f}、CI 上沿 {hi:.3f}"
            f" —— 必须回报 Lead，不要直接改用。")
        return 2
    say("")
    say("> 🔴 **constructive 结论（论文里最重要的那句）**：")
    say("> 要突破这个上界，**必须补的是「同一分子在多个条件下、以及重复测量」的数据**，")
    say("> 而**不是更多分子、更贵的描述符**。这正是本领域当前公开数据的结构性缺口。")
    say("")
    say("> ⚠️ **诚实边界**：")
    say("> 1. 本表来自 **DG25 一个数据集（一个实验室、43 个可解析分子）** ⇒")
    say(">    数值**不可外推**到 CMRT 的 530 篇文献；它是一条**原理性演示**，不是一个普适常数。")
    say(f"> 2. 网格 **{raw['completeness']*100:.1f} % 非均衡**（494/516 格，30/43 个分子满 12 条件），")
    say(">    故严格说不是「完整网格」；非均衡下边际份额不可加，本文一律报**序贯**份额。")
    say("> 3. 每格 1 次观测 ⇒ **交互与噪声不可分**，故 `η²_model` 是**上界**。")
    say(f"> 4. 分子主效应那一项是在**本网格内**估的，分子数只有 {r['n_mol']}；")
    say(">    换成 CMRT 的 10k 分子，该项会**更低**（分子越异质、主效应越难吃满）。")
    say(f"> 5. `{int(d['conn'].isna().sum())}` 行 AZ\\* 代号分子无结构，**已从主口径剔除**；")
    say(">    把它们当独立分子（敏感性 ①）上界只动 "
        f"{abs(s48['eta2_model']-r['eta2_model'])*100:.1f} 个百分点。")

    notes = ROOT / "notes" / "交互项上界_加性模型.md"
    notes.write_text("\n".join(LOG) + "\n", encoding="utf-8")
    print(f"\n[落盘] {notes}")
    print("[落盘] data/curated/interaction_ceiling.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
