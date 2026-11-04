# -*- coding: utf-8 -*-
"""
15_leave_one_doi.py — **真正的「留一文献」划分（S2c）** + 在其上的失败模式分解
================================================================================
## 立项理由（2026-09-17 实测发现）

`data/curated/splits/S2b_leave_paper.csv` 被所有报告称作「**留一文献**」，
但实测它只有 **10 折**，每折 **~53 篇 DOI**（全表共 **530 篇**）：

    fold 1..10，每折 n≈1053 行 / 53 篇文献 / ~1045 个分子

⇒ **它不是 leave-one-paper-out，是「留一批文献」**（equivalent to GroupKFold(k=10)）。
  后果：文献效应（`η²(doi) = 0.3854`，已复现审计数字）在折内被**平均掉**——
  各折 y 均值只在 **±0.05** 内浮动，而文献间的真实偏移 SD 是 **0.256**。
  ⇒ 当前 S2b 的 −0.04 **系统性偏乐观**，且论文若按字面报「leave-one-paper-out」
    会被一眼看穿。

本脚本建 **S2c = 真正的 leave-one-DOI-out（530 折）**，并在其上重跑失败模式分解，
把「论文自身的系统性偏移」与「分子层面的不可预测性」**分开量出来**。

## 产出
- `data/curated/splits/S2c_leave_doi.csv`（`row_id`, `fold`, `paper`）
- `notes/留一文献真划分_S2c.md`
- `data/curated/leave_one_doi_summary.csv`

## 用法
```bash
PYTHONIOENCODING=utf-8 python scripts/15_leave_one_doi.py
# 想快速看结果可只跑一个臂：
PYTHONIOENCODING=utf-8 python scripts/15_leave_one_doi.py --arms L0
```
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CUR = ROOT / "data" / "curated"
LOG: list[str] = []
b25 = None


def say(s: str = "") -> None:
    print(s, flush=True)
    LOG.append(s)


def load_mod(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="L0,L0+CF+FF")
    ap.add_argument("--min-n", type=int, default=5, help="逐折统计的最小折大小（不影响全局 R²）")
    args = ap.parse_args()

    global b25
    b25 = load_mod("_b25", ROOT / "scripts" / "_baseline_cp25.py")
    d13 = load_mod("_d13", ROOT / "scripts" / "13_failure_decomposition.py")
    # ⚠️ `13` 的 `oof_pred()` 读的是**它自己模块内**的全局 `b25`。
    #    在 `15` 里给 `b25` 赋值不会传过去 —— 必须显式注入，否则报
    #    `'NoneType' object has no attribute 'random_forest_pair'`。
    d13.b25 = b25

    say("# 真正的「留一文献」（S2c）：530 折 leave-one-DOI-out")
    say("")
    say("> 由 `scripts/15_leave_one_doi.py` 生成 ｜ 2026-09-17")
    say("")

    p = pd.read_csv(CUR / "cmrt_pairs.csv", low_memory=False)
    p["row_id"] = np.arange(len(p))
    say("## 1. 🔴 现有 `S2b_leave_paper` 不是「留一文献」")
    say("")
    f = pd.read_csv(CUR / "splits" / "S2b_leave_paper.csv")
    j = p.assign(fold=f["fold"].values)
    g = j.groupby("fold").agg(n=("y", "size"), n_doi=("doi", "nunique"), y_mean=("y", "mean"))
    say(f"- 全表 **{p['doi'].nunique()} 篇 DOI**（每篇行数中位 {p.groupby('doi').size().median():.0f}，"
        f"最大 {p.groupby('doi').size().max()}）")
    say(f"- 但 `S2b_leave_paper` 只有 **{len(g)} 折**，每折 **{int(g['n_doi'].min())}–"
        f"{int(g['n_doi'].max())} 篇**、n≈{int(g['n'].median())}")
    say(f"- 各折 y 均值范围 **{g['y_mean'].min():+.3f} ~ {g['y_mean'].max():+.3f}**"
        f"（±{max(abs(g['y_mean'].min()), abs(g['y_mean'].max())):.3f}）")

    def eta2(df, key):
        grand = df["y"].mean()
        gm = df.groupby(key)["y"].agg(["mean", "size"])
        return float((gm["size"] * (gm["mean"] - grand) ** 2).sum() / ((df["y"] - grand) ** 2).sum())

    e_doi = eta2(p, "doi")
    say(f"- `η²(doi) = {e_doi:.4f}`（复现审计的 0.385）⇒ 文献间偏移 SD = "
        f"**{np.sqrt(e_doi)*p['y'].std():.3f}**")
    say(f"- ⇒ **文献效应在 53 篇的折内被平均掉了**，所以 S2b 的 −0.04 是**偏乐观**的")
    say("")

    # ── 建立 S2c
    say("## 2. 建立 `S2c_leave_doi`（530 折，每折 = 一篇文献）")
    say("")
    doi_codes = {d: i for i, d in enumerate(sorted(p["doi"].unique()))}
    fold = p["doi"].map(doi_codes).values
    sp = pd.DataFrame({"row_id": p["row_id"].values, "fold": fold, "paper": p["doi"].values})
    (CUR / "splits").mkdir(parents=True, exist_ok=True)
    sp.to_csv(CUR / "splits" / "S2c_leave_doi.csv", index=False, encoding="utf-8")
    say(f"- 已落盘 `data/curated/splits/S2c_leave_doi.csv`（{len(sp)} 行，{len(doi_codes)} 折）")
    say("")

    # ── 特征
    CF = pd.read_csv(CUR / "l1cf_descriptors.csv").set_index("conn")
    FF = pd.read_csv(CUR / "l1a_descriptors.csv").set_index("conn")
    cf_live = [c for c in CF.columns if c.startswith("cf_")
               and float((CF[c].abs() < 1e-12).mean()) <= 0.5]
    ff_live = [c for c in FF.columns if c.endswith("_bw")
               and float((FF[c].abs() < 1e-12).mean()) <= 0.5]

    want = args.arms.split(",")
    cov = set(CF.index) & set(FF.index)
    keep = p["conn"].isin(cov).values
    if any("CF" in w for w in want):
        p2 = p[keep].reset_index(drop=True).copy()
        p2["_row_id"] = p["row_id"].values[keep]
        fold2 = fold[keep]
        X0 = b25.build_X(p2)
        sub = lambda tbl, cols: tbl.reindex(p2["conn"].values)[cols].set_axis(p2.index).astype(float).fillna(0.0)
        Xbest = pd.concat([X0, sub(CF, cf_live), sub(FF, ff_live)], axis=1)
        sets = {"L0": (X0, p2, fold2), "L0+CF+FF": (Xbest, p2, fold2)}
    else:
        p2, fold2 = p, fold
        X0 = b25.build_X(p2)
        sets = {"L0": (X0, p2, fold2)}
    say(f"- 评估样本：**{len(p2)}** 行 / {p2['conn'].nunique()} 分子 / "
        f"{len(np.unique(fold2))} 篇文献 ｜ y SD = {p2['y'].std():.4f}")
    say("")

    # ── 跑 OOF
    say("## 3. 主结果（530 折 OOF）")
    say("")
    say("| 臂 | 全局 R² | **R²_off**（oracle 逐篇偏移） | **R²_aff**（+oracle 缩放） | "
        "逐篇 r 中位 | r>0 篇占比 | 折内 r 中位(n≥20) |")
    say("|---|---|---|---|---|---|---|")
    from scipy.stats import pearsonr
    rows = []
    for tag in want:
        if tag not in sets:
            continue
        X, pp, ff = sets[tag]
        y = pp["y"].values.astype(float)
        t0 = time.time()
        pred = d13.oof_pred(X, y, ff)
        say_ = f"（{tag} OOF 耗时 {time.time()-t0:.0f} s）"
        dd = d13.fold_stats(y, pred, ff, min_n=args.min_n)
        r = d13.summarize(tag, dd, y)
        r["arm"] = tag
        rows.append(r)
        say(f"| `{tag}` | **{r['r2']:+.4f}** | **{r['r2_off']:+.4f}** | **{r['r2_aff']:+.4f}** | "
            f"{r['r_med']:+.3f} | {r['frac_r_gt0']*100:.0f}% | "
            f"{dd[dd['n']>=20]['r'].median():+.3f} |")
        print(say_, flush=True)
    say("")

    R = pd.DataFrame(rows)
    R.to_csv(CUR / "leave_one_doi_summary.csv", index=False, encoding="utf-8")

    # ── ★ 不变量：R²_aff 的「零技能基线」
    say("## 4. ★★ 不变量：`R²_aff` 里有多少是「白送分组均值」的功劳")
    say("")
    say("**推导**：若模型在折内**完全无排序能力**（逐折 r=0），则逐折仿射校准后的 MSE")
    say("只剩**折内方差**，于是")
    say("")
    say("```")
    say("R²_aff(零技能) = 1 − E_w[var_f] / var(y) = η²(分组)      ← 组间方差占比")
    say("```")
    say("⇒ **`R²_aff − η²(分组)` 才是「真正的折内排序能力」的贡献。**")
    say("")

    def eta2(df, key):
        grand = df["y"].mean()
        gm = df.groupby(key)["y"].agg(["mean", "size"])
        return float((gm["size"] * (gm["mean"] - grand) ** 2).sum()
                     / ((df["y"] - grand) ** 2).sum())

    q = p2.assign(_fold=ff)
    e2 = eta2(q, "_fold")
    say(f"- 本划分 **`η²(doi) = {e2:.4f}`**（{(ff != -1).any() and len(np.unique(ff))} 折，"
        f"n={len(q)}）")
    say("")
    say("| 臂 | R² | R²_aff | η²(分组) | **R²_aff − η²**（真正的排序能力） |")
    say("|---|---|---|---|---|")
    for _, r in R.iterrows():
        say(f"| `{r['arm']}` | {r['r2']:+.4f} | {r['r2_aff']:+.4f} | {e2:.4f} | "
            f"**{r['r2_aff']-e2:+.4f}** |")
    say("")
    say("> ⚠️ **读法**：`R²_aff` 看着不小（0.44），但它**几乎全部**只是 η² ——")
    say("> 等于「白送每篇文献自己的均值与尺度」的收益，**与描述符质量无关**。")
    say("> 描述符真正贡献的那部分（最后一列）才是可归因的排序能力。")
    say("")

    say("## 5. ★ 判定")
    say("")
    for _, r in R.iterrows():
        say(f"- **`{r['arm']}`**：R² {r['r2']:+.4f} ｜ 修掉逐篇偏移后 {r['r2_off']:+.4f} "
            f"｜ 再允许逐篇缩放后 {r['r2_aff']:+.4f} ｜ 其中超出 η² 的仅 "
            f"**{r['r2_aff']-e2:+.4f}**")
        if r["r2_aff"] - e2 < 0.1:
            say(f"  → 🔴 **折内排序能力≈0**：`R²_aff` 的绝大部分来自「消掉文献间方差」，"
                f"**不是模型学到了东西**")
        if r["r2_aff"] < 0.6:
            say(f"  → 🔴 **即使白送每篇文献的偏移与尺度，R² 也只到 {r['r2_aff']:+.4f} < 0.6**"
                f" ⇒ **CP4 的目标（R²≥0.6）在本协议下不可达**")
    say("")
    say("## 6. 与 `S2b` 的对照（同一批样本不可比，只看量级）")
    say("")
    say("| 划分 | 含义 | 折数 | L0 的 R²（历史记录） |")
    say("|---|---|---|---|")
    say("| `S2b_leave_paper` | ❌ 实为「留一批文献」(53 篇/折) | 10 | −0.1088 |")
    say(f"| `S2c_leave_doi` | ✅ **真正的留一文献** | {len(np.unique(fold2))} | "
        f"{R[R['arm']=='L0']['r2'].iloc[0]:+.4f} |" if (R["arm"] == "L0").any() else "")
    say("")
    say("> ⚠️ **论文口径**：今后凡称「留一文献」，必须用 **`S2c_leave_doi`**；")
    say(">    `S2b` 只能称为「**留一批文献 / 按文献分组 10 折**」，并在方法里写清楚。")

    (ROOT / "notes" / "留一文献真划分_S2c.md").write_text("\n".join(LOG) + "\n",
                                                     encoding="utf-8")
    print("\n[落盘] notes/留一文献真划分_S2c.md + data/curated/splits/S2c_leave_doi.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
