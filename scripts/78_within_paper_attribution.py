# -*- coding: utf-8 -*-
"""
78_within_paper_attribution.py — **H1：篇内 +0.185 到底来自「分子」还是「条件」？**
================================================================================
立项（2026-11-01 升级评估 manuscript_audit.md U1）：
  论文最核心的正面结论是「同一篇文献内，新分子 r 中位 = +0.185、80% 篇为正」。
  但 scripts/19 的篇内 50/50 切分**只按分子分组**，训练/测试**共享条件**
  （170 篇中位 10 个 (柱, ipr, 流速) 组合、4 根柱、5 个 ipr 水平）。
  ⇒ 模型可以用条件特征（ipr / speed / t0）预测，而这不是「分子信号」。
  本脚本用**同一切分、同一模型、只换特征族**做消融：
    M0  仅分子（25 维 RDKit 2D）
    M1  仅条件（ipr / speed / t0，3 维）
    M2  分子+条件（= L0 全 28 维，与 scripts/19 主臂逐位同口径）
    M3  分子+条件但不含 t0（ipr / speed；t0 与靶构造共享，见 U4）
  ★ 切分只生成一次，四个特征族**共用同一批切分**（否则配对 Wilcoxon 不成立）。
  报逐篇 Pearson r 中位 / 正例率 / EMO，并做逐篇配对 Wilcoxon（M2 vs M0）。

用法：PYTHONIOENCODING=utf-8 E:/python.exe scripts/78_within_paper_attribution.py
产出：data/curated/within_paper_attribution.csv
      notes/篇内信号归因_分子vs条件.md
"""
from __future__ import annotations
import importlib.util, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CUR = ROOT / "data" / "curated"
LOG: list[str] = []
SEED = 20260917
FIXED_ALPHA = 10.0   # 与 scripts/19 完全一致
REPS = 8             # 与 scripts/19 默认一致
MIN_N = 25           # 与 scripts/19 默认一致


def say(s=""):
    print(s, flush=True); LOG.append(s)


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def main() -> int:
    t_all = time.time()
    b25 = load_mod("_b25", ROOT / "scripts" / "_baseline_cp25.py")
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from scipy.stats import wilcoxon

    p = pd.read_csv(CUR / "cmrt_pairs.csv", low_memory=False)
    Xdf = b25.build_X(p).astype(float)
    n_na = int(Xdf.isna().sum().sum())
    if n_na:
        Xdf = Xdf.fillna(Xdf.median(numeric_only=True))
    mol_cols = [c for c in Xdf.columns if c.startswith("mol_")]
    cond_cols = [c for c in Xdf.columns if c.startswith("cond_")]
    cond_not0 = [c for c in cond_cols if c != "cond_t0"]
    arms = {
        "M0_mol_only": mol_cols,
        "M1_cond_only": cond_cols,
        "M2_mol_plus_cond": mol_cols + cond_cols,
        "M3_mol_plus_cond_no_t0": mol_cols + cond_not0,
    }
    y = p["y"].values.astype(float)
    emo = p["EMO"].values.astype(int)
    doi = p["doi"].values
    conn = p["conn"].values
    grp = {d: np.where(doi == d)[0] for d in np.unique(doi)}
    sizes = {d: len(v) for d, v in grp.items()}
    elig = [d for d, n in sizes.items() if n >= MIN_N]

    # ★ 切分只生成一次（复刻 scripts/19 的全局 RNG 顺序：按 elig 顺序、每篇 REPS 次），
    #   四个特征族共用 ⇒ 逐篇配对检验成立；M2 也应复现 scripts/19 的 +0.185。
    rng = np.random.default_rng(SEED)
    splits = []   # list of (doi, tr_idx, te_idx)
    for d in elig:
        idx = grp[d]
        mc = conn[idx]
        uc = np.unique(mc)
        for _ in range(REPS):
            cp = rng.permutation(len(uc))
            cut = max(1, len(uc) // 2)
            mask = np.isin(mc, list(set(uc[cp[:cut]])))
            tr, te = idx[mask], idx[~mask]
            if len(tr) > 0 and len(te) > 0:
                splits.append((d, tr, te))

    say("# H1：篇内信号归因（分子 vs 条件）")
    say("")
    say(f"> 由 `scripts/78_within_paper_attribution.py` 生成 ｜ 2026-11-01 ｜ 升级评估 H1")
    say(f"> 口径：与 `scripts/19` 主臂同口径（n≥{MIN_N} 的 {len(elig)} 篇、每篇 {REPS} 次 50/50 按 conn 分组切分、"
        f"Ridge(alpha={FIXED_ALPHA}) + LogReg(C=0.1)），**四个特征族共用同一批切分**（共 {len(splits)} 个切分）。")
    say(f"> 特征：分子 {len(mol_cols)} 维 / 条件 {len(cond_cols)} 维（{cond_cols}）/ 不含 t0 的条件 {len(cond_not0)} 维。")
    say("")

    def run_arm(cols):
        X0 = Xdf[cols].values
        per_paper = {d: {"r": [], "emo": [], "base": []} for d in elig}
        for d, tr, te in splits:
            yy = y[te]
            if yy.var() > 1e-8:
                reg = make_pipeline(StandardScaler(), Ridge(alpha=FIXED_ALPHA))
                reg.fit(X0[tr], y[tr]); pv = reg.predict(X0[te])
                if pv.std() > 1e-12:
                    per_paper[d]["r"].append(float(np.corrcoef(yy, pv)[0, 1]))
            if len(np.unique(emo[tr])) < 2:
                pc = np.full(len(te), int(emo[tr][0]))
            else:
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
                clf.fit(X0[tr], emo[tr]); pc = clf.predict(X0[te]).astype(int)
            per_paper[d]["emo"].append(float((pc == emo[te]).mean()))
            per_paper[d]["base"].append(float(max(emo[te].mean(), 1 - emo[te].mean())))
        rows = []
        for d in elig:
            v = per_paper[d]
            rows.append(dict(doi=d, n=len(grp[d]), arm=cols is not None and None,
                             r=float(np.mean(v["r"])) if v["r"] else np.nan,
                             emo=float(np.mean(v["emo"])), emo_base=float(np.mean(v["base"]))))
        return pd.DataFrame(rows)

    allR = []
    for arm, cols in arms.items():
        t0 = time.time()
        R = run_arm(cols); R["arm"] = arm
        allR.append(R)
        print(f"（{arm} 完成 {time.time()-t0:.0f} s）", flush=True)
    R = pd.concat(allR, ignore_index=True)
    R.to_csv(CUR / "within_paper_attribution.csv", index=False, encoding="utf-8")
    W = R.pivot(index="doi", columns="arm", values="r")

    say("## 1. 逐篇 Pearson r 中位（同一切分、同一模型，只换特征）")
    say("")
    say("| 特征族 | r 中位 | r>0 篇占比 | r 四分位 | EMO（按行加权） | EMO 基线 |")
    say("|---|---|---|---|---|---|")
    for arm in arms:
        sub = R[R.arm == arm]
        wgt = sub["n"].values / sub["n"].sum()
        say(f"| {arm} | **{sub['r'].median():+.4f}** | {(sub['r']>0).mean()*100:.0f}% | "
            f"{sub['r'].quantile(.25):+.3f} / {sub['r'].median():+.3f} / {sub['r'].quantile(.75):+.3f} | "
            f"{float((sub['emo']*wgt).sum())*100:.2f}% | {float((sub['emo_base']*wgt).sum())*100:.2f}% |")
    say("")
    say("> 锚点：M2（= scripts/19 主臂口径）的 r 中位应与 `within_paper_split.csv` 的 +0.185 一致；"
        "本表 M2 = "
        f"**{R[R.arm=='M2_mol_plus_cond']['r'].median():+.4f}**。")
    say("")
    say("## 2. 配对检验（逐篇 r，Wilcoxon 符号秩）")
    say("")
    for a, b in [("M2_mol_plus_cond", "M0_mol_only"), ("M1_cond_only", "M0_mol_only"),
                 ("M2_mol_plus_cond", "M3_mol_plus_cond_no_t0")]:
        x, yv = W[a].dropna(), W[b].dropna()
        common = x.index.intersection(yv.index)
        if len(common) >= 8:
            stat, pv = wilcoxon(x[common], yv[common])
            d = float((x[common] - yv[common]).median())
            say(f"- **{a} − {b}**：Δr 中位 **{d:+.4f}**，Wilcoxon p = **{pv:.4g}**（n={len(common)} 篇）")
    say("")
    m0 = W["M0_mol_only"].median(); m1 = W["M1_cond_only"].median()
    m2 = W["M2_mol_plus_cond"].median(); m3 = W["M3_mol_plus_cond_no_t0"].median()
    say("## 3. 判定")
    say("")
    say(f"- 仅分子 r 中位 **{m0:+.4f}**；仅条件 **{m1:+.4f}**；分子+条件 **{m2:+.4f}**；分子+条件(不含 t0) **{m3:+.4f}**")
    if abs(m2 - m3) > 0.03:
        say(f"- ⚠️ 去掉 t0 后 r 中位变化 {m3-m2:+.4f}（>0.03）⇒ **t0 泄漏（U4）在本臂上可检出**，必须披露/消融。")
    else:
        say(f"- ✅ 去掉 t0 后 r 中位变化 {m3-m2:+.4f}（<0.03）⇒ t0 泄漏在本臂上**不是主驱动**。")
    if m0 > 0.05 and m0 >= 0.7 * max(m2, 1e-9):
        say(f"- ✅ **仅分子仍显著为正（{m0:+.4f}，达 M2 的 {m0/m2*100:.0f}%）** ⇒ 篇内信号**主要来自分子**，"
            f"不能归因于条件共享；C3 的「分子信号真实存在」在控制条件后仍成立（以配对检验 p 值为准）。")
    elif m0 > 0.05:
        say(f"- 🟡 仅分子为正（{m0:+.4f}）但明显低于 M2（{m2:+.4f}）⇒ 篇内信号**部分**来自条件；C3 需加限定语。")
    else:
        say(f"- 🔴 仅分子 r 中位仅 {m0:+.4f} ⇒ 篇内 +0.185 **可能主要由条件驱动**，C3 的归因必须改写（诚实降级）。")
    say("")
    say(f"> 机时：{time.time()-t_all:.0f} s ｜ 落盘 `data/curated/within_paper_attribution.csv`（{len(R)} 行）")
    (ROOT / "notes" / "篇内信号归因_分子vs条件.md").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    print(f"[落盘] notes/篇内信号归因_分子vs条件.md ｜ {time.time()-t_all:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
