# -*- coding: utf-8 -*-
"""
06d_l1cf_signal_test.py — **L1-CF 与 L1a-FF 的同口径六臂消融**
================================================================================
`06b` 已判定 L1a-FF「有小而真的增益，但主口径不可靠，且收益来自偶量幅度」。
本脚本回答 **改描述符设计到底有没有用**：

| 臂 | 特征 | 作用 |
|---|---|---|
| `L0` | 28 维 2D + 条件 | 基线 |
| `L0+m_FF` | L0 + L1a-FF 的**偶量幅度** \\|D\\| | 旧设计的「无手性」部分（06b 的关键对照） |
| `L0+D_FF` | L0 + L1a-FF 正式（36 存活列 × 2） | 旧设计 |
| `L0+CF` | L0 + **L1-CF 存活列** | ★ 新设计（整个分子在局部手性坐标系下的分布） |
| `L0+CFm` | L0 + \\|L1-CF\\| | 新设计的偶量版 ⇒ 看**有符号的 ζ 列**是否真的在贡献 |
| `L0+CF+FF` | L0 + CF + FF | 两者是否互补（重叠度中位仅 0.363，预期互补） |

## 判据（先写死，与 `06b` 一致）
- **CF 必须明显优于 L0 与 `m` 臂，且 ΔR² 要大于折间 SD**，才算「换对了设计」；
- `L0+CFm ≈ L0+CF` ⇒ **有符号 ζ 列无用**，可以用更便宜的偶量版；
- `L0+CF > L0+D_FF` ⇒ **同样的几何代价下，新设计确实更好**。

## 用法
```bash
PYTHONIOENCODING=utf-8 python scripts/06d_l1cf_signal_test.py
```
产出：`notes/L1CF_下游信号消融.md` + `data/curated/l1cf_signal_ablation.csv`
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
ZERO_ABS = 1e-12
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


def run_cv_detail(X, y, folds, emo=None, permute=None):
    """同 `06b`：每折只训一次，额外回报逐折指标。"""
    from sklearn.metrics import accuracy_score, r2_score
    Xv = X.values.astype(float)
    rng = np.random.default_rng(0 if permute is None else permute)
    yy = rng.permutation(y) if permute is not None else y
    ee = rng.permutation(emo) if (permute is not None and emo is not None) else emo
    pred = np.full(len(y), np.nan)
    predc = np.zeros(len(y), dtype=int) if emo is not None else None
    per = []
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        if te.sum() == 0 or tr.sum() == 0:
            continue
        reg, clf = b25.random_forest_pair()
        reg.fit(Xv[tr], yy[tr])
        p_te = reg.predict(Xv[te])
        pred[te] = p_te
        per.append(float(r2_score(yy[te], p_te)))
        if emo is not None:
            clf.fit(Xv[tr], ee[tr])
            predc[te] = clf.predict(Xv[te])
    d = b25._score(yy, pred, ee, predc)
    a = np.array(per, float)
    d["r2_fold_mean"] = float(np.nanmean(a))
    d["r2_fold_sd"] = float(np.nanstd(a))
    d["r2_fold_min"] = float(np.nanmin(a))
    d["r2_fold_max"] = float(np.nanmax(a))
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ff", default=str(CUR / "l1a_descriptors.csv"))
    ap.add_argument("--cf", default=str(CUR / "l1cf_descriptors.csv"))
    ap.add_argument("--splits", default="S2_group_dup,S2b_leave_paper,S3_leave_column,S4_murcko")
    ap.add_argument("--dead-thresh", type=float, default=0.5)
    ap.add_argument("--perm", type=int, default=3)
    args = ap.parse_args()

    global b25
    b25 = load_mod("_b25", ROOT / "scripts" / "_baseline_cp25.py")

    say("# L1-CF vs L1a-FF：同口径六臂消融")
    say("")
    say("> 由 `scripts/06d_l1cf_signal_test.py` 生成 ｜ 2026-09-17")
    say("> 靶变量 `y = ln[k(S)/k(R)]` ｜ 模型超参、划分、评估与 `06b` **完全一致**")
    say("")

    p = pd.read_csv(CUR / "cmrt_pairs.csv")
    FF = pd.read_csv(args.ff)
    CF = pd.read_csv(args.cf)
    say("## 1. 数据")
    say("")
    say(f"- 主表 {len(p)} 行 / {p['conn'].nunique()} 分子")
    say(f"- L1a-FF 覆盖 **{len(FF)}** 分子 ｜ L1-CF 覆盖 **{len(CF)}** 分子")
    keep = p["conn"].isin(set(FF["conn"]) & set(CF["conn"]))
    n0 = len(p)
    p = p[keep.values].reset_index(drop=True).copy()
    p["_row_id"] = np.arange(n0)[keep.values]
    say(f"- 两者**共同覆盖**的行：{len(p)}（-{n0-len(p)}，{100*(n0-len(p))/n0:.1f}%）"
        f" ⇒ 所有臂在**同一批样本**上比较")
    say("")

    # ── 列筛选
    ff_bw = [c for c in FF.columns if c.endswith("_bw")]
    ff_live = [c for c in ff_bw
               if float((FF[c].abs() < ZERO_ABS).mean()) <= args.dead_thresh]
    cf_all = [c for c in CF.columns if c.startswith("cf_")]
    cf_live = [c for c in cf_all
               if float((CF[c].abs() < ZERO_ABS).mean()) <= args.dead_thresh]
    say("## 2. 列筛选（>50% 分子上为 0 的列剔除）")
    say("")
    say(f"- L1a-FF：{len(ff_bw)} → **{len(ff_live)}** 存活")
    say(f"- L1-CF：{len(cf_all)} → **{len(cf_live)}** 存活"
        f"（剔除 `{'`, `'.join(sorted(set(cf_all)-set(cf_live)))}`）")
    say("")

    kf = FF.set_index("conn"); kc = CF.set_index("conn")

    def m_(tbl, cols, names=None):
        sub = tbl.reindex(p["conn"].values)[cols]
        sub.index = p.index
        out = sub.astype(float).fillna(0.0)
        if names:
            out.columns = names
        return out

    X0 = b25.build_X(p)
    A = {
        "L0": X0,
        "L0+m_FF (旧·偶量)": pd.concat([X0, m_(kf, ff_live).abs()], axis=1),
        "L0+D_FF (旧)": pd.concat([X0, m_(kf, ff_live),
                                   m_(kf, [c[:-3] + "_sc" for c in ff_live])], axis=1),
        "L0+CF (新)": pd.concat([X0, m_(kc, cf_live)], axis=1),
        "L0+CFm (新·偶量)": pd.concat([X0, m_(kc, cf_live).abs()], axis=1),
    }
    A["L0+CF+FF"] = pd.concat([A["L0+CF (新)"], m_(kf, ff_live)], axis=1)
    say("## 3. 特征集")
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

    order = [("S2_group_dup", "S2 分组(分子×柱)"), ("S2b_leave_paper", "S2b 留一文献"),
             ("S3_leave_column", "S3 留一柱"), ("S4_murcko", "S4 留骨架")]
    order = [o for o in order if o[0] in set(args.splits.split(","))]

    say("## 4. 主结果")
    say("")
    say("| 划分 | 臂 | n | R² | ΔR² vs L0 | 逐折 R² 均值±SD | RMSE | EMO | EMO 基线 |")
    say("|---|---|---|---|---|---|---|---|---|")
    rows = []
    t0 = time.time()
    for fname, label in order:
        af = pd.read_csv(CUR / "splits" / f"{fname}.csv")["fold"].values
        folds = af if len(af) == len(p) else np.array([af[i] for i in p["_row_id"].values])
        base = None
        for lvl, X in A.items():
            r = run_cv_detail(X, y, folds, emo=emo)
            if lvl == "L0":
                base = r["r2"]
            r.update(split=label, arm=lvl, r2_delta=r["r2"] - base)
            rows.append(r)
            say(f"| {label} | `{lvl}` | {r['n']} | **{r['r2']:+.4f}** | "
                f"{r['r2_delta']:+.4f} | {r['r2_fold_mean']:+.4f} ± {r['r2_fold_sd']:.4f} | "
                f"{r['rmse']:.4f} | {r.get('emo_acc', float('nan'))*100:.1f}% | "
                f"{r.get('emo_base', float('nan'))*100:.1f}% |")
        say("")
    say(f"（建模总耗时 {time.time()-t0:.0f} s）")
    say("")
    df = pd.DataFrame(rows)
    df.to_csv(CUR / "l1cf_signal_ablation.csv", index=False, encoding="utf-8")

    say("## 5. 阴性对照（标签置换）")
    say("")
    say("| 划分 | 臂 | R²（打乱标签） | EMO |")
    say("|---|---|---|---|")
    for fname, label in order:
        if fname not in ("S2_group_dup", "S2b_leave_paper"):
            continue
        af = pd.read_csv(CUR / "splits" / f"{fname}.csv")["fold"].values
        folds = af if len(af) == len(p) else np.array([af[i] for i in p["_row_id"].values])
        for lvl in ("L0", "L0+D_FF (旧)", "L0+CF (新)"):
            rs, es = [], []
            for k in range(args.perm):
                r = b25.run_cv(A[lvl], y, folds, emo=emo, permute=500 + k)
                rs.append(r["r2"]); es.append(r.get("emo_acc", np.nan))
            say(f"| {label} | `{lvl}` | {np.mean(rs):+.4f} "
                f"({min(rs):+.3f} ~ {max(rs):+.3f}) | {np.mean(es)*100:.1f}% |")
        say("")

    say("## 6. ★ 判定")
    say("")
    piv = df.pivot_table(index="split", columns="arm", values="r2")
    sd = df.pivot_table(index="split", columns="arm", values="r2_fold_sd")
    for sp in piv.index:
        r = piv.loc[sp]
        g0 = r["L0"]
        gm = r["L0+m_FF (旧·偶量)"] - g0
        gd = r["L0+D_FF (旧)"] - g0
        gc = r["L0+CF (新)"] - g0
        gc2 = r["L0+CF (新)"] - r["L0+CFm (新·偶量)"]
        gcd = r["L0+CF (新)"] - r["L0+D_FF (旧)"]
        gcc = r["L0+CF+FF"] - max(r["L0+CF (新)"], r["L0+D_FF (旧)"])
        noise = max(sd.loc[sp].get("L0", 0.0), sd.loc[sp].get("L0+CF (新)", 0.0))
        verdict = []
        verdict.append("CF 增益**超折间噪声**" if gc > noise else
                       f"CF 增益**未超折间噪声**({gc:+.4f} vs {noise:.3f})")
        verdict.append("有符号 ζ 列**有贡献**" if gc2 > 0.015 else "有符号 ζ 列**无实质贡献**")
        verdict.append(("CF **优于**旧设计" if gcd > 0.01 else
                        "CF 与旧设计**同档**" if gcd > -0.01 else "CF **不如**旧设计"))
        verdict.append("CF+FF **互补**" if gcc > 0.01 else "合并**无增益**")
        say(f"- **{sp}**：L0 {g0:+.4f} ｜ +m_FF {gm:+.4f} ｜ +D_FF {gd:+.4f} ｜ "
            f"**+CF {gc:+.4f}** ｜ CF−FF {gcd:+.4f} ｜ 折间 SD {noise:.4f}")
        say(f"  → " + "；".join(verdict))
    say("")
    say("> ⚠️ 判定只用于决定下一步（是否上 DFT、是否换设计），不是论文结论。")

    (ROOT / "notes" / "L1CF_下游信号消融.md").write_text("\n".join(LOG) + "\n",
                                                     encoding="utf-8")
    print("\n[落盘] notes/L1CF_下游信号消融.md + data/curated/l1cf_signal_ablation.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
