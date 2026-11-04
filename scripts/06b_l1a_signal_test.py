# -*- coding: utf-8 -*-
"""
06b_l1a_signal_test.py — ★ L1a-FF 的**下游信号消融**（本项目最关键的判定实验）
================================================================================
`PROJECT_STATE.md` §5 ③ 的原话是「先用小样本确认描述符在下游模型上有信号」。

本脚本把它做成**可判定**的实验，并且在 2026-09-17 审计后**加了两个必需对照臂**。

## 为什么必须加对照臂（审计发现，见 `notes/L1a_描述符构建报告.md` §3）
`06` 的分支按 **CIP 秩降序**排列 ⇒ 局部伪标量的**符号被 R/S 锁死**（实测 92.8–97.2%）。
而本数据集的 `cip_S` **恒为 `S`**（10,516/10,525 行）—— 也就是「参考对映体」在整张表里
是同一个 CIP 标记。两条合起来意味着：

> **48 个伪标量的符号在整张表上近乎恒定 ⇒ 它们的「手性」信息是空的；
>  真正带信息的是它们的【幅度】 `|D|`，而 `|D|` 在镜面反射下【不变】= 无手性（偶）量。**

若这条成立，那么「必须上伪标量」这个立项前提在本任务上**是空的** ——
所以必须用「**只给偶量幅度**」的对照臂把这件事测死。

## 五个臂
| 臂 | 特征 | 作用 |
|---|---|---|
| `L0` | 分子 2D + 条件 | 基线（对齐 `_baseline_cp25`） |
| `L0+m` | L0 + **48 个 \\|D\\|（偶量 / 无手性）** | ★ **关键对照**：把「手性」全拿掉，只留 3D 幅度 |
| `L0+D` | L0 + 96 列（`*_bw` 有符号 + `*_sc` 符号共识） | **正式 L1a** |
| `L0+s` | L0 + 48 个 `sign(D)` ∈ {−1,0,+1} | ★ **纯符号对照**：若 ≈ L0，说明符号确实不含信息 |
| `L0+Dv` | L0 + 40 列（`--no-vol`，剔线性冗余） | 检验冗余列是否只是噪声 |

## 判据（先写死，避免事后解释）
- **`L0+D` 必须比 `L0` 明显好**，且**必须比 `L0+m` 明显好**，才算「伪标量有用」。
- 若 `L0+m ≈ L0+D > L0` ⇒ 收益全部来自**偶量 3D 描述**，伪标量框架是多余的。
- 若 `L0+s ≈ L0` ⇒ 符号不含信息（预期如此，因为 `cip_S` 恒定）。
- 阴性对照（打乱标签）必须回到 R² ≈ 0 且 EMO ≈ 50–57%。

## 用法
```bash
PYTHONIOENCODING=utf-8 python scripts/06b_l1a_signal_test.py
PYTHONIOENCODING=utf-8 python scripts/06b_l1a_signal_test.py --splits S2b_leave_paper,S3_leave_column
```
产出：`notes/L1a_下游信号消融.md` + `data/curated/l1a_signal_ablation.csv`
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
SEED = 42
ZERO_ABS = 1e-12
LOG: list[str] = []
b25 = None                      # 在 main() 里装载（`_baseline_cp25` 模块句柄）


def say(s: str = "") -> None:
    print(s, flush=True)
    LOG.append(s)


def load_baseline():
    """复用 `_baseline_cp25.py` 的特征与评估骨架（避免两套口径）"""
    spec = importlib.util.spec_from_file_location("_b25", ROOT / "scripts" / "_baseline_cp25.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def run_cv_detail(X, y, folds, emo=None, permute=None):
    """与 `_baseline_cp25.run_cv` 同构，但**额外回报逐折指标**。

    为什么必须报逐折：ΔR² 常常只有 +0.03 量级，而折间波动常有 ±0.05。
    只报一个汇总 R²，读者无法判断增益是不是噪声。**拟合次数不变**（每折仍只训一次）。
    """
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
        per.append(dict(fold=int(f), n_test=int(te.sum()),
                        r2=float(r2_score(yy[te], p_te))))
        if emo is not None:
            clf.fit(Xv[tr], ee[tr])
            pc = clf.predict(Xv[te])
            predc[te] = pc
            per[-1]["emo"] = float(accuracy_score(ee[te], pc))
    d = b25._score(yy, pred, ee, predc)
    r2s = np.array([q["r2"] for q in per], float)
    d["r2_fold_mean"] = float(np.nanmean(r2s))
    d["r2_fold_sd"] = float(np.nanstd(r2s))
    d["r2_fold_min"] = float(np.nanmin(r2s))
    d["r2_fold_max"] = float(np.nanmax(r2s))
    d["per_fold"] = per
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--l1a", default=str(CUR / "l1a_descriptors.csv"))
    ap.add_argument("--splits", default="S2_group_dup,S2b_leave_paper,S3_leave_column,S4_murcko")
    ap.add_argument("--dead-thresh", type=float, default=0.5,
                    help="某列在 >此比例的分子上为 0 则视为退化列，删除")
    ap.add_argument("--perm", type=int, default=3, help="阴性对照重复次数")
    ap.add_argument("--keep-uncovered", action="store_true",
                    help="保留没有 L1a 描述符的分子（默认**剔除**，否则 fillna(0) 会给模型"
                         "一个「此分子无覆盖」的伪特征，污染臂间对比）")
    args = ap.parse_args()

    global b25
    b25 = load_baseline()

    say("# L1a-FF 下游信号消融（★ 判定实验）")
    say("")
    say("> 由 `scripts/06b_l1a_signal_test.py` 生成 ｜ 2026-09-17")
    say("> 靶变量 `y = ln[k(S)/k(R)]`（定向）｜ 划分来自 `01_curate_cmrt.py`")
    say("")
    say("## 0. 判据（先写死）")
    say("")
    say("- `L0+D` 要**同时**胜过 `L0` 与 **`L0+m`**，才算「**伪标量**有用」；")
    say("- 若 **`L0+m ≈ L0+D` 且都 > `L0`** ⇒ 收益全来自**偶量 3D 幅度**，"
        "「必须用伪标量」这个前提在本任务上**是空的**；")
    say("- 若 `L0+s ≈ L0` ⇒ 符号不含信息（因 `cip_S` 恒定，预期如此）。")
    say("- ΔR² 只有 +0.0x 量级，**必须与逐折 SD 对比**才能判有无信号。")
    say("")

    p = pd.read_csv(CUR / "cmrt_pairs.csv")
    L1 = pd.read_csv(args.l1a)
    say("## 1. 数据")
    say("")
    u = p.drop_duplicates("conn")["conn"]
    cov = float(u.isin(set(L1["conn"])).mean()) if len(L1) else 0.0
    say(f"- 主表 **{len(p)}** 行 / **{len(u)}** 个分子；L1a 表 **{len(L1)}** 个分子"
        f" ｜ **覆盖主表分子 {cov*100:.2f}%**（未覆盖的多为硫立体中心，属设计性排除）")
    if not args.keep_uncovered:
        keep = p["conn"].isin(set(L1["conn"])).values
        n_before = len(p)
        p = p[keep].reset_index(drop=True).copy()
        p["_row_id"] = np.arange(n_before)[keep]      # ★ 划分靠 row_id 对齐，必须带着走
        say(f"- **已剔除无描述符覆盖的分子**：{n_before} → **{len(p)}** 行"
            f"（-{n_before-len(p)}，{100*(n_before-len(p))/n_before:.1f}%）"
            f" ｜ `--keep-uncovered` 可关闭此行为")
    else:
        p = p.copy()
        p["_row_id"] = np.arange(len(p))
    say(f"- `cip_S` 取值分布（这决定了「R/S 指示符」能否当特征）："
        f"`{p['cip_S'].value_counts().head(4).to_dict()}`")
    say("")

    # ── 特征矩阵：把 L1a 表按 conn 合并进主表
    cols_bw = [c for c in L1.columns if c.endswith("_bw")]
    cols_sc = [c for c in L1.columns if c.endswith("_sc")]
    base = [c[:-3] for c in cols_bw]

    # 退化列：在某列上 >dead-thresh 的分子取值为 0
    zf = {c: float((L1[c].abs() < ZERO_ABS).mean()) for c in cols_bw}
    dead = [c for c, v in zf.items() if v > args.dead_thresh]
    live = [c for c in cols_bw if c not in dead]
    say(f"## 2. 列体检")
    say("")
    say(f"- 退化列（>{args.dead_thresh*100:.0f}% 分子上为 0）**{len(dead)}** 个，已删除："
        f"`{'`, `'.join(dead[:12])}`{' …' if len(dead) > 12 else ''}")
    say(f"- 保留 **{len(live)}** 列")
    say("")

    key = L1.set_index("conn")
    def merge_cols(names):
        sub = key.reindex(p["conn"].values)[names]
        sub.index = p.index
        return sub.astype(float).fillna(0.0)

    X0 = b25.build_X(p)
    M_mag = merge_cols(live).abs()                       # 偶量幅度
    M_D = pd.concat([merge_cols(live), merge_cols([n[:-3] + "_sc" for n in live])], axis=1)
    M_Dv = merge_cols([c for c in live if not c.startswith("vol_")])
    M_s = merge_cols(live).apply(np.sign)                # 纯符号 ∈ {-1,0,+1}

    arms = {
        "L0": X0,
        "L0+m (偶量幅度)": pd.concat([X0, M_mag], axis=1),
        "L0+D (正式 L1a)": pd.concat([X0, M_D], axis=1),
        "L0+s (纯符号)": pd.concat([X0, M_s], axis=1),
        "L0+Dv (剔冗余)": pd.concat([X0, M_Dv], axis=1),
    }
    say("## 3. 特征集")
    say("")
    say("| 臂 | 维度 |")
    say("|---|---|")
    for k, v in arms.items():
        say(f"| `{k}` | {v.shape[1]} |")
    say("")

    y = p["y"].values.astype(float)
    emo = p["EMO"].values.astype(int)
    say(f"y 的 SD = {y.std():.4f}（R²=0 时的 RMSE 基准）｜ "
        f"EMO 多数类基线 = {max(emo.mean(), 1-emo.mean())*100:.1f}%")
    say("")

    order = [("S2_group_dup", "S2 分组(分子×柱)"), ("S2b_leave_paper", "S2b 留一文献"),
             ("S3_leave_column", "S3 留一柱"), ("S4_murcko", "S4 留骨架")]
    want = set(args.splits.split(","))
    order = [o for o in order if o[0] in want]

    say("## 4. 主结果")
    say("")
    say("> `R²` = 汇总 OOF；`逐折` = 各折 R² 的 均值±SD（**判断 ΔR² 是否大于折间噪声**）")
    say("")
    say("| 划分 | 臂 | n | R² | ΔR² vs L0 | 逐折 R² 均值±SD | 折范围 | RMSE | EMO | EMO 基线 |")
    say("|---|---|---|---|---|---|---|---|---|---|")
    rows = []
    t0 = time.time()
    for fname, label in order:
        fpath = CUR / "splits" / f"{fname}.csv"
        if not fpath.exists():
            continue
        all_f = pd.read_csv(fpath)["fold"].values
        use = p["_row_id"].values
        if len(all_f) != len(use):
            folds = np.array([all_f[i] for i in use])
        else:
            folds = all_f
        r0 = None
        for lvl, X in arms.items():
            r = run_cv_detail(X, y, folds, emo=emo)
            if r0 is None and lvl == "L0":
                r0 = r["r2"]
            r.update(split=label, arm=lvl, r2_delta=None if r0 is None else r["r2"] - r0)
            r.pop("per_fold", None)
            rows.append(r)
            d = "—" if lvl == "L0" else f"{(r['r2']-r0):+.4f}"
            say(f"| {label} | `{lvl}` | {r['n']} | **{r['r2']:+.4f}** | {d} | "
                f"{r['r2_fold_mean']:+.4f} ± {r['r2_fold_sd']:.4f} | "
                f"{r['r2_fold_min']:+.3f}~{r['r2_fold_max']:+.3f} | "
                f"{r['rmse']:.4f} | {r.get('emo_acc', float('nan'))*100:.1f}% | "
                f"{r.get('emo_base', float('nan'))*100:.1f}% |")
        say("")
    say(f"（建模总耗时 {time.time()-t0:.0f} s）")
    say("")
    df = pd.DataFrame(rows)
    df.to_csv(CUR / "l1a_signal_ablation.csv", index=False, encoding="utf-8")

    # ── 阴性对照
    say("## 5. 阴性对照（标签置换）")
    say("")
    say("| 划分 | 臂 | R²（打乱标签，均值±范围） | EMO |")
    say("|---|---|---|---|")
    for fname, label in order:
        if fname not in ("S2_group_dup", "S2b_leave_paper"):
            continue
        all_f = pd.read_csv(CUR / "splits" / f"{fname}.csv")["fold"].values
        folds = all_f if len(all_f) == len(p) else np.array([all_f[i] for i in p["_row_id"].values])
        for lvl in ("L0", "L0+m (偶量幅度)", "L0+D (正式 L1a)"):
            rs, es = [], []
            for k in range(args.perm):
                r = b25.run_cv(arms[lvl], y, folds, emo=emo, permute=1000 + k)
                rs.append(r["r2"]); es.append(r.get("emo_acc", np.nan))
            say(f"| {label} | `{lvl}` | {np.mean(rs):+.4f} "
                f"({min(rs):+.3f} ~ {max(rs):+.3f}) | {np.mean(es)*100:.1f}% |")
        say("")

    # ── 判定
    say("## 6. ★ 判定")
    say("")
    piv = df.pivot_table(index="split", columns="arm", values="r2")
    sd = df.pivot_table(index="split", columns="arm", values="r2_fold_sd")
    for sp in piv.index:
        a0 = piv.loc[sp].get("L0", np.nan)
        am = piv.loc[sp].get("L0+m (偶量幅度)", np.nan)
        ad = piv.loc[sp].get("L0+D (正式 L1a)", np.nan)
        as_ = piv.loc[sp].get("L0+s (纯符号)", np.nan)
        gain_d, gain_m = ad - a0, am - a0
        noise = max(sd.loc[sp].get("L0", 0.0), sd.loc[sp].get("L0+D (正式 L1a)", 0.0))
        if gain_d <= noise:
            verdict = f"**无可靠信号**（ΔR² {gain_d:+.4f} ≤ 折间 SD {noise:.3f}）"
        elif ad > am + 0.02:
            verdict = "**伪标量有增量**（D 明显优于 m 与 L0）"
        elif ad > am:
            verdict = "**幅度主导、符号有微弱增量**（D 略优于 m）"
        elif gain_m > 0.01:
            verdict = "**收益来自偶量幅度**（m ≥ D > L0）⇒ 伪标量框架在本任务上多余"
        else:
            verdict = "**无信号**（D ≈ m ≈ L0）"
        say(f"- **{sp}**：L0 {a0:+.4f} ｜ L0+m {am:+.4f}（{gain_m:+.4f}）｜ "
            f"L0+D {ad:+.4f}（{gain_d:+.4f}）｜ L0+s {as_:+.4f} ｜ 折间 SD {noise:.4f}")
        say(f"  → 判定：{verdict}")
    say("")
    say("> ⚠️ 判定只用于**决定下一步投不投算力**，不是论文结论。"
        "若结论是「无信号」，正确动作是**先改描述符**（换 DFT 矢量 / 加手性空腔描述符），"
        "而不是加大采样。")

    (ROOT / "notes" / "L1a_下游信号消融.md").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    print("\n[落盘] notes/L1a_下游信号消融.md + data/curated/l1a_signal_ablation.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
