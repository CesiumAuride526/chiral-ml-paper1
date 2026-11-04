# -*- coding: utf-8 -*-
"""
13_failure_decomposition.py — **把 S2b/S3/S4 的负 R² 拆成「偏移」还是「排序崩」**
================================================================================
## 为什么做这个

前四场把描述符侧反复加固（2D 天花板 +0.088 → L1-CF 也只能到 +0.15），
但**主口径 S2b 一直是负的**（−0.109 → −0.043）。在继续加算力之前必须先回答：

> S2b 的失败，是「**系统性偏移**」（模型整体偏了，只要知道被留出那篇文献的
>   条件/报告口径就能修正 ⇒ 属于**特征缺口**，可修）
> 还是「**排序崩**」（即使允许逐折做仿射校准，R² 仍然 ≤ 0 ⇒ 属于**根本限制**）？

## 判据（先写死）

对每个留出折 f（= 一篇文献 / 一根柱 / 一个骨架），用 OOF 预测算三个量：

| 量 | 定义 | 含义 |
|---|---|---|
| `bias_f` | mean(y_f) − mean(ŷ_f) | 系统性偏移 |
| `r_f` | Pearson(y_f, ŷ_f) | **折内排序能力**（偏移无关） |
| `ρ_f` | Spearman(y_f, ŷ_f) | 折内排序能力（秩，抗异常） |
| `R²_f` | 1 − MSE/var | 常规 |

再做两个**上界**（oracle）——它们不是模型，是「**若有人告诉我们该折的口径，最好能到多少**」：

- **R²_off** = 修掉每折 bias 后的 R² ⇒ 「**只知道一个常数偏移**」的天花板；
- **R²_aff** = 每折再做仿射校准（a·ŷ+b）后的 R² ⇒ 「**偏移 + 尺度**」的天花板。

⇒ **判定规则**：
- 若 `R²_off ≥ +0.3` ⇒ **失败主要是偏移** ⇒ 下一步是**补条件/协议特征**（可修）；
- 若 `R²_off ≤ 0 而 R²_aff ≥ +0.3` ⇒ **失败是尺度错配**（训练集 y 分布比测试集宽）；
- 若 **连 `R²_aff` 都 ≤ 0** ⇒ **折内排序本身没有信号** ⇒ 描述符侧再怎么加算力都无用。

## 用法
```bash
PYTHONIOENCODING=utf-8 python scripts/13_failure_decomposition.py
```
产出：`notes/失败模式分解.md` + `data/curated/failure_decomposition.csv`
"""
from __future__ import annotations

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


def oof_pred(X, y, folds):
    """返回 OOF 预测向量（与 `_baseline_cp25.run_cv` 同配置）。"""
    Xv = X.values.astype(float)
    pred = np.full(len(y), np.nan)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        if te.sum() == 0 or tr.sum() == 0:
            continue
        reg, _ = b25.random_forest_pair()
        reg.fit(Xv[tr], y[tr])
        pred[te] = reg.predict(Xv[te])
    return pred


def fold_stats(y, pred, folds, min_n=20):
    """逐折：n / y 分布 / 预测分布 / bias / r / ρ / R² / 仿射最优 R²。"""
    from scipy.stats import pearsonr, spearmanr
    out = []
    for f in np.unique(folds):
        m = (folds == f) & np.isfinite(pred)
        if m.sum() < min_n:
            continue
        yy, pp = y[m], pred[m]
        vy = yy.var()
        if vy <= 0:
            continue
        # 仿射最优（最小二乘 a,b）
        A = np.vstack([pp, np.ones_like(pp)]).T
        coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
        a, b = float(coef[0]), float(coef[1])
        mse = float(np.mean((yy - pp) ** 2))
        mse_off = float(np.mean((yy - (pp + yy.mean() - pp.mean())) ** 2))
        mse_aff = float(np.mean((yy - (a * pp + b)) ** 2))
        try:
            r = float(pearsonr(yy, pp)[0])
        except Exception:
            r = np.nan
        try:
            rho = float(spearmanr(yy, pp)[0])
        except Exception:
            rho = np.nan
        out.append(dict(fold=f, n=int(m.sum()), y_mean=float(yy.mean()),
                        y_sd=float(yy.std()), p_mean=float(pp.mean()),
                        p_sd=float(pp.std()), bias=float(yy.mean() - pp.mean()),
                        slope=a, r=r, rho=rho, r2=1 - mse / vy,
                        r2_off=1 - mse_off / vy, r2_aff=1 - mse_aff / vy))
    return pd.DataFrame(out)


def summarize(tag, d, y):
    """全局 R² + 逐折上界。"""
    from scipy.stats import pearsonr
    n_tot = int(d["n"].sum())
    w = d["n"].values / n_tot
    # ⚠️ 逐折 MSE = (1 − R²_f) · var_f（ddof=0，与 fold_stats 内一致）
    #    再按 n 加权还原全局 MSE。**不要**写成 Σ w·R²_f·var_f ——
    #    那样得到的是 1 + R²，会算出 > 1 的不可能值（本脚本初版就犯过这个错）。
    vf = d["y_sd"].values ** 2
    var_y = float(np.var(y))
    r2_glob = 1 - float(np.sum(w * (1 - d["r2"].values) * vf)) / var_y
    r2_off = 1 - float(np.sum(w * (1 - d["r2_off"].values) * vf)) / var_y
    r2_aff = 1 - float(np.sum(w * (1 - d["r2_aff"].values) * vf)) / var_y
    for nm, v in (("r2", r2_glob), ("r2_off", r2_off), ("r2_aff", r2_aff)):
        assert v <= 1.0 + 1e-9, f"{tag}: {nm} = {v} > 1 —— 公式写错了"
    return dict(split=tag, n=n_tot, n_fold=len(d), r2=r2_glob,
                r2_off=r2_off, r2_aff=r2_aff,
                b_iqr=float(d["bias"].quantile(.75) - d["bias"].quantile(.25)),
                b_med=float(d["bias"].median()),
                r_med=float(d["r"].median()), rho_med=float(d["rho"].median()),
                frac_r_gt0=float((d["r"] > 0).mean()),
                frac_aff_gt0=float((d["r2_aff"] > 0).mean()),
                frac_off_gt0=float((d["r2_off"] > 0).mean()))


def main() -> int:
    global b25
    b25 = load_mod("_b25", ROOT / "scripts" / "_baseline_cp25.py")

    say("# 失败模式分解：负 R² 是「偏移」还是「排序崩」？")
    say("")
    say("> 由 `scripts/13_failure_decomposition.py` 生成 ｜ 2026-09-17")
    say("> 特征 = `L0`（28 维 2D + 条件）｜ 模型 = 与 `06b/06d` **完全一致**的 GBM")
    say("> ⚠️ 本脚本用 **L0 全表**（10,525 行）跑，好处是不受 L1 覆盖率裁剪影响；")
    say(">    对照：`06d` 报的 S2b 三臂分别是 L0 −0.1088 / +CF −0.0473 / +CF+FF −0.0432，"
        "**本诊断的绝对水平只代表 L0 那一档**。")
    say("")

    p = pd.read_csv(CUR / "cmrt_pairs.csv", low_memory=False)
    X0 = b25.build_X(p)
    y_all = p["y"].values.astype(float)
    say(f"- 主表 {len(p)} 行 / {p['conn'].nunique()} 分子 ｜ SD(y) = {y_all.std():.4f} "
        f"｜ EMO 基线 = {max(p.EMO.mean(), 1-p.EMO.mean())*100:.1f}%")
    say(f"- **每分子行数中位 = {p.groupby('conn').size().median():.0f}**"
        f"（{p.groupby('conn').size().max()} 行封顶）⇒ **CMRT 每个对映体对只测一次，无重复测量**")
    say("")

    # ── ★ 对照组：最好臂 L0+CF+FF（覆盖子集），看描述符升级有没有改变失败模式
    try:
        import importlib.util as _iu
        spec = _iu.spec_from_file_location("_d6", ROOT / "scripts" / "06d_l1cf_signal_test.py")
        CF = pd.read_csv(CUR / "l1cf_descriptors.csv").set_index("conn")
        FF = pd.read_csv(CUR / "l1a_descriptors.csv").set_index("conn")
        cov = set(CF.index) & set(FF.index)
        m = p["conn"].isin(cov).values
        p2 = p[m].reset_index(drop=True).copy()
        p2["_row_id"] = np.arange(len(p))[m]
        cf = [c for c in CF.columns if c.startswith("cf_")]
        ff = [c for c in FF.columns if c.endswith("_bw")]
        def _sub(tbl, cols):
            s = tbl.reindex(p2["conn"].values)[cols]
            s.index = p2.index
            return s.astype(float).fillna(0.0)
        Xbest = pd.concat([b25.build_X(p2), _sub(CF, cf), _sub(FF, ff)], axis=1)
        say(f"- ★ 对照臂 `L0+CF+FF` 覆盖 **{m.sum()}** 行（{p2['conn'].nunique()} 分子，"
            f"{Xbest.shape[1]} 维）—— **所有臂在同一批样本上比较**")
        say("")
    except Exception as e:                                   # noqa: BLE001
        say(f"- ⚠️ 对照臂构建失败（{e}），只跑 L0")
        say("")
        Xbest, p2 = None, None

    order = [("S2_group_dup", "S2 分组(分子×柱)"), ("S2b_leave_paper", "S2b 留一文献"),
             ("S3_leave_column", "S3 留一柱"), ("S4_murcko", "S4 留骨架")]

    def run_set(tag: str, X, pp):
        rows, details = [], {}
        yy = pp["y"].values.astype(float)
        for fname, label in order:
            sf = CUR / "splits" / f"{fname}.csv"
            if not sf.exists():
                continue
            af = pd.read_csv(sf)["fold"].values
            folds = af if len(af) == len(pp) else np.array([af[i] for i in pp["_row_id"].values])
            pred = oof_pred(X, yy, folds)
            d = fold_stats(yy, pred, folds)
            details[label] = d
            r = summarize(label, d, yy)
            r["arm"] = tag
            rows.append(r)
        return pd.DataFrame(rows), details

    t0 = time.time()
    say("## 1. 主结果：全局 R² 与两个 oracle 上界")
    say("")
    say("| 臂 | 划分 | n | 折数 | R² 全局 | **R²_off**（修掉逐折偏移） | "
        "**R²_aff**（再允许逐折缩放） | 折内 r 中位 | 折内 ρ 中位 |")
    say("|---|---|---|---|---|---|---|---|---|")
    R_all, D_all = [], {}
    for tag, X, pp in ([("L0", X0, p)] +
                       ([("L0+CF+FF", Xbest, p2)] if Xbest is not None else [])):
        R, D = run_set(tag, X, pp)
        R_all.append(R)
        D_all[tag] = D
        for _, r in R.iterrows():
            say(f"| `{tag}` | {r['split']} | {int(r['n'])} | {int(r['n_fold'])} | "
                f"**{r['r2']:+.4f}** | **{r['r2_off']:+.4f}** | **{r['r2_aff']:+.4f}** | "
                f"{r['r_med']:+.3f} | {r['rho_med']:+.3f} |")
    say("")
    say(f"（OOF 建模总耗时 {time.time()-t0:.0f} s）")
    say("")

    R = pd.concat(R_all, ignore_index=True)
    say("| 臂 | 划分 | 逐折 bias 四分位距 | bias 中位 | 折内 r>0 占比 | 折内 R²_off>0 占比 | 折内 R²_aff>0 占比 |")
    say("|---|---|---|---|---|---|---|")
    for _, r in R.iterrows():
        say(f"| `{r['arm']}` | {r['split']} | {r['b_iqr']:.3f} | {r['b_med']:+.3f} | "
            f"{r['frac_r_gt0']*100:.0f}% | {r['frac_off_gt0']*100:.0f}% | {r['frac_aff_gt0']*100:.0f}% |")
    say("")

    # ── 判定
    say("## 2. ★ 判定（按先写死的规则）")
    say("")
    for _, r in R.iterrows():
        ro, ra = r["r2_off"], r["r2_aff"]
        if ro >= 0.3:
            v = f"**失败主要是【偏移】**（修掉逐折常数后 R²={ro:+.3f} ≥ 0.3）⇒ **补条件/协议特征可修**"
        elif ra >= 0.3:
            v = f"**失败是【尺度错配】**（offset 后仍 {ro:+.3f}，仿射后 {ra:+.3f}）⇒ 训练集 y 分布比测试集宽"
        elif ra <= 0.0:
            v = f"🔴 **【折内排序本身无信号】**（连仿射校准后 R²={ra:+.3f} ≤ 0）⇒ **加算力无用**"
        else:
            v = f"🟡 **混合**（offset 后 {ro:+.3f}，仿射后 {ra:+.3f}）：有部分排序信号但很弱"
        say(f"- **`{r['arm']}` / {r['split']}**：R² {r['r2']:+.4f} → 修偏移 {ro:+.4f} "
            f"→ 再修尺度 {ra:+.4f} ｜ {v}")
    say("")

    # ── 逐折明细（只打最关键的 S2b，两个臂）
    say("## 3. `S2b 留一文献` 逐折明细（按 n 降序）")
    for tag, D in D_all.items():
        if "S2b 留一文献" not in D:
            continue
        d = D["S2b 留一文献"].sort_values("n", ascending=False)
        say("")
        say(f"### 臂 `{tag}`")
        say("")
        say("| fold | n | y 均值 | y 的 SD | 预测均值 | 预测 SD | bias | 仿射斜率 | r | ρ | R² | R²_off | R²_aff |")
        say("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in d.head(12).iterrows():
            say(f"| {r['fold']} | {int(r['n'])} | {r['y_mean']:+.3f} | {r['y_sd']:.3f} | "
                f"{r['p_mean']:+.3f} | {r['p_sd']:.3f} | {r['bias']:+.3f} | {r['slope']:.3f} | "
                f"{r['r']:+.3f} | {r['rho']:+.3f} | {r['r2']:+.3f} | {r['r2_off']:+.3f} | {r['r2_aff']:+.3f} |")
    say("")

    # ── 尺度对比
    say("## 4. 尺度错配检查：预测被「收缩」了多少")
    say("")
    say("| 臂 | 划分 | y_sd 中位（逐折） | 预测 sd 中位 | 收缩比 = 预测 sd / y_sd |")
    say("|---|---|---|---|---|")
    for tag, D in D_all.items():
        for label, d in D.items():
            say(f"| `{tag}` | {label} | {d['y_sd'].median():.3f} | {d['p_sd'].median():.3f} | "
                f"{d['p_sd'].median()/d['y_sd'].median():.3f} |")
    say("")

    allrow = pd.concat([d.assign(split=k, arm=a)
                        for a, D in D_all.items() for k, d in D.items()])
    allrow.to_csv(CUR / "failure_decomposition.csv", index=False, encoding="utf-8")
    R.to_csv(CUR / "failure_decomposition_summary.csv", index=False, encoding="utf-8")
    (ROOT / "notes" / "失败模式分解.md").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    print("\n[落盘] notes/失败模式分解.md + data/curated/failure_decomposition*.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
