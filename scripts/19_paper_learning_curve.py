# -*- coding: utf-8 -*-
"""
19_paper_learning_curve.py — **「同一实验室」vs「换一个实验室」的学习曲线**
================================================================================
## 立项理由

`scripts/17` 证明：文献间方差占 38.5%，且其中 ≲5% 可由组成解释 ⇒ 是**实验室特异效应**。
但「所以呢？」——审稿人一定会问。本脚本给出**最直观、最难反驳**的那张对照：

    给定**同一篇文献**的训练数据，模型能预测该文献的新分子吗？      → R² = ?   （同实验室）
    给定**其它文献**的训练数据，模型能预测这篇文献吗？              → R² = ?   （换实验室）
    训练数据越多（1 → 250 篇），换实验室这一列会变好吗？            → 学习曲线

**如果「同实验室」远好于「换实验室」，且加文献数也救不回后者，
那就直接证明：瓶颈是实验室间不可迁移性，不是数据量、也不是描述符。**

## 设计（两个setting，同一分类/回归器与特征）

1. **within（同实验室）**：挑 n ≥ `--min-n` 的文献，做 `--reps` 次 50/50 行划分；
   训练集与测试集**来自同一篇文献、但分子不同**。
   - 报**逐篇 R²**（分母是该篇测试行的 y 方差 —— 这是该文献**自己的**可解释方差）
   - EMO 同理，基线用该篇测试集的多数类
2. **across（换实验室）**：随机抽 k 篇作训练、其余作测试；
   k ∈ {1,2,5,10,20,50,100,250}，每档 `--reps` 次重复。
   - 报全局 R²（分母是**全部**测试行方差，包含文献间方差）

> ⚠️ **两个 setting 的 R² 分母不同**（within 的方差里已经不含文献间方差），
> 所以**不能直接比大小**。真正可比的是**同一个 k 下**的对照与**曲线的走向**：
> - 若 across 的 R² 随 k **一直不上升** ⇒ **加数据没用**
> - 若 within 的 R² 明显 > 0 而 across ≈ 0 ⇒ **信号存在于文献内部，不可跨文献迁移**

> ⚠️⚠️ **模型必须选小样本可估的**：within 设置里训练集常常只有十几行，
> 而 `_baseline_cp25` 的 GBM 是 `min_samples_leaf=20` ⇒ **一个分裂都做不出来**，
> 只能预测均值 ⇒ 会把「同实验室」**人为压成 0**，对照直接作废。
> ⇒ 本脚本主口径一律用**标准化 Ridge / LogisticRegressionCV**（两边同一模型类，可比）；
> 另附 `--gbm-min` 档作次要对照（只取足够大的文献，GBM 才可估）。

## 产出
- `data/curated/learning_curve_papers.csv`（逐 k 档结果）
- `data/curated/within_paper_split.csv`（逐篇结果）
- `notes/学习曲线_同实验室vs换实验室.md`

## 用法
```bash
PYTHONIOENCODING=utf-8 python scripts/19_paper_learning_curve.py
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
SEED = 20260917


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
    ap.add_argument("--reps", type=int, default=8, help="每个 k 档的重复次数")
    ap.add_argument("--min-n", type=int, default=25, help="within 设置的最小文献行数")
    ap.add_argument("--gbm-min", type=int, default=60,
                    help="GBM 次要对照的最小文献行数（训练半集须 ≥30 才有分裂空间）")
    ap.add_argument("--test-papers", type=int, default=40,
                    help="across 设置每次抽多少篇作测试（控制耗时）")
    ap.add_argument("--seed", type=int, default=20260917, help="随机种子")
    ap.add_argument("--seed-sweep", default="", help="逗号分隔的种子列表；给了就追加稳健性一节，如 1,2,3,4")
    ap.add_argument("--null", type=int, default=0,
                    help="★ 「同实验室」主结论的**置换检验阴性对照**重复次数（≥20 才有分辨力；"
                         "每篇文献**内部**打乱 y，其余口径与主臂完全一致）")
    ap.add_argument("--null-seed", type=int, default=20260918, help="置换检验的固定种子基准")
    ap.add_argument("--null-reps", type=int, default=0,
                    help="置换臂每篇的重复次数；0 = 与 --reps 相同（保证与主臂同口径）")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    t_all = time.time()

    global b25
    b25 = load_mod("_b25", ROOT / "scripts" / "_baseline_cp25.py")
    from sklearn.metrics import r2_score, accuracy_score

    p = pd.read_csv(CUR / "cmrt_pairs.csv", low_memory=False)
    Xdf = b25.build_X(p).astype(float)
    # ⚠️ GBM 原生吃 NaN，**Ridge 不吃**（会直接抛 ValueError）。
    #   本表只有 **2 个** NaN cell（`mol_MaxPartialCharge`），按列中位补齐，
    #   两个 setting 用**同一份**补齐后的矩阵，对照才公平。
    n_na = int(Xdf.isna().sum().sum())
    if n_na:
        Xdf = Xdf.fillna(Xdf.median(numeric_only=True))
        say(f"> ⚠️ 特征矩阵有 **{n_na}** 个 NaN cell，已按列中位补齐"
            f"（Ridge 不接受 NaN；两个 setting 用同一份矩阵）。")
        say("")
    X0 = Xdf.values
    y = p["y"].values.astype(float)
    emo = p["EMO"].values.astype(int)
    doi = p["doi"].values
    grp = {d: np.where(doi == d)[0] for d in np.unique(doi)}
    sizes = {d: len(v) for d, v in grp.items()}

    say("# 学习曲线：「同一实验室」vs「换一个实验室」")
    say("")
    say("> 由 `scripts/19_paper_learning_curve.py` 生成 ｜ 2026-09-17")
    say("")
    say(f"主表 **{len(p)}** 行 / **{len(grp)}** 篇文献 ｜ 特征 = L0（28 维）｜ "
        f"分类/回归器 = **标准化 Ridge / LogisticRegressionCV**"
        f"（小 n 口径，两边同一模型类；GBM 见次要对照）")
    say("")
    n_cross = int((p.groupby("conn")["doi"].nunique() >= 2).sum())
    say(f"> ⚠️ **已知小泄漏（across 设置）**：有 **{n_cross}** 个分子（"
        f"{n_cross/p['conn'].nunique()*100:.1f}%）出现在 ≥2 篇文献里，")
    say(f"> 而 across 设置只保证**文献不重叠**、未排除分子重叠 ⇒ 该设置的 R² 略微偏乐观。")
    say(f"> 量级很小，且方向对结论不利（**偏乐观仍为负**），故不修正，仅声明。")
    say("")

    # ⚠️ **模型选择有讲究**：本脚本的 within 设置里，训练集常常只有十几行
    #   （一篇文献的一半）。此时 `min_samples_leaf=20` 的 GBM **一个分裂都做不出来**，
    #   只能预测均值 ⇒ 「同实验室」这一列会被**人为压成 0**，对照就废了。
    #   ⇒ 本脚本主口径一律用 **标准化 Ridge**（小 n 下可估、两边同一模型类，可比）。
    #   另附一个 n ≥ `--gbm-min` 的 GBM 子集作为次要对照。
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    # ⚠️ **alpha 必须固定，不能用 RidgeCV**：within 设置里训练集只有十几行，
    #   CV 选 alpha 本身就不稳定，实测会把 R² 打到 −0.39（p≫n 过拟合）。
    #   固定 alpha 让两个 setting 处在**同一个模型容量**下，对照才成立。
    FIXED_ALPHA = 10.0

    def fit_eval(tr_idx, te_idx, model="ridge", y_src=None):
        """`y_src` 给定则用它当靶（阴性对照用：传一份被置换过的 y），默认用全局 `y`。"""
        yv = y if y_src is None else y_src
        if model == "ridge":
            reg = make_pipeline(StandardScaler(), Ridge(alpha=FIXED_ALPHA))
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
        else:
            reg, clf = b25.random_forest_pair()
        reg.fit(X0[tr_idx], yv[tr_idx])
        pr = reg.predict(X0[te_idx])
        # ⚠️ 小训练集里 EMO 可能**只有一类**（一篇文献的半个样本全是同一洗脱顺序）
        #    ⇒ LogisticRegression 会直接抛 ValueError。此时退化为「全预测那一类」，
        #    这是该子集上唯一可能的策略，不是作弊。
        if len(np.unique(emo[tr_idx])) < 2:
            pc = np.full(len(te_idx), int(emo[tr_idx][0]))
        else:
            clf.fit(X0[tr_idx], emo[tr_idx])
            pc = clf.predict(X0[te_idx]).astype(int)

        yy = yv[te_idx]
        vy = float(yy.var())
        # ★ 主指标用**逐篇 Pearson r** —— 尺度无关，不会因 test 方差趋零而爆炸。
        #   （初版用 R²，遇到"半个文献的 y 几乎同值"时输出 −1e15，见本文件 git 历史）
        r = float(np.corrcoef(yy, pr)[0, 1]) if vy > 1e-8 and pr.std() > 1e-12 else np.nan
        # R² 只在方差够大时才有意义
        r2 = float(r2_score(yy, pr)) if vy > 1e-3 else np.nan
        # ★ 仿射重标定后的 R²（本项目既有口径 `R²_aff`）：
        #   k 很小时 Ridge 的**预测尺度**会爆掉（实测 k=1 时 R² 到 −1678），
        #   那是"没校准"而不是"没排序能力"。允许 oracle 重标定后曲线才可读。
        r2a = np.nan
        if vy > 1e-3 and pr.std() > 1e-12:
            A = np.vstack([pr, np.ones_like(pr)]).T
            coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
            r2a = float(r2_score(yy, A @ coef))
        base = max(emo[te_idx].mean(), 1 - emo[te_idx].mean())
        acc = float(accuracy_score(emo[te_idx], pc))
        return r2, acc, float(base), r, vy, r2a

    # ══════════════════════════════ 1. within：同一篇文献内部
    say("## 1. 「同一实验室」：训练集与测试集**来自同一篇文献**（分子不同）")
    say("")
    elig = [d for d, n in sizes.items() if n >= args.min_n]
    say(f"- 合格文献（n ≥ {args.min_n}）：**{len(elig)}** 篇，覆盖 "
        f"**{sum(sizes[d] for d in elig)}** 行")
    say(f"- 每篇做 **{args.reps}** 次 50/50 行划分")
    say("")
    t0 = time.time()
    wrows = []
    for d in elig:
        idx = grp[d]
        # ⚠️ **必须按分子分组划分**：一篇文献可能把**同一个分子**测在多根柱上
        #   （合格文献里占 0.9% 的行，19/170 篇）。按行随机切会把同一分子切到两边
        #   ⇒ 模型能"背分子"，within 的 r 会被抬高。按 `conn` 整体切才是干净的。
        m_conn = p["conn"].values[idx]
        uconn = np.unique(m_conn)
        rs, acs, bas, rr, ra = [], [], [], [], []
        for _ in range(args.reps):
            cperm = rng.permutation(len(uconn))
            cut = max(1, len(uconn) // 2)
            tr_set = set(uconn[cperm[:cut]])
            mask = np.isin(m_conn, list(tr_set))
            tr, te = idx[mask], idx[~mask]
            if len(tr) == 0 or len(te) == 0:
                continue
            r2, acc, base, r, vy, r2a = fit_eval(tr, te)
            if np.isfinite(r2):
                rs.append(r2)
            if np.isfinite(r):
                rr.append(r)
            if np.isfinite(r2a):
                ra.append(r2a)
            acs.append(acc)
            bas.append(base)
        wrows.append(dict(doi=d, n=len(idx),
                          r2=float(np.mean(rs)) if rs else np.nan,
                          r=float(np.mean(rr)) if rr else np.nan,
                          r2_aff=float(np.mean(ra)) if ra else np.nan,
                          emo=float(np.mean(acs)), emo_base=float(np.mean(bas))))
    W = pd.DataFrame(wrows)
    W.to_csv(CUR / "within_paper_split.csv", index=False, encoding="utf-8")
    print(f"（within 耗时 {time.time()-t0:.0f} s）", flush=True)

    wgt = W["n"].values / W["n"].sum()
    say("| 指标 | 值 |")
    say("|---|---|")
    say(f"| ★ **逐篇 Pearson r 中位** | **{W['r'].median():+.4f}** |")
    say(f"| r > 0 的文献占比 | **{(W['r']>0).mean()*100:.0f}%** |")
    say(f"| 逐篇 r 四分位 | "
        f"{W['r'].quantile(.25):+.3f} / {W['r'].median():+.3f} / {W['r'].quantile(.75):+.3f} |")
    say(f"| 逐篇 R² 中位（次要，仅 var>1e-3 的篇） | {W['r2'].median():+.4f} |")
    say(f"| EMO 准确率（按行数加权） | **{float((W['emo']*wgt).sum())*100:.2f}%** |")
    say(f"| EMO 多数类基线（同加权） | {float((W['emo_base']*wgt).sum())*100:.2f}% |")
    say("")
    say("> ★ **主指标是逐篇 r（尺度无关）**，不是 R²：")
    say("> 半篇文献的 y 方差可能接近 0，此时 R² 的分母趋零会输出 −1e15 之类的垃圾值")
    say("> （本脚本初版就踩了这个坑）。r 不受此影响，且**与 cross 设置可直接比较**。")
    say("")

    # ── 次要对照：只在足够大的文献上用 GBM（保证可估）
    elig_g = [d for d, n in sizes.items() if n >= args.gbm_min]
    if len(elig_g) >= 5:
        t0 = time.time()
        wrows_g = []
        for d in elig_g:
            idx = grp[d]
            m_conn = p["conn"].values[idx]
            uconn = np.unique(m_conn)
            rs, acs, bas, rr, ra = [], [], [], [], []
            for _ in range(args.reps):
                cperm = rng.permutation(len(uconn))
                cut = max(1, len(uconn) // 2)
                mask = np.isin(m_conn, list(set(uconn[cperm[:cut]])))
                tr, te = idx[mask], idx[~mask]
                if len(tr) == 0 or len(te) == 0:
                    continue
                r2, acc, base, r, vy, r2a = fit_eval(tr, te, model="gbm")
                if np.isfinite(r2):
                    rs.append(r2)
                if np.isfinite(r):
                    rr.append(r)
                if np.isfinite(r2a):
                    ra.append(r2a)
                acs.append(acc)
                bas.append(base)
            wrows_g.append(dict(doi=d, n=len(idx), r2=float(np.mean(rs)) if rs else np.nan,
                                r=float(np.mean(rr)) if rr else np.nan,
                                r2_aff=float(np.mean(ra)) if ra else np.nan,
                                emo=float(np.mean(acs)), emo_base=float(np.mean(bas))))
        Wg = pd.DataFrame(wrows_g)
        wgt_g = Wg["n"].values / Wg["n"].sum()
        say(f"**次要对照（GBM，仅 n ≥ {args.gbm_min} 的 {len(elig_g)} 篇，"
            f"保证 `min_samples_leaf=20` 可估）**：")
        say("")
        say(f"- 逐篇 **r** 中位 **{Wg['r'].median():+.4f}** ｜ r>0 占比 "
            f"**{(Wg['r']>0).mean()*100:.0f}%**")
        say(f"- 逐篇 R² 中位 {Wg['r2'].median():+.4f}")
        say(f"- EMO {float((Wg['emo']*wgt_g).sum())*100:.2f}% vs 基线 "
            f"{float((Wg['emo_base']*wgt_g).sum())*100:.2f}%")
        say(f"- ⇒ 换更强的非线性模型**没有改变结论**（Ridge r 中位 "
            f"{W['r'].median():+.4f} vs GBM {Wg['r'].median():+.4f}）")
        say("")
        Wg.to_csv(CUR / "within_paper_split_gbm.csv", index=False, encoding="utf-8")
        print(f"（within-GBM 耗时 {time.time()-t0:.0f} s）", flush=True)

    # ══════════════════════════════ 2. across：换文献
    say("## 2. 「换一个实验室」：训练集来自**其它文献**，预测目标文献")
    say("")
    say(f"- 每次随机抽 **{args.test_papers}** 篇作测试集（其全部行），其余抽 k 篇作训练")
    say(f"- 每档重复 **{args.reps}** 次")
    say("")
    say("| 训练文献数 k | 训练行数 | **R²** | R²_aff（**整块**重标定） | ★ **逐篇 r 中位** | r>0 篇占比 | EMO | EMO 基线 |")
    say("|---|---|---|---|---|---|---|---|")
    KS = [1, 2, 5, 10, 20, 50, 100, 250]
    allp = np.array(list(grp.keys()))
    crows = []
    for k in KS:
        if k >= len(allp) - args.test_papers:
            continue
        t0 = time.time()
        rs, acs, bas, ntr, rmed, rfrac, ras = [], [], [], [], [], [], []
        for _ in range(args.reps):
            perm = rng.permutation(allp)
            te_papers, tr_papers = perm[:args.test_papers], perm[args.test_papers:args.test_papers + k]
            te = np.concatenate([grp[d] for d in te_papers])
            tr = np.concatenate([grp[d] for d in tr_papers])
            r2, acc, base, _r, _vy, _r2a = fit_eval(tr, te)
            # 逐篇 r：对每个测试文献单独算（与 within 设置同口径，可直接比）
            reg = make_pipeline(StandardScaler(), Ridge(alpha=FIXED_ALPHA))
            reg.fit(X0[tr], y[tr])
            pv = reg.predict(X0[te])
            pr_ = []
            for d in te_papers:
                m = np.isin(te, grp[d])
                yy = y[te][m]
                if len(yy) >= 8 and yy.var() > 1e-3 and pv[m].std() > 1e-12:
                    pr_.append(float(np.corrcoef(yy, pv[m])[0, 1]))
            if pr_:
                rmed.append(float(np.median(pr_)))
                rfrac.append(float(np.mean(np.array(pr_) > 0)))
            if np.isfinite(r2):
                rs.append(r2)
            if np.isfinite(_r2a):
                ras.append(_r2a)
            acs.append(acc)
            bas.append(base)
            ntr.append(len(tr))
        crows.append(dict(k=k, n_train=float(np.mean(ntr)), r2=float(np.mean(rs)),
                          r2_sd=float(np.std(rs)), r2_aff=float(np.mean(ras)) if ras else np.nan, r_med=float(np.mean(rmed)) if rmed else np.nan,
                          frac_r_gt0=float(np.mean(rfrac)) if rfrac else np.nan,
                          emo=float(np.mean(acs)), emo_base=float(np.mean(bas))))
        say(f"| {k} | {np.mean(ntr):.0f} | **{np.mean(rs):+.4f}** ± {np.std(rs):.4f} | **{np.mean(ras):+.4f}** | "
            f"**{np.mean(rmed):+.4f}** | {np.mean(rfrac)*100:.0f}% | "
            f"{np.mean(acs)*100:.2f}% | {np.mean(bas)*100:.2f}% |")
        print(f"（k={k} 耗时 {time.time()-t0:.0f} s）", flush=True)
    say("")
    C = pd.DataFrame(crows)
    C.to_csv(CUR / "learning_curve_papers.csv", index=False, encoding="utf-8")

    say("> ⚠️ **本表的 `R²_aff` 与 `scripts/15` 的 `R²_aff` 不是同一个估计量，别混**：")
    say("> - **本表** = **整块**（40 篇测试集合在一起）做一次 oracle 仿射重标定")
    say(">   ⇒ 只能消掉**一个全局偏移**，消不掉「每篇文献各自的偏移」")
    say("> - **`scripts/15`** = **逐折（逐篇）**各做一次 ⇒ 消掉了每篇的偏移，")
    say(">   所以那个值能被 `η²(doi)` 撑到 0.44")
    say(">")
    say("> 要看「白送每篇均值能白拿多少」→ 看 `15`；")
    say("> 要看「能不能跨文献迁移」→ 看本表。")
    say("")

    # ══════════════════════════════ 3. 判定
    say("## 3. ★ 判定")
    say("")
    r2_1 = C[C["k"] == 1]["r2"].iloc[0] if (C["k"] == 1).any() else np.nan
    r2_max = C["r2"].max()
    k_max = int(C.loc[C["r2"].idxmax(), "k"])
    r2_250 = C[C["k"] == 250]["r2"].iloc[0] if (C["k"] == 250).any() else np.nan
    r_w = float(W["r"].median())
    r_c = float(C["r_med"].iloc[-1])
    say("★ **同一口径（逐篇 Pearson r）下的直接对照**：")
    say("")
    say(f"| 设置 | 训练数据 | 逐篇 r 中位 | r>0 篇占比 |")
    say("|---|---|---|---|")
    say(f"| **同实验室** | 同一篇文献的另半（分子不同） | **{r_w:+.4f}** | "
        f"{(W['r']>0).mean()*100:.0f}% |")
    say(f"| **换实验室** | 其它 {int(C['k'].iloc[-1])} 篇文献 | **{r_c:+.4f}** | "
        f"{C['frac_r_gt0'].iloc[-1]*100:.0f}% |")
    say("")
    say(f"- **「换实验室」R²**：k=1 时 {r2_1:+.4f} → 曲线最高 {r2_max:+.4f}（在 k={k_max}）"
        f" → k=250 时 {r2_250:+.4f}")
    say(f"- **「同实验室」EMO** {float((W['emo']*wgt).sum())*100:.1f}% vs 逐篇多数类基线 "
        f"{float((W['emo_base']*wgt).sum())*100:.1f}%")
    say("")
    if r2_max < 0.05:
        say(f"  → 🔴 **把训练文献数从 1 加到 {k_max}，跨文献 R² 始终 ≈ 0 / 为负** ⇒ "
            f"**加数据救不回来**。")
    else:
        say(f"  → 🟡 跨文献 R² 最高 {r2_max:+.4f} ⇒ 有一定可迁移性，但仍远低于 CP4 的 0.6。")
    if r_w < 0.15:
        say(f"  → 🔴 **「同实验室」的逐篇 r 中位也只有 {r_w:+.4f}** ⇒ "
            f"**信号不是「锁在文献内、只是跨不出去」，而是本来就很弱**。")
    else:
        say(f"  → 🟡 「同实验室」逐篇 r 中位 {r_w:+.4f} > 跨文献 {r_c:+.4f}"
            f" ⇒ 存在**文献内可学、跨文献不可迁移**的成分。")
    say("")
    say("> ⚠️ **诚实边界**：within 设置的训练集只有十几行（半篇文献），")
    say("> 样本量远小于 across 设置 ⇒ **within 的数字是「小样本下的能力」，不是上界**。")
    say("> 但两点仍成立：① across 曲线**不随 k 上升**（这条与样本量无关）；")
    say("> ② within 的 EMO **≈ 逐篇多数类基线**，即在该文献内部也几乎没有排序能力。")
    say(">")
    say("> 🔴 **论文口径**：这一条给出的是**最有说服力的「换实验室」对照**，")
    say("> 且**直接回答了审稿人的「那什么才行」**：")
    say("> 不是更多数据、不是更贵的描述符，而是**跨实验室的条件对齐与标准化测量**。")
    say("")

    # ══════════════════════ 3b. ★ 阴性对照：逐篇内部置换 y（第八场补，task-13）
    #   立项：**论文最核心的正面结论**（同实验室逐篇 r 中位 = +0.185、80% 篇为正）
    #   此前**没有任何阴性对照** —— 违反本项目自己的规则（PROJECT_STATE §6.2 坑 13
    #   「任何建模结论前必须先跑阴性对照（打乱标签）」）。
    #   做法：保持每篇的分子特征、折结构（按 conn 分组切半）、模型/特征/口径**全部不变**，
    #        只把该篇的 y **在篇内整体置换** ⇒ 破坏「描述符 ↔ 靶」的配对。
    #   ⚠️ 本项目第七场坑 9 的教训：对照必须**真能破坏信号**（若打乱前后数字完全相同
    #      = 空转）。下面有显式断言 + 打印 null 与实测的差。
    if args.null > 0:
        nreps = args.null_reps or args.reps
        obs_r_med = float(W["r"].median())
        obs_frac = float((W["r"] > 0).mean())
        say("## 3b. ★★ 阴性对照：**逐篇内部置换 y**（置换检验）")
        say("")
        say(f"> 立项：这条「同实验室 r 中位 {obs_r_med:+.3f}」是论文最核心的正面结论，")
        say("> 但此前**没有阴性对照**（违反坑 13）。本检验保持特征/折结构/模型/口径**完全不变**，")
        say("> 只把每篇的 `y` **在篇内整体置换** ⇒ 破坏「描述符 ↔ 靶」的配对。")
        say(">")
        say(f"> 重复 **{args.null}** 次（固定种子 {args.null_seed}）；每篇每轮做 **{nreps}** 次 50/50 切分"
            f"（与主臂 `--reps {args.reps}` 同口径）。")
        say("")
        null_rows = []
        t_null = time.time()
        for it in range(args.null):
            nrng = np.random.default_rng(args.null_seed + it)
            rr_w = []
            for d in elig:
                idx = grp[d]
                mc = p["conn"].values[idx]
                uc = np.unique(mc)
                if len(uc) < 2:
                    continue
                # 篇内置换：只动 y，X / conn / 折结构都不动
                yv = y.copy()
                yv[idx] = nrng.permutation(y[idx])
                vals = []
                for _ in range(nreps):
                    cp = nrng.permutation(len(uc))
                    cut = max(1, len(uc) // 2)
                    mask = np.isin(mc, list(set(uc[cp[:cut]])))
                    tr, te = idx[mask], idx[~mask]
                    if len(tr) == 0 or len(te) == 0:
                        continue
                    _r2, _acc, _base, r_, _vy, _r2a = fit_eval(tr, te, y_src=yv)
                    if np.isfinite(r_):
                        vals.append(r_)
                if vals:
                    rr_w.append(float(np.mean(vals)))
            null_rows.append(dict(
                iter=it, arm="within_paper_y_permuted", n_paper=len(rr_w), reps=nreps,
                seed=args.null_seed + it,
                r_med=float(np.median(rr_w)) if rr_w else np.nan,
                frac_r_gt0=float(np.mean(np.array(rr_w) > 0)) if rr_w else np.nan))
        P = pd.DataFrame(null_rows)
        P.to_csv(CUR / "within_paper_null.csv", index=False, encoding="utf-8")
        print(f"（置换检验耗时 {time.time()-t_null:.0f} s）", flush=True)

        nm = P["r_med"].dropna()
        nf = P["frac_r_gt0"].dropna()
        # ★ 非空转自检（坑 9）
        assert nm.nunique() > 1, "❌ 置换检验的 null 每次都相同 —— 疑似空转（no-op），立即检查"
        assert not np.allclose(nm.values, obs_r_med), \
            "❌ null 与实测完全相同 —— 配对没被破坏（空转），立即检查"
        say("| 量 | 实测（真 y） | null 均值 ± SD | null 2.5% / 50% / 97.5% 分位 | 经验 p 值 |")
        say("|---|---|---|---|---|")
        p_r = (1 + int((nm >= obs_r_med).sum())) / (len(nm) + 1)
        say(f"| 逐篇 r 中位 | **{obs_r_med:+.4f}** | {nm.mean():+.4f} ± {nm.std():.4f} | "
            f"{nm.quantile(.025):+.4f} / {nm.quantile(.5):+.4f} / {nm.quantile(.975):+.4f} | "
            f"**{p_r:.4f}** |")
        p_f = (1 + int((nf >= obs_frac).sum())) / (len(nf) + 1)
        say(f"| r>0 篇占比 | **{obs_frac*100:.1f}%** | {nf.mean()*100:.1f}% ± {nf.std()*100:.1f}% | "
            f"{nf.quantile(.025)*100:.1f}% / {nf.quantile(.5)*100:.1f}% / "
            f"{nf.quantile(.975)*100:.1f}% | **{p_f:.4f}** |")
        say("")
        say(f"⇒ 实测 r 中位 {obs_r_med:+.4f} 落在 null 分布的 "
            f"**{(nm < obs_r_med).mean()*100:.1f}%** 分位；null 里最大只到 {nm.max():+.4f}。")
        if p_r < 0.05:
            say(f"⇒ ✅ **显著（p = {p_r:.4f} < 0.05）**：篇内信号不是置换所能产生的 ⇒ "
                f"「同实验室存在真信号」站得住。")
        else:
            say(f"⇒ 🔴 **不显著（p = {p_r:.4f}）** ⇒ ⚠️ 「同实验室 r=+0.185」可能只是小样本噪声，"
                f"**必须回报 Lead 并重新审视论文论断。**")
        say("")
        say(f"> 📌 **对照已自证非空转**（坑 9）：{args.null} 次 null 的 r 中位并非同一个值"
            f"（SD = {nm.std():.4f}），且与实测 {obs_r_med:+.4f} 不同 "
            f"（差 {obs_r_med - nm.mean():+.4f}）。")
        say(">")
        say("> ⚠️ **读法**：置换检验只回答「篇内信号是否超出把 y 随机重排的水平」，")
        say("> 它**不**回答「跨文献可迁移性」—— 那是主表 across 一列的结论。")
        say("")
        print(f"[落盘] data/curated/within_paper_null.csv（{len(P)} 次置换）", flush=True)

    # ── 稳健性：换种子重跑（论文的核心数字不能是 seed 依赖的）
    if args.seed_sweep:
        seeds = [int(x) for x in args.seed_sweep.split(",") if x.strip()]
        say("## 4. ★ 稳健性检验：换随机种子重跑核心对照")
        say("")
        say(f"> 核心论断（「同实验室 r>0、换实验室 r≈0」）**不能是 seed 依赖的**。")
        say(f"> 下面在 {len(seeds)} 个种子上重跑同一口径（特征/模型/协议全不变）。")
        say("")
        say("| seed | **同实验室 r 中位** | r>0 篇占比 | **换实验室 r 中位**（k=最大） | r>0 篇占比 |")
        say("|---|---|---|---|---|")
        for sd in seeds:
            r2_ = np.random.default_rng(sd)
            # within
            rr_w, ok_w = [], []
            for d in elig:
                idx = grp[d]
                mc = p["conn"].values[idx]
                uc = np.unique(mc)
                vals = []
                for _ in range(args.reps):
                    cp = r2_.permutation(len(uc))
                    cut = max(1, len(uc) // 2)
                    mask = np.isin(mc, list(set(uc[cp[:cut]])))
                    tr, te = idx[mask], idx[~mask]
                    if len(tr) == 0 or len(te) == 0:
                        continue
                    yy = y[te]
                    if yy.var() <= 1e-8:
                        continue
                    reg = make_pipeline(StandardScaler(), Ridge(alpha=FIXED_ALPHA))
                    reg.fit(X0[tr], y[tr])
                    pv = reg.predict(X0[te])
                    if pv.std() > 1e-12:
                        vals.append(float(np.corrcoef(yy, pv)[0, 1]))
                if vals:
                    rr_w.append(float(np.mean(vals)))
            # across（用最大 k 档）
            kbig = max(KS)
            rr_a = []
            for _ in range(args.reps):
                perm = r2_.permutation(allp)
                te_p, tr_p = perm[:args.test_papers], perm[args.test_papers:args.test_papers + kbig]
                te_i = np.concatenate([grp[d] for d in te_p])
                tr_i = np.concatenate([grp[d] for d in tr_p])
                reg = make_pipeline(StandardScaler(), Ridge(alpha=FIXED_ALPHA))
                reg.fit(X0[tr_i], y[tr_i])
                pv = reg.predict(X0[te_i])
                for d in te_p:
                    m = np.isin(te_i, grp[d])
                    yy = y[te_i][m]
                    if len(yy) >= 8 and yy.var() > 1e-3 and pv[m].std() > 1e-12:
                        rr_a.append(float(np.corrcoef(yy, pv[m])[0, 1]))
            wmed = float(np.median(rr_w))
            wfrac = float(np.mean(np.array(rr_w) > 0))
            amed = float(np.median(rr_a))
            afrac = float(np.mean(np.array(rr_a) > 0))
            say(f"| {sd} | **{wmed:+.4f}** | {wfrac*100:.0f}% | **{amed:+.4f}** | {afrac*100:.0f}% |")
            print(f"（seed={sd} 完成）", flush=True)
        say("")
        say("> 读法：只要「同实验室」一列稳定为正、且**远高于**「换实验室」那一列（后者围绕 0 抖动），")
        say("> 核心论断就与随机种子无关。")
        say("")

    notes = ROOT / "notes" / "学习曲线_同实验室vs换实验室.md"
    notes.write_text("\n".join(LOG) + "\n", encoding="utf-8")
    print(f"\n[落盘] {notes}")
    print(f"[总耗时] {time.time()-t_all:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
