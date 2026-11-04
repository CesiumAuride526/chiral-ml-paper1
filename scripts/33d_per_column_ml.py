# -*- coding: utf-8 -*-
"""
33d_per_column_ml.py — 「按柱子分机器学习」到底有没有用？（回应 PI 建议）
========================================================================
PI 建议：*CMRT 还可以按柱子分机器学习的尝试*。
文献的通行做法正是「每根柱训练一个模型」，而本项目此前只跑过跨柱口径
（S3 留一柱：train 24 根柱 → 测第 25 根，EMO = 56.46%）。
本脚本把「按柱建模」跑穿 —— **每根柱单独跑一次当前的机器学习**，两个臂：

  A 臂（文献式·乐观）：柱内 **分子随机** 5 折 CV（×3 组种子）→ EMO
  B 臂（严格式）：柱内 **留一文献（DOI）** → EMO

## 口径（必须与 scripts/18、32 逐字一致，否则数字不可比）
- 模型：`_baseline_cp25.random_forest_pair()` 的分类器
  （`HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06,
  min_samples_leaf=20, random_state=42)`）
- 特征：`_baseline_cp25.build_X(p)`（L0：25 个 RDKit 2D 描述符 + cond_ipr/cond_speed/cond_t0）
- 标签：`p["EMO"]`（0/1 洗脱顺序）
- 基线：逐折多数类（样本量加权的 oracle 上界）+ 全局多数类
- 阴性对照：打乱标签（A 臂 3 次 / B 臂 2 次），自检非空转（第七场坑 9）

## 读数口径
- 「柱子的影响」= 逐柱 A/B 两臂的差异 + 柱间差异（含 η²(Column) 方差份额）
- 参照系：S3 留一柱（跨柱）EMO = 56.46%（`data/curated/s3_emo.csv`）

## 产出
- `data/curated/per_column_ml.csv`（逐柱表 + 汇总行）
- `notes/按柱建模_per_column_ML.md`（本文件）

## 用法
```bash
set PYTHONIOENCODING=utf-8
set OMP_NUM_THREADS=1        # ⚠️ HGB 多线程在受限环境会 PermissionError；单线程也够快
python scripts/33d_per_column_ml.py                 # 全量（含阴性对照）
python scripts/33d_per_column_ml.py --shuffleA 0 --shuffleB 0   # 跳过对照（快）
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
SEED = 20261009
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


def cv_emo(Xv, emo, folds):
    """逐折训练分类器 → OOF 预测 + 逐折多数类加权基线。

    与 `scripts/32_s3_emo.py:emo_oof` 同一口径（唯一区别：本函数还返回逐折明细）。
    """
    from sklearn.metrics import accuracy_score
    pred = np.full(len(emo), -1, dtype=int)
    per_fold = []
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        if te.sum() == 0 or tr.sum() == 0:
            continue
        _, clf = b25.random_forest_pair()
        clf.fit(Xv[tr], emo[tr])
        p = clf.predict(Xv[te]).astype(int)
        pred[te] = p
        per_fold.append((int(te.sum()), float(accuracy_score(emo[te], p)),
                         float(max(emo[te].mean(), 1 - emo[te].mean()))))
    ok = pred >= 0
    acc = float(accuracy_score(emo[ok], pred[ok])) if ok.any() else float("nan")
    w = np.array([r[0] for r in per_fold], dtype=float)
    base = float(np.average([r[2] for r in per_fold], weights=w)) if per_fold else float("nan")
    return pred, acc, base, len(per_fold)


def arm_a(Xv, emo, seeds):
    """柱内随机 5 折 × len(seeds) 组。返回 (逐种子 acc 列表, 逐种子基线列表)。"""
    from sklearn.model_selection import StratifiedKFold
    cnt = np.bincount(emo, minlength=2)
    k = min(5, int(cnt.min()))
    if k < 2:
        return None
    accs, bases = [], []
    for s in seeds:
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=s)
        folds = np.full(len(emo), -1, dtype=int)
        for i, (_, te) in enumerate(skf.split(np.zeros((len(emo), 1)), emo)):
            folds[te] = i
        _, acc, base, _ = cv_emo(Xv, emo, folds)
        accs.append(acc)
        bases.append(base)
    return accs, bases


def arm_b(Xv, emo, groups):
    """柱内留一文献（LeaveOneGroupOut on DOI）。"""
    codes = pd.factorize(groups)[0]
    if len(np.unique(codes)) < 2:
        return None
    _, acc, base, nf = cv_emo(Xv, emo, codes)
    return acc, base, nf, len(np.unique(codes))


def aggregate(rows, key) -> float:
    """按列样本量加权的总体 EMO。"""
    ns = np.array([r["n"] for r in rows], dtype=float)
    vs = np.array([r[key] for r in rows], dtype=float)
    ok = np.isfinite(vs)
    return float(np.average(vs[ok], weights=ns[ok]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103])
    ap.add_argument("--shuffleA", type=int, default=3, help="A 臂阴性对照次数")
    ap.add_argument("--shuffleB", type=int, default=2, help="B 臂阴性对照次数")
    args = ap.parse_args()
    t_all = time.time()

    global b25
    b25 = load_mod("_b25", ROOT / "scripts" / "_baseline_cp25.py")

    p = pd.read_csv(CUR / "cmrt_pairs.csv", low_memory=False)
    X0 = b25.build_X(p).astype(float)
    emo = p["EMO"].values.astype(int)
    y = p["y"].values.astype(float)
    cols = sorted(p["Column"].unique())

    # ── 结构自检
    assert p["Column"].notna().all() and p["doi"].notna().all()
    assert len(cols) == 25, f"柱数 {len(cols)} != 25"

    say("# 按柱建模（per-column ML）—— 每根柱单独跑一次当前的机器学习")
    say("")
    say("> 由 `scripts/33d_per_column_ml.py` 生成 ｜ 回应 PI 建议「CMRT 按柱子分机器学习」")
    say(f"> 口径与 `scripts/18`/`32` 逐字一致（HGB max_iter=250 / L0 28 维 / EMO 0-1）")
    say("")
    say(f"主表 **{len(p)}** 对 ｜ **{len(cols)}** 根柱 ｜ 全局 EMO 正类 {emo.mean()*100:.2f}% ｜ "
        f"A 臂 = 柱内分子随机 5 折（种子 {args.seeds}）｜ B 臂 = 柱内留一文献")
    say("")

    # ── η²(Column) 与 η²(doi)：柱 vs 文献 的方差份额（「柱子的影响」有多大）
    def eta2(keys):
        gm = y.mean()
        g = pd.DataFrame({"k": keys, "y": y}).groupby("k")["y"]
        ss_b = float((g.count() * (g.mean() - gm) ** 2).sum())
        ss_t = float(((y - gm) ** 2).sum())
        return ss_b / ss_t
    eta_col = eta2(p["Column"].values)
    eta_doi = eta2(p["doi"].values)
    say(f"方差份额（对 y = ln α）：**η²(Column) = {eta_col*100:.1f}%** ｜ "
        f"η²(doi) = {eta_doi*100:.1f}%（文献效应，参照）")
    say("")

    # ── 逐柱两臂
    rows = []
    say("## 1. 逐柱结果（A 臂 = 柱内随机；B 臂 = 柱内留一文献）")
    say("")
    say("| 柱 | n | 文献数 | EMO 正类 | y 均值 | **A 臂 EMO** | A 臂 oracle 基线 | **B 臂 EMO** | B 臂 oracle 基线 | B−A |")
    say("|---|---|---|---|---|---|---|---|---|---|")
    t0 = time.time()
    for c in cols:
        m = (p["Column"] == c).values
        Xc, ec, gc = X0[m], emo[m], p.loc[m, "doi"].values
        ra = arm_a(Xc.values.astype(float), ec, args.seeds)
        rb = arm_b(Xc.values.astype(float), ec, gc)
        a_acc = float(np.mean([v for v in ra[0]])) if ra else float("nan")
        a_sd = float(np.std([v for v in ra[0]])) if ra else float("nan")
        a_base = float(np.mean(ra[1])) if ra else float("nan")
        b_acc = rb[0] if rb else float("nan")
        b_base = rb[1] if rb else float("nan")
        rows.append(dict(column=c, n=int(m.sum()), n_doi=int(pd.Series(gc).nunique()),
                         emo_pos=float(ec.mean()), y_mean=float(y[m].mean()),
                         A_emo=a_acc, A_emo_sd=a_sd, A_base=a_base,
                         B_emo=b_acc, B_base=b_base, B_minus_A=(b_acc - a_acc)))
        say(f"| {c} | {m.sum()} | {pd.Series(gc).nunique()} | {ec.mean()*100:.1f}% | "
            f"{y[m].mean():+.3f} | **{a_acc*100:.1f}%** ± {a_sd*100:.1f} | {a_base*100:.1f}% | "
            f"**{b_acc*100:.1f}%** | {b_base*100:.1f}% | {(b_acc-a_acc)*100:+.1f} pp |")
    dt_main = time.time() - t0
    print(f"（两臂耗时 {dt_main:.0f} s）", flush=True)

    overall_a = aggregate(rows, "A_emo")
    overall_a_base = aggregate(rows, "A_base")
    overall_b = aggregate(rows, "B_emo")
    overall_b_base = aggregate(rows, "B_base")
    s3 = pd.read_csv(CUR / "s3_emo.csv")
    s3_acc = float(s3["emo_acc"].iloc[0])

    say("")
    say("## 2. 总体（按样本量加权）")
    say("")
    say("| 口径 | EMO | oracle 基线 | 相对基线 |")
    say("|---|---|---|---|")
    say(f"| **A 臂：柱内随机**（文献式） | **{overall_a*100:.2f}%** | {overall_a_base*100:.2f}% | "
        f"{(overall_a-overall_a_base)*100:+.2f} pp |")
    say(f"| **B 臂：柱内留一文献**（严格式） | **{overall_b*100:.2f}%** | {overall_b_base*100:.2f}% | "
        f"{(overall_b-overall_b_base)*100:+.2f} pp |")
    say(f"| 参照：S3 留一柱（跨柱，scripts/32） | {s3_acc*100:.2f}% | — | — |")
    say("")
    say(f"⇒ **A→B 掉 {(overall_a-overall_b)*100:.2f} pp** —— 柱内换一篇文献，同一根柱上信号几乎全部消失"
        f"（与全局'同实验室 vs 换实验室'的结论同型）。")
    say("")

    # ── 阴性对照
    say("## 3. 阴性对照（打乱标签）")
    say("")
    rng = np.random.default_rng(SEED)
    nullA, nullB = [], []
    if args.shuffleA > 0:
        for i in range(args.shuffleA):
            esh = rng.permutation(emo)
            v = []
            for c in cols:
                m = (p["Column"] == c).values
                ra = arm_a(X0[m].values.astype(float), esh[m], args.seeds[:1])
                if ra:
                    v.append((int(m.sum()), float(ra[0][0])))
            acc = float(np.average([a for _, a in v], weights=[n for n, _ in v]))
            nullA.append(acc)
        say(f"- A 臂打乱 {args.shuffleA} 次：**{np.mean(nullA)*100:.2f}% ± {np.std(nullA)*100:.2f}**")
    if args.shuffleB > 0:
        for i in range(args.shuffleB):
            esh = rng.permutation(emo)
            v = []
            for c in cols:
                m = (p["Column"] == c).values
                rb = arm_b(X0[m].values.astype(float), esh[m], p.loc[m, "doi"].values)
                if rb:
                    v.append((int(m.sum()), float(rb[0])))
            acc = float(np.average([a for _, a in v], weights=[n for n, _ in v]))
            nullB.append(acc)
        say(f"- B 臂打乱 {args.shuffleB} 次：**{np.mean(nullB)*100:.2f}% ± {np.std(nullB)*100:.2f}**")
    if nullA and abs(np.mean(nullA) - overall_a) < 1e-9:
        raise AssertionError("❌ 阴性对照与真值完全相同 —— 疑似空转（no-op）")
    if nullB and abs(np.mean(nullB) - overall_b) < 1e-9:
        raise AssertionError("❌ 阴性对照与真值完全相同 —— 疑似空转（no-op）")
    say("")

    # ── 落盘
    R = pd.DataFrame(rows)
    R.to_csv(CUR / "per_column_ml.csv", index=False, encoding="utf-8")
    say("## 4. 落盘")
    say("")
    say("- `data/curated/per_column_ml.csv`（逐柱表）")
    say("- `notes/按柱建模_per_column_ML.md`（本文件）")
    say(f"- 总耗时 {time.time()-t_all:.0f} s（两臂主体 {dt_main:.0f} s）")
    say("")
    (ROOT / "notes" / "按柱建模_per_column_ML.md").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    print(f"\n[落盘] data/curated/per_column_ml.csv  A={overall_a*100:.2f}%  B={overall_b*100:.2f}%"
          f"（S3={s3_acc*100:.2f}%）")
    print(f"[落盘] notes/按柱建模_per_column_ML.md")
    print(f"[总耗时] {time.time()-t_all:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
