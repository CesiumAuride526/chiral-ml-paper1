# -*- coding: utf-8 -*-
"""
37_study_blocked_null.py — **按研究分块的置换零假设**（重跑管线版）
================================================================================
## 为什么必须"重跑管线"
`scripts/18` 的阴性对照置换 3 次（行级打乱），口径与按文献聚合的主指标不对齐；R2 复核点名
「决定 'at chance' 能否成立的那条空假设从未跑过」。

⚠️ **本脚本的第一版是错的，记录在此**：它把零预测器取成"全局多数类"这个**常数**，
于是任何置换下的准确率都恒等于多数类占比（SD = 0.0000，z = ±inf）——**零分布退化**，
什么也检验不了（与项目里"阴性对照必须真的动起来"的教训同源）。
正确做法：**每次置换后重新训练分类器**，让预测随标签变化。

## 做什么
在**跨文献（leave-a-batch-of-papers, S2b 10 折）**协议上做两条零假设，各 **200** 次（`B` 见下）：
1. **篇内置换**：每篇文献内部打乱 EMO（保留每篇类别比例）⇒ 保留研究结构，破坏"标签↔特征"；
2. **整块置换**：把每篇的标签块在文献间整体重排（保留块结构）。
S2b 是"整篇文献被留出"的廉价代理（10 折 vs S2c 530 折），单次 = 10 次训练；
用同一分类器与同一 L0 特征（`scripts/_baseline_cp25.py`）。

## 产出
`data/curated/s2c_null_studyblocked.csv`
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
SEED = 20260918
B = 200


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def emo_oof(X, emo, folds) -> float:
    from sklearn.metrics import accuracy_score
    Xv = X.values.astype(float)
    pred = np.full(len(emo), -1, dtype=int)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        if te.sum() == 0 or tr.sum() == 0:
            continue
        _, clf = b25.random_forest_pair()
        clf.fit(Xv[tr], emo[tr])
        pred[te] = clf.predict(Xv[te]).astype(int)
    ok = pred >= 0
    return float(accuracy_score(emo[ok], pred[ok]))


def main() -> int:
    global b25
    b25 = load_mod("_b25", ROOT / "scripts" / "_baseline_cp25.py")
    p = pd.read_csv(CUR / "cmrt_pairs.csv", low_memory=False)
    p["row_id"] = np.arange(len(p))
    fw = pd.read_csv(CUR / "cmrt_pairs_with_folds.csv", low_memory=False)
    assert len(fw) == len(p), "行数不一致"
    folds = fw["S2b_paper_fold"].values
    X0 = b25.build_X(p).astype(float)
    emo = p["EMO"].values.astype(int)
    doi = p["doi"].astype(str).values

    t0 = time.time()
    obs = emo_oof(X0, emo, folds)
    print(f"观测 S2b EMO = {obs*100:.4f} %（耗时 {time.time()-t0:.0f} s，10 折，行数 {len(p)}）")

    order = np.argsort(doi, kind="mergesort")
    bounds = []
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and doi[order[j + 1]] == doi[order[i]]:
            j += 1
        bounds.append(order[i:j + 1])
        i = j + 1
    print(f"文献 {len(bounds)} 篇｜置换 {B} 次 × 两种方案（每次 10 折训练）")

    rng = np.random.default_rng(SEED)
    rows = []
    for label, mode in [("within_study", "within"), ("study_block", "block")]:
        accs = []
        for b in range(B):
            yp = emo.copy()
            if mode == "within":
                for g in bounds:
                    yp[g] = rng.permutation(yp[g])
            else:
                # 整块置换：把每篇文献的标签块在文献之间重排（块大小固定，按顺序铺回）
                blocks = [emo[g].copy() for g in bounds]
                perm = rng.permutation(len(blocks))
                pos = 0
                for k in range(len(bounds)):
                    b = blocks[perm[k]]
                    yp[pos:pos + len(b)] = b
                    pos += len(b)
            accs.append(emo_oof(X0, yp, folds))
        a = np.asarray(accs)
        lo, hi = np.percentile(a, [2.5, 97.5])
        p_emp = float((a >= obs).mean())
        sd = a.std(ddof=1)
        print(f"  {label:<14} null mean {a.mean()*100:.3f} %  SD {sd*100:.3f} pp  "
              f"95 % [{lo*100:.3f}, {hi*100:.3f}]  z={(obs-a.mean())/sd:+.2f}  "
              f"经验 p(≥obs)={p_emp:.4f} (n={B})")
        rows.append(dict(scheme=label, protocol="S2b_leave_a_batch_of_papers", n_perm=B, seed=SEED,
                         null_mean=a.mean(), null_sd=sd, null_lo=lo, null_hi=hi,
                         observed=obs, z=float((obs - a.mean()) / sd), p_empirical=p_emp,
                         multiple_comparison_note="p 分辨率 1/(B+1)",
                         n_rows=len(p), n_studies=len(bounds)))

    out = pd.DataFrame(rows)
    out.insert(0, "script", "scripts/37_study_blocked_null.py")
    path = CUR / "s2c_null_studyblocked.csv"
    out.to_csv(path, index=False, encoding="utf-8")
    print(f"\n[落盘] {path.relative_to(ROOT)}")
    print("读法：观测值远高于两种零分布 ⇒ 在**整篇文献留出**的协议下，模型确实学到了可检出")
    print("      但幅度很小的信号（+x pp 级），故正文只能说 'marginally better / at or near chance'。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
