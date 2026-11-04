# -*- coding: utf-8 -*-
"""
80_s2c_null_parallel.py — **H2：S2c（530 折真留一文献）上的 study-blocked 置换零假设**
================================================================================
立项（2026-11-01 升级评估 manuscript_audit.md U3）：
  论文 §5.6 / SI-S8.4 自陈：530 折 S2c 主协议**本身没有置换零假设**，只有 S2b（10 折）代理。
  本脚本补这一条：在 S2c 折索引上做**篇内置换**（保留每篇文献的类别比例，破坏标签↔特征配对），
  每次置换后**重新训练**（零分布必须重训，见 scripts/37 的教训），报 null 均值/SD/分位/z/经验 p。
⚠️ 530 折 × 每次置换 ≈ 8–18 分钟（单线程）⇒ 多进程并行；逐置换增量落盘，断电不丢。
用法：PYTHONIOENCODING=utf-8 E:/python.exe scripts/80_s2c_null_parallel.py --jobs 10 --perms 40
产出：data/curated/s2c_null_530fold.csv（增量） + notes/S2c_530折置换零假设.md（结束生成）
"""
from __future__ import annotations
import argparse, importlib.util, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CUR = ROOT / "data" / "curated"
SCR = ROOT / "scratch"
SEED = 20261101

_X = None; _EMO = None; _FOLDS = None; _DOI = None


def _init_worker():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    global _X, _EMO, _FOLDS, _DOI
    _X = np.load(SCR / "_h2_X.npy", allow_pickle=True).astype(np.float64)
    _EMO = np.load(SCR / "_h2_emo.npy", allow_pickle=True).astype(int)
    _FOLDS = np.load(SCR / "_h2_folds.npy", allow_pickle=True).astype(int)
    _DOI = np.load(SCR / "_h2_doi.npy", allow_pickle=True).astype(str)


def _perm_acc(args):
    b, seed, scheme = args
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score
    X, emo, folds, doi = _X, _EMO, _FOLDS, _DOI
    rng = np.random.default_rng(seed + b)
    yp = emo.copy()
    order = np.argsort(doi, kind="mergesort")
    groups = []
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and doi[order[j + 1]] == doi[order[i]]:
            j += 1
        groups.append(order[i:j + 1])
        i = j + 1
    if scheme == "within":
        # 篇内置换：保留每篇类别比例，破坏标签↔特征配对
        for g in groups:
            yp[g] = rng.permutation(yp[g])
    else:
        # 整块置换：把每篇的标签块在文献间整体重排（保留块结构，破坏研究↔类别平衡）
        blocks = [emo[g].copy() for g in groups]
        perm = rng.permutation(len(blocks))
        pos = 0
        for k in range(len(groups)):
            b_ = blocks[perm[k]]
            yp[pos:pos + len(b_)] = b_
            pos += len(b_)
    pred = np.full(len(emo), -1, dtype=int)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        if te.sum() == 0 or tr.sum() == 0:
            continue
        clf = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06,
                                             min_samples_leaf=20, random_state=42)
        clf.fit(X[tr], yp[tr])
        pred[te] = clf.predict(X[te]).astype(int)
    ok = pred >= 0
    return b, float(accuracy_score(emo[ok], pred[ok]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--perms", type=int, default=40)
    ap.add_argument("--scheme", choices=["within", "block"], default="within",
                    help="within=篇内置换（保守，主口径）；block=整块置换（破坏研究类别平衡）")
    args = ap.parse_args()
    import multiprocessing as mp
    from sklearn.metrics import accuracy_score

    global _X, _EMO, _FOLDS, _DOI
    spec = importlib.util.spec_from_file_location("_b25", ROOT / "scripts" / "_baseline_cp25.py")
    b25 = importlib.util.module_from_spec(spec); spec.loader.exec_module(b25)

    p = pd.read_csv(CUR / "cmrt_pairs.csv", low_memory=False)
    p["row_id"] = np.arange(len(p))
    sp = pd.read_csv(CUR / "splits" / "S2c_leave_doi.csv")
    p["fold"] = p["row_id"].map(sp.set_index("row_id")["fold"])
    assert p["fold"].notna().all()
    Xdf = b25.build_X(p).astype(float)
    if int(Xdf.isna().sum().sum()):
        Xdf = Xdf.fillna(Xdf.median(numeric_only=True))
    _X = Xdf.values.astype(np.float64)
    _EMO = p["EMO"].values.astype(int)
    _FOLDS = p["fold"].values.astype(int)
    _DOI = p["doi"].astype(str).values
    np.save(SCR / "_h2_X.npy", _X); np.save(SCR / "_h2_emo.npy", _EMO)
    np.save(SCR / "_h2_folds.npy", _FOLDS); np.save(SCR / "_h2_doi.npy", _DOI)
    print(f"数据就绪：{_X.shape} ｜ 折 {len(np.unique(_FOLDS))} ｜ 文献 {len(np.unique(_DOI))}", flush=True)

    out = CUR / ("s2c_null_530fold.csv" if args.scheme == "within" else "s2c_null_530fold_block.csv")
    if out.exists():
        old = pd.read_csv(out)
        done = set(old["perm"].astype(int).tolist())
        print(f"续跑：已有 {len(done)} 个置换", flush=True)
    else:
        done = set()
        pd.DataFrame(columns=["perm", "seed", "null_acc"]).to_csv(out, index=False, encoding="utf-8")

    obs = 0.528456057007126
    oofp = CUR / "upgrade_oof_preds.csv"
    if oofp.exists():
        o = pd.read_csv(oofp)
        ok = o["pred_S2c"] >= 0
        obs = float(accuracy_score(o["emo"][ok], o["pred_S2c"][ok]))
    print(f"观测 S2c EMO = {obs*100:.4f}%", flush=True)

    todo = [b for b in range(args.perms) if b not in done]
    t0 = time.time()
    if todo:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.jobs, initializer=_init_worker) as pool:
            for b, acc in pool.imap_unordered(_perm_acc, [(b, SEED, args.scheme) for b in todo]):
                pd.DataFrame([dict(perm=b, seed=SEED, null_acc=acc)]).to_csv(
                    out, mode="a", header=False, index=False, encoding="utf-8")
                done.add(b)
                el = time.time() - t0
                print(f"[{len(done)}/{args.perms}] perm={b} null_acc={acc*100:.4f}% "
                      f"（已用 {el/60:.1f} min）", flush=True)
    df = pd.read_csv(out)
    a = df["null_acc"].values
    print(f"\n=== 汇总：{len(a)} 个置换 | null {a.mean()*100:.3f}% ± {a.std(ddof=1)*100:.3f} pp | "
          f"95% [{np.percentile(a,2.5)*100:.3f}, {np.percentile(a,97.5)*100:.3f}] | 观测 {obs*100:.3f}% | "
          f"z={(obs-a.mean())/max(a.std(ddof=1),1e-12):+.2f} | p_emp={float((a>=obs).mean()):.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
