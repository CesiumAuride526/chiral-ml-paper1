# -*- coding: utf-8 -*-
"""
32_s3_emo.py — **S3（留一柱）的洗脱顺序准确率（EMO）—— 补上论文 Table 2 里 S3 行的可复现来源**
================================================================================
## 立项理由（第八场代码审计发现 #3）

论文 Table 2 有这一行：

    S3 留一柱（LOO-column，25 折）  EMO = 56.46%  ｜ 逐折多数类 oracle 基线 = 55.81%

但 `data/curated/s3_emo.csv` **在此之前没有任何脚本产出**：
`PROJECT_STATE` 把它归给 `scripts/31`，而 `scripts/31` 并不存在（`scripts/` 只有 00–30）。
后果有两层：
  ① 这个论文数字**没有可复现来源**（文件是手工/一次性产物）；
  ② `scripts/26_paper_number_gate.py:63-66` 的 S3 断言写成 `if _s3.exists():` ——
     文件一旦丢失，门禁**静默少查一项**（45 → 44 项），退出码仍是 0，照样报「✅ 全部通过」。

⇒ 本脚本按 `scripts/18_s2c_emo.py` 的**同一分类器、同一特征、同一 EMO 口径**，
在 S3（留一柱）上重算，并把结果落盘成与 `s2c_emo.csv` **列名完全一致**的表。

## 口径（必须与 `scripts/18` 逐字一致，否则数字不可比）
- 分类器：`_baseline_cp25.random_forest_pair()` 的第二个返回值
  = `HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, min_samples_leaf=20, random_state=42)`
- 特征：`_baseline_cp25.build_X(p)`（L0：25 个 RDKit 2D 分子描述符 + `cond_ipr/cond_speed/cond_t0`）
- 折：`data/curated/splits/S3_leave_column.csv`，**按 `row_id` 对齐**（PROJECT_STATE 坑 32）
- 三个基线：全局多数类 / 逐折多数类（样本量加权的 oracle 上界）/ EMO 准确率
- 另附**标签置换阴性对照**（默认 5 次），并断言它与真值不同（本项目第七场坑 9 的教训：对照必须真能破坏信号）

## 产出
- `data/curated/s3_emo.csv`（列与 `s2c_emo.csv` 完全一致：split,n,n_fold,emo_acc,emo_base_global,emo_base_perfold,acc_perfold_weighted）
- `notes/S3_EMO.md`

## 用法
```bash
set PYTHONIOENCODING=utf-8
set OMP_NUM_THREADS=1          # ⚠️ DSH 沙箱禁命名管道，HGB 多线程会 PermissionError [WinError 5]
python scripts/32_s3_emo.py
python scripts/32_s3_emo.py --shuffle 0     # 跳过阴性对照（快）
```
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CUR = ROOT / "data" / "curated"
SEED = 20260917          # ★ 与 scripts/18_s2c_emo.py 完全一致
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


def emo_oof(X, emo, folds) -> tuple[np.ndarray, dict]:
    """逐折训练分类器 → OOF 类别预测。返回 (pred, 统计)。

    ⚠️ 本函数**逐字复制**自 `scripts/18_s2c_emo.py:66-92`（口径一致是硬要求）。
    """
    from sklearn.metrics import accuracy_score
    Xv = X.values.astype(float)
    pred = np.full(len(emo), -1, dtype=int)
    per_fold = []
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        if te.sum() == 0 or tr.sum() == 0:
            continue
        _, clf = b25.random_forest_pair()
        # EMO 必须是 0/1 二分类
        clf.fit(Xv[tr], emo[tr])
        p = clf.predict(Xv[te]).astype(int)
        pred[te] = p
        per_fold.append((int(f), int(te.sum()), float(accuracy_score(emo[te], p)),
                         float(max(emo[te].mean(), 1 - emo[te].mean()))))
    ok = pred >= 0
    acc = float(accuracy_score(emo[ok], pred[ok]))
    maj_global = float(max(emo[ok].mean(), 1 - emo[ok].mean()))
    # 逐折多数类的样本量加权平均（oracle 上界）
    w = np.array([r[1] for r in per_fold], dtype=float)
    maj_fold = float(np.average([r[3] for r in per_fold], weights=w)) if len(per_fold) else np.nan
    acc_fold = float(np.average([r[2] for r in per_fold], weights=w)) if len(per_fold) else np.nan
    return pred, dict(n=int(ok.sum()), n_fold=len(per_fold), emo_acc=acc,
                      emo_base_global=maj_global, emo_base_perfold=maj_fold,
                      acc_perfold_weighted=acc_fold)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shuffle", type=int, default=5, help="阴性对照重复次数（0 = 跳过）")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    t_all = time.time()

    global b25
    b25 = load_mod("_b25", ROOT / "scripts" / "_baseline_cp25.py")

    p = pd.read_csv(CUR / "cmrt_pairs.csv", low_memory=False)
    p["row_id"] = np.arange(len(p))
    sp = pd.read_csv(CUR / "splits" / "S3_leave_column.csv")
    # ⚠️ 划分按 row_id 对齐，不是 pair_id（PROJECT_STATE 坑 32）
    fold_map = sp.set_index("row_id")["fold"]
    p["fold"] = p["row_id"].map(fold_map)
    assert p["fold"].notna().all(), "❌ S3 划分未覆盖全部行"
    assert sp["row_id"].is_unique, "❌ S3 划分的 row_id 不唯一"
    assert len(sp) == len(p), f"❌ S3 行数 {len(sp)} != 主表 {len(p)}"

    n_fold = int(p["fold"].nunique())
    n_col = int(sp["heldout_column"].nunique())
    # ★ 结构自检：柱 ↔ 折必须一一对应（否则「留一柱」名不副实）
    colmap = sp.groupby("heldout_column")["fold"].nunique()
    assert bool((colmap == 1).all()), "❌ 同一根柱跨多折 —— 不是真 LOO-column"
    assert n_fold == n_col, f"❌ 折数 {n_fold} != 柱数 {n_col}"

    X0 = b25.build_X(p).astype(float)
    emo = p["EMO"].values.astype(int)

    say("# S3（留一柱）的洗脱顺序准确率（EMO）")
    say("")
    say("> 由 `scripts/32_s3_emo.py` 生成 ｜ 配置与 `scripts/18_s2c_emo.py` **逐字一致**")
    say("> 立项：`data/curated/s3_emo.csv` 原先无任何脚本产出（第八场代码审计发现 #3）")
    say("")
    say(f"样本 **{len(p)}** 行 / **{n_fold}** 折（= {n_col} 根柱 LOO）｜ "
        f"全局 EMO 正类占比 = {emo.mean():.4f} ｜ 特征 {X0.shape[1]} 维")
    say("")

    t0 = time.time()
    _, r = emo_oof(X0, emo, p["fold"].values)
    dt = time.time() - t0
    say("## 1. S3 主结果（25 折 OOF，留一柱）")
    say("")
    say("| 指标 | 值 |")
    say("|---|---|")
    say(f"| **EMO 准确率** | **{r['emo_acc']*100:.4f}%** |")
    say(f"| 全局多数类基线 | {r['emo_base_global']*100:.4f}% |")
    say(f"| 逐折多数类基线（oracle，**上界**） | **{r['emo_base_perfold']*100:.4f}%** |")
    say(f"| 相对全局基线的增益 | {(r['emo_acc']-r['emo_base_global'])*100:+.2f} 个百分点 |")
    say(f"| 相对逐折 oracle 基线的增益 | "
        f"{(r['emo_acc']-r['emo_base_perfold'])*100:+.2f} 个百分点 |")
    say(f"| 折数 / 样本 | {r['n_fold']} / {r['n']} |")
    say(f"| 逐折准确率的样本量加权均值 | {r['acc_perfold_weighted']*100:.4f}% |")
    say("")
    print(f"（S3 EMO 耗时 {dt:.0f} s）", flush=True)

    # ── 与论文/旧文件的对照（只报差异，不改论文）
    say("## 2. 与论文 Table 2 / 旧 `s3_emo.csv` 的对照")
    say("")
    old = CUR / "s3_emo.csv"
    say("| 量 | 本次重算 | 论文 / 旧文件 | 差 |")
    say("|---|---|---|---|")
    say(f"| EMO 准确率 | **{r['emo_acc']*100:.4f}%** | 56.46%（论文 Table 2；旧 s3_emo.csv） "
        f"| {r['emo_acc']*100-56.46:+.4f} pp |")
    say(f"| 逐折多数类基线 | **{r['emo_base_perfold']*100:.4f}%** | 55.81%（论文）**旧文件没有这一列** "
        f"| {r['emo_base_perfold']*100-55.81:+.4f} pp |")
    say(f"| 全局多数类基线 | {r['emo_base_global']*100:.4f}% | 50.67%（旧文件） "
        f"| {r['emo_base_global']*100-50.67:+.4f} pp |")
    say("")

    # ── 阴性对照（本项目规则：任何建模数字都要先跑打乱标签）
    null_vals: list[float] = []
    if args.shuffle > 0:
        say("## 3. 阴性对照（打乱标签，同一 S3 协议）")
        say("")
        for i in range(args.shuffle):
            emo_sh = rng.permutation(emo)
            _, rs = emo_oof(X0, emo_sh, p["fold"].values)
            null_vals.append(float(rs["emo_acc"]))
            say(f"- 第 {i+1} 次打乱：EMO = **{rs['emo_acc']*100:.4f}%**")
        say("")
        say(f"⇒ 打乱后 **{np.mean(null_vals)*100:.4f}% ± {np.std(null_vals)*100:.4f}%**"
            f"（{args.shuffle} 次；全局多数类基线 {r['emo_base_global']*100:.4f}%）")
        # ★ 非空转自检：对照必须真的换数字（第七场坑 9）
        assert len(set(round(v, 12) for v in null_vals)) > 1 or abs(null_vals[0] - r["emo_acc"]) > 1e-9, \
            "❌ 阴性对照与真值完全相同 —— 疑似空转（no-op），立即停下检查"
        say(f"⇒ 真值 {r['emo_acc']*100:.2f}% 与 null 均值差 "
            f"**{(r['emo_acc']-np.mean(null_vals))*100:+.2f} pp**（对照已自检非空转）")
        say("")
    else:
        say("## 3. 阴性对照：本次 `--shuffle 0` 跳过 ⚠️（不得据此写作）")
        say("")

    # ── 落盘（列名与 s2c_emo.csv 完全一致，供 scripts/26 读取）
    row = dict(split="S3_leave_column", **r)
    R = pd.DataFrame([row])[
        ["split", "n", "n_fold", "emo_acc", "emo_base_global", "emo_base_perfold",
         "acc_perfold_weighted"]]
    R.to_csv(CUR / "s3_emo.csv", index=False, encoding="utf-8")
    import hashlib
    _sha = hashlib.sha256((CUR / "s3_emo.csv").read_bytes()).hexdigest()
    say("## 4. 落盘")
    say("")
    say("- `data/curated/s3_emo.csv`（列与 `s2c_emo.csv` 一致；`scripts/26` 读 `emo_acc`）")
    say(f"  - SHA256 = `{_sha}`（本脚本**无未固定随机性**：主臂逐折 HGB 的 `random_state=42` + "
        f"阴性对照 seed={SEED} ⇒ 同参数重跑应得到**同一 SHA256**）")
    say("- `notes/S3_EMO.md`（本文件）")
    say("- `scratch/audit8_code/s3_emo_shuffle.json`（阴性对照原始值）")
    say("")

    (ROOT / "notes" / "S3_EMO.md").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    (ROOT / "scratch" / "audit8_code").mkdir(parents=True, exist_ok=True)
    (ROOT / "scratch" / "audit8_code" / "s3_emo_shuffle.json").write_text(
        json.dumps(dict(real=r["emo_acc"], null=null_vals, n_fold=r["n_fold"],
                        seconds=dt, seed=SEED), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[落盘] data/curated/s3_emo.csv  EMO = {r['emo_acc']*100:.4f}%"
          f"（逐折基线 {r['emo_base_perfold']*100:.4f}%）")
    print(f"[落盘] notes/S3_EMO.md")
    print(f"[总耗时] {time.time()-t_all:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
