# -*- coding: utf-8 -*-
"""
18_s2c_emo.py — **真·留一文献（S2c）下的洗脱顺序准确率（EMO）**
================================================================================
## 立项理由

`notes/留一文献真划分_S2c.md`（`scripts/15`）只跑了**回归**臂（R²），
**没有跑分类臂**。而「洗脱顺序准确率」是本项目对外的**最扎眼数字**：

    前人的常用划分（随机 8:2）EMO = 73.8%
    「留一批文献」（S2b）        EMO = 53.5%
    真·留一文献（S2c）           EMO = **？**   ← 本脚本要补的就是这个数

⚠️ **不许拍脑袋填 52%**。`S2c` 的 `r>0 篇占比 = 52%` 是「**逐篇相关系数为正的比例**」，
与「逐样本分类准确率」**完全不是一回事**（一个是篇级、一个是行级）。
本脚本按与 `_baseline_cp25` 完全相同的分类器与配置实测。

## 口径
- 分类器：**本文件不写死超参**，一律取自 `_baseline_cp25.random_forest_pair()`。
  （历史上这里写死过 `max_iter=200`，与加载器实际值漂移；第八场审计后只保留「同配置」这个事实。）
- 折：`data/curated/splits/S2c_leave_doi.csv`（按 `row_id` 对齐）
  ⚠️ **折数不写死在文案里**。划分文件实测 **530 折**；本节历史上写死过 **523**
  （那是「描述符覆盖后可评估的文献数」这一**另一种口径**，不是划分折数）——
  三个数（530 划分 / 523 可评估文献 / 428 逐折统计覆盖）的定论见
  `notes/第八场_数字审计.md` §2。文案里一律从数据取。
- 报三个基线，缺一不可：
  1. **全局多数类**（全表 EMO 的多数类占比）
  2. **逐折多数类**（oracle 知道每折的类别比例——**上界**）
  3. **阴性对照**（打乱标签后重跑同一协议）

## 产出
- `data/curated/s2c_emo.csv`
- `notes/S2c_EMO.md`

## 用法
```bash
PYTHONIOENCODING=utf-8 python scripts/18_s2c_emo.py
PYTHONIOENCODING=utf-8 python scripts/18_s2c_emo.py --shuffle 2   # 阴性对照重复次数
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


def emo_oof(X, emo, folds) -> tuple[np.ndarray, dict]:
    """逐折训练分类器 → OOF 类别预测。返回 (pred, 统计)"""
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
    ap.add_argument("--shuffle", type=int, default=3)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    t_all = time.time()

    global b25
    b25 = load_mod("_b25", ROOT / "scripts" / "_baseline_cp25.py")

    p = pd.read_csv(CUR / "cmrt_pairs.csv", low_memory=False)
    p["row_id"] = np.arange(len(p))
    # ⚠️ **折号在 `cmrt_pairs_with_folds.csv`，不在 `cmrt_pairs.csv`。**
    #   初版只读 `cmrt_pairs`，下面的对照循环 `if col not in p.columns: continue`
    #   **静默跳过全部四个对照划分**，产出一张只有表头、没有行的空表（`notes/S2c_EMO.md` §2）。
    #   ⇒ 现在显式合并 + 缺列就报错，不许再静默跳过。
    fw = pd.read_csv(CUR / "cmrt_pairs_with_folds.csv", low_memory=False)
    FOLD_COLS = ["S1_random_fold", "S2_group_fold", "S2b_paper_fold", "S4_murcko_fold"]
    missing = [c for c in FOLD_COLS if c not in fw.columns]
    if missing:
        raise SystemExit(f"❌ 折列缺失：{missing}（应来自 cmrt_pairs_with_folds.csv）")
    if len(fw) != len(p):
        raise SystemExit(f"❌ 行数不一致：pairs={len(p)} folds={len(fw)}")
    for c in FOLD_COLS:
        p[c] = fw[c].values
    sp = pd.read_csv(CUR / "splits" / "S2c_leave_doi.csv")
    # ⚠️ 划分按 row_id 对齐，不是 pair_id（见 PROJECT_STATE 坑 32）
    fold_map = sp.set_index("row_id")["fold"]
    p["fold"] = p["row_id"].map(fold_map)
    assert p["fold"].notna().all(), "S2c 划分未覆盖全部行"

    X0 = b25.build_X(p).astype(float)
    emo = p["EMO"].values.astype(int)

    say("# 真·留一文献（S2c）下的洗脱顺序准确率（EMO）")
    say("")
    say("> 由 `scripts/18_s2c_emo.py` 生成 ｜ 2026-09-17")
    say("")
    say(f"样本 **{len(p)}** 行 / **{p['fold'].nunique()}** 折 ｜ "
        f"全局 EMO 正类占比 = {emo.mean():.4f}")
    say("")

    # ── S2c
    # ⚠️ 折数**从数据取**，不写死（项目里已因写死 523 而与本文件产出不一致）
    say(f"## 1. S2c（真·留一文献：划分 {p['fold'].nunique()} 折）")
    say("")
    t0 = time.time()
    _, r = emo_oof(X0, emo, p["fold"].values)
    say(f"| 指标 | 值 |")
    say("|---|---|")
    say(f"| **EMO 准确率** | **{r['emo_acc']*100:.2f}%** |")
    say(f"| 全局多数类基线 | {r['emo_base_global']*100:.2f}% |")
    say(f"| 逐折多数类基线（oracle，**上界**） | {r['emo_base_perfold']*100:.2f}% |")
    say(f"| 相对全局基线的增益 | {(r['emo_acc']-r['emo_base_global'])*100:+.2f} 个百分点 |")
    say(f"| 折数 / 样本 | {r['n_fold']} / {r['n']} |")
    say("")
    print(f"（S2c EMO 耗时 {time.time()-t0:.0f} s）", flush=True)

    # ── 其它划分对照（同一分类器、同一特征）
    say("## 2. 对照：其它划分（同分类器 / 同特征 L0）")
    say("")
    say("> ⚠️ **本表的数字与 `notes/CP2.5_基线预演.md` 不可逐字对照**：")
    say("> 两者分类器配置/折的产生方式不同（那里 S1 是单次 80/20，这里按 2 折 OOF）。")
    say("> **本表的价值是「同一分类器、同一特征下的单调梯度」**，不是绝对水平。")
    say("> 可对上的是 S2b：**这里 53.94% vs CP2.5 的 53.5%** —— 一致 ✓")
    say("")
    say("| 划分 | 折数 | EMO 准确率 | 全局多数类 | 逐折多数类(oracle) | 增益 |")
    say("|---|---|---|---|---|---|")
    rows = [dict(split="S2c_leave_doi", **r)]
    for name, col in [("S1_random", "S1_random_fold"), ("S2_group", "S2_group_fold"),
                      ("S2b_leave_paper", "S2b_paper_fold"), ("S4_murcko", "S4_murcko_fold")]:
        if col not in p.columns:      # 上面已保证存在；真缺了就报错，不静默跳过
            raise SystemExit(f"❌ 缺列 {col}")
        t0 = time.time()
        _, rr = emo_oof(X0, emo, p[col].values)
        rows.append(dict(split=name, **rr))
        say(f"| `{name}` | {rr['n_fold']} | **{rr['emo_acc']*100:.2f}%** | "
            f"{rr['emo_base_global']*100:.2f}% | {rr['emo_base_perfold']*100:.2f}% | "
            f"{(rr['emo_acc']-rr['emo_base_global'])*100:+.2f}pp |")
        print(f"（{name} 耗时 {time.time()-t0:.0f} s）", flush=True)
    say("")

    # ── 阴性对照
    say("## 3. 阴性对照（打乱标签，同一 S2c 协议）")
    say("")
    accs = []
    for i in range(args.shuffle):
        emo_sh = rng.permutation(emo)
        _, rs = emo_oof(X0, emo_sh, p["fold"].values)
        accs.append(rs["emo_acc"])
        say(f"- 第 {i+1} 次打乱：EMO = **{rs['emo_acc']*100:.2f}%**")
    say("")
    if accs:
        say(f"⇒ 打乱后 **{np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%**")
        say(f"⇒ S2c 的 {r['emo_acc']*100:.2f}% 比阴性对照高 "
            f"**{(r['emo_acc']-np.mean(accs))*100:+.2f} 个百分点**")
        # ★ 第八场修复（figures-fixer 的发现 D1）：阴性对照此前**只写在 markdown 里**，
        #   图 1 的灰带因此没有任何 CSV 来源。现在落盘，供图与门禁引用。
        sh = pd.DataFrame({"trial": np.arange(1, len(accs) + 1), "emo_acc": accs})
        sh.to_csv(CUR / "s2c_emo_shuffle.csv", index=False, encoding="utf-8")
        say("")
        say(f"> 阴性对照逐次结果已落盘：`data/curated/s2c_emo_shuffle.csv`"
            f"（{len(accs)} 次，seed={SEED}）")
    else:
        say("⇒ 本次 `--shuffle 0`：跳过阴性对照（未落盘 shuffle CSV）")
    say("")

    R = pd.DataFrame(rows)
    R.to_csv(CUR / "s2c_emo.csv", index=False, encoding="utf-8")

    say("## 4. ★ 判定")
    say("")
    gain = (r["emo_acc"] - r["emo_base_global"]) * 100
    if gain > 1.0:
        say(f"- ✅ **S2c 下仍有一点可学的排序信号**：{r['emo_acc']*100:.2f}% vs 基线 "
            f"{r['emo_base_global']*100:.2f}%（**+{gain:.2f}pp**）")
        say(f"  ⇒ 可写成「**信号真实存在但极弱**」，用于反驳「模型完全没学到东西」。")
    else:
        say(f"- 🔴 **S2c 下几乎没有可学分排序信号**：增益仅 {gain:+.2f}pp")
    say(f"- ⚠️ 论文口径：**EMO 必须与「逐折多数类基线」并报**——")
    say(f"  各文献的类别比例不同（有的文献几乎全是同一洗脱顺序），")
    say(f"  只报全局基线会低估难度。")
    say("")

    notes = ROOT / "notes" / "S2c_EMO.md"
    notes.write_text("\n".join(LOG) + "\n", encoding="utf-8")
    print(f"\n[落盘] {notes}")
    print(f"[落盘] data/curated/s2c_emo.csv")
    if accs:
        print(f"[落盘] data/curated/s2c_emo_shuffle.csv（阴性对照 {len(accs)} 次）")
    print(f"[总耗时] {time.time()-t_all:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
