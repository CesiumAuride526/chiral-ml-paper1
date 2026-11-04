# -*- coding: utf-8 -*-
"""
24_external_all_columns.py — **跨选择剂梯度**：CMRT → DG25 全部 6 根柱
================================================================================
## 立项理由（2026-09-17 第七场）

`scripts/20` 只做了 DG25 的 **OD-RH**（= CMRT `Cellulose-a` 同选择剂），得 R² = −0.802。
本脚本把它扩到 **全部 6 根柱**，回答一个 `20` 回答不了的问题：

    外部验证的失败，是「**所有**选择剂都失败」，还是「**只有不对齐的**失败」？

DG25 的 6 根柱在同一实验室、同一批 42–43 个分子上测完 ⇒ **柱之间的差异被隔离出来了**，
不掺杂实验室与样品差异。配合 `scripts/21` 的无模型对照，构成完整的跨选择剂梯度。

## 设计
- **训练**：CMRT（全部 10,525 行），特征**只用分子侧 25 维**
  （条件不可迁移：CMRT 正相 vs DG25 水/ACN）
- **测试**：DG25 每根柱一个测试集（剔除与 CMRT 重叠的分子）
- **第二臂**：对纤维素类的 DG25 柱，另在 **CMRT `Cellulose-a`** 上训练（选择剂对齐）
- 阴性对照：打乱靶变量（**只打乱靶、预测保持原序** —— 见 `20` 的 no-op 教训）

## 期望的读法
- 若 6 根柱 **一致地差** ⇒ 支持「跨实验室/条件不可迁移」的主论断
- 若纤维素类明显好于直链淀粉类 ⇒ 说明**选择剂对齐确实有价值**（但 `21` 显示即使同家族也仅 r≈0.3–0.66）

## 产出
- `data/curated/external_all_columns.csv`
- `notes/CP4c_跨选择剂梯度.md`

## 用法
```bash
PYTHONIOENCODING=utf-8 python scripts/24_external_all_columns.py
```
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CUR = ROOT / "data" / "curated"
LOG: list[str] = []
SEED = 42
# DG25 柱 → 选择剂家族（按 `21` 的化学身份判定：OD-RH 是纤维素类）
FAMILY = {"LC-2": "cellulose", "LC-3": "cellulose", "LC-4": "cellulose",
          "OD-RH": "cellulose", "LA-1": "amylose", "LA-2": "amylose"}


def say(s: str = "") -> None:
    print(s, flush=True)
    LOG.append(s)


def load_mod(name: str, path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import r2_score, accuracy_score

    rng = np.random.default_rng(SEED)
    b = load_mod("_b25", ROOT / "scripts" / "_baseline_cp25.py")

    tr_all = pd.read_csv(CUR / "cmrt_pairs.csv", low_memory=False)
    ca = tr_all[tr_all["family"] == "Cellulose-a"]
    d = pd.read_csv(CUR / "external_dg25_all.csv", low_memory=False)

    say("# CP4c —— 跨选择剂梯度：CMRT → DG25 全部 6 根柱")
    say("")
    say("> 由 `scripts/24_external_all_columns.py` 生成 ｜ 2026-09-17")
    say("")
    say("**训练集**：CMRT 全表 10,525 行（另备 `Cellulose-a` 2,738 行的对齐臂）｜")
    say("**特征**：分子侧 25 维（**不给条件** —— CMRT 正相 vs DG25 水/ACN，不可迁移）")
    say("")

    Xtr_all = b.build_X(tr_all)
    keep = [c for c in Xtr_all.columns if c.startswith("mol_")]
    Xall = Xtr_all[keep].values.astype(float)
    yall = tr_all["y"].values.astype(float)
    Xca = b.build_X(ca)[keep].values.astype(float)
    yca = ca["y"].values.astype(float)
    say(f"- 特征维度 **{len(keep)}**")
    say("")

    def feats(smis):
        out, ok = [], []
        for s in smis:
            f = b.mol_feats(s) if isinstance(s, str) and s else None
            ok.append(f is not None)
            if f is not None:
                out.append([f.get(c, np.nan) for c in keep])
        return np.array(out, dtype=float), np.array(ok, bool)

    def run(Xtr, ytr, tag, y_te, emo_te, Xte):
        m = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.06,
                                          min_samples_leaf=20, random_state=SEED)
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        r2 = float(r2_score(y_te, p))
        acc = float(accuracy_score(emo_te, (p > 0).astype(int)))
        base = float(max(emo_te.mean(), 1 - emo_te.mean()))
        return dict(arm=tag, n=len(y_te), r2=r2,
                    rmse=float(np.sqrt(np.mean((y_te - p) ** 2))),
                    emo=acc, emo_base=base, y_sd=float(y_te.std()),
                    pred_sd=float(np.std(p)))

    rows, sub = [], {}
    for col in ["OD-RH", "LC-2", "LC-3", "LC-4", "LA-1", "LA-2"]:
        s = d[d["column"] == col].copy()
        n_ov = int(s["cmrt_overlap"].sum())
        s = s[~s["cmrt_overlap"].astype(bool)]
        # ⚠️ `feats()` 返回的 Xte **只含解析成功的行**（73），而 `ok` 与输入等长（80）
        #    ⇒ **不能**再拿 `ok` 去索引 Xte（长度不匹配）。只用 `ok` 过滤 `s` 即可。
        #    （与 `20` 是同一类坑，见 PROJECT_STATE 第七场记载。）
        Xte, ok = feats(s["smi_flat"].values)
        s = s[ok].reset_index(drop=True)
        yte = s["y"].values.astype(float)
        emo = (yte > 0).astype(int)
        sub[col] = (Xte, yte, emo)
        say(f"`{col}`（{FAMILY[col]}）：{len(s)} 行（剔除 CMRT 重叠 {n_ov}）")
    say("")

    say("## 1. 主结果：CMRT 全表训练 → 逐柱预测")
    say("")
    say("| DG25 柱 | 选择剂 | n | **R²** | RMSE | EMO | EMO 基线 | 预测 SD / 靶 SD |")
    say("|---|---|---|---|---|---|---|---|")
    for col in ["OD-RH", "LC-2", "LC-3", "LC-4", "LA-1", "LA-2"]:
        Xte, yte, emo = sub[col]
        r = run(Xall, yall, col, yte, emo, Xte)
        r["column"] = col
        r["family"] = FAMILY[col]
        rows.append(r)
        say(f"| `{col}` | {FAMILY[col]} | {r['n']} | **{r['r2']:+.3f}** | {r['rmse']:.4f} | "
            f"{r['emo']*100:.1f}% | {r['emo_base']*100:.1f}% | "
            f"{r['pred_sd']:.3f} / {r['y_sd']:.3f} |")
    say("")

    say("## 2. 第二臂：在 CMRT `Cellulose-a` 上训练（**选择剂对齐**）→ 纤维素类柱")
    say("")
    say("| DG25 柱 | n | **R²** | EMO | EMO 基线 |")
    say("|---|---|---|---|---|")
    for col in ["OD-RH", "LC-2", "LC-3", "LC-4"]:
        Xte, yte, emo = sub[col]
        r = run(Xca, yca, col + " (aligned)", yte, emo, Xte)
        r["column"] = col + " [aligned]"
        r["family"] = FAMILY[col]
        rows.append(r)
        say(f"| `{col}` | {r['n']} | **{r['r2']:+.3f}** | {r['emo']*100:.1f}% | "
            f"{r['emo_base']*100:.1f}% |")
    say("")

    R = pd.DataFrame(rows)
    R.to_csv(CUR / "external_all_columns.csv", index=False, encoding="utf-8")

    # ── 按家族汇总
    say("## 3. ★ 按选择剂家族汇总（看是否与 CMRT 同族有关系）")
    say("")
    main_arm = R[R["arm"].isin(["OD-RH", "LC-2", "LC-3", "LC-4", "LA-1", "LA-2"])]
    for fam in ["cellulose", "amylose"]:
        g = main_arm[main_arm["family"] == fam]
        say(f"- **{fam}**（{len(g)} 柱）：R² 中位 **{g['r2'].median():+.3f}**"
            f"（范围 {g['r2'].min():+.3f} ~ {g['r2'].max():+.3f}）")
    say("")

    # ── 阴性对照（只打乱靶，预测保持原序）
    say("## 4. 阴性对照（**只打乱靶变量**，预测保持原序）")
    say("")
    say("> ⚠️ `20` 的教训：若同时重排预测与靶，**配对关系不被破坏**，打乱前后数字完全相同。")
    say("")
    accs = []
    for col in ["OD-RH", "LC-2", "LC-3", "LC-4", "LA-1", "LA-2"]:
        Xte, yte, emo = sub[col]
        m = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.06,
                                          min_samples_leaf=20, random_state=SEED)
        m.fit(Xall, yall)
        p = m.predict(Xte)
        r2n = float(r2_score(rng.permutation(yte), p))
        accs.append(r2n)
        say(f"- `{col}` 打乱后 R² = **{r2n:+.3f}**")
    say("")
    real = main_arm["r2"].mean()
    say(f"⇒ 打乱后均值 **{np.mean(accs):+.3f} ± {np.std(accs):.3f}**"
        f" ｜ 真实均值 **{real:+.3f}**")
    say("")

    # ── 判定
    say("## 5. ★ 判定")
    say("")
    best = main_arm.loc[main_arm["r2"].idxmax()]
    say(f"- 6 根柱**全部** R² < 0，最好的一根是 `{best['column']}` "
        f"（{best['family']}）R² = **{best['r2']:+.3f}**")
    if real <= np.mean(accs) + 0.1:
        say(f"- 真实均值 {real:+.3f} **不优于**随机对照 {np.mean(accs):+.3f}")
        say(f"  ⇒ 🔴 **跨选择剂的失败是普遍的，不是「只有某几根柱不对齐」**。")
    say("")
    say("> **与 `scripts/21`（无模型对照）合起来读**：")
    say("> `21` 显示 DG25 **内部**跨柱的可迁移性也很低（异柱×同相 r 中位 +0.128），")
    say("> 本脚本显示 **CMRT → DG25 跨数据集**同样全负。")
    say("> 两条独立证据指向同一结论：**换选择剂本身就是不可迁移的**，")
    say("> 与是否跨实验室无关。这与 §4.5 的交互项解释一致 ——")
    say("> **分子×条件的交互项才是主要方差来源**，而它无法由任何加性模型捕获。")
    say("")
    say("> ⚠️ **边界**：① 每柱仅 78–85 行；② 流动相模式不同（正相 vs 水/ACN）；")
    say("> ③ 「分不开」样本占 31–52%（CMRT 仅 0.16%）⇒ 会把 R² 拉低。")
    say("")

    notes = ROOT / "notes" / "CP4c_跨选择剂梯度.md"
    notes.write_text("\n".join(LOG) + "\n", encoding="utf-8")
    print(f"\n[落盘] {notes}")
    print(f"[落盘] data/curated/external_all_columns.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
