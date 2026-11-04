# -*- coding: utf-8 -*-
"""
20_external_odrh.py — **条件对齐的外部验证**：CMRT `Cellulose-a` → DG25 `OD-RH`
================================================================================
## 立项理由（2026-09-17 第七场之后）

`10_external_validate.py` 做的 CMRT → PE25 是**最大域迁移**（连柱名都查不到）。
第七场证明「信号不跨实验室迁移」之后，需要的是**把域差距缩到最小**的那种外部验证：

    DG25 的 **OD-RH** 与 CMRT 的 **`Cellulose-a` 家族**是**同一选择剂**
    （纤维素三(3,5-二甲基苯基氨基甲酸酯)：CMRT 里是 `OD` / `OD3` / `ODH` / `IB`）。

⇒ 本脚本是**「同一选择剂 + 不同实验室」**的最小域迁移测试。
**如果连这个都失败，「瓶颈是实验室而不是选择剂」就被钉死了。**

## 三个必须说清的口径
1. **选择剂对齐，流动相不对齐**：CMRT 是正相（hexane/i-PrOH），DG25 是水/ACN。
   且 DG25 只给了 `acidic`/`basic`，**拿不到 i-PrOH 比例** ⇒ **一行条件特征都不给**。
2. **两套训练的对照**：① 在**全部 CMRT** 上训练；② 只在 **CMRT `Cellulose-a`** 上训练
   （② 才是「选择剂对齐」的口径）。
3. **剔除与 CMRT 重叠的分子**（否则「外部验证」退化成内部测试）。

## 产出
- `data/curated/external_odrh_result.csv`
- `notes/CP4b_ODRH条件对齐外部验证.md`

## 用法
```bash
PYTHONIOENCODING=utf-8 python scripts/20_external_odrh.py
PYTHONIOENCODING=utf-8 python scripts/20_external_odrh.py --shuffle 5
```
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CUR = ROOT / "data" / "curated"
LOG: list[str] = []
SEED = 42


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
    ap.add_argument("--shuffle", type=int, default=5)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import r2_score, accuracy_score

    b = load_mod("_b25", ROOT / "scripts" / "_baseline_cp25.py")

    say("# CP4b —— 条件对齐的外部验证：CMRT `Cellulose-a` → DG25 `OD-RH`")
    say("")
    say("> 由 `scripts/20_external_odrh.py` 生成 ｜ 2026-09-17")
    say("")

    # ── 数据
    tr_all = pd.read_csv(CUR / "cmrt_pairs.csv", low_memory=False)
    te = pd.read_csv(CUR / "external_dg25_odrh.csv", low_memory=False)
    say("## 1. 口径对齐检查")
    say("")
    ca = tr_all[tr_all["family"] == "Cellulose-a"]
    say(f"- CMRT `Cellulose-a` 家族 = 柱 {sorted(ca['Column'].unique())}"
        f"（纤维素三(3,5-二甲基苯基氨基甲酸酯)），**{len(ca)}** 行")
    say(f"- DG25 `OD-RH` = 同一选择剂的**反相手性柱版本**，**{len(te)}** 行 / "
        f"{te['conn'].nunique()} 分子")
    say("- ⚠️ **选择剂对齐，流动相不对齐**：CMRT 正相（hexane/i-PrOH）vs DG25 水/ACN；")
    say("  DG25 只给 `acidic`/`basic`，**拿不到 i-PrOH 比例** ⇒ **不给任何条件特征**。")
    say("")

    n_leak = int(te["cmrt_overlap"].sum())
    say(f"- 剔除与 CMRT 重叠的 **{n_leak}** 行后剩 **{len(te)-n_leak}** 行")
    te = te[~te["cmrt_overlap"].astype(bool)].copy()
    say("")

    # ── 特征（只用分子侧）
    Xtr_all = b.build_X(tr_all)
    keep = [c for c in Xtr_all.columns if c.startswith("mol_")]
    Xca = b.build_X(ca)[keep].values.astype(float)
    Xall = Xtr_all[keep].values.astype(float)
    yca = ca["y"].values.astype(float)
    yall = tr_all["y"].values.astype(float)
    say(f"- 特征维度 **{len(keep)}**（**全部是分子侧**；刻意不含条件与 CSP —— 不可迁移）")
    say("")

    # 外部集分子描述符
    def feats_for(smis):
        """返回 (特征矩阵, 与输入**等长**的布尔掩码)。

        ⚠️ 掩码必须与输入等长 —— 初版只 append 成功的行，掩码长度 = 成功数，
        拿去做 `te[mask]` 会 ValueError（长度不匹配）。
        """
        out, ok = [], []
        for s in smis:
            f = b.mol_feats(s) if isinstance(s, str) and s else None
            ok.append(f is not None)
            if f is not None:
                out.append([f.get(c, np.nan) for c in keep])
        return np.array(out, dtype=float), np.array(ok, bool)

    smi = te["smi_flat"].values
    Xte, okm = feats_for(smi)
    te = te[okm].reset_index(drop=True)
    say(f"- 外部集 SMILES 解析成功 **{len(te)}/{len(smi)}** 行")
    say("")

    yte = te["y"].values.astype(float)
    emo_te = (yte > 0).astype(int)
    say(f"- 外部靶变量：y SD = **{yte.std():.4f}**（CMRT 全表 {tr_all['y'].std():.4f}、"
        f"`Cellulose-a` {yca.std():.4f}）")
    say(f"- 「分不开」（α≡1）占比 **{(~te['separated'].astype(bool)).mean()*100:.1f}%**"
        f"（CMRT 仅 0.16%）")
    say("")

    def run(Xtr, ytr, tag, y_te=None, emo_t=None, mask=None):
        """⚠️ `mask` 必须**同时**作用在预测上与靶上 ——
        初版只切了靶（`yte[sep]`）没切预测，长度不匹配直接报错。"""
        pred = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.06,
                                             min_samples_leaf=20, random_state=SEED)
        pred.fit(Xtr, ytr)
        p = pred.predict(Xte)
        if mask is not None:
            p = p[mask]
            y_te = yte[mask] if y_te is None else y_te
            emo_t = emo_te[mask] if emo_t is None else emo_t
        y_te = yte if y_te is None else y_te
        emo_t = emo_te if emo_t is None else emo_t
        r2 = float(r2_score(y_te, p))
        acc = float(accuracy_score(emo_t, (p > 0).astype(int)))
        base = float(max(emo_t.mean(), 1 - emo_t.mean()))
        say(f"| {tag} | {len(ytr)} | **{r2:+.3f}** | {float(np.sqrt(np.mean((y_te-p)**2))):.4f} | "
            f"**{acc*100:.1f}%** | {base*100:.1f}% |")
        return dict(train=tag, n_train=len(ytr), n_test=len(y_te), r2=r2,
                    rmse=float(np.sqrt(np.mean((y_te-p)**2))),
                    emo_acc=acc, emo_base=base, y_sd=float(y_te.std()))

    say("## 2. 主结果（预测全部外部行）")
    say("")
    say("| 训练集 | 训练 n | **R²** | RMSE | EMO | EMO 基线 |")
    say("|---|---|---|---|---|---|")
    rows = [run(Xall, yall, "全部 CMRT（跨家族）"),
            run(Xca, yca, "**CMRT `Cellulose-a`（选择剂对齐）**")]

    # ── 子集：真分开 / 逐流动相
    sep = te["separated"].astype(bool).values
    if sep.sum() > 5:
        say("")
        say("### 仅「真的分开」子集（α ≠ 1，n = %d）" % int(sep.sum()))
        say("")
        say("| 训练集 | 训练 n | **R²** | RMSE | EMO | EMO 基线 |")
        say("|---|---|---|---|---|---|")
        rows.append(run(Xall, yall, "全部 CMRT", mask=sep))
        rows.append(run(Xca, yca, "CMRT `Cellulose-a`", mask=sep))

    for ph in ["acidic", "basic"]:
        mk = (te["phase"].values == ph)
        if mk.sum() <= 10:
            continue
        say("")
        say(f"### 仅 {ph} 流动相（n = {int(mk.sum())}）")
        say("")
        say("| 训练集 | 训练 n | **R²** | RMSE | EMO | EMO 基线 |")
        say("|---|---|---|---|---|---|")
        rows.append(run(Xca, yca, f"CMRT `Cellulose-a`（{ph}）", mask=mk))

    R = pd.DataFrame(rows)
    R.to_csv(CUR / "external_odrh_result.csv", index=False, encoding="utf-8")

    # ── 阴性对照
    say("")
    say("## 3. 阴性对照（**打乱靶变量**，重跑「选择剂对齐」臂）")
    say("")
    say("> ⚠️ **初版这里是个 no-op**：用 `p[mask]` 与 `yte[mask]` 同时重排 ⇒")
    say("> **配对关系没被破坏**，打乱 5 次得到与原始**完全相同**的 R²（−0.802 ± 0.000），")
    say("> 看着「很干净」实则**什么都没检验**。正确做法是**只打乱靶、预测保持原序**。")
    say("")
    mnull = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.06,
                                          min_samples_leaf=20, random_state=SEED)
    mnull.fit(Xca, yca)
    p_null = mnull.predict(Xte)
    accs = []
    for i in range(args.shuffle):
        yp = rng.permutation(yte)
        r2n = float(r2_score(yp, p_null))
        accs.append(r2n)
        say(f"- 打乱第 {i+1} 次：R² = **{r2n:+.3f}**（EMO "
            f"{float(accuracy_score((yp>0).astype(int), (p_null>0).astype(int)))*100:.1f}%）")
    say("")
    say(f"⇒ 打乱后 R² = **{np.mean(accs):+.3f} ± {np.std(accs):.3f}**")

    # ── 判定
    say("")
    say("## 4. ★ 判定")
    say("")
    best = R["r2"].max()
    say(f"- 最好一项 R² = **{best:+.3f}**（CP4 判据 R² ≥ 0.6）")
    if best < 0.1:
        say(f"  → 🔴 **即使选择剂对齐（同一纤维素氨基甲酸酯）、且只做分子侧预测，")
        say(f"    R² 仍 ≤ {best:+.3f}** ⇒ **跨实验室迁移是瓶颈，不是选择剂、不是描述符。**")
        say(f"    这与 `scripts/19` 的「同实验室 r=0.185 / 换实验室 r≈0」互为独立佐证。")
    else:
        say(f"  → 🟡 R² 达 {best:+.3f} ⇒ 选择剂对齐确实改善迁移，但**仍远低于 CP4 的 0.6**。")
    say("")
    say("> ⚠️ **诚实边界**：")
    say(f"> 1. 外部集只有 **{len(te)}** 行 / {te['conn'].nunique()} 个分子 ⇒ 统计功效有限；")
    say("> 2. **流动相模式完全不同**（正相 vs 水/ACN）⇒ 这是「同选择剂 + 跨条件」的复合迁移，")
    say(">    失败**不能单独归因于实验室**；要说清是二者的混合；")
    say(f"> 3. 「分不开」样本占 **{(~te['separated'].astype(bool)).mean()*100:.0f}%**，")
    say(">    会把 R² 拉低（CMRT 几乎没有这类样本）—— 故必须并报「仅真分开」子集。")
    say("")

    notes = ROOT / "notes" / "CP4b_ODRH条件对齐外部验证.md"
    notes.write_text("\n".join(LOG) + "\n", encoding="utf-8")
    print(f"\n[落盘] {notes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
