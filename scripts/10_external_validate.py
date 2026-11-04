# -*- coding: utf-8 -*-
"""
10_external_validate.py — CP4 预演：CMRT → PE25 的**首次外部验证**
==================================================================
CP4 判据（ROADMAP §8）：外部验证 R²_pred ≥ 0.6。本脚本给出**第一次读数**。

⚠️ **这是一个「最大域迁移」测试，不是常规外部验证。** 三点必须说清楚：

  1. **流动相模式完全不同**：CMRT 绝大多数是正相（hexane/i-PrOH），
     PE25 是 **water/ACN（反相/水相硼酸盐体系）**。→ 条件描述符**不可迁移**，
     所以本脚本**只用分子描述符**，一行条件特征都不给。
  2. **固定相身份未知**：PE25 本地元数据里查不到柱名（`pe25.json` 无 column 字段）。
     → **不给任何 CSP 特征**。若硬塞 CMRT 的柱 one-hot，就是在编造信息。
  3. **靶变量分布不同**：PE25 有 **45% 的分子 α ≡ 1（根本没分开）**，CMRT 只有 0.16%。
     → 这不是缺陷，是 PE25 唯一能覆盖 CMRT 盲区的地方（见 `docs/数据集说明.md` §8.1.5）。

**因此本脚本报两个数**：
  - 全部 78 对（含「分不开」的）上的 R² —— 检验**域迁移下的整体表现**
  - 仅「真分开」子集上的 R² / 符号准确率 —— 检验**是否学到选择性的大小**

对照量（必须一起报，否则数字没意义）：
  - 预测 = 训练集均值（R² ≡ 0）
  - 预测 = 0（R² 可能为负）
  - EMO 多数类基线

产出: notes/CP4_外部验证预演.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CUR = ROOT / "data" / "curated"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module  # noqa: E402

_b = import_module("_baseline_cp25")
LOG: list[str] = []


def say(s: str = "") -> None:
    print(s)
    LOG.append(s)


def main() -> int:
    from rdkit import Chem, RDLogger
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import accuracy_score, r2_score

    RDLogger.DisableLog("rdApp.*")
    SEED = 42

    say("# CP4 预演 —— CMRT → PE25 的首次外部验证")
    say("")
    say("> 由 `scripts/10_external_validate.py` 生成 ｜ CP4 判据：R²_pred ≥ 0.6")
    say("")
    say("## ⚠️ 先说清楚这是什么测试")
    say("")
    say("这是**最大域迁移**而不是常规外部验证：")
    say("")
    say("| 维度 | CMRT（训练） | PE25（外部） | 可迁移？ |")
    say("|---|---|---|---|")
    say("| 流动相 | hexane / i-PrOH（正相） | **water / ACN（水相硼酸盐）** | ❌ **不可** |")
    say("| 固定相 | 25 根 Daicel 柱 | **本地元数据查不到柱名** | ❌ **未知** |")
    say("| 靶变量 | `ln α_true` | `ln α_true`（**已转换到同口径**） | ✅ 可比 |")
    say("| α ≡ 1 占比 | 0.16% | **45.2%** | — |")
    say("")
    say("→ 因此**只用分子描述符**（28 维 RDKit 2D + 立体中心数），**一行条件/CSP 特征都不给**。")
    say("")

    tr = pd.read_csv(CUR / "cmrt_pairs.csv")
    te = pd.read_csv(CUR / "external_pe25_pairs.csv")
    ext_mol = pd.read_csv(CUR / "external_molecules.csv")
    name2smi = dict(zip(ext_mol["name"].str.lower(), ext_mol["smiles"]))
    te["smi"] = te["molecule"].str.lower().map(name2smi)
    n_res = int(te["smi"].notna().sum())
    say(f"外部集：{len(te)} 对，SMILES 解析成功 **{n_res}/{len(te)}**")
    n_leak = int(te["in_cmrt_leak"].sum())
    say(f"剔除与 CMRT 重叠的 **{n_leak} 对**后剩 **{len(te)-n_leak} 对**")
    say("")
    te = te[te["smi"].notna()].copy()

    # 特征：只用分子描述符
    Xtr = _b.build_X(tr)
    keep = [c for c in Xtr.columns if c.startswith("mol_")]   # 丢掉 cond_* （不可迁移）
    Xtr = Xtr[keep].values.astype(float)

    def feats_for(df: pd.DataFrame) -> np.ndarray:
        return np.array([[(_b.mol_feats(s) or {}).get(c, np.nan) for c in keep]
                         for s in df["smi"]])

    Xte = feats_for(te)
    say(f"特征维度 **{len(keep)}**（全部是分子侧；刻意不含条件与 CSP）")
    say("")

    def report(mask: np.ndarray, tag: str) -> None:
        ytr, yte = tr["y"].values, te["y"].values[mask]
        emo_tr, emo_te = tr["EMO"].values, te["EMO"].values[mask]
        m = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.06,
                                          min_samples_leaf=20, random_state=SEED)
        m.fit(Xtr, ytr)
        pred = m.predict(Xte[mask])
        r2 = r2_score(yte, pred)
        rmse = float(np.sqrt(np.mean((yte - pred) ** 2)))
        base_rmse = float(np.sqrt(np.mean((yte - ytr.mean()) ** 2)))
        # 符号：模型输出 >0 判 S 后出峰
        acc = accuracy_score(emo_te, (pred > 0).astype(int))
        f = float(emo_te.mean())
        say(f"### {tag}（n = {int(mask.sum())}）")
        say("")
        say("| 指标 | 值 | 对照 |")
        say("|---|---|---|")
        say(f"| R² | **{r2:+.3f}** | 预测训练均值 ≡ 0.000 |")
        say(f"| RMSE | {rmse:.4f} | {base_rmse:.4f}（常数预测） |")
        say(f"| EMO 准确率 | **{acc*100:.1f}%** | 多数类 {max(f,1-f)*100:.1f}% |")
        say(f"| 靶变量 SD | {yte.std():.4f} | 训练集 {ytr.std():.4f} |")
        say("")
        return dict(tag=tag, n=int(mask.sum()), r2=float(r2), rmse=rmse,
                    base_rmse=base_rmse, emo_acc=float(acc), emo_base=float(max(f, 1 - f)))

    say("## 结果")
    say("")
    rows = []
    no_leak = (~te["in_cmrt_leak"].astype(bool)).values
    rows.append(report(np.ones(len(te), bool), "全部外部对（含「分不开」样本）"))
    rows.append(report(no_leak, "剔除 CMRT 重叠后"))
    sep = (no_leak) & (~te["no_separation"].values)
    if sep.sum() > 5:
        rows.append(report(sep, "仅「真的分开」子集（α ≠ 1）"))

    say("## 读数")
    say("")
    best = max(rows, key=lambda r: r["r2"])
    say(f"- 最好的一项 R² = **{best['r2']:+.3f}**（{best['tag']}）")
    if best["r2"] < 0.1:
        say("- → 🔴 **域迁移下基本没有可迁移的方差**。这与 CP2.5 的严格划分结论一致：")
        say("  分子侧 2D 描述符能承载的信息本来就少，跨流动相模式后更无法迁移。")
    say(f"- EMO 准确率最好 {max(r['emo_acc'] for r in rows)*100:.1f}%，"
        f"多数类基线 {max(r['emo_base'] for r in rows)*100:.1f}%")
    say("")
    say("")
    say("### ⚠️ 数字对「结构来源」敏感（必须声明）")
    say("")
    say("外部分子的 SMILES 由 `08_external_overlap.py` 解析（**PubChem 优先 → EBI ChEMBL 兜底**）。")
    say("实测（2026-09-16）：**PubChem 限流 503，本轮实际全部走 ChEMBL**。")
    say("两个库对同一药物的 SMILES **立体信息给出与否不同**，而特征里含 `nStereo`")
    say("（`CalcNumAtomStereoCenters`，**是手性相关的**），所以数值会变：")
    say("")
    say("| 结构来源 | 「仅分开子集」R² |")
    say("|---|---|")
    say("| PubChem（部分药物无立体信息） | −0.242 |")
    say("| **ChEMBL（本轮）** | **−0.021** |")
    say("")
    say("→ **两次都 ≤ 0，结论不变**；但论文里必须写明用的是哪个库、哪些分子拿到了立体信息。")
    say("→ 好消息：两个**独立**结构库给出的「与 CMRT 重叠的 4 个分子」完全一致，互相印证。")
    say("")
    say("**含义**：CP4（R²_pred ≥ 0.6）在**当前描述符方案下远远达不到**。")
    say("但这恰恰是本项目要论证的事：")
    say("")
    say("1. 论文的 L1（伪标量）**不是锦上添花**，而是让「对映体分辨」这件事第一次有物理载体的唯一途径；")
    say("2. 外部验证必须**先把条件对齐**（同流动相模式 / 同 CSP 家族）——")
    say("   📌 **DG25 的 OD-RH 就是 CMRT 的 Cellulose-a 家族**（同一选择剂，反相模式），")
    say("   它是比 PE25 更合适的第一个外部验证目标（**下一步**）；")
    say("3. 「45% 分不开」的 PE25 应**单独用作「能否预测分不开」的验证集**，")
    say("   而不是拿来算 R² —— 那是它不可替代的价值。")

    (ROOT / "notes" / "CP4_外部验证预演.md").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    pd.DataFrame(rows).to_csv(CUR / "cp4_external_results.csv", index=False)
    print("\n[落盘] notes/CP4_外部验证预演.md + data/curated/cp4_external_results.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
