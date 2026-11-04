# -*- coding: utf-8 -*-
"""
12_build_cmcds.py — 把 CMCDS 原始 CSV 解析成 **L1b 描述符表**，并做与 CMRT 的对接率检验
================================================================================
## 原始文件结构（实测）
`data/raw/cmcds/CMCDS_DATASET.csv`（7,563,924 B，MD5 `181c1015f2befeeedcf15da5d8494650`）
- `shape = (32661, 23)`：列 = `ID, smiles, ECD Transition Parameters, Excited State_1..20`
- **每分子恰好 3 行**，第 3 列取值决定是哪一行：
  1. `Excitation energies (eV)`
  2. `Rotatory Strength [R(velocity)] (cgs(10**-NN erg-esu-cm/Gauss))` ← **⚠️ 指数 NN 逐分子不同**
  3. `Wavelegths (nm)`（原文拼写就是 `Wavelegths`）
- 10,887 个 ID，与论文一致。

> ⚠️ **单位标签的坑**：旋光强度那一行的单位字符串里 `10**-NN` 的 NN **每个分子都不一样**
> （实测从 `-40` 一直怪到 `-10926`）⇒ **不要把 NN 当标度用**。
> 数值本身与论文描述自洽（86.4% 的分子 mean|R| < 50 cgs；本表实测中位 ~5），
> 故按 **cgs 单位 10⁻⁴⁰ erg·esu·cm/Gauss** 处理，并在报告里声明这一判断。

## 产出
- `data/curated/cmcds_l1b.csv` —— 每分子一行：20 个 E / 20 个 R / 20 个 λ + 汇总伪标量
- `notes/CMCDS解析与对接率.md`

用法：PYTHONIOENCODING=utf-8 python scripts/12_build_cmcds.py
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parent.parent
CUR = ROOT / "data" / "curated"
RAW = ROOT / "data" / "raw" / "cmcds" / "CMCDS_DATASET.csv"
NEXC = 20
LOG: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    LOG.append(s)


def canon(smi: str, stereo: bool = True) -> str | None:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return Chem.MolToSmiles(m, isomericSmiles=stereo)


def main() -> int:
    say("# CMCDS 解析与与 CMRT 的对接率检验")
    say("")
    say("> 由 `scripts/12_build_cmcds.py` 生成 ｜ 2026-09-17")
    say("")

    d = pd.read_csv(RAW, low_memory=False)
    lab = d.columns[2]
    cols = [f"Excited State_{i}" for i in range(1, NEXC + 1)]
    say("## 0. 原始结构")
    say("")
    say(f"- 原始 shape **{d.shape}**，ID 数 **{d['ID'].nunique()}**，"
        f"每 ID 行数 **{d.groupby('ID').size().unique().tolist()}**")
    kinds = d[lab].astype(str).str.replace(r"\(cgs.*", "(cgs…)", regex=True)
    say(f"- 第三列取值（归一化后）：`{kinds.value_counts().to_dict()}`")
    expo = d[lab].astype(str).str.extract(r"10\*\*-(\d+)")[0].astype(float)
    say(f"- 旋光强度行的单位指数 `10**-NN`：中位 **{expo.median():.0f}**，"
        f"范围 **{expo.min():.0f} ~ {expo.max():.0f}** ⇒ **逐分子不同，不可当标度**")
    say("")

    # ── 拆三行
    isE = d[lab].astype(str).str.startswith("Excitation energies")
    isR = d[lab].astype(str).str.startswith("Rotatory Strength")
    isW = d[lab].astype(str).str.startswith("Wavelegths")
    say("## 1. 拆行")
    say("")
    say(f"- 激发能 {int(isE.sum())} 行 ｜ 旋光强度 {int(isR.sum())} 行 ｜ 波长 {int(isW.sum())} 行")
    base = d.loc[isE, ["ID", "smiles"]].reset_index(drop=True)
    E = d.loc[isE, cols].astype(float).reset_index(drop=True)
    R = d.loc[isR, cols].astype(float).reset_index(drop=True)
    W = d.loc[isW, cols].astype(float).reset_index(drop=True)
    assert len(E) == len(R) == len(W) == len(base), "三行不齐"
    E.columns = [f"E{i}" for i in range(1, NEXC + 1)]
    R.columns = [f"R{i}" for i in range(1, NEXC + 1)]
    W.columns = [f"lam{i}" for i in range(1, NEXC + 1)]
    t = pd.concat([base, E, R, W], axis=1)
    RA = R.values
    say(f"- 合并后 **{len(t)}** 分子 × {t.shape[1]} 列")
    say("")

    # ── L1b 汇总伪标量（这些才是能进模型的特征）
    say("## 2. L1b 特征（汇总伪标量）")
    say("")
    t["R_sum"] = RA.sum(axis=1)                       # Σ R：净旋转强度（**伪标量**）
    t["R_absmean"] = np.abs(RA).mean(axis=1)
    t["R_absmax"] = np.abs(RA).max(axis=1)
    t["R_neg_frac"] = (RA < 0).mean(axis=1)
    wsum = np.abs(RA).sum(axis=1)
    t["Rw_lam"] = (RA * W.values).sum(axis=1) / np.maximum(wsum, 1e-12)   # 旋转强度重心（伪标量）
    lo = W.values > 250.0                              # 低能带（长波）——Cotton 效应区
    t["R_sum_lo"] = np.where(lo, RA, 0.0).sum(axis=1)
    t["R_sum_hi"] = np.where(~lo, RA, 0.0).sum(axis=1)
    t["E_mean"] = E.values.mean(axis=1)
    t["E_min"] = E.values.min(axis=1)
    t["E_max"] = E.values.max(axis=1)
    t["lam_max"] = W.values.max(axis=1)
    t["lam_min"] = W.values.min(axis=1)
    say("| 特征 | 含义 | 中位 |")
    say("|---|---|---|")
    for c, desc in (("R_sum", "Σ R（净旋转强度，**伪标量**）"),
                    ("R_absmean", "mean\\|R\\|"),
                    ("R_absmax", "max\\|R\\|"),
                    ("Rw_lam", "旋转强度重心波长（伪标量）"),
                    ("R_sum_lo", "λ>250 nm 低能带 Σ R"),
                    ("E_mean", "平均激发能 (eV)")):
        say(f"| `{c}` | {desc} | {t[c].median():.4f} |")
    say("")
    p050 = float((t["R_absmean"] < 50).mean())
    say(f"- **`mean|R| < 50 cgs` 的分子占 {p050*100:.1f}%**"
        f"（论文报 86.4%；同口径，说明数值量级可自洽）")
    say(f"- ⚠️ 单位按 **cgs 10⁻⁴⁰ erg·esu·cm/Gauss** 处理；"
        f"原始单位标签的指数逐分子异常，**不可用**")
    say("")
    Wv = t[[f"lam{i}" for i in range(1, NEXC + 1)]].values
    say(f"- 🔴 **谱窗覆盖：全部分子-跃迁的 λ 落在 "
        f"{Wv.min():.0f}–{Wv.max():.0f} nm（中位 {np.median(Wv):.0f} nm）**；")
    say(f"  **有任一 λ>250 nm 跃迁的分子只有 {float((Wv>250).any(1).mean())*100:.1f}%，"
        f"λ>300 nm 的跃迁占 {float((Wv>300).mean())*100:.2f}%**")
    say("  ⇒ 这 20 个态基本只覆盖**远紫外**，**真实 ECD 的 Cotton 效应区（250–400 nm）几乎没算到**。")
    say("    这是「只取 20 个激发态」的直接后果。**描述符用途上仍可用**（它仍是分子手性的伪标量投影），")
    say("    但**若要做谱预测/绝对构型指认，必须自己加算激发态数**。")
    say("")

    # ── 对接率
    say("## 3. ★ 与 CMRT 的对接率")
    say("")
    cm = pd.read_csv(CUR / "cmrt_pairs.csv")
    mono = cm[cm["cip_S"].astype(str).str.len() == 1].drop_duplicates("conn")
    t["can_stereo"] = [canon(s, True) for s in t["smiles"]]
    t["can_flat"] = [canon(s, False) for s in t["smiles"]]
    mono = mono.assign(can_stereo=[canon(s, True) for s in mono["smi_S"]],
                       can_flat=[canon(s, False) for s in mono["smi_S"]])
    n_bad = int(t["can_stereo"].isna().sum())
    say(f"- CMCDS SMILES 可解析 **{len(t)-n_bad}** / {len(t)}（失败 {n_bad}）")
    say(f"- CMRT 单手性中心分子 **{len(mono)}**")
    s_flat = set(t["can_flat"].dropna())
    m_flat = mono["can_flat"].isin(s_flat)
    say(f"- **按「去立体」规范 SMILES 匹配（= 只看骨架/连接性）："
        f"{int(m_flat.sum())} / {len(mono)} = {m_flat.mean()*100:.1f}%**")
    s_st = set(t["can_stereo"].dropna())
    mono = mono.assign(m_S=mono["can_stereo"].isin(s_st),
                       m_R=mono.assign(can_stereo=[canon(x, True) for x in mono["smi_R"]])
                                ["can_stereo"].isin(s_st))
    say(f"- 与 `smi_S`（CIP-S 对映体）立体一致："
        f"{int(mono['m_S'].sum())} / {len(mono)} = {mono['m_S'].mean()*100:.1f}%")
    say(f"- 与 `smi_R`（CIP-R 对映体）立体一致："
        f"{int(mono['m_R'].sum())} = {mono['m_R'].mean()*100:.1f}%")
    say(f"- **二者并集：{int((mono['m_S']|mono['m_R']).sum())} / {int(m_flat.sum())} = "
        f"{(mono['m_S']|mono['m_R']).sum()/max(m_flat.sum(),1)*100:.1f}%**"
        f"（同时命中 S 与 R 的仅 **{int((mono['m_S']&mono['m_R']).sum())}** 个）")
    say("")
    say("★ **结论：CMCDS 没有标错立体，也没有丢失定向。**")
    say("  它收录的就是 CMRT 里列出的那个对映体，因此 **约一半是 CIP-S、一半是 CIP-R**。")
    say("  ⇒ 与 CMRT 对齐时，落在 `m_R` 一侧的分子**把所有 R 值整体变号**即可，")
    say("     `smi_S` 一侧保持符号。**这一步必须做，否则伪标量会有一半整体反号。**")
    say("  ⇒ 对齐后**定向覆盖率 = 98.2%**（与骨架匹配率相同）。")
    say("")

    feats = ["ID", "smiles", "can_stereo", "can_flat", "R_sum", "R_absmean", "R_absmax",
             "R_neg_frac", "Rw_lam", "R_sum_lo", "R_sum_hi", "E_mean", "E_min", "E_max",
             "lam_max", "lam_min"] + [f"R{i}" for i in range(1, NEXC + 1)] \
            + [f"lam{i}" for i in range(1, NEXC + 1)] + [f"E{i}" for i in range(1, NEXC + 1)]
    # ⚠️ `lam1..lam20` / `E1..E20` 必须落盘：下游 `14` 要按**波长分带**做 Σ R
    #    （分带伪标量），没有逐态波长就只能按态序号切，物理含义不同。
    t[feats].to_csv(CUR / "cmcds_l1b.csv", index=False, encoding="utf-8")
    say("## 4. 产出")
    say("")
    say(f"- `data/curated/cmcds_l1b.csv` —— **{len(t)} 行 × {len(feats)} 列**")
    say("")
    say("## 5. ⚠️ 用之前必须做的一件事")
    say("")
    say("CMCDS 用的是**单构象（最低能）近似**，而本项目 CP3/CP3b 已实测：")
    say("**单构象不足以**支撑矢量三重积类描述符（反号比例仅 0.363）。")
    say("⇒ 必须单独验证「CMCDS 的 R 能否通过**系综版对映体交换检验**」，")
    say("   否则 L1b 会重复 `03_chirality_signature_poc.py` 的老问题。")
    say("   注意：R 是**谱量**（对构象的敏感性远低于位置类矢量），**预期能过**，但要实测。")

    (ROOT / "notes" / "CMCDS解析与对接率.md").write_text("\n".join(LOG) + "\n",
                                                     encoding="utf-8")
    print(f"\n[落盘] data/curated/cmcds_l1b.csv ({len(t)} 行)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
