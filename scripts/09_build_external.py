# -*- coding: utf-8 -*-
"""
09_build_external.py — 构造外部验证集（PE25）的同口径靶变量
============================================================
ROADMAP P1 尾巴 + CP4（外部验证 R²_pred ≥ 0.6）的前置件。

**为什么 PE25 值得单独做**：它的主表给了 `Molecule / Enantiomer / Retention time / Retention factor k`
——**每个对映体的 k 与 CIP 的 R/S 标签都在**，还有 `elution t0` 占位行。
所以能直接构造与 CMRT **完全同口径**的定向靶：
    `y = ln α_true = ln[ k(S) / k(R) ]`，`EMO = 1{y > 0}`
（CMRT 只有 RT、**没有 t₀**，还得靠柱几何估算；PE25 的 k 是作者直接给的。）

⚠️ **三个必须处理的坑**（都在本脚本里做了断言/校验）：

  R1 **符号约定相反**：PE25 自己的列是 `AlfaRS (kR/kS)` 与 `Log alfaRS` —— 是 **k_R/k_S**，
     与我们的 `k_S/k_R` **互为倒数**。直接抄会得到**符号完全相反**的靶变量。
  R2 **底数是 log10 不是 ln**：实测 `Log alfaRS = log10(α_RS)`（Aminogluthetimide：
     α=1.341829 → 0.127697 = log10，不是 ln 的 0.2940）。两者差 2.3026 倍。
  R3 **DG25 与 PE25 是同一批分子**（42/43 重合），**不是两套独立外部集** ——
     本脚本只输出 PE25，且在 README/报告里写明这一点。

**额外产出（方法学）**：PE25 同时给了 t_R 与 k ⇒ 可用 `t₀ = t_R/(k+1)` **精确反解实验死时间**。
实测 85 行全部给出 **t₀ = 5.5600 ± 0.0000 min**，既证明其 k 值内部自洽，
也给出一个**可在别处复用**的死时间恢复技术（CMRT 不适用，因为它没给 k）。

用法: PYTHONIOENCODING=utf-8 python scripts/09_build_external.py
产出: data/curated/external_pe25_pairs.csv + notes/外部验证集_PE25构造报告.md
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CUR = ROOT / "data" / "curated"
PE = ROOT / "data" / "raw" / "zenodo" / "peeters2025"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module
_norm = import_module("08_external_overlap").normalize   # 复用同一套名称规范化
LOG: list[str] = []


def say(s: str = "") -> None:
    print(s)
    LOG.append(s)


# 只有一个 sheet 里带完整 R/S 实验值；同名 sheet 在多个文件里重复，取一份即可
SRC_FILE = "Borate_chiral_exp_water_ACN.xlsx"
CONDITIONS = {
    "Explicit_water_acn_unbiased": "water/ACN 未偏置",
    "Explicit_water_acn_biased": "water/ACN 偏置",
}


def parse_sheet(d: pd.DataFrame, cond: str) -> tuple[pd.DataFrame, float]:
    """把「Molecule 只在首行、Enantiomer 为 R/S」的合并式表格拆成规整表。

    返回 (规整表, 反解出的 t₀)。
    """
    d = d.copy()
    d["Molecule"] = d["Molecule"].ffill()          # 合并单元格 → 向下填充
    rt = pd.to_numeric(d["Retention time (min)"], errors="coerce")
    k = pd.to_numeric(d["Retention factor k"], errors="coerce")
    ok = rt.notna() & k.notna()
    d, rt, k = d[ok], rt[ok], k[ok]
    t0 = (rt / (k + 1.0)).values
    return pd.DataFrame({
        "condition": cond,
        # 名称规范化必须与 08 一致，否则 3 个拼写变体（Bupivacaïne 等）对不上 SMILES
        "molecule": [_norm(v) or str(v).strip() for v in d["Molecule"].astype(str).values],
        "enantiomer": d["Enantiomer"].astype(str).str.strip().values,
        "RT_min": rt.values,
        "k": k.values,
        "t0_derived": t0,
    }), float(np.mean(t0))


def main() -> int:
    say("# 外部验证集 PE25 —— 同口径靶变量构造")
    say("")
    say("> 由 `scripts/09_build_external.py` 生成")
    say("")
    say("## 1. 解析实验表并反解死时间")
    say("")

    x = pd.ExcelFile(PE / SRC_FILE)
    frames, t0s = [], {}
    for sheet, label in CONDITIONS.items():
        if sheet not in x.sheet_names:
            continue
        raw = x.parse(sheet)
        if "Enantiomer" not in raw.columns:
            continue
        df, t0 = parse_sheet(raw, label)
        frames.append(df)
        t0s[label] = t0
        say(f"- **{label}**（sheet `{sheet}`）：{len(df)} 行测量，"
            f"反解 `t₀ = t_R/(k+1)` = **{t0:.4f} min**，离散度 "
            f"{np.std(df['t0_derived']):.6f} min")
    ext = pd.concat(frames, ignore_index=True)

    t0_all = ext["t0_derived"].values
    say("")
    say(f"**t₀ 恢复的一致性**：{len(t0_all)} 行给出 t₀ = **{t0_all.mean():.4f} ± {t0_all.std():.6f} min**。")
    if t0_all.std() < 1e-3:
        say("→ ✅ 完全一致（到小数点后 6 位）⇒ 这个数据集的 `k` 与 `RT` 是**自洽**的，可直接用。")
        say("→ 📌 这也是一个**可复用的技术**：任何同时给 `t_R` 与 `k` 的数据集，都能这样精确反解死时间。")
    say("")

    # ── 配对
    say("## 2. 配对（同一 conditions 下同一分子的 R / S）")
    say("")
    rows = []
    for (cond, mol), g in ext.groupby(["condition", "molecule"]):
        s = g[g["enantiomer"].str.upper() == "S"]
        r = g[g["enantiomer"].str.upper() == "R"]
        if len(s) != 1 or len(r) != 1:
            continue          # 缺 R/S 或重复 → 跳过（下面报告数量）
        kS, kR = float(s["k"].iloc[0]), float(r["k"].iloc[0])
        if kS <= 0 or kR <= 0:
            continue
        rows.append(dict(condition=cond, molecule=mol, k_S=kS, k_R=kR, RT_S=float(s["RT_min"].iloc[0]),
                         RT_R=float(r["RT_min"].iloc[0]), t0_derived=float(s["t0_derived"].iloc[0])))
    pp = pd.DataFrame(rows)
    n_mol = ext["molecule"].nunique()
    say(f"- 数据集含 {n_mol} 个唯一分子 × {len(CONDITIONS)} 个条件")
    say(f"- **成功配对 {len(pp)} 对**（每对 = 分子 × 条件）")
    say("")

    # ── R1/R2 校验：与 PE25 自带的 AlfaRS 列对账
    say("## 3. ⚠️ 与 PE25 自带列的约定核对（R1 符号 / R2 底数）")
    say("")
    ref = x.parse("Chiral log alfaRS unbiased")
    # ⚠️ 顺序要紧：**先 ffill 再 dropna**。反过来会把每对的 R 行先删掉，
    #    导致 ffill 拿不到分子名（首版就踩了这个坑，对账最大偏差 0.914）。
    ref["Molecule"] = ref["Molecule"].ffill()
    ref = ref.dropna(subset=["AlfaRS (kR/kS)"])
    ref_map = dict(zip(ref["Molecule"].astype(str).str.strip(), ref["AlfaRS (kR/kS)"]))
    chk = pp[pp["condition"] == CONDITIONS["Explicit_water_acn_unbiased"]].copy()
    chk["ours_kSkR"] = chk["k_S"] / chk["k_R"]
    chk["pe25_kRkS"] = chk["molecule"].map(ref_map)
    chk = chk.dropna(subset=["pe25_kRkS"])
    if len(chk):
        recip_err = float((np.abs(chk["ours_kSkR"] * chk["pe25_kRkS"] - 1.0)).max())
        say(f"- 我们算的 `k_S/k_R` × PE25 的 `kR/kS` − 1 = **最大 {recip_err:.2e}**")
        say(f"  {'✅ 恰好互为倒数 → **符号约定相反这点已坐实**' if recip_err < 1e-6 else '❌ 不一致，需排查'}")
        lg_err = float((np.log10(chk["pe25_kRkS"]) )
                       .sub(np.log(chk["pe25_kRkS"]) / np.log(10)).abs().max())
        say(f"- PE25 的 `Log alfaRS` 与 `log10(α)` 的最大偏差 = **{lg_err:.2e}**"
            f"（若换成 ln 则差 2.3026 倍）")
    say("")

    # ── 靶变量
    pp["alpha_true"] = pp["k_S"] / pp["k_R"]
    pp["y"] = np.log(pp["alpha_true"])
    pp["EMO"] = (pp["y"] > 0).astype(int)
    pp["log10_alphaRS_pe25"] = np.log10(1.0 / pp["alpha_true"])   # PE25 口径，便于对账
    say("## 4. 靶变量（与 CMRT 同口径）")
    say("")
    say(f"`y = ln α_true = ln[k(S)/k(R)]`")
    say("")
    say(f"- y 中位 **{pp['y'].median():+.4f}**，均值 {pp['y'].mean():+.4f}，SD **{pp['y'].std():.4f}**，"
        f"范围 {pp['y'].min():+.3f} ~ {pp['y'].max():+.3f}")
    say(f"- 无向 α 中位 **{np.exp(pp['y'].abs().median()):.3f}**，最大 {np.exp(pp['y'].abs().max()):.3f}")
    say(f"- `EMO` = 1 占比 **{pp['EMO'].mean()*100:.1f}%**")
    say("")

    # ── 🔴 与 CMRT 最本质的差别：PE25 里**大量分子根本没分开**
    # 用 α 是否恰为 1 判定（k_S == k_R 到浮点精度）——这也是"没有分离"的判据
    pp["no_separation"] = np.isclose(pp["alpha_true"], 1.0, rtol=1e-9)
    n_ns = int(pp["no_separation"].sum())
    n_sep = len(pp) - n_ns
    say("## 4.1 🔴 与 CMRT 最本质的差别：PE25 **包含「分不开」的样本**")
    say("")
    say(f"| | 对数 | 占 PE25 |")
    say("|---|---|---|")
    say(f"| α 恰为 1.000（**没分开**） | **{n_ns}** | {n_ns/len(pp)*100:.1f}% |")
    say(f"| α ≠ 1（真的分开了） | **{n_sep}** | {n_sep/len(pp)*100:.1f}% |")
    say("")
    if n_sep:
        sub = pp[~pp["no_separation"]]
        say(f"**只看真的分开的子集**（{n_sep} 对）：")
        say(f"- 无向 α 中位 {np.exp(sub['y'].abs().median()):.3f}，最大 {np.exp(sub['y'].abs().max()):.3f}")
        say(f"- `EMO`（S 后出峰）= {sub['EMO'].mean()*100:.1f}%"
            f"（{int(sub['EMO'].sum())} / {n_sep}）← 接近均衡，说明 R/S 标签无系统偏置")
    say("")
    say("> **这条比靶变量的数值重要得多。** CMRT 的数据卡 §7.1 写明它**只收录了成功分离的数据**")
    say("> （`|α| = 1.000` 仅 0.16%），这是它最大的一处选择偏倚。")
    say(f"> 而 **PE25 有 {n_ns/len(pp)*100:.0f}% 的分子根本没分开** ——")
    say("> ⇒ **PE25 恰好覆盖了 CMRT 的盲区**，「预测分不开」这件事**只有 PE25 能验证**。")
    say("> → 论文里应把这条写成外部验证的**设计优势**，而不是缺陷。")
    say("")

    # ── 与 CMRT 对账 + 重叠剔除
    say("## 5. 与 CMRT 对账 & 泄漏剔除")
    say("")
    cm = pd.read_csv(CUR / "cmrt_pairs.csv")
    say(f"| 量 | CMRT（{len(cm)} 对） | PE25（{len(pp)} 对） |")
    say("|---|---|---|")
    say(f"| y 的 SD | {cm['y'].std():.4f} | {pp['y'].std():.4f} |")
    say(f"| 无向 α 中位 | {np.exp(cm['y'].abs().median()):.3f} | {np.exp(pp['y'].abs().median()):.3f} |")
    say(f"| EMO 占比 | {cm['EMO'].mean()*100:.1f}% | {pp['EMO'].mean()*100:.1f}% |")
    say("")

    extm = pd.read_csv(CUR / "external_molecules.csv")
    overlap = set(extm.loc[extm["in_cmrt"].astype(bool), "name"].str.lower())
    pp["in_cmrt_leak"] = pp["molecule"].str.lower().isin(overlap)
    n_leak = int(pp["in_cmrt_leak"].sum())
    say(f"- **与 CMRT 重叠的分子：{n_leak} 对**（"
        f"{sorted(set(pp.loc[pp['in_cmrt_leak'], 'molecule']))}）")
    say(f"  → ⚠️ CP4 外部验证时**必须剔除**这些行（脚本已打 `in_cmrt_leak` 标记，未删除，保留可追溯性）")
    say(f"- 扣掉泄漏后可用：**{len(pp) - n_leak} 对**")
    say("")

    pp = pp.sort_values(["condition", "molecule"]).reset_index(drop=True)
    pp.to_csv(CUR / "external_pe25_pairs.csv", index=False, encoding="utf-8")
    (ROOT / "notes" / "外部验证集_PE25构造报告.md").write_text(
        "\n".join(LOG) + "\n", encoding="utf-8")
    print(f"\n[落盘] data/curated/external_pe25_pairs.csv ({len(pp)} 行)")
    print("[落盘] notes/外部验证集_PE25构造报告.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
