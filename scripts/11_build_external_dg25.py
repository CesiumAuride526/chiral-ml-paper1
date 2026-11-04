# -*- coding: utf-8 -*-
"""
11_build_external_dg25.py — ★ 构造 **DG25 外部验证集**（6 柱 × 2 流动相，定向靶）
================================================================================
ROADMAP 「下一步 ⑤」：*用 DG25 的 OD-RH（= CMRT 的 Cellulose-a 家族）重构外部验证*。

## 为什么 DG25 比 PE25 更适合做第一个外部验证
| | PE25 | **DG25** |
|---|---|---|
| 柱 | 只给了「borate」一个条件，**连柱名都查不到** | **6 根柱点名**（OD-RH / LA-1 / LA-2 / LC-2 / LC-3 / LC-4） |
| 与 CMRT 的关系 | 无 | **OD-RH = 纤维素三(3,5-二甲基苯基氨基甲酸酯)**，即 CMRT 的 `Cellulose-a` 家族（同为反相模式） |
| 流动相 | 单条件 | **酸性 / 碱性两套**（可做条件外推） |
| 靶变量 | `log10(kR/kS)`（需换算） | **`alfaRS` 直接给定向值**（含 <1 的值 ⇒ 不是「先出/后出」而是 **CIP 定向**） |

⚠️ **三个必须处理的坑**：
1. **本研究组的 `alfaRS` 是 `k_R/k_S`**（与 PE25 同源同约定），
   而我们的靶变量是 `y = ln α_true = ln[k(S)/k(R)]` —— **互为倒数**。`y = −ln(alfaRS)`。
   （本脚本用「<1 与 >1 混合出现、且 ≈1.000000 恰为『分不开』」这条内部证据做定向性证明。）
2. **DG25 与 PE25 是同一批分子（42/43 重合）** —— 二者**合起来只算一套**外部集，
   论文里绝不能报成两次独立验证（PROJECT_STATE §0 第 6 条）。
3. **`Predicted alfaRS` 列是他们模型的预测值 —— 必须丢掉**，
   否则会把别人的模型输出混进我们的「实验真值」。

用法: PYTHONIOENCODING=utf-8 python scripts/11_build_external_dg25.py
产出: data/curated/external_dg25_all.csv（长表，全 12 个条件）
      data/curated/external_dg25_odrh.csv（★ CMRT 同族子集，直接用于 CP4）
      notes/外部验证集_DG25构造报告.md
"""
from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CUR = ROOT / "data" / "curated"
DG = ROOT / "data" / "raw" / "zenodo" / "degauquier2025"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module  # noqa: E402

# 复用 08 的名称规范化（NFKD 去音标 + 大小写折叠）—— DG25 里有 `Bupivacaïne` 这种带分音符号的名字
_normalize = import_module("08_external_overlap").normalize

LOG: list[str] = []


def say(s: str = "") -> None:
    print(s)
    LOG.append(s)


# CMRT 与本外部集重合的分子（PROJECT_STATE §0 第 7 条）—— 外部验证前必须剔除
CMRT_OVERLAP = {"Ibuprofen", "Mandelic acid", "Modafinil", "Rolipram"}

PHASES = {
    "acidic": "Chiral models acidic mobile phase final.xlsx",
    "basic": "Chiral models basic mobile phase final.xlsx",
}
# 关心的柱；OD-RH 是唯一与 CMRT 同族的（纤维素氨基甲酸酯）
COLUMNS = ["OD-RH", "LA-1", "LA-2", "LC-2", "LC-3", "LC-4"]
ODRH = "OD-RH"


def load_phase(phase: str) -> pd.DataFrame:
    """读一个流动相文件里**全部** `MLR pred alfaRS <COL>` sheet 的**实验值**。"""
    path = DG / PHASES[phase]
    xl = pd.ExcelFile(path)
    rows = []
    for sh in xl.sheet_names:
        if not sh.startswith("MLR pred alfaRS "):
            continue
        col = sh.replace("MLR pred alfaRS ", "").strip()
        if col not in COLUMNS:
            continue
        df = xl.parse(sh)
        # ⚠️ 只要实验值，**丢掉 Predicted**
        exp_col = [c for c in df.columns if "xperimental" in c]
        if not exp_col:
            continue
        names_ = df.iloc[:, 0]                      # 第 1 列 = 分子名（表头是 Unnamed: 0）
        vals = pd.to_numeric(df[exp_col[0]], errors="coerce")
        for nm, v in zip(names_, vals):
            if pd.isna(nm) or pd.isna(v):
                continue
            rows.append(dict(name=str(nm).strip(), phase=phase, column=col,
                             alfaRS=float(v), src_sheet=sh))
    return pd.DataFrame(rows)


def main() -> int:
    say("# DG25 外部验证集（构造报告）")
    say("")

    d = pd.concat([load_phase(p) for p in PHASES], ignore_index=True)
    say(f"- 读取实验 α 记录：**{len(d)}** 条（{d['name'].nunique()} 分子 × "
        f"{d['column'].nunique()} 柱 × {d['phase'].nunique()} 流动相）")
    say("")

    # ── 健全性：α 必须 > 0
    bad = d[~(d["alfaRS"] > 0)]
    if len(bad):
        say(f"- ⚠️ 剔除 {len(bad)} 条非正 α：{bad['name'].unique()[:5].tolist()}")
        d = d[d["alfaRS"] > 0].copy()

    # ── 定向性证明（★ 不能靠假设）
    say("## 1. ★ 定向性证明（`alfaRS` 到底是 `k_R/k_S` 还是「后出/先出」）")
    say("")
    sep = d[d["alfaRS"] != 1.0]                      # 1.000000 精确等于「没分开」
    n_lt = int((sep["alfaRS"] < 1).sum())
    n_gt = int((sep["alfaRS"] > 1).sum())
    say(f"- 未分开（α 精确 = 1.000000）：**{int((d['alfaRS']==1.0).sum())}** / {len(d)} "
        f"（**{100*(d['alfaRS']==1.0).mean():.1f}%**）")
    say(f"- 已分开里：α < 1 有 **{n_lt}** 条，α > 1 有 **{n_gt}** 条")
    say("")
    if n_lt > 0 and n_gt > 0:
        say("⇒ ✅ **α 同时出现 < 1 与 > 1** ⇒ 它不是「后出/先出」（那必然 ≥ 1），")
        say("  而是 **CIP 定向的 `k_R/k_S`** —— 与 PE25 同约定（PROJECT_STATE §0 第 9 条）。")
        say("")
        say("  因此换成我们的口径：**`y = ln α_true = ln[k(S)/k(R)] = −ln(alfaRS)`**")
    else:
        say("⇒ 🔴 **只有单侧取值 —— 定向性存疑，先别用！**")
    say("")

    # ── 名称 → SMILES/conn
    ex = pd.read_csv(CUR / "external_molecules.csv")
    nm2 = {_normalize(n): r for n, r in zip(ex["name"], ex.to_dict("records"))}
    d["_k"] = d["name"].map(_normalize)
    d["conn"] = d["_k"].map(lambda k: nm2.get(k, {}).get("conn"))
    d["smi_flat"] = d["_k"].map(lambda k: nm2.get(k, {}).get("smiles"))
    d["in_cmrt"] = d["_k"].map(lambda k: nm2.get(k, {}).get("in_cmrt"))
    miss = d[d["conn"].isna()]["name"].unique()
    if len(miss):
        say(f"- ⚠️ **{len(miss)} 个名字没解析到结构**：{list(miss)[:8]}")
    else:
        say(f"- ✅ 全部 **{d['name'].nunique()}** 个分子都解析到了 SMILES/conn（复用 `external_molecules.csv`）")
    say("")

    # ── 靶变量
    d["y"] = -np.log(d["alfaRS"])           # ln(kS/kR)
    d["alpha"] = d["alfaRS"]
    d["separated"] = (d["alfaRS"] != 1.0).astype(int)
    d["cmrt_overlap"] = d["name"].isin(CMRT_OVERLAP)

    # ── 汇总
    say("## 2. 覆盖度（每个条件的可用单元数）")
    say("")
    say("| 流动相 | 柱 | 记录数 | 分子数 | 已分开 | **可用（去 CMRT 重合）** |")
    say("|---|---|---|---|---|---|")
    for (ph, col), g in d.groupby(["phase", "column"]):
        nn = len(g) - int(g["cmrt_overlap"].sum())
        say(f"| {ph} | **{col}** | {len(g)} | {g['name'].nunique()} | "
            f"{int(g['separated'].sum())} | **{nn}** |")
    say("")

    od = d[d["column"] == ODRH].copy()
    say(f"### ★ OD-RH（= CMRT `Cellulose-a` 家族，反相）")
    say("")
    say(f"- 两套流动相合计 **{len(od)}** 条记录 / **{od['name'].nunique()}** 分子")
    say(f"- 其中 **{int(od['cmrt_overlap'].sum())}** 条与 CMRT 分子重合 → 必须先剔除"
        f"（{sorted(set(od[od['cmrt_overlap']]['name']))}）")
    say(f"- 剔除后可用 **{len(od) - int(od['cmrt_overlap'].sum())}** 条")
    say("")
    say("| 条件 | n | 已分开 | y = ln(k_S/k_R) 均值 | SD |")
    say("|---|---|---|---|---|")
    for ph, g in od.groupby("phase"):
        say(f"| OD-RH / {ph} | {len(g)} | {int(g['separated'].sum())} | "
            f"{g['y'].mean():+.3f} | {g['y'].std():.3f} |")
    say("")

    # ── 与 PE25 的关系（**论文必须写明**）
    say("## 3. ⚠️ 与 PE25 的关系（论文必须写明）")
    say("")
    pe_path = CUR / "external_pe25_pairs.csv"
    if pe_path.exists():
        pe = pd.read_csv(pe_path)
        # ⚠️ 一律按**分子名规范化**比对（pe25 表不一定有 conn 列 —— 早前版本按 conn 比对
        #    静默得到「重合 0」，那正是「except 掩盖 bug」那类错误，故改为按名并**显式断言**）
        pe_names = set(pe["name"].map(_normalize)) if "name" in pe.columns else set()
        if not pe_names and "molecule" in pe.columns:
            pe_names = set(pe["molecule"].map(_normalize))
        if not pe_names:
            say(f"- 🔴 **无法比对**：`external_pe25_pairs.csv` 列名为 {list(pe.columns)[:8]}，"
                f"没有可用于比对的分子名列。**不要把「重合 0」当成结论。**")
        else:
            dg_names = set(od["_k"].dropna())
            both = pe_names & dg_names
            say(f"- PE25 分子 **{len(pe_names)}** 个 ｜ DG25-OD-RH 分子 **{len(dg_names)}** 个 ｜ "
                f"**重合 {len(both)} 个**（{100*len(both)/max(len(dg_names),1):.0f}%）")
            if len(both) >= 0.8 * len(dg_names):
                say("- ⇒ ✅ **重合度极高**：两者不是两套独立的外部验证集，**只能合并算一套**。")
                say("  论文里写「两次独立外部验证」会被审稿人一眼看穿。")
            else:
                say("- ⇒ ⚠️ 重合度低于预期，**与 PROJECT_STATE §0 第 6 条（42/43 重合）不符 —— "
                    "先查清再引用**。")
    say("")

    # ── 落盘
    keep = ["name", "conn", "smi_flat", "phase", "column", "alfaRS", "alpha", "y",
            "separated", "cmrt_overlap", "src_sheet"]
    d[keep].to_csv(CUR / "external_dg25_all.csv", index=False, encoding="utf-8")
    od[keep].to_csv(CUR / "external_dg25_odrh.csv", index=False, encoding="utf-8")
    say("## 4. 落盘")
    say("")
    say(f"- `data/curated/external_dg25_all.csv` —— **{len(d)}** 条（6 柱 × 2 流动相）")
    say(f"- `data/curated/external_dg25_odrh.csv` —— **{len(od)}** 条（★ CMRT 同族，CP4 用）")
    say("")
    say("### 下一步（CP4 用这个）")
    say("")
    say("1. 用 `external_dg25_odrh.csv` 剔掉 `cmrt_overlap`，做「CMRT(Cellulose-a) → DG25(OD-RH)」外部验证；")
    say("2. 再试「CMRT → LA/LC 系列」——**跨固定相骨架**，是 S3′ 失败后的真正考验；")
    say("3. 两套流动相给了**条件外推**的免费对照。")

    (ROOT / "notes" / "外部验证集_DG25构造报告.md").write_text(
        "\n".join(LOG) + "\n", encoding="utf-8")
    print(f"\n[落盘] data/curated/external_dg25_all.csv ({len(d)} 行)")
    print(f"[落盘] data/curated/external_dg25_odrh.csv ({len(od)} 行)")
    print("[落盘] notes/外部验证集_DG25构造报告.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
