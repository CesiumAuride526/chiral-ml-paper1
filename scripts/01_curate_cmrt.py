# -*- coding: utf-8 -*-
"""
01_curate_cmrt.py — CMRT 主表清洗（Paper 1 / M1 里程碑）
=========================================================
输入: data/raw/cmrt/cmrt_repo/dataset/All_column_charity.csv   (25,867 行)
      data/raw/cmrt/cmrt_repo/dataset/Column_info.xlsx         (25 根柱元数据)
输出: data/curated/cmrt_measurements.csv   记录级清洗表（可溯源、含所有 flag）
      data/curated/cmrt_pairs.csv          ★ 建模主表（一行 = 一个条件单元下的对映体对）
      data/curated/columns.csv             CSP 描述符表（25 根柱）
      data/curated/splits/*.csv            S1/S2/S2b/S3/S3p/S4 划分索引
      notes/01_curate_report.md            运行报告
      notes/异常复核样本.csv               需人工复核的行

规格来源: docs/计算方案.md §1.2（**2026-09-16 审计修正版**）+ docs/审计报告_2026-09-16.md §3
--------------------------------- 与 00_assess_cmrt.py 的关键差异（全是修正） ---------------------------------
 R1 靶变量必须**定向**: y = ln α_true = ln[k(S)/k(R)]，R/S 由 RDKit CIP 标签定，**不按洗脱顺序**。
    → 同时产出 EMO = 1{y>0}（S 后出峰）作辅助二分类靶。
 R2 α 用**色谱定义** α = k₂/k₁ = (t_R−t₀)/(t_R−t₀)，t₀ 必须估算（数据集无 t₀）。
 R3 对映体判定要**精确**：仅 InChIKey 连接层相同不够，必须剔除非对映体（diastereomer）。
    实测 609/10,968（5.6%）是 diastereomer — 审计当时把它们当对映体用了。
 R4 配对必须在**同一条件单元**内（同 iPrOH、同流速），否则 α 是两个条件的混合物。
 R5 DOI 修复（5% 记录把 '.' 写成了 '/'）。
 R6 划分除 S1–S5 外新增 **S2b 留一文献** 与 **S3′ 留一格因子外推**。
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import DataStructs, AllChem
from rdkit.ML.Cluster import Butina

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "cmrt" / "cmrt_repo" / "dataset"
CUR = ROOT / "data" / "curated"
SPLITS = CUR / "splits"
NOTES = ROOT / "notes"
CUR.mkdir(parents=True, exist_ok=True)
SPLITS.mkdir(parents=True, exist_ok=True)

SEED = 42
COL_ID_MM = 4.6          # 柱内径 (mm) —— Column_info.xlsx「内径」行，25 根柱统一
COL_LEN_MM = 250.0       # 柱长 (mm) —— 同上
POROSITY = 0.62          # V_m ≈ 0.62·V_c（审计 §3.1 采用的估算）
ALPHA_ABS_MAX = 20.0     # 审计 §3.9：α > 20 疑似抽取错误
RT_MAX = 60.0            # 审计 §3.9：RT > 60 min 标记不硬删
SPEED_MAX = 4.0          # 4.6 mm 柱上 > 4 mL/min 属异常（数据里有 13 mL/min）

# ── 取代基码的化学含义（★ 2026-09-16 已按 Daicel 官方产品页逐条核实，不再有猜测）──
# 命名规律（已用本数据集的 25 行 `particle_um` 实测校准）：
#   前缀 A* = 直链淀粉(amylose) / O* = 纤维素(cellulose)；
#   后缀 "3" → 3 μm、"H" → 5 μm、"U" → 1.6 μm（IAU）；
#   裸名（无后缀）：涂覆型 AD/OD/OJ/AS/AZ = **10 μm**，键合型 IA–IG = **5 μm**；
#   IA–IG 是**键合型**(immobilized)，AD/AS/AZ/OD/OJ/OZ 是**涂覆型**(coated)。
# 核实来源：daicelchiral.com / daicelchiraltech.cn 产品页 + Wikipedia「Chiral analysis」+ 文献摘要。
# ⚠️ 首版曾把 e 猜成 3,5-二甲基、把 g 记为「未确认」——都是错的，已更正。
SUBS_MEANING = {
    "a": "3,5-二甲基苯基氨基甲酸酯 (3,5-dimethylphenylcarbamate)"
         " — Amylose: AD/AD3/ADH/IA/IAH/IAU; Cellulose: IB/OD/OD3/ODH",
    "b": "3,5-二氯苯基氨基甲酸酯 (3,5-dichlorophenylcarbamate)"
         " — Amylose: IE; Cellulose: IC/IC3/ICH",
    "c": "4-甲基苯甲酸酯 (4-methylbenzoate) — Cellulose: OJ/OJ3/OJH",
    "d": "(S)-1-苯乙基氨基甲酸酯 ((S)-1-phenylethylcarbamate)"
         " — Amylose: AS/ASH  ⚠️ 取代基本身带手性中心",
    "e": "3-氯苯基氨基甲酸酯 (3-chlorophenylcarbamate) — Amylose: ID（单柱）",
    "f": "3-氯-4-甲基苯基氨基甲酸酯 (3-chloro-4-methylphenylcarbamate)"
         " — Amylose: AZ/IF/IF3; Cellulose: OZ3",
    "g": "3-氯-5-甲基苯基氨基甲酸酯 (3-chloro-5-methylphenylcarbamate) — Amylose: IG（单柱）",
}

# 取代基的化学结构（写成「游离氨基甲酸/羧酸」形式，即把与多糖相连的 O 端补 H）
# 用途：算取代基本身的 2D 结构描述符 —— 自建 CSP 描述符的核心输入
SUBS_SMILES = {
    "a": "Cc1cc(C)cc(NC(=O)O)c1",              # 3,5-二甲基苯基氨基甲酸
    "b": "O=C(O)Nc1cc(Cl)cc(Cl)c1",            # 3,5-二氯苯基氨基甲酸
    "c": "Cc1ccc(C(=O)O)cc1",                  # 4-甲基苯甲酸
    "d": "C[C@H](NC(=O)O)c1ccccc1",            # (S)-1-苯乙基氨基甲酸
    "e": "O=C(O)Nc1cc(Cl)ccc1",                # 3-氯苯基氨基甲酸
    "f": "O=C(O)Nc1cc(Cl)c(C)cc1",             # 3-氯-4-甲基苯基氨基甲酸
    "g": "Cc1cc(Cl)cc(NC(=O)O)c1",             # 3-氯-5-甲基苯基氨基甲酸
}

LOG: list[str] = []


def say(msg: str = "") -> None:
    print(msg)
    LOG.append(msg)


# ════════════════════════════════════════════════════════════════════
# Stage 0 — 载入与基础清洗
# ════════════════════════════════════════════════════════════════════
def load_raw() -> pd.DataFrame:
    r = pd.read_csv(RAW / "All_column_charity.csv")
    r.columns = [c.strip() for c in r.columns]
    for junk in ("Unnamed: 0", "index", ""):
        if junk in r.columns:
            r = r.drop(columns=[junk])
    say(f"[S0] 载入 {RAW.name}: {r.shape[0]} 行 × {r.shape[1]} 列  字段={list(r.columns)}")

    n0 = len(r)
    r = r.dropna(subset=["SMILES", "RT", "Column"]).copy()
    for c in ("RT", "Speed", "i-PrOH_proportion"):
        r[c] = pd.to_numeric(r[c], errors="coerce")
    r = r.dropna(subset=["RT", "Speed", "i-PrOH_proportion"])
    say(f"[S0] 丢空值 {n0} → {len(r)}")

    # ── R5: DOI 修复 ───────────────────────────────────────────────
    raw_doi = r["Literature"].astype(str).str.strip()

    def fix_doi(d: str) -> str:
        """把 DOI 后缀里被误写成 '/' 的分隔符还原为 '-'。仅在后缀含 '/' 时动手。"""
        m = re.match(r"^(10\.\d{4,9})/(.+)$", d)
        if not m:
            return d
        return f"{m.group(1)}/{m.group(2).replace('/', '-')}"

    r["doi"] = raw_doi.map(fix_doi)
    r["doi_raw"] = raw_doi
    n_fix = int((r["doi"] != r["doi_raw"]).sum())
    say(f"[S0] DOI 修复 {n_fix} 条 / {r.loc[r['doi'] != r['doi_raw'], 'doi'].nunique()} 个唯一 DOI")
    assert r["doi"].str.count("/").max() == 1, "DOI 修复后仍有多个斜杠"

    r = r[r["RT"] > 0].copy()  # 审计 §3.9：RT ≤ 0 共 1 条
    say(f"[S0] 剔除 RT ≤ 0 后 {len(r)}")
    return r


# ════════════════════════════════════════════════════════════════════
# Stage 1 — RDKit 规范化 + 手性指纹
# ════════════════════════════════════════════════════════════════════
def invert_stereo(mol: Chem.Mol) -> Chem.Mol:
    """镜像分子：翻转所有四面体手性中心的 CW/CCW。
    双键的 E/Z 是**相对**构型，镜面反射下不变，故 /b 层不动（已用 7 组对照实测验证）。"""
    m = Chem.RWMol(mol)
    for a in m.GetAtoms():
        t = a.GetChiralTag()
        if t == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
            a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
        elif t == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
            a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    return m.GetMol()


_TCACHE: dict[str, dict] = {}


def mol_info(smi: str) -> dict | None:
    """单分子全部派生化信息（带缓存）。"""
    if smi in _TCACHE:
        return _TCACHE[smi]
    m = Chem.MolFromSmiles(smi)
    if m is None:
        _TCACHE[smi] = None
        return None
    iso = Chem.MolToSmiles(m, isomericSmiles=True)
    mi = Chem.MolFromSmiles(iso)
    inv = invert_stereo(mi)
    # ⚠️ 本机 rdkit 2026.03.6 里 rdMolDescriptors.CalcMurckoScaffold **不存在**
    #    （调用即 AttributeError；审计脚本用 try/except 吞掉了 → 若照抄会静默全塌成一个值）
    #    正确 API: Chem.MurckoDecompose（无环分子的骨架为空串，属正常）
    murcko = Chem.MolToSmiles(Chem.MurckoDecompose(mi))
    inchi = Chem.MolToInchi(mi)
    # CIP 标签（新实现），配 canonical rank 排序 → 对映体间可比较
    ranks = list(Chem.CanonicalRankAtoms(mi, breakTies=True))
    centers = Chem.FindMolChiralCenters(mi, includeUnassigned=True, useLegacyImplementation=False)
    cip = tuple(c for _, c in sorted((ranks[i], c) for i, c in centers if c in ("R", "S")))
    # ECFP4 供化学类聚类用
    fp = AllChem.GetMorganFingerprintAsBitVect(mi, 2, nBits=2048)
    info = dict(
        smi_iso=iso,
        smi_flat=Chem.MolToSmiles(mi, isomericSmiles=False),
        smi_mirror=Chem.MolToSmiles(inv, isomericSmiles=True),
        inchikey=Chem.MolToInchiKey(mi),
        conn=Chem.MolToInchiKey(mi)[:14],
        murcko=murcko,
        cip=cip,
        _inchi=inchi,
        n_stereo=sum(1 for _, c in centers if c in ("R", "S")),
        n_stereo_undef=sum(1 for _, c in centers if c == "?"),
        fp=fp,
    )
    _TCACHE[smi] = info
    return info


def build_molecules(df: pd.DataFrame) -> pd.DataFrame:
    uniq = df["SMILES"].dropna().unique()
    cache = {s: mol_info(s) for s in uniq}
    bad = [s for s in uniq if cache[s] is None]
    say(f"[S1] 唯一 SMILES {len(uniq)}｜RDKit 解析失败 {len(bad)} ({len(bad)/len(uniq)*100:.2f}%)")
    for k in ("smi_iso", "smi_flat", "smi_mirror", "inchikey", "conn", "murcko",
              "cip", "n_stereo", "n_stereo_undef", "fp"):
        df[k] = df["SMILES"].map(lambda s: (cache.get(s) or {}).get(k))
    df = df.dropna(subset=["smi_iso", "conn"]).copy()
    say(f"[S1] 规范化后 {len(df)}｜唯一立体 SMILES {df['smi_iso'].nunique()}｜"
        f"唯一骨架(去立体) {df['smi_flat'].nunique()}｜Murcko 骨架 {df['murcko'].nunique()}")
    return df


# ════════════════════════════════════════════════════════════════════
# Stage 2 — t₀ / k
# ════════════════════════════════════════════════════════════════════
def add_retention(df: pd.DataFrame) -> pd.DataFrame:
    vc = math.pi * (COL_ID_MM / 20.0) ** 2 * (COL_LEN_MM / 10.0)  # cm³
    vm = POROSITY * vc
    r0 = COL_ID_MM / 2.0
    say(f"[S2] 柱几何 {COL_ID_MM} mm × {COL_LEN_MM} mm → V_c = {vc:.3f} cm³, "
        f"V_m = 0.62·V_c = {vm:.3f} cm³ (= {POROSITY*r0*r0*math.pi*COL_LEN_MM/1000:.4f})")
    say(f"[S2] t₀ [min] = {vm:.4f} / Speed [mL/min]  → t₀ 范围 "
        f"{vm/df['Speed'].max():.2f}–{vm/df['Speed'].min():.2f} min（Speed 最大/最小处）")
    df["t0"] = vm / df["Speed"]
    df["k"] = (df["RT"] - df["t0"]) / df["t0"]
    say(f"[S2] k ≤ 0 的记录 {int((df['k'] <= 0).sum())} 条 "
        f"({(df['k'] <= 0).mean()*100:.2f}%)  ← 审计 §3.9 报 109 条，量级一致")
    return df


# ════════════════════════════════════════════════════════════════════
# Stage 3 — 精确对映体配对
# ════════════════════════════════════════════════════════════════════
def are_enantiomers(a: dict, b: dict) -> tuple[bool, bool]:
    """双方法交叉验证。返回 (是否对映体, 两法是否一致)。"""
    if a["conn"] != b["conn"]:
        return False, True
    # 方法 1：立体翻转匹配（镜像分子 == 对方）
    v1 = a["smi_mirror"] == b["smi_iso"]
    # 方法 2：InChI /t 层全翻转 或 同 /t 异 /m
    ia, ib = a["_inchi"], b["_inchi"]
    ta = re.search(r"/t([^/]*)", ia)
    tb = re.search(r"/t([^/]*)", ib)
    ma = re.search(r"/m(\d)", ia)
    mb = re.search(r"/m(\d)", ib)
    ta = ta.group(1) if ta else ""
    tb = tb.group(1) if tb else ""
    ma = ma.group(1) if ma else ""
    mb = mb.group(1) if mb else ""
    flip = "".join({"+": "-", "-": "+"}.get(ch, ch) for ch in ta)
    v2 = (ta == tb and ma != mb) or (flip == tb and flip != "")
    return bool(v1), bool(v1 == v2)


def build_pairs(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """把记录级表折成「条件单元级」的对映体对表。

    条件单元 = (conn, Column, i-PrOH, Speed)。**先在同一单元内聚合重复测量**，
    再在同一单元内配两个对映体 —— 这样 α 才是同一条件下的选择性。
    """
    # 条件单元内的重复聚合（同分子同条件重复测量，审计 §3.5 实测 RSD 中位 2.74%）
    key = ["conn", "Column", "i-PrOH_proportion", "Speed", "smi_iso"]
    agg = df.groupby(key, dropna=False).agg(
        k_mean=("k", "mean"), k_sd=("k", "std"), k_n=("k", "size"),
        rt_mean=("RT", "mean"), t0=("t0", "first"),
        doi=("doi", lambda x: x.iloc[0]),
        murcko=("murcko", "first"), smi_flat=("smi_flat", "first"),
    ).reset_index()
    say(f"[S3] (分子, 柱, 条件) 条件单元 {len(agg)} 个（由 {len(df)} 条记录折叠）")

    grp = agg.groupby(["conn", "Column", "i-PrOH_proportion", "Speed"], dropna=False)
    n_cell_multi = sum(1 for _, g in grp if g["smi_iso"].nunique() >= 2)
    say(f"[S3] 含 ≥2 个立体异构体的条件单元 {n_cell_multi} 个")

    rows, n_cand, n_ent, n_disagree = [], 0, 0, 0
    sig_bad = 0
    n_skip_nonpos = 0
    for (conn, col, ipr, spd), g in grp:
        g2 = g.drop_duplicates(subset=["smi_iso"])
        if len(g2) != 2:
            continue
        n_cand += 1
        a, b = list(g2.itertuples())
        ia, ib = mol_info(a.smi_iso), mol_info(b.smi_iso)
        ent, agree = are_enantiomers(ia, ib)
        if not agree:
            n_disagree += 1
        if not ent:
            continue
        n_ent += 1
        # CIP 定向：签名字典序更大者（更 "S"）为 S 侧
        if tuple(ia["cip"]) >= tuple(ib["cip"]):
            s_side, r_side = a, b
            cip_s, cip_r = ia["cip"], ib["cip"]
        else:
            s_side, r_side = b, a
            cip_s, cip_r = ib["cip"], ia["cip"]
        # 对映体的 CIP 签名应逐位互补（R↔S）。不互补 = 多个手性中心因环对称产生
        # CanonicalRank 并列，定向退化成「按行序挑一个」→ 该对 y 的**符号语义不可信**。
        # 实测 10,525 对里仅 1 对（0.01%）。此时用规范化 SMILES 字典序兜底（确定且反对称），并打标。
        cip_ambiguous = 0
        if len(cip_s) != len(cip_r) or any(x == y for x, y in zip(cip_s, cip_r)):
            sig_bad += 1
            cip_ambiguous = 1
            if s_side.smi_iso < r_side.smi_iso:
                s_side, r_side = r_side, s_side
                cip_s, cip_r = cip_r, cip_s
        kS, kR = s_side.k_mean, r_side.k_mean
        if kS <= 0 or kR <= 0:
            n_skip_nonpos += 1
            continue
        alpha = kS / kR
        rows.append(dict(
            conn=conn, Column=col, murcko=a.murcko, smi_flat=a.smi_flat,
            ipr=ipr, speed=spd, t0=a.t0,
            smi_S=s_side.smi_iso, smi_R=r_side.smi_iso,
            cip_S="".join(cip_s), cip_R="".join(cip_r),
            k_S=kS, k_R=kR, kS_rsd=(s_side.k_sd / kS if pd.notna(s_side.k_sd) and kS else 0.0),
            kR_rsd=(r_side.k_sd / kR if pd.notna(r_side.k_sd) and kR else 0.0),
            n_rep=int(s_side.k_n + r_side.k_n),
            kS_n=int(s_side.k_n), kR_n=int(r_side.k_n),
            alpha_true=alpha, y=math.log(alpha), EMO=int(alpha > 1),
            # ⚠️ 辅助量必须与 y **同向**（正 = S 后出峰），否则符号无法与主靶对照
            rt_S=float(s_side.rt_mean), rt_R=float(r_side.rt_mean),
            dRT_over_t0=float((s_side.rt_mean - r_side.rt_mean) / a.t0),
            rt_ratio_spike=float(max(a.rt_mean, b.rt_mean) / min(a.rt_mean, b.rt_mean)),
            doi=a.doi, doi_raw=a.doi_raw if hasattr(a, "doi_raw") else a.doi,
            n_stereo=ia["n_stereo"], flag_cip_ambiguous=cip_ambiguous,
        ))
    say(f"[S3] 候选（同条件单元内两立体异构体）{n_cand}｜"
        f"**精确对映体 {n_ent}**｜剔除非对映体(diastereomer) {n_cand - n_ent} "
        f"({(n_cand-n_ent)/max(n_cand,1)*100:.1f}%)")
    p = pd.DataFrame(rows)

    # ── 未匹配上的对（两对映体不在同一条件单元，但同柱）→ 合并条件后配对并打标
    matched = set(zip(p["conn"], p["Column"], p["smi_S"], p["smi_R"]))
    fallback, n_skip_nonpos = [], 0
    for (conn, col), g in agg.groupby(["conn", "Column"]):
        # 把该柱上每个立体异构体的所有条件合并（k 取均值），再配对 —— 条件不再纯净，故单独打标
        per = g.groupby("smi_iso").agg(k_mean=("k_mean", "mean"), k_n=("k_n", "sum"),
                                       rt_mean=("rt_mean", "mean"), t0=("t0", "first"),
                                       murcko=("murcko", "first"), doi=("doi", "first"),
                                       smi_flat=("smi_flat", "first")).reset_index()
        if len(per) != 2:
            continue
        a, b = list(per.itertuples())
        ia, ib = mol_info(a.smi_iso), mol_info(b.smi_iso)
        if not are_enantiomers(ia, ib)[0]:
            continue
        s_side, r_side = (a, b) if tuple(ia["cip"]) >= tuple(ib["cip"]) else (b, a)
        cip_s, cip_r = (ia["cip"], ib["cip"]) if s_side is a else (ib["cip"], ia["cip"])
        cip_ambiguous = 0
        if len(cip_s) != len(cip_r) or any(x == y for x, y in zip(cip_s, cip_r)):
            sig_bad += 1
            cip_ambiguous = 1
            if s_side.smi_iso < r_side.smi_iso:
                s_side, r_side = r_side, s_side
                cip_s, cip_r = cip_r, cip_s
        if ((conn, col, s_side.smi_iso, r_side.smi_iso) in matched
                or (conn, col, r_side.smi_iso, s_side.smi_iso) in matched):
            continue
        kS, kR = s_side.k_mean, r_side.k_mean
        if kS <= 0 or kR <= 0:
            n_skip_nonpos += 1
            continue
        alpha = kS / kR
        fallback.append(dict(
            conn=conn, Column=col, murcko=a.murcko, smi_flat=a.smi_flat,
            ipr=np.nan, speed=np.nan, t0=a.t0,
            smi_S=s_side.smi_iso, smi_R=r_side.smi_iso,
            cip_S="".join(cip_s), cip_R="".join(cip_r),
            k_S=kS, k_R=kR, kS_rsd=0.0, kR_rsd=0.0,
            n_rep=int(s_side.k_n + r_side.k_n), kS_n=int(s_side.k_n), kR_n=int(r_side.k_n),
            alpha_true=alpha, y=math.log(alpha), EMO=int(alpha > 1),
            rt_S=float(s_side.rt_mean), rt_R=float(r_side.rt_mean),
            dRT_over_t0=float((s_side.rt_mean - r_side.rt_mean) / a.t0),
            rt_ratio_spike=float(max(a.rt_mean, b.rt_mean) / min(a.rt_mean, b.rt_mean)),
            doi=a.doi, doi_raw=a.doi, n_stereo=ia["n_stereo"], flag_cip_ambiguous=cip_ambiguous,
        ))
    fb = pd.DataFrame(fallback)
    if len(fb):
        fb["cond_matched"] = False
        say(f"[S3] 跨条件单元回退配对（同柱不同流动相/流速，α 条件不纯）{len(fb)} 对 —— 标记 cond_matched=False")
    say(f"[S3] 因 k ≤ 0 丢弃的对 {n_skip_nonpos}（主表剔除，保留在记录级表里）")
    say(f"[S3] 两法判定不一致 {n_disagree}｜CIP 签名非互补（定向退化为行序，已打标）{sig_bad}")
    p["cond_matched"] = True
    p = pd.concat([p, fb], ignore_index=True)

    stats = dict(n_records=len(df), n_units=len(agg), n_candidate=n_cand,
                 n_enantiomer=n_ent, n_diastereomer=n_cand - n_ent,
                 n_disagree=n_disagree, sig_bad=sig_bad, n_fallback=len(fb),
                 n_pairs_total=len(p))
    return p, stats


# ════════════════════════════════════════════════════════════════════
# Stage 4 — 柱元数据 / CSP 描述符表
# ════════════════════════════════════════════════════════════════════
def build_columns(p: pd.DataFrame) -> pd.DataFrame:
    x = pd.read_excel(RAW / "Column_info.xlsx", sheet_name="Sheet1")
    # ⚠️ 62 行里只有 25 行是柱；末尾还有「内径」「柱长」等元数据行，必须按三列齐全过滤
    x = x.dropna(subset=["Column", "Silica_surface", "取代基"])
    x = x[["Column", "Silica_surface", "填料粒径", "取代基"]]
    x.columns = ["Column", "surface", "particle_um", "subs_code"]
    x["surface"] = x["surface"].astype(str).str.strip()
    x["subs_code"] = x["subs_code"].astype(str).str.strip()
    x["particle_um"] = pd.to_numeric(x["particle_um"], errors="coerce")
    x = x[x["Column"].isin(p["Column"].unique())].copy()
    x["family"] = x["surface"] + "-" + x["subs_code"]
    x["subs_meaning"] = x["subs_code"].map(SUBS_MEANING)
    cnt = p.groupby("Column").size().rename("n_pairs")
    x = x.merge(cnt, on="Column", how="left")
    x["n_pairs"] = x["n_pairs"].fillna(0).astype(int)
    x = x.sort_values("n_pairs", ascending=False)
    say(f"[S4] 柱元数据 {len(x)} 根（Column_info.xlsx 62 行里只有 {len(x)} 行有数据）｜"
        f"家族 {x['family'].nunique()} 个")
    return x


# ════════════════════════════════════════════════════════════════════
# Stage 5 — 划分索引
# ════════════════════════════════════════════════════════════════════
def make_splits(p: pd.DataFrame, cols: pd.DataFrame) -> dict:
    """六套划分。每个文件列: row_id, fold, role（role: train/test，S3′ 用 task 拆分）。"""
    from sklearn.model_selection import GroupKFold

    def gkf(groups, k=5):
        """GroupKFold 的组数自适应封装（组数 < k 时 sklearn 会直接报错）。"""
        k = min(k, len(pd.unique(groups)))
        return GroupKFold(n_splits=max(k, 2)).split(p, groups=groups)

    p = p.reset_index(drop=True)
    p["row_id"] = p.index
    n = len(p)
    rng = np.random.default_rng(SEED)
    out = {}

    # ── S1 随机 8:2（故意不分组：这是要被我们自己证明高估的那一套）
    perm = rng.permutation(n)
    s1 = pd.DataFrame({"row_id": p["row_id"], "fold": 0})
    s1.loc[perm[: int(n * 0.8)], "fold"] = 1
    s1.loc[perm[int(n * 0.8):], "fold"] = 2
    out["S1_random"] = s1

    # ── S2 分组：GroupKFold on dup_group = (conn, Column)
    dg = (p["conn"] + "|" + p["Column"]).values
    s2 = pd.DataFrame({"row_id": p["row_id"], "fold": 0})
    for i, (_, te) in enumerate(gkf(dg, 5)):
        s2.loc[te, "fold"] = i + 1
    out["S2_group_dup"] = s2

    # ── S2b 留一文献：GroupKFold on DOI（🔴 审计新增；文献主效应占 30.0%）
    s2b = pd.DataFrame({"row_id": p["row_id"], "fold": 0})
    for i, (_, te) in enumerate(gkf(p["doi"].values, 10)):
        s2b.loc[te, "fold"] = i + 1
    s2b["paper"] = p["doi"].values
    out["S2b_leave_paper"] = s2b

    # ── S3 留一柱：fold = 柱序号（真正 LOO-column，供逐柱报告）
    order = p["Column"].astype("category").cat.categories
    s3 = pd.DataFrame({"row_id": p["row_id"], "fold": p["Column"].map(
        {c: i + 1 for i, c in enumerate(order)}).values, "heldout_column": p["Column"].values})
    out["S3_leave_column"] = s3

    # ── S3′ 留一格因子外推（🔴 核心）：3 组「同取代基、异骨架」匹配对
    fam = cols.set_index("Column")["family"].to_dict()
    p["family"] = p["Column"].map(fam)
    spec = {s: sorted(cols.loc[cols["subs_code"] == s, "Column"]) for s in ("a", "b", "f")}
    matched = []
    for s in ("a", "b", "f"):
        amyl = [c for c in spec[s] if c in set(cols.loc[cols["surface"] == "Amylose", "Column"])]
        cellu = [c for c in spec[s] if c in set(cols.loc[cols["surface"] == "Cellulose", "Column"])]
        if amyl and cellu:
            for tr, te, tag in ((amyl, cellu, "amylose→cellulose"), (cellu, amyl, "cellulose→amylose")):
                rid = p.loc[p["Column"].isin(tr + te), "row_id"]
                matched.append(pd.DataFrame({
                    "row_id": rid.values,
                    "task": f"S3p_{s}_{tag}",
                    "role": np.where(p.loc[rid, "Column"].isin(te).values, "test", "train"),
                    "subs_code": s,
                }))
    s3p = pd.concat(matched, ignore_index=True) if matched else pd.DataFrame()
    out["S3p_factor"] = s3p

    # ── S4 留一化学类：Murcko 骨架 GroupKFold（+ ECFP4 Butina 粗聚类）
    s4 = pd.DataFrame({"row_id": p["row_id"], "fold": 0, "murcko": p["murcko"].values})
    for i, (_, te) in enumerate(gkf(p["murcko"].values, 5)):
        s4.loc[te, "fold"] = i + 1
    out["S4_murcko"] = s4

    # ECFP4 + Butina 聚类 → 化学类（S4b）
    smis = p["smi_S"].unique()
    fps, ok = [], []
    for s in smis:
        info = mol_info(s)
        if info:
            fps.append(info["fp"])
            ok.append(s)
    sims = []
    for i in range(1, len(fps)):
        sims.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i]))
    # ⚠️ Butina 收的是**距离**下限三角阵；直接喂 Tanimoto 相似度会把所有点并成一簇。
    dists = [1.0 - s for s in sims]
    clus = Butina.ClusterData(dists, len(fps), 0.6, isDistData=True)  # 距离≤0.6 ⇔ 相似度≥0.4
    s2c = {}
    for ci, members in enumerate(clus):
        for mi in members:
            s2c[ok[mi]] = ci
    p["ecfp_cluster"] = p["smi_S"].map(s2c).fillna(-1).astype(int)
    say(f"[S5] ECFP4 (Tanimoto 距离 0.6) Butina 聚类 → {p['ecfp_cluster'].nunique()} 个化学类")
    s4b = pd.DataFrame({"row_id": p["row_id"], "fold": 0, "ecfp_cluster": p["ecfp_cluster"].values})
    for i, (_, te) in enumerate(gkf(p["ecfp_cluster"].values, 5)):
        s4b.loc[te, "fold"] = i + 1
    out["S4b_ecfp_class"] = s4b
    return out


# ════════════════════════════════════════════════════════════════════
# Stage 6 — 诊断
# ════════════════════════════════════════════════════════════════════
def eta2(df: pd.DataFrame, key: str, y: str = "y") -> float:
    gm = df[y].mean()
    sst = ((df[y] - gm) ** 2).sum()
    if sst == 0:
        return 0.0
    ssb = df.groupby(key, dropna=False)[y].agg(["mean", "size"]).eval("size * (mean - @gm)**2").sum()
    return float(ssb / sst)


def diagnostics(p: pd.DataFrame, df: pd.DataFrame) -> dict:
    d = {}
    say("\n[S6] ── 靶变量分布 ─────────────────────────────")
    y = p["y"]
    yabs = y.abs()
    say(f"      y = ln α_true（CIP 定向）  中位 {y.median():+.3f}  均值 {y.mean():+.3f}  "
        f"SD {y.std():.3f}  min {y.min():+.3f}  max {y.max():+.3f}")
    say(f"      |y| = |ln α|（无向）        中位 {yabs.median():.3f}  均值 {yabs.mean():.3f}  "
        f"→ 无向 α 中位 {math.exp(yabs.median()):.3f}  最大 {math.exp(yabs.max()):.3f}")
    say(f"      EMO = 1 (S 后出峰) 占比 {p['EMO'].mean()*100:.1f}%  ← 若≈50% 说明 CIP 定向无系统偏置")
    d.update(y_median=float(y.median()), y_sd=float(y.std()), yabs_median=float(yabs.median()),
             alpha_abs_median=float(math.exp(yabs.median())),
             emo_frac=float(p["EMO"].mean()), alpha_max=float(math.exp(yabs.max())))

    say("\n[S6] ── 方差分解（对新靶变量复算审计 §3.4）──────")
    for key, label in (("Column", "柱 Column"), ("murcko", "Murcko 骨架"),
                       ("doi", "文献 DOI"), ("conn", "精确分子 conn"), ("family", "CSP 家族")):
        e = eta2(p, key)
        d[f"eta2_{key}"] = e
        say(f"      {label:18s} 组数 {p[key].nunique():5d}   η² = {e*100:5.1f}%")

    say("\n[S6] ── 实验噪声底（对条件单元内重复测重复算）──")
    rep = df[df.duplicated(subset=["conn", "Column", "i-PrOH_proportion", "Speed", "smi_iso"], keep=False)]
    if len(rep):
        g = rep.groupby(["conn", "Column", "i-PrOH_proportion", "Speed", "smi_iso"])["k"]
        rsd = (g.std() / g.mean()).dropna()
        say(f"      重复组 {len(rsd)}｜k 的 RSD 中位 {rsd.median()*100:.2f}%（审计报 2.74%）")
        say(f"      → 传导到 ln α 的噪声 ≈ {math.sqrt(2)*rsd.median()*100:.2f}%，"
            f"靶变量 SD {y.std():.3f}，噪声/信号 ≈ {math.sqrt(2)*rsd.median()/y.std():.3f}")
        say(f"      → 重复性决定的 R² 上限 ≈ {1-(math.sqrt(2)*rsd.median()/y.std())**2:.3f}")
        d["k_rsd_median"] = float(rsd.median())

    say("\n[S6] ── 选择偏倚（数据卡必写）──────────────────")
    # ⚠️ 必须用**无向** |α| 判偏倚：定向靶 y 天然约一半为负，拿它统计会把「R 后出峰」误当「分不开」
    a = np.exp(yabs)
    say(f"      |α| = 1.000 的对 {int((a <= 1.0001).sum())} ({((a <= 1.0001).mean()*100):.2f}%)｜"
        f"|α| < 1.05 的 {int((a < 1.05).sum())} ({((a < 1.05).mean()*100):.1f}%)｜"
        f"|α| > 2 的 {int((a > 2).sum())} ({((a > 2).mean()*100):.1f}%)")
    say("      ⚠️ 数据集中**没有「分不开」的样本**（文献只报成功分离）→ 不可写成「指导选柱」")
    d["alpha_le_1"] = int((a <= 1.0001).sum())
    d["alpha_frac_lt_105"] = float((a < 1.05).mean())

    say("\n[S6] ── CSP 家族 × 数据量 ─────────────────────")
    fam = p.groupby("family").agg(n=("y", "size"), n_col=("Column", "nunique"),
                                  y_med=("y", "median")).sort_values("n", ascending=False)
    for f, r in fam.iterrows():
        say(f"      {f:16s} {int(r.n):5d} 对 / {int(r.n_col)} 柱  lnα 中位 {r.y_med:+.3f}")
    d["n_families"] = len(fam)
    return d


# ════════════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════════════
def main() -> None:
    say("#" * 78)
    say("# 01_curate_cmrt — CMRT 主表清洗（规格：计算方案 §1.2 审计修正版）")
    say("#" * 78)

    raw = load_raw()
    df = build_molecules(raw.copy())
    df = add_retention(df)

    # 异常 flag（审计 §3.9）
    df["flag_rt_gt_60"] = df["RT"] > RT_MAX
    df["flag_speed_out"] = df["Speed"] > SPEED_MAX
    df["flag_k_nonpos"] = df["k"] <= 0
    df["flag_undef_stereo"] = df["n_stereo_undef"].fillna(0) > 0
    say(f"[S2] flag: RT>60min {int(df['flag_rt_gt_60'].sum())}｜Speed>{SPEED_MAX} "
        f"{int(df['flag_speed_out'].sum())}｜k≤0 {int(df['flag_k_nonpos'].sum())}｜"
        f"立体未定义 {int(df['flag_undef_stereo'].sum())}")

    p, stats = build_pairs(df)

    p["flag_alpha_extreme"] = p["y"].abs() > math.log(ALPHA_ABS_MAX)
    # 文献报告两对映体 **完全相同** 的保留时间 → α 恰为 1。这类点 y≡0：
    #   · 回归里无信息（且暗示原文根本没分开或只给了一个 RT）
    #   · **EMO 分类任务必须剔除**（并列值没有"谁先出峰"可言，EMO 标签是任意的）
    p["flag_no_separation"] = np.isclose(p["alpha_true"], 1.0, rtol=1e-9)
    p["flag_cond_pooled"] = ~p["cond_matched"]
    p["flag_dup_pair"] = p.duplicated(subset=["conn", "Column", "smi_S"], keep=False)
    say(f"[S3] flag_cip_ambiguous（CIP 签名不互补，符号语义不可信）{int(p['flag_cip_ambiguous'].sum())} 对")
    say(f"[S3] flag_no_separation（α 恰为 1，EMO 无定义）{int(p['flag_no_separation'].sum())} 对"
        f"（{p['flag_no_separation'].mean()*100:.2f}%）")
    say(f"[S3] flag: |ln α| > ln{ALPHA_ABS_MAX:.0f} {int(p['flag_alpha_extreme'].sum())}｜"
        f"跨条件合并 {int(p['flag_cond_pooled'].sum())}｜同(分子,柱)多条条件 {int(p['flag_dup_pair'].sum())}")

    cols = build_columns(p)

    # 记录级清洗表落盘（含 flag，可溯源）
    keep = ["SMILES", "smi_iso", "smi_flat", "conn", "inchikey", "murcko", "cip", "n_stereo",
            "n_stereo_undef", "RT", "Speed", "i-PrOH_proportion", "Column", "t0", "k",
            "doi", "doi_raw"] + [c for c in df.columns if c.startswith("flag_")]
    df[keep].to_csv(CUR / "cmrt_measurements.csv", index=False, encoding="utf-8")
    say(f"\n[落盘] data/curated/cmrt_measurements.csv  ({len(df)} 行)")

    # 主表落盘
    p = p.sort_values(["Column", "conn"]).reset_index(drop=True)
    p.insert(0, "pair_id", ["P%05d" % i for i in range(len(p))])
    p = p.merge(cols[["Column", "surface", "subs_code", "particle_um", "family"]],
                on="Column", how="left", suffixes=("", "_c"))
    main_cols = ["pair_id", "conn", "murcko", "smi_S", "smi_R", "cip_S", "cip_R", "n_stereo",
                 "Column", "surface", "subs_code", "particle_um", "family",
                 "ipr", "speed", "t0", "k_S", "k_R", "alpha_true", "y", "EMO",
                 "rt_S", "rt_R", "dRT_over_t0", "rt_ratio_spike",
                 "n_rep", "kS_n", "kR_n", "kS_rsd", "kR_rsd",
                 "doi", "doi_raw", "cond_matched",
                 "flag_alpha_extreme", "flag_cond_pooled", "flag_dup_pair",
                 "flag_cip_ambiguous", "flag_no_separation"]
    p[main_cols].to_csv(CUR / "cmrt_pairs.csv", index=False, encoding="utf-8")
    say(f"[落盘] data/curated/cmrt_pairs.csv  ★ 主表 ({len(p)} 行 × {len(main_cols)} 列)")
    cols.to_csv(CUR / "columns.csv", index=False, encoding="utf-8")
    say(f"[落盘] data/curated/columns.csv  ({len(cols)} 根柱)")

    d = diagnostics(p, df)

    # 划分
    say("\n[S5] ── 生成划分索引 ─────────────────────────")
    sp = make_splits(p, cols)
    for name, dfo in sp.items():
        f = SPLITS / f"{name}.csv"
        dfo.to_csv(f, index=False)
        say(f"      {name:20s} → {f.name}  ({len(dfo)} 行)")
    # 主表也带上最常用两个划分的 fold（方便直接建模）
    p2 = p[["pair_id", "y", "EMO", "Column", "conn", "murcko", "doi", "family", "subs_code",
            "surface", "particle_um", "ipr", "speed", "flag_cond_pooled",
            "flag_no_separation", "flag_cip_ambiguous"]].copy()
    p2["S1_random_fold"] = sp["S1_random"]["fold"].values
    p2["S2_group_fold"] = sp["S2_group_dup"]["fold"].values
    p2["S2b_paper_fold"] = sp["S2b_leave_paper"]["fold"].values
    p2["S4_murcko_fold"] = sp["S4_murcko"]["fold"].values
    p2.to_csv(CUR / "cmrt_pairs_with_folds.csv", index=False, encoding="utf-8")
    say(f"      合并 fold 的主表 → data/curated/cmrt_pairs_with_folds.csv")

    # 异常复核样本
    rev = p[(p["flag_alpha_extreme"]) | (p["k_S"] <= 0) | (p["k_R"] <= 0)]
    rev[["pair_id", "conn", "smi_S", "smi_R", "Column", "ipr", "speed", "k_S", "k_R",
         "alpha_true", "doi"]].to_csv(NOTES / "异常复核样本.csv", index=False, encoding="utf-8")
    say(f"[落盘] notes/异常复核样本.csv  ({len(rev)} 行需人工复核)")

    # 统计 json
    allstats = dict(stats=stats, diag=d, columns=int(len(cols)),
                    families=int(cols["family"].nunique()))
    (CUR / "curation_stats.json").write_text(
        json.dumps(allstats, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # 报告
    (NOTES / "01_curate_report.md").write_text(
        "# 01_curate_cmrt 运行报告\n\n> 规格：`docs/计算方案.md` §1.2（2026-09-16 审计修正版）\n\n```\n"
        + "\n".join(LOG) + "\n```\n", encoding="utf-8")
    say(f"[落盘] notes/01_curate_report.md")
    say("\n完成。")


if __name__ == "__main__":
    main()
