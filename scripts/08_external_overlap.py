# -*- coding: utf-8 -*-
"""
08_external_overlap.py — 外部验证集（DG25 / PE25）与 CMRT 的**分子重叠检查**
============================================================================
ROADMAP 要求项（【照抄 B5-2 Kwon 的做法】）：外部验证前必须确认源/目标之间**没有分子重叠**，
否则「外部验证」是自欺欺人。

同时回答一个更根本的问题：**DG25 与 PE25 是不是同一个分子集？**
（两者作者、体系、分子名高度相似 → 若重合，就不能当成两套独立外部验证）

路径：
  1. 从 xlsx 抽分子名（DG25 的 *final* 表用真实药名；PE25 的 Molecule 列）
  2. 用 **PubChem PUG REST** 解析名称 → SMILES（NCI CACTUS 已实测 503/500 不可用）
     ⚠️ 本机 curl/Windows schannel 必须加 `--ssl-no-revoke`，否则 TLS 吊销检查失败
  3. 算 InChIKey **连接层（前 14 位）** —— 与 CMRT 主表的 `conn` 同一口径
  4. 报告三向重叠

用法: PYTHONIOENCODING=utf-8 python scripts/08_external_overlap.py
产出: data/curated/external_molecules.csv + notes/外部验证集_重叠检查.md
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
CUR = ROOT / "data" / "curated"
ZEN = ROOT / "data" / "raw" / "zenodo"
# ⚠️ 缓存**不能**放在 data/curated/ 里：那里是"可随时重建的产出物"，
#    而缓存是**外部服务查询结果**，重建一次要重新打 43 次 PubChem（会被限流 503）。
CACHE = ROOT / "data" / "external_name_to_smiles.json"
LOG: list[str] = []


def say(s: str = "") -> None:
    print(s)
    LOG.append(s)


# ────────────────────────────────────────────── PubChem 解析
# 名称规范化：源数据里有拼写变体（法/荷语系）与非分子条目
NAME_FIX = {
    "aminogluthetimide": "Aminoglutethimide",   # 常见拼写变体（少一个 e）
    "blebbistatine": "Blebbistatin",            # 法语尾音 e
    "bupivacaïne": "Bupivacaine",               # 去分音符
    "warfarin e": "Warfarin",
}
NON_MOLECULE = {"dode tijd", "dode tijd (min)", "dead time"}   # 荷兰语「死时间」，混进了首列


def normalize(name: str) -> str | None:
    """去首尾空白 + 去分音符 + 查拼写修正表；返回 None 表示不是分子条目。"""
    import unicodedata
    s = unicodedata.normalize("NFKD", str(name).strip())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    if s.lower() in NON_MOLECULE:
        return None
    return NAME_FIX.get(s.lower(), s)


def pubchem_smiles(name: str, tries: int = 3) -> str | None:
    """名称 → 异构 SMILES。PubChem 限速 ~5 req/s，这里天然慢于该速率。"""
    from urllib.parse import quote
    url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quote(name)}"
           f"/property/SMILES,ConnectivitySMILES/JSON")
    for k in range(tries):
        try:
            r = subprocess.run(
                ["curl", "-sS", "--ssl-no-revoke", "--max-time", "30", url],
                capture_output=True, text=True, encoding="utf-8", timeout=40)
            out = r.stdout or ""
            if r.returncode != 0 or not out.strip():
                time.sleep(2.0 * (k + 1))
                continue
            if "Too many requests" in out or "503" in out[:200]:
                # PubChem 限流：退避得更狠（实测一次突发 44 个请求就会触发）
                time.sleep(10.0 * (k + 1))
                continue
            j = json.loads(out)
            props = j["PropertyTable"]["Properties"][0]
            return props.get("SMILES") or props.get("ConnectivitySMILES")
        except Exception:
            time.sleep(2.0 * (k + 1))
    return None


def chembl_smiles(name: str, tries: int = 2) -> str | None:
    """备用源：EBI ChEMBL。独立于 PubChem，不受其限流影响。
    ⚠️ 返回的 `canonical_smiles` **可能不带立体信息**（外消旋药物常见）——
       做「连接层重叠检查」够用，但**不能**用来定 R/S 定向。"""
    from urllib.parse import quote
    url = (f"https://www.ebi.ac.uk/chembl/api/data/molecule/search?q={quote(name)}"
           f"&format=json&limit=1")
    for k in range(tries):
        try:
            r = subprocess.run(
                ["curl", "-sS", "--ssl-no-revoke", "--max-time", "30", url],
                capture_output=True, text=True, encoding="utf-8", timeout=40)
            out = r.stdout or ""
            if r.returncode != 0 or not out.strip():
                time.sleep(1.5 * (k + 1))
                continue
            j = json.loads(out)
            mols = j.get("molecules") or []
            if not mols:
                return None
            ms = mols[0].get("molecule_structures") or {}
            return ms.get("canonical_smiles")
        except Exception:
            time.sleep(1.5 * (k + 1))
    return None


# 🔴 熔断器：PubChem 一旦限流（503），**整轮跳过它**。
#    否则每个名字都要白等 3×10 s 退避 —— 实测 43 个名字会拖到 20 分钟以上。
_PUBCHEM_DEAD = False


def resolve_one(name: str) -> tuple[str | None, str]:
    """PubChem 优先 → ChEMBL 兜底（PubChem 限流时直接走 ChEMBL）。返回 (SMILES, 来源)。"""
    global _PUBCHEM_DEAD
    if not _PUBCHEM_DEAD:
        s = pubchem_smiles(name, tries=1)
        if s:
            return s, "pubchem"
        _PUBCHEM_DEAD = True
        print("    [!] PubChem 不可用（限流？）→ 本轮改用 ChEMBL")
    s = chembl_smiles(name)
    if s:
        return s, "chembl"
    return None, "none"


def resolve(names: list[str], force: bool = False) -> dict[str, str | None]:
    """带缓存的名称解析。`force=True` 时把缓存里的 None 也重试一遍。"""
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    todo = [n for n in names if force or n not in cache or cache.get(n) is None]
    if todo:
        say(f"需解析 **{len(todo)}** 个名称（缓存现有 {len(cache)} 条）…")
        src_count: dict[str, int] = {}
        for i, n in enumerate(todo, 1):
            smi, src = resolve_one(n)
            cache[n] = smi
            src_count[src] = src_count.get(src, 0) + 1
            if i % 5 == 0 or i == len(todo):
                print(f"    {i}/{len(todo)}  {n} → {str(smi)[:44]}")
                CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
            time.sleep(0.6)           # 礼貌限速（PubChem 突发会被 503）
        CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
        say("")
        say(f"解析来源统计：{src_count}  （`pubchem` = PubChem PUG REST，`chembl` = EBI 兜底）")
    return {n: cache[n] for n in names}


def conn_of(smi: str | None) -> str | None:
    if not smi:
        return None
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToInchiKey(m)[:14] if m else None


# ────────────────────────────────────────────── 名称抽取
def names_from_pe25() -> list[str]:
    x = pd.ExcelFile(ZEN / "peeters2025" / "Borate_chiral_exp_water_ACN.xlsx")
    out: set[str] = set()
    for s in x.sheet_names:
        d = x.parse(s)
        if "Molecule" in d.columns:
            for v in d["Molecule"].dropna().astype(str):
                n = normalize(v)
                if n:
                    out.add(n)
    return sorted(out)


def names_from_dg25() -> list[str]:
    out: dict[str, int] = {}
    for f in sorted(os.listdir(ZEN / "degauquier2025")):
        if not f.endswith(".xlsx"):
            continue
        x = pd.ExcelFile(ZEN / "degauquier2025" / f)
        for s in x.sheet_names:
            d = x.parse(s)
            if d.shape[1] == 0:
                continue
            col = d.iloc[:, 0].dropna().astype(str).str.strip()
            # 真实药名 = 含字母且不是 AZ1/OD-RH 这类内部编号
            real = [v for v in col if v and not v.upper().startswith("AZ")
                    and not v.upper().startswith(("OD", "LA-", "LC-", "OJ", "IA", "IB", "IC"))]
            for v in real:
                n = normalize(v)
                if n:
                    out[n] = out.get(n, 0) + 1
    return sorted(out)


# ────────────────────────────────────────────── main
def main() -> int:
    say("# 外部验证集（DG25 / PE25）与 CMRT 的分子重叠检查")
    say("")
    say("> 由 `scripts/08_external_overlap.py` 生成 ｜ 判据：**InChIKey 连接层（前 14 位）**")
    say("")

    p = pd.read_csv(CUR / "cmrt_pairs.csv")
    cmrt_conn = set(p["conn"].unique())
    say(f"CMRT 主表：{len(p)} 对 / **{len(cmrt_conn)} 个唯一骨架（conn）**")
    say("")

    pe = names_from_pe25()
    dg = names_from_dg25()
    say(f"## 名称抽取")
    say("")
    say(f"- **PE25**：{len(pe)} 个唯一分子名")
    say(f"- **DG25**：{len(dg)} 个唯一分子名（取自 `*final*` 表；`Achiral/Chiral descriptors` 表用的是 AZxx 内部编号）")
    say("")

    both = sorted(set(pe) & set(dg))
    say(f"### ⚠️ DG25 与 PE25 的重叠：**{len(both)} / {len(set(pe)|set(dg))}** 个分子同时出现在两边")
    say("")
    say(f"`{'`, `'.join(both[:20])}`" + (" …" if len(both) > 20 else ""))
    say("")
    if len(both) >= 0.5 * min(len(pe), len(dg)):
        say("→ 🔴 **两者高度重合，不是两套独立的外部验证集。**")
        say("  论文里**不能**把它们当成两次独立外部验证来报；应合并为一套，或明确说明关系。")
    say("")

    names = sorted(set(pe) | set(dg))
    say(f"## PubChem 解析（共 {len(names)} 个名称）")
    say("")
    mp = resolve(names)
    ok = {n: v for n, v in mp.items() if v}
    say("")
    say(f"解析成功 **{len(ok)}/{len(names)}**（{len(names)-len(ok)} 个失败，多为商品名/拼写变体/无立体信息）")
    bad = [n for n, v in mp.items() if not v]
    if bad:
        say("")
        say(f"失败清单：`{'`, `'.join(bad)}`")
    say("")

    rows = []
    for n in names:
        src = ("DG25" if n in dg else "") + ("+PE25" if n in pe and n in dg else ("PE25" if n in pe else ""))
        rows.append(dict(name=n, source=src, smiles=mp.get(n), conn=conn_of(mp.get(n))))
    ext = pd.DataFrame(rows)
    ext["in_cmrt"] = ext["conn"].isin(cmrt_conn)
    # 🔴 守卫：解析全失败时**必须报错退出**，不能静默写出「SMILES 全是 NaN」的表
    #    ——首版就是这么静默失败过，下游 09/10 拿到空表还以为一切正常。
    rate = float(ext["conn"].notna().mean())
    if rate < 0.5:
        say("")
        say(f"### ❌ 中止：PubChem 解析成功率仅 {rate*100:.0f}%"
            f"（{int(ext['conn'].notna().sum())}/{len(ext)}）")
        say("")
        say("大概率是 **PubChem 限流（HTTP 503 Too many requests）**。")
        say("处置：等几分钟后重跑（缓存 `data/external_name_to_smiles.json` 会保留已成功的条目），")
        say("**不要**让下游拿到这张表 —— 全 NaN 的 SMILES 会让 09/10 静默产出空结果。")
        (ROOT / "notes" / "外部验证集_重叠检查.md").write_text(
            "\n".join(LOG) + "\n", encoding="utf-8")
        raise SystemExit(2)
    ext.to_csv(CUR / "external_molecules.csv", index=False, encoding="utf-8")

    n_res = int(ext["conn"].notna().sum())
    n_hit = int(ext["in_cmrt"].sum())
    say("## 与 CMRT 的重叠（源/目标泄漏检查）")
    say("")
    say(f"- 可解析出 conn 的外部分子：**{n_res}**")
    say(f"- 其中 **{n_hit}** 个的连接层出现在 CMRT 主表中（**{n_hit/max(n_res,1)*100:.1f}%**）")
    say("")
    if n_hit:
        say("| 分子名 | 来源 | 是否在 CMRT |")
        say("|---|---|---|")
        for _, r in ext[ext["in_cmrt"]].iterrows():
            say(f"| {r['name']} | {r['source']} | ✅ |")
        say("")
        say(f"→ ⚠️ 这 {n_hit} 个分子**必须从外部验证集中剔除**（或反过来从 CMRT 训练集里剔除），")
        say("  否则「外部验证」会退化成「内部测试」。")
    else:
        say("→ ✅ 连接层上**无重叠**。但注意：连接层相同 ≠ 同一分子，")
        say("  连接层不同也**不能保证**分子不同（只是骨架不同）；反过来若有 0 命中也说明")
        say("  这套外部数据在化学空间上离 CMRT 较远 —— 对「外推能力」是更严苛的检验。")

    (ROOT / "notes" / "外部验证集_重叠检查.md").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    print(f"\n[落盘] data/curated/external_molecules.csv ({len(ext)} 行)")
    print("[落盘] notes/外部验证集_重叠检查.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
