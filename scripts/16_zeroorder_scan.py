# -*- coding: utf-8 -*-
"""
16_zeroorder_scan.py — **各代描述符与目标的零阶相关扫描**
================================================================================
## 为什么做这个

`06b`/`06d`/`14` 都是「**放进模型看 R²**」。但模型是黑箱：某个描述符块没带来 R²，
既可能是「它没信息」，也可能是「它有信息但被其他列盖住了」。
**零阶相关是最干净的补充判据**：一个手性描述符若连与目标的单变量相关都接近 0，
那「它携带手性信息」这个说法就直接被钉死了，不需要再争论模型。

## 扫描对象

| 族 | 有符号列（**伪标量**，应携带手性） | 偶量列（**无手性**，阴性对照） |
|---|---|---|
| L1b（CMCDS TD-DFT 旋转强度） | `R_sum`, `Rw_lam`, 分带 Σ R | `R_absmean`, `R_absmax`, `E_mean`, λ |
| L1-CF（MMFF 局部手性坐标系） | `*_z*` 等有符号列 | `*abs*` / 幅度列 |
| L1a-FF（MMFF 立体中心支矢量） | `*_bw`（`D`） | `*_c*` 偶量列 |

## 判据
- **伪标量列的 \\|r\\| 中位 ≈ 0** ⇒ 「手性信息没进到目标里」（本项目至今的实测）；
- 若**偶量列 \\|r\\| 明显大于伪标量列** ⇒ 信号来自**无手性**那部分（重复 `06b` 的结论）。

## 用法
```bash
PYTHONIOENCODING=utf-8 python scripts/16_zeroorder_scan.py
```
产出：`notes/零阶相关扫描.md` + `data/curated/zeroorder_scan.csv`
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parent.parent
CUR = ROOT / "data" / "curated"
LOG: list[str] = []
NEXC = 20


def say(s: str = "") -> None:
    print(s, flush=True)
    LOG.append(s)


def canon(s: str):
    m = Chem.MolFromSmiles(s)
    return None if m is None else Chem.MolToSmiles(m, isomericSmiles=True)


def build_l1b_aligned():
    """CMCDS → CMRT 的 S 参考（与 `14` 同一套口径，必须一致）。"""
    cm = pd.read_csv(CUR / "cmrt_pairs.csv", low_memory=False)
    cd = pd.read_csv(CUR / "cmcds_l1b.csv")
    mono = cm[cm["cip_S"].astype(str).str.len() == 1].drop_duplicates("conn").copy()
    mono["cs_S"] = [canon(s) for s in mono["smi_S"]]
    mono["cs_R"] = [canon(s) for s in mono["smi_R"]]
    cd["cs"] = [canon(s) for s in cd["smiles"]]
    SS, SR = set(mono["cs_S"].dropna()), set(mono["cs_R"].dropna())

    def side(c):
        a, b = c in SS, c in SR
        return 1 if (a and not b) else (-1 if (b and not a) else 0)

    k2c = {}
    for _, r in mono.iterrows():
        if pd.notna(r["cs_S"]):
            k2c.setdefault(r["cs_S"], r["conn"])
        if pd.notna(r["cs_R"]):
            k2c.setdefault(r["cs_R"], r["conn"])
    cd["conn"] = [k2c.get(c) for c in cd["cs"]]
    cd["align"] = [side(c) for c in cd["cs"]]
    cd = cd.dropna(subset=["conn"])
    cd = cd[cd["align"] != 0].drop_duplicates("conn").set_index("conn")
    sg = cd["align"].values
    lam = cd[[f"lam{i}" for i in range(1, NEXC + 1)]].values
    Rv = cd[[f"R{i}" for i in range(1, NEXC + 1)]].values * sg[:, None]
    out = pd.DataFrame(index=cd.index)
    out["L1b:R_sum"] = Rv.sum(axis=1)
    w = np.abs(Rv).sum(axis=1)
    out["L1b:Rw_lam"] = (Rv * lam).sum(axis=1) / np.maximum(w, 1e-12)
    for lo, hi in [(100, 150), (150, 200), (200, 250), (250, 300), (300, 999)]:
        m = (lam >= lo) & (lam < hi)
        out[f"L1b:Rb{lo}-{hi}"] = np.where(m, Rv, 0.0).sum(axis=1)
    out["L1b:absmean"] = np.abs(Rv).mean(axis=1)
    out["L1b:absmax"] = np.abs(Rv).max(axis=1)
    out["L1b:E_mean"] = cd["E_mean"].values
    out["L1b:lam_max"] = cd["lam_max"].values
    return out


def main() -> int:
    say("# 零阶相关扫描：各代描述符 vs 目标")
    say("")
    say("> 由 `scripts/16_zeroorder_scan.py` 生成 ｜ 2026-09-17")
    say("> `r = corr(列, y)`，y = ln[k(S)/k(R)] ｜ `r_emo = corr(列, EMO)`（洗脱顺序）")
    say("")

    p = pd.read_csv(CUR / "cmrt_pairs.csv", low_memory=False)
    blk = {"L1b": build_l1b_aligned()}
    CF = pd.read_csv(CUR / "l1cf_descriptors.csv").set_index("conn")
    FF = pd.read_csv(CUR / "l1a_descriptors.csv").set_index("conn")
    blk["L1-CF"] = CF[[c for c in CF.columns if c.startswith("cf_")]]
    blk["L1a-FF"] = FF[[c for c in FF.columns if c.endswith(("_bw", "_sc", "_c"))]]

    rows = []
    for fam, T in blk.items():
        m = p.merge(T, left_on="conn", right_index=True, how="inner")
        say(f"## {fam}（匹配 {len(m)} 行 / {m['conn'].nunique()} 分子）")
        say("")
        say("| 列 | r（与 y） | r（与 EMO） | |r| 排序 |")
        say("|---|---|---|---|")
        rec = []
        for c in T.columns:
            v = m[c].values.astype(float)
            if np.nanstd(v) <= 0:
                continue
            r = float(np.corrcoef(v, m["y"].values)[0, 1])
            re_ = float(np.corrcoef(v, m["EMO"].values)[0, 1])
            rec.append((c, r, re_))
        rec.sort(key=lambda t: -abs(t[1]))
        for i, (c, r, re_) in enumerate(rec[:12], 1):
            say(f"| `{c}` | {r:+.4f} | {re_:+.4f} | {i} |")
        a = np.array([abs(r) for _, r, _ in rec])
        say("")
        say(f"- **{fam}** 共 {len(rec)} 列 ｜ \\|r\\| 中位 **{np.median(a):.4f}** ｜ "
            f"最大 {a.max():.4f} ｜ \\|r\\|>0.05 的列占 **{(a>0.05).mean()*100:.0f}%**")
        say("")
        rows += [dict(family=fam, col=c, r=r, r_emo=re_) for c, r, re_ in rec]

    say("## ★ 汇总（把「有符号列」与「偶量列」分开看）")
    say("")
    D = pd.DataFrame(rows)

    def is_signed(fam, c):
        if fam == "L1b":
            return not c.startswith(f"{fam}:") or c.split(":")[1] in (
                "R_sum", "Rw_lam") or c.split(":")[1].startswith("Rb")
        if fam == "L1-CF":
            import re
            return not re.search(r"abs|mag|_n$|meanr|rho", c)
        return c.endswith("_bw")

    D["signed"] = [is_signed(f, c) for f, c in zip(D["family"], D["col"])]
    say("| 族 | 类型 | 列数 | \\|r\\| 中位 | \\|r\\| 最大 | \\|r\\|>0.05 占比 |")
    say("|---|---|---|---|---|---|")
    for (fam, sg), g in D.groupby(["family", "signed"]):
        a = g["r"].abs().values
        say(f"| {fam} | {'**有符号（伪标量）**' if sg else '偶量（无手性）'} | {len(g)} | "
            f"**{np.median(a):.4f}** | {a.max():.4f} | {(a>0.05).mean()*100:.0f}% |")
    say("")
    say("> 读法：**伪标量列的 \\|r\\| 中位若与偶量列同档甚至更低**，")
    say("> 就说明「手性信息」并没有比「无手性的形状/尺寸」提供更多关于目标的信息。")

    D.to_csv(CUR / "zeroorder_scan.csv", index=False, encoding="utf-8")
    (ROOT / "notes" / "零阶相关扫描.md").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    print("\n[落盘] notes/零阶相关扫描.md + data/curated/zeroorder_scan.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
