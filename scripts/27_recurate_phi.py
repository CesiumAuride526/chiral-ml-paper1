# -*- coding: utf-8 -*-
"""
27_recurate_phi.py — **在不同孔隙率下重跑完整清洗流程**
================================================================================
## 立项理由

`scripts/25` 只做了一半：它改了 `t₀` 并重算 `y`，但**没有重跑配对与筛选**。
而 `k > 0` 那一步、以及条件单元折叠，都会随 `t₀` 变化（`scripts/25` 的表里
「k ≤ 0 行数」从 0 变到 11 就是这个迹象）。

⇒ 本脚本把 `01_curate_cmrt.py` 在 **φ = 0.55 / 0.62 / 0.70** 下**整条重跑一遍**，
比较产出主表的：
  ① 对数  ② `y` 的分布  ③ `η²(doi)`（论文的核心量之一）  ④ 靶变量与基准的相关系数

**判据（事先写死）**：三个 φ 下 ① 对数差异 < 1%、② `η²(doi)` 差异 < 0.02，
则认为**「论文的核心数字不依赖孔隙率」**。

## 安全措施（重要）
本脚本**不会覆盖** `data/curated/` 下任何**既有**正式产物（主表 / 记录级表 / 划分 / 描述符表）：
它在导入 `01_curate_cmrt` 后，把模块级的 `CUR` / `NOTES` / `SPLITS`
**改指向 `scratch/_phi_<φ>/`**，再调用其 `main()`。
⇒ 正式主表**原封不动**，01 的全部临时产物落在 `scratch/` 下。

> ⚠️ **唯一的例外（第八场代码审计更正）**：本脚本自己会**新增/覆盖一个结果 CSV**
> `data/curated/recurate_phi.csv`（三档对照表，见 `27:150` 的 `R.to_csv(CUR / "recurate_phi.csv")`）。
> 原文写「不会覆盖 `data/curated/` 下的任何正式产物」，与这个文件的存在矛盾 ——
> 准确表述是：**不覆盖主表等既有正式产物；会写一个 `recurate_phi.csv`**。
> 若要让「零写入 `data/curated`」严格成立，把 `27:150` 的 `CUR` 换成 `SCRATCH` 即可
> （但门禁/文档要同步改路径，本轮不做）。

## 产出
- `data/curated/recurate_phi.csv`（三档对照表；**本脚本唯一写入 `data/curated` 的文件**）
- `notes/孔隙率重清洗对照.md`
- （临时）`scratch/_phi_<φ>/` 下 01 的整条产物

## 用法
```bash
PYTHONIOENCODING=utf-8 python scripts/27_recurate_phi.py
```
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CUR = ROOT / "data" / "curated"
SCRATCH = ROOT / "scratch"
LOG: list[str] = []
PHIS = [0.55, 0.62, 0.70]


def say(s: str = "") -> None:
    print(s, flush=True)
    LOG.append(s)


def eta2(df: pd.DataFrame, key: str) -> float:
    grand = df["y"].mean()
    g = df.groupby(key, observed=True)["y"].agg(["mean", "size"])
    return float((g["size"] * (g["mean"] - grand) ** 2).sum()
                 / ((df["y"] - grand) ** 2).sum())


def main() -> int:
    import io
    import contextlib

    t_all = time.time()
    say("# 孔隙率下的**完整重清洗**对照")
    say("")
    say("> 由 `scripts/27_recurate_phi.py` 生成 ｜ 2026-09-17")
    say("")
    say("`scripts/25` 只改了 `t₀` 与 `y`，**没有重跑配对与筛选**；本脚本把")
    say("`01_curate_cmrt.py` 在三个孔隙率下**整条重跑**，比较产出主表。")
    say("")
    say("> 🔒 **既有正式产物不受影响**：脚本把模块级 `CUR`/`NOTES`/`SPLITS` 改指向 "
        "`scratch/_phi_<φ>/` 后才调用 `main()`；唯一写入 `data/curated/` 的是本脚本自己的")
    say(">    `recurate_phi.csv`（对照表），**主表/划分/描述符表都不会被改**。")
    say("")

    base = pd.read_csv(CUR / "cmrt_pairs.csv", low_memory=False)
    say(f"- 基准（φ=0.62）主表：**{len(base)}** 行 / {base['doi'].nunique()} 篇 DOI ｜ "
        f"`η²(doi) = {eta2(base, 'doi'):.4f}`")
    say("")

    rows, tables = [], {}
    for phi in PHIS:
        t0 = time.time()
        out = SCRATCH / f"_phi_{phi}"
        (out / "splits").mkdir(parents=True, exist_ok=True)
        (out / "notes").mkdir(parents=True, exist_ok=True)

        spec = importlib.util.spec_from_file_location(
            f"_curate_{phi}", ROOT / "scripts" / "01_curate_cmrt.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # ★ 改指向临时目录 —— **绝不碰正式产物**
        mod.CUR = out
        mod.SPLITS = out / "splits"
        mod.NOTES = out / "notes"
        mod.POROSITY = phi

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            mod.main()

        p = pd.read_csv(out / "cmrt_pairs.csv", low_memory=False)
        tables[phi] = p
        say(f"（φ={phi} 重跑完成，{time.time()-t0:.0f} s，主表 **{len(p)}** 行）")
    say("")

    say("## 1. 三档对照")
    say("")
    # ⚠️ **必须按业务键 join，不能按行号比**：
    #   重跑后主表的**行序不同**（`pair_id` 是顺序标签，会重排），
    #   按位置比较会把「同一行」当成同一个对映体对 —— 初版就是这样得到
    #   33.9%/47.2% 的「符号翻转率」，那是**错位造成的假象，不是真翻转**。
    #   业务键 = `(conn, Column, ipr, speed)`（条件单元 + 分子），两表内均唯一。
    KEY = ["conn", "Column", "ipr", "speed"]
    say("> ⚠️ **对齐口径**：两表按业务键 `(conn, Column, ipr, speed)` **join** 后比较；"
        "**不能按行号**——重跑后行序不同，按位置比会得到虚假的「符号翻转率」。")
    say("")
    say("| φ | 对数 | 与基准对数比 | 与基准共有键 | **共有键上 y 的 r** | EMO 符号翻转 | **η²(doi)** | y SD |")
    say("|---|---|---|---|---|---|---|---|")
    bkey = base.copy()
    bkey["_k"] = list(zip(bkey.conn, bkey.Column, bkey.ipr, bkey.speed))
    bkey = bkey.drop_duplicates("_k").set_index("_k")
    n0 = len(base)
    e0 = eta2(base, "doi")
    for phi in PHIS:
        p = tables[phi]
        q = p.copy()
        q["_k"] = list(zip(q.conn, q.Column, q.ipr, q.speed))
        q = q.drop_duplicates("_k").set_index("_k")
        common = bkey.index.intersection(q.index)
        yb = bkey.loc[common, "y"].values.astype(float)
        yq = q.loc[common, "y"].values.astype(float)
        eb = bkey.loc[common, "EMO"].values.astype(float)
        eq = q.loc[common, "EMO"].values.astype(float)
        r = float(np.corrcoef(yq, yb)[0, 1]) if len(common) > 10 else np.nan
        flip = float(np.mean((eq > 0) != (eb > 0))) if len(common) > 10 else np.nan
        e = eta2(p, "doi")
        rows.append(dict(phi=phi, n=len(p), n_ratio=len(p) / n0, n_common=len(common),
                         r_vs_base=r, emo_flip=flip, eta2_doi=e, y_sd=float(p["y"].std())))
        say(f"| {phi} | **{len(p)}** | {len(p)/n0:.4f} | **{len(common)}** | **{r:.5f}** | "
            f"{flip*100:.2f}% | **{e:.4f}** | {p['y'].std():.4f} |")
    say("")
    say(f"> 共有键覆盖 **{rows[0]['n_common']}/{n0} = "
        f"{rows[0]['n_common']/n0*100:.2f}%**（φ=0.70 略少，因 `k>0` 筛选随 `t₀` 变化）。")
    say("")

    R = pd.DataFrame(rows)
    R.to_csv(CUR / "recurate_phi.csv", index=False, encoding="utf-8")

    # ── 判定
    say("## 2. ★ 判定（判据事先写死）")
    say("")
    maxdev_n = float(np.max(np.abs(R["n_ratio"] - 1)))
    maxdev_e = float(np.max(np.abs(R["eta2_doi"] - e0)))
    c1 = maxdev_n < 0.01
    c2 = maxdev_e < 0.02
    say("| 判据 | 实测 | 通过 |")
    say("|---|---|---|")
    max_flip = float(R["emo_flip"].max())
    c3 = max_flip < 0.05
    say(f"| 对数与基准差异 < 1% | 最大 {maxdev_n*100:.2f}% | {'✅' if c1 else '❌'} |")
    say(f"| `η²(doi)` 与基准差异 < 0.02 | 最大 {maxdev_e:.4f} | {'✅' if c2 else '❌'} |")
    say(f"| **共有键上 EMO 符号翻转 < 5%** | 最大 {max_flip*100:.2f}% | {'✅' if c3 else '❌'} |")
    say("")
    if c1 and c2 and c3:
        say("⇒ ✅ **三条判据全过** ⇒ **完整重清洗也不改变论文的核心数字**。")
        say("   可与 `scripts/25` 一起写进补充材料：**结论不依赖死时间估算**。")
    else:
        say("⇒ ⚠️ 有判据未过，须在论文中显式讨论。")
    say("")
    say(f"> 📌 **与 `scripts/25` 的区别**：`25` 只改 `t₀` 不改筛选（快，但 `k>0` 那步没重跑）；")
    say(f"> 本脚本**整条重跑**（含配对与筛选），是本项目的**最终口径**。")
    say(">")
    say("> 🔒 01 的临时产物在 `scratch/_phi_*/`，**可整目录删除**；")
    say(">    `data/curated/` 只有 `recurate_phi.csv`（本脚本的对照表）会被写，主表等既有一律不动。")
    say("")

    notes = ROOT / "notes" / "孔隙率重清洗对照.md"
    notes.write_text("\n".join(LOG) + "\n", encoding="utf-8")
    print(f"\n[落盘] {notes}")
    print(f"[落盘] data/curated/recurate_phi.csv")
    print(f"[总耗时] {time.time()-t_all:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
