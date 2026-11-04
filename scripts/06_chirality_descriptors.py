# -*- coding: utf-8 -*-
"""
06_chirality_descriptors.py — **立体中心锚定的伪标量描述符（力场级 L1a-FF）**
===============================================================================
产物：`data/curated/l1a_descriptors.csv` + `notes/L1a_描述符构建报告.md`

> ⚠️ **命名澄清（2026-09-17 审计后新增）**
> `docs/选题决策.md` §3.2 定义的 **L-a / L1a** 是「**量子化** ESEC」——
> 矢量来自 **DFT / xTB 量**（NPA 电荷、极化率张量、ALIE、LEA…）。
> 本脚本的矢量仍是 **RDKit / MMFF 级**（Gasteiger 电荷、质量、电负性…），
> 因此它的定位是 **L1a 的冒烟前置**，正式记法为 **`L1a-FF`**（force-field 级）。
> 论文里报「L1a」时不要用本表，要报 DFT 矢量版本。

## 一、它要解决的问题（来自 `03` / `04` / `04c`）

| 脚本 | 现象 |
|---|---|
| `03`（单构象） | 对映体交换反号比例仅 **0.363**（R/S 各自落到不同盆地） |
| `04`（MMFF 20 构象） | 提到 **0.947** |
| `04c`（GFN2 48 构象） | 仍只有 **0.937** —— **升级几何级别没有改善** |

`_proto_l1a.py` 找到了 `04c` 没量过的量：**在同一个对映体自己的系综内，
全局矢量三重积的符号就不稳**（题注见 `notes/CP3b_核查笔记`）。

## 二、机制（**2026-09-17 修正后的表述**）

* **全局矢量** `v_a = Σ_i w_i (r_i − r_center)` 由**整个分子形状**决定 ——
  一次单键旋转就能把它扫过大角度 ⇒ **符号确实是构象的属性**。这一段成立。
* 因此把原点移回**立体中心**、对四个取代基分支各取属性加权矢量
  `u_b = Σ_{i∈分支b} w_i (r_i − r_k)`，四面体框架刚性 ⇒ 符号应当稳定。

## 三、⚠️ 但「局部版稳定」这个结论**必须加三条限定**（审计实测，见报告 §2–§3）

1. **不稳定主要来自数值退化，不是构象**：全量 9,968 分子上，48 列里
   **12 列**在 >50% 的分子上**恒等于 0**（其中 **5 列**在 >90% 的分子上为 0）——
   全是 `***_ring` / `***_arom`：分支里没有成环/芳香原子 ⇒ 权重全零 ⇒ 零向量。
   `_proto_l1a2` / 本脚本 2026-09-17 前的诊断用 `np.sign(0)=0`，
   会把**恒零列也算成"符号不一致"**，把「全同号描述符占比」从 **75.8%** 压低到 62%。
   **本脚本分离这两个来源**（`n_zero_col` / `n_live` / `cons_med_nz`）。
2. **符号基本是 R/S 的函数**：用 CIP 秩降序排分支后，`vol_mass` 的符号与 CIP 标签
   一致率 **92.3%**（n = 9,968；小样本 36 分子上是 97.2%）。
   ⚠️ `_proto_l1a2` 报的 **54.8% 是用未排序的旧版算的，不可引用**，
   且「一致率低 = 携带更多信息」这个推论**方向是反的**。
   ⚠️ 剩下 7.7% 的分子**不是**排序 bug（CIP 秩无平局），而是**大柔性分支让支矢量摆动到翻号**
   （系综内一致度中位仅 0.80）。见报告 §3a。
3. **反号比例 1.000 是数学恒等式**，不是实验结果：因为 R/S 由同一去立体 SMILES 建来
   ⇒ 原子序一致 ⇒ 伪标量按定义整体变号。它只验证「原子序对齐」，**不与 CP3b 的 0.937 可比**
   （后者是跨系综配对后的比较）。

### 仍然成立的正面发现
* 48 列**不是**秩 1：不同权重族（尤其 `gast`/`absq`，电荷可负 ⇒ 支矢量可反向）
  给出**方向不同**的手性信息（实测存活列之间与 `vol_mass` 的符号一致率中位 **0.848**）。
* `vol_*` 与 `tp***_*` **存在严格线性冗余**：
  `vol_w = tp123_w − tp023_w + tp013_w − tp012_w`（同一权重下）。
  ⇒ 48 列的真实秩 ≤ **40**。下游建模用 `--no-vol` 出干净的 40 列版本。

## 四、48 个描述符的定义

| 家族 | 维度 | 定义 | 备注 |
|---|---|---|---|
| `vol_*` | 8 | `det[u₁−u₀, u₂−u₀, u₃−u₀]` | ⚠️ 与 `tp***` 线性冗余 |
| `voln_*` | 8 | 同上但三矢量先单位化（**纯角度手性**） | 独立 |
| `tp{ijl}_*` | 32 | 四选三分支的三重积 `u_i·(u_j×u_l)` | 独立；`tp012` 与 `vol` **反号**（排列奇偶性） |

八套原子权重 `w`：质量 / Gasteiger 电荷 / |电荷| / 度数 / 成环 / 芳香 / 电负性 / 计数。
分支按 **CIP 秩降序**排列（`Chem.ComputeAtomCIPRanks`；失败即**报错**，不再静默回退）。

## 五、用法
```bash
PYTHONIOENCODING=utf-8 python scripts/06_chirality_descriptors.py --n-mol 0 --workers 28
PYTHONIOENCODING=utf-8 python scripts/06_chirality_descriptors.py --n-mol 400 --no-vol   # 冒烟
```
`--n-mol 0` = 全部单手性中心分子（10,117 个，约 7 min @ 28 并发）。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "1")

from rdkit import Chem, RDLogger                                    # noqa: E402
from rdkit.Chem import AllChem                                      # noqa: E402

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
CUR = ROOT / "data" / "curated"
SEED = 0xC1A1
RT = 1.98720425864083e-3 * 298.15               # kcal/mol @ 298.15 K

# 八套原子标量权重的名字（顺序 = 列顺序，不要改）
W_NAMES = ("mass", "gast", "absq", "deg", "ring", "arom", "en", "cnt")
# Pauling 电负性（缺省 2.0）
EN = {1: 2.20, 3: 0.98, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98,
      11: 0.93, 14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16, 34: 2.55,
      35: 2.96, 53: 2.66}
BRANCH_TRIPLES = list(combinations(range(4), 3))     # (0,1,2) (0,1,3) (0,2,3) (1,2,3)

# 数值退化阈值：|D| < ZERO_ABS 视为「该构象下此描述符无定义」
ZERO_ABS = 1e-12
# ⚠️ 审计修正（2026-09-17）：旧版要求 **≥3 个构象**，把 492 个分子（4.9%）判为失败。
#    但「请求 30 构象、剪枝 0.5 Å 后只剩 1–2 个」是**刚性分子的正常结果**，
#    Boltzmann 加权在 1–2 个构象上依然良定义。**不要用构象数当质量门槛。**
MIN_CONF = 1


def descriptor_names() -> list[str]:
    names = [f"vol_{w}" for w in W_NAMES] + [f"voln_{w}" for w in W_NAMES]
    for i, j, l in BRANCH_TRIPLES:
        names += [f"tp{i}{j}{l}_{w}" for w in W_NAMES]
    return names


NAMES = descriptor_names()                            # 48
VOL_COLS = [n for n in NAMES if n.startswith("vol_")]  # 8 个冗余列


# ════════════════════════════════════════════════════════════ 几何与权重
def atom_weights(mol) -> np.ndarray:
    """(n_atom, 8) 原子标量权重。**与构象无关**，每分子只算一次。"""
    AllChem.ComputeGasteigerCharges(mol, nIter=20, throwOnParamFailure=False)
    gas = np.nan_to_num(
        np.array([a.GetDoubleProp("_GasteigerCharge")
                  if a.HasProp("_GasteigerCharge") else 0.0
                  for a in mol.GetAtoms()], dtype=float),
        nan=0.0, posinf=0.0, neginf=0.0)
    W = np.empty((mol.GetNumAtoms(), 8), dtype=float)
    W[:, 0] = [a.GetMass() for a in mol.GetAtoms()]
    W[:, 1] = gas
    W[:, 2] = np.abs(gas)
    W[:, 3] = [a.GetDegree() for a in mol.GetAtoms()]
    W[:, 4] = [float(a.IsInRing()) for a in mol.GetAtoms()]
    W[:, 5] = [float(a.GetIsAromatic()) for a in mol.GetAtoms()]
    W[:, 6] = [EN.get(a.GetAtomicNum(), 2.0) for a in mol.GetAtoms()]
    W[:, 7] = 1.0
    return W


def ordered_branches(mol, k: int):
    """立体中心 k 的四个取代基分支，**按 CIP 秩降序**（高优先级在前）。

    ⚠️ **排序只依赖分子图，不依赖坐标** —— 否则不同构象会排出不同顺序，
       描述符符号会自己翻。三级键：CIP 秩 → 对称等价类 → 规范全序。

    ⚠️ **CIP 秩是硬依赖**：`Chem.ComputeAtomCIPRanks` 失败时**直接抛错**。
       旧版 `except: cipr = CanonicalRankAtoms(...)` 会静默降级成「价键排序」，
       把符号与 R/S 的一致率从 97% 拉到随机的 55%（本项目已踩过这个坑）。
    """
    nbrs = [a.GetIdx() for a in mol.GetAtomWithIdx(k).GetNeighbors()]
    if len(nbrs) != 4:
        return None, None
    cipr = list(Chem.ComputeAtomCIPRanks(mol))          # 失败就抛错，不回退
    sym = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    canon = list(Chem.CanonicalRankAtoms(mol, breakTies=True))
    key = sorted(nbrs, key=lambda j: (-cipr[j], -sym[j], -canon[j]))
    nbr_set = set(nbrs)
    grps = []
    for start in key:
        seen, stack, grp = {k, start}, [start], [start]
        while stack:
            cur = stack.pop()
            for a in mol.GetAtomWithIdx(cur).GetNeighbors():
                j = a.GetIdx()
                if j in seen or j == k or (j in nbr_set and j != start):
                    continue
                seen.add(j)
                grp.append(j)
                stack.append(j)
        grps.append(np.array(grp, dtype=int))
    return grps, key


def local_pseudoscalars(pos: np.ndarray, W: np.ndarray, k: int, brs) -> np.ndarray:
    """(n_conf, n_atom, 3) → (n_conf, 48)。全向量化，单分子 < 1 ms。"""
    rel = pos - pos[:, k:k + 1, :]
    nc, nw = pos.shape[0], W.shape[1]
    U = np.zeros((nc, 4, nw, 3), dtype=float)
    for b, g in enumerate(brs):
        U[:, b] = np.einsum("cai,aw->cwi", rel[:, g, :], W[g])
    a, b2, c = U[:, 1] - U[:, 0], U[:, 2] - U[:, 0], U[:, 3] - U[:, 0]
    vol = np.einsum("cwi,cwi->cw", a, np.cross(b2, c))

    def unit(x):
        n = np.linalg.norm(x, axis=2, keepdims=True)
        return np.divide(x, n, out=np.zeros_like(x), where=n > 1e-12)
    voln = np.einsum("cwi,cwi->cw", unit(a), np.cross(unit(b2), unit(c)))
    tps = [np.einsum("cwi,cwi->cw", U[:, i], np.cross(U[:, j], U[:, l]))
           for i, j, l in BRANCH_TRIPLES]
    return np.concatenate([vol, voln] + tps, axis=1)


def boltzmann(e_kcal: np.ndarray) -> np.ndarray:
    w = np.exp(-(e_kcal - e_kcal.min()) / RT)
    s = w.sum()
    return w / s if s > 0 else np.full_like(w, 1.0 / len(w))


# ════════════════════════════════════════════════════════════ 符号共识（审计修正版）
def sign_consensus(w: np.ndarray, D: np.ndarray, zero_col: np.ndarray):
    """返回 (sc, sc_nz)。

    **审计修正**：`np.sign(0) = 0` 会把「恒零列」算成符号不一致（旧版的 bug）。
    这里把**恒零列**（该分子全部构象都为 0）的共识**置为 0 并单独计数**，
    另给 `sc_nz` —— 只在**有定义**的列上取中位，避免把退化混进「构象不稳定」。
    """
    sc = w @ np.sign(D)                       # 恒零列 → 0
    sc[zero_col] = 0.0
    live = ~zero_col
    return sc, (float(np.median(np.abs(sc[live]))) if live.any() else float("nan"))


# ════════════════════════════════════════════════════════════ 单分子作业
def compute_one(task: tuple) -> dict:
    """(conn, smi_S, cip_S, n_stereo, n_conf, prune, seed) → 一行描述符 + 诊断。"""
    conn, smi, cip_s, n_stereo, n_conf, prune, seed = task
    try:
        m0 = Chem.MolFromSmiles(smi)
        if m0 is None:
            return dict(conn=conn, ok=False, err="SMILES 解析失败")
        ctr = Chem.FindMolChiralCenters(m0, includeUnassigned=True,
                                        useLegacyImplementation=False)
        assigned = [i for i, c in ctr if c in ("R", "S")]
        if len(assigned) != 1:
            return dict(conn=conn, ok=False, err=f"已定手性中心数 = {len(assigned)}（只支持 1）")
        m = Chem.AddHs(m0)          # AddHs 把 H 追加在末尾 ⇒ 重原子下标不变
        k = assigned[0]
        brs, key = ordered_branches(m, k)
        if brs is None:
            z = m.GetAtomWithIdx(k).GetAtomicNum()
            kind = "非四配位立体中心（含孤对电子，如 N/S）" if z in (7, 15, 16) else "四配位以外"
            return dict(conn=conn, ok=False, err=f"立体中心邻居数 ≠ 4（{kind}，Z={z}）")

        ps = AllChem.ETKDGv3()
        ps.randomSeed = seed
        ps.numThreads = 1
        ps.pruneRmsThresh = prune          # ★ 显式剪枝（CP3b 坑：默认 -1 = 不剪枝）
        cids = list(AllChem.EmbedMultipleConfs(m, numConfs=n_conf, params=ps))
        if len(cids) < 1:
            return dict(conn=conn, ok=False, err="ETKDG 嵌入失败（0 构象）")
        res = AllChem.MMFFOptimizeMoleculeConfs(m, numThreads=1, maxIters=500)
        e = np.array([x[1] for x in res], dtype=float)
        good = np.isfinite(e)
        if good.sum() < MIN_CONF:
            return dict(conn=conn, ok=False, err=f"MMFF 有效构象 {int(good.sum())} < {MIN_CONF}")
        pos = np.array([m.GetConformer(c).GetPositions()
                        for c in cids], dtype=float)[good]
        e = e[good]

        W = atom_weights(m)
        D = local_pseudoscalars(pos, W, k, brs)        # (n_conf, 48)
        zero_col = (np.abs(D) < ZERO_ABS).all(axis=0)  # 该分子下此列恒零
        w = boltzmann(e)
        bw = w @ D                                     # Boltzmann 加权均值
        sc, cons_nz = sign_consensus(w, D, zero_col)

        live = ~zero_col
        npos = int((bw[live] > 0).sum())
        nneg = int((bw[live] < 0).sum())
        row = dict(conn=conn, ok=True, err="", cip_S=cip_s, n_stereo=n_stereo,
                   n_atom=m.GetNumAtoms(), n_conf=int(good.sum()),
                   e_span=float(e.max() - e.min()),
                   n_zero_col=int(zero_col.sum()),
                   n_live=int(live.sum()),
                   cons_median=float(np.median(np.abs(sc[live]))) if live.any() else float("nan"),
                   cons_min=float(np.min(np.abs(sc[live]))) if live.any() else float("nan"),
                   perf_frac=float(np.mean(np.abs(sc[live]) > 0.999)) if live.any() else float("nan"),
                   maj_frac=float(max(npos, nneg) / max(npos + nneg, 1)),
                   n_pos=npos, n_neg=nneg)
        for nm, v in zip(NAMES, bw):
            row[nm + "_bw"] = float(v)
        for nm, v in zip(NAMES, sc):
            row[nm + "_sc"] = float(v)
        return row
    except Exception as exc:                          # 不静默：原因写进表里
        return dict(conn=conn, ok=False, err=f"{type(exc).__name__}: {exc}")


# ════════════════════════════════════════════════════════════
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-mol", type=int, default=0,
                    help="分子数上限；0 = 全部单手性中心分子")
    ap.add_argument("--n-conf", type=int, default=30, help="ETKDG 请求构象数")
    ap.add_argument("--prune", type=float, default=0.5,
                    help="ETKDG pruneRmsThresh（Å）；显式剪枝避免重复盆地")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no-vol", action="store_true",
                    help="落盘时剔除 8 个与 tp*** 线性冗余的 vol_* 列（秩 ≤ 40）")
    args = ap.parse_args()

    log: list[str] = []

    def say(s: str = "") -> None:
        print(s, flush=True)
        log.append(s)

    p = pd.read_csv(CUR / "cmrt_pairs.csv")
    allm = p.drop_duplicates("conn")
    mono = allm[allm["cip_S"].astype(str).str.len() == 1]
    dropped = allm[allm["cip_S"].astype(str).str.len() != 1]
    if args.n_mol:
        mono = mono.sample(min(args.n_mol, len(mono)), random_state=args.seed)
    tasks = [(t.conn, t.smi_S, t.cip_S, int(t.n_stereo), args.n_conf, args.prune, args.seed)
             for t in mono.itertuples()]

    say("# L1a-FF 描述符构建报告（立体中心锚定的伪标量）")
    say("")
    say("> 由 `scripts/06_chirality_descriptors.py` 生成"
        " ｜ 2026-09-17 **审计修正版**")
    say("> **48 维 × {Boltzmann 均值 `*_bw`, 符号共识 `*_sc`}；"
        "矢量来自 RDKit/MMFF（力场级），不是 `选题决策.md` 的 DFT 级 L1a**")
    say("")
    say(f"- 分子数 **{len(tasks)}**（单手性中心，去重 conn）"
        f" ｜ 因手性中心数 ≠ 1 被剔除 **{len(dropped)}** 个"
        f"（占全部 {len(allm)} 的 {len(dropped)/max(len(allm),1)*100:.2f}%）")
    say(f"- ETKDG 请求 {args.n_conf} 构象、剪枝 {args.prune} Å、种子 {args.seed}"
        f" ｜ MMFF94 优化 ｜ 并发 **{args.workers}**")
    say("")

    t0 = time.time()
    rows = []
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(compute_one, t) for t in tasks]
        for n, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if n % 500 == 0 or n == len(futs):
                el = time.time() - t0
                print(f"  [{n}/{len(futs)}] {el:.0f}s 已用 / "
                      f"预计还要 {el/n*(len(futs)-n):.0f}s", flush=True)
    el = time.time() - t0

    d = pd.DataFrame(rows)
    ok = d[d["ok"]].copy()
    say("## 1. 产出规模")
    say("")
    say(f"- 成功 **{len(ok)}** / 失败 {len(d)-len(ok)} ｜ 总耗时 **{el:.0f} s**"
        f"（{el/max(len(d),1)*args.workers:.2f} s/分子·核）")
    if len(d) > len(ok):
        say("")
        say("失败原因分布：")
        say("")
        say("| 原因 | 个数 |")
        say("|---|---|")
        for r, c in d[~d["ok"]]["err"].value_counts().head(8).items():
            say(f"| {r} | {c} |")
    say("")
    say(f"- 剪枝后构象数：中位 **{ok['n_conf'].median():.0f}**"
        f"（请求 {args.n_conf}）｜ 范围 {ok['n_conf'].min():.0f}–{ok['n_conf'].max():.0f}")
    say(f"- 原子数（含 H）：中位 {ok['n_atom'].median():.0f}"
        f" ｜ 范围 {ok['n_atom'].min():.0f}–{ok['n_atom'].max():.0f}")
    say("")

    # ── §2 系综内符号一致度（**审计修正**：分离「恒零」与「真翻号」）
    say("## 2. ★ 系综内符号一致度（审计修正版：把**恒零列**拆出来）")
    say("")
    say("> **旧版（`_proto_l1a2` / 本脚本 2026-09-17 前）的 bug**：")
    say("> 用 `np.sign(D)` 统计一致性时，**恒等于 0 的列**也进入统计"
        "（`sign(0)=0` → 共识 0 → 记为「不一致」），")
    say("> 把「数值退化」错算成「构象导致符号翻转」。现分开报。")
    say("")
    say("| 指标 | 全局三重积（`_proto_l1a`） | 本方案（旧口径） | **本方案（拆开恒零后）** |")
    say("|---|---|---|---|")
    say(f"| 一致度中位 | 0.748 | 1.000 | **{ok['cons_median'].median():.3f}** |")
    say(f"| 一致度 > 0.999 的分子比例 | 20.5% | 62%（含恒零污染） | "
        f"**{(ok['cons_median'] > 0.999).mean()*100:.1f}%** |")
    nz = int(ok["n_zero_col"].median())
    say(f"| **恒零列数（中位）** | — | 未统计 | **{nz} / 48**"
        f"（{nz/48*100:.0f}%） |")
    say(f"| 有效列数（中位） | — | — | **{ok['n_live'].median():.0f} / 48** |")
    say("")
    zero_share = (ok[[c + "_bw" for c in NAMES]].abs() < ZERO_ABS).mean().sort_values(ascending=False)
    dead = zero_share[zero_share > 0.5]
    if len(dead):
        say(f"恒零（>50% 分子上为 0）的列 **{len(dead)}** 个："
            f"`{'`, `'.join(dead.index[:10])}`"
            f" ｜ 这些列**必须在建模前删除**")
    say("")
    say("⇒ 结论：**「局部锚定让符号稳定」在分离恒零后依然成立，"
        "但旧版报的 62% 是被退化列污染的**；")
    say("   机制描述应从「符号是构象/分子的属性」改为：")
    say("   **恒零列（权重族在该分子上无支撑）= 退化；其余列的符号基本由 R/S 锁死（见 §3）。**")
    say("")

    # ── §3 符号与 CIP 的关系（**新诊断**）
    say("## 3. ★ 符号与 CIP 标签的关系（上一场报 54.8% 的那条，是错的）")
    say("")
    bwc = [n + "_bw" for n in NAMES]
    ref = ok["vol_mass_bw"].values
    okcip = ok["cip_S"].isin(["R", "S"])
    sub = ok[okcip]
    agree = float(np.mean((sub["vol_mass_bw"] > 0) == (sub["cip_S"] == "S")))
    agree = max(agree, 1 - agree)
    say(f"- `vol_mass` 符号 vs CIP 标签一致率：**{agree*100:.1f}%**"
        f"（n = {len(sub)}）")
    say("  - **旧版报的 54.8% 出自 `_proto_l1a2.py`，那个版本分支按「原子下标」排、**"
        "根本没做 CIP 排序**，符号是任意的；")
    say("  - 且原文推论「一致率低 ⇒ 不是复述 R/S、携带构型之外的信息」**方向反了** ——"
        "一致率低意味着符号**没有被控制**，不是信息更多。")
    say(f"- 每分子「多数符号占比」`maj_frac` 中位：**{ok['maj_frac'].median():.3f}**"
        f"（1.0 = 48 列符号全一致 = 秩 1 塌缩；实测 < 1 ⇒ **不是秩 1**）")
    # ── ★ 不一致的那部分是什么？（2026-09-17 新增诊断）
    maj_sign = np.sign(ok["vol_mass_bw"].median())
    ok = ok.assign(_outlier=(np.sign(ok["vol_mass_bw"]) != maj_sign))
    off, on = ok[ok["_outlier"]], ok[~ok["_outlier"]]
    say(f"  约定：`vol_mass` 多数符号为 **{int(maj_sign):+d}** ⇒ "
        f"「与约定不符」**{len(off)}** 个（{len(off)/len(ok)*100:.1f}%）")
    say("")
    say("### 3a. ★ 那 7~8% 「不符合约定」的分子是什么？（新增）")
    say("")
    say("| 指标 | 符合约定 | **不符合** | 读法 |")
    say("|---|---|---|---|")
    say(f"| 系综内一致度中位 | {on['cons_median'].median():.3f} | "
        f"**{off['cons_median'].median():.3f}** | 不符合者的符号**在系综上真的会翻** |")
    say(f"| 原子数中位 | {on['n_atom'].median():.0f} | **{off['n_atom'].median():.0f}** | "
        f"不符合者**分子更大** |")
    say(f"| 剪枝后构象数中位 | {on['n_conf'].median():.0f} | **{off['n_conf'].median():.0f}** | "
        f"更高 = **更柔** |")
    say("")
    say("⇒ 机制：支矢量 `u_b = Σ_{i∈分支b} w_i(r_i − r_k)` 在**大而柔的分支**上会大幅摆动，")
    say("   足以把 `det[u₁−u₀, u₂−u₀, u₃−u₀]` 的符号扫过去。")
    say("   **这不否定「局部锚定」的动机，但说明它不是「拓扑不变量»**：")
    say("")
    say("   | 分子占比 | 符号的性质 | 结论 |")
    say("   |---|---|---|")
    say(f"   | **{100-len(off)/len(ok)*100:.1f}%** | 锁死在 R/S（= CIP 约定） | "
        f"符号**是常数** ⇒ `signed ≡ −(幅度)`，**不携带手性信息** |")
    say(f"   | **{len(off)/len(ok)*100:.1f}%** | 随构象摆动（一致度仅 "
        f"{off['cons_median'].median():.2f}） | 符号**是噪声**，也不是手性信息 |")
    say("")
    say("   ⇒ **两种情形下，这 48 列的「手性」都不是可用信息**；可用的只有**幅度**（偶量）。")
    say("   下游必须用 `06b` 的 `L0+m`（只用幅度）臂把这条钉死。")
    say("")
    say(f"- 存活列之间「与 `vol_mass` 同号率」中位："
        f"**{pd.Series([float(((np.sign(ok[c])==np.sign(ref))|(ok[c]==0)).mean()) for c in bwc[2:]]).median():.3f}**"
        f" ⇒ 不同权重族携带**方向不同**的手性信息（这是正面发现）")
    say("")
    per = []
    for c in bwc:
        per.append(dict(col=c[:-3],
                        zero_frac=float((ok[c].abs() < ZERO_ABS).mean()),
                        sign_eq_volmass=float(((np.sign(ok[c]) == np.sign(ref))
                                               | (ok[c] == 0)).mean()),
                        sc_med=float(ok[c[:-3] + "_sc"].abs().median()),
                        abs_med=float(ok[c].abs().median())))
    pd_ = pd.DataFrame(per)
    pd_.to_csv(CUR / "l1a_per_column.csv", index=False, encoding="utf-8")
    say(f"- 逐列体检表 → `data/curated/l1a_per_column.csv`")
    say("")

    # ── §3b 线性冗余
    say("### 3b. 线性冗余（**新发现**）")
    say("")
    say("同一权重下 `vol_w = tp123_w − tp023_w + tp013_w − tp012_w`"
        "（有符号体积是四个三阶子式的交替和）")
    say("⇒ 8 个 `vol_*` 列与 32 个 `tp***_*` **严格线性相关**；"
        "48 列的**真实秩 ≤ 40**。")
    say("另：`tp012_*` 与 `vol_*` 符号相反（排列奇偶性）——"
        "这是约定问题不是 bug，但**读表时不要误当成两种独立手性**。")
    say("")

    # ── §4 代数门禁（引用，不再当成「改进证据」）
    say("## 4. 代数门禁（`_proto_l1a3.py`，性质检验而非性能证据）")
    say("")
    say("| 门禁 | 结果 | 怎么读 |")
    say("|---|---|---|")
    say("| 镜面反射严格反号（48 × 全构象 × 25 分子） | ✅ 最大相对偏差 **0.0e+00** | 描述符**数学上确实是伪标量** |")
    say("| 减半策略：独立 R 系综 vs −(S 的描述符) | ✅ 反号比例中位 **1.000**"
        " | ⚠️ **数学恒等式**，只证明原子序对齐；**不是**相对 CP3b 0.937 的改进 |")
    say("| 幅度一致性（\\|D_R\\| vs \\|D_S\\| 相对偏差） | 中位 **0.070** | 真正的实测信息：以符号为主、幅度取 S 侧 |")
    say("| 分支排序在 R/S 两侧一致 | ✅ 10/10 | 支持减半策略可用 |")
    say("")
    say("> ⚠️ **审计结论**：上一场把「反号比例 1.000」与 CP3b 的 0.937 并列，读起来像「我们修好了」。")
    say("> 两者**不是同一个量**：0.937 是**跨系综配对后**的比较（含盆地不匹配），")
    say("> 1.000 是**定义使然**。论文里不能这样并列。")
    say("")

    # ── §5 描述符体检
    say("## 5. 描述符体检")
    say("")
    say("| 家族 | 维度 | 非零比例中位 | 符号共识\\|·\\|中位 |")
    say("|---|---|---|---|")
    for fam, sel in (("`vol_*` 分支有符号体积（冗余）", VOL_COLS),
                     ("`voln_*` 归一化（纯角度手性）",
                      [n for n in NAMES if n.startswith("voln_")]),
                     ("`tp***_*` 分支三重积", [n for n in NAMES if n.startswith("tp")])):
        v = float((ok[[s + "_bw" for s in sel]].abs() > ZERO_ABS).mean().median())
        sgn = float(ok[[s + "_sc" for s in sel]].abs().median().median())
        say(f"| {fam} | {len(sel)} | {v*100:.0f}% | {sgn:.3f} |")
    say("")

    # ── §6 产出
    keep_cols = [c for c in ok.columns
                 if c not in ("ok", "err")
                 and not (args.no_vol and c[:-3] in VOL_COLS)]
    out = ok[keep_cols]
    tag = ""
    if args.no_vol:
        tag = "_novol"
        say("（`--no-vol`：已剔除 8 个线性冗余的 `vol_*` 列）")
        say("")
    out.to_csv(CUR / f"l1a_descriptors{tag}.csv", index=False, encoding="utf-8")
    if len(d) > len(ok):
        d[~d["ok"]][["conn", "err"]].to_csv(
            CUR / "l1a_failures.csv", index=False, encoding="utf-8")

    say("## 6. 产出")
    say("")
    say(f"- `data/curated/l1a_descriptors{tag}.csv` —— **{len(out)} 行 × {out.shape[1]} 列**")
    if len(d) > len(ok):
        say(f"- `data/curated/l1a_failures.csv` —— {len(d)-len(ok)} 个失败分子及原因")
    say(f"- `data/curated/l1a_per_column.csv` —— 逐列退化率 / 符号一致率 / 共识")
    say("")
    say("**下一步**：`scripts/06b_l1a_signal_test.py` —— 四路消融")
    say("`L0 / L0+R-S / L0+m(偶) / L0+L1a`，在 S2b/S3/S4 上跑 + 阴性对照。")
    say("**描述符代数性质过关 ≠ 下游有信号**，必须实测；")
    say("且必须包含 **「只用 R/S 指示符」的对照臂** —— 因为 §3 已证明符号几乎完全由 R/S 决定。")
    say("")

    (ROOT / "notes" / "L1a_描述符构建报告.md").write_text(
        "\n".join(log) + "\n", encoding="utf-8")
    print(f"\n[落盘] data/curated/l1a_descriptors{tag}.csv ({len(out)} 行 × {out.shape[1]} 列)")
    print("[落盘] notes/L1a_描述符构建报告.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
