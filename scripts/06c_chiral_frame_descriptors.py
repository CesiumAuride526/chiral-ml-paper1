# -*- coding: utf-8 -*-
"""
06c_chiral_frame_descriptors.py — **L1-CF：局部手性坐标系形状描述符**
================================================================================
审计结论（`notes/审计_上一场_L1a方案_2026-09-17.md` §5）：
`06` 的 L1a-FF 只有「立体中心周围 4 个支矢量 × 8 种权重 = 48 列」，
**从不触及分子其余部分**，因此它拿到的信息 ≈ 偶量 3D 幅度，且 12/48 列结构性为 0。

本脚本换一个**原理上不同**的做法：**先把局部手性参考系建出来，再描述整个分子。**

## 一、为什么「整个分子在局部坐标系下的分布」才是对的方向

CSP 的手性识别发生在**分子与固定相接触的那一层**（约 3–8 Å），
所以模型需要知道的是「**分子的其余部分相对于立体中心的手性排布**」，
而不是「四个支矢量各自的朝向」。后者只是**四个数**，前者是**一个分布**。

## 二、参考系怎么建（只用拓扑 + 第一配位层几何）

取立体中心 `k` 的四个直接邻居，**按 CIP 秩降序**排为 `n0>n1>n2>n3`：

```
v_j = r_{n_j} − r_k
e1 = unit(v0)
e2 = unit(v1 − (v1·e1) e1)      # 正交化
e3 = e1 × e2                     # ← 伪矢量：镜面反射下 e3 → −e3
s  = sign(det[v0, v1, v2])       # ← 手性符号（与 R/S 同构，全表近乎常数）
```

对**每个重原子 i**（除 k）：

```
z_i = d_i · e3      (d_i = r_i − r_k)   ← 伪标量（镜面反号）
ζ_i = s · z_i                          ← **无手性**的「相对手性平面的高度」
ρ_i = √(|d_i|² − z_i²)                  ← 面内半径
r_i = |d_i|                             ← 径向距离
```

> 📌 **ζ 是偶量（镜面不变）——这是必然的，不是缺陷**：
> 任何伪标量都可写成 `s × (无手性函数)`，而本数据集 `cip_S` **恒为 `S`** ⇒ `s` 是常数。
> ⇒ **伪标量框架提供不了额外的手性信息**；能提供信息的是 `ζ`（有符号高度）这个
> **「在手性参考系里量出来的形状」**。这正是 L1a-FF 缺的东西。

## 三、描述符（**89 列**，全部以 `cf_` 为前缀）

> ⚠️ **第八场审计更正**：本行原写「≈87 列」，与落盘 `data/curated/l1cf_descriptors.csv`
> 实测的 **89** 个 `cf_*` 列不符。差额来自下表 C 块少列了两个：
> **`cf_rg_rel`**（C 块实际 **11** 列，不是 10）与全局符号列 **`cf_s`**。
> 即 56（A）+ 21（B）+ 11（C）+ 1（`cf_s`）= **89**。

| 块 | 内容 | 维度 |
|---|---|---|
| **A 原子分组矩** | 7 个化学分组（`all`/`C`/`het`/`arom`/`ring`/`hb`/`halo`）× 8 个泛函 | 56 |
| **B 有符号径向剖面** | 7 个径向壳（2.0/2.8/3.6/4.6/5.8/7.2/9.0 Å 起）× {质量, 质量·ζ, 绝对电荷·ζ} | 21 |
| **C 全局标量** | 原子数 / Rg / Rmax / 各分组半径 / 可旋转键 | 10 |

**A 的 8 个泛函**：`n`（计数）、`m1`（Σζ）、`m1n`（Σζ/(n·Rg)）、`m1abs`（Σ|ζ|/n）、
`m2`（Σζ²）、`m3`（Σζ³）、`meanrho`（Σρ/n）、`meanr`（Σr/n）。

**B 是关键**：`Σ_{r_i∈壳} m_i·ζ_i` 就是「**在某径向距离上，分子的质量偏向手性平面的哪一侧**」，
即一条**有符号的径向分布函数**。CSP 的接触层正好落在这几个壳里。

## 四、用法
```bash
PYTHONIOENCODING=utf-8 python scripts/06c_chiral_frame_descriptors.py --n-mol 0 --workers 28
```
产出：`data/curated/l1cf_descriptors.csv` + `notes/L1CF_描述符构建报告.md`
"""
from __future__ import annotations

import argparse
import os
import sys
import time
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
RT = 1.98720425864083e-3 * 298.15
ZERO_ABS = 1e-12
MIN_CONF = 1

HALO = {9, 17, 35, 53}
HBPOLAR = {7, 8, 16}                      # N/O/S：氢键 + 强极性
RING_CUTS = (2.0, 2.8, 3.6, 4.6, 5.8, 7.2, 9.0)      # 7 壳（最后一个为 >9 Å）

# 7 个化学分组（原子子集，不是权重族）
BIN_NAMES = ("all", "C", "het", "arom", "ring", "hb", "halo")
# 8 个泛函
FUN_NAMES = ("n", "m1", "m1n", "m1abs", "m2", "m3", "meanrho", "meanr")


def descriptor_names() -> list[str]:
    ns = []
    for b in BIN_NAMES:
        for f in FUN_NAMES:
            ns.append(f"cf_{b}_{f}")
    for j in range(len(RING_CUTS)):
        ns += [f"cf_sh{j}_mass", f"cf_sh{j}_mz", f"cf_sh{j}_qz"]
    ns += ["cf_n_heavy", "cf_n_het", "cf_n_arom", "cf_n_ring", "cf_n_rotb",
           "cf_rg", "cf_rmax", "cf_rmean", "cf_rg_het", "cf_rg_arom",
           "cf_rg_rel", "cf_s"]
    return ns


NAMES = descriptor_names()


def bin_masks(mol) -> np.ndarray:
    """(n_atom, 7) 布尔分组矩阵。**只依赖拓扑**，每分子算一次。"""
    n = mol.GetNumAtoms()
    M = np.zeros((n, len(BIN_NAMES)), dtype=bool)
    for a in mol.GetAtoms():
        i, z = a.GetIdx(), a.GetAtomicNum()
        heavy = z > 1
        M[i, 0] = heavy
        M[i, 1] = z == 6
        M[i, 2] = heavy and z not in (1, 6)
        M[i, 3] = heavy and a.GetIsAromatic()
        M[i, 4] = heavy and a.IsInRing()
        M[i, 5] = z in HBPOLAR
        M[i, 6] = z in HALO
    return M


def frame(pos, k, key, brs):
    """由**第一配位层**建局部手性参考系。

    返回 (e1, e2, e3, s)：(n_conf,3) 三个单位轴 + 手性符号 (n_conf,)。
    ⚠️ e3 是**伪矢量**（叉积），镜面反射下 `e3 → −e3`；这正好与 `s → −s` 抵消，
       使 ζ = s·(d·e3) 成为**偶量**。这不是 bug，见文件头说明。
    """
    rk = pos[:, k, :]                                        # (c,3) —— 注意别用 pos[:, k:k+1, :]
    v0 = pos[:, key[0], :] - rk                              # 否则 (c,3)−(c,1,3) 会广播成 (c,c,3)
    v1 = pos[:, key[1], :] - rk
    v2 = pos[:, key[2], :] - rk
    n0 = np.linalg.norm(v0, axis=1, keepdims=True)
    e1 = v0 / np.maximum(n0, 1e-12)
    proj = np.einsum("ci,ci->c", v1, e1)[:, None] * e1
    w = v1 - proj
    e2 = w / np.maximum(np.linalg.norm(w, axis=1, keepdims=True), 1e-12)
    e3 = np.cross(e1, e2)
    s = np.sign(np.einsum("ci,ci->c", v2, e3))
    s[s == 0] = 1.0
    return e1, e2, e3, s


def one_conf_descriptors(pos, k, key, M, mass, absq, n_rotb):
    """(n_conf, n_atom, 3) → (n_conf, n_desc)。全向量化。"""
    nc, na, _ = pos.shape
    e1, e2, e3, s = frame(pos, k, key, None)                 # (c,3) (c,)
    d = pos - pos[:, k:k + 1, :]                             # (c,a,3)
    z = np.einsum("cai,ci->ca", d, e3)                       # 伪标量高度
    zeta = z * s[:, None]                                    # ← 偶量
    r2 = (d ** 2).sum(axis=2)
    r = np.sqrt(r2)
    rho = np.sqrt(np.maximum(r2 - z ** 2, 0.0))
    rg = np.sqrt(np.mean(r2, axis=1))                        # (c,)
    rmax = r.max(axis=1)
    scale = np.maximum(rg, 1e-9)[:, None]

    cols = []
    for bi in range(len(BIN_NAMES)):
        mbool = M[:, bi]                                     # ⚠️ 必须用**布尔**做索引
        cnt = int(mbool.sum())
        if cnt < 1:
            cols += [np.zeros(nc)] * len(FUN_NAMES)
            continue
        zt = zeta[:, mbool]                                  # (c,kk)
        rh = rho[:, mbool]
        rr = r[:, mbool]
        cols += [
            np.full(nc, float(cnt)),
            zt.sum(axis=1),
            zt.sum(axis=1) / (cnt * scale[:, 0]),
            np.abs(zt).mean(axis=1),
            (zt ** 2).sum(axis=1),
            (zt ** 3).sum(axis=1),
            rh.mean(axis=1),
            rr.mean(axis=1),
        ]
    for j, cut in enumerate(RING_CUTS):
        # ⚠️ 最后一壳的上界是逐构象的 ⇒ 必须 [:, None] 让它变成 (c,1)，否则 (c,a) 与 (c,) 广播失败
        hi = (rmax * 10)[:, None] if j == len(RING_CUTS) - 1 else float(cut)
        lo = 0.0 if j == 0 else float(RING_CUTS[j - 1])
        sel = (r >= lo) & (r < hi)
        seld = sel.astype(float)
        mass_c = np.broadcast_to(mass[None, :], r.shape).astype(float, copy=True)
        absq_c = np.broadcast_to(absq[None, :], r.shape).astype(float, copy=True)
        cols += [
            np.einsum("ca,ca->c", seld, mass_c),
            np.einsum("ca,ca->c", seld * zeta, mass_c),
            np.einsum("ca,ca->c", seld * zeta, absq_c),
        ]
    heavy = M[:, 0]
    def rg_of(mask):
        sub = r2[:, mask]
        return np.sqrt(sub.mean(axis=1)) if sub.shape[1] else np.zeros(nc)
    cols += [
        np.full(nc, float(heavy.sum())),
        np.full(nc, float(M[:, 2].sum())),
        np.full(nc, float(M[:, 3].sum())),
        np.full(nc, float(M[:, 4].sum())),
        np.full(nc, float(n_rotb)),
        rg, rmax, r.mean(axis=1),
        rg_of(M[:, 2]), rg_of(M[:, 3]),
        rg / np.maximum(rmax, 1e-9),
        s,
    ]
    return np.stack(cols, axis=1)


# ════════════════════════════════════════════════════════════ 单分子作业
def compute_one(task: tuple) -> dict:
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
        m = Chem.AddHs(m0)
        k = assigned[0]
        nbrs = [a.GetIdx() for a in m.GetAtomWithIdx(k).GetNeighbors()]
        if len(nbrs) != 4:
            z = m.GetAtomWithIdx(k).GetAtomicNum()
            return dict(conn=conn, ok=False, err=f"非四配位立体中心（Z={z}）")
        cipr = list(Chem.ComputeAtomCIPRanks(m))            # 失败即抛错，不回退
        sym = list(Chem.CanonicalRankAtoms(mol=m, breakTies=False))
        can = list(Chem.CanonicalRankAtoms(mol=m, breakTies=True))
        key = sorted(nbrs, key=lambda j: (-cipr[j], -sym[j], -can[j]))

        ps = AllChem.ETKDGv3()
        ps.randomSeed = seed
        ps.numThreads = 1
        ps.pruneRmsThresh = prune
        cids = list(AllChem.EmbedMultipleConfs(m, numConfs=n_conf, params=ps))
        if len(cids) < 1:
            return dict(conn=conn, ok=False, err="ETKDG 嵌入失败（0 构象）")
        res = AllChem.MMFFOptimizeMoleculeConfs(m, numThreads=1, maxIters=500)
        e = np.array([x[1] for x in res], dtype=float)
        good = np.isfinite(e)
        if good.sum() < MIN_CONF:
            return dict(conn=conn, ok=False, err=f"MMFF 有效构象 {int(good.sum())} < {MIN_CONF}")
        pos = np.array([m.GetConformer(c).GetPositions() for c in cids], dtype=float)[good]
        e = e[good]

        AllChem.ComputeGasteigerCharges(m, nIter=20, throwOnParamFailure=False)
        absq = np.nan_to_num(np.array(
            [abs(a.GetDoubleProp("_GasteigerCharge")) if a.HasProp("_GasteigerCharge") else 0.0
             for a in m.GetAtoms()], dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        mass = np.array([a.GetMass() for a in m.GetAtoms()], dtype=float)
        n_rotb = rdMolDescriptors_nrotb(m)

        D = one_conf_descriptors(pos, k, key, bin_masks(m), mass, absq, n_rotb)
        w = np.exp(-(e - e.min()) / RT)
        w = w / w.sum()
        mean = w @ D

        row = dict(conn=conn, ok=True, err="", cip_S=cip_s, n_stereo=n_stereo,
                   n_atom=m.GetNumAtoms(), n_conf=int(good.sum()),
                   e_span=float(e.max() - e.min()),
                   s_med=float(np.median(np.sign(D[:, -1]))))
        for nm, v in zip(NAMES, mean):
            row[nm] = float(v)
        return row
    except Exception as exc:
        return dict(conn=conn, ok=False, err=f"{type(exc).__name__}: {exc}")


def rdMolDescriptors_nrotb(mol) -> int:
    return int(Chem.rdMolDescriptors.CalcNumRotatableBonds(mol))


# ════════════════════════════════════════════════════════════
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-mol", type=int, default=0)
    ap.add_argument("--n-conf", type=int, default=30)
    ap.add_argument("--prune", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    log: list[str] = []

    def say(s: str = "") -> None:
        print(s, flush=True)
        log.append(s)

    p = pd.read_csv(CUR / "cmrt_pairs.csv")
    allm = p.drop_duplicates("conn")
    mono = allm[allm["cip_S"].astype(str).str.len() == 1]
    if args.n_mol:
        mono = mono.sample(min(args.n_mol, len(mono)), random_state=args.seed)
    tasks = [(t.conn, t.smi_S, t.cip_S, int(t.n_stereo), args.n_conf, args.prune, args.seed)
             for t in mono.itertuples()]

    say("# L1-CF 描述符构建报告（局部手性坐标系形状描述符）")
    say("")
    say("> 由 `scripts/06c_chiral_frame_descriptors.py` 生成 ｜ 2026-09-17")
    say("> 设计动机与「为什么 ζ 是偶量、为什么这仍然是对的方向」见脚本 docstring")
    say("")
    say(f"- 分子数 **{len(tasks)}**（单手性中心）；维度 **{len(NAMES)}**")
    say(f"- ETKDG {args.n_conf} 构象 / 剪枝 {args.prune} Å / MMFF94 ｜ 并发 {args.workers}"
        f" ｜ 种子 {args.seed}")
    say("")

    t0 = time.time()
    rows = []
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(compute_one, t) for t in tasks]
        for n, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if n % 1000 == 0 or n == len(futs):
                el = time.time() - t0
                print(f"  [{n}/{len(futs)}] {el:.0f}s / 预计还要 {el/n*(len(futs)-n):.0f}s",
                      flush=True)
    el = time.time() - t0

    d = pd.DataFrame(rows)
    ok = d[d["ok"]].copy()
    say("## 1. 产出规模")
    say("")
    say(f"- 成功 **{len(ok)}** / 失败 {len(d)-len(ok)} ｜ 总耗时 **{el:.0f} s**")
    if len(d) > len(ok):
        say("")
        say("| 失败原因 | 个数 |")
        say("|---|---|")
        for r, c in d[~d["ok"]]["err"].value_counts().head(6).items():
            say(f"| {r} | {c} |")
    if not len(ok):
        (ROOT / "notes" / "L1CF_描述符构建报告.md").write_text(
            "\n".join(log) + "\n", encoding="utf-8")
        print("‼️ 无成功样本，已中止")
        return 1
    say(f"- 剪枝后构象数：中位 {ok['n_conf'].median():.0f}"
        f"（请求 {args.n_conf}）｜ 范围 {ok['n_conf'].min():.0f}–{ok['n_conf'].max():.0f}")
    say("")

    # ── 体检：与 L1a-FF 对比的「死列」问题是否解决
    bwc = [n for n in NAMES if n in ok.columns]
    zf = (ok[bwc].abs() < ZERO_ABS).mean().sort_values(ascending=False)
    dead = zf[zf > 0.5]
    say("## 2. 列体检（对比 L1a-FF 的 12/48 恒零列）")
    say("")
    say(f"- **恒零（>50% 分子上为 0）的列：{len(dead)} / {len(bwc)}**"
        + (f" → `{'`, `'.join(dead.index[:10])}`" if len(dead) else " ✅ 无"))
    say(f"- 零值率最高的一列：`{zf.index[0]}`（{zf.iloc[0]*100:.1f}%）")
    say("")
    # ζ 的稳定度：s_med 是否为常数（应然）
    say("## 3. 手性符号 `s`（预期全表近乎常数）")
    say("")
    vc = ok["s_med"].value_counts()
    say(f"- 取值分布：`{vc.to_dict()}`")
    say("  - **这条是「伪标量框架在本任务上多余」的直接证据**：`s` 几乎恒为 "
        f"{int(vc.index[0])} ⇒ 任何伪标量都退化成「常数 × 偶量」。")
    say("")

    # ── 与 L1a-FF 的信息量对比（用「列的标准差为零」简单判据 + 相关性）
    say("## 4. 与 L1a-FF 的重叠度（看是不是真的换了信息）")
    say("")
    l1 = CUR / "l1a_descriptors.csv"
    if l1.exists():
        L = pd.read_csv(l1)
        mg = ok[["conn"] + bwc].merge(L, on="conn", how="inner", suffixes=("_cf", "_ff"))
        cf = [c for c in mg.columns if c.startswith("cf_")]
        ff = [c for c in mg.columns if c.endswith("_bw") and not c.startswith("cf_")]
        if cf and ff:
            A = mg[cf].values
            B = mg[ff].values
            A = (A - A.mean(0)) / np.where(A.std(0) > 0, A.std(0), 1)
            B = (B - B.mean(0)) / np.where(B.std(0) > 0, B.std(0), 1)
            K = 8
            Am = A[:, np.argsort(-A.var(0))[:K]]
            Bm = B[:, np.argsort(-B.var(0))[:K]]
            C = np.abs(np.corrcoef(np.hstack([Am, Bm]).T)[:K, K:])
            sell = C.max(1)
            say(f"- 共同分子 **{len(mg)}** ｜ 取双方方差最大的各 {K} 列做 |Pearson| 上限匹配")
            say(f"- CF 每列在 FF 中的**最大绝对相关**：中位 **{np.median(sell):.3f}**，"
                f"最高 {sell.max():.3f}")
            say(f"- ⇒ |相关| < 0.7 的 CF 列占 **{(sell<0.7).mean()*100:.0f}%**"
                f"（越高说明越换了信息，而不是换皮）")
    say("")
    say("## 5. 产出")
    say("")
    out = ok.drop(columns=["ok", "err"])
    out.to_csv(CUR / "l1cf_descriptors.csv", index=False, encoding="utf-8")
    if len(d) > len(ok):
        d[~d["ok"]][["conn", "err"]].to_csv(
            CUR / "l1cf_failures.csv", index=False, encoding="utf-8")
    say(f"- `data/curated/l1cf_descriptors.csv` —— **{len(out)} 行 × {out.shape[1]} 列**")
    say("")
    say("**下一步**：`scripts/06d_l1cf_signal_test.py` —— 把 L1-CF 塞进 `06b` 的同一套五臂")
    say("消融（含 `L0+m` 偶量对照与逐折 SD），**判据不变**：CF 必须明显优于 L0 与 L0+m。")

    (ROOT / "notes" / "L1CF_描述符构建报告.md").write_text("\n".join(log) + "\n",
                                                        encoding="utf-8")
    print(f"\n[落盘] data/curated/l1cf_descriptors.csv ({len(out)} 行 × {out.shape[1]} 列)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
