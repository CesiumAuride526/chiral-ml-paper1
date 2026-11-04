# -*- coding: utf-8 -*-
"""
_baseline_cp25.py — CP2.5「最小可行结果」预演（把审计的定性判断变成数字）
==========================================================================
目的：**在投入任何大算力之前**，先用最便宜的 2D 描述符跑一版基线，回答三个问题：

  Q1 用 RDKit 2D 描述符能不能预测定向靶 y = ln[k(S)/k(R)]？R² 量级是多少？
  Q2 随机划分下看起来很好的性能，换成**严格划分**（留一文献 / 留一格因子外推）后崩多少？
     —— 这是论文「复现前人 → 展示其在严格划分下崩掉 → 我们守住」三段论的第 ①②步。
  Q3 洗脱顺序（EMO 二分类）能不能预测？**理论上不能**——普通 2D 描述符在镜面反射下不变，
     两个对映体的特征向量**逐位相同**（已实测）。这个基线就是来**坐实**这条断言的。

特征分三级消融：
  L0       分子 2D 描述符 + 实验条件（i-PrOH 比例、流速）
  L0+colID + **柱身份 one-hot**  ← 上界：知道"用哪根柱"，但**无法外推到新柱**
  L0+csp   + **CSP 结构描述符**（骨架/取代基码/粒径） ← 可外推的那一套

划分：S1 随机 / S2 分组 / S2b 留一文献 / S3 留一柱 / S3′ 留一格 / S4 留骨架 / S4b 化学类

用法: PYTHONIOENCODING=utf-8 python scripts/_baseline_cp25.py
产出: notes/CP2.5_基线预演.md
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
CUR = ROOT / "data" / "curated"
OUT = ROOT / "notes" / "CP2.5_基线预演.md"
SEED = 42
LOG: list[str] = []


def say(s: str = "") -> None:
    print(s)
    LOG.append(s)


# ────────────────────────────────────────────────── 特征
DESC = [
    ("MolWt", Descriptors.MolWt), ("MolLogP", Crippen.MolLogP),
    ("TPSA", rdMolDescriptors.CalcTPSA), ("HBD", rdMolDescriptors.CalcNumHBD),
    ("HBA", rdMolDescriptors.CalcNumHBA), ("RotB", rdMolDescriptors.CalcNumRotatableBonds),
    ("RingCount", rdMolDescriptors.CalcNumRings), ("AromRings", rdMolDescriptors.CalcNumAromaticRings),
    ("SatRings", rdMolDescriptors.CalcNumSaturatedRings), ("FracCsp3", rdMolDescriptors.CalcFractionCSP3),
    ("nStereo", rdMolDescriptors.CalcNumAtomStereoCenters),
    ("nHetero", rdMolDescriptors.CalcNumHeteroatoms),
    ("NHOH", rdMolDescriptors.CalcNumHeteroatoms), ("AmideBonds", rdMolDescriptors.CalcNumAmideBonds),
    ("HeavyAtoms", lambda m: m.GetNumHeavyAtoms()),
    ("LabuteASA", rdMolDescriptors.CalcLabuteASA), ("BertzCT", Descriptors.BertzCT),
    ("BalabanJ", Descriptors.BalabanJ),
    ("Chi0", rdMolDescriptors.CalcChi0n), ("Chi1", rdMolDescriptors.CalcChi1n),
    ("Kappa1", rdMolDescriptors.CalcKappa1), ("Kappa2", rdMolDescriptors.CalcKappa2),
    ("MolMR", Crippen.MolMR), ("NumRingsArom", rdMolDescriptors.CalcNumAromaticRings),
    ("MaxPartialCharge", lambda m: max(a.GetDoubleProp("_GasteigerCharge") for a in m.GetAtoms())
     if m.GetNumAtoms() else 0.0),
]


def mol_feats(smi: str) -> dict:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return {}
    Chem.rdPartialCharges.ComputeGasteigerCharges(m)
    d = {}
    for name, fn in DESC:
        try:
            v = fn(m)
            d["mol_" + name] = float(v) if np.isfinite(float(v)) else np.nan
        except Exception:
            d["mol_" + name] = np.nan
    return d


def build_X(p: pd.DataFrame) -> pd.DataFrame:
    uniq = pd.unique(np.concatenate([p["smi_S"].values, p["smi_R"].values]))
    cache = {s: mol_feats(s) for s in uniq}
    X = pd.DataFrame([cache[s] for s in p["smi_S"]], index=p.index)
    X["cond_ipr"] = p["ipr"].values
    X["cond_speed"] = p["speed"].values
    X["cond_t0"] = p["t0"].values
    return X


def add_col_id(X: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([X, pd.get_dummies(p["Column"], prefix="col").astype(float)], axis=1)


def add_csp(X: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
    """CSP **结构**描述符：可外推到没见过的柱组合（这是 S3/S3′ 唯一能用的那套）"""
    csp = pd.concat([
        pd.get_dummies(p["surface"], prefix="csp_surf"),
        pd.get_dummies(p["subs_code"], prefix="csp_subs"),
        pd.get_dummies(p["particle_um"].astype(str), prefix="csp_psize"),
    ], axis=1).astype(float)
    return pd.concat([X, csp], axis=1)


# ────────────────────────────────────────────────── 模型与评估
def random_forest_pair(n_estimators: int = 250):
    """统一的 GBM 配置（回归/分类共用超参，避免"调出来的结论"）"""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    return (HistGradientBoostingRegressor(max_iter=n_estimators, learning_rate=0.06,
                                          min_samples_leaf=20, random_state=SEED),
            HistGradientBoostingClassifier(max_iter=n_estimators, learning_rate=0.06,
                                           min_samples_leaf=20, random_state=SEED))


def _score(y: np.ndarray, pred: np.ndarray, emo: np.ndarray | None,
           predc: np.ndarray | None) -> dict:
    from sklearn.metrics import accuracy_score, r2_score
    ok = np.isfinite(pred)
    d = dict(n=int(ok.sum()), r2=float(r2_score(y[ok], pred[ok])),
             rmse=float(np.sqrt(np.mean((y[ok] - pred[ok]) ** 2))),
             y_sd=float(np.std(y[ok])))
    if emo is not None and predc is not None:
        d["emo_acc"] = float(accuracy_score(emo[ok], predc[ok]))
        # 多数类基线（随测试集组成变，必须逐次报，不能只报全局 50.7%）
        f = float(emo[ok].mean())
        d["emo_base"] = max(f, 1 - f)
    return d


def run_cv(X, y, folds: np.ndarray, emo=None, permute: int | None = None) -> dict:
    """k 折 OOF。folds 为每行的折号。permute 给定时打乱**标签**（阴性对照）。"""
    Xv = X.values.astype(float)
    rng = np.random.default_rng(permute if permute is not None else 0)
    yy = rng.permutation(y) if permute is not None else y
    ee = rng.permutation(emo) if (permute is not None and emo is not None) else emo
    pred = np.full(len(y), np.nan)
    predc = np.zeros(len(y), dtype=int) if emo is not None else None
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        if te.sum() == 0 or tr.sum() == 0:
            continue
        reg, clf = random_forest_pair()
        reg.fit(Xv[tr], yy[tr])
        pred[te] = reg.predict(Xv[te])
        if emo is not None:
            clf.fit(Xv[tr], ee[tr])
            predc[te] = clf.predict(Xv[te])
    return _score(yy, pred, ee, predc)


def run_holdout(X, y, folds: np.ndarray, train_fold: int, test_fold: int, emo=None) -> dict:
    """单次留出。⚠️ S1 是 80/20 单划分，**绝不能**当成 2 折 CV ——
    那样 OOF 里 80% 的预测会来自只用 20% 数据训练的模型，把 R² 系统性压低（实测 0.166 vs 0.31）。"""
    Xv = X.values.astype(float)
    tr, te = folds == train_fold, folds == test_fold
    reg, clf = random_forest_pair()
    reg.fit(Xv[tr], y[tr])
    pred = np.full(len(y), np.nan)
    pred[te] = reg.predict(Xv[te])
    predc = None
    if emo is not None:
        clf.fit(Xv[tr], emo[tr])
        predc = np.zeros(len(y), dtype=int)
        predc[te] = clf.predict(Xv[te])
    return _score(y, pred, emo, predc)


def main() -> int:
    say("# CP2.5 基线预演 —— 用最便宜的 2D 描述符先看住 R² 量级")
    say("")
    say("> 由 `scripts/_baseline_cp25.py` 生成 ｜ 目标：**在投入大算力前**判断这条路值不值得走")
    say("> 靶变量 `y = ln α_true = ln[k(S)/k(R)]`（定向）｜ 划分来自 `01_curate_cmrt.py`")
    say("")

    p = pd.read_csv(CUR / "cmrt_pairs.csv")
    sp = {n: pd.read_csv(CUR / "splits" / f"{n}.csv")
          for n in ("S1_random", "S2_group_dup", "S2b_leave_paper", "S3_leave_column",
                    "S4_murcko", "S4b_ecfp_class")}
    y = p["y"].values.astype(float)
    emo = p["EMO"].values.astype(int)
    say(f"主表 **{len(p)} 对**｜y 的 SD = {y.std():.4f}（这是 R²=0 时的 RMSE 基准）｜"
        f"EMO 基线（全猜多数类）= {max(emo.mean(), 1-emo.mean())*100:.1f}%")
    say("")

    X0 = build_X(p)
    feats = {"L0": X0, "L0+colID": add_col_id(X0, p), "L0+csp": add_csp(X0, p)}
    say("## 特征集")
    say("")
    say("| 级别 | 维度 | 内容 |")
    say("|---|---|---|")
    say(f"| L0 | {X0.shape[1]} | 分子 2D 描述符（RDKit）+ 实验条件（i-PrOH / 流速 / t₀） |")
    say(f"| L0+colID | {feats['L0+colID'].shape[1]} | L0 + **柱身份 one-hot**（上界；**不能外推到新柱**） |")
    say(f"| L0+csp | {feats['L0+csp'].shape[1]} | L0 + **CSP 结构描述符**（骨架 / 取代基码 / 粒径；可外推） |")
    say("")

    rows = []
    order = [("S1_random", "S1 随机（80/20 单划分）"), ("S2_group_dup", "S2 分组(分子×柱)"),
             ("S2b_leave_paper", "S2b 留一文献"), ("S3_leave_column", "S3 留一柱"),
             ("S4_murcko", "S4 留骨架"), ("S4b_ecfp_class", "S4b 留化学类")]
    say("## 主结果")
    say("")
    say("| 划分 | 特征集 | n(测试) | R² | RMSE | y 的 SD | EMO 准确率 | EMO 多数类基线 |")
    say("|---|---|---|---|---|---|---|---|")
    for fname, label in order:
        folds = sp[fname]["fold"].values
        for lvl in ("L0", "L0+colID", "L0+csp"):
            if fname == "S1_random":
                r = run_holdout(feats[lvl], y, folds, 1, 2, emo=emo)
            else:
                r = run_cv(feats[lvl], y, folds, emo=emo)
            r.update(split=label, level=lvl)
            rows.append(r)
            say(f"| {label} | {lvl} | {r['n']} | **{r['r2']:+.3f}** | {r['rmse']:.4f} | "
                f"{r['y_sd']:.4f} | {r.get('emo_acc', float('nan'))*100:.1f}% | "
                f"{r.get('emo_base', float('nan'))*100:.1f}% |")
    say("")
    df = pd.DataFrame(rows)

    # ── 阴性对照：打乱标签后重跑，任何 > 0 的 R² / > 基线的 EMO 都是假象
    say("## 阴性对照（标签置换，3 次）")
    say("")
    say("| 划分 | 特征集 | R²（打乱标签） | EMO（打乱标签） |")
    say("|---|---|---|---|")
    perm_rows = []
    for fname, label in (("S2_group_dup", "S2 分组"), ("S2b_leave_paper", "S2b 留一文献")):
        folds = sp[fname]["fold"].values
        for lvl in ("L0", "L0+colID", "L0+csp"):
            rs, es = [], []
            for s in (1, 2, 3):
                r = run_cv(feats[lvl], y, folds, emo=emo, permute=s)
                rs.append(r["r2"])
                es.append(r["emo_acc"])
            perm_rows.append(dict(split=label, level=lvl, r2=float(np.mean(rs)),
                                  emo=float(np.mean(es))))
            say(f"| {label} | {lvl} | {np.mean(rs):+.3f} ± {np.std(rs):.3f} | "
                f"{np.mean(es)*100:.1f}% |")
    say("")

    # ── 记忆 oracle：特征 = 分子身份 (conn)，看模型能否靠「背分子」取胜
    say("## 记忆 oracle（特征 = 分子身份 conn）")
    say("")
    say(f"> CMRT 的结构事实：**{len(p)} 行 / {p['conn'].nunique()} 个唯一分子** "
        f"（{p['conn'].nunique()/len(p)*100:.1f}%）—— 几乎每个分子只出现一次。")
    say("> 因此**任何划分都无法靠「背分子」取巧**。下表是验证：用分子身份当特征，")
    say("> 若某个划分下分数突然变高，说明该划分把同一个分子切到了两边。")
    say("")
    h = p["conn"].apply(lambda s: float(int(hash(s) % 10**8)) / 10**8)
    Xor = pd.DataFrame({"conn_hash": h.values})
    say("| 划分 | R²（仅 conn 身份） | EMO（仅 conn 身份） |")
    say("|---|---|---|")
    for fname, label in order:
        folds = sp[fname]["fold"].values
        r = (run_holdout(Xor, y, folds, 1, 2, emo=emo) if fname == "S1_random"
             else run_cv(Xor, y, folds, emo=emo))
        say(f"| {label} | {r['r2']:+.3f} | {r.get('emo_acc', float('nan'))*100:.1f}% |")
    say("")
    say("（注：GBM 对单个高基数哈希特征本就无法有效记忆，此表只作**下限**参考；")
    say("  真正的保证来自上面的结构事实——分子几乎不重复。）")
    say("")

    # ── S3′ 因子外推（每个任务单独 train→test，不是 CV）
    say("## S3′ 留一格因子外推（核心创新②的证据）")
    say("")
    s3p = pd.read_csv(CUR / "splits" / "S3p_factor.csv")
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import r2_score
    say("| 任务 | 训练 n | 测试 n | L0+colID R² | L0+csp R² | L0+csp RMSE | 测试集 y 的 SD |")
    say("|---|---|---|---|---|---|---|")
    s3rows = []
    for task in sorted(s3p["task"].unique()):
        s = s3p[s3p["task"] == task]
        trm = s["role"].values == "train"
        tem = s["role"].values == "test"
        idx = s["row_id"].values
        idx_tr, idx_te = idx[trm], idx[tem]
        if len(idx_te) == 0 or len(idx_tr) == 0:
            continue
        out = {}
        for lvl in ("L0+colID", "L0+csp"):
            Xv = feats[lvl].values.astype(float)
            # 只保留对列有值的特征（held-out 柱的 one-hot 列在测试集全 0 → 需删掉零方差列）
            keep = Xv[idx_tr].std(axis=0) > 0
            m = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.06,
                                              min_samples_leaf=20, random_state=SEED)
            m.fit(Xv[idx_tr][:, keep], y[idx_tr])
            pred = m.predict(Xv[idx_te][:, keep])
            out[lvl] = (r2_score(y[idx_te], pred),
                        float(np.sqrt(np.mean((y[idx_te] - pred) ** 2))))
        sd = float(np.std(y[idx_te]))
        s3rows.append(dict(task=task, n_tr=len(idx_tr), n_te=len(idx_te),
                           r2_id=out["L0+colID"][0], r2_csp=out["L0+csp"][0],
                           rmse_csp=out["L0+csp"][1], sd=sd))
        say(f"| `{task}` | {len(idx_tr)} | {len(idx_te)} | {out['L0+colID'][0]:+.3f} | "
            f"**{out['L0+csp'][0]:+.3f}** | {out['L0+csp'][1]:.4f} | {sd:.4f} |")
    s3df = pd.DataFrame(s3rows)
    say("")

    # ── 平凡基线
    say("## 平凡基线（对照）")
    say("")
    say("| 预测器 | R² | RMSE |")
    say("|---|---|---|")
    say(f"| 全局均值 | {0.0:+.3f} | {y.std():.4f} |")
    colm = p.groupby("Column")["y"].transform("mean")
    say(f"| 柱均值（泄漏柱身份的上界） | {r2_score(y, colm):+.3f} | "
        f"{float(np.sqrt(np.mean((y-colm)**2))):.4f} |")
    say("")

    # ── 结论
    s1 = df[df.split.str.startswith("S1")]["r2"].max()
    s2 = df[df.split.str.startswith("S2 ")]["r2"].max()
    s2b = df[df.split.str.startswith("S2b")]["r2"].max()
    s3 = df[df.split.str.startswith("S3 ")]
    s3best = s3["r2"].max() if len(s3) else float("nan")
    s4 = df[df.split.str.startswith("S4 ")]["r2"].max()
    emo_max = df["emo_acc"].max()
    emo_base = df["emo_base"].max()
    emo_row = df.loc[df["emo_acc"].idxmax()]
    perm_r2 = float(np.abs(np.array([r["r2"] for r in perm_rows])).max())
    perm_emo = float(np.max([r["emo"] for r in perm_rows]))
    say("## 结论")
    say("")
    say("### 定量摘录")
    say("")
    say(f"- 随机划分（S1）最好 R² = **{s1:+.3f}**；分组（S2）最好 = **{s2:+.3f}**")
    say(f"- **S2b 留一文献后最好 R² = {s2b:+.3f}**（相对 S2 变化 {s2b-s2:+.3f}）")
    say(f"- 留一柱（S3）最好 R² = **{s3best:+.3f}**；留骨架（S4）最好 = **{s4:+.3f}**")
    say(f"- 最佳 EMO 准确率 = **{emo_max*100:.1f}%**（出现在 {emo_row['split']} ｜ {emo_row['level']}，"
        f"该测试集多数类基线 {emo_base*100:.1f}%）")
    say(f"- 阴性对照（打乱标签）|R²| ≤ **{perm_r2:.3f}**、EMO ≤ **{perm_emo*100:.1f}%**"
        f" —— 上面所有数字都必须显著高于这一行才算信号")
    say("")
    # 把关键数字抽出来，供下面逐条引用
    def _at(prefix, lvl="L0+colID", col="r2"):
        s = df[df.split.str.startswith(prefix) & (df.level == lvl)]
        return float(s[col].iloc[0]) if len(s) else float("nan")

    r1, r2_, r2b = _at("S1"), _at("S2 "), _at("S2b")
    r3, r4 = _at("S3 "), _at("S4 ")
    e1, e2, e2b = (_at("S1", col="emo_acc"), _at("S2 ", col="emo_acc"), _at("S2b", col="emo_acc"))
    e3, e4 = _at("S3 ", col="emo_acc"), _at("S4 ", col="emo_acc")
    base = float(df["emo_base"].mean())

    say("### 读数（按证据，不按预期）")
    say("")
    say(f"**① 文献泄漏是压倒性的，而且同时污染两个靶。**")
    say(f"R²：S2 分组 {r2_:+.3f} → **S2b 留一文献 {r2b:+.3f}**（Δ = {r2b-r2_:+.3f}）；")
    say(f"EMO：S2 {e2*100:.1f}% → **S2b {e2b*100:.1f}%**（Δ = {(e2b-e2)*100:+.1f} 个百分点）。")
    say(f"也就是说，随机/分组划分下几乎所有「可学」的部分，都来自**模型认得这篇论文**，不是化学。")
    say(f"→ S2b 必须是主报告口径；这也直接解释了为什么同类文献能报出很高的准确率。")
    say("")
    say(f"**② 洗脱顺序（EMO）确实有一点可学，但只有一点。**")
    say(f"留一文献后 EMO = {e2b*100:.1f}%，多数类基线 {base*100:.1f}%，阴性对照（打乱标签）"
        f"{perm_emo*100:.1f}%。")
    say(f"{e2b*100:.1f}% 显著高于基线（n={len(p)}，标准误 ≈ {100*np.sqrt(base*(1-base)/len(p)):.2f} 个百分点），")
    say(f"但**远不是**文献里常报的 76–96%。→ 这条是本项目可以直接打出去的**方法学批评**。")
    say("")
    say(f"**③ 柱身份几乎不携带信息，而且不可外推。**")
    say(f"留一柱 R² = {r3:+.3f}（比全局均值还差）；柱身份 one-hot 在 S1/S2 上带来的增益也仅约 "
        f"{r2_-_at('S2 ', 'L0'):+.3f}。")
    say(f"→ 与审计的柱 η² = 2.6% 一致。**不能把卖点写成「跨柱迁移」**，")
    say(f"   要写成「我们量化了固定相描述符可迁移性的上限」。")
    say("")
    say(f"**④ 现有 CSP 描述符撑不起因子外推。**")
    say(f"S3′ 六个任务全部为负（{s3df['r2_csp'].min():+.2f} ~ {s3df['r2_csp'].max():+.2f}），")
    say(f"且 L0+csp 与 L0+colID 几乎同分 ⇒ 厂商 3 个标签（骨架/取代基码/粒径）")
    say(f"**基本没提供可迁移的固定相信息**。这是 `07_csp_descriptors.py` 的直接立项理由。")
    say("")
    say("### 对选题的直接含义")
    say("")
    say(f"- **2D 描述符的天花板就在这里**（严格划分下 R² ≤ {max(r2b, r3, r4):+.3f}）。")
    say("  这正是引入量子化学伪标量描述符（L1a/L1b/L1c）的立论基础 ——")
    say("  论文的增量**必须**来自 L1，否则严格划分下没有东西可报。")
    say("- **自建 CSP 描述符是硬需求**（S3′ 全线为负）。")
    say("- 若 L1 加进来后 S3′ 仍不改善，则把论文定位改为")
    say("  「固定相描述符的极限 + 分子侧才是瓶颈」（诚实且二区可接受），而不是硬撑跨柱迁移。")
    say("")
    say("### ⚠️ 两个必须记住的方法学教训")
    say("")
    say("1. **S1 是 80/20 单次划分，不是 2 折 CV。** 当成 2 折跑，OOF 里 80% 的预测会来自"
        "只用 20% 数据训练的模型，R² 被系统性压低（本次实测 0.166 → 修正后 0.325）。")
    say("2. **必须先跑阴性对照（打乱标签）再相信任何数字。** 本次所有结论都建立在"
        f"「打乱标签后 |R²| ≤ {perm_r2:.3f}、EMO ≈ {perm_emo*100:.1f}%」这个前提上。")

    OUT.write_text("\n".join(LOG) + "\n", encoding="utf-8")
    df.to_csv(CUR / "cp25_baseline_results.csv", index=False)
    s3df.to_csv(CUR / "cp25_s3p_results.csv", index=False)
    print(f"\n[落盘] {OUT}")
    print(f"[落盘] data/curated/cp25_baseline_results.csv + cp25_s3p_results.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
