# -*- coding: utf-8 -*-
"""
79_protocol_emo_ci.py — **H3：协议 EMO 的置信区间（按文献聚类 bootstrap）**
================================================================================
立项（2026-11-01 升级评估 manuscript_audit.md U2）：
  论文头条数字 52.85%（S2c 真留一文献）与「66.13 → 71.79 → 53.94 → 52.85」的协议梯度
  **全部没有置信区间**。审稿人必问「52.9% 的不确定度是多少」。
做法：
  1. 用与 scripts/18 **完全相同的分类器/特征/折**重算四套协议的 OOF 预测并落盘
     （S1 随机 2 折 / S2 分组 5 折 / S2b 留一批文献 10 折 / S2c 真留一文献 530 折）；
  2. 断言 S2c 复现值 = s2c_emo.csv 的 52.8456%（容差 0.1 pp）；
  3. **按文献（DOI）聚类 bootstrap** B=2000：整篇文献有放回重采样，
     报各协议 EMO 的 95% CI、以及配对差值（S2c−S2b、S2c−S2、S2c−S1）的 95% CI；
  4. 同时报「52.85% 相对全局多数类 50.67% 的增益 +2.18 pp」的 CI。

用法：PYTHONIOENCODING=utf-8 E:/python.exe scripts/79_protocol_emo_ci.py
产出：data/curated/upgrade_oof_preds.csv（逐行 OOF 预测）
      data/curated/protocol_emo_bootstrap.csv
      notes/协议EMO的置信区间.md
"""
from __future__ import annotations
import importlib.util, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CUR = ROOT / "data" / "curated"
LOG: list[str] = []
B = 2000
SEED = 20261101


def say(s=""):
    print(s, flush=True); LOG.append(s)


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def emo_oof(X, emo, folds, tag):
    """逐折训练 HGB 分类器（与 scripts/18 同一配置）→ OOF 预测。"""
    from sklearn.metrics import accuracy_score
    Xv = X.values.astype(float)
    pred = np.full(len(emo), -1, dtype=int)
    uf = np.unique(folds)
    t0 = time.time()
    for k, f in enumerate(uf):
        tr, te = folds != f, folds == f
        if te.sum() == 0 or tr.sum() == 0:
            continue
        _, clf = b25.random_forest_pair()
        clf.fit(Xv[tr], emo[tr])
        pred[te] = clf.predict(Xv[te]).astype(int)
        if k == 4:
            per = (time.time() - t0) / 5
            print(f"  [{tag}] {len(uf)} 折，前 5 折 {per:.2f} s/折 ⇒ 预计 {per*len(uf):.0f} s", flush=True)
    ok = pred >= 0
    acc = float(accuracy_score(emo[ok], pred[ok]))
    print(f"  [{tag}] 完成 {time.time()-t0:.0f} s，EMO={acc*100:.4f}%，覆盖 {int(ok.sum())}/{len(emo)}", flush=True)
    return pred, acc


def main() -> int:
    t_all = time.time()
    global b25
    b25 = load_mod("_b25", ROOT / "scripts" / "_baseline_cp25.py")

    p = pd.read_csv(CUR / "cmrt_pairs.csv", low_memory=False)
    p["row_id"] = np.arange(len(p))
    fw = pd.read_csv(CUR / "cmrt_pairs_with_folds.csv", low_memory=False)
    assert len(fw) == len(p)
    folds = {}
    for col, name in [("S1_random_fold", "S1"), ("S2_group_fold", "S2"), ("S2b_paper_fold", "S2b")]:
        folds[name] = fw[col].values
    sp = pd.read_csv(CUR / "splits" / "S2c_leave_doi.csv")
    p["fold_s2c"] = p["row_id"].map(sp.set_index("row_id")["fold"])
    assert p["fold_s2c"].notna().all()
    folds["S2c"] = p["fold_s2c"].values

    X0 = b25.build_X(p).astype(float)
    emo = p["EMO"].values.astype(int)
    doi = p["doi"].astype(str).values

    say("# 协议 EMO 的置信区间（按文献聚类 bootstrap）")
    say("")
    say(f"> 由 `scripts/79_protocol_emo_ci.py` 生成 ｜ 2026-11-01 ｜ 升级评估 H3 ｜ B={B} 次文献级重采样")
    say(f"> 分类器/特征/折与 `scripts/18` 完全一致（HistGradientBoosting 250 iter，L0 28 维）。")
    say("")
    preds = {}
    accs = {}
    for name in ["S1", "S2", "S2b", "S2c"]:
        pr, acc = emo_oof(X0, emo, folds[name], name)
        preds[name] = pr; accs[name] = acc
    out = pd.DataFrame({"row_id": p["row_id"].values, "doi": doi, "emo": emo,
                        "pred_S1": preds["S1"], "pred_S2": preds["S2"],
                        "pred_S2b": preds["S2b"], "pred_S2c": preds["S2c"]})
    out.to_csv(CUR / "upgrade_oof_preds.csv", index=False, encoding="utf-8")

    # 断言：复现 s2c_emo.csv
    ref = pd.read_csv(CUR / "s2c_emo.csv").set_index("split")["emo_acc"]
    checks = {"S1_random": ("S1", 0.6612826603325416), "S2_group": ("S2", 0.7179097387173397),
              "S2b_leave_paper": ("S2b", 0.5393824228028503), "S2c_leave_doi": ("S2c", 0.528456057007126)}
    for ref_name, (k, v) in checks.items():
        got = accs[k]
        assert abs(got - v) < 1e-3, f"{k} 复现失败：{got:.6f} vs {v:.6f}"
        say(f"- ✅ {k} 复现：{got*100:.4f}%（参考 {v*100:.4f}%）")
    say("")

    # ── 文献级聚类 bootstrap
    say("## 1. 按文献聚类 bootstrap（整篇文献有放回重采样）")
    say("")
    uniq = np.unique(doi)
    idx_by_doi = {d: np.where(doi == d)[0] for d in uniq}
    # 每个协议：每篇的 (n_i, correct_i)
    per = {}
    for k in ["S1", "S2", "S2b", "S2c"]:
        ok = preds[k] >= 0
        per[k] = np.array([(len(idx_by_doi[d]), int((preds[k][idx_by_doi[d]] == emo[idx_by_doi[d]]).sum()),
                            int(ok[idx_by_doi[d]].sum())) for d in uniq], dtype=float)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(uniq), size=(B, len(uniq)))
    boot = {k: np.empty(B) for k in per}
    for b in range(B):
        sel = draws[b]
        for k in per:
            n_i = per[k][sel, 0]; c_i = per[k][sel, 1]
            boot[k][b] = c_i.sum() / n_i.sum()
    rows = []
    for k in ["S1", "S2", "S2b", "S2c"]:
        a = boot[k]
        lo, hi = np.percentile(a, [2.5, 97.5])
        rows.append(dict(quantity=f"EMO_{k}", point=accs[k], lo=lo, hi=hi, sd=float(a.std(ddof=1))))
        say(f"| EMO {k} | **{accs[k]*100:.2f}%** | [{lo*100:.2f}, {hi*100:.2f}] | ±{a.std(ddof=1)*100:.2f} pp |")
    say("")
    say("## 2. 配对差值（同一批重采样文献）")
    say("")
    for a_name, b_name in [("S2c", "S2b"), ("S2c", "S2"), ("S2c", "S1"), ("S2b", "S2")]:
        d = boot[a_name] - boot[b_name]
        lo, hi = np.percentile(d, [2.5, 97.5])
        p_le0 = float((d <= 0).mean())
        rows.append(dict(quantity=f"delta_{a_name}_minus_{b_name}", point=accs[a_name]-accs[b_name],
                         lo=lo, hi=hi, sd=float(d.std(ddof=1))))
        say(f"| {a_name} − {b_name} | **{(accs[a_name]-accs[b_name])*100:+.2f} pp** | "
            f"[{lo*100:+.2f}, {hi*100:+.2f}] pp | P(Δ≤0)={p_le0:.4f} |")
    say("")
    # 相对全局多数类的增益
    maj = max(emo.mean(), 1 - emo.mean())
    d = boot["S2c"] - maj
    lo, hi = np.percentile(d, [2.5, 97.5])
    rows.append(dict(quantity="gain_S2c_vs_global_majority", point=accs["S2c"]-maj, lo=lo, hi=hi,
                     sd=float(d.std(ddof=1))))
    say(f"**S2c 相对全局多数类（{maj*100:.2f}%）的增益**：**{(accs['S2c']-maj)*100:+.2f} pp**，"
        f"95% CI [{lo*100:+.2f}, {hi*100:+.2f}] pp，P(增益≤0)={float((d<=0).mean()):.4f}")
    say("")
    say("> 读法：文献级聚类 bootstrap 保留「同篇文献内相关」结构，CI 比行级 bootstrap 更保守。")
    say("> 若 S2c 的 CI 下沿仍高于全局多数类，则「+2.18 pp 可检出」在不确定度口径下也站得住。")
    say("")
    pd.DataFrame(rows).to_csv(CUR / "protocol_emo_bootstrap.csv", index=False, encoding="utf-8")
    say(f"> 机时：{time.time()-t_all:.0f} s ｜ 落盘 `data/curated/protocol_emo_bootstrap.csv`、"
        f"`data/curated/upgrade_oof_preds.csv`（{len(out)} 行）")
    (ROOT / "notes" / "协议EMO的置信区间.md").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    print(f"[落盘] notes/协议EMO的置信区间.md ｜ {time.time()-t_all:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
