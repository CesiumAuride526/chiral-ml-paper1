# -*- coding: utf-8 -*-
"""
26_paper_number_gate.py — **论文数字门禁**
================================================================================
## 立项理由（2026-09-17 第七场）

本项目已有一道**数据门禁** `_qa_01_curate.py`（54 项，保证清洗规格没被改坏），
但**没有一道门禁守护论文里引用的数字**。本场就因此踩过至少两次：

  * **混用不同脚本的 EMO**：`_baseline_cp25` 的 S1=73.8% 与 `scripts/18` 的 S1=66.13%
    是两套配置，混着写进同一张表就会自相矛盾（已修正，见 `docs/选题决策.md` 证据 F）；
  * **把上一场的过度概括抄进论文**：零阶扫描「三代有符号列都不如偶量列」是错的
    （L1a-FF 实为 0.0189 vs 0.0141）。

⇒ 本脚本把**论文里引用的关键数字**写成断言，跑一遍即可确认「图/表/正文与落盘数据一致」。

## 用法
```bash
PYTHONIOENCODING=utf-8 python scripts/26_paper_number_gate.py
```
退出码 0 = 全过；非 0 = 有数字对不上，**改论文前先改这里**。

⚠️ **本脚本只读 CSV，不重算任何模型**。它守护的是「论文里写的数 = 落盘 CSV 里的数」，
不守护「CSV 里的数算得对」（那是各个 `17`–`25` 脚本自己的事）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CUR = ROOT / "data" / "curated"
OK, FAIL = [], []


def chk(name: str, got, want, tol=5e-4, note: str = "") -> None:
    try:
        good = abs(float(got) - float(want)) <= tol
    except Exception:
        good = False
    (OK if good else FAIL).append((name, got, want, note))


def chk_eq(name: str, got, want, note: str = "") -> None:
    (OK if got == want else FAIL).append((name, got, want, note))


def main() -> int:
    print("=" * 78)
    print("论文数字门禁：断言「论文里引用的数 = 落盘 CSV 里的数」")
    print("=" * 78)
    print()

    # ── 1. 协议梯度（必须整行取自 scripts/18）
    e = pd.read_csv(CUR / "s2c_emo.csv").set_index("split")
    chk("S1 随机 EMO (script 18)", e.loc["S1_random", "emo_acc"] * 100, 66.13, 0.01)
    chk("S2 分组 EMO (script 18)", e.loc["S2_group", "emo_acc"] * 100, 71.79, 0.01)
    chk("S2b 留一批 EMO", e.loc["S2b_leave_paper", "emo_acc"] * 100, 53.94, 0.01)
    chk("S2c 真·留一 EMO", e.loc["S2c_leave_doi", "emo_acc"] * 100, 52.85, 0.01)
    # S3 的 EMO 不在 s2c_emo.csv 里，由 scripts/32_s3_emo.py 产出。
    # 🔴 第八场修复：原先是 `if _s3.exists():` **条件断言** —— 文件一丢，门禁从 45 项
    #    静默降为 44 项**仍报 ✅ 且 exit 0**（代码审计实测三态）。论文 Table 2 引用了这个数，
    #    所以「找不到来源」必须 FAIL，不能少查一项。
    _s3 = CUR / "s3_emo.csv"
    if not _s3.exists():
        FAIL.append((
            "S3 EMO 的来源 CSV `data/curated/s3_emo.csv` **缺失**"
            "（论文 Table 2 引用 S3 = 56.46%；该文件由 `scripts/32_s3_emo.py` 产出）",
            "文件不存在", "存在", "第八场新增的硬依赖断言"))
    else:
        chk("S3 留一柱 EMO", pd.read_csv(_s3).loc[0, "emo_acc"] * 100, 56.46, 0.01)
    chk("S2c 逐折多数类 oracle 基线",
        e.loc["S2c_leave_doi", "emo_base_perfold"] * 100, 76.55, 0.01)
    chk("全局多数类基线", e.loc["S2c_leave_doi", "emo_base_global"] * 100, 50.67, 0.01)

    # ── 2. S2c 的 R²（两臂）
    lo = pd.read_csv(CUR / "leave_one_doi_summary.csv").set_index("arm")
    chk("S2c R² (L0)", lo.loc["L0", "r2"], -0.1073, 5e-4)
    chk("S2c R² (L0+CF+FF)", lo.loc["L0+CF+FF", "r2"], -0.0477, 5e-4)
    chk("S2c R²_aff (L0)", lo.loc["L0", "r2_aff"], 0.4393, 1e-3)

    # ── 3. 天花板分解 + bootstrap CI
    #  ★ 第八场 task-16 重算（原版有三个问题：55 行无结构分子被 pd.get_dummies 当成参照分子、
    #    ss 三项样本不一致不可加、点估计 549 行与 bootstrap 494 行不同 n）：
    #    主口径改为**剔除 55 行无结构分子（n=494）**、点估计与 bootstrap 同 n、**序贯分解**。
    #    结果：22.29% → **23.05%**，CI [17.98, 32.78] → **[17.06, 34.14]**。
    ic = pd.read_csv(CUR / "interaction_ceiling.csv").iloc[0]
    chk_eq("加性上界：样本 n（剔无结构分子）", int(ic.n_rows), 494)
    chk_eq("加性上界：理论格数", int(ic.n_theory), 516)
    chk("加性上界：网格完整度", float(ic.completeness), 0.9574, 1e-3)
    chk("点估计 n 与样本一致", float(ic.n), float(ic.n_rows), 0.5)
    chk_eq("bootstrap 重复次数", int(ic.boot_n), 200)
    chk("分子主效应 η²（序贯）", ic.eta2_mol, 0.2180, 1e-3)
    chk("条件主效应 η²（序贯）", ic.eta2_cond, 0.0125, 1e-3)
    chk("加性上界 η²_model", ic.eta2_model, 0.2305, 1e-3)
    chk("交互 + 噪声 η²_resid", ic.eta2_resid, 0.7695, 1e-3)
    chk("上界 bootstrap 下沿", ic.eta2_model_lo, 0.1706, 1e-3)
    chk("上界 bootstrap 上沿", ic.eta2_model_hi, 0.3414, 1e-3)
    chk("敏感性① 48 水平（549 行）上界", ic["_sens48_549_rows_eta2_model"], 0.2233, 1e-3)
    chk("敏感性② 平衡子网格（360 行）上界", ic["_sensbal_360_rows_eta2_model"], 0.2018, 1e-3)
    chk("敏感性③ 旧口径（549 行）上界", ic["_senslegacy_549_rows_eta2_model"], 0.2229, 1e-3)
    # 序贯分解必须自洽（边际份额不可加，但序贯三项必须加得起来）
    chk("序贯三项之和 = 1", ic.eta2_model + ic.eta2_resid, 1.0, 1e-6)

    # ── 4. 同/换实验室
    w = pd.read_csv(CUR / "within_paper_split.csv")
    chk("同实验室逐篇 r 中位", w.r.median(), 0.185, 1e-3)
    chk("同实验室 r>0 篇占比", (w.r > 0).mean(), 0.80, 0.01)
    lc = pd.read_csv(CUR / "learning_curve_papers.csv")
    chk("跨文献 r 中位（k=250）", lc[lc.k == 250].r_med.iloc[0], 0.0009, 1e-3)
    chk("跨文献 R²_aff 最大（任意 k）", lc.r2_aff.max(), 0.0283, 1e-3)

    # ── 5. 无模型条件对照
    cp = pd.read_csv(CUR / "condition_pairwise.csv")
    sc = cp[cp.same_col & ~cp.same_ph]
    dc = cp[~cp.same_col & cp.same_ph]
    both = cp[~cp.same_col & ~cp.same_ph]
    chk("同柱异相 r 中位", sc.r.median(), 0.560, 1e-3)
    chk("异柱同相 r 中位", dc.r.median(), 0.128, 1e-3)
    chk("异柱异相 r 中位", both.r.median(), 0.084, 1e-3)
    chk("异柱异相 符号一致率中位", both.sign_agree.median() * 100, 51.3, 0.1)

    # ── 6. 条件对齐外部验证
    od = pd.read_csv(CUR / "external_odrh_result.csv")
    al = od[od.train.str.contains("Cellulose-a", na=False) & (od.n_test == 73)]
    chk("OD-RH 选择剂对齐 R²", float(al.r2.iloc[0]), -0.802, 1e-3)

    # ── 7. 跨选择剂梯度
    ec = pd.read_csv(CUR / "external_all_columns.csv")
    m = ec[~ec.column.str.contains(r"\[aligned\]", regex=True, na=False)]
    chk_eq("跨选择剂：柱数", len(m), 6)
    chk("跨选择剂 最好 R²", m.r2.max(), -0.149, 1e-3)
    chk("跨选择剂 最差 R²", m.r2.min(), -1.441, 1e-3)
    chk("跨选择剂 均值", m.r2.mean(), -0.531, 1e-3)

    # ── 8. 死时间敏感性
    t0 = pd.read_csv(CUR / "t0_sensitivity.csv")
    chk_eq("φ 扫描点数", len(t0), 5)
    chk("φ 扫描 r 下沿", t0.r_vs_base.min(), 0.99444, 1e-4)
    chk("φ 扫描 EMO 翻转上沿", t0.emo_flip.max() * 100, 0.01, 0.005)
    chk_eq("φ 扫描 R² 全为负", bool((t0.r2 < 0).all()), True)

    # ── 9. 功效曲线
    pc = pd.read_csv(CUR / "offset_power_curve.csv")
    chk("种入 5% 的回收", pc[pc.planted == 0.05].recovered_mean.iloc[0], 0.0195, 1e-3)
    chk("种入 39% 的回收", pc[pc.planted == 0.39].recovered_mean.iloc[0], 0.3637, 1e-3)
    po = pd.read_csv(CUR / "paper_offset_predictability.csv").set_index("arm")
    chk("偏移可预测性（全组成）", po.loc["全组成（分子+条件）", "r2"], -0.0158, 1e-3)

    # ── 10. 数据集规模（论文正文到处引用）
    p = pd.read_csv(CUR / "cmrt_pairs.csv", low_memory=False)
    chk_eq("主表行数（对映体对）", len(p), 10525)
    chk_eq("文献数（DOI）", int(p.doi.nunique()), 530)
    chk_eq("柱数", int(p.Column.nunique()), 25)

    # ── 11. 描述符消融：**L1a-FF 是例外**（防止「三代都更差」的过度概括回归）
    z = pd.read_csv(CUR / "zeroorder_scan.csv")
    med = z.groupby(["family", "signed"]).r.apply(lambda s: s.abs().median())
    chk("L1-CF 偶量 |r| 中位", med[("L1-CF", False)], 0.0352, 1e-4)
    chk("L1-CF 有符号 |r| 中位", med[("L1-CF", True)], 0.0169, 1e-4)
    chk("L1b 偶量 |r| 中位", med[("L1b", False)], 0.0315, 1e-4)
    chk("L1b 有符号 |r| 中位", med[("L1b", True)], 0.0107, 1e-4)
    chk("L1a-FF 偶量 |r| 中位", med[("L1a-FF", False)], 0.0141, 1e-4)
    chk("L1a-FF 有符号 |r| 中位（**例外：更高**）", med[("L1a-FF", True)], 0.0189, 1e-4)
    if med[("L1a-FF", True)] <= med[("L1a-FF", False)]:
        FAIL.append(("L1a-FF 应是有符号 > 偶量的例外（若二者反转，"
                     "『三代都更差』的表述反而成立，须同步改论文）", None, None, ""))

    # ── 12. ★★ 第八场新增：**折数与整数口径**
    #    立项理由：手稿 6 处把 S2c 写成「523 folds」，而划分文件是 **530 折**。
    #    这道门禁的 45 项断言**没有一项**守护折数，`--trace` 又只匹配小数
    #    （旧正则 `\d+\.\d{1,5}`）⇒ 一个整数错数可以同时穿过两道门禁。
    #    现按第八场数字审计 §2 的三口径定论把它钉死。
    import csv as _csv

    SPL = CUR / "splits"

    def n_folds(fn: str, col: str = "fold") -> int:
        with (SPL / fn).open(encoding="utf-8") as fh:
            return len({row[col] for row in _csv.DictReader(fh)})

    n_s2c = n_folds("S2c_leave_doi.csv")
    chk_eq("S2c 划分折数（= 文献数）", n_s2c, 530)
    chk_eq("S2c EMO 臂折数 = 划分折数", int(e.loc["S2c_leave_doi", "n_fold"]), n_s2c)
    chk_eq("S1 折数", n_folds("S1_random.csv"), 2)
    chk_eq("S2 折数", n_folds("S2_group_dup.csv"), 5)
    chk_eq("S2b 折数", n_folds("S2b_leave_paper.csv"), 10)
    chk_eq("S3 折数", n_folds("S3_leave_column.csv"), 25)
    chk_eq("S4 折数", n_folds("S4_murcko.csv"), 5)
    chk_eq("S3′ 是长表（无 fold 列）",
           "fold" in pd.read_csv(SPL / "S3p_factor.csv", nrows=0).columns, False)
    # 三口径：530 划分 / 523 可评估文献 / 428 逐折统计覆盖 —— 论文必须写清引用的是哪一个
    lo_all = pd.read_csv(CUR / "leave_one_doi_summary.csv")
    chk_eq("S2c 逐折统计覆盖的折数", int(lo_all.n_fold.iloc[0]), 428)
    chk_eq("S2c 逐折统计覆盖的行数", int(lo_all.n.iloc[0]), 10193)

    # ── 13. ★★ 第八场新增：主表脏值与「论文曾写错」的量的回归守护
    chk_eq("DOI 分组数", int(p.doi.nunique()), 530)
    chk_eq("假 DOI 's' 的条数（源数据脏值，数据卡须披露）", int((p.doi == "s").sum()), 1)
    chk_eq("kS_rsd > 0 的行数（论文曾写 identically zero）", int((p.kS_rsd > 0).sum()), 396)
    chk_eq("kR_rsd > 0 的行数", int((p.kR_rsd > 0).sum()), 379)
    chk_eq("α ≡ 1 的行数", int(p.flag_no_separation.sum()), 17)

    # ── 14. ★★ 第八场新增：DG25 网格完整度与「未分开」占比
    #    论文曾写 "complete 43 × 12 grid"（实际 95.7 % 非均衡）与 "31–52 % non-separated"
    #    （实际逐柱 49–72 %、整体 58 %）。两处都由本条守护。
    dg = pd.read_csv(CUR / "external_dg25_all.csv")
    dg_ok = dg[dg.conn.notna()]
    n_cond = dg_ok.groupby(["column", "phase"]).ngroups
    chk_eq("DG25 长表行数", len(dg), 549)
    chk_eq("DG25 无结构分子行数（conn 为空）", int(dg.conn.isna().sum()), 55)
    chk_eq("DG25 非空单元格数", dg_ok.groupby(["conn", "column", "phase"]).ngroups, 494)
    chk_eq("DG25 条件数（柱 × 流动相）", n_cond, 12)
    per_mol = dg_ok.groupby("conn").apply(
        lambda g: g[["column", "phase"]].drop_duplicates().shape[0])
    chk_eq("DG25 覆盖全部条件的分子数", int((per_mol == n_cond).sum()), 30)
    dg_te = dg_ok[~dg_ok.cmrt_overlap.astype(bool)]
    ns_col = dg_te.groupby("column").separated.apply(lambda s: 1 - s.mean())
    chk_eq("外部测试集行数（剔重叠与无结构）", len(dg_te), 449)
    chk("外部测试集 未分开占比（整体）", float(1 - dg_te.separated.mean()), 0.5813, 1e-3)
    chk("外部测试集 未分开占比（逐柱下沿）", float(ns_col.min()), 0.4857, 1e-3)
    chk("外部测试集 未分开占比（逐柱上沿）", float(ns_col.max()), 0.7179, 1e-3)

    # ── 15. ★★ 第八场新增：**核心正面结论的阴性对照**
    #    论文的核心正面结论是「同实验室逐篇 r 中位 = +0.185、80% 文献为正」。
    #    第八场代码审计发现该结论**没有任何 null**（本项目自己的规矩：任何建模结论前必须先跑阴性对照）。
    #    task-13 补了置换检验；本断言把「实测值必须显著高于 null」这件事钉进门禁。
    wp = CUR / "within_paper_null.csv"
    if not wp.exists():
        FAIL.append((
            "核心正面结论缺少阴性对照 `data/curated/within_paper_null.csv`"
            "（无 null 的正面结论不合本项目规范；由 `scripts/19` 产出）",
            "文件不存在", "存在", "第八场新增的硬依赖断言"))
    else:
        wn = pd.read_csv(wp)
        chk_eq("置换检验重复次数", len(wn), 200)
        chk("阴性对照 r 中位均值", float(wn.r_med.mean()), -0.0118, 5e-3)
        chk("阴性对照 r>0 占比均值", float(wn.frac_r_gt0.mean()) * 100, 47.09, 0.5)
        chk_eq("实测 +0.185 高于全部 null 值（= 置换检验通过）",
               bool((wn.r_med < 0.185).all()), True)
        chk_eq("实测 80% 高于全部 null 值",
               bool((wn.frac_r_gt0 < 0.80).all()), True)

    # ── 16. ★★ 第八场新增：把原先「只写在 .md 里、门禁守护不到」的数纳入守护（task-20 落盘）
    #    这三张表的价值在于：文档里引用的 ±SD / 随机对照 / 种子扫描**第一次有了 CSV 出处**，
    #    以后改数据时它们会跟着响。
    pe = pd.read_csv(CUR / "external_pe25_result.csv")
    chk_eq("PE25 外部验证臂数", len(pe), 3)
    chk("PE25 全 84 对 R²", float(pe.r2.iloc[0]), -0.1687, 1e-3)
    chk("PE25 仅分开子集 R²", float(pe.r2.iloc[2]), -0.0207, 1e-3)
    nc = pd.read_csv(CUR / "negative_controls.csv")
    chk_eq("阴性对照条数", len(nc), 3)
    chk_eq("阴性对照 SD 口径统一为 ddof=0（否则 0.22→0.26 对不上）",
           bool((nc.sd_ddof == 0).all()), True)
    chk("S2c EMO 打乱对照均值", float(nc["mean"].iloc[0]), 50.17, 1e-2)
    ss = pd.read_csv(CUR / "seed_sensitivity.csv")
    chk_eq("种子扫描种子数", len(ss), 5)
    chk("种子扫描：同实验室 r 中位下沿", float(ss.within_r_median.min()), 0.1764, 1e-3)
    chk("种子扫描：换实验室 r 中位上沿", float(ss.across_r_median.max()), 0.0106, 1e-3)
    # 跨表组合断言：防「文档里的数」与「落盘 CSV 里的数」再次分家
    chk("手稿 §4.3 的 +0.185 = within_paper_split.csv 的中位",
        float(pd.read_csv(CUR / "within_paper_split.csv").r.median()), 0.185, 1e-3)
    chk("手稿 §4.6 的 OD-RH 打乱对照 = negative_controls 第 2 行",
        float(nc["mean"].iloc[1]), -0.7558, 1e-3)
    chk("手稿 §4.6 的跨柱随机对照 = negative_controls 第 3 行",
        float(nc["mean"].iloc[2]), -0.4653, 1e-3)

    # ── 输出
    print(f"通过 **{len(OK)}** 项，失败 **{len(FAIL)}** 项")
    print()
    if FAIL:
        print("❌ 失败明细：")
        for name, got, want, note in FAIL:
            print(f"   • {name}")
            print(f"       实测 {got}  期望 {want}  {note}")
        print()
        print("⇒ **论文里引用的这几个数字与落盘 CSV 不一致。**")
        print("   先确认是 CSV 变了还是论文写错了，再决定改哪边。")
        return 1
    print("✅ 全部通过：论文正文/图表引用的关键数字与 `data/curated/*.csv` 一致。")
    print()
    print("⚠️ 本门禁**不检查**「CSV 是否算得对」——那是 `17`–`25` 各自的责任；")
    print("   它只保证**写作时不会抄错数**。改数据后请一并重跑本脚本。")
    return 0


def trace_all(args=None) -> int:
    """**全量可追溯性检查**：把正文里出现的**每一个小数**回溯到源 CSV。

    比逐条断言更强 —— 断言只守护「我在意的那几十个数」，
    而这条能抓出**任何**凭空出现的数（例如我写错的、或从别处抄来的）。

    ⚠️ 已知白名单：软件版本号（如 `2026.03.6`）不是数据，按上下文排除。

    🔴 **第八场补充（整数盲区）**：原实现的正则是 `\\d+\\.\\d{1,5}`，
    **只匹配小数** ⇒ 「523 折」「530 篇」这类**整数**永远不会被追溯。
    手稿把 S2c 写成 523 折而划分文件是 530 折，正是从这个缝里漏过去的。
    现在补一节**整数追溯**（3–6 位），默认**只报告不判负**（整数里有大量非数据量：
    年份、期刊卷期、段落编号…），加 `--strict` 才计入退出码。
    """
    import glob
    import re

    import numpy as np
    import pandas as pd

    # 2026-10-12（整理场）：默认改指**当前正文**（两轮审稿修订稿 `_rev`）。旧稿 `编号引用.md`
    # 已作废，但此前 31 号门禁的追溯步一直默认跑在旧稿上 ⇒ 没守护真正的投稿正文。
    doc = Path(getattr(args, "doc", None) or (ROOT / "docs" / "Paper1_全文草稿_编号引用_rev.md"))
    if not doc.exists():
        print(f"❌ 找不到合成稿：{doc}"); return 1
    body = doc.read_text(encoding="utf-8").split("## References")[0]

    nums = {float(m.group(1)) for m in re.finditer(r"(?<![\w.])(\d+\.\d{1,5})(?![\w])", body)}

    pool: set[float] = set()
    for f in glob.glob(str(CUR / "*.csv")) + glob.glob(str(CUR / "splits" / "*.csv")):
        try:
            d = pd.read_csv(f, low_memory=False)
        except Exception:
            continue
        for c in d.columns:
            if pd.api.types.is_numeric_dtype(d[c]):
                for v in d[c].dropna().unique():
                    pool.add(round(float(v), 6)); pool.add(round(abs(float(v)), 6))
    pool |= {round(v * 100, 6) for v in list(pool)} | {round(v / 100, 6) for v in list(pool)}

    def traceable(x: float) -> bool:
        return any(abs(x - p) < 5e-4 for p in pool)

    untraced = sorted(x for x in nums if not traceable(x))
    # 软件版本号白名单
    untraced = [x for x in untraced if not any(
        f"{x}" in v for v in ("3.14.3", "2026.03.6", "1.9.0", "2.4.6", "3.0.3", "1.18.0"))]

    print(f"正文小数 **{len(nums)}** 个 ｜ 源 CSV 数字池 **{len(pool)}** 个")
    print(f"**不可追溯**：{len(untraced)} 个")
    for x in untraced:
        m = re.search(re.escape(str(x)), body)
        ctx = body[max(0, m.start() - 80):m.start() + 60].replace("\n", " ") if m else ""
        print(f"   ✗ {x:<12} …{ctx}…")
    if not untraced:
        print("✅ 正文里每一个数据小数都能回溯到 `data/curated/` 下的某个 CSV。")

    # ── ★★ 判据自检（null 对照）—— 第八场新增
    #    为什么必须有：Lead 独立实测（`scratch/audit8_lead/gate_trace_power.py`）
    #    把**随机小数**喂进同一判据，结果
    #      · 均匀 [0,1) 三位小数 → 2000/2000 = **100%** 被判「可追溯」
    #      · 均匀 [0,100) 两位小数 → **100%**
    #      · 对数均匀 1e-3..1e3  → **91.5%**
    #    ⇒ 这条判据对「凭空写出来的数」几乎没有免疫力：池子是 50 个 CSV 的**每一个**
    #      数值单元格再加 ×100/÷100（约 390 万个），任何三位小数都能撞上。
    #      所以「N 个小数 0 个不可追溯」**不能**当成「数字都有出处」的证据。
    #    ⇒ 本门禁自报判别力；`--strict` 下若判据已退化，直接判负。
    parr = np.array(sorted(pool), dtype=np.float64)

    def _trace_vec(xs: np.ndarray) -> np.ndarray:
        i = np.searchsorted(parr, xs)
        lo = np.clip(i - 1, 0, len(parr) - 1)
        hi = np.clip(i, 0, len(parr) - 1)
        return np.minimum(np.abs(xs - parr[lo]), np.abs(xs - parr[hi])) < 5e-4

    _rng = np.random.default_rng(20260918)
    null_sets = {
        "均匀[0,1) 三位小数": _rng.integers(0, 1000, 500) / 1000,
        "均匀[0,100) 两位小数": _rng.integers(0, 10000, 500) / 100,
        "对数均匀 1e-3..1e3": 10 ** _rng.uniform(-3, 3, 500),
    }
    print("\n★ 判据自检（null 对照：把**随机数**喂进同一判据，看会不会也被判「可追溯」）")
    worst = 0.0
    for k, v in null_sets.items():
        rate = float(_trace_vec(np.round(v, 6)).mean())
        worst = max(worst, rate)
        print(f"   {k:22s} 误判为可追溯 {rate*100:5.1f}%")
    weak = worst > 0.5
    if weak:
        print(f"   🔴 最高 **{worst*100:.0f}%** ⇒ 该判据**几乎无判别力**：")
        print("      「0 个不可追溯」≠「数字都有出处」，只说明池子太密。")
        print("      真正的守护是上面的逐条断言（含折数/口径/脏值；最新数量以本次运行打印的「通过 N 项」为准）。")
    else:
        print("   ✅ 判据仍有判别力。")

    # ── 整数追溯（第八场新增；默认只报告）
    ints = {int(m.group(1)) for m in re.finditer(r"(?<![\w.])(\d{3,6})(?![\w.])", body)}
    ipool: set[int] = set()
    for f in glob.glob(str(CUR / "*.csv")) + glob.glob(str(CUR / "splits" / "*.csv")):
        try:
            d = pd.read_csv(f, low_memory=False)
        except Exception:
            continue
        for c in d.columns:
            if pd.api.types.is_numeric_dtype(d[c]):
                for v in d[c].dropna().unique():
                    try:
                        ipool.add(int(v))
                    except (ValueError, OverflowError):
                        pass
                ipool.add(int(d[c].dropna().sum()))
                ipool.add(len(d))

    def int_traceable(x: int) -> bool:
        return x in ipool

    # 白名单：年份 / 卷期页码 / 段落与图表编号 / DOI 内数字 等「非数据整数」
    YEAR = set(range(1990, 2101))
    WHITELIST_CTX = (
        r"10\.\d{4,9}",              # DOI 里的数字
        r"(?:Table|Fig\.?|Note|S)\s*\d+",
        r"§\s*\d+",
        r"\b\d{4}\b",                # 年份
        r"\b\d+\s*(?:nm|cm|mm|µm|um|Da|g/mol|MHz|K|°C)\b",
    )
    odd = sorted(x for x in ints if not int_traceable(x))
    kept = []
    for x in odd:
        m = re.search(rf"(?<![\w.]){x}(?![\w.])", body)
        ctx = body[max(0, m.start() - 70):m.start() + 50].replace("\n", " ") if m else ""
        if x in YEAR:
            continue
        if any(re.search(p, ctx) for p in WHITELIST_CTX):
            continue
        kept.append((x, ctx))
    print(f"\n正文整数（3–6 位）**{len(ints)}** 个 ｜ 源整数池 **{len(ipool)}** 个 ｜ "
          f"未追溯且非白名单：**{len(kept)}** 个")
    for x, ctx in kept:
        print(f"   ✗ {x:<8} …{ctx}…")
    if not kept:
        print(f"✅（**弱判据**）正文整数扣掉白名单后都能在 CSV 里找到同值。")
        print(f"   ⚠️ 但整数池有 {len(ipool):,} 个值（含各列合计与行数）⇒ 同值几乎必然出现：")
        print(f"      手稿原来的错误数 **523** 也在池里（某个单元格恰好等于 523），所以这条")
        print(f"      **抓不到它**。整数层真正的守护是上面第 12–14 节的逐条断言。")

    if args is not None and getattr(args, "strict", False) and kept:
        print(f"\n[--strict] 有 {len(kept)} 个整数未追溯 ⇒ 判负。")
        return 1
    if args is not None and getattr(args, "strict", False) and weak:
        print("\n[--strict] 小数判据已退化（判据自检未过）⇒ 判负。")
        return 1
    return 1 if untraced else 0


if __name__ == "__main__":
    import argparse as _ap
    _p = _ap.ArgumentParser()
    _p.add_argument("--trace", action="store_true",
                    help="全量可追溯性检查（正文每个小数 + 整数的白名单外项 vs 源 CSV）")
    _p.add_argument("--strict", action="store_true",
                    help="配合 --trace：把「未追溯的整数」也计入退出码（默认只报告）")
    _p.add_argument("--doc", default=None,
                    help="配合 --trace：指定手稿路径（默认 docs/Paper1_全文草稿_编号引用_rev.md）。"
                         "★ 用途：在一份**人为改坏的副本**上跑，证明本门禁真的会失败（非空转自检）")
    _a = _p.parse_args()
    sys.exit(trace_all(_a) if _a.trace else main())
