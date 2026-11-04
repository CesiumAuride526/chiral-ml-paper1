# -*- coding: utf-8 -*-
"""38_make_figures_v3.py — **配图第三次修订（第十场：只修排版缺陷）**
================================================================================
## 相对 v2 的唯一改动 = 排版（**一个数字都没改**）
依据：`notes/审查报告_图件标注与引用编号_2026-09-20.md` §2.3 —— v2 七张图张张存在
文字互相压字 / 越界被裁（`save()` 用 `bbox_inches=None`，超宽文字直接在画布右缘切掉）。
本版逐张修掉，并把"文字包围盒两两相交 / 越出画布"变成 `audit()` 的**硬断言**
（v2 的自检只验 dpi 与声明宽度，所以这些缺陷 100 % 穿过去了）。

## v2 原文（保留对照）
34_make_figures_v2.py — **配图重做（第九场：两轮审稿后）**
================================================================================
## 为什么另起一个脚本
`23_make_figures.py` 的 `audit()` 按「5 张图 / 9 个输入 CSV」写死了验收断言。
本脚本是**重排版 + 加图**的独立实现，产物写进 `figures/v2/`，**不覆盖旧 fig1–fig5**，
因此两套图可以逐张对照（旧的仍在，审计记录也仍有效）。若要替换，改 `FIGDIR` 即可。

## 相对旧版的改动（全部数字仍只来自 data/curated/*.csv，不重算、不新增硬编码）
| 图 | 改动 |
|---|---|
| fig1 | (a) 阴性对照**可见化**：null 点 + 2.5–97.5 % 范围带（旧灰带 ΔRGB≤3 几乎看不清）＋ 全部直标 n；(b) 折数标签数据驱动，补 SD 说明，网格置底 |
| fig2 | 误差棒口径统一为**单侧 95 % t 区间**（旧图注写 ±2 SD、正文写 ±1 SD，互相矛盾）；标出可排除/不可排除档；标注观测线与"本功效下可排除的最小份额" |
| fig3 | (a) 直方图 → **雨云图**（半小提琴 + 散点 + 箱线），(b) 学习曲线加 SD 阴影带 + within 参考线；两面板等宽、加 n |
| fig4 | **改图型**：12×12 条件相关**热图**（连续色标，全篇首次出现）＋ 三类条件对的森林图（含 95 % CI）；解决旧版 (a)(b) 同色不同义 |
| fig5 | 补 **CI 带 + 双口径 bootstrap**（分子 / 分子+条件）＋ 三条敏感性区间带；条形加白缝 → 纹样区分 |
| fig6 | **新增：清洗流失漏斗** —— CMRT 原始记录 → 10,525 对的每一步（消解"简单减法不闭合"的疑虑） |
| fig7 | **新增：外部验证森林图** —— 6 根柱的 R² + 打乱对照，一图讲死"失败不比打乱好" |

## 用法
```bash
set PYTHONIOENCODING=utf-8
python scripts/34_make_figures_v2.py                 # PNG 600 dpi + PDF 矢量 + 预览 150 dpi
python scripts/34_make_figures_v2.py --vector none   # 只要位图
```
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
import zlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.text import Text

ROOT = Path(__file__).resolve().parent.parent
CUR = ROOT / "data" / "curated"
FIGDIR = ROOT / "figures" / "v3"

PROD_DPI = 600
PREVIEW_DPI = 150
MIN_PROD_DPI = 300
MAX_W_CM = 17.4
SOFT_W_CM = 17.0
TARGET_W_CM = 16.0        # 全部图统一声明宽度（双栏安全区，字号设计值即落版值）

# ── 配色（色盲友好 4 槽；CVD 最小 ΔE 已按旧审计口径验证过同族色相）────────────
BLUE, ORANGE, TEAL, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#8b5cf6"
INK, MUTED, GRID = "#1f2937", "#6b7280", "#e6e5e0"
BG = "#ffffff"
HATCH = {ORANGE: "\\\\\\\\", TEAL: "////", PURPLE: "...."}

VECTOR_SUFFIX = {"pdf": ".pdf", "eps": ".eps", "svg": ".svg"}

FIG_SOURCES = {
    "fig1_protocol": ("s2c_emo.csv", "s3_emo.csv", "leave_one_doi_summary.csv", "s2c_emo_shuffle.csv"),
    "fig2_power": ("offset_power_curve.csv", "paper_offset_predictability.csv"),
    "fig3_lab_transfer": ("within_paper_split.csv", "learning_curve_papers.csv"),
    "fig4_condition_grid": ("condition_pairwise.csv", "condition_pair_bootstrap.csv"),
    "fig5_variance": ("interaction_ceiling.csv", "review_round2_stats.csv"),
    # fig6 原来声明 curation_stats.json + cmrt_pairs.csv，但代码里 6 个数字是**写死的常量**、
    # 一个输入都没读（复核清单 D-17 的邻接发现）。已改为真读 curation_stats.json。
    "fig6_curation": ("curation_stats.json",),
    # fig7 的打乱对照取自 negative_controls.csv，不是 external_odrh_result.csv（D-17）。
    "fig7_external": ("external_all_columns.csv", "negative_controls.csv"),
}


def validate_sources() -> list[str]:
    """`FIG_SOURCES` 声明的每个输入都必须真的存在于 `data/curated/`。

    这里只查「文件在不在」。更严的一条 —— 「声明 vs 代码实际读取」是否一致 ——
    由 `scripts/_qa_fig_sources.py` 做 AST 静态核对（可重跑、非人工眼看）。
    """
    missing = []
    for name, srcs in FIG_SOURCES.items():
        for s in srcs:
            if not (CUR / s).exists():
                missing.append(f"{name} → {s}")
    return missing


# ────────────────────────────────────────────────────────────────────────────
# 样式 / 落盘 / 自检
# ────────────────────────────────────────────────────────────────────────────
def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Arial"],
        "font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.5,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
        "axes.edgecolor": "#9ca3af", "axes.linewidth": 0.8,
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK,
        "lines.linewidth": 1.8, "lines.markersize": 6,
        "grid.color": GRID, "grid.linewidth": 0.7,
        "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path",
        "figure.dpi": 120,
    })


def pin_pdf_epoch(name: str) -> None:
    """把 SOURCE_DATE_EPOCH 钉在该图输入的 mtime 上界 ⇒ PDF 可复现。"""
    if os.environ.get("SOURCE_DATE_EPOCH"):
        return
    ts = [os.path.getmtime(CUR / c) for c in FIG_SOURCES.get(name, ()) if (CUR / c).exists()]
    if ts:
        os.environ["SOURCE_DATE_EPOCH"] = str(int(max(ts)))


def png_probe(p: Path) -> dict:
    """标准库解析 PNG IHDR/pHYs（像素尺寸 + 声明 dpi）。"""
    raw = p.read_bytes()
    assert raw[:8] == b"\x89PNG\r\n\x1a\n", f"{p.name} 非 PNG"
    i, out = 8, {}
    while i < len(raw):
        ln = struct.unpack(">I", raw[i:i + 4])[0]
        typ = raw[i + 4:i + 8]
        data = raw[i + 8:i + 8 + ln]
        crc = struct.unpack(">I", raw[i + 8 + ln:i + 12 + ln])[0]
        assert crc == zlib.crc32(typ + data) & 0xFFFFFFFF, f"{p.name} chunk CRC 错"
        if typ == b"IHDR":
            out["w"], out["h"] = struct.unpack(">II", data[:8])
        if typ == b"pHYs":
            x, y, unit = struct.unpack(">IIB", data[:9])
            if unit == 1:
                out["dpi"] = x * 0.0254
        i += 12 + ln
    return out


TEXT_PROBLEMS: list[str] = []
_tick_cache: dict = {}


def _undrawn_text_ids(fig) -> set:
    """收集**不会被绘制**的文字的 id：

    ① `ax.axis("off")` 的坐标系里，文字 `get_visible()` 仍为 True，但整轴不画；
    ② 刻度值落在视窗外的刻度标签 —— matplotlib 保留该 artist 但不绘制。
       ⚠️ 这两种文字都**不是**通过 `t.axes` 能识别的（实测 tick 的 label1 与 findobj
       返回的对象不是同一个、且 `Text.axes` 为 None），所以必须从 axes 反查 id 集合。
    """
    undrawn: set = set()
    for ax in fig.axes:
        if not getattr(ax, "axison", True):
            # ⚠️ axis("off") 只关掉**轴**（脊/刻度/轴标），图内数据文字与标题照画 ⇒
            #    只把该轴的刻度标签与轴标题加入"不绘制"，不能整轴全排除（否则门禁对 fig6 全瞎）
            for _axis in (ax.xaxis, ax.yaxis):
                try:
                    _labs = list(_axis.get_ticklabels(which="both"))
                except TypeError:
                    _labs = list(_axis.get_ticklabels())
                _labs.append(_axis.get_label())
                for _lab in _labs:
                    undrawn.add(id(_lab))
            continue
        for nm, axis, lim in (("x", getattr(ax, "xaxis", None), ax.get_xlim()),
                              ("y", getattr(ax, "yaxis", None), ax.get_ylim())):
            if axis is None:
                continue
            lo, hi = sorted(lim)
            for tk in list(axis.get_major_ticks()) + list(axis.get_minor_ticks()):
                try:
                    inside = lo - 1e-9 <= float(tk.get_loc()) <= hi + 1e-9
                except Exception:                            # noqa: BLE001
                    inside = True
                if not inside:
                    for lab in (getattr(tk, "label1", None), getattr(tk, "label2", None)):
                        if lab is not None:
                            undrawn.add(id(lab))
    return undrawn


def text_overlap_report(fig, name: str) -> list[str]:
    """落盘**前**用文字包围盒查出"压字"（相交面积 > 较小者 10 %）与"越出画布"。

    三个必须避开的假阳性（实测踩过）：
      ① `Annotation.get_window_extent()` 含**引线箭头**（实测 87 px vs 文字 14 px）⇒ 取纯文字 bbox；
      ② `ax.axis("off")` 的坐标系里文字仍 `get_visible()==True`，但不会绘制；
      ③ 超出坐标轴视窗的刻度标签有 artist 但不绘制。
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    undrawn = _undrawn_text_ids(fig)
    seen, items = set(), []
    for t in fig.findobj(match=lambda o: hasattr(o, "get_text")):
        try:
            txt = t.get_text()
        except Exception:                                    # noqa: BLE001
            continue
        if not txt or not txt.strip() or not t.get_visible() or id(t) in undrawn:
            continue
        try:
            bb = Text.get_window_extent(t, renderer=r)
        except Exception:                                    # noqa: BLE001
            continue
        if bb.width <= 0 or bb.height <= 0:
            continue
        key = (txt, round(bb.x0, 1), round(bb.y0, 1), round(bb.x1, 1), round(bb.y1, 1))
        if key in seen:
            continue
        seen.add(key)
        items.append((txt, bb))

    out = []
    for txt, bb in items:
        if bb.x0 < -0.5 or bb.x1 > W + 0.5 or bb.y0 < -0.5 or bb.y1 > H + 0.5:
            out.append(f"{name}: 越出画布 x[{bb.x0:.0f},{bb.x1:.0f}] y[{bb.y0:.0f},{bb.y1:.0f}]"
                       f"（画布 {W:.0f}x{H:.0f}） {txt[:60]!r}")
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i][1], items[j][1]
            x0, x1 = max(a.x0, b.x0), min(a.x1, b.x1)
            y0, y1 = max(a.y0, b.y0), min(a.y1, b.y1)
            if x1 > x0 and y1 > y0:
                inter = (x1 - x0) * (y1 - y0)
                small = min(a.width * a.height, b.width * b.height)
                if small > 0 and inter / small > 0.10:
                    out.append(f"{name}: 压字 {inter/small*100:.0f}% {inter:.0f}px² "
                               f"{items[i][0][:42]!r} × {items[j][0][:42]!r}")
    return out


_CJK = re.compile(r"[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]")


def _cjk_text(fig) -> list[str]:
    """收集图内所有含 CJK 的文字（matplotlib 的 DejaVu Sans 无 CJK 字形 ⇒ 会渲染成豆腐块）。"""
    out: list[str] = []
    for t in fig.findobj(match=lambda o: hasattr(o, "get_text")):
        try:
            s = t.get_text()
        except Exception:                                    # noqa: BLE001
            continue
        if s and _CJK.search(s):
            out.append(s)
    return out


def save(fig, name: str, dpi: int, vector: tuple[str, ...]) -> list[Path]:
    """落盘：**固定整图宽度 + 显式尺寸**（`bbox_inches=None`）。

    为什么不自适应宽度（踩过，记录在此）：
    * `bbox_inches="tight"` 是按**文字**重新裁切的：图宽只有几英寸时，一段 7 pt 标注会把 bbox
      撑宽 1 英寸以上；文字是固定物理尺寸，于是「缩小 → bbox 更宽 → 再缩小」不收敛（实测震荡）。
    * 而且它会把图**纵向压扁**（实测 fig7 掉到 3.5 cm 高，标注互相压字）。
    ⇒ 统一：设计宽度 = 目标宽度（`TARGET_W_CM`），高度按设计比例，落盘用整图不裁切；
    左右留白由 `fit=` 显式设置，保证轴标/刻度不被切掉。
    """
    FIGDIR.mkdir(parents=True, exist_ok=True)
    bad = _cjk_text(fig)
    if bad:
        raise ValueError(f"{name}: 图内含 CJK 文字（会出豆腐块）：{bad[:3]}")

    w_design_cm, h_design_cm = fig.get_size_inches() * 2.54
    s = TARGET_W_CM / w_design_cm
    dpi_eff = int(round(dpi / s))
    fig.set_size_inches(TARGET_W_CM / 2.54, h_design_cm * s / 2.54)
    # ⚠️ 必须在改完尺寸之后**重跑一次布局**，否则坐标轴仍按旧尺寸排布 ⇒ 标题/标注互相压字
    try:
        fig.tight_layout(pad=0.5)
    except Exception:                                        # noqa: BLE001
        pass

    TEXT_PROBLEMS.extend(text_overlap_report(fig, name))

    outs: list[Path] = []
    prod = FIGDIR / f"{name}.png"
    prev = FIGDIR / f"{name}_preview.png"
    fig.savefig(prod, dpi=dpi_eff)
    fig.savefig(prev, dpi=PREVIEW_DPI)
    m = png_probe(prod)
    cm_w, cm_h = m["w"] / dpi_eff * 2.54, m["h"] / dpi_eff * 2.54
    outs += [prod, prev]
    for ext in vector:
        p = FIGDIR / f"{name}{VECTOR_SUFFIX[ext]}"
        fig.savefig(p, format=ext)
        outs.append(p)
    plt.close(fig)
    a, b = hashlib.md5(prod.read_bytes()).hexdigest(), hashlib.md5(prev.read_bytes()).hexdigest()
    assert a != b, f"{name}: 正式版与预览逐字节相同（dpi 未区分）"
    print(f"    ({name}) {cm_w:.2f} x {cm_h:.2f} cm @ {dpi_eff} dpi")
    return outs


def audit(files: list[Path]) -> int:
    """复核 dpi/宽度（图内 ASCII 由 save() 断言）。"""
    bad = 0
    print(f"\n{'file':<44}{'px':>12}{'dpi':>8}{'cm':>8}  md5")
    for p in sorted(files):
        if p.suffix.lower() != ".png":
            continue
        m = png_probe(p)
        dpi = m.get("dpi")
        cm = m["w"] / dpi * 2.54 if dpi else float("nan")
        tag = ""
        if "_preview" in p.name:
            pass
        else:
            if not dpi or dpi < MIN_PROD_DPI:
                tag = " ❌dpi"; bad += 1
            elif cm > MAX_W_CM:
                tag = " ❌宽"; bad += 1
        print(f"{p.name:<44}{m['w']:>6}x{m['h']:<5}{dpi or 0:>8.1f}{cm:>8.2f}  "
              f"{hashlib.md5(p.read_bytes()).hexdigest()[:8]}{tag}")
    print("✅ 正式版全部 dpi ≥ 300、声明宽度 ≤ 17.4 cm" if not bad else f"❌ {bad} 项不达标")
    if TEXT_PROBLEMS:
        print(f"\n❌ 图内文字问题 {len(TEXT_PROBLEMS)} 处（压字 / 越界，均会被读者看到）：")
        for m in TEXT_PROBLEMS:
            print("   -", m)
        bad += len(TEXT_PROBLEMS)
    else:
        print("✅ 图内文字：无压字、无越出画布")
    return 0 if not bad else 1


def note(ax, s: str, x=0.015, y=0.97, fontsize: float = 7.6, **kw):
    ax.text(x, y, s, transform=ax.transAxes, ha="left", va="top",
            fontsize=fontsize, color=MUTED, **kw)


# ────────────────────────────────────────────────────────────────────────────
# fig1 — 协议敏感性（EMO + 两臂 R²）
# ────────────────────────────────────────────────────────────────────────────
def fig1(dpi, vector):
    # s2c_emo.csv carries S1/S2/S2b/S2c/S4; S3 (leave-one-column) lives in s3_emo.csv.
    # The v3 script read only the former, so panel (a) silently plotted five protocols
    # while its caption promised six — fixed here by concatenating the S3 row.
    emo = pd.concat([pd.read_csv(CUR / "s2c_emo.csv"),
                     pd.read_csv(CUR / "s3_emo.csv")], ignore_index=True)
    lo = pd.read_csv(CUR / "leave_one_doi_summary.csv")
    sh = pd.read_csv(CUR / "s2c_emo_shuffle.csv")

    # ⚠️ v2 的隐藏 bug：CSV 的 split 是 "S1_random"/"S2_group"/… 这类长名，而这里写的是短代号
    #    ["S1","S2",…] ⇒ 一个都匹配不上，order 退化成 CSV 行序（S2c 打头）、x 轴直接印 5 个长名
    #    互相压字。v3 改成显式映射：排序回到"按协议编号"，轴标回到短代号（与表 3 / 图注一致）。
    SHORT = {"S1_random": "S1", "S2_group": "S2", "S3_leave_column": "S3",
             "S4_murcko": "S4", "S2b_leave_paper": "S2b", "S2c_leave_doi": "S2c"}
    e = emo.set_index("split")
    PREF = ["S1_random", "S2_group", "S3_leave_column", "S4_murcko",
            "S2b_leave_paper", "S2c_leave_doi"]
    order = [x for x in PREF if x in e.index] + [x for x in e.index if x not in PREF]
    ticklab = [SHORT.get(x, x) for x in order]
    vals = np.array([e.loc[x, "emo_acc"] * 100 for x in order])
    nrows = int(e["n"].iloc[0])
    base_glob = float(e["emo_base_global"].iloc[0]) * 100
    null = sh["emo_acc"].to_numpy() * 100
    # three permutations only: the meaningful band is their min–max range, not an
    # interpolated 2.5/97.5 % percentile (which is neither the min nor the max).
    lo95, hi95 = float(null.min()), float(null.max())
    folds = e["n_fold"].to_dict()

    fig, (a1, a2) = plt.subplots(
        1, 2, figsize=(15.4 / 2.54, 8.6 / 2.54), gridspec_kw=dict(width_ratios=[1.62, 1.0]))

    # ── (a) EMO 协议梯度 ───────────────────────────────────────────────
    x = np.arange(len(order))
    # 图例条目收短（v2 的 46 字条会把图例框撑到半个面板宽，压住数据点下的折数标注）
    a1.axhspan(lo95, hi95, color=BLUE, alpha=0.13, zorder=0,
               label=f"shuffled-label null band {lo95:.2f}–{hi95:.2f} %")
    a1.axhline(base_glob, color=MUTED, lw=1.1, ls="--", zorder=1,
               label=f"majority class {base_glob:.2f} %")
    a1.plot(x, vals, "-", color=INK, lw=1.4, zorder=2)
    a1.scatter(x, vals, s=64, color=BLUE, zorder=3, edgecolor="white", linewidth=0.8)
    for xi, v, s in zip(x, vals, order):
        a1.annotate(f"{v:.2f} %", (xi, v), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8.2, fontweight="bold", color=INK, zorder=4)
    a1.set_xticks(x)
    a1.set_xticklabels(ticklab, fontsize=9.0)
    a1.set_ylabel("elution-order accuracy (%)")
    a1.set_ylim(46.5, 80)
    a1.set_xlim(-0.55, len(order) - 0.45)
    a1.set_yticks([50, 55, 60, 65, 70, 75])
    a1.grid(axis="y", zorder=-1)
    a1.set_axisbelow(True)
    a1.set_title("(a)  Accuracy depends on the evaluation protocol", loc="left",
                 fontweight="bold", fontsize=9.8)
    # v2 把折数贴在坐标轴底部 ⇒ 与左下角图例压字；v3 改为贴在各自数据点正下方（白底垫字）
    for xi, vi, s in zip(x, vals, order):
        a1.annotate(f"{folds[s]}f", (xi, vi), textcoords="offset points", xytext=(0, -12),
                    ha="center", va="top", fontsize=6.8, color=MUTED, zorder=6,
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=0.5))
    # 锚点显式化：v2 用 vals[1]-vals[-1]（依赖 order 的下标），order 一改数字就变 ⇒ 按名字取
    _i_s1, _i_s4 = order.index("S1_random"), order.index("S4_murcko")
    a1.annotate(f"$-${vals[_i_s1]-vals[_i_s4]:.1f} pp", (x[_i_s1], vals[_i_s1]),
                textcoords="offset points", xytext=(6, -52), ha="left", va="top",
                fontsize=8.4, color=ORANGE, fontweight="bold")
    # v2/v3a 都放在轴内左下 ⇒ 图例框伸进数据点区，压住 S2b 下方的 "10f"（实测 81 %）
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.085), ncol=1, fontsize=7.2,
              handlelength=1.5, borderpad=0.25, labelspacing=0.25)
    note(a1, f"one model, one feature set (L0), {nrows:,} pairs", y=0.985)

    # ── (b) 两臂 R² ────────────────────────────────────────────────────
    arms = list(lo["arm"])
    r2 = lo["r2"].to_numpy() * 100
    nf = {a: int(r) for a, r in zip(lo["arm"], lo["n_fold"])}
    yb = np.arange(len(arms))
    cols = [BLUE, ORANGE]
    a2.axvline(0, color=INK, lw=1.0)
    a2.barh(yb, r2, height=0.46, color=cols[:len(arms)], zorder=3)
    for yi, v, a in zip(yb, r2, arms):
        a2.annotate(f"{v:.1f} %", (v, yi), textcoords="offset points",
                    xytext=(-6 if v < 0 else 6, 0), va="center",
                    ha="right" if v < 0 else "left", fontsize=8.4,
                    fontweight="bold", color=INK)
    a2.set_yticks(yb)
    a2.set_yticklabels([f"L0\n(28 descriptors)\n{nf[arms[0]]} folds",
                        f"L0 + chirality\ndescriptors\n{nf[arms[1]]} folds"], fontsize=7.6)
    a2.set_xlabel("R² under leave-one-paper-out (%)", fontsize=7.6)
    a2.set_xlim(min(r2) * 1.9, 6.0)
    a2.set_xticks([-20, -10, 0])          # v2 的刻度到 10（在视窗外）⇒ 无谓的越界判定来源
    a2.grid(axis="x", zorder=-1)
    a2.set_axisbelow(True)
    a2.set_title("(b)  Both arms negative", loc="left",
                 fontweight="bold", fontsize=9.0)
    # v2 把这条灰斜体注记放在轴内右下 (0.99, 0.02) ⇒ 压在 L0 的蓝条上；
    # 挪到左上 (0.02, 0.98) 又会压到 L0+chirality 的橙条（(b) 面板左右只剩约 95 px 空白，
    # 而文字宽 150 px）。该信息已完整写在论文图注里（"428 folds, 10,193 rows" +
    # "No fold-wise SD is shown ... the source table contains no SD column"）⇒ 图内删掉。

    fig.tight_layout(pad=0.6, w_pad=1.6)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.145, top=0.86, wspace=0.30)
    return save(fig, "fig1_protocol", dpi, vector)   # 高度见 figsize


# ────────────────────────────────────────────────────────────────────────────
# fig2 — 功效曲线（口径统一为单侧 95 % t 区间）
# ────────────────────────────────────────────────────────────────────────────
def fig2(dpi, vector):
    pc = pd.read_csv(CUR / "offset_power_curve.csv").sort_values("planted")
    po = pd.read_csv(CUR / "paper_offset_predictability.csv").set_index("arm")
    obs = float(po.loc["all", "r2"]) * 100 if "all" in po.index else float(po["r2"].min()) * 100
    draws = 5
    t95 = 2.131846786  # 单侧 95 %, df = draws-1 = 4

    x = pc["planted"].to_numpy() * 100
    m = pc["recovered_mean"].to_numpy() * 100
    s = pc["recovered_sd"].to_numpy() * 100

    fig, ax = plt.subplots(figsize=(15.5 / 2.54, 8.2 / 2.54))

    ax.axhline(obs, color=ORANGE, lw=1.8, zorder=3)
    ax.annotate(f"observed  {obs:.2f} %", (x.max() * 0.995, obs),
                textcoords="offset points", xytext=(-2, -11), ha="right",
                fontsize=8.6, color=ORANGE, fontweight="bold", zorder=5)

    ax.plot(x, x, ls=":", color=MUTED, lw=1.1, zorder=1,
            label="perfect recovery (y = x)")
    ax.errorbar(x, m, yerr=t95 * s, marker="o", color=BLUE, mfc=BLUE, mec="white",
                mew=0.9, ms=7, capsize=4, elinewidth=1.3, zorder=4,
                label=f"recovered (bar = one-sided 95 % t interval, {draws} draws)")
    for xi, mi, si in zip(x, m, s):
        ax.annotate(f"{mi:.2f} %", (xi, mi + t95 * si), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=8.0, fontweight="bold", color=INK)

    # 可排除 / 不可排除的分界
    excl = x[np.where((m - t95 * s) > obs)[0]]
    if len(excl):
        cut = excl.min()
        ax.axvspan(cut - 1.4, x.max() + 3.5, color=TEAL, alpha=0.10, zorder=0)
        # v2 放在 y=26.5（自顶往下）⇒ 把 20 % 档的数值标注 "15.58 %" 完全盖住（100 % 相交）
        # v3 移到左上那片真正的空白区（x>cut、y≈27–36 无任何数据点/标注）
        ax.text(cut + 1.0, 36.2,
                f"shares ≥ {cut:.0f} % of the\nbetween-paper variance\nare excluded at 95 %",
                fontsize=7.8, color="#0f766e", fontweight="bold", va="top", ha="left")
        ax.axvline(cut, color=TEAL, lw=1.2, ls="--", zorder=2)
    ax.annotate("5 % share: not excluded\n(interval reaches the observed value)",
                (5, m[0]), textcoords="offset points", xytext=(14, -30),
                fontsize=7.8, color=MUTED, ha="left",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9))

    ax.set_xlabel("planted composition-driven share of the between-paper variance (%)")
    ax.set_ylabel("share recovered by the same pipeline (%)")
    ax.set_xlim(0, 45)
    ax.set_ylim(-8, 44)
    ax.set_yticks([0, 10, 20, 30, 40])     # v2 自动刻度含 −10/50（都在视窗外）
    ax.grid(zorder=-1)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=8.0, handlelength=1.6, borderpad=0.3,
              bbox_to_anchor=(1.0, 0.15))   # v2 放在 0.06 ⇒ 与橙色 observed 标注压字 12 %
    ax.set_title("(2)  The between-paper offset is not composition-driven", loc="left",
                 fontweight="bold", fontsize=10)
    note(ax, "composition = molecular descriptors + i-PrOH fraction + flow + estimated t₀;\n"
             "stationary-phase identity is deliberately excluded", y=0.985)
    fig.tight_layout(pad=0.5)
    return save(fig, "fig2_power", dpi, vector)


# ────────────────────────────────────────────────────────────────────────────
# fig3 — 同实验室 vs 换实验室（雨云图 + 学习曲线）
# ────────────────────────────────────────────────────────────────────────────
def fig3(dpi, vector):
    w = pd.read_csv(CUR / "within_paper_split.csv")
    lc = pd.read_csv(CUR / "learning_curve_papers.csv")

    fig, (a1, a2) = plt.subplots(
        1, 2, figsize=(16.0 / 2.54, 8.5 / 2.54), gridspec_kw=dict(width_ratios=[1, 1.22]))

    # ── (a) 雨云图：within r 分布 ──────────────────────────────────────
    r = w["r"].to_numpy()
    med = float(np.median(r))
    pos = float((r > 0).mean() * 100)
    parts = a1.violinplot([r], positions=[0], widths=0.82, showextrema=False)
    for b in parts["bodies"]:
        b.set_facecolor(BLUE); b.set_alpha(0.22); b.set_edgecolor(BLUE); b.set_linewidth(0.9)
    jit = np.random.default_rng(7).normal(0, 0.055, len(r))
    a1.scatter(np.full(len(r), 0) + jit, r, s=9, color=BLUE, alpha=0.42,
               edgecolor="none", zorder=3)
    q1, q3 = np.percentile(r, [25, 75])
    a1.add_patch(Rectangle((-0.10, q1), 0.20, q3 - q1, facecolor=INK, alpha=0.85, zorder=4))
    a1.plot([-0.18, 0.18], [med, med], color="white", lw=2.0, zorder=5)
    a1.axhline(0, color=MUTED, lw=1.0, ls="--", zorder=2)
    a1.annotate(f"median r = +{med:.3f}\n{pos:.0f} % of papers positive",
                (0.30, med), fontsize=8.4, color=INK, fontweight="bold", va="center")
    a1.set_xlim(-0.55, 1.05)
    a1.set_ylim(-1.02, 1.02)
    a1.set_yticks([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])   # 去掉视窗外的 ±1.25
    a1.set_xticks([])
    a1.set_ylabel("per-paper Pearson r (within lab)", fontsize=9.0)
    a1.grid(axis="y", zorder=-1)
    a1.set_axisbelow(True)
    a1.set_title("(a)  Within-study prediction", loc="left",
                 fontweight="bold", fontsize=9.2)
    note(a1, f"{len(w)} papers with ≥25 rows\nmolecules split 50/50, 8 repeats", y=0.985)

    # ── (b) 学习曲线 ──────────────────────────────────────────────────
    k = lc["k"].to_numpy()
    rm = lc["r_med"].to_numpy()
    a2.axhline(med, color=ORANGE, lw=1.4, ls="--", zorder=2,
               label=f"within-study median (+{med:.3f})")
    a2.axhline(0, color=MUTED, lw=1.0, zorder=1)
    a2.plot(k, rm, "-o", color=BLUE, mfc=BLUE, mec="white", mew=0.9, ms=6.5,
            zorder=4, label="across-study median per-paper r")
    a2.fill_between(k, 0, rm, color=BLUE, alpha=0.10, zorder=0)
    for ki, ri in zip(k, rm):
        if ki in (1, 250):
            a2.annotate(f"{ri:+.3f}", (ki, ri), textcoords="offset points",
                        xytext=(0, -14 if ki == 1 else 8), ha="center",
                        fontsize=7.8, color=INK)
    a2.set_xscale("log")
    a2.set_xticks(k)
    # 8 个对数刻度挤在窄面板里会互相压字（实测最高 84 %）⇒ 旋转 40° 并缩小
    a2.set_xticklabels([str(v) for v in k], fontsize=7.2, rotation=40, ha="right")
    a2.set_xlabel("number of training papers k")
    a2.set_ylabel("median per-paper r on unseen papers")
    a2.set_ylim(-0.16, 0.30)
    a2.grid(zorder=-1)
    a2.set_axisbelow(True)
    a2.legend(loc="upper right", fontsize=8.0)
    a2.set_title("(b)  Across-lab transfer", loc="left",
                 fontweight="bold", fontsize=9.2)
    # v2：31 字 + 锚在 y=0.245 ⇒ 右端被画布切、又与 upper-right 图例条目相交 39 %
    a2.text(1.35, 0.185, "flat over a 250× data increase",
            fontsize=8.0, color=ORANGE, fontweight="bold")
    # v2 单行 80 字符 ⇒ 右侧被画布切掉；v3 折成两行并放到面板左下
    note(a2, "per-paper r (not R²): at k = 1\nthe scale is uncalibrated (R² reaches −1679)",
         x=0.015, y=0.10, fontsize=7.0)

    fig.tight_layout(pad=0.6)
    return save(fig, "fig3_lab_transfer", dpi, vector)


# ────────────────────────────────────────────────────────────────────────────
# fig4 — 条件相关热图 + 森林图（全篇首个连续色标）
# ────────────────────────────────────────────────────────────────────────────
def fig4(dpi, vector):
    cp = pd.read_csv(CUR / "condition_pairwise.csv")
    conds = sorted(set(cp["a"]) | set(cp["b"]))
    A = np.full((len(conds), len(conds)), np.nan)
    idx = {c: i for i, c in enumerate(conds)}
    for _, row in cp.iterrows():
        A[idx[row["a"]], idx[row["b"]]] = float(row["r"])
    np.fill_diagonal(A, 1.0)
    # 对称补全（CSV 只给每个无序对一行）
    for i, ci in enumerate(conds):
        for cj in conds[i + 1:]:
            a, b = A[idx[ci], idx[cj]], A[idx[cj], idx[ci]]
            if np.isnan(a) and not np.isnan(b):
                A[idx[ci], idx[cj]] = b
            elif np.isnan(b) and not np.isnan(a):
                A[idx[cj], idx[ci]] = a
    n_missing = int(np.isnan(A).sum())
    print(f"    (fig4) 条件对矩阵：{len(conds)}×{len(conds)}，缺失 {n_missing} 格（含对角线已填 1）")
    order = sorted(conds, key=lambda s: (s.split(" | ")[0], s.split(" | ")[1]))
    M = pd.DataFrame(A, index=conds, columns=conds).loc[order, order]
    Mv = np.array(M.values, dtype=float)

    fig, (a1, a2) = plt.subplots(
        1, 2, figsize=(16.5 / 2.54, 9.4 / 2.54), gridspec_kw=dict(width_ratios=[1.06, 1.0]))

    # ── (a) 热图 ──────────────────────────────────────────────────────
    vmax = float(np.nanmax(np.abs(Mv)))
    im = a1.imshow(Mv, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    # 短标签：只留柱名，相态用 a/b 后缀（" | acidic" → "a"，" | basic" → "b"），避免标签互相压字
    def shortlab(c: str) -> str:
        col, ph = c.split(" | ")
        return f"{col}{'a' if ph == 'acidic' else 'b'}"

    short = [shortlab(c) for c in order]
    a1.set_xticks(range(len(order))); a1.set_xticklabels(short, fontsize=7.0, rotation=90)
    a1.set_yticks(range(len(order))); a1.set_yticklabels(short, fontsize=7.0)
    for i in range(len(order)):
        for j in range(len(order)):
            v = Mv[i, j]
            if np.isnan(v):
                a1.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor="#f3f4f6", zorder=3))
                continue
            # v2 在 144 个格里印 5.2 pt 的 r×100（还会出现 "-0"）—— 实测基本读不出来。
            # v3 去掉格内数字：12×12 相关矩阵的信息由色标承担（图注里也没承诺格内数字）。
            _ = v
    for k in range(1, len(order)):
        if order[k].split(" | ")[0] != order[k - 1].split(" | ")[0]:
            a1.axhline(k - .5, color=INK, lw=1.4)
            a1.axvline(k - .5, color=INK, lw=1.4)
    cb = fig.colorbar(im, ax=a1, fraction=0.046, pad=0.03)
    cb.set_label("pairwise Pearson r", fontsize=8.0)
    cb.ax.tick_params(labelsize=7.2)
    a1.set_title("(a)  Condition-to-condition correlation", loc="left",
                 fontweight="bold", fontsize=9.6, pad=11)
    a1.text(0.0, 1.010, "cellulose: LC, OD-RH   |   amylose: LA",
            transform=a1.transAxes, fontsize=6.8, color=MUTED, va="bottom")
    a1.text(0.0, -0.34, "one observation per (compound, condition)\n"
                        "column labels: <column><a = acidic | b = basic>",
            transform=a1.transAxes, fontsize=6.8, color=MUTED, va="top")

    # ── (b) 森林图：三类条件对（**真·化合物级聚类 bootstrap**，见 scripts/36）──
    bst = pd.read_csv(CUR / "condition_pair_bootstrap.csv").set_index("group")
    groups = [("mobile phase only\n(same column)", "mobile phase only (same column)",
               cp[cp["same_col"] & ~cp["same_ph"]], TEAL),
              ("selector only\n(same mobile phase)", "selector only (same mobile phase)",
               cp[~cp["same_col"] & cp["same_ph"]], ORANGE),
              ("both changed", "both changed", cp[~cp["same_col"] & ~cp["same_ph"]], PURPLE)]
    rng = np.random.default_rng(11)
    ymax = len(groups) - 0.45
    for yi, (lab, key, sub, col) in enumerate(groups):
        rr = sub["r"].to_numpy()
        row = bst.loc[key]
        med_g, glo, ghi = float(row["median_r"]), float(row["ci_lo"]), float(row["ci_hi"])
        a2.plot([glo, ghi], [yi, yi], color=col, lw=2.4, solid_capstyle="round", zorder=3)
        a2.scatter(rr, np.full(len(rr), yi) + rng.normal(0, 0.045, len(rr)),
                   s=13, color=col, alpha=0.35, edgecolor="none", zorder=2)
        a2.scatter([med_g], [yi], s=64, color=col, edgecolor="white", lw=1.0, zorder=5)
        ytxt = yi + 0.34 if yi < len(groups) - 1 else yi - 0.46
        a2.text(0.99, (ytxt - (-0.55)) / (ymax - (-0.55)),
                f"r = {med_g:+.3f}   [{glo:+.3f}, {ghi:+.3f}]   n = {len(rr)}",
                transform=a2.transAxes, ha="right", va="center", fontsize=7.4, color=INK)
    a2.axvline(0, color=MUTED, lw=1.0, ls="--")
    a2.set_yticks(range(len(groups)))
    a2.set_yticklabels([g[0] for g in groups], fontsize=8.2)
    a2.set_ylim(-0.55, ymax)
    a2.set_xlim(-0.45, 1.15)
    a2.set_xticks([-0.5, 0.0, 0.5, 1.0])   # 去掉视窗外的 −1.0 / 1.5
    a2.set_xlabel("pairwise Pearson r", fontsize=8.6)
    a2.grid(axis="x", zorder=-1)
    a2.set_axisbelow(True)
    a2.set_title("(b)  What changes the ranking", loc="left", fontweight="bold",
                 fontsize=7.4, pad=11)
    # v2 这条说明从 (b) 面板左缘向右铺 ⇒ 右端被画布切掉；v3 改成右对齐（向左生长，落在空白处）
    a2.text(0.985, 1.010, "compound-clustered bootstrap, 2000 resamples; bars = 95 % CI",
            transform=a2.transAxes, fontsize=6.6, color=MUTED, va="bottom", ha="right")
    # v2/v3a 放在轴内右上 ⇒ 正好压在 "both changed" 那行的散点上；v3b 移到坐标区下方
    a2.text(1.0, -0.175, "selector-only and both-changed overlap:\n"
                        "selector contribution not resolvable",
            transform=a2.transAxes, fontsize=6.8, color=ORANGE, fontweight="bold",
            va="top", ha="right")
    fig.tight_layout(pad=0.6)
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.155, top=0.845, wspace=0.34)
    return save(fig, "fig4_condition_grid", dpi, vector)


# ────────────────────────────────────────────────────────────────────────────
# fig5 — 方差分解（CI 带 + 双口径 + 敏感性）
# ────────────────────────────────────────────────────────────────────────────
def fig5(dpi, vector):
    ic = pd.read_csv(CUR / "interaction_ceiling.csv").iloc[0]

    def g(k):
        return float(ic[k])

    fig, ax = plt.subplots(figsize=(16.0 / 2.54, 7.5 / 2.54))
    segs = [("molecule main effect", g("eta2_mol"), BLUE, ""),
            ("condition main\neffect", g("eta2_cond"), TEAL, "//"),
            ("molecule × condition interaction\n+ measurement noise", g("eta2_resid"),
             ORANGE, "\\\\")]
    left = 0.0
    ybar = 0.0
    h = 0.34
    for lab, v, col, hat in segs:
        ax.barh(ybar, v * 100, left=left * 100, height=h, color=col, hatch=hat,
                edgecolor="white", linewidth=0.8, zorder=3)
        xc = (left + v / 2) * 100
        if v > 0.05:
            # v2 的纹样（///、\\）会横穿白字，把 "76.95 %" 压得读不出来 ⇒ 加半透明白底
            ax.text(xc, ybar, f"{v*100:.2f} %", ha="center", va="center",
                    color="white", fontsize=8.6, fontweight="bold", zorder=4,
                    bbox=dict(facecolor=INK, alpha=0.55, edgecolor="none", pad=1.2))
        left += v
    # 小段引出标注
    ax.annotate(f"{g('eta2_cond')*100:.2f} %", (g("eta2_mol") * 100 + 0.4, ybar),
                textcoords="offset points", xytext=(0, -22), ha="left", fontsize=8.0,
                color="#0f766e", fontweight="bold",
                arrowprops=dict(arrowstyle="-", color="#0f766e", lw=0.9))

    # CI 带：分子-only 与 分子+条件 双口径（标注锚在轴内，避免撑宽 tight bbox）
    lo_m, hi_m = g("eta2_model_lo") * 100, g("eta2_model_hi") * 100
    alt = pd.read_csv(CUR / "review_round2_stats.csv")
    mo = alt[alt["analysis"] == "boot_molecule_only"]
    mj = alt[alt["analysis"] == "boot_molecule_and_condition"]
    if len(mo):
        # ⚠️ interaction_ceiling.csv 的 CI 是旧口径（分子只、但点估计百分位偏移）；
        #    正文/摘要用 scripts/33 重算的区间 ⇒ 图内必须读同一来源，否则图注与图内会打架。
        lo_m = float(mo["eta2_model_lo"].iloc[0]) * 100
        hi_m = float(mo["eta2_model_hi"].iloc[0]) * 100
    if len(mj):
        lo_j = float(mj["eta2_model_lo"].iloc[0]) * 100
        hi_j = float(mj["eta2_model_hi"].iloc[0]) * 100
    else:                                    # 兜底：只在本轮 CSV 缺失时用的旧值
        lo_j, hi_j = 18.6, 47.9
    yc = 0.42
    ax.plot([lo_m, hi_m], [yc, yc], color=BLUE, lw=3.2, solid_capstyle="butt", zorder=3)
    ax.scatter([g("eta2_model") * 100], [yc], color=INK, s=46, zorder=5)
    ax.text(0.985, 0.965, f"95 % bootstrap, molecules only:  {lo_m:.1f}–{hi_m:.1f} %",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.8, color=BLUE)
    ax.plot([lo_j, hi_j], [yc - 0.20, yc - 0.20], color=ORANGE, lw=3.2, alpha=0.9,
            solid_capstyle="butt", zorder=3)
    ax.text(0.985, 0.965 - 0.095, f"molecules + conditions jointly:  {lo_j:.1f}–{hi_j:.1f} %",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.8, color=ORANGE)
    ax.text(0.985, 0.965 - 0.19, "both schemes stay below 50 % of the variance",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.4, color=MUTED,
            style="italic")
    ax.axvline(50, color=MUTED, lw=1.0, ls=":")

    # 三条敏感性：一行菱形，标注在外侧右端（不再逐条居中，避免互相压字）
    sens = [("legacy 549-row coding", g("_senslegacy_549_rows_eta2_model") * 100),
            ("48-level coding", g("_sens48_549_rows_eta2_model") * 100),
            ("balanced 30-molecule sub-grid", g("_sensbal_360_rows_eta2_model") * 100)]
    ax.text(0.015, 0.905, "sensitivity codings (diamonds):\n"
                          f"  legacy 549-row coding  {g('_senslegacy_549_rows_eta2_model')*100:.2f} %\n"
                          f"  48-level coding  {g('_sens48_549_rows_eta2_model')*100:.2f} %\n"
                          f"  balanced 30-molecule sub-grid  {g('_sensbal_360_rows_eta2_model')*100:.2f} %",
            transform=ax.transAxes, fontsize=6.8, color=PURPLE, va="top", ha="left")
    for lab, v in sens:
        ax.scatter([v], [-0.44], marker="d", color=PURPLE, s=34, zorder=5,
                   edgecolor="white", linewidth=0.6)

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.95, 1.05)
    ax.set_yticks([])
    ax.set_xlabel("share of the variance in y = ln α (%)")
    ax.grid(axis="x", zorder=-1)
    ax.set_axisbelow(True)
    ax.set_title("(5)  What an additive (main-effects) description of the label can reach",
                 loc="left", fontweight="bold", fontsize=10)
    # v2 这条 62 字符 ⇒ 右端与右上角的 CI 说明相交（实测 27.5 %）；v3 收短到不再相碰，
    # 完整口径（"43 resolvable molecules × 12 conditions / 494 of 516 cells"）在论文图注里。
    note(ax, f"DG25: {int(g('n_mol'))} molecules x {int(g('n_cond'))} conditions; "
             f"{int(g('n'))}/{int(g('n_theory'))} cells "
             f"({g('completeness')*100:.1f} %)", y=0.995)
    fig.tight_layout(pad=0.5)
    return save(fig, "fig5_variance", dpi, vector)


# ────────────────────────────────────────────────────────────────────────────
# fig6 — 新增：清洗流失漏斗
# ────────────────────────────────────────────────────────────────────────────
def fig6(dpi, vector):
    # ── 2026-09-21 修补（复核清单 D-17 邻接项）────────────────────────────────
    # 原来这 6 个数字是**写死的常量**，而 FIG_SOURCES 却声明了两个输入文件 ——
    # 声明与代码对不上，图 6 因而完全不可机械核对。现在改为真读 curation_stats.json。
    # 起始行 25,867 = n_records(25,866) + 第 2 步丢掉的 1 行 RT ≤ 0（D-13 那处差 1 的来源）。
    N_RT_LE0 = 1
    st = json.loads((CUR / "curation_stats.json").read_text(encoding="utf-8"))["stats"]
    steps = [
        ("CMRT source records", st["n_records"] + N_RT_LE0, "as distributed"),
        ("RT ≤ 0 removed", st["n_records"], f"{N_RT_LE0} unphysical record"),
        ("condition units formed", st["n_units"],
         "(connectivity, column, i-PrOH, flow, enantiomer)"),
        ("candidate pairs", st["n_candidate"], "same connectivity block"),
        ("★ verified pairs", st["n_enantiomer"],
         f"{st['n_diastereomer']} ({st['n_diastereomer'] / st['n_candidate'] * 100:.1f} %)"
         " were diastereomers"),
        ("− k ≤ 0 removed", st["n_pairs_total"],
         f"{st['n_enantiomer'] - st['n_pairs_total']} pairs below estimated dead time"),
    ]
    # 与论文正文 / 其它脚本对过的期望值：任一不符就当场失败，而不是悄悄画出一张对不上的图
    _expect = [25867, 25866, 24624, 11179, 10582, 10525]
    _got = [v for _, v, _ in steps]
    assert _got == _expect, f"fig6 漏斗数字与论文不符：{_got} != {_expect}"
    assert st["n_enantiomer"] - st["n_pairs_total"] == 57
    fig, ax = plt.subplots(figsize=(16.0 / 2.54, 9.5 / 2.54))
    n = len(steps)
    top = steps[0][1]
    cmap = [BLUE, BLUE, TEAL, TEAL, ORANGE, ORANGE]
    for i, (lab, v, sub) in enumerate(steps):
        y = n - 1 - i
        w = v / top
        ax.add_patch(FancyBboxPatch((0.5 - w / 2, y - 0.35), w, 0.70,
                                    boxstyle="round,pad=0.006,rounding_size=0.02",
                                    facecolor=cmap[i], alpha=0.92, edgecolor="white",
                                    linewidth=1.0, zorder=3))
        ax.text(0.5, y + 0.115, f"{lab}", ha="center", va="center", color="white",
                fontsize=8.0, fontweight="bold", zorder=4)
        ax.text(0.5, y - 0.145, f"{v:,}", ha="center", va="center", color="white",
                fontsize=7.4, zorder=4)
        ax.text(0.5 + w / 2 + 0.02, y, sub, ha="left", va="center", fontsize=7.6,
                color=MUTED, zorder=4)
        if i > 0:
            delta = steps[i - 1][1] - v
            if delta:
                ax.annotate(f"−{delta:,}", (0.5, y + 0.42), ha="center", va="center",
                            fontsize=7.4, color=ORANGE, fontweight="bold", zorder=5)
    ax.set_xlim(-0.02, 1.32)
    ax.set_ylim(-0.75, n - 0.35)
    ax.axis("off")
    ax.set_title("(6)  From 25,867 source records to 10,525 verified enantiomer pairs",
                 loc="left", fontweight="bold", fontsize=10)
    ax.text(0.0, -0.62, "Steps are ordered as in Section 2.1; the 597 diastereomer rejections are the "
                        "dominant loss and are irreversible\n(they are not recovered later). "
                        "Step 3 (DOI repair, 1,307 records) changes provenance, not counts.",
            fontsize=7.4, color=MUTED, va="top")
    fig.tight_layout(pad=0.5)
    return save(fig, "fig6_curation", dpi, vector)


# ────────────────────────────────────────────────────────────────────────────
# fig7 — 新增：外部验证森林图（含打乱对照）
# ────────────────────────────────────────────────────────────────────────────
def fig7(dpi, vector):
    ea = pd.read_csv(CUR / "external_all_columns.csv")
    famcol = {"cellulose": BLUE, "amylose": TEAL}

    plain = ea[~ea["arm"].str.contains(r"\(aligned\)", regex=True)].reset_index(drop=True)
    aligned = ea[ea["arm"].str.contains(r"\(aligned\)", regex=True)].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(16.0 / 2.54, 12.8 / 2.54))
    # 每根柱占两个槽位：真实值（上）+ 打乱对照（下）
    for i, r in plain.iterrows():
        y = -(2 * i)
        c = famcol.get(r["family"], PURPLE)
        ax.scatter([float(r["r2"]) * 100], [y], marker="o", s=62, color=c,
                   edgecolor="white", linewidth=0.9, zorder=5)
        ax.annotate(f"{float(r['r2'])*100:.2f} %  (n = {int(r['n'])})",
                    (float(r["r2"]) * 100, y), textcoords="offset points",
                    xytext=(9, 0), ha="left", va="center", fontsize=8.0, color=INK,
                    zorder=7,
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=0.8))

    # 打乱对照：`external_all_columns.csv` 只给真实臂；对照值在 negative_controls.csv 里按**组**给
    # （OD-RH 对齐臂 5 次重复；六柱臂每柱 1 次 ⇒ 报的是柱间离散），所以图内不画逐柱三角，
    # 改为在图注与图内文字里给出。
    nc = pd.read_csv(CUR / "negative_controls.csv")
    ctrl_odrh = nc[nc["control"].str.contains("OD-RH", na=False)].iloc[0]
    ctrl_6 = nc[nc["control"].str.contains("DG25|6|六", na=False, regex=True)].iloc[0] \
        if len(nc) > 1 else ctrl_odrh

    # 对齐臂（同选择剂、更小训练集）单独一组放在最下方
    y0 = -(2 * len(plain)) - 1.9
    for i, r in aligned.iterrows():
        y = y0 - 0.62 * i
        ax.scatter([float(r["r2"]) * 100], [y], marker="*", s=105, color=ORANGE,
                   edgecolor="white", linewidth=0.5, zorder=6)
        ax.annotate(f"{float(r['r2'])*100:.2f} %  (n = {int(r['n'])})",
                    (float(r["r2"]) * 100, y), textcoords="offset points",
                    xytext=(9, 0), ha="left", va="center", fontsize=7.8, color=ORANGE,
                    zorder=7,
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=0.8))

    ax.axvline(0, color=INK, lw=1.1)
    labs, ticks = [], []
    for i, r in plain.iterrows():
        labs.append(str(r["column"]))
        ticks.append(-(2 * i))
    # v2：对齐臂刻度间距只有 0.62 单位 ⇒ 4 个 "xxx [aligned]" 标签互相压字 55.6 %；
    #     表头 "aligned (CMRT Cellulose-a)" 还挤在块中间。v3：间距 1.05，表头独占一行放在块上方。
    labs.append("aligned (CMRT Cellulose-a)")
    ticks.append(y0 + 1.05)
    for i, r in aligned.iterrows():
        labs.append(str(r["column"]))
        ticks.append(y0 - 1.05 * i)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labs, fontsize=8.4)
    ax.set_ylim(y0 - 1.05 * len(aligned) - 0.4, 0.9)
    ax.set_xlim(-205, 75)
    ax.set_xlabel("R² when CMRT is used to predict a De Gauquier column (%)")
    ax.grid(axis="x", zorder=-1)
    ax.set_axisbelow(True)
    # v2：标题 + 两行长说明 + "R² < 0" 全堆在坐标轴上方（y=1.075/1.020/0.997）⇒
    #     三行互相 100 %/45 %/39 % 相交，且右端全被画布切掉。v3：标题收短，说明移到轴**下方**
    #     折成 4 行（那里是整幅 16 cm 宽的自由区），并把关键词放进标题行。
    ax.set_title("(7)  External validation: every column is negative",
                 loc="left", fontweight="bold", fontsize=9.2)
    _notes = [
        "circles = all-CMRT arms (blue = cellulose, teal = amylose); no column beats a shuffle",
        "stars = selector-aligned arms, which use only CMRT Cellulose-a as training data",
        "R² < 0 means worse than predicting the mean",
        "shuffled-target control, OD-RH aligned "
        f"{ctrl_odrh['mean']:.3f} ± {ctrl_odrh['sd']:.3f} (5 repeats);",
        f"six-column control {ctrl_6['mean']:.3f} ± {ctrl_6['sd']:.3f} (one shuffle per column,",
        "so this is a between-column spread rather than a permutation SD)",
    ]
    # 起始位置放在 xlabel 之下（-0.075 会与 xlabel 相交 69 %）
    for _i, _ln in enumerate(_notes):
        ax.text(0.0, -0.125 - 0.042 * _i, _ln, transform=ax.transAxes,
                fontsize=6.6, color=MUTED, va="top")
    fig.tight_layout(pad=0.5)
    return save(fig, "fig7_external", dpi, vector)


FIGS = (fig1, fig2, fig3, fig4, fig5, fig6, fig7)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpi", type=int, default=PROD_DPI)
    ap.add_argument("--vector", default="pdf",
                    help="逗号分隔 pdf/eps/svg/none")
    args = ap.parse_args()
    vector = tuple(v.strip() for v in args.vector.split(",") if v.strip())
    if vector == ("none",):
        vector = ()
    for v in vector:
        if v not in VECTOR_SUFFIX:
            print(f"❌ 未知矢量格式 {v}"); return 2
    if args.dpi < MIN_PROD_DPI:
        print(f"❌ --dpi {args.dpi} < {MIN_PROD_DPI}，拒绝落盘"); return 2

    miss = validate_sources()
    if miss:
        print("❌ FIG_SOURCES 声明了 data/curated/ 里不存在的输入：")
        for m in miss:
            print(f"   {m}")
        return 2

    style()
    files: list[Path] = []
    fails = []
    for f in FIGS:
        try:
            pin_pdf_epoch(f.__name__)
            files += f(args.dpi, vector)
            print(f"  ✓ {f.__name__}")
        except Exception as ex:                                   # noqa: BLE001
            fails.append((f.__name__, repr(ex)))
            print(f"  ✗ {f.__name__}: {ex!r}")
    if fails:
        print("\n❌ 失败：", fails)
        return 1
    rc = audit(files)
    print(f"\n产出 {len(files)} 个文件 → {FIGDIR}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
