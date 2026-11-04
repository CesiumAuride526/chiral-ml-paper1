# -*- coding: utf-8 -*-
"""
31_run_all_gates.py — **一条命令跑完整门禁链**
================================================================================
## 为什么（2026-09-18 第八场）

本项目的门禁到第七场已经有三道，但**没有任何一处把它们串起来**：

| 门禁 | 守护什么 | 脚本 |
|---|---|---|
| ① 清洗门禁 | 主表 67 项 | `scripts/_qa_01_curate.py` |
| ② 论文数字门禁 | 论文引用的数与 `data/curated/*.csv` 一致（95 项断言，2026-10-12 实测） | `scripts/26_paper_number_gate.py` |
| ③ 全量小数可追溯 | 正文里**每一个小数**都能回到某个 CSV（默认已对准当前正文 `_rev.md`） | `scripts/26_paper_number_gate.py --trace` |

⇒ 本脚本按顺序跑一遍，**任何一项非零退出即整体非零退出**，并把子进程完整输出留档。

## 用法
```bash
PYTHONIOENCODING=utf-8 python scripts/31_run_all_gates.py          # 完整（含 01 重建，约 165 s）
PYTHONIOENCODING=utf-8 python scripts/31_run_all_gates.py --fast   # 跳过 01（开发迭代用，约 10 s）
PYTHONIOENCODING=utf-8 python scripts/31_run_all_gates.py --only numbers
```

## 为什么 `--fast` 合理
`01_curate_cmrt.py` 已在第八场被实测为**位级可复现**（`data/curated/` 52/52 文件 SHA256 不变，
见 `notes/第八场_代码审计.md`）。所以日常迭代只需 ② ③；**投稿前、改动 01 后必须跑完整链路**。

⚠️ 子进程环境固定 `OMP_NUM_THREADS=1`：本机沙箱下 sklearn/joblib 的线程池会去开命名管道，
被拒绝后抛 `PermissionError`（第八场实测）。
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRATCH = ROOT / "scratch"

# key, 说明, 命令, 是否耗时, 关键行解析（正则 → 显示名）
GATES = [
    dict(
        key="curate",
        label="① 01 清洗（从原始 CMRT 重建主表）",
        cmd=[sys.executable, "scripts/01_curate_cmrt.py"],
        slow=True,
        metrics=[("落盘/收尾", r"(写入|落盘|完成|rows|行).{0,60}")],
    ),
    dict(
        key="qa",
        label="② 数据门禁 _qa_01_curate（主表结构与划分完整性）",
        cmd=[sys.executable, "scripts/_qa_01_curate.py"],
        slow=False,
        metrics=[
            ("通过数", r"✓ 全部\s*(\d+)\s*项通过"),
            ("失败数", r"✗\s*(\d+)\s*项失败"),
        ],
    ),
    dict(
        key="numbers",
        label="③ 论文数字门禁（逐条断言：手稿数 = CSV 数）",
        cmd=[sys.executable, "scripts/26_paper_number_gate.py"],
        slow=False,
        metrics=[
            ("通过", r"通过\s*\*\*(\d+)\*\*\s*项"),
            ("失败", r"失败\s*\*\*(\d+)\*\*\s*项"),
        ],
    ),
    dict(
        key="trace",
        label="④ 可追溯性（小数 + 整数 + **判据自检**：随机数会不会也被判可追溯）",
        cmd=[sys.executable, "scripts/26_paper_number_gate.py", "--trace"],
        slow=False,
        metrics=[
            ("正文小数", r"正文小数\s*\*\*(\d+)\*\*"),
            ("不可追溯", r"不可追溯\*\*：\s*(\d+)\s*个"),
            ("判据误判率", r"误判为可追溯\s+([\d.]+)%"),
        ],
    ),
]


def run_one(g: dict, log_fh) -> dict:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")

    t0 = time.time()
    print(f"\n{'=' * 78}\n▶ {g['label']}\n  $ {' '.join(g['cmd'])}\n{'=' * 78}", flush=True)
    try:
        p = subprocess.run(
            g["cmd"], cwd=str(ROOT), env=env, capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        out, rc = (p.stdout or "") + (p.stderr or ""), p.returncode
    except Exception as e:  # 启动失败也要如实呈现
        out, rc = f"[启动失败] {type(e).__name__}: {e}\n", 127
    dt = time.time() - t0

    log_fh.write(f"\n\n{'#' * 78}\n# {g['label']}\n# cmd: {' '.join(g['cmd'])}\n"
                 f"# exit={rc}  {dt:.1f}s\n{'#' * 78}\n{out}")

    found = {}
    for name, pat in g["metrics"]:
        m = re.search(pat, out)
        if m:
            found[name] = m.group(1)
    tail = [ln.strip() for ln in out.strip().splitlines() if ln.strip()][-6:]
    print("\n".join(f"   {ln}" for ln in tail), flush=True)
    print(f"   ── exit={rc}  耗时 {dt:.1f}s" + (f"  {found}" if found else ""), flush=True)

    return dict(key=g["key"], label=g["label"], rc=rc, sec=dt, metrics=found)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="跳过 01 重建主表（约 159 s）")
    ap.add_argument("--only", default=None,
                    help="只跑一项：" + " / ".join(g["key"] for g in GATES))
    args = ap.parse_args()

    todo = GATES
    if args.only:
        todo = [g for g in GATES if g["key"] == args.only]
        if not todo:
            print(f"❌ 未知 --only {args.only!r}；可选：{[g['key'] for g in GATES]}")
            return 2
    elif args.fast:
        todo = [g for g in GATES if not g["slow"]]

    SCRATCH.mkdir(exist_ok=True)
    log = SCRATCH / f"gates_run_{datetime.now():%Y%m%d_%H%M%S}.txt"

    print("=" * 78)
    print("chiral_ml 全链路门禁")
    print(f"  步骤：{' → '.join(g['key'] for g in todo)}")
    print(f"  留档：{log.relative_to(ROOT)}")
    print("=" * 78)

    results = []
    with log.open("w", encoding="utf-8") as fh:
        fh.write(f"chiral_ml 全链路门禁  {datetime.now():%Y-%m-%d %H:%M:%S}\n"
                 f"步骤: {[g['key'] for g in todo]}\n")
        for g in todo:
            results.append(run_one(g, fh))

        # ── 汇总
        print("\n" + "=" * 78)
        print("汇总")
        print("=" * 78)
        print(f"{'步骤':34s} {'退出码':>6s} {'耗时(s)':>9s}  指标")
        print("-" * 78)
        for r in results:
            met = "  ".join(f"{k}={v}" for k, v in r["metrics"].items())
            print(f"{r['label'][:33]:34s} {r['rc']:>6d} {r['sec']:>9.1f}  {met}")
        bad = [r for r in results if r["rc"] != 0]
        print("-" * 78)
        if bad:
            print(f"❌ {len(bad)}/{len(results)} 项失败：{[r['key'] for r in bad]}")
            print("   先看留档里对应小节的完整输出，再决定改代码还是改论文。")
        else:
            print(f"✅ {len(results)}/{len(results)} 项全过。")
        print(f"\n完整输出：{log.relative_to(ROOT)}")
        fh.write(f"\n\n==== 汇总 ====\n" + "\n".join(
            f"{r['key']}: exit={r['rc']} {r['sec']:.1f}s {r['metrics']}" for r in results))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
