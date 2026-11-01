"""plot.py：P2 三张交付图。

用法：
  PY -m report.plot            # 扫描 arms/ 与 state/baseline/，输出 report/*.png
  PY -m report.plot --show    # 同时打开图片（macOS）

输出：
  report/fig1_main_curve.png    主曲线：sys accuracy vs 世代（三臂 + t=0 基线）
  report/fig2_attribution.png  归因瀑布：四变体分解（WH 臂逐代）
  report/fig3_cost.png          成本曲线：token 成本与 wallclock 逐代
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scratch" / "pylibs"))  # matplotlib 本地安装

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_config")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARM_COLORS = {"arm_W": "#1f77b4", "arm_H": "#2ca02c", "arm_WH": "#d62728"}
ARM_LABELS = {"arm_W": "arm W (weights only)", "arm_H": "arm H (harness only)",
              "arm_WH": "arm WH (co-evolve)"}


def _read(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def collect() -> dict:
    """收集全部 {arm: [(gen, accs, lineage), ...]} + baseline。"""
    data: dict[str, list] = {"arms": {}, "baseline": None}
    base = _read(ROOT / "state" / "baseline" / "eval.json")
    if base:
        data["baseline"] = base["variants"]["BASE_t0"]["accuracy_mean"]
    for arm_dir in sorted((ROOT / "arms").glob("arm_*")):
        points = []
        for gen_dir in sorted(arm_dir.glob("gen_*")):
            ev = _read(gen_dir / "eval" / "eval.json")
            lin = _read(gen_dir / "lineage.json")
            if not ev or not lin:
                continue
            points.append({
                "gen": int(gen_dir.name.split("_")[1]),
                "accs": {k: v["accuracy_mean"] for k, v in ev["variants"].items()},
                "std": {k: v["accuracy_std"] for k, v in ev["variants"].items()},
                "lineage": lin,
                "engine": ev.get("engine_stats", []),
                "hack": {k: v.get("hack_summary", {}) for k, v in ev["variants"].items()},
            })
        if points:
            data["arms"][arm_dir.name] = points
    return data


def fig1_main_curve(data: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    if data["baseline"] is not None:
        ax.axhline(data["baseline"], color="gray", ls="--", lw=1,
                   label=f"BASE t=0 ({data['baseline']:.3f})")
    for arm, pts in data["arms"].items():
        pts = sorted(pts, key=lambda p: p["gen"])
        xs = [p["gen"] for p in pts if p["gen"] >= 1]
        ys = [p["accs"].get("sys") for p in pts if p["gen"] >= 1]
        yerr = [p["std"].get("sys", 0) for p in pts if p["gen"] >= 1]
        if xs:
            ax.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3,
                        color=ARM_COLORS.get(arm), label=ARM_LABELS.get(arm, arm))
    ax.set_xlabel("generation")
    ax.set_ylabel("held-out accuracy (sys variant)")
    ax.set_title("Fig 1. Main curve: co-evolution vs single-channel arms")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig2_attribution(data: dict, out: Path) -> None:
    """四变体归因（WH 臂；W 臂显示 sys/w_only，H 臂显示 sys/h_core/h_full）。"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150,
                             gridspec_kw={"width_ratios": [2, 1.2]})
    ax = axes[0]
    wh = sorted(data["arms"].get("arm_WH", []), key=lambda p: p["gen"])
    wh = [p for p in wh if p["gen"] >= 1]
    variants = ["sys", "w_only", "h_core", "h_full"]
    colors = ["#d62728", "#1f77b4", "#ff7f0e", "#2ca02c"]
    width = 0.2
    for i, v in enumerate(variants):
        xs = [p["gen"] + (i - 1.5) * width for p in wh]
        ys = [p["accs"].get(v) for p in wh]
        ax.bar(xs, ys, width=width, color=colors[i], label=v)
    ax.set_xlabel("generation (arm_WH)")
    ax.set_ylabel("held-out accuracy")
    ax.set_title("Fig 2a. Four-variant attribution")
    ax.legend(ncol=2)
    ax.grid(alpha=0.3, axis="y")

    ax2 = axes[1]
    # 跨臂技能库通道净效应（h_full − h_core，每代一点）
    for arm in ("arm_WH", "arm_H"):
        pts = [p for p in sorted(data["arms"].get(arm, []), key=lambda p: p["gen"])
               if p["gen"] >= 1 and "h_full" in p["accs"] and "h_core" in p["accs"]]
        xs = [p["gen"] for p in pts]
        ys = [p["accs"]["h_full"] - p["accs"]["h_core"] for p in pts]
        if xs:
            ax2.plot(xs, ys, marker="s", color=ARM_COLORS.get(arm), label=arm)
    ax2.axhline(0, color="gray", ls=":")
    ax2.set_xlabel("generation")
    ax2.set_ylabel("h_full − h_core")
    ax2.set_title("Fig 2b. Skill-library channel effect")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig3_cost(data: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    for arm, pts in data["arms"].items():
        pts = sorted(pts, key=lambda p: p["gen"])
        xs, ys = [], []
        for p in pts:
            toks = sum(s.get("tokens_generated", 0) for s in p["engine"]) / 1000
            if toks:
                xs.append(p["gen"]); ys.append(toks)
        if xs:
            ax.plot(xs, ys, marker="o", color=ARM_COLORS.get(arm),
                    label=ARM_LABELS.get(arm, arm))
    ax.set_xlabel("generation")
    ax.set_ylabel("eval tokens (K)")
    ax.set_title("Fig 3. Cost curve (eval-side generation tokens)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    out_dir = ROOT / "report"
    out_dir.mkdir(exist_ok=True)
    data = collect()
    if not data["arms"] and data["baseline"] is None:
        print("plot.py: 尚无任何 arm/baseline 数据可画")
        return
    fig1_main_curve(data, out_dir / "fig1_main_curve.png")
    fig2_attribution(data, out_dir / "fig2_attribution.png")
    fig3_cost(data, out_dir / "fig3_cost.png")
    print(f"[plot] 三图已写入 {out_dir}/fig1_main_curve.png / fig2_attribution.png / fig3_cost.png")
    if args.show:
        for f in ("fig1_main_curve.png", "fig2_attribution.png", "fig3_cost.png"):
            os.system(f'open "{out_dir / f}"')


if __name__ == "__main__":
    main()
