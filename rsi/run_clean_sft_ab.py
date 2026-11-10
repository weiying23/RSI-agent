"""run_clean_sft_ab：评估格式统一后的三组 LoRA SFT 对照实验。

A: 100 iters / lr 1e-5
B:  50 iters / lr 5e-6
C: 200 iters / lr 1e-5

每组在独立 promo-val 100 题上出结果，最佳组再跑 650 混合 held-out。
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from . import config, train

EXPERIMENTS = [
    {"name": "A_100_lr1e5", "iters": 100, "lr": 1e-5},
    {"name": "B_50_lr5e6", "iters": 50, "lr": 5e-6},
    {"name": "C_200_lr1e5", "iters": 200, "lr": 1e-5},
]


def run(cmd: list[str], log_path: Path | None = None) -> None:
    print("[clean_ab] " + " ".join(cmd), flush=True)
    if log_path:
        with log_path.open("a", encoding="utf-8") as f:
            subprocess.run(cmd, cwd=str(config.ROOT), check=True, stdout=f, stderr=subprocess.STDOUT)
    else:
        subprocess.run(cmd, cwd=str(config.ROOT), check=True)


def main() -> None:
    config.sanitize_proxy_env()
    state = config.load_state()
    py = state.get("venv_python") or str(config.VENV_PYTHON)
    data_dir = config.ROOT / "data/sft_eval_format_v1"
    out_root = config.ROOT / "arms_sft/clean_sft_ab"
    out_root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Base 在同一 promo-val 上的参照，避免跨测试集比较。
    print("[clean_ab] 先评估 Base promo-val ...", flush=True)
    run([py, "-m", "rsi.eval_promo", "--adapter", "none",
         "--out-dir", str(out_root / "eval_base")])
    base = json.loads((out_root / "eval_base/promo_eval.json").read_text(encoding="utf-8"))

    results = []
    for exp in EXPERIMENTS:
        name = exp["name"]
        exp_dir = out_root / name
        exp_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[clean_ab] ===== 开始 {name} =====", flush=True)
        res = train.train(
            data_dir=data_dir,
            adapter_dir=exp_dir / "adapters",
            iters=exp["iters"],
            lr=exp["lr"],
            seed=20261110,
            log_file=exp_dir / "train.log",
        )
        if not res["ok"]:
            raise RuntimeError(f"训练失败：{exp_dir / 'train.log'}")

        run([py, "-m", "rsi.eval_promo", "--adapter", str(exp_dir / "adapters"),
             "--out-dir", str(exp_dir / "eval_promo")])
        ev = json.loads((exp_dir / "eval_promo/promo_eval.json").read_text(encoding="utf-8"))
        rec = {**exp, "train_seconds": res["seconds"], "promo": ev}
        results.append(rec)
        print(f"[clean_ab] ===== {name} 完成：accuracy={ev['accuracy']:.4f} "
              f"parse_fail={ev['parse_fail_rate']:.4f} train={res['seconds']:.1f}s =====\n",
              flush=True)
        (out_root / "partial_results.json").write_text(
            json.dumps({"base": base, "experiments": results}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # 先 accuracy，后 parse_fail，选择最佳组；held-out 不参与选择。
    best = max(results, key=lambda r: (r["promo"]["accuracy"], -r["promo"]["parse_fail_rate"]))
    print(f"[clean_ab] promo-val 最佳组：{best['name']}，开始最终 650 混合评估", flush=True)
    best_dir = out_root / best["name"]
    run([py, "-m", "rsi.eval_mixed", "--adapter", str(best_dir / "adapters"),
         "--out-dir", str(best_dir / "eval_mixed")])
    mixed = json.loads((best_dir / "eval_mixed/mixed_eval.json").read_text(encoding="utf-8"))

    # v7 已测过的 Base 混合基线；若存在则复用，不重复跑。
    base_mixed_path = config.ROOT / "arms_sft/arm_W_full_external/eval_base/mixed_eval.json"
    base_mixed = json.loads(base_mixed_path.read_text(encoding="utf-8")) if base_mixed_path.exists() else None

    summary = {
        "base_promo": base,
        "experiments": results,
        "best": best["name"],
        "best_mixed": mixed,
        "base_mixed": base_mixed,
        "elapsed_min": round((time.time()-t0)/60, 1),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# test-RSI 实验记录 v9 —— 评估格式统一后的三组 SFT",
        "",
        f"> 完成时间：{summary['finished_at']}",
        "> 数据：data/sft_eval_format_v1，1,500 条",
        "> 选择：promo-val 100 题，最佳组再做 650 混合 held-out。",
        "",
        "## 各组 promo-val 结果",
        "",
        "| 组 | iters | lr | accuracy | parse_fail |",
        "|---|---:|---:|---:|---:|",
        f"| Base | — | — | {base['accuracy']*100:.2f}% | {base['parse_fail_rate']*100:.2f}% |",
    ]
    for r in results:
        e = r["promo"]
        lines.append(f"| {r['name']} | {r['iters']} | {r['lr']} | "
                     f"{e['accuracy']*100:.2f}% | {e['parse_fail_rate']*100:.2f}% |")
    lines += [
        "",
        "## 最佳组最终混合评估",
        "",
        f"最佳组：**{best['name']}**",
        "",
        "| 指标 | Base | SFT |",
        "|---|---:|---:|",
    ]
    if base_mixed:
        lines += [
            f"| 原 held-out 450 | {base_mixed['original_accuracy']*100:.2f}% | {mixed['original_accuracy']*100:.2f}% |",
            f"| 外部中文 200 | {base_mixed['external_accuracy']*100:.2f}% | {mixed['external_accuracy']*100:.2f}% |",
            f"| 混合 650 | {base_mixed['mixed_accuracy']*100:.2f}% | {mixed['mixed_accuracy']*100:.2f}% |",
        ]
    else:
        lines += [
            f"| 原 held-out 450 | — | {mixed['original_accuracy']*100:.2f}% |",
            f"| 外部中文 200 | — | {mixed['external_accuracy']*100:.2f}% |",
            f"| 混合 650 | — | {mixed['mixed_accuracy']*100:.2f}% |",
        ]
    lines += [
        "",
        "## 初步分析",
        "",
        "- 数据已统一为评估格式：H0 system + 裸题 user + 直接推理 assistant。",
        "- 超参选择只在 promo-val 上进行，held-out 不参与调参。",
        "- 最终结果应与 harness-only 66% 对比，判断 SFT 是否值得继续。",
        "",
    ]
    (config.ROOT / "v9_clean_sft.md").write_text("\n".join(lines), encoding="utf-8")

    print("[clean_ab] 全部完成：" + json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    # 自动归档结果和代码，不上传 adapter。
    subprocess.run(["git", "add", "v9_clean_sft.md", "rsi/eval_promo.py", "rsi/run_clean_sft_ab.py"],
                   cwd=config.ROOT, check=True)
    subprocess.run(["git", "commit", "-m", "v9 evaluation-aligned SFT A/B/C results"], cwd=config.ROOT, check=True)
    subprocess.run(["git", "push", "origin", "main"], cwd=config.ROOT, check=True)


if __name__ == "__main__":
    main()
