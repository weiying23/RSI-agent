"""run_sft_next_plan：按复现→正则化→80:20 数据配比的顺序执行实验并逐组上传。

D：最佳 C 换 seed 复跑（scale=20）
E：同设置只改 LoRA scale=10
F：若 D/E 有组达到原 held-out>60%、混合>58%，做 80:20 数据实验

选择指标：accuracy - 0.25 * parse_fail_rate。
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from . import config, train


def run(cmd: list[str]) -> None:
    print("[next_plan] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(config.ROOT), check=True)


def push_doc(filename: str, text: str, message: str) -> None:
    path = config.ROOT / filename
    path.write_text(text, encoding="utf-8")
    subprocess.run(["git", "add", filename], cwd=config.ROOT, check=True)
    subprocess.run(["git", "commit", "-m", message], cwd=config.ROOT, check=True)
    subprocess.run(["git", "push", "origin", "main"], cwd=config.ROOT, check=True)


def experiment(name: str, iters: int, lr: float, scale: float, seed: int,
               data_dir: Path, out_root: Path, py: str) -> dict:
    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)
    print(f"[next_plan] ===== {name} 开始训练 =====", flush=True)
    res = train.train(
        data_dir=data_dir, adapter_dir=out / "adapters",
        iters=iters, lr=lr, seed=seed,
        log_file=out / "train.log", lora_scale=scale,
    )
    if not res["ok"]:
        raise RuntimeError(f"训练失败：{out / 'train.log'}")
    run([py, "-m", "rsi.eval_promo", "--adapter", str(out / "adapters"),
         "--out-dir", str(out / "eval_promo")])
    promo = json.loads((out / "eval_promo/promo_eval.json").read_text(encoding="utf-8"))
    run([py, "-m", "rsi.eval_mixed", "--adapter", str(out / "adapters"),
         "--out-dir", str(out / "eval_mixed")])
    mixed = json.loads((out / "eval_mixed/mixed_eval.json").read_text(encoding="utf-8"))
    rec = {
        "name": name, "iters": iters, "lr": lr, "scale": scale, "seed": seed,
        "train_seconds": res["seconds"], "promo": promo, "mixed": mixed,
        "composite": round(promo["accuracy"] - 0.25 * promo["parse_fail_rate"], 4),
        "data_dir": str(data_dir),
    }
    print(f"[next_plan] ===== {name} 完成：promo={promo['accuracy']:.4f} "
          f"parse_fail={promo['parse_fail_rate']:.4f} mixed={mixed['mixed_accuracy']:.4f} "
          f"composite={rec['composite']:.4f} =====", flush=True)
    return rec


def mixed_table(base: dict, rec: dict) -> str:
    m = rec["mixed"]
    return (
        "| 指标 | Base | SFT |\n|---|---:|---:|\n"
        f"| 原 held-out 450 | {base['original_accuracy']*100:.2f}% | {m['original_accuracy']*100:.2f}% |\n"
        f"| 外部中文 200 | {base['external_accuracy']*100:.2f}% | {m['external_accuracy']*100:.2f}% |\n"
        f"| 混合 650 | {base['mixed_accuracy']*100:.2f}% | {m['mixed_accuracy']*100:.2f}% |"
    )


def main() -> None:
    config.sanitize_proxy_env()
    state = config.load_state()
    py = state.get("venv_python") or str(config.VENV_PYTHON)
    out_root = config.ROOT / "arms_sft/next_plan"
    out_root.mkdir(parents=True, exist_ok=True)
    base_mixed = json.loads(
        (config.ROOT / "arms_sft/arm_W_full_external/eval_base/mixed_eval.json").read_text(encoding="utf-8")
    )
    data_dir = config.ROOT / "data/sft_eval_format_v1"
    t0 = time.time()

    # D：seed2 复现 C
    d = experiment("D_seed2_scale20", 200, 1e-5, 20.0, 20261111, data_dir, out_root, py)
    push_doc(
        "v10_sft_seed2.md",
        "# v10 — 最佳 C 换 seed 复跑\n\n"
        f"> 设置：200 iters，lr=1e-5，scale=20，seed=20261111。\n\n"
        f"promo accuracy={d['promo']['accuracy']*100:.2f}%，"
        f"parse_fail={d['promo']['parse_fail_rate']*100:.2f}%，"
        f"composite={d['composite']:.4f}。\n\n"
        + mixed_table(base_mixed, d)
        + "\n\n## 分析\n\n"
        "- 本组用于验证 C 的收益是否能在新 seed 下复现，不增加数据。\n"
        "- 与 C 的 62.89% 原 held-out、59.54% 混合成绩对照解读。\n",
        "v10 SFT best-C seed2 reproduction result",
    )

    # E：只改 scale=10，测试正则化
    e = experiment("E_scale10", 200, 1e-5, 10.0, 20261110, data_dir, out_root, py)
    push_doc(
        "v11_sft_scale10.md",
        "# v11 — LoRA scale=10 正则化实验\n\n"
        f"> 设置：200 iters，lr=1e-5，scale=10，其他与 C 相同。\n\n"
        f"promo accuracy={e['promo']['accuracy']*100:.2f}%，"
        f"parse_fail={e['promo']['parse_fail_rate']*100:.2f}%，"
        f"composite={e['composite']:.4f}。\n\n"
        + mixed_table(base_mixed, e)
        + "\n\n## 分析\n\n"
        "- scale 从 20 降为 10，目的是减小权重漂移并改善格式稳定性。\n"
        "- 选择指标为 accuracy − 0.25 × parse_fail_rate。\n",
        "v11 LoRA scale10 regularization result",
    )

    candidates = [d, e]
    best = max(candidates, key=lambda r: r["composite"])
    allow_mix = any(
        r["mixed"]["original_accuracy"] > 0.60 and r["mixed"]["mixed_accuracy"] > 0.58
        for r in candidates
    )
    summary = {"D": d, "E": e, "best_by_composite": best["name"], "allow_mix": allow_mix}

    if allow_mix:
        # F：80% 自生成 + 20% 中文 GSM8K，使用 composite 最优的 scale。
        run([py, "-m", "rsi.build_mix80_data"])
        f = experiment(
            "F_mix80_20", 200, 1e-5, best["scale"], 20261112,
            config.ROOT / "data/sft_mix80_20_v1", out_root, py,
        )
        summary["F"] = f
        push_doc(
            "v12_sft_mix80_20.md",
            "# v12 — 80:20 自生成 / 中文 GSM8K 数据实验\n\n"
            f"> 设置：200 iters，lr=1e-5，scale={best['scale']}，seed=20261112。\n"
            "> 训练集：80% 自生成评估格式数据 + 20% 中文 GSM8K 对齐数据。\n\n"
            f"promo accuracy={f['promo']['accuracy']*100:.2f}%，"
            f"parse_fail={f['promo']['parse_fail_rate']*100:.2f}%，"
            f"composite={f['composite']:.4f}。\n\n"
            + mixed_table(base_mixed, f)
            + "\n\n## 分析\n\n"
            "- 只有在 D/E 至少一组超过原 held-out 60% 和混合 58% 后才启动。\n"
            "- 本轮检验小比例外部数据是否能提供额外泛化收益。\n",
            "v12 80/20 self-generated and Chinese GSM8K SFT result",
        )
    else:
        push_doc(
            "v12_sft_next_decision.md",
            "# v12 — 80:20 数据实验跳过\n\n"
            "D/E 均未同时达到原 held-out >60% 和混合 >58% 的复现门槛，"
            "因此不继续外部数据混合实验，避免在不稳定权重通道上继续扩张。\n",
            "v12 skip 80/20 experiment due to unmet reproduction threshold",
        )

    summary["elapsed_min"] = round((time.time() - t0) / 60, 1)
    summary["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[next_plan] 全部结束：" + json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
