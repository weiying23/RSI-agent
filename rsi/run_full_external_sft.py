"""run_full_external_sft：全量外部 GSM8K + 小比例自生成正则的 SFT 实验。

流程：
  1. 组装训练集：全部中文 GSM8K 训练数据 + 自生成 verified 数据（小比例）
  2. 在混合测试集上评估 Base 基线
  3. LoRA SFT 训练
  4. 在混合测试集上评估 SFT
  5. 输出 summary.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from . import config, train


def run(cmd: list[str]) -> None:
    print("[run_full_external] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(config.ROOT), check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-data", type=Path,
                    default=config.ROOT / "arms_hints/arm_W/train_cum.jsonl")
    ap.add_argument("--external-data", type=Path,
                    default=config.ROOT / "data/external/gsm8k_zh_train.jsonl")
    ap.add_argument("--train-dir", type=Path,
                    default=config.ROOT / "data/sft_full_external_v1")
    ap.add_argument("--out-root", type=Path,
                    default=config.ROOT / "arms_sft/arm_W_full_external")
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=20261108)
    args = ap.parse_args()

    config.sanitize_proxy_env()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    py = config.load_state().get("venv_python") or str(config.VENV_PYTHON)

    # 1. build training data
    run([
        py, "-m", "rsi.build_balanced_train",
        "--self-data", str(args.self_data),
        "--external-data", str(args.external_data),
        "--all-external",
        "--out-dir", str(args.train_dir),
        "--seed", str(args.seed),
    ])

    # 2. baseline mixed eval
    t0 = time.time()
    run([
        py, "-m", "rsi.eval_mixed",
        "--adapter", "none",
        "--out-dir", str(out_root / "eval_base"),
    ])
    base = json.loads((out_root / "eval_base" / "mixed_eval.json").read_text(encoding="utf-8"))

    # 3. SFT
    adapter_dir = out_root / "adapters"
    res = train.train(
        data_dir=args.train_dir,
        adapter_dir=adapter_dir,
        iters=args.iters,
        lr=args.lr,
        seed=args.seed,
        log_file=out_root / "train.log",
    )
    if not res["ok"]:
        raise RuntimeError(f"SFT failed: {out_root / 'train.log'}")

    # 4. SFT mixed eval
    run([
        py, "-m", "rsi.eval_mixed",
        "--adapter", str(adapter_dir),
        "--out-dir", str(out_root / "eval_sft"),
    ])
    sft = json.loads((out_root / "eval_sft" / "mixed_eval.json").read_text(encoding="utf-8"))

    summary = {
        "base": base,
        "sft": sft,
        "train": {
            "iters": res["iters"],
            "seconds": res["seconds"],
            "data_manifest": str(args.train_dir / "manifest.json"),
        },
        "elapsed_min": round((time.time() - t0) / 60, 1),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[run_full_external] 完成", json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
