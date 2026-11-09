"""run_internal_sft：只使用内部自生成 RSR 数据训练 LoRA SFT。

流程：
  1. env_gen 生成足够数量的同域新题
  2. 基座模型 RSR-lite 生成训练数据（plan→solve→verify→rewrite→verify）
  3. 目标接受 1500 条，90/10 切 train/valid
  4. LoRA SFT
  5. 在混合测试集上评估，并复用 v7 的 Base 基线
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from . import config, env_gen, train
from .engine import Engine
from .rsr_data import rsr_generate_one


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempts", type=int, default=3000)
    ap.add_argument("--target", type=int, default=1500)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=20261109)
    ap.add_argument("--out-root", type=Path,
                    default=config.ROOT / "arms_sft/arm_W_internal_1500")
    args = ap.parse_args()

    config.sanitize_proxy_env()
    state = config.load_state()
    difficulty = state.get("difficulty") or 5
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # 1) 生成同域新题
    print(f"[internal_sft] 生成 {args.attempts} 道同域新题 ...", flush=True)
    probs = env_gen.sample(
        args.attempts, difficulty, role="pool", arm="rsr_internal",
        extra_seed=f"internal-rsr:{args.seed}",
    )
    prob_file = out_root / "rsr_pool.jsonl"
    env_gen.write_jsonl(prob_file, probs)
    print(f"[internal_sft] 题库生成完成：{len(probs)} -> {prob_file}", flush=True)

    # 2) RSR-lite 生成训练数据
    engine = Engine()
    rng = random.Random(args.seed)
    rng.shuffle(probs)
    accepted: list[dict] = []
    generated = 0
    print(f"[internal_sft] RSR-lite 开始，目标 {args.target} 条", flush=True)
    for i, p in enumerate(probs, 1):
        rec = rsr_generate_one(engine, p, timeout_sec=90)
        generated += 1
        if rec is not None:
            rec["id"] = p["id"]
            accepted.append(rec)
        if i % 100 == 0:
            print(f"[internal_sft] 已尝试 {i}/{len(probs)}，接受 {len(accepted)}/{args.target}，"
                  f"用时 {(time.time()-t_start)/60:.1f}min", flush=True)
        if len(accepted) >= args.target:
            print(f"[internal_sft] 达到目标 {args.target} 条，停止生成", flush=True)
            break
    engine.close()

    if len(accepted) < args.target:
        print(f"[internal_sft] 警告：只接受了 {len(accepted)}/{args.target}，继续用现有数据训练", flush=True)

    rng.shuffle(accepted)
    n_valid = max(1, int(len(accepted) * 0.1))
    valid = accepted[:n_valid]
    train_recs = accepted[n_valid:]
    data_dir = out_root / "data"
    train.write_data_files(data_dir, train_recs, valid)

    # 3) LoRA SFT
    print(f"[internal_sft] LoRA 训练：train={len(train_recs)} valid={len(valid)}", flush=True)
    adapter_dir = out_root / "adapters"
    res = train.train(
        data_dir=data_dir,
        adapter_dir=adapter_dir,
        iters=args.iters,
        lr=args.lr,
        seed=args.seed,
        log_file=out_root / "train.log",
    )
    if not res["ok"]:
        raise RuntimeError(f"训练失败：{out_root / 'train.log'}")

    # 4) SFT mixed eval
    py = state.get("venv_python") or str(config.VENV_PYTHON)
    import subprocess
    subprocess.run(
        [py, "-m", "rsi.eval_mixed", "--adapter", str(adapter_dir),
         "--out-dir", str(out_root / "eval_sft")],
        cwd=str(config.ROOT), check=True,
    )
    sft = json.loads((out_root / "eval_sft" / "mixed_eval.json").read_text(encoding="utf-8"))

    # 复用之前 Base 基线，不存在则重新跑
    base_path = config.ROOT / "arms_sft/arm_W_full_external/eval_base/mixed_eval.json"
    if base_path.exists():
        base = json.loads(base_path.read_text(encoding="utf-8"))
    else:
        subprocess.run(
            [py, "-m", "rsi.eval_mixed", "--adapter", "none",
             "--out-dir", str(out_root / "eval_base")],
            cwd=str(config.ROOT), check=True,
        )
        base = json.loads((out_root / "eval_base" / "mixed_eval.json").read_text(encoding="utf-8"))

    summary = {
        "attempts": generated,
        "accepted": len(accepted),
        "train_n": len(train_recs),
        "valid_n": len(valid),
        "train_res": {k: res[k] for k in ("iters", "seconds", "ok")},
        "base": base,
        "sft": sft,
        "elapsed_min": round((time.time() - t_start) / 60, 1),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[internal_sft] 完成", json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
