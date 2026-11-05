"""rl_grpo：MLX 上的极简 REINFORCE / GRPO-Lite LoRA 训练。

只做最核心的 group advantage policy gradient：
  - 每题采样 group_size 个答案；
  - reward = rsi.grade 的 0/1；
  - advantage = (r - mean(group)) / (std(group) + eps)；
  - loss = -advantage * mean_token_logprob，梯度只更新 LoRA 权重。

这是 RL 的最小实现，不含 critic 和显式 KL；用于快速验证权重通道
在 RL 更新下是否比 SFT 更稳。评估仍走 lockbox 独立进程。
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from . import config, grade, taskbank
from .genome import H0
from .timeouts import GenerationTimeout, wall_timeout

try:
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.tuner.utils import linear_to_lora_layers
except Exception as e:  # pragma: no cover
    raise ImportError("rl_grpo 需要 mlx_lm 环境") from e


def _prompt_tokens(tokenizer, question: str) -> list[int]:
    msgs = [
        {"role": "system", "content": H0().instruction},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_dict=False)


def _sample_completion(model, tokenizer, question: str, max_tokens: int,
                       timeout_sec: float) -> tuple[str, bool]:
    h0 = H0()
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": h0.instruction},
         {"role": "user", "content": question}],
        add_generation_prompt=True,
    )
    sampler = make_sampler(temp=h0.temperature, top_p=h0.top_p)
    try:
        with wall_timeout(timeout_sec, scope="call"):
            resp = generate(
                model, tokenizer, prompt=prompt,
                max_tokens=max_tokens, sampler=sampler,
            )
        return resp, True
    except GenerationTimeout:
        return "", False


def _rl_loss(model, batch, lengths, advantages):
    """batch: [B, L] token ids；lengths: [B,2]=(prompt_len, seq_len)。
    advantages: [B] stop_gradient 标量。"""
    inputs = batch[:, :-1]
    targets = batch[:, 1:]

    logits = model(inputs)

    steps = mx.arange(1, targets.shape[1] + 1)
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])

    ce = nn.losses.cross_entropy(logits, targets) * mask
    per_sample = ce.astype(mx.float32).sum(axis=1) / mx.maximum(mask.sum(axis=1), 1e-6)
    loss = (per_sample * advantages).mean()
    return loss, mask.sum()


def _pad_batch(tokenizer, rows: list[tuple[str, str]]):
    """rows: [(question, completion)]。返回 (batch, lengths, advantages)。"""
    max_len = 0
    processed = []
    for q, a in rows:
        msgs = [
            {"role": "system", "content": H0().instruction},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]
        full = tokenizer.apply_chat_template(msgs, return_dict=False)
        if full[-1] != tokenizer.eos_token_id:
            full = full + [tokenizer.eos_token_id]
        prompt_msgs = msgs[:-1]
        prompt_tokens = tokenizer.apply_chat_template(
            prompt_msgs, add_generation_prompt=True, return_dict=False
        )
        offset = len(prompt_tokens)
        if len(full) > args_max_seq_len:
            full = full[: args_max_seq_len]
        max_len = max(max_len, len(full))
        processed.append((full, offset))
    batch = np.zeros((len(rows), max_len), np.int32)
    lengths = []
    for i, (tokens, offset) in enumerate(processed):
        batch[i, :len(tokens)] = tokens
        lengths.append((offset, len(tokens)))
    return mx.array(batch), mx.array(lengths, mx.int32)


args_max_seq_len = 1024


def _save_adapter(model, out_dir: Path, model_name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(out_dir / "adapters.safetensors"), weights)
    (out_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "model": model_name,
                "fine_tune_type": "lora",
                "num_layers": 16,
                "lora_parameters": {"rank": 8, "dropout": 0.0, "scale": 20.0},
                "mask_prompt": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _eval_adapter(adapter_dir: Path, arms_root: Path, arm_name: str, gen: int,
                  eval_n: int, call_timeout: float, task_timeout: float) -> dict:
    spec = {
        "variants": [
            {"name": "sys", "adapter": str(adapter_dir), "genome": H0().to_dict()}
        ],
        "eval_n": eval_n,
        "reps": 1,
        "seed": gen,
        "call_timeout_sec": call_timeout,
        "task_timeout_sec": task_timeout,
        "runtime_mode": "checked",
    }
    out_dir = arms_root / arm_name / f"gen_{gen:03d}" / "eval"
    spec_path = arms_root / arm_name / f"gen_{gen:03d}" / "eval_spec.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    py = config.load_state().get("venv_python") or str(config.VENV_PYTHON)
    proc = subprocess.run(
        [py, str(config.LOCKBOX_DIR / "run_eval.py"),
         "--spec", str(spec_path), "--out-dir", str(out_dir)],
        cwd=str(config.ROOT),
        env={**__import__("os").environ, "HF_HUB_OFFLINE": "1"},
    )
    if proc.returncode != 0:
        raise RuntimeError("RL 适配器评估失败")
    return json.loads((out_dir / "eval.json").read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--pool", type=int, default=20)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--eval-n", type=int, default=100)
    ap.add_argument("--call-timeout", type=float, default=90.0)
    ap.add_argument("--task-timeout", type=float, default=180.0)
    ap.add_argument("--task-bank", type=Path, default=taskbank.DEFAULT_BANK_ROOT)
    ap.add_argument("--arms-root", type=Path, default=config.ROOT / "arms_rl")
    ap.add_argument("--arm-name", type=str, default="arm_W_RL")
    args = ap.parse_args()

    config.sanitize_proxy_env()
    state = config.load_state()
    model_name = state.get("model") or config.BASE_MODEL
    print(f"[rl] 加载模型 {model_name}")
    model, tokenizer = load(model_name)
    model.freeze()
    linear_to_lora_layers(
        model, 16,
        {"rank": 8, "dropout": 0.0, "scale": 20.0},
    )
    print(f"[rl] LoRA 已挂载，训练参数约 "
          f"{sum(v.size for _, v in tree_flatten(model.trainable_parameters()))/1e6:.2f}M")

    arms_root = Path(args.arms_root)
    arm_dir = arms_root / args.arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)

    h0 = H0()
    for rnd in range(1, args.rounds + 1):
        gen_dir = arm_dir / f"gen_{rnd:03d}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        probs = taskbank.load_split("train", rnd, root=args.task_bank)[: args.pool]
        print(f"[rl] round {rnd}: pool={len(probs)} group={args.group_size}")

        # 1) rollout
        rows = []
        stats_correct = 0
        t0 = time.time()
        for p in probs:
            group = []
            for _ in range(args.group_size):
                resp, ok = _sample_completion(
                    model, tokenizer, p["question"],
                    max_tokens=h0.max_tokens, timeout_sec=args.call_timeout,
                )
                r = 1 if (ok and grade.grade_one(p["answer_frac"], resp)["correct"]) else 0
                stats_correct += r
                group.append((p["question"], resp, r))
            rows.append(group)

        gen_time = time.time() - t0
        print(f"[rl] rollout 完成：{stats_correct}/{len(probs)*args.group_size} "
              f"正确，耗时 {gen_time/60:.1f} min")

        # 2) group advantage
        samples = []
        for group in rows:
            rs = [g[2] for g in group]
            mean_r = sum(rs) / len(rs)
            std_r = (sum((x-mean_r)**2 for x in rs)/len(rs)) ** 0.5 + 1e-6
            for q, a, r in group:
                adv = (r - mean_r) / std_r
                samples.append((q, a, adv))
        np.random.seed(1000 + rnd)
        idx = np.random.permutation(len(samples))

        # 3) policy gradient
        loss_value_and_grad = nn.value_and_grad(model, _rl_loss)
        optimizer = optim.Adam(learning_rate=args.lr)
        model.train()
        for epoch in range(args.epochs):
            for i in idx:
                q, a, adv = samples[i]
                batch, lengths = _pad_batch(tokenizer, [(q, a)])
                adv_arr = mx.array([adv], mx.float32)
                (loss, toks), grads = loss_value_and_grad(model, batch, lengths, adv_arr)
                optimizer.update(model, grads)
                mx.eval(model.parameters(), loss)
        model.eval()

        # 4) save adapter + eval
        adapter_dir = gen_dir / "adapters"
        _save_adapter(model, adapter_dir, model_name)
        print(f"[rl] round {rnd}: LoRA 已保存 -> {adapter_dir}")
        rep = _eval_adapter(adapter_dir, arms_root, args.arm_name, rnd,
                             args.eval_n, args.call_timeout, args.task_timeout)
        acc = rep["variants"]["sys"]["accuracy_mean"]
        print(f"[rl] round {rnd}: held-out sys={acc:.4f}")

        lineage = {
            "arm": args.arm_name, "gen": rnd, "rl": "grpo-lite",
            "pool": len(probs), "group_size": args.group_size,
            "epochs": args.epochs, "lr": args.lr,
            "correct_rate": round(stats_correct/(len(probs)*args.group_size), 4),
            "phase_c": {"sys": acc},
            "adapter": str(adapter_dir),
            "rollout_seconds": round(gen_time, 1),
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        (gen_dir / "lineage.json").write_text(
            json.dumps(lineage, ensure_ascii=False, indent=1), encoding="utf-8"
        )
    print(f"[rl] 全部 {args.rounds} 轮完成")


if __name__ == "__main__":
    main()
