"""train：mlx_lm lora 的参数化封装（复用用户 lora 流水线的配置核心）。

每代从基座重训 fresh adapter（STaR 式累积回放，HANDOFF §3.5 决策2）。
训练超参继承用户 config.yaml（Qwen2.5-3B rank8/scale20），仅参数化：
data / adapter_path / iters / lr / seed。
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from . import config


def build_train_config(
    data_dir: Path, adapter_dir: Path, iters: int, lr: float, seed: int
) -> dict:
    """生成 mlx_lm lora 的 config dict（镜像用户 lora/config.yaml）。"""
    return {
        "model": config.load_state().get("model") or config.BASE_MODEL,
        "train": True,
        "data": str(data_dir),
        "fine_tune_type": "lora",
        "optimizer": "adam",
        "num_layers": 16,
        "lora_parameters": {"rank": 8, "dropout": 0.0, "scale": 20.0},
        "batch_size": 2,
        "grad_accumulation_steps": 4,
        "max_seq_length": 1024,
        "grad_checkpoint": True,
        "mask_prompt": True,
        "iters": int(iters),
        "val_batches": 25,
        "learning_rate": float(lr),
        "steps_per_report": 20,
        "steps_per_eval": 200,
        "adapter_path": str(adapter_dir),
        "save_every": 100000,  # 只在结束时保存（fresh adapter，无中间检查点需求）
        "seed": int(seed),
    }


def train(
    data_dir: Path,
    adapter_dir: Path,
    iters: int = 400,
    lr: float = 5e-5,
    seed: int = 0,
    log_file: Path | None = None,
    timeout: float | None = None,
) -> dict:
    """运行一次 LoRA 训练。data_dir 需含 train.jsonl + valid.jsonl。"""
    import yaml

    data_dir = Path(data_dir)
    adapter_dir = Path(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_file or adapter_dir.parent / "train.log"

    cfg = build_train_config(data_dir, adapter_dir, iters, lr, seed)
    cfg_path = adapter_dir.parent / "lora_config.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f, allow_unicode=True)

    venv_py = config.load_state().get("venv_python") or str(config.VENV_PYTHON)
    config.sanitize_proxy_env()
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"  # 模型已缓存，禁止网络检查拖慢启动

    cmd = [venv_py, "-m", "mlx_lm", "lora", "-c", str(cfg_path)]
    t0 = time.time()
    with open(log_file, "w") as lf:
        lf.write(f"# cmd: {' '.join(cmd)}\n")
        lf.flush()
        proc = subprocess.run(
            cmd,
            stdout=lf,
            stderr=subprocess.STDOUT,
            cwd=str(config.ROOT),
            env=env,
            timeout=timeout,
        )
    seconds = time.time() - t0
    ok = proc.returncode == 0 and (adapter_dir / "adapters.safetensors").exists()
    return {
        "ok": ok,
        "returncode": proc.returncode,
        "adapter_dir": str(adapter_dir),
        "config": str(cfg_path),
        "log": str(log_file),
        "seconds": round(seconds, 1),
        "iters": iters,
    }


def write_data_files(data_dir: Path, train: list[dict], valid: list[dict]) -> dict:
    """写 train.jsonl / valid.jsonl（messages 格式，MLX 直读）。"""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, recs in (("train.jsonl", train), ("valid.jsonl", valid)):
        with open(data_dir / name, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"train": len(train), "valid": len(valid)}
