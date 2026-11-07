"""auto_sft_balanced：翻译完成后自动执行均衡中文 SFT 实验。

默认决策：
- heldout_zh = 200 道（从翻译集随机切出，只测不练）
- 训练外部数据 = 剩余翻译题
- 训练集 = 50% 自生成 verified（arms_hints）+ 50% 中文 GSM8K
- 模板均衡采样
- LoRA iters=300, lr=5e-5
- 评估：原 heldout 前 100 题 + heldout_zh 200 题
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from pathlib import Path

from . import config, grade, train
from .engine import Engine
from .genome import H0


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").split("\n") if x.strip()]


def wait_for_translation(out_path: Path, expected: int = 2261) -> None:
    while True:
        n = 0
        if out_path.exists():
            n = sum(1 for _ in out_path.open(encoding="utf-8") if _.strip())
        proc = subprocess.run(["pgrep", "-f", "translate_gsm8k"], capture_output=True, text=True)
        alive = proc.returncode == 0
        if (not alive and n > 0) or n >= expected:
            print(f"[auto] 翻译完成检测：lines={n} alive={alive}")
            return
        print(f"[auto] 等待翻译中：lines={n} alive={alive}")
        time.sleep(300)


def split_translated(src: Path, heldout_n: int, train_out: Path, heldout_out: Path, seed: int) -> None:
    recs = load_jsonl(src)
    rng = random.Random(seed)
    rng.shuffle(recs)
    heldout = recs[:heldout_n]
    train_recs = recs[heldout_n:]
    for path, data in [(heldout_out, heldout), (train_out, train_recs)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for rec in data:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[auto] 分割完成：train={len(train_recs)} heldout_zh={len(heldout)}")


def run_lockbox_eval(adapter: Path, eval_n: int, out_dir: Path, tag: str = "sys") -> dict:
    spec = {
        "variants": [{"name": tag, "adapter": str(adapter), "genome": H0().to_dict()}],
        "eval_n": eval_n,
        "reps": 1,
        "seed": 20261108,
        "call_timeout_sec": 90,
        "task_timeout_sec": 180,
        "runtime_mode": "checked",
    }
    spec_path = out_dir / f"eval_spec_{tag}.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    py = config.load_state().get("venv_python") or str(config.VENV_PYTHON)
    proc = subprocess.run(
        [py, str(config.LOCKBOX_DIR / "run_eval.py"),
         "--spec", str(spec_path), "--out-dir", str(out_dir / tag)],
        cwd=str(config.ROOT),
    )
    if proc.returncode != 0:
        raise RuntimeError("lockbox eval failed")
    return json.loads((out_dir / tag / "eval.json").read_text(encoding="utf-8"))


def eval_external(adapter: Path, heldout_path: Path, out_dir: Path) -> dict:
    recs = load_jsonl(heldout_path)
    engine = Engine(adapter_path=str(adapter), call_timeout_sec=90, task_timeout_sec=180)
    h0 = H0()
    results = []
    for rec in recs:
        q = ""
        ref = None
        for m in rec["messages"]:
            if m["role"] == "user":
                q = m["content"]
            elif m["role"] == "assistant":
                boxed, n = grade.extract_boxed(m["content"])
                if n > 0 and boxed is not None:
                    ref = boxed
        if not q or ref is None:
            continue
        resp, _ = engine.generate_one_checked(q, h0, shots=None, max_retry=h0.format_retry,
                                              timeout_sec=90)
        r = grade.grade_one(ref, resp)
        r["pid"] = rec.get("id", str(abs(hash(q)) % 10_000_000))
        results.append(r)
    engine.close()
    s = grade.summarize(results)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "external_eval.json").write_text(
        json.dumps({"n": len(results), "summary": s}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    return {"n": len(results), "accuracy": s["accuracy"], "parse_fail": s["parse_fail_rate"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-translation", action="store_true")
    ap.add_argument("--expected", type=int, default=2261)
    ap.add_argument("--translated", type=Path,
                    default=config.ROOT / "data/external/gsm8k_aligned_zh.jsonl")
    ap.add_argument("--self-data", type=Path,
                    default=config.ROOT / "arms_hints/arm_W/train_cum.jsonl")
    ap.add_argument("--heldout-n", type=int, default=200)
    ap.add_argument("--external-ratio", type=float, default=0.5)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--out-root", type=Path,
                    default=config.ROOT / "arms_sft" / "arm_W_balanced_v1")
    ap.add_argument("--seed", type=int, default=20261108)
    args = ap.parse_args()

    config.sanitize_proxy_env()
    if args.wait_translation:
        wait_for_translation(args.translated, args.expected)

    data_dir = config.ROOT / "data/sft_balanced_v1"
    train_zh = config.ROOT / "data/external/gsm8k_zh_train.jsonl"
    heldout_zh = config.ROOT / "data/external/gsm8k_zh_heldout.jsonl"
    split_translated(args.translated, args.heldout_n, train_zh, heldout_zh, args.seed)

    # build balanced train/valid
    subprocess.run(
        [config.load_state().get("venv_python") or str(config.VENV_PYTHON),
         "-m", "rsi.build_balanced_train",
         "--self-data", str(args.self_data),
         "--external-data", str(train_zh),
         "--external-ratio", str(args.external_ratio),
         "--balance-templates",
         "--out-dir", str(data_dir),
         "--seed", str(args.seed)],
        cwd=str(config.ROOT), check=True,
    )

    # train
    out_root = Path(args.out_root)
    adapter_dir = out_root / "adapters"
    log_file = out_root / "train.log"
    res = train.train(
        data_dir=data_dir,
        adapter_dir=adapter_dir,
        iters=args.iters,
        lr=args.lr,
        seed=args.seed,
        log_file=log_file,
    )
    if not res["ok"]:
        raise RuntimeError(f"SFT 训练失败：{log_file}")

    # eval original heldout 100
    lockbox_rep = run_lockbox_eval(adapter_dir, 100, out_root, "sys")
    acc_original = lockbox_rep["variants"]["sys"]["accuracy_mean"]

    # eval heldout_zh
    ext_rep = eval_external(adapter_dir, heldout_zh, out_root / "eval_zh")

    summary = {
        "acc_original_heldout_100": acc_original,
        "acc_heldout_zh": ext_rep,
        "adapter": str(adapter_dir),
        "train_data": str(data_dir),
        "train_res": {k: res[k] for k in ("iters", "seconds", "ok")},
        "external_ratio": args.external_ratio,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[auto] 完成", json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
