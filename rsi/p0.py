"""p0：一条命令跑通开环「生成→判分→蒸馏→训练→held-out 评估」。

P0 验收脚本（PROGRAM.md §5）。smoke 规模只验证管线正确性，
不产生任何结论性数字（结论只在全量 held-out + avg@3 下出，纪律§8）。

用法：
  python -m rsi.p0 --pool 20 --retry 2 --iters 40 --eval-n 60 --reps 1

前提：难度已标定（state/calibration.json）、held-out 已冻结。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from . import config, distill, env_gen, grade, train
from .engine import Engine
from .genome import H0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=20)
    ap.add_argument("--retry", type=int, default=2)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--eval-n", type=int, default=60)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--tag", type=str, default="p0_smoke")
    args = ap.parse_args()

    t_start = time.time()
    state = config.load_state()
    if not state.get("frozen", {}).get("heldout"):
        raise SystemExit("held-out 未冻结：先完成 rsi.calibrate + lockbox/freeze")
    difficulty = state.get("difficulty")
    print(f"[p0] 开环 smoke：pool={args.pool}×retry{args.retry} iters={args.iters} "
          f"eval_n={args.eval_n}×{args.reps} 难度={difficulty}")

    scratch = config.SCRATCH_DIR / args.tag
    scratch.mkdir(parents=True, exist_ok=True)

    # ---- 1. 出题（角色 pool，用后烧毁）----
    print("\n[p0] ① 生成题池 …")
    probs = env_gen.sample(args.pool, difficulty, role="pool", arm=args.tag,
                           extra_seed=f"{args.tag}-pool")
    env_gen.write_jsonl(scratch / "pool.jsonl", probs)
    print(f"    {len(probs)} 题 → {scratch/'pool.jsonl'}")

    # ---- 2. 基座生成（H_0，k 次采样，全轨迹归档）----
    print("\n[p0] ② 基座采样 …")
    engine = Engine()
    genome = H0()
    samples = engine.generate_batch(probs, genome, k=args.retry)
    est = engine.stats()
    env_gen.write_jsonl(
        scratch / "samples.jsonl",
        [{**s, "difficulty": difficulty} for s in samples],
    )
    print(f"    {len(samples)} 条采样，{est['tok_per_sec']} tok/s")

    # ---- 3. 判分 + 蒸馏 ----
    print("\n[p0] ③ 判分 + 蒸馏 …")
    dis = distill.distill(probs, samples, gen=0, seed=0)
    data_dir = scratch / "data"
    wrote = train.write_data_files(data_dir, dis["train"], dis["valid"])
    shot_pool_file = scratch / "shot_pool.jsonl"
    env_gen.write_jsonl(shot_pool_file, dis["shot_pool_add"])
    print(f"    {json.dumps(dis['stats'], ensure_ascii=False)}")
    print(f"    训练数据 → {data_dir}（train={wrote['train']} valid={wrote['valid']}）")
    if not dis["train"]:
        raise SystemExit("无验证通过样本——检查引擎输出或判分器（smoke 失败）")

    # ---- 4. LoRA 训练（fresh adapter from base）----
    print("\n[p0] ④ LoRA 训练 …")
    res = train.train(
        data_dir=data_dir,
        adapter_dir=scratch / "adapters",
        iters=args.iters,
        seed=0,
        log_file=scratch / "train.log",
    )
    print(f"    ok={res['ok']} {res['seconds']}s → {res['adapter_dir']}")
    if not res["ok"]:
        print(open(res["log"]).read()[-2000:])
        raise SystemExit("训练失败（见上）")

    # ---- 5. held-out 评估（独立进程，lockbox 门控）----
    print("\n[p0] ⑤ held-out 评估（lockbox 独立进程）…")
    spec = {
        "variants": [
            {"name": "BASE", "adapter": None, "genome": genome.to_dict()},
            {"name": "SMOKE", "adapter": str(scratch / "adapters"), "genome": genome.to_dict()},
        ],
        "eval_n": args.eval_n,
        "reps": args.reps,
        "seed": 0,
    }
    spec_path = scratch / "eval_spec.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    venv_py = state.get("venv_python") or str(config.VENV_PYTHON)
    proc = subprocess.run(
        [venv_py, str(config.LOCKBOX_DIR / "run_eval.py"),
         "--spec", str(spec_path), "--out-dir", str(scratch / "eval")],
        cwd=str(config.ROOT),
    )
    if proc.returncode != 0:
        raise SystemExit("评估失败")

    # ---- 6. 汇总 ----
    report = json.loads((scratch / "eval" / "eval.json").read_text(encoding="utf-8"))
    print("\n" + "=" * 62)
    print("[p0] 开环 smoke 结果（非结论性数字，仅管线验证）")
    print("=" * 62)
    for name, v in report["variants"].items():
        print(
            f"  {name:6s} accuracy = {v['accuracy_mean']:.4f} ± {v['accuracy_std']:.4f} "
            f"(n={report['eval_n']}×{report['reps']})"
        )
        hack = v["hack_summary"]
        print(
            f"         multi_boxed={hack['multi_boxed_rate']} "
            f"parse_fail={hack['parse_fail_rate']} "
            f"no_reasoning={hack['no_reasoning_rate']} "
            f"mean_chars={hack['mean_resp_chars']}"
        )
    total_min = (time.time() - t_start) / 60
    print(f"\n[p0] 完成，总耗时 {total_min:.1f} 分钟。全轨迹档案：{scratch}")
    print("[p0] P0 验收达成：生成→判分→蒸馏→训练→held-out 评估 一条命令跑通。")


if __name__ == "__main__":
    main()
