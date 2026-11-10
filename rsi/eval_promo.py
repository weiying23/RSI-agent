"""eval_promo：在独立 promo-val 上评估，供训练超参/检查点选择。

不使用 held-out 选择模型。默认使用 taskbank 的 promo_gen_004/005 共 100 题。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import config, grade, taskbank
from .engine import Engine
from .genome import H0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=str, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--shards", type=str, default="4,5")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    adapter = None if args.adapter in (None, "", "none", "None") else args.adapter
    problems = []
    for shard in args.shards.split(","):
        problems.extend(taskbank.load_split("promo", int(shard)))
    if args.limit:
        problems = problems[: args.limit]

    engine = Engine(adapter_path=adapter, call_timeout_sec=90, task_timeout_sec=180)
    h0 = H0()
    results = []
    responses = []
    t0 = time.time()
    for i, p in enumerate(problems, 1):
        timed_out = False
        try:
            resp, _ = engine.generate_one_checked(
                p["question"], h0, shots=None,
                max_retry=h0.format_retry, timeout_sec=90,
            )
        except Exception:
            resp, timed_out = "", True
        r = grade.grade_one(p["answer_frac"], resp)
        r.update({"pid": p["id"], "timed_out": timed_out})
        results.append(r)
        responses.append({"pid": p["id"], "response": resp, "timed_out": timed_out})
        if i % 25 == 0:
            print(f"[eval_promo] {i}/{len(problems)}", flush=True)
    engine.close()

    summary = grade.summarize(results)
    summary.update({
        "n": len(results),
        "adapter": adapter,
        "elapsed_min": round((time.time()-t0)/60, 1),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "promo_eval.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "responses.jsonl").open("w", encoding="utf-8") as f:
        for rec in responses:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("[eval_promo] " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
