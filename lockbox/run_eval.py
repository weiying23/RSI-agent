"""lockbox/run_eval.py：held-out 评估的执行器（唯一对 heldout 做生成的入口）。

用法（由 orchestrator / 人工以 subprocess 调用，绝不 import 进循环代码）：
  python lockbox/run_eval.py --spec spec.json --out-dir arms/arm_WH/gen_001/eval

spec 格式：
{
  "variants": [ ... ],
  "eval_n": 450,
  "reps": 3,
  "seed": 0,
  "call_timeout_sec": 90,
  "task_timeout_sec": 180
}

超时纪律：
- 每道题、每个 rep 都有独立任务截止时间；
- 超时响应写入 responses_repN.jsonl 并标记 timed_out=true；
- 超时计为错误，分母绝不删除；
- 续跑时默认不重跑已超时题；显式 --retry-timeouts 才重跑。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lockbox.common import load_problems
from rsi import grade
from rsi.engine import Engine
from rsi.genome import Genome
from rsi.timeouts import Deadline, GenerationTimeout


def _read_done(path: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if not path.exists():
        return done
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        obj = json.loads(ln)
        done[obj["pid"]] = obj
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--retry-timeouts", action="store_true",
                    help="续跑时重跑已标记超时的题（默认复用超时结果）")
    args = ap.parse_args()

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    problems = load_problems(spec.get("eval_n"))
    print(f"[run_eval] held-out 校验通过：{len(problems)} 题")

    call_timeout = float(spec.get("call_timeout_sec", 90.0))
    task_timeout = float(spec.get("task_timeout_sec", 180.0))
    runtime_mode = spec.get("runtime_mode", "bare")
    if runtime_mode not in ("bare", "checked"):
        raise ValueError("runtime_mode must be bare or checked")
    reps = int(spec.get("reps", 1))
    report: dict = {
        "eval_n": len(problems),
        "reps": reps,
        "call_timeout_sec": call_timeout,
        "task_timeout_sec": task_timeout,
        "runtime_mode": runtime_mode,
        "variants": {},
    }
    all_stats: list[dict] = []

    for v in spec["variants"]:
        name = v["name"]
        genome = Genome.from_dict(v["genome"])
        shots = v.get("shots") or []
        vdir = out_dir / name
        vdir.mkdir(parents=True, exist_ok=True)

        print(f"[run_eval] 变体 {name}: adapter={v.get('adapter') or 'BASE'}")
        engine = Engine(
            adapter_path=v.get("adapter"),
            call_timeout_sec=call_timeout,
            task_timeout_sec=task_timeout,
        )
        reps_acc: list[float] = []
        per_problem: dict[str, list[bool]] = {}
        rep_summaries: list[dict] = []
        total_timeouts = 0

        for r in range(1, reps + 1):
            results = []
            path_resp = vdir / f"responses_rep{r}.jsonl"
            done = _read_done(path_resp)
            mode = "a" if done else "w"
            with open(path_resp, mode, encoding="utf-8") as rf:
                for i, p in enumerate(problems):
                    rec = done.get(p["id"])
                    timed_out = False
                    elapsed = 0.0
                    timeout_scope = None

                    if rec is not None and rec.get("timed_out") and args.retry_timeouts:
                        rec = None
                        done.pop(p["id"], None)

                    if rec is not None:
                        resp = rec.get("response", "")
                        timed_out = bool(rec.get("timed_out"))
                        elapsed = float(rec.get("elapsed_sec", 0.0))
                        timeout_scope = rec.get("timeout_scope")
                    else:
                        deadline = Deadline(task_timeout)
                        t0 = time.monotonic()
                        try:
                            if runtime_mode == "checked":
                                resp, _tok = engine.generate_one_checked(
                                    p["question"], genome, shots,
                                    max_retry=genome.format_retry,
                                    timeout_sec=min(call_timeout, deadline.remaining),
                                    deadline=deadline,
                                )
                            else:
                                resp, _tok = engine.generate_one(
                                    p["question"], genome, shots,
                                    timeout_sec=min(call_timeout, deadline.remaining),
                                )
                            if deadline.expired:
                                raise GenerationTimeout(task_timeout, scope="task")
                        except GenerationTimeout as exc:
                            timed_out = True
                            timeout_scope = getattr(exc, "scope", "call")
                            resp = ""
                        elapsed = time.monotonic() - t0
                        out = {
                            "pid": p["id"],
                            "response": resp,
                            "timed_out": timed_out,
                            "status": "timeout" if timed_out else "ok",
                            "timeout_scope": timeout_scope,
                            "elapsed_sec": round(elapsed, 3),
                        }
                        rf.write(json.dumps(out, ensure_ascii=False) + "\n")
                        rf.flush()

                    res = grade.grade_one(p["answer_frac"], resp)
                    res.update({"pid": p["id"], "timed_out": timed_out,
                                "elapsed_sec": round(elapsed, 3),
                                "timeout_scope": timeout_scope})
                    results.append(res)
                    if timed_out:
                        total_timeouts += 1
                    if (i + 1) % 50 == 0:
                        print(f"  [{name}] rep{r}: {i+1}/{len(problems)}", flush=True)

            s = grade.summarize(results)
            s["timeout_count"] = sum(1 for x in results if x.get("timed_out"))
            s["timeout_rate"] = round(s["timeout_count"] / len(results), 4) if results else 0.0
            reps_acc.append(s["accuracy"])
            rep_summaries.append(s)
            for res in results:
                per_problem.setdefault(res["pid"], []).append(res["correct"])

        st = engine.stats()
        all_stats.append(st)
        engine.close()

        acc_mean = statistics.mean(reps_acc)
        acc_std = statistics.stdev(reps_acc) if len(reps_acc) > 1 else 0.0
        report["variants"][name] = {
            "adapter": v.get("adapter"),
            "genome": genome.to_dict(),
            "k_shots": len(shots),
            "accuracy_mean": round(acc_mean, 4),
            "accuracy_std": round(acc_std, 4),
            "per_rep_accuracy": [round(a, 4) for a in reps_acc],
            "hack_summary": rep_summaries[-1],
            "timeout_count": total_timeouts,
            "timeout_rate": round(total_timeouts / (len(problems) * reps), 4),
            "per_problem": {pid: vals for pid, vals in per_problem.items()},
        }
        print(f"  [{name}] accuracy = {acc_mean:.4f} ± {acc_std:.4f} "
              f"timeouts={total_timeouts}/{len(problems) * reps}")

    report["engine_stats"] = all_stats
    report["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out = out_dir / "eval.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[run_eval] 报告已写：{out}")


if __name__ == "__main__":
    main()
