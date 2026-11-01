"""lockbox/run_eval.py：held-out 评估的执行器（唯一对 heldout 做生成的入口）。

用法（由 orchestrator / 人工以 subprocess 调用，绝不 import 进循环代码）：
  python lockbox/run_eval.py --spec spec.json --out-dir arms/arm_WH/gen_001/eval

spec 格式：
{
  "variants": [                       # 四变体（HANDOFF §3.2 Phase C）
    {"name": "sys",    "adapter": "arms/.../adapters", "genome": {...}},
    {"name": "w_only", "adapter": "...",               "genome": {H_0, k_shots:0}},
    {"name": "h_core", "adapter": null,               "genome": {H_n, 关 few-shot}},
    {"name": "h_full", "adapter": null,               "genome": {H_n 完整}}
  ],
  "eval_n": 450,       # 评估题数（固定前缀子集）
  "reps": 3,           # avg@3
  "seed": 0
}

输出 eval.json：每变体 accuracy(mean±std) + hack 指标 + 逐题对错（供配对检验）。
评估期的全部生成归档于 eval 目录（轨迹档案），但永不进入蒸馏（纪律§4）。
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    problems = load_problems(spec.get("eval_n"))
    print(f"[run_eval] held-out 校验通过：{len(problems)} 题")

    report: dict = {"eval_n": len(problems), "reps": spec.get("reps", 1), "variants": {}}
    all_stats: list[dict] = []

    for v in spec["variants"]:
        name = v["name"]
        genome = Genome.from_dict(v["genome"])
        shots = v.get("shots") or []  # h_full 的 few-shot 由 spec 提供（属 H 的状态）
        vdir = out_dir / name
        vdir.mkdir(parents=True, exist_ok=True)

        print(f"[run_eval] 变体 {name}: adapter={v.get('adapter') or 'BASE'}")
        engine = Engine(adapter_path=v.get("adapter"))
        reps_acc: list[float] = []
        per_problem: dict[str, list[bool]] = {}
        rep_summaries: list[dict] = []

        for r in range(1, spec.get("reps", 1) + 1):
            results = []
            path_resp = vdir / f"responses_rep{r}.jsonl"
            # 断点续跑：已生成的响应直接复用（判分始终重新执行——单一判分实现）
            done: dict[str, str] = {}
            if path_resp.exists():
                for ln in path_resp.read_text(encoding="utf-8").splitlines():
                    if ln.strip():
                        o = json.loads(ln)
                        done[o["pid"]] = o["response"]
            with open(path_resp, "a" if done else "w", encoding="utf-8") as rf:
                for i, p in enumerate(problems):
                    if p["id"] in done:
                        resp = done[p["id"]]
                    else:
                        resp, _tok = engine.generate_one(p["question"], genome, shots)
                        rf.write(
                            json.dumps(
                                {"pid": p["id"], "response": resp},
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        rf.flush()
                    res = grade.grade_one(p["answer_frac"], resp)
                    res.update({"pid": p["id"]})
                    results.append(res)
                    if (i + 1) % 50 == 0:
                        print(f"  [{name}] rep{r}: {i+1}/{len(problems)}", flush=True)
            s = grade.summarize(results)
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
            "per_problem": {pid: vals for pid, vals in per_problem.items()},
        }
        print(f"  [{name}] accuracy = {acc_mean:.4f} ± {acc_std:.4f}")

    report["engine_stats"] = all_stats
    report["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out = out_dir / "eval.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[run_eval] 报告已写：{out}")


if __name__ == "__main__":
    main()
