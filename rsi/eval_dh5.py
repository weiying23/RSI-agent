"""eval_dh5：评估 D 权重 × H₅ harness 的协同效果。

测试集：
  - 原 held-out 450 题
  - 外部中文 heldout 200 题
  - 混合 650 题

新增评估：
  1. Base + H₅ core
  2. Base + H₅ full
  3. D + H₅ sys

复用已有结果：
  - Base + H₀
  - D + H₀
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from . import config, grade
from .engine import Engine
from .genome import Genome


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").split("\n") if x.strip()]


def build_variants(h5: Genome, shots: list[dict], d_adapter: Path) -> list[dict]:
    h_core = h5.__class__(**{**h5.to_dict(), "k_shots": 0, "shot_policy": "none"})
    return [
        {"name": "h_core", "adapter": None, "genome": h_core, "shots": []},
        {"name": "h_full", "adapter": None, "genome": h5, "shots": shots},
        {"name": "sys", "adapter": str(d_adapter), "genome": h5, "shots": shots},
    ]


def eval_original(variants: list[dict], out_dir: Path, py: str) -> dict:
    spec = {
        "eval_n": 450,
        "reps": 1,
        "seed": 20261112,
        "call_timeout_sec": 90,
        "task_timeout_sec": 180,
        "runtime_mode": "checked",
        "variants": [
            {
                "name": v["name"],
                "adapter": v["adapter"],
                "genome": v["genome"].to_dict(),
                "shots": v["shots"],
            }
            for v in variants
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    spec_path = out_dir / "eval_spec.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    subprocess.run(
        [py, str(config.LOCKBOX_DIR / "run_eval.py"),
         "--spec", str(spec_path), "--out-dir", str(out_dir)],
        cwd=str(config.ROOT), check=True,
    )
    rep = read_json(out_dir / "eval.json")
    return {
        name: {
            "accuracy": v["accuracy_mean"],
            "parse_fail": v["hack_summary"].get("parse_fail_rate"),
            "mean_resp_chars": v["hack_summary"].get("mean_resp_chars"),
        }
        for name, v in rep["variants"].items()
    }


def eval_external(variants: list[dict], heldout: Path, out_dir: Path) -> dict:
    recs = read_jsonl(heldout)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for v in variants:
        engine = Engine(
            adapter_path=v["adapter"],
            call_timeout_sec=90,
            task_timeout_sec=180,
        )
        rs = []
        responses = []
        t0 = time.time()
        for i, rec in enumerate(recs, 1):
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
            timed_out = False
            try:
                resp, _ = engine.generate_one_checked(
                    q, v["genome"], shots=v["shots"],
                    max_retry=v["genome"].format_retry,
                    timeout_sec=90,
                )
            except Exception:
                resp, timed_out = "", True
            r = grade.grade_one(ref, resp)
            r.update({"pid": str(i), "timed_out": timed_out})
            rs.append(r)
            responses.append({"pid": str(i), "response": resp, "timed_out": timed_out})
            if i % 50 == 0:
                print(f"[eval_dh5] external {v['name']} {i}/{len(recs)}", flush=True)
        engine.close()
        s = grade.summarize(rs)
        results[v["name"]] = {
            "accuracy": s["accuracy"],
            "parse_fail": s["parse_fail_rate"],
            "mean_resp_chars": s.get("mean_resp_chars"),
            "n": len(rs),
            "elapsed_min": round((time.time() - t0) / 60, 1),
        }
        (out_dir / f"{v['name']}_external_eval.json").write_text(
            json.dumps(results[v["name"]], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (out_dir / f"{v['name']}_external_responses.jsonl").write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in responses) + "\n",
            encoding="utf-8",
        )
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5-dir", type=Path,
                    default=config.ROOT / "arms_fixed/arm_H/gen_005")
    ap.add_argument("--d-adapter", type=Path,
                    default=config.ROOT / "arms_sft/next_plan/D_seed2_scale20/adapters")
    ap.add_argument("--external-heldout", type=Path,
                    default=config.ROOT / "data/external/gsm8k_zh_heldout.jsonl")
    ap.add_argument("--out-dir", type=Path,
                    default=config.ROOT / "arms_sft/dh5")
    args = ap.parse_args()

    config.sanitize_proxy_env()
    state = config.load_state()
    py = state.get("venv_python") or str(config.VENV_PYTHON)

    h5 = Genome.from_dict(read_json(args.h5_dir / "genome.json"))
    eval_spec = read_json(args.h5_dir / "eval_spec.json")
    shots = next(
        v.get("shots", []) for v in eval_spec["variants"] if v["name"] == "h_full"
    )
    variants = build_variants(h5, shots, args.d_adapter)

    print("[eval_dh5] 原 held-out 450 题评估 ...", flush=True)
    original = eval_original(variants, args.out_dir / "original", py)
    print("[eval_dh5] 外部中文 200 题评估 ...", flush=True)
    external = eval_external(variants, args.external_heldout, args.out_dir / "external")

    base = read_json(config.ROOT / "arms_sft/arm_W_full_external/eval_base/mixed_eval.json")
    d_h0 = read_json(args.d_adapter.parent / "eval_mixed/mixed_eval.json")

    summary = {"original": original, "external": external}
    for name in ("h_core", "h_full", "sys"):
        o = original[name]["accuracy"]
        e = external[name]["accuracy"]
        summary[name] = {
            "original_accuracy": o,
            "external_accuracy": e,
            "mixed_accuracy": round((o * 450 + e * 200) / 650, 4),
            "parse_fail_original": original[name]["parse_fail"],
            "parse_fail_external": external[name]["parse_fail"],
        }

    summary["base_h0"] = {
        "original_accuracy": base["original_accuracy"],
        "external_accuracy": base["external_accuracy"],
        "mixed_accuracy": base["mixed_accuracy"],
    }
    summary["d_h0"] = {
        "original_accuracy": d_h0["original_accuracy"],
        "external_accuracy": d_h0["external_accuracy"],
        "mixed_accuracy": d_h0["mixed_accuracy"],
    }
    summary["synergy"] = {
        "sys_vs_w_only": round(summary["sys"]["mixed_accuracy"] - summary["d_h0"]["mixed_accuracy"], 4),
        "sys_vs_h_full": round(summary["sys"]["mixed_accuracy"] - summary["h_full"]["mixed_accuracy"], 4),
        "sys_above_max_single": summary["sys"]["mixed_accuracy"] > max(
            summary["d_h0"]["mixed_accuracy"], summary["h_full"]["mixed_accuracy"]
        ),
    }
    summary["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[eval_dh5] " + json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
