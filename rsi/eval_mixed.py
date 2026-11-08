"""eval_mixed：在混合测试集上评估。

测试集 = 原 lockbox heldout（450 道自生成题库）+ 外部中文 GSM8K heldout（200 道）。

用法：
  python -m rsi.eval_mixed --adapter <path|none> --out-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from . import config, grade
from .engine import Engine
from .genome import H0


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").split("\n") if x.strip()]


def eval_lockbox(adapter: str | None, eval_n: int, out_dir: Path) -> dict:
    spec = {
        "variants": [{
            "name": "sys", "adapter": adapter, "genome": H0().to_dict()
        }],
        "eval_n": eval_n,
        "reps": 1,
        "seed": 20261108,
        "call_timeout_sec": 90,
        "task_timeout_sec": 180,
        "runtime_mode": "checked",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    spec_path = out_dir / "eval_spec.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    py = config.load_state().get("venv_python") or str(config.VENV_PYTHON)
    proc = subprocess.run(
        [py, str(config.LOCKBOX_DIR / "run_eval.py"),
         "--spec", str(spec_path), "--out-dir", str(out_dir / "lockbox")],
        cwd=str(config.ROOT),
    )
    if proc.returncode != 0:
        raise RuntimeError("lockbox eval failed")
    return json.loads((out_dir / "lockbox" / "eval.json").read_text(encoding="utf-8"))


def eval_external(adapter: str | None, heldout_path: Path, out_dir: Path) -> dict:
    recs = load_jsonl(heldout_path)
    engine = Engine(
        adapter_path=adapter,
        call_timeout_sec=90,
        task_timeout_sec=180,
    )
    h0 = H0()
    results = []
    resp_out = []
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
        try:
            resp, _ = engine.generate_one_checked(
                q, h0, shots=None,
                max_retry=h0.format_retry,
                timeout_sec=90,
            )
            timed_out = False
        except Exception as e:
            resp = ""
            timed_out = True
        r = grade.grade_one(ref, resp)
        r.update({"pid": str(i), "timed_out": timed_out})
        results.append(r)
        resp_out.append({"pid": str(i), "question": q, "response": resp,
                         "ref": ref, "correct": r["correct"]})
        if i % 50 == 0:
            print(f"[eval_mixed] external {i}/{len(recs)}", flush=True)
    engine.close()
    s = grade.summarize(results)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "external_eval.json").write_text(
        json.dumps({"n": len(results), "summary": s}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    (out_dir / "external_responses.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in resp_out) + "\n",
        encoding="utf-8",
    )
    return {"n": len(results), "accuracy": s["accuracy"],
            "parse_fail": s["parse_fail_rate"], "summary": s}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=str, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--eval-original-n", type=int, default=450)
    ap.add_argument("--external-heldout", type=Path,
                    default=config.ROOT / "data/external/gsm8k_zh_heldout.jsonl")
    args = ap.parse_args()

    adapter = None if args.adapter in (None, "", "none", "None") else args.adapter
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("[eval_mixed] 原 heldout 评估 ...", flush=True)
    lock = eval_lockbox(adapter, args.eval_original_n, out_dir)
    lock_acc = lock["variants"]["sys"]["accuracy_mean"]

    print("[eval_mixed] 外部中文 heldout 评估 ...", flush=True)
    ext = eval_external(adapter, args.external_heldout, out_dir)

    total_n = args.eval_original_n + ext["n"]
    combined = (lock_acc * args.eval_original_n + ext["accuracy"] * ext["n"]) / total_n
    summary = {
        "adapter": adapter,
        "original_n": args.eval_original_n,
        "original_accuracy": lock_acc,
        "external_n": ext["n"],
        "external_accuracy": ext["accuracy"],
        "external_parse_fail": ext["parse_fail"],
        "mixed_n": total_n,
        "mixed_accuracy": round(combined, 4),
        "elapsed_min": round((time.time() - t0) / 60, 1),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "mixed_eval.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[eval_mixed] " + json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
