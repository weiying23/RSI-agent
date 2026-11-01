"""baseline：W_base 在全量 held-out 上的 t=0 锚点（P0 收尾）。

用法：
  python -m rsi.baseline --eval-n 450 --reps 1

产出 state/baseline/eval.json：主曲线的 t=0 基线 + hack 指标基线。
"""
from __future__ import annotations

import argparse
import json
import subprocess

from . import config
from .genome import H0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-n", type=int, default=None)
    ap.add_argument("--reps", type=int, default=1)
    args = ap.parse_args()

    state = config.load_state()
    if not state.get("frozen", {}).get("heldout"):
        raise SystemExit("held-out 未冻结")

    out_dir = config.STATE_DIR / "baseline"
    spec = {
        "variants": [{"name": "BASE_t0", "adapter": None, "genome": H0().to_dict()}],
        "eval_n": args.eval_n,
        "reps": args.reps,
        "seed": 0,
    }
    spec_path = out_dir / "eval_spec.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")

    venv_py = state.get("venv_python") or str(config.VENV_PYTHON)
    proc = subprocess.run(
        [venv_py, str(config.LOCKBOX_DIR / "run_eval.py"),
         "--spec", str(spec_path), "--out-dir", str(out_dir)],
        cwd=str(config.ROOT),
    )
    if proc.returncode != 0:
        raise SystemExit("基线评估失败")
    rep = json.loads((out_dir / "eval.json").read_text(encoding="utf-8"))
    v = rep["variants"]["BASE_t0"]
    print(f"\n[baseline] t=0 锚点：accuracy = {v['accuracy_mean']:.4f} ± {v['accuracy_std']:.4f}")
    print(f"[baseline] 报告 → {out_dir / 'eval.json'}")


if __name__ == "__main__":
    main()
