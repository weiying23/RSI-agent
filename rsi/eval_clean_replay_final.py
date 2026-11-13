"""eval_clean_replay_final：从原始 W0/H0 开始的干净 replay，最终统一650题评估。"""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import config
from .genome import H0
from .eval_dh5 import eval_original, eval_external


def main() -> None:
    config.sanitize_proxy_env()
    state = config.load_state()
    py = state.get("venv_python") or str(config.VENV_PYTHON)
    out_root = config.ROOT / "arms_clean_replay/final_eval"
    out_root.mkdir(parents=True, exist_ok=True)

    adapter = config.ROOT / "arms_clean_replay/arm_WH/gen_002/adapters"
    variants = [{"name": "sys", "adapter": str(adapter), "genome": H0(), "shots": []}]

    print("[clean_replay_final] 原 held-out 450 评估 ...", flush=True)
    original = eval_original(variants, out_root / "original", py)
    print("[clean_replay_final] 外部中文 200 评估 ...", flush=True)
    external = eval_external(variants, config.ROOT / "data/external/gsm8k_zh_heldout.jsonl", out_root / "external")

    o = original["sys"]["accuracy"]
    e = external["sys"]["accuracy"]
    summary = {
        "sys": {
            "original_accuracy": o,
            "external_accuracy": e,
            "mixed_accuracy": round((o * 450 + e * 200) / 650, 4),
            "parse_fail_original": original["sys"]["parse_fail"],
            "parse_fail_external": external["sys"]["parse_fail"],
        },
        "baseline_h0": {
            "original_accuracy": 0.60,
            "external_accuracy": 0.535,
            "mixed_accuracy": 0.58,
        },
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[clean_replay_final] " + json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
