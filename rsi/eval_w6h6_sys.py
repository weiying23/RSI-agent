"""eval_w6h6_sys：只评估 W6+H6 的 sys 变体，复用已有响应。"""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import config
from .genome import Genome, select_shots
from .eval_dh5 import eval_original, eval_external


def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").split("\n") if x.strip()]


def main() -> None:
    config.sanitize_proxy_env()
    state = config.load_state()
    py = state.get("venv_python") or str(config.VENV_PYTHON)
    out_root = config.ROOT / "arms_sft/coevolution_v1"
    h6_info = read_json(out_root / "h6_best.json")
    h6 = Genome.from_dict(h6_info["genome"])
    shot_pool = read_jsonl(config.ROOT / "arms_fixed/arm_H/shot_pool.jsonl")
    shots = select_shots(shot_pool, h6)
    variants = [{
        "name": "sys",
        "adapter": str(out_root / "adapters"),
        "genome": h6,
        "shots": shots,
    }]
    print("[eval_w6h6_sys] 原 held-out 450 评估 ...", flush=True)
    original = eval_original(variants, out_root / "eval", py)
    print("[eval_w6h6_sys] 外部中文 200 评估 ...", flush=True)
    external = eval_external(variants, config.ROOT / "data/external/gsm8k_zh_heldout.jsonl", out_root / "eval_external")
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
        "h6": h6_info,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_root / "sys_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[eval_w6h6_sys] " + json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
