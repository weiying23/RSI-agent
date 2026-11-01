#!/usr/bin/env python3
"""status.py：外部 watchdog（Super-agent 思想的 10 行版）。

回答三个问题：账本状态如何、哪些世代目录还活着、有没有卡死。
用法：python status.py [--watch 秒]
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def snapshot() -> dict:
    from rsi import ledger

    out: dict = {"time": datetime.now().isoformat(timespec="seconds")}

    st = ROOT / "state" / "config.json"
    out["state"] = json.loads(st.read_text()) if st.exists() else "未初始化"
    out["ledger"] = ledger.stats()

    gens = []
    for arm_dir in sorted((ROOT / "arms").glob("arm_*")):
        for gen_dir in sorted(arm_dir.glob("gen_*")):
            markers = {
                "genome": (gen_dir / "genome.json").exists(),
                "samples": (gen_dir / "samples.jsonl").exists(),
                "train": (gen_dir / "train.jsonl").exists(),
                "adapters": (gen_dir / "adapters" / "adapters.safetensors").exists(),
                "eval": (gen_dir / "eval" / "eval.json").exists(),
                "lineage": (gen_dir / "lineage.json").exists(),
            }
            age_min = (time.time() - gen_dir.stat().st_mtime) / 60
            stage = sum(markers.values())
            gens.append(
                {
                    "arm": arm_dir.name,
                    "gen": gen_dir.name,
                    "stage": f"{stage}/6",
                    "idle_min": round(age_min, 1),
                    "stalled": age_min > 90 and stage < 6,
                }
            )
    out["generations"] = gens
    return out


def main() -> None:
    watch = 0
    if "--watch" in sys.argv:
        watch = int(sys.argv[sys.argv.index("--watch") + 1])
    while True:
        snap = snapshot()
        print(json.dumps(snap, ensure_ascii=False, indent=1))
        stalled = [g for g in snap["generations"] if g["stalled"]]
        if stalled:
            print(f"⚠ 卡死嫌疑：{stalled}", file=sys.stderr)
        if not watch:
            return
        time.sleep(watch)


if __name__ == "__main__":
    main()
