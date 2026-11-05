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
    arm_roots = [ROOT / "arms", ROOT / "arms_fixed"]
    arm_dirs = []
    for r in arm_roots:
        if r.exists():
            arm_dirs.extend(sorted(r.glob("arm_*")))
    for arm_dir in arm_dirs:
        for gen_dir in sorted(arm_dir.glob("gen_*")):
            phases = sum(1 for p in ("A", "B", "C", "D")
                         if (gen_dir / f".phase_{p}_done").exists())
            artifacts = {
                "genome": (gen_dir / "genome.json").exists(),
                "samples": (gen_dir / "samples.jsonl").exists(),
                "adapters": (gen_dir / "adapters" / "adapters.safetensors").exists(),
                "eval": (gen_dir / "eval" / "eval.json").exists(),
                "lineage": (gen_dir / "lineage.json").exists(),
            }
            age_min = (time.time() - gen_dir.stat().st_mtime) / 60
            gens.append(
                {
                    "arm": arm_dir.name,
                    "gen": gen_dir.name,
                    "phases": f"{phases}/4",
                    "artifacts": artifacts,
                    "idle_min": round(age_min, 1),
                    "stalled": age_min > 90 and phases < 4 and not artifacts["lineage"],
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
