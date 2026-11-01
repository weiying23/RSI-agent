"""lockbox/freeze.py：t=0 冻结 held-out（P0 一次性操作，之后永不再跑）。

流程：采样 450 题（角色=heldout，全局账本）→ 写 jsonl → 写 SHA256 →
state 标记 frozen。难度参数由 rsi.calibrate 的结果人工确认后传入——
冻结是一个用户可见的决策点，不静默执行。

用法：
  python lockbox/freeze.py --difficulty 3 --n 450
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lockbox.common import HELDOUT_FILE, HELDOUT_SHA, sha256_of
from rsi import config, env_gen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--difficulty", type=int, required=True, choices=range(1, 6))
    ap.add_argument("--n", type=int, default=450)
    args = ap.parse_args()

    state = config.load_state()
    if state.get("frozen", {}).get("heldout"):
        raise SystemExit("held-out 已冻结（state.frozen.heldout=true），拒绝重复冻结。")

    if HELDOUT_FILE.exists():
        raise SystemExit(f"{HELDOUT_FILE} 已存在，拒绝覆盖。")

    print(f"[freeze] 采样 {args.n} 题，难度 {args.difficulty}，角色 heldout …")
    problems = env_gen.sample(
        args.n, args.difficulty, role="heldout", arm="GLOBAL", extra_seed="heldout-v1"
    )

    env_gen.write_jsonl(HELDOUT_FILE, problems)
    digest = sha256_of(HELDOUT_FILE)
    HELDOUT_SHA.write_text(f"{digest}  heldout_v1.jsonl\n", encoding="utf-8")

    state["difficulty"] = args.difficulty
    state["heldout_size"] = len(problems)
    state["frozen"] = {"heldout": True, "sha256": digest, "frozen_at_levels": args.difficulty}
    config.save_state(state)

    dist = Counter(p["template"] for p in problems)
    print(f"[freeze] 完成：{len(problems)} 题已冻结")
    print(f"[freeze] sha256 = {digest}")
    print(f"[freeze] 模板分布：{dict(dist)}")
    print("[freeze] state.frozen.heldout = true（此后任何代码不得改动 heldout）")


if __name__ == "__main__":
    main()
