"""lockbox/grade_heldout.py：外部审计通道（校验和 + 独立判分）。

不生成、不训练——只对「已有输出文件」重新判分，用于：
  - 事后人工复核（拿任意一份 responses 文件来验分）
  - 演示防篡改纪律：判分算法 = rsi.grade 同一实现，包装层只做校验和

用法：
  python lockbox/grade_heldout.py --outputs responses.jsonl [--eval-n 450]

outputs 每行：{"id": "...", "response": "..."}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lockbox.common import load_problems
from rsi import grade


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", type=Path, required=True)
    ap.add_argument("--eval-n", type=int, default=None)
    ap.add_argument("--write", type=Path, default=None)
    args = ap.parse_args()

    problems = {p["id"]: p for p in load_problems(args.eval_n)}
    outputs = [
        json.loads(ln)
        for ln in args.outputs.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]

    results = []
    for o in outputs:
        p = problems[o["id"]]
        r = grade.grade_one(p["answer_frac"], o["response"])
        r["pid"] = o["id"]
        results.append(r)

    s = grade.summarize(results)
    print(json.dumps(s, ensure_ascii=False, indent=2))

    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(
            json.dumps({"summary": s, "results": results}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        print(f"明细已写：{args.write}")


if __name__ == "__main__":
    main()
