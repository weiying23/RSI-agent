"""convert_sft_format：把 RSR-lite 数据转换为与 held-out 评估一致的训练格式。

输入：arms_sft/arm_W_internal_1500/data/{train,valid}.jsonl
输出：data/sft_eval_format_v1/{train,valid,manifest}.jsonl/json

转换：
  1. user 删除 [解题提示]，只保留题目；
  2. assistant 删除“解题计划：...详细解答：”前缀，只保留最终详细解答；
  3. 开头加入与评估相同的 H0 system prompt；
  4. 用 rsr_pool 中的标准答案重新判分，并要求 lint 通过、单一 \boxed；
  5. 按题目去重、重新 90/10 切分。
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from . import config, grade
from .genome import H0


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").split("\n") if x.strip()]


def write_jsonl(path: Path, recs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def strip_hint(user: str) -> str:
    return user.split("\n\n[解题提示]", 1)[0].strip()


def strip_plan(assistant: str) -> str:
    for marker in ("详细解答：", "原始解答："):
        if marker in assistant:
            assistant = assistant.split(marker, 1)[1].strip()
            break
    else:
        assistant = assistant.strip()
    # 个别“详细解答”开头仍残留计划引导语，做最小限度清理，不改计算过程。
    for prefix in (
        "我们按照解题计划逐步推理：",
        "按照上述解题计划，",
        "按照解题计划，",
        "根据解题计划，",
    ):
        if assistant.startswith(prefix):
            assistant = assistant[len(prefix):].lstrip()
    return assistant


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path,
                    default=config.ROOT / "arms_sft/arm_W_internal_1500/data")
    ap.add_argument("--pool", type=Path,
                    default=config.ROOT / "arms_sft/arm_W_internal_1500/rsr_pool.jsonl")
    ap.add_argument("--out-dir", type=Path,
                    default=config.ROOT / "data/sft_eval_format_v1")
    ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--valid-frac", type=float, default=0.1)
    args = ap.parse_args()

    pool = {p["id"]: p for p in read_jsonl(args.pool)}
    raw = read_jsonl(args.input_dir / "train.jsonl") + read_jsonl(args.input_dir / "valid.jsonl")
    out: list[dict] = []
    seen: set[str] = set()
    rejected = Counter()
    system = H0().instruction

    for rec in raw:
        pid = rec.get("id")
        if pid not in pool:
            rejected["missing_pool_answer"] += 1
            continue
        messages = rec.get("messages", [])
        if len(messages) < 2:
            rejected["bad_messages"] += 1
            continue
        question = strip_hint(messages[0]["content"])
        solution = strip_plan(messages[-1]["content"])
        if not question or not solution:
            rejected["empty_content"] += 1
            continue
        if question in seen:
            rejected["duplicate_question"] += 1
            continue

        result = grade.grade_one(pool[pid]["answer_frac"], solution)
        if not result["correct"]:
            rejected["wrong_after_conversion"] += 1
            continue
        ok, reasons = grade.lint_one(solution, result)
        if not ok:
            rejected["lint_" + next((k for k, v in reasons.items() if v), "failed")] += 1
            continue

        seen.add(question)
        out.append({
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": question},
                {"role": "assistant", "content": solution},
            ],
            "id": pid,
            "template": rec.get("template"),
        })

    rng = random.Random(args.seed)
    rng.shuffle(out)
    n_valid = max(1, int(len(out) * args.valid_frac))
    valid = out[:n_valid]
    train = out[n_valid:]

    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "valid.jsonl", valid)
    manifest = {
        "input_records": len(raw),
        "output_records": len(out),
        "train_records": len(train),
        "valid_records": len(valid),
        "rejected": dict(rejected),
        "format": "system=H0, user=question_only, assistant=direct_solution_with_boxed",
        "pool": str(args.pool),
        "seed": args.seed,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
