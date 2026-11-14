"""prepare_warmstart_v19：把 1,350 条干净 SFT 数据拆成 80 条 shot pool + 1,270 条训练 replay。"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from . import config


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").split("\n") if x.strip()]


def write_jsonl(path: Path, recs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    src = config.ROOT / "data/sft_eval_format_v1"
    out = config.ROOT / "data/warmstart_v19"
    out.mkdir(parents=True, exist_ok=True)

    train = read_jsonl(src / "train.jsonl")
    valid = read_jsonl(src / "valid.jsonl")

    # 1) 按 template 分组
    by_t: dict[str, list[dict]] = defaultdict(list)
    for r in train:
        t = r.get("template") or "unknown"
        by_t[t].append(r)

    # 2) 每类选 8 条最短解答，保证 few-shot 简洁
    per_template = 8
    selected: list[dict] = []
    selected_ids: set[str] = set()
    for t, recs in by_t.items():
        recs = sorted(recs, key=lambda r: len(r["messages"][-1]["content"]))
        take = recs[:per_template]
        for r in take:
            if r.get("id") in selected_ids:
                continue
            selected_ids.add(r["id"])
            q = next(m["content"] for m in r["messages"] if m["role"] == "user")
            a = next(m["content"] for m in r["messages"] if m["role"] == "assistant")
            selected.append({
                "pid": r["id"],
                "question": q,
                "solution": a,
                "difficulty": 5,
                "template": t,
                "success_rate": 1.0,
                "gen": 0,
            })

    # 3) 轮转排列，让前几个 few-shot 来自不同模板
    by_t_sel: dict[str, list[dict]] = defaultdict(list)
    for s in selected:
        by_t_sel[s["template"]].append(s)
    order: list[dict] = []
    keys = sorted(by_t_sel)
    while any(by_t_sel[k] for k in keys):
        for k in keys:
            if by_t_sel[k]:
                order.append(by_t_sel[k].pop(0))

    # 4) 剩余 1,270 条作为训练 replay
    base_train = [r for r in train if r.get("id") not in selected_ids]

    write_jsonl(out / "shot_pool.jsonl", order)
    write_jsonl(out / "train.jsonl", base_train)
    write_jsonl(out / "valid.jsonl", valid)

    manifest = {
        "source_train": len(train),
        "source_valid": len(valid),
        "shot_pool": len(order),
        "train_replay": len(base_train),
        "per_template": per_template,
        "selection": "每个 template 选 8 条最短解答，并按 template 轮转排列",
        "seed": 20261115,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
