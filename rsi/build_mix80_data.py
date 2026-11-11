"""build_mix80_data：80% 自生成评估格式数据 + 20% 中文 GSM8K 对齐数据。"""
from __future__ import annotations

import argparse
import json
import random
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


def question(rec: dict) -> str:
    for m in rec["messages"]:
        if m["role"] == "user":
            return m["content"].strip()
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-dir", type=Path,
                    default=config.ROOT / "data/sft_eval_format_v1")
    ap.add_argument("--external", type=Path,
                    default=config.ROOT / "data/external/gsm8k_zh_train.jsonl")
    ap.add_argument("--out-dir", type=Path,
                    default=config.ROOT / "data/sft_mix80_20_v1")
    ap.add_argument("--seed", type=int, default=20261111)
    args = ap.parse_args()

    self_recs = read_jsonl(args.self_dir / "train.jsonl") + read_jsonl(args.self_dir / "valid.jsonl")
    ext_recs = read_jsonl(args.external)
    system = H0().instruction
    seen = set()
    self_clean = []
    for r in self_recs:
        q = question(r)
        if q and q not in seen:
            seen.add(q)
            self_clean.append(r)

    ext_clean = []
    for r in ext_recs:
        q = question(r)
        a = r["messages"][-1]["content"].strip()
        _, nbox = grade.extract_boxed(a)
        if not q or q in seen or nbox != 1 or len(a) < 100:
            continue
        seen.add(q)
        ext_clean.append({
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ],
            "translated": True,
        })

    n_ext = round(len(self_clean) * 0.2 / 0.8)
    rng = random.Random(args.seed)
    selected = rng.sample(ext_clean, min(n_ext, len(ext_clean)))
    all_recs = self_clean + selected
    rng.shuffle(all_recs)
    n_valid = max(1, int(len(all_recs) * 0.1))
    valid, train = all_recs[:n_valid], all_recs[n_valid:]

    write_jsonl(args.out_dir / "train.jsonl", train)
    write_jsonl(args.out_dir / "valid.jsonl", valid)
    manifest = {
        "self": len(self_clean), "external_selected": len(selected),
        "external_pool_after_filter": len(ext_clean),
        "train": len(train), "valid": len(valid),
        "self_ratio": round(len(self_clean)/(len(self_clean)+len(selected)), 4),
        "seed": args.seed,
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
