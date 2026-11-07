"""filter_gsm8k：从 GSM8K 中筛出与自生成模板相似的题。

规则是关键词/短语匹配，不调用 LLM。输出仍为 messages 格式，
且最终答案改写为 \\boxed{} 形式。每个样例可命中多个模板。
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from . import config
from .prepare_gsm8k import convert_answer


RULES = {
    "mult_chain": [r"\btimes\b", r"\bmultiplied by\b", r"\beach\b"],
    "frac_addsub": [r"\bfractions?\b", r"\bhalf\b", r"\bquarter\b", r"\bof a pizza", r"\bof a cake"],
    "lcm": [r"\bleast\s+common\s+multiple\b", r"\bLCM\b", r"\brepeated\s+every\b"],
    "gcd": [r"\bgreatest\s+common\s+(divisor|factor)\b", r"\bGCD\b", r"\bGCF\b"],
    "percent": [r"\bpercent\b", r"\b%\b", r"\binterest\b", r"\btax\b", r"\btip\b"],
    "average": [r"\baverage\b", r"\bmean\b", r"\bper\s+day\b", r"\bper\s+week\b"],
    "travel": [r"\bmiles\s+per\s+hour\b", r"\bkm\s+per\s+hour\b", r"\bspeed\b", r"\bdistance\b", r"\btravels?\b"],
    "rect_area": [r"\barea\b", r"\brectangle\b", r"\bperimeter\b", r"\bsquare\s+(feet|meters|inches|yards)\b", r"\bwide\b", r"\blong\b"],
    "discount": [r"\bdiscount\b", r"\bsale\s+price\b", r"\boriginal\s+price\b", r"\b% off\b", r"\bmarked down\b"],
    "chicken_rabbit": [r"\bchicken", r"\brabbit", r"\blegs?\b", r"\bhens?\b", r"\bcows?\b"],
}


def match_types(question: str) -> list[str]:
    q = question.lower()
    hits = []
    for t, patterns in RULES.items():
        if any(re.search(p, q) for p in patterns):
            hits.append(t)
    return hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_file", type=Path,
                    default=config.ROOT / "data/external/gsm8k_train.jsonl")
    ap.add_argument("--out", type=Path,
                    default=config.ROOT / "data/external/gsm8k_matched.jsonl")
    ap.add_argument("--min-hits", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    in_path = Path(args.in_file)
    out_path = Path(args.out)
    matched = []
    type_counts = {t: 0 for t in RULES}

    for line in in_path.read_text(encoding="utf-8").split("\n"):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        q = ""
        a = ""
        for m in rec["messages"]:
            if m["role"] == "user":
                q = m["content"]
            elif m["role"] == "assistant":
                a = m["content"]
        hits = match_types(q)
        if len(hits) >= args.min_hits:
            for h in hits:
                type_counts[h] += 1
            out_rec = {
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ],
                "match_types": hits,
            }
            matched.append(out_rec)
        if args.limit and len(matched) >= args.limit:
            break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in matched:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[filter_gsm8k] 匹配 {len(matched)} / 7473 条")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t:16s} {c}")
    print(f"[filter_gsm8k] 写入 -> {out_path}")


if __name__ == "__main__":
    main()
