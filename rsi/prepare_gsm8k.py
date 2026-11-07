"""prepare_gsm8k：下载 GSM8K train 并转成 test-RSI 训练 messages 格式。

转换规则：
- 只取 train split，test 留作评测；
- user 为原始英文题；
- assistant 为推理过程，去掉 <<expr=result>> 计算注释；
- GSM8K 的最终答案 "#### 72" 改写为 "\\boxed{72}"，与 rsi/grade.py 接口一致；
- 输出 data/external/gsm8k_train.jsonl。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time

from . import config


def clean_reasoning(text: str) -> str:
    text = re.sub(r"<<[^>]*>>", "", text)
    text = text.strip()
    return text


def convert_answer(answer: str) -> str:
    if "####" not in answer:
        return clean_reasoning(answer)

    reasoning, _, final_part = answer.partition("####")
    reasoning = clean_reasoning(reasoning)
    # 取 final line 里的数字部分；GSM8K 标准格式几乎是纯数字，但保留简单清理。
    final = final_part.strip().splitlines()[0].strip() if final_part.strip() else ""
    final = re.sub(r"[\$,]", "", final).strip()
    if not final:
        return reasoning
    return reasoning + f"\n\n\\boxed{{{final}}}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=None,
                    help="输出 JSONL 路径；默认 data/external/gsm8k_train.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="只转前 N 条，0 表示全部")
    args = ap.parse_args()

    config.sanitize_proxy_env()
    os.environ.setdefault("HF_HOME", "/tmp/hf_home")
    os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf_cache")
    os.environ["HF_HUB_OFFLINE"] = "0"

    from datasets import load_dataset

    print("[prepare_gsm8k] 加载 openai/gsm8k train ...")
    ds = load_dataset("openai/gsm8k", "main", split="train")
    if args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"[prepare_gsm8k] 共 {len(ds)} 条")

    out = args.out or str(config.ROOT / "data" / "external" / "gsm8k_train.jsonl")
    out_path = config.ROOT / out if not out.startswith("/") else __import__("pathlib").Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for ex in ds:
            q = ex["question"].strip()
            a = convert_answer(ex["answer"])
            if not q or not a:
                continue
            rec = {
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ]
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    print(f"[prepare_gsm8k] 写入 {count} 条 -> {out_path}")
    print(f"[prepare_gsm8k] 用时 {time.time()-start:.1f}s")


if __name__ == "__main__":
    main()
