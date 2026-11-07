"""translate_gsm8k：用本地 Qwen 把筛选后的 GSM8K 对齐题翻译成中文。

输出 data/external/gsm8k_aligned_zh.jsonl。
- 题目翻译成中文；
- 解题过程翻译成中文；
- 保留并校验 \\boxed{...} 格式；
- 翻译失败/超时则回退使用原英文条目，并打上 fallback 标记。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import config, grade
from .engine import Engine
from .genome import Genome
from .timeouts import GenerationTimeout


def _translate(engine: Engine, system: str, text: str, max_tokens: int = 512) -> str | None:
    g = Genome(
        instruction=system,
        k_shots=0,
        temperature=0.3,
        top_p=0.9,
        max_tokens=max_tokens,
        retry_k=1,
        format_retry=0,
        self_consistency_k=1,
    )
    try:
        resp, _ = engine.generate_one(text, g, timeout_sec=90.0)
        return resp.strip()
    except GenerationTimeout:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_file", type=Path,
                    default=config.ROOT / "data/external/gsm8k_aligned.jsonl")
    ap.add_argument("--out", type=Path,
                    default=config.ROOT / "data/external/gsm8k_aligned_zh.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    config.sanitize_proxy_env()
    recs = [json.loads(x) for x in Path(args.in_file).read_text(encoding="utf-8").split("\n") if x.strip()]
    if args.limit:
        recs = recs[: args.limit]
    print(f"[translate_gsm8k] 待翻译 {len(recs)} 条")

    engine = Engine()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    ok = 0
    fallback = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i, rec in enumerate(recs, 1):
            q_en = ""
            a_en = ""
            for m in rec["messages"]:
                if m["role"] == "user":
                    q_en = m["content"]
                elif m["role"] == "assistant":
                    a_en = m["content"]
            if not q_en or not a_en:
                continue

            q_zh = _translate(
                engine,
                "你是数学题翻译专家。把下面的英文数学题翻译成简洁、准确的中文数学题。只输出题目，不要解答，不要解释。",
                q_en,
            )
            a_zh = _translate(
                engine,
                "你是数学解题过程翻译专家。把下面的英文解题过程翻译成中文。保留所有数学公式，并且必须原样保留 \\boxed{...} 答案格式。只输出翻译后的解题过程。",
                a_en,
            )

            if q_zh and a_zh:
                # 校验 \boxed{} 格式；丢失则从英文原文找回答案并附加。
                boxed, n = grade.extract_boxed(a_zh)
                if n <= 0 or boxed is None:
                    orig_boxed, orig_n = grade.extract_boxed(a_en)
                    if orig_n > 0 and orig_boxed is not None:
                        a_zh = a_zh.strip() + f"\n\n\\boxed{{{orig_boxed}}}"
                    else:
                        a_zh = None
            if not (q_zh and a_zh):
                q_zh, a_zh = q_en, a_en
                fallback += 1
            else:
                ok += 1

            out_rec = {
                "messages": [
                    {"role": "user", "content": q_zh},
                    {"role": "assistant", "content": a_zh},
                ],
                "translated": bool(q_zh != q_en),
                "fallback": q_zh == q_en,
            }
            f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

            if i % 100 == 0:
                el = (time.time() - t0) / 60
                print(f"[translate_gsm8k] {i}/{len(recs)} ok={ok} fallback={fallback} {el:.1f}min", flush=True)

    el = (time.time() - t0) / 60
    print(f"[translate_gsm8k] 完成：ok={ok} fallback={fallback} 用时 {el:.1f} min")
    print(f"[translate_gsm8k] 写入 -> {out_path}")


if __name__ == "__main__":
    main()
