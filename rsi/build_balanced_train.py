"""build_balanced_train：组装均衡的 SFT 训练集。

来源：
  - 自生成中文 verified 数据（带 hints），默认 arms_hints/arm_W/train_cum.jsonl
  - 翻译后的中文 GSM8K 对齐数据，默认 data/external/gsm8k_aligned_zh.jsonl

规则：
  - 只保留中文数据；
  - 外部数据占比默认 35%，其余为自生成数据；
  - 去重（按 user question）；
  - 按 90/10 切 train/valid；
  - 输出 data/sft_balanced_v1/{train.jsonl,valid.jsonl,manifest.json}。
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from . import config


TEMPLATE_RULES_ZH = {
    "percent": [r"百分数", r"百分比", r"%"],
    "discount": [r"折扣", r"折"],
    "travel": [r"行驶", r"速度", r"千米", r"行程", r"距离"],
    "average": [r"平均数", r"平均"],
    "rect_area": [r"长方形", r"面积", r"长厘米", r"宽厘米"],
    "frac_addsub": [r"分数", r"通分", r"分母", r"分子"],
    "lcm": [r"最小公倍数", r"LCM"],
    "gcd": [r"最大公约数", r"GCD"],
    "mult_chain": [r"连续乘法", r"相乘", r"×"],
    "chicken_rabbit": [r"鸡", r"兔", r"腿"],
}


def infer_template_zh(text: str) -> str:
    for t, patterns in TEMPLATE_RULES_ZH.items():
        for pat in patterns:
            if re.search(pat, text):
                return t
    return "unknown"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").split("\n") if x.strip()]


def user_text(rec: dict) -> str:
    for m in rec.get("messages", []):
        if m["role"] == "user":
            return m["content"].strip()
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-data", type=Path,
                    default=config.ROOT / "arms_hints/arm_W/train_cum.jsonl")
    ap.add_argument("--external-data", type=Path,
                    default=config.ROOT / "data/external/gsm8k_aligned_zh.jsonl")
    ap.add_argument("--rsr-data", type=Path, default=None,
                    help="RSR 增强中文数据，可选")
    ap.add_argument("--balance-templates", action="store_true",
                    help="外部数据按题型配额采样，避免某类题过多")
    ap.add_argument("--external-ratio", type=float, default=0.35)
    ap.add_argument("--out-dir", type=Path,
                    default=config.ROOT / "data/sft_balanced_v1")
    ap.add_argument("--seed", type=int, default=20261108)
    ap.add_argument("--valid-frac", type=float, default=0.1)
    args = ap.parse_args()

    self_recs = load_jsonl(args.self_data)
    if args.rsr_data and Path(args.rsr_data).exists():
        rsr_recs = load_jsonl(args.rsr_data)
        self_recs = self_recs + rsr_recs
    ext_recs = load_jsonl(args.external_data)

    seen: set[str] = set()
    clean_self: list[dict] = []
    for rec in self_recs:
        q = user_text(rec)
        if not q or q in seen:
            continue
        seen.add(q)
        clean_self.append(rec)

    # 外部数据去重 + 去掉与自生成重复的题
    clean_ext: list[dict] = []
    for rec in ext_recs:
        q = user_text(rec)
        if not q or q in seen:
            continue
        seen.add(q)
        clean_ext.append(rec)

    # 目标外部数量 = 外部占比 / 自生成占比 * 自生成数量
    n_ext = int(round(len(clean_self) * args.external_ratio / max(1e-6, 1.0 - args.external_ratio)))
    n_ext = min(n_ext, len(clean_ext))
    rng = random.Random(args.seed)
    if args.balance_templates:
        by_tpl = defaultdict(list)
        for rec in clean_ext:
            tpl = infer_template_zh(user_text(rec))
            by_tpl[tpl].append(rec)
        tpls = list(by_tpl)
        per_tpl = max(1, n_ext // len(tpls)) if tpls else 0
        ext_selected = []
        for tpl in tpls:
            pool = by_tpl[tpl]
            take = min(len(pool), per_tpl)
            ext_selected += rng.sample(pool, take) if take else []
        # 配额外用剩余补充，优先模板数量少的
        remaining = n_ext - len(ext_selected)
        if remaining > 0:
            used_ids = {id(rec) for rec in ext_selected}
            rest = [rec for rec in clean_ext if id(rec) not in used_ids]
            if rest:
                ext_selected += rng.sample(rest, min(remaining, len(rest)))
    else:
        ext_selected = rng.sample(clean_ext, n_ext) if n_ext else []

    all_recs = clean_self + ext_selected
    rng.shuffle(all_recs)

    n_valid = max(1, int(len(all_recs) * args.valid_frac))
    valid_recs = all_recs[:n_valid]
    train_recs = all_recs[n_valid:]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "train.jsonl").open("w", encoding="utf-8") as f:
        for rec in train_recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with (out_dir / "valid.jsonl").open("w", encoding="utf-8") as f:
        for rec in valid_recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    manifest = {
        "self_source": str(args.self_data),
        "external_source": str(args.external_data),
        "self_count": len(clean_self),
        "external_count": len(clean_ext),
        "external_selected": n_ext,
        "train_count": len(train_recs),
        "valid_count": len(valid_recs),
        "external_ratio": args.external_ratio,
        "seed": args.seed,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[build_balanced_train]", json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
