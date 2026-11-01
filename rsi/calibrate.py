"""calibrate：基座难度标定（P0，一次性）。

把基座模型在 H_0 出厂基因组下的准确率标到 50–70% 区间
（太低无法自举，太高无提升空间——HANDOFF §3.8）。

用法：
  python -m rsi.calibrate --levels 2,3,4 --n 40

输出 state/calibration.json：逐级准确率 + 逐模板分解 + tok/s 实测。
标定集用后烧毁（角色 calib，全局账本排除，永不复用）。
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

from . import config, env_gen, grade
from .engine import Engine
from .genome import H0


def run_level(engine: Engine, d: int, n: int) -> dict:
    probs = env_gen.sample(n, d, role="calib", arm="GLOBAL", extra_seed=f"calib-d{d}")
    samples = engine.generate_batch(probs, H0(), k=1)
    # 全轨迹归档（纪律§5）：没有样本就无法复核标定
    out_samples = config.STATE_DIR / f"calibration_samples_d{d}.jsonl"
    config.STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_samples, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    prob_by_id = {p["id"]: p for p in probs}
    results = []
    for s in samples:
        r = grade.grade_one(prob_by_id[s["pid"]]["answer_frac"], s["response"])
        r["pid"] = s["pid"]
        r["template"] = s.get("template")
        results.append(r)
    summary = grade.summarize(results)
    per_template: dict[str, list[bool]] = {}
    for r in results:
        per_template.setdefault(r["template"], []).append(r["correct"])
    return {
        "difficulty": d,
        "n": n,
        "accuracy": summary["accuracy"],
        "hack": summary,
        "per_template": {
            k: round(statistics.mean(v), 3) for k, v in sorted(per_template.items())
        },
        "engine_stats": engine.stats(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=str, default="2,3,4")
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()
    levels = [int(x) for x in args.levels.split(",")]

    print(f"[calibrate] 基座难度标定：levels={levels} n={args.n}（H_0 出厂基因组）")
    engine = Engine()
    print(f"[calibrate] 模型加载 {engine.load_seconds:.1f}s")
    reports = []
    for d in levels:
        t0 = time.time()
        rep = run_level(engine, d, args.n)
        rep["wall_minutes"] = round((time.time() - t0) / 60, 1)
        reports.append(rep)
        print(
            f"[calibrate] 难度 {d}: accuracy={rep['accuracy']:.3f} "
            f"({rep['wall_minutes']}min, {rep['engine_stats']['tok_per_sec']} tok/s)"
        )
        for t, acc in rep["per_template"].items():
            print(f"    {t:16s} {acc:.2f}")

    # 推荐：最接近 60% 的级别（目标区间 50–70%）
    in_range = [r for r in reports if 0.5 <= r["accuracy"] <= 0.7]
    pool = in_range or reports
    rec = min(pool, key=lambda r: abs(r["accuracy"] - 0.6))

    out = {
        "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "levels": reports,
        "recommended_difficulty": rec["difficulty"],
        "recommended_accuracy": rec["accuracy"],
        "target_band": [0.5, 0.7],
        "note": "冻结前请人工确认 recommended_difficulty，然后运行 lockbox/freeze.py",
    }
    config.CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.CALIBRATION_FILE.write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"\n[calibrate] 推荐难度：{rec['difficulty']}（accuracy={rec['accuracy']:.3f}）")
    print(f"[calibrate] 报告已写：{config.CALIBRATION_FILE}")
    print("[calibrate] 下一步：确认后运行 python lockbox/freeze.py --difficulty <d>")


if __name__ == "__main__":
    main()
