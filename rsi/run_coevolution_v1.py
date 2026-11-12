"""run_coevolution_v1：从 (D,H5) 启动一代真正的权重×harness共进化。

流程：
  1. 冻结 D，搜索 H6
  2. 用 D+H6 生成新数据
  3. 训练 W6
  4. 在650题混合测试集上评估 W6×H6
  5. 生成 v16 文档并推送 GitHub
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from pathlib import Path

from . import config, distill, env_gen, grade, train
from .engine import Engine
from .genome import Genome, H0, select_shots
from .orchestrator import self_rewrite_instruction
from .eval_dh5 import eval_original, eval_external


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").split("\n") if x.strip()]


def write_jsonl(path: Path, recs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def evaluate_candidate(
    engine: Engine,
    genome: Genome,
    problems: list[dict],
    shot_pool: list[dict],
    tag: str,
    out_dir: Path,
) -> dict:
    """在 probe 上评估一个 harness 候选。"""
    shots = select_shots(shot_pool, genome)
    results = []
    responses = []
    t0 = time.time()
    for i, p in enumerate(problems, 1):
        try:
            resp, _ = engine.generate_one_checked(
                p["question"], genome, shots=shots,
                max_retry=genome.format_retry,
                timeout_sec=90,
            )
            timed_out = False
        except Exception:
            resp, timed_out = "", True
        r = grade.grade_one(p["answer_frac"], resp)
        r.update({"pid": p["id"], "timed_out": timed_out})
        results.append(r)
        responses.append({"pid": p["id"], "response": resp, "timed_out": timed_out})
    summary = grade.summarize(results)
    summary["composite"] = round(summary["accuracy"] - 0.25 * summary["parse_fail_rate"], 4)
    summary["elapsed_min"] = round((time.time() - t0) / 60, 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{tag}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_jsonl(out_dir / f"{tag}_responses.jsonl", responses)
    return summary


def build_h6_candidates(h5: Genome, failures: list[dict], engine: Engine, rng: random.Random):
    candidates = [
        ("h5_baseline", h5),
        ("h6_kshots3", h5.__class__(**{**h5.to_dict(), "k_shots": 3})),
        ("h6_sc3", h5.__class__(**{**h5.to_dict(), "self_consistency_k": 3})),
        ("h6_temp05", h5.__class__(**{**h5.to_dict(), "temperature": 0.5})),
    ]
    if failures:
        try:
            g_self = self_rewrite_instruction(engine, h5, failures, rng)
            candidates.append(("h6_self_rewrite", g_self))
        except Exception as e:
            print(f"[coevolve] self_rewrite 失败：{e}", flush=True)
    return candidates


def add_system_h0(records: list[dict]) -> list[dict]:
    out = []
    for rec in records:
        msgs = rec["messages"]
        if msgs and msgs[0]["role"] != "system":
            msgs = [{"role": "system", "content": H0().instruction}] + msgs
        out.append({**rec, "messages": msgs})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-adapter", type=Path,
                    default=config.ROOT / "arms_sft/next_plan/D_seed2_scale20/adapters")
    ap.add_argument("--h5-dir", type=Path,
                    default=config.ROOT / "arms_fixed/arm_H/gen_005")
    ap.add_argument("--out-root", type=Path,
                    default=config.ROOT / "arms_sft/coevolution_v1")
    ap.add_argument("--probe-n", type=int, default=30)
    ap.add_argument("--new-pool", type=int, default=100)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=20261113)
    args = ap.parse_args()

    config.sanitize_proxy_env()
    state = config.load_state()
    py = state.get("venv_python") or str(config.VENV_PYTHON)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 1) 读取 H5 和 shot pool
    h5 = Genome.from_dict(read_json(args.h5_dir / "genome.json"))
    eval_spec = read_json(args.h5_dir / "eval_spec.json")
    shots = next(v.get("shots", []) for v in eval_spec["variants"] if v["name"] == "h_full")
    shot_pool = read_jsonl(config.ROOT / "arms_fixed/arm_H/shot_pool.jsonl")

    # 2) 冻结 D，搜索 H6
    print("[coevolve] 加载 D 权重 ...", flush=True)
    engine = Engine(adapter_path=str(args.d_adapter), call_timeout_sec=90, task_timeout_sec=180)

    # probe：从固定题库 train split 5 随机抽 30 题，D 未在这些题上训练过
    all_probe = taskbank_probe = __import__("rsi.taskbank", fromlist=["load_split"]).load_split("train", 5)
    rng = random.Random(args.seed)
    rng.shuffle(all_probe)
    probe = all_probe[: args.probe_n]
    write_jsonl(out_root / "h6_probe.jsonl", probe)

    # 先评估 H5 baseline，收集失败样例给 self_rewrite
    print("[coevolve] 评估 H5 baseline ...", flush=True)
    base_summary = evaluate_candidate(engine, h5, probe, shot_pool, "h5_baseline", out_root / "h6_search")
    print(f"[coevolve] H5 baseline: acc={base_summary['accuracy']:.4f} "
          f"parse_fail={base_summary['parse_fail_rate']:.4f} "
          f"composite={base_summary['composite']:.4f}", flush=True)

    # 从 baseline 响应中提取失败样例
    failures = []
    base_responses = read_jsonl(out_root / "h6_search/h5_baseline_responses.jsonl")
    prob_by_id = {p["id"]: p for p in probe}
    for rec in base_responses:
        p = prob_by_id[rec["pid"]]
        r = grade.grade_one(p["answer_frac"], rec["response"])
        if not r["correct"] and r.get("extracted"):
            failures.append({
                "question": p["question"],
                "model_answer": r["extracted"][:40],
                "answer": p["answer"],
            })
        if len(failures) >= 5:
            break
    print(f"[coevolve] 收集到 {len(failures)} 个失败样例", flush=True)

    # 生成候选并评估
    candidates = build_h6_candidates(h5, failures, engine, rng)
    results = {"h5_baseline": {"genome": h5.to_dict(), "summary": base_summary}}
    for tag, cand in candidates[1:]:
        print(f"[coevolve] 评估 {tag} ...", flush=True)
        s = evaluate_candidate(engine, cand, probe, shot_pool, tag, out_root / "h6_search")
        results[tag] = {"genome": cand.to_dict(), "summary": s}
        print(f"[coevolve] {tag}: acc={s['accuracy']:.4f} "
              f"parse_fail={s['parse_fail_rate']:.4f} "
              f"composite={s['composite']:.4f}", flush=True)

    # 选择 composite 最高的候选
    best_tag = max(results, key=lambda k: results[k]["summary"]["composite"])
    best_h6 = Genome.from_dict(results[best_tag]["genome"])
    print(f"[coevolve] 最佳 H6: {best_tag} composite={results[best_tag]['summary']['composite']:.4f}", flush=True)
    (out_root / "h6_best.json").write_text(
        json.dumps({"tag": best_tag, "genome": best_h6.to_dict(),
                    "summary": results[best_tag]["summary"]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 3) 用 D+H6 生成新数据
    print(f"[coevolve] 生成 {args.new_pool} 道新题 ...", flush=True)
    new_probs = env_gen.sample(
        args.new_pool, state.get("difficulty", 5), role="pool", arm="coevolve_w6",
        extra_seed=f"coevolve-w6:{args.seed}",
    )
    write_jsonl(out_root / "w6_pool.jsonl", new_probs)

    print("[coevolve] 用 D + H6 生成解答 ...", flush=True)
    new_samples = engine.generate_batch(new_probs, best_h6, shot_pool=shot_pool)
    write_jsonl(out_root / "w6_samples.jsonl", new_samples)
    engine.close()

    print("[coevolve] 蒸馏新数据 ...", flush=True)
    dis = distill.distill(new_probs, new_samples, gen=6, seed=args.seed, with_hints=False)
    new_train = add_system_h0(dis["train"])
    new_valid = add_system_h0(dis["valid"])

    # 合并 D 的训练数据
    d_train = read_jsonl(config.ROOT / "data/sft_eval_format_v1/train.jsonl")
    d_valid = read_jsonl(config.ROOT / "data/sft_eval_format_v1/valid.jsonl")
    combined_train = d_train + new_train
    combined_valid = d_valid + new_valid
    data_dir = out_root / "data"
    train.write_data_files(data_dir, combined_train, combined_valid)
    print(f"[coevolve] W6 数据：train={len(combined_train)} valid={len(combined_valid)} "
          f"new_train={len(new_train)} new_valid={len(new_valid)}", flush=True)

    # 4) 训练 W6
    print("[coevolve] 训练 W6 ...", flush=True)
    adapter_dir = out_root / "adapters"
    res = train.train(
        data_dir=data_dir,
        adapter_dir=adapter_dir,
        iters=args.iters,
        lr=args.lr,
        seed=args.seed,
        log_file=out_root / "train.log",
    )
    if not res["ok"]:
        raise RuntimeError(f"W6 训练失败：{out_root / 'train.log'}")

    # 5) 评估 W6 × H6
    print("[coevolve] 评估 W6 × H6 ...", flush=True)
    variants = [
        {"name": "sys", "adapter": str(adapter_dir), "genome": best_h6, "shots": select_shots(shot_pool, best_h6)},
        {"name": "w_only", "adapter": str(adapter_dir), "genome": H0(), "shots": []},
        {"name": "h_core", "adapter": None,
         "genome": best_h6.__class__(**{**best_h6.to_dict(), "k_shots": 0, "shot_policy": "none"}),
         "shots": []},
        {"name": "h_full", "adapter": None, "genome": best_h6,
         "shots": select_shots(shot_pool, best_h6)},
    ]
    original = eval_original(variants, out_root / "eval", py)
    external = eval_external(variants, config.ROOT / "data/external/gsm8k_zh_heldout.jsonl", out_root / "eval_external")

    summary = {
        "h6_search": results,
        "best_h6": best_tag,
        "w6_train": {"iters": args.iters, "lr": args.lr, "seconds": res["seconds"],
                      "train_n": len(combined_train), "valid_n": len(combined_valid),
                      "new_train_n": len(new_train), "new_valid_n": len(new_valid)},
        "original": original,
        "external": external,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    # 计算 mixed
    for name in ("sys", "w_only", "h_core", "h_full"):
        o = original[name]["accuracy"]
        e = external[name]["accuracy"]
        summary[name] = {
            "original_accuracy": o,
            "external_accuracy": e,
            "mixed_accuracy": round((o * 450 + e * 200) / 650, 4),
        }
    (out_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[coevolve] 完成：" + json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    # 6) 生成 v16 文档并推送
    doc = [
        "# v16 —— (D,H₅) → (W₆,H₆) 一代共进化",
        "",
        f"> 完成时间：{summary['finished_at']}",
        "> 起点：D + H₅",
        "> 目标：验证权重×harness能否继续协同提升。",
        "",
        "## H6 搜索",
        "",
        "| 候选 | accuracy | parse_fail | composite |",
        "|---|---:|---:|---:|",
    ]
    for tag, r in results.items():
        s = r["summary"]
        doc.append(f"| {tag} | {s['accuracy']*100:.2f}% | {s['parse_fail_rate']*100:.2f}% | {s['composite']:.4f} |")
    doc += [
        "",
        f"最佳 H6：**{best_tag}**",
        "",
        "## W6 训练",
        "",
        f"- iters：{args.iters}",
        f"- lr：{args.lr}",
        f"- train：{len(combined_train)}",
        f"- valid：{len(combined_valid)}",
        f"- 新增 train：{len(new_train)}",
        f"- 新增 valid：{len(new_valid)}",
        "",
        "## 四变体结果",
        "",
        "| 变体 | 原 held-out 450 | 外部中文 200 | 混合 650 |",
        "|---|---:|---:|---:|",
    ]
    for name in ("sys", "w_only", "h_core", "h_full"):
        doc.append(f"| {name} | {summary[name]['original_accuracy']*100:.2f}% | "
                   f"{summary[name]['external_accuracy']*100:.2f}% | "
                   f"{summary[name]['mixed_accuracy']*100:.2f}% |")
    doc += [
        "",
        "## 结论",
        "",
        "- 见最终分析。",
        "",
        "## 文件",
        "",
        "- 汇总：`arms_sft/coevolution_v1/summary.json`",
        "- 日志：`logs/run_coevolution_v1.log`",
    ]
    (config.ROOT / "v16_coevolution_w6_h6.md").write_text("\n".join(doc), encoding="utf-8")

    subprocess.run(["git", "add", "v16_coevolution_w6_h6.md", "rsi/run_coevolution_v1.py"], cwd=config.ROOT, check=True)
    subprocess.run(["git", "commit", "-m", "v16 one-generation W6/H6 coevolution result"], cwd=config.ROOT, check=True)
    subprocess.run(["git", "push", "origin", "main"], cwd=config.ROOT, check=True)


if __name__ == "__main__":
    main()
