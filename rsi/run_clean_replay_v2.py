"""run_clean_replay_v2：从原始 W0/H0 开始，加入定向 harness 搜索的干净 replay v2。

改进点：
1. 先用 H0 在 100 道训练题上建立 shot pool；
2. H1 搜索显式测试 k_shots=1/2/3、hard_win/recent_success、self_consistency；
3. W1/W2 训练改用评估一致格式（H0 system + 裸题 + 直接解答）；
4. 第 2 代 probe 加入上代未解题，做失败条件化；
5. 最终统一在 650 题混合测试集上评估 sys，并在 100 题 held-out 上做四变体归因。
"""
from __future__ import annotations

import json
import random
import subprocess
import time
from pathlib import Path

from . import config, distill, grade, taskbank, train
from .engine import Engine
from .genome import Genome, H0, select_shots
from .eval_dh5 import eval_original, eval_external


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, recs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").split("\n") if x.strip()]


def add_system_h0(records: list[dict]) -> list[dict]:
    out = []
    for rec in records:
        msgs = rec["messages"]
        if msgs and msgs[0]["role"] != "system":
            msgs = [{"role": "system", "content": H0().instruction}] + msgs
        out.append({**rec, "messages": msgs})
    return out


def evaluate_candidate_v2(
    engine: Engine,
    genome: Genome,
    problems: list[dict],
    shot_pool: list[dict],
    tag: str,
    out_dir: Path,
) -> dict:
    shots = select_shots(shot_pool, genome)
    samples = engine.generate_batch(problems, genome, shot_pool=shot_pool)
    write_jsonl(out_dir / f"{tag}_samples.jsonl", samples)
    prob_by_id = {p["id"]: p for p in problems}
    by_pid: dict[str, list[dict]] = {}
    for s in samples:
        by_pid.setdefault(s["pid"], []).append(s)
    results = []
    solved_shots = []
    solved = set()
    for pid, atts in by_pid.items():
        p = prob_by_id[pid]
        corrects = []
        for s in atts:
            r = grade.grade_one(p["answer_frac"], s["response"])
            r["pid"] = pid
            results.append(r)
            if r["correct"]:
                corrects.append(s)
        if corrects:
            solved.add(pid)
            best = min(corrects, key=lambda s: len(s["response"]))
            solved_shots.append({
                "pid": pid,
                "question": p["question"],
                "solution": best["response"],
                "difficulty": p.get("difficulty"),
                "template": p.get("template"),
                "success_rate": len(corrects) / len(atts),
                "gen": 1,
            })
    summary = grade.summarize(results)
    summary["composite"] = round(summary["accuracy"] - 0.25 * summary["parse_fail_rate"], 4)
    summary["pass_at_k"] = len(solved) / len(problems)
    summary["solved_shots"] = solved_shots
    return summary


def build_h1_candidates(base: Genome, shot_pool: list[dict]) -> list[tuple[str, Genome]]:
    cands = [("baseline", base)]
    if shot_pool:
        cands += [
            ("k1_hard", base.__class__(**{**base.to_dict(), "k_shots": 1, "shot_policy": "hard_win"})),
            ("k2_hard", base.__class__(**{**base.to_dict(), "k_shots": 2, "shot_policy": "hard_win"})),
            ("k3_hard", base.__class__(**{**base.to_dict(), "k_shots": 3, "shot_policy": "hard_win"})),
            ("k2_recent", base.__class__(**{**base.to_dict(), "k_shots": 2, "shot_policy": "recent_success"})),
            ("k3_recent", base.__class__(**{**base.to_dict(), "k_shots": 3, "shot_policy": "recent_success"})),
            ("k2_hard_sc2", base.__class__(**{**base.to_dict(), "k_shots": 2, "shot_policy": "hard_win", "self_consistency_k": 2})),
            ("k3_hard_sc2", base.__class__(**{**base.to_dict(), "k_shots": 3, "shot_policy": "hard_win", "self_consistency_k": 2})),
        ]
    else:
        cands += [
            ("temp06", base.__class__(**{**base.to_dict(), "temperature": 0.6})),
            ("sc2", base.__class__(**{**base.to_dict(), "self_consistency_k": 2})),
            ("max512", base.__class__(**{**base.to_dict(), "max_tokens": 512})),
        ]
    return cands


def build_h2_candidates(base: Genome) -> list[tuple[str, Genome]]:
    cands = [("baseline", base)]
    k = base.k_shots
    if k < 3:
        cands.append(("k_plus1_hard", base.__class__(**{**base.to_dict(), "k_shots": k + 1, "shot_policy": "hard_win"})))
    if k > 1:
        cands.append(("k_minus1_hard", base.__class__(**{**base.to_dict(), "k_shots": k - 1, "shot_policy": "hard_win"})))
    cands += [
        ("sc_plus1", base.__class__(**{**base.to_dict(), "self_consistency_k": min(3, base.self_consistency_k + 1)})),
        ("temp06", base.__class__(**{**base.to_dict(), "temperature": 0.6})),
        ("top_p095", base.__class__(**{**base.to_dict(), "top_p": 0.95})),
        ("max512", base.__class__(**{**base.to_dict(), "max_tokens": 512})),
        ("recent_success", base.__class__(**{**base.to_dict(), "shot_policy": "recent_success"})),
    ]
    return cands


def search_harness(
    engine: Engine,
    base: Genome,
    problems: list[dict],
    shot_pool: list[dict],
    out_dir: Path,
    candidate_builder,
) -> tuple[Genome, str, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for tag, cand in candidate_builder(base, shot_pool):
        print(f"[clean_replay_v2] 评估 {tag} ...", flush=True)
        s = evaluate_candidate_v2(engine, cand, problems, shot_pool, tag, out_dir)
        results[tag] = {"genome": cand.to_dict(), "summary": s}
        print(f"[clean_replay_v2] {tag}: acc={s['accuracy']:.4f} "
              f"parse_fail={s['parse_fail_rate']:.4f} "
              f"composite={s['composite']:.4f}", flush=True)
    best_tag = max(results, key=lambda t: results[t]["summary"]["composite"])
    best_genome = Genome.from_dict(results[best_tag]["genome"])
    return best_genome, best_tag, results


def train_weight(
    adapter_dir: Path,
    data_dir: Path,
    iters: int,
    lr: float,
    seed: int,
    log_file: Path,
) -> dict:
    return train.train(
        data_dir=data_dir,
        adapter_dir=adapter_dir,
        iters=iters,
        lr=lr,
        seed=seed,
        log_file=log_file,
    )


def main() -> None:
    config.sanitize_proxy_env()
    state = config.load_state()
    py = state.get("venv_python") or str(config.VENV_PYTHON)
    root = config.ROOT
    out_root = root / "arms_clean_replay_v2"
    out_root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 固定题库
    train1 = taskbank.load_split("train", 1)
    train2 = taskbank.load_split("train", 2)
    probe1 = taskbank.load_split("probe", 1)
    probe2 = taskbank.load_split("probe", 2)
    rng = random.Random(20261114)
    rng.shuffle(train1)
    rng.shuffle(train2)
    rng.shuffle(probe1)
    rng.shuffle(probe2)

    shot_pool_probs = train1[:100]
    w1_train_probs = train1[100:200]
    w2_train_probs = train2[:100]
    h1_probe = probe1[:30]

    # 1) 用 H0 建立 shot pool
    print("[clean_replay_v2] Gen1：用 H0 在 100 题上建立 shot pool ...", flush=True)
    engine = Engine(adapter_path=None, call_timeout_sec=90, task_timeout_sec=180)
    h0 = H0()
    shot_samples = engine.generate_batch(shot_pool_probs, h0, shot_pool=[])
    write_jsonl(out_root / "gen1_shot_pool_samples.jsonl", shot_samples)
    shot_dis = distill.distill(shot_pool_probs, shot_samples, gen=1, seed=1)
    shot_pool = shot_dis["shot_pool_add"]
    write_jsonl(out_root / "shot_pool.jsonl", shot_pool)
    engine.close()
    print(f"[clean_replay_v2] shot pool size = {len(shot_pool)}", flush=True)

    # 2) H1 搜索
    print("[clean_replay_v2] Gen1：H1 定向搜索 ...", flush=True)
    engine = Engine(adapter_path=None, call_timeout_sec=90, task_timeout_sec=180)
    h1, h1_tag, h1_results = search_harness(
        engine, h0, h1_probe, shot_pool,
        out_root / "gen1_h_search", build_h1_candidates,
    )
    engine.close()
    (out_root / "h1.json").write_text(
        json.dumps({"tag": h1_tag, "genome": h1.to_dict(), "results": h1_results},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[clean_replay_v2] H1 = {h1_tag}", flush=True)

    # 3) W1 训练
    print("[clean_replay_v2] Gen1：W1 数据生成 + 训练 ...", flush=True)
    engine = Engine(adapter_path=None, call_timeout_sec=90, task_timeout_sec=180)
    w1_samples = engine.generate_batch(w1_train_probs, h1, shot_pool=shot_pool)
    write_jsonl(out_root / "gen1_w1_samples.jsonl", w1_samples)
    engine.close()
    w1_dis = distill.distill(w1_train_probs, w1_samples, gen=1, seed=2)
    w1_train = add_system_h0(w1_dis["train"])
    w1_valid = add_system_h0(w1_dis["valid"])
    w1_data_dir = out_root / "gen1_data"
    train.write_data_files(w1_data_dir, w1_train, w1_valid)
    w1_adapter = out_root / "gen1_adapters"
    w1_res = train_weight(w1_adapter, w1_data_dir, 200, 1e-5, 20261114, out_root / "gen1_train.log")
    if not w1_res["ok"]:
        raise RuntimeError("W1 训练失败")
    # 更新 shot pool
    shot_pool = shot_pool + w1_dis["shot_pool_add"]
    write_jsonl(out_root / "shot_pool.jsonl", shot_pool)

    # 4) H2 搜索：15道上代未解 + 15道新题
    unsolved = w1_dis["unsolved_pids"][:15]
    unsolved_probs = [p for p in w1_train_probs if p["id"] in unsolved]
    h2_probe = unsolved_probs + probe2[:30 - len(unsolved_probs)]
    write_jsonl(out_root / "gen2_probe.jsonl", h2_probe)
    print(f"[clean_replay_v2] Gen2：H2 搜索（{len(unsolved_probs)} 未解 + {len(h2_probe)-len(unsolved_probs)} 新题） ...", flush=True)
    engine = Engine(adapter_path=str(w1_adapter), call_timeout_sec=90, task_timeout_sec=180)
    h2, h2_tag, h2_results = search_harness(
        engine, h1, h2_probe, shot_pool,
        out_root / "gen2_h_search", lambda b, pool: build_h2_candidates(b),
    )
    engine.close()
    (out_root / "h2.json").write_text(
        json.dumps({"tag": h2_tag, "genome": h2.to_dict(), "results": h2_results},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[clean_replay_v2] H2 = {h2_tag}", flush=True)

    # 5) W2 训练
    print("[clean_replay_v2] Gen2：W2 数据生成 + 训练 ...", flush=True)
    engine = Engine(adapter_path=str(w1_adapter), call_timeout_sec=90, task_timeout_sec=180)
    w2_samples = engine.generate_batch(w2_train_probs, h2, shot_pool=shot_pool)
    write_jsonl(out_root / "gen2_w2_samples.jsonl", w2_samples)
    engine.close()
    w2_dis = distill.distill(w2_train_probs, w2_samples, gen=2, seed=3)
    w2_train = add_system_h0(w2_dis["train"])
    w2_valid = add_system_h0(w2_dis["valid"])
    w2_data_dir = out_root / "gen2_data"
    train.write_data_files(w2_data_dir, w2_train, w2_valid)
    w2_adapter = out_root / "gen2_adapters"
    w2_res = train_weight(w2_adapter, w2_data_dir, 200, 1e-5, 20261115, out_root / "gen2_train.log")
    if not w2_res["ok"]:
        raise RuntimeError("W2 训练失败")
    shot_pool = shot_pool + w2_dis["shot_pool_add"]
    write_jsonl(out_root / "shot_pool.jsonl", shot_pool)

    # 6) 最终评估
    print("[clean_replay_v2] 最终 650 题评估 ...", flush=True)
    final_shots = select_shots(shot_pool, h2)
    sys_variant = {"name": "sys", "adapter": str(w2_adapter), "genome": h2, "shots": final_shots}
    original_450 = eval_original([sys_variant], out_root / "final_eval_original_450", py, eval_n=450)
    external_200 = eval_external([sys_variant], root / "data/external/gsm8k_zh_heldout.jsonl", out_root / "final_eval_external")

    o = original_450["sys"]["accuracy"]
    e = external_200["sys"]["accuracy"]
    mixed = round((o * 450 + e * 200) / 650, 4)

    # 7) 100题四变体归因
    print("[clean_replay_v2] 100题四变体归因 ...", flush=True)
    variants_100 = [
        {"name": "sys", "adapter": str(w2_adapter), "genome": h2, "shots": final_shots},
        {"name": "w_only", "adapter": str(w2_adapter), "genome": H0(), "shots": []},
        {"name": "h_core", "adapter": None,
         "genome": h2.__class__(**{**h2.to_dict(), "k_shots": 0, "shot_policy": "none"}),
         "shots": []},
        {"name": "h_full", "adapter": None, "genome": h2, "shots": final_shots},
    ]
    attribution_100 = eval_original(variants_100, out_root / "attribution_100", py, eval_n=100)

    summary = {
        "h1": {"tag": h1_tag, "genome": h1.to_dict(), "search": h1_results},
        "h2": {"tag": h2_tag, "genome": h2.to_dict(), "search": h2_results},
        "w1_train": {"n_train": len(w1_train), "n_valid": len(w1_valid), "seconds": w1_res["seconds"]},
        "w2_train": {"n_train": len(w2_train), "n_valid": len(w2_valid), "seconds": w2_res["seconds"]},
        "final": {
            "original_accuracy": o,
            "external_accuracy": e,
            "mixed_accuracy": mixed,
            "parse_fail_original": original_450["sys"]["parse_fail"],
            "parse_fail_external": external_200["sys"]["parse_fail"],
        },
        "attribution_100": attribution_100,
        "baseline_h0": {
            "original_accuracy": 0.60,
            "external_accuracy": 0.535,
            "mixed_accuracy": 0.58,
        },
        "elapsed_min": round((time.time() - t0) / 60, 1),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[clean_replay_v2] " + json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
