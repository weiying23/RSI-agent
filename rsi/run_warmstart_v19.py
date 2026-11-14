"""run_warmstart_v19：1,350 条历史数据 warm-start + 80 条 shot pool + 每代 200 新题。"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

from . import config, distill, grade, taskbank, train
from .engine import Engine
from .genome import Genome, H0, select_shots
from .eval_dh5 import eval_original, eval_external
from .run_clean_replay_v2 import add_system_h0, evaluate_candidate_v2, search_harness


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").split("\n") if x.strip()]


def write_jsonl(path: Path, recs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_h1_candidates(base: Genome, shot_pool: list[dict]) -> list[tuple[str, Genome]]:
    cands = [("baseline", base)]
    if shot_pool:
        cands += [
            ("k1_hard", base.__class__(**{**base.to_dict(), "k_shots": 1, "shot_policy": "hard_win"})),
            ("k2_hard", base.__class__(**{**base.to_dict(), "k_shots": 2, "shot_policy": "hard_win"})),
            ("k3_hard", base.__class__(**{**base.to_dict(), "k_shots": 3, "shot_policy": "hard_win"})),
            ("k3_hard_sc2", base.__class__(**{**base.to_dict(), "k_shots": 3, "shot_policy": "hard_win", "self_consistency_k": 2})),
            ("temp06", base.__class__(**{**base.to_dict(), "temperature": 0.6})),
        ]
    else:
        cands += [
            ("temp06", base.__class__(**{**base.to_dict(), "temperature": 0.6})),
            ("sc2", base.__class__(**{**base.to_dict(), "self_consistency_k": 2})),
        ]
    return cands


def build_h2_candidates(base: Genome) -> list[tuple[str, Genome]]:
    cands = [("baseline", base)]
    if base.k_shots > 1:
        cands.append(("k_minus1", base.__class__(**{**base.to_dict(), "k_shots": base.k_shots - 1})))
    cands += [
        ("sc_plus1", base.__class__(**{**base.to_dict(), "self_consistency_k": min(3, base.self_consistency_k + 1)})),
        ("temp06", base.__class__(**{**base.to_dict(), "temperature": 0.6})),
        ("top_p095", base.__class__(**{**base.to_dict(), "top_p": 0.95})),
        ("max512", base.__class__(**{**base.to_dict(), "max_tokens": 512})),
    ]
    return cands


def unique_pool(pool: list[dict], used: set[str], n: int, rng: random.Random) -> list[dict]:
    recs = []
    seen = set()
    for p in pool:
        q = p["question"]
        if q in used or q in seen:
            continue
        seen.add(q)
        recs.append(p)
        if len(recs) == n:
            break
    return recs


def main() -> None:
    config.sanitize_proxy_env()
    state = config.load_state()
    py = state.get("venv_python") or str(config.VENV_PYTHON)
    out_root = config.ROOT / "arms_warmstart_v19"
    out_root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 1) 读取 warm-start 数据
    base_dir = config.ROOT / "data/warmstart_v19"
    base_train = read_jsonl(base_dir / "train.jsonl")
    base_valid = read_jsonl(base_dir / "valid.jsonl")
    shot_pool = read_jsonl(base_dir / "shot_pool.jsonl")
    print(f"[v19] warm-start train={len(base_train)} valid={len(base_valid)} shot_pool={len(shot_pool)}", flush=True)

    used_q = set()
    for rec in base_train + base_valid:
        used_q.add(next(m["content"] for m in rec["messages"] if m["role"] == "user"))
    for s in shot_pool:
        used_q.add(s["question"])

    # 2) 组合新题池：train gen3/4/5，去掉与历史数据重复的题
    rng = random.Random(20261116)
    train_pool = []
    for g in (3, 4, 5):
        train_pool.extend(taskbank.load_split("train", g))
    rng.shuffle(train_pool)

    probe_pool = []
    for g in (3, 4, 5):
        probe_pool.extend(taskbank.load_split("probe", g))
    rng.shuffle(probe_pool)

    # 3) Warm-start 训练 W1
    w1_data = out_root / "w1_data"
    train.write_data_files(w1_data, base_train, base_valid)
    print("[v19] 训练 W1（1,270 warm-start）...", flush=True)
    w1_adapter = out_root / "w1_adapters"
    w1_res = train.train(
        data_dir=w1_data, adapter_dir=w1_adapter,
        iters=200, lr=1e-5, seed=20261116,
        log_file=out_root / "w1_train.log",
    )
    if not w1_res["ok"]:
        raise RuntimeError("W1 训练失败")

    # 4) H1 搜索
    h1_probe = unique_pool(probe_pool, used_q, 25, rng)
    for p in h1_probe:
        used_q.add(p["question"])
    write_jsonl(out_root / "h1_probe.jsonl", h1_probe)
    print(f"[v19] H1 搜索，probe={len(h1_probe)} ...", flush=True)
    engine = Engine(adapter_path=str(w1_adapter), call_timeout_sec=90, task_timeout_sec=180)
    h0 = H0()
    h1, h1_tag, h1_results = search_harness(
        engine, h0, h1_probe, shot_pool,
        out_root / "h1_search", build_h1_candidates,
    )
    engine.close()
    (out_root / "h1.json").write_text(
        json.dumps({"tag": h1_tag, "genome": h1.to_dict(), "results": h1_results},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v19] H1 = {h1_tag}", flush=True)

    # 5) Gen1：生成 200 新题 + 训练 W2
    gen1_probs = unique_pool(train_pool, used_q, 200, rng)
    for p in gen1_probs:
        used_q.add(p["question"])
    write_jsonl(out_root / "gen1_new_probs.jsonl", gen1_probs)
    print(f"[v19] Gen1 生成 {len(gen1_probs)} 道新题 ...", flush=True)
    engine = Engine(adapter_path=str(w1_adapter), call_timeout_sec=90, task_timeout_sec=180)
    gen1_samples = engine.generate_batch(gen1_probs, h1, k=2, shot_pool=shot_pool)
    engine.close()
    write_jsonl(out_root / "gen1_samples.jsonl", gen1_samples)
    gen1_dis = distill.distill(gen1_probs, gen1_samples, gen=1, seed=20261117)
    gen1_train = add_system_h0(gen1_dis["train"])
    gen1_valid = add_system_h0(gen1_dis["valid"])
    print(f"[v19] Gen1 新增 train={len(gen1_train)} valid={len(gen1_valid)}", flush=True)

    cumulative_train = base_train + gen1_train
    w2_data = out_root / "w2_data"
    train.write_data_files(w2_data, cumulative_train, base_valid)
    print(f"[v19] 训练 W2，train={len(cumulative_train)} ...", flush=True)
    w2_adapter = out_root / "w2_adapters"
    w2_res = train.train(
        data_dir=w2_data, adapter_dir=w2_adapter,
        iters=200, lr=1e-5, seed=20261118,
        log_file=out_root / "w2_train.log",
    )
    if not w2_res["ok"]:
        raise RuntimeError("W2 训练失败")

    # 6) H2 搜索
    h2_probe = unique_pool(probe_pool, used_q, 25, rng)
    for p in h2_probe:
        used_q.add(p["question"])
    write_jsonl(out_root / "h2_probe.jsonl", h2_probe)
    print(f"[v19] H2 搜索，probe={len(h2_probe)} ...", flush=True)
    engine = Engine(adapter_path=str(w2_adapter), call_timeout_sec=90, task_timeout_sec=180)
    h2, h2_tag, h2_results = search_harness(
        engine, h1, h2_probe, shot_pool,
        out_root / "h2_search", lambda b, pool: build_h2_candidates(b),
    )
    engine.close()
    (out_root / "h2.json").write_text(
        json.dumps({"tag": h2_tag, "genome": h2.to_dict(), "results": h2_results},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v19] H2 = {h2_tag}", flush=True)

    # 7) Gen2：生成 200 新题 + 训练 W3
    gen2_probs = unique_pool(train_pool, used_q, 200, rng)
    for p in gen2_probs:
        used_q.add(p["question"])
    write_jsonl(out_root / "gen2_new_probs.jsonl", gen2_probs)
    print(f"[v19] Gen2 生成 {len(gen2_probs)} 道新题 ...", flush=True)
    engine = Engine(adapter_path=str(w2_adapter), call_timeout_sec=90, task_timeout_sec=180)
    gen2_samples = engine.generate_batch(gen2_probs, h2, k=2, shot_pool=shot_pool)
    engine.close()
    write_jsonl(out_root / "gen2_samples.jsonl", gen2_samples)
    gen2_dis = distill.distill(gen2_probs, gen2_samples, gen=2, seed=20261119)
    gen2_train = add_system_h0(gen2_dis["train"])
    gen2_valid = add_system_h0(gen2_dis["valid"])
    print(f"[v19] Gen2 新增 train={len(gen2_train)} valid={len(gen2_valid)}", flush=True)

    cumulative_train = cumulative_train + gen2_train
    w3_data = out_root / "w3_data"
    train.write_data_files(w3_data, cumulative_train, base_valid)
    print(f"[v19] 训练 W3，train={len(cumulative_train)} ...", flush=True)
    w3_adapter = out_root / "w3_adapters"
    w3_res = train.train(
        data_dir=w3_data, adapter_dir=w3_adapter,
        iters=200, lr=1e-5, seed=20261120,
        log_file=out_root / "w3_train.log",
    )
    if not w3_res["ok"]:
        raise RuntimeError("W3 训练失败")

    # 8) 最终 650 题评估
    print("[v19] 最终 650 题评估 ...", flush=True)
    final_shots = select_shots(shot_pool, h2)
    sys_variant = {"name": "sys", "adapter": str(w3_adapter), "genome": h2, "shots": final_shots}
    original_450 = eval_original([sys_variant], out_root / "final_eval_original_450", py, eval_n=450)
    external_200 = eval_external([sys_variant], config.ROOT / "data/external/gsm8k_zh_heldout.jsonl", out_root / "final_eval_external")

    o = original_450["sys"]["accuracy"]
    e = external_200["sys"]["accuracy"]
    mixed = round((o * 450 + e * 200) / 650, 4)

    # 9) 100 题四变体归因
    print("[v19] 100 题四变体归因 ...", flush=True)
    variants_100 = [
        {"name": "sys", "adapter": str(w3_adapter), "genome": h2, "shots": final_shots},
        {"name": "w_only", "adapter": str(w3_adapter), "genome": H0(), "shots": []},
        {"name": "h_core", "adapter": None,
         "genome": h2.__class__(**{**h2.to_dict(), "k_shots": 0, "shot_policy": "none"}),
         "shots": []},
        {"name": "h_full", "adapter": None, "genome": h2, "shots": final_shots},
    ]
    attribution_100 = eval_original(variants_100, out_root / "attribution_100", py, eval_n=100)

    summary = {
        "warm_start": {
            "base_train": len(base_train),
            "base_valid": len(base_valid),
            "shot_pool": len(shot_pool),
        },
        "h1": {"tag": h1_tag, "genome": h1.to_dict(), "search": h1_results},
        "h2": {"tag": h2_tag, "genome": h2.to_dict(), "search": h2_results},
        "gen1": {
            "n_new": len(gen1_probs),
            "n_train_added": len(gen1_train),
            "n_valid_added": len(gen1_valid),
            "cumulative_train": len(base_train) + len(gen1_train),
        },
        "gen2": {
            "n_new": len(gen2_probs),
            "n_train_added": len(gen2_train),
            "n_valid_added": len(gen2_valid),
            "cumulative_train": len(base_train) + len(gen1_train) + len(gen2_train),
        },
        "w1_train_seconds": w1_res["seconds"],
        "w2_train_seconds": w2_res["seconds"],
        "w3_train_seconds": w3_res["seconds"],
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
    print("[v19] " + json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
