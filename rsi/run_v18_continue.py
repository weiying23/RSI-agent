"""run_v18_continue：沿 V18 路线继续进化到 Gen3/Gen4。"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

from . import config, distill, grade, taskbank, train
from .engine import Engine
from .genome import Genome, H0, select_shots
from .eval_dh5 import eval_original, eval_external
from .run_clean_replay_v2 import add_system_h0, search_harness


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").split("\n") if x.strip()]


def write_jsonl(path: Path, recs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def used_pids_from(paths: list[Path]) -> set[str]:
    out = set()
    for p in paths:
        if not p.exists():
            continue
        for r in read_jsonl(p):
            if "pid" in r:
                out.add(r["pid"])
            elif "id" in r:
                out.add(r["id"])
    return out


def unsolved_probs(probs: list[dict], samples: list[dict], n: int) -> list[dict]:
    by_pid = {}
    for s in samples:
        by_pid.setdefault(s["pid"], []).append(s)
    pmap = {p["id"]: p for p in probs}
    out = []
    for pid, atts in by_pid.items():
        p = pmap.get(pid)
        if not p:
            continue
        ok = False
        for a in atts:
            r = grade.grade_one(p["answer_frac"], a["response"])
            if r["correct"]:
                ok = True
                break
        if not ok:
            out.append(p)
        if len(out) >= n:
            break
    return out


def build_h3_candidates(base: Genome, shot_pool: list[dict]) -> list[tuple[str, Genome]]:
    return [
        ("baseline", base),
        ("max512", base.__class__(**{**base.to_dict(), "max_tokens": 512})),
        ("sc3", base.__class__(**{**base.to_dict(), "self_consistency_k": 3})),
        ("temp05", base.__class__(**{**base.to_dict(), "temperature": 0.5})),
        ("k2_hard", base.__class__(**{**base.to_dict(), "k_shots": 2, "shot_policy": "hard_win"})),
        ("top_p095", base.__class__(**{**base.to_dict(), "top_p": 0.95})),
    ]


def build_h4_candidates(base: Genome, shot_pool: list[dict]) -> list[tuple[str, Genome]]:
    return build_h3_candidates(base, shot_pool)


def main() -> None:
    config.sanitize_proxy_env()
    state = config.load_state()
    py = state.get("venv_python") or str(config.VENV_PYTHON)
    src = config.ROOT / "arms_clean_replay_v2"
    out = config.ROOT / "arms_v18_continue"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 1) 读取 V18 的 W2/H2/shot pool/历史训练数据
    h2 = Genome.from_dict(json.loads((src / "h2.json").read_text(encoding="utf-8"))["genome"])
    shot_pool = read_jsonl(src / "shot_pool.jsonl")
    gen1_train = read_jsonl(src / "gen1_data/train.jsonl")
    gen1_valid = read_jsonl(src / "gen1_data/valid.jsonl")
    gen2_train = read_jsonl(src / "gen2_data/train.jsonl")
    gen2_valid = read_jsonl(src / "gen2_data/valid.jsonl")

    used = used_pids_from([
        src / "shot_pool.jsonl",
        src / "gen1_w1_samples.jsonl",
        src / "gen2_w2_samples.jsonl",
    ])

    # 2) Gen3：H3 搜索
    prev_probs = taskbank.load_split("train", 2)
    prev_samples = read_jsonl(src / "gen2_w2_samples.jsonl")
    hard = unsolved_probs(prev_probs, prev_samples, 15)
    fresh_pool = [p for p in taskbank.load_split("probe", 3) if p["id"] not in used]
    fresh = fresh_pool[:10]
    probe3 = hard + fresh
    write_jsonl(out / "h3_probe.jsonl", probe3)
    print(f"[v18_continue] Gen3 H3 搜索，probe={len(probe3)}", flush=True)

    w2_adapter = src / "gen2_adapters"
    engine = Engine(adapter_path=str(w2_adapter), call_timeout_sec=90, task_timeout_sec=180)
    h3, h3_tag, h3_results = search_harness(
        engine, h2, probe3, shot_pool,
        out / "h3_search", build_h3_candidates,
    )
    engine.close()
    (out / "h3.json").write_text(
        json.dumps({"tag": h3_tag, "genome": h3.to_dict(), "results": h3_results},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v18_continue] H3 = {h3_tag}", flush=True)

    # 3) Gen3：生成 200 新题 + 训练 W3
    train3_pool = [p for p in taskbank.load_split("train", 3) if p["id"] not in used]
    gen3_probs = train3_pool[:200]
    for p in gen3_probs:
        used.add(p["id"])
    write_jsonl(out / "gen3_new_probs.jsonl", gen3_probs)
    print(f"[v18_continue] Gen3 生成 {len(gen3_probs)} 道新题", flush=True)

    engine = Engine(adapter_path=str(w2_adapter), call_timeout_sec=90, task_timeout_sec=180)
    gen3_samples = engine.generate_batch(gen3_probs, h3, k=2, shot_pool=shot_pool)
    engine.close()
    write_jsonl(out / "gen3_samples.jsonl", gen3_samples)
    gen3_dis = distill.distill(gen3_probs, gen3_samples, gen=3, seed=20261121)
    gen3_train = add_system_h0(gen3_dis["train"])
    gen3_valid = add_system_h0(gen3_dis["valid"])
    print(f"[v18_continue] Gen3 新增 train={len(gen3_train)} valid={len(gen3_valid)}", flush=True)

    cum_train = gen1_train + gen2_train + gen3_train
    cum_valid = gen1_valid + gen2_valid + gen3_valid
    w3_data = out / "w3_data"
    train.write_data_files(w3_data, cum_train, cum_valid)
    print(f"[v18_continue] 训练 W3，train={len(cum_train)} valid={len(cum_valid)}", flush=True)
    w3_adapter = out / "w3_adapters"
    w3_res = train.train(
        data_dir=w3_data, adapter_dir=w3_adapter,
        iters=200, lr=1e-5, seed=20261122,
        log_file=out / "w3_train.log",
    )
    if not w3_res["ok"]:
        raise RuntimeError("W3 训练失败")

    # 更新 shot pool
    shot_pool = shot_pool + gen3_dis["shot_pool_add"]
    write_jsonl(out / "shot_pool.jsonl", shot_pool)

    # 4) Gen4：H4 搜索
    hard4 = unsolved_probs(gen3_probs, gen3_samples, 15)
    fresh4_pool = [p for p in taskbank.load_split("probe", 4) if p["id"] not in used]
    fresh4 = fresh4_pool[:10]
    probe4 = hard4 + fresh4
    write_jsonl(out / "h4_probe.jsonl", probe4)
    print(f"[v18_continue] Gen4 H4 搜索，probe={len(probe4)}", flush=True)

    engine = Engine(adapter_path=str(w3_adapter), call_timeout_sec=90, task_timeout_sec=180)
    h4, h4_tag, h4_results = search_harness(
        engine, h3, probe4, shot_pool,
        out / "h4_search", build_h4_candidates,
    )
    engine.close()
    (out / "h4.json").write_text(
        json.dumps({"tag": h4_tag, "genome": h4.to_dict(), "results": h4_results},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v18_continue] H4 = {h4_tag}", flush=True)

    # 5) Gen4：生成 200 新题 + 训练 W4
    train4_pool = [p for p in taskbank.load_split("train", 4) if p["id"] not in used]
    gen4_probs = train4_pool[:200]
    for p in gen4_probs:
        used.add(p["id"])
    write_jsonl(out / "gen4_new_probs.jsonl", gen4_probs)
    print(f"[v18_continue] Gen4 生成 {len(gen4_probs)} 道新题", flush=True)

    engine = Engine(adapter_path=str(w3_adapter), call_timeout_sec=90, task_timeout_sec=180)
    gen4_samples = engine.generate_batch(gen4_probs, h4, k=2, shot_pool=shot_pool)
    engine.close()
    write_jsonl(out / "gen4_samples.jsonl", gen4_samples)
    gen4_dis = distill.distill(gen4_probs, gen4_samples, gen=4, seed=20261123)
    gen4_train = add_system_h0(gen4_dis["train"])
    gen4_valid = add_system_h0(gen4_dis["valid"])
    print(f"[v18_continue] Gen4 新增 train={len(gen4_train)} valid={len(gen4_valid)}", flush=True)

    cum_train = cum_train + gen4_train
    cum_valid = cum_valid + gen4_valid
    w4_data = out / "w4_data"
    train.write_data_files(w4_data, cum_train, cum_valid)
    print(f"[v18_continue] 训练 W4，train={len(cum_train)} valid={len(cum_valid)}", flush=True)
    w4_adapter = out / "w4_adapters"
    w4_res = train.train(
        data_dir=w4_data, adapter_dir=w4_adapter,
        iters=200, lr=1e-5, seed=20261124,
        log_file=out / "w4_train.log",
    )
    if not w4_res["ok"]:
        raise RuntimeError("W4 训练失败")

    # 6) 最终 650 题评估
    print("[v18_continue] 最终 650 题评估", flush=True)
    final_shots = select_shots(shot_pool, h4)
    sys_variant = {"name": "sys", "adapter": str(w4_adapter), "genome": h4, "shots": final_shots}
    original_450 = eval_original([sys_variant], out / "final_eval_original_450", py, eval_n=450)
    external_200 = eval_external([sys_variant], config.ROOT / "data/external/gsm8k_zh_heldout.jsonl", out / "final_eval_external")

    o = original_450["sys"]["accuracy"]
    e = external_200["sys"]["accuracy"]
    mixed = round((o * 450 + e * 200) / 650, 4)

    # 7) 100 题四变体归因
    print("[v18_continue] 100 题四变体归因", flush=True)
    variants_100 = [
        {"name": "sys", "adapter": str(w4_adapter), "genome": h4, "shots": final_shots},
        {"name": "w_only", "adapter": str(w4_adapter), "genome": H0(), "shots": []},
        {"name": "h_core", "adapter": None,
         "genome": h4.__class__(**{**h4.to_dict(), "k_shots": 0, "shot_policy": "none"}),
         "shots": []},
        {"name": "h_full", "adapter": None, "genome": h4, "shots": final_shots},
    ]
    attribution_100 = eval_original(variants_100, out / "attribution_100", py, eval_n=100)

    summary = {
        "start": {"w": "v18_w2", "h": "v18_h2"},
        "h3": {"tag": h3_tag, "genome": h3.to_dict(), "search": h3_results},
        "h4": {"tag": h4_tag, "genome": h4.to_dict(), "search": h4_results},
        "gen3": {
            "n_new": len(gen3_probs), "n_train_added": len(gen3_train),
            "n_valid_added": len(gen3_valid),
            "cumulative_train": len(gen1_train) + len(gen2_train) + len(gen3_train),
        },
        "gen4": {
            "n_new": len(gen4_probs), "n_train_added": len(gen4_train),
            "n_valid_added": len(gen4_valid),
            "cumulative_train": len(gen1_train) + len(gen2_train) + len(gen3_train) + len(gen4_train),
        },
        "shot_pool_final": len(shot_pool),
        "w3_train_seconds": w3_res["seconds"],
        "w4_train_seconds": w4_res["seconds"],
        "final": {
            "original_accuracy": o,
            "external_accuracy": e,
            "mixed_accuracy": mixed,
            "parse_fail_original": original_450["sys"]["parse_fail"],
            "parse_fail_external": external_200["sys"]["parse_fail"],
        },
        "attribution_100": attribution_100,
        "elapsed_min": round((time.time() - t0) / 60, 1),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[v18_continue] " + json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
