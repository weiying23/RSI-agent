"""orchestrator：三臂外环状态机（P1 交付物）。

每代协议（arm_WH；W/H 臂为其子集，见 HANDOFF §3.2）：

  Phase A  harness 搜索（冻结 W_n）
    probe 30 题 = 上代未解题 + 新题各半（失败条件化，Self-HPO +1.5pp）
    候选 = 程序化扰动 ×2 + W_n 自改指令 ×1（自提议，S 字母的落点）
    评分 = pass@retry_k（数据产率语义）＋accuracy 并列记录
    H_{n+1} = argmax；全部候选（含落选）入档案
  Phase B  数据生成 + 训练（冻结 H_{n+1}）
    pool 200 新题 → k 采样 → verified 蒸馏 → 累积回放 → fresh LoRA → W_{n+1}
  Phase C  四变体 held-out 评估（lockbox 独立进程，循环代码不触碰 held-out）
    sys / w_only / h_core / h_full
  Phase D  晋升门（promo-val 烧毁型数据，held-out 绝不做选择）
    gain > margin δ 才晋级 W_{n+1}；否则保留 W_n（全记录）
    lineage.json = 归因 + 成本 + hack_flags + unsolved_pids

断点续跑：每阶段完成即写 marker 文件，重跑跳过已完成阶段。

用法：
  python -m rsi.orchestrator --arm WH --gens 5 [--gen-size standard|small]
  python -m rsi.orchestrator --arm H  --gens 5
  python -m rsi.orchestrator --arm W  --gens 5
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from pathlib import Path

from . import config, distill, env_gen, grade, taskbank, train
from .engine import Engine
from .genome import Genome, H0, perturb, select_shots

PROMOVAL_N = 50
PROBE_N = 20
MARGIN_DELTA = 0.02  # 晋升门 margin δ（命题1 噪声感知规则的工程近似）
EVAL_TREND_N = 100   # 趋势代：每代只测 sys 一个变体的考卷题数
EVAL_ATTR_N = 100    # 归因代：四变体全量考卷题数（只在指定代跑）

GEN_SIZE = {
    "standard": {"pool": 200, "iters": 120, "lr": 5e-5},
    "small": {"pool": 80, "iters": 80, "lr": 5e-5},
    "tiny": {"pool": 40, "iters": 60, "lr": 5e-5},
}

META_PROMPT = """你是一个数学解题策略研究员。你当前用于指导模型解数学题的指令是：

「{instruction}」

以下是最近的解题失败案例（模型在你的指令下做错了）：
{failures}

请分析这些失败的共同模式，然后重写这条解题指令，让模型能避开这些失败模式。
要求：
- 【硬性要求】必须原样保留「最终答案写入 \\boxed{{}} 中」这句结尾格式指令，一字不得删除或改写——这是判分系统的唯一接口，没有它答案全部判零分
- 保留原指令中有效的部分（分数写法要求等）
- 新指令要具体针对失败模式，不要泛泛而谈
- 长度不超过四句话
- 只输出新指令的文本本身，不要任何解释、引号或标记"""


def _marker(gen_dir: Path, phase: str) -> Path:
    return gen_dir / f".phase_{phase}_done"


def _done(gen_dir: Path, phase: str) -> bool:
    return _marker(gen_dir, phase).exists()


def _mark(gen_dir: Path, phase: str) -> None:
    _marker(gen_dir, phase).write_text(time.strftime("%Y-%m-%d %H:%M:%S"))


def _write_jsonl(path: Path, recs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _jdump(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def _subprocess_env() -> dict:
    import os

    config.sanitize_proxy_env()
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    return env


# ---------------------------------------------------------------------------
# Phase A：harness 搜索
# ---------------------------------------------------------------------------

def sanitize_instruction(text: str) -> str:
    """清洗模型自改的指令：剥代码块标记、引号包装、多余空白。"""
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`").lstrip("text\n").strip()
    for q in ("「", "」", '"', "'"):
        s = s.strip(q)
    # 去掉「新指令：」之类的包装
    for prefix in ("新指令：", "新指令:", "指令：", "指令:"):
        if s.startswith(prefix):
            s = s[len(prefix):].strip()
    return s


def self_rewrite_instruction(
    engine: Engine, genome: Genome, failures: list[dict], rng: random.Random
) -> Genome:
    """W_n 自改指令（Phase A 候选3，S 字母的落点）。"""
    lines = []
    for f in failures[:3]:
        lines.append(
            f"题目：{f['question']}\n模型答案：{f['model_answer']}（错误）\n正确答案：{f['answer']}"
        )
    prompt = META_PROMPT.format(
        instruction=genome.instruction, failures="\n\n".join(lines)
    )
    resp, _ = engine.generate_one(
        prompt,
        Genome(  # 元层用低温保守采样
            instruction="你是一个数学解题策略研究员，按要求输出。",
            temperature=0.4,
            top_p=0.9,
            max_tokens=256,
            retry_k=1,
        ),
    )
    new_instr = sanitize_instruction(resp)
    g = genome.__class__(**{**genome.to_dict(), "instruction": new_instr})
    g.validate()
    return g


def evaluate_candidate(
    engine: Engine, cand: Genome, probs: list[dict], tag: str, out_dir: Path,
    gen: int = 0, shot_pool: list[dict] | None = None,
) -> dict:
    """候选基因组在 probe 上的 pass@retry_k + accuracy。

    同时收集做对的解（每题保留最短正确解）→ 供几-shot 例题库积累。
    这是 arm_H（无 Phase B）few-shot 通道的唯一来源。
    """
    samples = engine.generate_batch(probs, cand, k=cand.retry_k, shot_pool=shot_pool)
    _write_jsonl(out_dir / f"{tag}_samples.jsonl", samples)
    prob_by_id = {p["id"]: p for p in probs}
    by_pid: dict[str, list[dict]] = {}
    for s in samples:
        by_pid.setdefault(s["pid"], []).append(s)
    solved: set[str] = set()
    results: list[dict] = []
    solved_shots: list[dict] = []
    for pid, atts in by_pid.items():
        p = prob_by_id[pid]
        corrects: list[dict] = []
        for s in atts:
            r = grade.grade_one(p["answer_frac"], s["response"])
            r["pid"] = pid
            results.append(r)
            if r["correct"]:
                corrects.append(s)
        if corrects:
            solved.add(pid)
            best = min(corrects, key=lambda s: len(s["response"]))
            solved_shots.append(
                {
                    "pid": pid,
                    "question": p["question"],
                    "solution": best["response"],
                    "difficulty": p.get("difficulty"),
                    "template": p.get("template"),
                    "success_rate": len(corrects) / len(atts),
                    "gen": gen,
                }
            )
    summary = grade.summarize(results)
    n_timeout = sum(1 for x in samples if x.get("status") == "timeout")
    summary["timeout_count"] = n_timeout
    summary["timeout_rate"] = round(n_timeout / len(samples), 4) if samples else 0.0
    return {
        "pass_at_k": len(solved) / len(probs),
        "attempt_accuracy": summary["accuracy"],
        "n_solved": len(solved),
        "solved_pids": sorted(solved),
        "solved_shots": solved_shots,
        "hack": summary,
        "timeout_count": n_timeout,
        "timeout_rate": round(n_timeout / len(samples), 4) if samples else 0.0,
    }


def _update_shot_pool(shot_pool_file: Path, new_shots: list[dict]) -> int:
    """把新做对的解合入例题库（按 pid 去重，保留最短解）。"""
    existing = _read_jsonl(shot_pool_file)
    by_pid = {s["pid"]: s for s in existing}
    for s in new_shots:
        if s["pid"] not in by_pid:
            by_pid[s["pid"]] = s
        elif len(s["solution"]) < len(by_pid[s["pid"]]["solution"]):
            by_pid[s["pid"]] = s
    _write_jsonl(shot_pool_file, list(by_pid.values()))
    return len(by_pid)


def run_phase_a(
    ctx: dict, engine: Engine, prev_unsolved: list[str], gen_dir: Path
) -> dict:
    """Phase A：返回 H_{n+1} 与全部候选档案。"""
    arm, state = ctx["arm"], ctx["state"]
    difficulty = state["difficulty"]
    n_probe = ctx["probe_n"]
    rng = random.Random(f"phaseA:{ctx['arm']}:{ctx['gen']}")

    # ---- probe 集：失败条件化（上代未解题优先）----
    # 题目来自一次性冻结的 task bank；三臂共享同一 probe 分片。
    prev_probs_by_id = {p["id"]: p for p in ctx.get("all_prev_pool", [])}
    hard_ids = [pid for pid in prev_unsolved if pid in prev_probs_by_id]
    rng.shuffle(hard_ids)
    hard_probs = [prev_probs_by_id[pid] for pid in hard_ids[: n_probe // 2]]
    n_fresh = n_probe - len(hard_probs)

    probe_shard = taskbank.load_split(
        "probe", ctx["gen"], root=ctx["task_bank_root"], expected_n=n_fresh
    )
    # fresh 部分在代内固定、三臂一致；hard 部分由各臂自己的失败轨迹决定。
    probe_rng = random.Random(f"taskbank-probe:{ctx['gen']}")
    fresh = probe_rng.sample(probe_shard, n_fresh)
    probs = hard_probs + fresh
    _write_jsonl(gen_dir / "probe.jsonl", probs)
    note = f"（{len(hard_probs)} 上代未解 + {n_fresh} 固定题库新题）"

    # ---- 候选：扰动 ×2 + 自改指令 ×1 ----
    h_n = ctx["genome_n"]
    candidates: list[tuple[str, Genome]] = [
        ("perturb_1", perturb(h_n, rng)),
        ("perturb_2", perturb(h_n, rng)),
    ]
    cand_dir = gen_dir / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)
    failures = ctx.get("recent_failures", [])
    if failures:
        try:
            g_self = self_rewrite_instruction(engine, h_n, failures, rng)
            candidates.append(("self_rewrite", g_self))
        except Exception as e:  # 自改失败不阻塞：降级为扰动
            _jdump(cand_dir / "self_rewrite_error.json", {"error": str(e)})
    else:
        # 无失败样本（首代）：第三个候选也用扰动
        candidates.append(("perturb_3", perturb(h_n, rng)))

    # ---- 逐候选评估（先测当前 H_n 作基线，归档回滚的参照）----
    shot_pool = _read_jsonl(ctx["shot_pool_file"])
    scores = {}
    base_sc = evaluate_candidate(engine, h_n, probs, "baseline_hn", cand_dir,
                                 gen=ctx["gen"], shot_pool=shot_pool)
    scores["baseline_hn"] = {"genome": h_n.to_dict(), "score": base_sc}
    print(f"    [baseline_hn] pass@k={base_sc['pass_at_k']:.3f} acc={base_sc['attempt_accuracy']:.3f}")
    for tag, cand in candidates:
        sc = evaluate_candidate(engine, cand, probs, tag, cand_dir,
                                gen=ctx["gen"], shot_pool=shot_pool)
        scores[tag] = {"genome": cand.to_dict(), "score": sc}
        print(f"    [{tag}] pass@k={sc['pass_at_k']:.3f} acc={sc['attempt_accuracy']:.3f}")
    best_tag = max(scores, key=lambda t: (scores[t]["score"]["pass_at_k"],
                                          scores[t]["score"]["attempt_accuracy"]))
    # 归档回滚：所有候选都不如当前 H_n，则回退保留 H_n
    if best_tag == "baseline_hn":
        h_next = h_n
        best_shots = []
        print(f"    Phase A{note}：候选全劣于 H_n → 回退保留 H_n")
    else:
        h_next = Genome.from_dict(scores[best_tag]["genome"])
        best_shots = scores[best_tag]["score"].get("solved_shots", [])
        print(f"    Phase A 完成{note} → H_{{n+1}} = {best_tag}")
    _jdump(cand_dir / "all_candidates.json", scores)
    _jdump(gen_dir / "genome.json", h_next.to_dict())
    # 胜出候选做对的题 → 积累进 few-shot 例题库（arm_H 无 Phase B，这是唯一来源）
    pool_size = _update_shot_pool(ctx["shot_pool_file"], best_shots) if best_shots else len(shot_pool)
    print(f"    （例题库 +{len(best_shots)}，共 {pool_size}）")
    return {"h_next": h_next, "best_tag": best_tag, "candidates": scores,
            "probe_note": note}


# ---------------------------------------------------------------------------
# Phase B：数据生成 + 训练
# ---------------------------------------------------------------------------

def run_phase_b(ctx: dict, engine: Engine, h_next: Genome, gen_dir: Path) -> dict:
    """Phase B：verified 蒸馏 → 累积回放 → fresh LoRA。"""
    arm, state = ctx["arm"], ctx["state"]
    difficulty = state["difficulty"]
    size = ctx["size"]

    # 固定训练题分片：三臂共享同一份题目清单；模型轨迹按臂独立生成。
    probs = taskbank.load_split(
        "train", ctx["gen"], root=ctx["task_bank_root"], expected_n=size["pool"]
    )
    if len(probs) > size["pool"]:
        probs = probs[: size["pool"]]
    _write_jsonl(gen_dir / "pool.jsonl", probs)

    shot_pool = _read_jsonl(ctx["shot_pool_file"])
    samples = engine.generate_batch(probs, h_next, shot_pool=shot_pool)
    _write_jsonl(gen_dir / "samples.jsonl", samples)

    dis = distill.distill(probs, samples, gen=ctx["gen"], seed=ctx["gen"],
                          with_hints=ctx.get("with_hints", False))
    # 累积回放：上一代累积 + 本代新增（账本保证 pid 天然不重复）
    prev_train = _read_jsonl(ctx["train_cum_file"])
    train_recs = prev_train + dis["train"]
    valid_recs = dis["valid"]

    # 外部数学训练数据混合（Plan B）：
    # external_mix = -1 表示与当轮自生成训练数据等量；>0 表示固定条数；0 关闭。
    ext_train_path = ctx.get("external_train")
    ext_mix = int(ctx.get("external_mix") or 0)
    ext_added = 0
    if ext_train_path and ext_mix != 0:
        ext_recs = _read_jsonl(Path(ext_train_path))
        if ext_recs:
            n_ext = len(train_recs) if ext_mix == -1 else min(ext_mix, len(ext_recs))
            rng_ext = random.Random(f"external-mix:{ctx['gen']}:{n_ext}")
            ext_sample = rng_ext.sample(ext_recs, n_ext)
            train_recs = train_recs + ext_sample
            if not valid_recs:
                valid_recs = ext_sample[: max(1, n_ext // 10)]
            ext_added = n_ext
    data_dir = gen_dir / "data"
    train.write_data_files(data_dir, train_recs, valid_recs)
    _write_jsonl(ctx["train_cum_file"], train_recs)  # 更新 arm 级累积
    _write_jsonl(ctx["shot_pool_file"],
                 _read_jsonl(ctx["shot_pool_file"]) + dis["shot_pool_add"])

    # 训练（fresh from base）
    res = train.train(
        data_dir=data_dir,
        adapter_dir=gen_dir / "adapters",
        iters=size["iters"],
        lr=size["lr"],
        seed=ctx["gen"],
        log_file=gen_dir / "train.log",
    )
    if not res["ok"]:
        raise RuntimeError(f"训练失败，见 {res['log']}")
    print(f"    Phase B 完成：pool={len(probs)} solved={dis['stats']['n_solved']} "
          f"self_train={len(prev_train)+len(dis['train'])} ext_added={ext_added} "
          f"total_train={len(train_recs)} train={res['seconds']}s")
    return {"distill_stats": dis["stats"], "train_res": res, "unsolved": dis["unsolved_pids"],
            "adapter": str(gen_dir / "adapters"), "external_added": ext_added}


# ---------------------------------------------------------------------------
# Phase C：四变体 held-out 评估
# ---------------------------------------------------------------------------

def run_phase_c(ctx: dict, h_next: Genome, w_candidate: str | None, gen_dir: Path,
                full_attribution: bool = False) -> dict:
    """lockbox 评估（趋势代只测 sys；归因代跑四变体，按臂裁剪 + 签名去重）。

    变体语义：
      arm_WH：sys / w_only / h_core / h_full（若 H_n 无 shots 则 h_core≡h_full 去重）
      arm_W ：只跑 sys（H 恒 H_0 无 shots → w_only≡sys，h_core/h_full≡BASE_t0）
      arm_H ：sys≡h_full / h_core（W 恒基座）
    full_attribution=False：每臂只跑核心 sys 变体（H 臂即 h_full），省算力。
    """
    arm = ctx["arm"]
    eval_n = ctx["eval_attr_n"] if full_attribution else ctx["eval_trend_n"]
    spec: dict = {
        "eval_n": eval_n,
        "reps": ctx["reps"],
        "seed": ctx["gen"],
        "call_timeout_sec": ctx["call_timeout_sec"],
        "task_timeout_sec": ctx["task_timeout_sec"],
        # 训练/候选生成使用运行时 harness（格式重试/投票/换策略），
        # held-out 评估也必须同一套，否则方向级 HPO 复现会高估或低估 harness。
        "runtime_mode": "checked",
    }

    shot_pool = _read_jsonl(ctx["shot_pool_file"])
    shots = select_shots(shot_pool, h_next)

    wanted: list[tuple[str, dict]] = []
    if not full_attribution:  # 趋势代：只核心变体
        if arm == "arm_H":
            wanted = [("h_full", {"adapter": None, "genome": h_next.to_dict(), "shots": shots})]
        else:
            wanted = [("sys", {"adapter": w_candidate, "genome": h_next.to_dict(), "shots": shots})]
    elif arm == "arm_WH":
        wanted = [
            ("sys", {"adapter": w_candidate, "genome": h_next.to_dict(), "shots": shots}),
            ("w_only", {"adapter": w_candidate, "genome": H0().to_dict()}),
            ("h_core", {"adapter": None,
                        "genome": {**h_next.to_dict(), "k_shots": 0, "shot_policy": "none"}}),
            ("h_full", {"adapter": None, "genome": h_next.to_dict(), "shots": shots}),
        ]
    elif arm == "arm_W":
        wanted = [("sys", {"adapter": w_candidate, "genome": H0().to_dict()})]
    else:  # arm_H
        wanted = [
            ("h_core", {"adapter": None,
                        "genome": {**h_next.to_dict(), "k_shots": 0, "shot_policy": "none"}}),
            ("h_full", {"adapter": None, "genome": h_next.to_dict(), "shots": shots}),
        ]

    # 配置签名去重：同一 (adapter, genome, n_shots) 只生成一次，结果别名复用
    seen: dict[str, str] = {}  # signature → variant name (首个)
    uniq: list[dict] = []
    alias: dict[str, str] = {}  # name → canonical name
    for name, v in wanted:
        sig = json.dumps([v["adapter"], v["genome"], len(v.get("shots") or [])],
                         sort_keys=True, ensure_ascii=False)
        if sig in seen:
            alias[name] = seen[sig]
        else:
            seen[sig] = name
            uniq.append({"name": name, **v})
    spec["variants"] = uniq

    spec_path = gen_dir / "eval_spec.json"
    _jdump(spec_path, {**spec, "aliases": alias, "full_attribution": full_attribution})
    venv_py = ctx["state"].get("venv_python") or str(config.VENV_PYTHON)
    phase_timeout = float(ctx.get("phase_timeout_sec") or 0.0)
    try:
        proc = subprocess.run(
            [venv_py, str(config.LOCKBOX_DIR / "run_eval.py"),
             "--spec", str(spec_path), "--out-dir", str(gen_dir / "eval")],
            cwd=str(config.ROOT), env=_subprocess_env(),
            timeout=(phase_timeout if phase_timeout > 0 else None),
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Phase C 超过阶段超时 {phase_timeout:.0f}s；响应文件已保留，可断点续跑。"
        ) from exc
    if proc.returncode != 0:
        raise RuntimeError("lockbox 评估失败")
    rep = json.loads((gen_dir / "eval" / "eval.json").read_text(encoding="utf-8"))
    accs = {k: v["accuracy_mean"] for k, v in rep["variants"].items()}
    # 别名回填（arm_W: sys≡w_only；arm_H: sys≡h_full；WH 无 shots 时 h_core≡h_full）
    if arm == "arm_H":
        accs["sys"] = accs.get("h_full")
    for name, canon in alias.items():
        if canon in accs:
            accs[name] = accs[canon]
    accs = {k: v for k, v in accs.items() if v is not None}
    print(f"    Phase C 完成（{len(uniq)} 变体 × {eval_n} 题）：{json.dumps(accs)}")
    return {"eval": rep, "accs": accs}


# ---------------------------------------------------------------------------
# Phase D：晋升门 + lineage
# ---------------------------------------------------------------------------

def run_phase_d(
    ctx: dict, phase_a: dict, phase_b: dict | None, phase_c: dict, gen_dir: Path
) -> dict:
    """晋升门：promo-val 上 W_{n+1} vs W_n（held-out 绝不做选择）。"""
    arm, state = ctx["arm"], ctx["state"]
    difficulty = state["difficulty"]
    h_next: Genome = phase_a["h_next"]

    # arm_H 无权重通道，无晋升门
    if phase_b is None:
        promoted, gate = False, {"skipped": "arm_H has no weight channel"}
    else:
        # 固定 promo 分片：三臂共享题目清单。
        probs = taskbank.load_split(
            "promo", ctx["gen"], root=ctx["task_bank_root"], expected_n=PROMOVAL_N
        )[:PROMOVAL_N]
        h_eval = h_next if arm == "arm_WH" else H0()
        # promo-val 快速门控：不注入 few-shot，候选同条件
        engine = Engine(
            adapter_path=phase_b["adapter"],
            call_timeout_sec=ctx["call_timeout_sec"],
            task_timeout_sec=ctx["task_timeout_sec"],
        )
        res_new = engine.generate_batch(probs, h_eval, k=1, shot_pool=None)
        engine.close()
        engine = Engine(
            adapter_path=ctx["w_current"],
            call_timeout_sec=ctx["call_timeout_sec"],
            task_timeout_sec=ctx["task_timeout_sec"],
        )
        res_old = engine.generate_batch(probs, h_eval, k=1, shot_pool=None)
        engine.close()

        def met_of(res):
            by = {p["id"]: p for p in probs}
            rs = [grade.grade_one(by[s["pid"]]["answer_frac"], s["response"])
                  for s in res]
            s = grade.summarize(rs)
            n_timeout = sum(1 for x in res if x.get("status") == "timeout")
            return {
                "accuracy": s["accuracy"],
                "parse_fail": s["parse_fail_rate"],
                "timeout_count": n_timeout,
                "timeout_rate": round(n_timeout / len(res), 4) if res else 0.0,
            }

        m_new, m_old = met_of(res_new), met_of(res_old)

        # 检查点继承：历史最优 adapter 纳入对比（CoEvo-RSI）
        best_file = ctx["arm_dir"] / "best_adapter.json"
        best_adapter = None
        if best_file.exists():
            best_adapter = json.loads(best_file.read_text(encoding="utf-8")).get("adapter")
        m_best = None
        if best_adapter and best_adapter != ctx["w_current"] and best_adapter != phase_b["adapter"]:
            engine = Engine(
                adapter_path=best_adapter,
                call_timeout_sec=ctx["call_timeout_sec"],
                task_timeout_sec=ctx["task_timeout_sec"],
            )
            res_best = engine.generate_batch(probs, h_eval, k=1, shot_pool=None)
            engine.close()
            m_best = met_of(res_best)

        # TIMR 门控：新权重 parse_fail 不显著上升（>5pp）才允许晋升
        timr_ok = m_new["parse_fail"] <= m_old["parse_fail"] + 0.05
        promoted = (m_new["accuracy"] - m_old["accuracy"]) > MARGIN_DELTA and timr_ok
        if promoted:
            w_used = phase_b["adapter"]
            _jdump(best_file, {"adapter": w_used, "accuracy": m_new["accuracy"],
                               "gen": ctx["gen"]})
        else:
            # 回退：在上一代与历史最优之间选更好的（历史继承）
            if m_best is not None and m_best["accuracy"] >= m_old["accuracy"]:
                w_used = best_adapter
            else:
                w_used = ctx["w_current"]

        gate = {"acc_new": m_new["accuracy"], "acc_old": m_old["accuracy"],
                "acc_best": (m_best or {}).get("accuracy"),
                "gain": round(m_new["accuracy"] - m_old["accuracy"], 4),
                "margin": MARGIN_DELTA,
                "timr_ok": timr_ok,
                "parse_fail_new": m_new["parse_fail"],
                "parse_fail_old": m_old["parse_fail"],
                "timeout_new": m_new["timeout_count"],
                "timeout_old": m_old["timeout_count"],
                "timeout_rate_new": m_new["timeout_rate"],
                "timeout_rate_old": m_old["timeout_rate"],
                "promoted": promoted}

    if phase_b is None:
        w_used = ctx["w_current"]

    lineage = {
        "arm": arm,
        "gen": ctx["gen"],
        "phase_a": {"best_tag": phase_a["best_tag"], "probe_note": phase_a["probe_note"],
                    "candidates": {k: {"pass_at_k": v["score"]["pass_at_k"],
                                       "acc": v["score"]["attempt_accuracy"],
                                       "genome": v["genome"]}
                                   for k, v in phase_a["candidates"].items()}},
        "phase_b": (phase_b or {}).get("distill_stats"),
        "phase_c": phase_c["accs"],
        "phase_d_gate": gate,
        "w_used": w_used,
        "h_next": h_next.to_dict(),
        "task_bank": {
            "root": str(ctx["task_bank_root"]),
            "manifest_sha256": ctx.get("task_bank_manifest_sha"),
            "train_split": f"train/gen_{ctx['gen']:03d}",
            "probe_split": f"probe/gen_{ctx['gen']:03d}",
            "promo_split": f"promo/gen_{ctx['gen']:03d}",
        },
        "timeouts": {
            "call_timeout_sec": ctx["call_timeout_sec"],
            "task_timeout_sec": ctx["task_timeout_sec"],
            "phase_timeout_sec": ctx["phase_timeout_sec"],
        },
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _jdump(gen_dir / "lineage.json", lineage)
    return lineage


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------

def run_generation(
    arm: str,
    gen: int,
    state: dict,
    size: dict,
    eval_n: int,
    reps: int,
    eval_attr_n: int = EVAL_ATTR_N,
    total_gens: int | None = None,
    arms_root: Path | str | None = None,
    task_bank_root: Path | str | None = None,
    task_bank_manifest_sha: str | None = None,
    probe_n: int = PROBE_N,
    call_timeout_sec: float = config.DEFAULT_CALL_TIMEOUT_SEC,
    task_timeout_sec: float = config.DEFAULT_TASK_TIMEOUT_SEC,
    phase_timeout_sec: float = config.DEFAULT_PHASE_TIMEOUT_SEC,
    with_hints: bool = False,
    external_train: str | None = None,
    external_mix: int = 0,
) -> None:
    arms_root = Path(arms_root) if arms_root else config.ARMS_DIR
    task_bank_root = Path(task_bank_root) if task_bank_root else taskbank.DEFAULT_BANK_ROOT
    arm_dir = arms_root / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    gen_dir = arm_dir / f"gen_{gen:03d}"
    gen_dir.mkdir(parents=True, exist_ok=True)
    ctx = {
        "arm": arm, "gen": gen, "state": state, "size": size,
        "eval_trend_n": eval_n, "eval_attr_n": eval_attr_n, "reps": reps,
        "probe_n": int(probe_n),
        "shot_pool_file": arm_dir / "shot_pool.jsonl",
        "train_cum_file": arm_dir / "train_cum.jsonl",
        "arm_dir": arm_dir,
        "arms_root": arms_root,
        "task_bank_root": task_bank_root,
        "task_bank_manifest_sha": task_bank_manifest_sha,
        "call_timeout_sec": float(call_timeout_sec),
        "task_timeout_sec": float(task_timeout_sec),
        "phase_timeout_sec": float(phase_timeout_sec),
        "with_hints": bool(with_hints),
        "external_train": external_train,
        "external_mix": external_mix,
    }
    # 归因代 = 最后一代码（该代四变体全量），其余代只测 sys 趋势
    full_attr = (gen == total_gens) if total_gens is not None else False
    t0 = time.time()

    # 本代起始状态
    if gen == 0:
        ctx["genome_n"] = H0()
        ctx["w_current"] = None  # 基座
        _jdump(gen_dir / "genome.json", H0().to_dict())
        _mark(gen_dir, "A"); _mark(gen_dir, "B"); _mark(gen_dir, "C"); _mark(gen_dir, "D")
        _jdump(gen_dir / "lineage.json", {"arm": arm, "gen": 0, "note": "初始状态 (W_base, H_0)"})
        print(f"[{arm}] gen_000 = 初始状态（基座 + H_0）")
        return

    prev_dir = arm_dir / f"gen_{gen-1:03d}"
    prev_lineage = json.loads((prev_dir / "lineage.json").read_text(encoding="utf-8"))
    # arm_W 的 harness 恒为 H_0，不需要读取上一代 genome；其余臂从上一代读取。
    if arm == "arm_W":
        ctx["genome_n"] = H0()
    else:
        prev_genome = prev_dir / "genome.json"
        if prev_genome.exists():
            ctx["genome_n"] = Genome.from_dict(
                json.loads(prev_genome.read_text(encoding="utf-8"))
            )
        else:
            ctx["genome_n"] = H0()
    ctx["w_current"] = prev_lineage.get("w_used")

    # 上代未解题 + 失败样例（失败条件化的原料）
    prev_samples = _read_jsonl(prev_dir / "samples.jsonl") if (prev_dir / "samples.jsonl").exists() else []
    prev_pool = _read_jsonl(prev_dir / "pool.jsonl") if (prev_dir / "pool.jsonl").exists() else []
    ctx["all_prev_pool"] = ctx.get("all_prev_pool", []) + prev_pool
    prob_by_id = {p["id"]: p for p in prev_pool}
    unsolved_prev = _solve_status(prev_pool, prev_samples)
    ctx["recent_failures"] = _failure_examples(prev_pool, prev_samples)

    # ---- Phase A ----
    if arm in ("arm_WH", "arm_H"):
        if not _done(gen_dir, "A"):
            print(f"[{arm}] gen_{gen:03d} Phase A：harness 搜索")
            engine = Engine(
                adapter_path=ctx["w_current"] if arm == "arm_WH" else None,
                call_timeout_sec=ctx["call_timeout_sec"],
                task_timeout_sec=ctx["task_timeout_sec"],
            )
            phase_a = run_phase_a(ctx, engine, unsolved_prev, gen_dir)
            engine.close()
            _mark(gen_dir, "A")
        else:
            phase_a = {
                "h_next": Genome.from_dict(
                    json.loads((gen_dir / "genome.json").read_text(encoding="utf-8"))
                ),
                "best_tag": "(resumed)", "candidates": {}, "probe_note": "(resumed)",
            }
            cand_file = gen_dir / "candidates" / "all_candidates.json"
            if cand_file.exists():
                raw = json.loads(cand_file.read_text(encoding="utf-8"))
                phase_a["candidates"] = {
                    k: {"pass_at_k": v["score"]["pass_at_k"],
                        "acc": v["score"]["attempt_accuracy"],
                        "genome": v["genome"]}
                    for k, v in raw.items()
                }
                phase_a["best_tag"] = max(
                    raw, key=lambda k: (raw[k]["score"]["pass_at_k"],
                                        raw[k]["score"]["attempt_accuracy"])
                )
    else:  # arm_W：H 恒为 H_0
        phase_a = {"h_next": H0(), "best_tag": "H_0_frozen", "candidates": {}, "probe_note": ""}
    h_next: Genome = phase_a["h_next"]

    # ---- Phase B ----
    if arm in ("arm_WH", "arm_W"):
        if not _done(gen_dir, "B"):
            print(f"[{arm}] gen_{gen:03d} Phase B：数据生成 + LoRA")
            engine = Engine(
                adapter_path=ctx["w_current"],
                call_timeout_sec=ctx["call_timeout_sec"],
                task_timeout_sec=ctx["task_timeout_sec"],
            )
            phase_b = run_phase_b(ctx, engine, h_next, gen_dir)
            engine.close()
            _mark(gen_dir, "B")
        else:
            phase_b = {
                "distill_stats": None, "train_res": None,
                "unsolved": [], "adapter": str(gen_dir / "adapters"),
            }
            pool = _read_jsonl(gen_dir / "pool.jsonl")
            samples = _read_jsonl(gen_dir / "samples.jsonl")
            if pool and samples:
                dis = distill.distill(pool, samples, gen=ctx["gen"], seed=ctx["gen"])
                phase_b["distill_stats"] = dis["stats"]
                phase_b["unsolved"] = dis["unsolved_pids"]
    else:
        phase_b = None

    # ---- Phase C ----
    if not _done(gen_dir, "C"):
        label = "四变体归因评估" if full_attr else "sys 趋势评估"
        print(f"[{arm}] gen_{gen:03d} Phase C：{label}")
        run_phase_c(ctx, h_next, phase_b["adapter"] if phase_b else None, gen_dir, full_attr)
        _mark(gen_dir, "C")
    phase_c_rep = json.loads((gen_dir / "eval" / "eval.json").read_text(encoding="utf-8"))
    phase_c = {"eval": phase_c_rep,
               "accs": {k: v["accuracy_mean"] for k, v in phase_c_rep["variants"].items()}}

    # ---- Phase D ----
    if not _done(gen_dir, "D"):
        print(f"[{arm}] gen_{gen:03d} Phase D：晋升门 + lineage")
        run_phase_d(ctx, phase_a, phase_b, phase_c, gen_dir)
        _mark(gen_dir, "D")

    lin = json.loads((gen_dir / "lineage.json").read_text(encoding="utf-8"))
    print(f"[{arm}] gen_{gen:03d} 完成（{(time.time()-t0)/60:.1f} min）"
          f" eval={json.dumps(lin.get('phase_c', {}))} gate={lin.get('phase_d_gate')}")


def _solve_status(probs: list[dict], samples: list[dict]) -> list[str]:
    """上一代 pool 中 k 次采样全失败的题。"""
    if not probs or not samples:
        return []
    by_pid: dict[str, list[bool]] = {}
    prob_by_id = {p["id"]: p for p in probs}
    for s in samples:
        if s["pid"] not in prob_by_id:
            continue
        r = grade.grade_one(prob_by_id[s["pid"]]["answer_frac"], s["response"])
        by_pid.setdefault(s["pid"], []).append(r["correct"])
    return [pid for pid, vals in by_pid.items() if not any(vals)]


def _failure_examples(probs: list[dict], samples: list[dict]) -> list[dict]:
    """提取失败样例（题目 + 模型错误答案 + 正确答案）供自改指令用。"""
    prob_by_id = {p["id"]: p for p in probs}
    out = []
    for s in samples:
        p = prob_by_id.get(s["pid"])
        if not p:
            continue
        r = grade.grade_one(p["answer_frac"], s["response"])
        if not r["correct"] and r["extracted"]:
            out.append({"question": p["question"], "model_answer": r["extracted"][:40],
                        "answer": p["answer"]})
        if len(out) >= 5:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["W", "H", "WH"], required=True)
    ap.add_argument("--gens", type=int, default=5,
                    help="更新代数；gen_000 是初始状态，实际目录为 gen_000..gen_N")
    ap.add_argument("--gen-size", choices=list(GEN_SIZE), default="small")
    ap.add_argument("--eval-n", type=int, default=EVAL_TREND_N,
                    help="趋势代考卷题数（sys 单变体）")
    ap.add_argument("--eval-attr-n", type=int, default=EVAL_ATTR_N,
                    help="归因代（最后一代）四变体考卷题数")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--task-bank", type=Path, default=taskbank.DEFAULT_BANK_ROOT,
                    help="一次性冻结的题库目录")
    ap.add_argument("--arms-root", type=Path, default=config.ARMS_DIR,
                    help="输出目录；新实验建议用 arms_fixed，避免覆盖旧结果")
    ap.add_argument("--probe-n", type=int, default=PROBE_N,
                    help="每个候选 harness 的 probe 题数")
    ap.add_argument("--with-hints", action="store_true",
                    help="LoRA 训练数据中给题目追加解题提示")
    ap.add_argument("--external-train", type=str, default=None,
                    help="外部数学训练 JSONL（messages 格式），用于和自生成数据混合")
    ap.add_argument("--external-mix", type=int, default=0,
                    help="每代混入外部数据的条数；-1 表示和当轮自生成训练数据同样多；0 关闭")
    ap.add_argument("--call-timeout", type=float, default=config.DEFAULT_CALL_TIMEOUT_SEC,
                    help="单次模型生成 wall-clock 上限（秒）")
    ap.add_argument("--task-timeout", type=float, default=config.DEFAULT_TASK_TIMEOUT_SEC,
                    help="单道题所有尝试/重试合计上限（秒）")
    ap.add_argument("--phase-timeout", type=float, default=config.DEFAULT_PHASE_TIMEOUT_SEC,
                    help="Phase C 子进程上限（秒）；0=不设")
    args = ap.parse_args()

    state = config.load_state()
    if not state.get("frozen", {}).get("heldout"):
        raise SystemExit("held-out 未冻结：先完成 P0（calibrate + freeze）")

    # 固定题库必须先存在且通过校验；实验中途禁止临时出题。
    manifest = taskbank.load_manifest(args.task_bank, verify=True)
    if manifest["gens"] < args.gens:
        raise SystemExit(
            f"题库只支持 {manifest['gens']} 代，但请求 {args.gens} 代；重建题库或减少 --gens"
        )
    if manifest["counts"]["probe_per_gen"] < args.probe_n:
        raise SystemExit(
            f"每代 probe 只有 {manifest['counts']['probe_per_gen']} 题，"
            f"不够 --probe-n {args.probe_n}"
        )

    size = GEN_SIZE[args.gen_size]
    if manifest["counts"]["train_per_gen"] < size["pool"]:
        raise SystemExit(
            f"每代训练题只有 {manifest['counts']['train_per_gen']} 题，"
            f"不够 gen-size={args.gen_size} 的 pool={size['pool']}"
        )
    eval_n = args.eval_n

    arm = f"arm_{args.arm}"
    print(
        f"[{arm}] 固定题库={args.task_bank} manifest={manifest['manifest_sha256'][:12]}… "
        f"call_timeout={args.call_timeout}s task_timeout={args.task_timeout}s "
        f"phase_timeout={args.phase_timeout}s"
    )
    for gen in range(0, args.gens + 1):
        run_generation(
            arm, gen, state, size, eval_n, args.reps,
            eval_attr_n=args.eval_attr_n,
            total_gens=args.gens,
            arms_root=args.arms_root,
            task_bank_root=args.task_bank,
            task_bank_manifest_sha=manifest["manifest_sha256"],
            probe_n=args.probe_n,
            call_timeout_sec=args.call_timeout,
            task_timeout_sec=args.task_timeout,
            phase_timeout_sec=args.phase_timeout,
            with_hints=args.with_hints,
            external_train=args.external_train,
            external_mix=args.external_mix,
        )
    print(f"[{arm}] 全部 {args.gens} 代完成")


if __name__ == "__main__":
    main()
