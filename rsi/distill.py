"""distill：验证通过的解 → 训练数据（messages 格式，兼容用户 MLX 配置）。

规则（HANDOFF §3.2 Phase B / §3.5 决策5）：
- 只收 判对 ∧ lint 过 的解
- 每题只留最短正确解（防风格坍缩）
- messages = [user: 题目, assistant: 解答]——不含 system：
  权重学的是「行为」，指令是 harness 侧的「框架」，这样归因才干净
- 90/10 切分；10% valid 仅作 MLX loss 监控，禁入任何结论（纪律§4）
- 未解出的题 → unsolved_pids，交给 Phase A 做失败条件化 probe（Self-HPO）
"""
from __future__ import annotations

import random

from . import grade


def distill(
    problems: list[dict],
    samples: list[dict],
    gen: int = 0,
    seed: int = 0,
    valid_frac: float = 0.1,
    existing_count: int = 0,
) -> dict:
    """蒸馏一批样本。

    problems: 题目记录（id/question/answer_frac/difficulty/...）
    samples: 引擎采样记录（pid/attempt/response）
    existing_count: 已有的累积训练条数（切分去偏用）
    返回 {"train","valid","shot_pool_add","unsolved_pids","stats"}
    """
    by_pid: dict[str, list[dict]] = {}
    for s in samples:
        by_pid.setdefault(s["pid"], []).append(s)

    prob_by_id = {p["id"]: p for p in problems}
    rng = random.Random(f"distill:{seed}:{gen}:{len(samples)}")

    train_recs: list[dict] = []
    valid_recs: list[dict] = []
    shot_add: list[dict] = []
    unsolved: list[str] = []
    n_correct_attempts = 0

    for pid, attempts in by_pid.items():
        p = prob_by_id[pid]
        graded = []
        for s in attempts:
            r = grade.grade_one(p["answer_frac"], s["response"])
            if r["correct"]:
                n_correct_attempts += 1
            ok, _reasons = grade.lint_one(s["response"], r)
            graded.append((s, r, ok and r["correct"]))
        winners = [g for g in graded if g[2]]
        if not winners:
            unsolved.append(pid)
            continue
        # 每题只留最短正确解
        s_best, r_best, _ = min(winners, key=lambda g: len(g[0]["response"]))
        rec = {
            "messages": [
                {"role": "user", "content": p["question"]},
                {"role": "assistant", "content": s_best["response"]},
            ]
        }
        if rng.random() < valid_frac:
            valid_recs.append(rec)
        else:
            train_recs.append(rec)
        shot_add.append(
            {
                "pid": pid,
                "question": p["question"],
                "solution": s_best["response"],
                "difficulty": p.get("difficulty"),
                "template": p.get("template"),
                "success_rate": sum(1 for g in graded if g[1]["correct"]) / len(graded),
                "gen": gen,
            }
        )

    n_attempts = len(samples)
    n_solved = len(by_pid) - len(unsolved)
    return {
        "train": train_recs,
        "valid": valid_recs,
        "shot_pool_add": shot_add,
        "unsolved_pids": unsolved,
        "stats": {
            "n_problems": len(by_pid),
            "n_attempts": n_attempts,
            "n_solved": n_solved,
            "solve_rate": round(n_solved / len(by_pid), 4) if by_pid else 0.0,
            "attempt_accuracy": round(n_correct_attempts / n_attempts, 4) if n_attempts else 0.0,
            "gen": gen,
        },
    }
