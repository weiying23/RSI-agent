"""distill：验证通过的解 → 训练数据（messages 格式，兼容用户 MLX 配置）。

规则（HANDOFF §3.2 Phase B / §3.5 决策5）：
- 只收 判对 ∧ lint 过 的解
- 【稳定性过滤】只收 success_rate=1.0 的题（k 次采样每次都对）——
  「蒙对一次」的解会教坏模型，是过拟合/灾难性遗忘的来源之一（RFT 信号）
- 每题只留最短正确解（防风格坍缩）
- messages = [user: 题目, assistant: 解答]——不含 system：
  权重学的是「行为」，指令是 harness 侧的「框架」，这样归因才干净
- 90/10 切分；10% valid 仅作 MLX loss 监控，禁入任何结论（纪律§4）
- 未稳定解出的题（全错 + 蒙对部分）→ unsolved_pids，交 Phase A 失败条件化
"""
from __future__ import annotations

import random

from . import grade

# ---------------------------------------------------------------------------
# 题型号提示：帮助 LoRA 数据中的模型建立“如何做对”的说明
# ---------------------------------------------------------------------------

TEMPLATE_HINTS = {
    "mult_chain": "这类题是连续乘法。按顺序逐步相乘，每乘一步检查一次中间结果，最后把乘积写入 \\boxed{}。",
    "frac_addsub": "这类题是分数加减。先找公分母通分，再对分子做加减；注意每一步结果都应为正，最后化成最简分数。",
    "lcm": "这类题求最小公倍数。不要直接相乘，除非几个数两两互质。可用质因数分解取每个质因子的最高次幂，或用公式 LCM(a,b)=a×b/GCD(a,b)。",
    "gcd": "这类题求最大公约数。用质因数分解取每个公共质因子的最低次幂，或用辗转相除法。",
    "percent": "这类题是百分数。先判断“单位 1”是否已知：已知单位 1 求部分用乘法；已知部分求单位 1 用除法。最后用分数或小数算。",
    "average": "这类题求平均数。先求总和，再除以个数；如果除不尽，保留最简分数。",
    "travel": "这类题是行程问题。核心公式是距离=速度×时间；如果求速度或时间，反过来用除法。注意时间可能是小数。",
    "rect_area": "这类题是长方形面积。面积=长×宽；如果长或宽发生变化，先算出变化后的新长度，再乘另一个不变的边。",
    "discount": "这类题是折扣问题。现价=原价×折扣数÷10；如果已知现价求原价，用除法。注意折扣可能是整数或几点几折。",
    "chicken_rabbit": "这类题是鸡兔同笼。设未知数，利用“总头数”和“总腿数”列方程；鸡 2 条腿，兔 4 条腿。解完后代回题目验证。",
}


def make_user_with_hint(question: str, template: str | None) -> str:
    """给训练题追加简短的解题提示，帮助模型建立正确解法。"""
    hint = TEMPLATE_HINTS.get(template or "")
    if not hint:
        return question
    return f"{question}\n\n[解题提示] {hint}"


def distill(
    problems: list[dict],
    samples: list[dict],
    gen: int = 0,
    seed: int = 0,
    valid_frac: float = 0.1,
    existing_count: int = 0,
    min_success_rate: float = 1.0,
    with_hints: bool = False,
) -> dict:
    """蒸馏一批样本。

    problems: 题目记录（id/question/answer_frac/difficulty/...）
    samples: 引擎采样记录（pid/attempt/response）
    existing_count: 已有的累积训练条数（切分去偏用）
    min_success_rate: 进训练集的最低稳定性（1.0 = 每次采样都对）
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
    n_stable = 0

    for pid, attempts in by_pid.items():
        p = prob_by_id[pid]
        graded = []
        for s in attempts:
            r = grade.grade_one(p["answer_frac"], s["response"])
            if r["correct"]:
                n_correct_attempts += 1
            ok, _reasons = grade.lint_one(s["response"], r)
            graded.append((s, r, ok and r["correct"]))
        sr = sum(1 for g in graded if g[1]["correct"]) / len(graded)
        winners = [g for g in graded if g[2]]  # correct ∧ lint
        # 稳定性过滤：蒙对的部分（sr<1.0）足以判对但不足以为师，弃为失败题
        if not winners or sr < min_success_rate:
            unsolved.append(pid)
            continue
        n_stable += 1
        # 每题只留最短正确解
        s_best, r_best, _ = min(winners, key=lambda g: len(g[0]["response"]))
        user_content = make_user_with_hint(p["question"], p.get("template")) if with_hints else p["question"]
        rec = {
            "messages": [
                {"role": "user", "content": user_content},
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
                "success_rate": sr,
                "gen": gen,
            }
        )

    n_attempts = len(samples)
    n_timeout = sum(1 for s in samples if s.get("status") == "timeout")
    n_solved = n_stable  # 稳定解出的题数（进训练集的口径）
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
            "stable_rate": round(n_stable / len(by_pid), 4) if by_pid else 0.0,
            "min_success_rate": min_success_rate,
            "timeout_attempts": n_timeout,
            "timeout_rate": round(n_timeout / n_attempts, 4) if n_attempts else 0.0,
            "gen": gen,
        },
    }
