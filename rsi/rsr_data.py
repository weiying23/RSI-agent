"""rsr_data：Terminal-RSI 式 SFT 数据增强。

对中文数学题执行：
  plan → solve → verify → rewrite → re-verify
只有最终答案正确、重写后仍正确且 lint 通过的样本才进入 SFT 训练集。
"""
from __future__ import annotations

from . import grade
from .distill import make_user_with_hint
from .genome import Genome


PLAN_GENOME = Genome(
    instruction=(
        "你是数学解题策略专家。请先只写一个简短的解题计划，"
        "说明这道题应该按什么步骤做。不要写出最终答案，不要写 \\boxed{}。"
    ),
    k_shots=0,
    temperature=0.3,
    top_p=0.9,
    max_tokens=256,
    retry_k=1,
    format_retry=0,
    self_consistency_k=1,
)

SOLVE_GENOME = Genome(
    instruction=(
        "请按照给定的解题计划，逐步推理并写出完整计算过程。"
        "最终答案写入 \\boxed{} 中。答案若为分数，用最简分数 a/b 表示；"
        "若为整数或小数，直接写出。"
    ),
    k_shots=0,
    temperature=0.7,
    top_p=0.9,
    max_tokens=512,
    retry_k=1,
    format_retry=1,
    self_consistency_k=1,
)

REWRITE_GENOME = Genome(
    instruction=(
        "你是数学题解答整理专家。请把给定的解题计划和解题过程，"
        "整理成一份干净、逐步、中文的最终解答。必须包含关键计算步骤，"
        "并把最终答案写入 \\boxed{} 中。只输出整理后的解答，不要解释。"
    ),
    k_shots=0,
    temperature=0.3,
    top_p=0.9,
    max_tokens=512,
    retry_k=1,
    format_retry=0,
    self_consistency_k=1,
)


def _generate(engine, genome: Genome, user: str, timeout_sec: float = 90.0) -> str | None:
    try:
        resp, _ = engine.generate_one(user, genome, timeout_sec=timeout_sec)
        return resp.strip()
    except Exception:
        return None


def rsr_generate_one(engine, problem: dict, timeout_sec: float = 90.0) -> dict | None:
    """对一道中文数学题执行 plan→solve→verify→rewrite→re-verify。

    返回 messages 格式样本，失败返回 None。
    """
    q = problem["question"]
    ref = problem["answer_frac"]
    template = problem.get("template")

    plan = _generate(engine, PLAN_GENOME, q, timeout_sec)
    if not plan:
        return None

    solve_user = f"{q}\n\n解题计划：\n{plan}"
    full = _generate(engine, SOLVE_GENOME, solve_user, timeout_sec)
    if not full:
        return None

    r_full = grade.grade_one(ref, full)
    if not r_full["correct"]:
        return None
    ok_full, _ = grade.lint_one(full, r_full)
    if not ok_full:
        return None

    rewrite_user = f"题目：{q}\n\n解题计划：\n{plan}\n\n原始解答：\n{full}"
    clean = _generate(engine, REWRITE_GENOME, rewrite_user, timeout_sec)
    if not clean:
        clean = full

    r_clean = grade.grade_one(ref, clean)
    if not r_clean["correct"]:
        # 重写失败/跑飞时，回退到已通过验证的原始解答，不丢弃正确样本。
        clean = full
        r_clean = grade.grade_one(ref, clean)
    ok_clean, _ = grade.lint_one(clean, r_clean)
    if not ok_clean:
        clean = full

    user_content = make_user_with_hint(q, template)
    assistant_content = f"解题计划：\n{plan}\n\n详细解答：\n{clean}"
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "template": template,
        "rsr": True,
    }


def generate_rsr_dataset(
    engine, problems: list[dict], out_path=None, timeout_sec: float = 90.0
) -> list[dict]:
    """批量生成 RSR 增强样本。"""
    recs: list[dict] = []
    for i, p in enumerate(problems, 1):
        rec = rsr_generate_one(engine, p, timeout_sec)
        if rec is not None:
            recs.append(rec)
        if i % 10 == 0:
            print(f"[rsr_data] {i}/{len(problems)} accepted={len(recs)}", flush=True)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        import json
        with out_path.open("w", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return recs
