"""genome：harness 基因组（L0 改进面的全部内容）。

H_0 = 「市面出厂设置」：无 few-shot、默认指令、temp 0.7。
arm_H 从这个原点进化，正好是市面产品的缩影（HANDOFF §3.3）。

改进面纪律：improver = (W, H) 的一切都在这里 + 权重，
改这里之外的任何东西（判分器/held-out/评估协议）都违反纪律§2–4。
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field, replace


INSTRUCTION_DEFAULT = (
    "解下面的数学题。请逐步推理，写出关键计算过程，"
    "最终答案写入 \\boxed{} 中。答案若为分数，用最简分数 a/b 表示；"
    "若为整数或小数，直接写出。"
)

SHOT_POLICIES = ("none", "recent_success", "hard_win")


@dataclass(frozen=True)
class Genome:
    instruction: str = INSTRUCTION_DEFAULT
    k_shots: int = 0
    shot_policy: str = "none"
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 384
    retry_k: int = 3

    # ---- 边界（HANDOFF §3.3）----
    BOUNDS = {
        "k_shots": (0, 3),
        "temperature": (0.3, 1.0),
        "top_p": (0.8, 0.95),
        "max_tokens": (256, 512),
        "retry_k": (1, 4),
    }

    def validate(self) -> None:
        if not self.instruction or len(self.instruction) < 8:
            raise ValueError("instruction 过短")
        if self.shot_policy not in SHOT_POLICIES:
            raise ValueError(f"未知 shot_policy: {self.shot_policy}")
        if self.k_shots > 0 and self.shot_policy == "none":
            raise ValueError("k_shots>0 需要非 none 的 shot_policy")
        for name, (lo, hi) in self.BOUNDS.items():
            v = getattr(self, name)
            if not (lo <= v <= hi):
                raise ValueError(f"{name}={v} 越界 [{lo}, {hi}]")

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("BOUNDS", None)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Genome":
        g = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__ and k != "BOUNDS"})
        g.validate()
        return g


def H0() -> Genome:
    """出厂设置基因组。"""
    return Genome()


# ---------------------------------------------------------------------------
# 变异算子（Phase A 的程序化候选；W 自改指令的候选由 engine 生成）
# ---------------------------------------------------------------------------

def perturb(base: Genome, rng: random.Random, n_params: int = 2) -> Genome:
    """对 1~n_params 个数值参数做随机扰动，返回新基因组（保持边界）。"""
    fields_avail = ["k_shots", "temperature", "top_p", "max_tokens", "retry_k"]
    k = rng.randint(1, n_params)
    chosen = rng.sample(fields_avail, k)
    changes: dict = {}
    for f in chosen:
        lo, hi = Genome.BOUNDS[f]
        cur = getattr(base, f)
        if f in ("k_shots", "retry_k"):  # 整数参数
            step = rng.choice([-2, -1, 1, 2])
            new = max(lo, min(hi, cur + step))
        elif f == "max_tokens":
            step = rng.choice([-128, -64, 64, 128])
            new = max(lo, min(hi, cur + step))
        else:  # 浮点参数
            step = rng.choice([-0.2, -0.1, 0.1, 0.2])
            new = round(max(lo, min(hi, cur + step)), 3)
        changes[f] = new
    # 温度扰动可能需要配套 shot_policy（k_shots>0 时）
    if changes.get("k_shots", base.k_shots) > 0 and base.shot_policy == "none":
        changes["shot_policy"] = rng.choice(["recent_success", "hard_win"])
    g = replace(base, **changes)
    g.validate()
    return g


# ---------------------------------------------------------------------------
# few-shot 示例选择（shot_policy 的实现）
# ---------------------------------------------------------------------------

def select_shots(pool: list[dict], genome: Genome) -> list[dict]:
    """从验证解池选 k_shots 个示例。

    pool 条目：{"question","solution","difficulty","success_rate","gen",...}
      recent_success: 最新优先（权重侧近期的成功风格）
      hard_win:       成功率最低的成功解优先（模型能力前缘，「难而不倒」）
    """
    if genome.k_shots <= 0 or not pool:
        return []
    if genome.shot_policy == "recent_success":
        ranked = sorted(pool, key=lambda e: (-e.get("gen", 0),))
    elif genome.shot_policy == "hard_win":
        ranked = sorted(
            pool, key=lambda e: (e.get("success_rate", 1.0), -e.get("difficulty", 3))
        )
    else:
        raise ValueError(f"shot_policy={genome.shot_policy} 不支持选例")
    return ranked[: genome.k_shots]
