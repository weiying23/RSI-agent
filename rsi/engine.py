"""engine：MLX 生成引擎（load once，批量采样）。

复用用户 lora 流水线的模式：mlx_lm.load() + apply_chat_template + generate()，
改进点 = 每代只 load 一次模型，循环生成（gen_math.py 模式的批量版）。

引擎是中性的测量基建（显微镜）：Phase A/B/C 三处共用同一实现，
它不属于任何一臂的改进面。
"""
from __future__ import annotations

import time
from typing import Callable

from . import config
from .genome import Genome, select_shots


class Engine:
    """加载一次模型，按基因组批量生成。"""

    def __init__(self, model_name: str | None = None, adapter_path: str | None = None):
        from mlx_lm import load

        config.sanitize_proxy_env()
        self.model_name = model_name or config.load_state().get("model") or config.BASE_MODEL
        self.adapter_path = adapter_path
        t0 = time.time()
        self.model, self.tok = load(self.model_name, adapter_path=adapter_path)
        self.load_seconds = time.time() - t0
        self.n_generated = 0
        self.tokens_generated = 0
        self.gen_seconds = 0.0

    # ------------------------------------------------------------------
    # prompt 构造
    # ------------------------------------------------------------------

    def build_messages(
        self, question: str, genome: Genome, shots: list[dict] | None = None
    ) -> list[dict]:
        msgs: list[dict] = [{"role": "system", "content": genome.instruction}]
        for s in shots or []:
            msgs.append({"role": "user", "content": s["question"]})
            msgs.append({"role": "assistant", "content": s["solution"]})
        msgs.append({"role": "user", "content": question})
        return msgs

    # ------------------------------------------------------------------
    # 生成
    # ------------------------------------------------------------------

    def generate_one(
        self, question: str, genome: Genome, shots: list[dict] | None = None
    ) -> tuple[str, int]:
        """返回 (response, token数)。"""
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=genome.temperature, top_p=genome.top_p)
        msgs = self.build_messages(question, genome, shots)
        prompt = self.tok.apply_chat_template(msgs, add_generation_prompt=True)
        t0 = time.time()
        resp = generate(
            self.model,
            self.tok,
            prompt=prompt,
            max_tokens=genome.max_tokens,
            sampler=sampler,
        )
        dt = time.time() - t0
        n_tok = len(self.tok.encode(resp))
        self.n_generated += 1
        self.tokens_generated += n_tok
        self.gen_seconds += dt
        return resp, n_tok

    def generate_one_checked(
        self, question: str, genome: Genome, shots: list[dict] | None = None,
        max_retry: int = 1,
    ) -> tuple[str, int]:
        """生成 + 运行时控制逻辑（7.3：格式强制 + 空答案拦截 + 换策略重试）。

        输出没写 \\boxed{} 或过短 → 升温换策略重试一次（harness 的运行时逻辑）。
        held-out 评估走 run_eval 的裸 generate_one，不带此逻辑——归因不虚高。
        """
        from . import grade as _grade

        resp, tok = self.generate_one(question, genome, shots)
        for _ in range(max_retry):
            boxed, n = _grade.extract_boxed(resp)
            if n > 0 and len(resp.strip()) >= 20:
                return resp, tok
            # 换策略重试：低温 + 短输出（重试只是补格式，不需完整推理）
            g2 = genome.__class__(
                **{**genome.to_dict(),
                   "temperature": 0.3,
                   "max_tokens": min(genome.max_tokens, 128)}
            )
            resp2, tok2 = self.generate_one(question, g2, shots)
            resp, tok = resp2, tok + tok2
        return resp, tok

    def generate_batch(
        self,
        problems: list[dict],
        genome: Genome,
        k: int | None = None,
        shot_pool: list[dict] | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> list[dict]:
        """对每题采 k 次（retry_k），返回样本记录（含失败轨迹，一等产出）。"""
        k = k if k is not None else genome.retry_k
        samples: list[dict] = []
        total = len(problems) * k
        done = 0
        for i, p in enumerate(problems):
            shots = select_shots(shot_pool or [], genome)
            for attempt in range(1, k + 1):
                resp, n_tok = self.generate_one_checked(p["question"], genome, shots)
                samples.append(
                    {
                        "pid": p["id"],
                        "attempt": attempt,
                        "response": resp,
                        "tokens": n_tok,
                        "template": p.get("template"),
                    }
                )
                done += 1
                if progress:
                    progress(done, total)
                _ = i
        return samples

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        tps = self.tokens_generated / self.gen_seconds if self.gen_seconds > 0 else 0.0
        return {
            "n_generated": self.n_generated,
            "tokens_generated": self.tokens_generated,
            "gen_seconds": round(self.gen_seconds, 1),
            "tok_per_sec": round(tps, 1),
            "adapter": self.adapter_path,
            "model": self.model_name,
            "load_seconds": round(self.load_seconds, 1),
        }

    def close(self) -> None:
        import mlx.core as mx

        self.model = None
        mx.clear_cache()
