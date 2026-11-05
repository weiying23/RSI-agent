"""engine：MLX 生成引擎（load once，批量采样）。

复用用户 lora 流水线的模式：mlx_lm.load() + apply_chat_template + generate()，
改进点 = 每代只 load 一次模型，循环生成（gen_math.py 模式的批量版）。

引擎是中性的测量基建（显微镜）：Phase A/B/C 三处共用同一实现，
它不属于任何一臂的改进面。

超时纪律
--------
- call_timeout_sec：单次模型生成 wall-clock 上限；
- task_timeout_sec：一道题所有 attempt / 运行时重试的合计上限；
- 超时样本必须记录为失败（response 为空），不能从分母剔除；
- 超时预算固定，不进入 Genome，不允许 harness 自己扩大。
"""
from __future__ import annotations

import time
from typing import Callable

from . import config
from .genome import Genome, select_shots
from .timeouts import (
    Deadline,
    GenerationTimeout,
    TaskDeadlineExceeded,
    remaining_or_raise,
    wall_timeout,
)


class Engine:
    """加载一次模型，按基因组批量生成。"""

    def __init__(
        self,
        model_name: str | None = None,
        adapter_path: str | None = None,
        call_timeout_sec: float | None = None,
        task_timeout_sec: float | None = None,
    ):
        from mlx_lm import load

        config.sanitize_proxy_env()
        self.model_name = model_name or config.load_state().get("model") or config.BASE_MODEL
        self.adapter_path = adapter_path
        self.call_timeout_sec = float(
            config.DEFAULT_CALL_TIMEOUT_SEC if call_timeout_sec is None else call_timeout_sec
        )
        self.task_timeout_sec = float(
            config.DEFAULT_TASK_TIMEOUT_SEC if task_timeout_sec is None else task_timeout_sec
        )
        t0 = time.time()
        self.model, self.tok = load(self.model_name, adapter_path=adapter_path)
        self.load_seconds = time.time() - t0
        self.n_generated = 0
        self.tokens_generated = 0
        self.gen_seconds = 0.0
        self.timeout_count = 0
        self.timeout_seconds = 0.0

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
        self,
        question: str,
        genome: Genome,
        shots: list[dict] | None = None,
        timeout_sec: float | None = None,
    ) -> tuple[str, int]:
        """返回 (response, token数)。

        单次 MLX 生成受 wall-clock timeout 约束。若超时抛 GenerationTimeout；
        调用方必须把该 attempt 记录为失败。
        """
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=genome.temperature, top_p=genome.top_p)
        msgs = self.build_messages(question, genome, shots)
        prompt = self.tok.apply_chat_template(msgs, add_generation_prompt=True)
        timeout_sec = self.call_timeout_sec if timeout_sec is None else float(timeout_sec)
        t0 = time.time()
        try:
            with wall_timeout(timeout_sec, scope="call"):
                resp = generate(
                    self.model,
                    self.tok,
                    prompt=prompt,
                    max_tokens=genome.max_tokens,
                    sampler=sampler,
                )
        except GenerationTimeout:
            dt = time.time() - t0
            self.timeout_count += 1
            self.timeout_seconds += dt
            raise
        dt = time.time() - t0
        n_tok = len(self.tok.encode(resp))
        self.n_generated += 1
        self.tokens_generated += n_tok
        self.gen_seconds += dt
        return resp, n_tok

    def generate_one_checked(
        self,
        question: str,
        genome: Genome,
        shots: list[dict] | None = None,
        max_retry: int | None = None,
        timeout_sec: float | None = None,
        deadline: Deadline | None = None,
    ) -> tuple[str, int]:
        """生成 + 运行时 harness：格式重试 + 可选自一致性投票。

        - format_retry：输出没写 \\boxed{} 或过短时，低温短输出重试；
        - self_consistency_k：独立采样 k 次，按规范化答案多数投票；
        - 两种逻辑都在任务截止时间内，超时向上抛出并记为失败。

        返回值是（最终选中的 response, 总生成 token 数）。
        """
        from . import grade as _grade

        samples: list[tuple[str, int]] = []
        k = max(1, int(genome.self_consistency_k))
        retries = genome.format_retry if max_retry is None else int(max_retry)

        for _ in range(k):
            resp, tok = self._generate_checked_single(
                question, genome, shots,
                max_retry=retries,
                timeout_sec=timeout_sec,
                deadline=deadline,
            )
            samples.append((resp, tok))

        if len(samples) == 1:
            return samples[0]

        # 答案多数投票：用统一 grade 归一化，避免 0.5 与 1/2 被当成不同答案。
        buckets: dict[object, list[tuple[str, int]]] = {}
        for resp, tok in samples:
            boxed, n = _grade.extract_boxed(resp)
            if n <= 0 or boxed is None:
                continue
            ans = _grade.normalize_answer(boxed)
            if ans is None:
                continue
            buckets.setdefault(ans, []).append((resp, tok))

        if not buckets:
            # 全部格式失败：返回最短响应，后续 distill / grade 会判错。
            return min(samples, key=lambda x: len(x[0]))

        best_answer = max(
            buckets,
            key=lambda a: (len(buckets[a]), -min(len(r) for r, _ in buckets[a])),
        )
        best_resp, best_tok = min(buckets[best_answer], key=lambda x: len(x[0]))
        total_tok = sum(tok for _, tok in samples)
        return best_resp, total_tok

    def _generate_checked_single(
        self,
        question: str,
        genome: Genome,
        shots: list[dict] | None,
        max_retry: int,
        timeout_sec: float | None,
        deadline: Deadline | None,
    ) -> tuple[str, int]:
        """单条候选轨迹：生成 + 格式失败重试。"""
        from . import grade as _grade

        def _call_once(g: Genome) -> tuple[str, int]:
            remaining = remaining_or_raise(deadline)
            per_call = timeout_sec
            if remaining is not None:
                per_call = remaining if per_call is None else min(float(per_call), remaining)
            resp, tok = self.generate_one(question, g, shots, timeout_sec=per_call)
            if deadline is not None and deadline.expired:
                raise TaskDeadlineExceeded(deadline.timeout_sec)
            return resp, tok

        resp, tok = _call_once(genome)
        for _ in range(max_retry):
            boxed, n = _grade.extract_boxed(resp)
            if n > 0 and len(resp.strip()) >= 20:
                return resp, tok
            # 换策略重试：低温 + 短输出（重试只补格式，不做完整推理）
            g2 = genome.__class__(
                **{**genome.to_dict(),
                   "temperature": 0.3,
                   "max_tokens": min(genome.max_tokens, 128)}
            )
            resp2, tok2 = _call_once(g2)
            resp, tok = resp2, tok + tok2
        return resp, tok

    def _generate_one_with_deadline(
        self,
        question: str,
        genome: Genome,
        shots: list[dict] | None,
        timeout_sec: float | None,
        deadline: Deadline | None,
    ) -> tuple[str, int]:
        remaining = remaining_or_raise(deadline)
        per_call = timeout_sec
        if remaining is not None:
            per_call = remaining if per_call is None else min(float(per_call), remaining)
        resp, tok = self.generate_one(question, genome, shots, timeout_sec=per_call)
        if deadline is not None and deadline.expired:
            raise TaskDeadlineExceeded(deadline.timeout_sec)
        return resp, tok

    def generate_batch(
        self,
        problems: list[dict],
        genome: Genome,
        k: int | None = None,
        shot_pool: list[dict] | None = None,
        progress: Callable[[int, int], None] | None = None,
        call_timeout_sec: float | None = None,
        task_timeout_sec: float | None = None,
    ) -> list[dict]:
        """对每题采 k 次（retry_k），返回样本记录（含失败/超时轨迹）。

        每个 attempt 都记录在 samples 中。若任务总预算耗尽，会为剩余
        attempt 写 timeout 占位记录，保证 distill 的稳定性分母仍为 k。
        """
        from . import grade as _grade

        k = k if k is not None else genome.retry_k
        call_timeout = self.call_timeout_sec if call_timeout_sec is None else float(call_timeout_sec)
        task_timeout = self.task_timeout_sec if task_timeout_sec is None else float(task_timeout_sec)
        samples: list[dict] = []
        total = len(problems) * k
        done = 0

        def _record(
            p: dict, attempt: int, response: str, tokens: int, status: str,
            elapsed: float, scope: str | None = None,
        ) -> None:
            nonlocal done
            boxed, n_box = _grade.extract_boxed(response)
            samples.append(
                {
                    "pid": p["id"],
                    "attempt": attempt,
                    "response": response,
                    "tokens": int(tokens),
                    "template": p.get("template"),
                    "status": status,
                    "timeout_scope": scope,
                    "elapsed_sec": round(float(elapsed), 3),
                    "format_ok": bool(n_box > 0 and len(response.strip()) >= 20),
                }
            )
            done += 1
            if progress:
                progress(done, total)

        for p in problems:
            shots = select_shots(shot_pool or [], genome)
            deadline = Deadline(task_timeout)
            for attempt in range(1, k + 1):
                if deadline.expired:
                    _record(p, attempt, "", 0, "timeout", deadline.elapsed, "task")
                    continue
                t0 = time.monotonic()
                try:
                    resp, n_tok = self.generate_one_checked(
                        p["question"],
                        genome,
                        shots,
                        max_retry=genome.format_retry,
                        timeout_sec=min(call_timeout, deadline.remaining),
                        deadline=deadline,
                    )
                except GenerationTimeout as exc:
                    _record(
                        p, attempt, "", 0, "timeout",
                        time.monotonic() - t0, getattr(exc, "scope", "call"),
                    )
                    continue
                _record(
                    p, attempt, resp, n_tok, "ok",
                    time.monotonic() - t0, None,
                )
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
            "timeout_count": self.timeout_count,
            "timeout_seconds": round(self.timeout_seconds, 1),
            "call_timeout_sec": self.call_timeout_sec,
            "task_timeout_sec": self.task_timeout_sec,
            "adapter": self.adapter_path,
            "model": self.model_name,
            "load_seconds": round(self.load_seconds, 1),
        }

    def close(self) -> None:
        import mlx.core as mx

        self.model = None
        mx.clear_cache()
