"""timeouts：任务/调用级超时基础设施。

设计原则
--------
- 超时是实验基础设施，不是 harness 可搜索参数；
- 所有臂、候选、世代使用同一预算；
- 超时任务必须算错，不能从分母里剔除；
- 超时事件写入轨迹，供审计和成本分析。

Python/MLX 的生成调用是同步的。这里用 Unix SIGALRM 在单次生成外
设置 wall-clock 上限；它适合 macOS/Linux 的主线程执行路径。若运行
环境无法使用信号，调用方仍应保留 max_tokens / retry 预算，并在每个
attempt 之间用 Deadline 做协作式超时。
"""
from __future__ import annotations

import signal
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field


class GenerationTimeout(TimeoutError):
    """单次模型调用超过 wall-clock 上限。"""

    def __init__(self, seconds: float, scope: str = "call"):
        self.seconds = float(seconds)
        self.scope = scope
        super().__init__(f"{scope} timeout after {seconds:.1f}s")


class TaskDeadlineExceeded(GenerationTimeout):
    """整个任务（含所有重试/工具步）超过截止时间。"""

    def __init__(self, seconds: float):
        super().__init__(seconds, scope="task")


@dataclass
class Deadline:
    """任务级截止时间，使用 monotonic 时钟。"""

    timeout_sec: float
    started_at: float = field(default_factory=time.monotonic)

    @property
    def deadline_at(self) -> float:
        return self.started_at + self.timeout_sec

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def remaining(self) -> float:
        return max(0.0, self.deadline_at - time.monotonic())

    @property
    def expired(self) -> bool:
        return self.remaining <= 0.0


def _can_use_sigalrm() -> bool:
    return (
        hasattr(signal, "SIGALRM")
        and hasattr(signal, "setitimer")
        and threading.current_thread() is threading.main_thread()
    )


@contextmanager
def wall_timeout(seconds: float | None, scope: str = "call"):
    """在代码块外设置 wall-clock 超时。

    - seconds <= 0 / None：不设超时；
    - 非主线程或平台不支持 SIGALRM：退化为无硬中断，由调用方在
      每次 attempt 之间检查 Deadline。
    """
    if seconds is None or float(seconds) <= 0:
        yield
        return
    if not _can_use_sigalrm():
        yield
        return

    seconds = float(seconds)
    old_handler = signal.getsignal(signal.SIGALRM)
    old_timer = signal.setitimer(signal.ITIMER_REAL, 0)

    def _handler(signum, frame):  # noqa: ANN001
        raise GenerationTimeout(seconds, scope=scope)

    signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        # 恢复外层可能存在的计时器（本项目暂不依赖嵌套，但保持正确性）。
        if old_timer[0] > 0:
            remaining = max(0.0, old_timer[0] - old_timer[1])
            if remaining > 0:
                signal.setitimer(signal.ITIMER_REAL, remaining)


def remaining_or_raise(deadline: Deadline | None) -> float | None:
    """返回任务剩余时间；已过期则抛 TaskDeadlineExceeded。"""
    if deadline is None:
        return None
    if deadline.expired:
        raise TaskDeadlineExceeded(deadline.timeout_sec)
    return deadline.remaining
