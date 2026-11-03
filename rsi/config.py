"""test-RSI 全局配置。

工程纪律：所有路径由 ROOT 推导，项目整体搬迁零改动。
冻结状态（master seed / 难度 / 模型）持久化在 state/config.json，
t=0 生成后任何代码不得改动其中的冻结字段。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# 项目根 = rsi/ 的上一级
ROOT = Path(__file__).resolve().parent.parent

# ---- 目录 ----
LOCKBOX_DIR = ROOT / "lockbox"
ARMS_DIR = ROOT / "arms"
STATE_DIR = ROOT / "state"
LEDGER_DIR = STATE_DIR / "ledger"
SCRATCH_DIR = ROOT / "scratch"
REPORT_DIR = ROOT / "report"

# ---- lockbox 资产（唯一结论性评估的题目与校验和）----
HELDOUT_FILE = LOCKBOX_DIR / "heldout_v1.jsonl"
HELDOUT_SHA = LOCKBOX_DIR / "heldout_v1.sha256"

# ---- 状态文件 ----
STATE_CONFIG = STATE_DIR / "config.json"
CALIBRATION_FILE = STATE_DIR / "calibration.json"
GLOBAL_LEDGER = LEDGER_DIR / "_global.json"

# ---- 环境 ----
# 复用用户已就绪的 MLX 虚拟环境（mlx_lm 0.31.3 / mlx 0.32.0 / py3.12）
VENV_PYTHON = Path("/Users/yingwei/Documents/code/lora/.venv/bin/python")
BASE_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"  # 已缓存，零下载

# ---- 数据纪律常量 ----
GRADE_TOL_ABS = 1e-6   # 答案绝对容差（HARDOFF §3.4）
GRADE_TOL_REL = 1e-9    # 相对容差兜底（大数值）
MIN_REASONING_CHARS = 30  # lint：boxed 前的最少推理字符数
GLOBAL_ROLES = {"heldout", "calib"}  # 全局排除的角色（跨臂共享账本）

# 数据池角色全集
ROLES = ("calib", "heldout", "probe", "pool", "promo-val")


def sanitize_proxy_env() -> None:
    """修复当前进程的代理环境变量（httpx 对 NO_PROXY 中带方括号的 IPv6 崩溃）。

    模型已全量缓存于 ~/.cache/huggingface，同时开启 HF_HUB_OFFLINE
    彻底绕开网络路径（加载更快，也不受代理故障影响）。
    """
    import re

    os.environ["HF_HUB_OFFLINE"] = "1"  # 强制覆盖：代理断连时确保走本地缓存
    for key in ("NO_PROXY", "no_proxy"):
        val = os.environ.get(key)
        if val and "[" in val:
            cleaned = re.sub(r",?\s*\[[^\]]*\]", "", val)
            os.environ[key] = cleaned


def load_state() -> dict:
    """读取冻结状态（master seed 等）。不存在时用默认值初始化并落盘。"""
    if STATE_CONFIG.exists():
        return json.loads(STATE_CONFIG.read_text(encoding="utf-8"))
    state = default_state()
    save_state(state)
    return state


def save_state(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_CONFIG.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, STATE_CONFIG)


def default_state() -> dict:
    """首次运行的初始状态（写入前需用户可见）。"""
    return {
        "master_seed": 20261101,
        "model": BASE_MODEL,
        "venv_python": str(VENV_PYTHON),
        # 以下字段由 P0 标定后冻结，代码不得覆盖非空冻结值
        "difficulty": None,       # 难度旋钮 1-5，标定后冻结
        "heldout_size": 450,      # 纪律§8：held-out ≥400
        "frozen": {
            "heldout": False,     # lockbox/freeze.py 冻结后置 True
        },
    }
