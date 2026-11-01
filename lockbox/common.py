"""lockbox 公共：held-out 校验和验证（改进面之外的不动点）。

lockbox/ 下的脚本是全系统唯一允许读取 heldout_v1.jsonl 的代码
（纪律§3：held-out 三不——不训练、不做选择、不被循环代码读取）。
校验和不符 → 拒绝出分。这是防篡改的最后一道防线。
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # rsi 包可导入（grade 是唯一判分实现）

HELDOUT_FILE = ROOT / "lockbox" / "heldout_v1.jsonl"
HELDOUT_SHA = ROOT / "lockbox" / "heldout_v1.sha256"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def verify_heldout() -> dict:
    """校验 held-out 完整性。任何不符 → 异常（fail closed）。"""
    if not HELDOUT_FILE.exists():
        raise FileNotFoundError("held-out 尚未冻结：先运行 lockbox/freeze.py")
    if not HELDOUT_SHA.exists():
        raise FileNotFoundError("held-out 校验和文件缺失：lockbox 不完整，拒绝出分")
    expected = HELDOUT_SHA.read_text(encoding="utf-8").split()[0].strip()
    actual = sha256_of(HELDOUT_FILE)
    if expected != actual:
        raise RuntimeError(
            f"held-out 校验和不符！expected={expected[:16]}… actual={actual[:16]}…\n"
            "题目文件被改动过——评估结果不可信，拒绝继续。"
        )
    problems = [
        json.loads(ln)
        for ln in HELDOUT_FILE.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    return {"n": len(problems), "problems": problems, "sha256": actual}


def load_problems(eval_n: int | None = None) -> list[dict]:
    data = verify_heldout()
    probs = data["problems"]
    if eval_n is not None:
        probs = probs[:eval_n]  # 固定前缀子集：跨臂/跨代可比
    return probs
