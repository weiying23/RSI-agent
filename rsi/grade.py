"""grade：全系统唯一的判分实现（纪律§2）。

lockbox/grade_heldout.py 只是本模块的只读包装 + SHA256 校验，
两者共享同一算法——不存在「训练用宽松判分器 / 评估用严格判分器」的错位。

能力：
- \\boxed{...} 抽取（含嵌套大括号、\\fbox、\\dfrac 等变体），统计 box 数
- 答案归一化：整数 / 小数 / 分数 / 带分数 / 百分号 / 万亿单位 / \\text 包装
- 数值比对：精确分数相等，或 |Δ|≤1e-6，或相对误差≤1e-9
- lint：训练样本资格（单一 boxed、最少推理、无退化）
- hack_flags：多 box 率、解析失败率、无推理直答率等（轨迹档案一等产出）
"""
from __future__ import annotations

import re
from fractions import Fraction

from . import config


# ---------------------------------------------------------------------------
# \boxed 抽取
# ---------------------------------------------------------------------------

def _extract_braced(text: str, start: int) -> tuple[str, int] | None:
    """从 text[start]（应为 '{'）开始提取配对大括号内容，返回 (content, end_next_idx)。"""
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i], i + 1
    return None  # 未闭合


def extract_boxed(text: str) -> tuple[str | None, int]:
    """返回（最后一个 \\boxed{...} 的内容, box 总数）。"""
    n = 0
    last: str | None = None
    for m in re.finditer(r"\\(?:boxed|fbox)\s*", text):
        got = _extract_braced(text, m.end())
        if got is None:
            # \boxed 42 无花括号的退化写法：取到行尾/空白
            tail = text[m.end() : m.end() + 64].strip()
            tail = re.split(r"[\s\n。；;]", tail)[0] if tail else ""
            if not tail:
                continue
            n += 1
            last = tail
            continue
        content, _end = got
        n += 1
        last = content
    return last, n


# ---------------------------------------------------------------------------
# 答案归一化
# ---------------------------------------------------------------------------

_CLEAN_PATTERNS = [
    (r"\\displaystyle", ""),
    (r"\\left|\\right", ""),
    (r"\\text\s*", ""),
    (r"\\mathrm|\\mathbf|\\mathit", ""),
    (r"\\!", ""),
    (r"\\[,;: ]", " "),
    (r"\\\\", " "),
    (r"\\approx|\\simeq", ""),
    (r"[≈≒≅]", ""),
    (r"^(?:答案|最终答案|ans(?:wer)?)\s*[=:：]?\s*", ""),
]
# 数值单位（直接剥离）——模型常见 \boxed{42千米}
_PLAIN_UNITS = [
    "平方厘米", "平方千米", "平方米", "千分之一", "千米", "公斤", "厘米",
    "小时", "分钟", "个", "只", "条", "元", "米", "人", "天", "张", "本",
]
# 乘性单位：1.2万 → 12000
_MULT_UNITS = [("亿", 100000000), ("万", 10000)]


def _frac_to_slash(s: str) -> str:
    """\\dfrac{3}{4} → 3/4。必须在任何大括号删除之前执行。"""
    for _ in range(3):  # 处理浅层嵌套
        new = re.sub(
            r"\\[dtc]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"\1/\2", s, flags=re.IGNORECASE
        )
        if new == s:
            break
        s = new
    # \frac 3 4 无括号形式
    s = re.sub(r"\\[dtc]?frac\s+(\d+)\s+(\d+)", r"\1/\2", s, flags=re.IGNORECASE)
    return s


def _try_parse(s: str) -> Fraction | None:
    """单一形式解析（不再改写字符串）。

    带分数必须带分隔符（「又」或至少一个空白）：
    "1又1/2" / "1 1/2" 是带分数，"17/15" 只是假分数——
    无分隔符形式绝不按带分数解释（否则 17/15 会被拆成 1+7/15）。
    """
    if not s:
        return None
    # 带分数（先于普通解析，且要求显式分隔符）
    for pat in (r"(-?\d+)\s*又\s*(\d+)\s*/\s*(\d+)", r"(-?\d+)\s+(\d+)\s*/\s*(\d+)"):
        m = re.fullmatch(pat, s)
        if m:
            whole, num, den = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if den == 0:
                return None
            sign = -1 if whole < 0 else 1
            return Fraction(whole) + sign * Fraction(num, den)
    t = s.replace(" ", "")  # 紧凑形式用于其余判定
    if not t:
        return None
    # 百分号：75% → 3/4
    if t.endswith("%"):
        try:
            return Fraction(t[:-1]) / 100
        except (ValueError, ZeroDivisionError):
            return None
    # 乘性单位：12万 / 1.2亿（Fraction 精确乘法）
    for suffix, mult in _MULT_UNITS:
        if t.endswith(suffix):
            try:
                return Fraction(t[: -len(suffix)]) * mult
            except (ValueError, ZeroDivisionError):
                return None
    # 普通分数 / 整数 / 小数（Fraction 原生支持 "3/4" "0.75" "-2"）
    try:
        return Fraction(t)
    except (ValueError, ZeroDivisionError):
        return None


def normalize_answer(raw: str) -> Fraction | None:
    """把 \\boxed 里的字符串归一化为精确 Fraction。失败返回 None。"""
    if raw is None:
        return None
    s = raw.strip()
    s = _frac_to_slash(s)  # 必须最先：\frac{a}{b} → a/b
    for pat, rep in _CLEAN_PATTERNS:
        s = re.sub(pat, rep, s, flags=re.IGNORECASE)
    s = s.replace("{", "").replace("}", "")
    s = s.replace(",", "").replace("\u00a0", " ")  # 保留普通空格：带分数分隔符
    s = s.strip("。．.;；:：$=+ \t\n")
    s = re.sub(r"^(?:是|为|即|得|等于)", "", s).strip()
    if not s:
        return None
    s = s.replace("−", "-").replace("—", "-")
    # 先按原样解析（带分数要求空格/又分隔符，见 _try_parse）
    v = _try_parse(s)
    if v is not None:
        return v
    # 失败再剥离数值单位后缀重试（单位剥离后重新尝试，含紧凑化）
    for unit in _PLAIN_UNITS:
        if s.endswith(unit):
            v = _try_parse(s[: -len(unit)])
            if v is not None:
                return v
            break
    return None


# ---------------------------------------------------------------------------
# 判分
# ---------------------------------------------------------------------------

def _matches(ans: Fraction, ref: Fraction) -> bool:
    if ans == ref:
        return True
    fa, fr = float(ans), float(ref)
    if abs(fa - fr) <= config.GRADE_TOL_ABS:
        return True
    denom = max(abs(fr), 1e-30)
    return abs(fa - fr) / denom <= config.GRADE_TOL_REL


def reasoning_chars(response: str, boxed_start: int | None = None) -> int:
    """boxed 之前的「推理字符量」（去空白）。boxed_start 缺省时自动定位。"""
    if boxed_start is None:
        m = None
        for m in re.finditer(r"\\(?:boxed|fbox)", response):
            pass
        boxed_start = m.start() if m else len(response)
    head = response[:boxed_start]
    return len(re.sub(r"\s", "", head))


def grade_one(ref_answer_frac: str, response: str) -> dict:
    """判分单条。ref_answer_frac 是 Problem.answer_frac（Fraction 的 str 形式）。"""
    ref = Fraction(ref_answer_frac)
    boxed, n_boxed = extract_boxed(response)
    flags: dict = {
        "n_boxed": n_boxed,
        "no_boxed": n_boxed == 0,
        "multi_boxed": n_boxed > 1,
        "resp_chars": len(response),
        "reasoning_chars": reasoning_chars(response),
    }
    if boxed is None:
        flags["parse_fail"] = True
        flags["no_reasoning_direct"] = flags["reasoning_chars"] < config.MIN_REASONING_CHARS
        return {"correct": False, "flags": flags, "extracted": None}
    ans = normalize_answer(boxed)
    flags["parse_fail"] = ans is None
    flags["no_reasoning_direct"] = flags["reasoning_chars"] < config.MIN_REASONING_CHARS
    correct = ans is not None and _matches(ans, ref)
    return {"correct": correct, "flags": flags, "extracted": boxed}


# ---------------------------------------------------------------------------
# lint：训练样本资格（Phase B 蒸馏前的过滤）
# ---------------------------------------------------------------------------

def _degenerate(response: str) -> bool:
    """退化输出检测：连续 6 行以上完全相同，视为重复坍缩。"""
    lines = [ln.strip() for ln in response.splitlines() if ln.strip()]
    run = best = 1
    for i in range(1, len(lines)):
        run = run + 1 if lines[i] == lines[i - 1] else 1
        best = max(best, run)
    return best >= 6


def lint_one(response: str, result: dict | None = None) -> tuple[bool, dict]:
    """训练资格：正确 ∧ 单一 boxed ∧ 有推理 ∧ 非退化 ∧ 长度 sane。"""
    r = result or {}
    boxed, n_boxed = extract_boxed(response)
    reasons = {
        "no_boxed": n_boxed == 0,
        "multi_boxed": n_boxed > 1,
        "too_short_reasoning": reasoning_chars(response) < config.MIN_REASONING_CHARS,
        "degenerate": _degenerate(response),
        "too_long": len(response) > 6000,
        "not_correct": (r.get("correct") is not True),
    }
    return (not any(reasons.values())), reasons


# ---------------------------------------------------------------------------
# 汇总（hack_flags 聚合）
# ---------------------------------------------------------------------------

def summarize(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {"n": 0}
    acc = sum(1 for r in results if r["correct"]) / n
    chars = [r["flags"]["resp_chars"] for r in results]
    return {
        "n": n,
        "accuracy": round(acc, 4),
        "mean_resp_chars": round(sum(chars) / n, 1),
        "multi_boxed_rate": round(sum(1 for r in results if r["flags"]["multi_boxed"]) / n, 4),
        "parse_fail_rate": round(sum(1 for r in results if r["flags"].get("parse_fail")) / n, 4),
        "no_reasoning_rate": round(
            sum(1 for r in results if r["flags"].get("no_reasoning_direct")) / n, 4
        ),
    }
