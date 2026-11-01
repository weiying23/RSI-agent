"""env_gen：程序化中文数学题生成器（唯一数据源）。

设计约束（HANDOFF §3.4）：
- ≥8 个模板，难度旋钮 1–5 映射操作数范围
- 标准答案由符号计算（整数 + Fraction）得出，零 LLM 参与
- 实例 ID = 模板名 + 参数的确定性哈希（去重与账本排除的键）
- 出题必经 sample()：查账本排除已用 ID，指派角色，写回账本

出题后题目即进入对应角色的生命周期（probe/pool/promo-val 用后语义上烧毁，
账本条目永久保留，保证同一实例永不以另一角色复用）。
"""
from __future__ import annotations

import hashlib
import json
import random
from fractions import Fraction
from math import gcd as _gcd

from . import config
from .ledger import Ledger

# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def frac_str(f: Fraction) -> str:
    """答案规范显示：整数直写，否则最简分数 a/b。"""
    f = Fraction(f)
    if f.denominator == 1:
        return str(f.numerator)
    return f"{f.numerator}/{f.denominator}"


def _zhe_str(z: Fraction) -> str:
    """折扣的中文读法：8 → 八折，7.5 → 七五折。z 必须是 x 或 x.5。"""
    digits = "零一二三四五六七八九"
    assert z.denominator in (1, 2)
    if z.denominator == 1:
        return f"{digits[z.numerator]}折"
    whole = z.numerator // 2
    half = 5  # x.5 折读作「x五折」
    return f"{digits[whole]}{digits[half]}折"


def make_id(template: str, params: dict) -> str:
    key = f"{template}|{json.dumps(params, sort_keys=True, ensure_ascii=False)}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 模板：每个返回 (params, question, Fraction answer)
# 参数必须完整进 params（实例 ID 的确定性来源）。
# ---------------------------------------------------------------------------

def t_mult_chain(rng: random.Random, d: int):
    n_ops = {1: 2, 2: 3, 3: 3, 4: 4, 5: 4}[d]
    lo, hi = {1: (2, 9), 2: (2, 9), 3: (3, 15), 4: (2, 12), 5: (5, 29)}[d]
    ops = [rng.randint(lo, hi) for _ in range(n_ops)]
    ans = 1
    for x in ops:
        ans *= x
    q = "计算：" + " × ".join(str(x) for x in ops) + " = ？"
    return {"ops": ops}, q, Fraction(ans)


def t_frac_addsub(rng: random.Random, d: int):
    n = 3 if d >= 4 else 2
    dlo, dhi = {1: (2, 6), 2: (2, 10), 3: (3, 14), 4: (3, 16), 5: (5, 24)}[d]
    # 构造时保证中间结果与最终结果都为正
    for _ in range(200):
        denoms = [rng.randint(dlo, dhi) for _ in range(n)]
        num1 = [rng.randint(1, max(2, denoms[0] - 1)) for _ in range(n)]
        fracs = [Fraction(num1[i], denoms[i]) for i in range(n)]
        ops = ["+"] + [rng.choice("+-") for _ in range(n - 1)]
        # 确保非整分数值且各不相同（去平凡）
        if all(f.denominator > 1 for f in fracs) and len(set(fracs)) == n:
            expr_val = fracs[0]
            ok = True
            for i in range(1, n):
                expr_val = expr_val + fracs[i] if ops[i] == "+" else expr_val - fracs[i]
                if expr_val <= 0:
                    ok = False
                    break
            if ok:
                parts = [f"{f.numerator}/{f.denominator}" for f in fracs]
                segs = [parts[0]]
                for i in range(1, n):
                    segs.append(f"{ops[i]} {parts[i]}")
                q = "计算：" + " ".join(segs) + " = ？"
                return {"fractions": [str(f) for f in fracs], "ops": ops}, q, expr_val
    raise RuntimeError("frac_addsub: 无法构造合法题目")


def t_lcm(rng: random.Random, d: int):
    n = 3 if d >= 4 else 2
    lo, hi = {1: (2, 12), 2: (3, 20), 3: (4, 30), 4: (3, 25), 5: (4, 45)}[d]
    for _ in range(200):
        nums = []
        while len(nums) < n:
            x = rng.randint(lo, hi)
            if x not in nums and x > 1:
                nums.append(x)
        if len(set(nums)) == n:
            a = nums[0]
            for b in nums[1:]:
                a = a * b // _gcd(a, b)
            if a > max(nums):  # lcm 非平凡
                q = f"求 {' 和 '.join(str(x) for x in nums)} 的最小公倍数。"
                return {"nums": nums}, q, Fraction(a)
    raise RuntimeError("lcm: 无法构造合法题目")


def t_gcd(rng: random.Random, d: int):
    n = 3 if d >= 4 else 2
    lo, hi = {1: (4, 24), 2: (6, 40), 3: (8, 60), 4: (12, 90), 5: (20, 150)}[d]
    for _ in range(200):
        nums = []
        while len(nums) < n:
            x = rng.randint(lo, hi)
            if x not in nums:
                nums.append(x)
        g = nums[0]
        for b in nums[1:]:
            g = _gcd(g, b)
        if g >= 2 and len(set(nums)) == n:
            q = f"求 {' 和 '.join(str(x) for x in nums)} 的最大公约数。"
            return {"nums": nums}, q, Fraction(g)
    raise RuntimeError("gcd: 无法构造合法题目")


def t_percent(rng: random.Random, d: int):
    if d == 1:
        N = rng.randrange(2, 11) * 20
        p = rng.randrange(1, 10) * 10
        q = f"一个数是 {N}，它的 {p}% 是多少？"
        return {"N": N, "p": p, "mode": "fwd"}, q, Fraction(N * p, 100)
    if d == 2:
        N = rng.randrange(2, 31) * 10
        p = rng.randrange(1, 20) * 5
        q = f"一个数是 {N}，它的 {p}% 是多少？"
        return {"N": N, "p": p, "mode": "fwd"}, q, Fraction(N * p, 100)
    if d == 3:  # 反向：已知部分求整体（答案恒为整数）
        p = rng.randrange(2, 20) * 5
        N = rng.randrange(4, 41) * 20  # 保证 N*p/100 为整数
        X = N * p // 100
        q = f"一个数的 {p}% 是 {X}，这个数是多少？"
        return {"N": N, "p": p, "mode": "rev"}, q, Fraction(N)
    if d == 4:
        N = rng.randrange(50, 501)
        p = rng.randrange(4, 97)
        q = f"一个数是 {N}，它的 {p}% 是多少？"
        return {"N": N, "p": p, "mode": "fwd"}, q, Fraction(N * p, 100)
    # d=5 反向，p 任意整数，X 带小数
    p = rng.randrange(4, 97)
    N = rng.randrange(100, 1001)
    X = Fraction(N * p, 100)
    xs = frac_str(X) if X.denominator != 1 else f"{float(X):g}"
    q = f"一个数的 {p}% 是 {xs}，这个数是多少？"
    return {"N": N, "p": p, "mode": "rev"}, q, Fraction(N)


def t_average(rng: random.Random, d: int):
    k = {1: 3, 2: 4, 3: 3, 4: 5, 5: 4}[d]
    lo, hi = {1: (1, 20), 2: (1, 50), 3: (1, 50), 4: (1, 99), 5: (10, 199)}[d]
    force_div = d <= 2
    for _ in range(200):
        nums = [rng.randint(lo, hi) for _ in range(k)]
        s = sum(nums)
        if force_div:
            if s % k != 0:
                continue
        elif s % k == 0:
            continue  # 高难度强制非整除（分数答案）
        q = f"求这{k}个数的平均数：" + "、".join(str(x) for x in nums) + "。"
        return {"nums": nums, "k": k}, q, Fraction(s, k)
    raise RuntimeError("average: 无法构造合法题目")


def t_travel(rng: random.Random, d: int):
    if d == 1:
        v = rng.randrange(1, 10) * 10
        t = rng.randint(1, 5)
        q = f"一辆汽车以每小时 {v} 千米的速度行驶了 {t} 小时，共行驶了多少千米？"
        return {"v": v, "t": t, "mode": "fwd"}, q, Fraction(v * t)
    if d == 2:
        v = rng.randint(12, 90)
        t = rng.randint(1, 6)
        q = f"一辆汽车以每小时 {v} 千米的速度行驶了 {t} 小时，共行驶了多少千米？"
        return {"v": v, "t": t, "mode": "fwd"}, q, Fraction(v * t)
    if d == 3:
        v = rng.randrange(5, 26) * 4
        t = Fraction(rng.choice([3, 5, 7, 9]), 4)  # 0.75,1.25,1.75,2.25 → 积为整数
        q = (
            f"一辆汽车以每小时 {v} 千米的速度行驶了 {float(t):g} 小时，"
            "共行驶了多少千米？"
        )
        return {"v": v, "t": str(t), "mode": "fwd"}, q, v * t
    if d == 4:  # 反向：求速度（构造保证整除）
        v = rng.randint(18, 95)
        t = rng.randint(2, 6)
        q = f"一辆汽车行驶了 {v * t} 千米用了 {t} 小时，它的速度是每小时多少千米？"
        return {"v": v, "t": t, "mode": "rev-v"}, q, Fraction(v)
    # d=5 半小时 → 答案可能带 .5
    v = rng.randint(40, 130)
    t = Fraction(rng.choice([1, 3, 5, 7, 9, 11]), 2)
    q = (
        f"一辆汽车以每小时 {v} 千米的速度行驶了 {float(t):g} 小时，"
        "共行驶了多少千米？"
    )
    return {"v": v, "t": str(t), "mode": "fwd"}, q, v * t


def t_rect_area(rng: random.Random, d: int):
    if d <= 2:
        lo, hi = {1: (2, 12), 2: (3, 20)}[d]
        a, b = rng.randint(lo, hi), rng.randint(lo, hi)
        if rng.random() < 0.5 or d == 1:
            q = f"一个长方形长 {a} 厘米，宽 {b} 厘米，它的面积是多少平方厘米？"
            return {"a": a, "b": b, "ask": "area"}, q, Fraction(a * b)
        q = f"一个长方形长 {a} 厘米，宽 {b} 厘米，它的周长是多少厘米？"
        return {"a": a, "b": b, "ask": "perim"}, q, Fraction(2 * (a + b))
    if d == 3 or d == 4:
        lo, hi = (5, 35) if d == 3 else (8, 60)
        a, b = rng.randint(lo, hi), rng.randint(lo, hi)
        x = rng.randint(2, 15)
        q = (
            f"一个长方形长 {a} 厘米，宽 {b} 厘米。若把长增加 {x} 厘米，"
            "新长方形的面积是多少平方厘米？"
        )
        return {"a": a, "b": b, "x": x, "ask": "grow", "grow_dim": "a"}, q, Fraction((a + x) * b)
    lo, hi = 11, 99
    a, b = rng.randint(lo, hi), rng.randint(lo, hi)
    x = rng.randint(10, 40)
    q = (
        f"一个长方形长 {a} 厘米，宽 {b} 厘米。若把宽增加 {x} 厘米，"
        "新长方形的面积是多少平方厘米？"
    )
    return {"a": a, "b": b, "x": x, "ask": "grow", "grow_dim": "b"}, q, Fraction(a * (b + x))


def t_discount(rng: random.Random, d: int):
    if d <= 3:
        P = {1: rng.randrange(1, 10) * 100, 2: rng.randrange(5, 51) * 20,
             3: rng.randrange(10, 201) * 10}[d]
        z = rng.choice([5, 6, 7, 8, 9])
        q = f"一件商品原价 {P} 元，现在打{_zhe_str(Fraction(z))}出售，现价是多少元？"
        return {"P": P, "z": z, "mode": "fwd"}, q, Fraction(P * z, 10)
    if d == 4:
        P = rng.randrange(25, 501)
        z = rng.choice([5, 6, 7, 8, 9])
        q = f"一件商品原价 {P} 元，现在打{_zhe_str(Fraction(z))}出售，现价是多少元？"
        return {"P": P, "z": z, "mode": "fwd"}, q, Fraction(P * z, 10)
    # d=5 半折/整折 + 反向（P 为 20 的倍数 → 任意 x.5 折与整折的现价都是整数）
    z = rng.choice([Fraction(n, 2) for n in range(9, 20)])  # 4.5 ~ 9.5
    P = rng.randrange(1, 101) * 20  # 20 ~ 2000
    X = P * z / 10
    assert X.denominator == 1
    q = f"一件商品打{_zhe_str(z)}后的售价是 {X.numerator} 元，它的原价是多少元？"
    return {"P": P, "z": str(z), "mode": "rev"}, q, Fraction(P)


def t_chicken_rabbit(rng: random.Random, d: int):
    lo, hi = {1: (2, 9), 2: (3, 15), 3: (5, 25), 4: (8, 40), 5: (12, 60)}[d]
    c = rng.randint(lo, hi)
    r = rng.randint(lo, hi)
    h, legs = c + r, 2 * c + 4 * r
    q = f"笼子里有鸡和兔共 {h} 只，一共有 {legs} 条腿。兔有多少只？"
    return {"c": c, "r": r}, q, Fraction(r)


TEMPLATES: dict[str, object] = {
    "mult_chain": t_mult_chain,
    "frac_addsub": t_frac_addsub,
    "lcm": t_lcm,
    "gcd": t_gcd,
    "percent": t_percent,
    "average": t_average,
    "travel": t_travel,
    "rect_area": t_rect_area,
    "discount": t_discount,
    "chicken_rabbit": t_chicken_rabbit,
}
TEMPLATE_NAMES = list(TEMPLATES)


# ---------------------------------------------------------------------------
# 采样入口（唯一出题通道）
# ---------------------------------------------------------------------------

def sample(
    n: int,
    difficulty: int,
    role: str,
    arm: str = "GLOBAL",
    extra_seed: str = "",
    template_subset: list[str] | None = None,
) -> list[dict]:
    """采 n 道难度 difficulty 的题，指派角色并登记账本。

    返回记录字段：id / template / difficulty / question / answer(规范串) /
    answer_frac(Fraction 精确串) / params / role / arm。
    """
    if role not in config.ROLES:
        raise ValueError(f"未知角色: {role}")
    if not (1 <= int(difficulty) <= 5):
        raise ValueError("difficulty 必须在 1-5")

    state = config.load_state()
    master = state.get("master_seed") or config.default_state()["master_seed"]
    ledger = Ledger(arm)

    names = template_subset or TEMPLATE_NAMES
    # 轮转 + 起点随机 → 模板配比均衡
    order = list(names)
    start = random.Random(f"{master}:{role}:{arm}:{len(ledger.entries)}:{extra_seed}").randrange(len(order))
    order = order[start:] + order[:start]

    rng = random.Random(f"{master}:{role}:{arm}:{len(ledger.entries)}:{extra_seed}:gen")
    out: list[dict] = []
    batch_ids: set[str] = set()
    attempts = 0
    max_attempts = n * 60 + 400
    # 用「尝试数」而非「成功数」轮转：模板参数空间耗尽时不会被活锁；
    # 连续失败过多的模板直接移出轮转。
    fail_streak: dict[str, int] = {t: 0 for t in order}
    active = list(order)
    while len(out) < n and attempts < max_attempts and active:
        tname = active[attempts % len(active)]
        attempts += 1
        params, q, ans = TEMPLATES[tname](rng, int(difficulty))  # type: ignore[misc]
        pid = make_id(tname, params)
        if pid in ledger or pid in batch_ids:
            fail_streak[tname] += 1
            if fail_streak[tname] > 300:
                active.remove(tname)
            continue
        fail_streak[tname] = 0
        ledger.assign(pid, role, {"template": tname, "difficulty": int(difficulty)})
        batch_ids.add(pid)
        out.append(
            {
                "id": pid,
                "template": tname,
                "difficulty": int(difficulty),
                "question": q,
                "answer": frac_str(ans),
                "answer_frac": str(Fraction(ans)),
                "params": params,
                "role": role,
                "arm": arm,
            }
        )
    if len(out) < n:
        raise RuntimeError(
            f"采样不足: 需要 {n}，只得到 {len(out)}（难度 {difficulty} 的参数空间过窄？）"
        )
    ledger.save()
    return out


def write_jsonl(path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
