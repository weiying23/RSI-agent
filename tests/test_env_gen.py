"""env_gen 单元测试：答案符号自洽、确定性 ID、账本排除。"""
from __future__ import annotations

import json
import random
import shutil
import sys
import unittest
from fractions import Fraction
from math import gcd, lcm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rsi import config
from rsi import env_gen


def verify_answer(rec: dict) -> bool:
    """独立于生成代码的答案复算（不同实现路径，防同源错误）。"""
    t, p, ans = rec["template"], rec["params"], Fraction(rec["answer_frac"])
    if t == "mult_chain":
        prod = 1
        for x in p["ops"]:
            prod *= x
        return prod == ans and ans > 0
    if t == "frac_addsub":
        frs = [Fraction(s) for s in p["fractions"]]
        val = frs[0]
        for i in range(1, len(frs)):
            val = val + frs[i] if p["ops"][i] == "+" else val - frs[i]
        return val == ans and ans > 0 and all(f.denominator > 1 for f in frs)
    if t == "lcm":
        v = lcm(*p["nums"])
        return v == ans and v > max(p["nums"]) and all(x > 1 for x in p["nums"])
    if t == "gcd":
        v = gcd(*p["nums"])
        return v == ans and v >= 2
    if t == "percent":
        if p["mode"] == "fwd":
            return Fraction(p["N"] * p["p"], 100) == ans
        return Fraction(p["N"]) == ans
    if t == "average":
        return Fraction(sum(p["nums"]), p["k"]) == ans
    if t == "travel":
        if p["mode"] == "fwd":
            return p["v"] * Fraction(p["t"]) == ans
        return Fraction(p["v"]) == ans
    if t == "rect_area":
        a, b = p["a"], p["b"]
        if p["ask"] == "area":
            return a * b == ans
        if p["ask"] == "perim":
            return 2 * (a + b) == ans
        if p["ask"] == "grow":
            if p["grow_dim"] == "a":
                return (a + p["x"]) * b == ans
            return a * (b + p["x"]) == ans
        return False
    if t == "discount":
        z = Fraction(p["z"]) if "/" in str(p["z"]) else Fraction(int(p["z"]))
        if p["mode"] == "fwd":
            return Fraction(p["P"]) * z / 10 == ans
        return Fraction(p["P"]) == ans
    if t == "chicken_rabbit":
        c, r = p["c"], p["r"]
        # 从题目文本反解 h 与 legs
        import re
        m = re.search(r"共 (\d+) 只，一共有 (\d+) 条腿", rec["question"])
        h, legs = int(m.group(1)), int(m.group(2))
        return h == c + r and legs == 2 * c + 4 * r and ans == r and c >= 1 and r >= 1
    return False


class TestTemplates(unittest.TestCase):
    def test_all_difficulties_symbolic_consistency(self):
        """每个模板 × 每个难度，答案必须符号自洽。"""
        rng = random.Random(7)
        for name, tfun in env_gen.TEMPLATES.items():
            for d in range(1, 6):
                for _ in range(30):
                    params, q, ans = tfun(rng, d)
                    rec = {
                        "template": name,
                        "params": params,
                        "question": q,
                        "answer_frac": str(ans),
                    }
                    self.assertTrue(
                        verify_answer(rec), f"{name} d={d} 自洽失败: {params} ans={ans}"
                    )

    def test_id_determinism(self):
        p1 = {"a": 1, "b": 2}
        p2 = {"b": 2, "a": 1}
        self.assertEqual(env_gen.make_id("t", p1), env_gen.make_id("t", p2))
        self.assertNotEqual(env_gen.make_id("t", p1), env_gen.make_id("t", {"a": 1, "b": 3}))

    def test_question_contains_params(self):
        rng = random.Random(3)
        for name, tfun in env_gen.TEMPLATES.items():
            params, q, ans = tfun(rng, 3)
            self.assertIsInstance(q, str)
            self.assertGreater(len(q), 8)
            self.assertIsInstance(ans, Fraction)


class TestSampling(unittest.TestCase):
    def setUp(self):
        # 每个测试用干净的临时 state 目录
        self.orig_root = config.ROOT
        import tempfile
        self.tmp = Path(tempfile.mkdtemp(prefix="rsi_test_"))
        config.STATE_DIR = self.tmp / "state"
        config.LEDGER_DIR = config.STATE_DIR / "ledger"
        config.GLOBAL_LEDGER = config.LEDGER_DIR / "_global.json"
        config.STATE_CONFIG = config.STATE_DIR / "config.json"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_sample_unique_and_ledgered(self):
        recs = env_gen.sample(40, 3, role="calib", arm="GLOBAL", extra_seed="t1")
        self.assertEqual(len(recs), 40)
        ids = [r["id"] for r in recs]
        self.assertEqual(len(set(ids)), 40)
        # 账本已登记
        ledger_data = json.loads(config.GLOBAL_LEDGER.read_text())
        for i in ids:
            self.assertIn(i, ledger_data)
            self.assertEqual(ledger_data[i]["role"], "calib")

    def test_exclusion_across_roles(self):
        """同一实例 ID 永不复用：第二次采样与第一次不相交。"""
        a = env_gen.sample(30, 3, role="calib", arm="GLOBAL", extra_seed="t2")
        b = env_gen.sample(30, 3, role="pool", arm="arm_WH", extra_seed="t3")
        ida, idb = {r["id"] for r in a}, {r["id"] for r in b}
        self.assertFalse(ida & idb, "跨角色实例泄漏！")

    def test_per_arm_isolation(self):
        """三臂是平行世界：arm_A 的 pool 与 arm_B 的 pool 互不相交。"""
        a = env_gen.sample(30, 3, role="pool", arm="arm_A", extra_seed="t4")
        b = env_gen.sample(30, 3, role="pool", arm="arm_B", extra_seed="t5")
        self.assertFalse({r["id"] for r in a} & {r["id"] for r in b})

    def test_sample_answers_valid(self):
        for d in (2, 4):
            for rec in env_gen.sample(20, d, role="calib", arm="GLOBAL", extra_seed=f"v{d}"):
                self.assertTrue(verify_answer(rec), f"d={d} {rec['id']} 失败")

    def test_invalid_role_rejected(self):
        with self.assertRaises(ValueError):
            env_gen.sample(5, 3, role="secret-eval")  # 未注册角色


if __name__ == "__main__":
    unittest.main(verbosity=2)
