"""grade 单元测试：boxed 抽取、归一化、容差、lint、汇总。"""
from __future__ import annotations

import sys
import unittest
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rsi import grade


class TestExtractBoxed(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(grade.extract_boxed(r"推理...\boxed{42}"), ("42", 1))

    def test_nested_braces(self):
        self.assertEqual(grade.extract_boxed(r"\boxed{\frac{3}{4}}"), (r"\frac{3}{4}", 1))

    def test_multiple_takes_last(self):
        content, n = grade.extract_boxed(r"先 \boxed{1} 再 \boxed{2448}")
        self.assertEqual((content, n), ("2448", 2))

    def test_fbox(self):
        self.assertEqual(grade.extract_boxed(r"\fbox{99}"), ("99", 1))

    def test_no_boxed(self):
        content, n = grade.extract_boxed("答案是 42")
        self.assertIsNone(content)
        self.assertEqual(n, 0)

    def test_unclosed(self):
        content, n = grade.extract_boxed(r"\boxed{42")
        self.assertEqual(n, 1)  # 退化路径仍计数


class TestNormalize(unittest.TestCase):
    def test_int(self):
        self.assertEqual(grade.normalize_answer("42"), Fraction(42))

    def test_negative(self):
        self.assertEqual(grade.normalize_answer("-7"), Fraction(-7))
        self.assertEqual(grade.normalize_answer("−7"), Fraction(-7))  # Unicode 减号

    def test_decimal(self):
        self.assertEqual(grade.normalize_answer("0.75"), Fraction(3, 4))

    def test_fraction(self):
        self.assertEqual(grade.normalize_answer("3/4"), Fraction(3, 4))

    def test_improper_fraction_not_mixed(self):
        """回归：假分数绝不能被拆成带分数（17/15 ≠ 1+7/15）。"""
        self.assertEqual(grade.normalize_answer("17/15"), Fraction(17, 15))
        self.assertEqual(grade.normalize_answer("11/8"), Fraction(11, 8))
        self.assertEqual(grade.normalize_answer("47/3"), Fraction(47, 3))
        self.assertEqual(grade.normalize_answer(r"\frac{17}{15}"), Fraction(17, 15))
        self.assertEqual(grade.normalize_answer(r"\dfrac{47}{3}"), Fraction(47, 3))

    def test_latex_frac(self):
        self.assertEqual(grade.normalize_answer(r"\dfrac{3}{4}"), Fraction(3, 4))
        self.assertEqual(grade.normalize_answer(r"\frac{3}{4}"), Fraction(3, 4))

    def test_mixed_number(self):
        self.assertEqual(grade.normalize_answer("1又1/2"), Fraction(3, 2))
        self.assertEqual(grade.normalize_answer("1 1/2"), Fraction(3, 2))

    def test_percent(self):
        self.assertEqual(grade.normalize_answer("75%"), Fraction(3, 4))

    def test_wan(self):
        self.assertEqual(grade.normalize_answer("1.2万"), Fraction(12000))

    def test_unit_suffix(self):
        self.assertEqual(grade.normalize_answer("42千米"), Fraction(42))
        self.assertEqual(grade.normalize_answer("3.5元"), Fraction(7, 2))
        self.assertEqual(grade.normalize_answer("56只"), Fraction(56))

    def test_text_wrapper(self):
        self.assertEqual(grade.normalize_answer(r"\text{42}"), Fraction(42))

    def test_answer_prefix(self):
        self.assertEqual(grade.normalize_answer("答案：42"), Fraction(42))
        self.assertEqual(grade.normalize_answer("=42"), Fraction(42))
        self.assertEqual(grade.normalize_answer("是42"), Fraction(42))

    def test_chinese_punct(self):
        self.assertEqual(grade.normalize_answer("42。"), Fraction(42))
        self.assertEqual(grade.normalize_answer("1,200"), Fraction(1200))

    def test_garbage_returns_none(self):
        self.assertIsNone(grade.normalize_answer("无法计算"))
        self.assertIsNone(grade.normalize_answer(""))
        self.assertIsNone(grade.normalize_answer(None))


class TestGradeOne(unittest.TestCase):
    def _resp(self, ans, prefix="我们逐步计算：\n首先处理乘法。\n最终得到\n"):
        return f"{prefix}\\boxed{{{ans}}}"

    def test_correct_int(self):
        r = grade.grade_one("2448", self._resp("2448"))
        self.assertTrue(r["correct"])
        self.assertFalse(r["flags"]["parse_fail"])

    def test_wrong(self):
        r = grade.grade_one("2448", self._resp("2447"))
        self.assertFalse(r["correct"])

    def test_fraction_equivalent_forms(self):
        r = grade.grade_one("3/4", self._resp("0.75"))
        self.assertTrue(r["correct"])
        r = grade.grade_one("3/4", self._resp(r"\dfrac{3}{4}"))
        self.assertTrue(r["correct"])

    def test_tolerance_repeating_decimal(self):
        # 1/3 ≈ 0.3333333（7位）在 1e-6 容差内
        r = grade.grade_one("1/3", self._resp("0.3333333"))
        self.assertTrue(r["correct"])

    def test_no_boxed(self):
        r = grade.grade_one("42", "答案是42")
        self.assertFalse(r["correct"])
        self.assertTrue(r["flags"]["no_boxed"])
        self.assertTrue(r["flags"]["parse_fail"])

    def test_multi_boxed_flag(self):
        r = grade.grade_one("42", "步骤一 \\boxed{1} 最后 \\boxed{42}")
        self.assertTrue(r["correct"])
        self.assertTrue(r["flags"]["multi_boxed"])

    def test_unit_answer(self):
        r = grade.grade_one("160", self._resp("160千米"))
        self.assertTrue(r["correct"])


class TestLint(unittest.TestCase):
    def test_good_solution(self):
        resp = "第一步计算括号内：3+4=7。\n第二步乘法：7×8=56。\n所以答案为\n\\boxed{56}"
        ok, reasons = grade.lint_one(resp, {"correct": True})
        self.assertTrue(ok, reasons)

    def test_short_reasoning_rejected(self):
        ok, reasons = grade.lint_one("\\boxed{56}", {"correct": True})
        self.assertFalse(ok)
        self.assertTrue(reasons["too_short_reasoning"])

    def test_degenerate_rejected(self):
        resp = "\n".join(["= = = = ="] * 8) + "\n\\boxed{56}"
        ok, reasons = grade.lint_one(resp, {"correct": True})
        self.assertFalse(ok)
        self.assertTrue(reasons["degenerate"])

    def test_incorrect_rejected(self):
        resp = "推理过程若干字符若干字符若干字符若干字符若干字符若干字符若干字符\n\\boxed{57}"
        ok, reasons = grade.lint_one(resp, {"correct": False})
        self.assertFalse(ok)
        self.assertTrue(reasons["not_correct"])


class TestSummarize(unittest.TestCase):
    def test_summary(self):
        rs = [
            grade.grade_one("1", "推理推理推理推理推理推理推理推理\n\\boxed{1}"),
            grade.grade_one("2", "推理推理推理推理推理推理推理推理\n\\boxed{3}"),
            grade.grade_one("3", "没有box"),
        ]
        s = grade.summarize(rs)
        self.assertEqual(s["n"], 3)
        self.assertAlmostEqual(s["accuracy"], round(1 / 3, 4))
        self.assertAlmostEqual(s["parse_fail_rate"], round(1 / 3, 4))


if __name__ == "__main__":
    unittest.main(verbosity=2)
