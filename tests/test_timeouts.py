"""超时基础设施测试。"""
from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rsi.timeouts import (
    Deadline,
    GenerationTimeout,
    TaskDeadlineExceeded,
    remaining_or_raise,
    wall_timeout,
)


class TestTimeouts(unittest.TestCase):
    def test_deadline(self):
        d = Deadline(10.0)
        self.assertGreater(d.remaining, 9.0)
        self.assertFalse(d.expired)

    def test_remaining_or_raise(self):
        d = Deadline(0.0)
        with self.assertRaises(TaskDeadlineExceeded):
            remaining_or_raise(d)

    def test_wall_timeout_raises(self):
        with self.assertRaises(GenerationTimeout):
            with wall_timeout(0.02, scope="call"):
                time.sleep(0.2)

    def test_wall_timeout_does_not_raise_for_fast_block(self):
        with wall_timeout(1.0, scope="call"):
            time.sleep(0.001)


if __name__ == "__main__":
    unittest.main(verbosity=2)
