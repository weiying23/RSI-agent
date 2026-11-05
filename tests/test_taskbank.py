"""固定题库测试：生成、哈希校验、角色隔离、读取。"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rsi import taskbank


class TestTaskBank(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="rsi_taskbank_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_generate_verify_and_load(self):
        manifest = taskbank.generate_bank(
            out_dir=self.tmp,
            gens=2,
            train_per_gen=12,
            probe_per_gen=7,
            promo_per_gen=6,
            difficulty=5,
            seed=123,
            force=True,
        )
        self.assertEqual(len(manifest["splits"]), 6)
        self.assertEqual(manifest["counts"]["total_generated"], 50)

        report = taskbank.verify_bank(self.tmp)
        self.assertTrue(report["ok"])
        self.assertEqual(report["n_generated"], 50)

        train = taskbank.load_split("train", 1, root=self.tmp)
        probe = taskbank.load_split("probe", 1, root=self.tmp)
        promo = taskbank.load_split("promo", 1, root=self.tmp)
        self.assertEqual(len(train), 12)
        self.assertEqual(len(probe), 7)
        self.assertEqual(len(promo), 6)

        all_ids = {r["id"] for r in train + probe + promo}
        self.assertEqual(len(all_ids), 25)

        # 同一分片重复读取必须完全一致
        self.assertEqual(train, taskbank.load_split("train", 1, root=self.tmp))

    def test_tamper_detected(self):
        taskbank.generate_bank(
            out_dir=self.tmp,
            gens=1,
            train_per_gen=8,
            probe_per_gen=5,
            promo_per_gen=5,
            difficulty=5,
            seed=7,
            force=True,
        )
        path = self.tmp / "train" / "train_gen_001.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        obj = json.loads(lines[0])
        obj["question"] += "（被篡改）"
        lines[0] = json.dumps(obj, ensure_ascii=False)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        with self.assertRaises(RuntimeError):
            taskbank.verify_bank(self.tmp)

    def test_expected_n_rejected(self):
        taskbank.generate_bank(
            out_dir=self.tmp,
            gens=1,
            train_per_gen=3,
            probe_per_gen=2,
            promo_per_gen=2,
            difficulty=5,
            seed=9,
            force=True,
        )
        with self.assertRaises(RuntimeError):
            taskbank.load_split("probe", 1, root=self.tmp, expected_n=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
