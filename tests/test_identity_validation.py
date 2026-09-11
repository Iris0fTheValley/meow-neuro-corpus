from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from derive_candidate_counts import derive_candidate_counts
from identity_validation_utils import assert_training_gate, bootstrap_mean_interval, wilson_interval


class IdentityValidationTests(unittest.TestCase):
    def test_wilson_interval_is_bounded_for_zero_and_one(self) -> None:
        zero = wilson_interval(0, 10)
        one = wilson_interval(10, 10)
        self.assertEqual(zero[0], 0.0)
        self.assertEqual(one[1], 1.0)
        self.assertLessEqual(zero[0], zero[1])
        self.assertLessEqual(one[0], one[1])

    def test_bootstrap_is_deterministic(self) -> None:
        self.assertEqual(bootstrap_mean_interval([0.0, 1.0, 1.0], seed=7), bootstrap_mean_interval([0.0, 1.0, 1.0], seed=7))

    def test_candidate_counts_are_row_level(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidates.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in [
                {"candidate_grade": "S", "training_candidate": False},
                {"candidate_grade": "A", "training_candidate": False},
                {"candidate_grade": "Q", "training_candidate": False},
            ]) + "\n", encoding="utf-8")
            counts = derive_candidate_counts(path)
            self.assertEqual(counts["row_count"], 3)
            self.assertEqual(counts["s_a_review_candidate_count"], 2)
            self.assertEqual(counts["training_candidate_count"], 0)

    def test_fail_closed_gate_rejects_true_candidate(self) -> None:
        with self.assertRaises(AssertionError):
            assert_training_gate([{"training_candidate": True}], prerequisites_pass=False)
        assert_training_gate([{"training_candidate": False}], prerequisites_pass=False)


if __name__ == "__main__":
    unittest.main()
