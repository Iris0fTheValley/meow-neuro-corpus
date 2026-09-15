from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_sft_v2_3_train_views as context_views
import run_sft_v2_3_context_sufficiency as context_closure


class ContextSufficiencyTests(unittest.TestCase):
    def pool_row(self, sample_id: str = "s1") -> dict:
        return {
            "sample_id": sample_id,
            "artifact_schema_version": "2.0.0",
            "pipeline_version": "sft-interaction-closure-v2.3.1-2026-09-13",
            "semantic_closure": {"state": "VERIFIED"},
            "context_turn_ids": ["c1"],
            "target_turn_ids": ["n1"],
            "transcript_qa": {
                "turns": [
                    {"turn_id": "c1", "role": "context", "speaker": "u", "text": "Whatever."},
                    {"turn_id": "n1", "role": "target", "speaker": "n", "text": "You should add more cats."},
                ]
            },
            "verified_pool_status": "VERIFIED_HARD_DEDUPED",
            "sampling_eligibility": "ELIGIBLE",
        }

    def test_request_exposes_only_final_selected_slice(self):
        request = context_closure.build_sufficiency_request(self.pool_row())
        self.assertIsNotNone(request)
        judge_input = request["judge_input"]
        self.assertEqual(judge_input["selected_context"][0]["text"], "Whatever.")
        self.assertEqual(judge_input["selected_target"][0]["text"], "You should add more cats.")
        forbidden = {key for key in judge_input if "diagnostic" in key.lower() or "timeline" in key.lower() or "source" in key.lower()}
        self.assertFalse(forbidden)

    def test_materialize_routes_incomplete_without_changing_semantic_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pool_path = root / "pool.jsonl"
            requests_path = root / "requests.jsonl"
            results_path = root / "results.jsonl"
            pool_path.write_text(json.dumps(self.pool_row()) + "\n", encoding="utf-8")
            context_closure.prepare(Namespace(pool=str(pool_path), requests=str(requests_path), prepare_report=str(root / "prepare.json")))
            request = json.loads(requests_path.read_text(encoding="utf-8"))
            result = {
                "schema_version": context_closure.semantic.SCHEMA_VERSION,
                "pipeline_version": context_closure.semantic.PIPELINE_VERSION,
                "context_sufficiency_version": context_closure.CONTEXT_SUFFICIENCY_VERSION,
                "sample_id": "s1",
                "request_sha256": request["request_sha256"],
                "source_materialization_sha256": request["source_materialization_sha256"],
                "judge_model": "fixture-model",
                "judge_prompt_version": context_closure.PROMPT_VERSION,
                "validated": {"valid": True, "state": "CONTEXT_INCOMPLETE", "confidence": 0.93, "reason": "needs omitted antecedent"},
            }
            results_path.write_text(json.dumps(result) + "\n", encoding="utf-8")
            args = Namespace(
                requests=str(requests_path), results=str(results_path), pool=str(pool_path), model="fixture-model",
                decisions_output=str(root / "decisions.jsonl"), closed_pool=str(root / "closed.jsonl"),
                interaction_pool=str(root / "interaction.jsonl"), reconstruction_candidates=str(root / "reconstruction.jsonl"),
                unsupported_quarantine=str(root / "unsupported.jsonl"), materialize_report=str(root / "materialize.json"),
            )
            report = context_closure.materialize(args)
            reconstructed = json.loads((root / "reconstruction.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(report["reconstruction_candidates"], 1)
            self.assertEqual(report["direct_interaction_pool"], 0)
            self.assertEqual(reconstructed["semantic_closure"]["state"], "VERIFIED")
            self.assertEqual(reconstructed["sampling_eligibility"], "CONTEXT_RECONSTRUCTION_ONLY")

    def test_train_view_gate_rejects_non_self_contained_rows(self):
        row = self.pool_row()
        row["context_sufficiency"] = {
            "context_sufficiency_version": context_closure.CONTEXT_SUFFICIENCY_VERSION,
            "state": "CONTEXT_INCOMPLETE",
            "selected_context_turn_ids": ["c1"],
            "selected_target_turn_ids": ["n1"],
        }
        with tempfile.TemporaryDirectory() as directory:
            pool = Path(directory) / "interaction_pool.jsonl"
            pool.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                context_views.assert_context_closed(pool)

    def test_context_incomplete_gets_bounded_trajectory_candidate_only(self):
        row = self.pool_row()
        timeline = {
            "_turns": [
                {"turn_id": "prior", "identity": "OTHER", "text": "Earlier real question.", "_start": 0.0, "_end": 1.0},
                {"turn_id": "c1", "identity": "OTHER", "text": "Whatever.", "_start": 1.5, "_end": 2.0},
                {"turn_id": "n1", "identity": "NEURO_FAMILY_HIGH", "text": "You should add more cats.", "_start": 2.5, "_end": 3.5},
            ]
        }
        candidate = context_closure.build_trajectory_candidate(row, timeline)
        self.assertEqual(candidate["recovery_status"], "BOUNDED_CONTEXT_RECOVERED")
        self.assertEqual(candidate["expanded_context_turn_ids"], ["prior", "c1"])
        self.assertEqual(candidate["training_eligibility"], "CONTEXT_RECONSTRUCTION_ONLY")
        self.assertTrue(candidate["semantic_truth_preserved"])


if __name__ == "__main__":
    unittest.main()

