from __future__ import annotations

"""Run bounded regressions and publish a machine-readable architecture gate."""

import io
import json
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import sft_interaction_semantic_closure as closure  # noqa: E402


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> None:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.discover(str(ROOT / "tests"), pattern="test_*.py"))
    suite.addTests(loader.loadTestsFromName("scripts.test_sft_semantic_closure_v2"))
    suite.addTests(loader.loadTestsFromName("scripts.test_sft_semantic_verified_v2_1"))
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    legacy_quality = load_json(ROOT / "reports" / "sft_v2_2" / "quality_report_v2_2.json")
    legacy_validation = load_json(ROOT / "reports" / "sft_v2_2" / "validation_report_v2_2.json")
    report = {
        "schema_version": closure.SCHEMA_VERSION,
        "pipeline_version": closure.PIPELINE_VERSION,
        "artifact_status": "ARCHITECTURE_REGRESSION_VALIDATED" if result.wasSuccessful() else "ARCHITECTURE_REGRESSION_FAILED",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "unit, synthetic regression, and existing lightweight v2.2 report evidence; no production model or corpus rerun",
        "tests": {"run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors), "pass": result.wasSuccessful()},
        "existing_artifact_baseline": {
            "status": "HISTORICAL_V2_2_NOT_REINTERPRETED_AS_V2_3",
            "structural_candidates": legacy_quality.get("v2_2_structural_candidates"),
            "target_reuse": legacy_validation.get("target_turn_reuse_count"),
            "prefix_ladder": legacy_validation.get("same_anchor_prefix_ladder_count"),
            "recording_family_leakage": legacy_validation.get("recording_family_leakage"),
            "primary_accepts_before_legacy_heuristic_guard": legacy_quality.get("model_semantic_accepted_before_independent_guard"),
            "legacy_heuristic_hard_rejects": legacy_quality.get("semantic_guard_rejected"),
        },
        "architecture_gates": {
            "IDENTITY_CLOSURE": "UNCHANGED_COMPATIBLE",
            "STRUCTURAL_INVARIANTS": "PASS" if result.wasSuccessful() else "FAIL",
            "SEMANTIC_STATE_MACHINE": "PASS" if result.wasSuccessful() else "FAIL",
            "STALE_JUDGE_RESULT_INVALIDATION": "PASS" if result.wasSuccessful() else "FAIL",
            "BOUNDED_CONTEXT_AND_EPISODE_SELECTION": "PASS" if result.wasSuccessful() else "FAIL",
            "SEMANTIC_PROVENANCE_REMATERIALIZATION": "PASS" if result.wasSuccessful() else "FAIL",
            "HEURISTIC_AUTHORITY": "ROUTING_EVIDENCE_ONLY",
            "TRANSCRIPT_REWRITE_PROHIBITION": "PASS" if result.wasSuccessful() else "FAIL",
            "RECORDING_BRIDGE_AUDIT": "PASS" if result.wasSuccessful() else "FAIL",
            "RECORDING_BRIDGE_SPLIT_ISOLATION": "PASS" if result.wasSuccessful() else "FAIL",
            "HARD_DEDUP_SCOPE": "RECORDING_PROVENANCE_AWARE",
            "INDEPENDENT_BEHAVIOR_REPETITION": "PRESERVED",
            "VERIFIED_POOL_ARCHITECTURE": "PASS" if result.wasSuccessful() else "FAIL",
            "TRAIN_SAMPLING_SEPARATION": "PASS" if result.wasSuccessful() else "FAIL",
            "SEALED_SPLIT_LINEAGE": "PASS" if result.wasSuccessful() else "FAIL",
            "SAMPLING_POLICY_SPLIT_AUTHORITY": "PASS" if result.wasSuccessful() else "FAIL",
            "TARGET_REUSE_REGRESSION_FIXTURES": 0,
            "PREFIX_LADDER_REGRESSION_FIXTURES": 0,
        },
        "production_gates": {
            "SEMANTIC_PRIMARY_COVERAGE": "NOT_YET_VALIDATED",
            "SEMANTIC_STRICT_OR_EQUIVALENT_VERIFICATION": "NOT_YET_VALIDATED",
            "UNRESOLVED_HIGH_RISK_SAMPLES": "NOT_YET_COUNTED",
            "JUDGE_CONFLICTS": "NOT_YET_COUNTED",
            "VERIFIED_POOL": "NOT_YET_CONSTRUCTED",
            "CROSS_SPLIT_RECORDING_LEAKAGE": "NOT_YET_VALIDATED",
            "READY": False,
        },
    }
    output = ROOT / "reports" / "sft_v2_3_architecture_validation.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not result.wasSuccessful():
        print(stream.getvalue(), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
