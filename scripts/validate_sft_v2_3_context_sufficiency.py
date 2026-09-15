from __future__ import annotations

"""Independent validator for the v2.3 context-sufficiency closure."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "datasets").exists() and (Path(__file__).resolve().parents[3] / "datasets").exists():
    ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = ROOT / "datasets" / "meow_v02_sft_v2_2_semantic_verified"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(1, str(SCRIPT_DIR))

import sft_interaction_semantic_closure as semantic  # noqa: E402
from run_sft_v2_3_context_sufficiency import ALLOWED_STATES, CONTEXT_SUFFICIENCY_VERSION, PROMPT_VERSION  # noqa: E402


def rows(path: Path) -> list[dict[str, Any]]:
    result = []
    if not path.exists():
        return result
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except Exception:
                continue
            if isinstance(value, dict):
                result.append(value)
    return result


def latest(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("sample_id")): row for row in rows(path)}


def validate_context(args: argparse.Namespace) -> dict[str, Any]:
    decisions = latest(Path(args.decisions))
    source_pool = latest(Path(args.source_pool))
    interaction_pool = latest(Path(args.interaction_pool))
    reconstruction = latest(Path(args.reconstruction_candidates))
    unsupported = latest(Path(args.unsupported_quarantine))
    requests = latest(Path(args.requests))

    verified_ids = set(source_pool)
    decision_ids = set(decisions)
    errors: list[dict[str, Any]] = []

    if decision_ids != verified_ids:
        errors.append({"reason": "DECISION_COVERAGE_MISMATCH", "missing": sorted(verified_ids - decision_ids)[:50], "extra": sorted(decision_ids - verified_ids)[:50]})

    state_counts = Counter()
    for sample_id, decision in decisions.items():
        state = str(decision.get("state") or "")
        state_counts[state] += 1
        if decision.get("schema_version") != semantic.SCHEMA_VERSION or decision.get("pipeline_version") != semantic.PIPELINE_VERSION:
            errors.append({"sample_id": sample_id, "reason": "VERSION_MISMATCH"})
        if decision.get("context_sufficiency_version") != CONTEXT_SUFFICIENCY_VERSION:
            errors.append({"sample_id": sample_id, "reason": "SUFFICIENCY_VERSION_MISMATCH"})
        if decision.get("judge_prompt_version") != PROMPT_VERSION:
            errors.append({"sample_id": sample_id, "reason": "PROMPT_VERSION_MISMATCH"})
        if state not in ALLOWED_STATES:
            errors.append({"sample_id": sample_id, "reason": "INVALID_STATE"})
        request = requests.get(sample_id) or {}
        judge_input = request.get("judge_input") or {}
        evidence_keys = {key for key in judge_input if "diagnostic" in key.lower() or "timeline" in key.lower() or "source" in key.lower() or "alternate" in key.lower()}
        if evidence_keys:
            errors.append({"sample_id": sample_id, "reason": "FORBIDDEN_EVIDENCE_EXPOSED", "keys": sorted(evidence_keys)})
        allowed_content_keys = {"selected_context", "selected_target"}
        contentish = {key for key in judge_input if key.endswith("_context") or key.endswith("_target") or "turns" in key.lower()}
        if not contentish.issubset(allowed_content_keys):
            errors.append({"sample_id": sample_id, "reason": "NON_SELECTED_CONTENT_EXPOSED", "keys": sorted(contentish - allowed_content_keys)})
        source_row = source_pool.get(sample_id)
        if source_row:
            if [str(value) for value in source_row.get("context_turn_ids") or []] != [str(value) for value in decision.get("selected_context_turn_ids") or []]:
                errors.append({"sample_id": sample_id, "reason": "SOURCE_CONTEXT_MISMATCH"})
            if [str(value) for value in source_row.get("target_turn_ids") or []] != [str(value) for value in decision.get("selected_target_turn_ids") or []]:
                errors.append({"sample_id": sample_id, "reason": "SOURCE_TARGET_MISMATCH"})

    self_contained_ids = {sample_id for sample_id, row in decisions.items() if row.get("state") == "SELF_CONTAINED"}
    incomplete_ids = {sample_id for sample_id, row in decisions.items() if row.get("state") == "CONTEXT_INCOMPLETE"}
    unsupported_ids = {sample_id for sample_id, row in decisions.items() if row.get("state") == "UNSUPPORTED_RELATION"}

    if set(interaction_pool) != self_contained_ids & set(source_pool):
        errors.append({"reason": "INTERACTION_POOL_PARTITION_MISMATCH", "missing": sorted((self_contained_ids & set(source_pool)) - set(interaction_pool))[:50], "extra": sorted(set(interaction_pool) - self_contained_ids)[:50]})
    if set(reconstruction) != incomplete_ids & set(source_pool):
        errors.append({"reason": "RECONSTRUCTION_PARTITION_MISMATCH"})
    if set(unsupported) != unsupported_ids & set(source_pool):
        errors.append({"reason": "UNSUPPORTED_PARTITION_MISMATCH"})
    if set(interaction_pool) & set(reconstruction) or set(interaction_pool) & set(unsupported) or set(reconstruction) & set(unsupported):
        errors.append({"reason": "PARTITION_OVERLAP"})
    for sample_id, row in reconstruction.items():
        candidate = row.get("trajectory_candidate") or {}
        base_context = [str(value) for value in row.get("context_turn_ids") or []]
        expanded_context = [str(value) for value in candidate.get("expanded_context_turn_ids") or []]
        target_ids = {str(value) for value in row.get("target_turn_ids") or []}
        if row.get("sampling_eligibility") != "CONTEXT_RECONSTRUCTION_ONLY" or not candidate.get("semantic_truth_preserved"):
            errors.append({"sample_id": sample_id, "reason": "TRAJECTORY_CANDIDATE_NOT_ISOLATED"})
        if candidate.get("status") != "PENDING_TRAJECTORY_REVIEW" or not set(base_context).issubset(expanded_context) or target_ids & set(expanded_context):
            errors.append({"sample_id": sample_id, "reason": "TRAJECTORY_CANDIDATE_ALIGNMENT"})

    view_errors = []
    if args.views:
        view_root = Path(args.views)
        if view_root.exists():
            view_rows = []
            for path in list(view_root.glob("train_*.jsonl")) + [view_root / "validation.jsonl", view_root / "sealed_eval.jsonl"]:
                view_rows.extend(rows(path))
            for row in view_rows:
                sample_id = str(row.get("sample_id"))
                sufficiency = row.get("context_sufficiency") or {}
                if sufficiency.get("state") != "SELF_CONTAINED" or sample_id not in self_contained_ids:
                    view_errors.append(sample_id)
            if view_errors:
                errors.append({"reason": "NON_SELF_CONTAINED_ROW_IN_VIEW", "sample_ids": sorted(set(view_errors))[:50]})

    gates = {
        "CONTEXT_SUFFICIENCY_COVERAGE": "PASS" if decision_ids == verified_ids else "FAIL",
        "DIAGNOSTIC_CONTEXT_ISOLATION": "PASS" if not any(error.get("reason") in {"FORBIDDEN_EVIDENCE_EXPOSED", "NON_SELECTED_CONTENT_EXPOSED"} for error in errors) else "FAIL",
        "FINAL_SELECTION_ALIGNMENT": "PASS" if not any(error.get("reason") in {"SOURCE_CONTEXT_MISMATCH", "SOURCE_TARGET_MISMATCH"} for error in errors) else "FAIL",
        "INTERACTION_VIEW_SELF_CONTAINED_ONLY": "PASS" if not any(error.get("reason") in {"INTERACTION_POOL_PARTITION_MISMATCH", "NON_SELF_CONTAINED_ROW_IN_VIEW"} for error in errors) else "FAIL",
        "RECONSTRUCTION_ROUTING": "PASS" if not any(error.get("reason") == "RECONSTRUCTION_PARTITION_MISMATCH" for error in errors) else "FAIL",
        "UNSUPPORTED_QUARANTINE_ROUTING": "PASS" if not any(error.get("reason") == "UNSUPPORTED_PARTITION_MISMATCH" for error in errors) else "FAIL",
        "PARTITION_EXCLUSIVITY": "PASS" if not any(error.get("reason") == "PARTITION_OVERLAP" for error in errors) else "FAIL",
        "TRAJECTORY_CANDIDATE_MATERIALIZATION": "PASS" if not any(error.get("reason", "").startswith("TRAJECTORY_CANDIDATE") for error in errors) else "FAIL",
    }
    report = {
        "schema_version": semantic.SCHEMA_VERSION,
        "pipeline_version": semantic.PIPELINE_VERSION,
        "context_sufficiency_version": CONTEXT_SUFFICIENCY_VERSION,
        "state_counts": dict(state_counts),
        "counts": {
            "verified_pool_required": len(verified_ids),
            "decisions": len(decisions),
            "source_verified_pool": len(source_pool),
            "interaction_pool": len(interaction_pool),
            "reconstruction_candidates": len(reconstruction),
            "unsupported_quarantine": len(unsupported),
        },
        "errors": errors[:100],
        "gates": gates,
        "pass": not errors and all(value == "PASS" for value in gates.values()),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate v2.3 context-sufficiency closure")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--requests", default=str(DEFAULT_DATASET / "context_sufficiency_requests_v2_3.jsonl"))
    parser.add_argument("--decisions", default=str(DEFAULT_DATASET / "context_sufficiency_decisions_v2_3.jsonl"))
    parser.add_argument("--source-pool", default=str(DEFAULT_DATASET / "verified_interaction_pool_v2_3.jsonl"))
    parser.add_argument("--interaction-pool", default=str(DEFAULT_DATASET / "interaction_view_pool_v2_3.jsonl"))
    parser.add_argument("--reconstruction-candidates", default=str(DEFAULT_DATASET / "context_reconstruction_candidates_v2_3.jsonl"))
    parser.add_argument("--unsupported-quarantine", default=str(DEFAULT_DATASET / "unsupported_relation_quarantine_v2_3.jsonl"))
    parser.add_argument("--views", default="")
    parser.add_argument("--report", default=str(DEFAULT_DATASET / "context_sufficiency_validation_v2_3.json"))
    args = parser.parse_args()
    print(json.dumps(validate_context(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
