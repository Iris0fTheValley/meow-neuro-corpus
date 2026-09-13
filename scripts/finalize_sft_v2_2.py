from __future__ import annotations

"""Resolve interaction closure and build the hard-deduplicated verified pool.

Sampling and split construction deliberately live in
``build_sft_v2_2_train_views.py``. This finalizer never treats a primary-model
PASS or a lexical heuristic as verified truth.
"""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "datasets" / "meow_v02_sft_v2_2_semantic_verified"
sys.path.insert(0, str(ROOT / "scripts"))
import sft_interaction_semantic_closure as closure  # noqa: E402
import sft_semantic_verified_v2_1 as legacy_core  # noqa: E402
import build_sft_v2_2_structural_candidates as structural_core  # noqa: E402


def load_jsonl(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                yield row


def latest(path: Path) -> dict[str, dict]:
    return {str(row.get("sample_id")): row for row in load_jsonl(path)}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _assert_result_schema(values: dict[str, dict], label: str) -> None:
    incompatible = [sample_id for sample_id, row in values.items() if row.get("schema_version") != closure.SCHEMA_VERSION or row.get("stage") != label]
    if incompatible:
        raise SystemExit(f"{label} contains {len(incompatible)} legacy/incompatible rows; rerun the v2.3 {label} stage")


def _identity_provenance(row: dict[str, Any], fusion: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    speaker = str((row.get("speaker_evidence") or {}).get("assistant_speaker") or "")
    result = fusion.get((str(row.get("source_id")), speaker)) or {}
    return {
        "result": result.get("identity") or row.get("identity"),
        "fusion_status": result.get("fusion_status"),
        "mapping_id": result.get("mapping_id"),
        "source_artifact": "identity_results/identity_mapping_multimodal_fusion_proxy.jsonl",
        "lineage_artifact": "reports/identity_closure_lineage.json",
        "source_prior_version": (result.get("source_prior") or {}).get("version"),
        "audio_consensus_gate": (result.get("audio_evidence") or {}).get("ensemble_gate"),
        "semantic_cannot_promote_without_audio": (result.get("semantic_evidence") or {}).get("cannot_promote_without_independent_audio"),
        "identity_unchanged_by_interaction_closure": True,
    }


def finalize(args: argparse.Namespace) -> dict:
    out = Path(args.output)
    structural_path = Path(args.structural)
    request_path = Path(args.requests)
    rows = list(load_jsonl(structural_path))
    if not rows:
        raise SystemExit(f"structural artifact is missing or empty: {structural_path}")
    incompatible_structural = [row.get("sample_id") for row in rows if row.get("artifact_schema_version") != structural_core.STRUCTURAL_SCHEMA_VERSION or row.get("pipeline_version") != structural_core.STRUCTURAL_PIPELINE_VERSION]
    if incompatible_structural:
        raise SystemExit(f"structural artifact has {len(incompatible_structural)} legacy/incompatible rows; rebuild Layer A")
    requests = latest(request_path)
    primary, strict, adjudication = latest(Path(args.primary)), latest(Path(args.strict)), latest(Path(args.adjudication))
    _assert_result_schema(primary, "primary")
    _assert_result_schema(strict, "strict")
    _assert_result_schema(adjudication, "adjudication")

    decisions = []
    for row in rows:
        sample_id = str(row.get("sample_id"))
        request_row = requests.get(sample_id) or {}
        request = request_row.get("request") or {}
        request_hash = request_row.get("request_sha256")
        if request.get("schema_version") != closure.SCHEMA_VERSION or not request_hash or request_hash != closure.payload_sha256(request):
            decisions.append({"schema_version": closure.SCHEMA_VERSION, "pipeline_version": closure.PIPELINE_VERSION, "sample_id": sample_id, "state": "MISSING_INVALID_EVIDENCE", "reason_code": "REQUEST_MISSING_VERSION_OR_HASH_INVALID", "risk_features": request.get("risk_features") or {}})
            continue
        evidence = []
        for value in (primary.get(sample_id), strict.get(sample_id), adjudication.get(sample_id)):
            if value and value.get("request_sha256") != request_hash:
                evidence.append({**value, "validated": {"valid": False, "accepted": False, "reason": "request_hash_mismatch"}})
            else:
                evidence.append(value)
        decisions.append(closure.resolve_semantic_state(sample_id, evidence[0], evidence[1], evidence[2], risk=request.get("risk_features") or {}))
    write_jsonl(out / "interaction_closure_decisions_v2_3.jsonl", decisions)
    status = closure.closure_status_report(decisions, len(rows))
    status["generated_at"] = datetime.now(timezone.utc).isoformat()
    status["pool_constructible"] = not status["pending"] and not status["missing_or_invalid_evidence"]
    write_json(out / "interaction_closure_status_v2_3.json", status)
    write_jsonl(out / "interaction_judge_conflicts_v2_3.jsonl", [row for row in decisions if row.get("state") == "JUDGE_CONFLICT"])
    write_jsonl(out / "interaction_ambiguous_v2_3.jsonl", [row for row in decisions if row.get("state") == "AMBIGUOUS"])
    if not status["pool_constructible"] and not args.allow_partial:
        raise SystemExit("interaction closure has pending or invalid evidence; reports were written but verified pool was not built")

    fusion = legacy_core.load_fusion()
    timelines = legacy_core.load_timelines({str(row.get("source_id")) for row in rows}, fusion)
    by_decision = {str(row.get("sample_id")): row for row in decisions}
    verified = []
    materialization_failures = []
    for row in rows:
        sample_id = str(row.get("sample_id"))
        materialized = closure.materialize_verified(row, by_decision[sample_id], timelines.get(str(row.get("source_id")), {}))
        if by_decision[sample_id].get("state") == "VERIFIED" and materialized is None:
            materialization_failures.append(sample_id)
            continue
        if materialized:
            materialized["identity_provenance"] = _identity_provenance(materialized, fusion)
            verified.append(materialized)
    if materialization_failures:
        status["pool_constructible"] = False
        status["materialization_failures"] = materialization_failures[:100]
        write_json(out / "interaction_closure_status_v2_3.json", status)
        raise SystemExit(f"{len(materialization_failures)} verified decisions could not resolve raw turn ids")
    if rows and not verified:
        status["pool_constructible"] = False
        status["reason"] = "NO_VERIFIED_INTERACTIONS"
        write_json(out / "interaction_closure_status_v2_3.json", status)
        raise SystemExit("no verified interactions remain after semantic closure")

    pairwise_edges = legacy_core.overlap_edges(verified)
    for edge in pairwise_edges:
        edge["schema_version"] = closure.SCHEMA_VERSION
        edge["pipeline_version"] = closure.PIPELINE_VERSION
    write_jsonl(out / "recording_overlap_edges_v2_3.jsonl", pairwise_edges)
    repair = closure.repair_recording_families(verified, pairwise_edges, min_supporting_pairs=args.min_recording_edge_support, max_component_size=args.max_recording_component_size)
    write_json(out / "recording_family_repair_v2_3.json", repair)
    pool, dedup = closure.provenance_aware_dedup(verified, repair["cluster_to_family"])
    for row in pool:
        existing = row.get("distribution_metadata") or {}
        row["distribution_metadata"] = {**closure.distribution_metadata(row), **existing}
        row["verified_pool_status"] = "VERIFIED_HARD_DEDUPED"
        row["sampling_eligibility"] = "ELIGIBLE"
        row.pop("split", None)
        row.pop("recommended_train", None)
    write_jsonl(out / "verified_interaction_pool_v2_3.jsonl", sorted(pool, key=lambda value: str(value.get("sample_id"))))
    write_json(out / "hard_dedup_audit_v2_3.json", dedup)

    state_counts = Counter(str(row.get("state")) for row in decisions)
    quality = {
        "schema_version": closure.SCHEMA_VERSION,
        "pipeline_version": closure.PIPELINE_VERSION,
        "artifact_status": "VERIFIED_INTERACTION_POOL" if status["pool_constructible"] else "PARTIAL_VERIFIED_INTERACTION_POOL",
        "identity_closure": "UNCHANGED_COMPATIBLE",
        "structural_candidates": len(rows),
        "semantic_state_counts": dict(state_counts),
        "semantic_primary_coverage": "PASS" if status["primary_coverage_pass"] else "PARTIAL",
        "semantic_strict_or_equivalent_verification": "PASS" if status["strict_or_equivalent_verification_pass"] else "PARTIAL",
        "unresolved_high_risk_samples": status["unresolved_high_risk_samples"],
        "judge_conflicts": status["judge_conflicts"],
        "ambiguous": status["ambiguous"],
        "missing_or_invalid_evidence": status["missing_or_invalid_evidence"],
        "semantic_verified_before_hard_dedup": len(verified),
        "verified_pool_rows": len(pool),
        "recording_pairwise_edges": len(pairwise_edges),
        "recording_edges_quarantined": len(repair["quarantined_edges"]),
        "recording_families": len(repair["recording_families"]),
        "hard_dedup_scope": "RECORDING_AND_PROVENANCE_AWARE",
        "independent_behavior_repetitions_preserved": dedup["cross_recording_behavior_repetitions_preserved"],
        "target_reuse": dedup["target_turn_reuse_count_after"],
        "prefix_ladder": dedup["prefix_ladder_count_after"],
        "verified_pool_constructible": status["pool_constructible"],
        "train_sampling_separated": True,
        "training_candidate_true_count": 0,
        "ready_for_first_sft": False,
    }
    write_json(out / "interaction_closure_quality_v2_3.json", quality)
    manifest = {
        "schema_version": closure.SCHEMA_VERSION,
        "pipeline_version": closure.PIPELINE_VERSION,
        "artifact_status": quality["artifact_status"],
        "files": {
            "structural_candidates": structural_path.name,
            "judge_requests": request_path.name,
            "primary_results": Path(args.primary).name,
            "strict_results": Path(args.strict).name,
            "adjudication_results": Path(args.adjudication).name,
            "closure_decisions": "interaction_closure_decisions_v2_3.jsonl",
            "judge_conflicts": "interaction_judge_conflicts_v2_3.jsonl",
            "ambiguous": "interaction_ambiguous_v2_3.jsonl",
            "recording_overlap_edges": "recording_overlap_edges_v2_3.jsonl",
            "recording_family_repair": "recording_family_repair_v2_3.json",
            "hard_dedup_audit": "hard_dedup_audit_v2_3.json",
            "verified_pool": "verified_interaction_pool_v2_3.jsonl",
            "quality": "interaction_closure_quality_v2_3.json",
            "status": "interaction_closure_status_v2_3.json",
        },
        "source_of_truth": {"timeline": "unique_timelines", "identity": "identity_results/identity_mapping_multimodal_fusion_proxy.jsonl", "identity_lineage": "reports/identity_closure_lineage.json"},
        "legacy_v2_2_artifacts": "HISTORICAL_REVIEW_ONLY_NOT_SEMANTICALLY_EQUIVALENT",
        "next_stage": "build_sft_v2_2_train_views.py",
        "training_candidate_true_count": 0,
        "ready_for_first_sft": False,
    }
    write_json(out / "interaction_closure_manifest_v2_3.json", manifest)
    return {"status": "FINALIZED", "quality": quality, "output": str(out)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve semantic evidence and build verified pool")
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--structural", default=str(DEFAULT_OUT / "structural_candidates_v2_2.jsonl"))
    parser.add_argument("--requests", default=str(DEFAULT_OUT / "interaction_judge_requests_v2_3.jsonl"))
    parser.add_argument("--primary", default=str(DEFAULT_OUT / "interaction_judge_primary_results_v2_3.jsonl"))
    parser.add_argument("--strict", default=str(DEFAULT_OUT / "interaction_judge_strict_results_v2_3.jsonl"))
    parser.add_argument("--adjudication", default=str(DEFAULT_OUT / "interaction_judge_adjudication_results_v2_3.jsonl"))
    parser.add_argument("--allow-partial", action="store_true", help="build a labelled partial pool while leaving readiness false")
    parser.add_argument("--min-recording-edge-support", type=int, default=2)
    parser.add_argument("--max-recording-component-size", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(finalize(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
