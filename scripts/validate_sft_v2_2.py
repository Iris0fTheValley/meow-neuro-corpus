from __future__ import annotations

"""Independent validator for interaction closure, verified pool, and views."""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "datasets" / "meow_v02_sft_v2_2_semantic_verified"
sys.path.insert(0, str(ROOT / "scripts"))
import sft_interaction_semantic_closure as closure  # noqa: E402


def rows(path: Path) -> list[dict]:
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


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def token_list(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(value or "").lower())


def response(row: dict) -> str:
    return " ".join(str(item.get("content") or "") for item in row.get("messages") or [] if item.get("role") == "assistant")


def contains(short: list[str], long: list[str]) -> bool:
    return bool(short) and len(short) <= len(long) and any(long[index : index + len(short)] == short for index in range(len(long) - len(short) + 1))


def validate(args: argparse.Namespace) -> dict:
    dataset = Path(args.dataset)
    pool = rows(Path(args.pool))
    decisions = rows(dataset / "interaction_closure_decisions_v2_3.jsonl")
    status = load_json(dataset / "interaction_closure_status_v2_3.json")
    dedup = load_json(dataset / "hard_dedup_audit_v2_3.json")
    repair = load_json(dataset / "recording_family_repair_v2_3.json")
    split_authority_path = Path(args.split_authority)
    split_authority = load_json(split_authority_path)
    schema_errors = []
    required = {"sample_id", "messages", "source_id", "canonical_recording_id", "recording_family_id", "context_turn_ids", "context_anchor_id", "target_turn_ids", "timestamps", "speaker_evidence", "identity_provenance", "semantic_closure", "semantic_qa", "transcript_qa", "distribution_metadata", "pipeline_version", "artifact_schema_version"}
    targets, anchors = defaultdict(list), defaultdict(list)
    provenance_alignment_errors = []
    for row in pool:
        missing = sorted(required - set(row))
        messages = row.get("messages") or []
        if missing or len(messages) != 2 or any(not str(item.get("content") or "").strip() for item in messages):
            schema_errors.append({"sample_id": row.get("sample_id"), "missing": missing, "message_count": len(messages)})
        if row.get("artifact_schema_version") != closure.SCHEMA_VERSION or row.get("pipeline_version") != closure.PIPELINE_VERSION:
            schema_errors.append({"sample_id": row.get("sample_id"), "reason": "VERSION_MISMATCH"})
        if (row.get("semantic_closure") or {}).get("state") != "VERIFIED" or (row.get("semantic_qa") or {}).get("status") != "VERIFIED":
            schema_errors.append({"sample_id": row.get("sample_id"), "reason": "NOT_SEMANTICALLY_VERIFIED"})
        if row.get("transcript_quality") != "PASS" or (row.get("transcript_qa") or {}).get("status") != "STRUCTURAL_PASS":
            schema_errors.append({"sample_id": row.get("sample_id"), "reason": "TRANSCRIPT_EVIDENCE_NOT_PASS"})
        identity = row.get("identity_provenance") or {}
        if not identity.get("identity_unchanged_by_interaction_closure") or not identity.get("source_artifact"):
            schema_errors.append({"sample_id": row.get("sample_id"), "reason": "IDENTITY_PROVENANCE_MISSING"})
        context_ids = [str(value) for value in row.get("context_turn_ids") or []]
        target_ids = [str(value) for value in row.get("target_turn_ids") or []]
        expected_ids = context_ids + target_ids
        transcript = row.get("transcript_qa") or {}
        transcript_turns = transcript.get("turns") or []
        transcript_ids = [str(turn.get("turn_id")) for turn in transcript_turns]
        transcript_roles = [str(turn.get("role")) for turn in transcript_turns]
        raw_indices = [int(turn.get("raw_timeline_index", -1)) for turn in transcript_turns]
        speaker = row.get("speaker_evidence") or {}
        boundary = row.get("boundary_provenance") or {}
        alignment_reasons = []
        if transcript_ids != expected_ids or transcript.get("final_turn_ids") != expected_ids or transcript.get("context_turn_ids") != context_ids or transcript.get("target_turn_ids") != target_ids:
            alignment_reasons.append("TRANSCRIPT_TURN_IDS")
        if transcript_roles != ["context"] * len(context_ids) + ["target"] * len(target_ids):
            alignment_reasons.append("TRANSCRIPT_ROLES")
        if row.get("raw_timeline_indices") != raw_indices or raw_indices != sorted(raw_indices):
            alignment_reasons.append("RAW_TIMELINE_INDICES")
        context_turns = transcript_turns[: len(context_ids)]
        target_turns = transcript_turns[len(context_ids) :]
        if speaker.get("context_speakers") != [str(turn.get("speaker") or "") for turn in context_turns] or speaker.get("context_identities") != [str(turn.get("identity") or "UNKNOWN") for turn in context_turns]:
            alignment_reasons.append("CONTEXT_SPEAKER_PROVENANCE")
        if speaker.get("assistant_turn_speakers") != [str(turn.get("speaker") or "") for turn in target_turns] or speaker.get("assistant_turn_identities") != [str(turn.get("identity") or "UNKNOWN") for turn in target_turns]:
            alignment_reasons.append("TARGET_SPEAKER_PROVENANCE")
        if [str(turn.get("turn_id")) for turn in boundary.get("turns") or []] != expected_ids or boundary.get("context_turn_ids") != context_ids or boundary.get("target_turn_ids") != target_ids:
            alignment_reasons.append("BOUNDARY_PROVENANCE")
        timestamps = row.get("timestamps") or {}
        first_stamp = (transcript_turns[0].get("timestamp") or {}) if transcript_turns else {}
        last_stamp = (transcript_turns[-1].get("timestamp") or {}) if transcript_turns else {}
        try:
            timestamps_aligned = float(timestamps.get("start")) == float(first_stamp.get("start")) and float(timestamps.get("end")) == float(last_stamp.get("end"))
        except (TypeError, ValueError):
            timestamps_aligned = False
        if not timestamps_aligned:
            alignment_reasons.append("TIMESTAMPS")
        if alignment_reasons:
            provenance_alignment_errors.append({"sample_id": row.get("sample_id"), "reasons": alignment_reasons})
        for target_id in row.get("target_turn_ids") or []:
            targets[(str(row.get("source_id")), str(target_id))].append(str(row.get("sample_id")))
        anchors[(str(row.get("source_id")), str(row.get("context_anchor_id")))].append(str(row.get("sample_id")))
    target_reuse = [value for value in targets.values() if len(value) > 1]
    duplicate_anchor = [value for value in anchors.values() if len(value) > 1]
    prefix_ladders = []
    by_anchor = defaultdict(list)
    for row in pool:
        by_anchor[(str(row.get("source_id")), str(row.get("context_anchor_id")))].append((row, token_list(response(row))))
    for anchor, values in by_anchor.items():
        for index, (left, left_tokens) in enumerate(values):
            for right, right_tokens in values[index + 1 :]:
                if left_tokens != right_tokens and (contains(left_tokens, right_tokens) or contains(right_tokens, left_tokens)):
                    prefix_ladders.append({"anchor": anchor, "left": left.get("sample_id"), "right": right.get("sample_id")})

    state_counts = Counter(str(row.get("state") or "UNKNOWN") for row in decisions)
    conflict_rows_in_pool = [row.get("sample_id") for row in pool if (row.get("semantic_closure") or {}).get("state") == "JUDGE_CONFLICT"]
    views = Path(args.views) if args.views else None
    view_report = {"status": "NOT_YET_VALIDATED", "family_leakage": {}, "sealed_eval_in_train": [], "split_constraint_violations": [], "lineage_errors": []}
    if views and views.exists():
        train_files = list(views.glob("train_*.jsonl"))
        train = rows(train_files[0]) if train_files else []
        validation, sealed = rows(views / "validation.jsonl"), rows(views / "sealed_eval.jsonl")
        families = {"train": {str(row.get("recording_family_id")) for row in train}, "validation": {str(row.get("recording_family_id")) for row in validation}, "sealed_eval": {str(row.get("recording_family_id")) for row in sealed}}
        leakage = {"train_validation": sorted(families["train"] & families["validation"]), "train_sealed_eval": sorted(families["train"] & families["sealed_eval"]), "validation_sealed_eval": sorted(families["validation"] & families["sealed_eval"])}
        sealed_ids = {str(row.get("sample_id")) for row in sealed}
        train_ids = {str(row.get("sample_id")) for row in train}
        current_assignments = split_authority.get("current_family_assignments") or {}
        constraint_violations = [relation for relation in repair.get("split_exclusion_relations") or [] if current_assignments.get(str(relation.get("family_a"))) != current_assignments.get(str(relation.get("family_b")))]
        lineage_errors = []
        member_assignments = split_authority.get("member_assignments") or {}
        for family, members in (split_authority.get("current_family_members") or {}).items():
            inherited = {member_assignments.get(str(member)) for member in members if str(member) in member_assignments}
            if len(inherited) != 1 or (inherited and next(iter(inherited)) != current_assignments.get(family)):
                lineage_errors.append({"family": family, "member_splits": sorted(value for value in inherited if value), "current_split": current_assignments.get(family)})
        manifest = load_json(views / "view_manifest.json")
        authority_reference_ok = manifest.get("split_authority_version") == split_authority.get("split_authority_version") and Path(str(manifest.get("split_authority") or "")) == split_authority_path
        valid_view = not any(leakage.values()) and not sealed_ids & train_ids and not constraint_violations and not lineage_errors and split_authority.get("status") == "ACTIVE" and authority_reference_ok
        view_report = {"status": "PASS" if valid_view else "FAIL", "counts": {"train": len(train), "validation": len(validation), "sealed_eval": len(sealed)}, "family_leakage": leakage, "sealed_eval_in_train": sorted(sealed_ids & train_ids), "split_constraint_violations": constraint_violations, "lineage_errors": lineage_errors, "split_authority_status": split_authority.get("status"), "shared_split_authority_reference": authority_reference_ok}

    gates = {
        "IDENTITY_CLOSURE": "PASS" if not any(error.get("reason") == "IDENTITY_PROVENANCE_MISSING" for error in schema_errors) else "FAIL",
        "STRUCTURAL_INVARIANTS": "PASS" if not schema_errors and not provenance_alignment_errors and not target_reuse and not duplicate_anchor and not prefix_ladders else "FAIL",
        "SEMANTIC_PRIMARY_COVERAGE": "PASS" if status.get("primary_coverage_pass") else "PARTIAL",
        "SEMANTIC_STRICT_OR_EQUIVALENT_VERIFICATION": "PASS" if status.get("strict_or_equivalent_verification_pass") else "PARTIAL",
        "JUDGE_CONFLICTS": "PASS" if not conflict_rows_in_pool else "FAIL",
        "TRANSCRIPT_RISK_PROVENANCE": "PASS" if not any(error.get("reason") == "TRANSCRIPT_EVIDENCE_NOT_PASS" for error in schema_errors) else "FAIL",
        "SEMANTIC_PROVENANCE_REMATERIALIZATION": "PASS" if not provenance_alignment_errors else "FAIL",
        "TARGET_REUSE": "PASS" if not target_reuse else "FAIL",
        "PREFIX_LADDER": "PASS" if not prefix_ladders else "FAIL",
        "CROSS_SPLIT_RECORDING_LEAKAGE": view_report["status"],
        "HARD_DEDUP_SCOPE": "PASS" if "recording" in str(dedup.get("policy") or "").lower() and dedup.get("target_turn_reuse_count_after") == 0 else "FAIL",
        "INDEPENDENT_BEHAVIOR_REPETITION": "PASS" if "cross-family" in str(dedup.get("policy") or "").lower() else "FAIL",
        "RECORDING_BRIDGE_AUDIT": "PASS" if "quarantined_edges" in repair and "cluster_pair_evidence" in repair else "FAIL",
        "RECORDING_BRIDGE_SPLIT_ISOLATION": "PASS" if view_report["status"] == "PASS" and not view_report["split_constraint_violations"] else "NOT_YET_VALIDATED" if view_report["status"] == "NOT_YET_VALIDATED" else "FAIL",
        "SEALED_SPLIT_LINEAGE": "PASS" if view_report["status"] == "PASS" and not view_report["lineage_errors"] else "NOT_YET_VALIDATED" if view_report["status"] == "NOT_YET_VALIDATED" else "FAIL",
        "SAMPLING_POLICY_SPLIT_AUTHORITY": "PASS" if view_report.get("shared_split_authority_reference") else "NOT_YET_VALIDATED" if view_report["status"] == "NOT_YET_VALIDATED" else "FAIL",
        "VERIFIED_POOL": "PASS" if status.get("pool_constructible") and not schema_errors else "PARTIAL",
        "TRAIN_SAMPLING": "PASS" if view_report["status"] == "PASS" else "NOT_YET_VALIDATED",
    }
    hard_fail = any(value == "FAIL" for value in gates.values())
    report = {
        "schema_version": closure.SCHEMA_VERSION,
        "pipeline_version": closure.PIPELINE_VERSION,
        "artifact_status": "VALIDATED_INTERACTION_CLOSURE_ARCHITECTURE",
        "sample_counts": {"decisions": len(decisions), "verified_pool": len(pool)},
        "semantic_state_counts": dict(state_counts),
        "unresolved_high_risk_samples": status.get("unresolved_high_risk_samples"),
        "judge_conflicts": status.get("judge_conflicts"),
        "schema_errors": schema_errors[:100],
        "provenance_alignment_errors": provenance_alignment_errors[:100],
        "target_turn_reuse_count": len(target_reuse),
        "duplicate_context_anchor_count": len(duplicate_anchor),
        "prefix_ladder_count": len(prefix_ladders),
        "conflict_rows_in_verified_pool": conflict_rows_in_pool,
        "view_validation": view_report,
        "gates": gates,
        "pass": not hard_fail and all(value == "PASS" for value in gates.values()),
    }
    output = Path(args.report) if args.report else dataset / "validation_report_v2_3.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate verified pool and optional train view")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--pool", default=str(DEFAULT_DATASET / "verified_interaction_pool_v2_3.jsonl"))
    parser.add_argument("--views", default="")
    parser.add_argument("--split-authority", default=str(DEFAULT_DATASET / "split_authority_v2_3.json"))
    parser.add_argument("--report", default="")
    args = parser.parse_args()
    print(json.dumps(validate(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
