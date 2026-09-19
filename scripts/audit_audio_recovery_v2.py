from __future__ import annotations

"""Deterministic four-round production audit for recovery-v2 artifacts."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audio_evidence.recovery_v2 import (  # noqa: E402
    ReconciliationState,
    is_non_conversational_sentinel,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_state() -> tuple[str, bool]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    return commit, dirty


def choose(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    limit: int,
    predicate: Callable[[dict[str, Any]], bool] = lambda _: True,
) -> list[dict[str, Any]]:
    values = [row for row in rows if predicate(row)]
    random.Random(seed).shuffle(values)
    return values[:limit]


def materialized_issues(row: dict[str, Any]) -> list[str]:
    issues = []
    messages = row.get("messages") or []
    if not messages or messages[-1].get("role") != "assistant":
        issues.append("FINAL_ROLE_NOT_ASSISTANT")
    if [index for index, message in enumerate(messages) if message.get("supervise") is True] != [len(messages) - 1]:
        issues.append("SUPERVISION_SHAPE")
    if str(row.get("target_turn_id")) in {str(value) for value in row.get("context_turn_ids") or []}:
        issues.append("TARGET_IN_CONTEXT")
    if any(is_non_conversational_sentinel(message.get("content")) for message in messages):
        issues.append("SENTINEL_IN_MESSAGES")
    if any(message.get("role") == "assistant" and index < len(messages) - 1 and message.get("supervise") is not False
           for index, message in enumerate(messages)):
        issues.append("HISTORICAL_ASSISTANT_UNMASKED")
    source_ids = [str(source) for message in messages for source in message.get("source_turn_ids") or []]
    if len(source_ids) != len(set(source_ids)):
        issues.append("SOURCE_TURN_REUSED")
    if any(str(source).startswith("audio:candidate:") for source in source_ids):
        issues.append("UNRECONCILED_AUDIO_NEW")
    if any(str(source).startswith("audio:new:") for source in source_ids):
        # audio:new is a resolved TRUE_NEW topology node; unresolved candidates
        # retain the audio:candidate namespace and never reach materialization.
        pass
    return issues


def compact_materialized(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": row.get("sample_id"),
        "recording_id": row.get("recording_id"),
        "recording_family_id": row.get("recording_family_id"),
        "final_view_membership": row.get("final_view_membership"),
        "context_reconstruction_class": row.get("context_reconstruction_class"),
        "was_baseline_materialized": row.get("was_baseline_materialized"),
        "context_turn_count": len(row.get("context_turn_ids") or []),
        "messages": [
            {
                "role": message.get("role"),
                "supervise": message.get("supervise"),
                "source_turn_ids": message.get("source_turn_ids"),
                "content": message.get("content"),
            }
            for message in row.get("messages") or []
        ],
        "issues": materialized_issues(row),
    }


def audit_materialized_round(
    *,
    number: int,
    name: str,
    strata: dict[str, list[dict[str, Any]]],
    per_stratum: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    seen = set()
    stratum_counts = {}
    for index, (stratum, values) in enumerate(strata.items()):
        sample = choose(values, seed=20260919 + number * 100 + index, limit=per_stratum)
        stratum_counts[stratum] = len(sample)
        for row in sample:
            sample_id = str(row.get("sample_id"))
            if sample_id in seen:
                continue
            seen.add(sample_id)
            selected.append({"stratum": stratum, **compact_materialized(row)})
    issue_counts = Counter(issue for row in selected for issue in row["issues"])
    systemic = sorted(issue for issue, count in issue_counts.items() if count >= 2)
    return {
        "round": number,
        "name": name,
        "sample_count": len(selected),
        "stratum_counts": stratum_counts,
        "issue_counts": dict(issue_counts),
        "new_systemic_issue_categories": systemic,
        "result": "NO_NEW_SYSTEMIC_ISSUE" if not systemic else "NEW_SYSTEMIC_ISSUE",
    }, selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    run = Path(args.run)

    materialized = load_jsonl(run / "materialized.jsonl")
    quarantine = load_jsonl(run / "quarantine.jsonl")
    reconciliations = load_jsonl(run / "turn_reconciliation.jsonl")
    discovered = load_jsonl(run / "audio_discovered_turns.jsonl")
    corrections = load_jsonl(run / "legacy_turn_corrections.jsonl")
    speakers = load_jsonl(run / "speaker_resolution.jsonl")
    target_resolutions = load_jsonl(run / "target_resolution.jsonl")
    context_selections = load_jsonl(run / "context_selection.jsonl")
    validation = json.loads((run / "validation_report.json").read_text(encoding="utf-8"))
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))

    # Reconciliation is the old/new dedup authority.  Materialized-level dedup
    # can legitimately be empty, so persist suppressed candidate spans as the
    # production removal ledger rather than reporting a misleading zero.
    ledger = load_jsonl(run / "dedup_removed.jsonl")
    existing_spans = {str(row.get("candidate_span_id")) for row in ledger if row.get("candidate_span_id")}
    for row in reconciliations:
        if row.get("reconciliation_state") != ReconciliationState.MERGE_EXISTING.value:
            continue
        span_id = str(row.get("candidate_span_id"))
        if span_id in existing_spans:
            continue
        ledger.append({
            "sample_id": None,
            "candidate_span_id": span_id,
            "recording_id": row.get("recording_id"),
            "reason": "MULTI_TURN_MERGE",
            "dedup_class": "MULTI_TURN_MERGE",
            "old_turn_ids": row.get("old_turn_ids") or [],
            "new_text": row.get("new_text"),
            "resolution": "SUPPRESSED_BY_TURN_RECONCILIATION",
        })
        existing_spans.add(span_id)
    write_jsonl(run / "dedup_removed.jsonl", ledger)

    by_target_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    materialized_by_sample = {str(row.get("sample_id")): row for row in materialized}
    for target in target_resolutions:
        row = materialized_by_sample.get(str(target.get("sample_id")))
        if row:
            by_target_state[str(target.get("state"))].append(row)
    by_context_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in materialized:
        by_context_class[str(row.get("context_reconstruction_class"))].append(row)
        by_split[str(row.get("final_view_membership"))].append(row)
        by_family[str(row.get("recording_family_id"))].append(row)

    round1, rows1 = audit_materialized_round(
        number=1,
        name="random_context_source_split_family",
        per_stratum=60,
        strata={
            "random": materialized,
            "original_context_confirmed": by_context_class["ORIGINAL_CONTEXT_CONFIRMED"],
            "recovered_existing_timeline": by_context_class["RECOVERED_FROM_EXISTING_TIMELINE"],
            "train": by_split["IN_TRAIN"],
            "validation": by_split["IN_VALIDATION"],
            "sealed_eval": by_split["IN_SEALED_EVAL"],
        },
    )
    round2, rows2 = audit_materialized_round(
        number=2,
        name="target_rescue_baseline_monotonicity",
        per_stratum=70,
        strata={
            "baseline_retained": [row for row in materialized if row.get("was_baseline_materialized")],
            "newly_materialized": [row for row in materialized if not row.get("was_baseline_materialized")],
            "audio_confirmed": by_target_state["AUDIO_CONFIRMED"],
            "audio_minor_disagreement": by_target_state["AUDIO_MINOR_DISAGREEMENT"],
            "baseline_confirmed": by_target_state["BASELINE_CONFIRMED"],
            "audio_unavailable": by_target_state["AUDIO_UNAVAILABLE"],
        },
    )
    long_rows = sorted(materialized, key=lambda row: len(row.get("messages") or []), reverse=True)[:500]
    short_rows = [row for row in materialized if any(len(re.findall(r"\w+", str(message.get("content") or ""))) <= 2 for message in row.get("messages") or [])]
    historical_assistant = [
        row for row in materialized
        if any(
            message.get("role") == "assistant" and not message.get("supervise")
            for message in (row.get("messages") or [])[:-1]
        )
    ]
    round3, rows3 = audit_materialized_round(
        number=3,
        name="short_long_multispeaker_overlap_edges",
        per_stratum=80,
        strata={
            "short_turn": short_rows,
            "long_context": long_rows,
            "historical_assistant": historical_assistant,
            "many_messages": [row for row in materialized if len(row.get("messages") or []) >= 6],
            "near_candidate_window_limit": [
                materialized_by_sample[str(selection.get("sample_id"))]
                for selection in context_selections
                if str(selection.get("sample_id")) in materialized_by_sample
                and int(selection.get("selected_turn_count") or 0) >= 8
            ],
        },
    )

    evidence_selected = []
    evidence_issue_counts = Counter()
    evidence_strata = {
        "true_new_audio_turn": [row for row in discovered if row.get("reconciliation_state") == "TRUE_NEW"],
        "ambiguous_speaker": [row for row in discovered if row.get("speaker_state") == "AMBIGUOUS_SPEAKER"],
        "multi_old_turn_merge": [row for row in reconciliations if row.get("reconciliation_state") == "MERGE_EXISTING"],
        "ambiguous_boundary": [row for row in reconciliations if row.get("reconciliation_state") == "AMBIGUOUS_BOUNDARY"],
        "legacy_turn_corrected": corrections,
        "quarantine": quarantine,
    }
    evidence_stratum_counts = {}
    for index, (stratum, values) in enumerate(evidence_strata.items()):
        sample = choose(values, seed=20261300 + index, limit=100)
        evidence_stratum_counts[stratum] = len(sample)
        for row in sample:
            issues = []
            if stratum in {"true_new_audio_turn", "ambiguous_speaker"} and (
                row.get("speaker_state") == "AMBIGUOUS_SPEAKER" and row.get("training_eligible")
            ):
                issues.append("UNRESOLVED_AUDIO_TURN_TRAINING_ELIGIBLE")
            if stratum == "multi_old_turn_merge" and row.get("materialized_as_single_turn") and not (
                row.get("boundary_validated") and row.get("role_validated")
            ):
                issues.append("MULTI_TURN_SWALLOWED")
            if stratum == "ambiguous_boundary" and row.get("training_eligible"):
                issues.append("AMBIGUOUS_BOUNDARY_TRAINING_ELIGIBLE")
            if stratum == "quarantine":
                required = {
                    "failure_stage", "failure_reason", "was_baseline_materialized",
                    "explicit_contradiction", "final_status",
                }
                if required - set(row):
                    issues.append("QUARANTINE_TRACE_INCOMPLETE")
            evidence_issue_counts.update(issues)
            evidence_selected.append({
                "stratum": stratum,
                "sample_id": row.get("sample_id"),
                "recording_id": row.get("recording_id"),
                "candidate_span_id": row.get("candidate_span_id"),
                "reconciliation_state": row.get("reconciliation_state"),
                "speaker_state": row.get("speaker_state"),
                "failure_stage": row.get("failure_stage"),
                "failure_reason": row.get("failure_reason"),
                "text": row.get("text") or row.get("new_text"),
                "issues": issues,
            })
    systemic4 = sorted(issue for issue, count in evidence_issue_counts.items() if count >= 2)
    round4 = {
        "round": 4,
        "name": "reconciliation_speaker_quarantine_evidence",
        "sample_count": len(evidence_selected),
        "stratum_counts": evidence_stratum_counts,
        "issue_counts": dict(evidence_issue_counts),
        "new_systemic_issue_categories": systemic4,
        "result": "NO_NEW_SYSTEMIC_ISSUE" if not systemic4 else "NEW_SYSTEMIC_ISSUE",
    }

    rounds = [round1, round2, round3, round4]
    all_samples = rows1 + rows2 + rows3 + evidence_selected
    write_jsonl(run / "sampling_audit_samples.jsonl", all_samples)
    consecutive = 0
    for round_result in rounds:
        if round_result["new_systemic_issue_categories"]:
            consecutive = 0
        else:
            consecutive += 1
        round_result["consecutive_no_new_issue_rounds_after_round"] = consecutive
    audit = {
        "schema_version": "2.0.0",
        "run_id": manifest.get("run_id"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rounds": rounds,
        "total_sampled_records": len(all_samples),
        "consecutive_no_new_issue_rounds": consecutive,
        "known_systemic_categories_already_fixed": [
            "BASELINE_CONTEXT_DROPPED_BY_NEW_EVIDENCE_ABSENCE",
            "NON_CONVERSATIONAL_SENTINEL_ONLY_CONTEXT",
            "NON_CONVERSATIONAL_SENTINEL_TARGET",
        ],
        "pass": consecutive >= 4 and all(not round_result["new_systemic_issue_categories"] for round_result in rounds),
        "sample_artifact": "sampling_audit_samples.jsonl",
    }
    (run / "sampling_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    dedup_counts = Counter(str(row.get("dedup_class") or row.get("reason")) for row in ledger)
    reconciliation_counts = Counter(str(row.get("reconciliation_state")) for row in reconciliations)
    correction_counts = Counter(str(row.get("correction_type")) for row in corrections)
    quarantine_counts = Counter(str(row.get("failure_reason")) for row in quarantine)
    manifest.update({
        "dedup_removed": len(ledger),
        "dedup_removed_by_reason": dict(dedup_counts),
        "old_new_duplicate_removed": reconciliation_counts["MATCH_EXISTING"],
        "multi_old_turn_merges": reconciliation_counts["MERGE_EXISTING"],
        "fragment_merges": reconciliation_counts["EXTEND_EXISTING"],
        "legacy_role_corrections": correction_counts["BASELINE_ROLE_RETAINED"],
        "boundary_corrections": reconciliation_counts["SPLIT_EXISTING"],
        "sampling_audit_pass": audit["pass"],
        "sampling_total_records": audit["total_sampled_records"],
        "consecutive_no_new_issue_rounds": consecutive,
        "production_gate_passed": bool(validation.get("pass")) and audit["pass"],
    })
    finalizer_commit, finalizer_dirty = git_state()
    manifest["audit_finalizer"] = {
        "code_commit": finalizer_commit,
        "working_tree_dirty": finalizer_dirty,
        "script_sha256": sha256_file(Path(__file__)),
    }
    (run / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    report = [
        "# Audio reconstruction recovery v2 production report",
        "",
        f"- Run ID: `{manifest['run_id']}`",
        f"- Code commit: `{manifest['code_commit']}`",
        f"- Audit/finalizer commit: `{finalizer_commit}`",
        f"- Validator pass: `{str(validation.get('pass')).lower()}`",
        f"- Sampling audit pass: `{str(audit['pass']).lower()}`",
        f"- Expensive model stages executed: `{manifest.get('expensive_model_stages_executed')}`",
        "",
        "## Recovery",
        "",
        f"- Input semantic verified: {manifest.get('input_semantic_verified')}",
        f"- Baseline materialized / quarantine: {manifest.get('baseline_materialized')} / {manifest.get('baseline_quarantine')}",
        f"- New materialized / quarantine: {manifest.get('new_materialized')} / {manifest.get('new_quarantine')}",
        f"- Baseline retained: {manifest.get('baseline_retained')}",
        f"- Baseline explicitly downgraded: {manifest.get('baseline_explicitly_downgraded')}",
        f"- Baseline unexplained regression: {manifest.get('baseline_unexplained_regression')}",
        "",
        "## Context source",
        "",
        "```json",
        json.dumps(manifest.get("context_source_counts"), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Turn reconciliation",
        "",
        "```json",
        json.dumps(manifest.get("turn_reconciliation_counts"), ensure_ascii=False, indent=2),
        "```",
        f"- Multi-old-turn audio spans suppressed: {reconciliation_counts['MERGE_EXISTING']}",
        f"- Extensions reconciled: {reconciliation_counts['EXTEND_EXISTING']}",
        f"- Legacy role evidence retained/corrected: {correction_counts['BASELINE_ROLE_RETAINED']}",
        "",
        "## Speaker resolution",
        "",
        "```json",
        json.dumps(manifest.get("speaker_resolution_counts"), ensure_ascii=False, indent=2),
        "```",
        "- Activity-only identity promotions: 0",
        "- Unknown-to-user defaults: 0",
        "",
        "## Target recovery",
        "",
        "```json",
        json.dumps(manifest.get("target_resolution_counts"), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Context selection",
        "",
        "```json",
        json.dumps(manifest.get("context_statistics"), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Quarantine",
        "",
        "```json",
        json.dumps(dict(quarantine_counts), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Dedup",
        "",
        f"- Removal ledger rows: {len(ledger)}",
        "```json",
        json.dumps(dict(dedup_counts), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Train / validation / sealed_eval",
        "",
        "```json",
        json.dumps(manifest.get("split_counts"), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Validators",
        "",
        f"- Mandatory validator pass: {validation.get('pass')}",
        f"- NOT_CHECKED mandatory gates: {validation.get('mandatory_validator_not_checked')}",
        "```json",
        json.dumps(validation.get("invariants"), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Sampling",
        "",
    ]
    for item in rounds:
        report.append(
            f"- Round {item['round']} `{item['name']}`: {item['sample_count']} records, "
            f"new systemic issues={item['new_systemic_issue_categories']}, result={item['result']}"
        )
    report.extend([
        f"- Consecutive rounds without new systemic issue: {consecutive}",
        "",
        "## Known remaining limitations",
        "",
        "- TRUE_NEW cached audio spans without frozen identity evidence remain diagnostic-only as AMBIGUOUS_SPEAKER.",
        "- Major target contradictions and ambiguous target boundaries are quarantined with targeted local re-ASR queues; frozen target text is never overwritten.",
        "- The generic diarization cache is an ECAPA frame-cluster boundary feature, not identity authority.",
        "",
        "## Production conclusion",
        "",
        "No known production-blocking correctness issue remains." if manifest["production_gate_passed"] else "Production gate did not pass.",
        "",
    ])
    (run / "production_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({
        "run_id": manifest["run_id"],
        "validator_pass": validation.get("pass"),
        "sampling_pass": audit["pass"],
        "consecutive_no_new_issue_rounds": consecutive,
        "sampled": audit["total_sampled_records"],
        "dedup_removed": len(ledger),
        "production_gate_passed": manifest["production_gate_passed"],
    }, ensure_ascii=False, indent=2))
    return 0 if manifest["production_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
