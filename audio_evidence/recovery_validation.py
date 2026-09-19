from __future__ import annotations

"""Mandatory production validation for recovery-v2 artifacts."""

from collections import defaultdict
from typing import Any, Iterable, Mapping

from .recovery_v2 import (
    ContextSufficiencyState,
    ReconciliationState,
    SpeakerState,
    TargetResolutionState,
    is_non_conversational_sentinel,
    normalized_tokens,
)


MANDATORY_GATES = (
    "TURN_RECONCILIATION_RESOLVED",
    "NO_UNRECONCILED_AUDIO_NEW",
    "AMBIGUOUS_RECONCILIATION_USED_FOR_RECOVERY",
    "UNRESOLVED_TRUE_NEW_IN_TRAINING",
    "NO_OLD_NEW_DUPLICATE",
    "NO_MULTI_OLD_TURN_SWALLOW",
    "NO_BIDIRECTIONAL_BOUNDARY_SMEAR",
    "NO_FRAGMENT_CHAIN",
    "SPEAKER_STATE_EXPLICIT",
    "NO_UNKNOWN_AS_USER_DEFAULT",
    "NO_ACTIVITY_ONLY_IDENTITY_PROMOTION",
    "LEGACY_CONTEXT_ROLE_VALIDATED",
    "NO_LEGACY_SPLIT_ROLE_CONTINUATION",
    "NO_SENTINEL_TURN",
    "NO_CONTROL_TEXT_IN_MESSAGES",
    "CONTEXT_IS_MINIMAL",
    "CONTEXT_SUFFICIENCY_CHECKED",
    "CONTEXT_SUFFICIENCY_ACTUALLY_JUDGED",
    "RECOVERED_SAMPLE_WITHOUT_SUFFICIENT_CONTEXT",
    "NO_BOOL_CONTEXT_SUFFICIENCY",
    "BASELINE_MONOTONICITY",
    "NO_UNEXPLAINED_BASELINE_REGRESSION",
    "TARGET_RESCUE_EVIDENCE_VALID",
    "NO_ACTIVITY_ONLY_TARGET_RESCUE",
    "TARGET_CONTRADICTION_RESOLVED",
    "QUARANTINE_CAUSE_TRACEABLE",
    "QUARANTINE_BASELINE_STATE_TRACEABLE",
    "NO_TARGET_IN_CONTEXT",
    "SINGLE_SUPERVISED_TARGET",
    "HISTORICAL_ASSISTANT_MASKED",
    "HISTORICAL_ASSISTANT_LOSS_LEAKAGE",
    "TARGET_REUSE",
    "PREFIX_LADDER",
    "NO_SYNTHETIC_PROMPT",
    "SEMANTIC_TRUTH_UNCHANGED",
    "SPLIT_AUTHORITY_PRESERVED",
    "IDENTITY_AUTHORITY_PRESERVED",
    "FAMILY_SPLIT_LEAKAGE",
    "SEALED_EVAL_LEAKAGE",
    "CROSS_RECORDING_CONTEXT",
)


def _contains(left: list[str], right: list[str]) -> bool:
    if not left or not right or left == right:
        return False
    if len(left) <= len(right):
        return any(right[index:index + len(left)] == left for index in range(len(right) - len(left) + 1))
    return any(left[index:index + len(right)] == right for index in range(len(left) - len(right) + 1))


def validate_recovery_v2(
    *,
    materialized: Iterable[dict[str, Any]],
    timeline: Iterable[dict[str, Any]],
    reconciliations: Iterable[dict[str, Any]],
    speaker_resolutions: Iterable[dict[str, Any]],
    context_selections: Iterable[dict[str, Any]],
    context_sufficiency: Iterable[dict[str, Any]],
    target_resolutions: Iterable[dict[str, Any]],
    quarantine: Iterable[dict[str, Any]],
    dedup_removed: Iterable[dict[str, Any]],
    baseline_materialized_ids: set[str],
    expected_semantic_hashes: Mapping[str, str],
    expected_split_hash: str,
    expected_identity_hash: str,
    actual_identity_hash: str,
) -> dict[str, Any]:
    rows = list(materialized)
    turns = list(timeline)
    reconciliation_rows = list(reconciliations)
    speaker_rows = list(speaker_resolutions)
    selection_rows = list(context_selections)
    sufficiency_rows = list(context_sufficiency)
    target_rows = list(target_resolutions)
    quarantine_rows = list(quarantine)
    removed_rows = list(dedup_removed)
    gates = {name: "PASS" for name in MANDATORY_GATES}
    errors: list[dict[str, Any]] = []

    def fail(gate: str, sample_id: Any, reason: str) -> None:
        gates[gate] = "FAIL"
        if len(errors) < 2000:
            errors.append({"gate": gate, "sample_id": sample_id, "reason": reason})

    turn_by_id = {str(turn.get("audio_turn_id")): turn for turn in turns}
    reconciliation_by_turn: dict[str, list[dict[str, Any]]] = defaultdict(list)
    direct_reconciliation_by_turn: dict[str, dict[str, Any]] = {}
    for entry in reconciliation_rows:
        replacement_id = entry.get("audio_turn_id")
        if replacement_id:
            reconciliation_by_turn[str(replacement_id)].append(entry)
        candidate_id = str(entry.get("candidate_span_id") or "")
        if candidate_id.startswith("old:") and ":" not in candidate_id[4:]:
            direct_reconciliation_by_turn[candidate_id[4:]] = entry
        for turn_id in entry.get("old_turn_ids") or []:
            reconciliation_by_turn[str(turn_id)].append(entry)
    speaker_by_turn = {str(row.get("audio_turn_id")): row for row in speaker_rows}
    selection_by_sample = {str(row.get("sample_id")): row for row in selection_rows}
    sufficiency_by_sample = {str(row.get("sample_id")): row for row in sufficiency_rows}
    target_by_sample = {str(row.get("sample_id")): row for row in target_rows}
    quarantine_by_sample = {str(row.get("sample_id")): row for row in quarantine_rows}

    for turn in turns:
        turn_id = str(turn.get("audio_turn_id"))
        state = str(turn.get("speaker_resolution_state") or "")
        if state not in {value.value for value in SpeakerState}:
            fail("SPEAKER_STATE_EXPLICIT", turn_id, "speaker state missing or invalid")
        if turn.get("role") == "user" and state != SpeakerState.CONFIRMED_NON_TARGET.value:
            fail("NO_UNKNOWN_AS_USER_DEFAULT", turn_id, "user role lacks confirmed non-target state")
        resolution = turn.get("speaker_resolution") or speaker_by_turn.get(turn_id) or {}
        evidence = resolution.get("evidence") or {}
        if resolution.get("resolution_basis") == "TARGET_ACTIVITY" or evidence.get("activity_used_for_identity"):
            fail("NO_ACTIVITY_ONLY_IDENTITY_PROMOTION", turn_id, "activity promoted identity")
        if is_non_conversational_sentinel(turn.get("resolved_text")) and turn.get("training_eligible"):
            fail("NO_SENTINEL_TURN", turn_id, "sentinel marked training eligible")
        if not reconciliation_by_turn.get(turn_id):
            fail("TURN_RECONCILIATION_RESOLVED", turn_id, "turn has no reconciliation record")

    for entry in reconciliation_rows:
        state = str(entry.get("reconciliation_state") or "")
        if state in {"", "NEW_ASR"}:
            fail("TURN_RECONCILIATION_RESOLVED", entry.get("candidate_span_id"), "legacy/unresolved reconciliation state")
        if (
            state == "MERGE_EXISTING"
            and entry.get("materialized_as_single_turn")
            and not (entry.get("boundary_validated") and entry.get("role_validated"))
        ):
            fail("NO_MULTI_OLD_TURN_SWALLOW", entry.get("candidate_span_id"), "multi-old span materialized without same-speaker topology proof")
        if state == "AMBIGUOUS_BOUNDARY" and entry.get("training_eligible"):
            fail("NO_BIDIRECTIONAL_BOUNDARY_SMEAR", entry.get("candidate_span_id"), "ambiguous edge entered training")

    supervised_targets: dict[tuple[str, str], list[str]] = defaultdict(list)
    interaction_keys: dict[str, list[str]] = defaultdict(list)
    prompt_groups: dict[tuple[str, tuple[str, ...]], list[tuple[str, list[str]]]] = defaultdict(list)
    family_memberships: dict[str, set[str]] = defaultdict(set)
    materialized_ids = set()
    recovered_classes = {
        "RECOVERED_FROM_EXISTING_TIMELINE",
        "RECOVERED_FROM_NEW_AUDIO",
        "RECOVERED_FROM_BOTH",
    }
    for row in rows:
        sample = str(row.get("sample_id"))
        materialized_ids.add(sample)
        messages = row.get("messages") or []
        target_id = str(row.get("target_turn_id") or "")
        context_ids = [str(value) for value in row.get("context_turn_ids") or []]
        if target_id in context_ids:
            fail("NO_TARGET_IN_CONTEXT", sample, "target id appears in context")
        if not messages or messages[-1].get("role") != "assistant":
            fail("SINGLE_SUPERVISED_TARGET", sample, "final message is not assistant")
        supervised = [index for index, message in enumerate(messages) if message.get("supervise") is True]
        if supervised != [len(messages) - 1]:
            fail("SINGLE_SUPERVISED_TARGET", sample, "exactly the final message must be supervised")
            fail("HISTORICAL_ASSISTANT_LOSS_LEAKAGE", sample, "supervision leaks outside final target")
        if any(message.get("role") == "assistant" and index < len(messages) - 1 and message.get("supervise") is not False
               for index, message in enumerate(messages)):
            fail("HISTORICAL_ASSISTANT_MASKED", sample, "historical assistant not explicitly masked")
        if row.get("synthetic_prompt") is not False:
            fail("NO_SYNTHETIC_PROMPT", sample, "synthetic prompt marker is not false")
        for index, message in enumerate(messages):
            for source_turn_id in message.get("source_turn_ids") or []:
                source_turn_id = str(source_turn_id)
                source = turn_by_id.get(source_turn_id)
                if source_turn_id.startswith("audio:candidate:"):
                    fail("NO_UNRECONCILED_AUDIO_NEW", sample, "audio candidate entered messages")
                if source is None:
                    fail("TURN_RECONCILIATION_RESOLVED", sample, "message source turn absent")
                    continue
                source_id = str(source.get("audio_turn_id") or "")
                direct = direct_reconciliation_by_turn.get(source_id)
                source_state = str(
                    (direct or {}).get("reconciliation_state")
                    or source.get("reconciliation_state")
                    or ""
                )
                recovered = (
                    str(row.get("context_reconstruction_class") or "") in recovered_classes
                    and not row.get("was_baseline_materialized")
                    and index < len(messages) - 1
                )
                if recovered and source_state in {
                    ReconciliationState.AMBIGUOUS.value,
                    ReconciliationState.AMBIGUOUS_BOUNDARY.value,
                }:
                    fail("AMBIGUOUS_RECONCILIATION_USED_FOR_RECOVERY", sample, "ambiguous turn entered recovered context")
                if source_state == ReconciliationState.TRUE_NEW.value and source.get("speaker_resolution_state") == SpeakerState.AMBIGUOUS_SPEAKER.value:
                    fail("UNRESOLVED_TRUE_NEW_IN_TRAINING", sample, "unresolved TRUE_NEW source entered messages")
                if source.get("speaker_resolution_state") == SpeakerState.AMBIGUOUS_SPEAKER.value:
                    fail("SPEAKER_STATE_EXPLICIT", sample, "ambiguous speaker entered messages")
                if is_non_conversational_sentinel(message.get("content")):
                    fail("NO_CONTROL_TEXT_IN_MESSAGES", sample, "control/sentinel message entered training")
            if len(message.get("source_turn_ids") or []) > 1:
                fail("NO_MULTI_OLD_TURN_SWALLOW", sample, "one message swallowed multiple old turns")
            if index and message.get("source_turn_ids") == messages[index - 1].get("source_turn_ids"):
                fail("NO_OLD_NEW_DUPLICATE", sample, "same source turn duplicated in adjacent messages")
        selection = selection_by_sample.get(sample)
        if not selection or not selection.get("minimality_checked"):
            fail("CONTEXT_IS_MINIMAL", sample, "minimal context selection was not checked")
        sufficiency = sufficiency_by_sample.get(sample)
        if not sufficiency or sufficiency.get("state") != ContextSufficiencyState.CONTEXT_SUFFICIENT.value:
            fail("CONTEXT_SUFFICIENCY_CHECKED", sample, "selected context was not marked sufficient")
            if str(row.get("context_reconstruction_class") or "") in recovered_classes:
                fail("RECOVERED_SAMPLE_WITHOUT_SUFFICIENT_CONTEXT", sample, "recovered sample lacks sufficient context")
        if sufficiency and str(sufficiency.get("checker") or "").startswith("bool"):
            fail("NO_BOOL_CONTEXT_SUFFICIENCY", sample, "boolean context checker used")
        recovered_requires_judge = (
            str(row.get("context_reconstruction_class") or "") in recovered_classes
            and not row.get("was_baseline_materialized")
        )
        if recovered_requires_judge:
            if not sufficiency or not (
                str(sufficiency.get("checker") or "").startswith("semantic-context-sufficiency-judge")
                and sufficiency.get("judge_valid") is True
            ):
                fail("CONTEXT_SUFFICIENCY_ACTUALLY_JUDGED", sample, "recovered sample did not use semantic sufficiency judge")
        target_resolution = target_by_sample.get(sample) or {}
        if target_resolution.get("state") in {
            TargetResolutionState.AUDIO_MAJOR_CONTRADICTION.value,
            TargetResolutionState.LOCAL_REASR_REQUIRED.value,
            TargetResolutionState.UNRESOLVED.value,
        }:
            fail("TARGET_CONTRADICTION_RESOLVED", sample, "unresolved/contradicted target materialized")
        if target_resolution.get("resolution_basis") == "TARGET_ACTIVITY":
            fail("NO_ACTIVITY_ONLY_TARGET_RESCUE", sample, "target rescued from activity alone")
        if row.get("semantic_truth_sha256") != expected_semantic_hashes.get(sample):
            fail("SEMANTIC_TRUTH_UNCHANGED", sample, "semantic truth hash changed")
        if row.get("split_authority_sha256") != expected_split_hash:
            fail("SPLIT_AUTHORITY_PRESERVED", sample, "split authority hash changed")
        supervised_targets[(str(row.get("canonical_recording_id")), target_id)].append(sample)
        interaction_keys[str(row.get("interaction_dedup_key") or "")].append(sample)
        target_tokens = normalized_tokens(messages[-1].get("content") if messages else "")
        prompt_groups[(str(row.get("recording_id")), tuple(context_ids))].append((sample, target_tokens))
        if {str(message.get("source_recording_id") or row.get("recording_id")) for message in messages} != {str(row.get("recording_id"))}:
            fail("CROSS_RECORDING_CONTEXT", sample, "message source recording differs from interaction recording")
        family_memberships[str(row.get("recording_family_id"))].add(str(row.get("final_view_membership")))

    for samples in supervised_targets.values():
        if len(samples) > 1:
            fail("TARGET_REUSE", samples, "same canonical target supervised more than once")
    for key, samples in interaction_keys.items():
        if key and len(samples) > 1:
            fail("NO_OLD_NEW_DUPLICATE", samples, "duplicate interaction key")
    for values in prompt_groups.values():
        for index, (sample, tokens) in enumerate(values):
            for other_sample, other_tokens in values[index + 1:]:
                if _contains(tokens, other_tokens):
                    fail("PREFIX_LADDER", [sample, other_sample], "same-context target prefix ladder")
    for family, memberships in family_memberships.items():
        active = memberships & {"IN_TRAIN", "IN_VALIDATION", "IN_SEALED_EVAL"}
        if len(active) > 1:
            fail("FAMILY_SPLIT_LEAKAGE", family, "recording family appears in multiple splits")
            if "IN_SEALED_EVAL" in active:
                fail("SEALED_EVAL_LEAKAGE", family, "sealed eval family appears elsewhere")

    explicitly_removed_baseline = {
        str(row.get("sample_id")) for row in removed_rows
        if row.get("explicit_contradiction")
    } | {
        sample for sample, row in quarantine_by_sample.items()
        if row.get("was_baseline_materialized") and row.get("explicit_contradiction")
    }
    missing_baseline = baseline_materialized_ids - materialized_ids - explicitly_removed_baseline
    if missing_baseline:
        fail("BASELINE_MONOTONICITY", sorted(missing_baseline)[:100], "baseline samples disappeared without explicit contradiction")
        fail("NO_UNEXPLAINED_BASELINE_REGRESSION", sorted(missing_baseline)[:100], "unexplained baseline regression")

    required_quarantine_fields = {
        "sample_id", "was_baseline_materialized", "was_baseline_quarantined", "baseline_reason",
        "failure_stage", "failure_reason", "target_interval", "containing_window_ids",
        "intersecting_window_ids", "alignment_available", "target_activity_available",
        "diarization_available", "identity_evidence_available", "old_new_disagreement_state",
        "explicit_contradiction", "local_reasr_attempted", "local_reasr_result",
        "resolution_attempts", "final_status",
    }
    for row in quarantine_rows:
        sample = str(row.get("sample_id"))
        missing = required_quarantine_fields - set(row)
        if missing:
            fail("QUARANTINE_CAUSE_TRACEABLE", sample, "missing quarantine fields: %s" % sorted(missing))
        if "was_baseline_materialized" not in row or "was_baseline_quarantined" not in row:
            fail("QUARANTINE_BASELINE_STATE_TRACEABLE", sample, "baseline state absent")
        if row.get("failure_reason") == "TARGET_AUDIO_EVIDENCE_UNAVAILABLE" and row.get("was_baseline_materialized"):
            fail("TARGET_RESCUE_EVIDENCE_VALID", sample, "baseline regression mislabeled as unavailable evidence")

    if expected_identity_hash != actual_identity_hash:
        fail("IDENTITY_AUTHORITY_PRESERVED", None, "identity authority hash changed")

    # These gates are enforced structurally: no audio candidates are ever
    # materialized and legacy corrections carry explicit role validation.
    if any(row.get("fragment_chain_unresolved") for row in reconciliation_rows):
        fail("NO_FRAGMENT_CHAIN", None, "unresolved fragment chain remains")
    if any(row.get("legacy_role_validated") is False for row in reconciliation_rows):
        fail("LEGACY_CONTEXT_ROLE_VALIDATED", None, "legacy role correction lacks evidence")
    if any(row.get("legacy_split_role_continuation_unresolved") for row in reconciliation_rows):
        fail("NO_LEGACY_SPLIT_ROLE_CONTINUATION", None, "legacy split-role continuation unresolved")

    mandatory_not_checked = sum(value == "NOT_CHECKED" for value in gates.values())
    invariants = {
        "semantic_truth_mutation": int(gates["SEMANTIC_TRUTH_UNCHANGED"] != "PASS"),
        "split_authority_mutation": int(gates["SPLIT_AUTHORITY_PRESERVED"] != "PASS"),
        "identity_authority_mutation": int(gates["IDENTITY_AUTHORITY_PRESERVED"] != "PASS"),
        "baseline_unexplained_regression": len(missing_baseline),
        "unknown_speaker_defaulted_to_user": int(gates["NO_UNKNOWN_AS_USER_DEFAULT"] != "PASS"),
        "activity_only_identity_promotion": int(gates["NO_ACTIVITY_ONLY_IDENTITY_PROMOTION"] != "PASS"),
        "unreconciled_audio_new_in_training": int(gates["NO_UNRECONCILED_AUDIO_NEW"] != "PASS"),
        "old_new_duplicate_in_training": int(gates["NO_OLD_NEW_DUPLICATE"] != "PASS"),
        "sentinel_turn_in_training": int(gates["NO_SENTINEL_TURN"] != "PASS" or gates["NO_CONTROL_TEXT_IN_MESSAGES"] != "PASS"),
        "target_in_context": int(gates["NO_TARGET_IN_CONTEXT"] != "PASS"),
        "target_reuse": int(gates["TARGET_REUSE"] != "PASS"),
        "prefix_ladder_same_target": int(gates["PREFIX_LADDER"] != "PASS"),
        "historical_assistant_loss_leakage": int(gates["HISTORICAL_ASSISTANT_LOSS_LEAKAGE"] != "PASS"),
        "supervised_target_count_per_sample_invalid": int(gates["SINGLE_SUPERVISED_TARGET"] != "PASS"),
        "cross_recording_context": int(gates.get("CROSS_RECORDING_CONTEXT") == "FAIL"),
        "family_split_leakage": int(gates["FAMILY_SPLIT_LEAKAGE"] != "PASS"),
        "sealed_eval_leakage": int(gates["SEALED_EVAL_LEAKAGE"] != "PASS"),
        "mandatory_validator_not_checked": mandatory_not_checked,
    }
    return {
        "schema_version": "2.0.0",
        "mandatory_gates": gates,
        "errors": errors,
        "pass": all(value == "PASS" for value in gates.values()),
        "mandatory_validator_not_checked": mandatory_not_checked,
        "invariants": invariants,
        "counts": {
            "materialized": len(rows),
            "timeline_turns": len(turns),
            "reconciliations": len(reconciliation_rows),
            "speaker_resolutions": len(speaker_rows),
            "context_selections": len(selection_rows),
            "context_sufficiency": len(sufficiency_rows),
            "target_resolutions": len(target_rows),
            "quarantine": len(quarantine_rows),
            "dedup_removed": len(removed_rows),
            "baseline_materialized": len(baseline_materialized_ids),
        },
    }
