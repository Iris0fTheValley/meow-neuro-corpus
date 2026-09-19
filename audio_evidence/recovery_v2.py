from __future__ import annotations

"""Fail-closed recovery primitives for audio-reconstruction-v1 recovery v2.

The functions in this module operate only on cached evidence.  They do not
create semantic, identity, or split authority.  In particular:

* target activity is an evidence feature, never an identity decision;
* generic diarization is a boundary feature, never a target identity decision;
* an unknown speaker is ambiguous, never implicitly a user;
* old/new text is reconciled before any role is materialized.
"""

from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum
import hashlib
import json
import re
from typing import Any, Iterable, Optional, Sequence


TARGET_IDENTITIES = {
    "NEURO",
    "EVIL_NEURO",
    "NEURO_FAMILY",
    "NEURO_FAMILY_HIGH",
    "NEURO_FAMILY_MEDIUM",
}
AMBIGUOUS_IDENTITIES = {"", "NONE", "NULL", "UNKNOWN", "AMBIGUOUS", "UNRESOLVED"}
SENTINEL_EXACT = {
    "filtered",
    "filter",
    "censored",
    "redacted",
    "content filtered",
    "message filtered",
    "audio filtered",
}


class ReconciliationState(str, Enum):
    MATCH_EXISTING = "MATCH_EXISTING"
    EXTEND_EXISTING = "EXTEND_EXISTING"
    SPLIT_EXISTING = "SPLIT_EXISTING"
    MERGE_EXISTING = "MERGE_EXISTING"
    TRUE_NEW = "TRUE_NEW"
    AMBIGUOUS = "AMBIGUOUS"
    AMBIGUOUS_BOUNDARY = "AMBIGUOUS_BOUNDARY"


class SpeakerState(str, Enum):
    CONFIRMED_TARGET = "CONFIRMED_TARGET"
    CONFIRMED_NON_TARGET = "CONFIRMED_NON_TARGET"
    AMBIGUOUS_SPEAKER = "AMBIGUOUS_SPEAKER"


class TargetResolutionState(str, Enum):
    BASELINE_CONFIRMED = "BASELINE_CONFIRMED"
    AUDIO_CONFIRMED = "AUDIO_CONFIRMED"
    AUDIO_MINOR_DISAGREEMENT = "AUDIO_MINOR_DISAGREEMENT"
    AUDIO_MAJOR_CONTRADICTION = "AUDIO_MAJOR_CONTRADICTION"
    AUDIO_UNAVAILABLE = "AUDIO_UNAVAILABLE"
    LOCAL_REASR_REQUIRED = "LOCAL_REASR_REQUIRED"
    UNRESOLVED = "UNRESOLVED"


class ContextSufficiencyState(str, Enum):
    CONTEXT_SUFFICIENT = "CONTEXT_SUFFICIENT"
    CONTEXT_INSUFFICIENT = "CONTEXT_INSUFFICIENT"
    CONTEXT_AMBIGUOUS = "CONTEXT_AMBIGUOUS"


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalized_tokens(text: Any) -> list[str]:
    return re.findall(r"[a-z0-9']+", str(text or "").lower())


def normalized_text(text: Any) -> str:
    return " ".join(normalized_tokens(text))


def is_non_conversational_sentinel(text: Any) -> bool:
    return normalized_text(text) in SENTINEL_EXACT


def sentinel_only_context(turns: Sequence[dict[str, Any]]) -> bool:
    return bool(turns) and all(
        is_non_conversational_sentinel(turn.get("resolved_text") or turn.get("text"))
        for turn in turns
    )


def interval_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def interval_iou(a0: float, a1: float, b0: float, b1: float) -> float:
    intersection = interval_overlap(a0, a1, b0, b1)
    return intersection / max(1e-9, max(a1, b1) - min(a0, b0))


def _contiguous_contains(container: Sequence[str], contained: Sequence[str]) -> bool:
    if not contained or len(contained) > len(container):
        return False
    return any(list(container[index:index + len(contained)]) == list(contained)
               for index in range(len(container) - len(contained) + 1))


def text_similarity(left: Any, right: Any) -> float:
    a, b = normalized_tokens(left), normalized_tokens(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    sequence = SequenceMatcher(a=a, b=b, autojunk=False).ratio()
    aset, bset = set(a), set(b)
    jaccard = len(aset & bset) / max(1, len(aset | bset))
    return max(sequence, jaccard)


def resolve_speaker_state(
    identity: Any,
    *,
    baseline_role: Optional[str] = None,
    frozen_semantic_role: Optional[str] = None,
    identity_evidence_available: bool = False,
    target_activity_available: bool = False,
    generic_diarization_available: bool = False,
) -> dict[str, Any]:
    """Resolve only from frozen identity or an already materialized baseline role.

    The activity and generic-diarization flags are recorded but deliberately do
    not participate in the decision.
    """
    label = str(identity or "").upper()
    evidence = {
        "frozen_identity": str(identity) if identity is not None else None,
        "identity_evidence_available": bool(identity_evidence_available),
        "target_activity_available": bool(target_activity_available),
        "generic_diarization_available": bool(generic_diarization_available),
        "frozen_semantic_role": frozen_semantic_role,
        "activity_used_for_identity": False,
        "generic_diarization_used_for_identity": False,
    }
    if label in TARGET_IDENTITIES:
        return {
            "speaker_state": SpeakerState.CONFIRMED_TARGET.value,
            "role": "assistant",
            "resolution_basis": "FROZEN_IDENTITY_AUTHORITY",
            "evidence": evidence,
        }
    if label not in AMBIGUOUS_IDENTITIES:
        return {
            "speaker_state": SpeakerState.CONFIRMED_NON_TARGET.value,
            "role": "user",
            "resolution_basis": "FROZEN_IDENTITY_AUTHORITY",
            "evidence": evidence,
        }
    if baseline_role in {"user", "assistant"}:
        state = SpeakerState.CONFIRMED_TARGET if baseline_role == "assistant" else SpeakerState.CONFIRMED_NON_TARGET
        return {
            "speaker_state": state.value,
            "role": baseline_role,
            "resolution_basis": "BASELINE_MATERIALIZED_ROLE_MONOTONICITY",
            "evidence": evidence,
        }
    if frozen_semantic_role == "user":
        return {
            "speaker_state": SpeakerState.CONFIRMED_NON_TARGET.value,
            "role": "user",
            "resolution_basis": "FROZEN_SEMANTIC_CONTEXT_ROLE",
            "evidence": evidence,
        }
    return {
        "speaker_state": SpeakerState.AMBIGUOUS_SPEAKER.value,
        "role": None,
        "resolution_basis": "INSUFFICIENT_IDENTITY_EVIDENCE",
        "evidence": evidence,
    }


def dominant_diarization_cluster(
    start: float,
    end: float,
    diarization: Iterable[dict[str, Any]],
) -> tuple[Optional[str], bool]:
    scores: dict[str, float] = {}
    overlap_seen = False
    for item in diarization:
        amount = interval_overlap(start, end, float(item.get("start", 0)), float(item.get("end", 0)))
        if amount <= 0:
            continue
        cluster = str(item.get("speaker_cluster") or "UNKNOWN")
        scores[cluster] = scores.get(cluster, 0.0) + amount
        overlap_seen = overlap_seen or bool(item.get("overlap"))
    if not scores:
        return None, overlap_seen
    ordered = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    if len(ordered) > 1 and ordered[1][1] >= ordered[0][1] * 0.8:
        return None, overlap_seen
    return ordered[0][0], overlap_seen


def assign_tokens_to_old_turns(
    tokens: Iterable[dict[str, Any]],
    old_turns: Sequence[dict[str, Any]],
    *,
    boundary_tolerance: float = 0.12,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Assign each token once using its midpoint and old interval boundaries.

    A token that straddles two adjacent old turns without a stable midpoint
    assignment is emitted as an ambiguous boundary token and cannot become a
    new training turn.
    """
    ordered_old = sorted(old_turns, key=lambda turn: (float(turn["start"]), float(turn["end"]), str(turn["turn_id"])))
    assigned = {str(turn["turn_id"]): [] for turn in ordered_old}
    unassigned: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    for raw in sorted(tokens, key=lambda item: (float(item.get("start", 0)), float(item.get("end", 0)), str(item.get("text") or ""))):
        token = dict(raw)
        start, end = float(token.get("start", 0)), float(token.get("end", 0))
        if end <= start or not normalized_tokens(token.get("text")):
            continue
        midpoint = (start + end) / 2.0
        candidates = [
            turn for turn in ordered_old
            if float(turn["start"]) - boundary_tolerance <= midpoint <= float(turn["end"]) + boundary_tolerance
        ]
        if not candidates:
            unassigned.append(token)
            continue
        ranked = sorted(
            candidates,
            key=lambda turn: (
                -interval_overlap(start, end, float(turn["start"]), float(turn["end"])),
                abs(midpoint - (float(turn["start"]) + float(turn["end"])) / 2.0),
                str(turn["turn_id"]),
            ),
        )
        best = ranked[0]
        if len(ranked) > 1:
            best_overlap = interval_overlap(start, end, float(best["start"]), float(best["end"]))
            second_overlap = interval_overlap(start, end, float(ranked[1]["start"]), float(ranked[1]["end"]))
            if best_overlap > 0 and second_overlap >= best_overlap * 0.9:
                token["candidate_old_turn_ids"] = [str(best["turn_id"]), str(ranked[1]["turn_id"])]
                ambiguous.append(token)
                continue
        token["assigned_old_turn_id"] = str(best["turn_id"])
        assigned[str(best["turn_id"])].append(token)
    return assigned, unassigned, ambiguous


def reconcile_old_turn(
    old_turn: dict[str, Any],
    assigned_tokens: Sequence[dict[str, Any]],
    *,
    is_frozen_target: bool = False,
) -> dict[str, Any]:
    old_text = str(old_turn.get("text") or old_turn.get("old_transcript") or "").strip()
    new_text = " ".join(str(token.get("text") or "").strip() for token in assigned_tokens).strip()
    old_tokens, new_tokens = normalized_tokens(old_text), normalized_tokens(new_text)
    similarity = text_similarity(old_text, new_text)
    if not new_tokens:
        state = ReconciliationState.MATCH_EXISTING
        reason = "NO_ATTRIBUTED_NEW_TOKENS_OLD_EVIDENCE_RETAINED"
        chosen_text, authority = old_text, "OLD_TRANSCRIPT"
        confidence = 0.55
    elif old_tokens == new_tokens or similarity >= 0.88:
        state = ReconciliationState.MATCH_EXISTING
        reason = "NORMALIZED_OR_FUZZY_OLD_NEW_MATCH"
        chosen_text, authority = old_text, "OLD_TRANSCRIPT"
        confidence = max(0.88, similarity)
    elif _contiguous_contains(new_tokens, old_tokens) and len(new_tokens) > len(old_tokens):
        extra_ratio = (len(new_tokens) - len(old_tokens)) / max(1, len(old_tokens))
        if not is_frozen_target and extra_ratio <= 0.5:
            state = ReconciliationState.EXTEND_EXISTING
            reason = "AUDIO_RECOVERS_BOUNDED_PREFIX_OR_SUFFIX"
            chosen_text, authority = new_text, "RECONCILED_TEXT"
            confidence = max(0.75, similarity)
        else:
            state = ReconciliationState.AMBIGUOUS
            reason = "TARGET_OR_LARGE_EXTENSION_REQUIRES_LOCAL_VERIFICATION"
            chosen_text, authority = old_text, "OLD_TRANSCRIPT"
            confidence = similarity
    elif _contiguous_contains(old_tokens, new_tokens) and similarity >= 0.55:
        state = ReconciliationState.MATCH_EXISTING
        reason = "NEW_ASR_MINOR_OMISSION_OLD_TEXT_RETAINED"
        chosen_text, authority = old_text, "OLD_TRANSCRIPT"
        confidence = max(0.7, similarity)
    else:
        state = ReconciliationState.AMBIGUOUS
        reason = "MAJOR_OR_UNCLASSIFIED_OLD_NEW_DISAGREEMENT"
        chosen_text, authority = old_text, "OLD_TRANSCRIPT"
        confidence = similarity
    return {
        "reconciliation_state": state.value,
        "old_turn_ids": [str(old_turn["turn_id"])],
        "old_text": old_text,
        "new_text": new_text or None,
        "chosen_text": chosen_text,
        "chosen_text_authority": authority,
        "similarity": round(similarity, 6),
        "confidence": round(float(confidence), 6),
        "resolution_reason": reason,
        "source_spans": [
            {"start": float(token["start"]), "end": float(token["end"]), "text": token.get("text")}
            for token in assigned_tokens
        ],
    }


def reconcile_span(
    span: dict[str, Any],
    overlapping_old_turns: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    text = str(span.get("text") or "").strip()
    old = sorted(overlapping_old_turns, key=lambda turn: (float(turn["start"]), str(turn["turn_id"])))
    if not old:
        state = ReconciliationState.TRUE_NEW
        reason = "NO_OLD_TIMELINE_TURN_OVERLAP"
    elif len(old) == 1:
        detail = reconcile_old_turn(old[0], span.get("tokens") or [])
        return {**detail, "candidate_span_id": span.get("candidate_span_id")}
    else:
        joined = " ".join(str(turn.get("text") or "").strip() for turn in old).strip()
        similarity = text_similarity(joined, text)
        state = ReconciliationState.MERGE_EXISTING if similarity >= 0.6 else ReconciliationState.AMBIGUOUS_BOUNDARY
        reason = "AUDIO_SPAN_COVERS_MULTIPLE_OLD_TURNS" if state == ReconciliationState.MERGE_EXISTING else "MULTI_OLD_TURN_SPAN_NOT_STABLY_DECOMPOSABLE"
    return {
        "candidate_span_id": span.get("candidate_span_id"),
        "reconciliation_state": state.value,
        "old_turn_ids": [str(turn["turn_id"]) for turn in old],
        "old_text": " ".join(str(turn.get("text") or "").strip() for turn in old).strip() or None,
        "new_text": text or None,
        "chosen_text": None if state != ReconciliationState.TRUE_NEW else text,
        "chosen_text_authority": None if state != ReconciliationState.TRUE_NEW else "RECONSTRUCTED_CANDIDATE",
        "similarity": round(text_similarity(" ".join(str(turn.get("text") or "") for turn in old), text), 6) if old else 0.0,
        "confidence": 0.8 if state == ReconciliationState.MERGE_EXISTING else 0.5,
        "resolution_reason": reason,
        "source_spans": [
            {"start": float(token["start"]), "end": float(token["end"]), "text": token.get("text")}
            for token in span.get("tokens") or []
        ],
    }


def build_candidate_spans(
    tokens: Iterable[dict[str, Any]],
    diarization: Iterable[dict[str, Any]],
    *,
    recording_id: str,
    window_ids: Sequence[str],
    max_gap: float = 1.25,
) -> list[dict[str, Any]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_cluster: Optional[str] = None
    for token in sorted(tokens, key=lambda item: (float(item["start"]), float(item["end"]))):
        start, end = float(token["start"]), float(token["end"])
        cluster, overlap = dominant_diarization_cluster(start, end, diarization)
        item = dict(token)
        item["boundary_speaker_cluster"] = cluster
        item["generic_overlap"] = overlap
        hard_break = bool(
            current
            and (
                start - float(current[-1]["end"]) > max_gap
                or overlap
                or current[-1].get("generic_overlap")
                or (cluster and current_cluster and cluster != current_cluster)
            )
        )
        if hard_break:
            groups.append(current)
            current = []
        current.append(item)
        current_cluster = cluster or current_cluster
    if current:
        groups.append(current)
    spans = []
    for index, group in enumerate(groups):
        text = " ".join(str(token.get("text") or "").strip() for token in group).strip()
        if not text:
            continue
        start, end = float(group[0]["start"]), float(group[-1]["end"])
        cluster, overlap = dominant_diarization_cluster(start, end, diarization)
        spans.append({
            "candidate_span_id": "audio:candidate:%s:%s" % (
                recording_id,
                canonical_sha256({"windows": sorted(window_ids), "start": round(start, 3), "end": round(end, 3), "text": normalized_text(text)})[:16],
            ),
            "recording_id": recording_id,
            "start": start,
            "end": end,
            "text": text,
            "tokens": group,
            "speaker_cluster": cluster,
            "generic_overlap": overlap,
            "source_window_ids": sorted(set(window_ids)),
            "candidate_index": index,
        })
    return spans


def resolve_target(
    *,
    old_text: Any,
    new_text: Any,
    baseline_materialized: bool,
    alignment_available: bool,
    identity_confirmed_target: bool,
    ambiguous_boundary: bool = False,
    aligned_token_count: int = 0,
) -> dict[str, Any]:
    old_value, new_value = str(old_text or "").strip(), str(new_text or "").strip()
    similarity = text_similarity(old_value, new_value)
    old_token_count = len(normalized_tokens(old_value))
    evidence_is_substantial = aligned_token_count >= max(3, min(8, int(old_token_count * 0.4)))
    if not old_value or not identity_confirmed_target:
        state = TargetResolutionState.UNRESOLVED
        reason = "FROZEN_TARGET_TEXT_OR_IDENTITY_AUTHORITY_MISSING"
        explicit = True
    elif ambiguous_boundary:
        state = TargetResolutionState.LOCAL_REASR_REQUIRED
        reason = "TARGET_BOUNDARY_AMBIGUOUS"
        explicit = False
    elif not alignment_available or not new_value:
        state = TargetResolutionState.BASELINE_CONFIRMED
        reason = "VERIFIED_TARGET_RETAINED_WITHOUT_NEW_AUDIO_CONFIRMATION"
        explicit = False
    elif similarity >= 0.86:
        state = TargetResolutionState.AUDIO_CONFIRMED
        reason = "TOKEN_ALIGNED_AUDIO_MATCH"
        explicit = False
    elif similarity >= 0.55:
        state = TargetResolutionState.AUDIO_MINOR_DISAGREEMENT
        reason = "MINOR_AUDIO_TEXT_DISAGREEMENT_FROZEN_TEXT_RETAINED"
        explicit = False
    elif not evidence_is_substantial:
        state = TargetResolutionState.AUDIO_UNAVAILABLE
        reason = "INSUFFICIENT_ALIGNED_TOKENS_FOR_CONTRADICTION"
        explicit = False
    else:
        state = TargetResolutionState.AUDIO_MAJOR_CONTRADICTION
        reason = "TOKEN_ALIGNED_AUDIO_MAJOR_CONTRADICTION"
        explicit = True
    return {
        "state": state.value,
        "baseline_materialized": bool(baseline_materialized),
        "old_text": old_value or None,
        "new_text": new_value or None,
        "chosen_text": old_value or None,
        "chosen_text_authority": "FROZEN_SEMANTIC_TARGET",
        "similarity": round(similarity, 6),
        "alignment_available": bool(alignment_available),
        "aligned_token_count": int(aligned_token_count),
        "aligned_evidence_substantial": evidence_is_substantial,
        "identity_confirmed_target": bool(identity_confirmed_target),
        "explicit_contradiction": explicit,
        "reason": reason,
    }


def deduplicate_recovery_rows(
    rows: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deduplicate while preserving already materialized baseline samples."""
    ordered = sorted(
        rows,
        key=lambda row: (
            not bool(row.get("was_baseline_materialized")),
            str(row.get("sample_id")),
        ),
    )
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    seen_targets: dict[tuple[str, str], str] = {}
    seen_interactions: dict[str, str] = {}
    seen_context_targets: dict[tuple[str, tuple[str, ...]], list[tuple[list[str], str]]] = {}
    for row in ordered:
        sample_id = str(row.get("sample_id"))
        target_key = (str(row.get("canonical_recording_id") or row.get("recording_id")), str(row.get("target_turn_id")))
        if target_key in seen_targets:
            removed.append({
                "sample_id": sample_id,
                "reason": "TARGET_REUSE",
                "dedup_class": "TARGET_REUSE",
                "kept_sample_id": seen_targets[target_key],
            })
            continue
        interaction_key = str(row.get("interaction_dedup_key") or "")
        if interaction_key and interaction_key in seen_interactions:
            removed.append({
                "sample_id": sample_id,
                "reason": "SAME_INTERACTION_DUPLICATE",
                "dedup_class": "SAME_INTERACTION_DUPLICATE",
                "kept_sample_id": seen_interactions[interaction_key],
            })
            continue
        group_key = (str(row.get("recording_id")), tuple(str(value) for value in row.get("context_turn_ids") or []))
        target_tokens = normalized_tokens((row.get("messages") or [{}])[-1].get("content"))
        ladder_match = None
        for prior_tokens, prior_sample in seen_context_targets.get(group_key, []):
            if target_tokens != prior_tokens and (
                _contiguous_contains(target_tokens, prior_tokens)
                or _contiguous_contains(prior_tokens, target_tokens)
            ):
                ladder_match = prior_sample
                break
        if ladder_match:
            removed.append({
                "sample_id": sample_id,
                "reason": "PREFIX_LADDER",
                "dedup_class": "PREFIX_LADDER",
                "kept_sample_id": ladder_match,
            })
            continue
        kept.append(row)
        seen_targets[target_key] = sample_id
        if interaction_key:
            seen_interactions[interaction_key] = sample_id
        seen_context_targets.setdefault(group_key, []).append((target_tokens, sample_id))
    return sorted(kept, key=lambda row: str(row.get("sample_id"))), removed


def context_sufficiency(
    context_turns: Sequence[dict[str, Any]],
    target_turn: dict[str, Any],
) -> dict[str, Any]:
    """Deterministic conservative checker using selected context and target only."""
    visible_context = [
        turn for turn in context_turns
        if turn.get("speaker_state") != SpeakerState.AMBIGUOUS_SPEAKER.value
        and not is_non_conversational_sentinel(turn.get("resolved_text"))
    ]
    target_text = str(target_turn.get("resolved_text") or "").strip()
    if not target_text:
        state, reason = ContextSufficiencyState.CONTEXT_INSUFFICIENT, "TARGET_TEXT_MISSING"
    elif not visible_context:
        state, reason = ContextSufficiencyState.CONTEXT_INSUFFICIENT, "NO_VISIBLE_CONTEXT"
    elif not any(turn.get("role") == "user" for turn in visible_context):
        state, reason = ContextSufficiencyState.CONTEXT_AMBIGUOUS, "NO_CONFIRMED_NON_TARGET_CONTEXT"
    else:
        last_end = float(visible_context[-1]["end"])
        target_start = float(target_turn["start"])
        if target_start - last_end > 8.0:
            state, reason = ContextSufficiencyState.CONTEXT_AMBIGUOUS, "LARGE_PRE_TARGET_GAP"
        else:
            state, reason = ContextSufficiencyState.CONTEXT_SUFFICIENT, "CONTIGUOUS_CONFIRMED_INTERACTION"
    return {
        "state": state.value,
        "checker": "deterministic-minimal-context-v2",
        "selected_context_turn_ids": [str(turn["audio_turn_id"]) for turn in visible_context],
        "selected_target_turn_id": str(target_turn["audio_turn_id"]),
        "visible_context_only": True,
        "hidden_history_accessed": False,
        "reason": reason,
    }


def select_minimal_context(
    candidates: Sequence[dict[str, Any]],
    target_turn: dict[str, Any],
    *,
    required_turn_ids: Optional[set[str]] = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return the first sufficient contiguous suffix, expanding backwards."""
    required = {str(value) for value in (required_turn_ids or set())}
    eligible = [
        turn for turn in sorted(candidates, key=lambda item: (float(item["start"]), str(item["audio_turn_id"])))
        if float(turn["end"]) <= float(target_turn["start"]) + 1e-6
        and turn.get("speaker_state") != SpeakerState.AMBIGUOUS_SPEAKER.value
        and not is_non_conversational_sentinel(turn.get("resolved_text"))
    ]
    last_result = context_sufficiency([], target_turn)
    for size in range(1, len(eligible) + 1):
        selected = eligible[-size:]
        ids = {str(turn["audio_turn_id"]) for turn in selected}
        if required and not required.issubset(ids):
            continue
        result = context_sufficiency(selected, target_turn)
        last_result = result
        if result["state"] == ContextSufficiencyState.CONTEXT_SUFFICIENT.value:
            result["candidate_turn_count"] = len(eligible)
            result["selected_turn_count"] = len(selected)
            result["minimality_checked_prefix_sizes"] = size
            return selected, result
    last_result["candidate_turn_count"] = len(eligible)
    last_result["selected_turn_count"] = 0
    last_result["minimality_checked_prefix_sizes"] = len(eligible)
    return [], last_result


@dataclass(frozen=True)
class QuarantineTrace:
    sample_id: str
    was_baseline_materialized: bool
    was_baseline_quarantined: bool
    baseline_reason: Optional[str]
    failure_stage: str
    failure_reason: str
    target_interval: Optional[dict[str, Any]]
    containing_window_ids: tuple[str, ...] = ()
    intersecting_window_ids: tuple[str, ...] = ()
    alignment_available: bool = False
    target_activity_available: bool = False
    diarization_available: bool = False
    identity_evidence_available: bool = False
    old_new_disagreement_state: Optional[str] = None
    explicit_contradiction: bool = False
    local_reasr_attempted: bool = False
    local_reasr_result: Optional[str] = None
    resolution_attempts: tuple[str, ...] = ()
    final_status: str = "QUARANTINED"

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "was_baseline_materialized": self.was_baseline_materialized,
            "was_baseline_quarantined": self.was_baseline_quarantined,
            "baseline_reason": self.baseline_reason,
            "failure_stage": self.failure_stage,
            "failure_reason": self.failure_reason,
            "target_interval": self.target_interval,
            "containing_window_ids": list(self.containing_window_ids),
            "intersecting_window_ids": list(self.intersecting_window_ids),
            "alignment_available": self.alignment_available,
            "target_activity_available": self.target_activity_available,
            "diarization_available": self.diarization_available,
            "identity_evidence_available": self.identity_evidence_available,
            "old_new_disagreement_state": self.old_new_disagreement_state,
            "explicit_contradiction": self.explicit_contradiction,
            "local_reasr_attempted": self.local_reasr_attempted,
            "local_reasr_result": self.local_reasr_result,
            "resolution_attempts": list(self.resolution_attempts),
            "final_status": self.final_status,
        }
