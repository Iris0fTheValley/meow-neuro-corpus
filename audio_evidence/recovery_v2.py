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
from collections import Counter
from difflib import SequenceMatcher
from enum import Enum
import hashlib
import json
import os
import re
import time
from bisect import bisect_left
from typing import Any, Callable, Iterable, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


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
    BOUNDARY_ECHO = "BOUNDARY_ECHO"
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


class ContextRelation(str, Enum):
    """Direction-aware relation between visible context and frozen target."""

    TARGET_RESPONDS_TO_CONTEXT = "TARGET_RESPONDS_TO_CONTEXT"
    CONTEXT_RESPONDS_TO_TARGET = "CONTEXT_RESPONDS_TO_TARGET"
    RELATED_BUT_NOT_RESPONSE = "RELATED_BUT_NOT_RESPONSE"
    UNRELATED = "UNRELATED"
    MISSING_IMMEDIATE_TRIGGER = "MISSING_IMMEDIATE_TRIGGER"


# These revisions are part of the semantic judge authority.  A cached result
# from a different value is not compatible, even when its visible text is the
# same.  Keep these separate from the audio evidence schema revision because a
# judge policy change must invalidate only judge decisions.
CONTEXT_JUDGE_SCHEMA_REVISION = "context-judge-schema-v2"
CONTEXT_JUDGE_PROMPT_REVISION = "context-judge-prompt-v5-static-rules"
CONTEXT_JUDGE_POLICY_REVISION = "context-judge-policy-v4-immediate-trigger"


RECOVERY_ALLOWED_RECONCILIATION_STATES = {
    ReconciliationState.MATCH_EXISTING.value,
    ReconciliationState.EXTEND_EXISTING.value,
    ReconciliationState.SPLIT_EXISTING.value,
    ReconciliationState.MERGE_EXISTING.value,
    ReconciliationState.TRUE_NEW.value,
}


def reconciliation_recovery_eligible(
    state: Any,
    *,
    boundary_validated: bool = True,
    role_validated: bool = True,
) -> bool:
    """Whether a reconciled turn may be used to recover a previously absent context.

    Baseline monotonic retention is deliberately handled separately.  In
    particular, an ambiguous audio hypothesis cannot rescue a sample that was
    previously context-incomplete.
    """
    value = str(state or "")
    if value not in RECOVERY_ALLOWED_RECONCILIATION_STATES:
        return False
    if value in {
        ReconciliationState.EXTEND_EXISTING.value,
        ReconciliationState.SPLIT_EXISTING.value,
        ReconciliationState.MERGE_EXISTING.value,
    }:
        return bool(boundary_validated and role_validated)
    return True


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


def boundary_echo_match(
    candidate_span: dict[str, Any],
    preceding_old_turns: Sequence[dict[str, Any]],
    *,
    max_candidate_tokens: int = 8,
    max_tail_tokens: int = 12,
    boundary_tolerance: float = 0.15,
) -> Optional[dict[str, Any]]:
    """Detect a new span that is only the tail of the preceding old turn.

    A short ASR span immediately after an old turn is not evidence of a new
    utterance when it exactly repeats the old turn's final words.  This is
    intentionally text/boundary evidence only; it never assigns a speaker.
    Longer spans are accepted only when their entire normalized token sequence
    is a suffix of the old turn.  The returned record is suitable for the
    reconciliation ledger and makes the rejection auditable.
    """
    candidate_text = str(candidate_span.get("text") or "").strip()
    candidate_tokens = normalized_tokens(candidate_text)
    if not candidate_tokens or len(candidate_tokens) > max_candidate_tokens:
        return None
    candidate_start = float(candidate_span.get("start") or 0.0)
    ordered = sorted(
        (
            turn for turn in preceding_old_turns
            if float(turn.get("end") or 0.0) <= candidate_start + boundary_tolerance
        ),
        key=lambda turn: (float(turn.get("end") or 0.0), str(turn.get("turn_id") or "")),
        reverse=True,
    )
    for old_turn in ordered:
        old_tokens = normalized_tokens(old_turn.get("text") or old_turn.get("resolved_text"))
        if not old_tokens:
            continue
        tail = old_tokens[-max_tail_tokens:]
        if len(candidate_tokens) <= len(tail) and tail[-len(candidate_tokens):] == candidate_tokens:
            return {
                "old_turn_id": str(old_turn.get("turn_id") or old_turn.get("audio_turn_id") or ""),
                "old_text": str(old_turn.get("text") or old_turn.get("resolved_text") or "").strip(),
                "candidate_text": candidate_text,
                "candidate_tokens": candidate_tokens,
                "match": "EXACT_NORMALIZED_SUFFIX",
                "boundary_gap_seconds": round(candidate_start - float(old_turn.get("end") or 0.0), 6),
            }
    return None


def _shared_scoped_cluster(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_keys = set(_scoped_keys(left.get("speaker_cluster_keys") or left.get("speaker_cluster")))
    right_keys = set(_scoped_keys(right.get("speaker_cluster_keys") or right.get("speaker_cluster")))
    return bool(left_keys & right_keys)


def merge_obvious_audio_fragments(
    spans: Sequence[dict[str, Any]],
    *,
    max_gap: float = 1.5,
) -> list[dict[str, Any]]:
    """Merge adjacent same-speaker ASR fragments before TRUE_NEW admission.

    Fragments are merged only when they share a window-scoped cluster and the
    boundary looks like a continuation (unfinished punctuation, a continuation
    conjunction, or a very short right fragment).  This prevents the topology
    classifier from deciding on individual ASR fragments.
    """
    ordered = sorted(spans, key=lambda item: (float(item.get("start") or 0.0), str(item.get("candidate_span_id") or "")))
    merged: list[dict[str, Any]] = []
    continuation_words = {
        "and", "or", "but", "so", "because", "to", "of", "for", "with", "that", "which", "then", "than",
    }
    for span in ordered:
        if not merged:
            merged.append(dict(span))
            continue
        previous = merged[-1]
        gap = float(span.get("start") or 0.0) - float(previous.get("end") or 0.0)
        left_text = str(previous.get("text") or "").strip()
        right_text = str(span.get("text") or "").strip()
        right_tokens = normalized_tokens(right_text)
        continuation = (
            gap <= max_gap
            and not previous.get("generic_overlap")
            and not span.get("generic_overlap")
            and _shared_scoped_cluster(previous, span)
            and (
                not re.search(r"[.!?]$", left_text)
                or (right_tokens and right_tokens[0] in continuation_words)
                or len(right_tokens) <= 2
            )
        )
        if not continuation:
            merged.append(dict(span))
            continue
        tokens = list(previous.get("tokens") or []) + list(span.get("tokens") or [])
        text = " ".join(value for value in (left_text, right_text) if value).strip()
        window_ids = sorted(set(previous.get("source_window_ids") or []) | set(span.get("source_window_ids") or []))
        cluster_keys = sorted(set(previous.get("speaker_cluster_keys") or []) | set(span.get("speaker_cluster_keys") or []))
        merged[-1] = {
            **previous,
            "candidate_span_id": "audio:candidate:merged:" + canonical_sha256({
                "left": previous.get("candidate_span_id"),
                "right": span.get("candidate_span_id"),
                "text": normalized_text(text),
            })[:16],
            "end": float(span.get("end") or previous.get("end") or 0.0),
            "text": text,
            "tokens": tokens,
            "source_window_ids": window_ids,
            "speaker_cluster_keys": cluster_keys,
            "speaker_cluster": cluster_keys[0] if len(cluster_keys) == 1 else None,
            "fragment_merge_count": int(previous.get("fragment_merge_count") or 0) + 1,
            "fragment_merge_provenance": [
                *(previous.get("fragment_merge_provenance") or [previous.get("candidate_span_id")]),
                span.get("candidate_span_id"),
            ],
        }
    return merged


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


def annotate_tokens_with_diarization(
    tokens: Iterable[dict[str, Any]],
    diarization: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach window-scoped generic boundary clusters to timed tokens.

    ECAPA cluster labels are local to a bounded window.  A label such as
    ``ECAPA_CLUSTER_00`` is therefore deliberately never emitted as an
    authority-bearing identifier by itself.  Tokens retain every overlapping
    ``window_id::local_cluster`` observation so downstream topology can demand
    agreement where a token is reconstructed from more than one window.
    """
    segments = sorted(
        [dict(item) for item in diarization],
        key=lambda item: (float(item.get("start", 0)), float(item.get("end", 0))),
    )
    active: list[dict[str, Any]] = []
    cursor = 0
    annotated = []
    for raw in sorted(tokens, key=lambda item: (float(item.get("start", 0)), float(item.get("end", 0)))):
        token = dict(raw)
        start, end = float(token.get("start", 0)), float(token.get("end", 0))
        token_window_ids = {str(value) for value in (token.get("source_window_ids") or []) if str(value)}
        while cursor < len(segments) and float(segments[cursor].get("start", 0)) < end:
            active.append(segments[cursor])
            cursor += 1
        active = [item for item in active if float(item.get("end", 0)) > start]
        scores: dict[str, float] = {}
        scoped_scores: dict[str, float] = {}
        overlap_seen = False
        for item in active:
            item_window_ids = {str(value) for value in (item.get("source_window_ids") or []) if str(value)}
            if token_window_ids and item_window_ids and not (token_window_ids & item_window_ids):
                continue
            amount = interval_overlap(start, end, float(item.get("start", 0)), float(item.get("end", 0)))
            if amount <= 0:
                continue
            cluster = str(item.get("speaker_cluster") or "UNKNOWN")
            scores[cluster] = scores.get(cluster, 0.0) + amount
            window_ids = [str(value) for value in (item.get("source_window_ids") or []) if str(value)]
            for window_id in window_ids:
                scoped = scoped_cluster_key(window_id, cluster)
                scoped_scores[scoped] = scoped_scores.get(scoped, 0.0) + amount
            overlap_seen = overlap_seen or bool(item.get("overlap"))
        ordered = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
        cluster = None
        if ordered and not (len(ordered) > 1 and ordered[1][1] >= ordered[0][1] * 0.8):
            cluster = ordered[0][0]
        token["boundary_speaker_cluster"] = cluster
        token["boundary_speaker_cluster_keys"] = sorted(scoped_scores)
        token["generic_overlap"] = overlap_seen
        annotated.append(token)
    return annotated


def scoped_cluster_key(window_id: Any, local_cluster_id: Any) -> str:
    """Canonical key for a diarization label that is only valid in one window."""
    return "%s::%s" % (str(window_id), str(local_cluster_id))


def _scoped_keys(value: Any) -> list[str]:
    """Accept only explicit window-scoped cluster identifiers.

    Bare ECAPA labels are intentionally discarded rather than upgraded through
    a recording-wide lookup.  This makes an absent scope fail closed.
    """
    values = value if isinstance(value, (list, tuple, set)) else [value]
    return sorted({str(item) for item in values if "::" in str(item)})


def build_cluster_speaker_anchors(
    diarization: Iterable[dict[str, Any]],
    resolved_old_turns: Iterable[dict[str, Any]],
    *,
    minimum_seconds: float = 1.0,
    dominance_ratio: float = 0.9,
) -> dict[str, dict[str, Any]]:
    """Anchor generic clusters to frozen speaker states through timed overlap.

    Generic diarization never creates identity. It can only carry an already
    frozen identity state within the *same bounded window* after accumulating
    dominant, non-conflicting overlap with turns resolved directly by frozen
    identity authority. Local labels must not cross window boundaries.
    """
    votes: dict[str, dict[str, float]] = {}
    supporting_turns: dict[str, dict[str, set[str]]] = {}
    trusted = [
        turn for turn in resolved_old_turns
        if (turn.get("speaker_resolution_state") or turn.get("speaker_state")) in {
            SpeakerState.CONFIRMED_TARGET.value,
            SpeakerState.CONFIRMED_NON_TARGET.value,
        }
        and (turn.get("speaker_resolution") or {}).get("resolution_basis") == "FROZEN_IDENTITY_AUTHORITY"
    ]
    for segment in diarization:
        cluster = str(segment.get("speaker_cluster") or "")
        window_ids = [str(value) for value in (segment.get("source_window_ids") or []) if str(value)]
        if not cluster or not window_ids or segment.get("overlap"):
            continue
        start, end = float(segment.get("start", 0)), float(segment.get("end", 0))
        if end <= start:
            continue
        for turn in trusted:
            amount = interval_overlap(start, end, float(turn["start"]), float(turn["end"]))
            if amount <= 0:
                continue
            state = str(turn.get("speaker_resolution_state") or turn.get("speaker_state"))
            for window_id in window_ids:
                cluster_key = scoped_cluster_key(window_id, cluster)
                votes.setdefault(cluster_key, {}).setdefault(state, 0.0)
                votes[cluster_key][state] += amount
                supporting_turns.setdefault(cluster_key, {}).setdefault(state, set()).add(str(turn["audio_turn_id"]))
    result: dict[str, dict[str, Any]] = {}
    for cluster, state_votes in votes.items():
        ordered = sorted(state_votes.items(), key=lambda pair: (-pair[1], pair[0]))
        winner, winner_seconds = ordered[0]
        total = sum(state_votes.values())
        ratio = winner_seconds / max(total, 1e-9)
        confirmed = winner_seconds >= minimum_seconds and ratio >= dominance_ratio
        result[cluster] = {
            "speaker_state": winner if confirmed else SpeakerState.AMBIGUOUS_SPEAKER.value,
            "role": (
                "assistant" if confirmed and winner == SpeakerState.CONFIRMED_TARGET.value
                else "user" if confirmed and winner == SpeakerState.CONFIRMED_NON_TARGET.value
                else None
            ),
            "resolution_basis": "FROZEN_IDENTITY_ANCHORED_DIARIZATION_CONTINUITY" if confirmed else "CONFLICTING_OR_INSUFFICIENT_CLUSTER_ANCHORS",
            "cluster": cluster,
            "cluster_scope": cluster.split("::", 1)[0],
            "local_cluster_id": cluster.split("::", 1)[1],
            "window_local_authority": True,
            "vote_seconds": {key: round(value, 6) for key, value in state_votes.items()},
            "dominance_ratio": round(ratio, 6),
            "supporting_old_turn_ids": sorted(supporting_turns.get(cluster, {}).get(winner, set())),
            "generic_diarization_created_identity": False,
        }
    return result


def resolve_anchored_cluster_speaker(
    cluster: Any,
    anchors: dict[str, dict[str, Any]],
    *,
    target_activity_available: bool = False,
) -> dict[str, Any]:
    keys = _scoped_keys(cluster)
    unscoped = bool(cluster) and not keys
    anchor_entries = [anchors.get(key) for key in keys]
    confirmed = {
        SpeakerState.CONFIRMED_TARGET.value,
        SpeakerState.CONFIRMED_NON_TARGET.value,
    }
    states = [str((entry or {}).get("speaker_state") or "") for entry in anchor_entries]
    all_scoped_confirmed = bool(keys) and len(anchor_entries) == len(keys) and all(state in confirmed for state in states)
    conflict = len(set(states)) > 1 or any(entry is None for entry in anchor_entries)
    if all_scoped_confirmed and len(set(states)) == 1:
        anchor = anchor_entries[0]
        return {
            "speaker_state": anchor["speaker_state"],
            "role": anchor["role"],
            "resolution_basis": anchor["resolution_basis"],
            "evidence": {
                "frozen_identity": None,
                "identity_evidence_available": True,
                "target_activity_available": bool(target_activity_available),
                "generic_diarization_available": True,
                "activity_used_for_identity": False,
                "generic_diarization_used_for_identity": False,
                "anchored_cluster_continuity_used": True,
                "window_local_cluster_authority": True,
                "scoped_cluster_keys": keys,
                "cluster_anchor_conflict": False,
                "cluster_anchors": anchor_entries,
            },
        }
    result = resolve_speaker_state(
        None,
        identity_evidence_available=False,
        target_activity_available=target_activity_available,
        generic_diarization_available=bool(keys),
    )
    result["evidence"].update({
        "anchored_cluster_continuity_used": False,
        "window_local_cluster_authority": bool(keys),
        "scoped_cluster_keys": keys,
        "cluster_anchor_conflict": conflict,
        "unscoped_cluster_authority": unscoped,
        "cluster_anchors": anchor_entries,
    })
    if conflict:
        result["resolution_basis"] = "CONFLICTING_WINDOW_LOCAL_CLUSTER_ANCHORS"
    elif keys:
        result["resolution_basis"] = "INSUFFICIENT_WINDOW_LOCAL_CLUSTER_ANCHORS"
    return result


def split_existing_turn(
    old_turn: dict[str, Any],
    assigned_tokens: Sequence[dict[str, Any]],
    cluster_anchors: dict[str, dict[str, Any]],
    *,
    is_frozen_target: bool = False,
    minimum_tokens_per_child: int = 2,
) -> list[dict[str, Any]]:
    """Create role-resolved children when timed tokens prove a speaker split."""
    if is_frozen_target:
        return []
    groups: list[list[dict[str, Any]]] = []
    for token in sorted(assigned_tokens, key=lambda item: (float(item["start"]), float(item["end"]))):
        cluster_keys = _scoped_keys(token.get("boundary_speaker_cluster_keys"))
        if not cluster_keys:
            continue
        if not groups:
            groups.append([])
        elif _split_token_relation(groups[-1][-1], token) == "different":
            groups.append([])
        groups[-1].append(token)
    if len(groups) < 2 or any(len(group) < minimum_tokens_per_child for group in groups):
        return []
    resolutions = [
        resolve_anchored_cluster_speaker(_group_cluster_keys(group), cluster_anchors)
        for group in groups
    ]
    states = [resolution["speaker_state"] for resolution in resolutions]
    if any(state == SpeakerState.AMBIGUOUS_SPEAKER.value for state in states) or len(set(states)) < 2:
        return []
    children = []
    for index, (group, resolution) in enumerate(zip(groups, resolutions)):
        text = " ".join(str(token.get("text") or "").strip() for token in group).strip()
        if not text or is_non_conversational_sentinel(text):
            return []
        children.append({
            "audio_turn_id": f"reconciled:split:{old_turn['turn_id']}:{index}",
            "start": float(group[0]["start"]),
            "end": float(group[-1]["end"]),
            "resolved_text": text,
            "old_turn_ids": [str(old_turn["turn_id"])],
            "reconciliation_state": ReconciliationState.SPLIT_EXISTING.value,
            "speaker_cluster": _group_cluster_keys(group)[0] if len(_group_cluster_keys(group)) == 1 else None,
            "speaker_cluster_keys": _group_cluster_keys(group),
            "speaker_resolution": resolution,
            "speaker_resolution_state": resolution["speaker_state"],
            "role": resolution["role"],
            "boundary_validated": True,
            "role_validated": True,
            "source_window_ids": sorted({
                str(window_id) for token in group for window_id in (token.get("source_window_ids") or [])
            }),
            "source_spans": [
                {
                    "start": float(token["start"]), "end": float(token["end"]), "text": token.get("text"),
                    "token_id": token.get("token_id"),
                    "source_window_ids": list(token.get("source_window_ids") or []),
                }
                for token in group
            ],
        })
    return children


def merge_existing_turns(
    span: dict[str, Any],
    old_turns: Sequence[dict[str, Any]],
    cluster_anchors: dict[str, dict[str, Any]],
    *,
    frozen_target_turn_ids: set[str],
) -> Optional[dict[str, Any]]:
    """Materialize a proven same-speaker merge as a replacement topology node."""
    ordered = sorted(old_turns, key=lambda turn: (float(turn["start"]), str(turn["turn_id"])))
    if len(ordered) < 2 or any(str(turn["turn_id"]) in frozen_target_turn_ids for turn in ordered):
        return None
    resolution = resolve_anchored_cluster_speaker(
        span.get("speaker_cluster_keys") or span.get("speaker_cluster"), cluster_anchors
    )
    if resolution["speaker_state"] == SpeakerState.AMBIGUOUS_SPEAKER.value or span.get("generic_overlap"):
        return None
    if text_similarity(" ".join(str(turn.get("text") or "") for turn in ordered), span.get("text")) < 0.6:
        return None
    turn_id = "reconciled:merge:" + canonical_sha256({
        "recording_id": span.get("recording_id"),
        "old_turn_ids": [str(turn["turn_id"]) for turn in ordered],
        "cluster": span.get("speaker_cluster"),
    })[:16]
    return {
        "audio_turn_id": turn_id,
        "start": float(ordered[0]["start"]),
        "end": float(ordered[-1]["end"]),
        "resolved_text": " ".join(str(turn.get("text") or "").strip() for turn in ordered).strip(),
        "old_turn_ids": [str(turn["turn_id"]) for turn in ordered],
        "reconciliation_state": ReconciliationState.MERGE_EXISTING.value,
        "speaker_cluster": span.get("speaker_cluster"),
        "speaker_cluster_keys": _scoped_keys(span.get("speaker_cluster_keys") or span.get("speaker_cluster")),
        "speaker_resolution": resolution,
        "speaker_resolution_state": resolution["speaker_state"],
        "role": resolution["role"],
        "boundary_validated": True,
        "role_validated": True,
        "source_spans": [
            {
                "start": float(token["start"]), "end": float(token["end"]), "text": token.get("text"),
                "token_id": token.get("token_id"),
                "source_window_ids": list(token.get("source_window_ids") or []),
            }
            for token in span.get("tokens") or []
        ],
    }


def _split_token_relation(left: dict[str, Any], right: dict[str, Any]) -> str:
    left_map = {key.split("::", 1)[0]: key.split("::", 1)[1] for key in _scoped_keys(left.get("boundary_speaker_cluster_keys"))}
    right_map = {key.split("::", 1)[0]: key.split("::", 1)[1] for key in _scoped_keys(right.get("boundary_speaker_cluster_keys"))}
    shared = set(left_map) & set(right_map)
    if not shared:
        return "unknown"
    return "same" if all(left_map[key] == right_map[key] for key in shared) else "different"


def _group_cluster_keys(group: Sequence[dict[str, Any]]) -> list[str]:
    by_window: dict[str, Counter[str]] = {}
    for token in group:
        for key in _scoped_keys(token.get("boundary_speaker_cluster_keys")):
            window_id, local_cluster = key.split("::", 1)
            by_window.setdefault(window_id, Counter())[local_cluster] += 1
    return sorted(scoped_cluster_key(window_id, votes.most_common(1)[0][0]) for window_id, votes in by_window.items())


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
    *,
    preceding_old_turns: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    text = str(span.get("text") or "").strip()
    old = sorted(overlapping_old_turns, key=lambda turn: (float(turn["start"]), str(turn["turn_id"])))
    if not old:
        echo = boundary_echo_match(span, preceding_old_turns)
        if echo:
            return {
                "candidate_span_id": span.get("candidate_span_id"),
                "reconciliation_state": ReconciliationState.BOUNDARY_ECHO.value,
                "old_turn_ids": [echo["old_turn_id"]],
                "old_text": echo["old_text"],
                "new_text": text or None,
                "chosen_text": None,
                "chosen_text_authority": None,
                "similarity": round(text_similarity(echo["old_text"], text), 6),
                "confidence": 0.98,
                "resolution_reason": "OLD_TURN_BOUNDARY_ECHO_NOT_TRUE_NEW",
                "boundary_echo": echo,
                "source_spans": [
                    {"start": float(token["start"]), "end": float(token["end"]), "text": token.get("text")}
                    for token in span.get("tokens") or []
                ],
            }
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
    # ``diarization`` is intentionally no longer aggregated by bare cluster
    # name. Tokens have already been annotated with window-local observations.
    del diarization

    def key_map(token: dict[str, Any]) -> dict[str, str]:
        result: dict[str, str] = {}
        for key in _scoped_keys(token.get("boundary_speaker_cluster_keys")):
            window_id, local_cluster = key.split("::", 1)
            result[window_id] = local_cluster
        return result

    def token_relation(left: dict[str, Any], right: dict[str, Any]) -> str:
        """same/different/unknown using only shared window-local labels."""
        left_map, right_map = key_map(left), key_map(right)
        shared = set(left_map) & set(right_map)
        if not shared:
            return "unknown"
        values = {left_map[window_id] == right_map[window_id] for window_id in shared}
        return "same" if values == {True} else "different"

    def span_cluster_keys(group: Sequence[dict[str, Any]]) -> list[str]:
        # A local window can contribute at most one dominant label to a span.
        votes: dict[str, Counter[str]] = {}
        for item in group:
            for window_id, local_cluster in key_map(item).items():
                votes.setdefault(window_id, Counter())[local_cluster] += 1
        return sorted(
            scoped_cluster_key(window_id, counter.most_common(1)[0][0])
            for window_id, counter in votes.items()
            if len(counter) == 1 or counter.most_common(2)[0][1] > counter.most_common(2)[1][1]
        )

    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for token in sorted(tokens, key=lambda item: (float(item["start"]), float(item["end"]))):
        start, end = float(token["start"]), float(token["end"])
        item = dict(token)
        hard_break = bool(
            current
            and (
                start - float(current[-1]["end"]) > max_gap
                or item.get("generic_overlap")
                or current[-1].get("generic_overlap")
                or token_relation(current[-1], item) == "different"
            )
        )
        if hard_break:
            groups.append(current)
            current = []
        current.append(item)
    if current:
        groups.append(current)
    spans = []
    for index, group in enumerate(groups):
        text = " ".join(str(token.get("text") or "").strip() for token in group).strip()
        if not text:
            continue
        start, end = float(group[0]["start"]), float(group[-1]["end"])
        cluster_keys = span_cluster_keys(group)
        source_window_ids = sorted({
            str(window_id)
            for token in group
            for window_id in (token.get("source_window_ids") or [])
        })
        # A cluster label has no recording-global meaning. Retain a readable
        # compatibility field only when exactly one scoped key supports span.
        cluster = cluster_keys[0] if len(cluster_keys) == 1 else None
        spans.append({
            "candidate_span_id": "audio:candidate:%s:%s" % (
                recording_id,
                canonical_sha256({"windows": source_window_ids, "start": round(start, 3), "end": round(end, 3), "text": normalized_text(text)})[:16],
            ),
            "recording_id": recording_id,
            "start": start,
            "end": end,
            "text": text,
            "tokens": group,
            "speaker_cluster": cluster,
            "speaker_cluster_keys": cluster_keys,
            "generic_overlap": any(bool(token.get("generic_overlap")) for token in group),
            "source_window_ids": source_window_ids,
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
    if is_non_conversational_sentinel(old_value):
        state = TargetResolutionState.UNRESOLVED
        reason = "NON_CONVERSATIONAL_SENTINEL_TARGET"
        explicit = True
    elif not old_value or not identity_confirmed_target:
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
    """Conservative visible-only checker with an explicit response direction.

    This function is a structural fallback and test helper.  Production
    recovery uses the directional semantic judge below; nevertheless, the
    fallback must never call temporal adjacency sufficient by itself.
    """
    visible_context = [
        turn for turn in context_turns
        if turn.get("speaker_state") != SpeakerState.AMBIGUOUS_SPEAKER.value
        and not is_non_conversational_sentinel(turn.get("resolved_text"))
    ]
    target_text = str(target_turn.get("resolved_text") or "").strip()
    relation = ContextRelation.UNRELATED.value
    immediate_trigger_turn_id = None
    if not target_text:
        state, reason = ContextSufficiencyState.CONTEXT_INSUFFICIENT, "TARGET_TEXT_MISSING"
        relation = ContextRelation.MISSING_IMMEDIATE_TRIGGER.value
    elif not visible_context:
        state, reason = ContextSufficiencyState.CONTEXT_INSUFFICIENT, "NO_VISIBLE_CONTEXT"
        relation = ContextRelation.MISSING_IMMEDIATE_TRIGGER.value
    elif not any(turn.get("role") == "user" for turn in visible_context):
        state, reason = ContextSufficiencyState.CONTEXT_AMBIGUOUS, "NO_CONFIRMED_NON_TARGET_CONTEXT"
        relation = ContextRelation.CONTEXT_RESPONDS_TO_TARGET.value
    else:
        last_end = float(visible_context[-1]["end"])
        target_start = float(target_turn["start"])
        if target_start - last_end > 8.0:
            state, reason = ContextSufficiencyState.CONTEXT_AMBIGUOUS, "LARGE_PRE_TARGET_GAP"
            relation = ContextRelation.MISSING_IMMEDIATE_TRIGGER.value
        else:
            last_user = next((turn for turn in reversed(visible_context) if turn.get("role") == "user"), None)
            trigger_tokens = normalized_tokens((last_user or {}).get("resolved_text"))
            target_tokens = normalized_tokens(target_text)
            overlap = set(trigger_tokens) & set(target_tokens)
            response_like = (
                len(target_tokens) <= 4
                or (target_tokens and target_tokens[0] in {"it", "they", "that", "this", "yes", "no", "sure", "probably", "maybe"})
                or bool(overlap)
            )
            if response_like and last_user:
                state, reason = ContextSufficiencyState.CONTEXT_SUFFICIENT, "VISIBLE_IMMEDIATE_USER_TRIGGER"
                relation = ContextRelation.TARGET_RESPONDS_TO_CONTEXT.value
                immediate_trigger_turn_id = str(last_user.get("audio_turn_id") or last_user.get("turn_id") or "")
            else:
                state, reason = ContextSufficiencyState.CONTEXT_AMBIGUOUS, "MISSING_IMMEDIATE_TRIGGER"
                relation = ContextRelation.MISSING_IMMEDIATE_TRIGGER.value
    return {
        "state": state.value,
        "relation": relation,
        "immediate_trigger_turn_id": immediate_trigger_turn_id,
        "checker": "deterministic-minimal-context-v2",
        "selected_context_turn_ids": [str(turn["audio_turn_id"]) for turn in visible_context],
        "selected_target_turn_id": str(target_turn["audio_turn_id"]),
        "visible_context_only": True,
        "hidden_history_accessed": False,
        "reason": reason,
    }


CONTEXT_JUDGE_SYSTEM_PROMPT = (
    "You are a narrow context-sufficiency judge for an already verified frozen assistant target. "
    "Use only the selected context and frozen target shown. Do not use hidden history, future turns, "
    "identity diagnostics, metadata, or external knowledge. Decide whether the selected context makes "
    "the target a plausible, understandable response. Temporal adjacency alone is never sufficient. "
    "First classify the direction of the interaction. TARGET_RESPONDS_TO_CONTEXT means the frozen "
    "assistant target answers a visible user trigger in the selected context. CONTEXT_RESPONDS_TO_TARGET "
    "means a visible context turn is an answer to the target or an unseen later trigger. "
    "RELATED_BUT_NOT_RESPONSE means topical relation without an answer relation. UNRELATED means no "
    "grounded relation. MISSING_IMMEDIATE_TRIGGER means the target looks like an answer but the actual "
    "immediate user question/request is absent. Only TARGET_RESPONDS_TO_CONTEXT may be sufficient. "
    "Use these exact state labels: CONTEXT_SUFFICIENT, CONTEXT_INSUFFICIENT, CONTEXT_AMBIGUOUS. "
    "Use these exact relation labels: TARGET_RESPONDS_TO_CONTEXT, CONTEXT_RESPONDS_TO_TARGET, "
    "RELATED_BUT_NOT_RESPONSE, UNRELATED, MISSING_IMMEDIATE_TRIGGER. "
    "Constraints: use no evidence outside selected_context and frozen_target; do not adjudicate or "
    "rewrite the frozen target; a short time gap does not establish a response relation; return JSON only. "
    "Return exactly one compact JSON object matching this schema: {state, relation, "
    "immediate_trigger_turn_id, confidence, reason}. immediate_trigger_turn_id must be a turn_id "
    "from selected_context whenever relation is TARGET_RESPONDS_TO_CONTEXT; confidence is a number 0..1; "
    "reason is a short explanation based only on shown text."
)


def build_context_judge_payload(
    context_turns: Sequence[dict[str, Any]],
    target_turn: dict[str, Any],
) -> dict[str, Any]:
    def compact(text: Any) -> str:
        # Keep the narrow judge request bounded while preserving the exact
        # selected turn ordering and role evidence.
        return str(text or "").strip()[:600]

    return {
        "selected_context": [
            {
                "turn_id": str(turn.get("audio_turn_id") or turn.get("turn_id") or ""),
                "role": str(turn.get("role") or ""),
                "text": compact(turn.get("resolved_text")),
            }
            for turn in context_turns
        ],
        "frozen_target": {
            "role": "assistant",
            "text": compact(target_turn.get("resolved_text")),
        },
    }


def parse_json_object(text: Any) -> Optional[dict[str, Any]]:
    value = str(text or "").strip()
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", value, flags=re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None


def call_context_sufficiency_judge(
    context_turns: Sequence[dict[str, Any]],
    target_turn: dict[str, Any],
    *,
    model: str,
    endpoint: str,
    timeout_seconds: int = 180,
    api_key_env: str = "",
) -> dict[str, Any]:
    payload = build_context_judge_payload(context_turns, target_turn)
    started = time.monotonic()
    retry_count = 0
    http_status = None
    is_chat_endpoint = endpoint.rstrip("/").endswith("/chat/completions")
    if is_chat_endpoint:
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": CONTEXT_JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "temperature": 0,
            "max_tokens": 256,
            "response_format": {"type": "json_object"},
            "stream": False,
        }
    else:
        prompt = (
            "<|im_start|>system\n" + CONTEXT_JUDGE_SYSTEM_PROMPT
            + "<|im_end|>\n<|im_start|>user\n"
            + json.dumps(payload, ensure_ascii=False)
            + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
        body = {
            "model": model,
            "prompt": prompt,
            "temperature": 0,
            "max_tokens": 128,
            "stop": ["<|im_end|>"],
            "stream": False,
        }
    endpoint_host = endpoint.lower()
    if "api.stepfun.com" in endpoint_host:
        # Step 3.5 Flash exposes reasoning_content separately and can spend a
        # small completion budget on it before emitting the required JSON.
        # Keep reasoning bounded so the visible judge object is not truncated.
        if is_chat_endpoint:
            if "/step_plan/" in endpoint_host:
                # Step Plan's router can spend substantial hidden-reasoning
                # budget before emitting the JSON object.  Telemetry from the
                # first Step Plan run showed finish_reason=length at 8192;
                # the documented Step Plan channel permits a much larger
                # max_tokens value, so do not truncate those responses.
                body["max_tokens"] = 32768
            else:
                body["max_tokens"] = 8192 if model == "step-3.7-flash" else 1024
            if model in {"step-3.5-flash", "step-3.5-flash-2603", "step-3.7-flash", "step-router-v1"}:
                body["reasoning_effort"] = "low"
        env_name = api_key_env or "STEPFUN_API_KEY"
        api_key = os.environ.get(env_name, "").strip()
        if not api_key:
            return {
                "state": ContextSufficiencyState.CONTEXT_AMBIGUOUS.value,
                "relation": ContextRelation.MISSING_IMMEDIATE_TRIGGER.value,
                "immediate_trigger_turn_id": None,
                "confidence": 0.0,
                "reason": f"JUDGE_API_KEY_MISSING:{env_name}",
                "checker": "semantic-context-sufficiency-judge-v4-directional",
                "judge_valid": False,
                "judge_input_sha256": canonical_sha256(payload),
                "visible_context_only": True,
                "hidden_history_accessed": False,
                "judge_telemetry": {
                    "latency_ms": round((time.monotonic() - started) * 1000, 3),
                    "retry_count": 0,
                    "http_status": None,
                    "error_type": "API_KEY_MISSING",
                    "finish_reason": None,
                    "prompt_tokens": None,
                    "cached_input_tokens": None,
                    "completion_tokens": None,
                    "reasoning_tokens": None,
                    "invalid_judge_output": True,
                },
            }
    else:
        api_key = ""
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(
        endpoint,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        for attempt in range(5):
            try:
                with urlopen(request, timeout=timeout_seconds) as response:
                    http_status = getattr(response, "status", None)
                    if not isinstance(http_status, int):
                        http_status = None
                    response_payload = json.loads(response.read().decode("utf-8"))
                break
            except HTTPError as exc:
                http_status = exc.code
                retryable = exc.code in {408, 409, 425, 429} or 500 <= exc.code < 600
                if not retryable or attempt == 4:
                    raise
                retry_count += 1
                retry_after = None
                if exc.headers is not None:
                    try:
                        retry_after = float(exc.headers.get("Retry-After"))
                    except (TypeError, ValueError):
                        retry_after = None
                time.sleep(max(5.0, min(60.0, retry_after or (5.0 * (2.0 ** attempt)))))
            except (URLError, TimeoutError, OSError):
                if attempt == 4:
                    raise
                time.sleep(min(30.0, 5.0 * (2.0 ** attempt)))
        choice = (response_payload.get("choices") or [{}])[0]
        parsed = parse_json_object(choice.get("text") or (choice.get("message") or {}).get("content"))
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        status = f":{exc.code}" if isinstance(exc, HTTPError) else ""
        return {
            "state": ContextSufficiencyState.CONTEXT_AMBIGUOUS.value,
            "relation": ContextRelation.MISSING_IMMEDIATE_TRIGGER.value,
            "immediate_trigger_turn_id": None,
            "confidence": 0.0,
            "reason": f"JUDGE_REQUEST_FAILED:{type(exc).__name__}{status}",
            "checker": "semantic-context-sufficiency-judge-v4-directional",
            "judge_valid": False,
            "judge_input_sha256": canonical_sha256(payload),
            "visible_context_only": True,
            "hidden_history_accessed": False,
            "judge_telemetry": {
                "latency_ms": round((time.monotonic() - started) * 1000, 3),
                "retry_count": retry_count,
                "http_status": http_status,
                "error_type": type(exc).__name__,
                "finish_reason": None,
                "prompt_tokens": None,
                "cached_input_tokens": None,
                "completion_tokens": None,
                "reasoning_tokens": None,
                "invalid_judge_output": True,
            },
        }
    usage = response_payload.get("usage") or {}
    prompt_details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
    choice = (response_payload.get("choices") or [{}])[0]
    finish_reason = choice.get("finish_reason")

    def usage_int(*keys: str) -> Optional[int]:
        for key in keys:
            value = usage.get(key)
            if value is None:
                value = prompt_details.get(key) if key in prompt_details else completion_details.get(key)
            try:
                return int(value) if value is not None else None
            except (TypeError, ValueError):
                return None
        return None

    base_telemetry = {
        "latency_ms": round((time.monotonic() - started) * 1000, 3),
        "retry_count": retry_count,
        "http_status": http_status,
        "error_type": None,
        "finish_reason": finish_reason,
        "prompt_tokens": usage_int("prompt_tokens", "input_tokens"),
        "cached_input_tokens": (
            usage_int("cached_tokens", "cache_read_input_tokens", "cached_input_tokens")
        ),
        "completion_tokens": usage_int("completion_tokens", "output_tokens"),
        "reasoning_tokens": usage_int("reasoning_tokens"),
        "invalid_judge_output": False,
    }
    state = str((parsed or {}).get("state") or "").upper()
    relation = str((parsed or {}).get("relation") or "").upper()
    immediate_trigger_turn_id = str((parsed or {}).get("immediate_trigger_turn_id") or "") or None
    try:
        confidence = float((parsed or {}).get("confidence"))
    except (TypeError, ValueError):
        confidence = -1.0
    valid = (
        state in {item.value for item in ContextSufficiencyState}
        and relation in {item.value for item in ContextRelation}
        and 0.0 <= confidence <= 1.0
    )
    if relation != ContextRelation.TARGET_RESPONDS_TO_CONTEXT.value:
        state = ContextSufficiencyState.CONTEXT_INSUFFICIENT.value
    if relation == ContextRelation.TARGET_RESPONDS_TO_CONTEXT.value and not immediate_trigger_turn_id:
        valid = False
    base_telemetry["invalid_judge_output"] = not valid
    return {
        "state": state if valid else ContextSufficiencyState.CONTEXT_AMBIGUOUS.value,
        "relation": relation if valid else ContextRelation.MISSING_IMMEDIATE_TRIGGER.value,
        "immediate_trigger_turn_id": immediate_trigger_turn_id if valid else None,
        "confidence": confidence if valid else 0.0,
        "reason": str((parsed or {}).get("reason") or "INVALID_JUDGE_OUTPUT")[:500],
        "checker": "semantic-context-sufficiency-judge-v4-directional",
        "judge_valid": valid,
        "judge_model": model,
        "judge_input_sha256": canonical_sha256(payload),
        "selected_context_turn_ids": [str(turn["audio_turn_id"]) for turn in context_turns],
        "selected_target_turn_id": str(target_turn["audio_turn_id"]),
        "visible_context_only": True,
        "hidden_history_accessed": False,
        "judge_telemetry": base_telemetry,
    }


def context_judge_cache_key(
    context_turns: Sequence[dict[str, Any]],
    target_turn: dict[str, Any],
    *,
    judge_model: str,
    judge_model_revision: str,
    prompt_revision: str = CONTEXT_JUDGE_PROMPT_REVISION,
    policy_revision: str = CONTEXT_JUDGE_POLICY_REVISION,
    schema_revision: str = CONTEXT_JUDGE_SCHEMA_REVISION,
) -> str:
    """Build the versioned, normalized cache key for a context judgement."""
    normalized_context = [
        {
            "role": str(turn.get("role") or ""),
            "text": normalized_text(turn.get("resolved_text") or turn.get("text")),
        }
        for turn in context_turns
    ]
    payload = {
        "normalized_context": normalized_context,
        "normalized_target": {
            "role": "assistant",
            "text": normalized_text(target_turn.get("resolved_text") or target_turn.get("text")),
        },
        "judge_model": str(judge_model),
        "judge_model_revision": str(judge_model_revision),
        "prompt_revision": str(prompt_revision),
        "policy_revision": str(policy_revision),
        "schema_revision": str(schema_revision),
    }
    return canonical_sha256(payload)


def select_minimal_context(
    candidates: Sequence[dict[str, Any]],
    target_turn: dict[str, Any],
    *,
    required_turn_ids: Optional[set[str]] = None,
    judge: Optional[Callable[[Sequence[dict[str, Any]], dict[str, Any]], dict[str, Any]]] = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return the first sufficient contiguous suffix, expanding backwards.

    Candidates are considered from the target backwards.  A suffix is accepted
    only for the direct response direction; related, reverse, and missing
    immediate-trigger judgements continue the search or fail closed.
    """
    required = {str(value) for value in (required_turn_ids or set())}
    eligible = [
        turn for turn in sorted(candidates, key=lambda item: (float(item["start"]), str(item["audio_turn_id"])))
        if float(turn["end"]) <= float(target_turn["start"]) + 1e-6
        and turn.get("speaker_state") != SpeakerState.AMBIGUOUS_SPEAKER.value
        and not is_non_conversational_sentinel(turn.get("resolved_text"))
    ]
    checker = judge or context_sufficiency
    def empty_context_result() -> dict[str, Any]:
        return {
            "state": ContextSufficiencyState.CONTEXT_INSUFFICIENT.value,
            "relation": ContextRelation.MISSING_IMMEDIATE_TRIGGER.value,
            "immediate_trigger_turn_id": None,
            "confidence": 0.0,
            "checker": "deterministic-empty-context-v1",
            "judge_valid": True,
            "selected_context_turn_ids": [],
            "selected_target_turn_id": str(target_turn["audio_turn_id"]),
            "visible_context_only": True,
            "hidden_history_accessed": False,
            "reason": "MISSING_IMMEDIATE_TRIGGER",
        }

    if not eligible:
        result = empty_context_result()
        result.update({
            "candidate_turn_count": 0,
            "selected_turn_count": 0,
            "minimality_checked_prefix_sizes": 0,
        })
        return [], result

    checked_sizes: list[int] = []
    last_result = empty_context_result()
    for size in range(1, len(eligible) + 1):
        selected = eligible[-size:]
        selected_ids = {str(turn["audio_turn_id"]) for turn in selected}
        if required and not required.issubset(selected_ids):
            continue
        checked_sizes.append(size)
        result = checker(selected, target_turn)
        last_result = result
        relation = str(result.get("relation") or "")
        # Test doubles and pre-directional callers may only return the old
        # state field.  The production semantic judge always emits relation.
        if not relation and result.get("state") == ContextSufficiencyState.CONTEXT_SUFFICIENT.value:
            relation = ContextRelation.TARGET_RESPONDS_TO_CONTEXT.value
            result = {**result, "relation": relation}
        trigger_id = result.get("immediate_trigger_turn_id")
        if (
            result.get("state") == ContextSufficiencyState.CONTEXT_SUFFICIENT.value
            and relation == ContextRelation.TARGET_RESPONDS_TO_CONTEXT.value
            and not trigger_id
        ):
            result = {
                **result,
                "state": ContextSufficiencyState.CONTEXT_INSUFFICIENT.value,
                "relation": ContextRelation.MISSING_IMMEDIATE_TRIGGER.value,
                "immediate_trigger_turn_id": None,
                "judge_valid": False,
                "reason": "MISSING_IMMEDIATE_TRIGGER",
            }
            relation = ContextRelation.MISSING_IMMEDIATE_TRIGGER.value
        elif (
            result.get("state") == ContextSufficiencyState.CONTEXT_SUFFICIENT.value
            and relation == ContextRelation.TARGET_RESPONDS_TO_CONTEXT.value
            and str(trigger_id) not in selected_ids
        ):
            # A judge may have reasoned about a larger hidden/previous suffix,
            # but that cannot authorize the current visible suffix.  Fail
            # closed and continue expanding until the trigger is actually in
            # the selected prefix.
            result = {
                **result,
                "state": ContextSufficiencyState.CONTEXT_INSUFFICIENT.value,
                "relation": ContextRelation.MISSING_IMMEDIATE_TRIGGER.value,
                "immediate_trigger_turn_id": None,
                "judge_valid": False,
                "reason": "IMMEDIATE_TRIGGER_OUTSIDE_SELECTED_SUFFIX",
            }
            relation = ContextRelation.MISSING_IMMEDIATE_TRIGGER.value
        last_result = result
        if (
            result["state"] == ContextSufficiencyState.CONTEXT_SUFFICIENT.value
            and relation == ContextRelation.TARGET_RESPONDS_TO_CONTEXT.value
            and result.get("immediate_trigger_turn_id")
            and str(result.get("immediate_trigger_turn_id")) in selected_ids
        ):
            result["candidate_turn_count"] = len(eligible)
            result["selected_turn_count"] = len(selected)
            result["minimality_checked_prefix_sizes"] = size
            return selected, result
    last_result["candidate_turn_count"] = len(eligible)
    last_result["selected_turn_count"] = 0
    last_result["minimality_checked_prefix_sizes"] = max(checked_sizes, default=0)
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
