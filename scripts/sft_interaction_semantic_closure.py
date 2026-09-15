from __future__ import annotations

"""Interaction semantic closure and provenance-aware corpus primitives.

This module intentionally starts after identity closure and structural
reconstruction.  It never infers speaker identity or rewrites transcript text.
The public functions are pure enough to exercise with synthetic fixtures.
"""

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable


SCHEMA_VERSION = "2.0.0"
PIPELINE_VERSION = "sft-interaction-closure-v2.3.1-2026-09-13"
PRIMARY_PROMPT_VERSION = "interaction-primary-v2.3-p1"
STRICT_PROMPT_VERSION = "interaction-strict-v2.3-p1"
ADJUDICATION_PROMPT_VERSION = "interaction-adjudication-v2.3-p1"

TARGETS = {"NEURO_FAMILY_HIGH", "NEURO_FAMILY_MEDIUM"}
MIN_TARGET_SPEAKER_CONFIDENCE = 0.70
ALLOWED_RELATIONS = {
    "DIRECT_RESPONSE",
    "CONTEXTUAL_RESPONSE",
    "GAME_STATE_RESPONSE",
    "SELF_CONTINUATION",
    "HIDDEN_TRIGGER_SUSPECTED",
    "WRONG_CONTEXT",
    "UNCLEAR",
}
ACCEPT_RELATIONS = {"DIRECT_RESPONSE", "CONTEXTUAL_RESPONSE", "GAME_STATE_RESPONSE"}
ALLOWED_EPISODES = {
    "KEEP_CURRENT",
    "MERGE_CONTINUATION",
    "SPLIT_HIDDEN_TRIGGER",
    "SPLIT_NEW_SPEECH_ACT",
    "SPLIT_EVENT_BOUNDARY",
    "UNCLEAR",
}
ACCEPT_EPISODES = {"KEEP_CURRENT", "MERGE_CONTINUATION"}
FINAL_STATES = {"VERIFIED", "REJECTED", "AMBIGUOUS", "JUDGE_CONFLICT", "MISSING_INVALID_EVIDENCE", "TERMINAL_INVALID_QUARANTINED"}

_STOPWORDS = set(
    "the a an and or but if then than to of in on at for from with as is are was were be been being i you he she it we they this that those these do did does can could would should will may might have has had not no yes just really very like my your our their his her its how what why when where who please tell explain imagine guess".split()
)
_TRIGGER_RE = re.compile(
    r"\?|\b(what|why|how|when|where|who|can|could|would|should|do|did|does|is|are|will|please|tell|explain|imagine|guess)\b",
    re.I,
)
_ACK_RE = re.compile(r"^(yes|no|okay|ok|right|sure|exactly|i think|i guess|well|that|yeah|oh|got that|let's go)\b", re.I)
_CONTINUATION_WORDS = {"and", "or", "because", "which", "that", "to", "then", "than", "as", "if", "of", "in", "with"}
_INDEPENDENT_STARTERS = {"i", "i'm", "im", "well", "now", "but", "also", "actually", "so", "okay", "ok", "wait", "you", "we", "they"}


def norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def tokens(value: Any) -> list[str]:
    return re.findall(r"[a-z0-9']+", norm(value))


def content_tokens(value: Any) -> set[str]:
    return {token for token in re.findall(r"[a-z]{3,}", norm(value)) if token not in _STOPWORDS}


def message_text(row: dict[str, Any]) -> str:
    return "\n".join(f"{item.get('role')}:{norm(item.get('content'))}" for item in row.get("messages") or [])


def payload_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def response_text(row: dict[str, Any]) -> str:
    messages = row.get("messages") or []
    return "\n".join(str(item.get("content") or "") for item in messages if item.get("role") == "assistant")


def snapshot(turn: dict[str, Any]) -> dict[str, Any]:
    return {
        "turn_id": str(turn.get("turn_id") or ""),
        "speaker": turn.get("speaker"),
        "identity": turn.get("identity", "UNKNOWN"),
        "speaker_confidence": turn.get("speaker_confidence"),
        "timestamp": turn.get("timestamp"),
        "gap_from_previous": round(float(turn.get("_gap_from_previous") or 0.0), 3),
        "event_boundary": bool(turn.get("event_boundary")),
        "bad_boundary": bool(turn.get("_bad_boundary") or turn.get("bad_boundary")),
        "asr_quality": turn.get("asr_quality"),
        "text": str(turn.get("text") or "")[:600],
    }


def _hard_bad_boundary(turn: dict[str, Any]) -> bool:
    return any(
        bool(turn.get(key))
        for key in (
            "_bad_boundary",
            "bad_boundary",
            "uncertain_transcription",
            "suspicious_transcription",
            "overlap",
            "malformed",
        )
    )


def target_turn_hard_eligibility(
    turn: dict[str, Any],
    *,
    expected_speaker: Any,
    expected_identity: Any,
    continuation: bool,
) -> tuple[bool, str]:
    """Apply the deterministic target evidence gate before semantic judgement.

    The same helper is used again by final materialization. Semantic judgement
    may decide episode membership, but it cannot upgrade raw training evidence.
    """
    identity = str(turn.get("identity") or "UNKNOWN")
    if identity not in TARGETS or (expected_identity and identity != str(expected_identity)):
        return False, "TARGET_IDENTITY_INELIGIBLE"
    if str(turn.get("speaker") or "") != str(expected_speaker or ""):
        return False, "TARGET_SPEAKER_MISMATCH"
    try:
        confidence = float(turn.get("speaker_confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence < MIN_TARGET_SPEAKER_CONFIDENCE:
        return False, "TARGET_SPEAKER_CONFIDENCE_LOW"
    if turn.get("text_qa") != "PASS":
        return False, "TARGET_TEXT_QA_FAIL"
    if turn.get("asr_quality") == "REVIEW":
        return False, "TARGET_ASR_REVIEW"
    if _hard_bad_boundary(turn):
        return False, "TARGET_BAD_BOUNDARY"
    if continuation and turn.get("event_boundary"):
        return False, "TARGET_EVENT_BOUNDARY"
    return True, "TARGET_MACHINE_ELIGIBLE"


def response_gap_limit(row: dict[str, Any]) -> float:
    """Use the structural response-gap policy; preserve 4s for old fixtures."""
    try:
        return max(0.0, float((row.get("episode_boundary") or {}).get("response_gap_policy_seconds", 4.0)))
    except (TypeError, ValueError):
        return 4.0


def _safe_context_turn(turn: dict[str, Any], next_turn: dict[str, Any], max_gap: float) -> bool:
    if turn.get("identity") in TARGETS:
        return False
    if turn.get("_bad_boundary") or turn.get("bad_boundary") or turn.get("event_boundary") or next_turn.get("event_boundary"):
        return False
    if turn.get("text_qa") == "FAIL" or turn.get("asr_quality") == "REVIEW":
        return False
    return float(next_turn.get("_start", 0.0)) - float(turn.get("_end", 0.0)) <= max_gap


def bounded_context_options(
    turns: list[dict[str, Any]],
    structural_context_ids: list[str],
    *,
    max_extension_turns: int = 2,
    max_context_turns: int = 5,
    max_gap_seconds: float = 8.0,
) -> tuple[list[dict[str, Any]], list[list[str]]]:
    """Return a bounded envelope and exact legal context selections.

    Legal selections are contiguous suffixes ending at the structural anchor.
    At most ``max_extension_turns`` clean preceding turns may be added.  The
    verifier therefore may shorten or slightly extend, but cannot search the
    timeline or borrow diagnostic turns silently.
    """
    by_id = {str(turn.get("turn_id")): index for index, turn in enumerate(turns)}
    positions = [by_id[value] for value in structural_context_ids if value in by_id]
    if not positions or len(positions) != len(structural_context_ids):
        return [], []
    if positions != list(range(min(positions), max(positions) + 1)):
        return [], []
    start, anchor = min(positions), max(positions)
    extension = 0
    cursor = start - 1
    while cursor >= 0 and extension < max_extension_turns and anchor - cursor + 1 <= max_context_turns:
        if not _safe_context_turn(turns[cursor], turns[cursor + 1], max_gap_seconds):
            break
        start = cursor
        extension += 1
        cursor -= 1
    envelope = turns[start : anchor + 1]
    ids = [str(turn.get("turn_id")) for turn in envelope]
    options = [ids[index:] for index in range(len(ids))]
    return [snapshot(turn) for turn in envelope], options


def risk_features(request: dict[str, Any]) -> dict[str, Any]:
    """Lexical/syntactic signals used only for routing and audit."""
    context = " ".join(str(turn.get("text") or "") for turn in request.get("candidate_context") or [])
    response_turns = request.get("candidate_response_episode") or []
    continuation = request.get("possible_continuation_fragments") or []
    response = " ".join(str(turn.get("text") or "") for turn in response_turns)
    lexical_overlap = sorted(content_tokens(context) & content_tokens(response))
    flags: list[str] = []
    if not lexical_overlap:
        flags.append("NO_CONTENT_TOKEN_OVERLAP")
    if not _TRIGGER_RE.search(context):
        flags.append("NO_QUESTION_OR_EXPLICIT_TRIGGER_PATTERN")
    if not _ACK_RE.search(response):
        flags.append("NO_ACKNOWLEDGEMENT_PATTERN")
    if response[:1].islower():
        flags.append("LOWERCASE_RESPONSE_START")
    for turn in continuation:
        text = str(turn.get("text") or "").strip()
        first = text.lower().split(" ")[0].strip("'\".,!?;:") if text else ""
        if first in _INDEPENDENT_STARTERS:
            flags.append("POSSIBLE_INDEPENDENT_SPEECH_ACT")
        elif text[:1].islower() or first in _CONTINUATION_WORDS:
            flags.append("POSSIBLE_GRAMMATICAL_CONTINUATION")
    envelope = request.get("context_evidence_envelope") or []
    structural = request.get("structural_context_turn_ids") or []
    if len(envelope) > len(structural):
        flags.append("CONTEXT_EXTENSION_AVAILABLE")
    if any(turn.get("asr_quality") == "REVIEW" or turn.get("bad_boundary") for turn in envelope + response_turns + continuation):
        flags.append("TRANSCRIPT_OR_BOUNDARY_RISK")
    return {
        "schema_version": SCHEMA_VERSION,
        "authority": "ROUTING_AND_EVIDENCE_ONLY",
        "cannot_decide_semantic_outcome": True,
        "flags": sorted(set(flags)),
        "content_token_overlap": lexical_overlap[:20],
        "strict_required": True,
    }


def build_request(
    row: dict[str, Any],
    timeline: dict[str, Any],
    source: dict[str, Any],
    *,
    max_context_extension: int = 2,
) -> dict[str, Any] | None:
    turns = list(timeline.get("_turns") or [])
    if not turns:
        return None
    by_id = {str(turn.get("turn_id")): turn for turn in turns}
    context_ids = [str(value) for value in row.get("context_turn_ids") or []]
    target_ids = [str(value) for value in row.get("target_turn_ids") or []]
    if not context_ids or not target_ids or any(value not in by_id for value in context_ids + target_ids):
        return None
    for turn in turns:
        turn["_bad_boundary"] = bool(
            turn.get("bad_boundary")
            or turn.get("uncertain_transcription")
            or turn.get("suspicious_transcription")
            or turn.get("overlap")
            or turn.get("malformed")
        )
    envelope, options = bounded_context_options(turns, context_ids, max_extension_turns=max_context_extension)
    if not envelope or not options:
        return None
    positions = {str(turn.get("turn_id")): index for index, turn in enumerate(turns)}
    last_target = max(positions[value] for value in target_ids)
    candidate_target_ids = list(target_ids)
    expected_speaker = by_id[target_ids[-1]].get("speaker")
    expected_identity = row.get("identity") or by_id[target_ids[-1]].get("identity")
    continuation_gap_limit = response_gap_limit(row)
    cursor = last_target + 1
    while cursor < len(turns) and len(candidate_target_ids) < len(target_ids) + 3:
        turn = turns[cursor]
        previous = turns[cursor - 1]
        gap = max(0.0, float(turn.get("_start", cursor)) - float(previous.get("_end", cursor - 1)))
        eligible, _ = target_turn_hard_eligibility(
            turn,
            expected_speaker=expected_speaker,
            expected_identity=expected_identity,
            continuation=True,
        )
        if not eligible or previous.get("event_boundary") or gap > continuation_gap_limit:
            break
        candidate_target_ids.append(str(turn.get("turn_id")))
        cursor += 1
    structural_context = [snapshot(by_id[value]) for value in context_ids]
    target = [snapshot(by_id[value]) for value in target_ids]
    continuation = [snapshot(by_id[value]) for value in candidate_target_ids[len(target_ids) :]]
    anchor_index = positions[context_ids[-1]]
    following_start = max(positions[candidate_target_ids[-1]] + 1, last_target + 1)
    request = {
        "schema_version": SCHEMA_VERSION,
        "task": "Select the observable context and coherent target episode, then judge their interaction relation.",
        "structural_context_turn_ids": context_ids,
        "allowed_context_selections": options,
        "offered_context_turn_ids": [str(turn.get("turn_id")) for turn in envelope],
        "current_target_turn_ids": target_ids,
        "candidate_target_turn_ids": candidate_target_ids,
        "preceding_raw_turns_diagnostic_only": [snapshot(turn) for turn in turns[max(0, min(positions[value] for value in context_ids) - 3) : min(positions[value] for value in context_ids)]],
        "context_evidence_envelope": envelope,
        "candidate_context": structural_context,
        "candidate_response_episode": target,
        "possible_continuation_fragments": continuation,
        "following_raw_turns_diagnostic_only": [snapshot(turn) for turn in turns[following_start : following_start + 3]],
        "source": {"source_id": row.get("source_id"), "title": source.get("title") or row.get("source_title") or "", "source_type": source.get("source_type") or source.get("platform") or "", "url": source.get("source_url") or row.get("source_url") or ""},
        "allowed_relation": sorted(ALLOWED_RELATIONS),
        "allowed_episode_decision": sorted(ALLOWED_EPISODES),
        "constraints": [
            "Select context only from one exact allowed_context_selections entry.",
            "Diagnostic neighboring turns never become training context unless their ids are in the selected allowed entry.",
            "Select target ids only as a contiguous prefix of candidate_target_turn_ids containing all current target ids.",
            "Do not invent a hidden trigger and do not rewrite transcript text.",
            "Time adjacency alone is not response evidence.",
            "Absurdity, teasing, abruptness, short replies, and topic hijack are not rejection reasons after an observable trigger.",
        ],
        "output_schema": {
            "relation": " | ".join(sorted(ALLOWED_RELATIONS)),
            "confidence": "number 0..1",
            "context_complete": "boolean for selected context itself",
            "response_complete": "boolean",
            "transcript_usable": "boolean; never supply corrected text",
            "episode_decision": " | ".join(sorted(ALLOWED_EPISODES)),
            "selected_context_turn_ids": "one exact array from allowed_context_selections",
            "selected_target_turn_ids": "contiguous prefix of candidate_target_turn_ids containing current_target_turn_ids",
            "reason": "short evidence-based reason",
        },
    }
    request["risk_features"] = risk_features(request)
    request["structural_anchor_index"] = anchor_index
    return {"sample_id": row.get("sample_id"), "request": request, "judge_input": request}


def parse_json_object(value: str) -> dict[str, Any] | None:
    value = str(value or "").strip()
    for start in [index for index, char in enumerate(value) if char == "{"]:
        for end in reversed([index for index, char in enumerate(value) if char == "}"]):
            if end <= start:
                continue
            try:
                parsed = json.loads(value[start : end + 1])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    return None


def _edit_distance_at_most_one(left: str, right: str) -> bool:
    """Recognize only a single source-prefix typo; never fuzzy-match turns."""
    left, right = str(left), str(right)
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        return sum(a != b for a, b in zip(left, right)) <= 1
    if len(left) > len(right):
        left, right = right, left
    index_left = index_right = differences = 0
    while index_left < len(left) and index_right < len(right):
        if left[index_left] == right[index_right]:
            index_left += 1
            index_right += 1
            continue
        differences += 1
        if differences > 1:
            return False
        index_right += 1
    return True


def _normalize_selection_id(value: str, offered: list[str]) -> tuple[str, dict[str, str] | None]:
    """Map only an unambiguous identifier typo back to an offered raw turn id."""
    value = str(value)
    offered = list(dict.fromkeys(str(item) for item in offered))
    if value in offered:
        return value, None
    casefold_matches = [item for item in offered if item.casefold() == value.casefold()]
    if len(casefold_matches) == 1:
        target = casefold_matches[0]
        return target, {"from": value, "to": target, "method": "CASEFOLD_UNIQUE"}
    marker = ":turn:"
    if marker not in value:
        return value, None
    value_prefix, value_turn = value.rsplit(marker, 1)
    suffix_matches = []
    for item in offered:
        if marker not in item:
            continue
        prefix, turn = item.rsplit(marker, 1)
        if turn == value_turn and _edit_distance_at_most_one(value_prefix.casefold(), prefix.casefold()):
            suffix_matches.append(item)
    if len(suffix_matches) == 1:
        target = suffix_matches[0]
        return target, {"from": value, "to": target, "method": "UNIQUE_TURN_SUFFIX_PREFIX_TYPO"}
    return value, None


def validate_judgement(result: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    parsed = result.get("parsed") if isinstance(result, dict) and "parsed" in result else result
    if not isinstance(parsed, dict):
        return {"valid": False, "accepted": False, "reason": "missing_json_object"}
    relation = str(parsed.get("relation") or "UNCLEAR").strip().upper()
    episode = str(parsed.get("episode_decision") or "UNCLEAR").strip().upper()
    try:
        confidence = float(parsed.get("confidence"))
    except (TypeError, ValueError):
        confidence = -1.0
    selected_context = [str(value) for value in parsed.get("selected_context_turn_ids") or []]
    selected_target = [str(value) for value in parsed.get("selected_target_turn_ids") or []]
    options = [[str(value) for value in option] for option in request.get("allowed_context_selections") or []]
    offered_target = [str(value) for value in request.get("candidate_target_turn_ids") or []]
    current_target = [str(value) for value in request.get("current_target_turn_ids") or []]
    normalizations = []
    offered_context = [value for option in options for value in option]
    normalized_context = []
    for value in selected_context:
        normalized, note = _normalize_selection_id(value, offered_context)
        normalized_context.append(normalized)
        if note:
            normalizations.append(note)
    normalized_target = []
    for value in selected_target:
        normalized, note = _normalize_selection_id(value, offered_target)
        normalized_target.append(normalized)
        if note:
            normalizations.append(note)
    selected_context = normalized_context
    selected_target = normalized_target
    target_prefix = selected_target == offered_target[: len(selected_target)] and selected_target[: len(current_target)] == current_target
    forbidden_rewrite_fields = {"corrected_transcript", "rewritten_transcript", "context_text", "response_text"} & set(parsed)
    schema_valid = (
        relation in ALLOWED_RELATIONS
        and episode in ALLOWED_EPISODES
        and 0.0 <= confidence <= 1.0
        and selected_context in options
        and bool(selected_target)
        and target_prefix
        and isinstance(parsed.get("context_complete"), bool)
        and isinstance(parsed.get("response_complete"), bool)
        and isinstance(parsed.get("transcript_usable"), bool)
        and not forbidden_rewrite_fields
    )
    evidence = {
        "relation": relation,
        "episode_decision": episode,
        "confidence": confidence,
        "context_complete": parsed.get("context_complete") is True,
        "response_complete": parsed.get("response_complete") is True,
        "transcript_usable": parsed.get("transcript_usable") is True,
        "selected_context_turn_ids": selected_context,
        "selected_target_turn_ids": selected_target,
        "reason": str(parsed.get("reason") or "")[:500],
    }
    if normalizations:
        evidence["selection_id_normalizations"] = normalizations
    if not schema_valid:
        reason = "transcript_rewrite_forbidden" if forbidden_rewrite_fields else "schema_or_bounded_turn_selection_invalid"
        return {"valid": False, "accepted": False, **evidence, "reason": reason}
    accepted = (
        relation in ACCEPT_RELATIONS
        and episode in ACCEPT_EPISODES
        and confidence >= 0.75
        and evidence["context_complete"]
        and evidence["response_complete"]
        and evidence["transcript_usable"]
    )
    return {"valid": True, "accepted": accepted, **evidence}


def _explicit_reject(value: dict[str, Any]) -> bool:
    return bool(value.get("valid")) and value.get("relation") in {"HIDDEN_TRIGGER_SUSPECTED", "WRONG_CONTEXT"} and float(value.get("confidence") or 0.0) >= 0.75


def _same_selection(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left.get("relation") == right.get("relation")
        and left.get("episode_decision") == right.get("episode_decision")
        and left.get("selected_context_turn_ids") == right.get("selected_context_turn_ids")
        and left.get("selected_target_turn_ids") == right.get("selected_target_turn_ids")
    )


def resolve_semantic_state(
    sample_id: str,
    primary: dict[str, Any] | None,
    strict: dict[str, Any] | None,
    adjudication: dict[str, Any] | None = None,
    *,
    risk: dict[str, Any] | None = None,
    quarantine: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve model evidence without allowing a heuristic to be the judge."""
    p = (primary or {}).get("validated") if primary and "validated" in primary else primary
    s = (strict or {}).get("validated") if strict and "validated" in strict else strict
    a = (adjudication or {}).get("validated") if adjudication and "validated" in adjudication else adjudication
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "sample_id": sample_id,
        "risk_features": risk or {},
        "primary": primary,
        "strict": strict,
        "adjudication": adjudication,
    }
    if quarantine and quarantine.get("state") == "TERMINAL_INVALID_QUARANTINED":
        quarantine_stage = str(quarantine.get("stage") or "primary").upper()
        return {
            **provenance,
            "state": "TERMINAL_INVALID_QUARANTINED",
            "reason_code": f"{quarantine_stage}_INVALID_TERMINAL_QUARANTINED",
            "quarantine": quarantine,
        }
    if not p:
        return {**provenance, "state": "PENDING_PRIMARY", "reason_code": "PRIMARY_MISSING"}
    if not p.get("valid"):
        return {**provenance, "state": "MISSING_INVALID_EVIDENCE", "reason_code": "PRIMARY_INVALID"}
    if _explicit_reject(p) and not s:
        return {**provenance, "state": "REJECTED", "reason_code": "PRIMARY_EXPLICIT_OBSERVABILITY_REJECT", "final_decision": p}
    if not p.get("accepted") and not s:
        return {**provenance, "state": "AMBIGUOUS", "reason_code": "PRIMARY_NON_ACCEPT_UNRESOLVED", "final_decision": p}
    if not s:
        return {**provenance, "state": "PENDING_STRICT", "reason_code": "SECOND_SEMANTIC_VERIFICATION_REQUIRED"}
    if not s.get("valid"):
        return {**provenance, "state": "MISSING_INVALID_EVIDENCE", "reason_code": "STRICT_INVALID"}
    if p.get("accepted") and s.get("accepted") and _same_selection(p, s):
        return {**provenance, "state": "VERIFIED", "reason_code": "SEMANTIC_PASSES_AGREE", "final_decision": s}
    if not p.get("accepted") and not s.get("accepted"):
        if _explicit_reject(p) and _explicit_reject(s) and p.get("relation") == s.get("relation"):
            return {**provenance, "state": "REJECTED", "reason_code": "SEMANTIC_PASSES_AGREE_REJECT", "final_decision": s}
        return {**provenance, "state": "AMBIGUOUS", "reason_code": "SEMANTIC_PASSES_NON_ACCEPT", "final_decision": s}
    if a:
        if not a.get("valid"):
            return {**provenance, "state": "MISSING_INVALID_EVIDENCE", "reason_code": "ADJUDICATION_INVALID"}
        if a.get("accepted"):
            return {**provenance, "state": "VERIFIED", "reason_code": "CONFLICT_ADJUDICATED_ACCEPT", "final_decision": a}
        if _explicit_reject(a):
            return {**provenance, "state": "REJECTED", "reason_code": "CONFLICT_ADJUDICATED_REJECT", "final_decision": a}
        return {**provenance, "state": "AMBIGUOUS", "reason_code": "CONFLICT_ADJUDICATED_UNCLEAR", "final_decision": a}
    return {**provenance, "state": "JUDGE_CONFLICT", "reason_code": "PRIMARY_STRICT_DISAGREEMENT"}


def materialize_verified(row: dict[str, Any], closure: dict[str, Any], timeline: dict[str, Any]) -> dict[str, Any] | None:
    if closure.get("state") != "VERIFIED":
        return None
    decision = closure.get("final_decision") or {}
    by_id = {str(turn.get("turn_id")): turn for turn in timeline.get("_turns") or []}
    context_ids = [str(value) for value in decision.get("selected_context_turn_ids") or []]
    target_ids = [str(value) for value in decision.get("selected_target_turn_ids") or []]
    if any(value not in by_id for value in context_ids + target_ids):
        return None
    context = [by_id[value] for value in context_ids]
    target = [by_id[value] for value in target_ids]
    context_indices = [int(turn.get("_index", -1)) for turn in context]
    target_indices = [int(turn.get("_index", -1)) for turn in target]
    if context_indices != list(range(min(context_indices), max(context_indices) + 1)) or target_indices != list(range(min(target_indices), max(target_indices) + 1)) or max(context_indices) >= min(target_indices):
        return None

    if any(_hard_bad_boundary(turn) or turn.get("text_qa") != "PASS" or turn.get("asr_quality") == "REVIEW" for turn in context) or any(turn.get("identity") in TARGETS for turn in context):
        return None
    assistant_speakers = {str(turn.get("speaker") or "") for turn in target}
    assistant_identities = {str(turn.get("identity") or "UNKNOWN") for turn in target}
    if len(assistant_speakers) != 1 or len(assistant_identities) != 1 or not assistant_identities.issubset(TARGETS) or (row.get("identity") and str(row.get("identity")) not in assistant_identities):
        return None
    expected_speaker = target[0].get("speaker")
    expected_identity = row.get("identity") or target[0].get("identity")
    if any(
        not target_turn_hard_eligibility(
            turn,
            expected_speaker=expected_speaker,
            expected_identity=expected_identity,
            continuation=index > 0,
        )[0]
        for index, turn in enumerate(target)
    ):
        return None
    confidences = [float(turn.get("speaker_confidence") or 0.0) for turn in target]
    continuation_gap_limit = response_gap_limit(row)
    if not confidences or any(
        max(0.0, float(target[index].get("_start", index)) - float(target[index - 1].get("_end", index - 1))) > continuation_gap_limit
        or target[index - 1].get("event_boundary")
        for index in range(1, len(target))
    ):
        return None

    def evidence_snapshot(turn: dict[str, Any], role: str) -> dict[str, Any]:
        return {
            "turn_id": str(turn.get("turn_id")),
            "role": role,
            "raw_timeline_index": int(turn.get("_index", -1)),
            "text": str(turn.get("text") or "").strip(),
            "timestamp": turn.get("timestamp"),
            "speaker": turn.get("speaker"),
            "identity": turn.get("identity", "UNKNOWN"),
            "speaker_confidence": turn.get("speaker_confidence"),
            "text_qa": turn.get("text_qa"),
            "text_qa_reasons": turn.get("text_qa_reasons") or [],
            "asr_quality": turn.get("asr_quality"),
            "asr_quality_reasons": turn.get("asr_quality_reasons") or [],
            "asr_metrics": turn.get("asr_metadata") or {},
            "boundary": {"bad_boundary": _hard_bad_boundary(turn), "event_boundary": bool(turn.get("event_boundary")), "gap_from_previous": round(float(turn.get("_gap_from_previous") or 0.0), 3)},
        }

    transcript_turns = [evidence_snapshot(turn, "context") for turn in context] + [evidence_snapshot(turn, "target") for turn in target]
    result = dict(row)
    result["messages"] = [
        {"role": "user", "content": "\n".join(str(turn.get("text") or "").strip() for turn in context)},
        {"role": "assistant", "content": "\n".join(str(turn.get("text") or "").strip() for turn in target)},
    ]
    result["context_turn_ids"] = context_ids
    result["context_anchor_id"] = context_ids[-1]
    result["target_turn_ids"] = target_ids
    result["response_episode_id"] = hashlib.sha256((str(row.get("source_id")) + "|" + "|".join(target_ids)).encode()).hexdigest()[:20]
    result["raw_timeline_indices"] = context_indices + target_indices
    result["timestamps"] = {"start": float(context[0].get("_start", (context[0].get("timestamp") or {}).get("start"))), "end": float(target[-1].get("_end", (target[-1].get("timestamp") or {}).get("end")))}
    result["speaker_evidence"] = {
        "assistant_speaker": next(iter(assistant_speakers)),
        "assistant_identity": next(iter(assistant_identities)),
        "assistant_confidence": min(confidences),
        "assistant_turn_speakers": [str(turn.get("speaker") or "") for turn in target],
        "assistant_turn_identities": [str(turn.get("identity") or "UNKNOWN") for turn in target],
        "assistant_turn_confidences": confidences,
        "context_speakers": [str(turn.get("speaker") or "") for turn in context],
        "context_identities": [str(turn.get("identity") or "UNKNOWN") for turn in context],
        "materialized_from_final_turn_selection": True,
        "interaction_semantics_cannot_change_identity": True,
    }
    result["transcript_qa"] = {
        "status": "STRUCTURAL_PASS",
        "source": "CURRENT_CANONICAL_TIMELINE",
        "context_turn_ids": context_ids,
        "target_turn_ids": target_ids,
        "final_turn_ids": context_ids + target_ids,
        "turns": transcript_turns,
        "all_selected_turns_machine_usable": True,
        "llm_did_not_rewrite_transcript": True,
    }
    result["boundary_provenance"] = {
        "context_turn_ids": context_ids,
        "target_turn_ids": target_ids,
        "turns": [{"turn_id": turn["turn_id"], **turn["boundary"]} for turn in transcript_turns],
        "materialized_from_final_turn_selection": True,
    }
    result["semantic_closure"] = closure
    result["semantic_qa"] = {"status": "VERIFIED", **{key: decision.get(key) for key in ("relation", "episode_decision", "confidence", "context_complete", "response_complete", "transcript_usable", "reason")}}
    result["episode_reconstruction"] = {
        "decision": decision.get("episode_decision"),
        "structural_target_turn_ids": row.get("target_turn_ids") or [],
        "selected_target_turn_ids": target_ids,
        "added_continuation_turn_ids": [value for value in target_ids if value not in set(row.get("target_turn_ids") or [])],
    }
    result["semantic_quality"] = "PASS"
    result["context_integrity"] = "PASS"
    result["response_episode_integrity"] = "PASS"
    result["transcript_quality"] = "PASS" if decision.get("transcript_usable") else "REVIEW"
    result["pipeline_version"] = PIPELINE_VERSION
    result["artifact_schema_version"] = SCHEMA_VERSION
    result["training_candidate"] = False
    result["evaluation_only"] = False
    return result


class DSU:
    def __init__(self, values: Iterable[str]):
        self.parent = {value: value for value in values}
        self.members = {value: {value} for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> str:
        a, b = self.find(left), self.find(right)
        if a == b:
            return a
        if len(self.members[a]) < len(self.members[b]):
            a, b = b, a
        self.parent[b] = a
        self.members[a] |= self.members.pop(b)
        return a


def repair_recording_families(
    rows: list[dict[str, Any]],
    pairwise_edges: list[dict[str, Any]],
    *,
    min_supporting_pairs: int = 2,
    max_component_size: int = 8,
) -> dict[str, Any]:
    """Merge recording clusters with auditable, bridge-resistant evidence."""
    clusters = sorted({str(row.get("canonical_recording_id")) for row in rows})
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for edge in pairwise_edges:
        left, right = sorted((str(edge.get("cluster_a")), str(edge.get("cluster_b"))))
        if left and right and left != right:
            grouped[(left, right)].append(edge)
    candidates = []
    for (left, right), evidence in grouped.items():
        strong = [edge for edge in evidence if edge.get("strong_merge_edge")]
        max_overlap = max((int(edge.get("overlap_length_tokens") or 0) for edge in evidence), default=0)
        max_containment = max((float(edge.get("containment_ratio") or 0.0) for edge in evidence), default=0.0)
        independent_pairs = len({(str(edge.get("sample_a")), str(edge.get("sample_b"))) for edge in strong})
        max_similarity = max((float(edge.get("similarity") or 0.0) for edge in evidence), default=0.0)
        ultra_strong = max_overlap >= 80 and max_containment >= 0.90 and max_similarity >= 0.90
        aggregate_strong = independent_pairs >= min_supporting_pairs and (max_overlap >= 30 or max_containment >= 0.85)
        eligible = ultra_strong or aggregate_strong
        candidates.append({"cluster_a": left, "cluster_b": right, "supporting_pair_count": independent_pairs, "max_overlap_tokens": max_overlap, "max_containment_ratio": round(max_containment, 6), "max_similarity": round(max_similarity, 6), "aggregate_strong": aggregate_strong, "ultra_strong_single_edge": ultra_strong, "eligible": eligible})
    candidates.sort(key=lambda edge: (-edge["supporting_pair_count"], -edge["max_overlap_tokens"], edge["cluster_a"], edge["cluster_b"]))
    dsu = DSU(clusters)
    accepted, quarantined = [], []
    for edge in candidates:
        if not edge["eligible"]:
            quarantined.append({**edge, "decision": "QUARANTINED_WEAK_EDGE"})
            continue
        left_root, right_root = dsu.find(edge["cluster_a"]), dsu.find(edge["cluster_b"])
        combined = len(dsu.members[left_root] | dsu.members[right_root])
        if left_root != right_root and combined > max_component_size:
            quarantined.append({**edge, "decision": "QUARANTINED_COMPONENT_BRIDGE", "proposed_component_size": combined})
            continue
        dsu.union(edge["cluster_a"], edge["cluster_b"])
        accepted.append({**edge, "decision": "MERGED"})
    root_to_members: dict[str, list[str]] = defaultdict(list)
    for cluster in clusters:
        root_to_members[dsu.find(cluster)].append(cluster)
    cluster_to_family: dict[str, str] = {}
    families = []
    for members in root_to_members.values():
        members = sorted(members)
        family_id = "rf_" + hashlib.sha256("|".join(members).encode()).hexdigest()[:16]
        cluster_to_family.update({member: family_id for member in members})
        families.append({"recording_family_id": family_id, "canonical_recording_ids": members})
    split_exclusion_relations = []
    for edge in quarantined:
        if edge.get("decision") != "QUARANTINED_COMPONENT_BRIDGE" or not edge.get("eligible"):
            continue
        family_a = cluster_to_family[str(edge["cluster_a"])]
        family_b = cluster_to_family[str(edge["cluster_b"])]
        if family_a != family_b:
            split_exclusion_relations.append({
                "family_a": family_a,
                "family_b": family_b,
                "cluster_a": edge["cluster_a"],
                "cluster_b": edge["cluster_b"],
                "reason": "ELIGIBLE_STRONG_EDGE_QUARANTINED_ONLY_FOR_COMPONENT_SIZE",
                "must_share_partition": True,
            })
    return {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "policy": {"aggregate_edge": f"supporting_pairs>={min_supporting_pairs} and (overlap>=30 or containment>=0.85)", "ultra_strong_single_edge": "overlap>=80 and containment>=0.90 and similarity>=0.90", "max_component_size": max_component_size},
        "pairwise_edge_count": len(pairwise_edges),
        "cluster_pair_evidence": candidates,
        "accepted_merge_edges": accepted,
        "quarantined_edges": quarantined,
        "split_exclusion_relations": split_exclusion_relations,
        "recording_families": sorted(families, key=lambda value: value["recording_family_id"]),
        "cluster_to_family": cluster_to_family,
    }


def _contains(short: list[str], long: list[str]) -> bool:
    return bool(short) and len(short) <= len(long) and any(long[index : index + len(short)] == short for index in range(len(long) - len(short) + 1))


def provenance_aware_dedup(rows: list[dict[str, Any]], family_map: dict[str, str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Remove data duplication while preserving cross-recording behavior."""
    audit: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "policy": "hard dedup is limited to identical provenance or one repaired recording family; cross-family text similarity is behavior-frequency metadata",
        "removed_by_reason": Counter(),
        "removed_examples": [],
        "cross_recording_behavior_repetitions_preserved": 0,
    }
    prepared = []
    for source in rows:
        row = dict(source)
        row["recording_family_id"] = family_map.get(str(row.get("canonical_recording_id")), str(row.get("canonical_recording_id")))
        prepared.append(row)
    exact_provenance: dict[tuple[Any, ...], dict[str, Any]] = {}
    exact_by_family: dict[tuple[str, str], dict[str, Any]] = {}
    anchors: dict[tuple[str, str], str] = {}
    targets: dict[tuple[str, str], str] = {}
    kept = []
    def remove(reason: str, row: dict[str, Any], prior: str | None = None) -> None:
        audit["removed_by_reason"][reason] += 1
        if len(audit["removed_examples"]) < 100:
            audit["removed_examples"].append({"reason": reason, "kept": prior, "removed": row.get("sample_id")})
    for row in sorted(prepared, key=lambda value: str(value.get("sample_id"))):
        family = str(row.get("recording_family_id"))
        provenance_key = (str(row.get("source_id")), tuple(row.get("context_turn_ids") or []), tuple(row.get("target_turn_ids") or []))
        text_key = message_text(row)
        if provenance_key in exact_provenance:
            remove("EXACT_INTERACTION_PROVENANCE", row, str(exact_provenance[provenance_key].get("sample_id")))
            continue
        prior_exact = exact_by_family.get((family, text_key))
        if prior_exact and str(prior_exact.get("source_id")) != str(row.get("source_id")):
            remove("SAME_RECORDING_EXACT_MESSAGES", row, str(prior_exact.get("sample_id")))
            continue
        source_id = str(row.get("source_id"))
        anchor = (source_id, str(row.get("context_anchor_id")))
        if anchor in anchors:
            remove("SAME_RECORDING_DUPLICATE_ANCHOR", row, anchors[anchor])
            continue
        duplicate_target = next((targets[(source_id, str(value))] for value in row.get("target_turn_ids") or [] if (source_id, str(value)) in targets), None)
        if duplicate_target:
            remove("SAME_RECORDING_TARGET_REUSE", row, duplicate_target)
            continue
        exact_provenance[provenance_key] = row
        exact_by_family[(family, text_key)] = row
        anchors[anchor] = str(row.get("sample_id"))
        for value in row.get("target_turn_ids") or []:
            targets[(source_id, str(value))] = str(row.get("sample_id"))
        kept.append(row)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in kept:
        by_family[str(row.get("recording_family_id"))].append(row)
    removed_ids: set[str] = set()
    for family_rows in by_family.values():
        ordered = sorted(family_rows, key=lambda value: (len(tokens(message_text(value))), str(value.get("sample_id"))))
        for index, short_row in enumerate(ordered):
            short_tokens = tokens(message_text(short_row))
            if len(short_tokens) < 8 or str(short_row.get("sample_id")) in removed_ids:
                continue
            for long_row in ordered[index + 1 :]:
                long_tokens = tokens(message_text(long_row))
                cross_source_reupload = str(short_row.get("source_id")) != str(long_row.get("source_id"))
                same_raw_anchor = str(short_row.get("source_id")) == str(long_row.get("source_id")) and str(short_row.get("context_anchor_id")) == str(long_row.get("context_anchor_id"))
                if (cross_source_reupload or same_raw_anchor) and _contains(short_tokens, long_tokens):
                    long_id = str(long_row.get("sample_id"))
                    removed_ids.add(long_id)
                    remove("SAME_RECORDING_INTERACTION_CONTAINMENT", long_row, str(short_row.get("sample_id")))
    final = [row for row in kept if str(row.get("sample_id")) not in removed_ids]
    cross_family_text: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in final:
        cross_family_text[norm(response_text(row))].append(row)
    for repeated in cross_family_text.values():
        families = sorted({str(row.get("recording_family_id")) for row in repeated})
        if len(families) < 2:
            continue
        cluster_id = "br_" + hashlib.sha256(norm(response_text(repeated[0])).encode()).hexdigest()[:16]
        for row in repeated:
            row.setdefault("distribution_metadata", {})["behavior_repeat_cluster"] = cluster_id
            row["distribution_metadata"]["behavior_repeat_independent_family_count"] = len(families)
        audit["cross_recording_behavior_repetitions_preserved"] += len(repeated)
    audit["removed_by_reason"] = dict(audit["removed_by_reason"])
    audit["rows_before"] = len(rows)
    audit["rows_after"] = len(final)
    target_counts: Counter[tuple[str, str]] = Counter()
    final_by_anchor: dict[tuple[str, str], list[list[str]]] = defaultdict(list)
    for row in final:
        source_id = str(row.get("source_id"))
        for value in row.get("target_turn_ids") or []:
            target_counts[(source_id, str(value))] += 1
        final_by_anchor[(source_id, str(row.get("context_anchor_id")))].append(tokens(response_text(row)))
    prefix_ladders = 0
    for values in final_by_anchor.values():
        for index, left in enumerate(values):
            for right in values[index + 1 :]:
                prefix_ladders += int(left != right and (_contains(left, right) or _contains(right, left)))
    audit["target_turn_reuse_count_after"] = sum(count > 1 for count in target_counts.values())
    audit["prefix_ladder_count_after"] = prefix_ladders
    return final, audit


def distribution_metadata(row: dict[str, Any]) -> dict[str, Any]:
    context = " ".join(str(item.get("content") or "") for item in row.get("messages") or [] if item.get("role") == "user")
    response = response_text(row)
    relation = str((row.get("semantic_qa") or {}).get("relation") or "UNKNOWN")
    return {
        "schema_version": "1.0.0",
        "authority": "ANALYSIS_AND_SAMPLING_ONLY",
        "interaction_type": relation,
        "context_token_count": len(tokens(context)),
        "response_token_count": len(tokens(response)),
        "context_speaker_class": "multi" if len((row.get("speaker_evidence") or {}).get("context_speakers") or []) > 1 else "single",
        "game_or_event_context": relation == "GAME_STATE_RESPONSE",
        "ai_self_reference": bool(re.search(r"\b(ai|artificial intelligence|neuro|evil)\b", response, re.I)),
        "question_back": "?" in response,
        "semantic_topic_shift_risk": "NO_CONTENT_TOKEN_OVERLAP" in (((row.get("semantic_closure") or {}).get("risk_features") or {}).get("flags") or []),
        "does_not_affect_identity_or_quality_gate": True,
    }


def closure_status_report(decisions: list[dict[str, Any]], structural_count: int) -> dict[str, Any]:
    counts = Counter(str(value.get("state") or "UNKNOWN") for value in decisions)
    terminal = sum(counts[state] for state in FINAL_STATES)
    primary_invalid = sum(value.get("reason_code") == "PRIMARY_INVALID" for value in decisions)
    strict_invalid = sum(value.get("reason_code") == "STRICT_INVALID" for value in decisions)
    primary_valid = 0
    primary_accepted = 0
    primary_rejected_nonaccepted = 0
    for value in decisions:
        primary = value.get("primary") or {}
        validated = primary.get("validated") if isinstance(primary, dict) else {}
        if validated and validated.get("valid"):
            primary_valid += 1
            if validated.get("accepted"):
                primary_accepted += 1
            else:
                primary_rejected_nonaccepted += 1
    unresolved_states = {"PENDING_PRIMARY", "PENDING_STRICT", "JUDGE_CONFLICT", "AMBIGUOUS", "MISSING_INVALID_EVIDENCE"}
    primary_terminal_quarantined = sum(
        value.get("state") == "TERMINAL_INVALID_QUARANTINED"
        and str(value.get("reason_code") or "").startswith("PRIMARY_")
        for value in decisions
    )
    strict_terminal_quarantined = sum(
        value.get("state") == "TERMINAL_INVALID_QUARANTINED"
        and str(value.get("reason_code") or "").startswith("STRICT_")
        for value in decisions
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "artifact_status": "INTERACTION_CLOSURE_STATUS",
        "structural_candidates": structural_count,
        "state_counts": dict(sorted(counts.items())),
        "pending": counts["PENDING_PRIMARY"] + counts["PENDING_STRICT"],
        "judge_conflicts": counts["JUDGE_CONFLICT"],
        "ambiguous": counts["AMBIGUOUS"],
        "missing_or_invalid_evidence": counts["MISSING_INVALID_EVIDENCE"],
        "terminal_quarantined": counts["TERMINAL_INVALID_QUARANTINED"],
        "unresolved_high_risk_samples": sum(1 for value in decisions if value.get("state") in unresolved_states and ((value.get("risk_features") or {}).get("flags"))),
        "primary_valid": primary_valid,
        "primary_accepted": primary_accepted,
        "primary_rejected_nonaccepted": primary_rejected_nonaccepted,
        "primary_terminal_quarantined": primary_terminal_quarantined,
        "strict_terminal_quarantined": strict_terminal_quarantined,
        "primary_invalid": primary_invalid,
        "strict_invalid": strict_invalid,
        "primary_coverage_pass": not counts["PENDING_PRIMARY"] and not primary_invalid,
        "strict_or_equivalent_verification_pass": not counts["PENDING_STRICT"] and not strict_invalid,
        "complete": terminal == structural_count and not counts["JUDGE_CONFLICT"] and not counts["MISSING_INVALID_EVIDENCE"],
    }
