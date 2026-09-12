from __future__ import annotations

import hashlib
import re
from typing import Any


TARGETS = {"NEURO_FAMILY_HIGH", "NEURO_FAMILY_MEDIUM"}
BAD_FLAGS = ("uncertain_transcription", "suspicious_transcription", "overlap", "malformed")
INDEPENDENT_STARTERS = {
    "anyway", "also", "now", "well", "wait", "okay", "ok", "so", "but", "however",
    "actually", "i", "i'm", "im", "we", "we're", "were", "you", "that's", "thats",
}
CONTINUATION_STARTERS = {"and", "or", "because", "which", "that", "to", "then", "than"}
GARBAGE_SINGLETONS = {"filtered", "heart", "underscore", "subtitle", "[music]", "[applause]"}


def normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def text_quality(text: Any, minimum_words: int = 1) -> tuple[str, list[str]]:
    value = normalize_text(text)
    reasons: list[str] = []
    if not value:
        return "FAIL", ["empty"]
    words = re.findall(r"[A-Za-z]{2,}(?:['-][A-Za-z]+)*|\d+", value)
    lower_words = [word.lower() for word in words]
    if len(words) < minimum_words and not re.search(r"[\u4e00-\u9fff\u3040-\u30ff]", value):
        reasons.append("too_few_words")
    if value.lower() in GARBAGE_SINGLETONS or (set(lower_words) & GARBAGE_SINGLETONS and len(words) <= 3):
        reasons.append("subtitle_or_asr_artifact")
    if re.fullmatch(r"[\W_]+", value, flags=re.UNICODE):
        reasons.append("symbol_only")
    if re.search(r"(?:https?://|www\.)\S+", value) and len(words) <= 2:
        reasons.append("url_artifact")
    if len(words) >= 5 and len(set(lower_words)) == 1:
        reasons.append("repeated_nonsense")
    punctuation = sum(1 for char in value if not char.isalnum() and not char.isspace() and char not in "'\".,?!-:;()")
    if value and punctuation / max(1, len(value)) > 0.35:
        reasons.append("token_soup")
    # A mixed Latin/CJK utterance is allowed; malformed is only inferred when
    # it is also symbol-heavy or lacks a lexical token.
    if not words and not re.search(r"[\u4e00-\u9fff\u3040-\u30ff]", value):
        reasons.append("no_lexical_content")
    return ("FAIL" if reasons else "PASS"), reasons


def is_bad_boundary(turn: dict[str, Any]) -> bool:
    return any(bool(turn.get(flag)) for flag in BAD_FLAGS)


def _bounds(turn: dict[str, Any], fallback: float) -> tuple[float, float]:
    timestamp = turn.get("timestamp") or {}
    try:
        start = float(timestamp.get("start"))
        end = float(timestamp.get("end"))
    except (TypeError, ValueError):
        start = fallback
        end = fallback
    return start, end


def _is_independent_start(text: str) -> bool:
    first = (normalize_text(text).lower().split(" ") or [""])[0].strip("'\".,!?;:")
    return first in INDEPENDENT_STARTERS


def can_continue_response(previous: dict[str, Any], current: dict[str, Any], max_gap_seconds: float) -> tuple[bool, str]:
    if previous.get("speaker") != current.get("speaker"):
        return False, "speaker_changed"
    if previous.get("identity") not in TARGETS or current.get("identity") not in TARGETS:
        return False, "identity_changed"
    if float(previous.get("speaker_confidence") or 0.0) < 0.70 or float(current.get("speaker_confidence") or 0.0) < 0.70:
        return False, "speaker_confidence_below_0.70"
    if is_bad_boundary(previous) or is_bad_boundary(current):
        return False, "bad_boundary"
    previous_start, previous_end = _bounds(previous, 0.0)
    current_start, _ = _bounds(current, previous_end)
    gap = max(0.0, current_start - previous_end)
    if gap > max_gap_seconds:
        return False, "gap_exceeded"
    if previous.get("event_boundary") or current.get("event_boundary"):
        return False, "event_boundary"
    previous_text = normalize_text(previous.get("text"))
    current_text = normalize_text(current.get("text"))
    if not previous_text or not current_text:
        return False, "empty_text"
    if _is_independent_start(current_text):
        return False, "new_discourse_act"
    if re.search(r"[.!?]\s*$", previous_text) and re.match(r"[A-Z\u4e00-\u9fff\u3040-\u30ff]", current_text):
        return False, "completed_sentence"
    first = current_text.lower().split(" ")[0].strip("'\".,!?;:")
    if first in CONTINUATION_STARTERS or current_text[:1].islower():
        return True, "syntactic_continuation"
    if not re.search(r"[.!?]\s*$", previous_text):
        return True, "unfinished_previous_turn"
    if gap <= min(0.65, max_gap_seconds):
        return True, "tight_same_speaker_fragment"
    return False, "independent_speech_act"


def _merge_messages(turns: list[dict[str, Any]], role: str) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    for turn in turns:
        text = normalize_text(turn.get("text"))
        if not text:
            continue
        if merged and merged[-1]["role"] == role:
            merged[-1]["content"] += "\n" + text
        else:
            merged.append({"role": role, "content": text})
    return merged


def _context_block(turns: list[dict[str, Any]], target_index: int, max_context_turns: int) -> tuple[list[int], str | None]:
    if target_index <= 0:
        return [], "no_preceding_turn"
    if is_bad_boundary(turns[target_index - 1]):
        return [], "bad_middle_boundary"
    if turns[target_index - 1].get("identity") in TARGETS:
        return [], "no_new_observable_trigger"
    indices = []
    cursor = target_index - 1
    while cursor >= 0 and len(indices) < max_context_turns:
        turn = turns[cursor]
        if is_bad_boundary(turn) or turn.get("identity") in TARGETS:
            break
        status, _ = text_quality(turn.get("text"), 1)
        if status != "PASS":
            break
        indices.append(cursor)
        cursor -= 1
    indices.reverse()
    if not indices:
        return [], "no_clean_context_anchor"
    return indices, None


def _judge(
    source_id: str,
    turns: list[dict[str, Any]],
    context_indices: list[int],
    target_indices: list[int],
    following_indices: list[int],
    trigger_gap_seconds: float,
    title: str = "",
) -> dict[str, Any]:
    context = [turns[index] for index in context_indices]
    targets = [turns[index] for index in target_indices]
    following = [turns[index] for index in following_indices]
    def snapshot(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{"turn_id": item.get("turn_id"), "speaker": item.get("speaker"), "identity": item.get("identity"), "timestamp": item.get("timestamp"), "text": normalize_text(item.get("text"))} for item in items]
    judge_input = {"preceding_raw_turns": snapshot(context[-3:]), "candidate_context": snapshot(context), "candidate_target_episode": snapshot(targets), "following_raw_turns": snapshot(following[:2]), "title": title}
    reasons: list[str] = []
    if not context or not targets:
        return {"status": "REJECT", "label": "UNCLEAR", "reasons": ["missing_context_or_target"], "judge_input": judge_input}
    if any(is_bad_boundary(turn) for turn in context + targets):
        reasons.append("bad_boundary_inside_candidate")
    if any(turn.get("speaker") == targets[0].get("speaker") for turn in context):
        reasons.append("TARGET_SELF_FRAGMENT")
    for turn in context:
        if text_quality(turn.get("text"), 1)[0] != "PASS":
            reasons.append("context_transcript_quality")
    merged_context_text = "\n".join(normalize_text(turn.get("text")) for turn in context)
    if text_quality(merged_context_text, 1)[0] != "PASS":
        reasons.append("context_transcript_quality")
    for turn in targets:
        if text_quality(turn.get("text"), 1)[0] != "PASS":
            reasons.append("target_transcript_quality")
    first_context_start, context_end = _bounds(context[-1], 0.0)
    target_start, target_end = _bounds(targets[0], 0.0)
    gap = max(0.0, target_start - context_end)
    if gap > trigger_gap_seconds:
        reasons.append("HIDDEN_TRIGGER_SUSPECTED")
    if reasons:
        label = "TARGET_SELF_FRAGMENT" if "TARGET_SELF_FRAGMENT" in reasons else "ASR_GARBAGE" if "context_transcript_quality" in reasons or "target_transcript_quality" in reasons else "UNCLEAR"
        return {"status": "REJECT", "label": label, "reasons": reasons, "context_turn_ids": [turn.get("turn_id") for turn in context], "target_turn_ids": [turn.get("turn_id") for turn in targets], "judge_input": judge_input}
    trigger_text = normalize_text(context[-1].get("text"))
    if trigger_text.endswith("?") or re.search(r"\b(what|why|how|where|when|can|could|would|do|did|is|are)\b", trigger_text.lower()):
        label = "DIRECT_RESPONSE"
    elif any(word in title.lower() for word in ("game", "gaming", "minecraft", "chess", "playing")):
        label = "VALID_GAME_STATE_RESPONSE"
    else:
        label = "VALID_CONTEXTUAL_RESPONSE"
    status = "REPAIR_FROM_RAW" if len(context_indices) > 1 or len(target_indices) > 1 else "ACCEPT"
    return {
        "status": status,
        "label": label,
        "reasons": ["raw_timeline_boundary_and_role_checks_passed"],
        "context_turn_ids": [turn.get("turn_id") for turn in context],
        "target_turn_ids": [turn.get("turn_id") for turn in targets],
        "following_turn_ids": [turns[index].get("turn_id") for index in following_indices],
        "judge_input": judge_input,
        "gap_seconds": round(gap, 3),
    }


def reconstruct_timeline(
    source_id: str,
    canonical_recording_id: str,
    turns: list[dict[str, Any]],
    *,
    trigger_gap_seconds: float = 10.0,
    continuation_gap_seconds: float = 1.5,
    max_context_turns: int = 3,
    title: str = "",
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, original in enumerate(turns):
        turn = dict(original)
        turn.setdefault("identity", "UNKNOWN")
        turn["_index"] = index
        turn["_start"], turn["_end"] = _bounds(turn, float(index))
        normalized.append(turn)
    candidates: list[dict[str, Any]] = []
    for index, target in enumerate(normalized):
        if target.get("identity") not in TARGETS:
            continue
        if is_bad_boundary(target) or text_quality(target.get("text"), 1)[0] != "PASS":
            continue
        if float(target.get("speaker_confidence") or 0.0) < 0.70:
            continue
        if index > 0 and normalized[index - 1].get("identity") in TARGETS:
            continued, _ = can_continue_response(normalized[index - 1], target, continuation_gap_seconds)
            if continued:
                continue
        context_indices, failure = _context_block(normalized, index, max_context_turns)
        if failure:
            continue
        if any(normalized[context_index].get("speaker") == target.get("speaker") for context_index in context_indices):
            continue
        target_indices = [index]
        continuation_reasons: list[str] = []
        cursor = index + 1
        while cursor < len(normalized):
            next_turn = normalized[cursor]
            if next_turn.get("identity") not in TARGETS:
                break
            if float(next_turn.get("speaker_confidence") or 0.0) < 0.70 or text_quality(next_turn.get("text"), 1)[0] != "PASS":
                break
            continued, reason = can_continue_response(normalized[target_indices[-1]], next_turn, continuation_gap_seconds)
            if not continued:
                break
            target_indices.append(cursor)
            continuation_reasons.append(reason)
            cursor += 1
        following_indices = list(range(cursor, min(len(normalized), cursor + 2)))
        semantic_qa = _judge(source_id, normalized, context_indices, target_indices, following_indices, trigger_gap_seconds, title)
        if semantic_qa["status"] == "REJECT":
            continue
        context_turns = [normalized[item] for item in context_indices]
        target_turns = [normalized[item] for item in target_indices]
        context_anchor_id = str(context_turns[-1].get("turn_id"))
        target_turn_ids = [str(item.get("turn_id")) for item in target_turns]
        response_episode_id = hashlib.sha256((source_id + "|" + "|".join(target_turn_ids)).encode("utf-8")).hexdigest()[:20]
        sample_id = hashlib.sha256((source_id + "|" + context_anchor_id + "|" + response_episode_id).encode("utf-8")).hexdigest()[:20]
        messages = _merge_messages(context_turns, "user") + _merge_messages(target_turns, "assistant")
        target_start = target_turns[0]["_start"]
        target_end = target_turns[-1]["_end"]
        candidate = {
            "sample_id": sample_id,
            "messages": messages,
            "source_id": source_id,
            "canonical_recording_id": canonical_recording_id,
            "context_anchor_id": context_anchor_id,
            "context_turn_id": context_anchor_id,
            "context_turn_ids": [str(item.get("turn_id")) for item in context_turns],
            "response_episode_id": response_episode_id,
            "target_turn_ids": target_turn_ids,
            "raw_timeline_indices": context_indices + target_indices,
            "timestamps": {"start": context_turns[0]["_start"], "end": target_end},
            "speaker_evidence": {
                "assistant_speaker": str(target_turns[0].get("speaker")),
                "assistant_identity": target_turns[0].get("identity"),
                "assistant_confidence": round(min(float(item.get("speaker_confidence") or 0.0) for item in target_turns), 6),
                "context_identities": [str(item.get("identity") or "UNKNOWN") for item in context_turns],
                "context_speakers": [str(item.get("speaker") or "") for item in context_turns],
            },
            "identity": target_turns[0].get("identity"),
            "identity_confidence": "high" if target_turns[0].get("identity") == "NEURO_FAMILY_HIGH" else "medium",
            "upstream_evidence_score": round(min(float(item.get("speaker_confidence") or 0.0) for item in target_turns), 6),
            "semantic_quality": "PASS",
            "context_integrity": "PASS",
            "transcript_quality": "PASS",
            "response_episode_integrity": "PASS",
            "semantic_qa": semantic_qa,
            "transcript_qa": {"status": "PASS", "missing_confidence_not_used_as_quality_evidence": any(item.get("transcription_confidence") in (None, 0, 0.0) for item in context_turns + target_turns)},
            "episode_boundary": {"continuation_reasons": continuation_reasons, "continuation_gap_seconds": continuation_gap_seconds, "trigger_gap_seconds": trigger_gap_seconds},
            "training_candidate": False,
            "evaluation_only": False,
            "pipeline_version": "sft-semantic-closure-v2-core-2026-09-12",
        }
        candidates.append(candidate)
    return candidates
