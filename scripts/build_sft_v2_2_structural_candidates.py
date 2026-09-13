from __future__ import annotations

"""Build the v2.2 Layer-A structural pool from canonical timelines.

The builder deliberately does not consume old conversation windows.  It reads
only canonical unique timelines plus the current fusion mapping, so updated
ASR/diarization can recover candidates without creating a second training path.
Semantic relation is left UNKNOWN for the local model judge.
"""

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TIMELINES = ROOT / "unique_timelines"
FUSION = ROOT / "identity_results" / "identity_mapping_multimodal_fusion_proxy.jsonl"
REGISTRY = ROOT / "source_registry.json"
MANIFEST = ROOT / "manifest" / "master_video_manifest.jsonl"
OLD_STRUCTURAL = ROOT / "datasets" / "meow_v02_sft_v2_1_semantic_verified" / "structural_candidates_v2.jsonl"
OUT_DIR = ROOT / "datasets" / "meow_v02_sft_v2_2_semantic_verified"
OUT = OUT_DIR / "structural_candidates_v2_2.jsonl"
FUNNEL = OUT_DIR / "structural_reconstruction_funnel.json"
TARGETS = {"NEURO_FAMILY_HIGH", "NEURO_FAMILY_MEDIUM"}
BAD_KEYS = ("uncertain_transcription", "suspicious_transcription", "overlap", "malformed")
GARBAGE = {"filtered", "heart", "underscore", "subtitle", "[music]", "[applause]"}


def norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def text_quality(text: Any) -> tuple[str, list[str]]:
    value = norm(text)
    if not value:
        return "FAIL", ["empty"]
    words = re.findall(r"[A-Za-z]{2,}(?:['-][A-Za-z]+)*|\d+", value)
    lower = [x.lower() for x in words]
    reasons: list[str] = []
    if value.lower() in GARBAGE:
        reasons.append("asr_or_subtitle_artifact")
    if len(words) >= 5 and len(set(lower)) == 1:
        reasons.append("repeated_nonsense")
    if re.fullmatch(r"[\W_]+", value, flags=re.UNICODE):
        reasons.append("symbol_only")
    if re.search(r"(?:https?://|www\.)\S+", value) and len(words) <= 3:
        reasons.append("url_artifact")
    # Preserve real code-switching; only flag script/token soup when lexical
    # content is absent or punctuation overwhelms the utterance.
    punctuation = sum(1 for c in value if not c.isalnum() and not c.isspace() and c not in "'\".,?!-:;()")
    if punctuation / max(1, len(value)) > 0.35:
        reasons.append("token_soup")
    if not words and not re.search(r"[\u4e00-\u9fff\u3040-\u30ff]", value):
        reasons.append("no_lexical_content")
    return ("FAIL" if reasons else "PASS"), reasons


def bounds(turn: dict[str, Any], fallback: float) -> tuple[float, float]:
    stamp = turn.get("timestamp") or {}
    try:
        start, end = float(stamp.get("start")), float(stamp.get("end"))
        if end < start:
            raise ValueError
        return start, end
    except (TypeError, ValueError):
        return fallback, fallback


def bad_boundary(turn: dict[str, Any]) -> bool:
    return any(bool(turn.get(key)) for key in BAD_KEYS)


def raw_asr_quality(turn: dict[str, Any]) -> tuple[str, list[str], dict[str, Any]]:
    segments = turn.get("raw_segments") or []
    no_speech: list[float] = []
    compression: list[float] = []
    avg_logprob: list[float] = []
    word_prob: list[float] = []
    for segment in segments:
        raw = segment.get("raw_asr") or segment
        for key, dest in (("no_speech_prob", no_speech), ("compression_ratio", compression), ("avg_logprob", avg_logprob)):
            try:
                dest.append(float(raw.get(key)))
            except (AttributeError, TypeError, ValueError):
                pass
        for word in raw.get("words") or []:
            try:
                word_prob.append(float(word.get("probability")))
            except (AttributeError, TypeError, ValueError):
                pass
    reasons: list[str] = []
    # These are QA signals, not identity thresholds.  Missing metadata is not
    # interpreted as bad; severe signals are conservatively quarantined.
    if no_speech and max(no_speech) >= 0.985 and len(norm(turn.get("text")).split()) > 2:
        reasons.append("high_no_speech_prob")
    if compression and max(compression) >= 3.5:
        reasons.append("high_compression_ratio")
    if avg_logprob and min(avg_logprob) <= -1.6 and len(norm(turn.get("text")).split()) >= 4:
        reasons.append("low_avg_logprob")
    if word_prob and len(word_prob) >= 3 and sum(word_prob) / len(word_prob) < 0.25:
        reasons.append("low_word_probability")
    return ("REVIEW" if reasons else "PASS"), reasons, {
        "no_speech_prob": max(no_speech) if no_speech else None,
        "compression_ratio": max(compression) if compression else None,
        "avg_logprob": min(avg_logprob) if avg_logprob else None,
        "mean_word_probability": sum(word_prob) / len(word_prob) if word_prob else None,
        "metadata_present": bool(no_speech or compression or avg_logprob or word_prob),
    }


def read_jsonl(path: Path):
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


def load_fusion() -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(row.get("source_id")), str(row.get("cluster"))): row for row in read_jsonl(FUSION)}


def load_registry() -> dict[str, dict[str, Any]]:
    try:
        payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    return {str(row.get("source_id")): row for row in payload.get("sources") or [] if row.get("source_id")}


def load_manifest() -> dict[str, dict[str, Any]]:
    result = {}
    for row in read_jsonl(MANIFEST):
        if row.get("source_id"):
            result[str(row["source_id"])] = row
    return result


def load_old_target_ids() -> set[str]:
    return {str(turn_id) for row in read_jsonl(OLD_STRUCTURAL) for turn_id in (row.get("target_turn_ids") or [])}


def canonical_recordings() -> dict[str, str]:
    result: dict[str, str] = {}
    for row in read_jsonl(ROOT / "reports" / "recording_content_clusters.jsonl"):
        cluster = str(row.get("recording_cluster_id") or "")
        for source_id in row.get("source_ids") or []:
            result[str(source_id)] = cluster
    return result


def percentile(values: list[float], fraction: float, default: float) -> float:
    values = sorted(x for x in values if math.isfinite(x))
    if not values:
        return default
    index = min(len(values) - 1, max(0, int(round((len(values) - 1) * fraction))))
    return values[index]


def snapshot(turn: dict[str, Any]) -> dict[str, Any]:
    return {
        "turn_id": turn.get("turn_id"),
        "speaker": turn.get("speaker"),
        "identity": turn.get("identity", "UNKNOWN"),
        "speaker_confidence": turn.get("speaker_confidence"),
        "timestamp": turn.get("timestamp"),
        "text": norm(turn.get("text")),
        "asr_quality": turn.get("asr_quality"),
        "bad_boundary": bad_boundary(turn),
    }


def merge_messages(turns: list[dict[str, Any]], role: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for turn in turns:
        value = norm(turn.get("text"))
        if not value:
            continue
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"] += "\n" + value
        else:
            messages.append({"role": role, "content": value})
    return messages


def prepare_timeline(source_id: str, payload: dict[str, Any], fusion: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(payload.get("turns") or []):
        turn = dict(raw)
        mapped = fusion.get((source_id, str(turn.get("speaker") or ""))) or {}
        turn["identity"] = str(mapped.get("identity") or turn.get("identity") or "UNKNOWN")
        turn["_index"] = index
        turn["_start"], turn["_end"] = bounds(turn, float(index))
        turn["text_qa"], turn["text_qa_reasons"] = text_quality(turn.get("text"))
        turn["asr_quality"], turn["asr_quality_reasons"], turn["asr_metadata"] = raw_asr_quality(turn)
        result.append(turn)
    for index, turn in enumerate(result):
        turn["_gap_from_previous"] = 0.0 if index == 0 else max(0.0, turn["_start"] - result[index - 1]["_end"])
    return result


def context_episode(turns: list[dict[str, Any]], target_index: int, max_gap: float, max_turns: int = 3) -> tuple[list[int], str | None]:
    if target_index <= 0:
        return [], "NO_OBSERVABLE_TRIGGER"
    immediate = turns[target_index - 1]
    if immediate.get("identity") in TARGETS:
        return [], "NO_NEW_OBSERVABLE_TRIGGER"
    if bad_boundary(immediate) or immediate.get("text_qa") != "PASS" or immediate.get("asr_quality") == "REVIEW":
        return [], "CONTEXT_EPISODE_BROKEN"
    selected = [target_index - 1]
    cursor = target_index - 2
    while cursor >= 0 and len(selected) < max_turns:
        current = turns[cursor]
        next_turn = turns[cursor + 1]
        if current.get("identity") in TARGETS or bad_boundary(current) or current.get("text_qa") != "PASS" or current.get("asr_quality") == "REVIEW":
            break
        if current.get("event_boundary") or next_turn.get("event_boundary"):
            break
        if max(0.0, next_turn["_start"] - current["_end"]) > max_gap:
            break
        selected.append(cursor)
        cursor -= 1
    selected.reverse()
    return selected, None


def response_continuation(previous: dict[str, Any], current: dict[str, Any], max_gap: float) -> tuple[bool, str]:
    if previous.get("speaker") != current.get("speaker") or current.get("identity") not in TARGETS:
        return False, "SPLIT_NEW_SPEECH_ACT"
    if bad_boundary(previous) or bad_boundary(current) or previous.get("event_boundary") or current.get("event_boundary"):
        return False, "SPLIT_EVENT"
    if previous.get("text_qa") != "PASS" or current.get("text_qa") != "PASS" or current.get("asr_quality") == "REVIEW":
        return False, "UNCLEAR"
    gap = max(0.0, current["_start"] - previous["_end"])
    if gap > max_gap:
        return False, "SPLIT_HIDDEN_TRIGGER"
    prev_text = norm(previous.get("text"))
    cur_text = norm(current.get("text"))
    first = cur_text.lower().split(" ")[0].strip("'\".,!?;:") if cur_text else ""
    if cur_text[:1].islower() or first in {"and", "or", "because", "which", "that", "to", "then", "than"} or not re.search(r"[.!?]\s*$", prev_text):
        return True, "MERGE_SYNTACTIC_CONTINUATION"
    # A very tight fragment with a short previous segment is often diarization
    # splitting, but a completed sentence after a pause is left as a boundary.
    if gap <= 0.55 and len(prev_text.split()) <= 8:
        return True, "MERGE_TIGHT_FRAGMENT"
    return False, "SPLIT_NEW_SPEECH_ACT"


def build(args: argparse.Namespace) -> dict[str, Any]:
    fusion = load_fusion()
    registry = load_registry()
    manifest = load_manifest()
    old_targets = load_old_target_ids()
    recordings = canonical_recordings()
    timelines: dict[str, list[dict[str, Any]]] = {}
    context_gaps: list[float] = []
    response_gaps: list[float] = []
    for path in sorted(TIMELINES.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        source_id = str(payload.get("source_id") or path.stem)
        turns = prepare_timeline(source_id, payload, fusion)
        timelines[source_id] = turns
        for i in range(1, len(turns)):
            gap = turns[i]["_gap_from_previous"]
            if gap <= 30 and turns[i - 1].get("identity") not in TARGETS and turns[i].get("identity") not in TARGETS and not bad_boundary(turns[i - 1]) and not bad_boundary(turns[i]):
                context_gaps.append(gap)
            if turns[i - 1].get("identity") in TARGETS and turns[i].get("identity") in TARGETS and turns[i - 1].get("speaker") == turns[i].get("speaker") and gap <= 10:
                response_gaps.append(gap)
    context_gap = max(3.0, min(12.0, percentile(context_gaps, 0.95, 6.0)))
    response_gap = max(0.8, min(3.0, percentile(response_gaps, 0.90, 1.5)))
    rows: list[dict[str, Any]] = []
    funnel = Counter()
    reason_counts = Counter()
    per_source = Counter()
    recovered_previous_rejects = 0
    for source_id, turns in sorted(timelines.items()):
        title = str((manifest.get(source_id) or registry.get(source_id) or {}).get("title") or "")
        recording_id = recordings.get(source_id) or "rc_source_" + hashlib.sha1(source_id.encode()).hexdigest()[:16]
        for index, target in enumerate(turns):
            funnel["raw_target_turns_seen"] += int(target.get("identity") in TARGETS)
            if target.get("identity") not in TARGETS:
                continue
            if float(target.get("speaker_confidence") or 0.0) < 0.70:
                reason_counts["IDENTITY_UNCERTAIN"] += 1
                continue
            if bad_boundary(target):
                reason_counts["RESPONSE_EPISODE_UNCLEAR"] += 1
                continue
            if target.get("text_qa") != "PASS" or target.get("asr_quality") == "REVIEW":
                reason_counts["ASR_GARBAGE"] += 1
                continue
            # A target immediately following target speech has no newly
            # observed trigger. It is offered later only as a continuation of
            # the first episode, never as a second prompt.
            if index > 0 and turns[index - 1].get("identity") in TARGETS:
                reason_counts["NO_NEW_OBSERVABLE_TRIGGER"] += 1
                continue
            context_indices, failure = context_episode(turns, index, context_gap)
            if failure:
                reason_counts[failure] += 1
                continue
            target_indices = [index]
            continuation_reasons: list[str] = []
            cursor = index + 1
            while cursor < len(turns) and turns[cursor].get("identity") in TARGETS:
                can_merge, reason = response_continuation(turns[target_indices[-1]], turns[cursor], response_gap)
                if not can_merge:
                    break
                target_indices.append(cursor)
                continuation_reasons.append(reason)
                cursor += 1
            context_turns = [turns[i] for i in context_indices]
            target_turns = [turns[i] for i in target_indices]
            if any(turn.get("speaker") == target.get("speaker") for turn in context_turns):
                reason_counts["TARGET_SELF_FRAGMENT"] += 1
                continue
            context_anchor_id = str(context_turns[-1].get("turn_id"))
            target_ids = [str(turn.get("turn_id")) for turn in target_turns]
            episode_id = hashlib.sha256((source_id + "|" + "|".join(target_ids)).encode()).hexdigest()[:20]
            sample_id = hashlib.sha256((source_id + "|" + context_anchor_id + "|" + episode_id).encode()).hexdigest()[:20]
            old_recovery = any(target_id not in old_targets for target_id in target_ids)
            if old_recovery:
                recovered_previous_rejects += 1
            rows.append({
                "sample_id": sample_id,
                "messages": merge_messages(context_turns, "user") + merge_messages(target_turns, "assistant"),
                "source_id": source_id,
                "canonical_recording_id": recording_id,
                "context_anchor_id": context_anchor_id,
                "context_turn_id": context_anchor_id,
                "context_turn_ids": [str(turn.get("turn_id")) for turn in context_turns],
                "response_episode_id": episode_id,
                "target_turn_ids": target_ids,
                "raw_timeline_indices": context_indices + target_indices,
                "timestamps": {"start": context_turns[0]["_start"], "end": target_turns[-1]["_end"]},
                "speaker_evidence": {
                    "assistant_speaker": target.get("speaker"),
                    "assistant_identity": target.get("identity"),
                    "assistant_confidence": min(float(turn.get("speaker_confidence") or 0.0) for turn in target_turns),
                    "context_identities": [str(turn.get("identity") or "UNKNOWN") for turn in context_turns],
                    "context_speakers": [str(turn.get("speaker") or "") for turn in context_turns],
                },
                "identity": target.get("identity"),
                "identity_confidence": "high" if target.get("identity") == "NEURO_FAMILY_HIGH" else "medium",
                "upstream_evidence_score": min(float(turn.get("speaker_confidence") or 0.0) for turn in target_turns),
                "semantic_quality": "UNKNOWN",
                "context_integrity": "UNKNOWN",
                "transcript_quality": "UNKNOWN",
                "response_episode_integrity": "UNKNOWN",
                "semantic_qa": {"status": "PENDING_MODEL_JUDGE"},
                "transcript_qa": {
                    "status": "STRUCTURAL_PASS",
                    "turns": [snapshot(turn) for turn in context_turns + target_turns],
                    "missing_confidence_not_used_as_quality_evidence": any(turn.get("transcription_confidence") in (None, 0, 0.0) for turn in context_turns + target_turns),
                },
                "episode_boundary": {
                    "context_gap_policy_seconds": context_gap,
                    "response_gap_policy_seconds": response_gap,
                    "continuation_reasons": continuation_reasons,
                    "candidate_response_turn_ids": target_ids,
                },
                "recovery": {"recovered_from_previous_reject": old_recovery, "source_state": "CURRENT_CANONICAL_TIMELINE"},
                "source_metadata": {"title": title, "participants": (manifest.get(source_id) or {}).get("participants") or []},
                "training_candidate": False,
                "evaluation_only": False,
                "pipeline_version": "sft-v2.2-structural-2026-09-13",
            })
            per_source[source_id] += 1
    # Deterministic uniqueness gates are applied before writing Layer A. They
    # are assertions, not silent repair: any collision is recorded and kept
    # out of the candidate pool.
    by_anchor: dict[str, str] = {}
    by_target: dict[str, str] = {}
    unique: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: str(item["sample_id"])):
        anchor = f"{row['canonical_recording_id']}|{row['context_anchor_id']}|{row['response_episode_id']}"
        targets = [str(x) for x in row.get("target_turn_ids") or []]
        if anchor in by_anchor or any(target_id in by_target for target_id in targets):
            reason_counts["STRUCTURAL_DUPLICATE_ANCHOR_OR_TARGET"] += 1
            continue
        by_anchor[anchor] = str(row["sample_id"])
        for target_id in targets:
            by_target[target_id] = str(row["sample_id"])
        unique.append(row)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as handle:
        for row in unique:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "schema_version": "1.0.0",
        "pipeline_version": "sft-v2.2-structural-2026-09-13",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input": {"timeline_files": len(timelines), "fusion_rows": len(fusion), "old_structural_rows": sum(1 for _ in read_jsonl(OLD_STRUCTURAL))},
        "policy": {"assistant_identities": sorted(TARGETS), "context_identities": "any_clean_observable_non_target_or_unknown", "old_artifacts_untouched": True},
        "dynamic_gap_calibration": {"context_gap_p95_clipped_seconds": context_gap, "response_gap_p90_clipped_seconds": response_gap, "context_gap_observations": len(context_gaps), "response_gap_observations": len(response_gaps)},
        "counts": {"raw_target_turns_seen": funnel["raw_target_turns_seen"], "structural_candidates": len(unique), "sources_with_candidates": len(per_source), "recovered_previous_reject_candidates": recovered_previous_rejects, "multi_segment_response_episodes": sum(1 for row in unique if len(row.get("target_turn_ids") or []) > 1)},
        "rejection_reasons": dict(reason_counts),
        "source_candidate_counts": dict(per_source),
        "output": str(OUT.relative_to(ROOT)).replace("\\", "/"),
    }
    FUNNEL.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUT))
    args = parser.parse_args()
    OUT = Path(args.output)
    if not OUT.is_absolute():
        OUT = ROOT / OUT
    report = build(args)
    summary = dict(report["counts"])
    summary["rejection_reasons"] = report["rejection_reasons"]
    summary["output"] = str(OUT)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
