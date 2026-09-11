from __future__ import annotations

"""Build canonical unique timelines and adaptive natural turns.

The existing diarization artifacts remain immutable.  This stage consumes one
ASR/diarization pair once, merges only locally compatible same-speaker ASR
fragments, and writes a separate artifact that is safe for later identity
mapping and conversation construction.
"""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from manifest_tools import ROOT, read_jsonl, safe_name, write_json


TERMINAL = re.compile(r"[.!?。！？…]$|(?:\[.*?\])$")
FRAGMENT_START = re.compile(r"^(?:and|but|or|so|because|then|which|that|to|of|a|the)\b", re.I)


def text_quality(segment: dict) -> tuple[str, float, bool]:
    text = re.sub(r"\s+", " ", str(segment.get("text") or "")).strip()
    avg_logprob = float(segment.get("avg_logprob") or -2.0)
    compression = float(segment.get("compression_ratio") or 0.0)
    no_speech = float(segment.get("no_speech_prob") or 0.0)
    repeated = len(text) >= 40 and len(set(re.findall(r"\w+", text.lower()))) <= 4
    suspicious = bool(compression >= 2.8 or no_speech >= 0.85 or repeated)
    confidence = max(0.0, min(1.0, 0.5 + avg_logprob / 4.0))
    return text, round(confidence, 4), suspicious


def should_merge(previous: dict, current: dict) -> tuple[bool, str]:
    if previous["speaker"] != current["speaker"]:
        return False, "speaker_change"
    gap = max(0.0, current["timestamp"]["start"] - previous["timestamp"]["end"])
    combined_duration = current["timestamp"]["end"] - previous["timestamp"]["start"]
    if combined_duration > 45.0:
        return False, "natural_turn_length_limit"
    if gap <= 0.8:
        return True, "asr_fragment_merge"
    previous_terminal = bool(TERMINAL.search(previous["text"]))
    current_fragment = bool(FRAGMENT_START.search(current["text"])) or len(current["text"].split()) <= 3
    if gap <= 2.5 and (not previous_terminal or current_fragment):
        return True, "same_speaker_adaptive_pause"
    return False, "pause_and_punctuation"


def reconstruct(row: dict) -> tuple[dict, dict]:
    asr = json.loads((ROOT / str(row["asr_path"])).read_text(encoding="utf-8"))
    diar = json.loads((ROOT / str(row["diarization_path"])).read_text(encoding="utf-8"))
    diar_by_id = {str(x.get("source_asr_segment_id")): x for x in diar.get("segments_with_speaker") or []}
    unique_segments: list[dict] = []
    seen_ids: set[str] = set()
    for segment in sorted(asr.get("segments") or [], key=lambda x: float(x.get("start") or 0)):
        segment_id = str(segment.get("id"))
        diarized = diar_by_id.get(segment_id)
        if diarized is None or segment_id in seen_ids:
            continue
        text, transcription_confidence, suspicious = text_quality(segment)
        if not text:
            continue
        seen_ids.add(segment_id)
        unique_segments.append(
            {
                "segment_id": segment_id,
                "timestamp": {"start": float(segment.get("start") or 0), "end": float(segment.get("end") or 0)},
                "speaker": diarized.get("cluster", "UNKNOWN"),
                "speaker_confidence": round(float(diarized.get("speaker_confidence") or 0), 4),
                "text": text,
                "transcription_confidence": transcription_confidence,
                "suspicious_transcription": suspicious,
                "raw_asr": segment,
                "raw_diarization": diarized,
            }
        )

    turns: list[dict] = []
    for segment in unique_segments:
        if not turns:
            turns.append({"segments": [segment], "speaker": segment["speaker"], "boundary_reason": "source_start"})
            continue
        previous = turns[-1]
        previous_view = {
            "speaker": previous["speaker"],
            "timestamp": previous["segments"][-1]["timestamp"],
            "text": previous["segments"][-1]["text"],
        }
        current_view = {"speaker": segment["speaker"], "timestamp": segment["timestamp"], "text": segment["text"]}
        merge, reason = should_merge(previous_view, current_view)
        if merge:
            previous["segments"].append(segment)
        else:
            turns.append({"segments": [segment], "speaker": segment["speaker"], "boundary_reason": reason})

    real_turns = []
    for index, turn in enumerate(turns):
        segments = turn["segments"]
        real_turns.append(
            {
                "turn_id": f"{row['source_id']}:turn:{index:06d}",
                "timestamp": {"start": segments[0]["timestamp"]["start"], "end": segments[-1]["timestamp"]["end"]},
                "speaker": turn["speaker"],
                "speaker_confidence": round(sum(x["speaker_confidence"] for x in segments) / len(segments), 4),
                "transcription_confidence": round(sum(x["transcription_confidence"] for x in segments) / len(segments), 4),
                "suspicious_transcription": any(x["suspicious_transcription"] for x in segments),
                "text": " ".join(x["text"] for x in segments),
                "turn_boundary": turn["boundary_reason"],
                "source_segment_ids": [x["segment_id"] for x in segments],
                "raw_segments": segments,
            }
        )
    timeline = {
        "schema_version": "0.1.0",
        "source_id": row["source_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "ANONYMOUS_CANONICAL_TIMELINE",
        "source_url": row.get("source_url"),
        "source_audio": row.get("audio_path"),
        "source_asr": row.get("asr_path"),
        "source_diarization": row.get("diarization_path"),
        "dedupe_policy": "one unique source ASR segment appears at most once; sliding-window membership is not canonical",
        "segment_count": len(unique_segments),
        "turn_count": len(real_turns),
        "turns": real_turns,
    }
    return timeline, {"segment_count": len(unique_segments), "turn_count": len(real_turns), "suspicious_segments": sum(x["suspicious_transcription"] for x in unique_segments)}


def choose_rows(rows: list[dict], limit: int) -> list[dict]:
    def score(row: dict) -> tuple[float, float]:
        title = str(row.get("title") or "").lower()
        participants = " ".join(str(x).lower() for x in (row.get("participants") or []))
        value = 0.0
        value += 20.0 if any(x in title for x in ("neuro", "evil", "vedal")) else 0.0
        value += 16.0 if any(x in participants for x in ("neuro", "evil", "vedal")) else 0.0
        value += 10.0 if row.get("transcript_available") else 0.0
        value += 4.0 if 3000 <= float(row.get("duration") or 0) <= 16000 else 0.0
        return (-value, float(row.get("duration") or 999999))

    candidates = [
        row for row in rows
        if row.get("diarization_status") == "done"
        and row.get("asr_path") and row.get("diarization_path")
        and (ROOT / str(row["asr_path"])).exists()
        and (ROOT / str(row["diarization_path"])).exists()
    ]
    return sorted(candidates, key=score)[: max(0, limit)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-id")
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()
    rows = read_jsonl(ROOT / "manifest" / "master_video_manifest.jsonl")
    if args.source_id:
        selected = [x for x in rows if x.get("source_id") == args.source_id]
    else:
        selected = choose_rows(rows, args.limit)
    timeline_dir = ROOT / "unique_timelines"
    turns_dir = ROOT / "real_turns"
    report_dir = ROOT / "reports"
    timeline_dir.mkdir(parents=True, exist_ok=True)
    turns_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    completed = []
    for row in selected:
        timeline, metrics = reconstruct(row)
        filename = f"{safe_name(str(row['source_id']))}.json"
        timeline_path = timeline_dir / filename
        turns_path = turns_dir / filename
        timeline_path.write_text(json.dumps(timeline, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        # The unique timeline is the provenance-heavy artifact.  Keep the
        # consumer-facing turns artifact lean so raw ASR/diarization payloads
        # are not duplicated on disk.
        lean_turns = [{k: v for k, v in turn.items() if k != "raw_segments"} for turn in timeline["turns"]]
        turns_path.write_text(json.dumps({**timeline, "turns": lean_turns}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_json(report_dir / f"natural_turn_metrics_{safe_name(str(row['source_id']))}.json", {"source_id": row["source_id"], **metrics, "status": "anonymous_identity_pending"})
        completed.append({"source_id": row["source_id"], **metrics})
    progress = {
        "schema_version": "0.1.0",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": "CONTINUE_WORKING",
        "completed": completed,
        "identity_mapping": "not_attempted_without_trusted_reference_bank",
    }
    write_json(report_dir / "natural_turn_progress.json", progress)
    print(json.dumps({"processed": len(completed), "completed": completed}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
