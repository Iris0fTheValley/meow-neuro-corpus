from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from manifest_tools import ROOT, safe_name, write_json


def lean_segment(row: dict, index: int) -> dict:
    return {
        "turn_id": index,
        "source_segment_id": row.get("source_segment_id", row.get("id", index)),
        "speaker": "UNKNOWN",
        "speaker_identity": "UNKNOWN",
        "speaker_confidence": 0.0,
        "text": str(row.get("text") or "").strip(),
        "timestamp": {
            "start": row.get("start"),
            "end": row.get("end"),
        },
        "transcription_confidence": None,
        "cleaning_status": row.get("cleaning_status", "unknown"),
        "cleaning_reasons": row.get("cleaning_reasons") or [],
    }


def context_dependency(segments: list[dict]) -> str:
    if not segments:
        return "unknown"
    first = str(segments[0].get("text") or "").lower()
    if any(first.startswith(x) for x in ("yes", "no", "why", "that", "it", "he", "she", "because", "yeah")):
        return "likely_context_dependent"
    if len(first.split()) <= 3:
        return "possible_context_dependent"
    return "low_or_unknown"


def make_window(source: dict, segments: list[dict], start: int, end: int, kind: str, reason: str) -> dict:
    selected = segments[start:end]
    preceding = segments[start - 1] if start > 0 else None
    following = segments[end] if end < len(segments) else None
    return {
        "conversation_id": f"{source['source_id']}:{kind}:{start:06d}",
        "schema_version": "0.1.0",
        "status": "TRANSCRIPT_ONLY_SPEAKER_UNKNOWN",
        "training_candidate": False,
        "candidate_grade": "Q",
        "candidate_reason": "public transcript only; speaker attribution and audio alignment pending",
        "source_video_id": source["source_id"],
        "source_url": source.get("source_url"),
        "title": source.get("title"),
        "platform": "twitchtranscripts",
        "participants": ["Neuro", "Evil", "Vedal"],
        "persona_scope": "uncertain",
        "start_time": selected[0]["timestamp"]["start"] if selected else None,
        "end_time": selected[-1]["timestamp"]["end"] if selected else None,
        "preceding_context": preceding,
        "following_context": following,
        "segments": selected,
        "identity_confidence": "unknown",
        "speaker_mapping": "not_attempted_without_audio_and_trusted_reference_bank",
        "transcription_confidence": None,
        "context_dependency": context_dependency(selected),
        "window_reason": reason,
        "window_type": kind,
        "transcript_segment_count": len(selected),
        "source_segment_ids": [x["source_segment_id"] for x in selected],
        "provenance": {
            "raw_path": source.get("raw_path"),
            "normalized_path": source.get("normalized_path"),
            "cleaning_status": "source_specific_cleaning_applied",
            "raw_preserved": True,
        },
    }


def windows_for(source: dict, segments: list[dict]) -> list[dict]:
    output = []
    # These are transcript segments, not inferred conversational turns. Keep
    # them isolated from diarized natural-turn data and never mark trainable.
    for start in range(0, len(segments), 8):
        end = min(len(segments), start + 8)
        if end - start >= 4:
            output.append(make_window(source, segments, start, end, "4_8_segments", "canonical_nonoverlap_transcript_window"))
    for start in range(0, len(segments), 16):
        end = min(len(segments), start + 16)
        if end - start >= 8:
            output.append(make_window(source, segments, start, end, "8_20_segments", "extended_transcript_retrieval"))
    for start in range(0, len(segments), 32):
        end = min(len(segments), start + 32)
        if end - start >= 20:
            output.append(make_window(source, segments, start, end, "20_plus_segments", "long_transcript_retrieval"))
    return output


def main() -> None:
    normalized_dir = ROOT / "normalized_subtitles" / "twitchtranscripts"
    output_dir = ROOT / "conversation_windows" / "twitchtranscripts"
    output_dir.mkdir(parents=True, exist_ok=True)
    combined = []
    per_source = {}
    for path in sorted(normalized_dir.glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        vod_id = str(doc.get("vod_id") or path.stem)
        source = {
            "source_id": f"twitchtranscripts:{vod_id}",
            "source_url": doc.get("source_url"),
            "title": doc.get("title"),
            "raw_path": doc.get("raw_path"),
            "normalized_path": str(path.relative_to(ROOT)),
        }
        segments = [lean_segment(row, i) for i, row in enumerate(doc.get("segments") or []) if str(row.get("text") or "").strip()]
        windows = windows_for(source, segments)
        per_source[source["source_id"]] = {
            "vod_id": vod_id,
            "title": source["title"],
            "segment_count": len(segments),
            "window_count": len(windows),
            "training_candidate_count": 0,
            "speaker_mapping": "unknown",
        }
        output_path = output_dir / f"{safe_name(vod_id)}.json"
        write_json(output_path, {
            "schema_version": "0.1.0",
            "source_id": source["source_id"],
            "status": "TRANSCRIPT_ONLY_SPEAKER_UNKNOWN",
            "raw_preserved": True,
            "windows": windows,
        })
        combined.extend(windows)

    combined_path = ROOT / "conversations" / "twitchtranscript_conversations_anonymous.jsonl"
    combined_path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in combined), encoding="utf-8")
    report = {
        "schema_version": "0.1.0",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": "CONTINUE_WORKING_TRANSCRIPT_ONLY_SPEAKER_UNKNOWN",
        "source_count": len(per_source),
        "window_count": len(combined),
        "window_type_counts": dict(Counter(x["window_type"] for x in combined)),
        "training_candidate_count": 0,
        "identity_mapping": "not_attempted_without_audio_and_trusted_reference_bank",
        "raw_preserved": True,
        "cleaning_report": "reports/twitchtranscript_cleaning.json",
        "per_source": per_source,
    }
    write_json(ROOT / "reports" / "twitchtranscript_conversation_progress.json", report)
    print(json.dumps({
        "source_count": report["source_count"],
        "window_count": report["window_count"],
        "window_type_counts": report["window_type_counts"],
        "training_candidate_count": 0,
    }, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
