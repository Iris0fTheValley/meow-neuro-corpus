from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from manifest_tools import ROOT, read_jsonl, safe_name, write_json


def context_dependency(turns: list[dict]) -> str:
    if not turns:
        return "unknown"
    first = str(turns[0].get("text") or "").lower()
    if any(first.startswith(x) for x in ("yes", "no", "why", "that", "it", "he", "she", "because", "yeah")):
        return "likely_context_dependent"
    if len(first.split()) <= 3:
        return "possible_context_dependent"
    return "low_or_unknown"


def lean_turn(turn: dict) -> dict:
    return {k: v for k, v in turn.items() if k != "raw_segments"}


def make_window(row: dict, turns: list[dict], start: int, end: int, kind: str, reason: str) -> dict:
    selected = [lean_turn(x) for x in turns[start:end]]
    preceding = lean_turn(turns[start - 1]) if start > 0 else None
    following = lean_turn(turns[end]) if end < len(turns) else None
    return {
        "conversation_id": f"{row['source_id']}:{kind}:{start:06d}",
        "schema_version": "0.1.0",
        "status": "ANONYMOUS_IDENTITY_PENDING",
        "training_candidate": False,
        "candidate_grade": "Q",
        "source_video_id": row["source_id"],
        "source_url": row.get("source_url"),
        "stream_date": row.get("stream_date_if_known") or row.get("upload_date"),
        "participants": row.get("participants") or [],
        "start_time": selected[0]["timestamp"]["start"] if selected else None,
        "end_time": selected[-1]["timestamp"]["end"] if selected else None,
        "preceding_context": preceding,
        "following_context": following,
        "turns": selected,
        "identity_confidence": "unknown",
        "transcription_confidence": round(sum(float(x.get("transcription_confidence") or 0) for x in selected) / max(1, len(selected)), 4),
        "context_dependency": context_dependency(selected),
        "window_reason": reason,
        "window_type": kind,
        "natural_turn_count": len(selected),
        "source_segment_ids": [sid for x in selected for sid in x.get("source_segment_ids", [])],
    }


def windows_for(row: dict, turns: list[dict]) -> list[dict]:
    output = []
    # Canonical non-overlapping primary windows.  These are the only windows
    # that could later be considered for training after identity validation.
    step = 8
    for start in range(0, len(turns), step):
        end = min(len(turns), start + step)
        if end - start >= 4:
            output.append(make_window(row, turns, start, end, "4_8_turns", "canonical_nonoverlap_primary"))
    # Larger retrieval/context windows are intentionally separate and remain
    # excluded from training until dedupe and identity QA are complete.
    for start in range(0, len(turns), 16):
        end = min(len(turns), start + 16)
        if end - start >= 8:
            output.append(make_window(row, turns, start, end, "8_20_turns", "extended_context_retrieval"))
    for start in range(0, len(turns), 32):
        end = min(len(turns), start + 32)
        if end - start >= 20:
            output.append(make_window(row, turns, start, end, "20_plus_turns", "long_context_retrieval"))
    return output


def main() -> None:
    progress_path = ROOT / "reports" / "natural_turn_progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    source_ids = {str(x["source_id"]) for x in progress.get("completed", [])}
    rows = {str(x["source_id"]): x for x in read_jsonl(ROOT / "manifest" / "master_video_manifest.jsonl")}
    out_dir = ROOT / "conversation_windows"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_windows = []
    per_source = {}
    for source_id in sorted(source_ids):
        row = rows.get(source_id)
        timeline_path = ROOT / "unique_timelines" / f"{safe_name(source_id)}.json"
        if not row or not timeline_path.exists():
            continue
        timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
        windows = windows_for(row, timeline.get("turns") or [])
        per_source[source_id] = {"windows": len(windows), "turns": len(timeline.get("turns") or [])}
        (out_dir / f"{safe_name(source_id)}.json").write_text(json.dumps({"source_id": source_id, "status": "ANONYMOUS_IDENTITY_PENDING", "windows": windows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        all_windows.extend(windows)

    combined = ROOT / "conversations" / "natural_conversations_anonymous.jsonl"
    combined.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in all_windows), encoding="utf-8")
    counts = Counter(x["window_type"] for x in all_windows)
    report = {
        "schema_version": "0.1.0",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": "CONTINUE_WORKING_ANONYMOUS_IDENTITY_PENDING",
        "source_count": len(per_source),
        "window_count": len(all_windows),
        "window_type_counts": dict(counts),
        "training_candidate_count": 0,
        "dedupe_status": "source-segment provenance recorded; cross-source dedupe pending",
        "identity_mapping": "not_attempted_without_trusted_reference_bank",
        "per_source": per_source,
    }
    write_json(ROOT / "reports" / "natural_conversation_progress.json", report)
    print(json.dumps({"source_count": report["source_count"], "window_count": report["window_count"], "window_type_counts": dict(counts), "training_candidate_count": 0}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
