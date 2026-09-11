from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from manifest_tools import ROOT, read_jsonl, safe_name, write_master


def features(text: str) -> dict:
    lower = text.lower()
    tokens = re.findall(r"\b[\w'’-]+\b", text)
    return {
        "response_length": len(tokens),
        "sentence_count": len(re.findall(r"[.!?]", text)) or int(bool(text.strip())),
        "self_reference": int(bool(re.search(r"\b(i|me|my|mine|myself|we|our|us)\b", lower))),
        "uncertainty": int(bool(re.search(r"\b(maybe|perhaps|probably|i think|i guess|not sure|might)\b", lower))),
        "repair": int(bool(re.search(r"\b(no,? no|i mean|wait|actually|or rather|sorry)\b", lower))),
        "assertiveness": int(bool(re.search(r"\b(definitely|obviously|of course|must|never|always)\b", lower))),
        "banter": int(bool(re.search(r"\b(lol|lmao|haha|chat|bro|dad|stupid|idiot)\b", lower))),
        "absurd_shift": int(bool(re.search(r"\b(universe|alien|robot|drone|pizza|turtle|moon)\b", lower))),
        "emotional_leak": int(bool(re.search(r"\b(love|hate|angry|sad|scared|cry|embarrass|happy)\b", lower))),
        "identity_reference": int(bool(re.search(r"\b(neuro|evil|vedal|ai|artificial|daughter|sister)\b", lower))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-id", action="append", required=True)
    args = parser.parse_args()
    rows = read_jsonl(ROOT / "manifest" / "master_video_manifest.jsonl")
    raw_out = ROOT / "conversations" / "conversations_raw.jsonl"
    clean_out = ROOT / "conversations" / "conversations_cleaned.jsonl"
    reject_out = ROOT / "datasets" / "rejected_segments.jsonl"
    stat = Counter()
    raw_records, clean_records, rejects = [], [], []
    for source_id in args.source_id:
        raw_path = ROOT / "raw_subtitles" / "library_of_ladev" / f"{safe_name(source_id)}.json"
        if not raw_path.exists():
            continue
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        segments = payload["payload"]["data"]["result"].get("subtitles", [])
        normalized_path = ROOT / "normalized_subtitles" / f"{safe_name(source_id)}.library_of_ladev.jsonl"
        normalized_path.parent.mkdir(parents=True, exist_ok=True)
        with normalized_path.open("w", encoding="utf-8", newline="\n") as handle:
            for item in segments:
                handle.write(json.dumps({"start": item.get("startTime"), "end": item.get("endTime"), "text": item.get("text", ""), "raw_text": item.get("text", ""), "source": "Library of Ladev public API", "source_segment_id": item.get("subtitleId")}, ensure_ascii=False) + "\n")
        windows = []
        current = []
        previous_end = None
        for item in segments:
            start, end = item.get("startTime"), item.get("endTime") or item.get("startTime")
            if previous_end is not None and start - previous_end > 45:
                if len(current) >= 4:
                    windows.extend([current[i:i+20] for i in range(0, len(current), 16) if len(current[i:i+20]) >= 4])
                current = []
            current.append(item)
            previous_end = end
        if len(current) >= 4:
            windows.extend([current[i:i+20] for i in range(0, len(current), 16) if len(current[i:i+20]) >= 4])
        for index, window in enumerate(windows):
            turns = [{
                "timestamp": {"start": item.get("startTime"), "end": item.get("endTime")},
                "speaker": "UNKNOWN",
                "speaker_confidence": None,
                "text": item.get("text", ""),
                "source": {"type": "human_or_curated_transcript_index", "attribution": "Library of Ladev public API", "segment_id": item.get("subtitleId"), "raw_path": str(raw_path.relative_to(ROOT))},
                "turn_boundary": "source_subtitle_segment",
                "interruption": False,
                "overlap": False,
                "uncertain_transcription": False,
            } for item in window]
            joined = " ".join(t["text"] for t in turns)
            f = features(joined)
            record = {
                "conversation_id": f"youtube:{source_id}:ladev:{index:05d}",
                "source_video_id": source_id,
                "source_url": f"https://www.youtube.com/watch?v={source_id}",
                "stream_date_if_known": next((r.get("stream_date_if_known") or r.get("upload_date") for r in rows if r.get("source_id") == source_id), None),
                "participants": [],
                "persona_scope": next((r.get("neuro_or_evil", "uncertain") for r in rows if r.get("source_id") == source_id), "uncertain"),
                "turns": turns,
                "quality": {**f, "multi_turn_value": min(1.0, len(turns) / 10.0), "asr_confidence": None, "speaker_confidence": 0.0},
                "training_candidate": False,
                "candidate_reason": "speaker diarization and Neuro/Evil/Vedal mapping pending",
            }
            raw_records.append(record)
            clean_records.append(record)
            rejects.append({"conversation_id": record["conversation_id"], "reason": "speaker_mapping_pending", "record": record})
            stat.update(f)
            stat["segments"] += 1
        for row in rows:
            if row.get("source_id") == source_id:
                row.update({"normalized_subtitle_path": str(normalized_path.relative_to(ROOT)), "transcript_conversation_status": "done", "transcript_conversation_count": len(windows)})
    for path, records in ((raw_out, raw_records), (clean_out, clean_records), (reject_out, rejects)):
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    stats_path = ROOT / "reports" / "forensic_style_stats.json"
    stats_path.write_text(json.dumps({"schema_version": "0.1.0", "scope": "Library of Ladev transcript records", "counts": dict(stat)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_master(rows)
    print(json.dumps({"source_ids": args.source_id, "raw_records": len(raw_records), "rejected": len(rejects), "segments": dict(stat)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

