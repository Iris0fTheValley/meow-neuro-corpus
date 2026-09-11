from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from manifest_tools import ROOT, read_jsonl, safe_name, write_master


def load_asr(row: dict) -> dict:
    return json.loads((ROOT / row["asr_path"]).read_text(encoding="utf-8"))


def style_features(text: str) -> dict:
    lower = text.lower()
    tokens = re.findall(r"\b[\w'’-]+\b", text)
    sentences = re.split(r"(?<=[.!?])\s+", text.strip()) if text.strip() else []
    return {
        "response_length": len(tokens),
        "sentence_count": len([s for s in sentences if s.strip()]),
        "self_reference": int(bool(re.search(r"\b(i|me|my|mine|myself|we|our|us)\b", lower))),
        "uncertainty": int(bool(re.search(r"\b(maybe|perhaps|probably|i think|i guess|not sure|might)\b", lower))),
        "repair": int(bool(re.search(r"\b(no,? no|i mean|wait|actually|or rather|sorry)\b", lower))),
        "assertiveness": int(bool(re.search(r"\b(definitely|obviously|of course|must|never|always)\b", lower))),
        "banter": int(bool(re.search(r"\b(lol|lmao|haha|dad|chat|bro|stupid|idiot)\b", lower))),
        "absurd_shift": int(bool(re.search(r"\b(universe|alien|robot|drone|pizza|turtle|moon)\b", lower))),
        "emotional_leak": int(bool(re.search(r"\b(love|hate|angry|sad|scared|cry|embarrass|happy)\b", lower))),
        "identity_reference": int(bool(re.search(r"\b(neuro|evil|vedal|ai|artificial|daughter|sister)\b", lower))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    rows = read_jsonl(ROOT / "manifest" / "master_video_manifest.jsonl")
    raw_path = ROOT / "conversations" / "conversations_raw.jsonl"
    clean_path = ROOT / "conversations" / "conversations_cleaned.jsonl"
    candidate_path = ROOT / "datasets" / "training_candidates.jsonl"
    rejected_path = ROOT / "datasets" / "rejected_segments.jsonl"
    stats = Counter()
    raw_records, clean_records, candidates, rejected = [], [], [], []
    for row in rows:
        if not row.get("asr_path"):
            continue
        if row.get("conversation_status") == "done" and not args.all:
            continue
        data = load_asr(row)
        turns = []
        for segment in data.get("segments", []):
            text = segment.get("text", "").strip()
            if not text:
                continue
            turns.append({
                "timestamp": {"start": segment["start"], "end": segment["end"]},
                "speaker": "UNKNOWN",
                "speaker_confidence": None,
                "text": text,
                "source": {"type": "local_asr", "path": row["asr_path"], "model": row.get("asr_model")},
                "turn_boundary": "asr_segment",
                "interruption": False,
                "overlap": False,
                "uncertain_transcription": (segment.get("avg_logprob") or -10) < -1.0,
                "raw_segment": segment,
            })
        if not turns:
            continue
        # Preserve contiguous context, but expose windows that are usable for later annotation.
        windows = []
        start = 0
        while start < len(turns):
            end = min(start + 20, len(turns))
            window = turns[start:end]
            if len(window) >= 4:
                windows.append(window)
            if end == len(turns):
                break
            start += 16
        for index, window in enumerate(windows):
            text = " ".join(t["text"] for t in window)
            features = style_features(text)
            record = {
                "conversation_id": f"{row['source_platform']}:{row['source_id']}:{index:04d}",
                "source_video_id": row["source_id"],
                "source_url": row["source_url"],
                "stream_date_if_known": row.get("stream_date_if_known") or row.get("upload_date"),
                "participants": row.get("participants", []),
                "persona_scope": row.get("neuro_or_evil", "uncertain"),
                "turns": window,
                "quality": {**features, "multi_turn_value": min(1.0, len(window) / 10.0), "speaker_confidence": 0.0, "asr_confidence": max(0.0, min(1.0, 1.0 + sum((t["raw_segment"].get("avg_logprob") or -2) for t in window) / len(window) / 2))},
                "training_candidate": False,
                "candidate_reason": "speaker mapping pending; raw ASR-only benchmark record",
            }
            raw_records.append(record)
            clean_records.append(record)
            rejected.append({"conversation_id": record["conversation_id"], "reason": "speaker_mapping_pending", "record": record})
            stats.update(features)
            stats["segments"] += 1
        row.update({"conversation_status": "done", "conversation_raw_count": len(windows), "conversation_raw_path": "conversations/conversations_raw.jsonl", "training_candidates_status": "pending_speaker_mapping"})
    for path, records in ((raw_path, raw_records), (clean_path, clean_records), (candidate_path, candidates), (rejected_path, rejected)):
        if records:
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_master(rows)
    stats_path = ROOT / "reports" / "forensic_style_stats.json"
    stats_path.write_text(json.dumps({"schema_version": "0.1.0", "scope": "benchmark ASR conversation records", "counts": dict(stats)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"raw_records": len(raw_records), "rejected": len(rejected), "stats": dict(stats)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

