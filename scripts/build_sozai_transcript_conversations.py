from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from manifest_tools import ROOT, read_jsonl, write_master


def features(text: str) -> dict:
    low = text.lower()
    return {
        "response_length": len(re.findall(r"\b[\w'’-]+\b", text)),
        "sentence_count": max(1, len(re.findall(r"[.!?]", text))),
        "self_reference": int(bool(re.search(r"\b(i|me|my|mine|myself|we|our|us)\b", low))),
        "uncertainty": int(bool(re.search(r"\b(maybe|perhaps|probably|i think|i guess|not sure|might)\b", low))),
        "repair": int(bool(re.search(r"\b(no,? no|i mean|wait|actually|or rather|sorry)\b", low))),
        "assertiveness": int(bool(re.search(r"\b(definitely|obviously|of course|must|never|always)\b", low))),
        "banter": int(bool(re.search(r"\b(lol|lmao|haha|chat|bro|dad|stupid|idiot)\b", low))),
        "absurd_shift": int(bool(re.search(r"\b(universe|alien|robot|drone|pizza|turtle|moon)\b", low))),
        "emotional_leak": int(bool(re.search(r"\b(love|hate|angry|sad|scared|cry|embarrass|happy)\b", low))),
        "identity_reference": int(bool(re.search(r"\b(neuro|evil|vedal|ai|artificial|daughter|sister)\b", low))),
    }


if __name__ == "__main__":
    rows = read_jsonl(ROOT / "manifest" / "master_video_manifest.jsonl")
    by_id = {r.get("source_id"): r for r in rows}
    records = []
    touched = defaultdict(int)
    for path in sorted((ROOT / "raw_subtitles" / "sozai").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        transcript = data.get("transcript")
        if not isinstance(transcript, list) or len(transcript) < 4:
            continue
        source_ids = data.get("youtube_ids") or []
        if not source_ids:
            continue
        source_id = source_ids[0]
        source_row = by_id.get(source_id)
        if not source_row:
            continue
        # Never replace a richer acoustic/diarized conversation with a
        # transcript-only mirror of the same source.
        if source_row.get("diarization_status") == "done":
            continue
        turns = []
        for item in transcript:
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            turns.append({"timestamp": {"start": float(item.get("start") or 0), "end": float(item.get("end") or item.get("start") or 0)}, "speaker": str(item.get("speaker") or "UNKNOWN"), "speaker_confidence": None, "text": text, "source": {"type": "sozai_public_transcript", "path": str(path.relative_to(ROOT)), "source_url": data.get("source_url"), "attribution": "SozAI public transcript"}, "turn_boundary": "source_transcript_segment", "interruption": False, "overlap": False, "uncertain_transcription": True, "raw_segment": item})
        windows = []
        for start in range(0, len(turns), 16):
            window = turns[start:start + 20]
            if len(window) >= 4:
                windows.append(window)
            if start + 20 >= len(turns):
                break
        for index, window in enumerate(windows):
            text = " ".join(t["text"] for t in window)
            feat = features(text)
            records.append({"conversation_id": f"youtube:{source_id}:sozai:{index:05d}", "source_video_id": source_id, "source_url": source_row["source_url"], "stream_date_if_known": source_row.get("stream_date_if_known") or source_row.get("upload_date"), "participants": source_row.get("participants", []), "persona_scope": source_row.get("neuro_or_evil", "uncertain"), "turns": window, "quality": {**feat, "multi_turn_value": min(1.0, len(window) / 10.0), "asr_confidence": 0.0, "speaker_confidence": 0.0, "context_dependency": 0.8}, "training_candidate": False, "candidate_reason": "public transcript only; original audio and validated speaker mapping pending"})
        touched[source_id] += len(windows)
        source_row.update({"transcript_status": "done", "transcript_path": str(path.relative_to(ROOT)), "transcript_segment_count": len(turns), "transcript_attribution": "SozAI public transcript", "conversation_transcript_only_count": len(windows)})
        if source_row.get("asr_status") not in {"done", "source_transcript"}:
            source_row["asr_status"] = "source_transcript"
        if source_row.get("conversation_status") != "done":
            source_row.update({"conversation_status": "done", "speaker_mapping_status": "uncertain", "speaker_mapping_note": "Transcript speaker labels are source labels only; no acoustic validation.", "training_candidates_status": "rejected_transcript_only"})
    for target in (ROOT / "conversations" / "conversations_raw.jsonl", ROOT / "conversations" / "conversations_cleaned.jsonl", ROOT / "datasets" / "rejected_segments.jsonl"):
        old = [json.loads(x) for x in target.read_text(encoding="utf-8").splitlines() if x.strip()] if target.exists() else []
        source_ids = set(touched)
        old = [x for x in old if x.get("source_video_id") not in source_ids and x.get("record", {}).get("source_video_id") not in source_ids]
        add = [{"conversation_id": r["conversation_id"], "reason": "transcript_only_speaker_unvalidated", "record": r} for r in records] if target.name == "rejected_segments.jsonl" else records
        target.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in old + add), encoding="utf-8")
    write_master(rows)
    print(json.dumps({"sources": len(touched), "records": len(records), "windows_by_source": dict(touched)}, ensure_ascii=False))
