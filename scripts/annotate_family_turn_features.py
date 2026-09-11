from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone

from manifest_tools import ROOT, read_jsonl, write_json


WORD = re.compile(r"\b[\w'’-]+\b", re.UNICODE)


def annotate(turn: dict, row: dict, index: int) -> dict:
    text = str(turn.get("text") or "").strip()
    lower = text.lower()
    words = WORD.findall(text)
    identity = str(turn.get("speaker_identity") or "UNKNOWN")
    return {
        "conversation_id": row.get("conversation_id"),
        "source_video_id": row.get("source_video_id"),
        "turn_index": index,
        "speaker_identity": identity,
        "identity_confidence": turn.get("identity_confidence", "unknown"),
        "mapping_status": turn.get("mapping_status"),
        "window_type": row.get("window_type"),
        "training_candidate": False,
        "char_count": len(text),
        "word_count": len(words),
        "question": int("?" in text),
        "first_person": int(bool(re.search(r"\b(i|me|my|mine|myself|we|our|us)\b", lower))),
        "second_person": int(bool(re.search(r"\b(you|your|yours|yourself)\b", lower))),
        "hedge": int(bool(re.search(r"\b(maybe|perhaps|probably|i think|i guess|not sure|might|could)\b", lower))),
        "assertion": int(bool(re.search(r"\b(definitely|obviously|of course|must|never|always|certainly)\b", lower))),
        "repair": int(bool(re.search(r"\b(no,? no|i mean|wait|actually|or rather|sorry|uh|um)\b", lower))),
        "banter": int(bool(re.search(r"\b(lol|lmao|haha|chat|bro|stupid|idiot|dummy)\b", lower))),
        "ai_reference": int(bool(re.search(r"\b(neuro|evil|vedal|ai|artificial intelligence|robot|computer|model)\b", lower))),
        "suspicious_transcription": bool(turn.get("suspicious_transcription")),
    }


def main() -> None:
    source = ROOT / "conversations" / "natural_conversations_family_mapped.jsonl"
    output = ROOT / "forensic_annotations" / "family_mapped_turn_surface_features.jsonl"
    records = []
    aggregate = defaultdict(lambda: {"turns": 0, "words": 0, "chars": 0, "first_person": 0, "second_person": 0, "question": 0, "hedge": 0, "assertion": 0, "repair": 0, "banter": 0, "ai_reference": 0, "suspicious": 0})
    for row in read_jsonl(source):
        for index, turn in enumerate(row.get("turns") or []):
            record = annotate(turn, row, index)
            records.append(record)
            bucket = aggregate[record["speaker_identity"]]
            bucket["turns"] += 1
            bucket["words"] += record["word_count"]
            bucket["chars"] += record["char_count"]
            for key in ("first_person", "second_person", "question", "hedge", "assertion", "repair", "banter", "ai_reference"):
                bucket[key] += record[key]
            bucket["suspicious"] += int(record["suspicious_transcription"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    for identity, bucket in aggregate.items():
        turns = max(1, bucket["turns"])
        bucket["mean_words"] = round(bucket["words"] / turns, 4)
        bucket["mean_chars"] = round(bucket["chars"] / turns, 4)
        bucket["suspicious_rate"] = round(bucket["suspicious"] / turns, 6)
    report = {
        "schema_version": "0.1.0",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": "FAMILY_MAPPED_CHEAP_FORENSIC_ONLY",
        "source": str(source.relative_to(ROOT)),
        "output": str(output.relative_to(ROOT)),
        "record_count": len(records),
        "training_candidate_count": 0,
        "identity_aggregate": dict(aggregate),
        "semantic_annotation_status": "deferred_until_final_gold_review",
        "policy": "Statistics are conditional on proxy identity mapping and are descriptive; they are not psychological or semantic labels.",
    }
    write_json(ROOT / "reports" / "family_mapped_surface_forensic_stats.json", report)
    print(json.dumps({"status": report["status"], "record_count": len(records), "identity_aggregate": dict(aggregate), "training_candidate_count": 0}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
