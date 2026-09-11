from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone

from manifest_tools import ROOT, read_jsonl, write_json


WORD = re.compile(r"\b[\w'’-]+\b", re.UNICODE)


def features(row: dict) -> dict:
    turns = row.get("turns") or []
    text = " ".join(str(turn.get("text") or "") for turn in turns).strip()
    lower = text.lower()
    words = WORD.findall(text)
    sentences = [x for x in re.split(r"(?<=[.!?])\s+", text) if x.strip()]
    return {
        "conversation_id": row.get("conversation_id"),
        "source_video_id": row.get("source_video_id"),
        "window_type": row.get("window_type"),
        "annotation_layer": "cheap_surface_only",
        "speaker_identity_status": "unknown_not_mapped",
        "training_candidate": False,
        "char_count": len(text),
        "word_count": len(words),
        "sentence_count": len(sentences),
        "turn_count": len(turns),
        "first_person_marker_count": len(re.findall(r"\b(i|me|my|mine|myself|we|our|us)\b", lower)),
        "second_person_marker_count": len(re.findall(r"\b(you|your|yours|yourself)\b", lower)),
        "question_marker_count": text.count("?") + len(re.findall(r"\b(why|how|what|when|where|who)\b", lower)),
        "hedge_marker_count": len(re.findall(r"\b(maybe|perhaps|probably|i think|i guess|not sure|might|could)\b", lower)),
        "assertion_marker_count": len(re.findall(r"\b(definitely|obviously|of course|must|never|always|certainly)\b", lower)),
        "repair_marker_count": len(re.findall(r"\b(no,? no|i mean|wait|actually|or rather|sorry|uh|um)\b", lower)),
        "banter_marker_count": len(re.findall(r"\b(lol|lmao|haha|chat|bro|stupid|idiot|dummy)\b", lower)),
        "ai_reference_count": len(re.findall(r"\b(neuro|evil|vedal|ai|artificial intelligence|robot|computer|model)\b", lower)),
        "disfluency_marker_count": len(re.findall(r"\b(uh|um|hmm|er|ah)\b", lower)),
        "response_length_bucket": "short" if len(words) < 40 else "medium" if len(words) < 160 else "long",
        "context_dependency": row.get("context_dependency", "unknown"),
    }


def main() -> None:
    source = ROOT / "conversations" / "natural_conversations_anonymous.jsonl"
    output = ROOT / "forensic_annotations" / "natural_anonymous_surface_features.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    records = [features(row) for row in read_jsonl(source)] if source.exists() else []
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    buckets = Counter(row["response_length_bucket"] for row in records)
    report = {
        "schema_version": "0.1.0",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": "ANONYMOUS_SURFACE_FEATURES_ONLY",
        "source": str(source.relative_to(ROOT)),
        "output": str(output.relative_to(ROOT)),
        "record_count": len(records),
        "training_candidate_count": 0,
        "identity_mapping": "not_attempted",
        "response_length_buckets": dict(buckets),
        "policy": "Cheap lexical features are descriptive only; semantic forensic labels require identity-mapped, QA-passed data.",
    }
    write_json(ROOT / "reports" / "natural_anonymous_surface_features.json", report)
    print(json.dumps({"status": report["status"], "record_count": report["record_count"], "training_candidate_count": 0, "response_length_buckets": dict(buckets)}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
