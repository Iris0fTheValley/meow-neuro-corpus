from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone

from manifest_tools import ROOT, read_jsonl, write_json


def main() -> None:
    rows = read_jsonl(ROOT / "conversations" / "retry_transcript_conversations_anonymous.jsonl")
    errors = []
    ids = Counter()
    for index, row in enumerate(rows):
        ids[str(row.get("conversation_id"))] += 1
        turns = row.get("turns") or []
        if not 4 <= len(turns) <= 8:
            errors.append({"index": index, "error": "turn_count_out_of_range"})
        if row.get("training_candidate") is not False or any(turn.get("speaker_identity") != "UNKNOWN" for turn in turns):
            errors.append({"index": index, "error": "identity_or_training_policy_violation"})
        if not row.get("source_segment_ids") or not all(turn.get("text") for turn in turns):
            errors.append({"index": index, "error": "missing_provenance_or_text"})
    duplicate_ids = sum(count - 1 for count in ids.values() if count > 1)
    if duplicate_ids:
        errors.append({"error": "duplicate_conversation_ids", "count": duplicate_ids})
    report = {
        "schema_version": "0.1.0",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not errors else "FAIL",
        "record_count": len(rows),
        "source_count": len({str(row.get("source_video_id")) for row in rows}),
        "training_candidate_count": sum(bool(row.get("training_candidate")) for row in rows),
        "error_count": len(errors),
        "errors": errors[:100],
        "policy": "Transcript-only fallback conversations remain anonymous and training-closed.",
    }
    write_json(ROOT / "reports" / "retry_transcript_conversation_qa.json", report)
    print(json.dumps({"status": report["status"], "record_count": report["record_count"], "source_count": report["source_count"], "training_candidate_count": report["training_candidate_count"], "error_count": report["error_count"]}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
