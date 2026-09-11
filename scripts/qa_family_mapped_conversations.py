from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone

from manifest_tools import ROOT, read_jsonl, write_json


ALLOWED = {"NEURO_FAMILY", "VEDAL", "UNKNOWN"}


def main() -> None:
    path = ROOT / "conversations" / "natural_conversations_family_mapped.jsonl"
    if not path.exists():
        raise SystemExit("family mapped conversation file is missing")
    errors = []
    ids = set()
    counts = Counter()
    records = 0
    for line_no, row in enumerate(read_jsonl(path), 1):
        records += 1
        cid = str(row.get("conversation_id") or "")
        if cid in ids:
            errors.append({"line": line_no, "kind": "duplicate_conversation_id", "id": cid})
        ids.add(cid)
        if row.get("training_candidate"):
            errors.append({"line": line_no, "kind": "training_candidate_true"})
        for turn in row.get("turns") or []:
            identity = turn.get("speaker_identity")
            counts[str(identity)] += 1
            if identity not in ALLOWED:
                errors.append({"line": line_no, "kind": "invalid_identity", "identity": identity})
        source_ids = row.get("source_segment_ids") or []
        if len(source_ids) != len(set(source_ids)):
            errors.append({"line": line_no, "kind": "duplicate_source_segment_ids"})
    report = {
        "schema_version": "0.1.0",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not errors else "FAIL",
        "record_count": records,
        "unique_conversation_ids": len(ids),
        "turn_identity_counts": dict(counts),
        "training_candidate_count": 0,
        "error_count": len(errors),
        "errors": errors[:100],
        "policy": "Proxy family mapping is accepted for analysis and QA; S/A promotion remains separately gated.",
    }
    write_json(ROOT / "reports" / "family_mapped_conversation_qa.json", report)
    progress_path = ROOT / "reports" / "family_mapped_conversation_progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        progress["status"] = "FAMILY_MAPPED_PROXY_QA_PASSED_TRAINING_CLOSED" if report["status"] == "PASS" else "FAMILY_MAPPED_PROXY_QA_FAILED"
        progress["qa_report"] = "reports/family_mapped_conversation_qa.json"
        write_json(progress_path, progress)
    print(json.dumps({"status": report["status"], "record_count": records, "turn_identity_counts": dict(counts), "error_count": len(errors), "training_candidate_count": 0}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
