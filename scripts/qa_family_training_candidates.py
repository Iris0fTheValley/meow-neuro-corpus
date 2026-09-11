from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone

from manifest_tools import ROOT, read_jsonl, write_json


def check(path, label: str, require_sa: bool) -> dict:
    errors = []
    ids = set()
    rows = 0
    grades = Counter()
    for line_no, row in enumerate(read_jsonl(path), 1):
        rows += 1
        cid = str(row.get("conversation_id") or "")
        if cid in ids:
            errors.append({"line": line_no, "kind": "duplicate_conversation_id"})
        ids.add(cid)
        grade = row.get("candidate_grade")
        grades[str(grade)] += 1
        if row.get("training_candidate"):
            errors.append({"line": line_no, "kind": "training_candidate_true"})
        if require_sa and grade not in {"S", "A"}:
            errors.append({"line": line_no, "kind": "non_sa_grade_in_sa_file", "grade": grade})
        if grade not in {"S", "A", "B", "C", "Q"}:
            errors.append({"line": line_no, "kind": "invalid_grade", "grade": grade})
        nested_ids = [turn.get("source_segment_ids") for turn in row.get("turns") or []]
        flat_ids = [item for values in nested_ids for item in (values or [])]
        if len(flat_ids) != len(set(flat_ids)):
            errors.append({"line": line_no, "kind": "duplicate_source_segment_ids"})
        identities = {turn.get("speaker_identity") for turn in row.get("turns") or []}
        if not identities <= {"NEURO_FAMILY", "VEDAL", "UNKNOWN"}:
            errors.append({"line": line_no, "kind": "invalid_identity", "identities": sorted(str(x) for x in identities)})
    return {"label": label, "path": str(path.relative_to(ROOT)), "status": "PASS" if not errors else "FAIL", "rows": rows, "grade_counts": dict(grades), "error_count": len(errors), "errors": errors[:100]}


def main() -> None:
    results = [
        check(ROOT / "datasets" / "family_training_candidates_review.jsonl", "all_grades", False),
        check(ROOT / "datasets" / "family_s_a_candidates_review.jsonl", "s_a_only", True),
    ]
    report = {
        "schema_version": "0.1.0",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(row["status"] == "PASS" for row in results) else "FAIL",
        "training_candidate_count": 0,
        "results": results,
        "policy": "Review candidates are not training candidates until final gold/QA promotion.",
    }
    write_json(ROOT / "reports" / "family_training_candidate_qa.json", report)
    print(json.dumps({"status": report["status"], "training_candidate_count": 0, "results": [{"label": x["label"], "status": x["status"], "rows": x["rows"], "error_count": x["error_count"]} for x in results]}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
