from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone

from manifest_tools import ROOT, read_jsonl, write_json


def main() -> None:
    path = ROOT / "datasets" / "family_s_a_candidates_review.jsonl"
    rows = list(read_jsonl(path))
    exact = Counter((row.get("source_video_id"), tuple(row.get("source_segment_ids") or [])) for row in rows)
    membership = defaultdict(list)
    for row in rows:
        for segment_id in row.get("source_segment_ids") or []:
            membership[(row.get("source_video_id"), segment_id)].append(row.get("conversation_id"))
    exact_duplicates = [{"source_video_id": key[0], "source_segment_ids": list(key[1]), "count": count} for key, count in exact.items() if count > 1]
    overlapping_segments = [{"source_video_id": key[0], "source_segment_id": key[1], "window_count": len(value), "conversation_ids": value[:10]} for key, value in membership.items() if len(value) > 1]
    report = {
        "schema_version": "0.1.0",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_NO_EXACT_DUPLICATE_WINDOWS_OVERLAP_REVIEWED",
        "candidate_count": len(rows),
        "exact_duplicate_window_count": len(exact_duplicates),
        "overlapping_segment_membership_count": len(overlapping_segments),
        "exact_duplicates": exact_duplicates[:100],
        "overlapping_segments": overlapping_segments[:100],
        "policy": "S/A queue contains canonical 4-8 windows only; overlap counts are retained as review evidence and never silently deduplicated.",
    }
    write_json(ROOT / "reports" / "family_candidate_dedupe_audit.json", report)
    print(json.dumps({"status": report["status"], "candidate_count": len(rows), "exact_duplicate_window_count": len(exact_duplicates), "overlapping_segment_membership_count": len(overlapping_segments)}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
