from __future__ import annotations

import json
from datetime import datetime, timezone

from manifest_tools import ROOT, write_json


def main() -> None:
    progress_path = ROOT / "reports" / "natural_turn_progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
    reviewed = []
    flagged = []
    for row in progress.get("completed", []):
        segments = int(row.get("segment_count") or 0)
        suspicious = int(row.get("suspicious_segments") or 0)
        ratio = round(suspicious / max(1, segments), 6)
        item = {**row, "suspicious_ratio": ratio, "review_status": "pending" if suspicious else "not_flagged"}
        reviewed.append(item)
        if suspicious >= 50 or ratio >= 0.05:
            flagged.append(item)
    report = {
        "schema_version": "0.1.0",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": "REVIEW_PENDING_NO_DELETIONS",
        "source_count": len(reviewed),
        "flagged_source_count": len(flagged),
        "policy": "Suspicious ASR is quarantined for review; this report does not delete, rewrite, or promote data.",
        "thresholds": {"suspicious_segments_at_least": 50, "suspicious_ratio_at_least": 0.05},
        "flagged_sources": sorted(flagged, key=lambda x: (-x["suspicious_ratio"], -x["suspicious_segments"])),
    }
    write_json(ROOT / "reports" / "natural_turn_quality_review.json", report)
    print(json.dumps({
        "status": report["status"],
        "source_count": report["source_count"],
        "flagged_source_count": report["flagged_source_count"],
        "flagged_sources": [{"source_id": x["source_id"], "suspicious_segments": x["suspicious_segments"], "suspicious_ratio": x["suspicious_ratio"]} for x in report["flagged_sources"][:20]],
    }, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
