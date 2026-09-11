from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone

from manifest_tools import ROOT, read_jsonl, write_json


def main() -> None:
    features_path = ROOT / "forensic_annotations" / "natural_anonymous_surface_features.jsonl"
    rows = list(read_jsonl(features_path)) if features_path.exists() else []
    manifest = {str(row.get("source_id")): row for row in read_jsonl(ROOT / "manifest" / "master_video_manifest.jsonl")}
    source_platform = Counter(str(manifest.get(str(row.get("source_video_id")), {}).get("source_platform") or "unknown") for row in rows)
    window_type = Counter(str(row.get("window_type") or "unknown") for row in rows)
    length = Counter(str(row.get("response_length_bucket") or "unknown") for row in rows)
    context = Counter(str(row.get("context_dependency") or "unknown") for row in rows)
    turn_count = Counter("4-8" if int(row.get("turn_count") or 0) <= 8 else "9-20" if int(row.get("turn_count") or 0) <= 20 else "21+" for row in rows)
    report = {
        "schema_version": "0.1.0",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": "ANONYMOUS_IDENTITY_PENDING",
        "record_count": len(rows),
        "source_count": len({row.get("source_video_id") for row in rows}),
        "training_candidate_count": 0,
        "identity_distribution": "deferred_until_gold_validated_mapping",
        "source_platform_counts": dict(source_platform),
        "window_type_counts": dict(window_type),
        "response_length_buckets": dict(length),
        "turn_count_buckets": dict(turn_count),
        "context_dependency_counts": dict(context),
        "coverage_gaps": [
            "Neuro/Evil/Vedal identity distributions unavailable",
            "identity-specific forensic distributions unavailable",
            "state and affect coverage require validated identity mapping and semantic annotation",
        ],
        "policy": "Metadata participant names are not used as speaker labels; unknown remains unknown.",
    }
    write_json(ROOT / "reports" / "anonymous_coverage_audit.json", report)
    write_json(ROOT / "coverage" / "anonymous_coverage_audit.json", report)
    print(json.dumps({"status": report["status"], "record_count": report["record_count"], "source_count": report["source_count"], "training_candidate_count": 0, "coverage_gaps": len(report["coverage_gaps"])}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
