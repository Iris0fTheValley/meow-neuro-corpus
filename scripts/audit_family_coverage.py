from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone

from manifest_tools import ROOT, read_jsonl, write_json


def main() -> None:
    manifest = {str(row.get("source_id")): row for row in read_jsonl(ROOT / "manifest" / "master_video_manifest.jsonl")}
    candidates = list(read_jsonl(ROOT / "datasets" / "family_training_candidates_review.jsonl"))
    source_stats = defaultdict(lambda: {"windows": 0, "family_turns": 0, "vedal_turns": 0, "unknown_turns": 0, "grades": Counter(), "platform": "unknown"})
    identity = Counter()
    grades = Counter()
    platform = Counter()
    for row in candidates:
        source_id = str(row.get("source_video_id"))
        stats = row.get("stats") or {}
        item = source_stats[source_id]
        item["windows"] += 1
        item["family_turns"] += int(stats.get("family_turn_count") or 0)
        item["vedal_turns"] += int(stats.get("vedal_turn_count") or 0)
        item["unknown_turns"] += int(stats.get("unknown_turn_count") or 0)
        item["grades"][str(row.get("candidate_grade"))] += 1
        item["platform"] = str(manifest.get(source_id, {}).get("source_platform") or "unknown")
        grades[str(row.get("candidate_grade"))] += 1
        platform[item["platform"]] += 1
        identity["NEURO_FAMILY"] += int(stats.get("family_turn_count") or 0)
        identity["VEDAL"] += int(stats.get("vedal_turn_count") or 0)
        identity["UNKNOWN"] += int(stats.get("unknown_turn_count") or 0)
    normalized_sources = []
    for source_id, item in source_stats.items():
        normalized_sources.append({
            "source_id": source_id,
            "platform": item["platform"],
            "windows": item["windows"],
            "family_turns": item["family_turns"],
            "vedal_turns": item["vedal_turns"],
            "unknown_turns": item["unknown_turns"],
            "grade_counts": dict(item["grades"]),
        })
    normalized_sources.sort(key=lambda row: (-row["windows"], row["source_id"]))
    report = {
        "schema_version": "0.1.0",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": "FAMILY_PROXY_COVERAGE_AUDIT_IDENTITY_FINAL_REVIEW_PENDING",
        "window_count": len(candidates),
        "source_count": len(normalized_sources),
        "identity_turn_counts": dict(identity),
        "grade_counts": dict(grades),
        "platform_window_counts": dict(platform),
        "top_sources_by_window_count": normalized_sources[:25],
        "training_candidate_count": 0,
        "coverage_gaps": [
            "semantic state/affect coverage is not yet annotated",
            "negative OTHER coverage remains incomplete",
            "proxy mapping still needs final gold review",
        ],
        "policy": "Coverage reports preserve unknown and do not infer identity from metadata.",
    }
    write_json(ROOT / "reports" / "family_coverage_audit.json", report)
    write_json(ROOT / "coverage" / "family_coverage_audit.json", report)
    print(json.dumps({"status": report["status"], "window_count": report["window_count"], "source_count": report["source_count"], "identity_turn_counts": dict(identity), "grade_counts": dict(grades), "training_candidate_count": 0}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
