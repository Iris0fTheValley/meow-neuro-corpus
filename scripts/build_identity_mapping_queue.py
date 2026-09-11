from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone

from manifest_tools import ROOT, read_jsonl, safe_name, write_json


def main() -> None:
    progress = json.loads((ROOT / "reports" / "natural_turn_progress.json").read_text(encoding="utf-8"))
    queue = []
    source_count = 0
    for item in progress.get("completed", []):
        source_id = str(item["source_id"])
        path = ROOT / "unique_timelines" / f"{safe_name(source_id)}.json"
        if not path.exists():
            continue
        timeline = json.loads(path.read_text(encoding="utf-8"))
        clusters = defaultdict(lambda: {"turn_count": 0, "segment_count": 0, "duration": 0.0, "confidence_sum": 0.0})
        for turn in timeline.get("turns") or []:
            cluster = str(turn.get("speaker") or "UNKNOWN")
            item_stats = clusters[cluster]
            segment_ids = turn.get("source_segment_ids") or []
            item_stats["turn_count"] += 1
            item_stats["segment_count"] += len(segment_ids)
            timestamp = turn.get("timestamp") or {}
            item_stats["duration"] += max(0.0, float(timestamp.get("end") or 0) - float(timestamp.get("start") or 0))
            item_stats["confidence_sum"] += float(turn.get("speaker_confidence") or 0)
        for cluster, stats in sorted(clusters.items()):
            queue.append({
                "mapping_id": f"{source_id}:{cluster}",
                "source_id": source_id,
                "cluster": cluster,
                "status": "PENDING_TRUSTED_REFERENCE",
                "identity": "UNKNOWN",
                "identity_confidence": "unknown",
                "best_score": None,
                "second_best_identity": None,
                "second_best_score": None,
                "margin": None,
                "matched_prototype": None,
                "reference_bank_version": "provisional-2026-09-11-v1",
                "evidence": {"source": "anonymous_diarization_cluster", "gold_validation": "pending"},
                "turn_count": stats["turn_count"],
                "segment_count": stats["segment_count"],
                "duration_seconds": round(stats["duration"], 3),
                "mean_speaker_confidence": round(stats["confidence_sum"] / max(1, stats["turn_count"]), 4),
            })
        source_count += 1
    output = ROOT / "identity_results" / "identity_mapping_pending.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in queue), encoding="utf-8")
    report = {
        "schema_version": "0.1.0",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PENDING_TRUSTED_REFERENCE",
        "source_count": source_count,
        "cluster_count": len(queue),
        "mapped_cluster_count": 0,
        "identity_counts": {"UNKNOWN": len(queue)},
        "output": str(output.relative_to(ROOT)),
        "policy": "No title or participant metadata is used as a speaker label; identity remains UNKNOWN until fixed gold verification is available.",
    }
    write_json(ROOT / "reports" / "identity_mapping_queue.json", report)
    print(json.dumps({"status": report["status"], "source_count": source_count, "cluster_count": len(queue), "mapped_cluster_count": 0}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
