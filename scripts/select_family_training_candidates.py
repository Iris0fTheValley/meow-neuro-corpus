from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone

from manifest_tools import ROOT, read_jsonl, write_json


def grade(row: dict) -> tuple[str, str, dict]:
    turns = row.get("turns") or []
    total = len(turns)
    family = sum(turn.get("speaker_identity") == "NEURO_FAMILY" for turn in turns)
    vedal = sum(turn.get("speaker_identity") == "VEDAL" for turn in turns)
    unknown = sum(turn.get("speaker_identity") == "UNKNOWN" for turn in turns)
    suspicious = sum(bool(turn.get("suspicious_transcription")) for turn in turns)
    family_ratio = family / max(1, total)
    unknown_ratio = unknown / max(1, total)
    suspicious_ratio = suspicious / max(1, total)
    stats = {
        "turn_count": total,
        "family_turn_count": family,
        "vedal_turn_count": vedal,
        "unknown_turn_count": unknown,
        "suspicious_turn_count": suspicious,
        "family_ratio": round(family_ratio, 4),
        "unknown_ratio": round(unknown_ratio, 4),
        "suspicious_ratio": round(suspicious_ratio, 4),
    }
    if row.get("window_type") != "4_8_turns":
        return "B", "extended/long retrieval window retained for context but excluded from S/A primary training review to prevent overlap", stats
    if family == 0 or unknown_ratio > 0.5 or suspicious_ratio > 0.25:
        return "Q", "no family target, too much unknown identity, or severe suspicious transcription", stats
    if row.get("window_type") == "4_8_turns" and family >= 2 and unknown == 0 and suspicious == 0 and family_ratio >= 0.5:
        return "S", "canonical primary window; family-dominant; no unknown or suspicious turns", stats
    if family >= 1 and unknown_ratio <= 0.2 and suspicious_ratio <= 0.1 and family_ratio >= 0.4:
        return "A", "family-dominant proxy-mapped window with bounded unknown/suspicious content", stats
    return "B", "useful family reference/research window but needs stronger review before training", stats


def main() -> None:
    source = ROOT / "conversations" / "natural_conversations_family_mapped.jsonl"
    quality_path = ROOT / "reports" / "natural_turn_quality_review.json"
    flagged_sources = set()
    if quality_path.exists():
        quality_report = json.loads(quality_path.read_text(encoding="utf-8"))
        flagged_sources = {
            str(item.get("source_id"))
            for item in quality_report.get("flagged_sources", [])
            if item.get("source_id")
        }
    all_output = ROOT / "datasets" / "family_training_candidates_review.jsonl"
    sa_output = ROOT / "datasets" / "family_s_a_candidates_review.jsonl"
    all_rows = []
    sa_rows = []
    counts = Counter()
    for row in read_jsonl(source):
        candidate_grade, reason, stats = grade(row)
        source_id = str(row.get("source_video_id") or "")
        if source_id in flagged_sources:
            candidate_grade = "Q"
            reason = "source-level ASR quality flag pending review; retain for audit but block S/A promotion"
            stats["source_quality_flagged"] = True
        else:
            stats["source_quality_flagged"] = False
        candidate = {
            "conversation_id": row.get("conversation_id"),
            "source_video_id": row.get("source_video_id"),
            "window_type": row.get("window_type"),
            "status": "REVIEW_REQUIRED_PROXY_MAPPING",
            "candidate_grade": candidate_grade,
            "training_candidate": False,
            "candidate_reason": reason,
            "identity_mapping": "NEURO_FAMILY | VEDAL | UNKNOWN",
            "identity_confidence": "proxy_pending_final_review",
            "source_segment_ids": row.get("source_segment_ids") or [],
            "stats": stats,
            "turns": row.get("turns") or [],
        }
        all_rows.append(candidate)
        counts[candidate_grade] += 1
        if candidate_grade in {"S", "A"}:
            sa_rows.append(candidate)
    all_output.parent.mkdir(parents=True, exist_ok=True)
    all_output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in all_rows), encoding="utf-8")
    sa_output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in sa_rows), encoding="utf-8")
    report = {
        "schema_version": "0.1.0",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": "S_A_REVIEW_CANDIDATES_PROXY_IDENTITY_PENDING",
        "input": str(source.relative_to(ROOT)),
        "all_output": str(all_output.relative_to(ROOT)),
        "s_a_output": str(sa_output.relative_to(ROOT)),
        "record_count": len(all_rows),
        "grade_counts": dict(counts),
        "s_a_review_candidate_count": len(sa_rows),
        "training_candidate_count": 0,
        "source_quality_flagged_count": len(flagged_sources),
        "policy": "S/A here means review priority only; source-level ASR flags force Q, and no record is training-ready until proxy identity, transcript QA, dedupe, and final human review pass.",
    }
    write_json(ROOT / "reports" / "family_training_candidate_progress.json", report)
    print(json.dumps({"status": report["status"], "record_count": len(all_rows), "grade_counts": dict(counts), "s_a_review_candidate_count": len(sa_rows), "training_candidate_count": 0}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
