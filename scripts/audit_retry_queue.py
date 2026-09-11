from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone

from manifest_tools import ROOT, read_jsonl, write_json


def category(error: str) -> str:
    text = error.lower()
    if "ssl" in text or "unexpected_eof" in text:
        return "network_ssl"
    if "timed out" in text or "timeout" in text:
        return "network_timeout"
    if "captcha" in text or "bot" in text or "sign in" in text:
        return "access_blocked"
    return "other_retryable"


def main() -> None:
    retryable = []
    for row in read_jsonl(ROOT / "manifest" / "master_video_manifest.jsonl"):
        if row.get("audio_download_status") != "retryable":
            continue
        error = str(row.get("audio_download_error") or row.get("asset_error") or row.get("audio_error") or "")
        title = str(row.get("title") or "")
        known_target_hint = bool(re.search(r"\b(neuro|evil|vedal)\b", title, re.I) or row.get("participants"))
        retryable.append({
            "source_id": row.get("source_id"),
            "source_platform": row.get("source_platform"),
            "source_url": row.get("source_url"),
            "title": title,
            "duration": row.get("duration"),
            "category": category(error),
            "known_target_metadata_hint": known_target_hint,
            "error": error,
            "next_action": "alternate_public_source_or_transcript_fallback; bounded_retry_only",
            "auth_bypass": False,
        })
    report = {
        "schema_version": "0.1.0",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": "RETRYABLE_ALTERNATE_SOURCE_PENDING",
        "retryable_count": len(retryable),
        "category_counts": dict(Counter(row["category"] for row in retryable)),
        "known_target_metadata_hint_count": sum(row["known_target_metadata_hint"] for row in retryable),
        "policy": "Do not treat blocked download as corpus failure; try bounded alternate public sources, then preserve unresolved status without auth bypass.",
        "items": retryable,
    }
    write_json(ROOT / "reports" / "retry_queue_review.json", report)
    write_json(ROOT / "retry_queue" / "retry_queue_review.json", report)
    print(json.dumps({"status": report["status"], "retryable_count": report["retryable_count"], "category_counts": report["category_counts"], "known_target_metadata_hint_count": report["known_target_metadata_hint_count"]}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
