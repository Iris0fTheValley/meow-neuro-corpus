from __future__ import annotations

import json

from acquire_assets import terminal_download_error
from manifest_tools import ROOT, read_jsonl, write_master, log_event


def main() -> None:
    rows = read_jsonl(ROOT / "manifest" / "master_video_manifest.jsonl")
    changed = []
    for row in rows:
        if row.get("audio_download_status") != "retryable":
            continue
        error = row.get("asset_error") or row.get("audio_error") or ""
        if not terminal_download_error(Exception(error)):
            continue
        row["audio_download_status"] = "blocked"
        row["audio_download_attempts"] = max(1, int(row.get("audio_download_attempts") or 0))
        row["audio_block_reason"] = "public source blocked/unavailable; no authentication bypass attempted"
        changed.append(row.get("source_id"))
    if changed:
        write_master(rows)
        log_event("download_failures_normalized", source_platform="youtube", blocked_count=len(changed), policy="bounded_retry_no_auth_bypass")
    print(json.dumps({"blocked": len(changed), "remaining_retryable": sum(r.get("audio_download_status") == "retryable" for r in rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
