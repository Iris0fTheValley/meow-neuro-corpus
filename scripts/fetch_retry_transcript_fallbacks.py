from __future__ import annotations

"""Fetch small public subtitle fallbacks for retryable sources; never fetch media here."""

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from manifest_tools import ROOT, write_json


def main() -> None:
    probe = json.loads((ROOT / "reports" / "retry_queue_proxy_probe.json").read_text(encoding="utf-8"))
    proxy = os.environ.get("MEOW_PUBLIC_PROXY", "http://127.0.0.1:7897")
    output_root = ROOT / "retry_queue" / "transcript_fallbacks"
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for item in probe.get("results", []):
        if not item.get("transcript_fallback_available"):
            continue
        source_id = str(item.get("source_id"))
        url = str(item.get("source_url"))
        before = {str(path) for path in output_root.glob(f"{source_id}.*")}
        command = [
            "..\\.venv-data\\Scripts\\python.exe", "-m", "yt_dlp",
            "--skip-download", "--write-subs", "--write-auto-subs",
            "--sub-langs", "en.*,zh.*", "--sub-format", "vtt",
            "--no-overwrites", "--ignore-errors", "--no-warnings",
            "--proxy", proxy, "--socket-timeout", "20", "--retries", "1",
            "--paths", str(output_root), "--output", f"{source_id}.%(ext)s", url,
        ]
        try:
            completed = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True, timeout=45)
            status = "FETCH_COMPLETED" if completed.returncode == 0 else "FETCH_FAILED"
            error_tail = (completed.stderr or completed.stdout)[-800:] if completed.returncode else None
        except subprocess.TimeoutExpired:
            status = "FETCH_TIMEOUT"
            error_tail = "timeout after 45 seconds"
        after = {str(path) for path in output_root.glob(f"{source_id}.*")}
        new_files = sorted(after - before)
        results.append({
            "source_id": source_id,
            "source_url": url,
            "status": status,
            "new_files": [str(Path(path).relative_to(ROOT)) for path in new_files],
            "error_tail": error_tail,
            "media_downloaded": False,
        })
    report = {
        "schema_version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "TRANSCRIPT_FALLBACK_FETCH_COMPLETE_MEDIA_NOT_DOWNLOADED",
        "attempt_count": len(results),
        "success_count": sum(bool(item["new_files"]) for item in results),
        "results": results,
        "policy": "Only public subtitle/automatic-caption files are fetched. No audio/video media, credentials, or auth bypass is used; original retryable manifest statuses remain unchanged.",
    }
    write_json(ROOT / "reports" / "retry_transcript_fallbacks.json", report)
    print(json.dumps({"status": report["status"], "attempt_count": report["attempt_count"], "success_count": report["success_count"]}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
