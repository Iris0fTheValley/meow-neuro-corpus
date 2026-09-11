from __future__ import annotations

"""Probe retryable URLs through the configured public proxy without downloading media."""

import json
import os
import subprocess
from collections import Counter
from datetime import datetime, timezone

from manifest_tools import ROOT, write_json


def main() -> None:
    report_path = ROOT / "reports" / "retry_queue_review.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    proxy = os.environ.get("MEOW_PUBLIC_PROXY", "http://127.0.0.1:7897")
    results = []
    for item in report.get("items", []):
        url = str(item.get("source_url") or "")
        command = [
            "..\\.venv-data\\Scripts\\python.exe", "-m", "yt_dlp",
            "--dump-single-json", "--skip-download", "--proxy", proxy,
            "--socket-timeout", "20", "--retries", "1", "--extractor-retries", "1",
            "--no-warnings", url,
        ]
        try:
            completed = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True, timeout=45)
        except subprocess.TimeoutExpired:
            results.append({"source_id": item.get("source_id"), "source_url": url, "status": "PROBE_TIMEOUT"})
            continue
        if completed.returncode != 0:
            results.append({
                "source_id": item.get("source_id"),
                "source_url": url,
                "status": "PROBE_FAILED",
                "error_tail": (completed.stderr or completed.stdout)[-800:],
            })
            continue
        try:
            metadata = json.loads(completed.stdout)
        except json.JSONDecodeError:
            results.append({"source_id": item.get("source_id"), "source_url": url, "status": "PROBE_INVALID_JSON"})
            continue
        audio_formats = [
            fmt for fmt in (metadata.get("formats") or [])
            if fmt.get("acodec") not in {None, "none"} and fmt.get("vcodec") == "none"
        ]
        results.append({
            "source_id": item.get("source_id"),
            "source_url": url,
            "title": metadata.get("title"),
            "duration": metadata.get("duration"),
            "status": "PROBE_AVAILABLE_AUDIO_FORMATS" if audio_formats else "PROBE_NO_AUDIO_FORMAT",
            "audio_format_ids": [str(fmt.get("format_id")) for fmt in audio_formats[:12]],
            "audio_exts": sorted({str(fmt.get("ext")) for fmt in audio_formats}),
            "best_audio_abr": max((float(fmt.get("abr") or 0) for fmt in audio_formats), default=0),
            "subtitle_languages": sorted((metadata.get("subtitles") or {}).keys()),
            "automatic_caption_languages": sorted((metadata.get("automatic_captions") or {}).keys()),
            "transcript_fallback_available": bool(metadata.get("subtitles") or metadata.get("automatic_captions")),
            "metadata_only": True,
        })
    summary = Counter(row["status"] for row in results)
    output = {
        "schema_version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PROXY_METADATA_PROBE_COMPLETE_NO_MEDIA_DOWNLOAD",
        "proxy": proxy,
        "probe_count": len(results),
        "status_counts": dict(summary),
        "results": results,
        "policy": "This probe only verifies public metadata/format availability through the configured proxy. It does not bypass auth, download media, or modify the manifest; retryable source state remains unchanged until a bounded download succeeds.",
    }
    write_json(ROOT / "reports" / "retry_queue_proxy_probe.json", output)
    write_json(ROOT / "retry_queue" / "retry_queue_proxy_probe.json", output)
    print(json.dumps({"status": output["status"], "probe_count": len(results), "status_counts": dict(summary)}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
