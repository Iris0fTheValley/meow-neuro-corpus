from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio_ffmpeg
import yt_dlp

from manifest_tools import ROOT, log_event, read_jsonl, safe_name, write_master


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--source-id", action="append")
    args = parser.parse_args()
    rows = read_jsonl(ROOT / "manifest" / "master_video_manifest.jsonl")
    folder = ROOT / "raw_subtitles"
    folder.mkdir(parents=True, exist_ok=True)
    done = 0
    for row in rows:
        if args.source_id and row.get("source_id") not in args.source_id:
            continue
        if not args.source_id and not args.all and done >= args.limit:
            break
        if row.get("subtitle_status") in {"done", "not_available", "blocked"}:
            continue
        opts = {
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["en.*", "zh.*", "ja.*"],
            "subtitlesformat": "vtt/srt/best",
            "outtmpl": str(folder / f"{safe_name(row['source_id'])}.%(ext)s"),
            "noplaylist": True,
            "ignoreerrors": True,
            "quiet": True,
            "ffmpeg_location": imageio_ffmpeg.get_ffmpeg_exe(),
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(row["source_url"], download=False)
            files = sorted(folder.glob(f"{safe_name(row['source_id'])}.*"))
            if files:
                row["subtitle_status"] = "done"
                row["subtitle_paths"] = [str(p.relative_to(ROOT)) for p in files]
                log_event("subtitle_complete", source_id=row["source_id"], files=row["subtitle_paths"])
            else:
                available = set((info or {}).get("subtitles", {}).keys()) | set((info or {}).get("automatic_captions", {}).keys())
                row["subtitle_status"] = "not_available" if not available else "blocked"
                row["subtitle_languages"] = sorted(available)
                log_event("subtitle_unavailable", source_id=row["source_id"], available=sorted(available))
        except Exception as exc:
            attempts = int(row.get("subtitle_attempts") or 0) + 1
            row["subtitle_attempts"] = attempts
            row["subtitle_status"] = "blocked" if attempts >= 2 else "retryable"
            row["subtitle_error"] = repr(exc)
            log_event("subtitle_blocked" if attempts >= 2 else "subtitle_failed", source_id=row.get("source_id"), attempts=attempts, error=repr(exc))
        done += 1
    write_master(rows)
    print(json.dumps({"processed": done, "total": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
