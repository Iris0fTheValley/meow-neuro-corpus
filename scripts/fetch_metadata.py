from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio_ffmpeg
import yt_dlp

from manifest_tools import ROOT, log_event, read_jsonl, sanitized_info, write_json, write_master


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--source-id", action="append")
    args = parser.parse_args()
    rows = read_jsonl(ROOT / "manifest" / "master_video_manifest.jsonl")
    out_dir = ROOT / "manifest" / "metadata"
    out_dir.mkdir(parents=True, exist_ok=True)
    processed = 0
    for row in rows:
        if args.source_id and row.get("source_id") not in args.source_id:
            continue
        if not args.source_id and not args.all and processed >= args.limit:
            break
        if row.get("metadata_status") in {"done", "blocked"}:
            continue
        try:
            opts = {"skip_download": True, "quiet": True, "no_warnings": True, "ffmpeg_location": imageio_ffmpeg.get_ffmpeg_exe()}
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(row["source_url"], download=False)
            safe = sanitized_info(info)
            write_json(out_dir / f"{row['source_platform']}_{row['source_id']}.json", safe)
            row.update({
                "title": info.get("title") or row.get("title"),
                "uploader": info.get("uploader") or row.get("uploader"),
                "upload_date": info.get("upload_date") or row.get("upload_date"),
                "duration": info.get("duration"),
                "description": info.get("description"),
                "participants": row.get("participants", []),
                "subtitle_languages": sorted(set((info.get("subtitles") or {}).keys()) | set((info.get("automatic_captions") or {}).keys())),
                "metadata_status": "done",
                "metadata_path": str((out_dir / f"{row['source_platform']}_{row['source_id']}.json").relative_to(ROOT)),
                "metadata_error": None,
            })
            log_event("metadata_complete", source_id=row["source_id"], source_url=row["source_url"])
        except Exception as exc:
            attempts = int(row.get("metadata_attempts") or 0) + 1
            row["metadata_attempts"] = attempts
            row["metadata_status"] = "blocked" if attempts >= 2 else "retryable"
            row["metadata_error"] = repr(exc)
            log_event("metadata_blocked" if attempts >= 2 else "metadata_failed", source_id=row.get("source_id"), source_url=row.get("source_url"), attempts=attempts, error=repr(exc))
        processed += 1
    write_master(rows)
    print(json.dumps({"processed": processed, "total": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
