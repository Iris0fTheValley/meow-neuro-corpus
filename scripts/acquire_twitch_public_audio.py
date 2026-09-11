from __future__ import annotations

import argparse
import json
import re

import imageio_ffmpeg
import yt_dlp

from manifest_tools import ROOT, read_jsonl, safe_name, write_master, log_event


def safe_error(exc: Exception) -> str:
    text = repr(exc)
    return re.sub(r"https?://\S+", "<redacted-url>", text)[:1000]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-id", required=True)
    args = ap.parse_args()
    rows = read_jsonl(ROOT / "manifest" / "master_video_manifest.jsonl")
    row = next((item for item in rows if item.get("source_id") == args.source_id), None)
    if row is None:
        raise SystemExit(f"unknown source id: {args.source_id}")
    folder = ROOT / "raw_audio"
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / f"{safe_name(args.source_id)}.%(ext)s"
    try:
        opts = {
            "format": "bestaudio/best",
            "outtmpl": str(output),
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "retries": 1,
            "fragment_retries": 1,
            "continuedl": True,
            "ffmpeg_location": imageio_ffmpeg.get_ffmpeg_exe(),
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(row["source_url"], download=False)
            safe_path = folder / f"{safe_name(args.source_id)}.{info.get('ext') or 'mp4'}"
            ydl.download([row["source_url"]])
        assets = sorted(folder.glob(f"{safe_name(args.source_id)}.*"))
        assets = [p for p in assets if p.suffix.lower() not in {".json", ".part", ".ytdl"}]
        if not assets:
            raise RuntimeError("download returned without a public audio asset")
        asset = assets[0]
        row.update({
            "title": info.get("title") or row.get("title"),
            "uploader": info.get("uploader") or row.get("uploader"),
            "upload_date": info.get("upload_date") or row.get("upload_date"),
            "duration": info.get("duration") or row.get("duration"),
            "audio_download_status": "done",
            "audio_path": str(asset.relative_to(ROOT)),
            "asset_bytes": asset.stat().st_size,
            "metadata_status": "done",
            "subtitle_languages": sorted(set(row.get("subtitle_languages", [])) | set((info.get("subtitles") or {}).keys())),
            "metadata_source": "public Twitch VOD page",
            "metadata_error": None,
        })
        log_event("asset_complete", kind="audio", source_id=args.source_id, path=row["audio_path"], bytes=row["asset_bytes"])
    except Exception as exc:
        row["audio_download_status"] = "blocked"
        row["audio_error"] = safe_error(exc)
        log_event("asset_blocked", kind="audio", source_id=args.source_id, error=row["audio_error"])
    write_master(rows)
    print(json.dumps({"source_id": args.source_id, "audio_download_status": row.get("audio_download_status"), "duration": row.get("duration")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
