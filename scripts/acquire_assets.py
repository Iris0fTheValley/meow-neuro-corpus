from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import imageio_ffmpeg
import yt_dlp

from manifest_tools import ROOT, log_event, read_jsonl, safe_name, write_master


DOWNLOAD_MIN_FREE_BYTES = 40 * 1024**3


def terminal_download_error(error: Exception) -> bool:
    text = str(error).lower()
    return any(token in text for token in (
        "sign in to confirm", "not a bot", "video unavailable", "private video",
        "has been removed", "http error 403", "403 forbidden", "forbidden",
    ))


def existing_asset(folder: Path, source_id: str) -> Path | None:
    matches = sorted(
        p for p in folder.glob(f"{safe_name(source_id)}.*")
        if p.suffix.lower() not in {".part", ".ytdl", ".tmp"}
    )
    return matches[0] if matches else None


def downloads_allowed(kind: str) -> bool:
    if kind != "audio":
        return True
    free_bytes = shutil.disk_usage(str(ROOT)).free
    if free_bytes >= DOWNLOAD_MIN_FREE_BYTES:
        return True
    log_event(
        "asset_acquisition_held_disk_low",
        kind=kind,
        free_bytes=free_bytes,
        threshold_bytes=DOWNLOAD_MIN_FREE_BYTES,
    )
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["audio", "video"], required=True)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--source-id", action="append")
    args = parser.parse_args()
    if args.kind == "video":
        log_event("video_acquisition_held_audio_only_policy")
        print(json.dumps({"kind": args.kind, "processed": 0, "held": "audio_only_policy"}, ensure_ascii=False))
        return
    if not downloads_allowed(args.kind):
        print(json.dumps({"kind": args.kind, "processed": 0, "held": "disk_low"}, ensure_ascii=False))
        return
    rows = read_jsonl(ROOT / "manifest" / "master_video_manifest.jsonl")
    folder = ROOT / ("raw_audio" if args.kind == "audio" else "raw_video")
    folder.mkdir(parents=True, exist_ok=True)
    done = 0
    for row in rows:
        status_key = f"{args.kind}_download_status"
        path_key = f"{args.kind}_path"
        if args.source_id and row.get("source_id") not in args.source_id:
            continue
        if not args.source_id and not args.all and done >= args.limit:
            break
        if row.get(status_key) == "done":
            # Repair stale manifest pointers left by interrupted yt-dlp runs;
            # a completed .part file is not a valid asset, but a sibling final
            # media file is safe to adopt without re-downloading.
            asset = existing_asset(folder, row["source_id"])
            if asset and row.get(path_key) != str(asset.relative_to(ROOT)):
                row[path_key] = str(asset.relative_to(ROOT))
                row["asset_bytes"] = asset.stat().st_size
                log_event("asset_path_repaired", kind=args.kind, source_id=row["source_id"], path=row[path_key], bytes=row["asset_bytes"])
            continue
        if existing := existing_asset(folder, row["source_id"]):
            row[status_key] = "done"
            row[path_key] = str(existing.relative_to(ROOT))
            continue
        if args.kind == "audio":
            fmt = "bestaudio/best"
            post = []
            tmpl = str(folder / f"{safe_name(row['source_id'])}.%(ext)s")
        else:
            fmt = "bestvideo[height<=480]+bestaudio/best[height<=480]/best"
            post = []
            tmpl = str(folder / f"{safe_name(row['source_id'])}.%(ext)s")
        opts = {
            "format": fmt,
            "outtmpl": tmpl,
            "noplaylist": True,
            "quiet": False,
            "no_warnings": False,
            "retries": 2,
            "fragment_retries": 2,
            "continuedl": True,
            "merge_output_format": "mkv",
            "ffmpeg_location": imageio_ffmpeg.get_ffmpeg_exe(),
            "postprocessors": post,
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([row["source_url"]])
            asset = existing_asset(folder, row["source_id"])
            if asset:
                row[status_key] = "done"
                row[path_key] = str(asset.relative_to(ROOT))
                row["asset_bytes"] = asset.stat().st_size
                log_event("asset_complete", kind=args.kind, source_id=row["source_id"], path=row[path_key], bytes=row["asset_bytes"])
            else:
                row[status_key] = "failed_permanent"
                row["asset_error"] = "download returned without a matching asset"
        except Exception as exc:
            row[status_key] = "blocked" if terminal_download_error(exc) else "retryable"
            row[f"{args.kind}_download_attempts"] = int(row.get(f"{args.kind}_download_attempts") or 0) + 1
            row["asset_error"] = repr(exc)
            log_event("asset_blocked" if row[status_key] == "blocked" else "asset_failed", kind=args.kind, source_id=row.get("source_id"), source_url=row.get("source_url"), error=repr(exc), attempts=row[f"{args.kind}_download_attempts"])
        done += 1
    write_master(rows)
    print(json.dumps({"kind": args.kind, "processed": done, "total": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
