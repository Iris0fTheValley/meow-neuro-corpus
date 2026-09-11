from __future__ import annotations

import argparse
import json
from pathlib import Path

import yt_dlp

from manifest_tools import ROOT, log_event, read_json, read_jsonl, merge_rows, write_master


def discover(url: str, limit: int | None) -> list[dict]:
    opts = {
        "extract_flat": "in_playlist",
        "skip_download": True,
        "quiet": True,
        "ignoreerrors": True,
        "playlistend": limit,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    rows = []
    for entry in (info or {}).get("entries", []) or []:
        if not entry:
            continue
        source_id = entry.get("id")
        webpage_url = entry.get("webpage_url") or entry.get("url")
        if not source_id or not webpage_url:
            continue
        rows.append({
            "source_platform": "youtube",
            "source_url": webpage_url,
            "source_id": source_id,
            "uploader": entry.get("channel") or entry.get("uploader"),
            "title": entry.get("title"),
            "upload_date": entry.get("upload_date"),
            "download_status": "pending",
            "processing_status": "pending",
            "discovery_source": url,
            "discovery_method": "yt_dlp_flat_playlist",
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-url", action="append")
    parser.add_argument("--all-seeded", action="store_true")
    parser.add_argument("--limit", type=int, default=250)
    args = parser.parse_args()

    registry = read_json(ROOT / "source_registry.json", {})
    urls = args.source_url or []
    if args.all_seeded:
        urls.extend(
            item["url"] for item in registry.get("sources", [])
            if item.get("platform") == "youtube" and item.get("kind") in {"official_channel", "official_vod_channel", "fan_archive"}
        )
    if not urls:
        raise SystemExit("pass --source-url or --all-seeded")

    discovered = []
    for url in dict.fromkeys(urls):
        try:
            rows = discover(url, args.limit)
            discovered.extend(rows)
            log_event("discovery_complete", source_url=url, count=len(rows))
        except Exception as exc:
            log_event("discovery_failed", source_url=url, error=repr(exc))

    initial = read_jsonl(ROOT / "manifest" / "initial_queue.jsonl")
    existing = read_jsonl(ROOT / "manifest" / "master_video_manifest.jsonl")
    write_master(merge_rows(existing, initial, discovered))
    print(json.dumps({"sources": len(dict.fromkeys(urls)), "discovered": len(discovered), "master": len(read_jsonl(ROOT / "manifest" / "master_video_manifest.jsonl"))}, ensure_ascii=False))


if __name__ == "__main__":
    main()

