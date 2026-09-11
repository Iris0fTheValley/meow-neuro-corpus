from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests

from manifest_tools import ROOT, read_jsonl, write_master, write_json, log_event


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-id", action="append", required=True)
    args = parser.parse_args()
    out_dir = ROOT / "raw_subtitles" / "library_of_ladev"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(ROOT / "manifest" / "master_video_manifest.jsonl")
    for source_id in args.source_id:
        response = requests.get("https://libraryofladev.com/api/search", params={"videoUrl": source_id, "fetchSize": 100}, timeout=60)
        if response.status_code != 200:
            log_event("ladev_video_unavailable", source_id=source_id, status=response.status_code, body=response.text[:500])
            continue
        payload = response.json()
        path = out_dir / f"{source_id}.json"
        write_json(path, {"source": "Library of Ladev public API", "source_url": "https://libraryofladev.com/", "api_query": {"videoUrl": source_id, "fetchSize": 100}, "payload": payload})
        result = payload.get("data", {}).get("result", {})
        for row in rows:
            if row.get("source_id") == source_id:
                row.update({
                    "transcript_available": True,
                    "transcript_source": "Library of Ladev public API",
                    "transcript_source_url": "https://libraryofladev.com/",
                    "transcript_raw_path": str(path.relative_to(ROOT)),
                    "transcript_segment_count": len(result.get("subtitles", [])),
                    "subtitle_status": "done",
                    "subtitle_languages": sorted(set(row.get("subtitle_languages", [])) | {"en"}),
                })
        log_event("ladev_video_complete", source_id=source_id, segments=len(result.get("subtitles", [])))
    write_master(rows)
    print(json.dumps({"requested": args.source_id, "saved": [str(p.relative_to(ROOT)) for p in out_dir.glob("*.json") if p.stem in args.source_id]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

