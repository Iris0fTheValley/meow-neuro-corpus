from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests

from manifest_tools import ROOT, log_event, merge_rows, read_jsonl, write_master, write_json


API = "https://libraryofladev.com/api/search"


def fetch_term(term: str, fetch_size: int, last_url: str | None = None) -> dict:
    params = {"text": term, "fetchSize": fetch_size}
    if last_url:
        params["lastUrl"] = last_url
    response = requests.get(API, params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--term", action="append", default=["Neuro", "Evil", "Vedal"])
    parser.add_argument("--fetch-size", type=int, default=100)
    parser.add_argument("--pages", type=int, default=3, help="Maximum pages per search term")
    args = parser.parse_args()
    snapshot_dir = ROOT / "sources" / "library-of-ladev" / "api_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    discovered = []
    for term in dict.fromkeys(args.term):
        last_url = None
        term_count = 0
        for page in range(1, max(1, args.pages) + 1):
            payload = fetch_term(term, args.fetch_size, last_url)
            write_json(snapshot_dir / f"search_{term.lower()}_page{page}.json", payload)
            result = payload.get("data", {}).get("result", [])
            for item in result:
                source_id = item.get("url")
                if not source_id:
                    continue
                discovered.append({
                    "source_platform": "youtube",
                    "source_url": f"https://www.youtube.com/watch?v={source_id}",
                    "source_id": source_id,
                    "title": item.get("title"),
                    "stream_date_if_known": item.get("date"),
                    "uploader": "Library of Ladev indexed source",
                    "discovery_source": "https://libraryofladev.com/api/search",
                    "discovery_query": term,
                    "transcript_source": "Library of Ladev public API",
                    "transcript_source_url": f"https://libraryofladev.com/?video={source_id}",
                    "transcript_available": True,
                    "transcript_total_matches": item.get("total"),
                    "subtitle_languages": ["en"],
                    "download_status": "pending",
                    "processing_status": "transcript_available",
                })
                term_count += 1
            next_url = payload.get("data", {}).get("lastUrl")
            log_event("ladev_search_page_complete", term=term, page=page, results=len(result), last_url=next_url)
            if payload.get("data", {}).get("noMoreResultsToFetch") or not next_url or next_url == last_url or not result:
                break
            last_url = next_url
        log_event("ladev_search_complete", term=term, results=term_count)
    existing = read_jsonl(ROOT / "manifest" / "master_video_manifest.jsonl")
    initial = read_jsonl(ROOT / "manifest" / "initial_queue.jsonl")
    known = {(str(row.get("source_platform") or ""), str(row.get("source_id") or row.get("source_url") or "")) for row in existing}
    new_discovered = [row for row in discovered if (str(row.get("source_platform") or ""), str(row.get("source_id") or row.get("source_url") or "")) not in known]
    write_master(merge_rows(existing, initial, new_discovered))
    write_json(ROOT / "sources" / "library-of-ladev" / "discovery_summary.json", {
        "terms": list(dict.fromkeys(args.term)), "pages_per_term": args.pages, "records_from_api": len(discovered),
        "new_manifest_leads": len(new_discovered), "skipped_existing_leads": len(discovered) - len(new_discovered),
        "raw_snapshots": [str(p.relative_to(ROOT)) for p in sorted(snapshot_dir.glob("search_*_page*.json"))],
        "policy": "API transcript text remains source-attributed and is not silently treated as ASR ground truth; align it to downloaded media when available.",
    })
    print(json.dumps({"terms": list(dict.fromkeys(args.term)), "records_from_api": len(discovered), "new_manifest_leads": len(new_discovered), "skipped_existing_leads": len(discovered) - len(new_discovered), "master": len(read_jsonl(ROOT / "manifest" / "master_video_manifest.jsonl"))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
