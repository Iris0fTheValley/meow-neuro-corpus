from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

from fetch_twitchtranscripts import extract_entries, windows
from manifest_tools import ROOT, merge_rows, read_json, read_jsonl, safe_name, write_json, write_master


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=8)
    args = ap.parse_args()
    lead_path = ROOT / "sources" / "archive_indexes" / "vod_leads.json"
    payload = read_json(lead_path, {})
    leads = payload.get("leads", [])[: max(0, args.limit)]
    session = requests.Session()
    session.headers.update({"User-Agent": "neuro-corpus-public-discovery/0.1"})
    page_dir = ROOT / "sources" / "twitchtranscripts" / "pages"
    raw_dir = ROOT / "raw_subtitles" / "twitchtranscripts"
    page_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(ROOT / "manifest" / "master_video_manifest.jsonl")
    discovered = []
    results = []
    for lead in leads:
        vod_id = str(lead["source_id"]).split(":", 1)[-1]
        url = f"https://www.twitchtranscripts.com/channel/vedal987/{vod_id}"
        item = {"source_id": lead["source_id"], "url": url, "discovery_source": lead.get("discovery_source")}
        try:
            response = session.get(url, timeout=30)
            entries = extract_entries(response.text) if response.ok else []
            item.update({"status_code": response.status_code, "segments": len(entries), "retrieved_at": datetime.now(timezone.utc).isoformat()})
            if entries:
                page_path = page_dir / f"{vod_id}.html"
                page_path.write_text(response.text, encoding="utf-8")
                raw_path = raw_dir / f"{vod_id}.json"
                write_json(raw_path, {"source": "TwitchTranscripts public page", "source_url": url, "vod_id": vod_id, "retrieved_at": item["retrieved_at"], "segments": entries})
                discovered.append({
                    "source_platform": "twitchtranscripts", "source_id": lead["source_id"], "source_url": url,
                    "title": None, "uploader": "TwitchTranscripts public archive", "discovery_method": "alternate_archive_lead",
                    "discovery_source": lead.get("discovery_source"), "transcript_source": "TwitchTranscripts public page",
                    "transcript_source_url": url, "transcript_available": True, "transcript_segment_count": len(entries),
                    "asr_status": "source_transcript", "audio_download_status": "pending", "processing_status": "transcript_only",
                    "language": "en", "subtitle_languages": ["en"], "participants": ["Neuro", "Evil", "Vedal"],
                    "training_candidate": False, "rights_note": "Public transcript snapshot; speaker/audio alignment remains unvalidated.",
                })
                item["status"] = "transcript_available"
            else:
                item["status"] = "blocked_or_no_transcript"
        except Exception as exc:
            item.update({"status": "blocked_or_error", "error": repr(exc)})
        results.append(item)
    if discovered:
        write_master(merge_rows(rows, discovered))
    frontier_path = ROOT / "discovery_frontier.json"
    frontier = read_json(frontier_path, {"frontier": []})
    by_id = {item.get("source_id"): item for item in results}
    for item in frontier.get("frontier", []):
        result = by_id.get(item.get("source_id"))
        if not result:
            continue
        if result.get("status") == "transcript_available":
            item["status"] = "transcript_available"
        elif result.get("status_code") == 404:
            item["status"] = "blocked"
            item["blocked_reason"] = "TwitchTranscripts public page returned HTTP 404"
    write_json(frontier_path, frontier)
    payload["last_probe"] = results
    write_json(lead_path, payload)
    print(json.dumps({"probed": len(results), "transcript_available": len(discovered)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
