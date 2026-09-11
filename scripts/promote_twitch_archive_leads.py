from __future__ import annotations

import argparse
import json

from manifest_tools import ROOT, merge_rows, read_json, read_jsonl, write_master


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=3)
    args = ap.parse_args()
    lead_payload = read_json(ROOT / "sources" / "archive_indexes" / "vod_leads.json", {})
    rows = read_jsonl(ROOT / "manifest" / "master_video_manifest.jsonl")
    existing = {str(row.get("source_id")) for row in rows}
    additions = []
    for lead in lead_payload.get("leads", [])[: max(0, args.limit)]:
        source_id = str(lead.get("source_id"))
        if not source_id or source_id in existing:
            continue
        additions.append({
            "source_platform": "twitch_archive",
            "source_id": source_id,
            "source_url": lead["url"],
            "title": None,
            "uploader": "vedal987 (public archive lead)",
            "discovery_method": "public_archive_index",
            "discovery_source": lead.get("discovery_source"),
            "participants": ["Neuro", "Evil", "Vedal"],
            "language": "en",
            "audio_download_status": "pending",
            "metadata_status": "pending",
            "subtitle_status": "pending",
            "processing_status": "pending",
            "training_candidate": False,
            "rights_note": "Public VOD lead; preserve archive attribution and verify source relationship before training use.",
        })
        existing.add(source_id)
    write_master(merge_rows(rows, additions))
    print(json.dumps({"promoted": len(additions), "source_ids": [x["source_id"] for x in additions]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
