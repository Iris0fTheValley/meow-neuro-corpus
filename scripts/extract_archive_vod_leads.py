from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from html import unescape

import requests

from manifest_tools import ROOT, read_json, write_json


PAGES = {
    "twitchnosub_vedal987": "https://twitchnosub.com/streamer/vedal987",
    "streamrecorder_vedal987": "https://streamrecorder.io/twitch/vedal987",
}
VOD_RE = re.compile(r"https?://(?:www\.)?twitch\.tv/videos/(\d+)")


def main() -> None:
    out = ROOT / "sources" / "archive_indexes"
    leads = []
    for name, url in PAGES.items():
        item = {"name": name, "url": url, "retrieved_at": datetime.now(timezone.utc).isoformat(), "links": []}
        try:
            response = requests.get(url, headers={"User-Agent": "neuro-corpus-public-discovery/0.1"}, timeout=30)
            text = unescape(response.text)
            ids = list(dict.fromkeys(VOD_RE.findall(text)))
            item.update({"status_code": response.status_code, "vod_ids": ids, "link_count": len(ids)})
            for vod_id in ids:
                lead = {
                    "type": "twitch_vod_lead",
                    "platform": "twitch_archive",
                    "source_id": f"vedal987:{vod_id}",
                    "url": f"https://www.twitch.tv/videos/{vod_id}",
                    "discovery_source": url,
                    "status": "pending",
                    "next_actions": ["verify public access", "check TwitchTranscripts or alternate archive", "deduplicate by date/duration/audio"],
                }
                item["links"].append(lead)
                leads.append(lead)
        except Exception as exc:
            item["error"] = repr(exc)
        write_json(out / f"{name}_vod_leads.json", item)
    write_json(out / "vod_leads.json", {"retrieved_at": datetime.now(timezone.utc).isoformat(), "leads": leads, "policy": "public links only; no authentication material"})
    frontier_path = ROOT / "discovery_frontier.json"
    frontier = read_json(frontier_path, {"frontier": []})
    existing = {str(item.get("url")) for item in frontier.get("frontier", [])}
    added = 0
    for lead in leads:
        if lead["url"] in existing:
            continue
        frontier.setdefault("frontier", []).append(lead)
        existing.add(lead["url"])
        added += 1
    frontier["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json(frontier_path, frontier)
    print(json.dumps({"pages": len(PAGES), "vod_leads": len(leads), "frontier_added": added}, ensure_ascii=False))


if __name__ == "__main__":
    main()
