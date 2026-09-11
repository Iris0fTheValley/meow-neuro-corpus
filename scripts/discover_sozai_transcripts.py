from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from html import unescape

import requests

from manifest_tools import ROOT, log_event, merge_rows, read_jsonl, write_master, write_json


URL = "https://sozai.app/transcripts/channel/neuro-sama-unofficial-vods/"
HEADERS = {"User-Agent": "neuro-public-corpus-discovery/0.1"}


if __name__ == "__main__":
    out_dir = ROOT / "sources" / "sozai"
    out_dir.mkdir(parents=True, exist_ok=True)
    response = requests.get(URL, headers=HEADERS, timeout=45)
    response.raise_for_status()
    html = response.text
    (out_dir / "neuro-sama-unofficial-vods.html").write_text(html, encoding="utf-8")
    links = []
    for href in re.findall(r'href=["\']([^"\']+)["\']', html):
        href = unescape(href)
        if "/transcripts/" in href or "/transcript/" in href:
            links.append(href if href.startswith("http") else "https://sozai.app" + href)
    links = list(dict.fromkeys(links))
    titles = [unescape(re.sub(r"\s+", " ", x)).strip() for x in re.findall(r"<(?:h2|h3|a)[^>]*>(.*?)</(?:h2|h3|a)>", html, flags=re.I | re.S)]
    snapshot = {"source": "SozAI public transcript index", "source_url": URL, "retrieved_at": datetime.now(timezone.utc).isoformat(), "page_title": "Neuro-sama Unofficial VODs — All Video Transcripts", "transcript_link_count": len(links), "transcript_links": links, "title_candidates": titles[:100]}
    write_json(out_dir / "index_snapshot.json", snapshot)
    log_event("sozai_transcript_index_complete", links=len(links), source_url=URL)
    # The index may not expose original YouTube IDs in every link. Preserve the
    # index as an expansion lead even when no safe manifest row can be derived.
    registry_path = ROOT / "source_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if not any(x.get("source_id") == "sozai_neuro_unofficial_vods" for x in registry.get("sources", [])):
        registry["sources"].append({"source_id": "sozai_neuro_unofficial_vods", "platform": "web", "kind": "public_transcript_index", "name": "SozAI Neuro-sama Unofficial VOD transcripts", "url": URL, "uploader": "SozAI index / Neuro-sama Unofficial VODs", "priority": "critical", "discovery_status": "expanded", "notes": "Public timestamped transcript index; preserve transcript attribution and locate original media before treating text as source truth."})
        registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    frontier_path = ROOT / "discovery_frontier.json"
    frontier = json.loads(frontier_path.read_text(encoding="utf-8"))
    if not any(x.get("url") == URL for x in frontier.get("frontier", [])):
        frontier["frontier"].append({"type": "transcript_index", "platform": "web", "url": URL, "status": "expanded", "next_actions": ["map transcript links to original public VOD IDs", "preserve transcript snapshots and attribution", "align only after media provenance is verified"]})
        frontier["updated_at"] = datetime.now(timezone.utc).date().isoformat()
        frontier_path.write_text(json.dumps(frontier, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status_code": response.status_code, "bytes": len(html), "transcript_links": len(links)}, ensure_ascii=False))
