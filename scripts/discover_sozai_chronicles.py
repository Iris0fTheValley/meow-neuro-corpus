from __future__ import annotations

import html
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import requests

from fetch_sozai_transcript_pages import YT_RE, extract_transcript, iso_duration_seconds
from manifest_tools import ROOT, merge_rows, read_jsonl, write_json, write_master, log_event


URL = "https://sozai.app/transcripts/channel/neuro-sama-chronicles/"
UA = {"User-Agent": "neuro-public-corpus-discovery/0.1"}


def fetch(url: str) -> tuple[str, str]:
    r = requests.get(url, headers=UA, timeout=45)
    r.raise_for_status()
    return url, r.text


if __name__ == "__main__":
    response = requests.get(URL, headers=UA, timeout=45)
    response.raise_for_status()
    links = []
    for href in re.findall(r'href=["\']([^"\']+)["\']', response.text):
        href = html.unescape(href)
        if "/transcript/" in href:
            links.append(href if href.startswith("http") else "https://sozai.app" + href)
    links = list(dict.fromkeys(links))
    write_json(ROOT / "sources" / "sozai" / "chronicles_index_snapshot.json", {"source": "SozAI public transcript index", "source_url": URL, "retrieved_at": datetime.now(timezone.utc).isoformat(), "transcript_links": links})
    pages = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(fetch, url) for url in links]
        for future in as_completed(futures):
            url, body = future.result()
            slug = url.rstrip("/").split("/")[-1]
            ids = list(dict.fromkeys(YT_RE.findall(html.unescape(body))))
            title_match = re.search(r"<title[^>]*>(.*?)</title>", body, re.I | re.S)
            title = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", title_match.group(1)))).strip() if title_match else slug
            transcript = extract_transcript(body)
            raw_path = None
            count = 0
            if isinstance(transcript, list):
                raw = ROOT / "raw_subtitles" / "sozai" / f"chronicles_{slug}.json"
                write_json(raw, {"source": "SozAI public transcript", "source_url": url, "retrieved_at": datetime.now(timezone.utc).isoformat(), "youtube_ids": ids, "title": title, "transcript": transcript})
                raw_path, count = str(raw.relative_to(ROOT)), len(transcript)
            pages.append({"url": url, "title": title, "youtube_ids": ids, "transcript_path": raw_path, "transcript_segment_count": count, "bytes": len(body)})
    rows = []
    for page in pages:
        for source_id in page["youtube_ids"]:
            rows.append({"source_platform": "youtube", "source_id": source_id, "source_url": f"https://www.youtube.com/watch?v={source_id}", "title": page["title"], "uploader": "Neuro-Sama Chronicles (SozAI transcript index)", "discovery_method": "public_transcript_index", "discovery_source": page["url"], "transcript_source": "SozAI public transcript", "transcript_source_url": page["url"], "transcript_available": True, "transcript_path": page["transcript_path"], "transcript_segment_count": page["transcript_segment_count"], "subtitle_languages": ["en"], "rights_note": "Public transcript index; highlight/edited status and original media provenance require verification.", "training_candidate": False})
    if rows:
        write_master(merge_rows(read_jsonl(ROOT / "manifest" / "master_video_manifest.jsonl"), rows))
    registry_path = ROOT / "source_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if not any(x.get("source_id") == "sozai_neuro_chronicles" for x in registry.get("sources", [])):
        registry["sources"].append({"source_id": "sozai_neuro_chronicles", "platform": "web", "kind": "public_transcript_index", "name": "SozAI Neuro-Sama Chronicles transcripts", "url": URL, "uploader": "SozAI index / Neuro-Sama Chronicles", "priority": "medium", "discovery_status": "expanded", "notes": "Public transcript index for highlights; preserve transcript attribution and do not treat edited highlights as full-stream evidence."})
        write_json(registry_path, registry)
    log_event("sozai_transcript_index_complete", source_url=URL, links=len(links), pages=len(pages), channel="neuro-sama-chronicles")
    print(json.dumps({"links": len(links), "pages": len(pages), "pages_with_transcript": sum(bool(x["transcript_path"]) for x in pages), "manifest_leads": len(rows)}, ensure_ascii=False))
