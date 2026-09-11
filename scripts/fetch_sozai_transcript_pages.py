from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from html import unescape

import requests

from manifest_tools import ROOT, log_event, merge_rows, read_jsonl, write_master, write_json


INDEX = ROOT / "sources" / "sozai" / "index_snapshot.json"
HEADERS = {"User-Agent": "neuro-public-corpus-discovery/0.1"}
YT_RE = re.compile(r"(?:youtube\.com/watch\?v=|youtu\.be/)([A-Za-z0-9_-]{11})")


def iso_duration_seconds(value: str | None) -> int | None:
    if not value or not value.startswith("PT"):
        return None
    total = 0
    for amount, unit in re.findall(r"(\d+)([HMS])", value[2:]):
        total += int(amount) * {"H": 3600, "M": 60, "S": 1}[unit]
    return total or None


def fetch(url: str) -> tuple[str, str]:
    response = requests.get(url, headers=HEADERS, timeout=45)
    response.raise_for_status()
    return url, response.text


def extract_transcript(html: str) -> object | None:
    for body in re.findall(r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', html, flags=re.I | re.S):
        try:
            data = json.loads(unescape(body))
        except Exception:
            continue
        objects = data if isinstance(data, list) else [data]
        for obj in objects:
            if isinstance(obj, dict) and obj.get("transcript"):
                return obj["transcript"]
    entries = []
    blocks = re.split(r'<div class=["\']transcript-entry["\']>', html, flags=re.I)[1:]
    for block in blocks:
        tm = re.search(r'data-time=["\']([0-9.]+)["\']', block)
        sm = re.search(r'<div class=["\']entry-speaker[^>]*>(.*?)</div>', block, flags=re.I | re.S)
        xm = re.search(r'<div class=["\']entry-text[^>]*>(.*?)</div>', block, flags=re.I | re.S)
        if not tm or not xm:
            continue
        text = re.sub(r"<br\s*/?>", " ", xm.group(1), flags=re.I)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\s+", " ", unescape(text)).strip()
        if text:
            entries.append({"speaker": re.sub(r"\s+", " ", unescape(sm.group(1))).strip() if sm else "Speaker A", "start": float(tm.group(1)), "text": text})
    if entries:
        for left, right in zip(entries, entries[1:]):
            left["end"] = right["start"]
        entries[-1]["end"] = entries[-1]["start"]
        return entries
    return None


if __name__ == "__main__":
    snapshot = json.loads(INDEX.read_text(encoding="utf-8"))
    urls = [u for u in snapshot.get("transcript_links", []) if "/transcript/" in u and u.rstrip("/").split("/")[-1] not in {"transcript", "transcripts"}]
    out_dir = ROOT / "sources" / "sozai" / "transcript_pages"
    out_dir.mkdir(parents=True, exist_ok=True)
    pages: list[dict] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fetch, url) for url in urls]
        for future in as_completed(futures):
            url, html = future.result()
            slug = url.rstrip("/").split("/")[-1]
            (out_dir / f"{slug}.html").write_text(html, encoding="utf-8")
            ids = list(dict.fromkeys(YT_RE.findall(unescape(html))))
            title_match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.I | re.S)
            title = re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", "", title_match.group(1)))).strip() if title_match else slug.replace("-", " ")
            transcript = extract_transcript(html)
            duration_match = re.search(r'"duration"\s*:\s*"(PT[0-9HMS]+)"', html)
            upload_match = re.search(r'"uploadDate"\s*:\s*"([0-9T:+-]+)"', html)
            transcript_path = None
            segment_count = 0
            if transcript is not None:
                transcript_path_obj = ROOT / "raw_subtitles" / "sozai" / f"{slug}.json"
                transcript_path_obj.parent.mkdir(parents=True, exist_ok=True)
                write_json(transcript_path_obj, {"source": "SozAI public transcript", "source_url": url, "retrieved_at": datetime.now(timezone.utc).isoformat(), "youtube_ids": ids, "title": title, "transcript": transcript})
                transcript_path = str(transcript_path_obj.relative_to(ROOT))
                segment_count = len(transcript) if isinstance(transcript, list) else 1
            pages.append({"url": url, "slug": slug, "title": title, "youtube_ids": ids, "bytes": len(html), "duration": iso_duration_seconds(duration_match.group(1) if duration_match else None), "upload_date": upload_match.group(1) if upload_match else None, "transcript_path": transcript_path, "transcript_segment_count": segment_count})
    pages.sort(key=lambda x: x["url"])
    write_json(ROOT / "sources" / "sozai" / "transcript_pages_summary.json", {"retrieved_at": datetime.now(timezone.utc).isoformat(), "pages": pages, "source": "SozAI public transcript pages"})
    rows = []
    for page in pages:
        for source_id in page["youtube_ids"]:
            rows.append({"source_platform": "youtube", "source_id": source_id, "source_url": f"https://www.youtube.com/watch?v={source_id}", "title": page["title"], "uploader": "Neuro-sama Unofficial VODs (SozAI transcript index)", "upload_date": page.get("upload_date"), "duration": page.get("duration"), "discovery_method": "public_transcript_index", "discovery_source": page["url"], "transcript_source": "SozAI public transcript", "transcript_source_url": page["url"], "transcript_available": True, "transcript_path": page["transcript_path"], "transcript_segment_count": page["transcript_segment_count"], "subtitle_languages": ["en"], "rights_note": "Public transcript index; retain source attribution and verify original media before training use."})
    if rows:
        write_master(merge_rows(read_jsonl(ROOT / "manifest" / "master_video_manifest.jsonl"), rows))
    log_event("sozai_transcript_pages_complete", pages=len(pages), pages_with_youtube_ids=sum(bool(p["youtube_ids"]) for p in pages), manifest_leads=len(rows))
    print(json.dumps({"pages": len(pages), "pages_with_youtube_ids": sum(bool(p["youtube_ids"]) for p in pages), "manifest_leads": len(rows)}, ensure_ascii=False))
