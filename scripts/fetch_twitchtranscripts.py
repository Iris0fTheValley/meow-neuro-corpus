from __future__ import annotations

import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import requests

from manifest_tools import ROOT, merge_rows, read_jsonl, safe_name, write_json, write_master, log_event


UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131 Safari/537.36"
CHANNEL = "https://www.twitchtranscripts.com/channel/vedal987"


def clean_markup(value: str) -> str:
    value = re.sub(r"<(script|style).*?</\1>", " ", value, flags=re.I | re.S)
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def seconds(code: str) -> float:
    parts = [int(x) for x in code.split(":")]
    if len(parts) == 2:
        return float(parts[0] * 60 + parts[1])
    return float(parts[0] * 3600 + parts[1] * 60 + parts[2])


def extract_entries(page_html: str) -> list[dict]:
    text = clean_markup(page_html)
    pattern = re.compile(r"\[((?:\d{1,2}:)?\d{2}:\d{2})\]\s*(.*?)(?=\[(?:\d{1,2}:)?\d{2}:\d{2}\]|$)", re.S)
    entries = []
    for index, match in enumerate(pattern.finditer(text)):
        body = re.sub(r"\s+", " ", match.group(2)).strip()
        if not body:
            continue
        start = seconds(match.group(1))
        entries.append({"id": index, "start": start, "end": start, "text": body, "source_segment_id": index})
    return entries


def channel_vods(index_html: str) -> list[tuple[str, str]]:
    found = []
    for vod_id, title in re.findall(r'"url"\s*:\s*"https://www\.twitchtranscripts\.com/channel/vedal987/(\d+)".*?"name"\s*:\s*"(.*?)"', index_html, re.S):
        pair = (vod_id, html.unescape(title))
        if pair not in found:
            found.append(pair)
    return found


def windows(segments: list[dict]) -> list[list[dict]]:
    return [segments[i:i + 20] for i in range(0, len(segments), 16) if len(segments[i:i + 20]) >= 4]


if __name__ == "__main__":
    session = requests.Session()
    session.headers.update({"User-Agent": UA})
    index_response = session.get(CHANNEL, timeout=45)
    index_response.raise_for_status()
    index_path = ROOT / "sources" / "twitchtranscripts" / "vedal987.html"
    write_json(ROOT / "sources" / "twitchtranscripts" / "index_snapshot.json", {
        "source_url": CHANNEL, "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "http_status": index_response.status_code, "vod_count": len(channel_vods(index_response.text)),
    })
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(index_response.text, encoding="utf-8")
    registry_path = ROOT / "source_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {"schema_version": "0.1.0", "sources": []}
    if not any(x.get("source_id") == "twitchtranscripts_vedal987" for x in registry.get("sources", [])):
        registry.setdefault("sources", []).append({"source_id": "twitchtranscripts_vedal987", "platform": "twitchtranscripts", "kind": "public_transcript_archive", "name": "TwitchTranscripts vedal987", "url": CHANNEL, "uploader": "TwitchTranscripts", "priority": "high", "discovery_status": "expanded", "notes": "Public transcript snapshots for vedal987; preserve attribution and do not infer speaker identity without audio."})
        write_json(registry_path, registry)
    frontier_path = ROOT / "discovery_frontier.json"
    frontier = json.loads(frontier_path.read_text(encoding="utf-8")) if frontier_path.exists() else {"schema_version": "0.1.0", "frontier": []}
    if not any(x.get("url") == CHANNEL for x in frontier.get("frontier", [])):
        frontier.setdefault("frontier", []).append({"type": "transcript_index", "platform": "twitchtranscripts", "url": CHANNEL, "status": "expanded", "next_actions": ["capture newly indexed public transcripts", "align with audio when available", "keep speaker identity uncertain until diarization"]})
        write_json(frontier_path, frontier)
    rows = read_jsonl(ROOT / "manifest" / "master_video_manifest.jsonl")
    new_rows, raw_records, fetched = [], [], []
    for vod_id, title in channel_vods(index_response.text):
        source_id = f"vedal987:{vod_id}"
        url = f"https://www.twitchtranscripts.com/channel/vedal987/{vod_id}"
        page = session.get(url, timeout=60)
        page.raise_for_status()
        segments = extract_entries(page.text)
        page_path = ROOT / "sources" / "twitchtranscripts" / "pages" / f"{vod_id}.html"
        page_path.parent.mkdir(parents=True, exist_ok=True)
        page_path.write_text(page.text, encoding="utf-8")
        raw_path = ROOT / "raw_subtitles" / "twitchtranscripts" / f"{vod_id}.json"
        write_json(raw_path, {"source": "TwitchTranscripts public page", "source_url": url, "retrieved_at": datetime.now(timezone.utc).isoformat(), "vod_id": vod_id, "title": title, "segments": segments})
        new_rows.append({
            "source_platform": "twitchtranscripts", "source_id": source_id, "source_url": url, "title": title,
            "uploader": "TwitchTranscripts public archive", "discovery_method": "public_channel_index",
            "discovery_source": CHANNEL, "transcript_source": "TwitchTranscripts public page",
            "transcript_source_url": url, "transcript_available": bool(segments), "transcript_segment_count": len(segments),
            "asr_status": "source_transcript", "audio_download_status": "pending", "processing_status": "transcript_only",
            "language": "en", "subtitle_languages": ["en"], "rights_note": "Public transcript snapshot; speaker identity and audio provenance remain unvalidated.",
        })
        for index, window in enumerate(windows(segments)):
            turns = [{"timestamp": {"start": x["start"], "end": x["end"]}, "speaker": "UNKNOWN", "speaker_confidence": None, "text": x["text"], "source": {"type": "public_transcript", "attribution": "TwitchTranscripts public page", "segment_id": x["source_segment_id"], "raw_path": str(raw_path.relative_to(ROOT))}, "turn_boundary": "source_transcript_segment", "interruption": False, "overlap": False, "uncertain_transcription": False} for x in window]
            raw_records.append({"conversation_id": f"twitchtranscripts:{safe_name(source_id)}:{index:05d}", "source_video_id": source_id, "source_url": url, "participants": ["Neuro", "Evil", "Vedal"], "persona_scope": "uncertain", "turns": turns, "quality": {"response_length": len(" ".join(t["text"] for t in turns).split()), "multi_turn_value": min(1.0, len(turns) / 10.0), "asr_confidence": None, "speaker_confidence": 0.0}, "training_candidate": False, "candidate_reason": "public transcript speaker attribution and audio alignment pending"})
        fetched.append({"source_id": source_id, "segments": len(segments), "windows": len(windows(segments))})
        log_event("transcript_snapshot_complete", source_id=source_id, source_url=url, segments=len(segments), attribution="TwitchTranscripts public page")
    raw_out = ROOT / "conversations" / "conversations_raw.jsonl"
    clean_out = ROOT / "conversations" / "conversations_cleaned.jsonl"
    reject_out = ROOT / "datasets" / "rejected_segments.jsonl"
    existing_ids = set()
    if raw_out.exists():
        for line in raw_out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    existing_ids.add(json.loads(line).get("conversation_id"))
                except json.JSONDecodeError:
                    continue
    raw_records = [record for record in raw_records if record.get("conversation_id") not in existing_ids]
    for path in (raw_out, clean_out):
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            for record in raw_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with reject_out.open("a", encoding="utf-8", newline="\n") as handle:
        for record in raw_records:
            handle.write(json.dumps({"conversation_id": record["conversation_id"], "reason": "transcript_only_speaker_unvalidated", "record": record}, ensure_ascii=False) + "\n")
    write_master(merge_rows(rows, new_rows))
    print(json.dumps({"vods": len(fetched), "fetched": fetched, "conversation_records": len(raw_records)}, ensure_ascii=False))
