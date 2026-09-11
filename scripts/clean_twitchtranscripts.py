from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone

from manifest_tools import ROOT, write_json


SEEK_IFRAME_SUFFIX = re.compile(r"\s*seekIframe\(\d+\)\)\(\)\"?>\s*$", re.I)
HTML_TAG = re.compile(r"<[^>]+>")


def clean_text(raw: str) -> tuple[str, list[str]]:
    reasons = []
    text = re.sub(r"\s+", " ", raw or "").strip()
    stripped = SEEK_IFRAME_SUFFIX.sub("", text).strip()
    if stripped != text:
        reasons.append("seekiframe_suffix_removed")
    text = HTML_TAG.sub(" ", stripped)
    text = re.sub(r"\s+", " ", text).strip()
    if text != stripped and "html_fragment_removed" not in reasons:
        reasons.append("html_fragment_removed")
    return text, reasons


def main() -> None:
    raw_dir = ROOT / "raw_subtitles" / "twitchtranscripts"
    clean_dir = ROOT / "normalized_subtitles" / "twitchtranscripts"
    quarantine_dir = ROOT / "quarantine" / "twitchtranscripts"
    clean_dir.mkdir(parents=True, exist_ok=True)
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    totals = Counter()
    files = []
    for raw_path in sorted(raw_dir.glob("*.json")):
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        cleaned_segments = []
        quarantined = []
        for segment in payload.get("segments") or []:
            raw_text = str(segment.get("text") or "")
            text, reasons = clean_text(raw_text)
            if not text or SEEK_IFRAME_SUFFIX.search(text) or "seekiframe" in text.lower():
                quarantined.append({"raw": segment, "reason": "unresolved_html_or_empty_after_source_cleaning"})
                totals["quarantined"] += 1
                continue
            record = {
                "id": segment.get("id"),
                "source_segment_id": segment.get("source_segment_id", segment.get("id")),
                "start": segment.get("start"),
                "end": segment.get("end"),
                "raw_text": raw_text,
                "text": text,
                "cleaning_reasons": reasons,
                "cleaning_status": "cleaned" if reasons else "unchanged",
            }
            cleaned_segments.append(record)
            totals["segments_kept"] += 1
            if reasons:
                totals["segments_modified"] += 1
                for reason in reasons:
                    totals[reason] += 1
        clean_payload = {
            "schema_version": "0.1.0",
            "cleaned_at": datetime.now(timezone.utc).isoformat(),
            "status": "SOURCE_SPECIFIC_CLEANED_RAW_PRESERVED",
            "source": payload.get("source"),
            "source_url": payload.get("source_url"),
            "vod_id": payload.get("vod_id"),
            "title": payload.get("title"),
            "raw_path": str((ROOT / "raw_subtitles" / "twitchtranscripts" / raw_path.name).relative_to(ROOT)),
            "cleaning_policy": "Remove only the known TwitchTranscripts seekIframe suffix and HTML fragments; unresolved records are quarantined.",
            "segments": cleaned_segments,
        }
        (clean_dir / raw_path.name).write_text(json.dumps(clean_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (quarantine_dir / raw_path.name).write_text(json.dumps({"source": payload.get("source"), "vod_id": payload.get("vod_id"), "records": quarantined}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        files.append({"file": raw_path.name, "raw_segments": len(payload.get("segments") or []), "cleaned_segments": len(cleaned_segments), "quarantined_segments": len(quarantined)})
    report = {
        "schema_version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "SOURCE_SPECIFIC_CLEANING_COMPLETE",
        "raw_preserved": True,
        "files": files,
        "totals": dict(totals),
        "cleaned_dir": "normalized_subtitles/twitchtranscripts",
        "quarantine_dir": "quarantine/twitchtranscripts",
    }
    write_json(ROOT / "reports" / "twitchtranscript_cleaning.json", report)
    print(json.dumps({"files": len(files), "totals": dict(totals)}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
