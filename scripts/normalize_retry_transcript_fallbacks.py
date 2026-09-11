from __future__ import annotations

"""Normalize fetched VTT fallback files into provenance-preserving anonymous segments."""

import html
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from manifest_tools import ROOT, write_json


TIMING = re.compile(r"^(\d{2}:\d{2}:\d{2}(?:\.\d{3})?)\s+-->\s+(\d{2}:\d{2}:\d{2}(?:\.\d{3})?)")
TAG = re.compile(r"<[^>]+>")


def seconds(value: str) -> float:
    hours, minutes, rest = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(rest)


def clean(lines: list[str]) -> str:
    text = " ".join(line.strip() for line in lines if line.strip())
    text = TAG.sub("", html.unescape(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main() -> None:
    input_root = ROOT / "retry_queue" / "transcript_fallbacks"
    output_path = ROOT / "retry_queue" / "transcript_fallbacks_normalized.jsonl"
    segments = []
    stats = Counter()
    source_counters = Counter()
    for path in sorted(input_root.glob("*.vtt")):
        source_id = path.name.split(".", 1)[0]
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        index = 0
        while index < len(lines):
            match = TIMING.match(lines[index].strip())
            if not match:
                index += 1
                continue
            start, end = seconds(match.group(1)), seconds(match.group(2))
            index += 1
            payload = []
            while index < len(lines) and lines[index].strip() and not TIMING.match(lines[index].strip()):
                if lines[index].strip() not in {"WEBVTT", "NOTE"}:
                    payload.append(lines[index])
                index += 1
            text = clean(payload)
            if not text or end <= start:
                stats["skipped_empty_or_invalid"] += 1
                continue
            segment_number = source_counters[source_id]
            source_counters[source_id] += 1
            segments.append({
                "source_id": source_id,
                "segment_id": f"{source_id}:vtt:{segment_number:06d}",
                "start": round(start, 3),
                "end": round(end, 3),
                "text": text,
                "speaker_identity": "UNKNOWN",
                "speaker_mapping_status": "anonymous_transcript_only",
                "transcript_source": str(Path("retry_queue") / "transcript_fallbacks" / path.name),
                "provenance": "public_subtitle_or_automatic_caption_fallback",
                "training_candidate": False,
            })
            stats["segments"] += 1
    output_path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in segments), encoding="utf-8")
    report = {
        "schema_version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "TRANSCRIPT_FALLBACK_NORMALIZED_ANONYMOUS",
        "input_file_count": len(list(input_root.glob("*.vtt"))),
        "segment_count": len(segments),
        "source_count": len({item["source_id"] for item in segments}),
        "output": str(output_path.relative_to(ROOT)),
        "training_candidate_count": 0,
        "stats": dict(stats),
        "policy": "Subtitle fallback segments remain anonymous and are not merged into family-mapped conversations or S/A candidates without audio and speaker QA.",
    }
    write_json(ROOT / "reports" / "retry_transcript_fallback_normalization.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
