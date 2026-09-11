from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone

from manifest_tools import ROOT, read_jsonl, write_json


PATTERNS = {
    "html_or_javascript": re.compile(r"(?:<\/?[a-z][^>]*>|seekiframe\s*\(|javascript:|(?:document|window)\.(?:location|parent|top|body|addEventListener|postMessage)|window\[['\"])", re.I),
    "music_marker": re.compile(r"\[\s*(?:music|♪|applause|laughter)\s*\]", re.I),
    "singing_marker": re.compile(r"\b(?:singing|sings|karaoke|lyrics|song)\b", re.I),
}


def transcript_kind(path: str) -> str:
    low = path.lower()
    if "ladev" in low:
        return "ladev_transcript"
    if "openai" in low:
        return "openai_whisper"
    if "faster" in low:
        return "faster_whisper"
    return "other"


def main() -> None:
    rows = read_jsonl(ROOT / "manifest" / "master_video_manifest.jsonl")
    total_segments = 0
    total_chars = 0
    file_counts = Counter()
    kind_counts = defaultdict(Counter)
    source_counts = defaultdict(Counter)
    examples = defaultdict(list)
    for row in rows:
        path_value = row.get("asr_path")
        if not path_value:
            continue
        path = ROOT / str(path_value)
        if not path.exists():
            continue
        kind = transcript_kind(str(path_value))
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            file_counts["unreadable"] += 1
            continue
        segments = payload.get("segments") or []
        previous_norm = None
        for segment in segments:
            text = re.sub(r"\s+", " ", str(segment.get("text") or "")).strip()
            if not text:
                continue
            total_segments += 1
            total_chars += len(text)
            source_id = str(row.get("source_id"))
            for name, pattern in PATTERNS.items():
                if pattern.search(text):
                    kind_counts[kind][name] += 1
                    source_counts[source_id][name] += 1
                    if len(examples[name]) < 12:
                        examples[name].append({"source_id": source_id, "path": str(path_value), "text": text[:300]})
            normalized = re.sub(r"[^a-z0-9]+", "", text.lower())
            words = re.findall(r"[a-z0-9']+", text.lower())
            if previous_norm and normalized == previous_norm and len(normalized) >= 12:
                kind_counts[kind]["consecutive_exact_repeat"] += 1
                source_counts[source_id]["consecutive_exact_repeat"] += 1
                if len(examples["consecutive_exact_repeat"]) < 12:
                    examples["consecutive_exact_repeat"].append({"source_id": source_id, "path": str(path_value), "text": text[:300]})
            if len(words) >= 12 and len(set(words)) <= max(3, len(words) // 8):
                kind_counts[kind]["low_diversity_repetition"] += 1
                source_counts[source_id]["low_diversity_repetition"] += 1
            previous_norm = normalized
        file_counts[kind] += 1

    high_value_sources = []
    for source_id, counts in source_counts.items():
        contamination = sum(counts.values())
        if contamination:
            high_value_sources.append({"source_id": source_id, "counts": dict(counts), "total_flags": contamination})
    high_value_sources.sort(key=lambda x: (-x["total_flags"], x["source_id"]))
    report = {
        "schema_version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "AUDIT_ONLY_NO_DELETIONS",
        "files_scanned": sum(file_counts.values()),
        "file_counts": dict(file_counts),
        "segments_scanned": total_segments,
        "characters_scanned": total_chars,
        "flags_by_transcript_kind": {k: dict(v) for k, v in kind_counts.items()},
        "flagged_source_count": len(high_value_sources),
        "highest_flag_sources": high_value_sources[:100],
        "examples": dict(examples),
        "policy": "Suspicious is not reject; raw transcripts remain unchanged and any cleaning must be source-specific and reversible.",
    }
    write_json(ROOT / "reports" / "transcript_contamination_audit.json", report)
    print(json.dumps({"files_scanned": report["files_scanned"], "segments_scanned": total_segments, "flagged_source_count": len(high_value_sources), "flags_by_transcript_kind": report["flags_by_transcript_kind"]}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
