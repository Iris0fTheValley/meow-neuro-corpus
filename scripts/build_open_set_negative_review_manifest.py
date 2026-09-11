from __future__ import annotations

"""Build a metadata-derived negative-reference review queue without assigning speaker gold labels."""

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from manifest_tools import ROOT, read_jsonl, write_json


PATTERNS = {
    "chat_tts": [r"chat\s*tts", r"chattts", r"text[- ]to[- ]speech", r"文字转语音", r"语音合成"],
    "singing": [r"sing(?:ing|er)?", r"karaoke", r"cover", r"唱歌", r"歌曲", r"歌回"],
    "game_voice": [r"game", r"minecraft", r"valorant", r"cyberpunk", r"hollow\s+knight", r"vr\s*chat", r"游戏", r"游玩"],
    "guest": [r"guest", r"collab", r"collaboration", r"interview", r"with\s+[a-z][a-z0-9_-]{2,}", r"嘉宾", r"联动", r"合作", r"访谈"],
}


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def compile_patterns() -> dict[str, list[re.Pattern[str]]]:
    return {bucket: [re.compile(pattern, re.IGNORECASE) for pattern in patterns] for bucket, patterns in PATTERNS.items()}


def main() -> None:
    patterns = compile_patterns()
    target_sources = set()
    for path in (
        ROOT / "speaker_refs" / "neuro_family_reference_candidates.jsonl",
        ROOT / "speaker_refs" / "vedal_provisional_reference_candidates.jsonl",
    ):
        for row in read_jsonl(path):
            if row.get("source_id"):
                target_sources.add(str(row["source_id"]))

    rows = read_jsonl(ROOT / "manifest" / "master_video_manifest.jsonl")
    entries = []
    counts = Counter()
    for row in rows:
        source_id = str(row.get("source_id") or "")
        if not source_id or source_id in target_sources:
            continue
        audio_path = row.get("audio_path")
        if not audio_path or not (ROOT / str(audio_path)).exists():
            continue
        metadata = load_json(ROOT / str(row.get("metadata_path"))) if row.get("metadata_path") else {}
        fields = [
            str(row.get("title") or ""),
            str(row.get("discovery_query") or ""),
            str(row.get("participants") or ""),
            str(metadata.get("title") or ""),
            str(metadata.get("fulltitle") or ""),
            str(metadata.get("description") or ""),
            str(metadata.get("tags") or ""),
        ]
        searchable = "\n".join(fields)
        matched = []
        evidence = {}
        for bucket, bucket_patterns in patterns.items():
            hits = sorted({match.group(0) for pattern in bucket_patterns if (match := pattern.search(searchable))})
            if hits:
                matched.append(bucket)
                evidence[bucket] = hits[:8]
        if not matched:
            continue
        # Category is a review bucket, never a speaker label. Preserve every hit for manual inspection.
        primary = matched[0]
        counts[primary] += 1
        entries.append({
            "review_id": f"negative_review:{source_id}",
            "source_id": source_id,
            "source_platform": row.get("source_platform"),
            "source_url": row.get("source_url"),
            "title": row.get("title"),
            "audio_path": audio_path,
            "duration": row.get("duration"),
            "candidate_bucket": primary,
            "matched_buckets": matched,
            "metadata_evidence": evidence,
            "metadata_only": True,
            "gold_label": None,
            "review_status": "PENDING_MANUAL_AUDIO_REVIEW",
            "use_for_benchmark": False,
            "review_instruction": "Listen to isolated clips and label each clip NEURO_FAMILY, VEDAL, OTHER, or UNKNOWN; source metadata is not a speaker label.",
        })

    entries.sort(key=lambda item: (item["candidate_bucket"], item["source_id"]))
    jsonl_path = ROOT / "speaker_refs" / "open_set_negative_review_manifest.jsonl"
    jsonl_path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in entries), encoding="utf-8")
    bucket_counts = {bucket: counts.get(bucket, 0) for bucket in PATTERNS}
    report = {
        "schema_version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "METADATA_CANDIDATES_PENDING_MANUAL_AUDIO_REVIEW",
        "candidate_count": len(entries),
        "bucket_counts": bucket_counts,
        "missing_buckets": [bucket for bucket, count in bucket_counts.items() if count == 0],
        "source_exclusion_count": len(target_sources),
        "output": str(jsonl_path.relative_to(ROOT)),
        "gold_label_count": 0,
        "benchmark_ready_count": 0,
        "policy": "Metadata keywords only create a review queue; no source or clip is treated as OTHER, guest, chat TTS, singing, or game voice until audio-level manual/external gold review.",
    }
    write_json(ROOT / "reports" / "open_set_negative_review_manifest.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
