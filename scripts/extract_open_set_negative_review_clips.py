from __future__ import annotations

"""Extract review-only clips for metadata-derived open-set negative candidates."""

import json
import os
import re
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from manifest_tools import ROOT, read_jsonl, safe_name, write_json


BAD_TEXT = re.compile(r"(?:\[\s*(?:music|applause|laughter)\s*\]|<[^>]+>|seekiframe|javascript:)", re.I)


def ffmpeg_binary() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        bundled = Path(local_app_data) / "oopz" / "ffmpeg.exe"
        if bundled.exists():
            return str(bundled)
    return "ffmpeg"


def usable(text: str) -> bool:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text or BAD_TEXT.search(text):
        return False
    return len(re.findall(r"[A-Za-z0-9']+", text)) >= 2


def windows(segments: list[dict], max_windows: int) -> list[dict]:
    usable_segments = [
        item for item in sorted(segments, key=lambda x: float(x.get("start") or 0))
        if float(item.get("speaker_confidence") or 0) >= 0.72 and usable(str(item.get("text") or ""))
    ]
    candidates = []
    for index, first in enumerate(usable_segments):
        start = float(first.get("start") or 0)
        end = float(first.get("end") or start)
        chosen = [first]
        for item in usable_segments[index + 1 :]:
            if item.get("cluster") != first.get("cluster"):
                break
            gap = float(item.get("start") or 0) - end
            if gap > 2.5 or float(item.get("end") or 0) - start > 45:
                break
            chosen.append(item)
            end = float(item.get("end") or end)
        duration = end - start
        if 12 <= duration <= 60 and len(chosen) >= 2:
            candidates.append({
                "cluster": str(first.get("cluster")),
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(duration, 3),
                "segment_count": len(chosen),
                "mean_speaker_confidence": round(sum(float(x.get("speaker_confidence") or 0) for x in chosen) / len(chosen), 4),
                "text_preview": " ".join(re.sub(r"\s+", " ", str(x.get("text") or "")).strip() for x in chosen)[:500],
            })
    selected = []
    for item in sorted(candidates, key=lambda x: (-x["mean_speaker_confidence"], -x["duration"])):
        if any(item["start"] < old["end"] and old["start"] < item["end"] for old in selected):
            continue
        selected.append(item)
        if len(selected) >= max_windows:
            break
    return sorted(selected, key=lambda x: x["start"])


def extract(audio_path: Path, output_path: Path, start: float, end: float) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [ffmpeg_binary(), "-hide_banner", "-loglevel", "error", "-y", "-ss", str(start), "-i", str(audio_path), "-t", str(end - start), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "flac", str(output_path)]
    try:
        subprocess.run(command, check=True, timeout=180, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
    return output_path.is_file() and output_path.stat().st_size > 1024


def main() -> None:
    manifest_path = ROOT / "speaker_refs" / "open_set_negative_review_manifest.jsonl"
    source_rows = {str(row.get("source_id")): row for row in read_jsonl(ROOT / "manifest" / "master_video_manifest.jsonl") if row.get("source_id")}
    source_candidates = read_jsonl(manifest_path)
    clips = []
    counts = defaultdict(int)
    skipped = defaultdict(int)
    for source in source_candidates:
        source_id = str(source["source_id"])
        row = source_rows.get(source_id, {})
        if row.get("diarization_status") != "done" or not row.get("diarization_path") or not row.get("audio_path"):
            skipped["missing_diarization_or_audio_metadata"] += 1
            continue
        audio_path = ROOT / str(row["audio_path"])
        diarization_path = ROOT / str(row["diarization_path"])
        if not audio_path.exists() or not diarization_path.exists():
            skipped["missing_audio_or_diarization_file"] += 1
            continue
        try:
            diarization = json.loads(diarization_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            skipped["invalid_diarization"] += 1
            continue
        by_cluster = defaultdict(list)
        for segment in diarization.get("segments_with_speaker") or []:
            by_cluster[str(segment.get("cluster"))].append(segment)
        candidates = []
        for cluster_segments in by_cluster.values():
            candidates.extend(windows(cluster_segments, max_windows=2))
        selected = []
        for item in sorted(candidates, key=lambda x: (-x["mean_speaker_confidence"], -x["duration"])):
            if any(item["start"] < old["end"] and old["start"] < item["end"] for old in selected):
                continue
            selected.append(item)
            if len(selected) >= 2:
                break
        for index, item in enumerate(sorted(selected, key=lambda x: x["start"])):
            bucket = str(source.get("candidate_bucket") or "other")
            clip_rel = Path("speaker_refs") / "open_set_negative_clips" / bucket / f"{safe_name(source_id)}__{index:02d}.flac"
            if not extract(audio_path, ROOT / clip_rel, item["start"], item["end"]):
                skipped["clip_extraction_failed"] += 1
                continue
            clips.append({
                "clip_id": f"negative:{source_id}:{index:02d}",
                "source_id": source_id,
                "candidate_bucket": bucket,
                "source_platform": source.get("source_platform"),
                "source_url": source.get("source_url"),
                "title": source.get("title"),
                "audio_path": row.get("audio_path"),
                "clip_path": str(clip_rel),
                "start": item["start"],
                "end": item["end"],
                "duration": item["duration"],
                "cluster": item["cluster"],
                "mean_speaker_confidence": item["mean_speaker_confidence"],
                "text_preview": item["text_preview"],
                "gold_label": None,
                "review_status": "PENDING_MANUAL_AUDIO_REVIEW",
                "use_for_benchmark": False,
                "metadata_only_source_hint": True,
                "review_instruction": "Assign exactly one of NEURO_FAMILY, VEDAL, OTHER, or UNKNOWN after listening; bucket is not a label.",
            })
            counts[bucket] += 1

    output_path = ROOT / "speaker_refs" / "open_set_negative_review_clips.jsonl"
    output_path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in clips), encoding="utf-8")
    bucket_counts = {bucket: counts.get(bucket, 0) for bucket in ("chat_tts", "singing", "game_voice", "guest")}
    report = {
        "schema_version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "CLIPS_PENDING_MANUAL_AUDIO_REVIEW",
        "source_candidate_count": len(source_candidates),
        "clip_count": len(clips),
        "bucket_counts": bucket_counts,
        "missing_buckets": [bucket for bucket, count in bucket_counts.items() if count == 0],
        "gold_label_count": 0,
        "benchmark_ready_count": 0,
        "output": str(output_path.relative_to(ROOT)),
        "skipped": dict(skipped),
        "policy": "These are review clips, not negative gold. Metadata buckets must never be treated as speaker identity without audio-level validation.",
    }
    write_json(ROOT / "reports" / "open_set_negative_review_clips.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
