from __future__ import annotations

"""Mine auditable speaker-reference candidates from existing acoustic outputs.

This stage deliberately does not assert speaker identity.  A label is only a
metadata hint from a source whose participant list contains exactly one known
identity.  The resulting clips are therefore provisional inputs to the later
embedding/gold-set benchmark, never training data or trusted mappings.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from manifest_tools import ROOT, read_jsonl, safe_name, write_json


KNOWN = {"neuro": "NEURO", "evil": "EVIL", "vedal": "VEDAL"}
BAD_TEXT = re.compile(
    r"(?:\[\s*(?:music|applause|laughter)\s*\]|<[^>]+>|seekiframe|javascript:)", re.I
)
SINGING_HINT = re.compile(r"\b(?:karaoke|singing|sings|song|bangers)\b", re.I)


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


def known_participant_hints(row: dict) -> list[str]:
    found: set[str] = set()
    for value in row.get("participants") or []:
        value_low = str(value).lower()
        for needle, identity in KNOWN.items():
            if needle in value_low:
                found.add(identity)
    return sorted(found)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def text_is_usable(text: str) -> bool:
    if not text or BAD_TEXT.search(text):
        return False
    words = re.findall(r"[A-Za-z0-9']+", text)
    if len(words) < 2:
        return False
    # Keep unusual speech, but reject obvious repeated ASR hallucinations.
    normalized = re.sub(r"[^a-z0-9 ]", "", text.lower())
    if len(normalized) >= 20 and len(set(normalized.split())) <= 3:
        return False
    return True


def make_windows(segments: list[dict], *, max_seconds: float = 45.0) -> list[dict]:
    usable = [
        s for s in sorted(segments, key=lambda x: float(x.get("start") or 0))
        if float(s.get("speaker_confidence") or 0) >= 0.78 and text_is_usable(clean_text(s.get("text")))
    ]
    windows: list[dict] = []
    for i, first in enumerate(usable):
        cluster = first.get("cluster")
        start = float(first.get("start") or 0)
        end = float(first.get("end") or start)
        selected = [first]
        for candidate in usable[i + 1 :]:
            if candidate.get("cluster") != cluster:
                break
            gap = float(candidate.get("start") or 0) - end
            if gap > 2.5 or float(candidate.get("end") or 0) - start > max_seconds:
                break
            selected.append(candidate)
            end = float(candidate.get("end") or end)
        duration = end - start
        if duration < 15.0 or duration > 60.0 or len(selected) < 2:
            continue
        text = " ".join(clean_text(x.get("text")) for x in selected)
        windows.append(
            {
                "cluster": cluster,
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(duration, 3),
                "segment_count": len(selected),
                "mean_speaker_confidence": round(
                    sum(float(x.get("speaker_confidence") or 0) for x in selected) / len(selected), 4
                ),
                "text": text,
                "source_asr_segment_ids": [x.get("source_asr_segment_id") for x in selected],
            }
        )
    # Non-overlapping, high-confidence clips give cleaner reference material.
    chosen: list[dict] = []
    for window in sorted(windows, key=lambda x: (-x["mean_speaker_confidence"], -x["duration"])):
        if any(window["start"] < x["end"] and x["start"] < window["end"] for x in chosen):
            continue
        chosen.append(window)
        if len(chosen) >= 4:
            break
    return sorted(chosen, key=lambda x: x["start"])


def extract_clip(audio_path: Path, output_path: Path, start: float, end: float) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_binary(), "-hide_banner", "-loglevel", "error", "-y",
        "-ss", str(start), "-i", str(audio_path), "-t", str(end - start),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "flac", str(output_path),
    ]
    try:
        subprocess.run(command, check=True, timeout=180, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
    return output_path.is_file() and output_path.stat().st_size > 1024


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-identity", type=int, default=24)
    ap.add_argument("--no-extract", action="store_true", help="write time ranges only; do not create FLAC clips")
    args = ap.parse_args()

    rows = read_jsonl(ROOT / "manifest" / "master_video_manifest.jsonl")
    candidates: list[dict] = []
    source_counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    seen_sources: set[str] = set()

    for row in rows:
        if row.get("diarization_status") != "done" or not row.get("audio_path") or not row.get("diarization_path"):
            continue
        hints = known_participant_hints(row)
        if len(hints) != 1:
            if "vedal" in {str(x).lower() for x in (row.get("participants") or [])}:
                skipped["vedal_multi_identity_source"] += 1
            continue
        identity = hints[0]
        if source_counts[identity] >= args.max_per_identity:
            continue
        audio_path = ROOT / str(row["audio_path"])
        diar_path = ROOT / str(row["diarization_path"])
        if not audio_path.exists() or not diar_path.exists():
            skipped["missing_audio_or_diarization"] += 1
            continue
        diar = load_json(diar_path)
        segments = diar.get("segments_with_speaker") or []
        counts = Counter(str(s.get("cluster")) for s in segments)
        total = sum(counts.values()) or 1
        cluster_duration = defaultdict(float)
        for s in segments:
            cluster_duration[str(s.get("cluster"))] += max(0.0, float(s.get("end") or 0) - float(s.get("start") or 0))
        if not counts:
            skipped["no_diarized_segments"] += 1
            continue
        dominant_cluster, dominant_count = counts.most_common(1)[0]
        dominant_share = dominant_count / total
        if dominant_share < 0.60:
            skipped["no_dominant_cluster"] += 1
            continue
        windows = make_windows([s for s in segments if str(s.get("cluster")) == dominant_cluster])
        if not windows:
            skipped["no_high_confidence_windows"] += 1
            continue
        if SINGING_HINT.search(str(row.get("title") or "")):
            # Singing is preserved elsewhere, but is a poor primary voiceprint.
            skipped["singing_or_karaoke_title"] += 1
            continue
        seen_sources.add(str(row["source_id"]))
        source_counts[identity] += 1
        for index, window in enumerate(windows):
            clip_rel = Path("speaker_refs") / "provisional_clips" / identity / f"{safe_name(str(row['source_id']))}__{index:02d}.flac"
            clip_path = ROOT / clip_rel
            extracted = True if args.no_extract else extract_clip(audio_path, clip_path, window["start"], window["end"])
            if not extracted:
                skipped["clip_extraction_failed"] += 1
                continue
            candidates.append(
                {
                    "candidate_id": f"{identity.lower()}:{row['source_id']}:{index:02d}",
                    "identity_hint": identity,
                    "candidate_status": "provisional_not_trusted",
                    "source_id": row["source_id"],
                    "source_platform": row.get("source_platform"),
                    "source_url": row.get("source_url"),
                    "title": row.get("title"),
                    "audio_path": row["audio_path"],
                    "clip_path": str(clip_rel),
                    "start": window["start"],
                    "end": window["end"],
                    "duration": window["duration"],
                    "cluster": window["cluster"],
                    "dominant_cluster_share": round(dominant_share, 4),
                    "dominant_cluster_duration_seconds": round(cluster_duration[dominant_cluster], 3),
                    "mean_speaker_confidence": window["mean_speaker_confidence"],
                    "segment_count": window["segment_count"],
                    "text_preview": window["text"][:500],
                    "evidence": {
                        "metadata_participant_hints": row.get("participants") or [],
                        "identity_hint_basis": "exactly one known identity in source participant metadata",
                        "acoustic_basis": "dominant anonymous ECAPA cluster with high-confidence contiguous speech",
                        "cross_source_validation": "pending",
                        "identity_verification": "pending_gold_benchmark",
                    },
                }
            )

    candidates.sort(key=lambda x: (x["identity_hint"], x["source_id"], x["start"]))
    out_path = ROOT / "speaker_refs" / "provisional_reference_candidates.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in candidates), encoding="utf-8")

    by_identity: dict[str, list[dict]] = defaultdict(list)
    for item in candidates:
        by_identity[item["identity_hint"]].append(item)
    bank = {
        "schema_version": "0.1.0",
        "speaker_reference_bank_version": "provisional-2026-09-11-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PROVISIONAL_PENDING_GOLD_VALIDATION",
        "policy": "Metadata hints never become trusted identity without cross-source embedding validation on a fixed gold set.",
        "identities": {
            identity: {
                "candidate_count": len(by_identity.get(identity, [])),
                "source_count": len({x["source_id"] for x in by_identity.get(identity, [])}),
                "trusted_clip_count": 0,
                "status": "reference_pending",
                "candidate_ids": [x["candidate_id"] for x in by_identity.get(identity, [])],
            }
            for identity in ("NEURO", "EVIL", "VEDAL")
        },
        "negative_other": {"status": "pending_construction", "candidate_count": 0},
        "provenance": {
            "manifest": "manifest/master_video_manifest.jsonl",
            "diarization": "SpeechBrain ECAPA anonymous clustering outputs",
            "extraction": "ffmpeg mono 16 kHz FLAC clips",
            "source_count": len(seen_sources),
            "skipped": dict(skipped),
        },
    }
    write_json(ROOT / "speaker_refs" / "reference_bank.json", bank)
    gold = {
        "schema_version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PENDING_EMBEDDING_VALIDATION",
        "policy": "All labels below are metadata-derived provisional labels; do not use for training until validated.",
        "identities": {
            identity: {"candidate_ids": [x["candidate_id"] for x in by_identity.get(identity, [])], "validated": False}
            for identity in ("NEURO", "EVIL", "VEDAL")
        },
        "negative_other": {"candidate_ids": [], "validated": False},
        "required_benchmark": [
            "cross-source precision proxy",
            "Neuro/Evil confusion",
            "known-vs-unknown false positives",
            "score and margin distributions",
            "runtime, VRAM, and RAM",
        ],
    }
    write_json(ROOT / "speaker_refs" / "gold_validation.json", gold)
    print(json.dumps({"candidates": len(candidates), "source_count": len(seen_sources), "by_identity": {k: len(v) for k, v in by_identity.items()}, "skipped": dict(skipped)}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
