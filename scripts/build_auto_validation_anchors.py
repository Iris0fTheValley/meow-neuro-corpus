from __future__ import annotations

"""Build automatic speaker-validation anchors from independent source evidence.

This artifact is deliberately separate from the human gold queue.  A row can be
an auto anchor only when metadata identifies the person, the transcript contains
a first-person self-identification, and diarization supplies a high-confidence
cluster for the same time interval.  The result is suitable for verifier
calibration and challenge-set construction, but it never marks training data.
"""

import argparse
import json
import re
import subprocess
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import imageio_ffmpeg
import soundfile as sf

from manifest_tools import ROOT, read_jsonl, safe_name, write_json


FAMILY_NAMES = {"neuro", "neuro-sama", "evil", "evil neuro"}
VEDAL_NAMES = {"vedal"}
IGNORE_NAMES = {"neuro-sama", "neuro", "evil", "evil neuro", "vedal"}
SELF_PREFIX = r"(?:i\s*['’]?m|i am|my name is|this is|hey(?: guys)?[, ]+i\s*['’]?m)"


def norm(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower()).strip()


def metadata_people(row: dict) -> set[str]:
    people = set()
    for value in row.get("participants") or []:
        name = norm(value)
        if not name or name in IGNORE_NAMES or re.fullmatch(r"[0-9\-#]+", name):
            continue
        if re.search(r"[a-z]", name):
            people.add(name)
    return people


def source_text(row: dict) -> str:
    return norm(" ".join(str(row.get(key) or "") for key in ("title", "discovery_query", "uploader")))


def family_metadata(row: dict) -> bool:
    text = source_text(row)
    participants = {norm(x) for x in row.get("participants") or []}
    return bool(participants & FAMILY_NAMES) and bool(re.search(r"\b(?:neuro|evil)\b", text))


def vedal_metadata(row: dict) -> bool:
    text = source_text(row)
    participants = {norm(x) for x in row.get("participants") or []}
    return "vedal" in participants and bool(re.search(r"\bvedal\b|\bdev(?:eloper)?\b|developer|creator", text))


def self_labels(text: str, people: set[str]) -> set[str]:
    low = norm(text)
    labels: set[str] = set()
    def valid(match: re.Match[str]) -> bool:
        before = low[max(0, match.start() - 32) : match.start()]
        return not re.search(r"\b(?:pretend|pretending|roleplay|call|called|like|quote)\s*$", before)

    family_pattern = re.compile(rf"\b{SELF_PREFIX}\s+(?:a\s+|an\s+)?(?:neuro(?:-sama)?|evil(?:\s+neuro)?)(?!['’]s)\b|\b(?:neuro|evil)\s+here\b")
    if any(valid(match) for match in family_pattern.finditer(low)):
        labels.add("NEURO_FAMILY")
    vedal_pattern = re.compile(rf"\b{SELF_PREFIX}\s+vedal(?!['’]s)\b|\bvedal here\b")
    if any(valid(match) for match in vedal_pattern.finditer(low)):
        labels.add("VEDAL")
    for person in people:
        escaped = re.escape(person)
        pattern = re.compile(rf"\b{SELF_PREFIX}\s+{escaped}(?!['’]s)\b|\b{escaped} here\b")
        if any(valid(match) for match in pattern.finditer(low)):
            labels.add("OTHER")
    return labels


def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def diar_match(segments: list[dict], start: float, end: float) -> dict | None:
    midpoint = (start + end) / 2.0
    candidates = []
    for segment in segments:
        s = float(segment.get("start") or 0.0)
        e = float(segment.get("end") or s)
        ov = overlap(start, end, s, e)
        if ov > 0 or s <= midpoint <= e:
            candidates.append((ov, float(segment.get("speaker_confidence") or 0.0), segment))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def extract_clip(audio: Path, start: float, end: float, output: Path) -> bool:
    duration = min(18.0, max(4.0, end - start))
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
        "-ss", str(max(0.0, start)), "-i", str(audio), "-t", str(duration),
        "-vn", "-sn", "-dn", "-ac", "1", "-ar", "16000", "-f", "flac", str(output),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        return result.returncode == 0 and output.exists() and output.stat().st_size > 1024
    except (OSError, subprocess.SubprocessError):
        return False


def read_transcript(source_id: str) -> list[dict]:
    paths = sorted((ROOT / "asr").glob(f"{safe_name(source_id)}.*.json"))
    preferred = [p for p in paths if "ladev-transcript" in p.name]
    preferred += [p for p in paths if p not in preferred and "openai" in p.name]
    for path in preferred:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        segments = data.get("segments") or []
        if segments:
            return segments
    return []


def existing_family_anchors(manifest: dict[str, dict]) -> list[dict]:
    anchors = []
    path = ROOT / "speaker_refs" / "neuro_family_reference_candidates.jsonl"
    if not path.exists():
        return anchors
    for row in read_jsonl(path):
        source = manifest.get(str(row.get("source_id")), {})
        if not family_metadata(source):
            continue
        if float(row.get("dominant_cluster_share") or 0.0) < 0.99:
            continue
        if float(row.get("mean_speaker_confidence") or 0.0) < 0.75:
            continue
        clip = ROOT / str(row.get("clip_path") or "")
        if not clip.exists():
            continue
        anchors.append({
            "anchor_id": f"family_meta:{row['candidate_id']}",
            "label": "NEURO_FAMILY",
            "source_id": row.get("source_id"),
            "clip_path": str(clip.relative_to(ROOT)),
            "start": row.get("start"),
            "end": row.get("end"),
            "cluster": row.get("cluster"),
            "text_preview": row.get("text_preview"),
            "evidence": {
                "metadata_participants": source.get("participants"),
                "metadata_title": source.get("title"),
                "metadata_family_signal": True,
                "acoustic_cluster_share": row.get("dominant_cluster_share"),
                "acoustic_confidence": row.get("mean_speaker_confidence"),
                "dual_model_gate_passed": bool((row.get("verification") or {}).get("dual_model_gate_passed")),
                "auto_anchor_not_gold": True,
            },
        })
    return anchors


def build_self_id_anchors(manifest: dict[str, dict], min_confidence: float) -> list[dict]:
    anchors = []
    for source_id, source in manifest.items():
        audio = ROOT / str(source.get("audio_path") or "")
        diar_path = ROOT / str(source.get("diarization_path") or "")
        if not audio.exists() or not diar_path.exists():
            continue
        people = metadata_people(source)
        text = source_text(source)
        if not people and "vedal" not in {norm(x) for x in source.get("participants") or []}:
            continue
        transcript = read_transcript(source_id)
        if not transcript:
            continue
        try:
            diar = json.loads(diar_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        diar_segments = diar.get("segments_with_speaker") or []
        for index, segment in enumerate(transcript):
            seg_text = str(segment.get("text") or "")
            labels = self_labels(seg_text, people)
            if not labels:
                continue
            start = float(segment.get("start") or 0.0)
            end = float(segment.get("end") or start)
            if end <= start:
                continue
            matched = diar_match(diar_segments, start, end)
            if not matched or float(matched.get("speaker_confidence") or 0.0) < min_confidence:
                continue
            cluster = str(matched.get("cluster") or "UNKNOWN")
            # Metadata must independently name the claimed identity.
            accepted = set()
            participant_names = {norm(x) for x in source.get("participants") or []}
            if "VEDAL" in labels and "vedal" in participant_names and vedal_metadata(source):
                accepted.add("VEDAL")
            if "NEURO_FAMILY" in labels and participant_names & FAMILY_NAMES and family_metadata(source):
                accepted.add("NEURO_FAMILY")
            if "OTHER" in labels:
                # A guest name is accepted only when it appears in the explicit
                # participant metadata and in first-person audio text. The
                # participant list is the independent source-level identity
                # evidence; title/query matching is optional because many
                # indexed source titles omit the guest name.
                if any(person in {norm(x) for x in source.get("participants") or []} for person in people):
                    accepted.add("OTHER")
            for label in sorted(accepted):
                clip_start = max(0.0, start - 1.5)
                clip_end = end + 1.5
                rel = Path("speaker_refs") / "auto_anchor_clips" / label / f"{safe_name(source_id)}__{index:04d}.flac"
                output = ROOT / rel
                output.parent.mkdir(parents=True, exist_ok=True)
                if not extract_clip(audio, clip_start, clip_end, output):
                    continue
                anchors.append({
                    "anchor_id": f"selfid:{label}:{source_id}:{index:04d}",
                    "label": label,
                    "source_id": source_id,
                    "clip_path": str(rel),
                    "start": clip_start,
                    "end": clip_end,
                    "cluster": cluster,
                    "text_preview": seg_text[:500],
                    "evidence": {
                        "metadata_participants": source.get("participants"),
                        "metadata_title": source.get("title"),
                        "metadata_query": source.get("discovery_query"),
                        "transcript_self_identification": seg_text,
                        "diarization_confidence": round(float(matched.get("speaker_confidence") or 0.0), 6),
                        "independent_evidence": ["explicit_metadata_identity", "first_person_transcript_identity", "overlapping_diarized_cluster"],
                        "auto_anchor_not_gold": True,
                    },
                })
    return anchors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-confidence", type=float, default=0.70)
    args = ap.parse_args()
    manifest = {str(row.get("source_id")): row for row in read_jsonl(ROOT / "manifest" / "master_video_manifest.jsonl") if row.get("record_type") != "manifest_header" and row.get("source_id")}
    anchors = existing_family_anchors(manifest) + build_self_id_anchors(manifest, args.min_confidence)
    dedup = {}
    for row in anchors:
        dedup[row["anchor_id"]] = row
    anchors = list(dedup.values())
    source_counts = Counter(str(row["source_id"]) for row in anchors)
    label_counts = Counter(str(row["label"]) for row in anchors)
    source_label_counts = defaultdict(set)
    for row in anchors:
        source_label_counts[row["label"]].add(str(row["source_id"]))
    out = ROOT / "speaker_refs" / "auto_validation_anchors.jsonl"
    out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in sorted(anchors, key=lambda x: (x["label"], str(x["source_id"]), x["anchor_id"]))) , encoding="utf-8")
    challenge = []
    negative_path = ROOT / "speaker_refs" / "open_set_negative_review_clips.jsonl"
    if negative_path.exists():
        for row in read_jsonl(negative_path):
            challenge.append({
                "challenge_id": row.get("clip_id"),
                "source_id": row.get("source_id"),
                "clip_path": row.get("clip_path"),
                "metadata_bucket": row.get("candidate_bucket"),
                "matched_buckets": row.get("matched_buckets"),
                "label": None,
                "gold_status": "UNLABELED_CHALLENGE_ONLY",
            })
    chat_tts_path = ROOT / "speaker_refs" / "open_set_chat_tts_challenge.jsonl"
    if chat_tts_path.exists():
        challenge.extend(read_jsonl(chat_tts_path))
    challenge_path = ROOT / "speaker_refs" / "open_set_challenge_set.jsonl"
    challenge_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in challenge), encoding="utf-8")
    report = {
        "schema_version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "AUTO_ANCHORS_READY_NOT_GOLD",
        "anchor_manifest": str(out.relative_to(ROOT)),
        "challenge_manifest": str(challenge_path.relative_to(ROOT)),
        "anchor_count": len(anchors),
        "anchor_label_counts": dict(label_counts),
        "anchor_source_counts": {label: len(sources) for label, sources in source_label_counts.items()},
        "cross_source_minimum_policy": "Calibration must use source-group holdout; no label is trusted from a single source.",
        "family_policy": "NEURO and EVIL are intentionally collapsed into NEURO_FAMILY.",
        "gold_policy": "Auto anchors are independent-evidence validation anchors, not human gold and never training promotion.",
        "challenge_clip_count": len(challenge),
        "challenge_bucket_counts": dict(Counter(str(row.get("metadata_bucket")) for row in challenge)),
        "source_count_total": len(source_counts),
    }
    write_json(ROOT / "reports" / "auto_validation_anchor_report.json", report)
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
