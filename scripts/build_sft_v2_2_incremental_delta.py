from __future__ import annotations

"""Incremental inventory for the v2.2 transcript/timeline refresh.

This is intentionally read-only with respect to corpus inputs.  It records the
current artifact state, compares it with the last v2.1 structural snapshot,
and emits a reproducible delta report before any structural rebuild is run.
"""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
V21 = ROOT / "datasets" / "meow_v02_sft_v2_1_semantic_verified"
OUT = ROOT / "reports" / "sft_v2_2_incremental_delta.json"
MANIFEST = ROOT / "manifest" / "master_video_manifest.jsonl"
REGISTRY = ROOT / "source_registry.json"
TIMELINES = ROOT / "unique_timelines"
ASR = ROOT / "asr"
IDENTITY = ROOT / "identity_results" / "identity_mapping_multimodal_fusion_proxy.jsonl"
CLUSTERS = ROOT / "reports" / "recording_content_clusters.jsonl"
TARGETS = {"NEURO_FAMILY_HIGH", "NEURO_FAMILY_MEDIUM"}


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except Exception:
                continue
            if isinstance(value, dict):
                value["_line_no"] = line_no
                yield value


def file_state(path: Path | None, baseline_cutoff: float) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"exists": False}
    stat = path.stat()
    return {
        "exists": True,
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "size": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "updated_since_v2_1": stat.st_mtime > baseline_cutoff,
    }


def source_from_asr_path(path: Path) -> str:
    name = path.name
    # Current canonical forms: source.backend.json and source.ladev-transcript.json.
    if name.endswith(".ladev-transcript.json"):
        return name[: -len(".ladev-transcript.json")]
    if name.endswith(".json"):
        return name[:-5].split(".", 1)[0]
    return path.stem


def asr_backend(path: Path, payload: dict[str, Any]) -> str:
    value = payload.get("backend") or payload.get("asr_backend") or payload.get("engine")
    if value:
        return str(value)
    if path.name.endswith(".ladev-transcript.json"):
        return "ladev-transcript"
    if ".openai-" in path.name:
        return "openai-whisper"
    return str(payload.get("model") or "unknown")


def transcript_metrics(path: Path) -> dict[str, Any]:
    payload = read_json(path, {})
    segments = payload.get("segments") if isinstance(payload, dict) else None
    if not isinstance(segments, list):
        # Some adapters wrap the list under transcript/utterances.
        segments = payload.get("utterances") if isinstance(payload, dict) else None
    if not isinstance(segments, list):
        segments = []
    no_speech: list[float] = []
    compression: list[float] = []
    word_prob: list[float] = []
    timestamp_anomalies = 0
    repeated = 0
    nonempty = 0
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        text = str(segment.get("text") or "").strip()
        if text:
            nonempty += 1
        for key, target in (("no_speech_prob", no_speech), ("compression_ratio", compression)):
            try:
                target.append(float(segment[key]))
            except (KeyError, TypeError, ValueError):
                pass
        words = segment.get("words") or []
        for word in words:
            try:
                word_prob.append(float(word.get("probability")))
            except (AttributeError, TypeError, ValueError):
                pass
        try:
            start = float(segment.get("start"))
            end = float(segment.get("end"))
            if end < start or start < 0:
                timestamp_anomalies += 1
        except (TypeError, ValueError):
            timestamp_anomalies += 1
        normalized = re.sub(r"\W+", " ", text.lower()).strip()
        if len(normalized.split()) >= 5 and len(set(normalized.split())) <= 1:
            repeated += 1

    def median(values: list[float]) -> float | None:
        if not values:
            return None
        values = sorted(values)
        mid = len(values) // 2
        return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2

    return {
        "segments": len(segments),
        "nonempty_segments": nonempty,
        "median_no_speech_prob": median(no_speech),
        "median_compression_ratio": median(compression),
        "median_word_probability": median(word_prob),
        "timestamp_anomalies": timestamp_anomalies,
        "repeated_segment_count": repeated,
    }


def canonical_id_map() -> dict[str, str]:
    result: dict[str, str] = {}
    for row in jsonl(CLUSTERS):
        cluster = str(row.get("recording_cluster_id") or "")
        for source_id in row.get("source_ids") or []:
            result[str(source_id)] = cluster
    return result


def load_old_snapshot() -> tuple[set[str], set[str], float]:
    old_sources: set[str] = set()
    old_targets: set[str] = set()
    mtimes = [V21.stat().st_mtime] if V21.exists() else []
    path = V21 / "structural_candidates_v2.jsonl"
    if path.exists():
        mtimes.append(path.stat().st_mtime)
        for row in jsonl(path):
            if row.get("source_id"):
                old_sources.add(str(row["source_id"]))
            for turn_id in row.get("target_turn_ids") or []:
                old_targets.add(str(turn_id))
    # v2.1's completed artifact directory is the last stable snapshot.  Files
    # written after that point are treated as changed, while unchanged sources
    # remain eligible for reuse.
    cutoff = max(mtimes) if mtimes else 0.0
    return old_sources, old_targets, cutoff


def manifest_records() -> dict[str, dict[str, Any]]:
    result = {}
    for row in jsonl(MANIFEST):
        source_id = row.get("source_id")
        if source_id:
            result[str(source_id)] = row
    return result


def build() -> dict[str, Any]:
    old_sources, old_targets, cutoff = load_old_snapshot()
    manifest = manifest_records()
    registry_payload = read_json(REGISTRY, {}) or {}
    registry = {str(row.get("source_id")): row for row in registry_payload.get("sources") or [] if row.get("source_id")}
    source_ids = set(manifest) | set(registry) | {p.stem for p in TIMELINES.glob("*.json")}
    fusion_rows = list(jsonl(IDENTITY))
    fusion_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in fusion_rows:
        if row.get("source_id"):
            fusion_by_source[str(row["source_id"])].append(row)
    cluster_map = canonical_id_map()
    asr_paths: dict[str, list[Path]] = defaultdict(list)
    backend_metrics: dict[str, dict[str, Any]] = defaultdict(lambda: {"files": 0, "sources": set(), "segments": 0, "nonempty_segments": 0, "timestamp_anomalies": 0, "repeated_segments": 0, "no_speech": [], "compression": [], "word_probability": []})
    for path in ASR.rglob("*.json"):
        sid = source_from_asr_path(path)
        asr_paths[sid].append(path)
        payload = read_json(path, {}) or {}
        backend = asr_backend(path, payload)
        metrics = transcript_metrics(path)
        bucket = backend_metrics[backend]
        bucket["files"] += 1
        bucket["sources"].add(sid)
        bucket["segments"] += metrics["segments"]
        bucket["nonempty_segments"] += metrics["nonempty_segments"]
        bucket["timestamp_anomalies"] += metrics["timestamp_anomalies"]
        bucket["repeated_segments"] += metrics["repeated_segment_count"]
        for source_key, metric_key in (("median_no_speech_prob", "no_speech"), ("median_compression_ratio", "compression"), ("median_word_probability", "word_probability")):
            if metrics[source_key] is not None:
                bucket[metric_key].append(metrics[source_key])

    per_source: list[dict[str, Any]] = []
    state_counts = Counter()
    current_target_count = 0
    current_resolved_target_turns: set[str] = set()
    old_target_sources: set[str] = set()
    for source_id in sorted(source_ids):
        entry = manifest.get(source_id, {})
        timeline_path = TIMELINES / f"{source_id}.json"
        timeline = read_json(timeline_path, {}) or {}
        turns = timeline.get("turns") if isinstance(timeline, dict) else []
        turns = turns if isinstance(turns, list) else []
        mapped = fusion_by_source.get(source_id, [])
        identities = Counter(str(row.get("identity") or "UNKNOWN") for row in mapped)
        target_speakers = {str(row.get("cluster")) for row in mapped if row.get("identity") in TARGETS}
        target_turns = [turn for turn in turns if str(turn.get("speaker")) in target_speakers]
        target_turn_ids = {str(turn.get("turn_id")) for turn in target_turns if turn.get("turn_id")}
        current_target_count += len(target_turns)
        current_resolved_target_turns.update(target_turn_ids)
        if source_id in old_sources:
            old_target_sources.add(source_id)
        artifact_paths: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for path in asr_paths.get(source_id, []):
            artifact_paths["asr"].append(file_state(path, cutoff))
        for key, manifest_key in (("transcript", "transcript_path"), ("diarization", "diarization_path"), ("audio", "audio_path")):
            raw = entry.get(manifest_key)
            path = ROOT / str(raw) if raw else None
            if path is not None:
                artifact_paths[key].append(file_state(path, cutoff))
        artifact_paths["timeline"].append(file_state(timeline_path, cutoff))
        changed = sorted({kind for kind, states in artifact_paths.items() for state in states if state.get("updated_since_v2_1")})
        has_audio = bool(entry.get("audio_download_status") == "done" or entry.get("audio_path"))
        has_transcript = bool(asr_paths.get(source_id) or entry.get("transcript_available") or entry.get("asr_status") == "done")
        has_diarization = bool(entry.get("diarization_status") == "done" or entry.get("diarization_path"))
        has_timeline = timeline_path.exists() and bool(turns)
        if source_id not in old_sources:
            # The v2.1 structural pool is not a complete source inventory.
            # Do not call every historically known-but-unused source new.
            # A source outside that pool is NEW_SOURCE only when a current
            # artifact was written after the v2.1 snapshot cutoff; otherwise
            # it is explicitly tracked as existing outside the prior SFT pool.
            state = "NEW_SOURCE" if changed else "EXISTING_OUTSIDE_V2_1_STRUCTURAL"
        elif "timeline" in changed and "diarization" in changed:
            state = "UPDATED_DIARIZATION"
        elif "timeline" in changed or "asr" in changed or "transcript" in changed:
            state = "UPDATED_TRANSCRIPT"
        elif "diarization" in changed:
            state = "NEW_DIARIZATION"
        elif mapped and source_id in old_sources:
            state = "UNCHANGED"
        else:
            state = "NEW_IDENTITY_RESOLUTION" if mapped else "UNCHANGED"
        state_counts[state] += 1
        per_source.append({
            "source_id": source_id,
            "previous_state": "PRESENT_IN_V2_1_STRUCTURAL" if source_id in old_sources else "NOT_IN_V2_1_STRUCTURAL",
            "current_state": state,
            "changed_artifacts": changed,
            "asr_backend_model": sorted({
                (
                    str((read_json(p, {}) or {}).get("backend") or asr_backend(p, read_json(p, {}) or {}) or "unknown"),
                    str((read_json(p, {}) or {}).get("model") or ""),
                )
                for p in asr_paths.get(source_id, [])
            }),
            "transcript_path": entry.get("transcript_path") or entry.get("asr_path"),
            "diarization_status": entry.get("diarization_status") or ("done" if has_diarization else "missing"),
            "identity_status": "TARGET_RESOLVED" if target_speakers else ("MAPPED_NON_TARGET" if mapped else "UNMAPPED"),
            "timeline_status": "USABLE" if has_timeline else "MISSING_OR_EMPTY",
            "canonical_recording_id": cluster_map.get(source_id),
            "has_audio": has_audio,
            "has_transcript": has_transcript,
            "has_diarization": has_diarization,
            "turn_count": len(turns),
            "resolved_target_speaker_count": len(target_speakers),
            "resolved_target_turn_count": len(target_turns),
            "artifact_states": dict(artifact_paths),
        })

    backend_report = {}
    for backend, bucket in sorted(backend_metrics.items()):
        def med(values):
            values = sorted(values)
            if not values: return None
            m = len(values) // 2
            return values[m] if len(values) % 2 else (values[m - 1] + values[m]) / 2
        backend_report[backend] = {
            "files": bucket["files"],
            "sources": len(bucket["sources"]),
            "segments": bucket["segments"],
            "nonempty_segments": bucket["nonempty_segments"],
            "timestamp_anomalies": bucket["timestamp_anomalies"],
            "repeated_segments": bucket["repeated_segments"],
            "median_no_speech_prob": med(bucket["no_speech"]),
            "median_compression_ratio": med(bucket["compression"]),
            "median_word_probability": med(bucket["word_probability"]),
        }

    eligible = [row for row in per_source if row["has_audio"] and row["has_transcript"] and row["has_diarization"] and row["timeline_status"] == "USABLE" and row["resolved_target_speaker_count"] > 0]
    old_structural_rows = sum(1 for _ in jsonl(V21 / "structural_candidates_v2.jsonl"))
    old_structural_sources = len(old_sources)
    report = {
        "schema_version": "1.0.0",
        "pipeline_version": "sft-v2.2-incremental-delta-2026-09-13",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline": {
            "source_artifact": "datasets/meow_v02_sft_v2_1_semantic_verified/structural_candidates_v2.jsonl",
            "structural_rows": old_structural_rows,
            "unique_sources": old_structural_sources,
            "unique_target_turns": len(old_targets),
            "snapshot_cutoff_utc": datetime.fromtimestamp(cutoff, timezone.utc).isoformat() if cutoff else None,
        },
        "current": {
            "manifest_records": len(manifest),
            "registry_sources": len(registry),
            "timeline_files": len(list(TIMELINES.glob("*.json"))),
            "fusion_rows": len(fusion_rows),
            "resolved_target_turns": current_target_count,
            "unique_resolved_target_turns": len(current_resolved_target_turns),
            "eligible_for_sft_reconstruction": len(eligible),
            "sources_with_audio": sum(1 for row in per_source if row["has_audio"]),
            "sources_with_transcript": sum(1 for row in per_source if row["has_transcript"]),
            "sources_with_diarization": sum(1 for row in per_source if row["has_diarization"]),
            "sources_with_usable_timeline": sum(1 for row in per_source if row["timeline_status"] == "USABLE"),
            "sources_with_resolved_target": sum(1 for row in per_source if row["resolved_target_speaker_count"] > 0),
        },
        "delta": {
            "state_counts": dict(state_counts),
            "new_sources": sum(1 for row in per_source if row["current_state"] == "NEW_SOURCE"),
            "existing_sources_outside_previous_structural_pool": sum(1 for row in per_source if row["current_state"] == "EXISTING_OUTSIDE_V2_1_STRUCTURAL"),
            "updated_or_new_artifact_sources": sum(1 for row in per_source if row["changed_artifacts"]),
            "new_target_turns_vs_v2_1_structural": len(current_resolved_target_turns - old_targets),
            "old_target_turns_still_present": len(current_resolved_target_turns & old_targets),
            "sources_with_new_or_updated_target_evidence": sum(1 for row in per_source if row["resolved_target_turn_count"] and row["current_state"] != "UNCHANGED"),
        },
        "asr_backend_quality": backend_report,
        "sources": per_source,
        "active_process_note": "Inventory may overlap an active ASR write; files observed in this snapshot are versioned by mtime and must be re-inventoried after the active source completes.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUT))
    args = parser.parse_args()
    OUT = Path(args.output)
    if not OUT.is_absolute():
        OUT = ROOT / OUT
    report = build()
    print(json.dumps({
        "output": str(OUT),
        "baseline_structural_rows": report["baseline"]["structural_rows"],
        "current_resolved_target_turns": report["current"]["resolved_target_turns"],
        "eligible_sources": report["current"]["eligible_for_sft_reconstruction"],
        "state_counts": report["delta"]["state_counts"],
        "backend_quality": report["asr_backend_quality"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
