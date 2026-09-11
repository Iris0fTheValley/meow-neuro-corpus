from __future__ import annotations

import json
import math
import subprocess
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import imageio_ffmpeg
import soundfile as sf
import torch
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from manifest_tools import ROOT, read_jsonl, write_json


MODEL = "iic/speech_eres2netv2_sv_zh-cn_16k-common"
DEFAULT_THRESHOLD = 0.2144
VERSION = "eres2netv2-cluster-remap-robust-median-2026-09-11-v3-audio-hard-negative-bank"


def norm(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype="float32")
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)


def window_plan(path: Path) -> list[tuple[float, float]]:
    duration = float(sf.info(path).duration)
    if duration <= 5.0:
        return [(0.0, duration)]
    width = min(8.0, duration)
    count = min(4, max(2, int(math.ceil(duration / 6.0))))
    starts = np.linspace(0.0, max(0.0, duration - width), count)
    return [(float(start), float(start + width)) for start in starts]


def extract_window(source: Path, start: float, end: float, output: Path) -> bool:
    duration = max(0.4, min(8.0, end - start))
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-ss", str(max(0.0, start)), "-i", str(source), "-t", str(duration), "-vn", "-sn", "-dn", "-ac", "1", "-ar", "16000", "-y", str(output)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        return result.returncode == 0 and output.exists() and output.stat().st_size > 1000
    except (OSError, subprocess.SubprocessError):
        return False


def select_cluster_turns(source_id: str, cluster: str, max_count: int = 8) -> list[tuple[float, float]]:
    timeline_path = ROOT / "unique_timelines" / f"{source_id}.json"
    if not timeline_path.exists():
        return []
    payload = json.loads(timeline_path.read_text(encoding="utf-8"))
    turns = []
    for turn in payload.get("turns") or []:
        if str(turn.get("speaker") or "UNKNOWN") != cluster:
            continue
        timestamp = turn.get("timestamp") or {}
        start = float(timestamp.get("start") or 0.0)
        end = float(timestamp.get("end") or start)
        if end - start >= 1.0:
            turns.append((start, end))
    if len(turns) <= max_count:
        return turns
    selected = sorted(turns, key=lambda pair: pair[1] - pair[0], reverse=True)[:4]
    stride = max(1, math.ceil(len(turns) / max(1, max_count - len(selected))))
    selected += turns[::stride]
    dedup = {f"{start:.3f}:{end:.3f}": (start, end) for start, end in selected}
    return list(dedup.values())[:max_count]


def robust_threshold(benchmark_path: Path) -> tuple[float, str]:
    if benchmark_path.exists():
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        clip_threshold = float(((benchmark.get("models") or {}).get("eres2netv2") or {}).get("selected_operating_point", {}).get("threshold") or DEFAULT_THRESHOLD)
    else:
        clip_threshold = DEFAULT_THRESHOLD
    audit_path = ROOT / "reports" / "identity_calibration_transfer_audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        selected = (((audit.get("segment_aggregation_calibration") or {}).get("median_margin") or {}).get("selected_zero_fp") or {}).get("threshold")
        if selected is not None:
            return float(selected), "source_disjoint_segment_median_zero_fp"
    return clip_threshold, "corrected_clip_level_operating_point"


def main() -> None:
    benchmark_path = ROOT / "reports" / "speaker_model_benchmark_identity_closure.json"
    threshold, threshold_source = robust_threshold(benchmark_path)
    mapping = [row for row in read_jsonl(ROOT / "identity_results" / "identity_mapping_family_proxy.jsonl") if (ROOT / str(row.get("proxy_clip_path") or "")).exists()]
    anchors = [row for row in read_jsonl(ROOT / "speaker_refs" / "auto_validation_anchors.jsonl") if row.get("label") == "NEURO_FAMILY" and (ROOT / str(row.get("clip_path") or "")).exists()]
    family_sources = {str(row.get("source_id")) for row in anchors}
    vedal = [row for row in read_jsonl(ROOT / "speaker_refs" / "vedal_provisional_reference_candidates.jsonl") if str(row.get("source_id")) not in family_sources and (ROOT / str(row.get("clip_path") or "")).exists()]
    hard_bank = ROOT / "speaker_refs" / "identity_closure" / "hard_negative_validation_bank.jsonl"
    if hard_bank.exists():
        guests = [row for row in read_jsonl(hard_bank) if row.get("evidence_tier") == "AUTO_TRUSTED" and row.get("bucket") == "named_guest" and (ROOT / str(row.get("clip_path") or "")).exists()]
    else:
        guests = [row for row in read_jsonl(ROOT / "speaker_refs" / "identity_closure" / "recurring_guest_negative_bank.jsonl") if (ROOT / str(row.get("clip_path") or "")).exists()]
    ref_rows = [{"kind": "family", "label": "NEURO_FAMILY", "source_id": str(row.get("source_id")), "clip_path": row["clip_path"]} for row in anchors]
    ref_rows += [{"kind": "vedal", "label": "VEDAL", "source_id": str(row.get("source_id")), "clip_path": row["clip_path"]} for row in vedal]
    ref_rows += [{"kind": "guest", "label": str(row.get("candidate_label") or row.get("label") or "OTHER_GUEST"), "source_id": str(row.get("source_id")), "clip_path": row["clip_path"]} for row in guests]
    query_rows = [{"kind": "query", "label": None, "source_id": str(row.get("source_id")), "cluster": row.get("cluster"), "mapping_id": row.get("mapping_id"), "clip_path": row["proxy_clip_path"]} for row in mapping]
    rows = ref_rows + query_rows
    verifier = pipeline(task=Tasks.speaker_verification, model=MODEL, device="cuda:0" if torch.cuda.is_available() else "cpu")
    # References stay unchanged. Queries are scored over temporary windows of
    # multiple existing diarized turns from the same cluster; no windows are
    # added to corpus manifests.
    query_windows = []
    manifest = {str(row.get("source_id")): row for row in read_jsonl(ROOT / "manifest" / "master_video_manifest.jsonl")}
    with tempfile.TemporaryDirectory(prefix="meow_eres_remap_", dir=str(ROOT / "tmp")) as temp_dir:
        temp_path = Path(temp_dir)
        for q_index, query in enumerate(query_rows):
            source = ROOT / str(manifest.get(query["source_id"], {}).get("audio_path") or "")
            made = []
            try:
                turns = select_cluster_turns(query["source_id"], str(query["cluster"])) if source.exists() else []
                if not turns:
                    source = ROOT / query["clip_path"]
                    turns = window_plan(source)
                for w_index, (start, end) in enumerate(turns):
                    out = temp_path / f"q_{q_index}_{w_index}.wav"
                    if extract_window(source, start, end, out):
                        made.append((out, start, end))
            except Exception:
                made = []
            if not made:
                source = ROOT / query["clip_path"]
                made = [(source, 0.0, float(sf.info(source).duration))]
            for path, start, end in made:
                query_windows.append({"query_index": q_index, "path": path, "start": start, "end": end})
        ref_paths = [str(ROOT / row["clip_path"]) for row in ref_rows]
        all_paths = ref_paths + [str(item["path"]) for item in query_windows]
        vectors = []
        for offset in range(0, len(all_paths), 64):
            vectors.append(norm(np.asarray(verifier(all_paths[offset:offset + 64], output_emb=True)["embs"], dtype="float32")))
        matrix = np.concatenate(vectors, axis=0)
    ref_end = len(ref_rows)
    ref_matrix = matrix[:ref_end]
    query_matrix = matrix[ref_end:]
    query_scores = defaultdict(list)
    for window_index, window in enumerate(query_windows):
        query = query_rows[window["query_index"]]
        source_id = query["source_id"]
        family_indices = [i for i, row in enumerate(ref_rows) if row["kind"] == "family" and row["source_id"] != source_id]
        vedal_indices = [i for i, row in enumerate(ref_rows) if row["kind"] == "vedal" and row["source_id"] != source_id]
        family_proto = norm(ref_matrix[family_indices].mean(axis=0, keepdims=True))[0]
        vedal_proto = norm(ref_matrix[vedal_indices].mean(axis=0, keepdims=True))[0]
        family_score = float(np.dot(query_matrix[window_index], family_proto))
        vedal_score = float(np.dot(query_matrix[window_index], vedal_proto))
        guest_scores = {}
        for label in sorted({row["label"] for row in ref_rows if row["kind"] == "guest"}):
            indices = [i for i, row in enumerate(ref_rows) if row["kind"] == "guest" and row["label"] == label and row["source_id"] != source_id]
            if indices:
                guest_proto = norm(ref_matrix[indices].mean(axis=0, keepdims=True))[0]
                guest_scores[label] = float(np.dot(query_matrix[window_index], guest_proto))
        guest_label, guest_score = max(guest_scores.items(), key=lambda pair: pair[1]) if guest_scores else (None, -1.0)
        query_scores[window["query_index"]].append({"family_score": family_score, "vedal_score": vedal_score, "best_guest_label": guest_label, "best_guest_score": guest_score, "family_margin": family_score - max(vedal_score, guest_score), "start": window["start"], "end": window["end"]})
    results = []; counts = Counter()
    for index, query in enumerate(query_rows):
        source_id = query["source_id"]
        scores = query_scores[index]
        family_score = float(np.median([row["family_score"] for row in scores]))
        vedal_score = float(np.median([row["vedal_score"] for row in scores]))
        guest_labels = sorted({row["best_guest_label"] for row in scores if row["best_guest_label"]})
        guest_medians = {label: float(np.median([row["best_guest_score"] for row in scores if row["best_guest_label"] == label])) for label in guest_labels}
        guest_label, guest_score = max(guest_medians.items(), key=lambda pair: pair[1]) if guest_medians else (None, -1.0)
        segment_margins = [float(row["family_margin"]) for row in scores]
        margin = float(np.median(segment_margins))
        vote_rate = sum(value >= threshold for value in segment_margins) / max(1, len(segment_margins))
        if margin >= threshold:
            identity = "NEURO_FAMILY"; confidence = "high" if margin >= threshold + 0.05 else "medium"
        elif guest_score > vedal_score and guest_score >= family_score:
            identity = "NON_TARGET_GUEST"; confidence = "medium"
        elif vedal_score >= family_score:
            identity = "NON_TARGET_KNOWN"; confidence = "medium"
        else:
            identity = "UNKNOWN"; confidence = "unknown"
        counts[identity] += 1
        results.append({"mapping_id": query["mapping_id"], "source_id": source_id, "cluster": query["cluster"], "identity": identity, "identity_confidence": confidence, "family_score": round(family_score, 6), "vedal_score": round(vedal_score, 6), "best_guest_label": guest_label, "best_guest_score": round(guest_score, 6), "family_margin": round(margin, 6), "segment_count": len(scores), "segment_margin_min": round(min(segment_margins), 6), "segment_margin_max": round(max(segment_margins), 6), "segment_vote_rate_at_threshold": round(vote_rate, 6), "aggregation": "median_segment_margin_over_temporary_proxy_windows", "model": MODEL, "threshold": threshold, "threshold_source": threshold_source, "status": "PROXY_ERES2NETV2_ROBUST_MEDIAN_MAPPING", "gold_validated": False, "training_candidate": False, "reference_source_disjoint": True, "reference_bank": "auto_validation_family_plus_source_disjoint_vedal_plus_auto_trusted_guest_hard_negatives"})
    output = ROOT / "identity_results" / "identity_mapping_eres2netv2_proxy.jsonl"
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results), encoding="utf-8")
    report = {"schema_version": "0.1.0", "created_at": datetime.now(timezone.utc).isoformat(), "status": "ERES2NETV2_CLUSTER_REMAP_ROBUST_MEDIAN_PROXY_COMPLETE", "version": VERSION, "model": MODEL, "threshold": threshold, "threshold_source": threshold_source, "query_cluster_count": len(results), "source_count": len({row["source_id"] for row in results}), "reference_counts": {"family": len(anchors), "vedal": len(vedal), "guest": len(guests)}, "identity_counts": dict(counts), "gold_validated": False, "training_candidate_count": 0, "promotion_decision": "DO_NOT_PROMOTE", "output": str(output.relative_to(ROOT)), "aggregation": "median over temporary 8-second windows from multiple existing diarized turns per cluster, with proxy-clip fallback; no corpus artifacts are retained", "policy": "Independent ERes2NetV2 remap uses source-disjoint auto-trusted hard negatives and the source-disjoint segment-median calibration; it does not promote any S/A row."}
    write_json(ROOT / "reports" / "identity_mapping_eres2netv2_proxy.json", report)
    print(json.dumps({"status": report["status"], "query_cluster_count": len(results), "identity_counts": dict(counts), "promotion_decision": report["promotion_decision"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
