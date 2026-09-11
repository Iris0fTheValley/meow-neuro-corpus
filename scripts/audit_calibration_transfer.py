from __future__ import annotations

"""Audit transfer from anchor calibration to the ERes2NetV2 cluster remap.

This is intentionally diagnostic only.  It does not change mappings,
thresholds, or candidate grades.  It reuses the exact ModelScope embedding
path used by the benchmark and compares the reference-bank/prototype rules
used by benchmark versus formal remap.  A small, source-disjoint sample of
existing diarized turns is scored to expose cluster-level aggregation loss.
"""

import json
import math
import subprocess
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import soundfile as sf
import torch

from benchmark_identity_models import load_records, modelscope_embeddings
from manifest_tools import ROOT, read_jsonl, write_json


MODEL = "iic/speech_eres2netv2_sv_zh-cn_16k-common"
KNOWN_FAMILY_SOURCES = {"8335sBz9aHU", "SJxG6ASjZeY", "GjIopQlnEUY", "FpMGqhh_yd8"}
MAX_SEGMENTS_PER_CLUSTER = 32


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-9))


def proto(matrix: np.ndarray, rows: list[dict], predicate, exclude_source: str | None = None) -> np.ndarray | None:
    indices = [i for i, row in enumerate(rows) if predicate(row) and (exclude_source is None or row["source_id"] != exclude_source)]
    if not indices:
        return None
    vector = matrix[indices].mean(axis=0)
    return vector / max(float(np.linalg.norm(vector)), 1e-9)


def score_with_rules(matrix: np.ndarray, rows: list[dict], query_index: int, family_mode: str, guest_mode: str) -> dict:
    query = rows[query_index]
    source_id = query["source_id"]
    family_pred = lambda row: row["label"] == "NEURO_FAMILY" and (family_mode == "all" or row["split"] == "bank_train")
    family = proto(matrix, rows, family_pred, source_id)
    vedal = proto(matrix, rows, lambda row: row["label"] == "VEDAL", source_id)
    guest_pred = lambda row: row.get("kind") in {"guest_candidate", "guest_auto_trusted"}
    guest = proto(matrix, rows, guest_pred, source_id)
    family_score = cosine(matrix[query_index], family) if family is not None else None
    vedal_score = cosine(matrix[query_index], vedal) if vedal is not None else None
    guest_score = cosine(matrix[query_index], guest) if guest is not None else None
    if guest_mode == "none":
        guest_score = None
    nontarget = max([x for x in (vedal_score, guest_score) if x is not None], default=-1.0)
    return {
        "family_score": round(family_score, 6) if family_score is not None else None,
        "vedal_score": round(vedal_score, 6) if vedal_score is not None else None,
        "guest_score": round(guest_score, 6) if guest_score is not None else None,
        "family_margin": round(family_score - nontarget, 6) if family_score is not None else None,
        "family_reference_mode": family_mode,
        "guest_reference_mode": guest_mode,
    }


def score_formal_rules(query_vector: np.ndarray, query_source: str, ref_matrix: np.ndarray, ref_rows: list[dict], guest_mode: str = "per_label") -> dict:
    family_indices = [i for i, row in enumerate(ref_rows) if row["kind"] == "family" and row["source_id"] != query_source]
    vedal_indices = [i for i, row in enumerate(ref_rows) if row["kind"] == "vedal" and row["source_id"] != query_source]
    family = ref_matrix[family_indices].mean(axis=0)
    family /= max(float(np.linalg.norm(family)), 1e-9)
    vedal = ref_matrix[vedal_indices].mean(axis=0)
    vedal /= max(float(np.linalg.norm(vedal)), 1e-9)
    family_score = cosine(query_vector, family)
    vedal_score = cosine(query_vector, vedal)
    guest_scores: dict[str, float] = {}
    labels = sorted({row["label"] for row in ref_rows if row["kind"] == "guest"})
    if guest_mode == "pooled":
        guest_indices = [i for i, row in enumerate(ref_rows) if row["kind"] == "guest" and row["source_id"] != query_source]
        if guest_indices:
            guest = ref_matrix[guest_indices].mean(axis=0)
            guest /= max(float(np.linalg.norm(guest)), 1e-9)
            guest_scores["__pooled__"] = cosine(query_vector, guest)
    else:
        for label in labels:
            indices = [i for i, row in enumerate(ref_rows) if row["kind"] == "guest" and row["label"] == label and row["source_id"] != query_source]
            if indices:
                guest = ref_matrix[indices].mean(axis=0)
                guest /= max(float(np.linalg.norm(guest)), 1e-9)
                guest_scores[label] = cosine(query_vector, guest)
    guest_label, guest_score = max(guest_scores.items(), key=lambda item: item[1]) if guest_scores else (None, -1.0)
    margin = family_score - max(vedal_score, guest_score)
    return {
        "family_score": round(family_score, 6),
        "vedal_score": round(vedal_score, 6),
        "best_guest_label": guest_label,
        "best_guest_score": round(guest_score, 6),
        "family_margin": round(margin, 6),
        "guest_reference_mode": guest_mode,
    }


def audio_info(path: Path) -> dict:
    try:
        info = sf.info(path)
        return {"samplerate": int(info.samplerate), "channels": int(info.channels), "duration_seconds": round(float(info.duration), 3), "frames": int(info.frames), "subtype": str(info.subtype)}
    except Exception as exc:
        return {"error": repr(exc)}


def extract_segment(audio_path: Path, start: float, end: float, output_path: Path) -> bool:
    duration = max(0.4, min(20.0, float(end) - float(start)))
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-ss", str(max(0.0, start)), "-i", str(audio_path), "-t", str(duration), "-vn", "-sn", "-dn", "-ac", "1", "-ar", "16000", "-y", str(output_path)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        return result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000
    except (OSError, subprocess.SubprocessError):
        return False


def select_turns(source_id: str, cluster: str) -> list[dict]:
    path = ROOT / "unique_timelines" / f"{source_id}.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    turns = []
    for turn in payload.get("turns") or []:
        if str(turn.get("speaker") or "UNKNOWN") != cluster:
            continue
        timestamp = turn.get("timestamp") or {}
        start = float(timestamp.get("start") or 0.0)
        end = float(timestamp.get("end") or start)
        if end > start:
            turns.append({"start": start, "end": end, "duration": end - start, "text": str(turn.get("text") or "")[:120]})
    if len(turns) <= MAX_SEGMENTS_PER_CLUSTER:
        return turns
    # Keep long turns plus evenly spaced turns so the audit sees both stable
    # and ordinary cluster material without expanding the corpus.
    longest = sorted(turns, key=lambda x: x["duration"], reverse=True)[:8]
    stride = max(1, math.ceil(len(turns) / (MAX_SEGMENTS_PER_CLUSTER - len(longest))))
    sampled = turns[::stride]
    combined = {f"{x['start']:.3f}:{x['end']:.3f}": x for x in longest + sampled}
    return list(combined.values())[:MAX_SEGMENTS_PER_CLUSTER]


def window_plan(path: Path) -> list[tuple[float, float]]:
    """Make a small deterministic window set without changing source files."""
    duration = float(sf.info(path).duration)
    if duration <= 5.0:
        return [(0.0, duration)]
    width = min(8.0, duration)
    count = min(4, max(2, int(math.ceil(duration / 6.0))))
    starts = np.linspace(0.0, max(0.0, duration - width), count)
    return [(round(float(start), 3), round(float(start + width), 3)) for start in starts]


def choose_zero_fp_threshold(rows: list[dict], score_key: str, family_key: str = "family") -> dict:
    positives = [row for row in rows if row["label"] == "NEURO_FAMILY"]
    negatives = [row for row in rows if row["label"] != "NEURO_FAMILY"]
    thresholds = sorted({float(row[score_key]) for row in rows})
    points = []
    for threshold in thresholds:
        tp = sum(float(row[score_key]) >= threshold for row in positives)
        fp = sum(float(row[score_key]) >= threshold for row in negatives)
        points.append({"threshold": round(threshold, 6), "family_accept": tp, "family_count": len(positives), "family_recall": round(tp / max(1, len(positives)), 6), "nontarget_fp": fp, "nontarget_count": len(negatives), "nontarget_fpr": round(fp / max(1, len(negatives)), 6)})
    zero = [row for row in points if row["nontarget_fp"] == 0]
    selected = max(zero, key=lambda row: (row["family_recall"], row["threshold"])) if zero else {}
    return {"selected_zero_fp": selected, "points": points, "family_count": len(positives), "nontarget_count": len(negatives)}


def main() -> None:
    benchmark_report = json.loads((ROOT / "reports" / "speaker_model_benchmark_identity_closure.json").read_text(encoding="utf-8"))
    threshold = float(benchmark_report["models"]["eres2netv2"]["selected_operating_point"]["threshold"])
    records = load_records()
    matrix = modelscope_embeddings(records, MODEL, "cuda:0" if torch.cuda.is_available() else "cpu")
    by_id = {row["record_id"]: i for i, row in enumerate(records)}
    saved_scores = [row for row in read_jsonl(ROOT / "speaker_refs" / "identity_closure" / "speaker_model_scores.jsonl") if row.get("model") == "eres2netv2"]
    saved_by_id = {row["record_id"]: row for row in saved_scores}
    validation = [row for row in records if row["label"] == "NEURO_FAMILY" and row["split"] == "validation"]

    anchor_comparison = []
    for row in validation:
        idx = by_id[row["record_id"]]
        old = saved_by_id.get(row["record_id"], {})
        benchmark_replay = score_with_rules(matrix, records, idx, "bank_train", "none")
        benchmark_guest_fixed = score_with_rules(matrix, records, idx, "bank_train", "pooled")
        formal_like_pooled = score_with_rules(matrix, records, idx, "all", "pooled")
        formal_like_per_label = score_with_rules(matrix, records, idx, "all", "per_label")
        anchor_comparison.append({"record_id": row["record_id"], "source_id": row["source_id"], "clip_path": row["clip_path"], "saved_benchmark": {k: old.get(k) for k in ("family_score", "vedal_score", "guest_score", "family_margin")}, "benchmark_replay": benchmark_replay, "benchmark_guest_fixed": benchmark_guest_fixed, "formal_like_pooled": formal_like_pooled, "formal_like_per_label": formal_like_per_label, "accept": {"saved_benchmark": float(old.get("family_margin") or -1.0) >= threshold, "formal_like_per_label": formal_like_per_label["family_margin"] >= threshold}})

    # Reproduce the formal remap reference bank exactly, using the same model
    # embeddings as the benchmark run.
    family = [row for row in records if row["label"] == "NEURO_FAMILY"]
    vedal = [row for row in records if row["label"] == "VEDAL" and row["source_id"] not in {x["source_id"] for x in family}]
    guests = [row for row in records if row.get("kind") == "guest_auto_trusted"]
    ref_rows = ([{"kind": "family", "label": "NEURO_FAMILY", "source_id": row["source_id"], "record_id": row["record_id"]} for row in family] + [{"kind": "vedal", "label": "VEDAL", "source_id": row["source_id"], "record_id": row["record_id"]} for row in vedal] + [{"kind": "guest", "label": row["label"], "source_id": row["source_id"], "record_id": row["record_id"]} for row in guests])
    ref_indices = [by_id[row["record_id"]] for row in ref_rows]
    ref_matrix = matrix[ref_indices]
    formal_anchor_exact = []
    for row in validation:
        idx = by_id[row["record_id"]]
        formal = score_formal_rules(matrix[idx], row["source_id"], ref_matrix, ref_rows, "per_label")
        formal_anchor_exact.append({"record_id": row["record_id"], "source_id": row["source_id"], **formal, "accepted_at_current_threshold": formal["family_margin"] >= threshold})

    current_remap = list(read_jsonl(ROOT / "identity_results" / "identity_mapping_eres2netv2_proxy.jsonl"))
    current_by_source = defaultdict(list)
    for row in current_remap:
        if row.get("source_id") in KNOWN_FAMILY_SOURCES:
            current_by_source[row["source_id"]].append(row)

    # Existing source audio and diarized turns only; these temporary clips are
    # deleted when the audit exits and are not added to any corpus manifest.
    manifest = {str(row.get("source_id")): row for row in read_jsonl(ROOT / "manifest" / "master_video_manifest.jsonl")}
    segment_summary = {}
    segment_paths = []
    segment_meta = []
    with tempfile.TemporaryDirectory(prefix="meow_transfer_audit_", dir=str(ROOT / "tmp")) as tmp:
        tmp_path = Path(tmp)
        for source_id in sorted(KNOWN_FAMILY_SOURCES):
            source_audio = ROOT / str(manifest.get(source_id, {}).get("audio_path") or "")
            for remap_row in current_by_source.get(source_id, []):
                cluster = str(remap_row.get("cluster"))
                turns = select_turns(source_id, cluster)
                for n, turn in enumerate(turns):
                    out = tmp_path / f"{source_id}_{cluster}_{n}.wav"
                    if source_audio.exists() and extract_segment(source_audio, turn["start"], turn["end"], out):
                        segment_paths.append(out)
                        segment_meta.append({"source_id": source_id, "cluster": cluster, "start": turn["start"], "end": turn["end"], "duration": turn["duration"], "text": turn["text"], "path": str(out)})
        if segment_paths:
            segment_rows = [{"record_id": f"segment:{i}", "source_id": meta["source_id"], "label": None, "split": "audit", "kind": "segment", "clip_path": str(path.relative_to(ROOT))} for i, (path, meta) in enumerate(zip(segment_paths, segment_meta))]
            segment_matrix = modelscope_embeddings(segment_rows, MODEL, "cuda:0" if torch.cuda.is_available() else "cpu")
            for meta, vector in zip(segment_meta, segment_matrix):
                scores = score_formal_rules(vector, meta["source_id"], ref_matrix, ref_rows, "per_label")
                meta.update(scores)
        for source_id in sorted(KNOWN_FAMILY_SOURCES):
            cluster_rows = [row for row in segment_meta if row["source_id"] == source_id]
            per_cluster = {}
            for cluster in sorted({row["cluster"] for row in cluster_rows}):
                rows = [row for row in cluster_rows if row["cluster"] == cluster]
                margins = [float(row["family_margin"]) for row in rows if row.get("family_margin") is not None]
                family_scores = [float(row["family_score"]) for row in rows if row.get("family_score") is not None]
                accepted = sum(x >= threshold for x in margins)
                per_cluster[cluster] = {"segment_count_scored": len(rows), "segment_family_score": {"min": round(min(family_scores), 6) if family_scores else None, "median": round(float(np.median(family_scores)), 6) if family_scores else None, "max": round(max(family_scores), 6) if family_scores else None}, "segment_family_margin": {"min": round(min(margins), 6) if margins else None, "median": round(float(np.median(margins)), 6) if margins else None, "max": round(max(margins), 6) if margins else None}, "segment_accept_count_at_current_threshold": accepted, "segment_accept_rate_at_current_threshold": round(accepted / max(1, len(margins)), 6), "cluster_aggregate_from_formal_remap": next((x for x in current_by_source.get(source_id, []) if x.get("cluster") == cluster), None)}
            segment_summary[source_id] = per_cluster

        # Calibrate a robust window aggregator on the same source-disjoint
        # positives and proxy negatives used by the corrected benchmark.
        calibration_rows = [row for row in records if row["label"] == "NEURO_FAMILY" and row["split"] == "validation"] + [row for row in records if row["split"] == "proxy_negative" and row.get("source_disjoint_to_family_bank", True)]
        calibration_window_paths = []
        calibration_window_meta = []
        for row in calibration_rows:
            source_path = ROOT / row["clip_path"]
            for n, (start, end) in enumerate(window_plan(source_path)):
                out = tmp_path / f"cal_{len(calibration_window_meta)}_{n}.wav"
                if extract_segment(source_path, start, end, out):
                    calibration_window_paths.append(out)
                    calibration_window_meta.append({"record_id": row["record_id"], "label": row["label"], "source_id": row["source_id"], "start": start, "end": end})
        if calibration_window_paths:
            calibration_rows_for_embedding = [{"record_id": f"cal_window:{i}", "source_id": meta["source_id"], "label": meta["label"], "split": "audit", "kind": "window", "clip_path": str(path.relative_to(ROOT))} for i, (path, meta) in enumerate(zip(calibration_window_paths, calibration_window_meta))]
            calibration_matrix = modelscope_embeddings(calibration_rows_for_embedding, MODEL, "cuda:0" if torch.cuda.is_available() else "cpu")
            grouped = defaultdict(list)
            for meta, vector in zip(calibration_window_meta, calibration_matrix):
                grouped[meta["record_id"]].append(score_formal_rules(vector, meta["source_id"], ref_matrix, ref_rows, "per_label")["family_margin"])
            aggregate_rows = []
            for row in calibration_rows:
                margins = [float(x) for x in grouped.get(row["record_id"], [])]
                if not margins:
                    continue
                aggregate_rows.append({"record_id": row["record_id"], "label": row["label"], "source_id": row["source_id"], "window_count": len(margins), "median_margin": round(float(np.median(margins)), 6), "trimmed_mean_margin": round(float(np.mean(sorted(margins)[1:-1] if len(margins) > 2 else margins)), 6), "vote_rate_at_current_threshold": round(sum(x >= threshold for x in margins) / len(margins), 6), "window_margins": [round(x, 6) for x in margins]})
            segment_aggregation_calibration = {"current_segment_margin_threshold": threshold, "row_count": len(aggregate_rows), "median_margin": choose_zero_fp_threshold(aggregate_rows, "median_margin"), "trimmed_mean_margin": choose_zero_fp_threshold(aggregate_rows, "trimmed_mean_margin"), "vote_rate_at_current_threshold": choose_zero_fp_threshold(aggregate_rows, "vote_rate_at_current_threshold"), "rows": aggregate_rows, "policy": "Calibration uses the corrected source-disjoint AUTO_TRUSTED family validation anchors plus source-disjoint proxy negatives; no challenge-only rows enter threshold fitting."}
        else:
            segment_aggregation_calibration = {"row_count": 0, "status": "NO_WINDOWS"}

    proxy_info = {}
    for row in current_remap:
        if row.get("source_id") in KNOWN_FAMILY_SOURCES:
            original = next((x for x in read_jsonl(ROOT / "identity_results" / "identity_mapping_family_proxy.jsonl") if x.get("source_id") == row.get("source_id") and x.get("cluster") == row.get("cluster")), None)
            if original:
                proxy_info[f"{row['source_id']}:{row['cluster']}"] = {"proxy": audio_info(ROOT / original["proxy_clip_path"]), "anchor_example_infos": [audio_info(ROOT / x["clip_path"]) for x in validation if x["source_id"] == row["source_id"]][:4]}

    report = {
        "schema_version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "CALIBRATION_TRANSFER_AUDIT_COMPLETE_NO_MUTATION",
        "model": MODEL,
        "current_threshold": threshold,
        "benchmark_report_threshold": threshold,
        "reference_counts": {"benchmark_family_total": len(family), "benchmark_family_bank_train": sum(row["label"] == "NEURO_FAMILY" and row["split"] == "bank_train" for row in records), "vedal": len(vedal), "guest_auto_trusted": len(guests)},
        "anchor_comparison": anchor_comparison,
        "formal_anchor_exact": formal_anchor_exact,
        "current_formal_remap_known_family_sources": {source: [{k: row.get(k) for k in ("cluster", "identity", "family_score", "vedal_score", "best_guest_label", "best_guest_score", "family_margin", "threshold")} for row in rows] for source, rows in current_by_source.items()},
        "segment_level_existing_source_audit": segment_summary,
        "segment_aggregation_calibration": segment_aggregation_calibration,
        "clip_preprocessing": proxy_info,
        "findings": [
            "Benchmark score rows have guest_score=null because benchmark_identity_models.py filters only kind=guest_candidate, while the current AUTO_TRUSTED guest bank uses kind=guest_auto_trusted.",
            "Benchmark family validation uses 26 bank_train anchors; formal remap uses all 42 family anchors with source exclusion.",
            "Formal remap uses per-guest-label max scores; benchmark (when guest predicate is fixed) uses one pooled guest prototype. These are different score transfer functions.",
            "Formal remap queries are 4-20 second 16 kHz mono proxy cluster clips; benchmark anchors are 15-22 second 16 kHz mono reference clips. Both ModelScope paths are otherwise the same embedding API, but the query material is not the same.",
            "This audit does not lower threshold, change mappings, or promote candidates.",
        ],
        "policy": "HUMAN_VERIFIED is not required for this diagnostic; source-disjoint AUTO_TRUSTED evidence is retained with provenance. No promotion follows from this report.",
    }
    write_json(ROOT / "reports" / "identity_calibration_transfer_audit.json", report)
    print(json.dumps({"status": report["status"], "threshold": threshold, "formal_anchor_accept": sum(x["accepted_at_current_threshold"] for x in formal_anchor_exact), "formal_anchor_count": len(formal_anchor_exact), "scored_segments": sum(x.get("segment_count_scored", 0) for value in segment_summary.values() for x in value.values()), "known_sources": sorted(segment_summary)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
