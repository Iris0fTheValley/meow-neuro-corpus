from __future__ import annotations

import argparse
import io
import json
import subprocess
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import soundfile as sf
import torch
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

from manifest_tools import ROOT, read_jsonl, safe_name, write_json


def read_clip(path: Path, seconds: float = 8.0) -> np.ndarray:
    signal, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    target = int(seconds * sample_rate)
    signal = np.asarray(signal[:target], dtype="float32")
    if len(signal) < target:
        signal = np.pad(signal, (0, target - len(signal)))
    return signal


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-9))


def extract_clip(audio_path: Path, start: float, end: float, output_path: Path) -> bool:
    duration = min(20.0, max(2.0, float(end) - float(start)))
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
        "-ss", str(max(0.0, float(start))), "-i", str(audio_path), "-t", str(duration),
        "-vn", "-sn", "-dn", "-ac", "1", "-ar", "16000", "-y", str(output_path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=45)
        return result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000
    except (OSError, subprocess.SubprocessError):
        return False


def representative_interval(source_id: str, cluster: str) -> tuple[float, float] | None:
    path = ROOT / "unique_timelines" / f"{safe_name(source_id)}.json"
    if not path.exists():
        return None
    timeline = json.loads(path.read_text(encoding="utf-8"))
    candidates = []
    for turn in timeline.get("turns") or []:
        if str(turn.get("speaker") or "UNKNOWN") != cluster:
            continue
        timestamp = turn.get("timestamp") or {}
        start = float(timestamp.get("start") or 0)
        end = float(timestamp.get("end") or start)
        if end > start:
            candidates.append((end - start, start, end, float(turn.get("speaker_confidence") or 0)))
    if not candidates:
        return None
    _, start, end, _ = max(candidates, key=lambda x: (x[0], x[3]))
    return start, end


def embed(encoder: EncoderClassifier, signals: list[np.ndarray], batch_size: int = 24) -> np.ndarray:
    output = []
    for offset in range(0, len(signals), batch_size):
        batch = np.asarray(signals[offset : offset + batch_size], dtype="float32")
        with torch.inference_mode():
            vectors = encoder.encode_batch(torch.from_numpy(batch)).squeeze(1).detach().cpu().numpy()
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)
        output.append(vectors)
    return np.concatenate(output, axis=0) if output else np.empty((0, 192), dtype="float32")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--family-manifest", default=None, help="Optional clean family calibration manifest")
    ap.add_argument("--vedal-manifest", default=None, help="Optional clean Vedal calibration manifest")
    ap.add_argument("--other-manifest", default=None, help="Optional clean OTHER calibration manifest")
    ap.add_argument("--ensemble-xvector", action="store_true", help="Require an X-vector non-target veto for family mapping")
    ap.add_argument("--xvector-margin", type=float, default=0.0)
    ap.add_argument("--reference-bank-version", default=None)
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    family_bank = json.loads((ROOT / "speaker_refs" / "reference_bank_neuro_family.json").read_text(encoding="utf-8"))
    family_rows = list(read_jsonl(ROOT / (args.family_manifest or family_bank["candidate_manifest"])))
    vedal_rows = list(read_jsonl(ROOT / (args.vedal_manifest or "speaker_refs/vedal_provisional_reference_candidates.jsonl")))
    other_rows = list(read_jsonl(ROOT / args.other_manifest)) if args.other_manifest else []
    manifest = {str(row.get("source_id")): row for row in read_jsonl(ROOT / "manifest" / "master_video_manifest.jsonl")}
    pending = list(read_jsonl(ROOT / "identity_results" / "identity_mapping_pending.jsonl"))
    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(ROOT / "models" / "speechbrain_ecapa"),
        run_opts={"device": args.device},
        local_strategy=LocalStrategy.COPY,
    )
    family_signals = [read_clip(ROOT / str(row["clip_path"])) for row in family_rows]
    vedal_signals = [read_clip(ROOT / str(row["clip_path"])) for row in vedal_rows if (ROOT / str(row["clip_path"])).exists()]
    family_vectors = embed(encoder, family_signals)
    vedal_vectors = embed(encoder, vedal_signals)
    other_signals = [read_clip(ROOT / str(row["clip_path"])) for row in other_rows if (ROOT / str(row["clip_path"])).exists()]
    other_vectors = embed(encoder, other_signals)
    xvector_encoder = None
    x_family_vectors = x_vedal_vectors = x_other_vectors = None
    if args.ensemble_xvector:
        xvector_encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb",
            savedir=str(ROOT / "models" / "speechbrain_xvector"),
            run_opts={"device": args.device},
            local_strategy=LocalStrategy.COPY,
        )
        x_family_vectors = embed(xvector_encoder, family_signals)
        x_vedal_vectors = embed(xvector_encoder, vedal_signals)
        x_other_vectors = embed(xvector_encoder, other_signals)
    source_to_family_indices = defaultdict(list)
    for index, row in enumerate(family_rows):
        source_to_family_indices[str(row["source_id"])].append(index)
    source_to_vedal_indices = defaultdict(list)
    for index, row in enumerate(vedal_rows):
        if index < len(vedal_vectors):
            source_to_vedal_indices[str(row["source_id"])].append(index)
    source_to_other_indices = defaultdict(list)
    for index, row in enumerate(other_rows):
        if index < len(other_vectors):
            source_to_other_indices[str(row["source_id"])].append(index)

    extraction_dir = ROOT / "identity_results" / "proxy_cluster_clips"
    extraction_dir.mkdir(parents=True, exist_ok=True)
    cluster_meta = []
    signals = []
    for row in pending:
        source_id = str(row["source_id"])
        cluster = str(row["cluster"])
        manifest_row = manifest.get(source_id, {})
        audio_path = ROOT / str(manifest_row.get("audio_path") or "")
        interval = representative_interval(source_id, cluster)
        output_path = extraction_dir / f"{safe_name(source_id)}__{safe_name(cluster)}.wav"
        ok = bool(interval and audio_path.exists() and extract_clip(audio_path, interval[0], interval[1], output_path))
        if ok:
            try:
                signals.append(read_clip(output_path))
            except Exception:
                ok = False
        if not ok:
            cluster_meta.append({"pending": row, "clip_path": None, "extraction_status": "failed"})
            continue
        cluster_meta.append({"pending": row, "clip_path": str(output_path.relative_to(ROOT)), "extraction_status": "ok"})

    cluster_vectors = embed(encoder, signals)
    x_cluster_vectors = embed(xvector_encoder, signals) if xvector_encoder is not None else None
    results = []
    vector_index = 0
    for meta in cluster_meta:
        pending_row = meta["pending"]
        if meta["extraction_status"] != "ok":
            results.append({**pending_row, "status": "PROXY_MAPPING_EXTRACTION_FAILED", "identity": "UNKNOWN", "identity_confidence": "unknown", "proxy_clip_path": None})
            continue
        source_id = str(pending_row["source_id"])
        vector = cluster_vectors[vector_index]
        x_vector = x_cluster_vectors[vector_index] if x_cluster_vectors is not None else None
        vector_index += 1
        family_indices = [i for i in range(len(family_rows)) if i not in source_to_family_indices.get(source_id, [])]
        if len(family_indices) < 10:
            family_indices = list(range(len(family_rows)))
        vedal_indices = [i for i in range(len(vedal_vectors)) if i not in source_to_vedal_indices.get(source_id, [])]
        if len(vedal_indices) < 5:
            vedal_indices = list(range(len(vedal_vectors)))
        family_proto = family_vectors[family_indices].mean(axis=0)
        family_proto /= max(float(np.linalg.norm(family_proto)), 1e-9)
        vedal_proto = vedal_vectors[vedal_indices].mean(axis=0)
        vedal_proto /= max(float(np.linalg.norm(vedal_proto)), 1e-9)
        other_indices = [i for i in range(len(other_vectors)) if i not in source_to_other_indices.get(source_id, [])]
        other_proto = None
        if other_indices:
            other_proto = other_vectors[other_indices].mean(axis=0)
            other_proto /= max(float(np.linalg.norm(other_proto)), 1e-9)
        family_score = cosine(vector, family_proto)
        vedal_score = cosine(vector, vedal_proto)
        other_score = cosine(vector, other_proto) if other_proto is not None else -1.0
        nontarget_score = max(vedal_score, other_score)
        margin = family_score - nontarget_score
        x_margin = None
        x_family_score = x_vedal_score = x_other_score = None
        if x_vector is not None:
            x_family_indices = [i for i in range(len(x_family_vectors)) if i not in source_to_family_indices.get(source_id, [])]
            if len(x_family_indices) < 10:
                x_family_indices = list(range(len(x_family_vectors)))
            x_vedal_indices = [i for i in range(len(x_vedal_vectors)) if i not in source_to_vedal_indices.get(source_id, [])]
            if len(x_vedal_indices) < 5:
                x_vedal_indices = list(range(len(x_vedal_vectors)))
            x_family_proto = x_family_vectors[x_family_indices].mean(axis=0)
            x_family_proto /= max(float(np.linalg.norm(x_family_proto)), 1e-9)
            x_vedal_proto = x_vedal_vectors[x_vedal_indices].mean(axis=0)
            x_vedal_proto /= max(float(np.linalg.norm(x_vedal_proto)), 1e-9)
            x_other_indices = [i for i in range(len(x_other_vectors)) if i not in source_to_other_indices.get(source_id, [])]
            x_other_proto = None
            if x_other_indices:
                x_other_proto = x_other_vectors[x_other_indices].mean(axis=0)
                x_other_proto /= max(float(np.linalg.norm(x_other_proto)), 1e-9)
            x_family_score = cosine(x_vector, x_family_proto)
            x_vedal_score = cosine(x_vector, x_vedal_proto)
            x_other_score = cosine(x_vector, x_other_proto) if x_other_proto is not None else -1.0
            x_margin = x_family_score - max(x_vedal_score, x_other_score)
        family_accept = margin >= args.margin and (x_margin is None or x_margin >= args.xvector_margin)
        if family_accept:
            identity, confidence, status = "NEURO_FAMILY", "high" if margin >= 0.05 else "medium", "PROXY_MAPPED_OPEN_SET"
        elif margin <= -args.margin:
            identity, confidence, status = "VEDAL", "high" if margin <= -0.05 else "medium", "PROXY_MAPPED_OPEN_SET"
        else:
            identity, confidence, status = "UNKNOWN", "unknown", "PROXY_UNKNOWN_LOW_MARGIN"
        results.append({
            **pending_row,
            "status": status,
            "identity": identity,
            "identity_confidence": confidence,
            "best_score": round(max(family_score, vedal_score), 6),
            "second_best_identity": "VEDAL" if family_score >= vedal_score else "NEURO_FAMILY",
            "second_best_score": round(min(family_score, vedal_score), 6),
            "margin": round(margin, 6),
            "family_vs_nontarget_margin": round(margin, 6),
            "matched_prototype": identity if identity != "UNKNOWN" else None,
            "reference_bank_version": args.reference_bank_version or family_bank["reference_bank_version"],
            "evidence": {
                "model": "speechbrain/spkrec-ecapa-voxceleb" if x_margin is None else ["speechbrain/spkrec-ecapa-voxceleb", "speechbrain/spkrec-xvect-voxceleb"],
                "open_set_margin_threshold": args.margin,
                "family_score": round(family_score, 6),
                "vedal_score": round(vedal_score, 6),
                "other_score": round(other_score, 6),
                "nontarget_score": round(nontarget_score, 6),
                "xvector_family_score": round(x_family_score, 6) if x_family_score is not None else None,
                "xvector_vedal_score": round(x_vedal_score, 6) if x_vedal_score is not None else None,
                "xvector_other_score": round(x_other_score, 6) if x_other_score is not None else None,
                "xvector_family_vs_nontarget_margin": round(x_margin, 6) if x_margin is not None else None,
                "xvector_margin_threshold": args.xvector_margin if x_margin is not None else None,
                "ensemble_family_accept": family_accept,
                "leave_source_out_for_reference_prototype": True,
                "gold_validated": False,
            },
            "proxy_clip_path": meta["clip_path"],
        })
    output = ROOT / "identity_results" / "identity_mapping_family_proxy.jsonl"
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results), encoding="utf-8")
    counts = Counter(row["identity"] for row in results)
    status_counts = Counter(row["status"] for row in results)
    report = {
        "schema_version": "0.1.0",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PROXY_MAPPING_COMPLETE_GOLD_PENDING",
        "model": ["speechbrain/spkrec-ecapa-voxceleb", "speechbrain/spkrec-xvect-voxceleb"] if args.ensemble_xvector else "speechbrain/spkrec-ecapa-voxceleb",
        "reference_bank": args.family_manifest or "speaker_refs/reference_bank_neuro_family.json",
        "source_count": len({row["source_id"] for row in results}),
        "cluster_count": len(results),
        "mapping_margin_threshold": args.margin,
        "identity_counts": dict(counts),
        "status_counts": dict(status_counts),
        "gold_validated": False,
        "training_candidate_count": 0,
        "output": str(output.relative_to(ROOT)),
        "policy": "Family mapping is allowed for Neuro/Evil; low-margin and extraction-failed clusters remain UNKNOWN. No S/A promotion before conversation QA.",
    }
    write_json(ROOT / "reports" / "identity_mapping_family_proxy.json", report)
    print(json.dumps({"status": report["status"], "source_count": report["source_count"], "cluster_count": report["cluster_count"], "identity_counts": dict(counts), "status_counts": dict(status_counts), "training_candidate_count": 0}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
