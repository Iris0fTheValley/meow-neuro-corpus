from __future__ import annotations

"""Score review-only negative clips against the promoted family/Vedal proxy bank."""

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np
import soundfile as sf
import torch
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

from manifest_tools import ROOT, read_jsonl, write_json


def read_clip(path, seconds: float = 8.0) -> np.ndarray:
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


def embed(encoder, rows: list[dict], batch_size: int = 24) -> np.ndarray:
    vectors = []
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset : offset + batch_size]
        signals = np.asarray([read_clip(ROOT / str(row["clip_path"])) for row in batch], dtype="float32")
        with torch.inference_mode():
            matrix = encoder.encode_batch(torch.from_numpy(signals)).squeeze(1).detach().cpu().numpy()
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)
        vectors.append(matrix)
    return np.concatenate(vectors, axis=0) if vectors else np.empty((0, 192), dtype="float32")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = ap.parse_args()
    negative_rows = read_jsonl(ROOT / "speaker_refs" / "open_set_negative_review_clips.jsonl")
    family_rows = read_jsonl(ROOT / "speaker_refs" / "neuro_family_reference_candidates.jsonl")
    vedal_rows = read_jsonl(ROOT / "speaker_refs" / "vedal_provisional_reference_candidates.jsonl")
    negative_rows = [row for row in negative_rows if (ROOT / str(row.get("clip_path"))).exists()]
    family_rows = [row for row in family_rows if (ROOT / str(row.get("clip_path"))).exists()]
    vedal_rows = [row for row in vedal_rows if (ROOT / str(row.get("clip_path"))).exists()]
    if not negative_rows or not family_rows or not vedal_rows:
        raise SystemExit("negative, family, and Vedal clips are all required")
    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(ROOT / "models" / "speechbrain_ecapa"),
        run_opts={"device": args.device},
        local_strategy=LocalStrategy.COPY,
    )
    family_matrix = embed(encoder, family_rows)
    vedal_matrix = embed(encoder, vedal_rows)
    negative_matrix = embed(encoder, negative_rows)
    family_proto = family_matrix.mean(axis=0)
    family_proto /= max(float(np.linalg.norm(family_proto)), 1e-9)
    vedal_proto = vedal_matrix.mean(axis=0)
    vedal_proto /= max(float(np.linalg.norm(vedal_proto)), 1e-9)
    scored = []
    for row, vector in zip(negative_rows, negative_matrix):
        family_score = cosine(vector, family_proto)
        vedal_score = cosine(vector, vedal_proto)
        scored.append({
            "clip_id": row.get("clip_id"),
            "source_id": row.get("source_id"),
            "candidate_bucket": row.get("candidate_bucket"),
            "clip_path": row.get("clip_path"),
            "family_score": round(family_score, 6),
            "vedal_score": round(vedal_score, 6),
            "family_vs_vedal_margin": round(family_score - vedal_score, 6),
            "review_status": row.get("review_status"),
            "metadata_only_source_hint": True,
            "gold_label": None,
        })
    threshold = 0.02
    by_bucket = defaultdict(list)
    for row in scored:
        by_bucket[str(row.get("candidate_bucket") or "unknown")].append(row)
    bucket_summary = {}
    for bucket, rows in sorted(by_bucket.items()):
        family_risk = [row for row in rows if row["family_vs_vedal_margin"] >= threshold]
        bucket_summary[bucket] = {
            "clip_count": len(rows),
            "potential_family_accept_count": len(family_risk),
            "potential_family_accept_rate": round(len(family_risk) / max(1, len(rows)), 6),
            "margin_p05": round(float(np.quantile([row["family_vs_vedal_margin"] for row in rows], 0.05)), 6),
            "margin_median": round(float(np.median([row["family_vs_vedal_margin"] for row in rows])), 6),
        }
    scored.sort(key=lambda row: row["family_vs_vedal_margin"], reverse=True)
    report = {
        "schema_version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "OPEN_SET_REVIEW_SCORE_TRIAGE_NOT_GOLD",
        "model": "speechbrain/spkrec-ecapa-voxceleb",
        "device": args.device,
        "family_reference_clip_count": len(family_rows),
        "vedal_reference_clip_count": len(vedal_rows),
        "negative_review_clip_count": len(scored),
        "family_margin_threshold": threshold,
        "bucket_summary": bucket_summary,
        "top_potential_family_false_positive_review": scored[:100],
        "all_scores": scored,
        "policy": "Metadata bucket is not a gold label. Potential family accepts are audio-review priority only and must not be counted as false positives or used for promotion until gold labels exist.",
    }
    write_json(ROOT / "reports" / "open_set_negative_review_scores.json", report)
    print(json.dumps({"status": report["status"], "negative_review_clip_count": len(scored), "bucket_summary": bucket_summary, "top_potential_family_accept_count": sum(row["family_vs_vedal_margin"] >= threshold for row in scored[:100])}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
