from __future__ import annotations

"""Benchmark provisional reference clips without promoting identity labels."""

import json
import os
import time
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

from manifest_tools import ROOT, write_json


IDENTITIES = ("NEURO", "EVIL", "VEDAL")

try:
    import psutil
except ImportError:
    psutil = None


def load_candidates(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-9)
    return float(np.dot(a, b) / denom)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-manifest", default="speaker_refs/provisional_reference_candidates.jsonl")
    ap.add_argument("--model-source", default="speechbrain/spkrec-ecapa-voxceleb")
    ap.add_argument("--model-dir", default="models/speechbrain_ecapa")
    ap.add_argument("--report-name", default="speaker_reference_benchmark.json")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = ap.parse_args()
    candidate_path = ROOT / args.candidate_manifest
    candidates = [x for x in load_candidates(candidate_path) if x.get("identity_hint") in IDENTITIES and (ROOT / str(x.get("clip_path"))).exists()]
    if not candidates:
        raise SystemExit("no provisional reference clips available")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    process = psutil.Process() if psutil is not None else None
    rss_before = process.memory_info().rss if process is not None else None
    started = time.perf_counter()
    encoder = EncoderClassifier.from_hparams(
        source=args.model_source,
        savedir=str(ROOT / args.model_dir),
        run_opts={"device": device},
        local_strategy=LocalStrategy.COPY,
    )
    matrix: list[np.ndarray] = []
    kept: list[dict] = []
    batch_size = 24
    for offset in range(0, len(candidates), batch_size):
        batch = candidates[offset : offset + batch_size]
        signals = np.asarray([read_clip(ROOT / str(x["clip_path"])) for x in batch], dtype="float32")
        with torch.inference_mode():
            embeddings = encoder.encode_batch(torch.from_numpy(signals)).squeeze(1).detach().cpu().numpy()
        embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-9)
        matrix.extend(embeddings)
        kept.extend(batch)
    matrix_np = np.asarray(matrix, dtype="float32")
    by_identity_sources: dict[str, set[str]] = defaultdict(set)
    for item in kept:
        by_identity_sources[str(item["identity_hint"])].add(str(item["source_id"]))

    per_clip = []
    confusion = Counter()
    margins: dict[str, list[float]] = defaultdict(list)
    scores: dict[str, list[float]] = defaultdict(list)
    for index, item in enumerate(kept):
        label = str(item["identity_hint"])
        # Leave the entire source out of the prototype to measure cross-source
        # consistency rather than memorization of a single recording.
        prototype_vectors: dict[str, np.ndarray] = {}
        for identity in IDENTITIES:
            usable = [
                j for j, other in enumerate(kept)
                if other.get("identity_hint") == identity and str(other.get("source_id")) != str(item.get("source_id"))
            ]
            if usable:
                prototype = matrix_np[usable].mean(axis=0)
                prototype /= max(float(np.linalg.norm(prototype)), 1e-9)
                prototype_vectors[identity] = prototype
        if label not in prototype_vectors or len(prototype_vectors) < 2:
            continue
        score_map = {identity: cosine(matrix_np[index], vector) for identity, vector in prototype_vectors.items()}
        ordered = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
        predicted, best_score = ordered[0]
        second_score = ordered[1][1]
        margin = best_score - second_score
        correct = predicted == label
        confusion[f"{label}->{predicted}"] += 1
        scores[label].append(best_score if correct else score_map[label])
        margins[label].append(margin)
        per_clip.append({
            "candidate_id": item["candidate_id"],
            "source_id": item["source_id"],
            "provisional_identity": label,
            "predicted_identity": predicted,
            "scores": {k: round(v, 6) for k, v in score_map.items()},
            "best_score": round(best_score, 6),
            "second_best_identity": ordered[1][0],
            "second_best_score": round(second_score, 6),
            "margin": round(margin, 6),
            "correct_against_provisional_label": correct,
            "reference_bank_version": "provisional-2026-09-11-v1",
            "validation_status": "proxy_only_not_trusted",
        })

    elapsed = time.perf_counter() - started
    metrics = {}
    for identity in IDENTITIES:
        rows = [x for x in per_clip if x["provisional_identity"] == identity]
        if not rows:
            metrics[identity] = {"clips_evaluated": 0, "independent_sources": len(by_identity_sources.get(identity, set()))}
            continue
        correct = sum(bool(x["correct_against_provisional_label"]) for x in rows)
        metrics[identity] = {
            "clips_evaluated": len(rows),
            "independent_sources": len(by_identity_sources.get(identity, set())),
            "proxy_accuracy": round(correct / len(rows), 6),
            "proxy_false_positives": sum(x["predicted_identity"] == identity and x["provisional_identity"] != identity for x in per_clip),
            "proxy_false_negatives": sum(x["predicted_identity"] != identity and x["provisional_identity"] == identity for x in rows),
            "mean_label_score": round(float(np.mean(scores[identity])), 6),
            "p05_label_score": round(float(np.quantile(scores[identity], 0.05)), 6),
            "mean_margin": round(float(np.mean(margins[identity])), 6),
            "p05_margin": round(float(np.quantile(margins[identity], 0.05)), 6),
        }

    peak_vram = None
    if device == "cuda":
        peak_vram = {
            "allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
    report = {
        "schema_version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PROXY_BENCHMARK_ONLY",
        "reference_bank_version": "provisional-2026-09-11-v1",
        "model": args.model_source,
        "candidate_manifest": args.candidate_manifest,
        "device": device,
        "clip_count": len(kept),
        "source_count": len({str(x["source_id"]) for x in kept}),
        "cross_source_leave_one_source_out": True,
        "metrics": metrics,
        "confusion": dict(confusion),
        "runtime_seconds": round(elapsed, 3),
        "runtime_seconds_per_clip": round(elapsed / max(1, len(kept)), 4),
        "ram_before_bytes": rss_before,
        "ram_after_bytes": process.memory_info().rss if process is not None else None,
        "peak_gpu_memory": peak_vram,
        "negative_other": {"status": "not_available", "reason": "No safely labeled non-target reference clips were present; do not interpret proxy accuracy as precision."},
        "promotion_decision": "DO_NOT_PROMOTE",
        "promotion_reason": "Labels are metadata-derived provisional hints and Vedal has no single-identity source; fixed negative gold set and independent validation remain incomplete.",
        "per_clip": per_clip,
    }
    report_path = ROOT / "reports" / args.report_name
    write_json(report_path, report)
    write_json(ROOT / "speaker_refs" / "reference_bank_benchmark.json", {
        "reference_bank_version": report["reference_bank_version"],
        "status": "PROXY_BENCHMARK_ONLY",
        "promotion_decision": report["promotion_decision"],
        "metrics": metrics,
        "confusion": dict(confusion),
        "runtime_seconds": report["runtime_seconds"],
        "peak_gpu_memory": peak_vram,
    })
    print(json.dumps({"status": report["status"], "device": device, "clip_count": len(kept), "source_count": report["source_count"], "metrics": metrics, "confusion": dict(confusion), "promotion_decision": report["promotion_decision"]}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
