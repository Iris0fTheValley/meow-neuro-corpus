from __future__ import annotations

"""Benchmark local SV baselines and 3D-Speaker models on closure anchors.

The negative bank is deliberately labelled as challenge/provisional.  This
script reports proxy operating points only and never changes identity mappings.
"""

import json
import math
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from manifest_tools import ROOT, read_jsonl, write_json


MODELS = {
    "ecapa_voxceleb": {"kind": "speechbrain", "source": "speechbrain/spkrec-ecapa-voxceleb", "savedir": "models/speechbrain_ecapa"},
    "xvector_voxceleb": {"kind": "speechbrain", "source": "speechbrain/spkrec-xvect-voxceleb", "savedir": "models/speechbrain_xvector"},
    "eres2netv2": {"kind": "modelscope", "source": "iic/speech_eres2netv2_sv_zh-cn_16k-common"},
    "campplus": {"kind": "modelscope", "source": "iic/speech_campplus_sv_zh-cn_16k-common"},
    "eres2net": {"kind": "modelscope", "source": "iic/speech_eres2net_sv_zh-cn_16k-common"},
}


def load_audio(path: Path, seconds: float = 8.0) -> np.ndarray:
    data, rate = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = np.asarray(data, dtype="float32")
    target = int(seconds * rate)
    data = data[:target]
    if len(data) < target:
        data = np.pad(data, (0, target - len(data)))
    return data


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-9))


def normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype="float32")
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)


def load_records() -> list[dict]:
    closure = ROOT / "speaker_refs" / "identity_closure"
    semantic = json.loads((closure / "semantic_identity_bank.json").read_text(encoding="utf-8"))
    family_source_ids = {str(row.get("source_id")) for row in semantic.get("records", []) if row.get("label") == "NEURO_FAMILY"}
    rows: list[dict] = []
    for row in semantic.get("records", []):
        clip = ROOT / str(next((a.get("clip_path") for a in read_jsonl(ROOT / "speaker_refs" / "auto_validation_anchors.jsonl") if a.get("anchor_id") == row.get("anchor_id")), ""))
        if clip.exists():
            rows.append({"record_id": row["anchor_id"], "source_id": str(row["source_id"]), "label": row["label"], "split": row["split"], "kind": "auto_anchor", "clip_path": str(clip.relative_to(ROOT)), "gold": False})
    vedal_path = ROOT / "speaker_refs" / "vedal_provisional_reference_candidates.jsonl"
    if vedal_path.exists():
        for index, row in enumerate(read_jsonl(vedal_path)):
            clip = ROOT / str(row.get("clip_path") or "")
            if clip.exists():
                source_id = str(row.get("source_id"))
                rows.append({"record_id": f"vedal_proxy:{row.get('candidate_id', index)}", "source_id": source_id, "label": "VEDAL", "split": "proxy_negative" if source_id not in family_source_ids else "proxy_negative_overlap", "kind": "vedal_proxy", "clip_path": str(clip.relative_to(ROOT)), "gold": False, "source_disjoint_to_family_bank": source_id not in family_source_ids})
    hard_bank_path = closure / "hard_negative_validation_bank.jsonl"
    if hard_bank_path.exists():
        for index, row in enumerate(read_jsonl(hard_bank_path)):
            clip = ROOT / str(row.get("clip_path") or "")
            if not clip.exists():
                continue
            tier = str(row.get("evidence_tier") or "")
            if tier == "AUTO_TRUSTED" and row.get("bucket") == "named_guest":
                rows.append({"record_id": row.get("record_id") or f"guest_auto:{index}", "source_id": str(row.get("source_id")), "label": str(row.get("label") or "OTHER"), "split": "proxy_negative", "kind": "guest_auto_trusted", "trust_tier": tier, "clip_path": str(clip.relative_to(ROOT)), "gold": False, "source_disjoint_to_family_bank": True})
            elif tier == "UNLABELED_STRESS_ONLY":
                bucket = str(row.get("bucket") or "unknown")
                rows.append({"record_id": row.get("record_id") or f"challenge:{index}", "source_id": str(row.get("source_id")), "label": f"CHALLENGE_{bucket.upper()}", "split": "challenge", "kind": "open_set_challenge", "metadata_bucket": bucket, "clip_path": str(clip.relative_to(ROOT)), "gold": False, "source_disjoint_to_family_bank": True})
    else:
        guest_path = closure / "recurring_guest_negative_bank.jsonl"
        if guest_path.exists():
            for row in read_jsonl(guest_path):
                clip = ROOT / str(row.get("clip_path") or "")
                if clip.exists():
                    rows.append({"record_id": row["candidate_id"], "source_id": str(row["source_id"]), "label": row["candidate_label"], "split": "proxy_negative", "kind": "guest_candidate", "clip_path": str(clip.relative_to(ROOT)), "gold": False})
        challenge_path = ROOT / "speaker_refs" / "open_set_challenge_set.jsonl"
        if challenge_path.exists():
            for index, row in enumerate(read_jsonl(challenge_path)):
                clip = ROOT / str(row.get("clip_path") or "")
                if clip.exists():
                    bucket = str(row.get("metadata_bucket") or "unknown").upper()
                    rows.append({"record_id": row.get("challenge_id") or f"challenge:{index}", "source_id": str(row.get("source_id")), "label": f"CHALLENGE_{bucket}", "split": "challenge", "kind": "open_set_challenge", "metadata_bucket": str(row.get("metadata_bucket") or "unknown"), "clip_path": str(clip.relative_to(ROOT)), "gold": False})
    return rows


def speechbrain_embeddings(rows: list[dict], source: str, savedir: str, device: str) -> np.ndarray:
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy
    encoder = EncoderClassifier.from_hparams(source=source, savedir=str(ROOT / savedir), run_opts={"device": device}, local_strategy=LocalStrategy.COPY)
    output = []
    for offset in range(0, len(rows), 16):
        batch = np.asarray([load_audio(ROOT / row["clip_path"]) for row in rows[offset:offset + 16]], dtype="float32")
        with torch.inference_mode():
            vectors = encoder.encode_batch(torch.from_numpy(batch)).squeeze(1).detach().cpu().numpy()
        output.append(normalize(vectors))
    return np.concatenate(output, axis=0) if output else np.empty((0, 192), dtype="float32")


def modelscope_embeddings(rows: list[dict], source: str, device: str) -> np.ndarray:
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    verifier = pipeline(task=Tasks.speaker_verification, model=source, device=device)
    output = []
    for offset in range(0, len(rows), 64):
        paths = [str(ROOT / row["clip_path"]) for row in rows[offset : offset + 64]]
        result = verifier(paths, output_emb=True)
        output.append(normalize(np.asarray(result["embs"], dtype="float32")))
    return np.concatenate(output, axis=0) if output else np.empty((0, 192), dtype="float32")


def prototype(matrix: np.ndarray, rows: list[dict], predicate, exclude_source: str | None = None) -> np.ndarray | None:
    indices = [i for i, row in enumerate(rows) if predicate(row) and (exclude_source is None or row["source_id"] != exclude_source)]
    if not indices:
        return None
    vector = matrix[indices].mean(axis=0)
    return vector / max(float(np.linalg.norm(vector)), 1e-9)


def score_rows(rows: list[dict], matrix: np.ndarray) -> list[dict]:
    scored = []
    for index, row in enumerate(rows):
        # Keep this scorer isomorphic with remap_clusters_eres2netv2_proxy.py:
        # all source-disjoint family anchors participate, and named guests are
        # scored per label with the strongest guest prototype as the veto.
        family = prototype(matrix, rows, lambda x: x["label"] == "NEURO_FAMILY", row["source_id"])
        vedal = prototype(matrix, rows, lambda x: x["label"] == "VEDAL", row["source_id"])
        guest_scores = {}
        for label in sorted({x["label"] for x in rows if x.get("kind") in {"guest_candidate", "guest_auto_trusted"}}):
            guest = prototype(matrix, rows, lambda x, label=label: x.get("kind") in {"guest_candidate", "guest_auto_trusted"} and x["label"] == label, row["source_id"])
            if guest is not None:
                guest_scores[label] = cosine(matrix[index], guest)
        family_score = cosine(matrix[index], family) if family is not None else None
        vedal_score = cosine(matrix[index], vedal) if vedal is not None else None
        guest_label, guest_score = max(guest_scores.items(), key=lambda item: item[1]) if guest_scores else (None, None)
        nontarget = max([value for value in (vedal_score, guest_score) if value is not None], default=-1.0)
        scored.append({**row, "family_score": round(family_score, 6) if family_score is not None else None, "vedal_score": round(vedal_score, 6) if vedal_score is not None else None, "guest_score": round(guest_score, 6) if guest_score is not None else None, "best_guest_label": guest_label, "family_margin": round(family_score - nontarget, 6) if family_score is not None else None, "reference_rule": "all_source_disjoint_family_plus_source_disjoint_vedal_plus_per_label_guest_max"})
    return scored


def operating_points(scored: list[dict]) -> dict:
    family_holdout = [r for r in scored if r["label"] == "NEURO_FAMILY" and r["split"] == "validation" and r["family_margin"] is not None]
    negatives = [r for r in scored if r["split"] == "proxy_negative" and r.get("source_disjoint_to_family_bank", True) and r["family_margin"] is not None]
    calibration_rows = family_holdout + negatives
    thresholds = sorted({round(float(r["family_margin"]), 4) for r in calibration_rows if r["family_margin"] is not None})
    points = []
    for threshold in thresholds:
        tp = sum(r["family_margin"] >= threshold for r in family_holdout)
        fp = sum(r["family_margin"] >= threshold for r in negatives)
        points.append({"threshold": threshold, "family_validation_count": len(family_holdout), "family_validation_accept": tp, "family_recall_proxy": round(tp / len(family_holdout), 6) if family_holdout else None, "nontarget_challenge_count": len(negatives), "nontarget_challenge_false_positive": fp, "nontarget_challenge_false_positive_rate": round(fp / len(negatives), 6) if negatives else None})
    zero_fp = [p for p in points if p["nontarget_challenge_false_positive"] == 0]
    selected = max(zero_fp, key=lambda p: (p["family_recall_proxy"] or -1, p["threshold"])) if zero_fp else min(points, key=lambda p: (p["nontarget_challenge_false_positive_rate"] or 1, -p["family_recall_proxy"] if p["family_recall_proxy"] is not None else 1)) if points else {}
    return {"selected_operating_point": selected, "family_holdout_count": len(family_holdout), "calibration_negative_count": len(negatives), "challenge_clip_count": sum(r["split"] == "challenge" for r in scored), "points": points, "reference_rule": "all source-disjoint family anchors; source-disjoint Vedal; per-label named guest max; no unlabeled challenge rows", "policy": "Auto-trusted/source-disjoint family anchors and provisional cross-source non-target anchors calibrate the verifier; unlabeled buckets are stress-only and never enter threshold fitting."}


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = load_records()
    if not rows:
        raise SystemExit("no closure benchmark clips")
    results = {}
    all_scores = []
    for name, config in MODELS.items():
        started = time.perf_counter()
        try:
            if config["kind"] == "speechbrain":
                matrix = speechbrain_embeddings(rows, config["source"], config["savedir"], device)
            else:
                matrix = modelscope_embeddings(rows, config["source"], "cuda:0" if torch.cuda.is_available() else "cpu")
            scored = score_rows(rows, matrix)
            report = operating_points(scored)
            selected_threshold = float((report.get("selected_operating_point") or {}).get("threshold") or 0.0)
            challenge_summary = {}
            for bucket in sorted({str(row.get("metadata_bucket") or "unknown") for row in scored if row.get("split") == "challenge"}):
                bucket_rows = [row for row in scored if row.get("split") == "challenge" and str(row.get("metadata_bucket") or "unknown") == bucket]
                accepted = sum(float(row.get("family_margin") or -1.0) >= selected_threshold for row in bucket_rows)
                challenge_summary[bucket] = {"clip_count": len(bucket_rows), "potential_family_accept_count": accepted, "potential_family_accept_rate": round(accepted / max(1, len(bucket_rows)), 6), "label_status": "UNLABELED_STRESS_ONLY"}
            report["challenge_summary"] = challenge_summary
            report.update({"status": "PROXY_BENCHMARK_COMPLETE", "model": config["source"], "device": device if config["kind"] == "speechbrain" else ("cuda:0" if torch.cuda.is_available() else "cpu"), "clip_count": len(rows), "source_count": len({r["source_id"] for r in rows}), "runtime_seconds": round(time.perf_counter() - started, 3), "label_counts": dict(Counter(r["label"] for r in rows))})
            results[name] = report
            all_scores.extend([{**row, "model": name} for row in scored])
        except Exception as exc:
            results[name] = {"status": "BENCHMARK_FAILED", "model": config["source"], "error": repr(exc), "runtime_seconds": round(time.perf_counter() - started, 3), "promotion_decision": "DO_NOT_PROMOTE"}
    closure = ROOT / "speaker_refs" / "identity_closure"
    (closure / "speaker_model_scores.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in all_scores), encoding="utf-8")
    report = {"schema_version": "0.1.0", "created_at": datetime.now(timezone.utc).isoformat(), "status": "IDENTITY_MODEL_PROXY_BENCHMARK_WITH_STRESS_AUDIT", "benchmark_version": "identity-closure-models-2026-09-11-v2", "clip_count": len(rows), "source_count": len({r["source_id"] for r in rows}), "label_counts": dict(Counter(r["label"] for r in rows)), "gold_label_count": 0, "auto_trusted_anchor_policy": "Source-disjoint multi-evidence AUTO_TRUSTED anchors are valid for calibration/validation; HUMAN_VERIFIED is a higher evidence tier, not a prerequisite.", "models": results, "validation_split": "family auto anchors plus source-disjoint provisional cross-source non-target anchors; open-set buckets are unlabeled stress-only", "promotion_decision": "DO_NOT_PROMOTE", "promotion_reason": "No final precision claim is made from 0/49; challenge buckets are explicitly separated from threshold fitting and require independent evidence before becoming labeled negatives."}
    write_json(ROOT / "reports" / "speaker_model_benchmark_identity_closure.json", report)
    print(json.dumps({"status": report["status"], "clip_count": len(rows), "models": {key: {k: value.get(k) for k in ("status", "selected_operating_point", "runtime_seconds")} for key, value in results.items()}, "promotion_decision": report["promotion_decision"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
