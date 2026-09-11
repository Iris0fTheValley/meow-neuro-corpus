from __future__ import annotations

"""Calibrate a target-vs-nontarget verifier on automatic anchors.

The calibration is source-group holdout.  It reports proxy operating points and
keeps metadata-only challenge buckets separate from labeled anchors.
"""

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
    signal, rate = sf.read(path, dtype="float32", always_2d=False)
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    target = int(seconds * rate)
    signal = np.asarray(signal[:target], dtype="float32")
    if len(signal) < target:
        signal = np.pad(signal, (0, target - len(signal)))
    return signal


def embed(encoder, signals: list[np.ndarray], batch_size: int = 24) -> np.ndarray:
    vectors = []
    for offset in range(0, len(signals), batch_size):
        batch = np.asarray(signals[offset : offset + batch_size], dtype="float32")
        with torch.inference_mode():
            value = encoder.encode_batch(torch.from_numpy(batch)).squeeze(1).detach().cpu().numpy()
        value /= np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-9)
        vectors.append(value)
    return np.concatenate(vectors, axis=0) if vectors else np.empty((0, 192), dtype="float32")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-9))


def prototype(matrix: np.ndarray, rows: list[dict], labels: set[str], source_exclude: str | None = None) -> np.ndarray | None:
    indices = [i for i, row in enumerate(rows) if row.get("label") in labels and (source_exclude is None or str(row.get("source_id")) != source_exclude)]
    if not indices:
        return None
    value = matrix[indices].mean(axis=0)
    value /= max(float(np.linalg.norm(value)), 1e-9)
    return value


def score_rows(matrix: np.ndarray, rows: list[dict], indices: list[int], use_leave_source_out: bool) -> list[dict]:
    out = []
    for index in indices:
        row = rows[index]
        exclude = str(row.get("source_id")) if use_leave_source_out else None
        family = prototype(matrix, rows, {"NEURO_FAMILY"}, exclude)
        vedal = prototype(matrix, rows, {"VEDAL"}, exclude)
        other = prototype(matrix, rows, {"OTHER"}, exclude)
        if family is None or vedal is None:
            continue
        family_score = cosine(matrix[index], family)
        vedal_score = cosine(matrix[index], vedal)
        other_score = cosine(matrix[index], other) if other is not None else None
        nontarget_score = max(vedal_score, other_score if other_score is not None else -1.0)
        out.append({
            "anchor_id": row.get("anchor_id"),
            "source_id": row.get("source_id"),
            "label": row.get("label"),
            "family_score": round(family_score, 6),
            "vedal_score": round(vedal_score, 6),
            "other_score": round(other_score, 6) if other_score is not None else None,
            "nontarget_score": round(nontarget_score, 6),
            "family_vs_nontarget_margin": round(family_score - nontarget_score, 6),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--score-challenge", action="store_true")
    ap.add_argument("--model-source", default="speechbrain/spkrec-ecapa-voxceleb")
    ap.add_argument("--model-dir", default="models/speechbrain_ecapa")
    ap.add_argument("--report-name", default="open_set_auto_anchor_calibration.json")
    args = ap.parse_args()
    rows = [row for row in read_jsonl(ROOT / "speaker_refs" / "open_set_calibration_candidates.jsonl") if (ROOT / str(row.get("clip_path") or "")).exists()]
    if not rows:
        raise SystemExit("calibration bank is empty")
    encoder = EncoderClassifier.from_hparams(
        source=args.model_source,
        savedir=str(ROOT / args.model_dir),
        run_opts={"device": args.device},
        local_strategy=LocalStrategy.COPY,
    )
    matrix = embed(encoder, [read_clip(ROOT / str(row["clip_path"])) for row in rows])
    indices = [i for i, row in enumerate(rows) if row.get("label") in {"NEURO_FAMILY", "VEDAL", "OTHER"}]
    per_clip = score_rows(matrix, rows, indices, True)
    thresholds = []
    for threshold in [round(x, 3) for x in np.arange(0.0, 0.501, 0.005)]:
        family = [row for row in per_clip if row["label"] == "NEURO_FAMILY"]
        vedal = [row for row in per_clip if row["label"] in {"VEDAL", "OTHER"}]
        family_accept = [row for row in family if row["family_vs_nontarget_margin"] >= threshold]
        vedal_accept = [row for row in vedal if row["family_vs_nontarget_margin"] >= threshold]
        thresholds.append({
            "margin_threshold": threshold,
            "family_count": len(family),
            "family_accept": len(family_accept),
            "family_recall_proxy": round(len(family_accept) / max(1, len(family)), 6),
            "vedal_count": len(vedal),
            "vedal_false_positive": len(vedal_accept),
            "vedal_false_positive_rate": round(len(vedal_accept) / max(1, len(vedal)), 6),
        })
    eligible = [row for row in thresholds if row["vedal_false_positive"] == 0 and row["family_recall_proxy"] >= 0.90]
    selected = max(eligible, key=lambda row: (row["family_recall_proxy"], row["margin_threshold"])) if eligible else min(thresholds, key=lambda row: (row["vedal_false_positive_rate"], -row["family_recall_proxy"]))
    challenge_summary = {}
    if args.score_challenge:
        challenge_rows = [row for row in read_jsonl(ROOT / "speaker_refs" / "open_set_challenge_set.jsonl") if (ROOT / str(row.get("clip_path") or "")).exists()]
        challenge_matrix = embed(encoder, [read_clip(ROOT / str(row["clip_path"])) for row in challenge_rows])
        challenge_scores = []
        family = prototype(matrix, rows, {"NEURO_FAMILY"})
        vedal = prototype(matrix, rows, {"VEDAL"})
        other = prototype(matrix, rows, {"OTHER"})
        for index, row in enumerate(challenge_rows):
            fs = cosine(challenge_matrix[index], family) if family is not None else None
            vs = cosine(challenge_matrix[index], vedal) if vedal is not None else None
            oscore = cosine(challenge_matrix[index], other) if other is not None else None
            nscore = max(vs if vs is not None else -1.0, oscore if oscore is not None else -1.0)
            margin = (fs - nscore) if fs is not None else None
            challenge_scores.append({**row, "family_score": round(fs, 6), "vedal_score": round(vs, 6), "other_score": round(oscore, 6) if oscore is not None else None, "nontarget_score": round(nscore, 6), "family_vs_nontarget_margin": round(margin, 6), "potential_family_accept": bool(margin is not None and margin >= selected["margin_threshold"])})
        model_tag = "xvector" if "xvect" in args.model_source else "ecapa"
        challenge_path = ROOT / "speaker_refs" / f"open_set_challenge_scores_{model_tag}_calibrated.jsonl"
        challenge_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in challenge_scores), encoding="utf-8")
        for bucket in sorted({str(row.get("metadata_bucket")) for row in challenge_scores}):
            bucket_rows = [row for row in challenge_scores if str(row.get("metadata_bucket")) == bucket]
            challenge_summary[bucket] = {
                "clip_count": len(bucket_rows),
                "potential_family_accept_count": sum(bool(row["potential_family_accept"]) for row in bucket_rows),
                "potential_family_accept_rate": round(sum(bool(row["potential_family_accept"]) for row in bucket_rows) / max(1, len(bucket_rows)), 6),
            }
    report = {
        "schema_version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "AUTO_ANCHOR_CALIBRATION_PROXY_NOT_GOLD",
        "model": args.model_source,
        "calibration_manifest": "speaker_refs/open_set_calibration_candidates.jsonl",
        "cross_source_leave_one_source_out": True,
        "anchor_count": len(rows),
        "anchor_label_counts": dict(Counter(row.get("label") for row in rows)),
        "evaluated_clip_count": len(per_clip),
        "threshold_sweep": thresholds,
        "selected_operating_point": selected,
        "challenge_summary": challenge_summary,
        "promotion_decision": "DO_NOT_PROMOTE",
        "promotion_reason": "Auto anchors are independent-evidence calibration data, not human gold; challenge buckets remain unlabeled and open-set precision is not yet proven.",
        "per_clip": per_clip,
    }
    write_json(ROOT / "reports" / args.report_name, report)
    print(json.dumps({"status": report["status"], "anchor_count": len(rows), "evaluated_clip_count": len(per_clip), "selected_operating_point": selected, "challenge_summary": challenge_summary}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
