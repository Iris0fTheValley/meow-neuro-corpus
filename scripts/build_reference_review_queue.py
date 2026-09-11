from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone

from manifest_tools import ROOT, read_jsonl, write_json


def load_per_clip(path):
    doc = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["candidate_id"]): row for row in doc.get("per_clip", [])}


def main() -> None:
    candidates = list(read_jsonl(ROOT / "speaker_refs" / "all_provisional_reference_candidates.jsonl"))
    ecapa = load_per_clip(ROOT / "reports" / "speaker_reference_benchmark_all_ecapa_cpu.json")
    xvector = load_per_clip(ROOT / "reports" / "speaker_reference_benchmark_all_xvector_cpu.json")
    selected = {str(row["candidate_id"]): row for row in read_jsonl(ROOT / "speaker_refs" / "reference_candidates_all_dual_model_agreement.jsonl")}
    queue = []
    for row in candidates:
        candidate_id = str(row["candidate_id"])
        e = ecapa.get(candidate_id, {})
        x = xvector.get(candidate_id, {})
        e_pred = e.get("predicted_identity")
        x_pred = x.get("predicted_identity")
        e_margin = e.get("margin")
        x_margin = x.get("margin")
        if candidate_id in selected:
            bucket = "dual_model_agreement"
        elif e_pred != row.get("identity_hint") or x_pred != row.get("identity_hint"):
            bucket = "model_disagreement_or_mismatch"
        else:
            bucket = "low_margin_or_unselected"
        queue.append({
            "review_id": f"human_review:{candidate_id}",
            "candidate_id": candidate_id,
            "review_status": "PENDING_HUMAN_REVIEW",
            "gold_label": None,
            "use_for_training": False,
            "review_bucket": bucket,
            "provisional_identity_hint": row.get("identity_hint"),
            "source_id": row.get("source_id"),
            "source_platform": row.get("source_platform"),
            "source_url": row.get("source_url"),
            "title": row.get("title"),
            "audio_path": row.get("audio_path"),
            "clip_path": row.get("clip_path"),
            "start": row.get("start"),
            "end": row.get("end"),
            "duration": row.get("duration"),
            "cluster": row.get("cluster"),
            "text_preview": row.get("text_preview"),
            "evidence": {
                "dominant_cluster_share": row.get("dominant_cluster_share"),
                "mean_speaker_confidence": row.get("mean_speaker_confidence"),
                "ecapa_prediction": e_pred,
                "ecapa_margin": e_margin,
                "xvector_prediction": x_pred,
                "xvector_margin": x_margin,
                "dual_model_agreement": candidate_id in selected,
                "candidate_status": row.get("candidate_status"),
            },
            "review_instruction": "Listen to the clip and assign exactly one of NEURO_FAMILY (Neuro or Evil), VEDAL, OTHER, or UNKNOWN; optionally note NEURO versus EVIL as a subtype, but never reject a family label because the subtype is uncertain.",
        })
    report = {
        "schema_version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PENDING_HUMAN_GOLD_REVIEW",
        "gold_label_policy": "gold_label remains null until a human or externally validated annotation is recorded",
        "candidate_count": len(queue),
        "reviewed_count": 0,
        "training_candidate_count": 0,
        "bucket_counts": dict(Counter(row["review_bucket"] for row in queue)),
        "negative_other_status": "not_available; OTHER/UNKNOWN is an allowed human review outcome",
        "target_label_space": ["NEURO_FAMILY", "VEDAL", "OTHER", "UNKNOWN"],
        "models": ["speechbrain/spkrec-ecapa-voxceleb", "speechbrain/spkrec-xvect-voxceleb"],
    }
    output = ROOT / "speaker_refs" / "reference_human_review_queue.jsonl"
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in queue), encoding="utf-8")
    write_json(ROOT / "speaker_refs" / "reference_human_review_queue.json", report)
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
