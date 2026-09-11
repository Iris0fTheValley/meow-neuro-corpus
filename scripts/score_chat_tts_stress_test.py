from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import torch
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from manifest_tools import ROOT, read_jsonl, write_json


MODEL = "iic/speech_eres2netv2_sv_zh-cn_16k-common"
DEFAULT_THRESHOLD = 0.2144


def normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype="float32")
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)


def main() -> None:
    benchmark_path = ROOT / "reports" / "speaker_model_benchmark_identity_closure.json"
    threshold = DEFAULT_THRESHOLD
    if benchmark_path.exists():
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        threshold = float(((benchmark.get("models") or {}).get("eres2netv2") or {}).get("selected_operating_point", {}).get("threshold") or DEFAULT_THRESHOLD)
    family_sources = {str(row.get("source_id")) for row in read_jsonl(ROOT / "speaker_refs" / "auto_validation_anchors.jsonl") if row.get("label") == "NEURO_FAMILY"}
    challenge = [row for row in read_jsonl(ROOT / "speaker_refs" / "open_set_chat_tts_challenge.jsonl") if str(row.get("source_id")) not in family_sources and (ROOT / str(row.get("clip_path") or "")).exists()]
    anchors = list(read_jsonl(ROOT / "speaker_refs" / "auto_validation_anchors.jsonl"))
    family = [row for row in anchors if row.get("label") == "NEURO_FAMILY" and (ROOT / str(row.get("clip_path") or "")).exists()]
    vedal = [row for row in read_jsonl(ROOT / "speaker_refs" / "vedal_provisional_reference_candidates.jsonl") if str(row.get("source_id")) not in family_sources and (ROOT / str(row.get("clip_path") or "")).exists()]
    hard_bank = ROOT / "speaker_refs" / "identity_closure" / "hard_negative_validation_bank.jsonl"
    guests = [row for row in read_jsonl(hard_bank) if row.get("evidence_tier") == "AUTO_TRUSTED" and row.get("bucket") == "named_guest" and (ROOT / str(row.get("clip_path") or "")).exists()] if hard_bank.exists() else list(read_jsonl(ROOT / "speaker_refs" / "identity_closure" / "recurring_guest_negative_bank.jsonl"))
    all_rows = [{"kind": "family", **row} for row in family] + [{"kind": "vedal", **row} for row in vedal] + [{"kind": "guest", **row} for row in guests] + [{"kind": "chat_tts", **row} for row in challenge]
    verifier = pipeline(task=Tasks.speaker_verification, model=MODEL, device="cuda:0" if torch.cuda.is_available() else "cpu")
    embeddings = normalize(np.asarray(verifier([str(ROOT / row["clip_path"]) for row in all_rows], output_emb=True)["embs"], dtype="float32"))
    # Match the corrected benchmark/remap scorer: all source-disjoint family
    # anchors and a per-label named-guest veto.  This bucket remains stress-only
    # and must never enter calibration.
    family_indices = [i for i, row in enumerate(all_rows) if row["kind"] == "family"]
    vedal_indices = [i for i, row in enumerate(all_rows) if row["kind"] == "vedal"]
    family_proto = normalize(embeddings[family_indices].mean(axis=0, keepdims=True))[0]
    vedal_proto = normalize(embeddings[vedal_indices].mean(axis=0, keepdims=True))[0]
    guest_labels = sorted({str(row.get("candidate_label") or row.get("label") or "OTHER_GUEST") for row in guests})
    guest_protos = {}
    for label in guest_labels:
        indices = [i for i, row in enumerate(all_rows) if row["kind"] == "guest" and str(row.get("candidate_label") or row.get("label") or "OTHER_GUEST") == label]
        if indices:
            guest_protos[label] = normalize(embeddings[indices].mean(axis=0, keepdims=True))[0]
    scored = []
    for index, row in enumerate(all_rows):
        if row["kind"] != "chat_tts":
            continue
        family_score = float(np.dot(embeddings[index], family_proto)); vedal_score = float(np.dot(embeddings[index], vedal_proto)); guest_scores = {label: float(np.dot(embeddings[index], vector)) for label, vector in guest_protos.items()}; guest_label, guest_score = max(guest_scores.items(), key=lambda item: item[1]) if guest_scores else (None, -1.0); margin = family_score - max(vedal_score, guest_score)
        scored.append({"challenge_id": row.get("challenge_id"), "source_id": row.get("source_id"), "cluster": row.get("cluster"), "clip_path": row.get("clip_path"), "family_score": round(family_score, 6), "vedal_score": round(vedal_score, 6), "best_guest_label": guest_label, "guest_score": round(guest_score, 6), "family_margin": round(margin, 6), "potential_family_accept_at_proxy_threshold": bool(margin >= threshold), "gold_label": None, "gold_status": "UNLABELED_CHALLENGE_ONLY", "source_disjoint_to_family_bank": True, "reference_rule": "all_source_disjoint_family_plus_source_disjoint_vedal_plus_per_label_guest_max"})
    out_dir = ROOT / "speaker_refs" / "identity_closure"
    out_path = out_dir / "chat_tts_stress_scores.jsonl"
    out_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in scored), encoding="utf-8")
    margins = sorted(float(row["family_margin"]) for row in scored)
    report = {"schema_version": "0.1.0", "created_at": datetime.now(timezone.utc).isoformat(), "status": "CHAT_TTS_UNLABELED_STRESS_TEST_CALIBRATED", "model": MODEL, "proxy_threshold": threshold, "source_disjoint_to_family_bank": True, "clip_count": len(scored), "source_count": len({row["source_id"] for row in scored}), "potential_family_accept_count": sum(row["potential_family_accept_at_proxy_threshold"] for row in scored), "potential_family_accept_rate": round(sum(row["potential_family_accept_at_proxy_threshold"] for row in scored) / max(1, len(scored)), 6), "margin_quantiles": {"p05": round(float(np.quantile(margins, 0.05)), 6) if margins else None, "median": round(float(np.median(margins)), 6) if margins else None, "p95": round(float(np.quantile(margins, 0.95)), 6) if margins else None}, "gold_label_count": 0, "use_for_calibration": False, "policy": "Explicit TTS mention creates a stress-test interval only; calibrated score is a safety audit, not a chat-TTS label or final precision claim. Overlapping clusters are quarantined from family promotion."}
    write_json(out_dir / "chat_tts_stress_test_report.json", report)
    print(json.dumps({"status": report["status"], "clip_count": report["clip_count"], "potential_family_accept_count": report["potential_family_accept_count"], "potential_family_accept_rate": report["potential_family_accept_rate"], "use_for_calibration": report["use_for_calibration"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
