from __future__ import annotations

"""Build an independent CHAT_TTS/OTHER_TTS rejection audit.

The layer is deliberately reject-only: it cannot promote a cluster to
NEURO_FAMILY.  Challenge intervals remain unlabeled stress evidence; named
family/guest anchors are used only to set a zero-anchor-false-reject boundary.
"""

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from manifest_tools import ROOT, read_jsonl, write_json


MODEL = "iic/speech_eres2netv2_sv_zh-cn_16k-common"
VERSION = "chat-tts-rejection-2026-09-12-v2"
KEYWORDS = {"tts", "text to speech", "text-to-speech", "google translate", "donation", "donate", "chat", "voice", "read"}


def normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype="float32")
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)


def source_split(source_id: str) -> str:
    return "holdout" if int(hashlib.sha1(source_id.encode("utf-8")).hexdigest()[:8], 16) % 5 == 0 else "bank_train"


def timeline_index() -> dict[str, dict[str, list[dict]]]:
    result = {}
    for path in (ROOT / "unique_timelines").glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        by_cluster = defaultdict(list)
        for turn in payload.get("turns") or []:
            by_cluster[str(turn.get("speaker") or "UNKNOWN")].append(turn)
        result[path.stem] = by_cluster
    return result


def context_features(turns: list[dict], start: float, end: float) -> dict:
    before = [turn for turn in turns if float((turn.get("timestamp") or {}).get("end") or 0) <= start and start - float((turn.get("timestamp") or {}).get("end") or 0) <= 30]
    after = [turn for turn in turns if float((turn.get("timestamp") or {}).get("start") or 0) >= end and float((turn.get("timestamp") or {}).get("start") or 0) - end <= 30]
    text = " ".join(str(turn.get("text") or "") for turn in before + after).casefold()
    hits = sorted({keyword for keyword in KEYWORDS if keyword in text})
    return {"preceding_turn_count": len(before), "following_turn_count": len(after), "context_keyword_hits": hits, "context_role_score": round(min(1.0, len(hits) / 3.0), 6)}


def main() -> None:
    closure = ROOT / "speaker_refs" / "identity_closure"
    chat_rows = [row for row in read_jsonl(ROOT / "speaker_refs" / "open_set_chat_tts_challenge.jsonl") if (ROOT / str(row.get("clip_path") or "")).exists()]
    family_rows = [row for row in read_jsonl(ROOT / "speaker_refs" / "auto_validation_anchors.jsonl") if row.get("label") == "NEURO_FAMILY" and (ROOT / str(row.get("clip_path") or "")).exists()]
    guest_rows = [row for row in read_jsonl(closure / "hard_negative_validation_bank.jsonl") if row.get("evidence_tier") == "AUTO_TRUSTED" and row.get("bucket") == "named_guest" and (ROOT / str(row.get("clip_path") or "")).exists()]
    fusion = list(read_jsonl(ROOT / "identity_results" / "identity_mapping_multimodal_fusion_proxy.jsonl"))
    family_proxy = {f"{row.get('source_id')}:{row.get('cluster')}": row for row in read_jsonl(ROOT / "identity_results" / "identity_mapping_family_proxy.jsonl")}
    timelines = timeline_index()
    priors = {str(row.get("source_id")): row for row in read_jsonl(closure / "expected_participants.jsonl")}
    source_registry = json.loads((ROOT / "source_registry.json").read_text(encoding="utf-8"))
    source_meta = {str(row.get("source_id")): row for row in source_registry.get("sources", [])}

    candidate_rows = []
    for row in fusion:
        proxy = family_proxy.get(f"{row.get('source_id')}:{row.get('cluster')}", {})
        path = ROOT / str(proxy.get("proxy_clip_path") or "")
        if path.exists():
            candidate_rows.append({"kind": "cluster", "record_id": f"{row.get('source_id')}:{row.get('cluster')}", "source_id": str(row.get("source_id")), "cluster": str(row.get("cluster")), "fusion_identity": row.get("identity"), "clip_path": str(path.relative_to(ROOT))})

    all_rows = []
    for row in chat_rows:
        all_rows.append({"kind": "chat_tts", "record_id": row["challenge_id"], "source_id": str(row["source_id"]), "clip_path": row["clip_path"]})
    for row in family_rows:
        all_rows.append({"kind": "family_anchor", "record_id": row.get("anchor_id"), "source_id": str(row["source_id"]), "clip_path": row["clip_path"]})
    for row in guest_rows:
        all_rows.append({"kind": "guest_anchor", "record_id": row.get("record_id"), "source_id": str(row["source_id"]), "clip_path": row["clip_path"]})
    all_rows.extend(candidate_rows)
    verifier = pipeline(task=Tasks.speaker_verification, model=MODEL, device="cuda:0" if torch.cuda.is_available() else "cpu")
    vectors = []
    paths = [str(ROOT / row["clip_path"]) for row in all_rows]
    for offset in range(0, len(paths), 64):
        vectors.append(normalize(np.asarray(verifier(paths[offset:offset + 64], output_emb=True)["embs"], dtype="float32")))
    matrix = np.concatenate(vectors, axis=0)
    by_kind = defaultdict(list)
    for index, row in enumerate(all_rows):
        by_kind[row["kind"]].append(index)
    train_tts = [i for i in by_kind["chat_tts"] if source_split(all_rows[i]["source_id"]) == "bank_train"]
    holdout_tts = [i for i in by_kind["chat_tts"] if source_split(all_rows[i]["source_id"]) == "holdout"]
    tts_proto = normalize(matrix[train_tts].mean(axis=0, keepdims=True))[0] if train_tts else normalize(matrix[by_kind["chat_tts"]].mean(axis=0, keepdims=True))[0]
    chat_anchor_sim = [float(np.dot(matrix[i], tts_proto)) for i in by_kind["family_anchor"]]
    guest_anchor_sim = [float(np.dot(matrix[i], tts_proto)) for i in by_kind["guest_anchor"]]
    train_tts_sim = [float(np.dot(matrix[i], tts_proto)) for i in train_tts]
    holdout_tts_sim = [float(np.dot(matrix[i], tts_proto)) for i in holdout_tts]
    zero_family_reject_threshold = max(chat_anchor_sim) + 1e-4 if chat_anchor_sim else 1.0
    challenge_by_key = defaultdict(list)
    for row in chat_rows:
        challenge_by_key[f"{row['source_id']}:{row.get('cluster')}"] .append(row)
    output = []
    for index, row in enumerate(candidate_rows):
        key = f"{row['source_id']}:{row['cluster']}"
        challenge = challenge_by_key.get(key, [])
        turns = timelines.get(row["source_id"], {}).get(row["cluster"], [])
        contexts = [context_features(turns, float(item.get("start") or 0), float(item.get("end") or 0)) for item in challenge]
        text = " ".join(str(item.get("text_preview") or "") for item in challenge).casefold()
        keyword_hits = sorted({keyword for keyword in KEYWORDS if keyword in text})
        prior = priors.get(row["source_id"], {})
        meta = source_meta.get(row["source_id"], {})
        sim = float(np.dot(matrix[len(all_rows) - len(candidate_rows) + index], tts_proto))
        # The challenge bank is intentionally unlabeled; similarity to the
        # bank cannot name a recurring voice without an independent anchor.
        repeated_voice = False
        explicit_event = bool(challenge)
        high_acoustic = sim >= zero_family_reject_threshold
        strong_reject = bool(high_acoustic and (explicit_event or repeated_voice))
        quarantine = bool(explicit_event)
        decision = "REJECT_NON_TARGET_TTS" if strong_reject else ("QUARANTINE_REVIEW" if quarantine else "NONE")
        output.append({"record_id": row["record_id"], "source_id": row["source_id"], "cluster": row["cluster"], "fusion_identity": row["fusion_identity"], "chat_tts_prototype_similarity": round(sim, 6), "chat_tts_rejection_threshold_zero_family_anchor_false_reject": round(zero_family_reject_threshold, 6), "tts_interval_count": len(challenge), "tts_total_seconds": round(sum(float(x.get("end") or 0) - float(x.get("start") or 0) for x in challenge), 3), "event_keyword_hits": keyword_hits, "context_features": contexts, "source_prior": {"possible": prior.get("possible", []), "confidence": prior.get("confidence"), "metadata_text_hits": sorted({keyword for keyword in KEYWORDS if keyword in json.dumps(meta, ensure_ascii=False).casefold()})}, "repeated_chat_tts_voice_evidence": {"available": False, "reason": "unlabeled challenge bank; no cross-source voice identity is promoted from stress-only clips"}, "rejection_decision": decision, "rejection_only_layer": True, "can_promote_family": False, "can_recalibrate_family": False, "gold_validated": False, "provenance": {"model": MODEL, "version": VERSION, "stress_only_chat_tts": True, "family_anchor_false_reject_control": "max_family_anchor_similarity_plus_epsilon", "not_used_for_family_promotion": True}})

    output_path = ROOT / "identity_results" / "chat_tts_rejection_scores.jsonl"
    output_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output), encoding="utf-8")
    family_candidates = [row for row in output if str(row["fusion_identity"]).startswith("NEURO_FAMILY")]
    report = {"schema_version": "0.1.0", "created_at": datetime.now(timezone.utc).isoformat(), "status": "CHAT_TTS_REJECTION_LAYER_AUDIT_COMPLETE_REJECT_ONLY", "version": VERSION, "model": MODEL, "candidate_cluster_count": len(output), "current_family_candidate_count": len(family_candidates), "chat_tts_clip_count": len(chat_rows), "chat_tts_source_count": len({row["source_id"] for row in chat_rows}), "chat_tts_source_disjoint_split": {"bank_train_count": len(train_tts), "holdout_count": len(holdout_tts), "bank_train_source_count": len({all_rows[i]['source_id'] for i in train_tts}), "holdout_source_count": len({all_rows[i]['source_id'] for i in holdout_tts})}, "acoustic_similarity": {"family_anchor_count": len(chat_anchor_sim), "family_anchor_max": round(max(chat_anchor_sim), 6) if chat_anchor_sim else None, "family_anchor_p95": round(float(np.quantile(chat_anchor_sim, 0.95)), 6) if chat_anchor_sim else None, "guest_anchor_count": len(guest_anchor_sim), "guest_anchor_median": round(float(np.median(guest_anchor_sim)), 6) if guest_anchor_sim else None, "chat_tts_train_median": round(float(np.median(train_tts_sim)), 6) if train_tts_sim else None, "chat_tts_holdout_median": round(float(np.median(holdout_tts_sim)), 6) if holdout_tts_sim else None, "zero_family_anchor_false_reject_threshold": round(zero_family_reject_threshold, 6)}, "decision_counts": dict(Counter(row["rejection_decision"] for row in output)), "family_candidate_decisions": dict(Counter(row["rejection_decision"] for row in family_candidates)), "family_candidate_reject_review_rows": [row["record_id"] for row in family_candidates if row["rejection_decision"] != "NONE"], "calibration_policy": "CHAT_TTS/OTHER_TTS layer is reject-only. Stress-only clips are not gold and cannot promote or recalibrate NEURO_FAMILY. Acoustic reject threshold is bounded by zero false reject on source-disjoint AUTO_TRUSTED family anchors.", "output": str(output_path.relative_to(ROOT)), "promotion_decision": "DO_NOT_PROMOTE"}
    write_json(ROOT / "reports" / "chat_tts_rejection_layer.json", report)
    print(json.dumps({"status": report["status"], "candidate_cluster_count": len(output), "current_family_candidate_count": len(family_candidates), "decision_counts": report["decision_counts"], "family_candidate_decisions": report["family_candidate_decisions"], "zero_family_anchor_false_reject_threshold": report["acoustic_similarity"]["zero_family_anchor_false_reject_threshold"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
