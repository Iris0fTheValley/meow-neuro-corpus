from __future__ import annotations

"""Promote only the strict, independently gated subset of production_v2.

This is a fail-closed promotion layer.  It does not change the family scorer,
does not promote source mappings, and never writes training_candidate=true.
Target turns must resolve to a current family cluster with no named-guest
review flag and no chat-TTS similarity at or above the zero-family-anchor
false-reject bound.
"""

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from manifest_tools import ROOT, write_json
from build_final_sft_artifacts import normalize, text_ok


V2 = ROOT / "datasets" / "meow_v02_sft_v1" / "production_v2_recovered"
OUT = ROOT / "datasets" / "meow_v02_sft_v1" / "final_sft_candidate_v1"
TIMELINES = ROOT / "unique_timelines"
FAMILY_AUDIT = ROOT / "reports" / "neuro_family_148_cluster_audit.jsonl"
TARGETS = {"NEURO_FAMILY_HIGH", "NEURO_FAMILY_MEDIUM"}
VERSION = "final-sft-candidate-2026-09-12-v1-fail-closed-promotion"


def load_rows() -> list[dict]:
    rows = []
    for name in ("train.jsonl", "validation.jsonl", "sealed_eval.jsonl"):
        path = V2 / name
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return rows


def split_clusters(rows: list[dict]) -> dict[str, set[str]]:
    counts = Counter(row["canonical_recording_id"] for row in rows)
    ordered = [cluster for cluster, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]
    if len(ordered) >= 9:
        pool = ordered[5:11] if len(ordered) >= 11 else ordered[3:9]
        return {"train": set(ordered) - set(pool), "validation": set(pool[:3]), "sealed_eval": set(pool[3:6])}
    if len(ordered) >= 6:
        pool = ordered[3:]
        midpoint = len(pool) // 2
        return {"train": set(ordered[:3]), "validation": set(pool[:midpoint]), "sealed_eval": set(pool[midpoint:])}
    if len(ordered) >= 3:
        return {"train": set(ordered[:-2]), "validation": {ordered[-2]}, "sealed_eval": {ordered[-1]}}
    return {"train": set(ordered), "validation": set(), "sealed_eval": set()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    risk = {}
    for line in FAMILY_AUDIT.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        rejection = item.get("chat_tts_rejection") or {}
        similarity = float(rejection.get("chat_tts_prototype_similarity") or 0.0)
        bound = float(rejection.get("chat_tts_rejection_threshold_zero_family_anchor_false_reject") or 999.0)
        risk[(str(item.get("source_id")), str(item.get("cluster")))] = {
            "guest_review": "high_named_guest_similarity_review" in (item.get("risk_flags") or []),
            "chat_tts_high": similarity >= bound,
            "chat_tts_similarity": similarity,
            "chat_tts_bound": bound,
            "fusion_identity": item.get("fusion_identity"),
            "fusion_confidence": item.get("fusion_confidence"),
            "eres_segment_vote_rate": item.get("eres_segment_vote_rate"),
        }

    turns = {}
    for path in TIMELINES.glob("*.json"):
        try:
            timeline = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        source_id = str(timeline.get("source_id") or path.stem)
        for turn in timeline.get("turns") or []:
            turns[(source_id, str(turn.get("turn_id")))] = turn

    rows = load_rows()
    accepted = {}
    exclusions = Counter()
    audit_lines = []
    for row in rows:
        reason = None
        target_key = None
        target_turn = None
        raw_recovery = row.get("raw_recovery") or {}
        if raw_recovery.get("target_turn_id"):
            target_turn = turns.get((str(row.get("source_id")), str(raw_recovery.get("target_turn_id"))))
            if target_turn is None:
                reason = "target_turn_not_resolvable"
            else:
                target_key = (str(row.get("source_id")), str(target_turn.get("speaker")))
        else:
            # Resolve legacy v1 rows by source, target end timestamp, and text.
            end_value = (row.get("timestamps") or {}).get("end")
            assistant_text = normalize((row.get("messages") or [{}, {}])[-1].get("content"))
            possible = []
            for (source_id, _), turn in turns.items():
                if source_id != str(row.get("source_id")):
                    continue
                try:
                    close = abs(float((turn.get("timestamp") or {}).get("end")) - float(end_value)) <= 1.0
                except (TypeError, ValueError):
                    close = False
                if close and normalize(turn.get("text")) and normalize(turn.get("text")) in assistant_text:
                    possible.append(turn)
            if len(possible) != 1:
                reason = "legacy_target_turn_not_uniquely_resolvable"
            else:
                target_turn = possible[0]
                target_key = (str(row.get("source_id")), str(target_turn.get("speaker")))
        gate = risk.get(target_key) if target_key else None
        if reason is None and gate is None:
            reason = "family_cluster_risk_unmapped"
        if reason is None and row.get("identity") not in TARGETS:
            reason = "assistant_identity_not_family"
        if reason is None and float(row.get("speaker_confidence") or 0.0) < 0.70:
            reason = "assistant_speaker_confidence_below_0.70"
        if reason is None and gate["guest_review"]:
            reason = "named_guest_similarity_review_quarantine"
        if reason is None and gate["chat_tts_high"]:
            reason = "chat_tts_similarity_at_or_above_family_anchor_bound"
        assistant = (row.get("messages") or [{}])[-1].get("content") or ""
        if reason is None and (not text_ok(assistant, 2) or len(assistant) > 500):
            reason = "assistant_text_quality"
        if reason is None and len(row.get("messages") or []) < 2:
            reason = "missing_context_message"
        if reason is not None:
            exclusions[reason] += 1
            audit_lines.append({"sample_id": row.get("sample_id"), "source_id": row.get("source_id"), "canonical_recording_id": row.get("canonical_recording_id"), "decision": "REJECT", "reason": reason, "target_key": list(target_key) if target_key else None, "risk": gate})
            continue
        dedup_key = "\n".join(str(item.get("role")) + ":" + normalize(item.get("content")) for item in row.get("messages") or [])
        row = dict(row)
        row["promotion"] = {"status": "FINAL_SFT_CANDIDATE", "identity_gate": "current_family_mapping_plus_cluster_audit", "guest_similarity_gate": "no_high_named_guest_similarity_review", "chat_tts_gate": "similarity_below_zero_family_anchor_false_reject_bound", "training_candidate": False, "promotion_pipeline_version": VERSION}
        row.pop("training_candidate", None)
        row["_dedup_key"] = dedup_key
        previous = accepted.get(dedup_key)
        if previous is None or float(row.get("quality_score") or 0.0) > float(previous.get("quality_score") or 0.0):
            accepted[dedup_key] = row

    rows = list(accepted.values())
    rows.sort(key=lambda item: (-float(item.get("quality_score") or 0.0), item.get("sample_id", "")))
    response_seen = set()
    deduped = []
    for row in rows:
        response_key = normalize((row.get("messages") or [{}])[-1].get("content"))
        if response_key in response_seen:
            exclusions["near_duplicate_response"] += 1
            continue
        response_seen.add(response_key)
        row.pop("_dedup_key", None)
        deduped.append(row)
    rows = deduped
    split_map = split_clusters(rows)
    splits = {"train": [], "validation": [], "sealed_eval": []}
    for row in rows:
        cluster = row["canonical_recording_id"]
        row["split"] = "sealed_eval" if cluster in split_map["sealed_eval"] else ("validation" if cluster in split_map["validation"] else "train")
        splits[row["split"]].append(row)
    for split, split_rows in splits.items():
        path = OUT / ("sealed_eval.jsonl" if split == "sealed_eval" else split + ".jsonl")
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in sorted(split_rows, key=lambda item: item["sample_id"])), encoding="utf-8")
    (OUT / "promotion_audit.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in audit_lines), encoding="utf-8")
    leakage = {"train_validation": sorted(split_map["train"] & split_map["validation"]), "train_sealed_eval": sorted(split_map["train"] & split_map["sealed_eval"]), "validation_sealed_eval": sorted(split_map["validation"] & split_map["sealed_eval"])}
    raw_funnel = json.loads((V2 / "rejection_funnel.json").read_text(encoding="utf-8"))
    quality = {"schema_version": "1.0.0", "artifact_status": "FINAL_SFT_CANDIDATE_FAIL_CLOSED", "pipeline_version": VERSION, "created_at": datetime.now(timezone.utc).isoformat(), "input_production_v2_samples": len(load_rows()), "production_v1_samples": raw_funnel.get("production_v1_samples"), "raw_high_confidence_target_utterances": raw_funnel.get("target_total"), "raw_target_utterances_with_recoverable_context": raw_funnel.get("target_with_recoverable_context"), "usable_target_response_rate": raw_funnel.get("usable_target_response_rate"), "raw_recovery_counts": raw_funnel.get("recovery_counts", {}), "raw_unusable_target_reasons": raw_funnel.get("unusable_target_reasons", {}), "sample_counts": {split: len(split_rows) for split, split_rows in splits.items()}, "total_samples": len(rows), "unique_sources": len({row["source_id"] for row in rows}), "unique_recording_clusters": len({row["canonical_recording_id"] for row in rows}), "split_recording_clusters": {split: sorted(split_map[split]) for split in split_map}, "identity_counts": dict(Counter(row.get("identity") for row in rows)), "removed_as_identity_uncertain": exclusions.get("family_cluster_risk_unmapped", 0) + exclusions.get("assistant_identity_not_family", 0) + exclusions.get("assistant_speaker_confidence_below_0.70", 0) + exclusions.get("target_turn_not_resolvable", 0) + exclusions.get("legacy_target_turn_not_uniquely_resolvable", 0), "removed_as_tts_guest": exclusions.get("named_guest_similarity_review_quarantine", 0) + exclusions.get("chat_tts_similarity_at_or_above_family_anchor_bound", 0), "removed_as_transcript_context_bad": exclusions.get("assistant_text_quality", 0) + exclusions.get("missing_context_message", 0), "removed_as_duplicate": exclusions.get("near_duplicate_response", 0), "exclusion_counts": dict(exclusions), "recording_cluster_leakage": leakage, "training_candidate_true_count": 0, "promotion_status": "CANDIDATE_ARTIFACT_ONLY_NO_SOURCE_PROMOTION", "estimated_speaker_contamination_risk": "low_relative_to_v2; all rows pass named-guest review and chat-TTS bound quarantine, but challenge clips remain unlabeled stress evidence", "estimated_context_error_risk": "low_to_moderate; inherited raw-timeline context gate and independent structural validator", "data_quality_ready": bool(rows) and len(split_map["validation"]) >= 2 and len(split_map["sealed_eval"]) >= 2 and not any(leakage.values()), "ready_for_first_sft": False, "readiness_reason": "Explicit current phase hold forbids SFT/promotion; artifact is fail-closed and ready for user quality review, but training_candidate remains false."}
    input_count = quality["input_production_v2_samples"]
    identity_removed = quality["removed_as_identity_uncertain"]
    tts_guest_removed = quality["removed_as_tts_guest"]
    transcript_removed = quality["removed_as_transcript_context_bad"]
    funnel = {
        "schema_version": "1.0.0",
        "pipeline_version": VERSION,
        "raw_recovery_reference": str((V2 / "rejection_funnel.json").relative_to(ROOT)),
        "stages": [
            {"stage": "initial_production_v2_candidates", "rows": input_count, "removed_rows": 0},
            {"stage": "assistant_identity_and_resolution_gate", "rows": input_count - identity_removed, "removed_rows": identity_removed},
            {"stage": "guest_tts_contamination_rejection_gate", "rows": input_count - identity_removed - tts_guest_removed, "removed_rows": tts_guest_removed},
            {"stage": "context_and_transcript_gate", "rows": input_count - identity_removed - tts_guest_removed - transcript_removed, "removed_rows": transcript_removed},
            {"stage": "exact_and_response_deduplication", "rows": len(rows), "removed_rows": quality["removed_as_duplicate"]},
            {"stage": "final_fail_closed_candidate", "rows": len(rows), "removed_rows": 0},
        ],
        "recording_cluster_leakage": leakage,
        "exclusion_counts": dict(exclusions),
        "raw_recovery_target_total": 82726,
        "raw_recovery_target_with_recoverable_context": 25476,
        "raw_recovery_usable_target_response_rate": 0.308645,
        "policy": "No assistant identity relaxation; context-side guest/TTS is allowed only when inherited raw reconstruction marked it coherent; training_candidate remains false.",
    }
    manifest = {"schema_version": "1.0.0", "artifact_status": "FINAL_SFT_CANDIDATE_FAIL_CLOSED", "pipeline_version": VERSION, "files": {"train": "train.jsonl", "validation": "validation.jsonl", "sealed_eval": "sealed_eval.jsonl", "dataset_manifest": "dataset_manifest.json", "quality_report": "quality_report.json", "validation_report": "validation_report.json", "promotion_audit": "promotion_audit.jsonl"}, "source_artifacts": {"production_v2": str(V2.relative_to(ROOT)), "family_cluster_audit": str(FAMILY_AUDIT.relative_to(ROOT))}, "recording_cluster_disjoint": not any(leakage.values()), "training_candidate_true_count": 0, "promotion_status": "CANDIDATE_ARTIFACT_ONLY_NO_SOURCE_PROMOTION"}
    delivery = {"schema_version": "1.0.0", "artifact_status": "FINAL_SFT_CANDIDATE_DELIVERY", "raw_high_confidence_target_utterances": quality["raw_high_confidence_target_utterances"], "raw_target_utterances_with_recoverable_context": quality["raw_target_utterances_with_recoverable_context"], "usable_target_response_rate": quality["usable_target_response_rate"], "final_train_samples": len(splits["train"]), "validation_samples": len(splits["validation"]), "sealed_eval_samples": len(splits["sealed_eval"]), "unique_sources": quality["unique_sources"], "unique_recording_clusters": quality["unique_recording_clusters"], "removed_as_identity_uncertain": quality["removed_as_identity_uncertain"], "removed_as_tts_guest": quality["removed_as_tts_guest"], "removed_as_transcript_context_bad": quality["removed_as_transcript_context_bad"], "removed_as_duplicate": quality["removed_as_duplicate"], "estimated_speaker_contamination_risk": quality["estimated_speaker_contamination_risk"], "estimated_context_error_risk": quality["estimated_context_error_risk"], "READY_FOR_FIRST_SFT": "NO", "readiness_reason": quality["readiness_reason"], "paths": {"dataset": str(OUT.relative_to(ROOT)), "manifest": str((OUT / "dataset_manifest.json").relative_to(ROOT)), "quality_report": str((OUT / "quality_report.json").relative_to(ROOT))}}
    write_json(OUT / "quality_report.json", quality)
    write_json(OUT / "rejection_funnel.json", funnel)
    write_json(OUT / "dataset_manifest.json", manifest)
    write_json(OUT / "final_delivery_report.json", delivery)
    print(json.dumps({"status": "FINAL_SFT_CANDIDATE_BUILT", "sample_counts": quality["sample_counts"], "sources": quality["unique_sources"], "recording_clusters": quality["unique_recording_clusters"], "exclusions": exclusions, "data_quality_ready": quality["data_quality_ready"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
