from __future__ import annotations

import json
from datetime import datetime, timezone

from derive_candidate_counts import derive_candidate_counts, iter_jsonl
from manifest_tools import ROOT, write_json


def load(path: str) -> dict:
    target = ROOT / path
    return json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}


def derive_mapping_counts(path: str) -> dict:
    counts = {}
    total = 0
    training_true = 0
    for row in iter_jsonl(ROOT / path):
        total += 1
        identity = str(row.get("identity") or "UNKNOWN")
        counts[identity] = counts.get(identity, 0) + 1
        if row.get("training_candidate") is True:
            training_true += 1
    return {"row_count": total, "identity_counts": counts, "training_candidate_count": training_true, "source": path}


def main() -> None:
    checkpoint = load("checkpoints/latest.json")
    gates = checkpoint.get("quality_gates", {})
    refs = load("speaker_refs/reference_human_review_queue.json")
    natural = load("reports/natural_turn_progress.json")
    conversations = load("reports/natural_conversation_progress.json")
    qa = load("reports/conversation_artifact_qa.json")
    identity_queue = load("reports/identity_mapping_queue.json")
    retry_probe = load("reports/retry_queue_proxy_probe.json")
    retry_transcript = load("reports/retry_transcript_fallbacks.json")
    retry_normalized = load("reports/retry_transcript_fallback_normalization.json")
    retry_conversations = load("reports/retry_transcript_conversation_progress.json")
    retry_conversation_qa = load("reports/retry_transcript_conversation_qa.json")
    family_bank = load("speaker_refs/reference_bank_neuro_family.json")
    family_mapping = load("reports/identity_mapping_family_proxy.json")
    family_conversations = load("reports/family_mapped_conversation_progress.json")
    family_conversation_qa = load("reports/family_mapped_conversation_qa.json")
    family_candidates = load("reports/family_training_candidate_progress.json")
    family_candidates_qa = load("reports/family_training_candidate_qa.json")
    family_gold = load("speaker_refs/gold_validation_neuro_family.json")
    negative_review = load("reports/open_set_negative_review_manifest.json")
    negative_clips = load("reports/open_set_negative_review_clips.json")
    negative_scores = load("reports/open_set_negative_review_scores.json")
    auto_anchor = load("reports/auto_validation_anchor_report.json")
    calibration_bank = load("reports/open_set_calibration_bank.json")
    calibration = load("reports/open_set_auto_anchor_calibration.json")
    calibration_xvector = load("reports/open_set_auto_anchor_calibration_xvector.json")
    ensemble_calibration = load("reports/open_set_ensemble_calibration.json")
    participant_prior = load("speaker_refs/identity_closure/expected_participants_report.json")
    guest_negative = load("speaker_refs/identity_closure/recurring_guest_negative_bank_report.json")
    guest_prototypes = load("speaker_refs/identity_closure/recurring_guest_negative_prototypes_report.json")
    semantic_bank = load("speaker_refs/identity_closure/semantic_identity_bank.json")
    fusion = load("reports/identity_mapping_multimodal_fusion_proxy.json")
    eres2netv2_remap = load("reports/identity_mapping_eres2netv2_proxy.json")
    model_benchmark = load("reports/speaker_model_benchmark_identity_closure.json")
    fusion_candidates = load("reports/family_training_candidate_fusion_progress.json")
    chat_tts_stress = load("speaker_refs/identity_closure/chat_tts_stress_test_report.json")
    hard_negative_bank = load("speaker_refs/identity_closure/hard_negative_validation_bank_report.json")
    chat_tts_rejection = load("reports/chat_tts_rejection_layer.json")
    fusion_audio_audit = load("reports/identity_fusion_audio_consensus_audit.json")
    validation_boundary = load("reports/identity_validation_boundary_audit.json")
    recording_clusters = load("reports/recording_content_cluster_audit.json")
    lineage = load("reports/identity_closure_lineage.json")
    derived_candidates = derive_candidate_counts()
    derived_fusion_mapping = derive_mapping_counts("identity_results/identity_mapping_multimodal_fusion_proxy.jsonl")
    derived_eres_mapping = derive_mapping_counts("identity_results/identity_mapping_eres2netv2_proxy.jsonl")
    lineage_complete = bool(lineage) and all(
        node.get("artifact_status") == "current"
        for name, node in lineage.get("nodes", {}).items()
        if name in {"recording_content_clusters", "identity_validation_boundary", "eres2netv2_cluster_remap", "multimodal_fusion_mapping", "post_fusion_candidate_regrade"}
    )
    invariants = {
        "candidate_report_matches_row_level": derived_candidates["s_a_review_candidate_count"] == fusion_candidates.get("s_a_review_candidate_count"),
        "candidate_training_flags_zero": derived_candidates["training_candidate_count"] == 0,
        "fusion_mapping_training_flags_zero": derived_fusion_mapping["training_candidate_count"] == 0,
        "eres_mapping_training_flags_zero": derived_eres_mapping["training_candidate_count"] == 0,
        "recording_cluster_report_present": bool(recording_clusters),
        "validation_boundary_report_present": bool(validation_boundary),
        "lineage_complete": lineage_complete,
    }
    prerequisites_pass = all(invariants.values()) and validation_boundary.get("family_positive_validation", {}).get("evidence_sufficiency") is True
    training_candidate_count = derived_candidates["training_candidate_count"] if prerequisites_pass else 0
    report = {
        "schema_version": "0.1.0",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": "CONTINUE_WORKING",
        "readiness": "NOT_READY_FOR_SFT",
        "stopping_conditions": {
            "trusted_neuro_reference": "AUTO_TRUSTED_source_disjoint_anchor_bank_active; HUMAN_VERIFIED_optional_higher_evidence",
            "trusted_evil_reference": "AUTO_TRUSTED_source_disjoint_anchor_bank_active; HUMAN_VERIFIED_optional_higher_evidence",
            "trusted_vedal_reference": "cross_source_auto_trusted_challenge_bank_expanding; human gold not required",
            "gold_validation_set": "AUTO_TRUSTED_source_disjoint_validation_allowed; HUMAN_VERIFIED_not_a_pipeline_prerequisite",
            "speaker_verification_precision": "audio-consensus-gated proxy active; expanded hard-negative stress audit pending; no 0/49 claim used as final proof",
            "identity_mapping_backlog": "1068 clusters remapped with ERes2NetV2 consensus gate; recording/content split and independent-positive evidence audit pending",
            "recording_content_split": "recording/content cluster audit present; overlap or insufficient independent positive evidence blocks closure",
            "positive_validation": validation_boundary.get("family_positive_validation", {}).get("status", "missing"),
            "canonical_timelines": "partial_validation_tranche",
            "transcript_contamination": "source_specific_cleaning_plus_review_queues",
            "natural_turn_reconstruction": "partial_validation_tranche",
            "conversation_generation": "family_mapped_proxy_qa_passed_training_closed",
            "forensic_annotation": "cheap_surface_only_identity_semantics_pending",
            "duplicate_handling": "review_pending_no_deletions",
            "s_and_a_candidates": "review_candidates_generated_training_closed",
            "coverage_audit": "anonymous_only_identity_gaps_open",
            "quarantine_reasons": "present",
            "retry_queue": "26_alternate_source_or_transcript_fallback_pending",
            "discovery_saturation": "not_proven",
            "high_value_processing_queue": "nonempty",
            "final_corpus_audit": "not_complete",
            "meow_v02_data_readiness": "this_interim_report_only",
        },
        "evidence": {
            "checkpoint_state": gates.get("state"),
            "reference_status": gates.get("reference_bank_status"),
            "trusted_clip_count": gates.get("trusted_clip_count", 0),
            "neuro_family_mapping_ready_clip_count": family_bank.get("mapping_ready_clip_count", 0),
            "neuro_family_gold_validated_trusted_count": family_bank.get("gold_validated_trusted_count", 0),
            "neuro_family_gold_validation_manifest_status": family_gold.get("status"),
            "neuro_family_gold_validation_candidate_count": sum(family_gold.get("population_counts", {}).values()),
            "neuro_family_gold_labeled_count": family_gold.get("gold_labeled_count", 0),
            "open_set_negative_review_status": negative_review.get("status"),
            "open_set_negative_review_candidate_count": negative_review.get("candidate_count", 0),
            "open_set_negative_review_missing_buckets": negative_review.get("missing_buckets", []),
            "open_set_negative_review_clip_status": negative_clips.get("status"),
            "open_set_negative_review_clip_count": negative_clips.get("clip_count", 0),
            "open_set_negative_gold_labeled_count": negative_clips.get("gold_label_count", 0),
            "open_set_negative_score_triage_status": negative_scores.get("status"),
            "open_set_negative_score_triage_clip_count": negative_scores.get("negative_review_clip_count", 0),
            "open_set_negative_score_triage_bucket_summary": negative_scores.get("bucket_summary", {}),
            "auto_validation_anchor_status": auto_anchor.get("status"),
            "auto_validation_anchor_count": auto_anchor.get("anchor_count", 0),
            "auto_validation_anchor_label_counts": auto_anchor.get("anchor_label_counts", {}),
            "auto_validation_anchor_source_counts": auto_anchor.get("anchor_source_counts", {}),
            "open_set_calibration_bank_status": calibration_bank.get("status"),
            "open_set_calibration_bank_label_counts": calibration_bank.get("label_counts", {}),
            "open_set_calibration_excluded_confounds_count": calibration_bank.get("excluded_confounds_count", 0),
            "open_set_ecapa_calibration_status": calibration.get("status"),
            "open_set_ecapa_selected_operating_point": calibration.get("selected_operating_point", {}),
            "open_set_xvector_calibration_status": calibration_xvector.get("status"),
            "open_set_xvector_selected_operating_point": calibration_xvector.get("selected_operating_point", {}),
            "open_set_ensemble_calibration_status": ensemble_calibration.get("status"),
            "open_set_ensemble_calibration_operating_point": {
                "ecapa_margin_threshold": ensemble_calibration.get("ecapa_margin_threshold"),
                "xvector_margin_threshold": ensemble_calibration.get("xvector_margin_threshold"),
                "family_recall_proxy": ensemble_calibration.get("family_recall_proxy"),
                "nontarget_false_positive_rate": ensemble_calibration.get("nontarget_ensemble_false_positive_rate"),
            },
            "identity_closure_participant_prior_status": participant_prior.get("status", "missing"),
            "identity_closure_participant_prior_source_count": participant_prior.get("source_count", 0),
            "identity_closure_guest_negative_status": guest_negative.get("status", "missing"),
            "identity_closure_guest_negative_candidate_count": guest_negative.get("candidate_count", 0),
            "identity_closure_guest_negative_source_counts": guest_negative.get("candidate_source_counts", {}),
            "identity_closure_guest_prototype_status": guest_prototypes.get("status", "missing"),
            "identity_closure_guest_prototype_count": guest_prototypes.get("prototype_count", 0),
            "identity_closure_guest_prototype_promotion_decision": guest_prototypes.get("promotion_decision", "DO_NOT_PROMOTE"),
            "identity_closure_semantic_bank_status": semantic_bank.get("status", "missing"),
            "identity_closure_semantic_bank_record_count": len(semantic_bank.get("records", [])),
            "identity_closure_fusion_status": fusion.get("status", "missing"),
            "identity_closure_fusion_cluster_count": derived_fusion_mapping["row_count"],
            "identity_closure_fusion_identity_counts": derived_fusion_mapping["identity_counts"],
            "identity_closure_eres2netv2_remap_status": eres2netv2_remap.get("status", "missing"),
            "identity_closure_eres2netv2_remap_model": eres2netv2_remap.get("model"),
            "identity_closure_eres2netv2_remap_cluster_count": derived_eres_mapping["row_count"],
            "identity_closure_eres2netv2_remap_identity_counts": derived_eres_mapping["identity_counts"],
            "identity_closure_eres2netv2_remap_promotion_decision": eres2netv2_remap.get("promotion_decision", "DO_NOT_PROMOTE"),
            "identity_closure_eres2netv2_remap_training_candidate_count": eres2netv2_remap.get("training_candidate_count", 0),
            "identity_model_benchmark_status": model_benchmark.get("status", "missing"),
            "identity_model_benchmark_clip_count": model_benchmark.get("clip_count", 0),
            "identity_model_benchmark_models": {name: {
                "status": value.get("status"),
                "selected_operating_point": value.get("selected_operating_point", {}),
            } for name, value in (model_benchmark.get("models") or {}).items()},
            "identity_model_benchmark_promotion_decision": model_benchmark.get("promotion_decision", "DO_NOT_PROMOTE"),
            "identity_closure_fusion_candidate_status": fusion_candidates.get("status", "missing"),
            "identity_closure_fusion_candidate_count": derived_candidates["s_a_review_candidate_count"],
            "identity_closure_fusion_training_candidate_count": training_candidate_count,
            "identity_closure_chat_tts_stress_status": chat_tts_stress.get("status", "missing"),
            "identity_closure_chat_tts_stress_clip_count": chat_tts_stress.get("clip_count", 0),
            "identity_closure_chat_tts_stress_potential_family_accept_rate": chat_tts_stress.get("potential_family_accept_rate"),
            "identity_closure_chat_tts_stress_use_for_calibration": chat_tts_stress.get("use_for_calibration", False),
            "identity_closure_chat_tts_rejection_layer_status": chat_tts_rejection.get("status", "missing"),
            "identity_closure_chat_tts_rejection_layer_version": chat_tts_rejection.get("version"),
            "identity_closure_chat_tts_rejection_candidate_cluster_count": chat_tts_rejection.get("candidate_cluster_count", 0),
            "identity_closure_chat_tts_rejection_current_family_candidate_count": chat_tts_rejection.get("current_family_candidate_count", 0),
            "identity_closure_chat_tts_rejection_decision_counts": chat_tts_rejection.get("decision_counts", {}),
            "identity_closure_chat_tts_rejection_family_decision_counts": chat_tts_rejection.get("family_candidate_decisions", {}),
            "identity_closure_chat_tts_rejection_calibration_policy": chat_tts_rejection.get("calibration_policy"),
            "identity_closure_chat_tts_rejection_promotion_effect": "REJECTION_ONLY_NO_FAMILY_PROMOTION",
            "identity_closure_hard_negative_bank_status": hard_negative_bank.get("status", "missing"),
            "identity_closure_hard_negative_bank_record_count": hard_negative_bank.get("record_count", 0),
            "identity_closure_hard_negative_auto_trusted_count": hard_negative_bank.get("auto_trusted_named_guest_count", 0),
            "identity_closure_hard_negative_unlabeled_stress_count": hard_negative_bank.get("unlabeled_stress_clip_count", 0),
            "identity_closure_hard_negative_unlabeled_bucket_counts": hard_negative_bank.get("unlabeled_stress_bucket_counts", {}),
            "identity_closure_hard_negative_human_gold_is_promotion_blocker": hard_negative_bank.get("human_gold_is_promotion_blocker", False),
            "identity_closure_fusion_audio_audit_status": fusion_audio_audit.get("status", "missing"),
            "identity_closure_fusion_audio_audit_family_count": fusion_audio_audit.get("fusion_family_count", 0),
            "identity_closure_fusion_audio_audit_family_supported_by_eres2netv2": fusion_audio_audit.get("fusion_family_supported_by_eres2netv2", 0),
            "identity_closure_fusion_audio_audit_semantic_or_metadata_only_count": fusion_audio_audit.get("semantic_or_metadata_only_family_promotion_count", 0),
            "family_mapping_identity_counts": family_mapping.get("identity_counts", {}),
            "family_mapped_conversation_count": family_conversations.get("record_count", 0),
            "family_mapped_conversation_qa": family_conversation_qa.get("status"),
            "family_s_a_review_candidate_count": derived_candidates["s_a_review_candidate_count"],
            "family_training_candidate_qa": family_candidates_qa.get("status"),
            "human_review_candidates": refs.get("candidate_count", 0),
            "human_reviewed_count": refs.get("reviewed_count", 0),
            "natural_turn_sources": len(natural.get("completed", [])),
            "natural_conversation_windows": conversations.get("window_count", 0),
            "conversation_artifact_qa": qa.get("status"),
            "training_candidate_count": training_candidate_count,
            "current_operating_threshold": 0.163198,
            "historical_benchmark_threshold": 0.1205,
            "current_post_fusion_s_a_count": derived_candidates["s_a_review_candidate_count"],
            "historical_pre_fusion_s_a_count": family_candidates.get("s_a_review_candidate_count", 0),
            "derived_candidate_counts": derived_candidates,
            "identity_validation_boundary": validation_boundary,
            "recording_content_cluster_audit": recording_clusters,
            "lineage": lineage,
            "invariants": invariants,
            "training_gate_closed": not prerequisites_pass,
            "human_gold_is_promotion_blocker": False,
            "auto_trusted_anchor_calibration_enabled": True,
            "auto_trusted_anchor_policy": "source-disjoint multi-evidence anchors may calibrate and validate; HUMAN_VERIFIED is a higher evidence tier, not a prerequisite",
            "identity_mapping_pending_clusters": identity_queue.get("cluster_count", 0),
            "identity_mapping_mapped_clusters": identity_queue.get("mapped_cluster_count", 0),
            "identity_mapping_family_proxy_clusters": family_mapping.get("cluster_count", 0),
            "identity_mapping_family_proxy_sources": family_mapping.get("source_count", 0),
            "retry_queue_proxy_probe_status": retry_probe.get("status"),
            "retry_queue_proxy_probe_count": retry_probe.get("probe_count", 0),
            "retry_queue_proxy_probe_status_counts": retry_probe.get("status_counts", {}),
            "retry_queue_proxy_transcript_fallback_count": sum(1 for item in retry_probe.get("results", []) if item.get("transcript_fallback_available")),
            "retry_transcript_fallback_status": retry_transcript.get("status"),
            "retry_transcript_fallback_attempt_count": retry_transcript.get("attempt_count", 0),
            "retry_transcript_fallback_success_count": retry_transcript.get("success_count", 0),
            "retry_transcript_normalization_status": retry_normalized.get("status"),
            "retry_transcript_normalized_segment_count": retry_normalized.get("segment_count", 0),
            "retry_transcript_normalized_source_count": retry_normalized.get("source_count", 0),
            "retry_transcript_conversation_status": retry_conversations.get("status"),
            "retry_transcript_conversation_window_count": retry_conversations.get("window_count", 0),
            "retry_transcript_conversation_qa_status": retry_conversation_qa.get("status"),
            "retry_transcript_conversation_qa_error_count": retry_conversation_qa.get("error_count", 0),
        },
        "next_high_value_actions": [
            "expand metadata-confirmed, cross-source Vedal and OTHER auto anchors",
            "add explicitly identified guest/chat-TTS/singing/game-voice challenge anchors; keep unlabeled mode buckets out of precision claims",
            "rerun calibrated open-set mapping after anchor bank expansion",
            "recompute provisional S/A candidates after each verifier calibration; keep training_candidate=false until final promotion gates",
            "rerun forensic semantic annotation after final gold review",
            "continue natural turn reconstruction/QA on the remaining high-value backlog",
            "resolve alternate public sources for retryable high-value queue",
            "expand recurring guest candidates into independently verified cross-source acoustic prototypes",
            "evaluate ERes2NetV2/CAM++/ERes2Net on a larger source-disjoint non-target bank before remapping promotion",
            "verify recurring guest candidates per cluster and grow independent negative coverage before any S/A promotion",
        ],
    }
    write_json(ROOT / "reports" / "meow_v02_data_readiness_interim.json", report)
    print(json.dumps({"status": report["status"], "readiness": report["readiness"], "trusted_clip_count": report["evidence"]["trusted_clip_count"], "natural_turn_sources": report["evidence"]["natural_turn_sources"], "training_candidate_count": training_candidate_count, "invariants": invariants}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
