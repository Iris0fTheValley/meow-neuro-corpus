from __future__ import annotations

import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from manifest_tools import ROOT, read_jsonl, write_json


def main() -> None:
    rows = read_jsonl(ROOT / "manifest" / "master_video_manifest.jsonl")
    statuses = Counter()
    for row in rows:
        statuses[row.get("processing_status", "pending")] += 1
        for field in ("metadata_status", "audio_download_status", "video_download_status", "subtitle_status", "asr_status", "diarization_status"):
            if row.get(field):
                statuses[f"{field}:{row[field]}"] += 1
    minimal_complete = sum(1 for r in rows if all(r.get(k) in {"done", "not_available"} for k in ("metadata_status", "audio_download_status", "asr_status")))
    full_complete = sum(1 for r in rows if all([
        r.get("metadata_status") in {"done", "not_available"},
        r.get("audio_download_status") in {"done", "not_available"},
        r.get("asr_status") in {"done", "source_transcript", "not_available"},
        r.get("diarization_status") == "done",
        r.get("speaker_mapping_status") in {"attempted", "attempted_anonymous", "mapped", "uncertain"},
        r.get("conversation_status") == "done",
        r.get("quality_metrics_path"),
    ]))
    reference_bank_path = ROOT / "speaker_refs" / "reference_bank_all_dual_model.json"
    reference_bank = json.loads(reference_bank_path.read_text(encoding="utf-8")) if reference_bank_path.exists() else {}
    natural_turn_path = ROOT / "reports" / "natural_turn_progress.json"
    natural_turns = json.loads(natural_turn_path.read_text(encoding="utf-8")) if natural_turn_path.exists() else {}
    natural_conversation_path = ROOT / "reports" / "natural_conversation_progress.json"
    natural_conversations = json.loads(natural_conversation_path.read_text(encoding="utf-8")) if natural_conversation_path.exists() else {}
    contamination_path = ROOT / "reports" / "transcript_contamination_audit.json"
    contamination = json.loads(contamination_path.read_text(encoding="utf-8")) if contamination_path.exists() else {}
    twitch_path = ROOT / "reports" / "twitchtranscript_conversation_progress.json"
    twitch = json.loads(twitch_path.read_text(encoding="utf-8")) if twitch_path.exists() else {}
    conversation_qa_path = ROOT / "reports" / "conversation_artifact_qa.json"
    conversation_qa = json.loads(conversation_qa_path.read_text(encoding="utf-8")) if conversation_qa_path.exists() else {}
    review_queue_path = ROOT / "speaker_refs" / "reference_human_review_queue.json"
    review_queue = json.loads(review_queue_path.read_text(encoding="utf-8")) if review_queue_path.exists() else {}
    natural_quality_path = ROOT / "reports" / "natural_turn_quality_review.json"
    natural_quality = json.loads(natural_quality_path.read_text(encoding="utf-8")) if natural_quality_path.exists() else {}
    duplicate_review_path = ROOT / "reports" / "duplicate_group_review.json"
    duplicate_review = json.loads(duplicate_review_path.read_text(encoding="utf-8")) if duplicate_review_path.exists() else {}
    surface_features_path = ROOT / "reports" / "natural_anonymous_surface_features.json"
    surface_features = json.loads(surface_features_path.read_text(encoding="utf-8")) if surface_features_path.exists() else {}
    coverage_path = ROOT / "reports" / "anonymous_coverage_audit.json"
    coverage = json.loads(coverage_path.read_text(encoding="utf-8")) if coverage_path.exists() else {}
    retry_queue_path = ROOT / "reports" / "retry_queue_review.json"
    retry_queue = json.loads(retry_queue_path.read_text(encoding="utf-8")) if retry_queue_path.exists() else {}
    retry_probe_path = ROOT / "reports" / "retry_queue_proxy_probe.json"
    retry_probe = json.loads(retry_probe_path.read_text(encoding="utf-8")) if retry_probe_path.exists() else {}
    retry_transcript_path = ROOT / "reports" / "retry_transcript_fallbacks.json"
    retry_transcript = json.loads(retry_transcript_path.read_text(encoding="utf-8")) if retry_transcript_path.exists() else {}
    retry_normalized_path = ROOT / "reports" / "retry_transcript_fallback_normalization.json"
    retry_normalized = json.loads(retry_normalized_path.read_text(encoding="utf-8")) if retry_normalized_path.exists() else {}
    retry_conversation_path = ROOT / "reports" / "retry_transcript_conversation_progress.json"
    retry_conversations = json.loads(retry_conversation_path.read_text(encoding="utf-8")) if retry_conversation_path.exists() else {}
    retry_conversation_qa_path = ROOT / "reports" / "retry_transcript_conversation_qa.json"
    retry_conversation_qa = json.loads(retry_conversation_qa_path.read_text(encoding="utf-8")) if retry_conversation_qa_path.exists() else {}
    readiness_path = ROOT / "reports" / "meow_v02_data_readiness_interim.json"
    readiness = json.loads(readiness_path.read_text(encoding="utf-8")) if readiness_path.exists() else {}
    identity_queue_path = ROOT / "reports" / "identity_mapping_queue.json"
    identity_queue = json.loads(identity_queue_path.read_text(encoding="utf-8")) if identity_queue_path.exists() else {}
    family_bank_path = ROOT / "speaker_refs" / "reference_bank_neuro_family.json"
    family_bank = json.loads(family_bank_path.read_text(encoding="utf-8")) if family_bank_path.exists() else {}
    family_mapping_path = ROOT / "reports" / "identity_mapping_family_proxy.json"
    family_mapping = json.loads(family_mapping_path.read_text(encoding="utf-8")) if family_mapping_path.exists() else {}
    family_conversation_path = ROOT / "reports" / "family_mapped_conversation_progress.json"
    family_conversations = json.loads(family_conversation_path.read_text(encoding="utf-8")) if family_conversation_path.exists() else {}
    family_qa_path = ROOT / "reports" / "family_mapped_conversation_qa.json"
    family_qa = json.loads(family_qa_path.read_text(encoding="utf-8")) if family_qa_path.exists() else {}
    family_candidates_path = ROOT / "reports" / "family_training_candidate_progress.json"
    family_candidates = json.loads(family_candidates_path.read_text(encoding="utf-8")) if family_candidates_path.exists() else {}
    family_candidates_qa_path = ROOT / "reports" / "family_training_candidate_qa.json"
    family_candidates_qa = json.loads(family_candidates_qa_path.read_text(encoding="utf-8")) if family_candidates_qa_path.exists() else {}
    family_ecapa_path = ROOT / "reports" / "speaker_reference_neuro_family_ecapa_open_set.json"
    family_ecapa = json.loads(family_ecapa_path.read_text(encoding="utf-8")) if family_ecapa_path.exists() else {}
    family_forensic_path = ROOT / "reports" / "family_mapped_surface_forensic_stats.json"
    family_forensic = json.loads(family_forensic_path.read_text(encoding="utf-8")) if family_forensic_path.exists() else {}
    family_dedupe_path = ROOT / "reports" / "family_candidate_dedupe_audit.json"
    family_dedupe = json.loads(family_dedupe_path.read_text(encoding="utf-8")) if family_dedupe_path.exists() else {}
    family_coverage_path = ROOT / "reports" / "family_coverage_audit.json"
    family_coverage = json.loads(family_coverage_path.read_text(encoding="utf-8")) if family_coverage_path.exists() else {}
    family_hard_negative_path = ROOT / "reports" / "neuro_family_hard_negative_quarantine.json"
    family_hard_negative = json.loads(family_hard_negative_path.read_text(encoding="utf-8")) if family_hard_negative_path.exists() else {}
    family_gold_path = ROOT / "speaker_refs" / "gold_validation_neuro_family.json"
    family_gold = json.loads(family_gold_path.read_text(encoding="utf-8")) if family_gold_path.exists() else {}
    negative_review_path = ROOT / "reports" / "open_set_negative_review_manifest.json"
    negative_review = json.loads(negative_review_path.read_text(encoding="utf-8")) if negative_review_path.exists() else {}
    negative_clips_path = ROOT / "reports" / "open_set_negative_review_clips.json"
    negative_clips = json.loads(negative_clips_path.read_text(encoding="utf-8")) if negative_clips_path.exists() else {}
    negative_scores_path = ROOT / "reports" / "open_set_negative_review_scores.json"
    negative_scores = json.loads(negative_scores_path.read_text(encoding="utf-8")) if negative_scores_path.exists() else {}
    auto_anchor_path = ROOT / "reports" / "auto_validation_anchor_report.json"
    auto_anchor = json.loads(auto_anchor_path.read_text(encoding="utf-8")) if auto_anchor_path.exists() else {}
    calibration_bank_path = ROOT / "reports" / "open_set_calibration_bank.json"
    calibration_bank = json.loads(calibration_bank_path.read_text(encoding="utf-8")) if calibration_bank_path.exists() else {}
    calibration_path = ROOT / "reports" / "open_set_auto_anchor_calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8")) if calibration_path.exists() else {}
    calibration_xvector_path = ROOT / "reports" / "open_set_auto_anchor_calibration_xvector.json"
    calibration_xvector = json.loads(calibration_xvector_path.read_text(encoding="utf-8")) if calibration_xvector_path.exists() else {}
    ensemble_calibration_path = ROOT / "reports" / "open_set_ensemble_calibration.json"
    ensemble_calibration = json.loads(ensemble_calibration_path.read_text(encoding="utf-8")) if ensemble_calibration_path.exists() else {}
    identity_closure_dir = ROOT / "speaker_refs" / "identity_closure"
    participant_prior_report_path = identity_closure_dir / "expected_participants_report.json"
    participant_prior_report = json.loads(participant_prior_report_path.read_text(encoding="utf-8")) if participant_prior_report_path.exists() else {}
    guest_negative_report_path = identity_closure_dir / "recurring_guest_negative_bank_report.json"
    guest_negative_report = json.loads(guest_negative_report_path.read_text(encoding="utf-8")) if guest_negative_report_path.exists() else {}
    guest_prototype_report_path = identity_closure_dir / "recurring_guest_negative_prototypes_report.json"
    guest_prototype_report = json.loads(guest_prototype_report_path.read_text(encoding="utf-8")) if guest_prototype_report_path.exists() else {}
    semantic_bank_path = identity_closure_dir / "semantic_identity_bank.json"
    semantic_bank = json.loads(semantic_bank_path.read_text(encoding="utf-8")) if semantic_bank_path.exists() else {}
    fusion_path = ROOT / "reports" / "identity_mapping_multimodal_fusion_proxy.json"
    fusion = json.loads(fusion_path.read_text(encoding="utf-8")) if fusion_path.exists() else {}
    eres2netv2_remap_path = ROOT / "reports" / "identity_mapping_eres2netv2_proxy.json"
    eres2netv2_remap = json.loads(eres2netv2_remap_path.read_text(encoding="utf-8")) if eres2netv2_remap_path.exists() else {}
    model_benchmark_path = ROOT / "reports" / "speaker_model_benchmark_identity_closure.json"
    model_benchmark = json.loads(model_benchmark_path.read_text(encoding="utf-8")) if model_benchmark_path.exists() else {}
    fusion_candidates_path = ROOT / "reports" / "family_training_candidate_fusion_progress.json"
    fusion_candidates = json.loads(fusion_candidates_path.read_text(encoding="utf-8")) if fusion_candidates_path.exists() else {}
    chat_tts_stress_path = identity_closure_dir / "chat_tts_stress_test_report.json"
    chat_tts_stress = json.loads(chat_tts_stress_path.read_text(encoding="utf-8")) if chat_tts_stress_path.exists() else {}
    chat_tts_rejection_path = ROOT / "reports" / "chat_tts_rejection_layer.json"
    chat_tts_rejection = json.loads(chat_tts_rejection_path.read_text(encoding="utf-8")) if chat_tts_rejection_path.exists() else {}
    hard_negative_bank_path = identity_closure_dir / "hard_negative_validation_bank_report.json"
    hard_negative_bank = json.loads(hard_negative_bank_path.read_text(encoding="utf-8")) if hard_negative_bank_path.exists() else {}
    fusion_audio_audit_path = ROOT / "reports" / "identity_fusion_audio_consensus_audit.json"
    fusion_audio_audit = json.loads(fusion_audio_audit_path.read_text(encoding="utf-8")) if fusion_audio_audit_path.exists() else {}
    payload = {
        "schema_version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "videos_discovered": len(rows),
        "videos_minimal_pipeline_complete": minimal_complete,
        "videos_full_pipeline_complete": full_complete,
        "transcript_available_rows": sum(1 for r in rows if r.get("transcript_available")),
        "status_counts": dict(sorted(statuses.items())),
        "disk_free_bytes": shutil.disk_usage(ROOT).free,
        "quality_gates": {
            "state": "CONTINUE_WORKING",
            "human_gold_is_promotion_blocker": False,
            "auto_trusted_anchor_calibration_enabled": True,
            "auto_trusted_anchor_policy": "Source-disjoint multi-evidence AUTO_TRUSTED anchors are valid for calibration and validation; HUMAN_VERIFIED is optional higher-evidence provenance, not a prerequisite.",
            "reference_bank_status": family_bank.get("status", reference_bank.get("status", "missing")),
            "reference_candidate_counts": {
                "NEURO_FAMILY": family_bank.get("mapping_ready_clip_count", 0),
                "VEDAL": reference_bank.get("by_identity", {}).get("VEDAL", 0),
            },
            "legacy_subtype_reference_candidate_counts": reference_bank.get("by_identity", {}),
            "trusted_clip_count": reference_bank.get("trusted_clip_count", 0),
            "natural_turn_source_count": len(natural_turns.get("completed", [])),
            "natural_conversation_window_count": natural_conversations.get("window_count", 0),
            "transcript_audit_status": contamination.get("status", "missing"),
            "transcript_flagged_source_count": contamination.get("flagged_source_count", 0),
            "twitchtranscript_status": twitch.get("status", "missing"),
            "twitchtranscript_source_count": twitch.get("source_count", 0),
            "twitchtranscript_window_count": twitch.get("window_count", 0),
            "twitchtranscript_training_candidate_count": twitch.get("training_candidate_count", 0),
            "conversation_artifact_qa_status": conversation_qa.get("status", "missing"),
            "conversation_artifact_qa_error_count": sum(x.get("error_count", 0) for x in conversation_qa.get("layers", [])),
            "reference_human_review_status": review_queue.get("status", "missing"),
            "reference_human_review_candidate_count": review_queue.get("candidate_count", 0),
            "reference_human_reviewed_count": review_queue.get("reviewed_count", 0),
            "natural_turn_quality_status": natural_quality.get("status", "missing"),
            "natural_turn_quality_flagged_source_count": natural_quality.get("flagged_source_count", 0),
            "duplicate_group_review_status": duplicate_review.get("status", "missing"),
            "duplicate_group_count": duplicate_review.get("group_count", 0),
            "surface_feature_annotation_status": surface_features.get("status", "missing"),
            "surface_feature_record_count": surface_features.get("record_count", 0),
            "anonymous_coverage_status": coverage.get("status", "missing"),
            "anonymous_coverage_gap_count": len(coverage.get("coverage_gaps", [])),
            "retry_queue_status": retry_queue.get("status", "missing"),
            "retryable_queue_count": retry_queue.get("retryable_count", 0),
            "retry_queue_proxy_probe_status": retry_probe.get("status", "missing"),
            "retry_queue_proxy_probe_count": retry_probe.get("probe_count", 0),
            "retry_queue_proxy_probe_status_counts": retry_probe.get("status_counts", {}),
            "retry_queue_proxy_transcript_fallback_count": sum(1 for item in retry_probe.get("results", []) if item.get("transcript_fallback_available")),
            "retry_transcript_fallback_status": retry_transcript.get("status", "missing"),
            "retry_transcript_fallback_attempt_count": retry_transcript.get("attempt_count", 0),
            "retry_transcript_fallback_success_count": retry_transcript.get("success_count", 0),
            "retry_transcript_normalization_status": retry_normalized.get("status", "missing"),
            "retry_transcript_normalized_segment_count": retry_normalized.get("segment_count", 0),
            "retry_transcript_normalized_source_count": retry_normalized.get("source_count", 0),
            "retry_transcript_conversation_status": retry_conversations.get("status", "missing"),
            "retry_transcript_conversation_window_count": retry_conversations.get("window_count", 0),
            "retry_transcript_conversation_qa_status": retry_conversation_qa.get("status", "missing"),
            "retry_transcript_conversation_qa_error_count": retry_conversation_qa.get("error_count", 0),
            "meow_v02_readiness_status": readiness.get("readiness", "missing"),
            "identity_mapping_queue_status": identity_queue.get("status", "missing"),
            "identity_mapping_pending_cluster_count": identity_queue.get("cluster_count", 0),
            "identity_mapping_mapped_cluster_count": identity_queue.get("mapped_cluster_count", 0),
            "identity_mapping_family_proxy_cluster_count": family_mapping.get("cluster_count", 0),
            "identity_mapping_family_proxy_source_count": family_mapping.get("source_count", 0),
            "neuro_family_bank_status": family_bank.get("status", "missing"),
            "neuro_family_mapping_ready_clip_count": family_bank.get("mapping_ready_clip_count", 0),
            "neuro_family_gold_validated_trusted_count": family_bank.get("gold_validated_trusted_count", 0),
            "neuro_family_mapping_status": family_mapping.get("status", "pending"),
            "neuro_family_mapped_cluster_count": family_mapping.get("identity_counts", {}).get("NEURO_FAMILY", 0),
            "vedal_mapped_cluster_count": family_mapping.get("identity_counts", {}).get("VEDAL", 0),
            "family_mapped_conversation_status": family_conversations.get("status", "pending"),
            "family_mapped_conversation_count": family_conversations.get("record_count", 0),
            "family_mapped_conversation_qa_status": family_qa.get("status", "pending"),
            "family_mapped_conversation_qa_error_count": family_qa.get("error_count", 0),
            "family_training_candidate_status": family_candidates.get("status", "pending"),
            "family_s_a_review_candidate_count": family_candidates.get("s_a_review_candidate_count", 0),
            "family_training_candidate_qa_status": family_candidates_qa.get("status", "pending"),
            "family_open_set_ecapa_vedal_false_positive_count": family_ecapa.get("selected_vedal_false_positive_count", 0),
            "family_forensic_status": family_forensic.get("status", "pending"),
            "family_forensic_turn_count": family_forensic.get("record_count", 0),
            "family_candidate_dedupe_status": family_dedupe.get("status", "pending"),
            "family_candidate_exact_duplicate_window_count": family_dedupe.get("exact_duplicate_window_count", 0),
            "family_candidate_overlap_membership_count": family_dedupe.get("overlapping_segment_membership_count", 0),
            "family_coverage_status": family_coverage.get("status", "pending"),
            "family_coverage_window_count": family_coverage.get("window_count", 0),
            "family_hard_negative_quarantine_status": family_hard_negative.get("status", "pending"),
            "family_hard_negative_quarantined_cluster_count": family_hard_negative.get("changed_family_to_unknown_count", 0),
            "family_gold_validation_status": family_gold.get("status", "missing"),
            "family_gold_validation_candidate_count": sum(family_gold.get("population_counts", {}).values()),
            "family_gold_labeled_count": family_gold.get("gold_labeled_count", 0),
            "open_set_negative_review_status": negative_review.get("status", "missing"),
            "open_set_negative_review_candidate_count": negative_review.get("candidate_count", 0),
            "open_set_negative_review_missing_buckets": negative_review.get("missing_buckets", []),
            "open_set_negative_review_clip_status": negative_clips.get("status", "missing"),
            "open_set_negative_review_clip_count": negative_clips.get("clip_count", 0),
            "open_set_negative_gold_labeled_count": negative_clips.get("gold_label_count", 0),
            "open_set_negative_score_triage_status": negative_scores.get("status", "missing"),
            "open_set_negative_score_triage_clip_count": negative_scores.get("negative_review_clip_count", 0),
            "open_set_negative_score_triage_bucket_summary": negative_scores.get("bucket_summary", {}),
            "auto_validation_anchor_status": auto_anchor.get("status", "missing"),
            "auto_validation_anchor_count": auto_anchor.get("anchor_count", 0),
            "auto_validation_anchor_label_counts": auto_anchor.get("anchor_label_counts", {}),
            "auto_validation_anchor_source_counts": auto_anchor.get("anchor_source_counts", {}),
            "open_set_calibration_bank_status": calibration_bank.get("status", "missing"),
            "open_set_calibration_bank_label_counts": calibration_bank.get("label_counts", {}),
            "open_set_calibration_excluded_confounds_count": calibration_bank.get("excluded_confounds_count", 0),
            "open_set_ecapa_calibration_status": calibration.get("status", "missing"),
            "open_set_ecapa_selected_operating_point": calibration.get("selected_operating_point", {}),
            "open_set_xvector_calibration_status": calibration_xvector.get("status", "missing"),
            "open_set_xvector_selected_operating_point": calibration_xvector.get("selected_operating_point", {}),
            "open_set_ensemble_calibration_status": ensemble_calibration.get("status", "missing"),
            "open_set_ensemble_calibration_operating_point": {
                "ecapa_margin_threshold": ensemble_calibration.get("ecapa_margin_threshold"),
                "xvector_margin_threshold": ensemble_calibration.get("xvector_margin_threshold"),
                "family_recall_proxy": ensemble_calibration.get("family_recall_proxy"),
                "nontarget_false_positive_rate": ensemble_calibration.get("nontarget_ensemble_false_positive_rate"),
            },
            "identity_closure_participant_prior_status": participant_prior_report.get("status", "missing"),
            "identity_closure_participant_prior_source_count": participant_prior_report.get("source_count", 0),
            "identity_closure_guest_negative_status": guest_negative_report.get("status", "missing"),
            "identity_closure_guest_negative_candidate_count": guest_negative_report.get("candidate_count", 0),
            "identity_closure_guest_negative_source_counts": guest_negative_report.get("candidate_source_counts", {}),
            "identity_closure_guest_prototype_status": guest_prototype_report.get("status", "missing"),
            "identity_closure_guest_prototype_count": guest_prototype_report.get("prototype_count", 0),
            "identity_closure_guest_prototype_promotion_decision": guest_prototype_report.get("promotion_decision", "DO_NOT_PROMOTE"),
            "identity_closure_semantic_bank_status": semantic_bank.get("status", "missing"),
            "identity_closure_semantic_bank_record_count": len(semantic_bank.get("records", [])),
            "identity_closure_fusion_status": fusion.get("status", "missing"),
            "identity_closure_fusion_cluster_count": fusion.get("cluster_count", 0),
            "identity_closure_fusion_identity_counts": fusion.get("identity_counts", {}),
            "identity_closure_eres2netv2_remap_status": eres2netv2_remap.get("status", "missing"),
            "identity_closure_eres2netv2_remap_model": eres2netv2_remap.get("model"),
            "identity_closure_eres2netv2_remap_cluster_count": eres2netv2_remap.get("query_cluster_count", 0),
            "identity_closure_eres2netv2_remap_identity_counts": eres2netv2_remap.get("identity_counts", {}),
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
            "identity_closure_fusion_candidate_count": fusion_candidates.get("s_a_review_candidate_count", 0),
            "identity_closure_fusion_training_candidate_count": fusion_candidates.get("training_candidate_count", 0),
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
        },
        "notes": [
            "A video is not considered full-pipeline complete until subtitles/ASR, diarization, speaker mapping, conversation reconstruction, and quality metrics exist.",
            "This checkpoint is intentionally conservative and does not infer speaker identity from titles.",
        ],
    }
    write_json(ROOT / "checkpoints" / f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", payload)
    write_json(ROOT / "checkpoints" / "latest.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
