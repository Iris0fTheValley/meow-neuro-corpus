from __future__ import annotations

"""Audit current NEURO_FAMILY fusion clusters without changing labels."""

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone

from manifest_tools import ROOT, read_jsonl, write_json


VERSION = "family-cluster-audit-2026-09-12-v1"


def main() -> None:
    closure = ROOT / "speaker_refs" / "identity_closure"
    fusion_rows = [row for row in read_jsonl(ROOT / "identity_results" / "identity_mapping_multimodal_fusion_proxy.jsonl") if str(row.get("identity", "")).startswith("NEURO_FAMILY")]
    eres = {f"{row.get('source_id')}:{row.get('cluster')}": row for row in read_jsonl(ROOT / "identity_results" / "identity_mapping_eres2netv2_proxy.jsonl")}
    family_proxy = {f"{row.get('source_id')}:{row.get('cluster')}": row for row in read_jsonl(ROOT / "identity_results" / "identity_mapping_family_proxy.jsonl")}
    tts = {str(row.get("record_id")): row for row in read_jsonl(ROOT / "identity_results" / "chat_tts_rejection_scores.jsonl")}
    priors = {str(row.get("source_id")): row for row in read_jsonl(closure / "expected_participants.jsonl")}
    recurring_guests = {f"{row.get('source_id')}:{row.get('cluster')}": row for row in read_jsonl(closure / "recurring_guest_negative_bank.jsonl")}
    timelines = {}
    for path in (ROOT / "unique_timelines").glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        by_cluster = defaultdict(list)
        for turn in payload.get("turns") or []:
            by_cluster[str(turn.get("speaker") or "UNKNOWN")].append(turn)
        timelines[path.stem] = by_cluster
    challenge_buckets = defaultdict(set)
    for path in [ROOT / "speaker_refs" / "open_set_challenge_set.jsonl", ROOT / "speaker_refs" / "open_set_chat_tts_challenge.jsonl"]:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            challenge_buckets[f"{row.get('source_id')}:{row.get('cluster')}"] .add(str(row.get("metadata_bucket") or "chat_tts"))

    output = []
    for row in fusion_rows:
        source_id = str(row.get("source_id")); cluster = str(row.get("cluster")); key = f"{source_id}:{cluster}"
        eres_row = eres.get(key, {}); proxy = family_proxy.get(key, {}); tts_row = tts.get(key, {})
        prior = priors.get(source_id, {}); turns = timelines.get(source_id, {}).get(cluster, [])
        timestamps = [(float((turn.get("timestamp") or {}).get("start") or 0), float((turn.get("timestamp") or {}).get("end") or 0)) for turn in turns]
        timestamps = [(start, end) for start, end in timestamps if end > start]
        named_guests = [label for label in (prior.get("explicit_participant_labels") or []) if label not in {"NEURO_FAMILY", "VEDAL"}]
        guest_similarity = float(eres_row.get("best_guest_score") or -1.0)
        risk_flags = []
        if tts_row.get("rejection_decision") == "REJECT_NON_TARGET_TTS":
            risk_flags.append("strong_chat_tts_reject")
        elif tts_row.get("rejection_decision") == "QUARANTINE_REVIEW":
            risk_flags.append("chat_tts_quarantine")
        if guest_similarity >= 0.70:
            risk_flags.append("high_named_guest_similarity_review")
        if named_guests:
            risk_flags.append("explicit_guest_participant_prior")
        if prior.get("explicit_vedal_participant") or "VEDAL" in (prior.get("explicit_participant_labels") or []):
            risk_flags.append("explicit_vedal_participant_prior")
        if challenge_buckets.get(key):
            risk_flags.extend(f"challenge_bucket:{bucket}" for bucket in sorted(challenge_buckets[key]))
        if recurring_guests.get(key):
            risk_flags.append("recurring_guest_bank_conflict")
        output.append({"record_id": key, "source_id": source_id, "cluster": cluster, "fusion_identity": row.get("identity"), "fusion_confidence": row.get("identity_confidence"), "eres_family_score": eres_row.get("family_score"), "eres_family_margin": eres_row.get("family_margin"), "eres_best_guest_label": eres_row.get("best_guest_label"), "eres_best_guest_score": eres_row.get("best_guest_score"), "eres_segment_count": eres_row.get("segment_count"), "eres_segment_vote_rate": eres_row.get("segment_vote_rate_at_threshold"), "chat_tts_rejection": tts_row, "semantic_evidence": row.get("semantic_evidence"), "role_evidence": row.get("role_evidence"), "source_prior": row.get("source_prior"), "turn_count": len(turns), "temporal_span_seconds": round(max((end for _, end in timestamps), default=0) - min((start for start, _ in timestamps), default=0), 3), "challenge_buckets": sorted(challenge_buckets.get(key, set())), "risk_flags": sorted(set(risk_flags)), "provenance": {"version": VERSION, "gold_validated": False, "training_candidate": False, "source_disjoint_validation": True}})

    report = {"schema_version": "0.1.0", "created_at": datetime.now(timezone.utc).isoformat(), "status": "NEURO_FAMILY_148_CLUSTER_AUDIT_COMPLETE_NO_MUTATION", "version": VERSION, "cluster_count": len(output), "identity_counts": dict(Counter(row["fusion_identity"] for row in output)), "risk_flag_counts": dict(Counter(flag for row in output for flag in row["risk_flags"])), "high_guest_similarity_count": sum("high_named_guest_similarity_review" in row["risk_flags"] for row in output), "chat_tts_reject_count": sum("strong_chat_tts_reject" in row["risk_flags"] for row in output), "chat_tts_quarantine_count": sum("chat_tts_quarantine" in row["risk_flags"] for row in output), "explicit_guest_prior_count": sum("explicit_guest_participant_prior" in row["risk_flags"] for row in output), "explicit_vedal_prior_count": sum("explicit_vedal_participant_prior" in row["risk_flags"] for row in output), "challenge_bucket_counts": dict(Counter(bucket for row in output for bucket in row["challenge_buckets"])), "priority_review_clusters": sorted(output, key=lambda row: (len(row["risk_flags"]), float(row.get("eres_best_guest_score") or -1.0)), reverse=True)[:100], "output": "reports/neuro_family_148_cluster_audit.jsonl", "policy": "Audit only. No cluster is promoted or downgraded here. Semantic, role, and metadata evidence remain auxiliary; any future reject must be independently justified and provenance-preserving."}
    out_path = ROOT / "reports" / "neuro_family_148_cluster_audit.jsonl"
    out_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output), encoding="utf-8")
    write_json(ROOT / "reports" / "neuro_family_148_cluster_audit.json", report)
    print(json.dumps({"status": report["status"], "cluster_count": len(output), "risk_flag_counts": report["risk_flag_counts"], "priority_review_count": len(report["priority_review_clusters"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
