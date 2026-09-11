from __future__ import annotations

"""Regrade conversation review candidates with the new cluster fusion proxy."""

import json
from collections import Counter
from datetime import datetime, timezone

from manifest_tools import ROOT, read_jsonl, write_json


def main() -> None:
    fusion = {f"{row.get('source_id')}:{row.get('cluster')}": row for row in read_jsonl(ROOT / "identity_results" / "identity_mapping_multimodal_fusion_proxy.jsonl")}
    tts_by_source: dict[str, list[dict]] = {}
    for challenge in read_jsonl(ROOT / "speaker_refs" / "open_set_chat_tts_challenge.jsonl"):
        tts_by_source.setdefault(str(challenge.get("source_id")), []).append(challenge)
    source = ROOT / "conversations" / "natural_conversations_anonymous.jsonl"
    all_output = ROOT / "datasets" / "family_training_candidates_fusion_review.jsonl"
    sa_output = ROOT / "datasets" / "family_s_a_candidates_fusion_review.jsonl"
    all_output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter(); s_a_count = 0; record_count = 0
    with all_output.open("w", encoding="utf-8") as all_handle, sa_output.open("w", encoding="utf-8") as sa_handle:
        for row in read_jsonl(source):
            source_id = str(row.get("source_video_id") or "")
            mapped_turns = []
            identity_counts = Counter()
            suspicious = 0
            chat_tts_overlap = 0
            for turn in row.get("turns") or []:
                mapping = fusion.get(f"{source_id}:{turn.get('speaker')}")
                identity = str(mapping.get("identity") if mapping else "UNKNOWN")
                confidence = str(mapping.get("identity_confidence") if mapping else "unknown")
                suspicious += int(bool(turn.get("suspicious_transcription")))
                timestamp = turn.get("timestamp") or {}
                turn_start = float(timestamp.get("start") or 0.0)
                turn_end = float(timestamp.get("end") or turn_start)
                tts_overlap = any(
                    str(challenge.get("cluster")) == str(turn.get("speaker"))
                    and turn_start < float(challenge.get("end") or 0.0)
                    and float(challenge.get("start") or 0.0) < turn_end
                    for challenge in tts_by_source.get(source_id, [])
                )
                chat_tts_overlap += int(tts_overlap)
                identity_counts[identity] += 1
                mapped_turns.append({**turn, "speaker_identity": identity, "identity_confidence": confidence, "mapping_status": mapping.get("fusion_status") if mapping else "MISSING_FUSION_MAPPING", "fusion_reason": mapping.get("reason") if mapping else "missing cluster mapping", "chat_tts_challenge_overlap": tts_overlap, "hard_negative_provenance": "open_set_chat_tts_challenge:mention_only_unlabeled" if tts_overlap else None})
            total = len(mapped_turns)
            family = identity_counts["NEURO_FAMILY_HIGH"] + identity_counts["NEURO_FAMILY_MEDIUM"]
            unknown = identity_counts["UNKNOWN"]
            non_target = identity_counts["NON_TARGET_GUEST"] + identity_counts["NON_TARGET_KNOWN"]
            family_ratio = family / max(1, total)
            unknown_ratio = unknown / max(1, total)
            suspicious_ratio = suspicious / max(1, total)
            if row.get("window_type") != "4_8_turns":
                grade = "B"; reason = "extended context retained for research; excluded from primary S/A review"
            elif non_target or chat_tts_overlap or family == 0 or unknown_ratio > 0.5 or suspicious_ratio > 0.25:
                grade = "Q"; reason = "known non-target conflict, chat-TTS challenge overlap, no family target, too much unknown identity, or severe transcription risk"
            elif family >= 2 and unknown == 0 and suspicious == 0 and family_ratio >= 0.5:
                grade = "S"; reason = "fusion proxy family-dominant primary review window; final identity validation still required"
            elif family >= 1 and unknown_ratio <= 0.2 and suspicious_ratio <= 0.1 and family_ratio >= 0.4:
                grade = "A"; reason = "fusion proxy family-dominant review window with bounded uncertainty"
            else:
                grade = "B"; reason = "useful family proxy window but identity confidence or coverage is insufficient for S/A review"
            candidate = {**row, "status": "REVIEW_REQUIRED_MULTIMODAL_FUSION_PROXY", "candidate_grade": grade, "training_candidate": False, "candidate_reason": reason, "identity_mapping": "NEURO_FAMILY_HIGH | NEURO_FAMILY_MEDIUM | NON_TARGET_KNOWN | NON_TARGET_GUEST | UNKNOWN", "identity_confidence": "fusion_proxy_pending_final_validation", "speaker_identity_counts": dict(identity_counts), "mapped_turn_count": family + non_target, "unknown_turn_count": unknown, "non_target_turn_count": non_target, "stats": {"turn_count": total, "family_turn_count": family, "family_high_turn_count": identity_counts["NEURO_FAMILY_HIGH"], "family_medium_turn_count": identity_counts["NEURO_FAMILY_MEDIUM"], "vedal_turn_count": identity_counts["NON_TARGET_KNOWN"], "guest_turn_count": identity_counts["NON_TARGET_GUEST"], "unknown_turn_count": unknown, "suspicious_turn_count": suspicious, "chat_tts_challenge_overlap_count": chat_tts_overlap, "family_ratio": round(family_ratio, 4), "unknown_ratio": round(unknown_ratio, 4), "suspicious_ratio": round(suspicious_ratio, 4)}, "turns": mapped_turns}
            all_handle.write(json.dumps(candidate, ensure_ascii=False) + "\n")
            counts[grade] += 1; record_count += 1
            if grade in {"S", "A"}:
                sa_handle.write(json.dumps(candidate, ensure_ascii=False) + "\n"); s_a_count += 1
    report = {"schema_version": "0.1.0", "updated_at": datetime.now(timezone.utc).isoformat(), "status": "S_A_REVIEW_CANDIDATES_AUDIO_CONSENSUS_TTS_QUARANTINED", "input": str(source.relative_to(ROOT)), "all_output": str(all_output.relative_to(ROOT)), "s_a_output": str(sa_output.relative_to(ROOT)), "record_count": record_count, "grade_counts": dict(counts), "s_a_review_candidate_count": s_a_count, "training_candidate_count": 0, "gold_validated": False, "hard_negative_chat_tts_policy": "Any window turn overlapping an open-set chat-TTS challenge interval is quarantined to Q; challenge intervals remain mention-only and unlabeled.", "policy": "Regrading is cluster-level and proxy-only; S/A remain review priority labels and no row is training_candidate until source-disjoint non-target validation and final promotion gates pass."}
    write_json(ROOT / "reports" / "family_training_candidate_fusion_progress.json", report)
    print(json.dumps({"status": report["status"], "record_count": record_count, "grade_counts": dict(counts), "s_a_review_candidate_count": s_a_count, "training_candidate_count": 0}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
