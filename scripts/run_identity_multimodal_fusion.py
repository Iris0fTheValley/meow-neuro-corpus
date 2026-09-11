from __future__ import annotations

"""Fuse cluster-level evidence into a conservative, versioned proxy mapping.

This is a remapping/regrading pass.  It never overwrites the existing proxy
mapping and never sets training_candidate=true.  Any strong evidence conflict
resolves to UNKNOWN rather than a forced target label.
"""

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone

from manifest_tools import ROOT, read_jsonl, safe_name, write_json


VERSION = "multimodal-fusion-proxy-2026-09-12-v2-chat-tts-rejection"
SEMANTIC_CONFIRMATION_THRESHOLD = 0.35
NUMERIC_STYLE = ("word_count", "content_word_count", "first_person_count", "second_person_count", "question_count", "hedge_count", "assertion_count", "repair_false_start_count", "address_count", "reaction_count")


def style(text: str) -> dict[str, float]:
    low = str(text or "").casefold()
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", low)
    return {
        "word_count": float(len(words)), "content_word_count": float(len([w for w in words if w not in {"a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "he", "her", "i", "if", "in", "is", "it", "me", "my", "of", "on", "or", "our", "so", "that", "the", "their", "there", "they", "this", "to", "us", "was", "we", "were", "will", "with", "you", "your"}])) ,
        "first_person_count": float(len(re.findall(r"\b(?:i|i'm|i've|i'll|me|my|mine|we|us|our)\b", low))),
        "second_person_count": float(len(re.findall(r"\b(?:you|your|yours|u)\b", low))),
        "question_count": float(str(text or "").count("?")),
        "hedge_count": float(len(re.findall(r"\b(?:maybe|perhaps|probably|i think|i guess|kind of|sort of|might|could)\b", low))),
        "assertion_count": float(len(re.findall(r"\b(?:obviously|definitely|clearly|must|will|is|are)\b", low))),
        "repair_false_start_count": float(len(re.findall(r"\b(?:i mean|sorry|wait|actually|or rather|uh+|um+)\b", low))),
        "address_count": float(len(re.findall(r"\b(?:chat|guys|everyone|people|friend|buddy|sir|ma'am)\b", low))),
        "reaction_count": float(len(re.findall(r"\b(?:oh|wow|what|why|no|yes|yeah|haha|lol|yay|ugh|huh)\b", low))),
    }


def load_timelines() -> dict[str, dict[str, list[dict]]]:
    result = {}
    for path in (ROOT / "unique_timelines").glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        by_cluster = defaultdict(list)
        for turn in data.get("turns") or []:
            by_cluster[str(turn.get("speaker") or "UNKNOWN")].append(turn)
        result[str(data.get("source_id") or path.stem)] = by_cluster
    return result


def aggregate_style(turns: list[dict]) -> dict:
    values = [style(turn.get("text") or "") for turn in turns if str(turn.get("text") or "").strip()]
    if not values:
        result = {key: 0.0 for key in NUMERIC_STYLE}
        result["turn_count"] = 0
        return result
    result = {key: round(sum(value[key] for value in values) / len(values), 6) for key in NUMERIC_STYLE}
    result["turn_count"] = len(values)
    return result


def style_similarity(features: dict, family_means: dict) -> float | None:
    if not features.get("turn_count") or not family_means:
        return None
    differences = []
    for key in NUMERIC_STYLE:
        scale = max(1.0, float(family_means.get(key) or 0.0))
        differences.append(abs(float(features.get(key) or 0.0) - float(family_means.get(key) or 0.0)) / scale)
    return round(1.0 / (1.0 + sum(differences) / len(differences)), 6)


def main() -> None:
    closure = ROOT / "speaker_refs" / "identity_closure"
    mapping = list(read_jsonl(ROOT / "identity_results" / "identity_mapping_family_proxy.jsonl"))
    eres_mapping = {f"{row.get('source_id')}:{row.get('cluster')}": row for row in read_jsonl(ROOT / "identity_results" / "identity_mapping_eres2netv2_proxy.jsonl")}
    tts_rejection = {str(row.get("record_id")): row for row in read_jsonl(ROOT / "identity_results" / "chat_tts_rejection_scores.jsonl")}
    tts_cluster_keys = {f"{row.get('source_id')}:{row.get('cluster')}" for row in read_jsonl(ROOT / "speaker_refs" / "open_set_chat_tts_challenge.jsonl")}
    priors = {str(row.get("source_id")): row for row in read_jsonl(closure / "expected_participants.jsonl")}
    guest_candidates = {f"{row.get('source_id')}:{row.get('cluster')}": row for row in read_jsonl(closure / "recurring_guest_negative_bank.jsonl")}
    semantic = json.loads((closure / "semantic_identity_bank.json").read_text(encoding="utf-8"))
    family_means = (semantic.get("labels") or {}).get("NEURO_FAMILY", {}).get("feature_means", {})
    timelines = load_timelines()
    output = []
    counts = Counter()
    conflicts = Counter()
    for row in mapping:
        source_id = str(row.get("source_id")); cluster = str(row.get("cluster")); key = f"{source_id}:{cluster}"
        prior = priors.get(source_id, {}); evidence = row.get("evidence") or {}
        margin = float(row.get("family_vs_nontarget_margin") or evidence.get("family_vs_nontarget_margin") or -1.0)
        xmargin = evidence.get("xvector_family_vs_nontarget_margin")
        current = str(row.get("identity") or "UNKNOWN")
        eres = eres_mapping.get(key, {})
        eres_identity = str(eres.get("identity") or "UNKNOWN")
        eres_margin = float(eres.get("family_margin") or -1.0)
        cluster_turns = timelines.get(source_id, {}).get(cluster, [])
        features = aggregate_style(cluster_turns)
        semantic_score = style_similarity(features, family_means)
        guest_conflict = key in guest_candidates
        chat_tts_conflict = key in tts_cluster_keys
        explicit_family = "NEURO_FAMILY" in (prior.get("explicit_participant_labels") or [])
        explicit_vedal = "VEDAL" in (prior.get("explicit_participant_labels") or [])
        named_guests = [label for label in (prior.get("explicit_participant_labels") or []) if label not in {"NEURO_FAMILY", "VEDAL"}]
        guest_prior_ambiguity = bool(named_guests and explicit_family)
        audio_family = current == "NEURO_FAMILY" and margin >= 0.135 and (xmargin is None or float(xmargin) >= 0.0)
        eres_family = eres_identity == "NEURO_FAMILY"
        audio_consensus_family = audio_family and eres_family
        audio_vedal = current == "VEDAL" and margin <= -0.135 and (xmargin is None or float(xmargin) <= 0.0)
        tts_evidence = tts_rejection.get(key, {})
        tts_decision = str(tts_evidence.get("rejection_decision") or "NONE")
        if tts_decision == "REJECT_NON_TARGET_TTS":
            identity = "NON_TARGET_TTS"; confidence = "high"; reason = "independent chat-TTS rejection layer has strong acoustic prototype match plus explicit TTS event evidence; reject-only layer cannot promote family"
            conflicts["chat_tts_strong_reject"] += 1
        elif chat_tts_conflict or tts_decision == "QUARANTINE_REVIEW":
            identity = "UNKNOWN"; confidence = "unknown"; reason = "cluster overlaps an open-set chat-TTS challenge interval; quarantined from family promotion pending direct TTS/non-target evidence"
            conflicts["chat_tts_cluster_quarantine"] += 1
        elif guest_conflict:
            identity = "NON_TARGET_GUEST"; confidence = "medium"; reason = "recurring guest candidate conflict; cluster kept out of family promotion"
            conflicts["guest_candidate_conflict"] += 1
        elif eres_identity == "NON_TARGET_GUEST":
            identity = "NON_TARGET_GUEST"; confidence = "medium"; reason = "independent ERes2NetV2 guest prototype veto; fusion family evidence rejected"
            conflicts["eres_guest_veto"] += 1
        elif eres_identity == "NON_TARGET_KNOWN":
            identity = "NON_TARGET_KNOWN"; confidence = "medium"; reason = "independent ERes2NetV2 non-target veto; fusion family evidence rejected"
            conflicts["eres_known_nontarget_veto"] += 1
        elif guest_prior_ambiguity and audio_consensus_family:
            identity = "UNKNOWN"; confidence = "unknown"; reason = "explicit family and guest participant priors coexist; audio supports family but cluster-level guest assignment is unresolved, so high-precision fusion rejects promotion"
            conflicts["explicit_guest_family_ambiguity"] += 1
        elif audio_consensus_family and explicit_family and not explicit_vedal and semantic_score is not None and semantic_score >= SEMANTIC_CONFIRMATION_THRESHOLD:
            identity = "NEURO_FAMILY_HIGH"; confidence = "high"; reason = "leave-source-out ECAPA/X-vector plus independent ERes2NetV2 consensus + explicit family participant prior + cluster style consistency"
        elif audio_consensus_family and not explicit_vedal:
            identity = "NEURO_FAMILY_MEDIUM"; confidence = "medium"; reason = "leave-source-out ECAPA/X-vector plus independent ERes2NetV2 consensus; participant prior or semantic role evidence incomplete"
        elif audio_family and not eres_family:
            identity = "UNKNOWN"; confidence = "unknown"; reason = "fusion family evidence disagrees with independent ERes2NetV2; semantic/metadata evidence is not allowed to promote"
            conflicts["eres_family_disagreement"] += 1
        elif audio_vedal and explicit_vedal:
            identity = "NON_TARGET_KNOWN"; confidence = "medium"; reason = "Vedal proxy margin + explicit Vedal participant prior; not human gold"
        elif current == "VEDAL" and explicit_vedal:
            identity = "NON_TARGET_KNOWN"; confidence = "low"; reason = "Vedal proxy mapping with weak margin; retained as non-target review recommendation"
        else:
            identity = "UNKNOWN"; confidence = "unknown"; reason = "audio/semantic/source evidence conflict or insufficient independent evidence"
        counts[identity] += 1
        output.append({
            "mapping_id": row.get("mapping_id"), "source_id": source_id, "cluster": cluster,
            "identity": identity, "identity_confidence": confidence, "training_candidate": False,
            "fusion_status": "PROXY_RECOMMENDATION_NOT_GOLD", "reason": reason,
            "audio_evidence": {"proxy_identity": current, "family_margin": margin, "xvector_margin": xmargin, "ensemble_gate": audio_family or audio_vedal, "eres2netv2_identity": eres_identity, "eres2netv2_family_margin": eres_margin, "eres2netv2_consensus_family": eres_family},
            "semantic_evidence": {"cluster_turn_count": features.get("turn_count"), "style_similarity_to_family": semantic_score, "bank_version": semantic.get("version"), "confirmation_threshold": SEMANTIC_CONFIRMATION_THRESHOLD, "threshold_type": "heuristic_auxiliary_confirmation", "cannot_promote_without_independent_audio": True},
            "role_evidence": {"explicit_family_participant": explicit_family, "explicit_vedal_participant": explicit_vedal, "explicit_guest_labels": named_guests},
            "source_prior": {"possible": prior.get("possible", []), "confidence": prior.get("confidence"), "version": VERSION},
            "negative_guest_evidence": guest_candidates.get(key),
            "chat_tts_rejection_evidence": tts_evidence,
            "provenance": {"source_disjoint_validation": True, "gold_validated": False, "neuro_evil_subtype_separation_required": False, "promotion_eligible": False},
        })
    out_path = ROOT / "identity_results" / "identity_mapping_multimodal_fusion_proxy.jsonl"
    out_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output), encoding="utf-8")
    report = {"schema_version": "0.1.0", "updated_at": datetime.now(timezone.utc).isoformat(), "status": "MULTIMODAL_FUSION_PROXY_AUDIO_CONSENSUS_GATED", "version": VERSION, "source_count": len({row["source_id"] for row in output}), "cluster_count": len(output), "identity_counts": dict(counts), "conflict_counts": dict(conflicts), "semantic_bank_version": semantic.get("version"), "negative_bank_version": VERSION, "eres2netv2_mapping_version": "identity_mapping_eres2netv2_proxy", "gold_validated": False, "training_candidate_count": 0, "promotion_decision": "DO_NOT_PROMOTE", "output": str(out_path.relative_to(ROOT)), "policy": "Fusion family promotion requires independent ERes2NetV2 audio consensus; semantic, role, and metadata evidence cannot override ERes2NetV2 non-target or UNKNOWN disagreement. Every output remains non-training proxy."}
    write_json(ROOT / "reports" / "identity_mapping_multimodal_fusion_proxy.json", report)
    print(json.dumps({"status": report["status"], "cluster_count": report["cluster_count"], "identity_counts": dict(counts), "training_candidate_count": 0}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
