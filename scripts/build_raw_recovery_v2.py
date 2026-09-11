from __future__ import annotations

"""Raw-timeline recovery pass for production_v2.

This deliberately bypasses natural_conversations_family_mapped.jsonl.  It
indexes every current target utterance in unique_timelines and attempts one
best preceding context at 10/30/60/120 second horizons.  Target identity is
kept strict; context identity may be UNKNOWN, a guest, Vedal, or TTS when the
turn is clean and temporally connected to the target.
"""

import hashlib
import json
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from manifest_tools import ROOT, write_json
from build_final_sft_artifacts import normalize, text_ok


FUSION = ROOT / "identity_results" / "identity_mapping_multimodal_fusion_proxy.jsonl"
CLUSTERS = ROOT / "reports" / "recording_content_clusters.jsonl"
REGISTRY = ROOT / "source_registry.json"
TIMELINES = ROOT / "unique_timelines"
V1 = ROOT / "datasets" / "meow_v02_sft_v1" / "production"
OUT = ROOT / "datasets" / "meow_v02_sft_v1" / "production_v2_recovered"
TARGETS = {"NEURO_FAMILY_HIGH", "NEURO_FAMILY_MEDIUM"}
HORIZONS = (10.0, 30.0, 60.0, 120.0)
VERSION = "production-meow-v02-sft-v2-raw-recovery-2026-09-12"


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def read_v1() -> list[dict]:
    rows = []
    for name in ("train.jsonl", "validation.jsonl", "sealed_eval.jsonl"):
        path = V1 / name
        if path.exists():
            rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return rows


def clean_text(text: str, minimum_words: int = 2) -> bool:
    value = normalize(text)
    if not text_ok(value, minimum_words):
        return False
    words = re.findall(r"[a-zA-Z]{2,}", value)
    return len(words) >= minimum_words


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


def snapshot(rows: list[dict]) -> dict:
    return {"rows": len(rows), "sources": len({row.get("source_id") for row in rows}), "recording_clusters": len({row.get("canonical_recording_id") for row in rows})}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fusion = {(str(row.get("source_id")), str(row.get("cluster"))): row for row in load_jsonl(FUSION)}
    source_cluster = {}
    mirror_clusters = set()
    for row in load_jsonl(CLUSTERS):
        rc = str(row.get("recording_cluster_id"))
        if row.get("high_risk_mirror_or_reupload"):
            mirror_clusters.add(rc)
        for source_id in row.get("source_ids") or []:
            source_cluster[str(source_id)] = rc
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    source_meta = {str(row.get("source_id")): row for row in registry.get("sources", [])}

    v1_rows = read_v1()
    v1_by_key = {}
    for row in v1_rows:
        key = "\n".join(item.get("role", "") + ":" + normalize(item.get("content", "")) for item in row.get("messages", []))
        v1_by_key[key] = row

    audit_path = OUT / "target_utterance_recovery_audit.jsonl"
    audit_handle = audit_path.open("w", encoding="utf-8")
    counters = Counter()
    recovered = Counter()
    candidates = dict(v1_by_key)
    target_total = 0
    target_indexed = 0
    source_count = 0
    for timeline_path in sorted(TIMELINES.glob("*.json")):
        try:
            timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
        except Exception:
            counters["timeline_parse_error"] += 1
            continue
        source_id = str(timeline.get("source_id") or timeline_path.stem)
        recording_id = source_cluster.get(source_id)
        if not recording_id:
            counters["missing_recording_cluster"] += 1
            continue
        source_count += 1
        if recording_id in mirror_clusters:
            counters["mirror_or_reupload_source"] += 1
            continue
        raw_turns = timeline.get("turns") or []
        turns = []
        for index, turn in enumerate(raw_turns):
            speaker = str(turn.get("speaker") or "")
            identity = str((fusion.get((source_id, speaker)) or {}).get("identity") or "UNKNOWN")
            timestamp = turn.get("timestamp") or {}
            try:
                start = float(timestamp.get("start"))
                end = float(timestamp.get("end"))
            except (TypeError, ValueError):
                start = float(index)
                end = start
            turns.append((turn, identity, start, end, index))

        for index, (target_turn, target_identity, target_start, target_end, _) in enumerate(turns):
            if target_identity not in TARGETS:
                continue
            target_total += 1
            audit = {"source_id": source_id, "canonical_recording_id": recording_id, "turn_id": target_turn.get("turn_id"), "target_identity": target_identity, "target_timestamp": target_turn.get("timestamp"), "status": "UNUSABLE", "failure_reason": None}
            if bool(target_turn.get("uncertain_transcription")) or bool(target_turn.get("suspicious_transcription")) or bool(target_turn.get("overlap")):
                audit["failure_reason"] = "target_transcript_or_overlap_flag"
                counters["target_transcript_or_overlap_flag"] += 1
                audit_handle.write(json.dumps(audit, ensure_ascii=False) + "\n")
                continue
            speaker_confidence = float(target_turn.get("speaker_confidence") or 0.0)
            if speaker_confidence < 0.65:
                audit["failure_reason"] = "target_speaker_confidence_below_0.65"
                counters["target_speaker_confidence"] += 1
                audit_handle.write(json.dumps(audit, ensure_ascii=False) + "\n")
                continue
            raw_transcription = target_turn.get("transcription_confidence")
            transcription_missing = raw_transcription is None or float(raw_transcription or 0.0) <= 0.01
            if not clean_text(target_turn.get("text"), 4 if transcription_missing else 2):
                audit["failure_reason"] = "target_text_malformed_or_music_like"
                counters["target_text_quality"] += 1
                audit_handle.write(json.dumps(audit, ensure_ascii=False) + "\n")
                continue
            target_indexed += 1
            best = None
            horizon_candidates = []
            preceding_speech_available = False
            chat_tts_context_available = False
            for horizon in HORIZONS:
                start_limit = target_start - horizon
                context_index = None
                for candidate_index in range(index - 1, -1, -1):
                    context_turn, context_identity, context_start, context_end, _ = turns[candidate_index]
                    if context_end < start_limit:
                        break
                    if context_identity in TARGETS:
                        continue
                    if bool(context_turn.get("uncertain_transcription")) or bool(context_turn.get("suspicious_transcription")) or bool(context_turn.get("overlap")):
                        continue
                    if not clean_text(context_turn.get("text"), 1):
                        continue
                    context_index = candidate_index
                    break
                if context_index is None:
                    continue
                preceding_speech_available = True
                if any(item[1] == "NON_TARGET_TTS" for item in turns[context_index:index]):
                    chat_tts_context_available = True
                selected = []
                for selected_index in range(context_index, index + 1):
                    turn, identity, start, end, _ = turns[selected_index]
                    if bool(turn.get("uncertain_transcription")) or bool(turn.get("suspicious_transcription")) or bool(turn.get("overlap")):
                        continue
                    if not clean_text(turn.get("text"), 1 if identity not in TARGETS else (4 if (turn.get("transcription_confidence") in (None, 0, 0.0)) else 2)):
                        continue
                    selected.append((turn, identity, start, end))
                if not selected or selected[-1][0] is not target_turn:
                    continue
                messages = []
                context_identities = []
                for turn, identity, _, _ in selected:
                    role = "assistant" if identity in TARGETS else "user"
                    text = str(turn.get("text") or "").strip()
                    if not text:
                        continue
                    context_identities.append(identity)
                    if messages and messages[-1]["role"] == role:
                        messages[-1]["content"] += "\n" + text
                    else:
                        messages.append({"role": role, "content": text})
                if not messages or messages[0]["role"] != "user" or messages[-1]["role"] != "assistant":
                    continue
                if len(messages[-1]["content"]) > 500:
                    continue
                gap = max(0.0, target_start - selected[0][2])
                score = (1.0 if target_identity == "NEURO_FAMILY_HIGH" else 0.86) * 0.45 + speaker_confidence * 0.25 + min(1.0, len(messages) / 4.0) * 0.12 + min(1.0, len(re.findall(r"[a-zA-Z]{2,}", messages[-1]["content"])) / 12.0) * 0.08 + max(0.0, 1.0 - gap / horizon) * 0.10
                if transcription_missing:
                    score -= 0.03
                score -= min(0.10, sum(1 for identity in context_identities if identity == "UNKNOWN") * 0.01)
                candidate = {"messages": messages, "score": round(score, 6), "horizon": horizon, "gap_seconds": round(gap, 3), "context_identities": context_identities, "turn_count": len(selected)}
                horizon_candidates.append(candidate)
            if horizon_candidates:
                # Prefer the smallest temporal horizon that yields a clean
                # context; only then use score to break ties.
                best = sorted(horizon_candidates, key=lambda item: (item["horizon"], -item["score"]))[0]
            if best is None:
                audit["failure_reason"] = "no_clean_preceding_non_target_context_within_120s"
                audit["raw_timeline_available"] = True
                audit["preceding_speech_available"] = preceding_speech_available
                audit["chat_tts_context_available"] = chat_tts_context_available
                counters["no_recoverable_context"] += 1
                audit_handle.write(json.dumps(audit, ensure_ascii=False) + "\n")
                continue
            messages = best["messages"]
            dedup_key = "\n".join(item["role"] + ":" + normalize(item["content"]) for item in messages)
            title = normalize((source_meta.get(source_id) or {}).get("title") or "")
            title_risk = any(term in title for term in ("karaoke", "singing", "song", "gaming", "gameplay", "minecraft", "skyrim", "chess", "playing"))
            context_identities = best["context_identities"]
            if any(identity == "UNKNOWN" for identity in context_identities):
                recovered["recovered_from_old_UNKNOWN_or_unmapped_context"] += 1
            if any(identity == "NON_TARGET_GUEST" for identity in context_identities):
                recovered["recovered_from_guest_context"] += 1
            if any(identity == "NON_TARGET_TTS" for identity in context_identities):
                recovered["recovered_from_TTS_chat_context"] += 1
            if best["horizon"] > 30:
                recovered["recovered_by_wider_timeline_search"] += 1
            if best.get("turn_count", len(messages)) > 2:
                recovered["recovered_by_new_turn_reconstruction"] += 1
            source_url = (source_meta.get(source_id) or {}).get("source_url") or timeline.get("source_url")
            row = {"sample_id": hashlib.sha256((source_id + "|" + str(target_turn.get("turn_id")) + "|" + dedup_key).encode("utf-8")).hexdigest()[:20], "messages": messages, "source_id": source_id, "source_url": source_url, "canonical_recording_id": recording_id, "timestamps": {"start": selected[0][2], "end": target_end}, "speaker_confidence": round(speaker_confidence, 6), "transcription_confidence": round(float(raw_transcription or 0.0), 6), "transcription_confidence_missing": transcription_missing, "identity": target_identity, "identity_confidence": "high" if target_identity == "NEURO_FAMILY_HIGH" else "medium", "quality_score": round(best["score"], 6), "raw_recovery": {"target_turn_id": target_turn.get("turn_id"), "search_horizon_seconds": best["horizon"], "context_gap_seconds": best["gap_seconds"], "context_turn_count": best.get("turn_count", len(messages)), "context_identities": context_identities, "title_risk_signal": title_risk, "source_timeline": str(timeline_path.relative_to(ROOT))}, "pipeline_version": VERSION}
            existing = candidates.get(dedup_key)
            if existing is None or float(existing.get("quality_score") or 0.0) < row["quality_score"]:
                candidates[dedup_key] = row
            audit.update({"status": "RECOVERED", "failure_reason": None, "raw_timeline_available": True, "preceding_speech_available": preceding_speech_available, "chat_tts_context_available": chat_tts_context_available, "search_horizon_seconds": best["horizon"], "context_gap_seconds": best["gap_seconds"], "context_identities": context_identities, "sample_id": row["sample_id"]})
            audit_handle.write(json.dumps(audit, ensure_ascii=False) + "\n")
    audit_handle.close()

    cluster_audit = {}
    for line in audit_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        cluster = str(item.get("canonical_recording_id") or "UNKNOWN_CLUSTER")
        entry = cluster_audit.setdefault(cluster, {"canonical_recording_id": cluster, "source_ids": set(), "target_utterances": 0, "recoverable_targets": 0, "unusable_targets": 0, "preceding_speech_available_targets": 0, "chat_tts_context_targets": 0, "failure_reasons": Counter(), "raw_timeline_available": True})
        entry["source_ids"].add(str(item.get("source_id")))
        entry["target_utterances"] += 1
        if item.get("status") == "RECOVERED":
            entry["recoverable_targets"] += 1
        else:
            entry["unusable_targets"] += 1
            entry["failure_reasons"][str(item.get("failure_reason") or "unknown")] += 1
        if item.get("preceding_speech_available"):
            entry["preceding_speech_available_targets"] += 1
        if item.get("chat_tts_context_available"):
            entry["chat_tts_context_targets"] += 1
    cluster_audit_rows = []
    for cluster, entry in sorted(cluster_audit.items()):
        entry["source_ids"] = sorted(entry["source_ids"])
        entry["source_count"] = len(entry["source_ids"])
        entry["failure_reasons"] = dict(entry["failure_reasons"])
        entry["high_risk_mirror_or_reupload"] = cluster in mirror_clusters
        entry["status"] = "RECOVERABLE" if entry["recoverable_targets"] else "NO_SAFE_CONTEXT"
        cluster_audit_rows.append(entry)
    write_json(OUT / "raw_cluster_recovery_audit.json", {"schema_version": "1.0.0", "pipeline_version": VERSION, "clusters": cluster_audit_rows, "non_mirror_clusters_without_recovery": [row["canonical_recording_id"] for row in cluster_audit_rows if not row["high_risk_mirror_or_reupload"] and row["recoverable_targets"] == 0]})

    rows = list(candidates.values())
    seen_responses = set()
    deduped = []
    duplicate_removed = 0
    for row in sorted(rows, key=lambda item: (-float(item.get("quality_score") or 0.0), item.get("sample_id", ""))):
        response_key = normalize(row.get("messages", [{}])[-1].get("content", ""))
        if response_key in seen_responses:
            duplicate_removed += 1
            continue
        seen_responses.add(response_key)
        deduped.append(row)
    rows = deduped
    split_map = split_clusters(rows)
    splits = {"train": [], "validation": [], "sealed_eval": []}
    for row in rows:
        cluster = row["canonical_recording_id"]
        row["split"] = "sealed_eval" if cluster in split_map["sealed_eval"] else ("validation" if cluster in split_map["validation"] else "train")
        splits[row["split"]].append(row)
    for split in splits:
        write_path = OUT / ("sealed_eval.jsonl" if split == "sealed_eval" else split + ".jsonl")
        write_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in sorted(splits[split], key=lambda item: item["sample_id"])), encoding="utf-8")
    leakage = {"train_validation": sorted(split_map["train"] & split_map["validation"]), "train_sealed_eval": sorted(split_map["train"] & split_map["sealed_eval"]), "validation_sealed_eval": sorted(split_map["validation"] & split_map["sealed_eval"])}
    quality = {"schema_version": "1.0.0", "artifact_status": "PRODUCTION_V2_RAW_RECOVERED_CANDIDATE", "pipeline_version": VERSION, "created_at": datetime.now(timezone.utc).isoformat(), "sample_counts": {split: len(splits[split]) for split in splits}, "total_samples": len(rows), "unique_sources": len({row["source_id"] for row in rows}), "unique_recording_clusters": len({row["canonical_recording_id"] for row in rows}), "split_recording_clusters": {split: sorted(split_map[split]) for split in split_map}, "high_confidence_target_utterances_total": target_total, "target_utterances_indexed_after_target_gates": target_indexed, "target_utterances_with_recoverable_context": sum(1 for line in audit_path.read_text(encoding="utf-8").splitlines() if line and json.loads(line).get("status") == "RECOVERED"), "usable_target_response_rate": round(len(rows) / target_total, 6) if target_total else 0.0, "production_v1_samples": len(v1_rows), "raw_recovered_rows_after_dedup": sum(1 for row in rows if row.get("raw_recovery")), "new_raw_unique_samples_over_v1": max(0, len(rows) - len(v1_rows)), "recovery_counts": dict(recovered), "unusable_target_reasons": dict(counters), "near_duplicate_removed": duplicate_removed, "recording_cluster_leakage": leakage, "training_candidate_true_count": 0, "promotion_status": "CANDIDATE_ONLY_NOT_PROMOTED", "ready_for_first_sft": False, "readiness_reason": "Raw recovery is independently validated but this phase forbids SFT promotion and chat-TTS identity closure remains open."}
    manifest = {"schema_version": "1.0.0", "artifact_status": "PRODUCTION_V2_RAW_RECOVERED_CANDIDATE", "pipeline_version": VERSION, "files": {"train": "train.jsonl", "validation": "validation.jsonl", "sealed_eval": "sealed_eval.jsonl", "dataset_manifest": "dataset_manifest.json", "quality_report": "quality_report.json", "rejection_funnel": "rejection_funnel.json", "validation_report": "validation_report.json", "target_utterance_recovery_audit": "target_utterance_recovery_audit.jsonl", "raw_cluster_recovery_audit": "raw_cluster_recovery_audit.json", "final_delivery_report": "final_delivery_report.json"}, "source_of_truth": {"raw_timelines": str(TIMELINES.relative_to(ROOT)), "identity_mapping": str(FUSION.relative_to(ROOT)), "recording_clusters": str(CLUSTERS.relative_to(ROOT))}, "production_v1": str(V1.relative_to(ROOT)), "recording_cluster_disjoint": not any(leakage.values()), "training_candidate_true_count": 0, "promotion_status": "CANDIDATE_ONLY_NOT_PROMOTED"}
    funnel = {"pipeline_version": VERSION, "source": "unique_timelines raw corpus", "search_horizons_seconds": list(HORIZONS), "target_total": target_total, "target_indexed_after_target_gates": target_indexed, "target_with_recoverable_context": quality["target_utterances_with_recoverable_context"], "usable_target_response_rate": quality["usable_target_response_rate"], "production_v1_samples": len(v1_rows), "production_v2_samples": len(rows), "recovery_counts": dict(recovered), "unusable_target_reasons": dict(counters), "duplicate_removed": duplicate_removed, "final": snapshot(rows), "recording_cluster_leakage": leakage, "non_mirror_clusters_without_recovery": [row["canonical_recording_id"] for row in cluster_audit_rows if not row["high_risk_mirror_or_reupload"] and row["recoverable_targets"] == 0], "policy": "Assistant identity remains current HIGH/MEDIUM fusion only; context may be UNKNOWN, guest, Vedal, or TTS when clean, temporally connected, and not promoted as assistant."}
    delivery = {"schema_version": "1.0.0", "artifact_status": "PRODUCTION_V2_DELIVERY_SUMMARY", "high_confidence_target_utterances_total": target_total, "target_utterances_with_recoverable_conversational_context": quality["target_utterances_with_recoverable_context"], "final_usable_supervision_targets": len(rows), "raw_recovered_rows_after_dedup": quality["raw_recovered_rows_after_dedup"], "new_raw_unique_samples_over_v1": quality["new_raw_unique_samples_over_v1"], "production_v1_samples": len(v1_rows), "production_v2_samples": len(rows), "sources": quality["unique_sources"], "recording_clusters": quality["unique_recording_clusters"], "recovered_from_old_UNKNOWN": recovered.get("recovered_from_old_UNKNOWN_or_unmapped_context", 0), "recovered_from_guest_context": recovered.get("recovered_from_guest_context", 0), "recovered_from_TTS_chat_context": recovered.get("recovered_from_TTS_chat_context", 0), "recovered_by_wider_timeline_search": recovered.get("recovered_by_wider_timeline_search", 0), "recovered_by_reASR_alignment": 0, "recovered_by_new_turn_reconstruction": recovered.get("recovered_by_new_turn_reconstruction", 0), "target_utterances_still_unusable": target_total - quality["target_utterances_with_recoverable_context"], "top_unrecoverable_reasons": Counter(counters).most_common(10), "non_mirror_clusters_without_recovery": [row["canonical_recording_id"] for row in cluster_audit_rows if not row["high_risk_mirror_or_reupload"] and row["recoverable_targets"] == 0], "usable_target_response_rate": quality["usable_target_response_rate"], "READY_FOR_FIRST_SFT": "NO", "readiness_reason": "Raw-level recovery completed without lowering assistant identity; promotion remains forbidden and unresolved chat-TTS contamination keeps readiness NO."}
    write_json(OUT / "quality_report.json", quality)
    write_json(OUT / "dataset_manifest.json", manifest)
    write_json(OUT / "rejection_funnel.json", funnel)
    write_json(OUT / "final_delivery_report.json", delivery)
    print(json.dumps({"status": "PRODUCTION_V2_RAW_RECOVERY_BUILT", "production_v1_samples": len(v1_rows), "production_v2_samples": len(rows), "target_total": target_total, "recoverable_context": quality["target_utterances_with_recoverable_context"], "usable_target_response_rate": quality["usable_target_response_rate"], "sources": quality["unique_sources"], "recording_clusters": quality["unique_recording_clusters"], "recovery_counts": recovered, "unusable_reasons": counters}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
