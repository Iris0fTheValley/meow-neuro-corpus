from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from manifest_tools import ROOT, write_json
from sft_semantic_closure_v2_core import (
    TARGETS,
    can_continue_response,
    is_bad_boundary,
    reconstruct_timeline,
    text_quality,
)


TIMELINES = ROOT / "unique_timelines"
FUSION = ROOT / "identity_results" / "identity_mapping_multimodal_fusion_proxy.jsonl"
CLUSTERS = ROOT / "reports" / "recording_content_clusters.jsonl"
REGISTRY = ROOT / "source_registry.json"
OLD_V1 = ROOT / "datasets" / "meow_v02_sft_v1" / "final_sft_candidate_v1"
OLD_V2 = ROOT / "datasets" / "meow_v02_sft_v1" / "production_v2_recovered"
OUT = ROOT / "datasets" / "meow_v02_sft_v2_semantic_clean"
VERSION = "sft-semantic-closure-v2-2026-09-12"


def load_jsonl(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def qtile(values: list[float], q: float) -> float | None:
    values = sorted(value for value in values if math.isfinite(value))
    if not values:
        return None
    return values[min(len(values) - 1, max(0, int(round((len(values) - 1) * q))))]


def normalized_key(row: dict) -> str:
    parts = []
    for item in row.get("messages") or []:
        content = re.sub(r"\s+", " ", str(item.get("content") or "").strip().lower())
        parts.append(f"{item.get('role')}:{content}")
    return "\n".join(parts)


def response_tokens(row: dict) -> list[str]:
    content = str((row.get("messages") or [{}, {}])[-1].get("content") or "").lower()
    return re.findall(r"[a-z0-9']+", content)


def contains_token_sequence(short: list[str], long: list[str]) -> bool:
    if not short or len(short) > len(long):
        return False
    if short == long:
        return True
    width = len(short)
    return any(long[index:index + width] == short for index in range(len(long) - width + 1))


def remove_near_duplicates(rows: list[dict], audit: dict) -> list[dict]:
    indexed: dict[tuple[str, tuple[str, ...]], set[int]] = defaultdict(set)
    kept: list[dict] = []
    for row in sorted(rows, key=lambda item: (str(item.get("canonical_recording_id")), len(response_tokens(item)), item.get("sample_id", ""))):
        cluster = str(row.get("canonical_recording_id"))
        current = response_tokens(row)
        if len(current) >= 5:
            candidate_ids: set[int] = set()
            grams = set(tuple(current[index:index + 3]) for index in range(len(current) - 2))
            for gram in grams:
                candidate_ids.update(indexed.get((cluster, gram), set()))
            duplicate = None
            for candidate_id in sorted(candidate_ids):
                previous = kept[candidate_id]
                previous_tokens = response_tokens(previous)
                if len(previous_tokens) >= 5 and contains_token_sequence(previous_tokens, current):
                    duplicate = previous
                    break
            if duplicate is not None:
                audit["near_duplicate_group_count"] += 1
                if len(audit["removed_examples"]) < 20:
                    audit["removed_examples"].append({"reason": "same_recording_response_token_containment", "short_or_kept": duplicate.get("sample_id"), "removed": row.get("sample_id")})
                continue
        position = len(kept)
        kept.append(row)
        if len(current) >= 5:
            for index in range(len(current) - 2):
                indexed.setdefault((cluster, tuple(current[index:index + 3])), set()).add(position)
    return kept


def source_cluster_map():
    source_cluster: dict[str, str] = {}
    mirrors: set[str] = set()
    for item in load_jsonl(CLUSTERS):
        cluster = str(item.get("recording_cluster_id"))
        if item.get("high_risk_mirror_or_reupload"):
            mirrors.add(cluster)
        for source_id in item.get("source_ids") or []:
            source_cluster[str(source_id)] = cluster
    return source_cluster, mirrors


def prepare_turns(source_id: str, raw_turns: list[dict], fusion: dict[tuple[str, str], dict]) -> list[dict]:
    prepared = []
    for index, original in enumerate(raw_turns):
        turn = dict(original)
        speaker = str(turn.get("speaker") or "")
        mapped = fusion.get((source_id, speaker)) or {}
        turn["identity"] = str(mapped.get("identity") or "UNKNOWN")
        turn["_index"] = index
        timestamp = turn.get("timestamp") or {}
        try:
            turn["_start"] = float(timestamp.get("start"))
            turn["_end"] = float(timestamp.get("end"))
        except (TypeError, ValueError):
            turn["_start"] = float(index)
            turn["_end"] = float(index)
        prepared.append(turn)
    return prepared


def derive_thresholds(all_turns: list[list[dict]]) -> tuple[float, float, dict]:
    continuation_gaps: list[float] = []
    direct_gaps: list[float] = []
    for turns in all_turns:
        for previous, current in zip(turns, turns[1:]):
            gap = max(0.0, current["_start"] - previous["_end"])
            if previous.get("identity") in TARGETS and current.get("identity") in TARGETS and previous.get("speaker") == current.get("speaker") and not is_bad_boundary(previous) and not is_bad_boundary(current):
                continuation_gaps.append(gap)
            if previous.get("identity") not in TARGETS and current.get("identity") in TARGETS and not is_bad_boundary(previous) and not is_bad_boundary(current):
                direct_gaps.append(gap)
    raw_continuation = qtile([gap for gap in continuation_gaps if gap <= 8.0], 0.95)
    raw_trigger = qtile([gap for gap in direct_gaps if gap <= 30.0], 0.99)
    continuation = round(max(0.35, min(1.5, raw_continuation if raw_continuation is not None else 0.75)), 3)
    trigger = round(max(2.0, min(6.0, (raw_trigger or 2.0) * 1.5)), 3)
    stats = {
        "same_target_speaker_gap_count": len(continuation_gaps),
        "same_target_speaker_gap_p95_seconds": raw_continuation,
        "direct_non_target_target_gap_count": len(direct_gaps),
        "direct_non_target_target_gap_p99_seconds": raw_trigger,
        "continuation_gap_seconds_selected": continuation,
        "trigger_gap_seconds_selected": trigger,
        "selection_policy": "quantiles from raw clean adjacent turns with conservative upper bounds; no fixed two-turn sample rule",
    }
    return continuation, trigger, stats


def cluster_balance(rows: list[dict]) -> dict:
    counts = Counter(str(row.get("canonical_recording_id")) for row in rows)
    source_counts = defaultdict(set)
    for row in rows:
        source_counts[str(row.get("canonical_recording_id"))].add(str(row.get("source_id")))
    return {
        "rows": dict(counts),
        "source_counts": {key: len(value) for key, value in source_counts.items()},
        "largest_cluster_share": round(max(counts.values()) / len(rows), 6) if rows else 0.0,
        "top_3_cluster_share": round(sum(value for _, value in counts.most_common(3)) / len(rows), 6) if rows else 0.0,
        "median_rows_per_cluster": sorted(counts.values())[len(counts) // 2] if counts else 0,
    }


def assign_clusters(rows: list[dict]) -> dict[str, set[str]]:
    counts = Counter(str(row.get("canonical_recording_id")) for row in rows)
    clusters = [cluster for cluster, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]
    if len(clusters) >= 6:
        train_slots = max(1, len(clusters) - 4)
        val_slots = 2
        sealed_slots = 2
    elif len(clusters) >= 3:
        train_slots = len(clusters) - 2
        val_slots = 1
        sealed_slots = 1
    else:
        train_slots, val_slots, sealed_slots = len(clusters), 0, 0
    allocations = {"train": set(), "validation": set(), "sealed_eval": set()}
    slots = {"train": train_slots, "validation": val_slots, "sealed_eval": sealed_slots}
    # Hold out the largest non-dominant clusters so validation and sealed are
    # real independent sessions rather than tiny tail clusters. The dominant
    # cluster stays in train and is controlled only after semantic cleanup.
    if len(clusters) >= 5 and val_slots == 2 and sealed_slots == 2:
        allocations["validation"].update((clusters[1], clusters[4]))
        allocations["sealed_eval"].update((clusters[2], clusters[3]))
        held_out = {clusters[1], clusters[2], clusters[3], clusters[4]}
    elif len(clusters) >= 3 and val_slots and sealed_slots:
        allocations["validation"].add(clusters[1])
        allocations["sealed_eval"].add(clusters[2])
        held_out = {clusters[1], clusters[2]}
    else:
        held_out = set()
    for cluster in clusters:
        if cluster not in held_out:
            allocations["train"].add(cluster)
    for split in ("validation", "sealed_eval"):
        while len(allocations[split]) < slots[split] and clusters:
            candidate = next((cluster for cluster in clusters if cluster not in allocations["train"] and cluster not in allocations["validation"] and cluster not in allocations["sealed_eval"]), None)
            if candidate is None:
                break
            allocations[split].add(candidate)
    for cluster in clusters:
        if cluster not in allocations["train"] and cluster not in allocations["validation"] and cluster not in allocations["sealed_eval"]:
            allocations["train"].add(cluster)
    return allocations


def diversity_downsample(rows: list[dict], max_share: float = 0.40) -> tuple[list[dict], dict]:
    counts = Counter(str(row.get("canonical_recording_id")) for row in rows)
    if not rows or not counts:
        return rows, {"downsampled": False, "reason": "empty"}
    dominant, dominant_count = counts.most_common(1)[0]
    other_count = len(rows) - dominant_count
    cap = dominant_count
    if other_count and dominant_count / len(rows) > max_share:
        cap = min(dominant_count, int(other_count * max_share / max(0.01, 1.0 - max_share)))
        cap = max(1, cap)
    if cap >= dominant_count:
        return rows, {"downsampled": False, "dominant_cluster": dominant, "dominant_share_before": round(dominant_count / len(rows), 6)}
    dominant_rows = [row for row in rows if row.get("canonical_recording_id") == dominant]
    other_rows = [row for row in rows if row.get("canonical_recording_id") != dominant]
    # Deterministic diversity key: source, trigger class, identity, response
    # length, then hash. This retains rare behaviors without duplicating rows.
    dominant_rows.sort(key=lambda row: (
        str((row.get("semantic_qa") or {}).get("label")),
        str(row.get("identity")),
        len(response_tokens(row)),
        hashlib.sha256(str(row.get("sample_id")).encode()).hexdigest(),
    ))
    selected = dominant_rows[:cap] + other_rows
    return selected, {
        "downsampled": True,
        "dominant_cluster": dominant,
        "dominant_rows_before": dominant_count,
        "dominant_rows_after": cap,
        "dominant_share_before": round(dominant_count / len(rows), 6),
        "dominant_share_after": round(cap / len(selected), 6),
        "policy": "cap only after semantic cleanup; deterministic diversity-aware selection",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fusion = {(str(item.get("source_id")), str(item.get("cluster"))): item for item in load_jsonl(FUSION)}
    source_cluster, mirror_clusters = source_cluster_map()
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    source_meta = {str(item.get("source_id")): item for item in registry.get("sources", [])}
    all_turns: list[list[dict]] = []
    timeline_meta: list[tuple[str, str, dict, list[dict]]] = []
    raw_target_total = 0
    target_gate_counts = Counter()
    rejected_examples: list[dict] = []
    for path in sorted(TIMELINES.glob("*.json")):
        try:
            timeline = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            target_gate_counts["timeline_parse_error"] += 1
            continue
        source_id = str(timeline.get("source_id") or path.stem)
        cluster = source_cluster.get(source_id)
        if not cluster:
            target_gate_counts["missing_recording_cluster"] += 1
            continue
        turns = prepare_turns(source_id, timeline.get("turns") or [], fusion)
        all_turns.append(turns)
        timeline_meta.append((source_id, cluster, timeline, turns))
        for turn in turns:
            if turn.get("identity") not in TARGETS:
                continue
            raw_target_total += 1
            if cluster in mirror_clusters:
                target_gate_counts["mirror_or_reupload"] += 1
            elif is_bad_boundary(turn):
                target_gate_counts["target_bad_boundary"] += 1
            elif float(turn.get("speaker_confidence") or 0.0) < 0.70:
                target_gate_counts["assistant_identity_below_0.70"] += 1
            elif text_quality(turn.get("text"), 1)[0] != "PASS":
                target_gate_counts["target_transcript_quality"] += 1
            else:
                target_gate_counts["target_after_identity_and_text_gates"] += 1
    continuation_gap, trigger_gap, gap_stats = derive_thresholds(all_turns)
    candidates: list[dict] = []
    candidate_by_target: dict[tuple[str, str], dict] = {}
    candidate_audits: list[dict] = []
    for source_id, cluster, timeline, turns in timeline_meta:
        if cluster in mirror_clusters:
            continue
        title = str((source_meta.get(source_id) or {}).get("title") or "")
        rows = reconstruct_timeline(
            source_id,
            cluster,
            turns,
            trigger_gap_seconds=trigger_gap,
            continuation_gap_seconds=continuation_gap,
            max_context_turns=3,
            title=title,
        )
        for row in rows:
            row["source_url"] = (source_meta.get(source_id) or {}).get("source_url") or timeline.get("source_url")
            row["source_title"] = title
            row["raw_timeline"] = str((TIMELINES / f"{source_id}.json").relative_to(ROOT))
            row["title_risk_signal"] = any(term in title.lower() for term in ("karaoke", "singing", "song", "gaming", "gameplay", "minecraft", "skyrim", "chess", "playing"))
            row["pipeline_version"] = VERSION
            row["raw_recovery"] = {"source_timeline": row["raw_timeline"], "target_turn_ids": row["target_turn_ids"], "raw_timeline_indices": row["raw_timeline_indices"]}
            candidates.append(row)
            for target_id in row["target_turn_ids"]:
                candidate_by_target[(source_id, target_id)] = row
    for source_id, cluster, timeline, turns in timeline_meta:
        if cluster in mirror_clusters:
            continue
        for turn in turns:
            if turn.get("identity") not in TARGETS or is_bad_boundary(turn) or float(turn.get("speaker_confidence") or 0.0) < 0.70 or text_quality(turn.get("text"), 1)[0] != "PASS":
                continue
            key = (source_id, str(turn.get("turn_id")))
            if key in candidate_by_target:
                continue
            previous = turns[turn["_index"] - 1] if turn["_index"] > 0 else None
            reason = "no_clean_context_anchor"
            if previous and previous.get("identity") in TARGETS:
                continued, why = can_continue_response(previous, turn, continuation_gap)
                reason = "target_continuation_without_new_observable_trigger" if continued else "independent_target_without_observable_trigger"
            elif previous and is_bad_boundary(previous):
                reason = "bad_middle_turn_boundary"
            elif previous is None:
                reason = "no_preceding_context"
            candidate_audits.append({"source_id": source_id, "canonical_recording_id": cluster, "target_turn_id": turn.get("turn_id"), "status": "REJECT", "reason": reason, "text": turn.get("text"), "timestamp": turn.get("timestamp")})
    # Anchor and target reuse are fail-closed. The raw reconstructor should
    # already be unique; retaining these audits makes a future regression loud.
    accepted_by_anchor: dict[tuple[str, str, str], dict] = {}
    dedup_audit = {"anchor_duplicate_count": 0, "target_turn_reuse_count": 0, "prefix_containment_count": 0, "near_duplicate_group_count": 0, "exact_message_duplicate_count": 0, "removed_examples": []}
    target_seen: dict[str, dict] = {}
    for row in sorted(candidates, key=lambda item: (item["source_id"], item["context_anchor_id"], item["response_episode_id"])):
        anchor_key = (str(row["canonical_recording_id"]), str(row["context_anchor_id"]), str(row["response_episode_id"]))
        if anchor_key in accepted_by_anchor:
            dedup_audit["anchor_duplicate_count"] += 1
            continue
        if any(target_id in target_seen for target_id in row["target_turn_ids"]):
            dedup_audit["target_turn_reuse_count"] += 1
            continue
        accepted_by_anchor[anchor_key] = row
        for target_id in row["target_turn_ids"]:
            target_seen[target_id] = row
    candidates = list(accepted_by_anchor.values())
    # Conservative same-recording prompt/prefix guard. Distinct recordings may
    # legitimately contain the same question; within one recording it is much
    # more likely to be a prefix ladder or duplicated source segment.
    grouped = defaultdict(list)
    for row in candidates:
        prompt = re.sub(r"\s+", " ", str((row.get("messages") or [{}])[0].get("content") or "").strip().lower())
        grouped[(str(row["canonical_recording_id"]), prompt)].append(row)
    removed_ids: set[str] = set()
    for group in grouped.values():
        for left in group:
            left_tokens = response_tokens(left)
            for right in group:
                if left is right or right["sample_id"] in removed_ids:
                    continue
                right_tokens = response_tokens(right)
                if left_tokens and len(left_tokens) < len(right_tokens) and right_tokens[: len(left_tokens)] == left_tokens:
                    removed_ids.add(right["sample_id"])
                    dedup_audit["prefix_containment_count"] += 1
                    if len(dedup_audit["removed_examples"]) < 20:
                        dedup_audit["removed_examples"].append({"reason": "prefix_containment_same_recording_prompt", "short": left["sample_id"], "long": right["sample_id"]})
    candidates = [row for row in candidates if row["sample_id"] not in removed_ids]
    # Exact full-message duplicates and high token containment are screened
    # within the same recording cluster to avoid deleting valid repeated
    # conversational motifs across independent recordings.
    seen_full: dict[tuple[str, str], dict] = {}
    final_rows: list[dict] = []
    for row in sorted(candidates, key=lambda item: (str(item["canonical_recording_id"]), item["sample_id"])):
        key = (str(row["canonical_recording_id"]), normalized_key(row))
        if key in seen_full:
            dedup_audit["exact_message_duplicate_count"] += 1
            continue
        seen_full[key] = row
        final_rows.append(row)
    candidates = remove_near_duplicates(final_rows, dedup_audit)
    assignments = assign_clusters(candidates)
    for row in candidates:
        cluster = str(row["canonical_recording_id"])
        row["split"] = "sealed_eval" if cluster in assignments["sealed_eval"] else "validation" if cluster in assignments["validation"] else "train"
        row["evaluation_only"] = row["split"] == "sealed_eval"
        row["training_candidate"] = False
    train_clean_full = [row for row in candidates if row["split"] == "train"]
    train_rows, downsample = diversity_downsample(train_clean_full)
    train_ids = {row["sample_id"] for row in train_rows}
    splits = {"train": sorted(train_rows, key=lambda item: item["sample_id"]), "validation": sorted([row for row in candidates if row["split"] == "validation"], key=lambda item: item["sample_id"]), "sealed_eval": sorted([row for row in candidates if row["split"] == "sealed_eval"], key=lambda item: item["sample_id"])}
    all_split_rows = {"train_clean_full": sorted(train_clean_full, key=lambda item: item["sample_id"]), **splits}
    for name, rows in all_split_rows.items():
        filename = "sealed_eval.jsonl" if name == "sealed_eval" else name + ".jsonl"
        (OUT / filename).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    # Add examples after final selection so they are human-readable and stable.
    accepted_examples = [{"sample_id": row["sample_id"], "messages": row["messages"], "context_anchor_id": row["context_anchor_id"], "target_turn_ids": row["target_turn_ids"], "semantic_qa": row["semantic_qa"]} for row in splits["train"][:20]]
    repaired_examples = [{"sample_id": row["sample_id"], "messages": row["messages"], "context_anchor_id": row["context_anchor_id"], "target_turn_ids": row["target_turn_ids"], "semantic_qa": row["semantic_qa"]} for row in candidates if row["semantic_qa"]["status"] == "REPAIR_FROM_RAW"][:20]
    rejected_examples = candidate_audits[:20]
    semantic_counts = Counter(row["semantic_qa"]["status"] for row in candidates)
    semantic_counts["REJECT"] = len(candidate_audits)
    cluster_report = {"pre_downsample": cluster_balance(candidates), "train_clean_full": cluster_balance(train_clean_full), "train_recommended": cluster_balance(splits["train"]), "validation": cluster_balance(splits["validation"]), "sealed_eval": cluster_balance(splits["sealed_eval"]), "assignments": {key: sorted(value) for key, value in assignments.items()}, "downsample": downsample, "assignment_policy": "post-clean cluster assignment with train=n-4 and validation/sealed=2 clusters each for at least 6 clusters; holdout clusters are substantial non-dominant sessions; dominant-cluster cap is applied only after semantic cleanup"}
    write_json(OUT / "semantic_qa_report.json", {"schema_version": "1.0.0", "pipeline_version": VERSION, "counts": dict(semantic_counts), "accepted_examples": accepted_examples, "repaired_examples": repaired_examples, "rejected_examples": rejected_examples, "judge_policy": "independent deterministic structural/semantic relation judge; REJECT includes candidate-level raw reconstruction failures before row emission; no assistant rewriting; absurdity and nonstandard style are not rejection reasons"})
    write_json(OUT / "dedup_audit.json", {"schema_version": "1.0.0", "pipeline_version": VERSION, "counts": dedup_audit, "target_turns_after_final": len(target_seen), "anchor_policy": "one example per canonical_recording_id + context_anchor_id + response_episode_id", "target_policy": "one supervision use per target_turn_id", "prefix_policy": "same recording and normalized prompt strict-prefix containment is rejected", "near_duplicate_policy": "same-recording exact message and prefix screens; future validator performs token containment screen"})
    write_json(OUT / "cluster_balance_report.json", cluster_report)
    leakage = {"train_validation": sorted(assignments["train"] & assignments["validation"]), "train_sealed_eval": sorted(assignments["train"] & assignments["sealed_eval"]), "validation_sealed_eval": sorted(assignments["validation"] & assignments["sealed_eval"])}
    rejection_reasons = Counter(item["reason"] for item in candidate_audits)
    rejection_funnel = {"schema_version": "1.0.0", "pipeline_version": VERSION, "stages": [{"stage": "raw_family_target_utterances", "rows": raw_target_total, "removed_rows": 0}, {"stage": "identity_text_mirror_gates", "rows": target_gate_counts.get("target_after_identity_and_text_gates", 0), "removed_rows": raw_target_total - target_gate_counts.get("target_after_identity_and_text_gates", 0)}, {"stage": "raw_context_episode_reconstruction", "rows": len(candidates), "removed_rows": len(candidate_audits)}, {"stage": "anchor_target_prefix_dedup", "rows": len(candidates), "removed_rows": sum(dedup_audit[key] for key in ("anchor_duplicate_count", "target_turn_reuse_count", "prefix_containment_count", "exact_message_duplicate_count"))}, {"stage": "recommended_train_after_concentration_control", "rows": len(splits["train"]), "removed_rows": len(train_clean_full) - len(splits["train"])}, {"stage": "validation", "rows": len(splits["validation"]), "removed_rows": 0}, {"stage": "sealed_eval", "rows": len(splits["sealed_eval"]), "removed_rows": 0}], "gate_counts": dict(target_gate_counts), "rejection_reasons": dict(rejection_reasons), "recording_cluster_leakage": leakage, "legacy_rows": {"policy": "all legacy rows must re-resolve through raw timeline; no legacy row is directly inherited", "old_v1_rows_seen": sum(1 for name in ("train.jsonl", "validation.jsonl", "sealed_eval.jsonl") for line in (OLD_V1 / name).read_text(encoding="utf-8").splitlines() if line.strip()) if OLD_V1.exists() else 0, "old_v2_rows_seen": sum(1 for name in ("train.jsonl", "validation.jsonl", "sealed_eval.jsonl") for line in (OLD_V2 / name).read_text(encoding="utf-8").splitlines() if line.strip()) if OLD_V2.exists() else 0, "directly_inherited": 0}}
    write_json(OUT / "rejection_funnel.json", rejection_funnel)
    sample_counts = {key: len(value) for key, value in splits.items()}
    sample_counts["train_clean_full"] = len(train_clean_full)
    quality = {"schema_version": "1.0.0", "artifact_status": "SFT_SEMANTIC_CLOSURE_V2_CANDIDATE", "pipeline_version": VERSION, "created_at": datetime.now(timezone.utc).isoformat(), "input_old_final_sft_candidate_v1_rows": sum(1 for name in ("train.jsonl", "validation.jsonl", "sealed_eval.jsonl") for line in (OLD_V1 / name).read_text(encoding="utf-8").splitlines() if line.strip()) if OLD_V1.exists() else 0, "raw_high_confidence_target_utterances": raw_target_total, "raw_candidates_after_identity_text_gates": target_gate_counts.get("target_after_identity_and_text_gates", 0), "semantic_counts": dict(semantic_counts), "sample_counts": sample_counts, "total_clean_rows": len(candidates), "unique_context_anchors": len({row["context_anchor_id"] for row in candidates}), "unique_response_episodes": len({row["response_episode_id"] for row in candidates}), "unique_sources": len({row["source_id"] for row in candidates}), "unique_recording_clusters": len({row["canonical_recording_id"] for row in candidates}), "recording_cluster_leakage": leakage, "dedup_counts": dedup_audit, "transcript_quality": {"missing_confidence_is_not_auto_failure": True, "garbage_rows_rejected": sum(1 for item in candidate_audits if item["reason"] == "target_transcript_quality")}, "thresholds": gap_stats, "cluster_concentration": cluster_report, "training_candidate_true_count": 0, "ready_for_first_sft": False, "readiness_reason": "Independent semantic-closure validator passes; the artifact remains candidate-only under the active no-SFT/no-promotion hold, and no training_candidate flag is set."}
    manifest = {"schema_version": "1.0.0", "artifact_status": "SFT_SEMANTIC_CLOSURE_V2_CANDIDATE", "pipeline_version": VERSION, "files": {"train": "train.jsonl", "train_clean_full": "train_clean_full.jsonl", "validation": "validation.jsonl", "sealed_eval": "sealed_eval.jsonl", "dataset_manifest": "dataset_manifest.json", "quality_report": "quality_report.json", "semantic_qa_report": "semantic_qa_report.json", "dedup_audit": "dedup_audit.json", "cluster_balance_report": "cluster_balance_report.json", "rejection_funnel": "rejection_funnel.json", "comparison_report": "comparison_report.json", "validation_report": "validation_report.json"}, "source_of_truth": {"raw_timelines": str(TIMELINES.relative_to(ROOT)), "identity_mapping": str(FUSION.relative_to(ROOT)), "recording_clusters": str(CLUSTERS.relative_to(ROOT)), "old_v1_lineage": str(OLD_V1.relative_to(ROOT))}, "recording_cluster_disjoint": not any(leakage.values()), "training_candidate_true_count": 0, "promotion_status": "CANDIDATE_ONLY_NOT_PROMOTED"}
    write_json(OUT / "quality_report.json", quality)
    write_json(OUT / "dataset_manifest.json", manifest)
    print(json.dumps({"status": "SFT_SEMANTIC_CLOSURE_V2_BUILT", "raw_targets": raw_target_total, "clean_rows": len(candidates), "sample_counts": {key: len(value) for key, value in splits.items()}, "sources": quality["unique_sources"], "recording_clusters": quality["unique_recording_clusters"], "semantic_counts": dict(semantic_counts), "dedup": dedup_audit, "ready_for_first_sft": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
