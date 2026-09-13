from __future__ import annotations

"""Finalize v2.2 after the local semantic judge completes.

This reuses only the tested v2.1 forensic primitives for episode validation,
overlap evidence, and dedup; all outputs and report names are v2.2-specific.
"""

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets" / "meow_v02_sft_v2_2_semantic_verified"
STRUCTURAL = OUT / "structural_candidates_v2_2.jsonl"
RESULTS = OUT / "semantic_judge_results_v2_2.jsonl"
RUN = OUT / "semantic_judge_run_v2_2.json"
sys.path.insert(0, str(ROOT / "scripts"))
import sft_semantic_verified_v2_1 as core  # noqa: E402

core.VERSION = "sft-v2.2-semantic-verified-2026-09-13"
core.JUDGE_PROMPT_VERSION = "semantic-relation-v2.2-incremental-p1"


def load_jsonl(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                yield row


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def balanced_split(rows: list[dict]) -> dict[str, set[str]]:
    counts = Counter(str(row.get("recording_family_id")) for row in rows)
    families = [family for family, _ in counts.most_common()]
    if len(families) < 6:
        return {"train": set(families), "validation": set(), "sealed_eval": set()}
    # Keep the two largest families in train, but reserve non-trivial
    # independent families. Choosing the smallest families creates a formally
    # disjoint yet useless 1-5-row validation/eval set.
    # Keep the two largest families in train; otherwise one large recording
    # family can consume an entire holdout split and force over-aggressive
    # train downsampling.
    holdout_pool = families[2:]
    target_each = sum(counts.values()) * 0.08
    validation: set[str] = set()
    sealed: set[str] = set()
    validation_count = sealed_count = 0
    for family in holdout_pool:
        if validation_count >= target_each and sealed_count >= target_each and len(validation) >= 3 and len(sealed) >= 3:
            break
        if validation_count <= sealed_count:
            validation.add(family)
            validation_count += counts[family]
        else:
            sealed.add(family)
            sealed_count += counts[family]
    return {"train": set(families) - validation - sealed, "validation": validation, "sealed_eval": sealed}


def diversity_cap_all(rows: list[dict], max_share: float = 0.40) -> tuple[list[dict], dict]:
    """Cap every dominant family, not only the first one.

    The older v2.1 helper capped one family.  With two large independent
    recordings, that can leave the second family above the intended share.
    """
    current = list(rows)
    before = Counter(str(row.get("recording_family_id")) for row in current)
    actions = []
    while current:
        counts = Counter(str(row.get("recording_family_id")) for row in current)
        family, count = counts.most_common(1)[0]
        share = count / len(current)
        if share <= max_share or len(counts) <= 1:
            break
        other = len(current) - count
        cap = max(1, int(other * max_share / (1 - max_share)))
        if cap >= count:
            break
        family_rows = [row for row in current if str(row.get("recording_family_id")) == family]
        # Stable diversity-preserving order: spread source ids first, then
        # retain a range of response lengths and deterministic tie-breaks.
        family_rows.sort(key=lambda row: (str(row.get("source_id")), len(core.response_tokens(row)), str(row.get("sample_id"))))
        keep_ids = {str(row.get("sample_id")) for row in family_rows[:cap]}
        current = [row for row in current if str(row.get("recording_family_id")) != family or str(row.get("sample_id")) in keep_ids]
        actions.append({"family": family, "before": count, "after": cap, "share_before": round(share, 6)})
    after_counts = Counter(str(row.get("recording_family_id")) for row in current)
    return current, {"applied": bool(actions), "actions": actions, "before": dict(before), "after": dict(after_counts), "largest_share_before": round(before.most_common(1)[0][1] / len(rows), 6) if rows else 0.0, "largest_share_after": round(after_counts.most_common(1)[0][1] / len(current), 6) if current else 0.0}


_STOPWORDS = set("the a an and or but if then than to of in on at for from with as is are was were be been being i you he she it we they this that those these do did does can could would should will may might have has had not no yes just really very like my your our their his her its how what why when where who please tell explain imagine guess".split())
_TRIGGER_RE = re.compile(r"\?|\b(what|why|how|when|where|who|can|could|would|should|do|did|does|is|are|will|please|tell|explain|imagine|guess)\b", re.I)
_ACK_RE = re.compile(r"^(yes|no|okay|ok|right|sure|exactly|i think|i guess|well|that|yeah|oh|got that|let's go)\b", re.I)
_CONTINUATION_WORDS = {"and", "or", "because", "which", "that", "to", "then", "than", "as", "if", "of", "in", "with"}
_INDEPENDENT_STARTERS = {"i", "i'm", "im", "well", "now", "but", "also", "actually", "so", "okay", "ok", "wait", "you", "we", "they"}


def _content_tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z]{3,}", str(value or "").lower()) if token not in _STOPWORDS}


def context_relation_guard(row: dict) -> str | None:
    messages = row.get("messages") or []
    context = " ".join(str(item.get("content") or "") for item in messages if item.get("role") == "user")
    response = " ".join(str(item.get("content") or "") for item in messages if item.get("role") == "assistant")
    if _content_tokens(context) & _content_tokens(response):
        return None
    if _TRIGGER_RE.search(context) or _ACK_RE.search(response):
        return None
    # A lowercase response can be a genuine transcript continuation; the raw
    # timeline/episode guard handles whether its boundary is valid.
    if response[:1].islower():
        return None
    if len(re.findall(r"[A-Za-z]+", response)) <= 2:
        return None
    return "SEMANTIC_RELATION_UNCLEAR_NO_OBSERVABLE_TEXTUAL_LINK"


def episode_continuity_guard(row: dict, timelines: dict[str, dict[str, Any]]) -> str | None:
    target_ids = [str(value) for value in row.get("target_turn_ids") or []]
    if len(target_ids) < 2:
        return None
    turns = (timelines.get(str(row.get("source_id"))) or {}).get("_turns") or []
    by_id = {str(turn.get("turn_id")): turn for turn in turns}
    selected = [by_id[value] for value in target_ids if value in by_id]
    if len(selected) != len(target_ids):
        return "RESPONSE_EPISODE_TURN_MISSING"
    for previous, current in zip(selected, selected[1:]):
        previous_text = str(previous.get("text") or "").strip()
        current_text = str(current.get("text") or "").strip()
        first = current_text.lower().split(" ")[0].strip("'\".,!?;:") if current_text else ""
        last = previous_text.lower().split()[-1].strip("'\".,!?;:") if previous_text else ""
        if first in _INDEPENDENT_STARTERS:
            return "RESPONSE_EPISODE_INDEPENDENT_SPEECH_ACT"
        if not (current_text[:1].islower() or first in _CONTINUATION_WORDS or last in _CONTINUATION_WORDS):
            return "RESPONSE_EPISODE_INDEPENDENT_SPEECH_ACT"
    return None


def finalize() -> dict:
    rows = list(load_jsonl(STRUCTURAL))
    results = {str(row.get("sample_id")): row for row in load_jsonl(RESULTS)}
    fusion = core.load_fusion()
    timelines = core.load_timelines({str(row.get("source_id")) for row in rows}, fusion)
    accepted, episode_audit = core.episode_rows(rows, results, timelines)
    model_accepted_count = len(accepted)
    semantic_guard_rejected = 0
    guarded: list[dict] = []
    for row in accepted:
        guard_reason = context_relation_guard(row) or episode_continuity_guard(row, timelines)
        if guard_reason:
            semantic_guard_rejected += 1
            episode_audit.append({"sample_id": row.get("sample_id"), "source_id": row.get("source_id"), "canonical_recording_id": row.get("canonical_recording_id"), "current_target_turn_ids": row.get("target_turn_ids") or [], "outcome": "REJECT", "reason": guard_reason, "guard": "independent_post_judge_semantic_guard"})
            continue
        guarded.append(row)
    accepted = guarded
    write_jsonl(OUT / "episode_reconstruction_audit_v2_2.jsonl", episode_audit)
    # Preserve Layer B before any recording-level or duplicate removal.
    for row in accepted:
        row["pipeline_version"] = core.VERSION
        row["training_candidate"] = False
        row["evaluation_only"] = False
        row["semantic_quality"] = "PASS" if (row.get("semantic_qa") or {}).get("status") == "MODEL_ACCEPT" else "UNKNOWN"
        row["context_integrity"] = "PASS" if (row.get("semantic_qa") or {}).get("context_complete") is True else "UNKNOWN"
        row["transcript_quality"] = "PASS" if (row.get("semantic_qa") or {}).get("transcript_usable") is True and (row.get("transcript_qa") or {}).get("status") == "STRUCTURAL_PASS" else "UNKNOWN"
        row["response_episode_integrity"] = "PASS" if (row.get("episode_reconstruction") or {}).get("decision") in {"KEEP_CURRENT", "MERGE_CONTINUATION"} else "UNKNOWN"
    write_jsonl(OUT / "semantic_verified_candidates_v2_2.jsonl", sorted(accepted, key=lambda item: str(item.get("sample_id"))))

    edges = core.overlap_edges(accepted)
    write_jsonl(OUT / "recording_overlap_edges_v2_2.jsonl", edges)
    repair = core.repair_recording_families(accepted, edges)
    repair["pipeline_version"] = core.VERSION
    repair["pairwise_edges_preserved"] = True
    write_json(OUT / "recording_cluster_repair_v2_2.json", repair)
    clean, dedup = core.global_dedup(accepted, repair["cluster_to_family"])
    write_json(OUT / "global_dedup_audit_v2_2.json", dedup)

    assignments = balanced_split(clean)
    train_full: list[dict] = []
    validation: list[dict] = []
    sealed: list[dict] = []
    for row in clean:
        family = str(row.get("recording_family_id"))
        split = "sealed_eval" if family in assignments["sealed_eval"] else "validation" if family in assignments["validation"] else "train"
        row["split"] = split
        row["evaluation_only"] = split == "sealed_eval"
        row["training_candidate"] = False
        (sealed if split == "sealed_eval" else validation if split == "validation" else train_full).append(row)
    train, cap_report = diversity_cap_all(train_full)
    train_ids = {str(row.get("sample_id")) for row in train}
    for row in train_full:
        row["recommended_train"] = str(row.get("sample_id")) in train_ids
    for row in validation + sealed:
        row["recommended_train"] = False
    for name, data in (("train_clean_full.jsonl", train_full), ("train.jsonl", train), ("validation.jsonl", validation), ("sealed_eval.jsonl", sealed)):
        write_jsonl(OUT / name, sorted(data, key=lambda item: str(item.get("sample_id"))))

    rejected = Counter()
    for item in episode_audit:
        if item.get("outcome") == "ACCEPT":
            continue
        relation = str((item.get("judge") or {}).get("relation") or "")
        reason = str(item.get("reason") or "other_semantic_rejected")
        if relation == "HIDDEN_TRIGGER_SUSPECTED" or reason == "hidden_trigger_suspected":
            rejected["HIDDEN_TRIGGER"] += 1
        elif relation == "WRONG_CONTEXT" or reason == "wrong_context":
            rejected["WRONG_CONTEXT"] += 1
        elif relation == "UNCLEAR":
            rejected["UNCLEAR"] += 1
        else:
            rejected[reason] += 1
    families_by_split = {"train": sorted({str(row.get("recording_family_id")) for row in train}), "validation": sorted({str(row.get("recording_family_id")) for row in validation}), "sealed_eval": sorted({str(row.get("recording_family_id")) for row in sealed})}
    leakage = {"train_validation": sorted(set(families_by_split["train"]) & set(families_by_split["validation"])), "train_sealed_eval": sorted(set(families_by_split["train"]) & set(families_by_split["sealed_eval"])), "validation_sealed_eval": sorted(set(families_by_split["validation"]) & set(families_by_split["sealed_eval"]))}
    family_counts = Counter(str(row.get("recording_family_id")) for row in train)
    largest = family_counts.most_common(1)[0][1] / len(train) if train else 0.0
    top3 = sum(value for _, value in family_counts.most_common(3)) / len(train) if train else 0.0
    quality = {
        "schema_version": "1.0.0",
        "artifact_status": "SFT_SEMANTIC_VERIFIED_V2_2_CANDIDATE",
        "pipeline_version": core.VERSION,
        "judge_model": sorted({str(row.get("judge_model")) for row in results.values() if row.get("judge_model")}),
        "judge_prompt_version": core.JUDGE_PROMPT_VERSION,
        "v2_2_structural_candidates": len(rows),
        "semantic_accepted": len(accepted),
        "model_semantic_accepted_before_independent_guard": model_accepted_count,
        "model_semantic_rejected": len(rows) - model_accepted_count,
        "semantic_rejected": len(rows) - len(accepted),
        "semantic_guard_rejected": semantic_guard_rejected,
        "hidden_trigger_rejected": rejected.get("HIDDEN_TRIGGER", 0),
        "wrong_context_rejected": rejected.get("WRONG_CONTEXT", 0),
        "response_episodes": len(accepted),
        "single_segment_episodes": sum(len(row.get("target_turn_ids") or []) == 1 for row in accepted),
        "multi_segment_episodes": sum(len(row.get("target_turn_ids") or []) > 1 for row in accepted),
        "global_exact_duplicate": dedup.get("global_exact_message_duplicate_count", 0),
        "global_response_containment": dedup.get("global_response_containment_count", 0),
        "cross_cluster_overlap_pairs": len(edges),
        "recording_families": len(repair.get("recording_families") or []),
        "final_train": len(train),
        "final_train_clean_full": len(train_full),
        "final_validation": len(validation),
        "final_sealed_eval": len(sealed),
        "unique_sources": len({str(row.get("source_id")) for row in train + validation + sealed}),
        "unique_underlying_recording_families": len({str(row.get("recording_family_id")) for row in train + validation + sealed}),
        "largest_train_recording_family_share": round(largest, 6),
        "top_3_train_recording_family_share": round(top3, 6),
        "target_reuse": dedup.get("target_turn_reuse_count", 0),
        "duplicate_context_anchor": dedup.get("duplicate_context_anchor_count", 0),
        "prefix_ladder": dedup.get("prefix_ladder_count", 0),
        "cross_split_content_leakage": leakage,
        "quality_layers": {"structural": True, "semantic": len(accepted) > 0 and not rejected.get("judge_request_error"), "transcript": all(row.get("transcript_quality") == "PASS" for row in accepted), "episode": all(row.get("response_episode_integrity") == "PASS" for row in accepted), "recording_independence": not any(leakage.values())},
        "training_candidate_true_count": 0,
        "ready_for_first_sft": False,
        "readiness_reason": "v2.2 is a semantic-verified candidate; active no-SFT/no-promotion hold remains in force and training_candidate is false.",
        "diversity_cap": cap_report,
    }
    write_json(OUT / "quality_report_v2_2.json", quality)
    write_json(OUT / "rejection_funnel_v2_2.json", {"schema_version": "1.0.0", "pipeline_version": core.VERSION, "stages": [{"stage": "structural_candidates_v2_2", "rows": len(rows), "removed_rows": 0}, {"stage": "semantic_relation_and_episode_judge", "rows": len(accepted), "removed_rows": len(rows) - len(accepted)}, {"stage": "recording_overlap_repair", "rows": len(accepted), "removed_rows": 0, "cross_cluster_overlap_pairs": len(edges)}, {"stage": "global_dedup", "rows": len(clean), "removed_rows": len(accepted) - len(clean)}, {"stage": "final_train", "rows": len(train), "removed_rows": len(train_full) - len(train)}, {"stage": "validation", "rows": len(validation), "removed_rows": 0}, {"stage": "sealed_eval", "rows": len(sealed), "removed_rows": 0}], "rejection_reasons": dict(rejected)})
    write_json(OUT / "dataset_manifest_v2_2.json", {"schema_version": "1.0.0", "artifact_status": quality["artifact_status"], "pipeline_version": core.VERSION, "files": {"structural_candidates": "structural_candidates_v2_2.jsonl", "semantic_verified_candidates": "semantic_verified_candidates_v2_2.jsonl", "train": "train.jsonl", "train_clean_full": "train_clean_full.jsonl", "validation": "validation.jsonl", "sealed_eval": "sealed_eval.jsonl", "semantic_judge_results": "semantic_judge_results_v2_2.jsonl", "episode_reconstruction_audit": "episode_reconstruction_audit_v2_2.jsonl", "recording_overlap_edges": "recording_overlap_edges_v2_2.jsonl", "recording_cluster_repair": "recording_cluster_repair_v2_2.json", "global_dedup_audit": "global_dedup_audit_v2_2.json", "quality_report": "quality_report_v2_2.json", "rejection_funnel": "rejection_funnel_v2_2.json"}, "source_of_truth": {"raw_timelines": "unique_timelines", "fusion": "identity_results/identity_mapping_multimodal_fusion_proxy.jsonl", "incremental_delta": "reports/sft_v2_2_incremental_delta.json"}, "training_candidate_true_count": 0, "ready_for_first_sft": False})
    return {"status": "FINALIZED", "quality": quality, "output": str(OUT)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.wait:
        while True:
            try:
                state = json.loads(RUN.read_text(encoding="utf-8"))
            except Exception:
                state = {}
            if state.get("status") == "COMPLETED":
                break
            time.sleep(args.poll_seconds)
    if not RUN.exists() or json.loads(RUN.read_text(encoding="utf-8")).get("status") != "COMPLETED":
        raise SystemExit("semantic judge is not completed")
    print(json.dumps(finalize(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
