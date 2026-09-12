from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from manifest_tools import ROOT, write_json


DATASET = ROOT / "datasets" / "meow_v02_sft_v2_semantic_clean"
OLD_ROOT_CAUSE = ROOT / "reports" / "sft_reconstruction_v1_root_cause.json"


def norm(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def load(name: str) -> list[dict]:
    return [json.loads(line) for line in (DATASET / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    old = json.loads(OLD_ROOT_CAUSE.read_text(encoding="utf-8"))
    quality = json.loads((DATASET / "quality_report.json").read_text(encoding="utf-8"))
    funnel = json.loads((DATASET / "rejection_funnel.json").read_text(encoding="utf-8"))
    semantic = json.loads((DATASET / "semantic_qa_report.json").read_text(encoding="utf-8"))
    dedup = json.loads((DATASET / "dedup_audit.json").read_text(encoding="utf-8"))
    clusters = json.loads((DATASET / "cluster_balance_report.json").read_text(encoding="utf-8"))
    validation = json.loads((DATASET / "validation_report.json").read_text(encoding="utf-8"))
    train = load("train.jsonl")
    prompts = defaultdict(list)
    for row in train:
        prompts[norm((row.get("messages") or [{}])[0].get("content"))].append(row)
    report = {
        "schema_version": "1.0.0",
        "pipeline_version": quality["pipeline_version"],
        "old_final_sft_candidate_v1": {
            "input_rows": old["rows"],
            "unique_context_anchors": old.get("unique_user_prompts"),
            "final_train_rows": old["rows"],
            "duplicate_prompt_rows": old["rows_in_duplicate_prompt_groups"],
            "anchor_duplicate_count": old["context_anchor_duplicate_count"],
            "prefix_containment_count": old["prefix_containment_pair_count"],
            "target_turn_reuse_count": old["target_turn_reuse_count"],
            "recording_clusters": old["recording_cluster_rows"],
            "largest_cluster_share": old["largest_cluster_share"],
            "top_3_cluster_share": old["top_3_cluster_share"],
        },
        "new_semantic_clean_v2": {
            "input_rows": quality["raw_candidates_after_identity_text_gates"],
            "unique_context_anchors": quality["unique_context_anchors"],
            "final_train_rows": len(train),
            "duplicate_prompt_rows": sum(len(group) for group in prompts.values() if len(group) > 1),
            "anchor_duplicate_count": validation["duplicate_context_anchor_count"],
            "prefix_containment_count": validation["prefix_containment_count"],
            "target_turn_reuse_count": validation["target_turn_reuse_count"],
            "semantic_accept": semantic["counts"].get("ACCEPT", 0),
            "semantic_repair": semantic["counts"].get("REPAIR_FROM_RAW", 0),
            "semantic_reject": semantic["counts"].get("REJECT", 0),
            "asr_garbage_rejected": funnel.get("gate_counts", {}).get("target_transcript_quality", 0) + funnel["rejection_reasons"].get("context_transcript_quality", 0),
            "local_re_asr_recovered": 0,
            "recording_clusters": quality["unique_recording_clusters"],
            "train_recording_clusters": len(clusters["assignments"]["train"]),
            "validation_clusters": len(clusters["assignments"]["validation"]),
            "sealed_clusters": len(clusters["assignments"]["sealed_eval"]),
            "largest_train_cluster_share": clusters["train_recommended"]["largest_cluster_share"],
            "top_3_train_cluster_share": clusters["train_recommended"]["top_3_cluster_share"],
            "recording_leakage": validation["recording_cluster_leakage"],
            "near_duplicate_leakage": validation["near_duplicate_screen"]["response_containment_count"] + validation["prefix_containment_count"],
            "training_candidate_true_count": quality["training_candidate_true_count"],
            "ready_for_first_sft": quality["ready_for_first_sft"],
        },
        "interpretation": "v2 models one observable context anchor to one response episode, rejects bad middle turns as boundaries, re-resolves all rows from raw timelines, and applies anchor/target/prefix/response-containment screens before split assignment.",
    }
    write_json(DATASET / "comparison_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
