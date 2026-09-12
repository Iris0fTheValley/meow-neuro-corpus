from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from manifest_tools import ROOT, write_json
from sft_semantic_closure_v2_core import TARGETS, text_quality


DATASET = ROOT / "datasets" / "meow_v02_sft_v2_semantic_clean"
SPLITS = {"train": "train.jsonl", "validation": "validation.jsonl", "sealed_eval": "sealed_eval.jsonl"}


def norm(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", norm(value))


def contains_sequence(short: list[str], long: list[str]) -> bool:
    if not short or len(short) > len(long):
        return False
    if short == long:
        return True
    width = len(short)
    return any(long[index:index + width] == short for index in range(len(long) - width + 1))


def row_response(row: dict) -> str:
    return str((row.get("messages") or [{}, {}])[-1].get("content") or "")


def load_rows() -> dict[str, list[dict]]:
    rows = {}
    for split, filename in SPLITS.items():
        path = DATASET / filename
        rows[split] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows


def main() -> None:
    rows_by_split = load_rows()
    all_rows = [(split, row) for split, rows in rows_by_split.items() for row in rows]
    schema_errors = []
    suspicious = []
    anchor_groups = defaultdict(list)
    target_groups = defaultdict(list)
    episode_groups = defaultdict(list)
    full_groups = defaultdict(list)
    prompt_groups = defaultdict(list)
    cluster_by_split = defaultdict(set)
    near_duplicate_groups = []
    for split, row in all_rows:
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 2 or messages[0].get("role") != "user" or messages[-1].get("role") != "assistant":
            schema_errors.append({"split": split, "sample_id": row.get("sample_id"), "reason": "invalid_role_order"})
        if row.get("identity") not in TARGETS:
            schema_errors.append({"split": split, "sample_id": row.get("sample_id"), "reason": "assistant_identity_not_target"})
        if float((row.get("speaker_evidence") or {}).get("assistant_confidence") or 0.0) < 0.70:
            schema_errors.append({"split": split, "sample_id": row.get("sample_id"), "reason": "assistant_confidence_below_0.70"})
        if row.get("training_candidate") is not False:
            schema_errors.append({"split": split, "sample_id": row.get("sample_id"), "reason": "training_candidate_not_false"})
        if split == "sealed_eval" and row.get("evaluation_only") is not True:
            schema_errors.append({"split": split, "sample_id": row.get("sample_id"), "reason": "sealed_eval_not_evaluation_only"})
        if not row.get("context_turn_id") or not row.get("context_anchor_id") or not row.get("response_episode_id") or not row.get("target_turn_ids"):
            schema_errors.append({"split": split, "sample_id": row.get("sample_id"), "reason": "missing_raw_provenance"})
        if row.get("semantic_qa", {}).get("status") not in {"ACCEPT", "REPAIR_FROM_RAW"}:
            suspicious.append({"split": split, "sample_id": row.get("sample_id"), "reason": "semantic_qa_failure"})
        if row.get("semantic_quality") != "PASS" or row.get("context_integrity") != "PASS" or row.get("transcript_quality") != "PASS" or row.get("response_episode_integrity") != "PASS":
            suspicious.append({"split": split, "sample_id": row.get("sample_id"), "reason": "quality_gate_failure"})
        for message in messages or []:
            if text_quality(message.get("content"), 1)[0] != "PASS":
                suspicious.append({"split": split, "sample_id": row.get("sample_id"), "reason": "transcript_garbage"})
        indices = row.get("raw_timeline_indices") or []
        if indices and indices != list(range(min(indices), max(indices) + 1)):
            suspicious.append({"split": split, "sample_id": row.get("sample_id"), "reason": "skipped_boundary_stitching"})
        if len(row.get("target_turn_ids") or []) != len(set(row.get("target_turn_ids") or [])):
            suspicious.append({"split": split, "sample_id": row.get("sample_id"), "reason": "duplicate_target_inside_episode"})
        cluster = str(row.get("canonical_recording_id"))
        cluster_by_split[split].add(cluster)
        anchor_key = (cluster, str(row.get("context_anchor_id")))
        anchor_groups[anchor_key].append(row)
        for target_id in row.get("target_turn_ids") or []:
            target_groups[str(target_id)].append(row)
        episode_groups[str(row.get("response_episode_id"))].append(row)
        full_groups[(cluster, "\n".join(f"{m.get('role')}:{norm(m.get('content'))}" for m in messages or []))].append(row)
        prompt = norm((messages or [{}])[0].get("content"))
        prompt_groups[(cluster, prompt)].append(row)
    for key, group in anchor_groups.items():
        if len(group) > 1:
            suspicious.append({"reason": "duplicate_context_anchor", "anchor": key, "sample_ids": [row.get("sample_id") for row in group]})
    for key, group in target_groups.items():
        if len(group) > 1:
            suspicious.append({"reason": "target_turn_reuse", "target_turn_id": key, "sample_ids": [row.get("sample_id") for row in group]})
    for key, group in episode_groups.items():
        if len(group) > 1:
            suspicious.append({"reason": "duplicate_response_episode", "response_episode_id": key, "sample_ids": [row.get("sample_id") for row in group]})
    for key, group in full_groups.items():
        if len(group) > 1:
            suspicious.append({"reason": "exact_duplicate", "sample_ids": [row.get("sample_id") for row in group]})
    prefix_count = 0
    for key, group in prompt_groups.items():
        responses = [(row, tokens(row_response(row))) for row in group]
        for left, left_tokens in responses:
            for right, right_tokens in responses:
                if left is right or len(left_tokens) < 5 or len(left_tokens) >= len(right_tokens):
                    continue
                if contains_sequence(left_tokens, right_tokens):
                    prefix_count += 1
                    if len(near_duplicate_groups) < 20:
                        near_duplicate_groups.append({"reason": "prefix_or_token_containment", "cluster": key[0], "prompt": key[1], "short": left.get("sample_id"), "long": right.get("sample_id")})
    response_rows = sorted(all_rows, key=lambda item: (str(item[1].get("canonical_recording_id")), len(tokens(row_response(item[1]))), item[1].get("sample_id", "")))
    response_index = defaultdict(set)
    response_containment_count = 0
    for position, (_, row) in enumerate(response_rows):
        cluster = str(row.get("canonical_recording_id"))
        current = tokens(row_response(row))
        if len(current) >= 5:
            candidate_ids = set()
            for index in range(len(current) - 2):
                candidate_ids.update(response_index.get((cluster, tuple(current[index:index + 3])), set()))
            for candidate_id in candidate_ids:
                _, previous = response_rows[candidate_id]
                previous_tokens = tokens(row_response(previous))
                if len(previous_tokens) >= 5 and contains_sequence(previous_tokens, current):
                    response_containment_count += 1
                    if len(near_duplicate_groups) < 20:
                        near_duplicate_groups.append({"reason": "same-recording-response-token-containment", "short": previous.get("sample_id"), "long": row.get("sample_id")})
                    break
        if len(current) >= 5:
            for index in range(len(current) - 2):
                response_index.setdefault((cluster, tuple(current[index:index + 3])), set()).add(position)
    split_leakage = {
        "train_validation": sorted(cluster_by_split["train"] & cluster_by_split["validation"]),
        "train_sealed_eval": sorted(cluster_by_split["train"] & cluster_by_split["sealed_eval"]),
        "validation_sealed_eval": sorted(cluster_by_split["validation"] & cluster_by_split["sealed_eval"]),
    }
    train_counts = Counter(str(row.get("canonical_recording_id")) for row in rows_by_split["train"])
    report = {
        "schema_version": "1.0.0",
        "artifact_status": "VALIDATED_SFT_SEMANTIC_CLOSURE_V2",
        "dataset_dir": str(DATASET.relative_to(ROOT)),
        "sample_counts": {split: len(rows) for split, rows in rows_by_split.items()},
        "total_samples": len(all_rows),
        "json_schema_errors": schema_errors,
        "suspicious_rows": suspicious,
        "duplicate_context_anchor_count": sum(1 for group in anchor_groups.values() if len(group) > 1),
        "target_turn_reuse_count": sum(1 for group in target_groups.values() if len(group) > 1),
        "duplicate_response_episode_count": sum(1 for group in episode_groups.values() if len(group) > 1),
        "exact_duplicate_count": sum(1 for group in full_groups.values() if len(group) > 1),
        "prefix_containment_count": prefix_count,
        "near_duplicate_screen": {"method": "same-recording normalized-prompt and response token-sequence containment", "group_examples": near_duplicate_groups, "response_containment_count": response_containment_count},
        "recording_cluster_leakage": split_leakage,
        "recording_cluster_counts_train": dict(train_counts),
        "largest_train_cluster_share": round(max(train_counts.values()) / len(rows_by_split["train"],), 6) if train_counts and rows_by_split["train"] else 0.0,
        "training_candidate_true_count": sum(1 for _, row in all_rows if row.get("training_candidate") is True),
        "pass": not schema_errors and not suspicious and prefix_count == 0 and response_containment_count == 0 and not any(split_leakage.values()),
    }
    write_json(DATASET / "validation_report.json", report)
    print(json.dumps({"status": "PASS" if report["pass"] else "FAIL", "sample_counts": report["sample_counts"], "schema_errors": len(schema_errors), "suspicious": len(suspicious), "prefix_containment": prefix_count, "leakage": split_leakage, "training_candidate_true_count": report["training_candidate_true_count"]}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
