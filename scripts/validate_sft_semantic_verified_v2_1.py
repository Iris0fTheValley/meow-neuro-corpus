from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

try:
    from manifest_tools import ROOT, write_json
    from sft_semantic_closure_v2_core import TARGETS, text_quality
except ModuleNotFoundError:
    from scripts.manifest_tools import ROOT, write_json
    from scripts.sft_semantic_closure_v2_core import TARGETS, text_quality


DATASET = ROOT / "datasets" / "meow_v02_sft_v2_1_semantic_verified"
SPLITS = {"train": "train.jsonl", "validation": "validation.jsonl", "sealed_eval": "sealed_eval.jsonl"}


def load_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def norm(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", norm(value))


def contains(short: list[str], long: list[str]) -> bool:
    if not short or len(short) > len(long):
        return False
    return any(long[index:index + len(short)] == short for index in range(len(long) - len(short) + 1))


def main() -> None:
    rows_by_split = {split: load_jsonl(DATASET / filename) for split, filename in SPLITS.items()}
    all_rows = [(split, row) for split, rows in rows_by_split.items() for row in rows]
    schema_errors = []
    suspicious = []
    anchors = defaultdict(list)
    targets = defaultdict(list)
    episodes = defaultdict(list)
    full = defaultdict(list)
    split_families = defaultdict(set)
    prompt_groups = defaultdict(list)
    for split, row in all_rows:
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) != 2 or messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
            schema_errors.append({"sample_id": row.get("sample_id"), "reason": "message_schema"})
        if row.get("identity") not in TARGETS or float((row.get("speaker_evidence") or {}).get("assistant_confidence") or 0.0) < 0.70:
            schema_errors.append({"sample_id": row.get("sample_id"), "reason": "assistant_identity_gate"})
        if row.get("training_candidate") is not False:
            schema_errors.append({"sample_id": row.get("sample_id"), "reason": "training_candidate_not_false"})
        if split == "sealed_eval" and row.get("evaluation_only") is not True:
            schema_errors.append({"sample_id": row.get("sample_id"), "reason": "sealed_eval_flag"})
        for key in ("context_turn_id", "response_episode_id", "target_turn_ids", "recording_family_id"):
            if not row.get(key):
                schema_errors.append({"sample_id": row.get("sample_id"), "reason": f"missing_{key}"})
        for key in ("semantic_quality", "context_integrity", "transcript_quality", "response_episode_integrity"):
            if row.get(key) != "PASS":
                suspicious.append({"sample_id": row.get("sample_id"), "reason": f"{key}_not_pass"})
        qa = row.get("semantic_qa") or {}
        if qa.get("relation") not in {"DIRECT_RESPONSE", "CONTEXTUAL_RESPONSE", "GAME_STATE_RESPONSE"} or not qa.get("judge_model") or float(qa.get("confidence") or 0) < 0.75:
            suspicious.append({"sample_id": row.get("sample_id"), "reason": "semantic_judge_evidence"})
        for message in messages or []:
            if text_quality(message.get("content"), 1)[0] != "PASS":
                suspicious.append({"sample_id": row.get("sample_id"), "reason": "transcript_garbage"})
        target_ids = [str(value) for value in row.get("target_turn_ids") or []]
        if len(target_ids) != len(set(target_ids)):
            suspicious.append({"sample_id": row.get("sample_id"), "reason": "duplicate_target_inside_episode"})
        family = str(row.get("recording_family_id"))
        split_families[split].add(family)
        anchors[(family, str(row.get("context_anchor_id")))].append(row)
        episodes[str(row.get("response_episode_id"))].append(row)
        for target_id in target_ids:
            targets[target_id].append(row)
        full[(family, "\n".join(f"{m.get('role')}:{norm(m.get('content'))}" for m in messages or []))].append(row)
        prompt = norm((messages or [{}])[0].get("content"))
        prompt_groups[(family, prompt)].append(row)
    for key, group in anchors.items():
        if len(group) > 1:
            suspicious.append({"reason": "duplicate_context_anchor", "key": key})
    for key, group in targets.items():
        if len(group) > 1:
            suspicious.append({"reason": "target_turn_reuse", "target": key})
    for key, group in episodes.items():
        if len(group) > 1:
            suspicious.append({"reason": "duplicate_response_episode", "episode": key})
    exact_count = sum(1 for group in full.values() if len(group) > 1)
    if exact_count:
        suspicious.append({"reason": "global_exact_duplicate", "count": exact_count})
    prefix_count = 0
    for key, group in prompt_groups.items():
        values = [(row, tokens((row.get("messages") or [{}, {}])[-1].get("content"))) for row in group]
        for left, left_tokens in values:
            for right, right_tokens in values:
                if left is right or len(left_tokens) < 5 or len(left_tokens) >= len(right_tokens):
                    continue
                if contains(left_tokens, right_tokens):
                    prefix_count += 1
    leakage = {"train_validation": sorted(split_families["train"] & split_families["validation"]), "train_sealed_eval": sorted(split_families["train"] & split_families["sealed_eval"]), "validation_sealed_eval": sorted(split_families["validation"] & split_families["sealed_eval"])}
    overlap_edges = load_jsonl(DATASET / "recording_overlap_edges.jsonl")
    row_split = {str(row.get("sample_id")): split for split, rows in rows_by_split.items() for row in rows}
    strong_cross_split = []
    for edge in overlap_edges:
        if edge.get("strong_merge_edge") and row_split.get(str(edge.get("sample_a"))) != row_split.get(str(edge.get("sample_b"))):
            strong_cross_split.append(edge)
    report = {
        "schema_version": "1.0.0",
        "artifact_status": "VALIDATED_SFT_SEMANTIC_VERIFIED_V2_1",
        "sample_counts": {split: len(rows) for split, rows in rows_by_split.items()},
        "v2_structural_candidates": len(load_jsonl(DATASET / "structural_candidates_v2.jsonl")),
        "json_schema_errors": schema_errors,
        "suspicious_rows": suspicious,
        "duplicate_context_anchor_count": sum(1 for group in anchors.values() if len(group) > 1),
        "target_turn_reuse_count": sum(1 for group in targets.values() if len(group) > 1),
        "duplicate_response_episode_count": sum(1 for group in episodes.values() if len(group) > 1),
        "global_exact_duplicate_count": exact_count,
        "same_anchor_prefix_ladder_count": prefix_count,
        "recording_family_leakage": leakage,
        "cross_split_strong_overlap_count": len(strong_cross_split),
        "quality_layers": {"structural": not schema_errors, "semantic": not any(item.get("reason") == "semantic_judge_evidence" for item in suspicious), "transcript": not any(item.get("reason") == "transcript_garbage" for item in suspicious), "episode": not any(item.get("reason") == "response_episode_integrity_not_pass" for item in suspicious), "recording_independence": not any(leakage.values()) and not strong_cross_split},
        "training_candidate_true_count": sum(1 for _, row in all_rows if row.get("training_candidate") is True),
    }
    report["pass"] = not schema_errors and not suspicious and not any(leakage.values()) and not strong_cross_split
    write_json(DATASET / "validation_report.json", report)
    print(json.dumps({"status": "PASS" if report["pass"] else "FAIL", "sample_counts": report["sample_counts"], "schema_errors": len(schema_errors), "suspicious": len(suspicious), "prefix_ladder": prefix_count, "target_reuse": report["target_turn_reuse_count"], "leakage": leakage, "cross_split_strong_overlap": len(strong_cross_split), "training_candidate_true_count": report["training_candidate_true_count"]}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
