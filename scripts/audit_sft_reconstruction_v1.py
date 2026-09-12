from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from manifest_tools import ROOT, write_json


INPUT = ROOT / "datasets" / "meow_v02_sft_v1" / "final_sft_candidate_v1" / "train.jsonl"
OUT = ROOT / "reports" / "sft_reconstruction_v1_root_cause.json"


def norm(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def main() -> None:
    rows = [json.loads(line) for line in INPUT.read_text(encoding="utf-8").splitlines() if line.strip()]
    prompts: dict[str, list[dict]] = defaultdict(list)
    anchors: dict[tuple[str, str], list[dict]] = defaultdict(list)
    targets: dict[str, list[dict]] = defaultdict(list)
    clusters = Counter()
    prefix_pairs = []
    for row in rows:
        messages = row.get("messages") or []
        prompt = norm(messages[0].get("content") if messages else "")
        response = norm(messages[-1].get("content") if messages else "")
        prompts[prompt].append(row)
        recovery = row.get("raw_recovery") or {}
        target_id = str(recovery.get("target_turn_id") or "")
        anchors[(str(row.get("source_id")), target_id)].append(row)
        if target_id:
            targets[target_id].append(row)
        clusters[str(row.get("canonical_recording_id"))] += 1
    for prompt, group in prompts.items():
        responses = [(row, norm((row.get("messages") or [])[-1].get("content"))) for row in group]
        for left, left_response in responses:
            for right, right_response in responses:
                if left is right or not left_response or left_response == right_response:
                    continue
                if len(left_response) < len(right_response) and right_response.startswith(left_response):
                    prefix_pairs.append({"prompt": prompt, "short_sample_id": left.get("sample_id"), "long_sample_id": right.get("sample_id")})
    duplicate_prompt_rows = sum(len(group) for group in prompts.values() if len(group) > 1)
    result = {
        "schema_version": "1.0.0",
        "artifact": str(INPUT.relative_to(ROOT)),
        "rows": len(rows),
        "unique_user_prompts": len(prompts),
        "duplicate_prompt_groups": sum(1 for group in prompts.values() if len(group) > 1),
        "rows_in_duplicate_prompt_groups": duplicate_prompt_rows,
        "rows_in_duplicate_prompt_groups_fraction": round(duplicate_prompt_rows / len(rows), 6) if rows else 0.0,
        "context_anchor_duplicate_count": sum(1 for group in anchors.values() if len(group) > 1),
        "target_turn_reuse_count": sum(1 for key, group in targets.items() if key and len(group) > 1),
        "prefix_containment_pair_count": len(prefix_pairs),
        "prefix_containment_examples": prefix_pairs[:20],
        "recording_cluster_rows": dict(clusters),
        "largest_cluster_share": round(max(clusters.values()) / len(rows), 6) if rows else 0.0,
        "top_3_cluster_share": round(sum(value for _, value in clusters.most_common(3)) / len(rows), 6) if rows else 0.0,
        "root_causes": [
            "target_turn_independent_sample_generation",
            "same_role_target_concatenation_without_response_episode_boundary",
            "bad_or_uncertain_middle_turns_skipped_as_transparent",
            "exact_message_and_response-only_deduplication",
            "rank_based_cluster_split_and_concentration",
        ],
    }
    write_json(OUT, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
