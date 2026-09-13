from __future__ import annotations

"""Independent validator for the v2.2 semantic-verified candidate."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets" / "meow_v02_sft_v2_2_semantic_verified"


def rows(path: Path):
    if not path.exists():
        return []
    result = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except Exception:
                continue
            if isinstance(value, dict):
                result.append(value)
    return result


def norm(value) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def tokens(value) -> list[str]:
    return re.findall(r"[a-z0-9]+", norm(value))


def message_key(row: dict) -> str:
    return "\n".join(f"{item.get('role')}:{norm(item.get('content'))}" for item in row.get("messages") or [])


def response(row: dict) -> str:
    messages = row.get("messages") or []
    return str(messages[-1].get("content") or "") if messages else ""


def contains(short: list[str], long: list[str]) -> bool:
    if not short or len(short) > len(long):
        return False
    return any(long[i:i + len(short)] == short for i in range(len(long) - len(short) + 1))


def validate() -> dict:
    split_rows = {name: rows(OUT / file) for name, file in (("train", "train.jsonl"), ("validation", "validation.jsonl"), ("sealed_eval", "sealed_eval.jsonl"))}
    all_rows = [row for values in split_rows.values() for row in values]
    schema_errors = []
    required = {"sample_id", "messages", "source_id", "canonical_recording_id", "recording_family_id", "context_anchor_id", "response_episode_id", "target_turn_ids", "timestamps", "speaker_evidence", "semantic_qa", "transcript_qa", "pipeline_version"}
    for row in all_rows:
        missing = sorted(required - set(row))
        messages = row.get("messages") or []
        if missing or len(messages) < 2 or any(not str(item.get("content") or "").strip() for item in messages):
            schema_errors.append({"sample_id": row.get("sample_id"), "missing": missing, "message_count": len(messages)})
    anchor_ids = defaultdict(list)
    target_ids = defaultdict(list)
    exact = defaultdict(list)
    response_rows = []
    for row in all_rows:
        anchor_ids[(str(row.get("recording_family_id")), str(row.get("context_anchor_id")))].append(row.get("sample_id"))
        for target_id in row.get("target_turn_ids") or []:
            target_ids[str(target_id)].append(row.get("sample_id"))
        exact[message_key(row)].append(row.get("sample_id"))
        response_rows.append((row, tokens(response(row))))
    duplicate_anchor = [values for values in anchor_ids.values() if len(values) > 1]
    target_reuse = [values for values in target_ids.values() if len(values) > 1]
    exact_dup = [values for values in exact.values() if len(values) > 1]
    prefix = []
    by_anchor = defaultdict(list)
    for row, toks in response_rows:
        by_anchor[(str(row.get("recording_family_id")), str(row.get("context_anchor_id")))].append((row, toks))
    for key, values in by_anchor.items():
        for i, (left, left_tokens) in enumerate(values):
            for right, right_tokens in values[i + 1:]:
                if left_tokens != right_tokens and (contains(left_tokens, right_tokens) or contains(right_tokens, left_tokens)):
                    prefix.append({"anchor": key, "left": left.get("sample_id"), "right": right.get("sample_id")})
    strong_cross_split = []
    # Independent split leakage screen: shared 12-token contiguous text across
    # different splits. Short common phrases are ignored.
    split_sequences = []
    for split, values in split_rows.items():
        for row in values:
            seq = tokens(" ".join(str(item.get("content") or "") for item in row.get("messages") or []))
            if len(seq) >= 12:
                split_sequences.append((split, row, seq))
    shingle_index = defaultdict(list)
    for split, row, seq in split_sequences:
        for i in range(len(seq) - 11):
            shingle_index[tuple(seq[i:i + 12])].append((split, row, seq))
    seen_pairs = set()
    for values in shingle_index.values():
        for i, left in enumerate(values):
            for right in values[i + 1:]:
                if left[0] == right[0]:
                    continue
                pair = tuple(sorted((str(left[1].get("sample_id")), str(right[1].get("sample_id")))))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                strong_cross_split.append({"split_a": left[0], "split_b": right[0], "sample_a": pair[0], "sample_b": pair[1], "shared_tokens": 12})
    family_sets = {split: {str(row.get("recording_family_id")) for row in values} for split, values in split_rows.items()}
    family_leakage = {"train_validation": sorted(family_sets["train"] & family_sets["validation"]), "train_sealed_eval": sorted(family_sets["train"] & family_sets["sealed_eval"]), "validation_sealed_eval": sorted(family_sets["validation"] & family_sets["sealed_eval"])}
    training_flags = [row.get("sample_id") for row in all_rows if row.get("training_candidate") is True]
    quality_layers = {
        "structural": not schema_errors and not duplicate_anchor and not target_reuse and not prefix,
        "semantic": all((row.get("semantic_qa") or {}).get("status") == "MODEL_ACCEPT" for row in all_rows),
        "transcript": all(row.get("transcript_quality") == "PASS" for row in all_rows),
        "episode": all(row.get("response_episode_integrity") == "PASS" for row in all_rows),
        "recording_independence": not any(family_leakage.values()) and not strong_cross_split,
    }
    report = {
        "schema_version": "1.0.0",
        "artifact_status": "VALIDATED_SFT_SEMANTIC_VERIFIED_V2_2",
        "sample_counts": {split: len(values) for split, values in split_rows.items()},
        "schema_errors": schema_errors[:100],
        "duplicate_context_anchor_count": len(duplicate_anchor),
        "target_turn_reuse_count": len(target_reuse),
        "global_exact_duplicate_count": len(exact_dup),
        "same_anchor_prefix_ladder_count": len(prefix),
        "recording_family_leakage": family_leakage,
        "cross_split_strong_overlap_count": len(strong_cross_split),
        "cross_split_strong_overlap_examples": strong_cross_split[:100],
        "quality_layers": quality_layers,
        "training_candidate_true_count": len(training_flags),
        "pass": all(quality_layers.values()) and not training_flags,
    }
    (OUT / "validation_report_v2_2.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUT))
    parser.parse_args()
    report = validate()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
