from __future__ import annotations

"""Independent structural and contamination checks for the final SFT artifact."""

import hashlib
import json
import re
import argparse
from collections import Counter
from pathlib import Path

from manifest_tools import ROOT, write_json


DATASET = ROOT / "datasets" / "final_meow_v02_sft_v1"
SPLITS = {"train": "train.jsonl", "validation": "validation.jsonl", "sealed_eval": "sealed_eval.jsonl"}


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").casefold()).strip()


def load(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            row["_line_number"] = line_number
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DATASET)
    args = parser.parse_args()
    dataset = args.dataset_dir if args.dataset_dir.is_absolute() else ROOT / args.dataset_dir
    rows_by_split = {split: load(dataset / filename) for split, filename in SPLITS.items()}
    all_rows = [row for rows in rows_by_split.values() for row in rows]
    exact_keys = Counter()
    response_keys = Counter()
    invalid_schema = []
    suspicious = []
    for split, rows in rows_by_split.items():
        for row in rows:
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages or messages[0].get("role") != "user" or messages[-1].get("role") != "assistant":
                invalid_schema.append({"split": split, "sample_id": row.get("sample_id"), "reason": "invalid_role_sequence"})
                continue
            if not str(messages[-1].get("content") or "").strip():
                invalid_schema.append({"split": split, "sample_id": row.get("sample_id"), "reason": "empty_assistant"})
            key = "\n".join(str(item.get("role")) + ":" + norm(item.get("content")) for item in messages)
            exact_keys[hashlib.sha256(key.encode("utf-8")).hexdigest()] += 1
            response_keys[norm(messages[-1].get("content"))] += 1
            text = norm(messages[-1].get("content"))
            if text in {"music", "[music]", "♪", "♫", "🎵", "🎶"} or len(re.findall(r"[a-zA-Z]{2,}", text)) < 2:
                suspicious.append({"split": split, "sample_id": row.get("sample_id"), "reason": "garbage_or_music_like_assistant"})
            if row.get("identity") not in {"NEURO_FAMILY_HIGH", "NEURO_FAMILY_MEDIUM"}:
                suspicious.append({"split": split, "sample_id": row.get("sample_id"), "reason": "non_target_identity"})
            if row.get("training_candidate") is True:
                suspicious.append({"split": split, "sample_id": row.get("sample_id"), "reason": "training_candidate_true_forbidden"})

    clusters = {split: {row.get("canonical_recording_id") for row in rows} for split, rows in rows_by_split.items()}
    leakage = {
        "train_validation": sorted(clusters["train"] & clusters["validation"]),
        "train_sealed_eval": sorted(clusters["train"] & clusters["sealed_eval"]),
        "validation_sealed_eval": sorted(clusters["validation"] & clusters["sealed_eval"]),
    }
    duplicate_keys = sorted(key for key, count in exact_keys.items() if count > 1)
    repeated_responses = sorted((key, count) for key, count in response_keys.items() if count > 1)
    try:
        dataset_label = str(dataset.relative_to(ROOT))
    except ValueError:
        dataset_label = str(dataset)
    report = {
        "schema_version": "1.0.0",
        "artifact_status": "VALIDATED_FINAL_FIRST_SFT_CANDIDATE_DATASET",
        "dataset_dir": dataset_label,
        "sample_counts": {split: len(rows) for split, rows in rows_by_split.items()},
        "total_samples": len(all_rows),
        "json_schema_errors": invalid_schema,
        "suspicious_rows": suspicious,
        "exact_duplicate_count": len(duplicate_keys),
        "near_duplicate_screen": {"method": "normalized_assistant_response_collision", "repeated_response_key_count": len(repeated_responses), "repeated_response_keys_top10": repeated_responses[:10]},
        "recording_cluster_leakage": leakage,
        "training_candidate_true_count": sum(1 for row in all_rows if row.get("training_candidate") is True),
        "pass": not invalid_schema and not suspicious and not duplicate_keys and not any(leakage.values()),
    }
    write_json(dataset / "validation_report.json", report)
    try:
        validation_path = str((dataset / "validation_report.json").relative_to(ROOT))
    except ValueError:
        validation_path = str(dataset / "validation_report.json")
    print(json.dumps({"status": "PASS" if report["pass"] else "FAIL", "sample_counts": report["sample_counts"], "exact_duplicates": report["exact_duplicate_count"], "repeated_response_keys": len(repeated_responses), "leakage": leakage, "suspicious_rows": len(suspicious), "validation_report": validation_path}, ensure_ascii=False, indent=2))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
