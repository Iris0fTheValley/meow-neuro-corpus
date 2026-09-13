from __future__ import annotations

"""Build reproducible train views from the verified interaction pool.

Quality closure is upstream. This module may sample or balance eligible facts,
but cannot upgrade identity, transcript, or semantic verification status.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "datasets" / "meow_v02_sft_v2_2_semantic_verified"
sys.path.insert(0, str(ROOT / "scripts"))
import sft_interaction_semantic_closure as closure  # noqa: E402

SAMPLING_SCHEMA_VERSION = "1.0.0"
SAMPLING_VERSION = "verified-pool-views-v1-2026-09-13"


def load_jsonl(path: Path) -> list[dict]:
    values = []
    if not path.exists():
        return values
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except Exception:
                continue
            if isinstance(value, dict):
                values.append(value)
    return values


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def family_splits(rows: list[dict], validation_share: float = 0.08, sealed_share: float = 0.08) -> dict[str, str]:
    counts = Counter(str(row.get("recording_family_id")) for row in rows)
    total = sum(counts.values())
    ordered = sorted(counts, key=lambda family: hashlib.sha256(("split-v1|" + family).encode()).hexdigest())
    assignment = {family: "train" for family in ordered}
    if len(ordered) < 6:
        return assignment
    targets = {"validation": total * validation_share, "sealed_eval": total * sealed_share}
    allocated = Counter()
    # Largest families stay in train; independent holdouts are filled from the
    # deterministic remainder and are sealed before any sampling policy runs.
    largest = {family for family, _ in counts.most_common(2)}
    for family in [value for value in ordered if value not in largest]:
        candidates = [name for name in ("validation", "sealed_eval") if allocated[name] < targets[name]]
        if not candidates:
            break
        split = min(candidates, key=lambda name: (allocated[name] / max(1.0, targets[name]), name))
        assignment[family] = split
        allocated[split] += counts[family]
    return assignment


def apply_policy(rows: list[dict], policy: str, max_per_family: int) -> list[dict]:
    eligible = [row for row in rows if row.get("verified_pool_status") == "VERIFIED_HARD_DEDUPED" and row.get("sampling_eligibility") == "ELIGIBLE"]
    if policy == "natural_frequency":
        return eligible
    if policy == "high_precision":
        return [row for row in eligible if float((row.get("semantic_qa") or {}).get("confidence") or 0.0) >= 0.90 and row.get("transcript_quality") == "PASS"]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in eligible:
        grouped[str(row.get("recording_family_id"))].append(row)
    selected = []
    for family, family_rows in sorted(grouped.items()):
        family_rows.sort(key=lambda row: hashlib.sha256((policy + "|" + str(row.get("sample_id"))).encode()).hexdigest())
        selected.extend(family_rows[:max_per_family])
    return selected


def stable_family_splits(rows: list[dict], manifest_path: Path, validation_share: float, sealed_share: float, rebuild: bool = False) -> tuple[dict[str, str], bool]:
    fresh = family_splits(rows, validation_share, sealed_share)
    if rebuild or not manifest_path.exists():
        return fresh, False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"existing split manifest is unreadable: {exc}")
    assignment: dict[str, str] = {}
    for split in ("train", "validation", "sealed_eval"):
        for family in (manifest.get("families") or {}).get(split) or []:
            if family in assignment:
                raise SystemExit(f"recording family appears in multiple persisted splits: {family}")
            assignment[str(family)] = split
    for family, split in fresh.items():
        assignment.setdefault(family, split)
    return assignment, True


def build(args: argparse.Namespace) -> dict:
    pool = load_jsonl(Path(args.pool))
    if not pool:
        raise SystemExit(f"verified pool is missing or empty: {args.pool}")
    incompatible = [row.get("sample_id") for row in pool if row.get("artifact_schema_version") != closure.SCHEMA_VERSION]
    if incompatible:
        raise SystemExit(f"verified pool schema mismatch for {len(incompatible)} rows")
    output = Path(args.output)
    assignments, reused_splits = stable_family_splits(pool, output / "view_manifest.json", args.validation_share, args.sealed_share, args.rebuild_splits)
    partitioned = defaultdict(list)
    for source in pool:
        row = dict(source)
        split = assignments[str(row.get("recording_family_id"))]
        row["split"] = split
        row["evaluation_only"] = split == "sealed_eval"
        row["sampling_policy"] = {"schema_version": SAMPLING_SCHEMA_VERSION, "version": SAMPLING_VERSION, "policy": args.policy}
        partitioned[split].append(row)
    sampled_train = apply_policy(partitioned["train"], args.policy, args.max_per_family)
    write_jsonl(output / f"train_{args.policy}.jsonl", sorted(sampled_train, key=lambda row: str(row.get("sample_id"))))
    write_jsonl(output / "validation.jsonl", sorted(partitioned["validation"], key=lambda row: str(row.get("sample_id"))))
    write_jsonl(output / "sealed_eval.jsonl", sorted(partitioned["sealed_eval"], key=lambda row: str(row.get("sample_id"))))
    manifest = {
        "schema_version": SAMPLING_SCHEMA_VERSION,
        "sampling_version": SAMPLING_VERSION,
        "source_pool_schema_version": closure.SCHEMA_VERSION,
        "source_pool_pipeline_version": closure.PIPELINE_VERSION,
        "policy": {"name": args.policy, "max_per_family": args.max_per_family if args.policy == "recording_balanced" else None},
        "quality_and_sampling_separated": True,
        "sealed_eval_selected_before_sampling": True,
        "persisted_split_assignment_reused": reused_splits,
        "counts": {"verified_pool": len(pool), "train_eligible_before_policy": len(partitioned["train"]), "train_view": len(sampled_train), "validation": len(partitioned["validation"]), "sealed_eval": len(partitioned["sealed_eval"])},
        "families": {name: sorted(family for family, split in assignments.items() if split == name) for name in ("train", "validation", "sealed_eval")},
        "files": {"train": f"train_{args.policy}.jsonl", "validation": "validation.jsonl", "sealed_eval": "sealed_eval.jsonl"},
        "training_candidate_true_count": 0,
        "ready_for_first_sft": False,
    }
    (output / "view_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a sampling view from the verified interaction pool")
    parser.add_argument("--pool", default=str(DEFAULT_DATASET / "verified_interaction_pool_v2_3.jsonl"))
    parser.add_argument("--output", default=str(DEFAULT_DATASET / "views" / "natural_frequency"))
    parser.add_argument("--policy", choices=["natural_frequency", "recording_balanced", "high_precision"], default="natural_frequency")
    parser.add_argument("--max-per-family", type=int, default=1000)
    parser.add_argument("--validation-share", type=float, default=0.08)
    parser.add_argument("--sealed-share", type=float, default=0.08)
    parser.add_argument("--rebuild-splits", action="store_true", help="explicitly replace persisted family assignments; never use after sealed-eval tuning")
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
