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

SAMPLING_SCHEMA_VERSION = "2.0.0"
SAMPLING_VERSION = "verified-pool-views-v2-2026-09-13"
SPLIT_AUTHORITY_VERSION = "recording-lineage-split-authority-v1-2026-09-13"


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


def _family_members(rows: list[dict], repair: dict) -> dict[str, set[str]]:
    present = {str(row.get("recording_family_id")) for row in rows}
    result = {
        str(item.get("recording_family_id")): {str(value) for value in item.get("canonical_recording_ids") or []}
        for item in repair.get("recording_families") or []
        if str(item.get("recording_family_id")) in present
    }
    for row in rows:
        result.setdefault(str(row.get("recording_family_id")), set()).add(str(row.get("canonical_recording_id")))
    return result


def _constraint_groups(families: set[str], relations: list[dict]) -> list[set[str]]:
    parent = {family: family for family in families}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for relation in relations:
        if not relation.get("must_share_partition"):
            continue
        left, right = str(relation.get("family_a")), str(relation.get("family_b"))
        if left not in parent or right not in parent:
            continue
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a
    groups: dict[str, set[str]] = defaultdict(set)
    for family in families:
        groups[find(family)].add(family)
    return list(groups.values())


def derive_split_authority(rows: list[dict], repair: dict, existing: dict | None, validation_share: float = 0.08, sealed_share: float = 0.08) -> tuple[dict[str, str], dict]:
    """Resolve split authority from stable recording members and exclusions."""
    existing = existing or {}
    if existing and (existing.get("schema_version") != SAMPLING_SCHEMA_VERSION or existing.get("split_authority_version") != SPLIT_AUTHORITY_VERSION):
        raise SystemExit("split authority schema/version mismatch; migrate explicitly instead of reinterpreting it")
    members = _family_members(rows, repair)
    historical = {str(member): str(split) for member, split in (existing.get("member_assignments") or {}).items()}
    fresh = family_splits(rows, validation_share, sealed_share)
    relations = list(repair.get("split_exclusion_relations") or [])
    assignments: dict[str, str] = {}
    conflicts = []
    priority = {"train": 0, "validation": 1, "sealed_eval": 2}
    for group in _constraint_groups(set(members), relations):
        inherited = {historical[member] for family in group for member in members[family] if member in historical}
        if len(inherited) > 1:
            conflicts.append({"state": "SPLIT_LINEAGE_CONFLICT", "recording_families": sorted(group), "underlying_members": sorted({member for family in group for member in members[family]}), "historical_splits": sorted(inherited)})
            for family in group:
                assignments[family] = "quarantine"
            continue
        if inherited:
            split = next(iter(inherited))
        else:
            split = max((fresh.get(family, "train") for family in group), key=lambda value: priority[value])
        for family in group:
            assignments[family] = split
    updated_members = dict(historical)
    if not conflicts:
        for family, split in assignments.items():
            for member in members[family]:
                updated_members[member] = split
    authority = {
        "schema_version": SAMPLING_SCHEMA_VERSION,
        "split_authority_version": SPLIT_AUTHORITY_VERSION,
        "status": "SPLIT_LINEAGE_CONFLICT" if conflicts else "ACTIVE",
        "authority_domain": "RECORDING_LINEAGE_AND_SPLIT_EXCLUSION",
        "member_assignments": updated_members,
        "current_family_members": {family: sorted(values) for family, values in sorted(members.items())},
        "current_family_assignments": assignments,
        "split_exclusion_relations": relations,
        "conflicts": conflicts,
        "sealed_isolation_precedes_sampling": True,
    }
    return assignments, authority


def build(args: argparse.Namespace) -> dict:
    pool = load_jsonl(Path(args.pool))
    if not pool:
        raise SystemExit(f"verified pool is missing or empty: {args.pool}")
    incompatible = [row.get("sample_id") for row in pool if row.get("artifact_schema_version") != closure.SCHEMA_VERSION]
    if incompatible:
        raise SystemExit(f"verified pool schema mismatch for {len(incompatible)} rows")
    output = Path(args.output) if args.output else DEFAULT_DATASET / "views" / args.policy
    repair_path = Path(args.recording_repair)
    try:
        repair = json.loads(repair_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"recording repair is missing or unreadable: {exc}")
    if repair.get("schema_version") != closure.SCHEMA_VERSION:
        raise SystemExit("recording repair schema mismatch")
    authority_path = Path(args.split_authority)
    try:
        existing_authority = json.loads(authority_path.read_text(encoding="utf-8")) if authority_path.exists() else {}
    except Exception as exc:
        raise SystemExit(f"split authority is unreadable: {exc}")
    assignments, authority = derive_split_authority(pool, repair, existing_authority, args.validation_share, args.sealed_share)
    authority_path.parent.mkdir(parents=True, exist_ok=True)
    authority_path.write_text(json.dumps(authority, ensure_ascii=False, indent=2), encoding="utf-8")
    if authority["conflicts"]:
        raise SystemExit(f"SPLIT_LINEAGE_CONFLICT: {len(authority['conflicts'])} conflict group(s); no view was built")
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
        "split_authority": str(authority_path),
        "split_authority_version": SPLIT_AUTHORITY_VERSION,
        "shared_dataset_level_split_authority": True,
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
    parser.add_argument("--output", default="", help="defaults to dataset/views/<policy>")
    parser.add_argument("--policy", choices=["natural_frequency", "recording_balanced", "high_precision"], default="natural_frequency")
    parser.add_argument("--max-per-family", type=int, default=1000)
    parser.add_argument("--validation-share", type=float, default=0.08)
    parser.add_argument("--sealed-share", type=float, default=0.08)
    parser.add_argument("--recording-repair", default=str(DEFAULT_DATASET / "recording_family_repair_v2_3.json"))
    parser.add_argument("--split-authority", default=str(DEFAULT_DATASET / "split_authority_v2_3.json"))
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
