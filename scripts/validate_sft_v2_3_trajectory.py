from __future__ import annotations

"""Validator for reconstructed trajectory rows and the combined train view."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def validate(args: argparse.Namespace) -> dict[str, Any]:
    trajectories = load(Path(args.trajectory_pool))
    pending = load(Path(args.pending))
    quarantine = load(Path(args.quarantine))
    combined = load(Path(args.combined_pool))
    view_root = Path(args.view)
    view_rows = []
    for path in list(view_root.glob("train_*.jsonl")) + [view_root / "validation.jsonl", view_root / "sealed_eval.jsonl"]:
        view_rows.extend(load(path))
    errors: list[dict[str, Any]] = []
    target_seen: dict[str, str] = {}
    for row in trajectories:
        sid = str(row.get("sample_id"))
        if row.get("verified_pool_status") != "VERIFIED_RECONSTRUCTED_CONTEXT" or row.get("sampling_eligibility") != "TRAJECTORY_ELIGIBLE":
            errors.append({"sample_id": sid, "reason": "TRAJECTORY_STATUS_INVALID"})
        tc = row.get("trajectory_reconstruction") or {}
        if tc.get("status") != "VERIFIED_RECONSTRUCTED_CONTEXT" or not tc.get("semantic_truth_preserved"):
            errors.append({"sample_id": sid, "reason": "TRAJECTORY_VERIFICATION_MISSING"})
        context_ids = [str(value) for value in row.get("context_turn_ids") or []]
        target_ids = [str(value) for value in row.get("target_turn_ids") or []]
        tqa = row.get("transcript_qa") or {}
        if context_ids + target_ids != [str(value) for value in tqa.get("final_turn_ids") or []]:
            errors.append({"sample_id": sid, "reason": "FINAL_TURN_PROVENANCE_MISMATCH"})
        if len(tqa.get("turns") or []) != len(context_ids) + len(target_ids):
            errors.append({"sample_id": sid, "reason": "TRANSCRIPT_PROVENANCE_COUNT_MISMATCH"})
        if not tqa.get("all_selected_turns_machine_usable") or tqa.get("semantic_verifier_cannot_rewrite_transcript") is not True:
            errors.append({"sample_id": sid, "reason": "MACHINE_GATE_OR_REWRITE_FLAG_INVALID"})
        for tid in target_ids:
            if tid in target_seen:
                errors.append({"sample_id": sid, "reason": "TARGET_TURN_REUSE", "other": target_seen[tid]})
            target_seen[tid] = sid
    trajectory_ids = {str(row.get("sample_id")) for row in trajectories}
    if trajectory_ids & {str(row.get("sample_id")) for row in pending + quarantine}:
        errors.append({"reason": "TRAJECTORY_PARTITION_OVERLAP"})
    if len(combined) != len({str(row.get("sample_id")) for row in combined}):
        errors.append({"reason": "COMBINED_SAMPLE_DUPLICATE"})
    view_trajectory_ids = {str(row.get("sample_id")) for row in view_rows if row.get("sampling_eligibility") == "TRAJECTORY_ELIGIBLE"}
    if not view_trajectory_ids.issubset(trajectory_ids):
        errors.append({"reason": "VIEW_CONTAINS_UNKNOWN_TRAJECTORY"})
    report = {
        "schema_version": "2.0.0",
        "trajectory_version": "trajectory-reconstruction-v1-2026-09-16",
        "counts": {"trajectory_pool": len(trajectories), "pending": len(pending), "quarantine": len(quarantine), "combined_pool": len(combined), "view_rows": len(view_rows), "view_trajectory_rows": len(view_trajectory_ids)},
        "target_turn_reuse": sum(1 for e in errors if e.get("reason") == "TARGET_TURN_REUSE"),
        "errors": errors[:100],
        "gates": {
            "TRAJECTORY_CONTEXT_VERIFICATION": "PASS" if not any(e.get("reason") in {"TRAJECTORY_STATUS_INVALID", "TRAJECTORY_VERIFICATION_MISSING"} for e in errors) else "FAIL",
            "TRAJECTORY_PROVENANCE_ALIGNMENT": "PASS" if not any(e.get("reason", "").endswith("MISMATCH") for e in errors) else "FAIL",
            "TRAJECTORY_MACHINE_GATE": "PASS" if not any(e.get("reason") == "MACHINE_GATE_OR_REWRITE_FLAG_INVALID" for e in errors) else "FAIL",
            "TRAJECTORY_TARGET_REUSE": "PASS" if not any(e.get("reason") == "TARGET_TURN_REUSE" for e in errors) else "FAIL",
            "TRAJECTORY_VIEW_ISOLATION": "PASS" if not any(e.get("reason") in {"TRAJECTORY_PARTITION_OVERLAP", "VIEW_CONTAINS_UNKNOWN_TRAJECTORY"} for e in errors) else "FAIL",
        },
    }
    report["pass"] = not errors and all(value == "PASS" for value in report["gates"].values())
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-pool", required=True)
    parser.add_argument("--pending", required=True)
    parser.add_argument("--quarantine", required=True)
    parser.add_argument("--combined-pool", required=True)
    parser.add_argument("--view", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
