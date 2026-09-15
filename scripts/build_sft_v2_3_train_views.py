from __future__ import annotations

"""Context-closed train-view entry point for v2.3.

Delegates split authority and sampling to the frozen v2.2-named builder, but
refuses any pool row that has not passed the v2.3 context-sufficiency closure.
"""

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "datasets").exists() and (Path(__file__).resolve().parents[3] / "datasets").exists():
    ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = ROOT / "datasets" / "meow_v02_sft_v2_2_semantic_verified"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(1, str(SCRIPT_DIR))

import build_sft_v2_2_train_views as legacy_views  # noqa: E402
import sft_interaction_semantic_closure as semantic  # noqa: E402
from run_sft_v2_3_context_sufficiency import CONTEXT_SUFFICIENCY_VERSION  # noqa: E402


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


def assert_context_closed(pool_path: Path) -> dict:
    rows = load_jsonl(pool_path)
    if not rows:
        raise SystemExit(f"context-closed interaction pool is missing or empty: {pool_path}")
    errors = []
    for row in rows:
        sample_id = str(row.get("sample_id"))
        sufficiency = row.get("context_sufficiency") or {}
        if row.get("artifact_schema_version") != semantic.SCHEMA_VERSION:
            errors.append({"sample_id": sample_id, "reason": "SCHEMA_VERSION_MISMATCH"})
        if sufficiency.get("context_sufficiency_version") != CONTEXT_SUFFICIENCY_VERSION:
            errors.append({"sample_id": sample_id, "reason": "CONTEXT_SUFFICIENCY_VERSION_MISSING"})
        if sufficiency.get("state") != "SELF_CONTAINED":
            errors.append({"sample_id": sample_id, "reason": "NOT_SELF_CONTAINED"})
        if [str(value) for value in row.get("context_turn_ids") or []] != [str(value) for value in sufficiency.get("selected_context_turn_ids") or []]:
            errors.append({"sample_id": sample_id, "reason": "CONTEXT_SELECTION_MISMATCH"})
        if [str(value) for value in row.get("target_turn_ids") or []] != [str(value) for value in sufficiency.get("selected_target_turn_ids") or []]:
            errors.append({"sample_id": sample_id, "reason": "TARGET_SELECTION_MISMATCH"})
    if errors:
        raise SystemExit(f"context-closed pool failed validation for {len(errors)} rows; examples={errors[:10]}")
    return {"rows": len(rows), "context_sufficiency_version": CONTEXT_SUFFICIENCY_VERSION}


def build(args: argparse.Namespace) -> dict:
    context_report = assert_context_closed(Path(args.pool))
    manifest = legacy_views.build(args)
    manifest["context_sufficiency_required"] = True
    manifest["context_sufficiency_version"] = CONTEXT_SUFFICIENCY_VERSION
    manifest["context_closed_source_rows"] = context_report["rows"]
    manifest["source_pool"] = str(Path(args.pool))
    output = Path(args.output) if args.output else DEFAULT_DATASET / "views" / args.policy
    (output / "view_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v2.3 train views from SELF_CONTAINED interaction rows only")
    parser.add_argument("--pool", default=str(DEFAULT_DATASET / "interaction_view_pool_v2_3.jsonl"))
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
