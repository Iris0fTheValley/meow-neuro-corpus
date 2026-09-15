from __future__ import annotations

"""Unified v2.3 validator: frozen interaction invariants + context sufficiency."""

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "datasets").exists() and (Path(__file__).resolve().parents[3] / "datasets").exists():
    ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = ROOT / "datasets" / "meow_v02_sft_v2_2_semantic_verified"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(1, str(SCRIPT_DIR))

import validate_sft_v2_2 as core_validator  # noqa: E402
import validate_sft_v2_3_context_sufficiency as context_validator  # noqa: E402


def validate(args: argparse.Namespace) -> dict:
    dataset = Path(args.dataset)
    core_report = core_validator.validate(
        Namespace(
            dataset=str(dataset),
            pool=args.pool,
            views=args.views,
            split_authority=args.split_authority,
            strict_quarantine=args.strict_quarantine,
            report=str(dataset / "validation_report_core_v2_3.json"),
        )
    )
    context_report = context_validator.validate_context(
        Namespace(
            decisions=args.context_decisions,
            source_pool=args.pool,
            interaction_pool=args.interaction_pool,
            reconstruction_candidates=args.reconstruction_candidates,
            unsupported_quarantine=args.unsupported_quarantine,
            requests=args.context_requests,
            views=args.views,
            report=str(dataset / "context_sufficiency_validation_v2_3.json"),
        )
    )
    report = {
        "artifact_status": "VALIDATED_V2_3_WITH_CONTEXT_SUFFICIENCY",
        "core": core_report,
        "context_sufficiency": context_report,
        "gates": {
            **{f"CORE::{key}": value for key, value in (core_report.get("gates") or {}).items()},
            **{f"CONTEXT::{key}": value for key, value in (context_report.get("gates") or {}).items()},
        },
        "pass": bool(core_report.get("pass")) and bool(context_report.get("pass")),
    }
    output = Path(args.report) if args.report else dataset / "validation_report_v2_3.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate full v2.3 production including context sufficiency")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--pool", default=str(DEFAULT_DATASET / "verified_interaction_pool_v2_3.jsonl"))
    parser.add_argument("--interaction-pool", default=str(DEFAULT_DATASET / "interaction_view_pool_v2_3.jsonl"))
    parser.add_argument("--context-requests", default=str(DEFAULT_DATASET / "context_sufficiency_requests_v2_3.jsonl"))
    parser.add_argument("--context-decisions", default=str(DEFAULT_DATASET / "context_sufficiency_decisions_v2_3.jsonl"))
    parser.add_argument("--reconstruction-candidates", default=str(DEFAULT_DATASET / "context_reconstruction_candidates_v2_3.jsonl"))
    parser.add_argument("--unsupported-quarantine", default=str(DEFAULT_DATASET / "unsupported_relation_quarantine_v2_3.jsonl"))
    parser.add_argument("--views", default="")
    parser.add_argument("--split-authority", default=str(DEFAULT_DATASET / "split_authority_v2_3.json"))
    parser.add_argument("--strict-quarantine", default="")
    parser.add_argument("--report", default="")
    args = parser.parse_args()
    print(json.dumps(validate(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
