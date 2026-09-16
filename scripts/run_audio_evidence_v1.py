from __future__ import annotations

"""Bounded entry points for audio-evidence-v1 contracts.

This CLI intentionally has no corpus-wide command. Model execution is wired by
injecting adapters into ``AudioEvidencePipeline`` from an environment-specific
production runner. The CLI only plans explicitly supplied interactions;
validation is the Python API ``audio_evidence.validation.validate_artifacts``.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audio_evidence.planning import AudioWindowPlanner  # noqa: E402


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="M.E.O.W. audio-evidence-v1 bounded architecture entry")
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="plan/merge only the supplied interaction JSONL")
    plan.add_argument("--interactions", required=True)
    plan.add_argument("--output", required=True)
    plan.add_argument("--padding-before", type=float, default=0.0)
    plan.add_argument("--padding-after", type=float, default=0.0)
    plan.add_argument("--merge-gap", type=float, default=0.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "plan":
        planner = AudioWindowPlanner(args.padding_before, args.padding_after, args.merge_gap)
        requests = [planner.request_from_interaction(row) for row in read_jsonl(Path(args.interactions))]
        result = planner.merge(requests)
        atomic_json(Path(args.output), result)
        print(json.dumps({"status": "PASS", "windows": len(result["windows"]), "samples": len(result["sample_window_mappings"]), "output": args.output}, ensure_ascii=False))


if __name__ == "__main__":
    main()
