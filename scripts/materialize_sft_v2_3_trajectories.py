from __future__ import annotations

"""Materialize bounded canonical-context trajectory candidates.

This is a downstream materializer, not a new semantic policy.  It accepts only
CONTEXT_INCOMPLETE rows whose bounded canonical recovery was independently
verified with the frozen context-sufficiency judge.  Final turn provenance is
rebuilt from the canonical timeline; candidate snapshots are never treated as
training truth.
"""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "datasets").exists() and (Path(__file__).resolve().parents[3] / "datasets").exists():
    ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = ROOT / "datasets" / "meow_v02_sft_v2_2_semantic_verified"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(1, str(SCRIPT_DIR))

import build_sft_v2_2_structural_candidates as structural_core  # noqa: E402
import sft_interaction_semantic_closure as semantic  # noqa: E402
import sft_semantic_verified_v2_1 as judge_core  # noqa: E402
from run_sft_v2_3_context_sufficiency import (  # noqa: E402
    CONTEXT_SUFFICIENCY_VERSION,
    PROMPT_VERSION,
    result_compatibility,
)

TRAJECTORY_VERSION = "trajectory-reconstruction-v1-2026-09-16"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def latest(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("sample_id")): row for row in load_jsonl(path)}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _bad_boundary(turn: dict[str, Any]) -> bool:
    return structural_core.bad_boundary(turn)


def _snapshot(turn: dict[str, Any], role: str, raw_index: int) -> dict[str, Any]:
    start = float(turn.get("_start", (turn.get("timestamp") or {}).get("start", raw_index)))
    end = float(turn.get("_end", (turn.get("timestamp") or {}).get("end", start)))
    return {
        "turn_id": str(turn.get("turn_id") or ""),
        "role": role,
        "raw_timeline_index": int(turn.get("_index", raw_index)),
        "text": str(turn.get("text") or ""),
        "timestamp": {"start": start, "end": end},
        "speaker": str(turn.get("speaker") or ""),
        "identity": turn.get("identity", "UNKNOWN"),
        "speaker_confidence": turn.get("speaker_confidence"),
        "text_qa": turn.get("text_qa"),
        "text_qa_reasons": turn.get("text_qa_reasons") or [],
        "asr_quality": turn.get("asr_quality"),
        "asr_quality_reasons": turn.get("asr_quality_reasons") or [],
        "asr_metrics": turn.get("asr_metadata") or {},
        "boundary": {
            "bad_boundary": _bad_boundary(turn),
            "event_boundary": bool(turn.get("event_boundary")),
            "gap_from_previous": round(float(turn.get("_gap_from_previous") or 0.0), 3),
        },
    }


def _canonical_turns(timeline: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(turn.get("turn_id")): turn for turn in timeline.get("_turns") or []}


def _annotate(timeline: dict[str, Any]) -> None:
    previous_end = None
    for index, turn in enumerate(timeline.get("_turns") or []):
        turn["text_qa"], turn["text_qa_reasons"] = structural_core.text_quality(turn.get("text"))
        turn["asr_quality"], turn["asr_quality_reasons"], turn["asr_metadata"] = structural_core.raw_asr_quality(turn)
        turn["_bad_boundary"] = structural_core.bad_boundary(turn)
        start = float(turn.get("_start", index))
        end = float(turn.get("_end", index))
        turn["_gap_from_previous"] = 0.0 if previous_end is None else max(0.0, start - previous_end)
        turn["_index"] = int(turn.get("_index", index))
        previous_end = end


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    candidates = latest(Path(args.candidates))
    source_pool = latest(Path(args.pool))
    requests = latest(Path(args.requests))
    results = latest(Path(args.verification_results))
    if not candidates:
        raise SystemExit(f"trajectory candidates missing or empty: {args.candidates}")
    if not source_pool:
        raise SystemExit(f"source pool missing or empty: {args.pool}")
    recovered = {
        sample_id: row
        for sample_id, row in candidates.items()
        if (row.get("trajectory_candidate") or {}).get("recovery_status") == "BOUNDED_CONTEXT_RECOVERED"
        and (row.get("trajectory_candidate") or {}).get("added_context_turn_ids")
    }
    fusion = judge_core.load_fusion()
    timelines = judge_core.load_timelines({str(row.get("source_id")) for row in recovered.values()}, fusion)
    for timeline in timelines.values():
        _annotate(timeline)
    trajectory: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    counts = Counter()
    for sample_id, candidate in sorted(recovered.items()):
        tc = candidate.get("trajectory_candidate") or {}
        request = requests.get(sample_id)
        result = results.get(sample_id)
        if not request or not result:
            pending.append({"sample_id": sample_id, "reason": "RECONSTRUCTED_CONTEXT_RESULT_MISSING"})
            counts["missing_result"] += 1
            continue
        compatible, reason = result_compatibility(result, request, args.model)
        if not compatible:
            pending.append({"sample_id": sample_id, "reason": f"RECONSTRUCTED_CONTEXT_{reason}"})
            counts["stale_or_invalid_result"] += 1
            continue
        state = str((result.get("validated") or {}).get("state") or "")
        if state == "CONTEXT_INCOMPLETE":
            pending.append({"sample_id": sample_id, "reason": "RECONSTRUCTED_CONTEXT_STILL_INCOMPLETE"})
            counts["still_incomplete"] += 1
            continue
        if state == "UNSUPPORTED_RELATION":
            quarantine.append({"sample_id": sample_id, "reason": "RECONSTRUCTED_CONTEXT_UNSUPPORTED_RELATION", "semantic_result": result.get("validated") or {}})
            counts["unsupported_relation"] += 1
            continue
        if state != "SELF_CONTAINED":
            pending.append({"sample_id": sample_id, "reason": "RECONSTRUCTED_CONTEXT_UNKNOWN_STATE"})
            counts["unknown_state"] += 1
            continue
        timeline = timelines.get(str(candidate.get("source_id")))
        by_id = _canonical_turns(timeline or {})
        context_ids = [str(value) for value in tc.get("expanded_context_turn_ids") or []]
        target_ids = [str(value) for value in tc.get("target_turn_ids") or []]
        if not context_ids or not target_ids or any(value not in by_id for value in context_ids + target_ids):
            quarantine.append({"sample_id": sample_id, "reason": "CANONICAL_TURN_NOT_FOUND"})
            counts["missing_canonical_turn"] += 1
            continue
        selected_context = [by_id[value] for value in context_ids]
        selected_target = [by_id[value] for value in target_ids]
        invalid_reason = None
        for turn in selected_context + selected_target:
            if not turn.get("text_qa") == "PASS":
                invalid_reason = "TEXT_QA_FAIL"
                break
            if turn.get("asr_quality") == "REVIEW":
                invalid_reason = "ASR_REVIEW"
                break
            if _bad_boundary(turn) or turn.get("event_boundary"):
                invalid_reason = "BOUNDARY_INVALID"
                break
        if invalid_reason:
            quarantine.append({"sample_id": sample_id, "reason": f"FINAL_MACHINE_GATE_{invalid_reason}"})
            counts["final_machine_gate_reject"] += 1
            continue
        context_snapshots = [_snapshot(turn, "context", int(turn.get("_index", 0))) for turn in selected_context]
        target_snapshots = [_snapshot(turn, "target", int(turn.get("_index", 0))) for turn in selected_target]
        final_turns = context_snapshots + target_snapshots
        context_text = "\n".join(value["text"] for value in context_snapshots if value["text"])
        target_text = "\n".join(value["text"] for value in target_snapshots if value["text"])
        source_tqa = candidate.get("transcript_qa") or {}
        row = dict(candidate)
        row.update(
            {
                "context_turn_ids": context_ids,
                "target_turn_ids": target_ids,
                "messages": [{"role": "user", "content": context_text}, {"role": "assistant", "content": target_text}],
                "transcript_qa": {
                    "status": "RECONSTRUCTED_CONTEXT_PASS",
                    "source": "CURRENT_CANONICAL_TIMELINE",
                    "context_turn_ids": context_ids,
                    "target_turn_ids": target_ids,
                    "final_turn_ids": context_ids + target_ids,
                    "turns": final_turns,
                    "all_selected_turns_machine_usable": True,
                    "semantic_verifier_cannot_rewrite_transcript": True,
                },
                "speaker_evidence": {
                    "assistant_speaker": target_snapshots[0]["speaker"],
                    "assistant_identity": target_snapshots[0]["identity"],
                    "assistant_turn_speakers": [value["speaker"] for value in target_snapshots],
                    "assistant_turn_identities": [value["identity"] for value in target_snapshots],
                    "assistant_turn_confidences": [value["speaker_confidence"] for value in target_snapshots],
                    "context_speakers": [value["speaker"] for value in context_snapshots],
                    "context_identities": [value["identity"] for value in context_snapshots],
                    "context_turn_confidences": [value["speaker_confidence"] for value in context_snapshots],
                    "materialized_from_final_turn_selection": True,
                    "interaction_semantics_cannot_change_identity": True,
                },
                "timestamps": {"start": final_turns[0]["timestamp"]["start"], "end": final_turns[-1]["timestamp"]["end"]},
                "trajectory_reconstruction": {
                    "trajectory_version": TRAJECTORY_VERSION,
                    "status": "VERIFIED_RECONSTRUCTED_CONTEXT",
                    "base_context_turn_ids": [str(value) for value in tc.get("base_context_turn_ids") or []],
                    "added_context_turn_ids": [str(value) for value in tc.get("added_context_turn_ids") or []],
                    "expanded_context_turn_ids": context_ids,
                    "semantic_truth_preserved": True,
                    "verification_result": result.get("validated") or {},
                    "verification_request_sha256": request.get("request_sha256"),
                    "canonical_timeline_source": "CURRENT_CANONICAL_TIMELINE",
                },
                "verified_pool_status": "VERIFIED_RECONSTRUCTED_CONTEXT",
                "sampling_eligibility": "TRAJECTORY_ELIGIBLE",
                "context_sufficiency": {**(candidate.get("context_sufficiency") or {}), "reconstructed_context_verified": True},
                "source_transcript_qa_status": source_tqa.get("status"),
            }
        )
        row.pop("trajectory_candidate", None)
        trajectory.append(row)
        counts["trajectory_eligible"] += 1
    write_jsonl(Path(args.trajectory_pool), sorted(trajectory, key=lambda row: str(row.get("sample_id"))))
    write_jsonl(Path(args.pending), pending)
    write_jsonl(Path(args.quarantine), quarantine)
    report = {
        "schema_version": semantic.SCHEMA_VERSION,
        "pipeline_version": semantic.PIPELINE_VERSION,
        "trajectory_version": TRAJECTORY_VERSION,
        "context_sufficiency_version": CONTEXT_SUFFICIENCY_VERSION,
        "status": "MATERIALIZED",
        "recovered_candidates": len(recovered),
        "trajectory_pool_rows": len(trajectory),
        "pending_rows": len(pending),
        "quarantine_rows": len(quarantine),
        "counts": dict(counts),
        "diagnostic_turns_promoted_to_training": 0,
        "canonical_timeline_rematerialized": True,
        "semantic_verifier_rewrites_transcript": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(Path(args.report), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize verified bounded trajectory reconstructions")
    parser.add_argument("--candidates", default=str(DEFAULT_DATASET / "context_reconstruction_candidates_v2_3.jsonl"))
    parser.add_argument("--pool", default=str(DEFAULT_DATASET / "verified_interaction_pool_v2_3.jsonl"))
    parser.add_argument("--requests", default=str(DEFAULT_DATASET / "reconstructed_context_verification_requests_v2_3.jsonl"))
    parser.add_argument("--verification-results", default=str(DEFAULT_DATASET / "reconstructed_context_verification_results_v2_3.jsonl"))
    parser.add_argument("--trajectory-pool", default=str(DEFAULT_DATASET / "trajectory_pool_v2_3.jsonl"))
    parser.add_argument("--pending", default=str(DEFAULT_DATASET / "trajectory_pending_v2_3.jsonl"))
    parser.add_argument("--quarantine", default=str(DEFAULT_DATASET / "trajectory_quarantine_v2_3.jsonl"))
    parser.add_argument("--report", default=str(DEFAULT_DATASET / "trajectory_materialization_v2_3.json"))
    parser.add_argument("--model", default="qwen3.8-27b-efficientthink-simpo-lynnstyle")
    args = parser.parse_args()
    print(json.dumps(materialize(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
