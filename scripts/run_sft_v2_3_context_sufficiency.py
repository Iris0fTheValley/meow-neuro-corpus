from __future__ import annotations

"""Context-sufficiency closure for v2.3 verified interactions.

This stage is intentionally downstream of semantic truth closure.  Its judge sees
only the exact context and target that would be materialized for training.  It
never receives diagnostic turns, source metadata, alternate context options, or
canonical timeline neighbors.
"""

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "datasets").exists() and (Path(__file__).resolve().parents[3] / "datasets").exists():
    ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = ROOT / "datasets" / "meow_v02_sft_v2_2_semantic_verified"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(SCRIPT_DIR))

import sft_interaction_semantic_closure as semantic  # noqa: E402
import sft_semantic_verified_v2_1 as judge_core  # noqa: E402

CONTEXT_SUFFICIENCY_VERSION = "context-sufficiency-v1-2026-09-16"
PROMPT_VERSION = "context-sufficiency-v2.3-p1"
ALLOWED_STATES = {"SELF_CONTAINED", "CONTEXT_INCOMPLETE", "UNSUPPORTED_RELATION"}
DEFAULT_RESULTS = DEFAULT_DATASET / "context_sufficiency_results_v2_3.jsonl"
SYSTEM_PROMPT = (
    "You are a narrow context-sufficiency verifier for already semantically-verified dialogue. "
    "Judge only the selected context and selected target shown in this request. No diagnostic turns, source metadata, "
    "timeline neighbors, alternate context, hidden events, or external knowledge are available or allowed. "
    "SELF_CONTAINED means the selected context alone is sufficient to make the selected target a plausible response. "
    "CONTEXT_INCOMPLETE means the target plausibly depends on omitted conversational context and the selected slice is not independently trainable. "
    "UNSUPPORTED_RELATION means the selected pair itself does not form a plausible response relation even after allowing normal conversational ellipsis; "
    "do not invent missing events to rescue it. Do not rewrite transcript text. Return exactly one JSON object matching output_schema."
)


def load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except Exception:
                continue
            if isinstance(value, dict):
                yield value


def latest(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("sample_id")): row for row in load_jsonl(path)}


def write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def payload_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _minimal_turn(turn: dict[str, Any]) -> dict[str, str]:
    return {
        "turn_id": str(turn.get("turn_id") or ""),
        "speaker": str(turn.get("speaker") or ""),
        "text": str(turn.get("text") or ""),
    }


def _recovery_bad_boundary(turn: dict[str, Any]) -> bool:
    return any(bool(turn.get(key)) for key in ("bad_boundary", "_bad_boundary", "uncertain_transcription", "suspicious_transcription", "overlap", "malformed"))


def build_trajectory_candidate(pool_row: dict[str, Any], timeline: dict[str, Any] | None, *, max_recovery_turns: int = 4, max_gap_seconds: float = 8.0) -> dict[str, Any]:
    """Attach bounded canonical context recovery without upgrading training state."""
    context_ids = [str(value) for value in pool_row.get("context_turn_ids") or []]
    target_ids = [str(value) for value in pool_row.get("target_turn_ids") or []]
    turns = list((timeline or {}).get("_turns") or [])
    try:
        max_gap_seconds = max(0.0, float((pool_row.get("episode_boundary") or {}).get("context_gap_policy_seconds", max_gap_seconds)))
    except (TypeError, ValueError):
        max_gap_seconds = 8.0
    by_id = {str(turn.get("turn_id")): turn for turn in turns}
    base = {
        "status": "PENDING_TRAJECTORY_REVIEW",
        "source": "CURRENT_CANONICAL_TIMELINE",
        "semantic_truth_preserved": True,
        "training_eligibility": "CONTEXT_RECONSTRUCTION_ONLY",
        "base_context_turn_ids": context_ids,
        "target_turn_ids": target_ids,
        "expanded_context_turn_ids": context_ids,
        "diagnostic_turns_promoted_to_training": 0,
    }
    if not turns or any(value not in by_id for value in context_ids + target_ids):
        base["recovery_status"] = "CANONICAL_TIMELINE_UNAVAILABLE"
        return base
    positions = {turn_id: index for index, turn_id in enumerate(by_id)}
    context_positions = [positions[value] for value in context_ids]
    target_positions = [positions[value] for value in target_ids]
    if context_positions != list(range(min(context_positions), max(context_positions) + 1)) or min(context_positions) >= min(target_positions):
        base["recovery_status"] = "BASE_SELECTION_NOT_CONTIGUOUS"
        return base
    start = min(context_positions)
    cursor = start - 1
    added = 0
    while cursor >= 0 and added < max_recovery_turns:
        current = turns[cursor]
        following = turns[cursor + 1]
        gap = max(0.0, float(following.get("_start", cursor + 1)) - float(current.get("_end", cursor)))
        if current.get("identity") in semantic.TARGETS or _recovery_bad_boundary(current) or current.get("event_boundary") or following.get("event_boundary") or gap > max_gap_seconds:
            break
        start = cursor
        cursor -= 1
        added += 1
    expanded = turns[start : max(context_positions) + 1]
    base["recovery_status"] = "BOUNDED_CONTEXT_RECOVERED" if added else "NO_ADDITIONAL_CLEAN_CONTEXT"
    base["expanded_context_turn_ids"] = [str(turn.get("turn_id")) for turn in expanded]
    base["added_context_turn_ids"] = [str(turn.get("turn_id")) for turn in expanded if str(turn.get("turn_id")) not in context_ids]
    base["expanded_context"] = [_minimal_turn(turn) for turn in expanded]
    return base


def build_sufficiency_request(pool_row: dict[str, Any]) -> dict[str, Any] | None:
    """Build a judge payload from the exact materialized pool slice only."""
    if (pool_row.get("semantic_closure") or {}).get("state") != "VERIFIED":
        return None
    context_ids = [str(value) for value in pool_row.get("context_turn_ids") or []]
    target_ids = [str(value) for value in pool_row.get("target_turn_ids") or []]
    transcript_turns = list((pool_row.get("transcript_qa") or {}).get("turns") or [])
    by_id = {str(turn.get("turn_id") or ""): turn for turn in transcript_turns if isinstance(turn, dict)}
    if not context_ids or not target_ids or any(turn_id not in by_id for turn_id in context_ids + target_ids):
        return None
    selected_context = [_minimal_turn(by_id[turn_id]) for turn_id in context_ids]
    selected_target = [_minimal_turn(by_id[turn_id]) for turn_id in target_ids]
    # Hash the exact materialized text+turn selection so cached sufficiency
    # results cannot survive a rematerialization change.
    source_materialization = {
        "sample_id": str(pool_row.get("sample_id")),
        "context_turn_ids": context_ids,
        "target_turn_ids": target_ids,
        "selected_context": selected_context,
        "selected_target": selected_target,
    }
    payload = {
        "schema_version": semantic.SCHEMA_VERSION,
        "context_sufficiency_version": CONTEXT_SUFFICIENCY_VERSION,
        "task": "Judge whether this exact final training slice is self-sufficient.",
        "selected_context": selected_context,
        "selected_target": selected_target,
        "allowed_state": sorted(ALLOWED_STATES),
        "constraints": [
            "Use only selected_context and selected_target as conversational evidence.",
            "Do not infer or invent omitted turns, chat messages, donations, UI events, game events, or visual state.",
            "Do not rewrite, correct, summarize, or expand transcript text.",
            "Judge self-sufficiency of the materialized slice, not persona quality or training value.",
        ],
        "output_schema": {
            "state": "SELF_CONTAINED | CONTEXT_INCOMPLETE | UNSUPPORTED_RELATION",
            "confidence": "number 0..1",
            "reason": "short evidence-based reason about the shown pair only",
        },
    }
    return {
        "sample_id": str(pool_row.get("sample_id")),
        "source_materialization_sha256": payload_sha256(source_materialization),
        "selected_context_turn_ids": context_ids,
        "selected_target_turn_ids": target_ids,
        "judge_input": payload,
        "request_sha256": payload_sha256(payload),
        "artifact_schema_version": semantic.SCHEMA_VERSION,
        "pipeline_version": semantic.PIPELINE_VERSION,
        "context_sufficiency_version": CONTEXT_SUFFICIENCY_VERSION,
    }


def validate_result(raw: dict[str, Any] | None) -> dict[str, Any]:
    parsed = raw.get("parsed") if isinstance(raw, dict) and "parsed" in raw else raw
    if not isinstance(parsed, dict):
        return {"valid": False, "reason": "missing_json_object"}
    state = str(parsed.get("state") or "").strip().upper()
    try:
        confidence = float(parsed.get("confidence"))
    except (TypeError, ValueError):
        confidence = -1.0
    forbidden = {"corrected_transcript", "rewritten_transcript", "context_text", "target_text", "response_text"} & set(parsed)
    valid = state in ALLOWED_STATES and 0.0 <= confidence <= 1.0 and not forbidden
    result = {
        "valid": valid,
        "state": state,
        "confidence": confidence,
        "reason": str(parsed.get("reason") or "")[:500],
    }
    if not valid:
        result["reason_code"] = "transcript_rewrite_forbidden" if forbidden else "schema_invalid"
    return result


def result_compatibility(result: dict[str, Any] | None, request: dict[str, Any], model: str) -> tuple[bool, str]:
    if not result:
        return False, "MISSING"
    expected = {
        "sample_id": str(request.get("sample_id")),
        "request_sha256": request.get("request_sha256"),
        "source_materialization_sha256": request.get("source_materialization_sha256"),
        "schema_version": semantic.SCHEMA_VERSION,
        "pipeline_version": semantic.PIPELINE_VERSION,
        "context_sufficiency_version": CONTEXT_SUFFICIENCY_VERSION,
        "judge_prompt_version": PROMPT_VERSION,
        "judge_model": model,
    }
    for key, expected_value in expected.items():
        if result.get(key) != expected_value:
            return False, f"STALE_{key.upper()}"
    if not (result.get("validated") or {}).get("valid"):
        return False, "INVALID_RESULT"
    return True, "COMPATIBLE"


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    source_pool = list(load_jsonl(Path(args.pool)))
    if not source_pool:
        raise SystemExit(f"verified pool is missing or empty: {args.pool}")
    prepared: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for row in sorted(source_pool, key=lambda value: str(value.get("sample_id"))):
        sample_id = str(row.get("sample_id"))
        value = build_sufficiency_request(row)
        if value is None:
            missing.append({"sample_id": sample_id, "reason": "MATERIALIZED_SELECTION_NOT_REMATERIALIZABLE"})
            continue
        prepared.append(value)
    if missing:
        raise SystemExit(f"context-sufficiency prepare failed for {len(missing)} verified-pool samples; examples={missing[:10]}")
    write_jsonl(Path(args.requests), prepared)
    report = {
        "schema_version": semantic.SCHEMA_VERSION,
        "pipeline_version": semantic.PIPELINE_VERSION,
        "context_sufficiency_version": CONTEXT_SUFFICIENCY_VERSION,
        "status": "PREPARED",
        "verified_pool_required": len(source_pool),
        "prepared": len(prepared),
        "diagnostic_turns_exposed_to_judge": 0,
        "source": "verified_interaction_pool_v2_3 exact materialized transcript",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(Path(args.prepare_report), report)
    return report


def _selected(args: argparse.Namespace, values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = {value.strip() for value in str(args.sample_ids or "").split(",") if value.strip()}
    if args.sample_ids_file:
        wanted.update(line.strip() for line in Path(args.sample_ids_file).read_text(encoding="utf-8").splitlines() if line.strip())
    if wanted:
        values = [row for row in values if str(row.get("sample_id")) in wanted]
    shard_count = max(1, int(getattr(args, "shard_count", 1) or 1))
    shard_index = int(getattr(args, "shard_index", 0) or 0)
    if not 0 <= shard_index < shard_count:
        raise SystemExit(f"invalid shard index {shard_index} for shard count {shard_count}")
    if shard_count > 1:
        values = [
            row for row in values
            if int(hashlib.sha256(str(row.get("sample_id")).encode()).hexdigest(), 16) % shard_count == shard_index
        ]
    return values[: args.max_items] if args.max_items else values


def _judge_destination(args: argparse.Namespace) -> Path:
    destination = Path(args.results)
    shard_count = max(1, int(getattr(args, "shard_count", 1) or 1))
    shard_index = int(getattr(args, "shard_index", 0) or 0)
    if shard_count > 1 and destination == DEFAULT_RESULTS:
        return destination.with_name(f"context_sufficiency_results_v2_3.part-{shard_index:02d}-of-{shard_count:02d}.jsonl")
    return destination


def judge(args: argparse.Namespace) -> dict[str, Any]:
    requests = list(load_jsonl(Path(args.requests)))
    selected = _selected(args, requests)
    destination = _judge_destination(args)
    existing = latest(destination) if destination.exists() else {}
    counters = Counter()
    previous_prompt = judge_core.SYSTEM_PROMPT
    try:
        judge_core.SYSTEM_PROMPT = SYSTEM_PROMPT
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("a" if destination.exists() else "w", encoding="utf-8") as handle:
            for request in selected:
                sample_id = str(request.get("sample_id"))
                prior = existing.get(sample_id)
                compatible, reason = result_compatibility(prior, request, args.model)
                if compatible:
                    counters["skipped_existing"] += 1
                    continue
                if prior:
                    counters["rerun_" + reason.lower()] += 1
                output = None
                for attempt in range(1, args.max_attempts + 1):
                    raw = judge_core.call_judge(request["judge_input"], model=args.model, endpoint=args.endpoint)
                    validated = validate_result(raw)
                    output = {
                        "schema_version": semantic.SCHEMA_VERSION,
                        "pipeline_version": semantic.PIPELINE_VERSION,
                        "context_sufficiency_version": CONTEXT_SUFFICIENCY_VERSION,
                        "sample_id": sample_id,
                        "request_sha256": request.get("request_sha256"),
                        "source_materialization_sha256": request.get("source_materialization_sha256"),
                        "judge_model": args.model,
                        "judge_prompt_version": PROMPT_VERSION,
                        "validated": validated,
                        "raw": raw,
                        "judge_input": request["judge_input"],
                        "attempts": attempt,
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                    }
                    if validated.get("valid"):
                        break
                    if attempt < args.max_attempts:
                        time.sleep(0.5 * attempt)
                handle.write(json.dumps(output, ensure_ascii=False) + "\n")
                handle.flush()
                counters["processed"] += 1
                counters["valid"] += int(bool((output or {}).get("validated", {}).get("valid")))
                counters["invalid"] += int(not bool((output or {}).get("validated", {}).get("valid")))
                state = str((output or {}).get("validated", {}).get("state") or "")
                if state in ALLOWED_STATES:
                    counters[state] += 1
                if args.sleep:
                    time.sleep(args.sleep)
    finally:
        judge_core.SYSTEM_PROMPT = previous_prompt
    report = {
        "schema_version": semantic.SCHEMA_VERSION,
        "pipeline_version": semantic.PIPELINE_VERSION,
        "context_sufficiency_version": CONTEXT_SUFFICIENCY_VERSION,
        "status": "COMPLETED",
        "selected": len(selected),
        **dict(counters),
        "model": args.model,
        "prompt_version": PROMPT_VERSION,
        "results": str(destination),
        "shard_index": int(getattr(args, "shard_index", 0) or 0),
        "shard_count": max(1, int(getattr(args, "shard_count", 1) or 1)),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(Path(args.run_report), report)
    return report


def merge(args: argparse.Namespace) -> dict[str, Any]:
    dataset = Path(args.dataset)
    pattern = str(args.merge_glob or "context_sufficiency_results_v2_3.part-*-of-*.jsonl")
    shard_paths = sorted(dataset.glob(pattern))
    if not shard_paths:
        raise SystemExit(f"no context-sufficiency shard results match {pattern} in {dataset}")
    merged_rows: list[dict[str, Any]] = []
    sample_sources: dict[str, set[str]] = {}
    for path in shard_paths:
        for row in load_jsonl(path):
            sample_id = str(row.get("sample_id"))
            sample_sources.setdefault(sample_id, set()).add(path.name)
            merged_rows.append(row)
    overlap = {sample_id: sorted(paths) for sample_id, paths in sample_sources.items() if len(paths) > 1}
    if overlap:
        raise SystemExit(f"sample ids overlap across shard files; refusing ambiguous merge: {list(overlap.items())[:10]}")
    merged_rows.sort(key=lambda row: (str(row.get("sample_id")), str(row.get("completed_at") or "")))
    destination = Path(args.results)
    write_jsonl(destination, merged_rows)
    report = {
        "schema_version": semantic.SCHEMA_VERSION,
        "pipeline_version": semantic.PIPELINE_VERSION,
        "context_sufficiency_version": CONTEXT_SUFFICIENCY_VERSION,
        "status": "MERGED",
        "shards": [path.name for path in shard_paths],
        "rows": len(merged_rows),
        "unique_samples": len(sample_sources),
        "results": str(destination),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(Path(args.merge_report), report)
    return report


def status(args: argparse.Namespace) -> dict[str, Any]:
    requests = latest(Path(args.requests))
    results = latest(Path(args.results))
    compatible: set[str] = set()
    stale: dict[str, str] = {}
    for sample_id, request in requests.items():
        ok, reason = result_compatibility(results.get(sample_id), request, args.model)
        if ok:
            compatible.add(sample_id)
        else:
            stale[sample_id] = reason
    state_counts = Counter(
        str((results[sample_id].get("validated") or {}).get("state") or "UNKNOWN") for sample_id in compatible
    )
    report = {
        "schema_version": semantic.SCHEMA_VERSION,
        "pipeline_version": semantic.PIPELINE_VERSION,
        "context_sufficiency_version": CONTEXT_SUFFICIENCY_VERSION,
        "required": len(requests),
        "valid_current": len(compatible),
        "missing_or_stale": len(stale),
        "state_counts": dict(state_counts),
        "complete": len(compatible) == len(requests) and not stale,
        "stale_reason_counts": dict(Counter(stale.values())),
        "stale_examples": sorted(stale)[:50],
    }
    write_json(Path(args.status_report), report)
    return report


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    requests = latest(Path(args.requests))
    results = latest(Path(args.results))
    source_pool = list(load_jsonl(Path(args.pool)))
    if not source_pool:
        raise SystemExit(f"verified pool is missing or empty: {args.pool}")
    source_by_id = {str(row.get("sample_id")): row for row in source_pool}
    timeline_by_source: dict[str, dict[str, Any]] = {}
    if any(str((results.get(sample_id, {}).get("validated") or {}).get("state")) == "CONTEXT_INCOMPLETE" for sample_id in source_by_id):
        try:
            fusion = judge_core.load_fusion()
            timeline_by_source = judge_core.load_timelines({str(row.get("source_id")) for row in source_pool}, fusion)
        except (OSError, ValueError, json.JSONDecodeError):
            timeline_by_source = {}
    pool_ids = set(source_by_id)
    missing_requests = sorted(pool_ids - set(requests))
    if missing_requests:
        raise SystemExit(f"{len(missing_requests)} verified-pool samples lack context-sufficiency requests")
    decisions: list[dict[str, Any]] = []
    invalid: list[dict[str, str]] = []
    for sample_id in sorted(requests):
        ok, reason = result_compatibility(results.get(sample_id), requests[sample_id], args.model)
        if not ok:
            invalid.append({"sample_id": sample_id, "reason": reason})
            continue
        validated = results[sample_id]["validated"]
        decisions.append(
            {
                "schema_version": semantic.SCHEMA_VERSION,
                "pipeline_version": semantic.PIPELINE_VERSION,
                "context_sufficiency_version": CONTEXT_SUFFICIENCY_VERSION,
                "sample_id": sample_id,
                "state": validated["state"],
                "confidence": validated["confidence"],
                "reason": validated.get("reason") or "",
                "request_sha256": requests[sample_id]["request_sha256"],
                "source_materialization_sha256": requests[sample_id]["source_materialization_sha256"],
                "selected_context_turn_ids": requests[sample_id]["selected_context_turn_ids"],
                "selected_target_turn_ids": requests[sample_id]["selected_target_turn_ids"],
                "judge_model": args.model,
                "judge_prompt_version": PROMPT_VERSION,
            }
        )
    if invalid:
        raise SystemExit(f"context-sufficiency closure incomplete/invalid for {len(invalid)} samples; examples={invalid[:10]}")
    by_decision = {row["sample_id"]: row for row in decisions}
    closed_pool: list[dict[str, Any]] = []
    interaction_pool: list[dict[str, Any]] = []
    incomplete: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    for sample_id, source in sorted(source_by_id.items()):
        decision = by_decision[sample_id]
        context_ids = [str(value) for value in source.get("context_turn_ids") or []]
        target_ids = [str(value) for value in source.get("target_turn_ids") or []]
        if context_ids != decision["selected_context_turn_ids"] or target_ids != decision["selected_target_turn_ids"]:
            raise SystemExit(f"context-sufficiency selection provenance mismatch for {sample_id}")
        row = dict(source)
        row["context_sufficiency"] = decision
        state = decision["state"]
        row["sampling_eligibility"] = "ELIGIBLE" if state == "SELF_CONTAINED" else "CONTEXT_RECONSTRUCTION_ONLY" if state == "CONTEXT_INCOMPLETE" else "QUARANTINED_UNSUPPORTED_RELATION"
        closed_pool.append(row)
        if state == "SELF_CONTAINED":
            interaction_pool.append(row)
        elif state == "CONTEXT_INCOMPLETE":
            row["trajectory_candidate"] = build_trajectory_candidate(
                row,
                timeline_by_source.get(str(row.get("source_id"))),
                max_recovery_turns=int(getattr(args, "max_recovery_turns", 4) or 4),
                max_gap_seconds=float(getattr(args, "max_recovery_gap", 8.0) or 8.0),
            )
            incomplete.append(row)
        else:
            unsupported.append(row)
    write_jsonl(Path(args.decisions_output), decisions)
    write_jsonl(Path(args.closed_pool), closed_pool)
    write_jsonl(Path(args.interaction_pool), interaction_pool)
    write_jsonl(Path(args.reconstruction_candidates), incomplete)
    write_jsonl(Path(args.unsupported_quarantine), unsupported)
    counts = Counter(row["state"] for row in decisions)
    report = {
        "schema_version": semantic.SCHEMA_VERSION,
        "pipeline_version": semantic.PIPELINE_VERSION,
        "context_sufficiency_version": CONTEXT_SUFFICIENCY_VERSION,
        "status": "MATERIALIZED",
        "source_verified_pool": len(source_pool),
        "context_sufficiency_required": len(requests),
        "state_counts": dict(counts),
        "context_closed_pool": len(closed_pool),
        "direct_interaction_pool": len(interaction_pool),
        "reconstruction_candidates": len(incomplete),
        "unsupported_relation_quarantine": len(unsupported),
        "diagnostic_evidence_promoted_to_training_context": 0,
        "ready_for_interaction_train_views": len(interaction_pool) > 0,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(Path(args.materialize_report), report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v2.3 selected-context sufficiency closure")
    parser.add_argument("command", choices=["prepare", "judge", "merge", "status", "materialize"])
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--requests", default=str(DEFAULT_DATASET / "context_sufficiency_requests_v2_3.jsonl"))
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--pool", default=str(DEFAULT_DATASET / "verified_interaction_pool_v2_3.jsonl"))
    parser.add_argument("--decisions-output", default=str(DEFAULT_DATASET / "context_sufficiency_decisions_v2_3.jsonl"))
    parser.add_argument("--closed-pool", default=str(DEFAULT_DATASET / "verified_interaction_pool_v2_3_context_closed.jsonl"))
    parser.add_argument("--interaction-pool", default=str(DEFAULT_DATASET / "interaction_view_pool_v2_3.jsonl"))
    parser.add_argument("--reconstruction-candidates", default=str(DEFAULT_DATASET / "context_reconstruction_candidates_v2_3.jsonl"))
    parser.add_argument("--unsupported-quarantine", default=str(DEFAULT_DATASET / "unsupported_relation_quarantine_v2_3.jsonl"))
    parser.add_argument("--prepare-report", default=str(DEFAULT_DATASET / "context_sufficiency_prepare_v2_3.json"))
    parser.add_argument("--run-report", default=str(DEFAULT_DATASET / "context_sufficiency_run_v2_3.json"))
    parser.add_argument("--status-report", default=str(DEFAULT_DATASET / "context_sufficiency_status_v2_3.json"))
    parser.add_argument("--materialize-report", default=str(DEFAULT_DATASET / "context_sufficiency_materialization_v2_3.json"))
    parser.add_argument("--model", default="qwen3.8-27b-efficientthink-simpo-lynnstyle")
    parser.add_argument("--endpoint", default="http://127.0.0.1:1234/v1/completions")
    parser.add_argument("--sample-ids", default="")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--merge-glob", default="context_sufficiency_results_v2_3.part-*-of-*.jsonl")
    parser.add_argument("--merge-report", default=str(DEFAULT_DATASET / "context_sufficiency_merge_v2_3.json"))
    parser.add_argument("--sample-ids-file", default="")
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-recovery-turns", type=int, default=4)
    parser.add_argument("--max-recovery-gap", type=float, default=8.0)
    parser.add_argument("--sleep", type=float, default=0.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        result = prepare(args)
    elif args.command == "judge":
        result = judge(args)
    elif args.command == "merge":
        result = merge(args)
    elif args.command == "status":
        result = status(args)
    else:
        result = materialize(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
