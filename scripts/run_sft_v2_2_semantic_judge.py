from __future__ import annotations

"""Resumable primary/strict/adjudication interaction judge entry point.

The v2.2 filename remains the supported entry point. New artifacts are schema
v2.0 and cannot be confused with the legacy single-judge v2.2 results.
"""

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets" / "meow_v02_sft_v2_2_semantic_verified"
STRUCTURAL = OUT / "structural_candidates_v2_2.jsonl"
REQUESTS = OUT / "interaction_judge_requests_v2_3.jsonl"
MODEL_SOURCE_PATH = r"J:\AI friend\MEOW Qwen3.5\Qwen3.8-27B-EfficientThink-SimPO-Q4-LynnStyle.gguf"
MODEL_DEFAULT = "qwen3.8-27b-efficientthink-simpo-lynnstyle"
PRIMARY_QUARANTINE_FILENAME = "interaction_judge_primary_quarantine_v2_3.jsonl"
STRICT_QUARANTINE_FILENAME = "interaction_judge_strict_quarantine_v2_3.jsonl"

# These ids were explicitly reviewed by the production operator after the
# bounded retry budget was exhausted.  They are the only primary invalids that
# this lifecycle command may quarantine without a new review artifact.
REVIEWED_TERMINAL_PRIMARY_INVALID_IDS = frozenset(
    {
        "08798f7812e92b718f29",
        "194196a714e847c0840a",
        "8b1a6bbd75eb9c80f03e",
    }
)

sys.path.insert(0, str(ROOT / "scripts"))
import build_sft_v2_2_structural_candidates as structural_core  # noqa: E402
import sft_interaction_semantic_closure as closure  # noqa: E402
import sft_semantic_verified_v2_1 as v21  # noqa: E402

SYSTEM_PROMPTS = {
    "primary": (
        "You are a conservative interaction-forensics judge. Select only offered raw turn ids. "
        "The selected context itself must observably explain the response; time adjacency is not evidence. "
        "Do not reject Neuro for absurdity, teasing, abruptness, brevity, or topic hijack after a real trigger. "
        "Never infer identity, invent hidden events, or rewrite transcript. Return exactly one JSON object matching output_schema."
    ),
    "strict": (
        "You are a separate strict interaction verifier. Diagnostic preceding/following turns cannot substitute for selected context. "
        "Suspected chat, donation, UI, or visual triggers must be HIDDEN_TRIGGER_SUSPECTED or UNCLEAR. "
        "Merge only grammatical or unmistakable same-thought continuation. Select only legal ids, never rewrite transcript, and return one JSON object."
    ),
    "adjudication": (
        "You are the final evidence adjudicator for a documented judge conflict. Decide only from supplied raw-turn evidence and bounded selections. "
        "Do not compromise between judges, borrow diagnostic turns, invent events, infer identity, or rewrite transcript. UNCLEAR is valid. Return one JSON object."
    ),
}
PROMPT_VERSIONS = {"primary": closure.PRIMARY_PROMPT_VERSION, "strict": closure.STRICT_PROMPT_VERSION, "adjudication": closure.ADJUDICATION_PROMPT_VERSION}


def load_rows(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                yield row


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def result_path(stage: str) -> Path:
    return OUT / f"interaction_judge_{stage}_results_v2_3.jsonl"


def quarantine_path() -> Path:
    return OUT / PRIMARY_QUARANTINE_FILENAME


def strict_quarantine_path() -> Path:
    return OUT / STRICT_QUARANTINE_FILENAME


def run_path(stage: str) -> Path:
    return OUT / f"interaction_judge_{stage}_run_v2_3.json"


def latest_results(path: Path) -> dict[str, dict]:
    return {str(row.get("sample_id")): row for row in load_rows(path)}


def result_history(path: Path, sample_id: str) -> list[dict]:
    return [row for row in load_rows(path) if str(row.get("sample_id")) == str(sample_id)]


def invalid_lifecycle(
    current: dict | None,
    history: list[dict],
    *,
    retry_budget: int,
    terminal_eligible: bool,
) -> dict:
    """Classify an invalid result without altering or coercing its payload."""
    validation = (current or {}).get("validated") or {}
    retry_count = max(0, len(history) - 1)
    result = {
        "state": "VALID" if validation.get("valid") else "RETRYABLE_INVALID",
        "retry_count": retry_count,
        "retry_budget": retry_budget,
        "latest_reason": validation.get("reason") or "invalid_result",
    }
    if not validation.get("valid") and terminal_eligible and retry_count >= retry_budget:
        result["state"] = "TERMINAL_INVALID_QUARANTINED"
    return result


def load_terminal_quarantine(path: Path | None = None) -> dict[str, dict]:
    values = latest_results(path or quarantine_path())
    return {sample_id: row for sample_id, row in values.items() if row.get("state") == "TERMINAL_INVALID_QUARANTINED"}


def _sample_ids_arg(args: argparse.Namespace) -> set[str]:
    values = {value.strip() for value in str(getattr(args, "sample_ids", "") or "").split(",") if value.strip()}
    sample_ids_file = str(getattr(args, "sample_ids_file", "") or "")
    if sample_ids_file:
        values.update(line.strip() for line in Path(sample_ids_file).read_text(encoding="utf-8").splitlines() if line.strip())
    return values


def quarantine(args: argparse.Namespace) -> dict:
    """Write explicitly reviewed terminal invalids to a fail-closed artifact."""
    stage = str(getattr(args, "stage", "primary") or "primary")
    if stage not in {"primary", "strict"}:
        raise SystemExit("quarantine supports only primary or strict stage")
    requests = {str(row.get("sample_id")): row for row in load_rows(Path(args.requests))}
    source_path = Path(args.primary_results) if stage == "primary" and args.primary_results else result_path(stage)
    source = latest_results(source_path)
    requested_ids = _sample_ids_arg(args) or set(REVIEWED_TERMINAL_PRIMARY_INVALID_IDS)
    if stage == "strict" and not _sample_ids_arg(args):
        requested_ids = set()
    retry_budget = max(0, int(args.retry_budget))
    rows = []
    skipped = []
    for sample_id in sorted(requested_ids):
        request_row = requests.get(sample_id)
        current = source.get(sample_id)
        history = result_history(source_path, sample_id)
        lifecycle = invalid_lifecycle(
            current,
            history,
            retry_budget=retry_budget,
            terminal_eligible=(
                (stage == "primary" and (sample_id in REVIEWED_TERMINAL_PRIMARY_INVALID_IDS or bool(_sample_ids_arg(args))))
                or (stage == "strict" and bool(_sample_ids_arg(args)))
            ),
        )
        if not request_row or not current or lifecycle["state"] != "TERMINAL_INVALID_QUARANTINED":
            skipped.append({"sample_id": sample_id, "state": lifecycle["state"], "retry_count": lifecycle["retry_count"]})
            continue
        rows.append(
            {
                "schema_version": closure.SCHEMA_VERSION,
                "pipeline_version": closure.PIPELINE_VERSION,
                "artifact_status": "TERMINAL_INVALID_QUARANTINED",
                "state": "TERMINAL_INVALID_QUARANTINED",
                "stage": stage,
                "sample_id": sample_id,
                "request_sha256": request_row.get("request_sha256"),
                "invalid_reason": (current.get("validated") or {}).get("reason") or "invalid_result",
                "retry_count": lifecycle["retry_count"],
                "retry_budget": retry_budget,
                "retry_history": [
                    {
                        "request_sha256": value.get("request_sha256"),
                        "validated": value.get("validated"),
                        "attempts": value.get("attempts"),
                        "completed_at": value.get("completed_at"),
                        "raw": value.get("raw"),
                    }
                    for value in history
                ],
                "latest_raw_result": current.get("raw"),
                "latest_result": current,
                "quarantined_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    destination = Path(args.quarantine) if args.quarantine else (quarantine_path() if stage == "primary" else strict_quarantine_path())
    existing = load_terminal_quarantine(destination)
    existing.update({str(row["sample_id"]): row for row in rows})
    write_jsonl(destination, [existing[sample_id] for sample_id in sorted(existing)])
    report = {
        "schema_version": closure.SCHEMA_VERSION,
        "pipeline_version": closure.PIPELINE_VERSION,
        "status": "COMPLETED",
        "artifact": str(destination),
        "retry_budget": retry_budget,
        "requested": len(requested_ids),
        "terminal_quarantined": len(rows),
        "skipped": skipped,
        "sample_ids": sorted(row["sample_id"] for row in rows),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(OUT / f"interaction_judge_{stage}_quarantine_run_v2_3.json", report)
    return report


def result_compatibility(result: dict | None, request_row: dict, stage: str, model: str) -> tuple[bool, str]:
    """Return whether a cached result is valid for this exact stage request."""
    if not result:
        return False, "MISSING"
    expected = {
        "sample_id": str(request_row.get("sample_id")),
        "request_sha256": request_row.get("request_sha256"),
        "schema_version": closure.SCHEMA_VERSION,
        "pipeline_version": closure.PIPELINE_VERSION,
        "stage": stage,
        "judge_prompt_version": PROMPT_VERSIONS[stage],
        "judge_model": model,
    }
    for key, value in expected.items():
        actual = str(result.get(key)) if key == "sample_id" else result.get(key)
        if actual != value:
            return False, f"STALE_{key.upper()}"
    if not (result.get("validated") or {}).get("valid"):
        return False, "INVALID_RESULT"
    return True, "COMPATIBLE"


def _annotate_timeline(timeline: dict) -> None:
    previous_end = None
    for index, turn in enumerate(timeline.get("_turns") or []):
        turn["text_qa"], turn["text_qa_reasons"] = structural_core.text_quality(turn.get("text"))
        turn["asr_quality"], turn["asr_quality_reasons"], turn["asr_metadata"] = structural_core.raw_asr_quality(turn)
        turn["_gap_from_previous"] = 0.0 if previous_end is None else max(0.0, float(turn.get("_start", index)) - previous_end)
        previous_end = float(turn.get("_end", index))


def prepare(args: argparse.Namespace) -> dict:
    fusion, registry = v21.load_fusion(), v21.load_registry()
    rows = list(load_rows(Path(args.structural)))
    if not rows:
        raise SystemExit(f"structural artifact is missing or empty: {args.structural}")
    incompatible = [row.get("sample_id") for row in rows if row.get("artifact_schema_version") != structural_core.STRUCTURAL_SCHEMA_VERSION or row.get("pipeline_version") != structural_core.STRUCTURAL_PIPELINE_VERSION]
    if incompatible:
        raise SystemExit(f"structural artifact has {len(incompatible)} legacy/incompatible rows; rerun build_sft_v2_2_structural_candidates.py")
    timelines = v21.load_timelines({str(row.get("source_id")) for row in rows}, fusion)
    for timeline in timelines.values():
        _annotate_timeline(timeline)
    missing, count = [], 0
    Path(args.requests).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.requests).open("w", encoding="utf-8") as handle:
        for row in rows:
            source_id = str(row.get("source_id"))
            request = closure.build_request(row, timelines.get(source_id, {}), registry.get(source_id, {}), max_context_extension=args.max_context_extension)
            if request is None:
                missing.append({"sample_id": row.get("sample_id"), "reason": "MISSING_OR_INVALID_BOUNDED_RAW_EVIDENCE"})
                continue
            request.update({"artifact_schema_version": closure.SCHEMA_VERSION, "pipeline_version": closure.PIPELINE_VERSION})
            request["request_sha256"] = closure.payload_sha256(request["request"])
            handle.write(json.dumps(request, ensure_ascii=False) + "\n")
            count += 1
    report = {
        "schema_version": closure.SCHEMA_VERSION,
        "pipeline_version": closure.PIPELINE_VERSION,
        "status": "PREPARED",
        "structural_rows": len(rows),
        "judge_requests": count,
        "missing_or_invalid_raw_evidence": len(missing),
        "missing_examples": missing[:100],
        "stages": ["primary", "strict", "adjudication"],
        "prompt_versions": PROMPT_VERSIONS,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(OUT / "interaction_closure_prepare_v2_3.json", report)
    return report


def _selected_requests(args: argparse.Namespace, requests: list[dict]) -> list[dict]:
    if args.sample_ids:
        wanted = {value.strip() for value in args.sample_ids.split(",") if value.strip()}
        return [row for row in requests if str(row.get("sample_id")) in wanted]
    if args.sample_ids_file:
        wanted = {line.strip() for line in Path(args.sample_ids_file).read_text(encoding="utf-8").splitlines() if line.strip()}
        return [row for row in requests if str(row.get("sample_id")) in wanted]
    return requests[: args.max_items] if args.max_items else requests


def strict_required_sample_ids(requests: list[dict], primary_results: dict[str, dict], expected_model: str, quarantined_ids: set[str] | None = None) -> set[str]:
    request_by_id = {str(row.get("sample_id")): row for row in requests}
    quarantined_ids = quarantined_ids or set()
    return {
        sample_id
        for sample_id, row in primary_results.items()
        if sample_id in request_by_id
        and sample_id not in quarantined_ids
        and result_compatibility(row, request_by_id[sample_id], "primary", expected_model)[0]
        and (row.get("validated") or {}).get("accepted")
    }


def judge(args: argparse.Namespace) -> dict:
    stage = args.stage
    available = list(load_rows(Path(args.requests)))
    targeted = bool(args.sample_ids or args.sample_ids_file)
    if stage == "strict" and not targeted:
        primary_results = latest_results(result_path("primary"))
        expected_primary_model = args.primary_model or args.model
        required = strict_required_sample_ids(available, primary_results, expected_primary_model, set(load_terminal_quarantine()))
        required -= set(load_terminal_quarantine(strict_quarantine_path()))
        available = [row for row in available if str(row.get("sample_id")) in required]
    selected = _selected_requests(args, available)
    destination = Path(args.results) if args.results else result_path(stage)
    existing = latest_results(destination) if destination.exists() else {}
    if args.fresh and destination.exists():
        destination.unlink()
        existing = {}
    if stage == "adjudication" and not args.sample_ids and not args.sample_ids_file:
        raise SystemExit("adjudication requires targeted sample ids from the conflict report")
    counters = Counter()
    previous_prompt = v21.SYSTEM_PROMPT
    try:
        v21.SYSTEM_PROMPT = SYSTEM_PROMPTS[stage]
        with destination.open("a" if destination.exists() else "w", encoding="utf-8") as handle:
            for item in selected:
                sample_id = str(item.get("sample_id"))
                prior = existing.get(sample_id)
                compatible, cache_reason = result_compatibility(prior, item, stage, args.model)
                if compatible:
                    counters["skipped_existing"] += 1
                    continue
                if prior and cache_reason.startswith("STALE_"):
                    counters["stale_rerun"] += 1
                elif prior:
                    counters["invalid_rerun"] += 1
                request_payload = dict(item.get("request") or {})
                if request_payload.get("schema_version") != closure.SCHEMA_VERSION:
                    counters["invalid_request_schema"] += 1
                    continue
                output = None
                for attempt in range(1, args.max_attempts + 1):
                    raw = v21.call_judge(request_payload, model=args.model, endpoint=args.endpoint)
                    validation = closure.validate_judgement(raw, request_payload)
                    output = {"schema_version": closure.SCHEMA_VERSION, "pipeline_version": closure.PIPELINE_VERSION, "sample_id": sample_id, "stage": stage, "request_sha256": item.get("request_sha256"), "judge_model": args.model, "judge_prompt_version": PROMPT_VERSIONS[stage], "validated": validation, "raw": raw, "judge_input": request_payload, "attempts": attempt, "completed_at": datetime.now(timezone.utc).isoformat()}
                    if validation.get("valid"):
                        break
                    if attempt < args.max_attempts:
                        time.sleep(0.5 * attempt)
                handle.write(json.dumps(output, ensure_ascii=False) + "\n")
                handle.flush()
                counters["processed"] += 1
                counters["valid"] += int(bool((output or {}).get("validated", {}).get("valid")))
                counters["accepted"] += int(bool((output or {}).get("validated", {}).get("accepted")))
                counters["invalid"] += int(not bool((output or {}).get("validated", {}).get("valid")))
                if args.sleep:
                    time.sleep(args.sleep)
    finally:
        v21.SYSTEM_PROMPT = previous_prompt
    result = {"schema_version": closure.SCHEMA_VERSION, "pipeline_version": closure.PIPELINE_VERSION, "status": "COMPLETED", "stage": stage, "selected": len(selected), **dict(counters), "model": args.model, "prompt_version": PROMPT_VERSIONS[stage], "results": str(destination), "completed_at": datetime.now(timezone.utc).isoformat()}
    run_report = getattr(args, "run_report", "")
    write_json(Path(run_report) if run_report else run_path(stage), result)
    return result


def status(args: argparse.Namespace) -> dict:
    requests = list(load_rows(Path(args.requests)))
    request_by_id = {str(row.get("sample_id")): row for row in requests}
    requested = {str(row.get("sample_id")) for row in requests}
    report = {"schema_version": closure.SCHEMA_VERSION, "pipeline_version": closure.PIPELINE_VERSION, "requests": len(requests), "stages": {}}
    primary_values = latest_results(result_path("primary"))
    primary_model = args.primary_model or args.model
    strict_model = args.strict_model or args.model
    adjudication_model = args.adjudication_model or args.model
    primary_quarantine = load_terminal_quarantine()
    strict_quarantine = load_terminal_quarantine(strict_quarantine_path())
    strict_required = strict_required_sample_ids(requests, primary_values, primary_model, set(primary_quarantine))
    decision_rows = latest_results(OUT / "interaction_closure_decisions_v2_3.jsonl")
    adjudication_required = {sample_id for sample_id, row in decision_rows.items() if row.get("state") == "JUDGE_CONFLICT"}
    required_by_stage = {"primary": requested, "strict": strict_required, "adjudication": adjudication_required}
    model_by_stage = {"primary": primary_model, "strict": strict_model, "adjudication": adjudication_model}
    for stage in ("primary", "strict", "adjudication"):
        values = latest_results(result_path(stage))
        required = required_by_stage[stage]
        compatible = {sample_id for sample_id in required if result_compatibility(values.get(sample_id), request_by_id.get(sample_id, {}), stage, model_by_stage[stage])[0]}
        stale = {sample_id for sample_id in required if sample_id in values and not result_compatibility(values.get(sample_id), request_by_id.get(sample_id, {}), stage, model_by_stage[stage])[0]}
        missing = required - set(values)
        quarantine_ids = set(primary_quarantine) if stage == "primary" else set(strict_quarantine) if stage == "strict" else set()
        quarantined = required & quarantine_ids
        if stage == "primary":
            quarantined = set(sample_id for sample_id in primary_quarantine if sample_id in required)
            stale = stale - quarantined
        elif stage == "strict":
            stale = stale - quarantined
        missing = (required - set(values)) - quarantined
        valid_accepted = {sample_id for sample_id in compatible if (values.get(sample_id) or {}).get("validated", {}).get("accepted")}
        valid_nonaccepted = compatible - valid_accepted
        stale_reasons = Counter(result_compatibility(values.get(sample_id), request_by_id.get(sample_id, {}), stage, model_by_stage[stage])[1] for sample_id in stale)
        report["stages"][stage] = {"required": len(required), "valid_current": len(compatible), "accepted": len(valid_accepted), "rejected_nonaccepted": len(valid_nonaccepted), "stale_or_invalid": len(stale), "retryable_invalid": len(stale), "terminal_quarantined": len(quarantined), "stale_reason_counts": dict(stale_reasons), "stale_examples": sorted(stale)[:50], "missing": len(missing), "missing_examples": sorted(missing)[:50], "rerun_required": len(stale | missing), "execution_complete": len(compatible) + len(quarantined) == len(required) and not stale and not missing}
        if stage == "primary":
            report["stages"][stage]["primary_valid"] = len(compatible)
            report["stages"][stage]["primary_accepted"] = len(valid_accepted)
            report["stages"][stage]["primary_rejected_nonaccepted"] = len(valid_nonaccepted)
            report["stages"][stage]["primary_terminal_quarantined"] = len(quarantined)
        if stage == "strict":
            report["stages"][stage]["strict_terminal_quarantined"] = len(quarantined)
    write_json(OUT / "interaction_judge_execution_status_v2_3.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Interaction semantic closure workflow")
    parser.add_argument("command", choices=["prepare", "judge", "status", "quarantine"])
    parser.add_argument("--stage", choices=["primary", "strict", "adjudication"], default="primary")
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--primary-model", default="", help="expected primary model when routing/status checks another stage")
    parser.add_argument("--strict-model", default="", help="expected strict model for status")
    parser.add_argument("--adjudication-model", default="", help="expected adjudication model for status")
    parser.add_argument("--endpoint", default="http://127.0.0.1:1234/v1/completions")
    parser.add_argument("--structural", default=str(STRUCTURAL))
    parser.add_argument("--requests", default=str(REQUESTS))
    parser.add_argument("--results", default="")
    parser.add_argument("--run-report", default="", help="per-worker run report path; keeps sharded workers from sharing the canonical report")
    parser.add_argument("--primary-results", default="", help="primary result artifact for quarantine lifecycle")
    parser.add_argument("--quarantine", default="", help="terminal quarantine artifact for the selected stage")
    parser.add_argument("--fresh", action="store_true", help="explicitly discard only the selected stage result file")
    parser.add_argument("--retry-invalid", action="store_true")
    parser.add_argument("--sample-ids", default="")
    parser.add_argument("--sample-ids-file", default="")
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-budget", type=int, default=5)
    parser.add_argument("--max-context-extension", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()
    result = prepare(args) if args.command == "prepare" else judge(args) if args.command == "judge" else status(args) if args.command == "status" else quarantine(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
