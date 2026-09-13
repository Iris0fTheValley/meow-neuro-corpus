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
MODEL_DEFAULT = "qwen35-9b-persona-epoch4-nothink"

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
        "You are an independent strict interaction verifier. Diagnostic preceding/following turns cannot substitute for selected context. "
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


def result_path(stage: str) -> Path:
    return OUT / f"interaction_judge_{stage}_results_v2_3.jsonl"


def run_path(stage: str) -> Path:
    return OUT / f"interaction_judge_{stage}_run_v2_3.json"


def latest_results(path: Path) -> dict[str, dict]:
    return {str(row.get("sample_id")): row for row in load_rows(path)}


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


def judge(args: argparse.Namespace) -> dict:
    stage = args.stage
    available = list(load_rows(Path(args.requests)))
    targeted = bool(args.sample_ids or args.sample_ids_file)
    if stage == "strict" and not targeted:
        primary_results = latest_results(result_path("primary"))
        required = {sample_id for sample_id, row in primary_results.items() if (row.get("validated") or {}).get("accepted")}
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
                if prior and not args.retry_invalid:
                    counters["skipped_existing"] += 1
                    continue
                if prior and args.retry_invalid and (prior.get("validated") or {}).get("valid"):
                    counters["skipped_valid"] += 1
                    continue
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
    write_json(run_path(stage), result)
    return result


def status(args: argparse.Namespace) -> dict:
    requests = list(load_rows(Path(args.requests)))
    requested = {str(row.get("sample_id")) for row in requests}
    report = {"schema_version": closure.SCHEMA_VERSION, "pipeline_version": closure.PIPELINE_VERSION, "requests": len(requests), "stages": {}}
    primary_values = latest_results(result_path("primary"))
    strict_required = {sample_id for sample_id, row in primary_values.items() if (row.get("validated") or {}).get("accepted")}
    decision_rows = latest_results(OUT / "interaction_closure_decisions_v2_3.jsonl")
    adjudication_required = {sample_id for sample_id, row in decision_rows.items() if row.get("state") == "JUDGE_CONFLICT"}
    required_by_stage = {"primary": requested, "strict": strict_required, "adjudication": adjudication_required}
    for stage in ("primary", "strict", "adjudication"):
        values = latest_results(result_path(stage))
        valid = {sample_id for sample_id, row in values.items() if (row.get("validated") or {}).get("valid")}
        required = required_by_stage[stage]
        report["stages"][stage] = {"required": len(required), "valid": len(valid & required), "invalid": len((set(values) - valid) & required), "missing": len(required - set(values))}
    write_json(OUT / "interaction_judge_execution_status_v2_3.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Interaction semantic closure workflow")
    parser.add_argument("command", choices=["prepare", "judge", "status"])
    parser.add_argument("--stage", choices=["primary", "strict", "adjudication"], default="primary")
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--endpoint", default="http://127.0.0.1:1234/v1/chat/completions")
    parser.add_argument("--structural", default=str(STRUCTURAL))
    parser.add_argument("--requests", default=str(REQUESTS))
    parser.add_argument("--results", default="")
    parser.add_argument("--fresh", action="store_true", help="explicitly discard only the selected stage result file")
    parser.add_argument("--retry-invalid", action="store_true")
    parser.add_argument("--sample-ids", default="")
    parser.add_argument("--sample-ids-file", default="")
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-context-extension", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()
    result = prepare(args) if args.command == "prepare" else judge(args) if args.command == "judge" else status(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
