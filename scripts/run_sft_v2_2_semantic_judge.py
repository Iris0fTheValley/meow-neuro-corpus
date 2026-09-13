from __future__ import annotations

"""Prepare and run the actual local semantic relation judge for v2.2."""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets" / "meow_v02_sft_v2_2_semantic_verified"
STRUCTURAL = OUT / "structural_candidates_v2_2.jsonl"
REQUESTS = OUT / "semantic_judge_requests_v2_2.jsonl"
RESULTS = OUT / "semantic_judge_results_v2_2.jsonl"
RUN = OUT / "semantic_judge_run_v2_2.json"

sys.path.insert(0, str(ROOT / "scripts"))
import sft_semantic_verified_v2_1 as v21  # noqa: E402

v21.JUDGE_PROMPT_VERSION = "semantic-relation-v2.2-incremental-p1"
MODEL_DEFAULT = "qwen35-9b-persona-epoch4-nothink"
STRICT_SYSTEM_PROMPT = (
    "You are an independent, strict conversation-forensics verifier. "
    "Judge only whether the exact candidate_context is an observable trigger for the exact candidate response episode. "
    "The preceding and following raw turns are diagnostic evidence only; they cannot substitute for a missing trigger or be silently inserted into candidate_context. "
    "Reject or mark UNCLEAR when the candidate context and response are merely adjacent, when a topic appears only in omitted neighboring turns, or when a hidden chat/donation/UI trigger is plausible but not observed. "
    "Neuro's absurdity, teasing, abrupt style, nonstandard grammar, and topic hijack are allowed only after a real observable interaction trigger. "
    "For episode merging, require explicit same-thought or grammatical continuation; a new self-contained sentence, new speech act, or new topic must be split or marked UNCLEAR even if the same speaker continues. "
    "Return only the requested JSON object and never rewrite transcript text."
)


def load_rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                yield row


def prepare() -> dict:
    fusion = v21.load_fusion()
    registry = v21.load_registry()
    rows = list(load_rows(STRUCTURAL))
    timelines = v21.load_timelines({str(row.get("source_id")) for row in rows}, fusion)
    REQUESTS.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    missing = 0
    with REQUESTS.open("w", encoding="utf-8") as handle:
        for row in rows:
            request = v21.make_judge_request(row, timelines.get(str(row.get("source_id")), {}), registry)
            if request is None:
                missing += 1
                continue
            request["request"]["judge_prompt_version"] = v21.JUDGE_PROMPT_VERSION
            request["judge_model"] = MODEL_DEFAULT
            request["judge_version"] = "sft-v2.2-semantic-judge-2026-09-13"
            handle.write(json.dumps(request, ensure_ascii=False) + "\n")
            count += 1
    report = {"status": "PREPARED", "structural_rows": len(rows), "judge_requests": count, "missing_raw_timeline": missing, "model": MODEL_DEFAULT, "prompt_version": v21.JUDGE_PROMPT_VERSION, "created_at": datetime.now(timezone.utc).isoformat()}
    (OUT / "semantic_judge_prepare_v2_2.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def latest_results() -> dict[str, dict]:
    result = {}
    if not RESULTS.exists():
        return result
    for row in load_rows(RESULTS):
        result[str(row.get("sample_id"))] = row
    return result


def judge(args: argparse.Namespace) -> dict:
    requests = list(load_rows(REQUESTS))
    results_path = Path(args.results) if args.results else (OUT / ("semantic_judge_results_v2_2_strict.jsonl" if args.strict else "semantic_judge_results_v2_2.jsonl"))
    run_path = OUT / ("semantic_judge_strict_run_v2_2.json" if args.strict else "semantic_judge_run_v2_2.json")
    existing = {}
    if args.retry_invalid and results_path.exists():
        for row in load_rows(results_path):
            existing[str(row.get("sample_id"))] = row
    if not args.retry_invalid and results_path.exists():
        results_path.unlink()
    processed = 0
    valid = 0
    accepted = 0
    errors = 0
    mode = "a" if args.retry_invalid else "w"
    system_prompt_previous = v21.SYSTEM_PROMPT
    prompt_previous = v21.JUDGE_PROMPT_VERSION
    if args.strict:
        v21.SYSTEM_PROMPT = STRICT_SYSTEM_PROMPT
        v21.JUDGE_PROMPT_VERSION = "semantic-relation-v2.2-independent-strict-p1"
    try:
      with results_path.open(mode, encoding="utf-8") as handle:
        selected_requests = requests
        if args.sample_ids:
            wanted = {value.strip() for value in args.sample_ids.split(",") if value.strip()}
            selected_requests = [item for item in requests if str(item.get("sample_id")) in wanted]
        elif args.max_items:
            selected_requests = requests[:args.max_items]
        for item in selected_requests:
            sample_id = str(item.get("sample_id"))
            prior = existing.get(sample_id)
            if prior and prior.get("validated", {}).get("valid") and not args.retry_invalid:
                continue
            if prior and prior.get("validated", {}).get("valid") and args.retry_invalid:
                continue
            request_payload = dict(item["request"])
            if args.strict:
                request_payload["strict_verifier_constraints"] = [
                    "Candidate context must itself contain the observable trigger; do not borrow a topic from preceding_raw_turns.",
                    "If the response is only plausibly related through an unobserved chat, donation, UI, or visual event, use HIDDEN_TRIGGER_SUSPECTED or UNCLEAR.",
                    "Do not accept a candidate merely because it is adjacent in time or because a preceding raw turn makes it seem related.",
                    "Only merge a target fragment when it is a grammatical or unmistakable same-thought continuation; otherwise select only the current target or mark UNCLEAR.",
                ]
            output = None
            for attempt in range(3):
                try:
                    raw = v21.call_judge(request_payload, model=args.model, endpoint=args.endpoint)
                    validation = v21.validate_judgement(raw, request_payload)
                    if validation.get("valid"):
                        valid += 1
                        accepted += int(bool(validation.get("accepted")))
                    else:
                        errors += 1
                    output = {"sample_id": sample_id, "judge_model": args.model, "judge_version": "sft-v2.2-independent-strict-2026-09-13" if args.strict else "sft-v2.2-semantic-judge-2026-09-13", "judge_prompt_version": v21.JUDGE_PROMPT_VERSION, "validated": validation, "raw": raw, "judge_input": request_payload, "attempts": attempt + 1, "completed_at": datetime.now(timezone.utc).isoformat()}
                    break
                except Exception as exc:
                    if attempt == 2:
                        errors += 1
                        output = {"sample_id": sample_id, "judge_model": args.model, "judge_version": "sft-v2.2-independent-strict-2026-09-13" if args.strict else "sft-v2.2-semantic-judge-2026-09-13", "judge_prompt_version": v21.JUDGE_PROMPT_VERSION, "validated": {"valid": False, "accepted": False, "reason": "judge_request_error", "error": str(exc)[:500]}, "judge_input": request_payload, "attempts": attempt + 1, "completed_at": datetime.now(timezone.utc).isoformat()}
                    else:
                        time.sleep(0.5 * (attempt + 1))
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")
            handle.flush()
            processed += 1
            if processed % 50 == 0:
                run_path.write_text(json.dumps({"status": "RUNNING", "total": len(selected_requests), "processed": processed, "valid": valid, "accepted": accepted, "errors": errors, "model": args.model, "prompt_version": v21.JUDGE_PROMPT_VERSION, "updated_at": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False, indent=2), encoding="utf-8")
            if args.sleep:
                time.sleep(args.sleep)
      result = {"status": "COMPLETED", "total": len(selected_requests), "processed": processed, "valid": valid, "accepted": accepted, "errors": errors, "model": args.model, "prompt_version": v21.JUDGE_PROMPT_VERSION, "completed_at": datetime.now(timezone.utc).isoformat()}
      run_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
      v21.SYSTEM_PROMPT = system_prompt_previous
      v21.JUDGE_PROMPT_VERSION = prompt_previous
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "judge"])
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--endpoint", default="http://127.0.0.1:1234/v1/chat/completions")
    parser.add_argument("--retry-invalid", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--results", default="")
    parser.add_argument("--sample-ids", default="")
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(judge(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
