from __future__ import annotations

"""Stratified semantic content audit of final materialized rows.

The audit is review evidence, not a quality gate replacement.  The model sees
only canonical selected turns and the final materialized messages for each
sample; no diagnostic neighbours or hidden metadata are supplied.
"""

import argparse
import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

PROMPT_VERSION = "semantic-content-audit-v1-2026-09-16"
SCHEMA_VERSION = "2.0.0"
SYSTEM_PROMPT = (
    "You are a strict semantic content auditor for a speech-derived dialogue corpus. "
    "Inspect only the canonical selected turns and the final materialized messages shown. "
    "Do not use external knowledge, infer hidden events, or rewrite transcript text. "
    "Neuro may be absurd, terse, sharp, or topic-shifting; do not mark those traits as errors. "
    "Flag only clear context-response failure, speaker/trigger mismatch, mechanical trajectory stitching, "
    "near-duplicate supervision, or transcript rewriting. Return exactly one JSON object matching output_schema."
)


def load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except Exception:
            continue
        if isinstance(value, dict):
            result.append(value)
    return result


def sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _risk(row: dict[str, Any]) -> tuple[int, float, str]:
    suff = row.get("context_sufficiency") or {}
    closure = row.get("semantic_closure") or {}
    flags = (closure.get("risk_features") or {}).get("flags") or []
    return (len(flags), float(suff.get("confidence") or 1.0), str(row.get("sample_id")))


def build_sample_sets(interaction: list[dict[str, Any]], trajectory: list[dict[str, Any]], high_precision: list[dict[str, Any]], sample_size: int, seed: int) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    categories: dict[str, set[str]] = {}
    def add(rows: list[dict[str, Any]], category: str) -> None:
        for row in rows:
            sid = str(row.get("sample_id"))
            by_id[sid] = row
            categories.setdefault(sid, set()).add(category)
    ordered = sorted(interaction, key=lambda row: str(row.get("sample_id")))
    rng = random.Random(seed)
    random_rows = ordered[:]
    rng.shuffle(random_rows)
    add(random_rows[:sample_size], "random_interaction")
    add(sorted(interaction, key=_risk, reverse=True)[:sample_size], "high_risk_interaction")
    add(sorted(trajectory, key=lambda row: str(row.get("sample_id")))[:sample_size], "trajectory")
    add(sorted(high_precision, key=lambda row: str(row.get("sample_id")))[:sample_size], "high_precision")
    # A second ordinary slice makes the baseline explicit without requiring a
    # new random seed or hidden reviewer choices.
    add(ordered[sample_size : 2 * sample_size], "ordinary_interaction")
    return [
        {
            "sample_id": sid,
            "categories": sorted(categories[sid]),
            "row": by_id[sid],
        }
        for sid in sorted(by_id)
    ]


def build_requests(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    requests = []
    for sample in samples:
        row = sample["row"]
        tqa = row.get("transcript_qa") or {}
        # Keep the canonical/raw evidence, but avoid copying large ASR metric
        # blobs into the audit prompt.  This prevents endpoint context errors
        # while preserving the exact selected turn IDs, speakers, identities,
        # timestamps, transcript text, and machine QA flags.
        selected_turns = [
            {
                "turn_id": turn.get("turn_id"),
                "role": turn.get("role"),
                "speaker": turn.get("speaker"),
                "identity": turn.get("identity"),
                "text": turn.get("text"),
                "timestamp": turn.get("timestamp"),
                "text_qa": turn.get("text_qa"),
                "asr_quality": turn.get("asr_quality"),
                "boundary": turn.get("boundary"),
            }
            for turn in tqa.get("turns") or []
        ]
        payload = {
            "schema_version": SCHEMA_VERSION,
            "prompt_version": PROMPT_VERSION,
            "sample_id": sample["sample_id"],
            "categories": sample["categories"],
            "canonical_selected_turns": selected_turns,
            "final_materialized_messages": row.get("messages") or [],
            "output_schema": {
                "verdict": "PASS | FAIL | UNCLEAR",
                "confidence": "number 0..1",
                "context_response_relation": "REAL | WEAK_OR_MISSING | UNCLEAR",
                "speaker_trigger_match": "PASS | FAIL | UNCLEAR",
                "trajectory_continuity": "PASS | FAIL | NOT_APPLICABLE | UNCLEAR",
                "near_duplicate_suspected": "boolean",
                "transcript_rewrite_suspected": "boolean",
                "reason": "short evidence-based reason",
            },
        }
        requests.append({"sample_id": sample["sample_id"], "categories": sample["categories"], "audit_input": payload, "request_sha256": sha(payload), "pipeline_version": row.get("pipeline_version"), "artifact_schema_version": row.get("artifact_schema_version")})
    return requests


def parse_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def call(request_payload: dict[str, Any], model: str, endpoint: str) -> dict[str, Any]:
    prompt = "<|im_start|>system\n" + SYSTEM_PROMPT + "<|im_end|>\n<|im_start|>user\n" + json.dumps(request_payload, ensure_ascii=False) + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    body = {"model": model, "prompt": prompt, "temperature": 0, "max_tokens": 384, "stop": ["<|im_end|>"], "stream": False}
    req = Request(endpoint, data=json.dumps(body, ensure_ascii=False).encode("utf-8"), headers={"Content-Type": "application/json; charset=utf-8"}, method="POST")
    try:
        with urlopen(req, timeout=180) as response:
            payload = json.loads(response.read().decode("utf-8"))
        choice = payload.get("choices", [{}])[0]
        text = str(choice.get("text") or (choice.get("message") or {}).get("content") or "")
        parsed = parse_object(text)
        valid = isinstance(parsed, dict) and str(parsed.get("verdict") or "") in {"PASS", "FAIL", "UNCLEAR"} and 0.0 <= float(parsed.get("confidence")) <= 1.0
        return {"valid": valid, "parsed": parsed, "raw_content": text, "usage": payload.get("usage") or {}}
    except Exception as exc:
        return {"valid": False, "error": str(exc)[:500]}


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.dataset)
    interaction = load(root / "interaction_view_pool_v2_3.jsonl")
    trajectory = load(root / "trajectory_pool_v2_3.jsonl")
    high_precision = load(root / "views" / "high_precision_context_closed" / "train_high_precision.jsonl")
    samples = build_sample_sets(interaction, trajectory, high_precision, args.sample_size, args.seed)
    requests = build_requests(samples)
    req_path = Path(args.requests)
    req_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in requests), encoding="utf-8")
    result_path = Path(args.results)
    existing = {str(row.get("sample_id")): row for row in load(result_path)}
    results: list[dict[str, Any]] = []
    def one(request: dict[str, Any]) -> dict[str, Any]:
        prior = existing.get(str(request["sample_id"]))
        if prior and prior.get("request_sha256") == request["request_sha256"] and prior.get("valid"):
            return prior
        output = call(request["audit_input"], args.model, args.endpoint)
        return {"sample_id": request["sample_id"], "categories": request["categories"], "request_sha256": request["request_sha256"], "model": args.model, "prompt_version": PROMPT_VERSION, **output, "completed_at": datetime.now(timezone.utc).isoformat()}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(one, request) for request in requests]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: str(row.get("sample_id")))
    result_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results), encoding="utf-8")
    valid = [row for row in results if row.get("valid")]
    verdicts = {}
    for row in valid:
        verdict = str((row.get("parsed") or {}).get("verdict") or "UNKNOWN")
        verdicts[verdict] = verdicts.get(verdict, 0) + 1
    report = {"schema_version": SCHEMA_VERSION, "prompt_version": PROMPT_VERSION, "model": args.model, "sample_size_per_stratum": args.sample_size, "seed": args.seed, "rows": len(requests), "valid_results": len(valid), "invalid_results": len(results) - len(valid), "verdict_counts": verdicts, "categories": {category: sum(category in row.get("categories", []) for row in requests) for category in {c for row in requests for c in row.get("categories", [])}}, "failures": [{"sample_id": row.get("sample_id"), "parsed": row.get("parsed"), "categories": row.get("categories")} for row in valid if (row.get("parsed") or {}).get("verdict") == "FAIL"], "generated_at": datetime.now(timezone.utc).isoformat()}
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--model", default="qwen3.8-27b-efficientthink-simpo-lynnstyle")
    parser.add_argument("--endpoint", default="http://127.0.0.1:1234/v1/completions")
    parser.add_argument("--sample-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=230916)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
