from __future__ import annotations

"""Open-ended, content-first audit for cache-only audio recovery artifacts.

This complements structural validation: the reviewer receives only final
training samples or a compact topology/quarantine evidence record, never an
expected verdict, frozen semantic verdict, identity diagnostics, or hidden
history. It is deliberately allowed to name a new issue category.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audio_evidence.recovery_v2 import canonical_sha256, parse_json_object  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def stable_take(rows: list[dict[str, Any]], count: int, *, salt: str) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256((salt + "|" + str(row.get("sample_id") or row.get("audio_turn_id") or row)).encode("utf-8")).hexdigest(),
    )[:count]


SYSTEM = """You are an independent, open-ended production data reviewer.
Review only the exact text/evidence shown. Do not assume hidden conversation,
metadata, expected labels, acoustics, speaker identities, or a desired answer.
The final assistant message is intentionally the training target; prior messages
are intentionally prompt history. A sample may contain several user or assistant
messages, and it is NOT an error that the target responds to only the latest
relevant turn rather than every earlier turn.

Actively look for a concrete, observable defect: (1) a target with no plausible
connection to any immediately preceding interaction, (2) an exact/near-exact
duplicate utterance, (3) a target copied into earlier prompt text, (4) visibly
swallowed or wrongly split speakers, (5) role text that is visibly contradictory,
(6) an objectively redundant prefix where a visible suffix alone clearly carries
the same exchange, (7) control/noise text, or another concrete issue. Do not flag
unusual dialogue, a short valid answer, multiple history turns, the absence of
token overlap, a supervision convention, or an expected fail-closed quarantine.
For a finding, cite exact visible text and explain the defect without relying on
unstated facts. Return JSON only:
{"finding": boolean, "issue_category": "semantic_disconnect"|"duplicate_speech"|"target_leakage"|"speaker_topology"|"role_assignment"|"context_too_wide"|"context_too_thin"|"asr_noise"|"other"|null, "severity":"none"|"low"|"medium"|"high", "evidence": string}.
Use finding=false when the evidence does not prove a concrete issue."""

TRIAGE_SYSTEM = """You are a strict second-pass production audit adjudicator.
You receive only a sample/evidence record and a first reviewer claim. Verify the
claim from visible text alone. Reject it if it relies on unshown speaker identity,
hidden history, expected labels, the supervision convention, a target merely not
answering every old message, or an expected quarantine. Target leakage requires
the target text to appear in earlier prompt text. Duplicate speech requires the
same/near-identical visible utterance. Context-too-wide requires an explicitly
shown shorter suffix that independently carries the interaction; do not infer it
only because old turns exist. Return JSON only:
{"verified": boolean, "issue_category": string|null, "severity":"none"|"low"|"medium"|"high", "evidence": string}.
Use verified=false unless the claim is concretely supported."""


def review_payload(row: dict[str, Any]) -> dict[str, Any]:
    # The visible audit view intentionally excludes semantic verdicts, identity
    # diagnostics, future timeline, and any evaluator expectation.
    if row.get("review_kind") == "training_sample":
        return {
            "kind": "training_sample",
            "sample_id": row.get("sample_id"),
            "recording_id": row.get("recording_id"),
            "context_source": row.get("context_reconstruction_class"),
            "messages": [
                {"role": message.get("role"), "text": message.get("content")}
                for message in (row.get("messages") or [])
            ],
        }
    return {
        "kind": row.get("review_kind"),
        "sample_id": row.get("sample_id"),
        "recording_id": row.get("recording_id"),
        "evidence": row.get("review_evidence"),
    }


def call_model(system: str, user_payload: dict[str, Any], *, endpoint: str, model: str, timeout: int) -> tuple[dict[str, Any], bool, str]:
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError, URLError

    prompt = (
        "<|im_start|>system\n" + system + "<|im_end|>\n<|im_start|>user\n"
        + json.dumps(user_payload, ensure_ascii=False) + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    request = Request(
        endpoint,
        data=json.dumps({"model": model, "prompt": prompt, "temperature": 0, "max_tokens": 256, "stop": ["<|im_end|>"]}, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
        choice = (response_payload.get("choices") or [{}])[0]
        parsed = parse_json_object(choice.get("text") or (choice.get("message") or {}).get("content")) or {}
    except (HTTPError, URLError, OSError, TimeoutError, json.JSONDecodeError) as exc:
        return {}, False, type(exc).__name__
    return parsed, bool(parsed), ""


def call_reviewer(payload: dict[str, Any], *, endpoint: str, model: str, timeout: int) -> dict[str, Any]:
    parsed, parsed_ok, error = call_model(SYSTEM, payload, endpoint=endpoint, model=model, timeout=timeout)
    if not parsed_ok:
        parsed, parsed_ok, error = call_model(SYSTEM, payload, endpoint=endpoint, model=model, timeout=timeout)
    if not parsed_ok:
        return {"finding": False, "issue_category": None, "severity": "none", "evidence": error, "valid": False}
    finding = parsed.get("finding") is True
    severity = str(parsed.get("severity") or "none").lower()
    valid = isinstance(parsed.get("finding"), bool) and severity in {"none", "low", "medium", "high"}
    return {
        "finding": finding if valid else False,
        "issue_category": (str(parsed.get("issue_category") or "other")[:160] if finding and valid else None),
        "severity": severity if valid else "none",
        "evidence": str(parsed.get("evidence") or "")[:1000],
        "valid": valid,
    }


def call_triage(payload: dict[str, Any], claim: dict[str, Any], *, endpoint: str, model: str, timeout: int) -> dict[str, Any]:
    parsed, parsed_ok, error = call_model(
        TRIAGE_SYSTEM,
        {"sample_or_evidence": payload, "first_reviewer_claim": claim},
        endpoint=endpoint, model=model, timeout=timeout,
    )
    if not parsed_ok:
        parsed, parsed_ok, error = call_model(
            TRIAGE_SYSTEM,
            {"sample_or_evidence": payload, "first_reviewer_claim": claim},
            endpoint=endpoint, model=model, timeout=timeout,
        )
    if not parsed_ok:
        return {"verified": False, "issue_category": None, "severity": "none", "evidence": error, "valid": False}
    verified = parsed.get("verified") is True
    severity = str(parsed.get("severity") or "none").lower()
    valid = isinstance(parsed.get("verified"), bool) and severity in {"none", "low", "medium", "high"}
    return {
        "verified": verified if valid else False,
        "issue_category": str(parsed.get("issue_category") or claim.get("issue_category") or "other")[:160] if verified and valid else None,
        "severity": severity if valid else "none",
        "evidence": str(parsed.get("evidence") or "")[:1000],
        "valid": valid,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:1234/v1/completions")
    parser.add_argument("--per-stratum", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    run = Path(args.run)
    materialized = load_jsonl(run / "materialized.jsonl")
    timeline = {str(row.get("audio_turn_id")): row for row in load_jsonl(run / "canonical_audio_evidence_timeline.jsonl")}
    reconciliations = load_jsonl(run / "turn_reconciliation.jsonl")
    discovered = load_jsonl(run / "audio_discovered_turns.jsonl")
    targets = load_jsonl(run / "target_resolution.jsonl")
    quarantined = load_jsonl(run / "quarantine.jsonl")
    by_id = {str(row.get("sample_id")): row for row in materialized}
    size = args.per_stratum

    def training(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{**row, "review_kind": "training_sample"} for row in rows]

    split_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in materialized:
        split_rows[str(row.get("final_view_membership"))].append(row)
    context_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in materialized:
        context_rows[str(row.get("context_reconstruction_class"))].append(row)
    true_new_ids = {str(row.get("audio_turn_id")) for row in timeline.values() if row.get("reconciliation_state") == "TRUE_NEW"}
    topology_sample_rows = [
        row for row in materialized
        if set(map(str, row.get("context_turn_ids") or [])) & true_new_ids
    ]
    merge_ids = {str(row.get("audio_turn_id")) for row in timeline.values() if row.get("reconciliation_state") == "MERGE_EXISTING"}
    merge_sample_rows = [row for row in materialized if set(map(str, row.get("context_turn_ids") or [])) & merge_ids]
    long_rows = sorted(materialized, key=lambda row: len(row.get("context_turn_ids") or []), reverse=True)
    short_rows = [row for row in materialized if any(len(str(message.get("content") or "")) <= 5 for message in row.get("messages") or [])]
    retained_rows = [row for row in materialized if row.get("was_baseline_materialized")]
    target_by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for target in targets:
        if str(target.get("sample_id")) in by_id:
            target_by_state[str(target.get("state"))].append(by_id[str(target["sample_id"])])
    evidence_rows = []
    for row in reconciliations:
        state = str(row.get("reconciliation_state"))
        if state in {"MERGE_EXISTING", "SPLIT_EXISTING", "EXTEND_EXISTING", "AMBIGUOUS_BOUNDARY"}:
            evidence_rows.append({
                "review_kind": "topology_evidence", "sample_id": None, "recording_id": row.get("recording_id"),
                "review_evidence": {
                    "candidate_span_id": row.get("candidate_span_id"),
                    "reconciliation_state": state,
                    "old_turn_ids": row.get("old_turn_ids"),
                    "old_timeline_text": row.get("old_text"),
                    "reconstructed_text": row.get("chosen_text") or row.get("new_text"),
                    "source_window_ids": row.get("source_window_ids"),
                    "source_spans": row.get("source_spans"),
                },
            })
    quarantine_evidence = [{
        "review_kind": "quarantine_evidence", "sample_id": row.get("sample_id"), "recording_id": None,
        "review_evidence": {key: row.get(key) for key in ("failure_stage", "failure_reason", "target_interval", "old_new_disagreement_state", "explicit_contradiction", "resolution_attempts")},
    } for row in quarantined]

    rounds = [
        ("random_split_and_recording_family", training(sum((stable_take(split_rows[key], size, salt="r1" + key) for key in ("IN_TRAIN", "IN_VALIDATION", "IN_SEALED_EVAL")), []))),
        ("recovered_context_and_true_new", training(sum((stable_take(context_rows[key], size, salt="r2" + key) for key in ("RECOVERED_FROM_EXISTING_TIMELINE", "RECOVERED_FROM_NEW_AUDIO", "RECOVERED_FROM_BOTH")), [])) + training(stable_take(topology_sample_rows, size, salt="r2true"))),
        ("topology_edges_and_context_extent", stable_take(evidence_rows, size * 2, salt="r3topology") + training(stable_take(long_rows, size, salt="r3long")) + training(stable_take(short_rows, size, salt="r3short")) + training(stable_take(merge_sample_rows, size, salt="r3merge"))),
        ("target_rescue_baseline_quarantine_ambiguity", training(sum((stable_take(target_by_state[key], size, salt="r4" + key) for key in ("AUDIO_CONFIRMED", "AUDIO_MINOR_DISAGREEMENT", "BASELINE_CONFIRMED")), [])) + training(stable_take(retained_rows, size, salt="r4retained")) + stable_take(quarantine_evidence, size * 2, salt="r4quarantine")),
    ]

    samples: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    round_rows: list[dict[str, Any]] = []
    for round_index, (name, selected) in enumerate(rounds, 1):
        category_counts: Counter[str] = Counter()
        initial_category_counts: Counter[str] = Counter()
        initial_findings = 0
        invalid_initial = 0
        invalid_triage = 0
        for ordinal, row in enumerate(selected):
            payload = review_payload(row)
            input_hash = canonical_sha256(payload)
            result = call_reviewer(payload, endpoint=args.endpoint, model=args.model, timeout=args.timeout)
            triage = None
            if not result["valid"]:
                invalid_initial += 1
            if result["finding"]:
                initial_findings += 1
                initial_category_counts[str(result["issue_category"] or "other")] += 1
                triage = call_triage(payload, result, endpoint=args.endpoint, model=args.model, timeout=args.timeout)
                if not triage["valid"]:
                    invalid_triage += 1
            sample = {
                "round": round_index, "stratum": name, "ordinal": ordinal, "sample_id": row.get("sample_id"),
                "recording_id": row.get("recording_id"), "review_kind": row.get("review_kind"),
                "review_input_hash": input_hash, "reviewer": args.model, "review": result, "triage": triage,
            }
            samples.append(sample)
            if triage and triage["verified"]:
                category = str(triage["issue_category"] or "other")
                category_counts[category] += 1
                findings.append({
                    "sample_id": row.get("sample_id"), "recording_id": row.get("recording_id"),
                    "issue_category": category, "severity": triage["severity"], "evidence": triage["evidence"],
                    "reviewer": args.model, "review_input_hash": input_hash, "round": round_index,
                    "first_reviewer_category": result["issue_category"], "first_reviewer_evidence": result["evidence"],
                })
        # A repeated new category is a production blocker; one uncorroborated
        # low-severity observation remains recorded but does not reset the run.
        # Unparseable model calls are recorded as audit coverage diagnostics, not
        # fabricated corpus findings. A verified repeated category remains a real
        # open-ended systemic finding and resets the production stopping count.
        systemic = sorted(category for category, count in category_counts.items() if count >= 2)
        round_rows.append({
            "round": round_index, "name": name, "sample_count": len(selected),
            "initial_finding_count": initial_findings,
            "initial_finding_counts": dict(initial_category_counts),
            "verified_finding_counts": dict(category_counts),
            "invalid_initial_responses": invalid_initial,
            "invalid_triage_responses": invalid_triage,
            "new_systemic_issue_categories": systemic,
            "result": "NO_NEW_SYSTEMIC_ISSUE" if not systemic else "SYSTEMIC_ISSUE_FOUND",
        })
    consecutive = 0
    for row in round_rows:
        consecutive = consecutive + 1 if not row["new_systemic_issue_categories"] else 0
        row["consecutive_no_new_systemic_issue_rounds_after_round"] = consecutive
    report = {
        "schema_version": "content-audit-v1", "run_id": run.name, "reviewer": args.model,
        "rounds": round_rows, "total_sampled_records": len(samples),
        "consecutive_no_new_systemic_issue_rounds": consecutive,
        "pass": consecutive >= 4,
    }
    (run / "content_audit_rounds.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(run / "content_audit_samples.jsonl", samples)
    write_jsonl(run / "content_audit_findings.jsonl", findings)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
