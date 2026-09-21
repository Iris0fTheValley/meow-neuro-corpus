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
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audio_evidence.recovery_v2 import (  # noqa: E402
    canonical_sha256,
    is_non_conversational_sentinel,
    normalized_text,
    parse_json_object,
)


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
Review only the exact visible messages shown. Do not assume hidden conversation,
metadata, expected labels, acoustics, speaker identities, or a desired answer.
The ordered message list has one and only one target: the final message marked
is_target=true. Every earlier message is context, not a separate candidate
response to audit. Never report a defect about an earlier context message merely
because it does not answer an even earlier message. This rule is especially
important when context contains assistant-to-assistant turns.

Judge whether the final target has a plausible, grounded connection to the
visible context. A valid target may be elliptical, self-referential, a game-state
update, a continuation of an implied shared topic, or a response to an earlier
relevant turn rather than the last message. Do not require token overlap, a direct
question/answer form, an explicit acknowledgement, or a response to every old
message. Do flag a semantic disconnect only when the final target itself is a
clear non sequitur or visibly depends on an absent immediate trigger/antecedent;
do not flag a surprising name, awkward wording, or an unstated real-world fact
by itself.

Also look for a concrete, observable defect: (1) an exact/near-exact duplicate
utterance, (2) the final target copied into earlier prompt text, (3) visibly
swallowed or wrongly split speakers, (4) role text that is visibly contradictory,
(5) an objectively redundant prefix where a visible suffix alone clearly carries
the same exchange, (6) control/noise text, or another concrete issue. Do not flag
unusual dialogue, a short valid answer, multiple history turns, a supervision
convention, or an expected fail-closed quarantine.
For a finding, cite exact visible text and explain the defect without relying on
unstated facts. Return JSON only:
{"finding": boolean, "issue_category": "semantic_disconnect"|"duplicate_speech"|"target_leakage"|"speaker_topology"|"role_assignment"|"context_too_wide"|"context_too_thin"|"asr_noise"|"other"|null, "severity":"none"|"low"|"medium"|"high", "evidence": string}.
Use finding=false when the evidence does not prove a concrete issue."""

TRIAGE_SYSTEM = """You are a strict second-pass production audit adjudicator.
You receive only a sample/evidence record and a first reviewer claim. The final
message marked is_target=true is the only target; earlier messages are context
and must not be judged as if they were replies to earlier context. Verify the
claim from visible text alone. Reject it if it relies on unshown speaker
identity, hidden history, expected labels, the supervision convention, a target
merely not answering every old message, or an expected quarantine. A semantic
disconnect must be about the final target itself and require a clear non sequitur
or an absent immediate trigger, not merely an elliptical or indirect reply.
Target leakage requires the target text to appear in earlier prompt text.
Duplicate speech requires the same/near-identical visible utterance.
Context-too-wide requires an explicitly shown shorter suffix that independently
carries the interaction; do not infer it only because old turns exist. Return JSON only:
{"verified": boolean, "issue_category": string|null, "severity":"none"|"low"|"medium"|"high", "evidence": string}.
Use verified=false unless the claim is concretely supported."""


def review_payload(row: dict[str, Any]) -> dict[str, Any]:
    # The visible audit view intentionally excludes semantic verdicts, identity
    # diagnostics, future timeline, and any evaluator expectation.
    if row.get("review_kind") == "training_sample":
        messages = [
            {"role": message.get("role"), "text": message.get("content")}
            for message in (row.get("messages") or [])
        ]
        if messages:
            messages[-1]["is_target"] = True
        return {
            "kind": "training_sample",
            "sample_id": row.get("sample_id"),
            "recording_id": row.get("recording_id"),
            "messages": messages,
        }
    return {
        "kind": row.get("review_kind"),
        "sample_id": row.get("sample_id"),
        "recording_id": row.get("recording_id"),
        "evidence": row.get("review_evidence"),
    }


def call_model(
    system: str,
    user_payload: dict[str, Any],
    *,
    endpoint: str,
    model: str,
    timeout: int,
    api_key_env: str = "STEPFUN_API_KEY",
) -> tuple[dict[str, Any], bool, str]:
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError, URLError

    is_chat_endpoint = "/chat/completions" in endpoint or "/step_plan/" in endpoint
    if is_chat_endpoint:
        api_key = os.environ.get(api_key_env, "").strip()
        if not api_key:
            return {}, False, "missing_api_key"
        request_body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            "temperature": 0,
            "max_tokens": 32768 if "/step_plan/" in endpoint else 256,
            "response_format": {"type": "json_object"},
        }
        if "/step_plan/" in endpoint:
            request_body["reasoning_effort"] = "low"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json; charset=utf-8",
        }
    else:
        prompt = (
            "<|im_start|>system\n" + system + "<|im_end|>\n<|im_start|>user\n"
            + json.dumps(user_payload, ensure_ascii=False) + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
        request_body = {"model": model, "prompt": prompt, "temperature": 0, "max_tokens": 256, "stop": ["<|im_end|>"]}
        headers = {"Content-Type": "application/json; charset=utf-8"}
    request = Request(
        endpoint,
        data=json.dumps(request_body, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
        choice = (response_payload.get("choices") or [{}])[0]
        parsed = parse_json_object((choice.get("message") or {}).get("content") or choice.get("text")) or {}
    except (HTTPError, URLError, OSError, TimeoutError, json.JSONDecodeError) as exc:
        return {}, False, type(exc).__name__
    return parsed, bool(parsed), ""


def call_reviewer(payload: dict[str, Any], *, endpoint: str, model: str, timeout: int, api_key_env: str = "STEPFUN_API_KEY") -> dict[str, Any]:
    parsed, parsed_ok, error = call_model(SYSTEM, payload, endpoint=endpoint, model=model, timeout=timeout, api_key_env=api_key_env)
    if not parsed_ok:
        parsed, parsed_ok, error = call_model(SYSTEM, payload, endpoint=endpoint, model=model, timeout=timeout, api_key_env=api_key_env)
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


def call_triage(payload: dict[str, Any], claim: dict[str, Any], *, endpoint: str, model: str, timeout: int, api_key_env: str = "STEPFUN_API_KEY") -> dict[str, Any]:
    parsed, parsed_ok, error = call_model(
        TRIAGE_SYSTEM,
        {"sample_or_evidence": payload, "first_reviewer_claim": claim},
        endpoint=endpoint, model=model, timeout=timeout, api_key_env=api_key_env,
    )
    if not parsed_ok:
        parsed, parsed_ok, error = call_model(
            TRIAGE_SYSTEM,
            {"sample_or_evidence": payload, "first_reviewer_claim": claim},
            endpoint=endpoint, model=model, timeout=timeout, api_key_env=api_key_env,
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


def corroborate_finding(
    payload: dict[str, Any],
    claim: dict[str, Any],
    triage: dict[str, Any] | None,
    *,
    context_state: str | None,
    source_row: dict[str, Any],
    turn_intervals: dict[str, tuple[float, float]],
) -> dict[str, Any] | None:
    """Confirm an open-ended reviewer hypothesis from the visible artifact.

    The reviewer is deliberately free to name a new concern, but a corpus-level
    issue cannot be manufactured by an ungrounded model assertion.  Semantic
    hypotheses are corroborated only when the selected-context sufficiency
    artifact itself disagrees; lexical claims require literal visible evidence.
    This is an evidence gate, not a lexical semantic-rejection policy.
    """
    if not claim.get("finding") or not triage or not triage.get("verified"):
        return None
    category = str(triage.get("issue_category") or claim.get("issue_category") or "other").strip().lower().replace("-", "_").replace(" ", "_")
    evidence = str(triage.get("evidence") or claim.get("evidence") or "")
    messages = payload.get("messages") if payload.get("kind") == "training_sample" else None
    if messages:
        texts = [str(message.get("text") or "") for message in messages]
        normalized = [normalized_text(text) for text in texts]
        target = normalized[-1] if normalized else ""
        history = normalized[:-1]
        if category == "target_leakage":
            if target and len(target) >= 4 and any(target == prior for prior in history):
                return {"category": category, "evidence": "Final target text exactly occurs in earlier prompt text."}
            return None
        if category == "duplicate_speech":
            # Natural spoken repeats are retained.  A duplicate requires the
            # same normalized text *and* overlapping source intervals (or a
            # reused source turn), not merely repeated words in adjacent turns.
            visible_messages = source_row.get("messages") or []
            for right_index, text in enumerate(history):
                if len(text) < 8:
                    continue
                for left_index, prior in enumerate(history[:right_index]):
                    if prior != text:
                        continue
                    left_ids = set(map(str, visible_messages[left_index].get("source_turn_ids") or []))
                    right_ids = set(map(str, visible_messages[right_index].get("source_turn_ids") or []))
                    if left_ids & right_ids:
                        return {"category": category, "evidence": f"Repeated prompt text reuses source turn(s): {sorted(left_ids & right_ids)}."}
                    for left_id in left_ids:
                        for right_id in right_ids:
                            left_span, right_span = turn_intervals.get(left_id), turn_intervals.get(right_id)
                            if left_span and right_span and min(left_span[1], right_span[1]) > max(left_span[0], right_span[0]):
                                return {"category": category, "evidence": f"Repeated prompt text has overlapping source intervals: {left_id}, {right_id}."}
            return None
        if category in {"asr_noise", "asr_control_noise", "control_noise"}:
            sentinels = [text for text in texts if is_non_conversational_sentinel(text)]
            if sentinels:
                return {"category": "asr_noise", "evidence": f"Visible non-conversational sentinel: {sentinels[0]!r}"}
            return None
        if category in {"semantic_disconnect", "context_too_thin", "context_too_wide"}:
            # Production context-judge verdicts are intentionally unavailable
            # to this reviewer.  The independent reviewer/triage pair must
            # ground the claim in the final visible messages alone.
            if evidence.strip():
                return {"category": category, "evidence": evidence[:1000]}
            return None
        # A role/topology claim on a final sample requires an actual topology
        # disagreement artifact; role text alone cannot establish identity.
        return None
    if payload.get("kind") == "topology_evidence":
        evidence_row = payload.get("evidence") or {}
        if category in {"speaker_topology", "role_assignment"} and evidence_row.get("reconciliation_state") == "AMBIGUOUS_BOUNDARY":
            return {"category": category, "evidence": "Visible topology evidence is explicitly AMBIGUOUS_BOUNDARY."}
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:1234/v1/completions")
    parser.add_argument("--api-key-env", default="STEPFUN_API_KEY")
    parser.add_argument("--per-stratum", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--reuse-existing-reviews", action="store_true", help="Reuse matching review inputs after tightening corroboration only.")
    args = parser.parse_args()
    run = Path(args.run)
    materialized = load_jsonl(run / "materialized.jsonl")
    timeline = {str(row.get("audio_turn_id")): row for row in load_jsonl(run / "canonical_audio_evidence_timeline.jsonl")}
    turn_intervals = {
        turn_id: (float(row.get("start") or 0.0), float(row.get("end") or 0.0))
        for turn_id, row in timeline.items()
        if row.get("start") is not None and row.get("end") is not None
    }
    targets = load_jsonl(run / "target_resolution.jsonl")
    prior_reviews = {}
    if args.reuse_existing_reviews:
        for row in load_jsonl(run / "content_audit_samples.jsonl"):
            if row.get("review_input_hash"):
                prior_reviews[str(row["review_input_hash"])] = row
    by_id = {str(row.get("sample_id")): row for row in materialized}
    size = args.per_stratum

    def training(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{**row, "review_kind": "training_sample"} for row in rows]

    split_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in materialized:
        split_rows[str(row.get("final_view_membership"))].append(row)
    long_rows = sorted(materialized, key=lambda row: len(row.get("context_turn_ids") or []), reverse=True)
    retained_rows = [row for row in materialized if row.get("was_baseline_materialized")]
    newly_materialized_rows = [row for row in materialized if not row.get("was_baseline_materialized")]
    true_new_ids = {str(row.get("audio_turn_id")) for row in timeline.values() if row.get("reconciliation_state") == "TRUE_NEW"}
    true_new_rows = [
        row for row in materialized
        if set(map(str, row.get("context_turn_ids") or [])) & true_new_ids
    ]
    assistant_to_assistant_rows = [
        row for row in materialized
        if len(row.get("messages") or []) >= 2
        and row["messages"][-2].get("role") == "assistant"
    ]
    target_by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for target in targets:
        if str(target.get("sample_id")) in by_id:
            target_by_state[str(target.get("state"))].append(by_id[str(target["sample_id"])])
    rounds = [
        # Every round has independent random samples from all final splits;
        # additional strata target the known reconstruction failure surfaces.
        ("random_splits_true_new_long_assistant_to_assistant_baseline", training(
            stable_take(split_rows["IN_TRAIN"], size, salt="r1train")
            + stable_take(split_rows["IN_VALIDATION"], size, salt="r1validation")
            + stable_take(split_rows["IN_SEALED_EVAL"], size, salt="r1sealed")
            + stable_take(true_new_rows, size, salt="r1true_new")
            + stable_take(long_rows, size, salt="r1long")
            + stable_take(assistant_to_assistant_rows, size, salt="r1assistant_assistant")
            + stable_take(retained_rows, size, salt="r1baseline_retained")
            + stable_take(newly_materialized_rows, size, salt="r1newly_materialized")
        )),
        ("random_splits_true_new_long_assistant_to_assistant_baseline", training(
            stable_take(split_rows["IN_TRAIN"], size, salt="r2train")
            + stable_take(split_rows["IN_VALIDATION"], size, salt="r2validation")
            + stable_take(split_rows["IN_SEALED_EVAL"], size, salt="r2sealed")
            + stable_take(true_new_rows, size, salt="r2true_new")
            + stable_take(long_rows, size, salt="r2long")
            + stable_take(assistant_to_assistant_rows, size, salt="r2assistant_assistant")
            + stable_take(retained_rows, size, salt="r2baseline_retained")
            + stable_take(newly_materialized_rows, size, salt="r2newly_materialized")
        )),
        ("random_splits_true_new_long_assistant_to_assistant_baseline", training(
            stable_take(split_rows["IN_TRAIN"], size, salt="r3train")
            + stable_take(split_rows["IN_VALIDATION"], size, salt="r3validation")
            + stable_take(split_rows["IN_SEALED_EVAL"], size, salt="r3sealed")
            + stable_take(true_new_rows, size, salt="r3true_new")
            + stable_take(long_rows, size, salt="r3long")
            + stable_take(assistant_to_assistant_rows, size, salt="r3assistant_assistant")
            + stable_take(retained_rows, size, salt="r3baseline_retained")
            + stable_take(newly_materialized_rows, size, salt="r3newly_materialized")
        )),
        ("random_splits_true_new_long_assistant_to_assistant_baseline", training(
            stable_take(split_rows["IN_TRAIN"], size, salt="r4train")
            + stable_take(split_rows["IN_VALIDATION"], size, salt="r4validation")
            + stable_take(split_rows["IN_SEALED_EVAL"], size, salt="r4sealed")
            + stable_take(true_new_rows, size, salt="r4true_new")
            + stable_take(long_rows, size, salt="r4long")
            + stable_take(assistant_to_assistant_rows, size, salt="r4assistant_assistant")
            + stable_take(retained_rows, size, salt="r4baseline_retained")
            + stable_take(newly_materialized_rows, size, salt="r4newly_materialized")
        )),
    ]

    samples: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    hypotheses: list[dict[str, Any]] = []
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
            prior = prior_reviews.get(input_hash)
            result = dict(prior.get("review") or {}) if prior else call_reviewer(
                payload, endpoint=args.endpoint, model=args.model, timeout=args.timeout, api_key_env=args.api_key_env,
            )
            triage = dict(prior.get("triage") or {}) if prior and prior.get("triage") else None
            if not result["valid"]:
                invalid_initial += 1
            if result["finding"]:
                initial_findings += 1
                initial_category_counts[str(result["issue_category"] or "other")] += 1
                triage = triage or call_triage(
                    payload, result, endpoint=args.endpoint, model=args.model, timeout=args.timeout, api_key_env=args.api_key_env,
                )
                if not triage["valid"]:
                    invalid_triage += 1
            sample = {
                "round": round_index, "stratum": name, "ordinal": ordinal, "sample_id": row.get("sample_id"),
                "recording_id": row.get("recording_id"), "review_kind": row.get("review_kind"),
                "review_input_hash": input_hash, "reviewer": args.model, "review": result, "triage": triage,
            }
            samples.append(sample)
            if result["finding"]:
                hypotheses.append({
                    "sample_id": row.get("sample_id"), "recording_id": row.get("recording_id"),
                    "initial_category": result.get("issue_category"), "initial_evidence": result.get("evidence"),
                    "triage": triage, "reviewer": args.model, "review_input_hash": input_hash, "round": round_index,
                })
            confirmed = corroborate_finding(
                payload, result, triage,
                context_state=None,
                source_row=row,
                turn_intervals=turn_intervals,
            )
            if confirmed:
                category = str(confirmed["category"])
                category_counts[category] += 1
                findings.append({
                    "sample_id": row.get("sample_id"), "recording_id": row.get("recording_id"),
                    "issue_category": category, "severity": triage["severity"], "evidence": confirmed["evidence"],
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
    write_jsonl(run / "content_audit_hypotheses.jsonl", hypotheses)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
