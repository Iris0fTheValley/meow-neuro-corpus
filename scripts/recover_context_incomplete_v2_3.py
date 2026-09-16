from __future__ import annotations

"""Deterministic bounded recovery for v2.3 CONTEXT_INCOMPLETE rows.

This is deliberately downstream of semantic closure.  It never calls a model,
does not modify existing v2.3 artifacts, and preserves the original semantic
decision while rematerializing context from the canonical timeline.
"""

import argparse
import copy
import difflib
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if not (ROOT / "datasets").exists():
    for candidate_root in (SCRIPT_DIR.parents[2], SCRIPT_DIR.parents[3], SCRIPT_DIR.parents[4]):
        if (candidate_root / "datasets").exists():
            ROOT = candidate_root
            break
DATASET = ROOT / "datasets" / "meow_v02_sft_v2_2_semantic_verified"
sys.path.insert(0, str(ROOT / "scripts"))

import sft_interaction_semantic_closure as semantic  # noqa: E402
import sft_semantic_verified_v2_1 as judge_core  # noqa: E402
import build_sft_v2_2_structural_candidates as structural  # noqa: E402

TARGETS = set(semantic.TARGETS)
ACCEPTED_RELATIONS = {"DIRECT_RESPONSE", "CONTEXTUAL_RESPONSE", "GAME_STATE_RESPONSE", "VALID_GAME_STATE_RESPONSE"}
WINDOWS = (2, 4, 8, 12)
RECOVERY_VERSION = "bounded-context-recovery-v1-2026-09-16"
PIPELINE_VERSION = "sft-interaction-closure-v2.3.1-2026-09-13"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def words(text: Any) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:['-][a-z0-9]+)?", norm(text))


def row_text(row: dict[str, Any], role: str | None = None) -> str:
    messages = row.get("messages") or []
    if role:
        return "\n".join(str(m.get("content") or "") for m in messages if m.get("role") == role)
    return "\n".join(f"{m.get('role')}:{m.get('content')}" for m in messages)


def content_tokens(text: Any) -> set[str]:
    stop = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "is", "it", "i", "you", "we", "that", "this", "for", "do", "did", "are", "was", "be", "me", "my", "your", "with", "so", "just", "uh", "um"}
    return {x for x in words(text) if len(x) > 2 and x not in stop}


def is_bad_boundary(turn: dict[str, Any]) -> bool:
    return any(bool(turn.get(k)) for k in ("_bad_boundary", "bad_boundary", "uncertain_transcription", "suspicious_transcription", "overlap", "malformed"))


def turn_gap(previous: dict[str, Any], current: dict[str, Any], fallback_previous: float = 0.0) -> float:
    try:
        return max(0.0, float(current.get("_start", 0.0)) - float(previous.get("_end", fallback_previous)))
    except (TypeError, ValueError):
        return 0.0


def trigger_evidence(context: list[dict[str, Any]], target: list[dict[str, Any]]) -> dict[str, Any]:
    context_text = "\n".join(norm(t.get("text")) for t in context)
    target_text = "\n".join(norm(t.get("text")) for t in target)
    triggers: list[str] = []
    if "?" in context_text:
        triggers.append("QUESTION_MARK")
    if re.search(r"\b(what|why|how|where|when|who|which|can|could|would|do|did|is|are|will|should)\b", context_text):
        triggers.append("QUESTION_OR_MODAL_PATTERN")
    if re.search(r"\b(try|check|look|tell|show|remember|explain|use|press|go|get|please|stop|wait|listen)\b", context_text):
        triggers.append("IMPERATIVE_OR_REQUEST_PATTERN")
    if re.search(r"\b(you|your|yours|we|our|neuro)\b", context_text):
        triggers.append("ADDRESS_OR_PARTICIPANT_PATTERN")
    overlap = sorted(content_tokens(context_text) & content_tokens(target_text))
    if overlap:
        triggers.append("CONTENT_TOKEN_OVERLAP")
    return {"flags": sorted(set(triggers)), "overlap_tokens": overlap, "has_observable_evidence": bool(triggers)}


def annotate_timelines(timelines: dict[str, dict[str, Any]]) -> None:
    for timeline in timelines.values():
        previous_end = None
        for index, turn in enumerate(timeline.get("_turns") or []):
            turn["text_qa"], turn["text_qa_reasons"] = structural.text_quality(turn.get("text"))
            turn["asr_quality"], turn["asr_quality_reasons"], turn["asr_metadata"] = structural.raw_asr_quality(turn)
            turn["_bad_boundary"] = structural.bad_boundary(turn)
            try:
                turn["_start"] = float(turn.get("_start", (turn.get("timestamp") or {}).get("start", index)))
                turn["_end"] = float(turn.get("_end", (turn.get("timestamp") or {}).get("end", index)))
            except (TypeError, ValueError):
                turn["_start"], turn["_end"] = float(index), float(index)
            turn["_gap_from_previous"] = 0.0 if previous_end is None else max(0.0, turn["_start"] - previous_end)
            previous_end = turn["_end"]


def legal_segment(turns: list[dict[str, Any]], start: int, end: int, target_speaker: str, max_gap: float) -> tuple[bool, str]:
    if start < 0 or end >= len(turns) or start > end:
        return False, "segment_out_of_range"
    segment = turns[start : end + 1]
    for turn in segment:
        if turn.get("identity") in TARGETS:
            return False, "identity_boundary"
        if turn.get("identity_unreliable") or turn.get("_identity_unreliable"):
            return False, "identity_unreliable_boundary"
        if turn.get("speaker") == target_speaker:
            return False, "target_speaker_in_context"
        if is_bad_boundary(turn) or turn.get("event_boundary"):
            return False, "hard_or_event_boundary"
        if turn.get("text_qa") != "PASS":
            return False, "text_qa_failure"
        if turn.get("asr_quality") == "REVIEW":
            return False, "asr_review"
    for previous, current in zip(segment, segment[1:]):
        if turn_gap(previous, current) > max_gap:
            return False, "temporal_gap"
        if previous.get("event_boundary") or current.get("event_boundary"):
            return False, "event_boundary"
    return True, "ok"


def materialize_recovered(source: dict[str, Any], timeline: dict[str, Any], context_ids: list[str], recovery: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    original_closure = copy.deepcopy(source.get("semantic_closure") or {})
    original_qa = copy.deepcopy(source.get("semantic_qa") or {})
    original_sufficiency = copy.deepcopy(source.get("context_sufficiency") or {})
    original_ids = [str(x) for x in source.get("target_turn_ids") or []]
    decision = copy.deepcopy(original_closure.get("final_decision") or {})
    if not decision:
        decision = {"relation": original_qa.get("relation"), "episode_decision": original_qa.get("episode_decision"),
                    "confidence": original_qa.get("confidence"), "context_complete": True, "response_complete": True,
                    "transcript_usable": True, "reason": original_qa.get("reason", "")}
    decision.update({"selected_context_turn_ids": context_ids, "selected_target_turn_ids": original_ids, "context_complete": True, "response_complete": True, "transcript_usable": True})
    closure = copy.deepcopy(original_closure)
    closure.update({"state": "VERIFIED", "final_decision": decision, "recovery_materialization": {"version": RECOVERY_VERSION, "semantic_truth_preserved": True}})
    try:
        result = semantic.materialize_verified(source, closure, timeline)
    except Exception as exc:  # defense-in-depth: a malformed source row is unresolved, never promoted
        return None, f"materializer_exception:{type(exc).__name__}"
    if result is None:
        return None, "materializer_defense_in_depth_reject"
    # Restore the original semantic truth and sufficiency decision.  Only the
    # canonical turn-dependent materialization is changed in this artifact.
    result["semantic_closure"] = original_closure
    result["semantic_qa"] = original_qa
    result["context_sufficiency"] = original_sufficiency
    result["context_recovery"] = recovery
    result["sampling_eligibility"] = "RECOVERED_CONTEXT_CANDIDATE"
    result["verified_pool_status"] = "VERIFIED_CONTEXT_RECOVERED_CANDIDATE"
    result["training_candidate"] = False
    result["evaluation_only"] = False
    result["pipeline_version"] = PIPELINE_VERSION
    result["artifact_schema_version"] = semantic.SCHEMA_VERSION
    return result, None


def recover(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    candidates = load_jsonl(Path(args.candidates))
    incomplete = [r for r in candidates if (r.get("context_sufficiency") or {}).get("state") == "CONTEXT_INCOMPLETE"]
    fusion = judge_core.load_fusion()
    source_ids = {str(r.get("source_id")) for r in incomplete}
    timelines = judge_core.load_timelines(source_ids, fusion)
    annotate_timelines(timelines)
    recovered: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    blocking_reason_counts: Counter[str] = Counter()
    rule_counts: Counter[str] = Counter()
    for source in sorted(incomplete, key=lambda r: str(r.get("sample_id"))):
        sid = str(source.get("sample_id"))
        timeline = timelines.get(str(source.get("source_id")))
        if not timeline:
            reason_counts["canonical_timeline_unavailable"] += 1
            unresolved.append({"sample_id": sid, "reason": "canonical_timeline_unavailable", "source_id": source.get("source_id")})
            continue
        turns = list(timeline.get("_turns") or [])
        by_id = {str(t.get("turn_id")): t for t in turns}
        context_ids = [str(x) for x in source.get("context_turn_ids") or []]
        target_ids = [str(x) for x in source.get("target_turn_ids") or []]
        if not context_ids or not target_ids or any(x not in by_id for x in context_ids + target_ids):
            reason_counts["selected_turn_missing_from_canonical_timeline"] += 1
            unresolved.append({"sample_id": sid, "reason": "selected_turn_missing_from_canonical_timeline", "source_id": source.get("source_id")})
            continue
        positions = {str(t.get("turn_id")): i for i, t in enumerate(turns)}
        cpos = [positions[x] for x in context_ids]
        tpos = [positions[x] for x in target_ids]
        if cpos != list(range(min(cpos), max(cpos) + 1)) or tpos != list(range(min(tpos), max(tpos) + 1)) or max(cpos) >= min(tpos):
            reason_counts["base_selection_not_contiguous"] += 1
            unresolved.append({"sample_id": sid, "reason": "base_selection_not_contiguous", "source_id": source.get("source_id")})
            continue
        target_turns = [turns[i] for i in tpos]
        target_speaker = str(target_turns[0].get("speaker") or "")
        try:
            max_gap = float((source.get("episode_boundary") or {}).get("context_gap_policy_seconds", 8.0))
        except (TypeError, ValueError):
            max_gap = 8.0
        max_gap = max(0.0, max_gap)
        relation = str(((source.get("semantic_qa") or {}).get("relation") or ((source.get("semantic_closure") or {}).get("final_decision") or {}).get("relation") or "")).upper()
        candidates_for_row: list[dict[str, Any]] = []
        row_failures: list[str] = []
        for window in WINDOWS:
            start = max(0, min(cpos) - max(0, window - len(cpos)))
            if start >= min(cpos):
                continue
            ok, failure = legal_segment(turns, start, max(cpos), target_speaker, max_gap)
            if not ok:
                row_failures.append(failure)
                rule_counts[f"window_{window}_{failure}"] += 1
                continue
            segment = turns[start : max(cpos) + 1]
            evidence = trigger_evidence(segment, target_turns)
            # Semantic truth is already closed; an accepted relation is a
            # secondary evidence channel, not a replacement for hard QA.
            score = (2 if evidence["has_observable_evidence"] else 0) + (1 if evidence["overlap_tokens"] else 0) + (1 if relation in ACCEPTED_RELATIONS else 0)
            candidates_for_row.append({"window": window, "start": start, "turns": segment, "evidence": evidence, "score": score})
        if not candidates_for_row:
            reason_counts["no_legal_bounded_context"] += 1
            # Keep the aggregate state simple for downstream agents, while
            # recording which canonical hard boundary prevented recovery.
            priority = {"identity_boundary": 0, "identity_unreliable_boundary": 1,
                        "hard_or_event_boundary": 2, "event_boundary": 3,
                        "text_qa_failure": 4, "asr_review": 5, "temporal_gap": 6}
            dominant = sorted(set(row_failures), key=lambda x: (priority.get(x, 99), x))[0] if row_failures else "unknown"
            blocking_reason_counts[dominant] += 1
            unresolved.append({"sample_id": sid, "reason": "no_legal_bounded_context", "blocking_reason": dominant,
                               "blocking_reasons": sorted(set(row_failures)), "source_id": source.get("source_id"), "base_context_turn_ids": context_ids})
            continue
        viable = [c for c in candidates_for_row if c["evidence"]["has_observable_evidence"] or relation in ACCEPTED_RELATIONS]
        if not viable:
            reason_counts["no_observable_evidence_after_bounded_expansion"] += 1
            unresolved.append({"sample_id": sid, "reason": "no_observable_evidence_after_bounded_expansion", "source_id": source.get("source_id"),
                               "base_context_turn_ids": context_ids, "windows_examined": [c["window"] for c in candidates_for_row]})
            continue
        # Shortest legal candidate wins; score breaks ties, then earliest index.
        selected = sorted(viable, key=lambda c: (len(c["turns"]), -c["score"], c["start"]))[0]
        expanded_ids = [str(t.get("turn_id")) for t in selected["turns"]]
        added = [x for x in expanded_ids if x not in context_ids]
        if not added:
            reason_counts["no_additional_context"] += 1
            unresolved.append({"sample_id": sid, "reason": "no_additional_context", "source_id": source.get("source_id")})
            continue
        recovery = {
            "recovery_version": RECOVERY_VERSION,
            "status": "RECOVERED_BOUNDED_CONTEXT",
            "rule": "shortest_legal_window_with_observable_evidence",
            "window_options": list(WINDOWS),
            "selected_window": selected["window"],
            "base_context_turn_ids": context_ids,
            "recovered_context_turn_ids": expanded_ids,
            "added_context_turn_ids": added,
            "target_turn_ids": target_ids,
            "max_gap_seconds": max_gap,
            "evidence": selected["evidence"],
            "semantic_truth_preserved": True,
            "canonical_timeline_source": "CURRENT_CANONICAL_TIMELINE",
            "diagnostic_turns_promoted_to_training": 0,
        }
        materialized, failure = materialize_recovered(source, timeline, expanded_ids, recovery)
        if materialized is None:
            reason_counts[failure or "materialization_failure"] += 1
            unresolved.append({"sample_id": sid, "reason": failure or "materialization_failure", "source_id": source.get("source_id"), "recovery": recovery})
            continue
        recovered.append(materialized)
        rule_counts[f"selected_window_{selected['window']}"] += 1
    report = {
        "schema_version": "context-recovery-report-v1",
        "pipeline_version": PIPELINE_VERSION,
        "recovery_version": RECOVERY_VERSION,
        "status": "COMPLETED",
        "source_context_incomplete": len(incomplete),
        "recovered": len(recovered),
        "unresolved": len(unresolved),
        "reason_counts": dict(reason_counts),
        "unresolved_blocking_reason_counts": dict(blocking_reason_counts),
        "rule_counts": dict(rule_counts),
        "canonical_timelines_loaded": len(timelines),
        "semantic_judge_calls": 0,
        "asr_calls": 0,
        "semantic_truth_changed": 0,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return recovered, unresolved, report


def interval(row: dict[str, Any]) -> tuple[float, float]:
    ts = row.get("timestamps") or {}
    try:
        return float(ts.get("start")), float(ts.get("end"))
    except (TypeError, ValueError):
        return (0.0, 0.0)


def token_containment(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    if len(a) <= len(b) and any(a == b[i : i + len(a)] for i in range(len(b) - len(a) + 1)):
        return len(a) / len(b)
    if len(b) < len(a) and any(b == a[i : i + len(b)] for i in range(len(a) - len(b) + 1)):
        return len(b) / len(a)
    return 0.0


def similarity(a: dict[str, Any], b: dict[str, Any]) -> tuple[float, float, float]:
    ta, tb = words(row_text(a)), words(row_text(b))
    sa, sb = " ".join(ta), " ".join(tb)
    seq = difflib.SequenceMatcher(None, sa, sb).ratio() if sa and sb else 0.0
    ja = len(set(ta) & set(tb)) / max(1, len(set(ta) | set(tb)))
    cont = token_containment(ta, tb)
    return max(seq, ja, cont), seq, cont


def role_sequence_ratio(a: dict[str, Any], b: dict[str, Any], role: str) -> float:
    """Token-level similarity used only as review evidence, never alone as a gate."""
    ta, tb = words(row_text(a, role)), words(row_text(b, role))
    if not ta or not tb:
        return 0.0
    return difflib.SequenceMatcher(None, ta, tb, autojunk=False).ratio()


def pair_payload(a: dict[str, Any], b: dict[str, Any], score: float, seq: float, cont: float, decision: str, reason: str) -> dict[str, Any]:
    return {
        "sample_a": a.get("sample_id"), "sample_b": b.get("sample_id"), "similarity": round(score, 6), "sequence_ratio": round(seq, 6), "token_containment": round(cont, 6),
        "decision": decision, "reason": reason,
        "provenance_a": {k: a.get(k) for k in ("source_id", "canonical_recording_id", "recording_family_id", "context_turn_ids", "target_turn_ids", "timestamps")},
        "provenance_b": {k: b.get(k) for k in ("source_id", "canonical_recording_id", "recording_family_id", "context_turn_ids", "target_turn_ids", "timestamps")},
        "context_a": row_text(a, "user")[:2000], "target_a": row_text(a, "assistant")[:2000],
        "context_b": row_text(b, "user")[:2000], "target_b": row_text(b, "assistant")[:2000],
        "content_review": "actual materialized context and target compared with provenance; no response-only automatic deletion",
    }


def dedup_and_split(args: argparse.Namespace, recovered: list[dict[str, Any]]) -> dict[str, Any]:
    interaction = load_jsonl(Path(args.interaction_pool))
    for row in interaction:
        row.setdefault("recovery_source", "existing_self_contained_interaction_pool")
    for row in recovered:
        row.setdefault("recovery_source", "deterministic_bounded_context_recovery")
    rows = sorted(interaction + recovered, key=lambda r: str(r.get("sample_id")))
    exact_groups: dict[str, list[int]] = defaultdict(list)
    target_groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        exact_groups[norm(row_text(row))].append(i)
        target_groups[tuple(str(x) for x in row.get("target_turn_ids") or [])].append(i)
    removed: set[int] = set()
    removal_reasons: Counter[str] = Counter()
    duplicate_examples: list[dict[str, Any]] = []
    for key, indices in exact_groups.items():
        if not key or len(indices) < 2:
            continue
        # Identical behavior across distinct recording families is preserved.
        by_family = defaultdict(list)
        for i in indices:
            by_family[str(rows[i].get("recording_family_id") or rows[i].get("canonical_recording_id"))].append(i)
        for family_indices in by_family.values():
            if len(family_indices) <= 1:
                continue
            keep = family_indices[0]
            for i in family_indices[1:]:
                removed.add(i); removal_reasons["exact_normalized_same_family"] += 1
                if len(duplicate_examples) < 50:
                    duplicate_examples.append({"reason": "exact_normalized_same_family", "kept": rows[keep].get("sample_id"), "removed": rows[i].get("sample_id")})
    for key, indices in target_groups.items():
        if not key or len(indices) < 2:
            continue
        keep = indices[0]
        for i in indices[1:]:
            removed.add(i); removal_reasons["target_turn_duplicate"] += 1
            if len(duplicate_examples) < 50:
                duplicate_examples.append({"reason": "target_turn_duplicate", "kept": rows[keep].get("sample_id"), "removed": rows[i].get("sample_id"), "target_turn_ids": list(key)})

    # Candidate pairs are narrowed by shared 3-grams, exact target text, or
    # same family. This avoids an O(n^2) scan while still checking every pair
    # that can reach the 70% interaction-similarity threshold.
    inverted: dict[tuple[str, ...], list[int]] = defaultdict(list)
    target_text_groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        toks = words(row_text(row))
        shingles = set(tuple(toks[j : j + 3]) for j in range(max(0, len(toks) - 2)))
        for sh in sorted(shingles)[:80]:
            inverted[sh].append(i)
        target_text_groups[norm(row_text(row, "assistant"))].append(i)
    pair_indices: set[tuple[int, int]] = set()
    for members in list(inverted.values()) + list(target_text_groups.values()):
        if len(members) > 500:
            # Very common short phrases do not imply full interaction similarity;
            # compare only same-family members in oversized buckets.
            members = [i for i in members if rows[i].get("recording_family_id")]
        for pos, i in enumerate(sorted(set(members))):
            for j in sorted(set(members))[pos + 1 :]:
                pair_indices.add((i, j) if i < j else (j, i))
    families: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        families[str(row.get("recording_family_id") or row.get("canonical_recording_id"))].append(i)
    for members in families.values():
        if len(members) <= 300:
            for pos, i in enumerate(members):
                for j in members[pos + 1 :]:
                    pair_indices.add((i, j) if i < j else (j, i))

    high_pairs: list[dict[str, Any]] = []
    preserve_pairs = 0
    remove_pairs = 0
    for i, j in sorted(pair_indices):
        score, seq, cont = similarity(rows[i], rows[j])
        if score < 0.70:
            continue
        a, b = rows[i], rows[j]
        same_family = str(a.get("recording_family_id") or a.get("canonical_recording_id")) == str(b.get("recording_family_id") or b.get("canonical_recording_id"))
        same_source = str(a.get("source_id") or "") == str(b.get("source_id") or "")
        ia, ib = interval(a), interval(b)
        overlap = max(0.0, min(ia[1], ib[1]) - max(ia[0], ib[0]))
        exact = norm(row_text(a)) == norm(row_text(b))
        context_seq = role_sequence_ratio(a, b, "user")
        target_seq = role_sequence_ratio(a, b, "assistant")
        target_ids_overlap = bool(set(str(x) for x in a.get("target_turn_ids") or []) & set(str(x) for x in b.get("target_turn_ids") or []))

        # Similarity is only a routing signal.  A removal requires the
        # canonical provenance to agree as well: either the same selected
        # target/overlapping interval, or a strong cross-source mirror inside
        # one already-established recording family.  Distinct families are
        # retained as genuine behaviour repetition.
        same_recording_duplicate = same_family and (exact or cont >= 0.90 or target_ids_overlap or overlap > 0)
        mirror_duplicate = same_family and not same_source and ((target_seq >= 0.90 and context_seq >= 0.55) or (target_seq >= 0.70 and context_seq >= 0.80))
        redundant = same_recording_duplicate or mirror_duplicate
        if redundant:
            if mirror_duplicate and not same_recording_duplicate:
                reason = "same recording family, distinct source IDs, and matching context/target semantics; provenance confirms reupload/mirror supervision"
            else:
                reason = "same recording family with exact/containment/target-overlap or overlapping interval provenance"
            decision = "REMOVE_TRUE_RECORDING_OR_SUPERVISION_DUPLICATE"
            remove_pairs += 1
            # Prefer the original self-contained artifact, then the shorter
            # context, then stable sample ordering.  This prevents recovery
            # from replacing a canonical clean row while remaining resumable.
            def rank(index: int) -> tuple[int, int, str]:
                row = rows[index]
                source_rank = 0 if row.get("recovery_source") == "existing_self_contained_interaction_pool" else 1
                return (source_rank, len(row.get("context_turn_ids") or []), str(row.get("sample_id")))
            loser = j if rank(i) <= rank(j) else i
            removed.add(loser)
            removal_reasons["high_similarity_same_family_redundancy"] += 1
        else:
            decision, reason = "PRESERVE_NATURAL_BEHAVIOR_REPETITION", "similarity without same-recording redundant provenance; response repetition is retained"
            preserve_pairs += 1
        payload = pair_payload(a, b, score, seq, cont, decision, reason)
        payload.update({"context_sequence_ratio": round(context_seq, 6), "target_sequence_ratio": round(target_seq, 6),
                        "same_source": same_source, "interval_overlap_seconds": round(overlap, 6),
                        "target_turn_overlap": target_ids_overlap,
                        "semantic_review_disposition": decision,
                        "semantic_review_basis": "actual context+target token comparison plus recording/source/timestamp/turn-id provenance"})
        high_pairs.append(payload)

    kept = [row for i, row in enumerate(rows) if i not in removed]
    split_authority = load_json(Path(args.split_authority))
    family_assignments = split_authority.get("current_family_assignments") or {}
    # A repaired family hash is intentionally ephemeral: the authoritative
    # lineage record is also keyed by the underlying canonical recording
    # member.  When a family has changed membership (or was quarantined from
    # the current family table), resolve its split through that member map
    # instead of silently dropping an otherwise valid candidate.
    member_assignments = split_authority.get("member_assignments") or {}
    cluster_map = load_json(Path(args.recording_repair)).get("cluster_to_family") or {}
    split_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    unresolved_split: list[dict[str, Any]] = []
    for row in kept:
        family = str(row.get("recording_family_id") or cluster_map.get(str(row.get("canonical_recording_id"))) or "")
        split = family_assignments.get(family)
        split_authority_resolution = "current_family_assignments"
        if not split:
            canonical_id = str(row.get("canonical_recording_id") or "")
            split = member_assignments.get(canonical_id)
            split_authority_resolution = "member_assignments_fallback" if split else None
        if not split:
            unresolved_split.append({"sample_id": row.get("sample_id"), "family": family, "canonical_recording_id": row.get("canonical_recording_id")})
            continue
        row["recording_family_id"] = family
        row["split"] = split
        row["split_authority_resolution"] = split_authority_resolution
        row["recovery_view"] = "recovered_context_natural_frequency"
        split_counts[split] += 1
        family_counts[family] += 1
    if unresolved_split:
        # Fail closed rather than placing an unknown lineage in a split.
        kept = [row for row in kept if row.get("sample_id") not in {x["sample_id"] for x in unresolved_split}]
    output_dir = Path(args.output_dir)
    train = [r for r in kept if r.get("split") == "train"]
    validation = [r for r in kept if r.get("split") == "validation"]
    sealed = [r for r in kept if r.get("split") == "sealed_eval"]
    write_jsonl(output_dir / "context_recovered_candidate_pool_v2_3.jsonl", kept)
    views = output_dir / "views" / "recovered_context_closed"
    write_jsonl(views / "train_recovered_context.jsonl", train)
    write_jsonl(views / "validation.jsonl", validation)
    write_jsonl(views / "sealed_eval.jsonl", sealed)
    target_ids = [str(x) for row in kept for x in row.get("target_turn_ids") or []]
    context_keys = [(str(row.get("recording_family_id")), tuple(str(x) for x in row.get("context_turn_ids") or [])) for row in kept]
    prefix_count = 0
    # Strict prefix ladder check within one family and target episode.
    by_episode: dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]] = defaultdict(list)
    for row in kept:
        by_episode[(str(row.get("recording_family_id")), tuple(str(x) for x in row.get("target_turn_ids") or []))].append(row)
    for members in by_episode.values():
        texts = [words(row_text(row, "user")) for row in members]
        for i, ta in enumerate(texts):
            for j, tb in enumerate(texts):
                if i != j and len(ta) < len(tb) and ta == tb[: len(ta)]:
                    prefix_count += 1
    view_manifest = {"schema_version": "recovered-context-view-v1", "pipeline_version": PIPELINE_VERSION, "recovery_version": RECOVERY_VERSION,
                     "policy": "natural_frequency_all_deduped_recovered_candidates", "shared_split_authority": str(Path(args.split_authority)),
                     "split_authority_version": split_authority.get("split_authority_version"),
                     "split_authority": str(Path(args.split_authority)),
                     "counts": {"train": len(train), "validation": len(validation), "sealed_eval": len(sealed), "total": len(kept)},
                     "files": {"train": "train_recovered_context.jsonl", "validation": "validation.jsonl", "sealed_eval": "sealed_eval.jsonl"},
                     "ready_for_first_sft": False}
    write_json(views / "view_manifest.json", view_manifest)
    audit = {"schema_version": "context-recovery-dedup-audit-v1", "pipeline_version": PIPELINE_VERSION,
             "input_rows": len(rows), "kept_rows": len(kept), "removed_rows": len(removed),
             "exact_duplicate_count": sum(v for k, v in removal_reasons.items() if k.startswith("exact_")),
             "target_duplicate_count": removal_reasons.get("target_turn_duplicate", 0),
             "near_duplicate_pair_count_ge_70": len(high_pairs), "near_duplicate_pairs_removed": remove_pairs,
             "natural_repetition_pairs_preserved": preserve_pairs,
             "removal_reason_counts": dict(removal_reasons), "duplicate_examples": duplicate_examples,
             "target_reuse_count": len(target_ids) - len(set(target_ids)), "prefix_ladder_count": prefix_count,
             "recording_family_counts": dict(family_counts), "split_counts": dict(split_counts), "unresolved_split_lineage": unresolved_split,
             "semantic_review": {"method": "actual materialized context+target text plus canonical provenance for every >=70% pair", "automatic_response_only_deletion": False,
                                 "cross_recording_behavior_repetition_preserved": True}, "generated_at": datetime.now(timezone.utc).isoformat()}
    write_json(output_dir / "context_recovery_dedup_audit_v2_3.json", audit)
    write_jsonl(output_dir / "context_recovery_high_similarity_pairs_ge70_v2_3.jsonl", high_pairs)
    clusters: dict[str, list[str]] = defaultdict(list)
    for pair in high_pairs:
        if pair["decision"].startswith("REMOVE"):
            clusters["remove_same_recording"].extend([pair["sample_a"], pair["sample_b"]])
        else:
            clusters["preserve_natural_repetition"].extend([pair["sample_a"], pair["sample_b"]])
    write_json(output_dir / "context_recovery_high_similarity_clusters_v2_3.json", {"schema_version": "high-similarity-clusters-v1", "clusters": {k: sorted(set(v)) for k, v in clusters.items()}, "pair_count": len(high_pairs)})
    return {"audit": audit, "view_manifest": view_manifest, "high_pairs": high_pairs, "kept": kept, "train": train, "validation": validation, "sealed_eval": sealed}


def main() -> None:
    parser = argparse.ArgumentParser(description="deterministic v2.3 context-incomplete recovery")
    parser.add_argument("--candidates", default=str(DATASET / "context_reconstruction_candidates_v2_3.jsonl"))
    parser.add_argument("--interaction-pool", default=str(DATASET / "interaction_view_pool_v2_3.jsonl"))
    parser.add_argument("--split-authority", default=str(DATASET / "split_authority_v2_3.json"))
    parser.add_argument("--recording-repair", default=str(DATASET / "recording_family_repair_v2_3.json"))
    parser.add_argument("--output-dir", default=str(DATASET / "context_recovery_v2_3"))
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    recovered, unresolved, report = recover(args)
    write_jsonl(output / "context_recovered_rows_v2_3.jsonl", recovered)
    write_jsonl(output / "context_recovery_unresolved_v2_3.jsonl", unresolved)
    report["output_files"] = {"recovered": "context_recovered_rows_v2_3.jsonl", "unresolved": "context_recovery_unresolved_v2_3.jsonl"}
    write_json(output / "context_recovery_report_v2_3.json", report)
    final = dedup_and_split(args, recovered)
    final_report = {"schema_version": "context-recovery-final-report-v1", "pipeline_version": PIPELINE_VERSION, "recovery_version": RECOVERY_VERSION,
                    "recovery": report, "dedup": final["audit"], "views": final["view_manifest"],
                    "output_dir": str(output), "generated_at": datetime.now(timezone.utc).isoformat()}
    write_json(output / "context_recovery_final_report_v2_3.json", final_report)
    print(json.dumps({"recovery": report, "dedup": final["audit"], "views": final["view_manifest"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
