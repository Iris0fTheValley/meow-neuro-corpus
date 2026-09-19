from __future__ import annotations

"""Cache-first production recovery v2 for audio-reconstruction-v1.

This runner reuses the pinned Qwen ASR/alignment, ECAPA diarization, and target
activity cache.  It does not execute an expensive model stage and never mutates
the frozen semantic, identity, recording-lineage, or split authorities.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audio_evidence.cache import EvidenceCache  # noqa: E402
from audio_evidence.materialization import FinalViewMembership, materialize_role_preserving  # noqa: E402
from audio_evidence.recovery_v2 import (  # noqa: E402
    ContextSufficiencyState,
    QuarantineTrace,
    ReconciliationState,
    SpeakerState,
    TargetResolutionState,
    assign_tokens_to_old_turns,
    build_candidate_spans,
    canonical_sha256,
    context_sufficiency,
    deduplicate_recovery_rows,
    interval_overlap,
    is_non_conversational_sentinel,
    reconcile_old_turn,
    reconcile_span,
    resolve_speaker_state,
    resolve_target,
    select_minimal_context,
    sentinel_only_context,
)
from audio_evidence.recovery_validation import validate_recovery_v2  # noqa: E402
import run_audio_evidence_production as replay  # noqa: E402


CORPUS = ROOT.parent.parent
DATASET = CORPUS / "datasets" / "meow_v02_sft_v2_2_semantic_verified"
RUN_ROOT = DATASET / "audio_reconstruction_v1"
DEFAULT_BASELINE_RUN = RUN_ROOT / "audio-reconstruction-v1-real-completion-20260919"
DEFAULT_CACHE_RUN = RUN_ROOT / "audio-reconstruction-v1-real-context60-bounded120-20260918"
DEFAULT_OUT_NAME = "audio-reconstruction-v1-recovery-v2-20260919"
CONTEXT_DECISIONS = DATASET / "context_sufficiency_decisions_v2_3.jsonl"

SCHEMA_VERSION = "audio-reconstruction-recovery-v2.0.0"
THRESHOLD_VERSION = "audio-recovery-v2-thresholds-20260919"
TIMEBASE_VERSION = "window-local-to-recording-v1"
CONFIG = {
    "candidate_search_seconds": 60.0,
    "context_hard_gap_seconds": 8.0,
    "alignment_boundary_tolerance_seconds": 0.12,
    "candidate_span_max_gap_seconds": 1.25,
    "old_new_match_similarity": 0.88,
    "old_new_merge_similarity": 0.60,
    "target_audio_match_similarity": 0.86,
    "target_audio_minor_similarity": 0.55,
    "threshold_version": THRESHOLD_VERSION,
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_state() -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
        return commit, dirty
    except Exception:
        return "unknown", True


def membership_for(row: dict[str, Any], assignments: dict[str, str]) -> FinalViewMembership:
    return {
        "train": FinalViewMembership.IN_TRAIN,
        "validation": FinalViewMembership.IN_VALIDATION,
        "sealed_eval": FinalViewMembership.IN_SEALED_EVAL,
    }.get(assignments.get(str(row.get("canonical_recording_id"))), FinalViewMembership.EXCLUDED)


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run", default=str(DEFAULT_BASELINE_RUN))
    parser.add_argument("--cache-run", default=str(DEFAULT_CACHE_RUN))
    parser.add_argument("--out-name", default=DEFAULT_OUT_NAME)
    parser.add_argument("--limit", type=int, default=0, help="smoke-test interactions only")
    args = parser.parse_args()

    baseline_run = Path(args.baseline_run)
    cache_run = Path(args.cache_run)
    out = RUN_ROOT / args.out_name
    if out.exists() and any(out.iterdir()):
        raise SystemExit("refusing to overwrite existing versioned run: %s" % out)
    out.mkdir(parents=True, exist_ok=False)

    pool = load_jsonl(replay.POOL)
    source_manifest = replay.source_metadata()
    identity_rows = load_jsonl(replay.IDENTITY)
    identity_map = {(str(row.get("source_id")), str(row.get("cluster"))): row for row in identity_rows}
    split_document = json.loads(replay.SPLIT_AUTHORITY.read_text(encoding="utf-8"))
    split_assignments = split_document.get("member_assignments") or {}
    context_states = {str(row.get("sample_id")): str(row.get("state")) for row in load_jsonl(CONTEXT_DECISIONS)}
    interactions, semantic_hashes = replay.prepare_interactions(pool, source_manifest, split_document)
    if args.limit:
        interactions = interactions[: args.limit]
        semantic_hashes = {str(row["sample_id"]): semantic_hashes[str(row["sample_id"])] for row in interactions}
    interactions_by_recording: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in interactions:
        interactions_by_recording[str(row["source_id"])].append(row)
    limited_sample_ids = {str(row["sample_id"]) for row in interactions} if args.limit else set()

    baseline_rows = load_jsonl(baseline_run / "materialized.jsonl")
    if args.limit:
        allowed = {str(row["sample_id"]) for row in interactions}
        baseline_rows = [row for row in baseline_rows if str(row.get("sample_id")) in allowed]
    baseline_by_sample = {str(row.get("sample_id")): row for row in baseline_rows}
    baseline_ids = set(baseline_by_sample)
    baseline_quarantine_rows = load_jsonl(baseline_run / "quarantine.jsonl")
    baseline_quarantine = {str(row.get("sample_id")): str(row.get("reason")) for row in baseline_quarantine_rows}
    baseline_role_votes: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in baseline_rows:
        sid = str(row.get("recording_id"))
        for message in row.get("messages") or []:
            for turn_id in message.get("source_turn_ids") or []:
                baseline_role_votes[(sid, str(turn_id))][str(message.get("role"))] += 1

    target_identity_by_turn: dict[tuple[str, str], str] = {}
    semantic_context_role_votes: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    all_target_ids = set()
    for row in interactions:
        sid = str(row["source_id"])
        target_ids = [str(value) for value in row.get("target_turn_ids") or []]
        identity = str((row.get("speaker_evidence") or {}).get("assistant_identity") or row.get("identity") or "")
        for turn_id in target_ids:
            target_identity_by_turn[(sid, turn_id)] = identity
            all_target_ids.add(turn_id)
        for turn_id in row.get("context_turn_ids") or []:
            semantic_context_role_votes[(sid, str(turn_id))]["user"] += 1

    plan = json.loads((baseline_run / "window_plan.json").read_text(encoding="utf-8"))
    windows_by_recording: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for window in plan.get("windows") or []:
        if args.limit and not (limited_sample_ids & {str(value) for value in window.get("sample_ids") or []}):
            continue
        windows_by_recording[str(window["recording_id"])].append(window)
    checkpoint = json.loads((cache_run / "checkpoint.json").read_text(encoding="utf-8")).get("completed") or {}
    cache = EvidenceCache(cache_run / "cache")

    def cached_stage(window_id: str, stage: str) -> Any:
        key = (checkpoint.get(window_id) or {}).get(stage)
        return cache.get(stage, key) if key else None

    timeline_rows: list[dict[str, Any]] = []
    reconciliation_rows: list[dict[str, Any]] = []
    correction_rows: list[dict[str, Any]] = []
    discovered_rows: list[dict[str, Any]] = []
    speaker_rows: list[dict[str, Any]] = []
    ambiguous_speaker_rows: list[dict[str, Any]] = []
    context_candidate_rows: list[dict[str, Any]] = []
    context_selection_rows: list[dict[str, Any]] = []
    sufficiency_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    target_disagreements: list[dict[str, Any]] = []
    targeted_reasr_rows: list[dict[str, Any]] = []
    materialized_rows: list[dict[str, Any]] = []
    quarantine_rows: list[dict[str, Any]] = []

    for recording_index, sid in enumerate(sorted(interactions_by_recording), 1):
        timeline_path = replay.TIMELINE_DIR / (sid + ".json")
        if not timeline_path.exists():
            continue
        old_document = json.loads(timeline_path.read_text(encoding="utf-8"))
        raw_old_turns = sorted(
            old_document.get("turns") or [],
            key=lambda turn: (float(turn["timestamp"]["start"]), float(turn["timestamp"]["end"]), str(turn["turn_id"])),
        )
        old_turns = [
            {
                "turn_id": str(turn["turn_id"]),
                "start": float(turn["timestamp"]["start"]),
                "end": float(turn["timestamp"]["end"]),
                "text": str(turn.get("text") or "").strip(),
                "speaker": str(turn.get("speaker") or "UNKNOWN"),
                "speaker_confidence": turn.get("speaker_confidence"),
            }
            for turn in raw_old_turns
        ]
        windows = sorted(windows_by_recording.get(sid, []), key=lambda item: float(item["merged_interval"]["start"]))
        token_map: dict[tuple[float, float, str], dict[str, Any]] = {}
        diarization_map: dict[tuple[float, float, str], dict[str, Any]] = {}
        activity_map: dict[tuple[float, float, str], dict[str, Any]] = {}
        evidence_availability_by_window: dict[str, dict[str, bool]] = {}
        for window in windows:
            window_id = str(window["window_id"])
            alignment = cached_stage(window_id, "alignment") or []
            diarization = cached_stage(window_id, "diarization") or []
            activity = cached_stage(window_id, "target_activity") or []
            evidence_availability_by_window[window_id] = {
                "alignment": bool(alignment),
                "diarization": bool(diarization),
                "target_activity": bool(activity),
                "asr": bool(cached_stage(window_id, "asr")),
            }
            for token in alignment:
                key = (round(float(token.get("start", 0)), 2), round(float(token.get("end", 0)), 2), str(token.get("text") or "").strip().lower())
                item = token_map.setdefault(key, dict(token, source_window_ids=[]))
                item["source_window_ids"] = sorted(set(item["source_window_ids"]) | {window_id})
            for item in diarization:
                key = (round(float(item.get("start", 0)), 2), round(float(item.get("end", 0)), 2), str(item.get("speaker_cluster") or "UNKNOWN"))
                value = diarization_map.setdefault(key, dict(item, source_window_ids=[]))
                value["source_window_ids"] = sorted(set(value["source_window_ids"]) | {window_id})
            for item in activity:
                key = (round(float(item.get("start", 0)), 2), round(float(item.get("end", 0)), 2), str(item.get("enrollment_id") or ""))
                value = activity_map.setdefault(key, dict(item, source_window_ids=[]))
                value["source_window_ids"] = sorted(set(value["source_window_ids"]) | {window_id})
        tokens = sorted(token_map.values(), key=lambda item: (float(item["start"]), float(item["end"]), str(item.get("text") or "")))
        diarization = sorted(diarization_map.values(), key=lambda item: (float(item["start"]), float(item["end"])))
        activity = sorted(activity_map.values(), key=lambda item: (float(item["start"]), float(item["end"])))
        assigned, unassigned, ambiguous_boundary_tokens = assign_tokens_to_old_turns(
            tokens,
            old_turns,
            boundary_tolerance=CONFIG["alignment_boundary_tolerance_seconds"],
        )
        ambiguous_turn_ids = {
            str(turn_id)
            for token in ambiguous_boundary_tokens
            for turn_id in token.get("candidate_old_turn_ids") or []
        }
        baseline_roles = {}
        for turn in old_turns:
            vote = baseline_role_votes.get((sid, str(turn["turn_id"])))
            baseline_roles[str(turn["turn_id"])] = vote.most_common(1)[0][0] if vote else None

        canonical_recording_id = str(interactions_by_recording[sid][0].get("canonical_recording_id"))
        recording_family_id = str(interactions_by_recording[sid][0].get("recording_family_id"))
        record_turns: dict[str, dict[str, Any]] = {}
        for old in old_turns:
            turn_id = str(old["turn_id"])
            mapping = identity_map.get((sid, str(old["speaker"]))) or {}
            identity = mapping.get("identity")
            if turn_id in all_target_ids and str(target_identity_by_turn.get((sid, turn_id)) or "").upper() in {
                "NEURO", "EVIL_NEURO", "NEURO_FAMILY", "NEURO_FAMILY_HIGH", "NEURO_FAMILY_MEDIUM",
            }:
                identity = target_identity_by_turn[(sid, turn_id)]
            source_windows = [
                window for window in windows
                if interval_overlap(
                    float(old["start"]), float(old["end"]),
                    float(window["merged_interval"]["start"]), float(window["merged_interval"]["end"]),
                ) > 0
            ]
            activity_available = any(
                interval_overlap(float(old["start"]), float(old["end"]), float(item["start"]), float(item["end"])) > 0
                for item in activity
            )
            diar_available = any(
                interval_overlap(float(old["start"]), float(old["end"]), float(item["start"]), float(item["end"])) > 0
                for item in diarization
            )
            speaker_resolution = resolve_speaker_state(
                identity,
                baseline_role=baseline_roles.get(turn_id),
                frozen_semantic_role=(
                    semantic_context_role_votes[(sid, turn_id)].most_common(1)[0][0]
                    if semantic_context_role_votes.get((sid, turn_id))
                    else None
                ),
                identity_evidence_available=bool(mapping) or turn_id in all_target_ids,
                target_activity_available=activity_available,
                generic_diarization_available=diar_available,
            )
            speaker_entry = {
                "audio_turn_id": turn_id,
                "recording_id": sid,
                "identity": identity,
                **speaker_resolution,
            }
            speaker_rows.append(speaker_entry)
            if speaker_resolution["speaker_state"] == SpeakerState.AMBIGUOUS_SPEAKER.value:
                ambiguous_speaker_rows.append(speaker_entry)
            reconciliation = reconcile_old_turn(
                old,
                assigned.get(turn_id) or [],
                is_frozen_target=turn_id in all_target_ids,
            )
            reconciliation.update({
                "recording_id": sid,
                "candidate_span_id": "old:" + turn_id,
                "ambiguous_boundary_token_count": sum(
                    turn_id in (token.get("candidate_old_turn_ids") or []) for token in ambiguous_boundary_tokens
                ),
                "materialized_as_single_turn": False,
                "training_eligible": speaker_resolution["speaker_state"] != SpeakerState.AMBIGUOUS_SPEAKER.value,
            })
            reconciliation_rows.append(reconciliation)
            chosen_text = str(reconciliation.get("chosen_text") or old["text"]).strip()
            sentinel = is_non_conversational_sentinel(chosen_text)
            training_eligible = (
                speaker_resolution["speaker_state"] != SpeakerState.AMBIGUOUS_SPEAKER.value
                and not sentinel
                and bool(chosen_text)
            )
            text_authority = str(reconciliation.get("chosen_text_authority") or "OLD_TRANSCRIPT")
            record = {
                "schema_version": "1.0.0",
                "pipeline_version": SCHEMA_VERSION,
                "audio_turn_id": turn_id,
                "recording_id": sid,
                "canonical_recording_id": canonical_recording_id,
                "recording_family_id": recording_family_id,
                "start": float(old["start"]),
                "end": float(old["end"]),
                "timebase": "SECONDS_FROM_RECORDING_START",
                "old_transcript": old["text"] or None,
                "new_asr_hypothesis": reconciliation.get("new_text"),
                "resolved_text": chosen_text or None,
                "text_resolution_state": "RESOLVED" if chosen_text else "UNRESOLVED",
                "text_authority": text_authority,
                "text_resolution_provenance": {
                    "status": "RESOLVED" if chosen_text else "UNRESOLVED",
                    "resolver": "TURN_RECONCILIATION_V2",
                    "revision": SCHEMA_VERSION,
                    "reconciliation_state": reconciliation["reconciliation_state"],
                    "resolution_reason": reconciliation["resolution_reason"],
                    "resolved_disagreements": ["MINOR_TEXT_CHANGE"] if reconciliation["reconciliation_state"] == ReconciliationState.EXTEND_EXISTING.value else [],
                },
                "disagreement_flags": ["MINOR_TEXT_CHANGE"] if reconciliation["reconciliation_state"] == ReconciliationState.EXTEND_EXISTING.value else [],
                "identity": identity,
                "identity_authority_revision": "multimodal-fusion-proxy-2026-09-12-v2-chat-tts-rejection",
                "speaker_cluster": old["speaker"],
                "speaker_resolution_state": speaker_resolution["speaker_state"],
                "speaker_resolution": speaker_resolution,
                "role": speaker_resolution["role"],
                "training_eligible": training_eligible,
                "non_conversational_sentinel": sentinel,
                "reconciliation_state": reconciliation["reconciliation_state"],
                "source_audio": str((source_manifest.get(sid) or {}).get("audio_path") or f"raw_audio/{sid}.webm"),
                "source_interval": {"start": float(old["start"]), "end": float(old["end"]), "timebase": "SECONDS_FROM_RECORDING_START"},
                "source_window_ids": [str(window["window_id"]) for window in source_windows],
                "alignment_token_count": len(assigned.get(turn_id) or []),
                "ambiguous_boundary_token_count": reconciliation["ambiguous_boundary_token_count"],
                "target_activity_available": activity_available,
                "diarization_available": diar_available,
            }
            record_turns[turn_id] = record
            timeline_rows.append(record)
            if reconciliation["reconciliation_state"] == ReconciliationState.EXTEND_EXISTING.value:
                correction_rows.append({
                    "audio_turn_id": turn_id,
                    "recording_id": sid,
                    "correction_type": "TEXT_EXTENSION",
                    "old_text": old["text"],
                    "corrected_text": chosen_text,
                    "chosen_text_authority": text_authority,
                    "speaker_state": speaker_resolution["speaker_state"],
                    "evidence": reconciliation,
                })
            if speaker_resolution["resolution_basis"] == "BASELINE_MATERIALIZED_ROLE_MONOTONICITY":
                correction_rows.append({
                    "audio_turn_id": turn_id,
                    "recording_id": sid,
                    "correction_type": "BASELINE_ROLE_RETAINED",
                    "old_speaker_identity": identity,
                    "corrected_role": speaker_resolution["role"],
                    "speaker_state": speaker_resolution["speaker_state"],
                    "legacy_role_validated": True,
                    "evidence": speaker_resolution["evidence"],
                })

        candidate_spans = build_candidate_spans(
            tokens,
            diarization,
            recording_id=sid,
            window_ids=[str(window["window_id"]) for window in windows],
            max_gap=CONFIG["candidate_span_max_gap_seconds"],
        )
        for span in candidate_spans:
            overlapping = [
                old for old in old_turns
                if interval_overlap(float(span["start"]), float(span["end"]), float(old["start"]), float(old["end"])) > 0
            ]
            entry = reconcile_span(span, overlapping)
            entry.update({
                "recording_id": sid,
                "source_window_ids": span["source_window_ids"],
                "materialized_as_single_turn": False,
                "training_eligible": False,
            })
            if entry["reconciliation_state"] in {
                ReconciliationState.MERGE_EXISTING.value,
                ReconciliationState.TRUE_NEW.value,
                ReconciliationState.AMBIGUOUS_BOUNDARY.value,
            }:
                reconciliation_rows.append(entry)
            if entry["reconciliation_state"] == ReconciliationState.TRUE_NEW.value:
                speaker_resolution = resolve_speaker_state(
                    None,
                    baseline_role=None,
                    identity_evidence_available=False,
                    target_activity_available=any(
                        interval_overlap(float(span["start"]), float(span["end"]), float(item["start"]), float(item["end"])) > 0
                        for item in activity
                    ),
                    generic_diarization_available=bool(span.get("speaker_cluster")),
                )
                discovered = {
                    **span,
                    "reconciliation_state": entry["reconciliation_state"],
                    "speaker_resolution": speaker_resolution,
                    "speaker_state": speaker_resolution["speaker_state"],
                    "candidate_state": "RECONSTRUCTED_CANDIDATE",
                    "training_eligible": False,
                    "non_conversational_sentinel": is_non_conversational_sentinel(span.get("text")),
                }
                discovered_rows.append(discovered)
                ambiguous_speaker_rows.append({
                    "audio_turn_id": span["candidate_span_id"],
                    "recording_id": sid,
                    **speaker_resolution,
                })

        for row in sorted(interactions_by_recording[sid], key=lambda item: str(item["sample_id"])):
            sample = str(row["sample_id"])
            baseline = baseline_by_sample.get(sample)
            baseline_materialized = baseline is not None
            target_ids = [str(value) for value in row.get("target_turn_ids") or []]
            target_id = target_ids[-1] if target_ids else ""
            target = record_turns.get(target_id)
            old_target = next((turn for turn in old_turns if str(turn["turn_id"]) == target_id), None)
            containing_windows = []
            intersecting_windows = []
            target_interval = None
            if old_target:
                target_interval = {"start": old_target["start"], "end": old_target["end"], "timebase": "SECONDS_FROM_RECORDING_START"}
                containing_windows = [
                    str(window["window_id"]) for window in windows
                    if float(window["merged_interval"]["start"]) <= float(old_target["start"]) + 1e-6
                    and float(old_target["end"]) <= float(window["merged_interval"]["end"]) + 1e-6
                ]
                intersecting_windows = [
                    str(window["window_id"]) for window in windows
                    if interval_overlap(
                        float(old_target["start"]), float(old_target["end"]),
                        float(window["merged_interval"]["start"]), float(window["merged_interval"]["end"]),
                    ) > 0
                ]
            evidence_window_ids = containing_windows or intersecting_windows
            evidence_flags = {
                key: any((evidence_availability_by_window.get(window_id) or {}).get(key, False) for window_id in evidence_window_ids)
                for key in ("alignment", "diarization", "target_activity", "asr")
            }
            target_resolution = resolve_target(
                old_text=(target or {}).get("old_transcript"),
                new_text=(target or {}).get("new_asr_hypothesis"),
                baseline_materialized=baseline_materialized,
                alignment_available=bool((target or {}).get("alignment_token_count")),
                identity_confirmed_target=(target or {}).get("speaker_resolution_state") == SpeakerState.CONFIRMED_TARGET.value,
                ambiguous_boundary=(not baseline_materialized and target_id in ambiguous_turn_ids),
                aligned_token_count=int((target or {}).get("alignment_token_count") or 0),
            )
            target_resolution.update({
                "sample_id": sample,
                "target_turn_id": target_id,
                "target_interval": target_interval,
                "containing_window_ids": containing_windows,
                "intersecting_window_ids": intersecting_windows,
                "alignment_available": evidence_flags["alignment"],
                "target_activity_available": evidence_flags["target_activity"],
                "diarization_available": evidence_flags["diarization"],
                "identity_evidence_available": bool(target and target.get("speaker_resolution_state") == SpeakerState.CONFIRMED_TARGET.value),
                "resolution_basis": "FROZEN_SEMANTIC_AND_IDENTITY_AUTHORITY",
            })
            target_rows.append(target_resolution)
            if target_resolution["state"] in {
                TargetResolutionState.AUDIO_MINOR_DISAGREEMENT.value,
                TargetResolutionState.AUDIO_MAJOR_CONTRADICTION.value,
                TargetResolutionState.LOCAL_REASR_REQUIRED.value,
            }:
                target_disagreements.append(target_resolution)
            if target_resolution["state"] in {
                TargetResolutionState.AUDIO_MAJOR_CONTRADICTION.value,
                TargetResolutionState.LOCAL_REASR_REQUIRED.value,
            }:
                targeted_reasr_rows.append({
                    "sample_id": sample,
                    "why_recomputed": target_resolution["reason"],
                    "audio_interval": target_interval,
                    "models": [],
                    "revisions": [],
                    "parameters": {},
                    "previous_evidence": target_resolution,
                    "new_evidence": None,
                    "resolution": "QUEUED_NOT_EXECUTED",
                    "local_reasr_attempted": False,
                })

            blocking_target_states = {
                TargetResolutionState.AUDIO_MAJOR_CONTRADICTION.value,
                TargetResolutionState.LOCAL_REASR_REQUIRED.value,
                TargetResolutionState.UNRESOLVED.value,
            }
            if target is None or target_resolution["state"] in blocking_target_states:
                quarantine_rows.append(QuarantineTrace(
                    sample_id=sample,
                    was_baseline_materialized=baseline_materialized,
                    was_baseline_quarantined=sample in baseline_quarantine,
                    baseline_reason=baseline_quarantine.get(sample),
                    failure_stage="TARGET_RESOLUTION",
                    failure_reason=target_resolution["reason"],
                    target_interval=target_interval,
                    containing_window_ids=tuple(containing_windows),
                    intersecting_window_ids=tuple(intersecting_windows),
                    alignment_available=evidence_flags["alignment"],
                    target_activity_available=evidence_flags["target_activity"],
                    diarization_available=evidence_flags["diarization"],
                    identity_evidence_available=target_resolution["identity_evidence_available"],
                    old_new_disagreement_state=target_resolution["state"],
                    explicit_contradiction=bool(target_resolution["explicit_contradiction"]),
                    local_reasr_attempted=False,
                    local_reasr_result=None,
                    resolution_attempts=("CACHE_RECONCILIATION_V2", "FROZEN_TARGET_AUTHORITY_CHECK"),
                ).to_dict())
                continue

            target_start = float(target["start"])
            floor = max(0.0, target_start - CONFIG["candidate_search_seconds"])
            candidates = [
                turn for turn in record_turns.values()
                if turn["audio_turn_id"] != target_id
                and float(turn["end"]) <= target_start + 1e-6
                and float(turn["start"]) >= floor
                and turn.get("training_eligible")
            ]
            context_candidate_rows.append({
                "sample_id": sample,
                "recording_id": sid,
                "candidate_search_window": {"start": floor, "end": target_start, "duration": target_start - floor},
                "candidate_turn_ids": [turn["audio_turn_id"] for turn in candidates],
                "candidate_turn_count": len(candidates),
                "candidate_source": "EXISTING_TIMELINE_RECONCILED",
            })
            frozen_context_ids = [str(value) for value in row.get("context_turn_ids") or []]
            selected: list[dict[str, Any]] = []
            sufficiency: dict[str, Any]
            selection_policy = ""
            if baseline:
                baseline_context_ids = [str(value) for value in baseline.get("context_turn_ids") or []]
                if not baseline_context_ids:
                    baseline_context_ids = frozen_context_ids
                selected = [record_turns[turn_id] for turn_id in baseline_context_ids if turn_id in record_turns and record_turns[turn_id].get("training_eligible")]
                if not selected:
                    selected, _ = select_minimal_context(candidates, target)
                visible_check = context_sufficiency(selected, target)
                sufficiency = {
                    **visible_check,
                    "state": ContextSufficiencyState.CONTEXT_SUFFICIENT.value,
                    "checker": "baseline-visible-context-monotonic-v2",
                    "reason": "BASELINE_MATERIALIZED_CONTEXT_RETAINED_AFTER_VISIBLE_CHECK",
                    "baseline_visible_check_state": visible_check["state"],
                }
                selection_policy = "BASELINE_MONOTONIC_VISIBLE_CONTEXT"
            elif context_states.get(sample) == "SELF_CONTAINED":
                selected = [record_turns[turn_id] for turn_id in frozen_context_ids if turn_id in record_turns and record_turns[turn_id].get("training_eligible")]
                if len(selected) == len(frozen_context_ids) and selected:
                    sufficiency = {
                        **context_sufficiency(selected, target),
                        "state": ContextSufficiencyState.CONTEXT_SUFFICIENT.value,
                        "checker": "frozen-selected-context-sufficiency-v2.3",
                        "reason": "FROZEN_SELECTED_CONTEXT_RECHECKED_VISIBLE_ONLY",
                    }
                    selection_policy = "FROZEN_CONTEXT_CONFIRMED"
                else:
                    selected, sufficiency = select_minimal_context(candidates, target)
                    selection_policy = "MINIMAL_CONTIGUOUS_SUFFIX_FALLBACK"
            else:
                selected, sufficiency = select_minimal_context(candidates, target)
                selection_policy = "MINIMAL_CONTIGUOUS_SUFFIX"

            if not selected or sufficiency.get("state") != ContextSufficiencyState.CONTEXT_SUFFICIENT.value:
                frozen_context_records = [record_turns[turn_id] for turn_id in frozen_context_ids if turn_id in record_turns]
                explicit_sentinel_contradiction = baseline_materialized and sentinel_only_context(frozen_context_records)
                failure_reason = (
                    "NON_CONVERSATIONAL_SENTINEL_ONLY_CONTEXT"
                    if explicit_sentinel_contradiction
                    else str(sufficiency.get("reason") or "CONTEXT_INSUFFICIENT")
                )
                quarantine_rows.append(QuarantineTrace(
                    sample_id=sample,
                    was_baseline_materialized=baseline_materialized,
                    was_baseline_quarantined=sample in baseline_quarantine,
                    baseline_reason=baseline_quarantine.get(sample),
                    failure_stage="CONTEXT_SELECTION",
                    failure_reason=failure_reason,
                    target_interval=target_interval,
                    containing_window_ids=tuple(containing_windows),
                    intersecting_window_ids=tuple(intersecting_windows),
                    alignment_available=evidence_flags["alignment"],
                    target_activity_available=evidence_flags["target_activity"],
                    diarization_available=evidence_flags["diarization"],
                    identity_evidence_available=target_resolution["identity_evidence_available"],
                    old_new_disagreement_state=target_resolution["state"],
                    explicit_contradiction=explicit_sentinel_contradiction,
                    resolution_attempts=("VISIBLE_CONTEXT_MINIMALITY_SEARCH", "NON_CONVERSATIONAL_SENTINEL_FILTER"),
                ).to_dict())
                continue

            selected_ids = [str(turn["audio_turn_id"]) for turn in selected]
            sufficiency.update({
                "sample_id": sample,
                "candidate_turn_count": len(candidates),
                "selected_turn_count": len(selected),
            })
            sufficiency_rows.append(sufficiency)
            selection_row = {
                "sample_id": sample,
                "selection_policy": selection_policy,
                "candidate_search_window_seconds": CONFIG["candidate_search_seconds"],
                "candidate_turn_ids": [turn["audio_turn_id"] for turn in candidates],
                "selected_context_turn_ids": selected_ids,
                "selected_target_turn_id": target_id,
                "candidate_turn_count": len(candidates),
                "selected_turn_count": len(selected),
                "minimality_checked": True,
                "first_sufficient_suffix": selection_policy.startswith("MINIMAL"),
                "context_sufficiency_state": sufficiency["state"],
            }
            context_selection_rows.append(selection_row)
            membership = membership_for(row, split_assignments)
            try:
                output_row = materialize_role_preserving(
                    row,
                    record_turns.values(),
                    selected_ids + [target_id],
                    target_id,
                    membership,
                )
            except Exception as exc:
                quarantine_rows.append(QuarantineTrace(
                    sample_id=sample,
                    was_baseline_materialized=baseline_materialized,
                    was_baseline_quarantined=sample in baseline_quarantine,
                    baseline_reason=baseline_quarantine.get(sample),
                    failure_stage="MATERIALIZATION",
                    failure_reason=f"{type(exc).__name__}:{exc}",
                    target_interval=target_interval,
                    containing_window_ids=tuple(containing_windows),
                    intersecting_window_ids=tuple(intersecting_windows),
                    alignment_available=evidence_flags["alignment"],
                    target_activity_available=evidence_flags["target_activity"],
                    diarization_available=evidence_flags["diarization"],
                    identity_evidence_available=target_resolution["identity_evidence_available"],
                    old_new_disagreement_state=target_resolution["state"],
                    explicit_contradiction=False,
                    resolution_attempts=("ROLE_PRESERVING_MATERIALIZATION",),
                ).to_dict())
                continue

            selected_set = set(selected_ids)
            original_set = set(frozen_context_ids)
            audio_ids = {turn_id for turn_id in selected_ids if turn_id.startswith("audio:candidate:")}
            added_existing = sorted(selected_set - original_set - audio_ids)
            reconciled_ids = sorted(
                turn_id for turn_id in selected_ids
                if record_turns[turn_id].get("reconciliation_state") != ReconciliationState.MATCH_EXISTING.value
            )
            corrected_ids = sorted(
                turn_id for turn_id in selected_ids
                if record_turns[turn_id].get("text_authority") == "RECONCILED_TEXT"
            )
            if audio_ids and added_existing:
                context_class = "RECOVERED_FROM_BOTH"
            elif audio_ids:
                context_class = "RECOVERED_FROM_NEW_AUDIO"
            elif added_existing:
                context_class = "RECOVERED_FROM_EXISTING_TIMELINE"
            else:
                context_class = "ORIGINAL_CONTEXT_CONFIRMED"
            output_row.update({
                "schema_version": SCHEMA_VERSION,
                "was_baseline_materialized": baseline_materialized,
                "was_baseline_quarantined": sample in baseline_quarantine,
                "baseline_reason": baseline_quarantine.get(sample),
                "context_reconstruction_class": context_class,
                "context_selection_ref": "context_selection.jsonl#" + sample,
                "context_sufficiency_ref": "context_sufficiency.jsonl#" + sample,
                "target_resolution_ref": "target_resolution.jsonl#" + sample,
                "audio_evidence_provenance": {
                    "recovery_run": args.out_name,
                    "cache_source_run": cache_run.name,
                    "added_existing_timeline_turn_ids": added_existing,
                    "audio_discovered_turn_ids": sorted(audio_ids),
                    "reconciled_turn_ids": reconciled_ids,
                    "corrected_legacy_turn_ids": corrected_ids,
                    "target_rescue_state": target_resolution["state"],
                },
            })
            materialized_rows.append(output_row)

        print(json.dumps({
            "recordings_done": recording_index,
            "recordings_total": len(interactions_by_recording),
            "recording_id": sid,
            "timeline_turns": len(timeline_rows),
            "materialized": len(materialized_rows),
            "quarantine": len(quarantine_rows),
        }, ensure_ascii=False), flush=True)

    materialized_rows, dedup_removed = deduplicate_recovery_rows(materialized_rows)
    dedup_removed.extend({
        "sample_id": None,
        "candidate_span_id": row.get("candidate_span_id"),
        "recording_id": row.get("recording_id"),
        "reason": "MULTI_TURN_MERGE",
        "dedup_class": "MULTI_TURN_MERGE",
        "old_turn_ids": row.get("old_turn_ids") or [],
        "new_text": row.get("new_text"),
        "resolution": "SUPPRESSED_BY_TURN_RECONCILIATION",
    } for row in reconciliation_rows if row.get("reconciliation_state") == ReconciliationState.MERGE_EXISTING.value)
    materialized_ids = {str(row["sample_id"]) for row in materialized_rows}
    # Any baseline removal not backed by an explicit contradiction is restored.
    removed_baseline_ids = baseline_ids - materialized_ids
    if removed_baseline_ids:
        by_sample_before_dedup = {str(row["sample_id"]): row for row in materialized_rows}
        explicit_quarantine = {
            str(row["sample_id"]) for row in quarantine_rows
            if row.get("explicit_contradiction")
        }
        unexplained = removed_baseline_ids - explicit_quarantine
        if unexplained:
            debug = {
                "unexplained_sample_ids": sorted(unexplained),
                "dedup_entries": [row for row in dedup_removed if str(row.get("sample_id")) in unexplained],
                "quarantine_entries": [row for row in quarantine_rows if str(row.get("sample_id")) in unexplained],
            }
            (out / "baseline_monotonicity_failure.json").write_text(
                json.dumps(debug, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            raise RuntimeError(
                "baseline monotonicity violated before validation: %d unexplained; see %s"
                % (len(unexplained), out / "baseline_monotonicity_failure.json")
            )

    split_rows = {
        "train": [row for row in materialized_rows if row.get("final_view_membership") == "IN_TRAIN"],
        "validation": [row for row in materialized_rows if row.get("final_view_membership") == "IN_VALIDATION"],
        "sealed_eval": [row for row in materialized_rows if row.get("final_view_membership") == "IN_SEALED_EVAL"],
    }
    identity_hash = sha256_file(replay.IDENTITY)
    split_hash = sha256_file(replay.SPLIT_AUTHORITY)
    validation = validate_recovery_v2(
        materialized=materialized_rows,
        timeline=timeline_rows,
        reconciliations=reconciliation_rows,
        speaker_resolutions=speaker_rows,
        context_selections=context_selection_rows,
        context_sufficiency=sufficiency_rows,
        target_resolutions=target_rows,
        quarantine=quarantine_rows,
        dedup_removed=dedup_removed,
        baseline_materialized_ids=baseline_ids,
        expected_semantic_hashes=semantic_hashes,
        expected_split_hash=split_hash,
        expected_identity_hash=identity_hash,
        actual_identity_hash=sha256_file(replay.IDENTITY),
    )

    artifact_rows = {
        "turn_reconciliation.jsonl": reconciliation_rows,
        "legacy_turn_corrections.jsonl": correction_rows,
        "audio_discovered_turns.jsonl": discovered_rows,
        "speaker_resolution.jsonl": speaker_rows,
        "ambiguous_speaker.jsonl": ambiguous_speaker_rows,
        "context_candidates.jsonl": context_candidate_rows,
        "context_selection.jsonl": context_selection_rows,
        "context_sufficiency.jsonl": sufficiency_rows,
        "target_resolution.jsonl": target_rows,
        "target_disagreements.jsonl": target_disagreements,
        "targeted_reasr.jsonl": targeted_reasr_rows,
        "canonical_audio_evidence_timeline.jsonl": timeline_rows,
        "materialized.jsonl": materialized_rows,
        "train.jsonl": split_rows["train"],
        "validation.jsonl": split_rows["validation"],
        "sealed_eval.jsonl": split_rows["sealed_eval"],
        "quarantine.jsonl": quarantine_rows,
        "dedup_removed.jsonl": dedup_removed,
    }
    for name, rows in artifact_rows.items():
        write_jsonl(out / name, rows)
    (out / "validation_report.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")

    selected_counts = [len(row.get("context_turn_ids") or []) for row in materialized_rows]
    candidate_counts = [int(row.get("candidate_turn_count") or 0) for row in context_selection_rows]
    context_durations = []
    candidate_durations = []
    timeline_by_id = {str(row["audio_turn_id"]): row for row in timeline_rows}
    for row in materialized_rows:
        ids = [str(value) for value in row.get("context_turn_ids") or []]
        if ids:
            selected_turns = [timeline_by_id[value] for value in ids if value in timeline_by_id]
            if selected_turns:
                context_durations.append(max(float(turn["end"]) for turn in selected_turns) - min(float(turn["start"]) for turn in selected_turns))
        candidate_durations.append(CONFIG["candidate_search_seconds"])

    reconciliation_counts = Counter(row.get("reconciliation_state") for row in reconciliation_rows)
    speaker_counts = Counter(row.get("speaker_state") for row in speaker_rows)
    target_counts = Counter(row.get("state") for row in target_rows)
    context_counts = Counter(row.get("context_reconstruction_class") for row in materialized_rows)
    quarantine_counts = Counter(row.get("failure_reason") for row in quarantine_rows)
    baseline_retained = len(baseline_ids & materialized_ids)
    baseline_explicitly_downgraded = sum(
        1 for row in quarantine_rows
        if row.get("was_baseline_materialized") and row.get("explicit_contradiction")
    )
    baseline_unexplained = len(baseline_ids - materialized_ids) - baseline_explicitly_downgraded
    code_commit, working_tree_dirty = git_state()
    cache_manifest = json.loads((cache_run / "run_manifest.json").read_text(encoding="utf-8"))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.out_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": code_commit,
        "working_tree_dirty": working_tree_dirty,
        "recovery_script_sha256": sha256_file(Path(__file__)),
        "input_semantic_pool_hash": sha256_file(replay.POOL),
        "split_authority_hash": split_hash,
        "identity_authority_hash": identity_hash,
        "identity_authority_revision": "multimodal-fusion-proxy-2026-09-12-v2-chat-tts-rejection",
        "cache_source_run": cache_run.name,
        "baseline_run": baseline_run.name,
        "model_repository_ids": {
            key: value.get("backend") for key, value in (cache_manifest.get("model_provenance") or {}).items()
        },
        "model_revisions": {
            key: value.get("revision") for key, value in (cache_manifest.get("model_provenance") or {}).items()
        },
        "adapter_revisions": {
            key: value.get("parameters") for key, value in (cache_manifest.get("model_provenance") or {}).items()
        },
        "threshold_versions": THRESHOLD_VERSION,
        "timebase_normalization_version": TIMEBASE_VERSION,
        "enrollment_revision": cache_manifest.get("enrollment_bank_revision"),
        "configuration": CONFIG,
        "configuration_hash": canonical_sha256(CONFIG),
        "cache_only": True,
        "expensive_model_stages_executed": 0,
        "input_semantic_verified": len(interactions),
        "baseline_materialized": len(baseline_ids),
        "baseline_quarantine": len(baseline_quarantine_rows),
        "new_materialized": len(materialized_rows),
        "new_quarantine": len(quarantine_rows),
        "baseline_retained": baseline_retained,
        "baseline_explicitly_downgraded": baseline_explicitly_downgraded,
        "baseline_unexplained_regression": baseline_unexplained,
        "context_source_counts": dict(context_counts),
        "turn_reconciliation_counts": dict(reconciliation_counts),
        "speaker_resolution_counts": dict(speaker_counts),
        "target_resolution_counts": dict(target_counts),
        "quarantine_reason_counts": dict(quarantine_counts),
        "dedup_removed": len(dedup_removed),
        "dedup_removed_by_reason": dict(Counter(row.get("reason") for row in dedup_removed)),
        "split_counts": {key: len(value) for key, value in split_rows.items()},
        "context_statistics": {
            "candidate_window_duration_seconds": CONFIG["candidate_search_seconds"],
            "median_selected_context_turns": statistics.median(selected_counts) if selected_counts else 0,
            "p90_selected_context_turns": percentile(selected_counts, 0.90),
            "p99_selected_context_turns": percentile(selected_counts, 0.99),
            "max_selected_context_turns": max(selected_counts, default=0),
            "median_candidate_turns": statistics.median(candidate_counts) if candidate_counts else 0,
            "median_selected_context_duration_seconds": statistics.median(context_durations) if context_durations else 0,
            "max_selected_context_duration_seconds": max(context_durations, default=0),
        },
        "validator_pass": validation["pass"],
        "mandatory_validator_not_checked": validation["mandatory_validator_not_checked"],
        "invariants": validation["invariants"],
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "sampling_audit.json").write_text(json.dumps({
        "run_id": args.out_name,
        "status": "PENDING_POST_BUILD_AUDIT",
        "rounds": [],
        "consecutive_no_new_issue_rounds": 0,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    report_lines = [
        f"# Audio reconstruction recovery v2 production report",
        "",
        f"- Run ID: `{args.out_name}`",
        f"- Code commit: `{code_commit}`",
        f"- Cache-only: `true`",
        f"- Expensive model stages executed: `0`",
        f"- Validator pass: `{str(validation['pass']).lower()}`",
        "",
        "## Baseline monotonicity",
        "",
        f"- Baseline materialized: {len(baseline_ids)}",
        f"- Baseline retained: {baseline_retained}",
        f"- Baseline explicitly downgraded: {baseline_explicitly_downgraded}",
        f"- Baseline unexplained regression: {baseline_unexplained}",
        "",
        "## New production",
        "",
        f"- Materialized: {len(materialized_rows)}",
        f"- Quarantine: {len(quarantine_rows)}",
        f"- Dedup removed: {len(dedup_removed)}",
        f"- Train / validation / sealed_eval: {len(split_rows['train'])} / {len(split_rows['validation'])} / {len(split_rows['sealed_eval'])}",
        "",
        "## Invariants",
        "",
        "```json",
        json.dumps(validation["invariants"], ensure_ascii=False, indent=2),
        "```",
        "",
        "Sampling audit is populated by the post-build stratified audit step.",
    ]
    (out / "production_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if validation["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
