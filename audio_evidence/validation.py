from __future__ import annotations

from collections import Counter, defaultdict
import re
from typing import Any, Dict, Iterable, List, Optional

from .contracts import AUDIO_EVIDENCE_SCHEMA_VERSION, TIMEBASE_NORMALIZATION_VERSION, Timebase, canonical_sha256
from .enrollment import EnrollmentBank, VerificationStatus


GATE_NAMES = (
    "ENROLLMENT_SOURCE_CONFIRMED", "ENROLLMENT_PROVENANCE_COMPLETE", "NO_UNVERIFIED_ENROLLMENT",
    "NO_CIRCULAR_ENROLLMENT_BOOTSTRAP", "ENROLLMENT_NO_UNRESOLVED_OVERLAP", "AUDIO_WINDOW_PROVENANCE",
    "AUDIO_INTERVAL_NO_CROSS_RECORDING", "TIMESTAMP_TIMEBASE_CONSISTENT", "TARGET_ACTIVITY_EVIDENCE_TRACEABLE",
    "DIARIZATION_TRACEABLE", "TSE_SOURCE_TRACEABLE", "ASR_SOURCE_TRACEABLE", "ALIGNMENT_SOURCE_TRACEABLE",
    "TRANSCRIPT_DISAGREEMENT_RECORDED", "TEXT_AUTHORITY_RESOLVED", "NO_SILENT_TRANSCRIPT_OVERWRITE", "ROLE_PRESERVING_CONTEXT",
    "HISTORICAL_ASSISTANT_MASKED", "HISTORICAL_ASSISTANT_LOSS_LEAKAGE", "TARGET_REUSE", "PREFIX_LADDER",
    "NO_SYNTHETIC_PROMPT", "TRAIN_AUTHORITY_CONSISTENT", "SPLIT_AUTHORITY_PRESERVED", "SEMANTIC_TRUTH_UNCHANGED",
)


def _fail(gates: Dict[str, str], errors: List[Dict[str, Any]], gate: str, sample: Any, reason: str) -> None:
    gates[gate] = "FAIL"
    errors.append({"gate": gate, "sample_id": sample, "reason": reason})


def validate_artifacts(
    bank: EnrollmentBank,
    windows: Iterable[Dict[str, Any]],
    timeline: Iterable[Dict[str, Any]],
    materialized: Iterable[Dict[str, Any]],
    *,
    expected_semantic_hashes: Optional[Dict[str, str]] = None,
    expected_split_hash: Optional[str] = None,
    allow_synthetic: bool = False,
) -> Dict[str, Any]:
    windows = list(windows)
    timeline = list(timeline)
    materialized = list(materialized)
    gates = {name: "PASS" for name in GATE_NAMES}
    errors: List[Dict[str, Any]] = []
    if expected_semantic_hashes is None:
        gates["SEMANTIC_TRUTH_UNCHANGED"] = "NOT_CHECKED"
        errors.append({"gate": "SEMANTIC_TRUTH_UNCHANGED", "sample_id": None, "reason": "expected semantic authority hashes were not supplied"})
    if expected_split_hash is None:
        gates["SPLIT_AUTHORITY_PRESERVED"] = "NOT_CHECKED"
        errors.append({"gate": "SPLIT_AUTHORITY_PRESERVED", "sample_id": None, "reason": "expected split authority hash was not supplied"})
    if not bank.references:
        _fail(gates, errors, "ENROLLMENT_SOURCE_CONFIRMED", None, "enrollment bank is empty")
    for reference in bank.references.values():
        sample = reference.enrollment_id
        if reference.verification_status != VerificationStatus.CONFIRMED_TARGET:
            _fail(gates, errors, "NO_UNVERIFIED_ENROLLMENT", sample, "bank contains non-confirmed reference")
        if reference.synthetic_test_only and not allow_synthetic:
            _fail(gates, errors, "ENROLLMENT_SOURCE_CONFIRMED", sample, "synthetic reference is forbidden outside tests")
        if not reference.source_provenance or not reference.verification_evidence or not reference.checksum:
            _fail(gates, errors, "ENROLLMENT_PROVENANCE_COMPLETE", sample, "missing enrollment trace")
        if reference.overlap_status != "NO_OVERLAP_CONFIRMED":
            _fail(gates, errors, "ENROLLMENT_NO_UNRESOLVED_OVERLAP", sample, reference.overlap_status)
        if any(item.authority_domain.upper() in {"TARGET_ACTIVITY", "PVAD", "NEW_DIARIZATION", "TSE", "NEW_ASR", "SEMANTIC_CONTENT", "PERSONA_STYLE", "ENROLLMENT_SIMILARITY"} for item in reference.verification_evidence):
            _fail(gates, errors, "NO_CIRCULAR_ENROLLMENT_BOOTSTRAP", sample, "new-pipeline evidence appears in enrollment trust")
    if materialized and not windows:
        _fail(gates, errors, "AUDIO_WINDOW_PROVENANCE", None, "materialized rows have no window artifact")
    if materialized and not timeline:
        _fail(gates, errors, "ROLE_PRESERVING_CONTEXT", None, "materialized rows have no evidence timeline")
    window_recordings = {}
    for window in windows:
        sample = window.get("window_id")
        if not window.get("source_audio") or not window.get("audio_checksum") or not window.get("recording_id") or not window.get("canonical_recording_id") or not window.get("recording_family_id") or not window.get("merged_interval"):
            _fail(gates, errors, "AUDIO_WINDOW_PROVENANCE", sample, "missing source mapping")
        elif len(str(window.get("audio_checksum"))) != 64:
            _fail(gates, errors, "AUDIO_WINDOW_PROVENANCE", sample, "source audio checksum is not SHA-256")
        window_recordings[str(sample)] = (str(window.get("recording_id")), str(window.get("canonical_recording_id")), str(window.get("recording_family_id")))
        interval = window.get("merged_interval") or {}
        if interval.get("timebase") != Timebase.RECORDING_SECONDS.value:
            _fail(gates, errors, "TIMESTAMP_TIMEBASE_CONSISTENT", sample, "unknown timebase")
        try:
            if float(interval.get("end")) <= float(interval.get("start")):
                raise ValueError
        except (TypeError, ValueError):
            _fail(gates, errors, "AUDIO_INTERVAL_NO_CROSS_RECORDING", sample, "invalid/reversed interval")
    evidence_turn_ids = set()
    timeline_by_id = {}
    for turn in timeline:
        sample = turn.get("audio_turn_id")
        evidence_turn_ids.add(str(sample))
        timeline_by_id[str(sample)] = turn
        if turn.get("schema_version") != AUDIO_EVIDENCE_SCHEMA_VERSION:
            _fail(gates, errors, "AUDIO_WINDOW_PROVENANCE", sample, "timeline schema mismatch")
        if turn.get("timebase") != Timebase.RECORDING_SECONDS.value:
            _fail(gates, errors, "TIMESTAMP_TIMEBASE_CONSISTENT", sample, "timeline timebase unknown")
        try:
            if float(turn.get("end")) <= float(turn.get("start")):
                raise ValueError
        except (TypeError, ValueError):
            _fail(gates, errors, "TIMESTAMP_TIMEBASE_CONSISTENT", sample, "timeline interval is missing or reversed")
        source_interval = turn.get("source_interval") or {}
        try:
            inside_source = source_interval.get("timebase") == Timebase.RECORDING_SECONDS.value and float(source_interval["start"]) <= float(turn["start"]) < float(turn["end"]) <= float(source_interval["end"])
        except (KeyError, TypeError, ValueError):
            inside_source = False
        if not inside_source:
            _fail(gates, errors, "TIMESTAMP_TIMEBASE_CONSISTENT", sample, "timeline turn is outside or detached from its bounded source interval")
        for evidence_field in ("target_activity_evidence", "diarization_evidence", "alignment"):
            for item in turn.get(evidence_field) or []:
                try:
                    item_inside = (
                        item.get("timebase") == Timebase.RECORDING_SECONDS.value
                        and float(source_interval["start"]) <= float(item["start"]) < float(item["end"]) <= float(source_interval["end"])
                    )
                    source_timebase = Timebase(str(item["source_timebase"]))
                    conversion = item["timebase_conversion"]
                    expected_offset = float(source_interval["start"]) if source_timebase == Timebase.WINDOW_LOCAL_SECONDS else 0.0
                    conversion_valid = conversion.get("version") == TIMEBASE_NORMALIZATION_VERSION and float(conversion.get("offset_seconds")) == expected_offset
                except (KeyError, TypeError, ValueError):
                    item_inside = False
                    conversion_valid = False
                if not item_inside or not conversion_valid:
                    _fail(gates, errors, "TIMESTAMP_TIMEBASE_CONSISTENT", sample, "%s is not normalized and bounded in recording-global time" % evidence_field)
        turn_authority = (str(turn.get("recording_id")), str(turn.get("canonical_recording_id")), str(turn.get("recording_family_id")))
        if turn.get("window_id") and window_recordings.get(str(turn.get("window_id"))) != turn_authority:
            _fail(gates, errors, "AUDIO_INTERVAL_NO_CROSS_RECORDING", sample, "window and turn recording disagree")
        if turn.get("target_activity_evidence") and not turn.get("enrollment_provenance"):
            _fail(gates, errors, "TARGET_ACTIVITY_EVIDENCE_TRACEABLE", sample, "activity lacks enrollment trace")
        if turn.get("speaker_cluster") and not (turn.get("model_provenance") or {}).get("diarization"):
            _fail(gates, errors, "DIARIZATION_TRACEABLE", sample, "cluster lacks model trace")
        if turn.get("optional_tse_asr_hypothesis"):
            tse_enrollment = (turn.get("tse_source") or {}).get("enrollment_id")
            if not tse_enrollment or tse_enrollment not in bank.references:
                _fail(gates, errors, "TSE_SOURCE_TRACEABLE", sample, "TSE result lacks confirmed enrollment trace")
        if turn.get("new_asr_hypothesis") is not None and not (turn.get("asr_source") or {}).get("audio_checksum"):
            _fail(gates, errors, "ASR_SOURCE_TRACEABLE", sample, "ASR lacks audio provenance")
        if turn.get("alignment") and not turn.get("new_asr_hypothesis"):
            _fail(gates, errors, "ALIGNMENT_SOURCE_TRACEABLE", sample, "alignment has no source transcript")
        if turn.get("new_asr_hypothesis") is not None and not turn.get("disagreement_flags"):
            _fail(gates, errors, "TRANSCRIPT_DISAGREEMENT_RECORDED", sample, "old/new comparison absent")
        if turn.get("text_resolution_state") not in {"RESOLVED", "UNRESOLVED"}:
            _fail(gates, errors, "TEXT_AUTHORITY_RESOLVED", sample, "text resolution state missing")
        if turn.get("text_resolution_state") == "RESOLVED" and (not turn.get("text_authority") or not turn.get("resolved_text") or not turn.get("text_resolution_provenance")):
            _fail(gates, errors, "TEXT_AUTHORITY_RESOLVED", sample, "resolved text lacks authority/provenance")
        if turn.get("old_transcript") and turn.get("old_transcript_overwritten"):
            _fail(gates, errors, "NO_SILENT_TRANSCRIPT_OVERWRITE", sample, "old transcript was mutated")
    supervised_targets = defaultdict(list)
    prompt_signatures = defaultdict(list)
    interaction_keys = defaultdict(list)
    for row in materialized:
        sample = str(row.get("sample_id"))
        messages = row.get("messages") or []
        if not messages or any(message.get("role") not in {"user", "assistant"} for message in messages):
            _fail(gates, errors, "ROLE_PRESERVING_CONTEXT", sample, "invalid role sequence")
        if any(not message.get("source_turn_ids") or (evidence_turn_ids and any(str(turn_id) not in evidence_turn_ids for turn_id in message.get("source_turn_ids") or [])) for message in messages):
            _fail(gates, errors, "ROLE_PRESERVING_CONTEXT", sample, "message is not traceable to timeline turns")
        for message in messages:
            source_turns = [timeline_by_id.get(str(turn_id)) for turn_id in message.get("source_turn_ids") or []]
            source_turns = [turn for turn in source_turns if turn is not None]
            if source_turns and any(turn.get("text_resolution_state") != "RESOLVED" for turn in source_turns):
                _fail(gates, errors, "TEXT_AUTHORITY_RESOLVED", sample, "materialized message references unresolved text evidence")
            if source_turns and str(message.get("content") or "") != "\n".join(str(turn.get("resolved_text") or "") for turn in source_turns):
                _fail(gates, errors, "TEXT_AUTHORITY_RESOLVED", sample, "materialized text differs from resolved timeline authority")
            if source_turns and any(
                str(turn.get("recording_id")) != str(row.get("recording_id"))
                or str(turn.get("canonical_recording_id")) != str(row.get("canonical_recording_id"))
                or str(turn.get("recording_family_id")) != str(row.get("recording_family_id"))
                for turn in source_turns
            ):
                _fail(gates, errors, "ROLE_PRESERVING_CONTEXT", sample, "materialized message crosses recording authority")
        supervised = [message for message in messages if message.get("supervise")]
        if len(supervised) != (0 if row.get("supervision_state") == "NO_SUPERVISION" else 1) or (supervised and supervised[0] is not messages[-1]):
            _fail(gates, errors, "HISTORICAL_ASSISTANT_LOSS_LEAKAGE", sample, "supervision is not restricted to final target")
        if any(message.get("role") == "assistant" and message is not messages[-1] and message.get("supervise") is not False for message in messages):
            _fail(gates, errors, "HISTORICAL_ASSISTANT_MASKED", sample, "historical assistant is not explicitly masked")
        if row.get("synthetic_prompt") is not False:
            _fail(gates, errors, "NO_SYNTHETIC_PROMPT", sample, "synthetic prompt marker present")
        target = str(row.get("target_turn_id") or "")
        if supervised:
            supervised_targets[(row.get("canonical_recording_id") or row.get("recording_id"), target)].append(sample)
        interaction_keys[str(row.get("interaction_dedup_key") or "")].append(sample)
        target_text = str(supervised[0].get("content") or "") if supervised else ""
        prompt_signatures[(row.get("recording_id"), tuple(row.get("context_turn_ids") or []))].append((sample, target, target_text))
        if row.get("final_view_membership_authority") != "FINAL_VIEW_MEMBERSHIP" or row.get("legacy_field_semantics") != "SOURCE_SAMPLING_ELIGIBILITY_ONLY":
            _fail(gates, errors, "TRAIN_AUTHORITY_CONSISTENT", sample, "membership authority/legacy semantics ambiguous")
        if not row.get("semantic_truth_ref") or not row.get("semantic_truth_sha256"):
            _fail(gates, errors, "SEMANTIC_TRUTH_UNCHANGED", sample, "semantic authority reference/hash missing")
        elif expected_semantic_hashes is not None and row.get("semantic_truth_sha256") != expected_semantic_hashes.get(sample):
            _fail(gates, errors, "SEMANTIC_TRUTH_UNCHANGED", sample, "semantic truth reference changed")
        if not row.get("split_authority_ref") or not row.get("split_authority_sha256"):
            _fail(gates, errors, "SPLIT_AUTHORITY_PRESERVED", sample, "split authority reference/hash missing")
        elif expected_split_hash is not None and row.get("split_authority_sha256") != expected_split_hash:
            _fail(gates, errors, "SPLIT_AUTHORITY_PRESERVED", sample, "split authority reference changed")
    for key, samples in supervised_targets.items():
        if len(samples) > 1:
            _fail(gates, errors, "TARGET_REUSE", samples, "same recording turn supervised more than once")
    for key, samples in interaction_keys.items():
        if key and len(samples) > 1:
            _fail(gates, errors, "TARGET_REUSE", samples, "same canonical interaction/provenance/context/target duplicated")
    # Prefix reuse is evaluated only among supervised targets sharing the same
    # interaction context; masked historical assistant text is intentionally ignored.
    for _, values in prompt_signatures.items():
        for index, (left_sample, left_target, left_text) in enumerate(values):
            left = re.findall(r"[a-z0-9']+", left_text.lower())
            for right_sample, right_target, right_text in values[index + 1:]:
                right = re.findall(r"[a-z0-9']+", right_text.lower())
                containment = left != right and ((left and len(left) <= len(right) and any(right[i:i + len(left)] == left for i in range(len(right) - len(left) + 1))) or (right and len(right) <= len(left) and any(left[i:i + len(right)] == right for i in range(len(left) - len(right) + 1))))
                if left_target == right_target or containment:
                    _fail(gates, errors, "PREFIX_LADDER", [left_sample, right_sample], "same-context supervised target ladder")
    return {"schema_version": "1.0.0", "gates": gates, "errors": errors, "pass": all(value == "PASS" for value in gates.values()), "counts": {"enrollment": len(bank.references), "windows": len(windows), "timeline_turns": len(evidence_turn_ids), "materialized": len(materialized)}}
