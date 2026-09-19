from __future__ import annotations

"""Cache-only diagnosis of the audio completion gap.

This tool never imports a model adapter and never decodes audio.  It reads the
frozen inputs plus the content-addressed evidence cache of a completed
window-inference run and reports:

* exact cache-key presence per stage and window (cache-hit proof);
* the real per-turn new-ASR text reconstructed from cached alignment evidence;
* why materialization fails today vs. what would resolve under the v2.3
  frozen-verified-target text authority;
* the resulting target disagreement distribution.

Usage:
    python scripts/diagnose_audio_completion_gap.py [--run-id RUN] [--limit N]
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audio_evidence.cache import EvidenceCache  # noqa: E402
from audio_evidence.contracts import TIMEBASE_NORMALIZATION_VERSION, Timebase  # noqa: E402
from audio_evidence.planning import AudioWindowPlanner  # noqa: E402
from audio_evidence.transcript import (  # noqa: E402
    TextResolutionState,
    automatic_text_resolution,
    detect_disagreement,
    resolve_frozen_target_text,
)
import run_audio_evidence_production as replay  # noqa: E402

CORPUS = ROOT.parent.parent
DATASET = CORPUS / "datasets" / "meow_v02_sft_v2_2_semantic_verified"
POOL = DATASET / "verified_interaction_pool_v2_3.jsonl"
SPLIT = DATASET / "split_authority_v2_3.json"
CONTEXT_DECISIONS = DATASET / "context_sufficiency_decisions_v2_3.jsonl"
IDENTITY = CORPUS / "identity_results" / "identity_mapping_multimodal_fusion_proxy.jsonl"
TIMELINE_DIR = CORPUS / "unique_timelines"
MANIFEST = CORPUS / "manifest" / "master_video_manifest.jsonl"
RUN_ROOT = DATASET / "audio_reconstruction_v1"
DEFAULT_RUN = "audio-reconstruction-v1-real-context60-bounded120-20260918"

# Frozen adapter provenance that participates in the cache key.
ACTIVITY_REVISION = "ecapa-target-activity-2026-09-17"
DIARIZATION_REVISION = "ecapa-frame-diarization-2026-09-17"
ASR_REVISION = "Qwen/Qwen3-ASR-1.7B@7278e1e70fe206f11671096ffdd38061171dd6e5"
ALIGNMENT_REVISION = "Qwen/Qwen3-ForcedAligner-0.6B@c7cbfc2048c462b0d63a45797104fc9db3ad62b7"
ACTIVITY_PARAMETERS = {
    "target_activity_threshold": 0.62,
    "target_activity_threshold_version": "ecapa-target-activity-v1",
    "frame_seconds": 1.5,
    "hop_seconds": 0.75,
    "waveform_sample_rate": 16000,
}
ASR_PARAMETERS = {"return_time_stamps": True, "raw_waveform": True, "sample_rate": 16000}
ALIGNMENT_PARAMETERS = {"raw_waveform": True, "sample_rate": 16000}
ENROLLMENT_REVISION = "audio-enrollment-v1-20260917"


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def stage_key(audio_checksum: str, interval: dict, revision: str, parameters: dict, enrollment_revision: str) -> str:
    cache_parameters = dict(parameters)
    cache_parameters["timebase_normalization_version"] = TIMEBASE_NORMALIZATION_VERSION
    return EvidenceCache.key(audio_checksum, dict(interval), revision, cache_parameters, enrollment_revision)


def bound_windows(plan: dict, max_seconds: float = 120.0) -> dict:
    """Split long merged unions so ASR never receives an unbounded interval."""
    windows, mappings = [], []
    by_old = defaultdict(list)
    for mapping in plan.get("sample_window_mappings", []):
        by_old[mapping["window_id"]].append(mapping)
    for old in plan.get("windows", []):
        start = float(old["merged_interval"]["start"])
        end = float(old["merged_interval"]["end"])
        chunks = []
        cursor = start
        while cursor < end - 1e-6:
            chunk_end = min(end, cursor + max_seconds)
            identity = f"{old['window_id']}|{cursor:.6f}|{chunk_end:.6f}|bounded-120s-v1"
            wid = "aw_" + hashlib.sha256(identity.encode()).hexdigest()[:20]
            chunks.append((wid, cursor, chunk_end))
            windows.append({**old, "window_id": wid, "merged_interval": {"start": cursor, "end": chunk_end, "timebase": Timebase.RECORDING_SECONDS.value}, "sample_ids": [], "policy_version": "audio-window-planner-bounded-120s-v1"})
            cursor = chunk_end
        for mapping in by_old.get(old["window_id"], []):
            req = mapping.get("requested_interval") or {}
            midpoint = (float(req.get("start", start)) + float(req.get("end", end))) / 2.0
            chosen = next((x for x in chunks if x[1] <= midpoint <= x[2]), chunks[-1])
            mappings.append({**mapping, "window_id": chosen[0]})
            for window in windows:
                if window["window_id"] == chosen[0]:
                    window["sample_ids"] = sorted(set(window["sample_ids"]) | {mapping["sample_id"]})
    return {"schema_version": "1.0.0", "policy_version": "audio-window-planner-bounded-120s-v1", "windows": windows, "sample_window_mappings": sorted(mappings, key=lambda x: (x["sample_id"], x["window_id"]))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=DEFAULT_RUN)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    run = RUN_ROOT / args.run_id
    cache = EvidenceCache(run / "cache")

    pool = load_jsonl(POOL)
    manifest = {str(r["source_id"]): r for r in load_jsonl(MANIFEST) if r.get("source_id")}
    identity_rows = load_jsonl(IDENTITY)
    identity_map = {(str(r.get("source_id")), str(r.get("cluster"))): r for r in identity_rows}
    context_states = {str(r.get("sample_id")): str(r.get("state")) for r in load_jsonl(CONTEXT_DECISIONS)}
    source_ids = {str(r["source_id"]) for r in pool}
    timelines = {sid: json.loads((TIMELINE_DIR / f"{sid}.json").read_text(encoding="utf-8")) for sid in source_ids if (TIMELINE_DIR / f"{sid}.json").exists()}

    interactions = []
    for row in pool:
        sid = str(row["source_id"])
        meta = manifest.get(sid)
        if not meta or sid not in timelines:
            continue
        audio_path = CORPUS / str(meta.get("audio_path") or f"raw_audio/{sid}.webm")
        checksum = str(meta.get("audio_content_hash") or "")
        if len(checksum) != 64 or not audio_path.exists():
            continue
        row = dict(row)
        row.update({"recording_id": sid, "source_audio": str(audio_path), "source_audio_checksum": checksum})
        interactions.append(row)

    for row in interactions:
        ts = dict(row.get("timestamps") or {})
        start, end = float(ts["start"]), float(ts["end"])
        incomplete = context_states.get(str(row["sample_id"])) == "CONTEXT_INCOMPLETE" or not bool((row.get("semantic_qa") or {}).get("context_complete", True))
        ts["start"] = max(0.0, start - (60.0 if incomplete else 2.0))
        ts["end"] = end + 5.0
        ts["timebase"] = Timebase.RECORDING_SECONDS.value
        row["timestamps"] = ts
        row["context_incomplete_input"] = incomplete

    planner = AudioWindowPlanner(padding_before=0.0, padding_after=0.0, merge_gap=1.0)
    plan = bound_windows(planner.merge([planner.request_from_interaction(x) for x in interactions]), max_seconds=120.0)
    window_lookup = {w["window_id"]: w for w in plan["windows"]}
    window_for_sample = {m["sample_id"]: m["window_id"] for m in plan["sample_window_mappings"]}
    # ---- exact cache-key presence (cache-hit proof, no model execution) ----
    with tempfile.TemporaryDirectory() as scratch:
        replay.OUT = Path(scratch)
        bank = replay.build_enrollment_bank(identity_rows, manifest)
    enrollment_id = sorted(bank.references)[0]
    enrollment = bank.confirmed(enrollment_id)
    enrollment_revision = "%s:%s:%s:%s" % (bank.revision, enrollment.enrollment_id, enrollment.checksum, enrollment.embedding_revision)
    present, missing = Counter(), Counter()
    missing_examples = defaultdict(list)
    expected_asr_keys = {}
    checkpoint_path = run / "checkpoint.json"
    if checkpoint_path.exists():
        for window_id, stages in (json.loads(checkpoint_path.read_text(encoding="utf-8")).get("completed") or {}).items():
            if (stages or {}).get("asr"):
                expected_asr_keys[str(window_id)] = stages["asr"]

    # Per-stage cache-hit proof: recompute every key the pipeline will request
    # and compare it with the key the source run durably recorded.
    checkpoint_document = json.loads(checkpoint_path.read_text(encoding="utf-8")).get("completed") if checkpoint_path.exists() else {}
    key_mismatches = defaultdict(list)
    stage_key_matches = Counter()
    stage_key_absent = Counter()
    for window in plan["windows"]:
        window_id = str(window["window_id"])
        audio_checksum, interval = str(window["audio_checksum"]), dict(window["merged_interval"])
        recorded = checkpoint_document.get(window_id) or {}
        specs = [("target_activity", ACTIVITY_REVISION, ACTIVITY_PARAMETERS), ("diarization", DIARIZATION_REVISION, ACTIVITY_PARAMETERS)]
        if recorded.get("asr"):
            specs.append(("asr", ASR_REVISION, {**ASR_PARAMETERS, "source_audio_checksum": audio_checksum}))
        if recorded.get("alignment"):
            asr_payload = cache.get("asr", recorded["asr"]) if recorded.get("asr") else None
            hypothesis_text = str((asr_payload or {}).get("text") or "")
            specs.append(("alignment", ALIGNMENT_REVISION, {**ALIGNMENT_PARAMETERS, "source_audio_checksum": audio_checksum, "text_sha256": hashlib.sha256(hypothesis_text.encode("utf-8")).hexdigest()}))
        for stage, revision, parameters in specs:
            expected_key = recorded.get(stage)
            computed_key = stage_key(audio_checksum, interval, revision, parameters, enrollment_revision)
            if expected_key is None:
                stage_key_absent[stage] += 1
                continue
            if computed_key != expected_key:
                if len(key_mismatches[stage]) < 3:
                    key_mismatches[stage].append(window_id)
                continue
            if cache.get(stage, computed_key) is None:
                stage_key_absent[stage] += 1
                if len(missing_examples[stage]) < 3:
                    missing_examples[stage].append(window_id)
            else:
                stage_key_matches[stage] += 1
    for window in plan["windows"]:
        audio_checksum, interval = str(window["audio_checksum"]), dict(window["merged_interval"])
        asr_parameters = dict(ASR_PARAMETERS)
        asr_parameters["source_audio_checksum"] = audio_checksum
        key = stage_key(audio_checksum, interval, ASR_REVISION, asr_parameters, enrollment_revision)
        value = cache.get("asr", key)
        if value is None:
            missing["asr"] += 1
            if len(missing_examples["asr"]) < 3:
                missing_examples["asr"].append(window["window_id"])
        else:
            present["asr"] += 1

    report = {
        "run": args.run_id,
        "windows": len(plan["windows"]),
        "interactions": len(interactions),
        "enrollment_revision": enrollment_revision,
        "asr_cache_key_present": present["asr"],
        "asr_cache_key_missing": missing["asr"],
        "asr_cache_key_missing_examples": missing_examples["asr"],
        "stage_key_matches": dict(stage_key_matches),
        "stage_key_recorded_mismatch_examples": {stage: values for stage, values in key_mismatches.items()},
        "stage_key_absent": dict(stage_key_absent),
    }
    if os.environ.get("MEOW_DEBUG_CACHE_KEY"):
        first = plan["windows"][0]
        report["debug_first_window"] = {
            "window_id": first["window_id"],
            "merged_interval": first["merged_interval"],
            "computed_asr_key": stage_key(str(first["audio_checksum"]), dict(first["merged_interval"]), ASR_REVISION, {**ASR_PARAMETERS, "source_audio_checksum": str(first["audio_checksum"])}, enrollment_revision),
            "expected_asr_key": expected_asr_keys.get(first["window_id"]),
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    # ---- real per-turn text reconstruction for the target turn ------------
    windows_by_recording = defaultdict(list)
    for candidate in plan["windows"]:
        windows_by_recording[candidate["recording_id"]].append((float(candidate["merged_interval"]["start"]), float(candidate["merged_interval"]["end"]), candidate))

    def containing_windows(recording_id, start, end):
        for window_start, window_end, candidate in windows_by_recording.get(recording_id, []):
            if window_start <= start and end <= window_end:
                yield candidate

    mappings_by_sample = defaultdict(list)
    for mapping in plan["sample_window_mappings"]:
        mappings_by_sample[mapping["sample_id"]].append(mapping)

    samples = interactions if not args.limit else interactions[: args.limit]
    current_reasons = Counter()
    frozen_outcomes = Counter()
    frozen_disagreements = Counter()
    old_authority = match_new = 0
    unresolved_examples = []
    for row in samples:
        target_ids = [str(x) for x in row.get("target_turn_ids") or []]
        if not target_ids:
            current_reasons["TARGET_TURN_PROVENANCE_MISSING"] += 1
            continue
        target_id = target_ids[-1]
        timeline = timelines.get(row["recording_id"], {})
        target_record = next((t for t in timeline.get("turns", []) if str(t.get("turn_id")) == target_id), None)
        if target_record is None:
            current_reasons["TARGET_TURN_PROVENANCE_MISSING"] += 1
            continue
        window_id = window_for_sample.get(row["sample_id"])
        if window_id is None:
            current_reasons["TARGET_AUDIO_EVIDENCE_UNAVAILABLE"] += 1
            continue
        # Follow the runner's materialization search: the target's evidence may
        # live in another bounded chunk that fully contains the target turn.
        candidate_ids = [window_id] + [other["window_id"] for other in containing_windows(row["recording_id"], float(target_record["timestamp"]["start"]), float(target_record["timestamp"]["end"]))]
        chosen = None
        for candidate_id in candidate_ids:
            candidate = window_lookup[candidate_id]
            window_start = float(candidate["merged_interval"]["start"])
            window_end = float(candidate["merged_interval"]["end"])
            audio_key_candidate = stage_key(str(candidate["audio_checksum"]), dict(candidate["merged_interval"]), ASR_REVISION, {**ASR_PARAMETERS, "source_audio_checksum": str(candidate["audio_checksum"])}, enrollment_revision)
            if cache.get("asr", audio_key_candidate) is not None:
                chosen = (candidate_id, candidate, cache.get("asr", audio_key_candidate))
                break
        if chosen is None:
            current_reasons["TARGET_AUDIO_EVIDENCE_UNAVAILABLE"] += 1
            continue
        window_id, window, asr_payload = chosen
        window_start, window_end = float(window["merged_interval"]["start"]), float(window["merged_interval"]["end"])
        target_start = float(target_record["timestamp"]["start"])
        target_end = min(float(target_record["timestamp"]["end"]), window_end)
        hypothesis_text = str(asr_payload.get("text") or "")
        # Rebuild the window's source turns exactly as the runner does.
        source_turns = []
        for turn in timeline.get("turns", []):
            start, end = float(turn["timestamp"]["start"]), float(turn["timestamp"]["end"])
            if start < window_end and end > window_start:
                source_turns.append({"audio_turn_id": str(turn.get("turn_id")), "start": max(start, window_start), "end": min(end, window_end), "old_transcript": turn.get("text")})
        multiple = len(source_turns) > 1
        target_new = hypothesis_text if not multiple else None
        alignment = None
        if multiple:
            alignment_parameters = dict(ALIGNMENT_PARAMETERS)
            alignment_parameters["source_audio_checksum"] = str(window["audio_checksum"])
            alignment_parameters["text_sha256"] = hashlib.sha256(hypothesis_text.encode("utf-8")).hexdigest()
            alignment_key = stage_key(str(window["audio_checksum"]), dict(window["merged_interval"]), ALIGNMENT_REVISION, alignment_parameters, enrollment_revision)
            alignment = cache.get("alignment", alignment_key)
            if alignment:
                aligned = [str(item.get("text") or "") for item in alignment if target_start <= (float(item["start"]) + float(item["end"])) / 2.0 <= target_end]
                target_new = " ".join(aligned).strip() or None
        target_old = str(target_record.get("text") or "")
        unattributed_target_span = multiple and not target_new
        disagreement = detect_disagreement(target_old, target_new) if target_new is not None else (detect_disagreement(target_old, target_old, boundary_changed=True) if unattributed_target_span else None)
        current = automatic_text_resolution(target_old, target_new, disagreement)
        frozen = resolve_frozen_target_text(target_old, target_new, disagreement, no_new_asr_evidence=False)
        if current["text_resolution_state"] != TextResolutionState.RESOLVED.value:
            current_reasons["TARGET_TEXT_UNRESOLVED"] += 1
        if frozen["text_resolution_state"] == TextResolutionState.RESOLVED.value:
            frozen_outcomes[(frozen.get("text_resolution_provenance") or {}).get("resolver", "?")] += 1
            frozen_disagreements[(disagreement.value if disagreement else "NO_NEW_TEXT")] += 1
            if frozen.get("text_authority") == "OLD_TRANSCRIPT":
                old_authority += 1
            else:
                match_new += 1
        else:
            frozen_outcomes["UNRESOLVED"] += 1
            if len(unresolved_examples) < 20:
                unresolved_examples.append({"sample_id": row["sample_id"], "window_id": window_id, "target_turn_id": target_id, "disagreement": (disagreement.value if disagreement else None), "reason": (frozen.get("text_resolution_provenance") or {}).get("reason"), "old_len": len(target_old), "new_len": len(target_new or "")})

    summary = {
        "current_quarantine_reasons": dict(current_reasons.most_common()),
        "frozen_target_outcomes": dict(frozen_outcomes.most_common()),
        "frozen_target_disagreements": dict(frozen_disagreements.most_common()),
        "old_transcript_authority_used": old_authority,
        "new_asr_equivalent_text_used": match_new,
        "unresolved_examples": unresolved_examples,
    }
    (run / "reports").mkdir(parents=True, exist_ok=True)
    (run / "reports" / "completion_gap_diagnosis.json").write_text(json.dumps({**report, **summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
