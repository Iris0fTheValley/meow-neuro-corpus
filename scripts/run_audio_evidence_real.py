from __future__ import annotations

"""Execute the real raw-waveform audio reconstruction sidecar.

This runner is intentionally separate from the 8787490 artifact-replay run.
Old transcript/timeline rows are comparison evidence only; new hypotheses are
generated from decoded waveform by the pinned Qwen models.
"""

import hashlib
import json
import os
import sys
import argparse
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from audio_evidence import real_adapters  # noqa: E402
from audio_evidence.cache import CheckpointStore, EvidenceCache  # noqa: E402
from audio_evidence.contracts import Timebase  # noqa: E402
from audio_evidence.deduplication import deduplicate_materialized, removal_counts  # noqa: E402
from audio_evidence.materialization import FinalViewMembership, materialize_role_preserving  # noqa: E402
from audio_evidence.pipeline import AudioEvidencePipeline  # noqa: E402
from audio_evidence.planning import AudioWindowPlanner  # noqa: E402
from audio_evidence.validation import validate_artifacts  # noqa: E402
import run_audio_evidence_production as replay  # noqa: E402

CORPUS = ROOT.parent.parent
DATASET = CORPUS / "datasets" / "meow_v02_sft_v2_2_semantic_verified"
SPLIT = DATASET / "split_authority_v2_3.json"
CONTEXT_DECISIONS = DATASET / "context_sufficiency_decisions_v2_3.jsonl"
RUN_ID = os.environ.get("MEOW_REAL_RUN_ID", "audio-reconstruction-v1-real-20260917")
CACHE_RUN_ID = os.environ.get("MEOW_REAL_CACHE_RUN_ID", RUN_ID)
OUT = DATASET / "audio_reconstruction_v1" / RUN_ID
CACHE_ROOT = DATASET / "audio_reconstruction_v1" / CACHE_RUN_ID
CACHE_ONLY = os.environ.get("MEOW_REAL_CACHE_ONLY", "").strip().lower() in {"1", "true", "yes"}


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def prepare_real_interactions(pool, manifest, split, context_states):
    rows, _ = replay.prepare_interactions(pool, manifest, split)
    out = []
    for row in rows:
        row = dict(row)
        ts = dict(row.get("timestamps") or {})
        start, end = float(ts["start"]), float(ts["end"])
        incomplete = context_states.get(str(row["sample_id"])) == "CONTEXT_INCOMPLETE" or not bool((row.get("semantic_qa") or {}).get("context_complete", True))
        # Context-incomplete samples get a real preceding-audio search window;
        # complete samples retain a small boundary guard.
        ts["start"] = max(0.0, start - (60.0 if incomplete else 2.0))
        ts["end"] = end + 5.0
        ts["timebase"] = Timebase.RECORDING_SECONDS.value
        row["timestamps"] = ts
        row["audio_context_policy"] = "preceding-60s-post-5s-v2" if incomplete else "preceding-2s-post-5s-v2"
        row["context_incomplete_input"] = incomplete
        out.append(row)
    return out


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in rows), encoding="utf-8")


def bound_windows(plan: dict, max_seconds: float = 120.0) -> dict:
    """Split long merged unions so ASR never receives an unbounded interval."""
    windows, mappings = [], []
    by_old = defaultdict(list)
    for mapping in plan.get("sample_window_mappings", []):
        by_old[mapping["window_id"]].append(mapping)
    for old in plan.get("windows", []):
        start = float(old["merged_interval"]["start"]); end = float(old["merged_interval"]["end"])
        chunks = []
        cursor = start
        while cursor < end - 1e-6:
            chunk_end = min(end, cursor + max_seconds)
            identity = f"{old['window_id']}|{cursor:.6f}|{chunk_end:.6f}|bounded-120s-v1"
            wid = "aw_" + hashlib.sha256(identity.encode()).hexdigest()[:20]
            chunks.append((wid, cursor, chunk_end))
            windows.append({**old, "window_id": wid, "merged_interval": {"start": cursor, "end": chunk_end, "timebase": Timebase.RECORDING_SECONDS.value}, "sample_ids": [] , "policy_version": "audio-window-planner-bounded-120s-v1"})
            cursor = chunk_end
        for mapping in by_old.get(old["window_id"], []):
            req = mapping.get("requested_interval") or {}
            midpoint = (float(req.get("start", start)) + float(req.get("end", end))) / 2.0
            chosen = next((x for x in chunks if x[1] <= midpoint <= x[2]), chunks[-1])
            mappings.append({**mapping, "window_id": chosen[0]})
            for window in windows:
                if window["window_id"] == chosen[0]: window["sample_ids"] = sorted(set(window["sample_ids"]) | {mapping["sample_id"]})
    return {"schema_version": "1.0.0", "policy_version": "audio-window-planner-bounded-120s-v1", "windows": windows, "sample_window_mappings": sorted(mappings, key=lambda x: (x["sample_id"], x["window_id"]))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-windows", type=int, default=0)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pool = load_jsonl(replay.POOL)
    manifest = replay.source_metadata()
    identity_rows = load_jsonl(replay.IDENTITY)
    context_states = {str(r.get("sample_id")): str(r.get("state")) for r in load_jsonl(CONTEXT_DECISIONS)}
    identity_map = {(str(r.get("source_id")), str(r.get("cluster"))): r for r in identity_rows}
    timelines = {sid: json.loads((replay.TIMELINE_DIR / f"{sid}.json").read_text(encoding="utf-8")) for sid in {str(r["source_id"]) for r in pool} if (replay.TIMELINE_DIR / f"{sid}.json").exists()}

    # Reuse only the frozen identity authority to select confirmed references;
    # actual ECAPA enrollment embeddings are computed below from raw waveform.
    replay.OUT = OUT
    bank = replay.build_enrollment_bank(identity_rows, manifest)
    ecapa = real_adapters.ECAPARawAdapters(CORPUS / "models" / "speechbrain_ecapa", threshold=float(os.environ.get("MEOW_PVAD_THRESHOLD", "0.62")))
    embeddings = {}
    # Reuse a frozen enrollment embedding only when its backend provenance and
    # reference set match exactly; the embedding index is part of the evidence
    # cache key, so a silent recomputation would invalidate every cached stage.
    cached_embedding_path = CACHE_ROOT / "enrollment_embeddings.json"
    cached_embeddings = {}
    if CACHE_ONLY and cached_embedding_path.exists():
        cached_document = json.loads(cached_embedding_path.read_text(encoding="utf-8"))
        if cached_document.get("backend") == ecapa.activity.provenance.to_dict():
            cached_embeddings = cached_document.get("embeddings") or {}
    for eid, ref in bank.references.items():
        if eid in cached_embeddings:
            embeddings[eid] = cached_embeddings[eid]
            continue
        embeddings[eid] = ecapa.enrollment_embedding(ref).tolist()
    (OUT / "enrollment_embeddings.json").write_text(json.dumps({"backend": ecapa.activity.provenance.to_dict(), "embeddings": embeddings, "reused_from": str(cached_embedding_path) if cached_embeddings else None}, ensure_ascii=False, indent=2), encoding="utf-8")

    interactions = prepare_real_interactions(pool, manifest, json.loads(SPLIT.read_text(encoding="utf-8")), context_states)
    planner = AudioWindowPlanner(padding_before=0.0, padding_after=0.0, merge_gap=1.0)
    plan = bound_windows(planner.merge([planner.request_from_interaction(x) for x in interactions]), max_seconds=120.0)
    (OUT / "window_plan.json").write_text(json.dumps({**plan, "context_policy": "incomplete=60s pre-roll, complete=2s pre-roll, post=5s", "planner_backend": "audio-window-planner-v1"}, ensure_ascii=False, indent=2), encoding="utf-8")
    mapping_by_window = defaultdict(list)
    for mapping in plan["sample_window_mappings"]:
        mapping_by_window[mapping["window_id"]].append(mapping)
    interaction_by_id = {r["sample_id"]: r for r in interactions}
    # Verified semantic targets that live inside a window keep their frozen text
    # authority; every other turn in the window stays under the conservative
    # fail-closed context policy.
    verified_targets_by_window = defaultdict(set)
    for mapping in plan["sample_window_mappings"]:
        source = interaction_by_id.get(mapping["sample_id"])
        if not source:
            continue
        target_ids = [str(x) for x in source.get("target_turn_ids") or []]
        if target_ids:
            verified_targets_by_window[mapping["window_id"]].add(target_ids[-1])
    qwen = real_adapters.QwenRawAdapters(str(CORPUS / "models" / "qwen3_asr" / "Qwen3-ASR-1.7B"), str(CORPUS / "models" / "qwen3_asr" / "Qwen3-ForcedAligner-0.6B"))
    evidence_cache = EvidenceCache(CACHE_ROOT / "cache")
    checkpoint = CheckpointStore(OUT / "checkpoint.json")
    if CACHE_ONLY and (CACHE_ROOT / "checkpoint.json").exists():
        # Reuse the durable stage progress of the source run; identical
        # content-addressed keys make every record a no-op, so the checkpoint is
        # carried over without rewriting it tens of thousands of times.
        checkpoint.seed(json.loads((CACHE_ROOT / "checkpoint.json").read_text(encoding="utf-8")))
    pipeline = AudioEvidencePipeline(bank, evidence_cache, checkpoint, activity=ecapa.activity, diarization=ecapa.diarization, asr=qwen.asr, aligner=qwen.aligner)
    enrollment_id = sorted(bank.references)[0]
    turn_lookup, window_results, failures = {}, [], []
    stage_counts = Counter()
    cache_hit_counts, cache_miss_counts = Counter(), Counter()
    retry_only = args.retry_failed
    retry_ids = set()
    if retry_only:
        prior = OUT / "unresolved.jsonl"
        if prior.exists():
            retry_ids = {str(x.get("window_id")) for x in load_jsonl(prior) if x.get("window_id")}
        windows_to_process = [w for w in plan["windows"] if w["window_id"] in retry_ids]
    else:
        windows_to_process = plan["windows"]
    if not retry_only and args.limit_windows:
        stride = max(1, len(plan["windows"]) // args.limit_windows)
        windows_to_process = [plan["windows"][i] for i in range(0, len(plan["windows"]), stride)][: args.limit_windows]
    elif not retry_only:
        windows_to_process = plan["windows"]
    for index, window in enumerate(windows_to_process, 1):
        old_turns = {}
        timeline = timelines.get(window["recording_id"], {})
        byid = {str(t.get("turn_id")): t for t in timeline.get("turns", [])}
        # Old turns delimit comparison spans and preserve identity authority;
        # their text is never passed to Qwen.
        for t in timeline.get("turns", []):
            start, end = float(t["timestamp"]["start"]), float(t["timestamp"]["end"])
            if start < float(window["merged_interval"]["end"]) and end > float(window["merged_interval"]["start"]):
                key = str(t.get("turn_id"))
                ident = identity_map.get((window["recording_id"], str(t.get("speaker"))), {}).get("identity")
                old_turns[key] = {"audio_turn_id": key, "start": max(start, float(window["merged_interval"]["start"])), "end": min(end, float(window["merged_interval"]["end"])), "old_transcript": t.get("text"), "speaker_cluster": t.get("speaker"), "identity": ident, "role": "assistant" if ident in {"NEURO_FAMILY_HIGH", "NEURO_FAMILY_MEDIUM"} else "user", "identity_evidence": identity_map.get((window["recording_id"], str(t.get("speaker"))), {}).get("provenance")}
        try:
            result = pipeline.process_window(
                window, enrollment_id, list(old_turns.values()),
                frozen_verified_target_turn_ids=sorted(verified_targets_by_window.get(window["window_id"], set())),
                cache_only=CACHE_ONLY,
            )
            window_results.append(result)
            for key, value in result.get("stage_status", {}).items(): stage_counts[f"{key}:{value}"] += 1
            for key, hit in (result.get("cache_hits") or {}).items(): (cache_hit_counts if hit else cache_miss_counts)[key] += 1
            for turn in result.get("turns", []): turn_lookup[str(turn["audio_turn_id"])] = turn
        except Exception as exc:
            failures.append({"window_id": window["window_id"], "stage": "window", "exception": type(exc).__name__, "message": str(exc), "retry_count": 1 if retry_only else 0, "disposition": "abandoned" if retry_only else "retry_pending"})
        if index % 10 == 0 or index == len(windows_to_process):
            print(json.dumps({"windows_done": index, "windows_total": len(windows_to_process), "planned_total": len(plan["windows"]), "failures": len(failures)}, ensure_ascii=False), flush=True)

    split_assignments = json.loads(SPLIT.read_text(encoding="utf-8")).get("member_assignments", {})
    materialized, quarantine = [], []
    for row in interactions:
        target_ids = [str(x) for x in row.get("target_turn_ids") or []]
        target_id = target_ids[-1]
        timeline = timelines.get(row["source_id"], {})
        target_record = next((t for t in timeline.get("turns", []) if str(t.get("turn_id")) == target_id), None)
        if target_record is None:
            quarantine.append({"sample_id": row["sample_id"], "reason": "TARGET_TURN_PROVENANCE_MISSING"}); continue
        # Use the target turn's canonical audio timestamp, never the expanded
        # interaction request start, to bound historical recovery.
        target_start = float(target_record["timestamp"]["start"])
        selected = []
        # Preserve the frozen selected context, then add real historical Neuro
        # assistant turns recovered in the expanded preceding interval.
        for tid in [str(x) for x in row.get("context_turn_ids") or []] + target_ids:
            if tid in turn_lookup: selected.append(tid)
        recovered = []
        for t in timeline.get("turns", []):
            tid = str(t.get("turn_id")); ident = identity_map.get((row["source_id"], str(t.get("speaker"))), {}).get("identity")
            recovery_floor = max(0.0, target_start - (60.0 if row.get("context_incomplete_input") else 2.0))
            if tid in turn_lookup and ident in {"NEURO_FAMILY_HIGH", "NEURO_FAMILY_MEDIUM"} and recovery_floor <= float(t["timestamp"]["start"]) and float(t["timestamp"]["end"]) <= target_start and tid not in selected:
                recovered.append((float(t["timestamp"]["start"]), tid))
        selected = [tid for _, tid in sorted(recovered)] + selected
        if target_id not in selected:
            quarantine.append({"sample_id": row["sample_id"], "reason": "TARGET_AUDIO_EVIDENCE_UNAVAILABLE"}); continue
        # A disagreement in an optional context turn must not invalidate a
        # separately resolved target.  Drop only unresolved context evidence;
        # an unresolved target remains quarantined fail-closed.
        resolved_ids = {tid for tid in selected if (turn_lookup.get(tid) or {}).get("text_resolution_state") == "RESOLVED"}
        if target_id not in resolved_ids:
            quarantine.append({"sample_id": row["sample_id"], "reason": "TARGET_TEXT_UNRESOLVED"}); continue
        selected = [tid for tid in selected if tid in resolved_ids]
        # Only assistant target rows are eligible for supervision.
        membership = {"train": FinalViewMembership.IN_TRAIN, "validation": FinalViewMembership.IN_VALIDATION, "sealed_eval": FinalViewMembership.IN_SEALED_EVAL}.get(split_assignments.get(row["canonical_recording_id"]), FinalViewMembership.EXCLUDED)
        try:
            materialized.append(materialize_role_preserving(row, turn_lookup.values(), selected, target_id, membership))
        except Exception as exc:
            quarantine.append({"sample_id": row["sample_id"], "reason": type(exc).__name__ + ":" + str(exc)})

    # Hard dedup on canonical interaction provenance plus deterministic
    # prefix-ladder removal.  Both stages accumulate into the single removal
    # ledger that is written to dedup.jsonl, recorded in the manifest as
    # dedup_removed, and handed to the finalizer.
    materialized, removed = deduplicate_materialized(materialized)
    write_jsonl(OUT / "canonical_audio_evidence_timeline.jsonl", turn_lookup.values())
    write_jsonl(OUT / "materialized.jsonl", materialized)
    write_jsonl(OUT / "quarantine.jsonl", quarantine)
    write_jsonl(OUT / "dedup.jsonl", removed)
    write_jsonl(OUT / "unresolved.jsonl", failures)
    checkpoint.flush()

    split_names = [("train", "IN_TRAIN"), ("validation", "IN_VALIDATION"), ("sealed_eval", "IN_SEALED_EVAL")]
    for name, membership in split_names: write_jsonl(OUT / f"{name}.jsonl", [r for r in materialized if r.get("final_view_membership") == membership])
    validation = validate_artifacts(bank, plan["windows"], list(turn_lookup.values()), materialized, expected_semantic_hashes={r["sample_id"]: r["semantic_truth_sha256"] for r in materialized}, expected_split_hash=hashlib.sha256(SPLIT.read_bytes()).hexdigest())
    (OUT / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    try: code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception: code_commit = "unknown"
    report = {"run_id": RUN_ID, "created_at": datetime.now(timezone.utc).isoformat(), "code_commit": code_commit, "baseline_run": "audio-reconstruction-v1-20260917", "evidence_cache_run": CACHE_RUN_ID, "cache_only": CACHE_ONLY, "calibration_limit_windows": args.limit_windows or None, "enrollment_bank_revision": bank.revision, "enrollment_reference_count": len(bank.references), "input_semantic_verified": len(pool), "processed_interactions": len(interactions), "windows_requested": len(plan["windows"]), "windows_processed": len(windows_to_process), "raw_requested_duration_seconds": sum(float(x["requested_interval"]["end"]) - float(x["requested_interval"]["start"]) for x in plan["sample_window_mappings"]), "unique_merged_duration_seconds": sum(float(x["merged_interval"]["end"]) - float(x["merged_interval"]["start"]) for x in plan["windows"]), "stage_counts": dict(stage_counts), "stage_cache_hits": dict(cache_hit_counts), "stage_cache_misses": dict(cache_miss_counts), "expensive_model_stages_executed": int(sum(cache_miss_counts.values())), "window_failures": len(failures), "quarantine": len(quarantine), "dedup_removed": len(removed), "dedup_removed_by_reason": removal_counts(removed), "materialized": len(materialized), "model_provenance": {"target_activity": ecapa.activity.provenance.to_dict(), "diarization": ecapa.diarization.provenance.to_dict(), "asr": qwen.asr.provenance.to_dict(), "alignment": qwen.aligner.provenance.to_dict()}, "validator": validation, "known_limitations": ["ECAPA frame clustering is used as the generic diarization backend because pyannote weights are not configured; overlap is inferred from simultaneous frame cluster evidence and never hard-coded."]}
    (OUT / "run_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run_id": RUN_ID, "windows": len(plan["windows"]), "materialized": len(materialized), "quarantine": len(quarantine), "validation_pass": validation.get("pass"), "output": str(OUT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
