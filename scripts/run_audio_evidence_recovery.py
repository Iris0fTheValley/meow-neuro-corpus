from __future__ import annotations

"""Deterministic recovery and audio-timeline reconstruction pass.

This pass is deliberately separate from the frozen ``real-completion`` run.  It
consumes its bounded-window cache, fixes target->window authority by containment,
merges evidence from every containing window, and emits a new sidecar namespace.
No semantic or split authority is rewritten.  A target-local inference adapter
can be plugged in later; cache evidence is always attempted first.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from audio_evidence.cache import EvidenceCache  # noqa: E402
from audio_evidence.contracts import Timebase  # noqa: E402
from audio_evidence.deduplication import deduplicate_materialized, removal_counts  # noqa: E402
from audio_evidence.materialization import FinalViewMembership, materialize_role_preserving  # noqa: E402
from audio_evidence.transcript import detect_disagreement  # noqa: E402
import run_audio_evidence_production as replay  # noqa: E402

CORPUS = ROOT.parent.parent
DATASET = CORPUS / "datasets" / "meow_v02_sft_v2_2_semantic_verified"
RUN_ROOT = DATASET / "audio_reconstruction_v1"
SOURCE_RUN = RUN_ROOT / "audio-reconstruction-v1-real-completion-20260919"
CACHE_RUN = RUN_ROOT / "audio-reconstruction-v1-real-context60-bounded120-20260918"
OUT_NAME = "audio-reconstruction-v1-recovery-20260919"

NEURO = {"NEURO", "EVIL_NEURO", "NEURO_FAMILY", "NEURO_FAMILY_HIGH", "NEURO_FAMILY_MEDIUM"}


def load_jsonl(path: Path):
    return [json.loads(x) for x in path.open(encoding="utf-8") if x.strip()]


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--source-run", default=str(SOURCE_RUN))
    ap.add_argument("--cache-run", default=str(CACHE_RUN))
    ap.add_argument("--out-name", default=OUT_NAME)
    args = ap.parse_args()
    source_run, cache_run, out = Path(args.source_run), Path(args.cache_run), RUN_ROOT / args.out_name
    out.mkdir(parents=True, exist_ok=True)

    pool = load_jsonl(replay.POOL)
    manifest = replay.source_metadata()
    identity_rows = load_jsonl(replay.IDENTITY)
    identity_map = {(str(r.get("source_id")), str(r.get("cluster"))): r for r in identity_rows}
    split = json.loads(replay.SPLIT_AUTHORITY.read_text(encoding="utf-8"))
    context_decisions = DATASET / "context_sufficiency_decisions_v2_3.jsonl"
    context_states = {str(r.get("sample_id")): str(r.get("state")) for r in load_jsonl(context_decisions)}
    interactions, _ = replay.prepare_interactions(pool, manifest, split)
    if args.limit:
        interactions = interactions[: args.limit]
    timelines = {}
    for r in interactions:
        sid = str(r["source_id"])
        path = replay.TIMELINE_DIR / (sid + ".json")
        if path.exists():
            timelines[sid] = json.loads(path.read_text(encoding="utf-8"))
    target_ids = {str(x) for r in interactions for x in (r.get("target_turn_ids") or [])}
    incomplete_ids = {sid for sid, state in context_states.items() if state == "CONTEXT_INCOMPLETE"}

    plan = json.loads((source_run / "window_plan.json").read_text(encoding="utf-8"))
    checkpoint = json.loads((cache_run / "checkpoint.json").read_text(encoding="utf-8")).get("completed") or {}
    cache = EvidenceCache(cache_run / "cache")
    windows = plan.get("windows", [])
    by_recording = defaultdict(list)
    for w in windows:
        by_recording[str(w["recording_id"])].append(w)

    def stage(w, name):
        key = (checkpoint.get(str(w["window_id"])) or {}).get(name)
        return cache.get(name, key) if key else None

    # Materialize a per-window evidence view once.  The cache payload is already
    # in recording-global seconds; retain all source windows for provenance.
    evidence = {}
    for w in windows:
        evidence[str(w["window_id"])] = {
            "window": w, "asr": stage(w, "asr"), "alignment": stage(w, "alignment"),
            "activity": stage(w, "target_activity") or [], "diarization": stage(w, "diarization") or [],
        }

    def containing(sid, start, end):
        return [w for w in by_recording.get(str(sid), [])
                if float(w["merged_interval"]["start"]) <= start + 1e-6 and end <= float(w["merged_interval"]["end"]) + 1e-6]

    def intersecting(sid, start, end):
        return [w for w in by_recording.get(str(sid), []) if overlap(start, end, float(w["merged_interval"]["start"]), float(w["merged_interval"]["end"])) > 0]

    def identity(sid, speaker):
        return (identity_map.get((str(sid), str(speaker))) or {}).get("identity")

    # Fixed target authority: containment first, then best evidence completeness.
    target_windows = defaultdict(list)
    diagnostics = []
    for row in interactions:
        sid, sample = str(row["source_id"]), str(row["sample_id"])
        timeline = timelines.get(sid, {})
        tids = [str(x) for x in row.get("target_turn_ids") or []]
        tid = tids[-1] if tids else ""
        target = next((t for t in timeline.get("turns", []) if str(t.get("turn_id")) == tid), None)
        if target is None:
            diagnostics.append({"sample_id": sample, "recording_id": sid, "target_turn_id": tid, "original_quarantine_reason": "TARGET_TURN_PROVENANCE_MISSING", "final_disposition": "FINAL_TARGET_UNRESOLVED"})
            continue
        ts, te = float(target["timestamp"]["start"]), float(target["timestamp"]["end"])
        candidates = containing(sid, ts, te)
        if not candidates:
            candidates = intersecting(sid, ts, te)
        scored = []
        for w in candidates:
            ev = evidence[str(w["window_id"])]
            al = [x for x in (ev["alignment"] or []) if overlap(ts, te, float(x.get("start", 0)), float(x.get("end", 0))) > 0]
            act = [x for x in ev["activity"] if overlap(ts, te, float(x.get("start", 0)), float(x.get("end", 0))) > 0]
            diar = [x for x in ev["diarization"] if overlap(ts, te, float(x.get("start", 0)), float(x.get("end", 0))) > 0]
            scored.append((len(candidates) == len(containing(sid, ts, te)) and len(containing(sid, ts, te)) > 0, len(al), len(act), len(diar), str(w["window_id"]), w, al, act, diar))
        scored.sort(key=lambda x: tuple(int(v) if isinstance(v, bool) else v for v in x[:4]), reverse=True)
        chosen = scored[0] if scored else None
        if chosen:
            target_windows[sample] = [x[5] for x in scored]
        old_text = str(target.get("text") or "")
        diagnostics.append({
            "sample_id": sample, "recording_id": sid, "target_turn_id": tid,
            "original_quarantine_reason": "TARGET_TEXT_UNRESOLVED" if sample in {str(x.get("sample_id")) for x in load_jsonl(source_run / "quarantine.jsonl")} else None,
            "target_old_text": old_text, "target_new_asr_text": (evidence[chosen[4]]["asr"] or {}).get("text") if chosen and evidence[chosen[4]]["asr"] else None,
            "target_old_interval": {"start": ts, "end": te},
            "target_windows": [str(x["window_id"]) for x in [z[5] for z in scored]],
            "complete_containing_window": bool(containing(sid, ts, te)),
            "target_activity_evidence": bool(chosen and chosen[7]), "diarization_evidence": bool(chosen and chosen[8]),
            "alignment_evidence": len(chosen[6]) if chosen else 0, "asr_evidence": bool(chosen and evidence[chosen[4]]["asr"]),
            "possible_window_mapping_bug": bool(chosen and containing(sid, ts, te)),
            "possible_alignment_attribution_failure": bool(chosen and not chosen[6] and evidence[chosen[4]]["asr"]),
            "possible_speaker_conflict": False, "possible_boundary_conflict": bool(chosen and not chosen[6]),
            "cache_evidence": bool(chosen and (chosen[5] or chosen[6] or chosen[7] or chosen[8])),
        })

    # Old turns are retained as evidence records; multiple window observations
    # are merged without last-write-wins.  New turns are generated only from
    # aligned audio spans that do not already belong to an old turn.
    turn_lookup = {}
    source_cache_windows = defaultdict(list)
    for sid, ws in by_recording.items():
        source_cache_windows[sid] = ws
    for sid, tl in timelines.items():
        for old in tl.get("turns", []):
            tid = str(old.get("turn_id")); s, e = float(old["timestamp"]["start"]), float(old["timestamp"]["end"])
            ident = identity(sid, old.get("speaker")); role = "assistant" if ident in NEURO else "user"
            obs = []
            source_windows = [w for w in source_cache_windows.get(sid, []) if overlap(s, e, float(w["merged_interval"]["start"]), float(w["merged_interval"]["end"])) > 0]
            for w in source_windows:
                ev = evidence[str(w["window_id"])]
                obs.extend(x for x in (ev["alignment"] or []) if overlap(s, e, float(x.get("start", 0)), float(x.get("end", 0))) > 0)
            text = str(old.get("text") or "")
            if obs:
                new_text = " ".join(str(x.get("text") or "") for x in sorted(obs, key=lambda z: float(z.get("start", 0)))).strip()
                disagreement = detect_disagreement(text, new_text)
            else:
                new_text, disagreement = None, None
            is_target = tid in target_ids and text
            if is_target and text:
                # Frozen target authority wins when the target has bounded audio
                # support.  This explicitly treats long-window attribution drift
                # as recoverable local evidence, never as a semantic mutation.
                has_support = bool(obs) or any(overlap(s, e, float(x.get("start", 0)), float(x.get("end", 0))) > 0 for w in source_windows for x in evidence[str(w["window_id"])]["activity"])
                resolved = has_support
                authority = "OLD_TRANSCRIPT" if resolved else None
                state = "RESOLVED" if resolved else "UNRESOLVED"
                provenance = {"status": state, "resolver": "CACHE_ONLY_TARGET_RESCUE_V1", "revision": "audio-evidence-target-rescue-v1", "disagreement": disagreement.value if disagreement else None, "resolved_disagreements": [disagreement.value] if disagreement else [], "source_windows": [str(w["window_id"]) for w in source_windows], "new_asr_role": "INDEPENDENT_AUDIO_CONFIRMATION_EVIDENCE"}
                resolved_text = text if resolved else None
            elif text:
                # Context text remains conservative but usable when it is backed
                # by aligned audio or the frozen old turn itself.
                resolved, authority, state, provenance, resolved_text = True, "OLD_TRANSCRIPT", "RESOLVED", {"status": "RESOLVED", "resolver": "CONTEXT_OLD_TRANSCRIPT_WITH_AUDIO_EVIDENCE_V1", "resolved_disagreements": [disagreement.value] if disagreement else [], "source_windows": [str(w["window_id"]) for w in source_windows]}, text
            else:
                resolved, authority, state, provenance, resolved_text = False, None, "UNRESOLVED", None, None
            turn_lookup[tid] = {"audio_turn_id": tid, "recording_id": sid, "canonical_recording_id": next((r.get("canonical_recording_id") for r in interactions if str(r["source_id"]) == sid), None), "recording_family_id": next((r.get("recording_family_id") for r in interactions if str(r["source_id"]) == sid), None), "window_id": source_windows[0]["window_id"] if source_windows else None, "start": s, "end": e, "timebase": Timebase.RECORDING_SECONDS.value, "old_transcript": text or None, "new_asr_hypothesis": new_text, "resolved_text": resolved_text, "text_resolution_state": state, "text_authority": authority, "text_resolution_provenance": provenance, "disagreement_flags": [disagreement.value] if disagreement else [], "identity": ident or "NON_TARGET_GUEST", "role": role, "speaker_cluster": old.get("speaker"), "source_audio": str((manifest.get(sid) or {}).get("audio_path") or f"raw_audio/{sid}.webm"), "source_interval": {"start": s, "end": e, "timebase": Timebase.RECORDING_SECONDS.value}, "asr_source": {"source_windows": [str(w["window_id"]) for w in source_windows]} if new_text else None, "alignment": obs[:1000], "target_activity_evidence": [], "diarization_evidence": [], "model_provenance": {"asr": {"backend": "qwen3-asr-cache", "revision": "Qwen/Qwen3-ASR-1.7B@7278e1e70fe206f11671096ffdd38061171dd6e5"}, "alignment": {"backend": "qwen3-forced-aligner-cache", "revision": "Qwen/Qwen3-ForcedAligner-0.6B@c7cbfc2048c462b0d63a45797104fc9db3ad62b7"}, "recovery": "audio-evidence-target-rescue-v1"}}

    # Reconstruct genuinely new speech turns from aligned tokens.  Segmentation
    # is deterministic: same anonymous cluster and <=1.25s internal silence.
    for sid, ws in source_cache_windows.items():
        if sid not in timelines:
            continue
        for w in ws:
            ev = evidence[str(w["window_id"])]
            tokens = sorted(ev["alignment"] or [], key=lambda x: float(x.get("start", 0)))
            groups, cur = [], []
            for tok in tokens:
                s, e = float(tok.get("start", 0)), float(tok.get("end", 0))
                if e <= s or not str(tok.get("text") or "").strip():
                    continue
                cluster = next((str(d.get("speaker_cluster")) for d in ev["diarization"] if overlap(s, e, float(d.get("start", 0)), float(d.get("end", 0))) > 0), "UNKNOWN")
                if cur and (s - float(cur[-1]["end"]) > 1.25 or cluster != cur[-1]["cluster"]):
                    groups.append(cur); cur = []
                cur.append({"text": str(tok["text"]).strip(), "start": s, "end": e, "cluster": cluster})
            if cur: groups.append(cur)
            for idx, group in enumerate(groups):
                s, e = group[0]["start"], group[-1]["end"]
                if e - s < 0.25: continue
                if any(overlap(s, e, float(t["start"]), float(t["end"])) / max(0.01, e - s) >= 0.5 for t in turn_lookup.values() if t["recording_id"] == sid):
                    continue
                cluster = group[0]["cluster"]
                ident = identity(sid, cluster)
                target_activity_overlap = any(overlap(s, e, float(a.get("start", 0)), float(a.get("end", 0))) >= 0.5 * (e - s) for a in ev["activity"])
                role = "assistant" if ident in NEURO or target_activity_overlap else ("user" if cluster != "UNKNOWN" else "ambiguous")
                if target_activity_overlap and ident not in NEURO:
                    ident = "NEURO_FAMILY_ACTIVITY_SUPPORTED"
                if role == "ambiguous": continue
                nid = "audio:new:%s:%04d" % (w["window_id"], idx)
                text = " ".join(x["text"] for x in group).strip()
                turn_lookup[nid] = {"audio_turn_id": nid, "recording_id": sid, "canonical_recording_id": next((r.get("canonical_recording_id") for r in interactions if str(r["source_id"]) == sid), None), "recording_family_id": next((r.get("recording_family_id") for r in interactions if str(r["source_id"]) == sid), None), "window_id": w["window_id"], "start": s, "end": e, "timebase": Timebase.RECORDING_SECONDS.value, "old_transcript": None, "new_asr_hypothesis": text, "resolved_text": text, "text_resolution_state": "RESOLVED", "text_authority": "NEW_ASR", "text_resolution_provenance": {"status": "RESOLVED", "resolver": "AUDIO_TIMELINE_SEGMENT_AGGREGATION_V1", "revision": "audio-context-reconstruction-v1", "source_waveform": str((manifest.get(sid) or {}).get("audio_path") or f"raw_audio/{sid}.webm"), "source_window": w["window_id"], "alignment_token_count": len(group)}, "disagreement_flags": [], "identity": ident or "NON_TARGET_GUEST", "role": role, "speaker_cluster": cluster, "source_audio": str((manifest.get(sid) or {}).get("audio_path") or f"raw_audio/{sid}.webm"), "source_interval": {"start": s, "end": e, "timebase": Timebase.RECORDING_SECONDS.value}, "asr_source": {"source_window": w["window_id"], "waveform": str((manifest.get(sid) or {}).get("audio_path") or f"raw_audio/{sid}.webm")}, "alignment": group, "target_activity_evidence": [], "diarization_evidence": [], "model_provenance": {"asr": {"backend": "qwen3-asr-cache", "revision": "Qwen/Qwen3-ASR-1.7B@7278e1e70fe206f11671096ffdd38061171dd6e5"}, "alignment": {"backend": "qwen3-forced-aligner-cache", "revision": "Qwen/Qwen3-ForcedAligner-0.6B@c7cbfc2048c462b0d63a45797104fc9db3ad62b7"}}}

    # Role-preserving materialization over the full semantic pool.
    by_source = defaultdict(list)
    for t in turn_lookup.values(): by_source[str(t["recording_id"])].append(t)
    source_quarantine = {str(r.get("sample_id")): str(r.get("reason")) for r in load_jsonl(source_run / "quarantine.jsonl")}
    materialized, quarantine = [], []
    class_counts = Counter(); rescue_counts = Counter()
    for row in interactions:
        sample, sid = str(row["sample_id"]), str(row["source_id"])
        tids = [str(x) for x in row.get("target_turn_ids") or []]; target_id = tids[-1] if tids else ""
        target = turn_lookup.get(target_id)
        if not target or target.get("text_resolution_state") != "RESOLVED":
            quarantine.append({"sample_id": sample, "reason": "FINAL_TARGET_UNRESOLVED", "original_reason": source_quarantine.get(sample, "TARGET_AUDIO_EVIDENCE_UNAVAILABLE")}); rescue_counts["final_unresolved"] += 1; continue
        ts = float(target["start"]); incomplete = sample in incomplete_ids or not bool((row.get("semantic_qa") or {}).get("context_complete", True))
        floor = max(0.0, ts - (60.0 if incomplete else 2.0))
        frozen_ids = [str(x) for x in row.get("context_turn_ids") or []]
        eligible = [t for t in by_source[sid] if t["audio_turn_id"] != target_id and float(t["end"]) <= ts + 1e-6 and float(t["start"]) >= floor and t.get("text_resolution_state") == "RESOLVED"]
        selected = {t["audio_turn_id"] for t in eligible}
        selected.update(tid for tid in frozen_ids if tid in turn_lookup and turn_lookup[tid].get("text_resolution_state") == "RESOLVED")
        ordered = sorted((turn_lookup[x] for x in selected), key=lambda t: (float(t["start"]), t["audio_turn_id"]))
        selected_ids = [t["audio_turn_id"] for t in ordered] + [target_id]
        if incomplete and not ordered:
            quarantine.append({"sample_id": sample, "reason": "CONTEXT_INCOMPLETE_NO_AUDIO_CONTEXT", "original_reason": source_quarantine.get(sample)}); rescue_counts["context_still_incomplete"] += 1; continue
        membership = {"train": FinalViewMembership.IN_TRAIN, "validation": FinalViewMembership.IN_VALIDATION, "sealed_eval": FinalViewMembership.IN_SEALED_EVAL}.get(split.get("member_assignments", {}).get(str(row.get("canonical_recording_id"))), FinalViewMembership.EXCLUDED)
        try:
            out_row = materialize_role_preserving(row, turn_lookup.values(), selected_ids, target_id, membership)
        except Exception as exc:
            quarantine.append({"sample_id": sample, "reason": type(exc).__name__ + ":" + str(exc)}); continue
        original_context = set(frozen_ids)
        added = [x for x in selected if x not in original_context]
        if incomplete and added:
            kind = "CONTEXT_AUDIO_RECOVERED"
        elif added:
            kind = "CONTEXT_AUDIO_ENHANCED"
        else:
            kind = "ORIGINAL_CONTEXT_CONFIRMED"
        out_row["context_reconstruction_class"] = kind
        out_row["context_sufficiency_recheck"] = {"checker": "deterministic-context-sufficiency-v1", "state": "SELF_CONTAINED" if (not incomplete or bool(ordered)) else "CONTEXT_INCOMPLETE", "has_real_context": bool(ordered), "target_is_final_supervised_turn": True}
        out_row["audio_evidence_provenance"] = {"recovery_run": args.out_name, "source_cache_run": cache_run.name, "new_context_turn_ids": added, "target_rescue": sample in source_quarantine}
        materialized.append(out_row); class_counts[kind] += 1
        rescue_counts["target_rescued"] += int(sample in source_quarantine)

    materialized, removed = deduplicate_materialized(materialized)
    write_jsonl(out / "canonical_audio_evidence_timeline.jsonl", sorted(turn_lookup.values(), key=lambda t: (t["recording_id"], float(t["start"]), t["audio_turn_id"])))
    write_jsonl(out / "materialized.jsonl", materialized); write_jsonl(out / "quarantine.jsonl", quarantine); write_jsonl(out / "dedup.jsonl", removed); write_jsonl(out / "diagnostic_breakdown.jsonl", diagnostics)
    for name, membership in (("train", "IN_TRAIN"), ("validation", "IN_VALIDATION"), ("sealed_eval", "IN_SEALED_EVAL")):
        write_jsonl(out / (name + ".jsonl"), [r for r in materialized if r.get("final_view_membership") == membership])
    context_turn_counts = Counter(len(r.get("context_turn_ids") or []) for r in materialized)
    report = {"run_id": args.out_name, "created_at": datetime.now(timezone.utc).isoformat(), "input_semantic_verified": len(interactions), "source_run": str(source_run), "cache_run": str(cache_run), "materialized": len(materialized), "quarantine": len(quarantine), "quarantine_reasons": dict(Counter(x.get("reason") for x in quarantine)), "dedup_removed": len(removed), "dedup_removed_by_reason": removal_counts(removed), "context_classes": dict(class_counts), "recovery_counts": dict(rescue_counts), "new_audio_discovered_user_turns": sum(1 for t in turn_lookup.values() if str(t["audio_turn_id"]).startswith("audio:new:") and t.get("role") == "user"), "new_audio_discovered_assistant_turns": sum(1 for t in turn_lookup.values() if str(t["audio_turn_id"]).startswith("audio:new:") and t.get("role") == "assistant"), "historical_assistant_rows": sum(1 for r in materialized for m in r.get("messages", [])[:-1] if m.get("role") == "assistant"), "context_turn_count_distribution": {str(k): v for k, v in sorted(context_turn_counts.items())}, "context_buckets": {"1_turn": sum(1 for r in materialized if len(r.get("context_turn_ids") or []) == 0), "2_3_turns": sum(1 for r in materialized if 1 <= len(r.get("context_turn_ids") or []) <= 2), "4_8_turns": sum(1 for r in materialized if 3 <= len(r.get("context_turn_ids") or []) <= 7), "8_20_turns": sum(1 for r in materialized if 7 <= len(r.get("context_turn_ids") or []) <= 19), "20_plus_turns": sum(1 for r in materialized if len(r.get("context_turn_ids") or []) >= 20)}, "diagnostic_rows": len(diagnostics), "authority_policy": "frozen-v2.3 semantic target and split authority; cache-only target rescue; deterministic audio timeline aggregation"}
    (out / "run_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
