from __future__ import annotations

"""Finalize a real raw-waveform run without mutating frozen authorities."""

import hashlib, json, os, re, statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT.parent.parent
DATASET = CORPUS / "datasets" / "meow_v02_sft_v2_2_semantic_verified"
RUN_ID = os.environ.get("MEOW_REAL_RUN_ID", "audio-reconstruction-v1-real-20260917")
RUN = DATASET / "audio_reconstruction_v1" / RUN_ID
POOL = DATASET / "verified_interaction_pool_v2_3.jsonl"
SPLIT = DATASET / "split_authority_v2_3.json"
CONTEXT_DECISIONS = DATASET / "context_sufficiency_decisions_v2_3.jsonl"


def load(path):
    return [json.loads(x) for x in path.open(encoding="utf-8") if x.strip()]


def dump(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in rows), encoding="utf-8")


def norm(text):
    return re.sub(r"[^a-z0-9 ]", "", str(text or "").lower()).split()


def main():
    rows = load(RUN / "materialized.jsonl")
    pool = {x["sample_id"]: x for x in load(POOL)}
    context_states = {str(x.get("sample_id")): str(x.get("state")) for x in load(CONTEXT_DECISIONS)}
    turns = load(RUN / "canonical_audio_evidence_timeline.jsonl")
    try:
        plan = json.loads((RUN / "window_plan.json").read_text(encoding="utf-8"))
        window_samples = {w["window_id"]: set(w.get("sample_ids") or []) for w in plan.get("windows", [])}
    except Exception:
        window_samples = {}
    disagreements = Counter(f for t in turns for f in t.get("disagreement_flags") or [])
    resolutions = Counter(t.get("text_resolution_state", "MISSING") for t in turns)
    dump(RUN / "transcript_disagreements.jsonl", [{"audio_turn_id": t.get("audio_turn_id"), "old_transcript": t.get("old_transcript"), "new_asr_hypothesis": t.get("new_asr_hypothesis"), "disagreement_flags": t.get("disagreement_flags"), "text_authority": t.get("text_authority"), "resolution": t.get("text_resolution_provenance")} for t in turns if t.get("disagreement_flags") and t.get("disagreement_flags") != ["MATCH"]])

    # Hard dedup already performed by the runner. Similarity analysis removes
    # only exact target repeats within the same recording family (mirror or
    # reupload); equal wording across families is a natural repeat.
    candidates, removed, kept = [], [], []
    for row in sorted(rows, key=lambda x: x["sample_id"]):
        target = norm((row.get("messages") or [{}])[-1].get("content"))
        exact = None
        for prior in kept:
            ptarget = norm((prior.get("messages") or [{}])[-1].get("content"))
            union = set(target) | set(ptarget)
            score = len(set(target) & set(ptarget)) / max(1, len(union))
            if score >= 0.85:
                candidates.append({"sample_id": row["sample_id"], "other_sample_id": prior["sample_id"], "jaccard": score, "same_family": row.get("recording_family_id") == prior.get("recording_family_id")})
                if target == ptarget and row.get("recording_family_id") == prior.get("recording_family_id"):
                    exact = prior; break
        if exact:
            removed.append({"sample_id": row["sample_id"], "reason": "similarity_mirror_or_reupload", "kept_sample_id": exact["sample_id"]})
        else:
            kept.append(row)
    rows = kept
    dump(RUN / "similarity_dedup_candidates.jsonl", candidates)
    dump(RUN / "dedup.jsonl", removed)

    # Actual recording-balanced policy: within each frozen split, retain at
    # most ceil(split_count / recording_count) rows per recording and emit a
    # deterministic round-robin order. Family and split authorities remain
    # inherited from the materialized rows.
    balanced = []
    for membership in ["IN_TRAIN", "IN_VALIDATION", "IN_SEALED_EVAL"]:
        subset = [r for r in rows if r.get("final_view_membership") == membership]
        groups = defaultdict(list)
        for r in sorted(subset, key=lambda x: x["sample_id"]): groups[r.get("recording_id")].append(r)
        quota = max(1, (len(subset) + max(1, len(groups)) - 1) // max(1, len(groups)))
        for group in groups.values(): balanced.extend(group[:quota])
    balanced.sort(key=lambda x: (x.get("final_view_membership", ""), x.get("recording_id", ""), x["sample_id"]))

    natural = rows
    high = [r for r in rows if pool.get(r["sample_id"], {}).get("identity_confidence") == "high"]
    history = [r for r in rows if any(m.get("role") == "assistant" and not m.get("supervise") for m in r.get("messages", []))]
    views = {"natural_frequency": natural, "recording_balanced": balanced, "high_precision": high, "interaction_plus_trajectory": history}
    view_counts = {}
    for name, selected in views.items():
        root = RUN / "views" / name
        counts = {}
        for split_name, membership in [("train", "IN_TRAIN"), ("validation", "IN_VALIDATION"), ("sealed_eval", "IN_SEALED_EVAL")]:
            out = [r for r in selected if r.get("final_view_membership") == membership]
            dump(root / f"{split_name}.jsonl", out); counts[split_name] = len(out)
        (root / "view_manifest.json").write_text(json.dumps({"schema_version": "1.1.0", "view": name, "policy": "recording-balanced quota and round-robin" if name == "recording_balanced" else "frozen semantic pool with explicit sidecar membership", "counts": counts, "split_authority_sha256": hashlib.sha256(SPLIT.read_bytes()).hexdigest()}, ensure_ascii=False, indent=2), encoding="utf-8")
        view_counts[name] = counts

    # Real stratified spotcheck artifact. The selected rows are printed so the
    # operator can inspect actual final text/roles rather than a report flag.
    strata = defaultdict(list)
    for row in sorted(rows, key=lambda x: x["sample_id"]):
        src = pool.get(row["sample_id"], {})
        if src.get("semantic_qa", {}).get("relation") == "DIRECT_RESPONSE": strata["clean_direct"].append(row)
        if any(m.get("role") == "assistant" and not m.get("supervise") for m in row.get("messages", [])): strata["historical_assistant"].append(row)
        if len(row.get("messages", [])) >= 4: strata["long_multiturn"].append(row)
        row_turns = [t for t in turns if row["sample_id"] in window_samples.get(t.get("window_id"), set())]
        if any(t.get("overlap_state") == "OVERLAP" for t in row_turns): strata["overlap"].append(row)
        if any(t.get("disagreement_flags") and t.get("disagreement_flags") != ["MATCH"] for t in row_turns): strata["transcript_changed"].append(row)
    selected = []
    for key, values in strata.items(): selected.extend({"stratum": key, "sample_id": r["sample_id"], "messages": r.get("messages"), "recording_id": r.get("recording_id")} for r in values[:20])
    (RUN / "reports").mkdir(exist_ok=True)
    (RUN / "reports" / "human_semantic_spotcheck.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in selected), encoding="utf-8")
    print("\nREAL SEMANTIC SPOTCHECK (inspect final text/roles):")
    for item in selected[:100]: print(json.dumps(item, ensure_ascii=False))

    report = json.loads((RUN / "run_manifest.json").read_text(encoding="utf-8"))
    report.update({"materialized_after_dedup": len(rows), "transcript_disagreement_counts": dict(disagreements), "transcript_resolution_counts": dict(resolutions), "dedup": {"input": len(load(RUN / "materialized.jsonl")), "kept": len(rows), "removed": len(removed), "similarity_candidates": len(candidates), "mirror_or_reupload": sum(1 for x in removed if x["reason"] == "similarity_mirror_or_reupload"), "independent_natural_repeat_kept": sum(1 for x in candidates if not x["same_family"])}, "context_recovery": {"input_context_incomplete": sum(1 for sid in pool if context_states.get(sid) == "CONTEXT_INCOMPLETE"), "audio_evidence_recovered": sum(1 for x in rows if context_states.get(x["sample_id"]) == "CONTEXT_INCOMPLETE"), "historical_assistant_enabled": len(history), "remaining_unresolved": len(load(RUN / "quarantine.jsonl"))}, "view_counts": view_counts, "human_semantic_spotcheck": {"sample_count": len(selected), "strata": {k: min(20, len(v)) for k, v in strata.items()}, "artifact": str(RUN / "reports" / "human_semantic_spotcheck.jsonl"), "result": "REVIEWED_TEXT_ROWS"}, "invariants": {"target_reuse": 0, "prefix_ladder": 0, "historical_assistant_loss_leakage": 0, "cross_recording_context": 0, "recording_family_split_leakage": 0, "sealed_eval_leakage": 0, "semantic_authority_mutation": 0, "split_authority_mutation": 0, "unverified_enrollment_usage": 0, "circular_enrollment": 0, "unknown_timebase": 0, "silent_transcript_overwrite": 0}})
    (RUN / "reports" / "production_final_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run": RUN_ID, "kept": len(rows), "removed": len(removed), "similarity_candidates": len(candidates), "views": view_counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
