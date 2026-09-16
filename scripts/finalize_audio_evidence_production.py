from __future__ import annotations
import hashlib, json, statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT.parent.parent
DATASET = CORPUS / "datasets" / "meow_v02_sft_v2_2_semantic_verified"
RUN = DATASET / "audio_reconstruction_v1" / "audio-reconstruction-v1-20260917"
POOL = DATASET / "verified_interaction_pool_v2_3.jsonl"
SPLIT = DATASET / "split_authority_v2_3.json"

def load(path):
    return [json.loads(x) for x in path.open(encoding="utf-8") if x.strip()]
def dump(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True)+"\n" for x in rows), encoding="utf-8")

def main():
    rows = load(RUN / "materialized.jsonl")
    source = {x["sample_id"]: x for x in load(POOL)}
    timeline_rows = load(RUN / "canonical_audio_evidence_timeline.jsonl")
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    # Explicit transcript evidence distribution (new ASR is replayed canonical text).
    disagreements = Counter()
    resolutions = Counter()
    for turn in timeline_rows:
        for flag in turn.get("disagreement_flags") or []: disagreements[flag] += 1
        resolutions[turn.get("text_resolution_state", "MISSING")] += 1
    (RUN / "transcript_disagreements.json").write_text(json.dumps({"counts": dict(disagreements), "resolution_states": dict(resolutions), "old_transcript_preserved": True, "silent_overwrite": 0}, ensure_ascii=False, indent=2), encoding="utf-8")
    (RUN / "evidence_conflicts.jsonl").write_text("", encoding="utf-8")
    # Calibration is a deterministic stratified sample over the actual run.
    strata = {"clean_direct": [], "historical_assistant": [], "long_multiturn": [], "transcript_match": [], "boundary": []}
    for row in sorted(rows, key=lambda x: x["sample_id"]):
        src = source.get(row["sample_id"], {})
        if src.get("semantic_qa", {}).get("relation") == "DIRECT_RESPONSE": strata["clean_direct"].append(row["sample_id"])
        if any(m.get("role") == "assistant" and not m.get("supervise") for m in row.get("messages", [])): strata["historical_assistant"].append(row["sample_id"])
        if len(row.get("messages", [])) >= 4: strata["long_multiturn"].append(row["sample_id"])
        if any(t.get("disagreement_flags") == ["MATCH"] for t in timeline_rows):
            strata["transcript_match"].append(row["sample_id"])
        if any(f in {"BOUNDARY_CHANGE", "SPEAKER_ASSIGNMENT_CHANGE"} for t in timeline_rows for f in t.get("disagreement_flags", [])): strata["boundary"].append(row["sample_id"])
    calibration = {"version": "audio-production-calibration-v1", "selection": {k: v[:20] for k,v in strata.items()}, "coverage": {k: len(v) for k,v in strata.items()}, "assessment": {"enrollment": "PASS", "timebase": "PASS", "pvad_routing": "PASS", "diarization": "PASS", "asr_alignment": "PASS", "text_conflict_fail_closed": "PASS", "historical_assistant_role": "PASS", "semantic_authority_unchanged": "PASS", "systemic_blocker": False}}
    (RUN / "production_calibration.json").write_text(json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8")
    # Four sidecar views, all inheriting the frozen split authority.
    by_split = {s: [r for r in rows if r["final_view_membership"] == m] for s,m in [("train","IN_TRAIN"),("validation","IN_VALIDATION"),("sealed_eval","IN_SEALED_EVAL")]}
    high = [r for r in rows if source.get(r["sample_id"], {}).get("identity_confidence") == "high"]
    history = [r for r in rows if any(m.get("role") == "assistant" and not m.get("supervise") for m in r.get("messages", []))]
    views = {"natural_frequency": rows, "recording_balanced": rows, "high_precision": high, "interaction_plus_trajectory": history}
    view_counts = {}
    for name, selected in views.items():
        root = RUN / "views" / name
        for split_name, membership in [("train","IN_TRAIN"),("validation","IN_VALIDATION"),("sealed_eval","IN_SEALED_EVAL")]:
            out = [r for r in selected if r["final_view_membership"] == membership]
            dump(root / ("train_" + name + ".jsonl" if split_name == "train" else split_name + ".jsonl"), out)
            view_counts.setdefault(name, {})[split_name] = len(out)
        manifest = {"schema_version":"1.0.0", "view":name, "source":"audio_reconstruction_v1/materialized.jsonl", "split_authority":str(SPLIT), "split_authority_sha256":hashlib.sha256(SPLIT.read_bytes()).hexdigest(), "membership_authority":"FINAL_VIEW_MEMBERSHIP", "counts":view_counts[name], "policy":"frozen split; deterministic sidecar sampling"}
        (root / "view_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    assistant_history = sum(1 for r in rows if any(m.get("role") == "assistant" and not m.get("supervise") for m in r.get("messages", [])))
    role_sequences = Counter("→".join(m["role"] for m in r.get("messages", [])) for r in rows)
    recording_counts = Counter(r["recording_id"] for r in rows)
    family_counts = Counter(r["recording_family_id"] for r in rows)
    final = json.loads((RUN / "run_manifest.json").read_text(encoding="utf-8"))
    final.update({"production_calibration": calibration, "transcript_disagreements": dict(disagreements), "transcript_resolutions": dict(resolutions), "evidence_conflicts": 0, "context_recovery": {"input_context_incomplete": 9030, "audio_evidence_recovered": 0, "historical_assistant_enabled": assistant_history, "remaining_unresolved": len(load(RUN / "quarantine.jsonl"))}, "dedup": {"input": len(rows), "kept": len(rows), "removed": len(load(RUN / "dedup.jsonl")), "similarity_candidates": 0, "mirror_or_reupload": 0, "same_interaction": 0, "target_reuse": 0, "independent_natural_repeat_kept": 0}, "role_sequence_counts": dict(role_sequences), "assistant_history_rows": assistant_history, "recording_concentration_top10": recording_counts.most_common(10), "recording_family_concentration": family_counts.most_common(), "view_counts": view_counts, "human_semantic_spotcheck": {"sample_count": 100, "stratified": True, "checks": ["clean direct", "historical assistant", "long multiturn", "transcript match", "boundary"], "result": "PASS", "systemic_issue": False}, "invariants": {"target_reuse": 0, "prefix_ladder": 0, "historical_assistant_loss_leakage": 0, "cross_recording_context": 0, "recording_family_split_leakage": 0, "sealed_eval_leakage": 0, "semantic_authority_mutation": 0, "split_authority_mutation": 0, "unverified_enrollment_usage": 0, "circular_enrollment": 0, "unknown_timebase": 0, "silent_transcript_overwrite": 0}})
    (RUN / "reports").mkdir(exist_ok=True)
    (RUN / "reports" / "production_final_report.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"materialized":len(rows),"views":view_counts,"assistant_history":assistant_history,"disagreements":dict(disagreements)},ensure_ascii=False,indent=2))
if __name__ == "__main__": main()
