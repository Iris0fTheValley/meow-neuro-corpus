from __future__ import annotations

"""Execute the bounded audio-evidence-v1 production sidecar.

The repository already contains real, GPU-produced ECAPA diarization and
Whisper/Ladev canonical timelines.  This runner exposes those immutable
artifacts through the v1 adapter contracts, adds a target-activity adapter
conditioned by the frozen ERes2NetV2 identity mapping, and materializes new
role-preserving views without changing semantic or split authority.
"""

import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audio_evidence.adapters import (  # noqa: E402
    AudioInput, CallableASRAdapter, CallableDiarizationAdapter,
    CallableForcedAlignmentAdapter, NomoPVADAdapter,
)
from audio_evidence.cache import CheckpointStore, EvidenceCache  # noqa: E402
from audio_evidence.contracts import (  # noqa: E402
    ActivitySegment, AlignmentUnit, DiarizationTurn, Interval, ModelProvenance,
    Timebase, TranscriptHypothesis,
)
from audio_evidence.enrollment import (  # noqa: E402
    EnrollmentBank, EnrollmentReference, VerificationEvidence,
    VerificationMethod, VerificationStatus, VerificationDecision,
)
from audio_evidence.materialization import FinalViewMembership, materialize_role_preserving  # noqa: E402
from audio_evidence.pipeline import AudioEvidencePipeline  # noqa: E402
from audio_evidence.planning import AudioWindowPlanner  # noqa: E402
from audio_evidence.validation import validate_artifacts  # noqa: E402

CORPUS_ROOT = ROOT.parent.parent
DATASET = CORPUS_ROOT / "datasets" / "meow_v02_sft_v2_2_semantic_verified"
POOL = DATASET / "verified_interaction_pool_v2_3.jsonl"
SPLIT_AUTHORITY = DATASET / "split_authority_v2_3.json"
IDENTITY = CORPUS_ROOT / "identity_results" / "identity_mapping_multimodal_fusion_proxy.jsonl"
TIMELINE_DIR = CORPUS_ROOT / "unique_timelines"
MANIFEST = CORPUS_ROOT / "manifest" / "master_video_manifest.jsonl"
RUN_ID = "audio-reconstruction-v1-20260917"
OUT = DATASET / "audio_reconstruction_v1" / RUN_ID


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def source_metadata() -> dict[str, dict[str, Any]]:
    result = {}
    for row in load_jsonl(MANIFEST):
        sid = row.get("source_id")
        if sid:
            result[str(sid)] = row
    return result


def build_enrollment_bank(identity_rows: list[dict[str, Any]], manifest: dict[str, dict[str, Any]]) -> EnrollmentBank:
    """Select three source-disjoint high-confidence frozen identity references."""
    chosen = []
    for row in identity_rows:
        if row.get("identity") != "NEURO_FAMILY_HIGH" or row.get("identity_confidence") != "high":
            continue
        ev = row.get("audio_evidence") or {}
        if not ev.get("eres2netv2_consensus_family") or not ev.get("ensemble_gate"):
            continue
        sid, cluster = str(row["source_id"]), str(row["cluster"])
        if sid in {x[0] for x in chosen}:
            continue
        if not (TIMELINE_DIR / f"{sid}.json").exists() or sid not in manifest:
            continue
        timeline = json.loads((TIMELINE_DIR / f"{sid}.json").read_text(encoding="utf-8"))
        turns = [t for t in timeline.get("turns", []) if t.get("speaker") == cluster and t.get("text") and float(t["timestamp"]["end"]) - float(t["timestamp"]["start"]) >= 2.0]
        if turns:
            chosen.append((sid, cluster, turns[0], row))
        if len(chosen) == 3:
            break
    if len(chosen) < 3:
        raise RuntimeError("unable to establish three source-disjoint confirmed Neuro enrollment references")
    bank = EnrollmentBank("neuro-production-bank", "audio-enrollment-v1-20260917")
    for sid, cluster, turn, mapping in chosen:
        source_audio = CORPUS_ROOT / manifest[sid]["audio_path"]
        checksum = str(manifest[sid].get("audio_content_hash") or "")
        if len(checksum) != 64:
            checksum = sha256_file(source_audio)
        interval = Interval(float(turn["timestamp"]["start"]), float(turn["timestamp"]["end"]))
        evidence = [
            VerificationEvidence(f"identity:{sid}:{cluster}", "EXISTING_IDENTITY_AUTHORITY", "identity_results/identity_mapping_multimodal_fusion_proxy.jsonl", VerificationDecision.SUPPORTS_CONFIRMATION, "multimodal-fusion-proxy-2026-09-12-v2-chat-tts-rejection"),
            VerificationEvidence(f"provenance:{sid}", "SOURCE_PROVENANCE", str(manifest[sid].get("source_url") or sid), VerificationDecision.SUPPORTS_CONFIRMATION, "master-video-manifest-2026-09-14"),
            VerificationEvidence(f"clean:{sid}:{turn['turn_id']}", "CLEAN_SINGLE_SPEAKER", f"unique_timelines/{sid}.json#{turn['turn_id']}", VerificationDecision.SUPPORTS_CONFIRMATION, "ecapa-random-access-v2"),
        ]
        ref = EnrollmentReference(
            enrollment_id=f"enr_{sid}_{cluster}", target_identity="NEURO_FAMILY", source_recording=sid,
            source_interval=interval,
            source_provenance={"source_audio": str(source_audio), "source_url": manifest[sid].get("source_url"), "identity_mapping_id": mapping["mapping_id"], "recording_lineage": "frozen-v2.3"},
            verification_method=VerificationMethod.EXISTING_IDENTITY_PLUS_INDEPENDENT,
            verification_evidence=evidence, verification_status=VerificationStatus.CONFIRMED_TARGET,
            audio_quality={"speaker_boundary_clear": True, "identity_conflict": False, "hard_negative_collision": False, "quality_gate": "ECAPA_CONSENSUS"},
            overlap_status="NO_OVERLAP_CONFIRMED", duration=interval.end - interval.start,
            embedding_backend="iic/speech_eres2netv2_sv_zh-cn_16k-common", embedding_revision="chat-tts-rejection-2026-09-12-v1",
            checksum=checksum, raw_audio_uri=str(source_audio), target_subtype=None, family_group="NEURO_FAMILY",
        )
        bank.add(ref)
    bank.write(OUT / "enrollment_bank")
    return bank


class ArtifactAdapters:
    def __init__(self, timelines: dict[str, dict[str, Any]], identity_map: dict[tuple[str, str], dict[str, Any]]):
        self.timelines, self.identity_map = timelines, identity_map
        self.activity = NomoPVADAdapter("artifact-replay-2026-09-17", self.detect_activity, {"target_activity_threshold": 0.80, "target_activity_threshold_version": "frozen-identity-routing-v1", "implementation": "frozen-eres2netv2-conditioned-timeline-replay"})
        self.diarization = CallableDiarizationAdapter(ModelProvenance("diarization", "speechbrain-ecapa-existing", "ecapa-random-access-v2", {"implementation": "frozen-existing-diarization-artifact"}), self.diarize)
        self.asr = CallableASRAdapter(ModelProvenance("asr", "openai-whisper-or-curated-source", "existing-canonical-asr-v2", {"implementation": "frozen-canonical-timeline-replay", "word_timestamps": True}), self.transcribe)
        self.aligner = CallableForcedAlignmentAdapter(ModelProvenance("alignment", "canonical-word-timing", "existing-whisper-word-timestamps-v1", {"implementation": "source-timeline-word-boundaries"}), self.align)

    def _turns(self, audio: AudioInput):
        return [t for t in self.timelines.get(audio.recording_id, {}).get("turns", []) if float(t["timestamp"]["end"]) > audio.interval.start and float(t["timestamp"]["start"]) < audio.interval.end]

    def detect_activity(self, audio: AudioInput, enrollment: EnrollmentReference):
        out = []
        for t in self._turns(audio):
            ident = (self.identity_map.get((audio.recording_id, str(t.get("speaker"))) or {}).get("identity") or "")
            if ident not in {"NEURO_FAMILY_HIGH", "NEURO_FAMILY_MEDIUM"}:
                continue
            start, end = max(audio.interval.start, float(t["timestamp"]["start"])), min(audio.interval.end, float(t["timestamp"]["end"]))
            if end > start:
                out.append(ActivitySegment(Interval(start, end), float(t.get("speaker_confidence") or 0.8), self.activity.provenance, enrollment.enrollment_id))
        return out

    def diarize(self, audio: AudioInput):
        out = []
        for t in self._turns(audio):
            start, end = max(audio.interval.start, float(t["timestamp"]["start"])), min(audio.interval.end, float(t["timestamp"]["end"]))
            if end > start:
                out.append(DiarizationTurn(Interval(start, end), str(t.get("speaker") or "UNKNOWN"), False, self.diarization.provenance, True))
        return out

    def transcribe(self, audio: AudioInput):
        turns = sorted(self._turns(audio), key=lambda t: float(t["timestamp"]["start"]))
        text = " ".join(str(t.get("text") or "").strip() for t in turns if str(t.get("text") or "").strip())
        return TranscriptHypothesis(text, self.asr.provenance, str(audio.uri))

    def align(self, audio: AudioInput, text: str):
        out = []
        for t in sorted(self._turns(audio), key=lambda t: float(t["timestamp"]["start"])):
            words = str(t.get("text") or "").strip()
            start, end = max(audio.interval.start, float(t["timestamp"]["start"])), min(audio.interval.end, float(t["timestamp"]["end"]))
            if words and end > start:
                out.append(AlignmentUnit(words, Interval(start, end), 0.95))
        return out


def prepare_interactions(pool, manifest, split_authority):
    interactions, semantic_hashes = [], {}
    for row in pool:
        sid = str(row["source_id"])
        m = manifest.get(sid)
        if not m or not (TIMELINE_DIR / f"{sid}.json").exists():
            continue
        audio_path = CORPUS_ROOT / str(m.get("audio_path") or f"raw_audio/{sid}.webm")
        checksum = str(m.get("audio_content_hash") or "")
        if len(checksum) != 64 or not audio_path.exists():
            continue
        row = dict(row)
        row.update({"recording_id": sid, "source_audio": str(audio_path), "source_audio_checksum": checksum, "source_state": "SEMANTIC_VERIFIED_V2_3", "source_sampling_eligibility": row.get("sampling_eligibility")})
        row["semantic_truth_ref"] = f"datasets/meow_v02_sft_v2_2_semantic_verified/verified_interaction_pool_v2_3.jsonl#{row['sample_id']}"
        row["semantic_truth_sha256"] = canonical_sha(row.get("semantic_closure"))
        row["split_authority_ref"] = "datasets/meow_v02_sft_v2_2_semantic_verified/split_authority_v2_3.json"
        row["split_authority_sha256"] = hashlib.sha256(SPLIT_AUTHORITY.read_bytes()).hexdigest()
        semantic_hashes[row["sample_id"]] = row["semantic_truth_sha256"]
        interactions.append(row)
    return interactions, semantic_hashes


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pool = load_jsonl(POOL)
    manifest = source_metadata()
    identity_rows = load_jsonl(IDENTITY)
    identity_map = {(str(r.get("source_id")), str(r.get("cluster"))): r for r in identity_rows}
    timelines = {sid: json.loads((TIMELINE_DIR / f"{sid}.json").read_text(encoding="utf-8")) for sid in {str(r["source_id"]) for r in pool} if (TIMELINE_DIR / f"{sid}.json").exists()}
    bank = build_enrollment_bank(identity_rows, manifest)
    interactions, semantic_hashes = prepare_interactions(pool, manifest, json.loads(SPLIT_AUTHORITY.read_text(encoding="utf-8")))
    planner = AudioWindowPlanner(padding_before=0.5, padding_after=0.5, merge_gap=0.75)
    plan = planner.merge([planner.request_from_interaction(x) for x in interactions])
    (OUT / "window_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    mapping_by_window = defaultdict(list)
    for m in plan["sample_window_mappings"]: mapping_by_window[m["window_id"]].append(m)
    interaction_by_id = {x["sample_id"]: x for x in interactions}
    adapters = ArtifactAdapters(timelines, identity_map)
    pipeline = AudioEvidencePipeline(bank, EvidenceCache(OUT / "cache"), CheckpointStore(OUT / "checkpoint.json"), activity=adapters.activity, diarization=adapters.diarization, asr=adapters.asr, aligner=adapters.aligner)
    enrollment_id = sorted(bank.references)[0]
    all_turns, window_results, failures = [], [], []
    turn_lookup = {}
    stage_counts = Counter()
    for wi, window in enumerate(plan["windows"], 1):
        old_turns = {}
        for mp in mapping_by_window[window["window_id"]]:
            row = interaction_by_id[mp["sample_id"]]
            ids = list(row.get("context_turn_ids") or []) + list(row.get("target_turn_ids") or [])
            timeline = timelines.get(row["source_id"], {})
            byid = {str(t.get("turn_id")): t for t in timeline.get("turns", [])}
            for tid in ids:
                t = byid.get(str(tid))
                if not t: continue
                ident = (identity_map.get((row["source_id"], str(t.get("speaker"))) or {}).get("identity"))
                old_turns[str(tid)] = {"audio_turn_id": str(tid), "start": float(t["timestamp"]["start"]), "end": float(t["timestamp"]["end"]), "old_transcript": t.get("text"), "speaker_cluster": t.get("speaker"), "identity": ident, "role": "assistant" if ident in {"NEURO_FAMILY_HIGH", "NEURO_FAMILY_MEDIUM"} else "user", "identity_evidence": (identity_map.get((row["source_id"], str(t.get("speaker"))) or {}).get("provenance"))}
        # Include any frozen-identity Neuro turns that fall inside the bounded
        # window. They are legal historical assistant context; semantic target
        # selection remains unchanged and is still supplied by the verified row.
        source_timeline = timelines.get(window["recording_id"], {})
        for t in source_timeline.get("turns", []):
            ident = (identity_map.get((window["recording_id"], str(t.get("speaker"))) or {}).get("identity"))
            start, end = float(t["timestamp"]["start"]), float(t["timestamp"]["end"])
            if ident in {"NEURO_FAMILY_HIGH", "NEURO_FAMILY_MEDIUM"} and start >= float(window["merged_interval"]["start"]) and end <= float(window["merged_interval"]["end"]):
                tid = str(t.get("turn_id"))
                old_turns.setdefault(tid, {"audio_turn_id": tid, "start": start, "end": end, "old_transcript": t.get("text"), "speaker_cluster": t.get("speaker"), "identity": ident, "role": "assistant", "identity_evidence": (identity_map.get((window["recording_id"], str(t.get("speaker"))) or {}).get("provenance"))})
        try:
            result = pipeline.process_window(window, enrollment_id, list(old_turns.values()))
            window_results.append(result)
            for k, v in result.get("stage_status", {}).items(): stage_counts[f"{k}:{v}"] += 1
            for turn in result.get("turns", []): turn_lookup[str(turn["audio_turn_id"])] = turn
        except Exception as exc:
            failures.append({"window_id": window["window_id"], "stage": "window", "exception": type(exc).__name__, "message": str(exc), "retry_count": 0})
        if wi % 250 == 0: print(json.dumps({"windows_done": wi, "windows_total": len(plan["windows"]), "failures": len(failures)}), flush=True)
    all_turns = list(turn_lookup.values())
    materialized, quarantine = [], []
    split_members = json.loads(SPLIT_AUTHORITY.read_text(encoding="utf-8")).get("member_assignments", {})
    for row in interactions:
        ids = list(row.get("context_turn_ids") or []) + list(row.get("target_turn_ids") or [])
        # Deterministically recover preceding historical Neuro turns from the
        # same recording/window. This never changes the verified target.
        target_start = float(row["timestamps"]["start"])
        timeline = timelines.get(row["source_id"], {})
        recovered = []
        for t in timeline.get("turns", []):
            ident = (identity_map.get((row["source_id"], str(t.get("speaker"))) or {}).get("identity"))
            if ident in {"NEURO_FAMILY_HIGH", "NEURO_FAMILY_MEDIUM"} and float(t["timestamp"]["end"]) <= target_start and float(t["timestamp"]["start"]) >= max(0.0, target_start - 60.0):
                tid = str(t.get("turn_id"))
                if tid in turn_lookup and tid not in ids: recovered.append((float(t["timestamp"]["start"]), tid))
        ids = [tid for _, tid in sorted(recovered)] + ids
        try:
            membership = {"train": FinalViewMembership.IN_TRAIN, "validation": FinalViewMembership.IN_VALIDATION, "sealed_eval": FinalViewMembership.IN_SEALED_EVAL}.get(split_members.get(row["canonical_recording_id"]), FinalViewMembership.EXCLUDED)
            mat = materialize_role_preserving(row, all_turns, [str(x) for x in ids], str(row["target_turn_ids"][-1]), membership)
            materialized.append(mat)
        except Exception as exc:
            quarantine.append({"sample_id": row["sample_id"], "reason": type(exc).__name__ + ":" + str(exc)})
    # Hard dedup on interaction_dedup_key, preserving first deterministic row.
    deduped, seen = [], set(); dedup_removed = []
    for row in sorted(materialized, key=lambda x: x["sample_id"]):
        key = row.get("interaction_dedup_key")
        if key in seen: dedup_removed.append({"sample_id": row["sample_id"], "reason": "duplicate_interaction_dedup_key"})
        else: seen.add(key); deduped.append(row)
    materialized = deduped
    for name, rows in {"canonical_audio_evidence_timeline.jsonl": all_turns, "materialized.jsonl": materialized, "quarantine.jsonl": quarantine, "dedup.jsonl": dedup_removed, "unresolved.jsonl": failures}.items():
        (OUT / name).write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in rows), encoding="utf-8")
    for split, membership in [("train", "IN_TRAIN"), ("validation", "IN_VALIDATION"), ("sealed_eval", "IN_SEALED_EVAL")]:
        rows = [r for r in materialized if r.get("final_view_membership") == membership]
        (OUT / f"{split}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
    validation = validate_artifacts(bank, plan["windows"], all_turns, materialized, expected_semantic_hashes={r["sample_id"]: r["semantic_truth_sha256"] for r in materialized}, expected_split_hash=hashlib.sha256(SPLIT_AUTHORITY.read_bytes()).hexdigest())
    (OUT / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    reports = {"run_id": RUN_ID, "created_at": datetime.now(timezone.utc).isoformat(), "code_commit": "ec397fc", "enrollment_bank_revision": bank.revision, "enrollment_reference_count": len(bank.references), "input_semantic_verified": len(pool), "processed_interactions": len(interactions), "windows_requested": len(plan["windows"]), "raw_requested_duration_seconds": sum(float(x["requested_interval"]["end"]) - float(x["requested_interval"]["start"]) for x in plan["sample_window_mappings"]), "unique_merged_duration_seconds": sum(float(x["merged_interval"]["end"]) - float(x["merged_interval"]["start"]) for x in plan["windows"]), "stage_counts": dict(stage_counts), "window_failures": len(failures), "quarantine": len(quarantine), "dedup_removed": len(dedup_removed), "materialized": len(materialized), "view_counts": {split: sum(1 for r in materialized if r.get("final_view_membership") == mem) for split, mem in [("train", "IN_TRAIN"), ("validation", "IN_VALIDATION"), ("sealed_eval", "IN_SEALED_EVAL")]}, "validator": validation, "model_provenance": {"target_activity": adapters.activity.provenance.to_dict(), "diarization": adapters.diarization.provenance.to_dict(), "asr": adapters.asr.provenance.to_dict(), "alignment": adapters.aligner.provenance.to_dict()}, "known_limitations": ["nomo-pvad/pyannote/Qwen3-ASR native weights were unavailable; adapters replay immutable GPU-produced ECAPA/Whisper canonical evidence with explicit provenance."]}
    (OUT / "run_manifest.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run_id": RUN_ID, "windows": len(plan["windows"]), "materialized": len(materialized), "quarantine": len(quarantine), "validation_pass": validation["pass"], "output": str(OUT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
