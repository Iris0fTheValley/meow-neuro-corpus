from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audio_evidence.adapters import (  # noqa: E402
    AudioInput, CallableASRAdapter, CallableDiarizationAdapter, CallableEnrollmentEmbeddingAdapter,
    CallableForcedAlignmentAdapter, CallableTargetActivityAdapter, CallableTSEAdapter,
)
from audio_evidence.cache import CheckpointStore, EvidenceCache  # noqa: E402
from audio_evidence.contracts import (  # noqa: E402
    ActivitySegment, AlignmentUnit, ContractError, DiarizationTurn, Interval,
    ModelProvenance, Timebase, TranscriptHypothesis, interval_from_dict, to_recording_interval,
)
from audio_evidence.deduplication import deduplicate_materialized, removal_counts  # noqa: E402
from audio_evidence.enrollment import (  # noqa: E402
    EnrollmentBank, EnrollmentReference, VerificationEvidence, VerificationMethod,
    VerificationStatus, VerificationDecision,
)
from audio_evidence.materialization import (  # noqa: E402
    FinalViewMembership, belongs_to_train, export_prompt_completion, materialize_role_preserving,
)
from audio_evidence.pipeline import AudioEvidencePipeline  # noqa: E402
from audio_evidence.planning import AudioWindowPlanner, WindowRequest  # noqa: E402
from audio_evidence.routing import AmbiguityRouter, RouteReason, TargetActivityPolicy  # noqa: E402
from audio_evidence.transcript import (  # noqa: E402
    TextResolutionState, TranscriptDisagreement, automatic_text_resolution,
    detect_disagreement, resolve_frozen_target_text,
)
from audio_evidence.validation import validate_artifacts  # noqa: E402
from scripts.run_audio_evidence_v1 import build_parser  # noqa: E402


SHA = hashlib.sha256(b"fixture").hexdigest()


def evidence(domain, decision=VerificationDecision.SUPPORTS_CONFIRMATION):
    return VerificationEvidence("ev-" + domain.lower(), domain, "artifact://frozen-v2.3", decision, "frozen-r1")


def confirmed_reference(identity="NEURO_FAMILY", enrollment_id="enr-1", synthetic=False):
    if synthetic:
        method = VerificationMethod.SYNTHETIC_FIXTURE
        items = [evidence("SOURCE_PROVENANCE")]
    else:
        method = VerificationMethod.EXISTING_IDENTITY_PLUS_INDEPENDENT
        items = [evidence("EXISTING_IDENTITY_AUTHORITY"), evidence("SOURCE_PROVENANCE"), evidence("CLEAN_SINGLE_SPEAKER")]
    return EnrollmentReference(
        enrollment_id=enrollment_id, target_identity=identity, source_recording="rec-gold",
        source_interval=Interval(10.0, 14.0), source_provenance={"source_id": "official", "immutable": True},
        verification_method=method, verification_evidence=items,
        verification_status=VerificationStatus.CONFIRMED_TARGET,
        audio_quality={"speaker_boundary_clear": True, "identity_conflict": False, "hard_negative_collision": False},
        overlap_status="NO_OVERLAP_CONFIRMED", duration=4.0, embedding_backend="ERes2NetV2",
        embedding_revision="frozen-r1", checksum=SHA, raw_audio_uri="source://rec-gold#10,14",
        synthetic_test_only=synthetic,
    )


def interaction(sample="s1"):
    return {
        "sample_id": sample, "recording_id": "rec-1", "canonical_recording_id": "canon-1",
        "recording_family_id": "family-1", "source_audio": "audio://rec-1", "target_turn_ids": ["t4"],
        "source_audio_checksum": SHA,
        "timestamps": {"start": 0.0, "end": 4.0, "timebase": "SECONDS_FROM_RECORDING_START"},
        "training_candidate": False, "semantic_truth_ref": "semantic://s1", "semantic_truth_sha256": "semhash",
        "split_authority_ref": "split://v2.3", "split_authority_sha256": "splithash", "provenance": {"source": "v2.3"},
    }


def window_fixture(start=0.0, end=4.0):
    return {
        "window_id": "w1", "recording_id": "rec-1", "canonical_recording_id": "canon-1", "recording_family_id": "family-1",
        "source_audio": "audio://rec-1", "audio_checksum": SHA, "merged_interval": Interval(start, end).to_dict(),
    }


def timeline(overlap=False):
    result = []
    for turn_id, role, identity, text in (("t1", "user", "GUEST", "A"), ("t2", "assistant", "NEURO_FAMILY", "B"), ("t3", "user", "GUEST", "C"), ("t4", "assistant", "NEURO_FAMILY", "D")):
        result.append({
            "audio_turn_id": turn_id, "recording_id": "rec-1", "canonical_recording_id": "canon-1", "recording_family_id": "family-1",
            "role": role, "identity": identity, "old_transcript": text, "new_asr_hypothesis": None,
            "disagreement_flags": [], "text_resolution_state": "RESOLVED", "text_authority": "OLD_TRANSCRIPT",
            "resolved_text": text, "text_resolution_provenance": {"status": "RESOLVED", "resolver": "EXISTING_TRANSCRIPT_AUTHORITY", "revision": "fixture"},
        })
    return result


class EnrollmentTests(unittest.TestCase):
    def test_confirmed_independent_reference_enters_bank(self):
        bank = EnrollmentBank("neuro-bank", "r1")
        bank.add(confirmed_reference())
        self.assertEqual(bank.confirmed("enr-1").target_identity, "NEURO_FAMILY")

    def test_rejected_and_unverified_cannot_enter_bank(self):
        for status in (VerificationStatus.REJECTED, VerificationStatus.UNVERIFIED):
            value = confirmed_reference()
            object.__setattr__(value, "verification_status", status)
            with self.assertRaises(ContractError):
                EnrollmentBank("b", "r").add(value)

    def test_circular_enrollment_rejected(self):
        with self.assertRaises(ContractError):
            evidence("PVAD")

    def test_weak_identity_only_confirmation_rejected(self):
        with self.assertRaises(ContractError):
            EnrollmentReference(
                enrollment_id="bad", target_identity="NEURO_FAMILY", source_recording="r", source_interval=Interval(0, 1),
                source_provenance={"x": 1}, verification_method=VerificationMethod.EXISTING_IDENTITY_PLUS_INDEPENDENT,
                verification_evidence=[evidence("EXISTING_IDENTITY_AUTHORITY")], verification_status=VerificationStatus.CONFIRMED_TARGET,
                audio_quality={"speaker_boundary_clear": True}, overlap_status="NO_OVERLAP_CONFIRMED", duration=1,
                embedding_backend="ERes2NetV2", embedding_revision="r", checksum=SHA, raw_audio_uri="audio://bad")

    def test_negative_or_inconclusive_evidence_cannot_confirm(self):
        for decision in (VerificationDecision.REJECTS_CONFIRMATION, VerificationDecision.INCONCLUSIVE):
            with self.assertRaises(ContractError):
                EnrollmentReference(
                    enrollment_id="negative", target_identity="NEURO_FAMILY", source_recording="r", source_interval=Interval(0, 1),
                    source_provenance={"x": 1}, verification_method=VerificationMethod.EXISTING_IDENTITY_PLUS_INDEPENDENT,
                    verification_evidence=[evidence("EXISTING_IDENTITY_AUTHORITY", decision), evidence("SOURCE_PROVENANCE"), evidence("CLEAN_SINGLE_SPEAKER")],
                    verification_status=VerificationStatus.CONFIRMED_TARGET,
                    audio_quality={"speaker_boundary_clear": True, "identity_conflict": False, "hard_negative_collision": False},
                    overlap_status="NO_OVERLAP_CONFIRMED", duration=1, embedding_backend="ERes2NetV2", embedding_revision="r",
                    checksum=SHA, raw_audio_uri="audio://negative")

    def test_contaminated_and_guest_enrollment_fail_closed(self):
        value = confirmed_reference(identity="GUEST")
        with self.assertRaises(ContractError):
            EnrollmentBank("b", "r").add(value)
        with self.assertRaises(ContractError):
            EnrollmentReference(
                enrollment_id="overlap", target_identity="NEURO_FAMILY", source_recording="r", source_interval=Interval(0, 1),
                source_provenance={"x": 1}, verification_method=VerificationMethod.HUMAN_GOLD,
                verification_evidence=[evidence("HUMAN_GOLD")], verification_status=VerificationStatus.CONFIRMED_TARGET,
                audio_quality={"speaker_boundary_clear": True}, overlap_status="UNRESOLVED", duration=1,
                embedding_backend="e", embedding_revision="r", checksum=SHA, raw_audio_uri="audio://bad")

    def test_embedding_production_is_confirmed_only_and_traceable(self):
        bank = EnrollmentBank("bank", "r1")
        bank.add(confirmed_reference())
        producer = CallableEnrollmentEmbeddingAdapter(ModelProvenance("enrollment_embedding", "ERes2NetV2", "frozen-r1"), lambda ref: {"embedding_uri": "embeddings/enr-1.bin", "embedding_checksum": SHA})
        result = bank.produce_embedding("enr-1", producer)
        self.assertEqual(result["source_audio_checksum"], SHA)
        self.assertEqual(result["source_interval"]["start"], 10.0)
        synthetic = EnrollmentBank("test", "r1")
        synthetic.add(confirmed_reference(synthetic=True))
        with self.assertRaises(ContractError):
            synthetic.produce_embedding("enr-1", producer)


class PlannerAndCacheTests(unittest.TestCase):
    def test_interval_merge_and_reversible_mapping(self):
        planner = AudioWindowPlanner(merge_gap=0.5)
        a = planner.request_from_interaction(interaction("s1"))
        second = interaction("s2")
        second["timestamps"] = {"start": 4.4, "end": 6.0, "timebase": "SECONDS_FROM_RECORDING_START"}
        result = planner.merge([planner.request_from_interaction(second), a])
        self.assertEqual(len(result["windows"]), 1)
        self.assertEqual({item["sample_id"] for item in result["sample_window_mappings"]}, {"s1", "s2"})

    def test_no_cross_recording_merge(self):
        planner = AudioWindowPlanner(merge_gap=10)
        a = planner.request_from_interaction(interaction("s1"))
        other = interaction("s2")
        other.update({"recording_id": "rec-2", "canonical_recording_id": "canon-2", "source_audio": "audio://rec-2"})
        result = planner.merge([a, planner.request_from_interaction(other)])
        self.assertEqual(len(result["windows"]), 2)

    def test_unknown_timebase_and_reversal_fail(self):
        bad = interaction()
        bad["timestamps"] = {"start": 2, "end": 1, "timebase": "SECONDS_FROM_RECORDING_START"}
        with self.assertRaises(ContractError):
            AudioWindowPlanner().request_from_interaction(bad)
        bad["timestamps"] = {"start": 1, "end": 2, "timebase": "FRAMES"}
        with self.assertRaises(ContractError):
            AudioWindowPlanner().request_from_interaction(bad)

    def test_cache_key_invalidation_and_atomic_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = EvidenceCache(Path(directory))
            key1 = cache.key(SHA, Interval(0, 1).to_dict(), "model-r1", {"x": 1}, "enr-r1")
            key2 = cache.key(SHA, Interval(0, 1).to_dict(), "model-r2", {"x": 1}, "enr-r1")
            self.assertNotEqual(key1, key2)
            threshold_key = cache.key(SHA, Interval(0, 1).to_dict(), "model-r1", {"target_activity_threshold": 0.8, "target_activity_threshold_version": "frozen-v1"}, "enr-r1")
            other_threshold_key = cache.key(SHA, Interval(0, 1).to_dict(), "model-r1", {"target_activity_threshold": 0.9, "target_activity_threshold_version": "frozen-v1"}, "enr-r1")
            self.assertNotEqual(threshold_key, other_threshold_key)
            calls = []
            self.assertEqual(cache.get_or_compute("asr", key1, lambda: calls.append(1) or {"x": 1})[1], False)
            self.assertEqual(cache.get_or_compute("asr", key1, lambda: calls.append(2) or {"x": 2})[1], True)
            self.assertEqual(calls, [1])

    def test_concurrent_cache_miss_computes_expensive_value_once(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = EvidenceCache(Path(directory), lock_wait_seconds=2.0)
            key = cache.key(SHA, Interval(0, 1).to_dict(), "model-r1", {}, "enr-r1")
            barrier = threading.Barrier(2)
            calls = 0
            calls_lock = threading.Lock()
            results = []

            def compute():
                nonlocal calls
                with calls_lock:
                    calls += 1
                time.sleep(0.05)
                return {"value": "computed"}

            def worker():
                barrier.wait()
                results.append(cache.get_or_compute("asr", key, compute))

            threads = [threading.Thread(target=worker) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(calls, 1)
            self.assertEqual(sorted(hit for _, hit in results), [False, True])
            self.assertTrue(all(value == {"value": "computed"} for value, _ in results))

    def test_stale_cache_and_checkpoint_locks_recover_but_live_lock_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = EvidenceCache(root / "cache", stale_lock_seconds=0)
            key = cache.key(SHA, Interval(0, 1).to_dict(), "r1", {}, "enr-r1")
            stage = root / "cache" / "asr"
            stage.mkdir(parents=True)
            stale = stage / (key + ".lockdir")
            stale.mkdir()
            (stale / "owner.json").write_text(json.dumps({"pid": 99999999, "created_at": 0, "purpose": "crashed"}), encoding="utf-8")
            cache.put("asr", key, {"ok": True})
            self.assertEqual(cache.get("asr", key), {"ok": True})

            checkpoint_path = root / "checkpoint.json"
            checkpoint_lock = checkpoint_path.with_suffix(".json.lockdir")
            checkpoint_lock.mkdir()
            (checkpoint_lock / "owner.json").write_text(json.dumps({"pid": 99999999, "created_at": 0, "purpose": "crashed"}), encoding="utf-8")
            checkpoint = CheckpointStore(checkpoint_path, stale_lock_seconds=0)
            checkpoint.record("w1", "asr", key)
            self.assertEqual(checkpoint.load()["completed"]["w1"]["asr"], key)

            live = stage / ("live.lockdir")
            live.mkdir()
            (live / "owner.json").write_text(json.dumps({"pid": os.getpid(), "created_at": 0, "purpose": "live"}), encoding="utf-8")
            with self.assertRaises(ContractError):
                from audio_evidence.cache import _acquire_lock
                _acquire_lock(live, 0, "contender")


class RoutingAndTranscriptTests(unittest.TestCase):
    def setUp(self):
        self.model = ModelProvenance("diarization", "mock", "r1")

    def test_clean_default_and_overlap_tse_route(self):
        clean = [DiarizationTurn(Interval(0, 1), "S0", False, self.model)]
        self.assertEqual(AmbiguityRouter().decide([], clean).reason_code, RouteReason.CLEAN_SINGLE_SPEAKER)
        overlap = [DiarizationTurn(Interval(0, 1), "S0", True, self.model)]
        activity_model = ModelProvenance("target_activity", "mock", "r1")
        activity = [ActivitySegment(Interval(0.2, 0.8), 0.9, activity_model, "enr-1")]
        self.assertEqual(AmbiguityRouter(TargetActivityPolicy(0.8, "fixture-v1")).decide(activity, overlap).reason_code, RouteReason.TARGET_OVERLAP)

    def test_sequential_speakers_do_not_trigger_tse(self):
        activity_model = ModelProvenance("target_activity", "mock", "r1")
        activity = [ActivitySegment(Interval(2.0, 3.0), 0.9, activity_model, "enr-1")]
        sequential = [
            DiarizationTurn(Interval(0, 1), "S0", False, self.model),
            DiarizationTurn(Interval(2, 3), "S1", False, self.model),
        ]
        decision = AmbiguityRouter(TargetActivityPolicy(0.8, "fixture-v1")).decide(activity, sequential)
        self.assertFalse(decision.use_tse)
        self.assertEqual(decision.reason_code, RouteReason.CLEAN_SINGLE_SPEAKER)

    def test_threshold_filters_low_probability_but_routes_high_confidence_overlap(self):
        activity_model = ModelProvenance("target_activity", "mock", "r1")
        overlap = [DiarizationTurn(Interval(0, 1), "S0", True, self.model)]
        router = AmbiguityRouter(TargetActivityPolicy(0.8, "frozen-routing-v1"))
        low = router.decide([ActivitySegment(Interval(0.2, 0.8), 0.2, activity_model, "enr-1")], overlap)
        high = router.decide([ActivitySegment(Interval(0.2, 0.8), 0.9, activity_model, "enr-1")], overlap)
        self.assertFalse(low.use_tse)
        self.assertEqual(high.reason_code, RouteReason.TARGET_OVERLAP)
        self.assertEqual(high.to_dict()["target_activity_policy"], {"threshold": 0.8, "version": "frozen-routing-v1"})

    def test_activity_without_explicit_threshold_fails_closed(self):
        activity_model = ModelProvenance("target_activity", "mock", "r1")
        with self.assertRaisesRegex(ContractError, "explicit versioned"):
            AmbiguityRouter().decide([ActivitySegment(Interval(0, 1), 0.9, activity_model, "enr-1")], [])
        with self.assertRaises(ContractError):
            TargetActivityPolicy(0.0, "invalid-zero-threshold")

    def test_timebase_conversion_and_unknown_timebase_fail_closed(self):
        planned = Interval(1800, 1860)
        self.assertEqual(to_recording_interval(Interval(5, 10, Timebase.WINDOW_LOCAL_SECONDS), planned), Interval(1805, 1810))
        with self.assertRaises(ContractError):
            to_recording_interval(Interval(59, 61, Timebase.WINDOW_LOCAL_SECONDS), planned)
        with self.assertRaises(ContractError):
            interval_from_dict({"start": 5, "end": 10, "timebase": "FRAMES"})

    def test_transcript_disagreement_classes(self):
        self.assertEqual(detect_disagreement("hello there", "hello there"), TranscriptDisagreement.MATCH)
        self.assertEqual(detect_disagreement("", "hello"), TranscriptDisagreement.MISSING_OLD_SPEECH)
        self.assertEqual(detect_disagreement("hello", ""), TranscriptDisagreement.MISSING_NEW_SPEECH)
        self.assertEqual(detect_disagreement("one two three", "completely different words"), TranscriptDisagreement.MAJOR_TRANSCRIPT_CONFLICT)
        self.assertEqual(detect_disagreement("x", "y", boundary_changed=True), TranscriptDisagreement.BOUNDARY_CHANGE)


class CliContractTests(unittest.TestCase):
    def test_cli_advertises_only_implemented_bounded_plan_command(self):
        parser = build_parser()
        subparsers = next(action for action in parser._actions if hasattr(action, "choices") and action.choices)
        self.assertEqual(set(subparsers.choices), {"plan"})
        self.assertNotIn("validate", parser.format_help().lower())


class MaterializationTests(unittest.TestCase):
    def test_guest_target_multiple_guests_and_multiple_assistant_history(self):
        values = [
            {"audio_turn_id": "a1", "recording_id": "rec-1", "role": "user", "identity": "GUEST_A", "old_transcript": "guest one"},
            {"audio_turn_id": "a2", "recording_id": "rec-1", "role": "assistant", "identity": "NEURO_FAMILY", "old_transcript": "history one"},
            {"audio_turn_id": "a3", "recording_id": "rec-1", "role": "user", "identity": "GUEST_B", "old_transcript": "guest two"},
            {"audio_turn_id": "a4", "recording_id": "rec-1", "role": "assistant", "identity": "NEURO_FAMILY", "old_transcript": "history two"},
            {"audio_turn_id": "a5", "recording_id": "rec-1", "role": "user", "identity": "GUEST_A", "old_transcript": "guest three"},
            {"audio_turn_id": "a6", "recording_id": "rec-1", "role": "assistant", "identity": "NEURO_FAMILY", "old_transcript": "target"},
        ]
        for value in values:
            value.update({
                "canonical_recording_id": "canon-1", "recording_family_id": "family-1", "disagreement_flags": [],
                "text_resolution_state": "RESOLVED", "text_authority": "OLD_TRANSCRIPT", "resolved_text": value["old_transcript"],
                "text_resolution_provenance": {"status": "RESOLVED", "resolver": "EXISTING_TRANSCRIPT_AUTHORITY", "revision": "fixture"},
            })
        row = materialize_role_preserving(interaction(), values, ["a1", "a2", "a3", "a4", "a5", "a6"], "a6", FinalViewMembership.IN_TRAIN)
        self.assertEqual([item["role"] for item in row["messages"]], ["user", "assistant", "user", "assistant", "user", "assistant"])
        self.assertEqual(sum(bool(item["supervise"]) for item in row["messages"]), 1)

    def test_role_preservation_and_loss_mask(self):
        row = materialize_role_preserving(interaction(), timeline(), ["t1", "t2", "t3", "t4"], "t4", FinalViewMembership.IN_TRAIN)
        self.assertEqual([m["role"] for m in row["messages"]], ["user", "assistant", "user", "assistant"])
        self.assertEqual([m["supervise"] for m in row["messages"]], [False, False, False, True])
        exported = export_prompt_completion(row)
        self.assertEqual(exported["historical_assistant_loss_leakage"], 0)
        self.assertEqual(exported["prompt_messages"][1]["role"], "assistant")
        self.assertTrue(belongs_to_train(row))

    def test_neuro_history_can_start_context(self):
        row = materialize_role_preserving(interaction(), timeline(), ["t2", "t3", "t4"], "t4", FinalViewMembership.IN_TRAIN)
        self.assertEqual(row["messages"][0]["role"], "assistant")
        self.assertFalse(row["messages"][0]["supervise"])

    def test_cross_recording_and_duplicate_target_fail(self):
        values = timeline()
        values[0]["recording_id"] = "other"
        with self.assertRaises(ContractError):
            materialize_role_preserving(interaction(), values, ["t1", "t4"], "t4", FinalViewMembership.IN_TRAIN)
        with self.assertRaises(ContractError):
            materialize_role_preserving(interaction(), timeline(), ["t1", "t4", "t4"], "t4", FinalViewMembership.IN_TRAIN)

    def test_all_selected_turns_from_wrong_recording_fail(self):
        values = [dict(turn, recording_id="wrong-rec", canonical_recording_id="wrong-canon", recording_family_id="wrong-family") for turn in timeline()]
        with self.assertRaisesRegex(ContractError, "does not match interaction"):
            materialize_role_preserving(interaction(), values, ["t1", "t4"], "t4", FinalViewMembership.IN_TRAIN)

    def test_unresolved_transcript_conflicts_fail_closed(self):
        for flag in ("MAJOR_TRANSCRIPT_CONFLICT", "BOUNDARY_CHANGE", "SPEAKER_ASSIGNMENT_CHANGE", "MINOR_TEXT_CHANGE"):
            values = timeline()
            values[-1].update({
                "new_asr_hypothesis": "completely different", "disagreement_flags": [flag],
                "text_resolution_state": "UNRESOLVED", "text_authority": None, "resolved_text": None,
                "text_resolution_provenance": None,
            })
            with self.subTest(flag=flag), self.assertRaisesRegex(ContractError, "unresolved"):
                materialize_role_preserving(interaction(), values, ["t3", "t4"], "t4", FinalViewMembership.IN_TRAIN)

    def test_no_synthetic_prompt_for_missing_text(self):
        values = timeline()
        values[0]["old_transcript"] = ""
        values[0]["resolved_text"] = ""
        with self.assertRaises(ContractError):
            materialize_role_preserving(interaction(), values, ["t1", "t4"], "t4", FinalViewMembership.IN_TRAIN)


class IntegrationSmokeTests(unittest.TestCase):
    def adapters(self, overlap=False, counts=None):
        counts = counts if counts is not None else {}
        def count(name):
            counts[name] = counts.get(name, 0) + 1
        activity_model = ModelProvenance("target_activity", "nomo-pvad-mock", "r1", {"target_activity_threshold": 0.8, "target_activity_threshold_version": "fixture-v1"})
        diar_model = ModelProvenance("diarization", "pyannote-community-1-mock", "r1")
        asr_model = ModelProvenance("asr", "qwen3-asr-mock", "r1")
        align_model = ModelProvenance("alignment", "qwen3-aligner-mock", "r1")
        tse_model = ModelProvenance("tse", "wesep-mock", "r1")
        activity = CallableTargetActivityAdapter(activity_model, lambda audio, enr: (count("activity") or [ActivitySegment(Interval(1, 2), 0.99, activity_model, enr.enrollment_id)]))
        diar = CallableDiarizationAdapter(diar_model, lambda audio: (count("diar") or [DiarizationTurn(Interval(0, 4), "S0", overlap, diar_model)]))
        tse = CallableTSEAdapter(tse_model, lambda audio, enr: AudioInput("audio://extracted", hashlib.sha256(b"extract").hexdigest(), audio.recording_id, audio.interval))
        asr = CallableASRAdapter(asr_model, lambda audio: TranscriptHypothesis("new transcript", asr_model, audio.uri))
        align = CallableForcedAlignmentAdapter(align_model, lambda audio, text: [AlignmentUnit("new", Interval(0, 1), 0.9)])
        return activity, diar, tse, asr, align

    def test_mock_end_to_end_clean_and_cache_resume(self):
        bank = EnrollmentBank("test-bank", "test-r1")
        bank.add(confirmed_reference(synthetic=True))
        counts = {}
        activity, diar, tse, asr, align = self.adapters(counts=counts)
        with tempfile.TemporaryDirectory() as directory:
            pipeline = AudioEvidencePipeline(bank, EvidenceCache(Path(directory) / "cache"), CheckpointStore(Path(directory) / "checkpoint.json"), activity=activity, diarization=diar, tse=tse, asr=asr, aligner=align, allow_synthetic=True)
            window = window_fixture()
            old = [{"audio_turn_id": "t4", "role": "assistant", "identity": "NEURO_FAMILY", "old_transcript": "old transcript", "start": 0, "end": 4}]
            first = pipeline.process_window(window, "enr-1", old)
            second = pipeline.process_window(window, "enr-1", old)
            self.assertEqual(first["route"]["reason_code"], "CLEAN_SINGLE_SPEAKER")
            self.assertEqual(second["cache_hits"], {"target_activity": True, "diarization": True, "asr": True, "alignment": True})
            self.assertEqual(counts, {"activity": 1, "diar": 1})
            self.assertEqual(first["turns"][0]["old_transcript"], "old transcript")
            self.assertEqual(first["turns"][0]["new_asr_hypothesis"], "new transcript")

    def test_planner_window_artifact_is_direct_pipeline_input(self):
        bank = EnrollmentBank("test-bank", "test-r1")
        bank.add(confirmed_reference(synthetic=True))
        observed = []
        activity_model = ModelProvenance("target_activity", "mock", "r1", {"target_activity_threshold": 0.8, "target_activity_threshold_version": "fixture-v1"})
        activity = CallableTargetActivityAdapter(activity_model, lambda audio, enr: observed.append(audio.interval.to_dict()) or [ActivitySegment(audio.interval, 0.99, activity_model, enr.enrollment_id)])
        planner = AudioWindowPlanner()
        plan = planner.merge([planner.request_from_interaction(interaction())])
        planned_window = plan["windows"][0]
        with tempfile.TemporaryDirectory() as directory:
            pipeline = AudioEvidencePipeline(bank, EvidenceCache(Path(directory) / "cache"), CheckpointStore(Path(directory) / "cp.json"), activity=activity, diarization=None, tse=None, asr=None, aligner=None, allow_synthetic=True)
            result = pipeline.process_window(planned_window, "enr-1", [{"audio_turn_id": "t4", "old_transcript": "old", "start": 0, "end": 4}])
            with self.assertRaisesRegex(ValueError, "inside the exact planned"):
                pipeline.process_window(planned_window, "enr-1", [{"audio_turn_id": "bad", "old_transcript": "outside", "start": -1, "end": 4}])
        self.assertEqual(observed, [planned_window["merged_interval"]])
        self.assertEqual(result["turns"][0]["source_interval"], planned_window["merged_interval"])

    def test_overlap_routes_tse_with_enrollment_provenance(self):
        bank = EnrollmentBank("test-bank", "test-r1")
        bank.add(confirmed_reference(synthetic=True))
        activity, diar, tse, asr, align = self.adapters(overlap=True)
        with tempfile.TemporaryDirectory() as directory:
            pipeline = AudioEvidencePipeline(bank, EvidenceCache(Path(directory) / "cache"), CheckpointStore(Path(directory) / "cp.json"), activity=activity, diarization=diar, tse=tse, asr=asr, aligner=align, allow_synthetic=True)
            window = window_fixture()
            result = pipeline.process_window(window, "enr-1", [{"audio_turn_id": "t4", "old_transcript": "old", "start": 0, "end": 4}])
            self.assertEqual(result["route"]["reason_code"], "TARGET_OVERLAP")
            self.assertEqual(result["turns"][0]["tse_source"]["enrollment_id"], "enr-1")

    def test_window_local_adapter_times_are_normalized_to_recording_global(self):
        bank = EnrollmentBank("test-bank", "test-r1")
        bank.add(confirmed_reference(synthetic=True))
        activity_model = ModelProvenance("target_activity", "mock-pvad", "r1", {"target_activity_threshold": 0.8, "target_activity_threshold_version": "frozen-fixture-v1"})
        diar_model = ModelProvenance("diarization", "mock-diar", "r1")
        asr_model = ModelProvenance("asr", "mock-asr", "r1")
        align_model = ModelProvenance("alignment", "mock-align", "r1")
        local = Interval(5, 10, Timebase.WINDOW_LOCAL_SECONDS)
        activity = CallableTargetActivityAdapter(activity_model, lambda audio, enr: [ActivitySegment(local, 0.95, activity_model, enr.enrollment_id)])
        diar = CallableDiarizationAdapter(diar_model, lambda audio: [DiarizationTurn(local, "S0", False, diar_model)])
        asr = CallableASRAdapter(asr_model, lambda audio: TranscriptHypothesis("bounded words", asr_model, audio.uri))
        align = CallableForcedAlignmentAdapter(align_model, lambda audio, text: [AlignmentUnit("bounded words", local, 0.9)])
        window = window_fixture(1800, 1860)
        with tempfile.TemporaryDirectory() as directory:
            pipeline = AudioEvidencePipeline(bank, EvidenceCache(Path(directory) / "cache"), CheckpointStore(Path(directory) / "cp.json"), activity=activity, diarization=diar, tse=None, asr=asr, aligner=align, allow_synthetic=True)
            result = pipeline.process_window(window, "enr-1", [{"audio_turn_id": "t4", "old_transcript": "bounded words", "start": 1800, "end": 1860}])
        turn = result["turns"][0]
        for field in ("target_activity_evidence", "diarization_evidence", "alignment"):
            self.assertEqual((turn[field][0]["start"], turn[field][0]["end"]), (1805.0, 1810.0), field)
            self.assertEqual(turn[field][0]["timebase"], "SECONDS_FROM_RECORDING_START")
            self.assertEqual(turn[field][0]["source_timebase"], "WINDOW_LOCAL_SECONDS")
            self.assertEqual(turn[field][0]["timebase_conversion"]["offset_seconds"], 1800.0)

    def test_pipeline_rejects_adapter_interval_outside_bounded_window(self):
        bank = EnrollmentBank("test-bank", "test-r1")
        bank.add(confirmed_reference(synthetic=True))
        activity_model = ModelProvenance("target_activity", "mock", "r1", {"target_activity_threshold": 0.8, "target_activity_threshold_version": "fixture-v1"})
        activity = CallableTargetActivityAdapter(activity_model, lambda audio, enr: [ActivitySegment(Interval(59, 61, Timebase.WINDOW_LOCAL_SECONDS), 0.9, activity_model, enr.enrollment_id)])
        with tempfile.TemporaryDirectory() as directory:
            pipeline = AudioEvidencePipeline(bank, EvidenceCache(Path(directory) / "cache"), CheckpointStore(Path(directory) / "cp.json"), activity=activity, diarization=None, tse=None, asr=None, aligner=None, allow_synthetic=True)
            with self.assertRaisesRegex(ContractError, "outside"):
                pipeline.process_window(window_fixture(1800, 1860), "enr-1", [{"audio_turn_id": "t4", "old_transcript": "old", "start": 1800, "end": 1860}])

    def test_validator_accepts_complete_role_preserving_chain(self):
        bank = EnrollmentBank("bank", "r1")
        bank.add(confirmed_reference())
        window = window_fixture()
        evidence_turns = []
        for index, source in enumerate(timeline()):
            evidence_turns.append({
                "schema_version": "1.0.0", "recording_id": "rec-1", "audio_turn_id": source["audio_turn_id"], "window_id": "w1",
                "canonical_recording_id": "canon-1", "recording_family_id": "family-1",
                "start": float(index), "end": float(index + 1), "timebase": "SECONDS_FROM_RECORDING_START", "role": source["role"], "identity": source["identity"],
                "source_interval": Interval(0, 4).to_dict(),
                "old_transcript": source["old_transcript"], "new_asr_hypothesis": source["old_transcript"], "alignment": None,
                "speaker_cluster": "S0", "target_activity_evidence": [{"start": float(index), "end": float(index + 1), "timebase": "SECONDS_FROM_RECORDING_START", "source_timebase": "SECONDS_FROM_RECORDING_START", "timebase_conversion": {"version": "window-local-to-recording-v1", "offset_seconds": 0.0}}], "diarization_evidence": [], "enrollment_provenance": {"enrollment_id": "enr-1"},
                "model_provenance": {"diarization": {"revision": "r1"}}, "asr_source": {"audio_checksum": SHA},
                "disagreement_flags": ["MATCH"], "old_transcript_overwritten": False,
                "text_resolution_state": "RESOLVED", "text_authority": "NEW_ASR", "resolved_text": source["old_transcript"],
                "text_resolution_provenance": {"status": "RESOLVED", "resolver": "EXACT_OLD_NEW_MATCH", "revision": "fixture"},
            })
        row = materialize_role_preserving(interaction(), evidence_turns, ["t1", "t2", "t3", "t4"], "t4", FinalViewMembership.IN_TRAIN)
        report = validate_artifacts(bank, [window], evidence_turns, [row], expected_semantic_hashes={"s1": "semhash"}, expected_split_hash="splithash")
        self.assertTrue(report["pass"], report["errors"])

    def test_validator_detects_loss_leakage_and_semantic_mutation(self):
        bank = EnrollmentBank("bank", "r1")
        bank.add(confirmed_reference())
        row = materialize_role_preserving(interaction(), timeline(), ["t1", "t2", "t3", "t4"], "t4", FinalViewMembership.IN_TRAIN)
        row["messages"][1]["supervise"] = True
        report = validate_artifacts(bank, [], [], [row], expected_semantic_hashes={"s1": "different"})
        self.assertEqual(report["gates"]["HISTORICAL_ASSISTANT_LOSS_LEAKAGE"], "FAIL")
        self.assertEqual(report["gates"]["SEMANTIC_TRUTH_UNCHANGED"], "FAIL")

    def test_missing_expected_authority_hashes_are_not_checked_and_block_pass(self):
        bank = EnrollmentBank("bank", "r1")
        bank.add(confirmed_reference())
        row = materialize_role_preserving(interaction(), timeline(), ["t3", "t4"], "t4", FinalViewMembership.IN_TRAIN)
        report = validate_artifacts(bank, [], [], [row])
        self.assertEqual(report["gates"]["SEMANTIC_TRUTH_UNCHANGED"], "NOT_CHECKED")
        self.assertEqual(report["gates"]["SPLIT_AUTHORITY_PRESERVED"], "NOT_CHECKED")
        self.assertFalse(report["pass"])

    def test_graceful_degradation_preserves_old_text(self):
        bank = EnrollmentBank("test-bank", "test-r1")
        bank.add(confirmed_reference(synthetic=True))
        with tempfile.TemporaryDirectory() as directory:
            pipeline = AudioEvidencePipeline(bank, EvidenceCache(Path(directory) / "cache"), CheckpointStore(Path(directory) / "cp.json"), activity=None, diarization=None, tse=None, asr=None, aligner=None, allow_synthetic=True)
            window = window_fixture()
            result = pipeline.process_window(window, "enr-1", [{"audio_turn_id": "t4", "old_transcript": "keep me", "start": 0, "end": 4}])
            self.assertEqual(result["turns"][0]["old_transcript"], "keep me")
            self.assertIsNone(result["turns"][0]["new_asr_hypothesis"])
            self.assertIsNone(result["turns"][0]["alignment"])
            self.assertEqual(result["stage_status"]["asr"], "UNAVAILABLE")

    def test_overlap_without_tse_is_explicitly_unresolved(self):
        bank = EnrollmentBank("test-bank", "test-r1")
        bank.add(confirmed_reference(synthetic=True))
        activity, diar, _, asr, align = self.adapters(overlap=True)
        with tempfile.TemporaryDirectory() as directory:
            pipeline = AudioEvidencePipeline(bank, EvidenceCache(Path(directory) / "cache"), CheckpointStore(Path(directory) / "cp.json"), activity=activity, diarization=diar, tse=None, asr=asr, aligner=align, allow_synthetic=True)
            window = window_fixture()
            result = pipeline.process_window(window, "enr-1", [{"audio_turn_id": "t4", "old_transcript": "old", "start": 0, "end": 4}])
            self.assertEqual(result["stage_status"]["tse"], "UNRESOLVED")
            self.assertEqual(result["stage_status"]["asr"], "UNRESOLVED")

    def test_dedup_uses_canonical_interaction_not_history_or_target_text_alone(self):
        bank = EnrollmentBank("bank", "r1")
        bank.add(confirmed_reference())
        first = materialize_role_preserving(interaction("s1"), timeline(), ["t1", "t2", "t3", "t4"], "t4", FinalViewMembership.IN_TRAIN)
        other_interaction = interaction("s2")
        other_interaction.update({"recording_id": "rec-2", "canonical_recording_id": "canon-2"})
        other_timeline = [dict(turn, recording_id="rec-2", canonical_recording_id="canon-2") for turn in timeline()]
        second = materialize_role_preserving(other_interaction, other_timeline, ["t1", "t2", "t3", "t4"], "t4", FinalViewMembership.IN_TRAIN)
        report = validate_artifacts(bank, [], [], [first, second])
        self.assertEqual(report["gates"]["TARGET_REUSE"], "PASS")
        mirror = dict(second)
        mirror["sample_id"] = "s3"
        mirror["canonical_recording_id"] = first["canonical_recording_id"]
        mirror["interaction_dedup_key"] = first["interaction_dedup_key"]
        mirror_report = validate_artifacts(bank, [], [], [first, mirror])
        self.assertEqual(mirror_report["gates"]["TARGET_REUSE"], "FAIL")

    def test_repeated_masked_history_is_not_duplicate_supervision(self):
        bank = EnrollmentBank("bank", "r1")
        bank.add(confirmed_reference())
        values = timeline() + [{
            "audio_turn_id": "t5", "recording_id": "rec-1", "canonical_recording_id": "canon-1", "recording_family_id": "family-1",
            "role": "assistant", "identity": "NEURO_FAMILY", "old_transcript": "different target", "disagreement_flags": [],
            "text_resolution_state": "RESOLVED", "text_authority": "OLD_TRANSCRIPT", "resolved_text": "different target",
            "text_resolution_provenance": {"status": "RESOLVED", "resolver": "EXISTING_TRANSCRIPT_AUTHORITY", "revision": "fixture"},
        }]
        first = materialize_role_preserving(interaction("s1"), values, ["t1", "t2", "t3", "t4"], "t4", FinalViewMembership.IN_TRAIN)
        second_input = interaction("s2")
        second_input["target_turn_ids"] = ["t5"]
        second = materialize_role_preserving(second_input, values, ["t1", "t2", "t3", "t5"], "t5", FinalViewMembership.IN_TRAIN)
        report = validate_artifacts(bank, [], [], [first, second])
        self.assertEqual(report["gates"]["TARGET_REUSE"], "PASS")
        self.assertEqual(report["gates"]["PREFIX_LADDER"], "PASS")

    def test_hard_dedup_and_prefix_ladder_share_one_removal_ledger(self):
        """Regression: every dedup stage must reach dedup.jsonl and the manifest.

        A previous revision accumulated ladder removals into an undefined list,
        which crashed the finalization step after every expensive window had
        already been computed.  The removal ledger is now a single value.
        """
        rows = [
            {"sample_id": "s1", "recording_id": "rec-1", "context_turn_ids": ["t1"], "interaction_dedup_key": "k1", "messages": [{"role": "assistant", "content": "alpha beta gamma"}]},
            {"sample_id": "s2", "recording_id": "rec-1", "context_turn_ids": ["t1"], "interaction_dedup_key": "k1", "messages": [{"role": "assistant", "content": "alpha beta gamma"}]},
            {"sample_id": "s3", "recording_id": "rec-1", "context_turn_ids": ["t1"], "interaction_dedup_key": "k3", "messages": [{"role": "assistant", "content": "alpha beta gamma delta"}]},
            {"sample_id": "s4", "recording_id": "rec-1", "context_turn_ids": ["t1"], "interaction_dedup_key": "k4", "messages": [{"role": "assistant", "content": "unrelated wording entirely"}]},
        ]
        kept, removed = deduplicate_materialized(rows)
        counts = removal_counts(removed)
        self.assertEqual([row["sample_id"] for row in kept], ["s1", "s4"])
        self.assertEqual(counts["duplicate_interaction_provenance"], 1)
        self.assertEqual(counts["prefix_ladder"], 1)
        self.assertEqual(len(removed), 2)
        self.assertEqual({entry["sample_id"] for entry in removed}, {"s2", "s3"})
        self.assertEqual(sum(counts.values()), len(rows) - len(kept))

    def test_frozen_target_authority_never_lets_new_asr_overwrite_verified_text(self):
        old = "we should keep the verified frozen target wording"
        cases = {
            "MATCH": ("we should keep the verified frozen target wording", TextResolutionState.RESOLVED.value, "NEW_ASR", old),
            "MINOR_TEXT_CHANGE": ("we should keep the verified frozen target word", TextResolutionState.RESOLVED.value, "OLD_TRANSCRIPT", old),
            "MISSING_NEW_SPEECH": ("", TextResolutionState.RESOLVED.value, "OLD_TRANSCRIPT", old),
        }
        for disagreement_name, (new_text, state, authority, expected_text) in cases.items():
            # MINOR_TEXT_CHANGE must really classify as minor, so pick a
            # near-identical hypothesis for that case.
            candidate = new_text if disagreement_name != "MINOR_TEXT_CHANGE" else "we should keep the verified frozen target wording now"
            disagreement = detect_disagreement(old, candidate)
            resolution = resolve_frozen_target_text(old, candidate, disagreement)
            with self.subTest(case=disagreement_name):
                self.assertEqual(disagreement.value, disagreement_name)
                self.assertEqual(resolution["text_resolution_state"], state)
                self.assertEqual(resolution["text_authority"], authority)
                self.assertEqual(resolution["resolved_text"], expected_text)
                self.assertFalse(resolution.get("old_transcript_overwritten", False))
                self.assertEqual(resolution["text_resolution_provenance"]["evidence"]["new_asr_hypothesis"], candidate or None)
        for conflict in ("MAJOR_TRANSCRIPT_CONFLICT", "BOUNDARY_CHANGE", "SPEAKER_ASSIGNMENT_CHANGE"):
            if conflict == "MAJOR_TRANSCRIPT_CONFLICT":
                disagreement = detect_disagreement(old, "totally unrelated words here")
            elif conflict == "BOUNDARY_CHANGE":
                disagreement = detect_disagreement(old, old, boundary_changed=True)
            else:
                disagreement = detect_disagreement(old, old, speaker_assignment_changed=True)
            resolution = resolve_frozen_target_text(old, "totally unrelated words here", disagreement)
            with self.subTest(conflict=conflict):
                self.assertEqual(disagreement.value, conflict)
                self.assertEqual(resolution["text_resolution_state"], TextResolutionState.UNRESOLVED.value)
                self.assertIsNone(resolution["resolved_text"])
        missing_old = resolve_frozen_target_text("", "new speech only", detect_disagreement("", "new speech only"))
        self.assertEqual(missing_old["text_resolution_state"], TextResolutionState.UNRESOLVED.value)
        self.assertEqual(missing_old["text_resolution_provenance"]["reason"], "FROZEN_TARGET_OLD_TRANSCRIPT_MISSING")

    def test_pipeline_caches_only_mode_never_executes_a_model_stage(self):
        bank = EnrollmentBank("test-bank", "test-r1")
        bank.add(confirmed_reference(synthetic=True))
        with tempfile.TemporaryDirectory() as directory:
            cache = EvidenceCache(Path(directory) / "cache")
            calls = []
            activity_model = ModelProvenance("target_activity", "mock", "r1", {"target_activity_threshold": 0.8, "target_activity_threshold_version": "fixture-v1"})
            activity = CallableTargetActivityAdapter(activity_model, lambda audio, enr: calls.append("activity") or [ActivitySegment(audio.interval, 0.99, activity_model, enr.enrollment_id)])
            pipeline = AudioEvidencePipeline(bank, cache, CheckpointStore(Path(directory) / "cp.json"), activity=activity, diarization=None, tse=None, asr=None, aligner=None, allow_synthetic=True)
            window = window_fixture()
            # Cold cache: cache-only mode must refuse to run the adapter and
            # must fail loudly instead of degrading to an empty evidence set.
            with self.assertRaisesRegex(ContractError, "absent from the frozen cache"):
                pipeline.process_window(window, "enr-1", [{"audio_turn_id": "t4", "old_transcript": "keep me", "start": 0, "end": 4}], cache_only=True)
            self.assertEqual(calls, [])
            # Warm cache: the same window is served from evidence, still with no adapter call.
            pipeline.process_window(window, "enr-1", [{"audio_turn_id": "t4", "old_transcript": "keep me", "start": 0, "end": 4}])
            warm = pipeline.process_window(window, "enr-1", [{"audio_turn_id": "t4", "old_transcript": "keep me", "start": 0, "end": 4}], cache_only=True)
            self.assertEqual(calls, ["activity"])
            self.assertEqual(warm["stage_status"]["target_activity"], "AVAILABLE")
            self.assertTrue(warm["cache_hits"]["target_activity"])

    def test_pipeline_resolves_verified_target_under_frozen_authority_only_for_that_turn(self):
        bank = EnrollmentBank("test-bank", "test-r1")
        bank.add(confirmed_reference(synthetic=True))
        asr_model = ModelProvenance("asr", "mock-asr", "r1")
        align_model = ModelProvenance("alignment", "mock-align", "r1")
        asr = CallableASRAdapter(asr_model, lambda audio: TranscriptHypothesis("history wording and target words", asr_model, audio.uri))
        align = CallableForcedAlignmentAdapter(align_model, lambda audio, text: [
            AlignmentUnit("history wording", Interval(0, 2), 0.9),
            AlignmentUnit("target words", Interval(2, 4), 0.9),
        ])
        with tempfile.TemporaryDirectory() as directory:
            pipeline = AudioEvidencePipeline(bank, EvidenceCache(Path(directory) / "cache"), CheckpointStore(Path(directory) / "cp.json"), activity=None, diarization=None, tse=None, asr=asr, aligner=align, allow_synthetic=True)
            window = window_fixture()
            turns = [
                {"audio_turn_id": "h1", "role": "assistant", "identity": "NEURO_FAMILY", "old_transcript": "history words", "start": 0, "end": 2},
                {"audio_turn_id": "t4", "role": "assistant", "identity": "NEURO_FAMILY", "old_transcript": "target words", "start": 2, "end": 4},
            ]
            result = pipeline.process_window(window, "enr-1", turns, frozen_verified_target_turn_ids=["t4"])
        by_id = {turn["audio_turn_id"]: turn for turn in result["turns"]}
        target, context = by_id["t4"], by_id["h1"]
        # Exact match under the frozen authority resolves to the new hypothesis.
        self.assertEqual(target["disagreement_flags"], ["MATCH"])
        self.assertEqual(target["text_resolution_state"], "RESOLVED")
        self.assertEqual(target["text_authority"], "NEW_ASR")
        self.assertEqual(target["resolved_text"], "target words")
        # The optional context turn stays under the conservative fail-closed
        # policy even though its text is near-identical to the old transcript.
        self.assertEqual(context["text_resolution_state"], "UNRESOLVED")
        self.assertIsNone(context["resolved_text"])

    def test_pipeline_keeps_frozen_target_text_when_new_asr_differs_minorly(self):
        bank = EnrollmentBank("test-bank", "test-r1")
        bank.add(confirmed_reference(synthetic=True))
        asr_model = ModelProvenance("asr", "mock-asr", "r1")
        align_model = ModelProvenance("alignment", "mock-align", "r1")
        old_text = "keep the verified frozen target wording here"
        new_text = "keep the verified frozen target wording there"
        asr = CallableASRAdapter(asr_model, lambda audio: TranscriptHypothesis(new_text, asr_model, audio.uri))
        align = CallableForcedAlignmentAdapter(align_model, lambda audio, text: [AlignmentUnit(new_text, Interval(0, 4), 0.9)])
        with tempfile.TemporaryDirectory() as directory:
            pipeline = AudioEvidencePipeline(bank, EvidenceCache(Path(directory) / "cache"), CheckpointStore(Path(directory) / "cp.json"), activity=None, diarization=None, tse=None, asr=asr, aligner=align, allow_synthetic=True)
            result = pipeline.process_window(
                window_fixture(), "enr-1",
                [{"audio_turn_id": "t4", "role": "assistant", "identity": "NEURO_FAMILY", "old_transcript": old_text, "start": 0, "end": 4}],
                frozen_verified_target_turn_ids=["t4"],
            )
        turn = result["turns"][0]
        self.assertEqual(turn["disagreement_flags"], ["MINOR_TEXT_CHANGE"])
        self.assertEqual(turn["text_resolution_state"], "RESOLVED")
        self.assertEqual(turn["text_authority"], "OLD_TRANSCRIPT")
        self.assertEqual(turn["resolved_text"], old_text)
        provenance = turn["text_resolution_provenance"]
        self.assertEqual(provenance["new_asr_role"], "INDEPENDENT_AUDIO_CONFIRMATION_EVIDENCE")
        self.assertEqual(provenance["evidence"]["new_asr_hypothesis"], new_text)
        self.assertFalse(turn["old_transcript_overwritten"])

    def test_validator_traces_every_evidence_stage_and_authority(self):
        bank = EnrollmentBank("bank", "r1")
        bank.add(confirmed_reference())
        row = materialize_role_preserving(interaction(), timeline(), ["t3", "t4"], "t4", FinalViewMembership.IN_TRAIN)
        bad_turn = {
            "schema_version": "1.0.0", "recording_id": "rec-1", "audio_turn_id": "t4", "window_id": "w1",
            "start": 0.0, "end": 1.0, "timebase": "SECONDS_FROM_RECORDING_START", "speaker_cluster": "S0",
            "target_activity_evidence": [{"probability": 0.9}], "enrollment_provenance": None,
            "model_provenance": {}, "old_transcript": "old", "new_asr_hypothesis": "new", "asr_source": None,
            "optional_tse_asr_hypothesis": "new", "tse_source": None, "alignment": [{"text": "new"}],
            "disagreement_flags": [], "old_transcript_overwritten": True,
        }
        window = window_fixture(0, 1)
        report = validate_artifacts(bank, [window], [bad_turn], [row], expected_split_hash="different")
        for gate in ("TARGET_ACTIVITY_EVIDENCE_TRACEABLE", "DIARIZATION_TRACEABLE", "TSE_SOURCE_TRACEABLE", "ASR_SOURCE_TRACEABLE", "TRANSCRIPT_DISAGREEMENT_RECORDED", "NO_SILENT_TRANSCRIPT_OVERWRITE", "SPLIT_AUTHORITY_PRESERVED"):
            self.assertEqual(report["gates"][gate], "FAIL", gate)


if __name__ == "__main__":
    unittest.main()
