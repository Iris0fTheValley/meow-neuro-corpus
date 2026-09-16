from __future__ import annotations

import hashlib
import json
import sys
import tempfile
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
    ModelProvenance, TranscriptHypothesis,
)
from audio_evidence.enrollment import (  # noqa: E402
    EnrollmentBank, EnrollmentReference, VerificationEvidence, VerificationMethod,
    VerificationStatus,
)
from audio_evidence.materialization import (  # noqa: E402
    FinalViewMembership, belongs_to_train, export_prompt_completion, materialize_role_preserving,
)
from audio_evidence.pipeline import AudioEvidencePipeline  # noqa: E402
from audio_evidence.planning import AudioWindowPlanner, WindowRequest  # noqa: E402
from audio_evidence.routing import AmbiguityRouter, RouteReason  # noqa: E402
from audio_evidence.transcript import TranscriptDisagreement, detect_disagreement  # noqa: E402
from audio_evidence.validation import validate_artifacts  # noqa: E402


SHA = hashlib.sha256(b"fixture").hexdigest()


def evidence(domain):
    return VerificationEvidence("ev-" + domain.lower(), domain, "artifact://frozen-v2.3", "PASS", "frozen-r1")


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
        "timestamps": {"start": 0.0, "end": 4.0, "timebase": "SECONDS_FROM_RECORDING_START"},
        "training_candidate": False, "semantic_truth_ref": "semantic://s1", "semantic_truth_sha256": "semhash",
        "split_authority_ref": "split://v2.3", "split_authority_sha256": "splithash", "provenance": {"source": "v2.3"},
    }


def timeline(overlap=False):
    return [
        {"audio_turn_id": "t1", "recording_id": "rec-1", "role": "user", "identity": "GUEST", "old_transcript": "A"},
        {"audio_turn_id": "t2", "recording_id": "rec-1", "role": "assistant", "identity": "NEURO_FAMILY", "old_transcript": "B"},
        {"audio_turn_id": "t3", "recording_id": "rec-1", "role": "user", "identity": "GUEST", "old_transcript": "C"},
        {"audio_turn_id": "t4", "recording_id": "rec-1", "role": "assistant", "identity": "NEURO_FAMILY", "old_transcript": "D"},
    ]


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
            calls = []
            self.assertEqual(cache.get_or_compute("asr", key1, lambda: calls.append(1) or {"x": 1})[1], False)
            self.assertEqual(cache.get_or_compute("asr", key1, lambda: calls.append(2) or {"x": 2})[1], True)
            self.assertEqual(calls, [1])


class RoutingAndTranscriptTests(unittest.TestCase):
    def setUp(self):
        self.model = ModelProvenance("diarization", "mock", "r1")

    def test_clean_default_and_overlap_tse_route(self):
        clean = [DiarizationTurn(Interval(0, 1), "S0", False, self.model)]
        self.assertEqual(AmbiguityRouter().decide([], clean).reason_code, RouteReason.CLEAN_SINGLE_SPEAKER)
        overlap = [DiarizationTurn(Interval(0, 1), "S0", True, self.model)]
        self.assertEqual(AmbiguityRouter().decide([], overlap).reason_code, RouteReason.TARGET_OVERLAP)

    def test_transcript_disagreement_classes(self):
        self.assertEqual(detect_disagreement("hello there", "hello there"), TranscriptDisagreement.MATCH)
        self.assertEqual(detect_disagreement("", "hello"), TranscriptDisagreement.MISSING_OLD_SPEECH)
        self.assertEqual(detect_disagreement("hello", ""), TranscriptDisagreement.MISSING_NEW_SPEECH)
        self.assertEqual(detect_disagreement("one two three", "completely different words"), TranscriptDisagreement.MAJOR_TRANSCRIPT_CONFLICT)
        self.assertEqual(detect_disagreement("x", "y", boundary_changed=True), TranscriptDisagreement.BOUNDARY_CHANGE)


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

    def test_no_synthetic_prompt_for_missing_text(self):
        values = timeline()
        values[0]["old_transcript"] = ""
        with self.assertRaises(ContractError):
            materialize_role_preserving(interaction(), values, ["t1", "t4"], "t4", FinalViewMembership.IN_TRAIN)


class IntegrationSmokeTests(unittest.TestCase):
    def adapters(self, overlap=False, counts=None):
        counts = counts if counts is not None else {}
        def count(name):
            counts[name] = counts.get(name, 0) + 1
        activity_model = ModelProvenance("target_activity", "nomo-pvad-mock", "r1")
        diar_model = ModelProvenance("diarization", "pyannote-community-1-mock", "r1")
        asr_model = ModelProvenance("asr", "qwen3-asr-mock", "r1")
        align_model = ModelProvenance("alignment", "qwen3-aligner-mock", "r1")
        tse_model = ModelProvenance("tse", "wesep-mock", "r1")
        activity = CallableTargetActivityAdapter(activity_model, lambda audio, enr: (count("activity") or [ActivitySegment(Interval(1, 2), 0.99, activity_model, enr.enrollment_id)]))
        diar = CallableDiarizationAdapter(diar_model, lambda audio: (count("diar") or [DiarizationTurn(Interval(0, 4), "S0", overlap, diar_model)]))
        tse = CallableTSEAdapter(tse_model, lambda audio, enr: AudioInput("audio://extracted", hashlib.sha256(b"extract").hexdigest(), audio.recording_id))
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
            window = {"window_id": "w1", "recording_id": "rec-1", "source_audio": "audio://rec-1", "audio_checksum": SHA, "merged_interval": Interval(0, 4).to_dict()}
            old = [{"audio_turn_id": "t4", "role": "assistant", "identity": "NEURO_FAMILY", "old_transcript": "old transcript", "start": 0, "end": 4}]
            first = pipeline.process_window(window, "enr-1", old)
            second = pipeline.process_window(window, "enr-1", old)
            self.assertEqual(first["route"]["reason_code"], "CLEAN_SINGLE_SPEAKER")
            self.assertEqual(second["cache_hits"], {"target_activity": True, "diarization": True, "asr": True, "alignment": True})
            self.assertEqual(counts, {"activity": 1, "diar": 1})
            self.assertEqual(first["turns"][0]["old_transcript"], "old transcript")
            self.assertEqual(first["turns"][0]["new_asr_hypothesis"], "new transcript")

    def test_overlap_routes_tse_with_enrollment_provenance(self):
        bank = EnrollmentBank("test-bank", "test-r1")
        bank.add(confirmed_reference(synthetic=True))
        activity, diar, tse, asr, align = self.adapters(overlap=True)
        with tempfile.TemporaryDirectory() as directory:
            pipeline = AudioEvidencePipeline(bank, EvidenceCache(Path(directory) / "cache"), CheckpointStore(Path(directory) / "cp.json"), activity=activity, diarization=diar, tse=tse, asr=asr, aligner=align, allow_synthetic=True)
            window = {"window_id": "w1", "recording_id": "rec-1", "source_audio": "audio://rec-1", "audio_checksum": SHA, "merged_interval": Interval(0, 4).to_dict()}
            result = pipeline.process_window(window, "enr-1", [{"audio_turn_id": "t4", "old_transcript": "old", "start": 0, "end": 4}])
            self.assertEqual(result["route"]["reason_code"], "TARGET_OVERLAP")
            self.assertEqual(result["turns"][0]["tse_source"]["enrollment_id"], "enr-1")

    def test_validator_accepts_complete_role_preserving_chain(self):
        bank = EnrollmentBank("bank", "r1")
        bank.add(confirmed_reference())
        window = {"window_id": "w1", "recording_id": "rec-1", "source_audio": "audio://rec-1", "merged_interval": Interval(0, 4).to_dict()}
        evidence_turns = []
        for index, source in enumerate(timeline()):
            evidence_turns.append({
                "schema_version": "1.0.0", "recording_id": "rec-1", "audio_turn_id": source["audio_turn_id"], "window_id": "w1",
                "start": float(index), "end": float(index + 1), "timebase": "SECONDS_FROM_RECORDING_START", "role": source["role"], "identity": source["identity"],
                "old_transcript": source["old_transcript"], "new_asr_hypothesis": source["old_transcript"], "alignment": None,
                "speaker_cluster": "S0", "target_activity_evidence": [{"x": 1}], "enrollment_provenance": {"enrollment_id": "enr-1"},
                "model_provenance": {"diarization": {"revision": "r1"}}, "asr_source": {"audio_checksum": SHA},
                "disagreement_flags": ["MATCH"], "old_transcript_overwritten": False,
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

    def test_graceful_degradation_preserves_old_text(self):
        bank = EnrollmentBank("test-bank", "test-r1")
        bank.add(confirmed_reference(synthetic=True))
        with tempfile.TemporaryDirectory() as directory:
            pipeline = AudioEvidencePipeline(bank, EvidenceCache(Path(directory) / "cache"), CheckpointStore(Path(directory) / "cp.json"), activity=None, diarization=None, tse=None, asr=None, aligner=None, allow_synthetic=True)
            window = {"window_id": "w1", "recording_id": "rec-1", "source_audio": "audio://rec-1", "audio_checksum": SHA, "merged_interval": Interval(0, 4).to_dict()}
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
            window = {"window_id": "w1", "recording_id": "rec-1", "source_audio": "audio://rec-1", "audio_checksum": SHA, "merged_interval": Interval(0, 4).to_dict()}
            result = pipeline.process_window(window, "enr-1", [{"audio_turn_id": "t4", "old_transcript": "old", "start": 0, "end": 4}])
            self.assertEqual(result["stage_status"]["tse"], "UNRESOLVED")
            self.assertEqual(result["stage_status"]["asr"], "UNRESOLVED")

    def test_dedup_uses_canonical_interaction_not_history_or_target_text_alone(self):
        bank = EnrollmentBank("bank", "r1")
        bank.add(confirmed_reference())
        first = materialize_role_preserving(interaction("s1"), timeline(), ["t1", "t2", "t3", "t4"], "t4", FinalViewMembership.IN_TRAIN)
        other_interaction = interaction("s2")
        other_interaction.update({"recording_id": "rec-2", "canonical_recording_id": "canon-2"})
        other_timeline = [dict(turn, recording_id="rec-2") for turn in timeline()]
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
        values = timeline() + [{"audio_turn_id": "t5", "recording_id": "rec-1", "role": "assistant", "identity": "NEURO_FAMILY", "old_transcript": "different target"}]
        first = materialize_role_preserving(interaction("s1"), values, ["t1", "t2", "t3", "t4"], "t4", FinalViewMembership.IN_TRAIN)
        second_input = interaction("s2")
        second_input["target_turn_ids"] = ["t5"]
        second = materialize_role_preserving(second_input, values, ["t1", "t2", "t3", "t5"], "t5", FinalViewMembership.IN_TRAIN)
        report = validate_artifacts(bank, [], [], [first, second])
        self.assertEqual(report["gates"]["TARGET_REUSE"], "PASS")
        self.assertEqual(report["gates"]["PREFIX_LADDER"], "PASS")

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
        window = {"window_id": "w1", "recording_id": "rec-1", "source_audio": "audio://rec-1", "merged_interval": Interval(0, 1).to_dict()}
        report = validate_artifacts(bank, [window], [bad_turn], [row], expected_split_hash="different")
        for gate in ("TARGET_ACTIVITY_EVIDENCE_TRACEABLE", "DIARIZATION_TRACEABLE", "TSE_SOURCE_TRACEABLE", "ASR_SOURCE_TRACEABLE", "TRANSCRIPT_DISAGREEMENT_RECORDED", "NO_SILENT_TRANSCRIPT_OVERWRITE", "SPLIT_AUTHORITY_PRESERVED"):
            self.assertEqual(report["gates"][gate], "FAIL", gate)


if __name__ == "__main__":
    unittest.main()
