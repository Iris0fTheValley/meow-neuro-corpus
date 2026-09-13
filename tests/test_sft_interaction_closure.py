from __future__ import annotations

import sys
import json
import tempfile
from argparse import Namespace
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import sft_interaction_semantic_closure as closure
import build_sft_v2_2_structural_candidates as structural
import build_sft_v2_2_train_views as views
import finalize_sft_v2_2 as finalizer


def judgement(
    relation="DIRECT_RESPONSE",
    episode="KEEP_CURRENT",
    context=("c1",),
    target=("n1",),
    confidence=0.92,
    **overrides,
):
    value = {
        "relation": relation,
        "episode_decision": episode,
        "confidence": confidence,
        "context_complete": True,
        "response_complete": True,
        "transcript_usable": True,
        "selected_context_turn_ids": list(context),
        "selected_target_turn_ids": list(target),
        "reason": "fixture evidence",
    }
    value.update(overrides)
    return value


def request():
    return {
        "allowed_context_selections": [["p1", "c1"], ["c1"]],
        "current_target_turn_ids": ["n1"],
        "candidate_target_turn_ids": ["n1", "n2"],
    }


def validated(**kwargs):
    return closure.validate_judgement(judgement(**kwargs), request())


class SemanticClosureStateTests(unittest.TestCase):
    def test_direct_response_requires_two_agreeing_semantic_judges(self):
        primary = validated()
        pending = closure.resolve_semantic_state("s", primary, None)
        self.assertEqual(pending["state"], "PENDING_STRICT")
        final = closure.resolve_semantic_state("s", primary, validated())
        self.assertEqual(final["state"], "VERIFIED")

    def test_temporal_adjacency_without_relation_is_not_verified(self):
        primary = validated(relation="UNCLEAR", confidence=0.55)
        strict = validated(relation="UNCLEAR", confidence=0.88)
        self.assertEqual(closure.resolve_semantic_state("s", primary, strict)["state"], "AMBIGUOUS")

    def test_absurd_topic_shift_with_observable_trigger_is_preserved(self):
        risk = {"flags": ["NO_CONTENT_TOKEN_OVERLAP"], "authority": "ROUTING_AND_EVIDENCE_ONLY"}
        decision = closure.resolve_semantic_state("s", validated(), validated(), risk=risk)
        self.assertEqual(decision["state"], "VERIFIED")

    def test_diagnostic_neighbor_cannot_be_selected_as_context(self):
        invalid = closure.validate_judgement(judgement(context=("diagnostic", "c1")), request())
        self.assertFalse(invalid["valid"])

    def test_new_speech_act_merge_disagreement_is_explicit_conflict(self):
        primary = validated(episode="MERGE_CONTINUATION", target=("n1", "n2"))
        strict = validated(episode="SPLIT_NEW_SPEECH_ACT", target=("n1",))
        self.assertEqual(closure.resolve_semantic_state("s", primary, strict)["state"], "JUDGE_CONFLICT")

    def test_grammatical_continuation_can_merge(self):
        primary = validated(episode="MERGE_CONTINUATION", target=("n1", "n2"))
        strict = validated(episode="MERGE_CONTINUATION", target=("n1", "n2"))
        self.assertEqual(closure.resolve_semantic_state("s", primary, strict)["state"], "VERIFIED")

    def test_llm_cannot_supply_rewritten_transcript(self):
        value = judgement(corrected_transcript="more natural words")
        result = closure.validate_judgement(value, request())
        self.assertFalse(result["valid"])
        self.assertEqual(result["reason"], "transcript_rewrite_forbidden")

    def test_request_hash_is_stable_and_content_sensitive(self):
        left = {"b": [2, 3], "a": 1}
        self.assertEqual(closure.payload_sha256(left), closure.payload_sha256({"a": 1, "b": [2, 3]}))
        self.assertNotEqual(closure.payload_sha256(left), closure.payload_sha256({"a": 2, "b": [2, 3]}))

    def test_context_selection_is_bounded_and_contiguous(self):
        turns = [
            {"turn_id": "far", "identity": "OTHER", "_start": 0, "_end": 1},
            {"turn_id": "p1", "identity": "OTHER", "_start": 2, "_end": 3},
            {"turn_id": "c1", "identity": "OTHER", "_start": 4, "_end": 5},
        ]
        envelope, options = closure.bounded_context_options(turns, ["c1"], max_extension_turns=1)
        self.assertEqual([turn["turn_id"] for turn in envelope], ["p1", "c1"])
        self.assertEqual(options, [["p1", "c1"], ["c1"]])

    def test_bad_boundary_cannot_be_context(self):
        turns = [
            {"turn_id": "c", "identity": "OTHER", "text_qa": "PASS", "asr_quality": "PASS", "overlap": True, "_start": 0, "_end": 1},
            {"turn_id": "n", "identity": "NEURO_FAMILY_HIGH", "_start": 1.1, "_end": 2},
        ]
        indices, reason = structural.context_episode(turns, 1, 6.0)
        self.assertEqual(indices, [])
        self.assertEqual(reason, "CONTEXT_EPISODE_BROKEN")

    def test_long_gap_context_is_not_joined(self):
        turns = [
            {"turn_id": "old", "identity": "OTHER", "text_qa": "PASS", "asr_quality": "PASS", "_start": 0, "_end": 1},
            {"turn_id": "anchor", "identity": "OTHER", "text_qa": "PASS", "asr_quality": "PASS", "_start": 20, "_end": 21},
            {"turn_id": "n", "identity": "NEURO_FAMILY_HIGH", "_start": 21.1, "_end": 22},
        ]
        indices, reason = structural.context_episode(turns, 2, 6.0)
        self.assertIsNone(reason)
        self.assertEqual(indices, [1])

    def test_target_without_observable_trigger_is_not_seeded(self):
        turns = [{"turn_id": "n", "identity": "NEURO_FAMILY_HIGH", "_start": 0, "_end": 1}]
        indices, reason = structural.context_episode(turns, 0, 6.0)
        self.assertEqual(indices, [])
        self.assertEqual(reason, "NO_OBSERVABLE_TRIGGER")

    def test_structural_continuation_heuristic_cannot_merge_seed(self):
        turns = [
            {"turn_id": "n1", "identity": "NEURO_FAMILY_HIGH", "speaker": "n", "text": "this is", "text_qa": "PASS", "asr_quality": "PASS", "_start": 0, "_end": 1},
            {"turn_id": "n2", "identity": "NEURO_FAMILY_HIGH", "speaker": "n", "text": "a continuation", "text_qa": "PASS", "asr_quality": "PASS", "_start": 1.1, "_end": 2},
        ]
        seed, offered, risks = structural.structural_response_seed(turns, 0, 3.0)
        self.assertEqual(seed, [0])
        self.assertEqual(offered, ["n2"])
        self.assertTrue(risks[0].startswith("RISK_MERGE_"))


class DedupAndRecordingTests(unittest.TestCase):
    def row(self, sample, family_source, response, *, cluster=None, context="c", target="n"):
        return {
            "sample_id": sample,
            "source_id": family_source,
            "canonical_recording_id": cluster or family_source,
            "context_turn_ids": [context],
            "context_anchor_id": context,
            "target_turn_ids": [target],
            "messages": [{"role": "user", "content": "prompt"}, {"role": "assistant", "content": response}],
        }

    def test_same_recording_exact_duplicate_is_removed(self):
        values = [self.row("a", "src-a", "same", context="c1", target="n1"), self.row("b", "src-b", "same", cluster="cluster-a", context="c2", target="n2")]
        values[0]["canonical_recording_id"] = "cluster-a"
        kept, audit = closure.provenance_aware_dedup(values, {"cluster-a": "family-a"})
        self.assertEqual(len(kept), 1)
        self.assertEqual(audit["removed_by_reason"]["SAME_RECORDING_EXACT_MESSAGES"], 1)

    def test_independent_recordings_same_phrase_are_preserved_as_behavior(self):
        values = [self.row("a", "src-a", "same phrase", target="n1"), self.row("b", "src-b", "same phrase", target="n2")]
        kept, audit = closure.provenance_aware_dedup(values, {"src-a": "family-a", "src-b": "family-b"})
        self.assertEqual(len(kept), 2)
        self.assertEqual(audit["cross_recording_behavior_repetitions_preserved"], 2)
        self.assertTrue(all((row.get("distribution_metadata") or {}).get("behavior_repeat_cluster") for row in kept))

    def test_distinct_raw_episodes_in_one_recording_are_not_text_deduped(self):
        values = [self.row("a", "src-a", "same phrase", context="c1", target="n1"), self.row("b", "src-a", "same phrase", context="c2", target="n2")]
        kept, _ = closure.provenance_aware_dedup(values, {"src-a": "family-a"})
        self.assertEqual(len(kept), 2)

    def test_same_recording_response_containment_is_removed(self):
        short = "one two three four five six seven eight"
        long = short + " nine ten"
        values = [self.row("a", "src-a", short, cluster="clip-a", context="c1", target="n1"), self.row("b", "src-b", long, cluster="vod-b", context="c2", target="n2")]
        kept, audit = closure.provenance_aware_dedup(values, {"clip-a": "family-a", "vod-b": "family-a"})
        self.assertEqual(len(kept), 1)
        self.assertEqual(audit["removed_by_reason"]["SAME_RECORDING_INTERACTION_CONTAINMENT"], 1)

    def test_partial_reupload_needs_aggregate_or_ultra_strong_evidence(self):
        values = [self.row("a", "s1", "x", cluster="a"), self.row("b", "s2", "y", cluster="b")]
        edges = [
            {"cluster_a": "a", "cluster_b": "b", "sample_a": "a1", "sample_b": "b1", "strong_merge_edge": True, "overlap_length_tokens": 31, "containment_ratio": 0.8},
            {"cluster_a": "a", "cluster_b": "b", "sample_a": "a2", "sample_b": "b2", "strong_merge_edge": True, "overlap_length_tokens": 32, "containment_ratio": 0.8},
        ]
        repair = closure.repair_recording_families(values, edges)
        self.assertEqual(repair["cluster_to_family"]["a"], repair["cluster_to_family"]["b"])

    def test_multiple_weak_common_phrase_edges_do_not_union(self):
        values = [self.row("a", "s1", "x", cluster="a"), self.row("b", "s2", "y", cluster="b")]
        edges = [
            {"cluster_a": "a", "cluster_b": "b", "sample_a": f"a{i}", "sample_b": f"b{i}", "strong_merge_edge": True, "overlap_length_tokens": 20, "containment_ratio": 0.7, "similarity": 0.7}
            for i in range(4)
        ]
        repair = closure.repair_recording_families(values, edges)
        self.assertNotEqual(repair["cluster_to_family"]["a"], repair["cluster_to_family"]["b"])
        self.assertTrue(any(edge["decision"] == "QUARANTINED_WEAK_EDGE" for edge in repair["quarantined_edges"]))

    def test_giant_component_bridge_is_quarantined(self):
        values = [self.row(name, name, name, cluster=name) for name in ("a", "b", "c")]
        edges = [
            {"cluster_a": "a", "cluster_b": "b", "sample_a": "a1", "sample_b": "b1", "strong_merge_edge": True, "overlap_length_tokens": 90, "containment_ratio": 0.95, "similarity": 0.95},
            {"cluster_a": "b", "cluster_b": "c", "sample_a": "b2", "sample_b": "c2", "strong_merge_edge": True, "overlap_length_tokens": 90, "containment_ratio": 0.95, "similarity": 0.95},
        ]
        repair = closure.repair_recording_families(values, edges, max_component_size=2)
        self.assertTrue(any(edge["decision"] == "QUARANTINED_COMPONENT_BRIDGE" for edge in repair["quarantined_edges"]))


class CorpusSelectionTests(unittest.TestCase):
    def test_recording_families_are_split_before_sampling(self):
        rows = []
        for family_index in range(8):
            for row_index in range(3):
                rows.append({"sample_id": f"s-{family_index}-{row_index}", "recording_family_id": f"f-{family_index}", "verified_pool_status": "VERIFIED_HARD_DEDUPED", "sampling_eligibility": "ELIGIBLE"})
        assignment = views.family_splits(rows)
        train = [row for row in rows if assignment[row["recording_family_id"]] == "train"]
        selected = views.apply_policy(train, "recording_balanced", 1)
        sealed_families = {family for family, split in assignment.items() if split == "sealed_eval"}
        self.assertTrue(sealed_families)
        self.assertFalse({row["recording_family_id"] for row in selected} & sealed_families)
        self.assertLessEqual(max(Counter(row["recording_family_id"] for row in selected).values()), 1)

    def test_persisted_sealed_assignment_is_reused(self):
        rows = [{"recording_family_id": f"f-{index}"} for index in range(7)]
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "view_manifest.json"
            manifest.write_text(json.dumps({"families": {"train": ["f-0"], "validation": ["f-1"], "sealed_eval": ["f-2"]}}), encoding="utf-8")
            assignment, reused = views.stable_family_splits(rows, manifest, 0.08, 0.08)
        self.assertTrue(reused)
        self.assertEqual(assignment["f-2"], "sealed_eval")


class FinalizerIntegrationTests(unittest.TestCase):
    def test_verified_pool_is_distinct_from_train_view(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            structural_path = root / "structural.jsonl"
            requests_path = root / "requests.jsonl"
            primary_path = root / "primary.jsonl"
            strict_path = root / "strict.jsonl"
            adjudication_path = root / "adjudication.jsonl"
            output = root / "output"
            row = {
                "sample_id": "s1",
                "source_id": "src",
                "canonical_recording_id": "recording",
                "context_turn_ids": ["c1"],
                "context_anchor_id": "c1",
                "target_turn_ids": ["n1"],
                "speaker_evidence": {"assistant_speaker": "speaker-neuro", "context_speakers": ["speaker-user"]},
                "identity": "NEURO_FAMILY_HIGH",
                "transcript_qa": {"status": "STRUCTURAL_PASS", "turns": []},
                "messages": [{"role": "user", "content": "old prompt"}, {"role": "assistant", "content": "old response"}],
                "artifact_schema_version": structural.STRUCTURAL_SCHEMA_VERSION,
                "pipeline_version": structural.STRUCTURAL_PIPELINE_VERSION,
            }
            request_value = {"schema_version": closure.SCHEMA_VERSION, "risk_features": {"flags": []}}
            request_hash = closure.payload_sha256(request_value)
            decision = validated()
            request_row = {"sample_id": "s1", "request": request_value, "request_sha256": request_hash}
            primary = {"schema_version": closure.SCHEMA_VERSION, "sample_id": "s1", "stage": "primary", "request_sha256": request_hash, "validated": decision}
            strict = {"schema_version": closure.SCHEMA_VERSION, "sample_id": "s1", "stage": "strict", "request_sha256": request_hash, "validated": decision}
            for path, value in ((structural_path, row), (requests_path, request_row), (primary_path, primary), (strict_path, strict)):
                path.write_text(json.dumps(value) + "\n", encoding="utf-8")
            timeline = {"_turns": [
                {"turn_id": "c1", "text": "observable prompt", "timestamp": {"start": 1, "end": 2}, "identity": "OTHER", "speaker": "speaker-user"},
                {"turn_id": "n1", "text": "observed response", "timestamp": {"start": 2, "end": 3}, "identity": "NEURO_FAMILY_HIGH", "speaker": "speaker-neuro"},
            ]}
            fusion = {("src", "speaker-neuro"): {"identity": "NEURO_FAMILY_HIGH", "mapping_id": "src:speaker-neuro", "fusion_status": "PROXY", "audio_evidence": {"ensemble_gate": True}, "semantic_evidence": {"cannot_promote_without_independent_audio": True}}}
            args = Namespace(output=str(output), structural=str(structural_path), requests=str(requests_path), primary=str(primary_path), strict=str(strict_path), adjudication=str(adjudication_path), allow_partial=False, min_recording_edge_support=2, max_recording_component_size=8)
            with patch.object(finalizer.legacy_core, "load_fusion", return_value=fusion), patch.object(finalizer.legacy_core, "load_timelines", return_value={"src": timeline}), patch.object(finalizer.legacy_core, "overlap_edges", return_value=[]):
                result = finalizer.finalize(args)
            pool = finalizer.load_jsonl(output / "verified_interaction_pool_v2_3.jsonl")
            pool_rows = list(pool)
            self.assertEqual(result["status"], "FINALIZED")
            self.assertEqual(len(pool_rows), 1)
            self.assertEqual(pool_rows[0]["messages"][0]["content"], "observable prompt")
            self.assertFalse((output / "train.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
