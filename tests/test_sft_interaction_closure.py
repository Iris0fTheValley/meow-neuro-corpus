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
import run_sft_v2_2_semantic_judge as judge_workflow
import validate_sft_v2_2 as pool_validator


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
    def continuation_request(self, **continuation_overrides):
        base_target = {
            "identity": "NEURO_FAMILY_HIGH",
            "speaker": "n",
            "speaker_confidence": 0.95,
            "text_qa": "PASS",
            "asr_quality": "PASS",
        }
        continuation = {
            **base_target,
            "turn_id": "n2",
            "text": "and this continues",
            "_start": 2.1,
            "_end": 3.0,
            "timestamp": {"start": 2.1, "end": 3.0},
            **continuation_overrides,
        }
        timeline = {"_turns": [
            {"turn_id": "c1", "identity": "OTHER", "speaker": "u", "speaker_confidence": 0.95, "text": "prompt", "text_qa": "PASS", "asr_quality": "PASS", "_start": 0.0, "_end": 1.0, "timestamp": {"start": 0.0, "end": 1.0}},
            {**base_target, "turn_id": "n1", "text": "answer", "_start": 1.1, "_end": 2.0, "timestamp": {"start": 1.1, "end": 2.0}},
            continuation,
        ]}
        row = {
            "sample_id": "s",
            "source_id": "src",
            "identity": "NEURO_FAMILY_HIGH",
            "context_turn_ids": ["c1"],
            "target_turn_ids": ["n1"],
            "episode_boundary": {"response_gap_policy_seconds": 3.0},
        }
        return closure.build_request(row, timeline, {})["request"]

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

    def test_text_qa_failed_continuation_is_not_offered_to_judge(self):
        value = self.continuation_request(text_qa="FAIL")
        self.assertEqual(value["candidate_target_turn_ids"], ["n1"])
        self.assertEqual(value["possible_continuation_fragments"], [])

    def test_low_confidence_continuation_is_not_offered_to_judge(self):
        value = self.continuation_request(speaker_confidence=0.69)
        self.assertEqual(value["candidate_target_turn_ids"], ["n1"])
        self.assertEqual(value["possible_continuation_fragments"], [])

    def test_asr_review_continuation_is_not_offered_to_judge(self):
        value = self.continuation_request(asr_quality="REVIEW")
        self.assertEqual(value["candidate_target_turn_ids"], ["n1"])
        self.assertEqual(value["possible_continuation_fragments"], [])

    def test_clean_continuation_is_offered_but_not_machine_merged(self):
        value = self.continuation_request()
        self.assertEqual(value["candidate_target_turn_ids"], ["n1", "n2"])
        self.assertEqual([turn["turn_id"] for turn in value["possible_continuation_fragments"]], ["n2"])
        self.assertEqual(value["current_target_turn_ids"], ["n1"])

    def test_other_hard_ineligible_continuations_are_not_offered(self):
        cases = {
            "identity": {"identity": "OTHER"},
            "speaker": {"speaker": "different"},
            "bad_boundary": {"overlap": True},
            "event_boundary": {"event_boundary": True},
            "temporal_bound": {"_start": 5.1, "timestamp": {"start": 5.1, "end": 6.0}},
        }
        for name, overrides in cases.items():
            with self.subTest(name=name):
                value = self.continuation_request(**overrides)
                self.assertEqual(value["candidate_target_turn_ids"], ["n1"])
                self.assertEqual(value["possible_continuation_fragments"], [])

    def test_selected_continuation_cannot_bypass_asr_gate(self):
        row = {"source_id": "src", "target_turn_ids": ["n1"], "transcript_qa": {"status": "STRUCTURAL_PASS"}}
        semantic = {"state": "VERIFIED", "final_decision": judgement(context=("c1",), target=("n1", "n2"), episode="MERGE_CONTINUATION")}
        base = {"text_qa": "PASS", "asr_quality": "PASS", "speaker_confidence": 0.95, "speaker": "n", "identity": "NEURO_FAMILY_HIGH"}
        timeline = {"_turns": [
            {**base, "turn_id": "c1", "_index": 0, "_start": 0, "_end": 1, "text": "prompt", "identity": "OTHER", "speaker": "u", "timestamp": {"start": 0, "end": 1}},
            {**base, "turn_id": "n1", "_index": 1, "_start": 1, "_end": 2, "text": "answer", "timestamp": {"start": 1, "end": 2}},
            {**base, "turn_id": "n2", "_index": 2, "_start": 2, "_end": 3, "text": "bad continuation", "asr_quality": "REVIEW", "timestamp": {"start": 2, "end": 3}},
        ]}
        self.assertIsNone(closure.materialize_verified(row, semantic, timeline))

    def test_self_continuation_is_not_a_verified_user_response_pair(self):
        primary = validated(relation="SELF_CONTINUATION")
        strict = validated(relation="SELF_CONTINUATION")
        self.assertEqual(closure.resolve_semantic_state("s", primary, strict)["state"], "AMBIGUOUS")


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
        self.assertTrue(repair["split_exclusion_relations"])
        repaired_rows = [{**row, "recording_family_id": repair["cluster_to_family"][row["canonical_recording_id"]]} for row in values]
        assignments, _ = views.derive_split_authority(repaired_rows, repair, None)
        relation = repair["split_exclusion_relations"][0]
        self.assertEqual(assignments[relation["family_a"]], assignments[relation["family_b"]])


class CorpusSelectionTests(unittest.TestCase):
    def repair(self, family_members, relations=None):
        return {
            "schema_version": closure.SCHEMA_VERSION,
            "recording_families": [{"recording_family_id": family, "canonical_recording_ids": members} for family, members in family_members.items()],
            "split_exclusion_relations": relations or [],
        }

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

    def test_old_sealed_lineage_survives_family_id_change(self):
        rows = [{"recording_family_id": "new-family", "canonical_recording_id": member} for member in ("A", "B", "C")]
        existing = {"schema_version": views.SAMPLING_SCHEMA_VERSION, "split_authority_version": views.SPLIT_AUTHORITY_VERSION, "member_assignments": {"A": "sealed_eval", "B": "sealed_eval"}}
        assignment, authority = views.derive_split_authority(rows, self.repair({"new-family": ["A", "B", "C"]}), existing)
        self.assertEqual(assignment["new-family"], "sealed_eval")
        self.assertEqual(authority["member_assignments"]["C"], "sealed_eval")

    def test_train_and_sealed_lineage_merge_is_explicit_conflict(self):
        rows = [{"recording_family_id": "merged", "canonical_recording_id": member} for member in ("A", "B")]
        existing = {"schema_version": views.SAMPLING_SCHEMA_VERSION, "split_authority_version": views.SPLIT_AUTHORITY_VERSION, "member_assignments": {"A": "train", "B": "sealed_eval"}}
        assignment, authority = views.derive_split_authority(rows, self.repair({"merged": ["A", "B"]}), existing)
        self.assertEqual(assignment["merged"], "quarantine")
        self.assertEqual(authority["status"], "SPLIT_LINEAGE_CONFLICT")

    def test_quarantined_strong_bridge_cannot_cross_split(self):
        rows = [{"recording_family_id": "fa", "canonical_recording_id": "A"}, {"recording_family_id": "fb", "canonical_recording_id": "B"}]
        relation = {"family_a": "fa", "family_b": "fb", "must_share_partition": True, "reason": "ELIGIBLE_STRONG_EDGE_QUARANTINED_ONLY_FOR_COMPONENT_SIZE"}
        assignment, _ = views.derive_split_authority(rows, self.repair({"fa": ["A"], "fb": ["B"]}, [relation]), None)
        self.assertEqual(assignment["fa"], assignment["fb"])

    def test_weak_quarantine_does_not_create_split_constraint(self):
        groups = views._constraint_groups({"fa", "fb"}, [])
        self.assertEqual({frozenset(group) for group in groups}, {frozenset({"fa"}), frozenset({"fb"})})

    def test_sampling_policies_reuse_one_dataset_split_authority(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pool_path, repair_path, authority_path = root / "pool.jsonl", root / "repair.json", root / "split_authority.json"
            rows = [{"sample_id": f"s-{index}", "recording_family_id": f"f-{index}", "canonical_recording_id": f"c-{index}", "artifact_schema_version": closure.SCHEMA_VERSION, "verified_pool_status": "VERIFIED_HARD_DEDUPED", "sampling_eligibility": "ELIGIBLE", "semantic_qa": {"confidence": 0.95}, "transcript_quality": "PASS"} for index in range(7)]
            pool_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            repair_path.write_text(json.dumps(self.repair({f"f-{index}": [f"c-{index}"] for index in range(7)})), encoding="utf-8")
            def args(policy):
                return Namespace(pool=str(pool_path), output=str(root / policy), policy=policy, max_per_family=1, validation_share=0.08, sealed_share=0.08, recording_repair=str(repair_path), split_authority=str(authority_path))
            natural = views.build(args("natural_frequency"))
            balanced = views.build(args("recording_balanced"))
        self.assertEqual(natural["split_authority"], balanced["split_authority"])
        self.assertEqual(natural["families"]["sealed_eval"], balanced["families"]["sealed_eval"])


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
                "context_turn_ids": ["c2", "c3"],
                "context_anchor_id": "c3",
                "target_turn_ids": ["n1"],
                "speaker_evidence": {"assistant_speaker": "speaker-neuro", "context_speakers": ["speaker-user-2", "speaker-user-3"]},
                "identity": "NEURO_FAMILY_HIGH",
                "transcript_qa": {"status": "STRUCTURAL_PASS", "turns": []},
                "messages": [{"role": "user", "content": "old prompt"}, {"role": "assistant", "content": "old response"}],
                "artifact_schema_version": structural.STRUCTURAL_SCHEMA_VERSION,
                "pipeline_version": structural.STRUCTURAL_PIPELINE_VERSION,
            }
            request_value = {"schema_version": closure.SCHEMA_VERSION, "risk_features": {"flags": []}}
            request_hash = closure.payload_sha256(request_value)
            decision = {"valid": True, "accepted": True, **judgement(context=("c1", "c2", "c3"), target=("n1", "n2"), episode="MERGE_CONTINUATION")}
            request_row = {"sample_id": "s1", "request": request_value, "request_sha256": request_hash, "artifact_schema_version": closure.SCHEMA_VERSION, "pipeline_version": closure.PIPELINE_VERSION}
            primary = {"schema_version": closure.SCHEMA_VERSION, "pipeline_version": closure.PIPELINE_VERSION, "sample_id": "s1", "stage": "primary", "request_sha256": request_hash, "judge_prompt_version": closure.PRIMARY_PROMPT_VERSION, "judge_model": "fixture", "validated": decision}
            strict = {"schema_version": closure.SCHEMA_VERSION, "pipeline_version": closure.PIPELINE_VERSION, "sample_id": "s1", "stage": "strict", "request_sha256": request_hash, "judge_prompt_version": closure.STRICT_PROMPT_VERSION, "judge_model": "fixture", "validated": decision}
            for path, value in ((structural_path, row), (requests_path, request_row), (primary_path, primary), (strict_path, strict)):
                path.write_text(json.dumps(value) + "\n", encoding="utf-8")
            def turn(turn_id, text, index, identity, speaker, confidence=0.95):
                return {"turn_id": turn_id, "text": text, "timestamp": {"start": index, "end": index + 0.9}, "_start": index, "_end": index + 0.9, "_index": index, "identity": identity, "speaker": speaker, "speaker_confidence": confidence, "text_qa": "PASS", "asr_quality": "PASS", "asr_metadata": {"avg_logprob": -0.2}}
            timeline = {"_turns": [
                turn("c1", "extended observable setup", 0, "OTHER", "speaker-user-1"),
                turn("c2", "observable prompt", 1, "OTHER", "speaker-user-2"),
                turn("c3", "more context", 2, "OTHER", "speaker-user-3"),
                turn("n1", "observed response", 3, "NEURO_FAMILY_HIGH", "speaker-neuro"),
                turn("n2", "and continuation", 4, "NEURO_FAMILY_HIGH", "speaker-neuro", 0.91),
            ]}
            fusion = {("src", "speaker-neuro"): {"identity": "NEURO_FAMILY_HIGH", "mapping_id": "src:speaker-neuro", "fusion_status": "PROXY", "audio_evidence": {"ensemble_gate": True}, "semantic_evidence": {"cannot_promote_without_independent_audio": True}}}
            args = Namespace(output=str(output), structural=str(structural_path), requests=str(requests_path), primary=str(primary_path), strict=str(strict_path), adjudication=str(adjudication_path), allow_partial=False, min_recording_edge_support=2, max_recording_component_size=8)
            with patch.object(finalizer.legacy_core, "load_fusion", return_value=fusion), patch.object(finalizer.legacy_core, "load_timelines", return_value={"src": timeline}), patch.object(finalizer.legacy_core, "overlap_edges", return_value=[]):
                result = finalizer.finalize(args)
            pool = finalizer.load_jsonl(output / "verified_interaction_pool_v2_3.jsonl")
            pool_rows = list(pool)
            self.assertEqual(result["status"], "FINALIZED")
            self.assertEqual(len(pool_rows), 1)
            final_row = pool_rows[0]
            self.assertEqual(final_row["context_turn_ids"], ["c1", "c2", "c3"])
            self.assertEqual(final_row["target_turn_ids"], ["n1", "n2"])
            self.assertEqual(final_row["raw_timeline_indices"], [0, 1, 2, 3, 4])
            self.assertEqual(final_row["transcript_qa"]["final_turn_ids"], ["c1", "c2", "c3", "n1", "n2"])
            self.assertEqual(final_row["speaker_evidence"]["context_speakers"], ["speaker-user-1", "speaker-user-2", "speaker-user-3"])
            self.assertEqual(final_row["speaker_evidence"]["assistant_turn_confidences"], [0.95, 0.91])
            self.assertEqual([turn["turn_id"] for turn in final_row["boundary_provenance"]["turns"]], ["c1", "c2", "c3", "n1", "n2"])
            validation = pool_validator.validate(Namespace(dataset=str(output), pool=str(output / "verified_interaction_pool_v2_3.jsonl"), views="", split_authority=str(output / "split_authority_v2_3.json"), report=str(output / "targeted_validation.json")))
            self.assertEqual(validation["gates"]["SEMANTIC_PROVENANCE_REMATERIALIZATION"], "PASS")
            self.assertEqual(validation["provenance_alignment_errors"], [])
            self.assertFalse((output / "train.jsonl").exists())


class JudgeResumeTests(unittest.TestCase):
    def request_row(self, digest="new-hash"):
        return {"sample_id": "s1", "request_sha256": digest, "request": {"schema_version": closure.SCHEMA_VERSION, **request()}}

    def result_row(self, digest="new-hash", stage="primary", model=judge_workflow.MODEL_DEFAULT):
        return {
            "sample_id": "s1",
            "request_sha256": digest,
            "schema_version": closure.SCHEMA_VERSION,
            "pipeline_version": closure.PIPELINE_VERSION,
            "stage": stage,
            "judge_prompt_version": judge_workflow.PROMPT_VERSIONS[stage],
            "judge_model": model,
            "validated": {"valid": True, "accepted": True},
        }

    def test_stale_request_hash_requires_rerun(self):
        compatible, reason = judge_workflow.result_compatibility(self.result_row("old-hash"), self.request_row(), "primary", judge_workflow.MODEL_DEFAULT)
        self.assertFalse(compatible)
        self.assertEqual(reason, "STALE_REQUEST_SHA256")

    def test_compatible_result_resumes(self):
        compatible, reason = judge_workflow.result_compatibility(self.result_row(), self.request_row(), "primary", judge_workflow.MODEL_DEFAULT)
        self.assertTrue(compatible)
        self.assertEqual(reason, "COMPATIBLE")

    def test_strict_routing_rejects_stale_primary(self):
        required = judge_workflow.strict_required_sample_ids([self.request_row()], {"s1": self.result_row("old-hash")}, judge_workflow.MODEL_DEFAULT)
        self.assertEqual(required, set())

    def test_status_reports_stale_as_rerun_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request_path = root / "requests.jsonl"
            request_path.write_text(json.dumps(self.request_row()) + "\n", encoding="utf-8")
            (root / "interaction_judge_primary_results_v2_3.jsonl").write_text(json.dumps(self.result_row("old-hash")) + "\n", encoding="utf-8")
            args = Namespace(requests=str(request_path), model=judge_workflow.MODEL_DEFAULT, primary_model="", strict_model="", adjudication_model="")
            with patch.object(judge_workflow, "OUT", root):
                report = judge_workflow.status(args)
        self.assertEqual(report["stages"]["primary"]["stale_or_invalid"], 1)
        self.assertEqual(report["stages"]["primary"]["rerun_required"], 1)

    def _judge_args(self, root, request_path, result_path):
        return Namespace(stage="primary", requests=str(request_path), sample_ids="", sample_ids_file="", max_items=0, results=str(result_path), fresh=False, retry_invalid=False, model=judge_workflow.MODEL_DEFAULT, primary_model="", endpoint="unused", max_attempts=1, sleep=0.0)

    def test_judge_reruns_stale_result_and_latest_overrides_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request_path, result_path = root / "requests.jsonl", root / "results.jsonl"
            request_path.write_text(json.dumps(self.request_row()) + "\n", encoding="utf-8")
            result_path.write_text(json.dumps(self.result_row("old-hash")) + "\n", encoding="utf-8")
            with patch.object(judge_workflow, "OUT", root), patch.object(judge_workflow.v21, "call_judge", return_value={"parsed": judgement()}) as call:
                report = judge_workflow.judge(self._judge_args(root, request_path, result_path))
            latest = judge_workflow.latest_results(result_path)["s1"]
        self.assertEqual(call.call_count, 1)
        self.assertEqual(report["stale_rerun"], 1)
        self.assertEqual(latest["request_sha256"], "new-hash")

    def test_judge_skips_compatible_cached_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request_path, result_path = root / "requests.jsonl", root / "results.jsonl"
            request_path.write_text(json.dumps(self.request_row()) + "\n", encoding="utf-8")
            result_path.write_text(json.dumps(self.result_row()) + "\n", encoding="utf-8")
            with patch.object(judge_workflow, "OUT", root), patch.object(judge_workflow.v21, "call_judge") as call:
                report = judge_workflow.judge(self._judge_args(root, request_path, result_path))
        self.assertEqual(call.call_count, 0)
        self.assertEqual(report["skipped_existing"], 1)


if __name__ == "__main__":
    unittest.main()
