from __future__ import annotations

import unittest
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from audio_evidence.recovery_v2 import (
    CONTEXT_JUDGE_POLICY_REVISION,
    CONTEXT_JUDGE_PROMPT_REVISION,
    CONTEXT_JUDGE_SCHEMA_REVISION,
    ContextRelation,
    ContextSufficiencyState,
    ReconciliationState,
    SpeakerState,
    TargetResolutionState,
    assign_tokens_to_old_turns,
    build_cluster_speaker_anchors,
    build_candidate_spans,
    context_sufficiency,
    is_non_conversational_sentinel,
    merge_existing_turns,
    merge_obvious_audio_fragments,
    build_context_judge_payload,
    call_context_sufficiency_judge,
    context_judge_cache_key,
    reconciliation_recovery_eligible,
    reconcile_old_turn,
    reconcile_span,
    resolve_speaker_state,
    resolve_anchored_cluster_speaker,
    resolve_target,
    select_minimal_context,
    sentinel_only_context,
    split_existing_turn,
)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_audio_evidence_recovery_v2 import CachedContextJudge
from content_audit_audio_recovery_v4 import call_reviewer, corroborate_finding, review_payload


def old(turn_id: str, start: float, end: float, text: str, role: str = "user") -> dict:
    return {"turn_id": turn_id, "start": start, "end": end, "text": text, "role": role}


def token(text: str, start: float, end: float) -> dict:
    return {"text": text, "start": start, "end": end}


def context_turn(turn_id: str, start: float, end: float, role: str, text: str) -> dict:
    state = SpeakerState.CONFIRMED_TARGET if role == "assistant" else SpeakerState.CONFIRMED_NON_TARGET
    return {
        "audio_turn_id": turn_id,
        "start": start,
        "end": end,
        "role": role,
        "resolved_text": text,
        "speaker_state": state.value,
    }


class ContentAuditRegressionTests(unittest.TestCase):
    def test_content_audit_hides_supervision_marker_from_reviewer(self):
        payload = review_payload({
            "review_kind": "training_sample",
            "sample_id": "audit-sample",
            "recording_id": "recording",
            "context_reconstruction_class": "RECOVERED_FROM_NEW_AUDIO",
            "messages": [
                {"role": "user", "content": "Are you coming tomorrow?", "supervise": False},
                {"role": "assistant", "content": "Probably.", "supervise": True},
            ],
        })
        self.assertEqual(payload["messages"], [
            {"role": "user", "text": "Are you coming tomorrow?"},
            {"role": "assistant", "text": "Probably."},
        ])

    def test_content_audit_invalid_model_output_is_coverage_diagnostic_not_finding(self):
        payload = {"kind": "training_sample", "messages": []}
        with patch("content_audit_audio_recovery_v4.call_model", return_value=({}, False, "TimeoutError")):
            result = call_reviewer(payload, endpoint="http://unused", model="test", timeout=1)
        self.assertFalse(result["finding"])
        self.assertFalse(result["valid"])

    def test_content_audit_requires_visible_target_leakage_evidence(self):
        payload = {"kind": "training_sample", "messages": [
            {"role": "user", "text": "Are you coming tomorrow?"},
            {"role": "assistant", "text": "Probably."},
        ]}
        claim = {"finding": True, "issue_category": "target_leakage", "evidence": "hallucinated"}
        triage = {"verified": True, "issue_category": "target_leakage", "severity": "high"}
        self.assertIsNone(corroborate_finding(payload, claim, triage, context_state="CONTEXT_SUFFICIENT", source_row={}, turn_intervals={}))
        payload["messages"].insert(0, {"role": "user", "text": "Probably."})
        source_row = {"messages": [
            {"source_turn_ids": ["old"]}, {"source_turn_ids": ["old"]}, {"source_turn_ids": ["target"]},
        ]}
        confirmed = corroborate_finding(payload, claim, triage, context_state="CONTEXT_SUFFICIENT", source_row=source_row, turn_intervals={})
        self.assertEqual(confirmed["category"], "target_leakage")

    def test_content_audit_keeps_natural_repeat_without_overlapping_source(self):
        payload = {"kind": "training_sample", "messages": [
            {"role": "user", "text": "Hold on a second."},
            {"role": "user", "text": "Hold on a second."},
            {"role": "assistant", "text": "Okay."},
        ]}
        claim = {"finding": True, "issue_category": "duplicate_speech", "evidence": "same text"}
        triage = {"verified": True, "issue_category": "duplicate_speech", "severity": "medium"}
        row = {"messages": [
            {"source_turn_ids": ["first"]}, {"source_turn_ids": ["second"]}, {"source_turn_ids": ["target"]},
        ]}
        self.assertIsNone(corroborate_finding(payload, claim, triage, context_state="CONTEXT_SUFFICIENT", source_row=row, turn_intervals={"first": (1.0, 2.0), "second": (2.1, 3.0)}))


class TurnReconciliationRegressionTests(unittest.TestCase):
    def test_old_new_exact_duplicate_is_match_existing(self):
        result = reconcile_old_turn(
            old("t1", 0, 2, "I have not, no."),
            [token("have", 0.3, 0.6), token("not", 0.6, 0.9), token("No", 0.9, 1.2)],
        )
        self.assertEqual(result["reconciliation_state"], ReconciliationState.MATCH_EXISTING.value)
        self.assertEqual(result["chosen_text"], "I have not, no.")

    def test_old_turn_tail_echo_is_not_true_new(self):
        result = reconcile_span(
            {
                "candidate_span_id": "echo",
                "start": 2.2,
                "end": 2.6,
                "text": "list",
                "tokens": [token("list", 2.2, 2.6)],
            },
            [],
            preceding_old_turns=[old("old", 0, 2, "I've already added Atlantis and your mom to my list.")],
        )
        self.assertEqual(result["reconciliation_state"], ReconciliationState.BOUNDARY_ECHO.value)
        self.assertFalse(reconciliation_recovery_eligible(result["reconciliation_state"]))
        self.assertEqual(result["boundary_echo"]["old_turn_id"], "old")

    def test_obvious_same_utterance_fragments_merge_before_admission(self):
        spans = [
            {
                "candidate_span_id": "a",
                "start": 0,
                "end": 0.8,
                "text": "I already added",
                "tokens": [token("I already added", 0, 0.8)],
                "speaker_cluster_keys": ["w::C0"],
                "source_window_ids": ["w"],
            },
            {
                "candidate_span_id": "b",
                "start": 1.1,
                "end": 1.8,
                "text": "Atlantis to my list",
                "tokens": [token("Atlantis to my list", 1.1, 1.8)],
                "speaker_cluster_keys": ["w::C0"],
                "source_window_ids": ["w"],
            },
        ]
        merged = merge_obvious_audio_fragments(spans)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["text"], "I already added Atlantis to my list")
        self.assertEqual(merged[0]["fragment_merge_count"], 1)

    def test_new_span_covering_two_old_turns_is_merge_not_third_turn(self):
        values = [old("a", 0, 1, "old A"), old("b", 1, 2, "old B")]
        span = {"candidate_span_id": "c", "text": "old A old B", "tokens": [token("old A", 0, 1), token("old B", 1, 2)]}
        result = reconcile_span(span, values)
        self.assertEqual(result["reconciliation_state"], ReconciliationState.MERGE_EXISTING.value)
        self.assertIsNone(result["chosen_text"])
        self.assertEqual(result["old_turn_ids"], ["a", "b"])

    def test_cross_role_merge_keeps_internal_old_boundaries(self):
        values = [old("a", 0, 1, "assistant A", "assistant"), old("b", 1, 2, "user B", "user")]
        span = {"candidate_span_id": "c", "text": "assistant A user B", "tokens": [token("assistant A", 0, 1), token("user B", 1, 2)]}
        result = reconcile_span(span, values)
        self.assertEqual(result["reconciliation_state"], ReconciliationState.MERGE_EXISTING.value)
        self.assertEqual(len(result["old_turn_ids"]), 2)
        self.assertFalse(result.get("materialized_as_single_turn", False))

    def test_fragment_chain_groups_before_role_materialization(self):
        diarization = [{"start": 0, "end": 3, "speaker_cluster": "S0", "overlap": False}]
        spans = build_candidate_spans(
            [token("I have", 0, 0.8), token("a better idea", 1.0, 2.0)],
            diarization,
            recording_id="r",
            window_ids=["w"],
        )
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]["text"], "I have a better idea")

    def test_bidirectional_edge_smear_tokens_are_reassigned_by_time(self):
        turns = [old("left", 0, 1.4, "Magnifying glass"), old("right", 1.5, 3.0, "What's in the chamber")]
        assigned, unassigned, ambiguous = assign_tokens_to_old_turns(
            [
                token("Magnifying", 0.2, 0.7),
                token("glass", 0.8, 1.2),
                token("What's", 1.6, 1.9),
                token("in", 1.9, 2.1),
                token("the chamber", 2.1, 2.8),
            ],
            turns,
        )
        self.assertFalse(unassigned)
        self.assertFalse(ambiguous)
        self.assertEqual([item["text"] for item in assigned["left"]], ["Magnifying", "glass"])
        self.assertEqual([item["text"] for item in assigned["right"]], ["What's", "in", "the chamber"])

    def test_legacy_split_role_continuation_requires_frozen_identity(self):
        confirmed = resolve_speaker_state("NEURO_FAMILY_HIGH", generic_diarization_available=True)
        unknown = resolve_speaker_state(None, generic_diarization_available=True)
        self.assertEqual(confirmed["speaker_state"], SpeakerState.CONFIRMED_TARGET.value)
        self.assertEqual(confirmed["role"], "assistant")
        self.assertEqual(unknown["speaker_state"], SpeakerState.AMBIGUOUS_SPEAKER.value)
        self.assertIsNone(unknown["role"])

    def test_ambiguous_reconciliation_cannot_recover_new_context(self):
        result = reconcile_old_turn(
            old("t1", 0, 2, "I don't think that's a good idea."),
            [token("Let's", 0, 0.3), token("go inside", 0.3, 0.8), token("the cave now", 0.8, 1.5)],
        )
        self.assertEqual(result["reconciliation_state"], ReconciliationState.AMBIGUOUS.value)
        self.assertFalse(reconciliation_recovery_eligible(result["reconciliation_state"]))

    def test_cluster_anchor_can_resolve_true_new_without_diarization_becoming_identity(self):
        old_turn = context_turn("anchor", 0, 2, "user", "Known guest speech")
        old_turn["speaker_resolution"] = {"resolution_basis": "FROZEN_IDENTITY_AUTHORITY"}
        diarization = [{"start": 0, "end": 2, "speaker_cluster": "C0", "source_window_ids": ["w0"], "overlap": False}]
        anchors = build_cluster_speaker_anchors(diarization, [old_turn])
        result = resolve_anchored_cluster_speaker("w0::C0", anchors)
        self.assertEqual(result["speaker_state"], SpeakerState.CONFIRMED_NON_TARGET.value)
        self.assertFalse(result["evidence"]["generic_diarization_used_for_identity"])
        self.assertTrue(result["evidence"]["anchored_cluster_continuity_used"])

    def test_split_existing_creates_role_resolved_children(self):
        anchor_user = context_turn("u", 0, 1, "user", "Known guest")
        anchor_user["speaker_resolution"] = {"resolution_basis": "FROZEN_IDENTITY_AUTHORITY"}
        anchor_assistant = context_turn("a", 2, 3, "assistant", "Known target")
        anchor_assistant["speaker_resolution"] = {"resolution_basis": "FROZEN_IDENTITY_AUTHORITY"}
        anchors = build_cluster_speaker_anchors(
            [
                {"start": 0, "end": 1, "speaker_cluster": "U", "source_window_ids": ["w"], "overlap": False},
                {"start": 2, "end": 3, "speaker_cluster": "A", "source_window_ids": ["w"], "overlap": False},
            ],
            [anchor_user, anchor_assistant],
            minimum_seconds=0.5,
        )
        # Each local diarization label is explicitly scoped to the window.
        values = [
            {**token("What's your", 10.0, 10.5), "boundary_speaker_cluster_keys": ["w::A"], "source_window_ids": ["w"]},
            {**token("plan this time", 10.5, 11.0), "boundary_speaker_cluster_keys": ["w::A"], "source_window_ids": ["w"]},
            {**token("My plan is", 11.1, 11.6), "boundary_speaker_cluster_keys": ["w::U"], "source_window_ids": ["w"]},
            {**token("to explore more", 11.6, 12.2), "boundary_speaker_cluster_keys": ["w::U"], "source_window_ids": ["w"]},
        ]
        children = split_existing_turn(old("broken", 10, 12.3, "combined"), values, anchors)
        self.assertEqual([child["role"] for child in children], ["assistant", "user"])
        self.assertTrue(all(reconciliation_recovery_eligible(child["reconciliation_state"]) for child in children))
        self.assertEqual(children[0]["source_window_ids"], ["w"])

    def test_merge_existing_materializes_corrected_topology(self):
        anchor = context_turn("a", 0, 2, "assistant", "Known target")
        anchor["speaker_resolution"] = {"resolution_basis": "FROZEN_IDENTITY_AUTHORITY"}
        anchors = build_cluster_speaker_anchors(
            [{"start": 0, "end": 2, "speaker_cluster": "A", "source_window_ids": ["w"], "overlap": False}],
            [anchor],
            minimum_seconds=0.5,
        )
        turns = [
            old("x", 10, 11, "not even God", "assistant"),
            old("y", 11, 12, "will save you from my wrath.", "user"),
        ]
        span = {
            "recording_id": "r",
            "speaker_cluster_keys": ["w::A"],
            "generic_overlap": False,
            "text": "not even God will save you from my wrath",
            "tokens": [token("not even God", 10, 11), token("will save you from my wrath", 11, 12)],
        }
        merged = merge_existing_turns(span, turns, anchors, frozen_target_turn_ids=set())
        self.assertIsNotNone(merged)
        self.assertEqual(merged["role"], "assistant")
        self.assertEqual(merged["reconciliation_state"], ReconciliationState.MERGE_EXISTING.value)

    def test_same_local_cluster_name_cannot_cross_window_identity(self):
        guest = context_turn("guest", 0, 2, "user", "Known guest")
        guest["speaker_resolution"] = {"resolution_basis": "FROZEN_IDENTITY_AUTHORITY"}
        target = context_turn("target", 10, 12, "assistant", "Known target")
        target["speaker_resolution"] = {"resolution_basis": "FROZEN_IDENTITY_AUTHORITY"}
        anchors = build_cluster_speaker_anchors(
            [
                {"start": 0, "end": 2, "speaker_cluster": "ECAPA_CLUSTER_00", "source_window_ids": ["window-a"], "overlap": False},
                {"start": 10, "end": 12, "speaker_cluster": "ECAPA_CLUSTER_00", "source_window_ids": ["window-b"], "overlap": False},
            ],
            [guest, target],
            minimum_seconds=0.5,
        )
        self.assertNotEqual(anchors["window-a::ECAPA_CLUSTER_00"]["speaker_state"], anchors["window-b::ECAPA_CLUSTER_00"]["speaker_state"])
        result = resolve_anchored_cluster_speaker(
            ["window-a::ECAPA_CLUSTER_00", "window-b::ECAPA_CLUSTER_00"], anchors,
        )
        self.assertEqual(result["speaker_state"], SpeakerState.AMBIGUOUS_SPEAKER.value)
        self.assertTrue(result["evidence"]["cluster_anchor_conflict"])

    def test_bare_local_cluster_label_is_fail_closed(self):
        result = resolve_anchored_cluster_speaker("ECAPA_CLUSTER_00", {})
        self.assertEqual(result["speaker_state"], SpeakerState.AMBIGUOUS_SPEAKER.value)
        self.assertTrue(result["evidence"]["unscoped_cluster_authority"])


class SpeakerAndSentinelRegressionTests(unittest.TestCase):
    def test_standalone_filtered_is_sentinel_but_natural_sentence_is_not(self):
        self.assertTrue(is_non_conversational_sentinel("Filtered."))
        self.assertFalse(is_non_conversational_sentinel("That word was filtered yesterday."))
        self.assertTrue(sentinel_only_context([{"resolved_text": "Filtered."}]))
        self.assertFalse(sentinel_only_context([{"resolved_text": "Filtered."}, {"resolved_text": "Real speech."}]))

    def test_unknown_speaker_does_not_default_to_user(self):
        result = resolve_speaker_state(None)
        self.assertEqual(result["speaker_state"], SpeakerState.AMBIGUOUS_SPEAKER.value)
        self.assertIsNone(result["role"])

    def test_activity_only_evidence_cannot_promote_target_identity(self):
        result = resolve_speaker_state(None, target_activity_available=True)
        self.assertEqual(result["speaker_state"], SpeakerState.AMBIGUOUS_SPEAKER.value)
        self.assertFalse(result["evidence"]["activity_used_for_identity"])

    def test_baseline_role_is_retained_monotonically_without_new_identity(self):
        result = resolve_speaker_state(None, baseline_role="user", target_activity_available=False)
        self.assertEqual(result["speaker_state"], SpeakerState.CONFIRMED_NON_TARGET.value)
        self.assertEqual(result["resolution_basis"], "BASELINE_MATERIALIZED_ROLE_MONOTONICITY")

    def test_frozen_semantic_context_role_is_not_unknown_defaulting(self):
        result = resolve_speaker_state(None, frozen_semantic_role="user")
        self.assertEqual(result["speaker_state"], SpeakerState.CONFIRMED_NON_TARGET.value)
        self.assertEqual(result["resolution_basis"], "FROZEN_SEMANTIC_CONTEXT_ROLE")
        self.assertFalse(result["evidence"]["activity_used_for_identity"])


class TargetAndContextRegressionTests(unittest.TestCase):
    def test_stepfun_chat_endpoint_uses_json_mode_and_bearer_key(self):
        context = [context_turn("question", 0, 1, "user", "Are you coming tomorrow?")]
        target = context_turn("target", 1.1, 1.5, "assistant", "Probably.")
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = json.dumps({
            "choices": [{"message": {"content": json.dumps({
                "state": "CONTEXT_SUFFICIENT",
                "relation": "TARGET_RESPONDS_TO_CONTEXT",
                "immediate_trigger_turn_id": "question",
                "confidence": 0.95,
                "reason": "direct answer",
            })}}]
        }).encode("utf-8")
        with patch.dict(os.environ, {"STEPFUN_API_KEY": "test-secret"}, clear=False):
            with patch("audio_evidence.recovery_v2.urlopen", return_value=response) as open_url:
                result = call_context_sufficiency_judge(
                    context,
                    target,
                    model="step-3.7-flash",
                    endpoint="https://api.stepfun.com/v1/chat/completions",
                )
        self.assertEqual(result["state"], ContextSufficiencyState.CONTEXT_SUFFICIENT.value)
        request = open_url.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer test-secret")
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(body["reasoning_effort"], "low")
        self.assertEqual(body["max_tokens"], 8192)
        self.assertEqual(body["messages"][0]["role"], "system")

    def test_stepfun_without_key_fails_closed_without_network_call(self):
        context = [context_turn("question", 0, 1, "user", "Are you coming tomorrow?")]
        target = context_turn("target", 1.1, 1.5, "assistant", "Probably.")
        with patch.dict(os.environ, {}, clear=True):
            with patch("audio_evidence.recovery_v2.urlopen") as open_url:
                result = call_context_sufficiency_judge(
                    context,
                    target,
                    model="step-1-8k",
                    endpoint="https://api.stepfun.com/v1/chat/completions",
                )
        self.assertEqual(result["reason"], "JUDGE_API_KEY_MISSING:STEPFUN_API_KEY")
        open_url.assert_not_called()

    def test_lexically_disjoint_context_reaches_semantic_judge(self):
        context = [context_turn("question", 0, 1, "user", "Are you coming tomorrow?")]
        target = context_turn("target", 1.1, 1.5, "assistant", "Probably.")
        with TemporaryDirectory() as directory:
            with patch("run_audio_evidence_recovery_v2.call_context_sufficiency_judge") as call:
                call.return_value = {
                    "state": ContextSufficiencyState.CONTEXT_SUFFICIENT.value,
                    "checker": "semantic-context-sufficiency-judge-v3",
                    "judge_valid": True,
                }
                result = CachedContextJudge(Path(directory) / "judge.jsonl", model="test", endpoint="http://unused")(context, target)
        self.assertEqual(result["state"], ContextSufficiencyState.CONTEXT_SUFFICIENT.value)
        call.assert_called_once()

    def test_baseline_target_without_new_audio_remains_confirmed(self):
        result = resolve_target(
            old_text="Frozen target.",
            new_text=None,
            baseline_materialized=True,
            alignment_available=False,
            identity_confirmed_target=True,
        )
        self.assertEqual(result["state"], TargetResolutionState.BASELINE_CONFIRMED.value)
        self.assertFalse(result["explicit_contradiction"])

    def test_explicit_substantial_contradiction_is_not_rescued_by_activity(self):
        result = resolve_target(
            old_text="I like apples and quiet afternoons.",
            new_text="Completely unrelated chamber warning now.",
            baseline_materialized=True,
            alignment_available=True,
            identity_confirmed_target=True,
            aligned_token_count=8,
        )
        self.assertEqual(result["state"], TargetResolutionState.AUDIO_MAJOR_CONTRADICTION.value)
        self.assertTrue(result["explicit_contradiction"])

    def test_sentinel_target_is_quarantined_without_semantic_rewrite(self):
        result = resolve_target(
            old_text="Filtered.",
            new_text="different audio words",
            baseline_materialized=True,
            alignment_available=True,
            identity_confirmed_target=True,
            aligned_token_count=4,
        )
        self.assertEqual(result["state"], TargetResolutionState.UNRESOLVED.value)
        self.assertEqual(result["reason"], "NON_CONVERSATIONAL_SENTINEL_TARGET")
        self.assertEqual(result["chosen_text"], "Filtered.")
        self.assertTrue(result["explicit_contradiction"])

    def test_minimal_context_stops_at_first_sufficient_suffix(self):
        candidates = [
            context_turn(f"old-{index}", float(index), float(index) + 0.4, "user" if index % 2 == 0 else "assistant", f"old {index}")
            for index in range(17)
        ]
        candidates.extend([
            context_turn("needed-user", 18.0, 18.8, "user", "What is in the chamber?"),
            context_turn("history-assistant", 19.0, 19.5, "assistant", "Let me look."),
            context_turn("last-user", 20.0, 20.8, "user", "Can you tell me now?"),
        ])
        target = context_turn("target", 21.0, 22.0, "assistant", "It is empty.")
        selected, result = select_minimal_context(candidates, target)
        self.assertEqual(result["state"], ContextSufficiencyState.CONTEXT_SUFFICIENT.value)
        self.assertLessEqual(len(selected), 3)
        self.assertEqual(selected[-1]["audio_turn_id"], "last-user")

    def test_context_sufficiency_does_not_treat_nonempty_as_automatically_sufficient(self):
        target = context_turn("target", 20, 21, "assistant", "Sure.")
        only_assistant = [context_turn("history", 19, 19.5, "assistant", "Earlier.")]
        result = context_sufficiency(only_assistant, target)
        self.assertEqual(result["state"], ContextSufficiencyState.CONTEXT_AMBIGUOUS.value)
        self.assertFalse(result["hidden_history_accessed"])

    def test_context_sufficiency_marks_missing_immediate_trigger(self):
        context = [context_turn("old", 0, 1, "user", "Hacker mans are the worst hackers of all time.")]
        target = context_turn("target", 1.1, 2, "assistant", "I did, with Blender, a free 3D modeling software.")
        result = context_sufficiency(context, target)
        self.assertEqual(result["relation"], ContextRelation.MISSING_IMMEDIATE_TRIGGER.value)
        self.assertNotEqual(result["state"], ContextSufficiencyState.CONTEXT_SUFFICIENT.value)

    def test_context_judge_payload_requires_direction_and_trigger(self):
        context = [context_turn("q", 0, 1, "user", "Where is the password?")]
        target = context_turn("target", 1, 2, "assistant", "It is on page two.")
        payload = build_context_judge_payload(context, target)
        self.assertIn(ContextRelation.TARGET_RESPONDS_TO_CONTEXT.value, payload["allowed_relations"])
        self.assertIn("relation", payload["output_schema"])
        self.assertIn("immediate_trigger_turn_id", payload["output_schema"])
        self.assertEqual(payload["selected_context"][0]["turn_id"], "q")

    def test_judge_cache_key_changes_for_any_authority_revision(self):
        context = [context_turn("q", 0, 1, "user", "Where is the password?")]
        target = context_turn("target", 1, 2, "assistant", "It is on page two.")
        base = context_judge_cache_key(
            context, target, judge_model="m", judge_model_revision="r",
            prompt_revision=CONTEXT_JUDGE_PROMPT_REVISION,
            policy_revision=CONTEXT_JUDGE_POLICY_REVISION,
            schema_revision=CONTEXT_JUDGE_SCHEMA_REVISION,
        )
        changed = context_judge_cache_key(
            context, target, judge_model="m", judge_model_revision="r",
            prompt_revision="changed-prompt",
            policy_revision=CONTEXT_JUDGE_POLICY_REVISION,
            schema_revision=CONTEXT_JUDGE_SCHEMA_REVISION,
        )
        self.assertNotEqual(base, changed)

    def test_target_is_never_selected_as_context(self):
        target = context_turn("target", 5, 6, "assistant", "Answer")
        candidates = [
            context_turn("user", 3, 4, "user", "Question"),
            target,
        ]
        selected, result = select_minimal_context(candidates, target)
        self.assertEqual(result["state"], ContextSufficiencyState.CONTEXT_SUFFICIENT.value)
        self.assertNotIn("target", [turn["audio_turn_id"] for turn in selected])

    def test_semantic_judge_rejects_temporally_adjacent_unrelated_context(self):
        target = context_turn("target", 2, 3, "assistant", "Yeah, the password is on the second page.")
        candidates = [context_turn("pizza", 0, 1, "user", "I like pizza.")]

        def judge(context, _target):
            related = any("password" in turn["resolved_text"].lower() for turn in context)
            return {
                "state": (
                    ContextSufficiencyState.CONTEXT_SUFFICIENT.value
                    if related else ContextSufficiencyState.CONTEXT_INSUFFICIENT.value
                ),
                "checker": "test-semantic-judge",
                "judge_valid": True,
            }

        selected, result = select_minimal_context(candidates, target, judge=judge)
        self.assertFalse(selected)
        self.assertEqual(result["state"], ContextSufficiencyState.CONTEXT_INSUFFICIENT.value)

    def test_semantic_judge_accepts_minimal_password_context(self):
        target = context_turn("target", 4, 5, "assistant", "Yeah, the password is on the second page.")
        candidates = [
            context_turn("where", 0, 1, "user", "Where is the password?"),
            context_turn("history", 1.2, 2, "assistant", "I think it was written somewhere."),
            context_turn("page", 2.2, 3, "user", "Which page?"),
        ]

        def judge(context, _target):
            sufficient = any("password" in turn["resolved_text"].lower() for turn in context) and any(
                "page" in turn["resolved_text"].lower() for turn in context
            )
            return {
                "state": (
                    ContextSufficiencyState.CONTEXT_SUFFICIENT.value
                    if sufficient else ContextSufficiencyState.CONTEXT_INSUFFICIENT.value
                ),
                "checker": "test-semantic-judge",
                "judge_valid": True,
            }

        selected, result = select_minimal_context(candidates, target, judge=judge)
        self.assertEqual(result["state"], ContextSufficiencyState.CONTEXT_SUFFICIENT.value)
        self.assertEqual([turn["audio_turn_id"] for turn in selected], ["where", "history", "page"])


if __name__ == "__main__":
    unittest.main()
