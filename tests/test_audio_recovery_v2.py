from __future__ import annotations

import unittest

from audio_evidence.recovery_v2 import (
    ContextSufficiencyState,
    ReconciliationState,
    SpeakerState,
    TargetResolutionState,
    assign_tokens_to_old_turns,
    build_candidate_spans,
    context_sufficiency,
    is_non_conversational_sentinel,
    reconcile_old_turn,
    reconcile_span,
    resolve_speaker_state,
    resolve_target,
    select_minimal_context,
    sentinel_only_context,
)


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


class TurnReconciliationRegressionTests(unittest.TestCase):
    def test_old_new_exact_duplicate_is_match_existing(self):
        result = reconcile_old_turn(
            old("t1", 0, 2, "I have not, no."),
            [token("have", 0.3, 0.6), token("not", 0.6, 0.9), token("No", 0.9, 1.2)],
        )
        self.assertEqual(result["reconciliation_state"], ReconciliationState.MATCH_EXISTING.value)
        self.assertEqual(result["chosen_text"], "I have not, no.")

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

    def test_target_is_never_selected_as_context(self):
        target = context_turn("target", 5, 6, "assistant", "Answer")
        candidates = [
            context_turn("user", 3, 4, "user", "Question"),
            target,
        ]
        selected, result = select_minimal_context(candidates, target)
        self.assertEqual(result["state"], ContextSufficiencyState.CONTEXT_SUFFICIENT.value)
        self.assertNotIn("target", [turn["audio_turn_id"] for turn in selected])


if __name__ == "__main__":
    unittest.main()
