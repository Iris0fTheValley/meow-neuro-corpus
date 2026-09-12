from __future__ import annotations

import unittest

from scripts.sft_semantic_closure_v2_core import reconstruct_timeline


def turn(turn_id, speaker, identity, start, end, text, **flags):
    item = {
        "turn_id": turn_id,
        "speaker": speaker,
        "identity": identity,
        "timestamp": {"start": start, "end": end},
        "text": text,
        "speaker_confidence": 0.95,
        "transcription_confidence": 0.9,
        "turn_boundary": "pause_and_punctuation",
    }
    item.update(flags)
    return item


class SemanticClosureRegressionTests(unittest.TestCase):
    def test_prefix_ladder_one_anchor_one_episode(self):
        turns = [
            turn("u1", "GUEST", "NON_TARGET_GUEST", 0, 2, "What should we do next?"),
            turn("n1", "NEURO", "NEURO_FAMILY_HIGH", 2.5, 4, "We can try the left door."),
            turn("n2", "NEURO", "NEURO_FAMILY_HIGH", 5, 6.5, "The blue one looks safer."),
            turn("n3", "NEURO", "NEURO_FAMILY_HIGH", 7.5, 9, "I am going in now."),
        ]
        rows = reconstruct_timeline("source", "recording", turns)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["context_turn_id"], "u1")
        self.assertEqual(rows[0]["target_turn_ids"], ["n1"])

    def test_removed_middle_turn_is_a_boundary(self):
        turns = [
            turn("u1", "GUEST", "NON_TARGET_GUEST", 0, 2, "Did you see the message?"),
            turn("n1", "NEURO", "NEURO_FAMILY_HIGH", 2.5, 4, "Yes, I saw it."),
            turn("bad", "GUEST", "NON_TARGET_GUEST", 4.2, 5, "Filtered Heart _", suspicious_transcription=True),
            turn("n2", "NEURO", "NEURO_FAMILY_HIGH", 5.2, 7, "It was very strange."),
        ]
        rows = reconstruct_timeline("source", "recording", turns)
        self.assertEqual([row["target_turn_ids"] for row in rows], [["n1"]])

    def test_hidden_trigger_like_transition_is_not_continuation(self):
        turns = [
            turn("u1", "GUEST", "NON_TARGET_GUEST", 0, 2, "Are you ready?"),
            turn("n1", "NEURO", "NEURO_FAMILY_HIGH", 2.2, 4, "Yes, I am ready."),
            turn("n2", "NEURO", "NEURO_FAMILY_HIGH", 4.8, 6.5, "Anyway, chess is fun."),
        ]
        rows = reconstruct_timeline("source", "recording", turns)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["target_turn_ids"], ["n1"])

    def test_garbage_context_is_rejected(self):
        turns = [
            turn("u1", "GUEST", "NON_TARGET_GUEST", 0, 2, "Filtered Heart _", suspicious_transcription=True),
            turn("n1", "NEURO", "NEURO_FAMILY_HIGH", 2.2, 4, "I cannot read that."),
        ]
        self.assertEqual(reconstruct_timeline("source", "recording", turns), [])

    def test_valid_direct_pair_is_retained(self):
        turns = [
            turn("u1", "GUEST", "NON_TARGET_GUEST", 0, 2, "Where is my sister Yui?"),
            turn("n1", "NEURO", "NEURO_FAMILY_HIGH", 2.3, 4.5, "Yui's not here right now."),
        ]
        rows = reconstruct_timeline("source", "recording", turns)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["semantic_qa"]["status"], "ACCEPT")

    def test_valid_gameplay_context_is_retained(self):
        turns = [
            turn("u1", "VEDAL", "VEDAL", 0, 3, "The boss is at the gate."),
            turn("n1", "NEURO", "NEURO_FAMILY_MEDIUM", 3.4, 5.5, "I can see it, move left."),
        ]
        rows = reconstruct_timeline("source", "recording", turns)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["semantic_qa"]["status"], "ACCEPT")

    def test_unknown_same_speaker_is_not_user_context(self):
        turns = [
            turn("u1", "NEURO", "UNKNOWN", 0, 2, "I mean the other one."),
            turn("n1", "NEURO", "NEURO_FAMILY_HIGH", 2.2, 4, "No, the blue one."),
        ]
        self.assertEqual(reconstruct_timeline("source", "recording", turns), [])


if __name__ == "__main__":
    unittest.main()
