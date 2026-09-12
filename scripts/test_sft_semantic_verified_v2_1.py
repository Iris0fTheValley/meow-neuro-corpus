from __future__ import annotations

import unittest

try:
    from sft_semantic_verified_v2_1 import validate_judgement, overlap_edges
except ModuleNotFoundError:
    from scripts.sft_semantic_verified_v2_1 import validate_judgement, overlap_edges


class SemanticVerifiedV21Tests(unittest.TestCase):
    def request(self):
        return {"current_target_turn_ids": ["n1"], "candidate_target_turn_ids": ["n1", "n2"]}

    def test_accepts_direct_response_with_explicit_selection(self):
        result = {"parsed": {"relation": "DIRECT_RESPONSE", "confidence": 0.91, "context_complete": True, "response_complete": True, "transcript_usable": True, "episode_decision": "KEEP_CURRENT", "selected_target_turn_ids": ["n1"], "reason": "question and answer"}}
        validated = validate_judgement(result, self.request())
        self.assertTrue(validated["valid"])
        self.assertTrue(validated["accepted"])

    def test_rejects_wrong_context(self):
        result = {"parsed": {"relation": "WRONG_CONTEXT", "confidence": 0.95, "context_complete": True, "response_complete": True, "transcript_usable": True, "episode_decision": "KEEP_CURRENT", "selected_target_turn_ids": ["n1"], "reason": "unrelated"}}
        validated = validate_judgement(result, self.request())
        self.assertTrue(validated["valid"])
        self.assertFalse(validated["accepted"])

    def test_rejects_unoffered_target(self):
        result = {"parsed": {"relation": "DIRECT_RESPONSE", "confidence": 0.95, "context_complete": True, "response_complete": True, "transcript_usable": True, "episode_decision": "MERGE_CONTINUATION", "selected_target_turn_ids": ["n1", "n3"], "reason": "bad selection"}}
        validated = validate_judgement(result, self.request())
        self.assertFalse(validated["valid"])

    def test_rejects_missing_quality_evidence(self):
        result = {"parsed": {"relation": "DIRECT_RESPONSE", "confidence": 0.95, "context_complete": False, "response_complete": True, "transcript_usable": True, "episode_decision": "KEEP_CURRENT", "selected_target_turn_ids": ["n1"], "reason": "incomplete"}}
        validated = validate_judgement(result, self.request())
        self.assertFalse(validated["valid"])

    def test_allows_explicit_continuation_selection(self):
        result = {"parsed": {"relation": "CONTEXTUAL_RESPONSE", "confidence": 0.86, "context_complete": True, "response_complete": True, "transcript_usable": True, "episode_decision": "MERGE_CONTINUATION", "selected_target_turn_ids": ["n1", "n2"], "reason": "same uninterrupted reply"}}
        validated = validate_judgement(result, self.request())
        self.assertTrue(validated["valid"])
        self.assertTrue(validated["accepted"])

    def test_cross_cluster_partial_overlap_is_detected(self):
        rows = [
            {"sample_id": "a", "source_id": "clip", "canonical_recording_id": "cluster-a", "messages": [{"role": "user", "content": "one two three four five six seven eight nine ten eleven twelve"}, {"role": "assistant", "content": "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty"}]},
            {"sample_id": "b", "source_id": "vod", "canonical_recording_id": "cluster-b", "messages": [{"role": "user", "content": "zero one two three four five six seven eight nine ten eleven twelve"}, {"role": "assistant", "content": "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty"}]},
        ]
        edges = overlap_edges(rows)
        self.assertTrue(edges)
        self.assertTrue(any(edge["strong_merge_edge"] for edge in edges))


if __name__ == "__main__":
    unittest.main()
