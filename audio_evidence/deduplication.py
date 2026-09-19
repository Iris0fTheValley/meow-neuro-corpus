from __future__ import annotations

"""Deterministic deduplication for materialized role-preserving rows.

Two independent stages run in a fixed order and accumulate into a single
removed ledger:

1. hard dedup on ``interaction_dedup_key`` (identical canonical interaction
   provenance, context and target);
2. prefix-ladder dedup among rows sharing the same recording and the same
   context turn set, where one supervised target is a token-prefix/suffix
   containment of another.

The returned ``removed`` list is the single authoritative removal ledger: the
run writes it to ``dedup.jsonl``, records its length in the run manifest as
``dedup_removed``, and passes it to the finalizer.  Keeping one list (instead
of separate per-stage lists) is what prevents a stage's removals from being
silently dropped from the manifest and the finalizer input.
"""

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple


def _target_tokens(row: Dict[str, Any]) -> List[str]:
    messages = row.get("messages") or [{}]
    return str(messages[-1].get("content") or "").lower().split()


def _is_containment(left: List[str], right: List[str]) -> bool:
    if not left or not right or left == right:
        return False
    if len(left) <= len(right):
        return any(right[index:index + len(left)] == left for index in range(len(right) - len(left) + 1))
    return any(left[index:index + len(right)] == right for index in range(len(left) - len(right) + 1))


def deduplicate_materialized(rows: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return ``(kept_rows, removed_rows)`` under the frozen dedup policy."""
    removed: List[Dict[str, Any]] = []
    deduped, seen = [], set()
    for row in sorted(rows, key=lambda x: x["sample_id"]):
        key = row.get("interaction_dedup_key")
        if key in seen:
            removed.append({"sample_id": row["sample_id"], "reason": "duplicate_interaction_provenance"})
        else:
            seen.add(key)
            deduped.append(row)

    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in deduped:
        groups[(row.get("recording_id"), tuple(row.get("context_turn_ids") or []))].append(row)
    keep_ids = set()
    for _, values in groups.items():
        ordered = sorted(values, key=lambda x: x["sample_id"])
        for index, left in enumerate(ordered):
            left_tokens = _target_tokens(left)
            drop = False
            for right in ordered[:index]:
                if _is_containment(left_tokens, _target_tokens(right)):
                    removed.append({"sample_id": left["sample_id"], "reason": "prefix_ladder", "kept_sample_id": right["sample_id"]})
                    drop = True
                    break
            if not drop:
                keep_ids.add(left["sample_id"])
    return [row for row in deduped if row["sample_id"] in keep_ids], removed


def removal_counts(removed: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for entry in removed:
        counts[str(entry.get("reason"))] = counts.get(str(entry.get("reason")), 0) + 1
    return counts
