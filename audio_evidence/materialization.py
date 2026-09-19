from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any, Dict, Iterable, List

from .contracts import ContractError, MATERIALIZATION_VERSION, canonical_sha256
from .transcript import MATERIALIZATION_BLOCKING_FLAGS, TextAuthority, TextResolutionState


class FinalViewMembership(str, Enum):
    IN_TRAIN = "IN_TRAIN"
    IN_VALIDATION = "IN_VALIDATION"
    IN_SEALED_EVAL = "IN_SEALED_EVAL"
    EXCLUDED = "EXCLUDED"


class SupervisionState(str, Enum):
    FINAL_TARGET_ONLY = "FINAL_TARGET_ONLY"
    NO_SUPERVISION = "NO_SUPERVISION"


def role_for_turn(turn: Dict[str, Any]) -> str:
    explicit = str(turn.get("role") or "")
    if explicit in {"user", "assistant"}:
        return explicit
    identity = str(turn.get("identity") or "")
    return "assistant" if identity in {"NEURO", "EVIL_NEURO", "NEURO_FAMILY", "NEURO_FAMILY_HIGH", "NEURO_FAMILY_MEDIUM"} else "user"


def materialize_role_preserving(
    interaction: Dict[str, Any],
    timeline_turns: Iterable[Dict[str, Any]],
    selected_turn_ids: List[str],
    final_target_turn_id: str,
    membership: FinalViewMembership,
) -> Dict[str, Any]:
    by_id = {str(turn.get("audio_turn_id") or turn.get("turn_id")): turn for turn in timeline_turns}
    if not selected_turn_ids or selected_turn_ids[-1] != final_target_turn_id:
        raise ContractError("final target must be the last selected turn")
    if len(selected_turn_ids) != len(set(selected_turn_ids)):
        raise ContractError("a target/context turn cannot be selected twice")
    selected = []
    for turn_id in selected_turn_ids:
        if turn_id not in by_id:
            raise ContractError("selected turn is absent from the canonical audio evidence timeline")
        selected.append(by_id[turn_id])
    recording_ids = {str(turn.get("recording_id")) for turn in selected}
    if len(recording_ids) != 1 or "" in recording_ids:
        raise ContractError("cross-recording context is forbidden")
    interaction_recording = str(interaction.get("recording_id") or "")
    interaction_canonical = str(interaction.get("canonical_recording_id") or "")
    interaction_family = str(interaction.get("recording_family_id") or "")
    if not all((interaction_recording, interaction_canonical, interaction_family)):
        raise ContractError("interaction recording authority is incomplete")
    if recording_ids != {interaction_recording}:
        raise ContractError("selected timeline recording does not match interaction recording authority")
    if {str(turn.get("canonical_recording_id") or "") for turn in selected} != {interaction_canonical}:
        raise ContractError("selected timeline canonical recording does not match interaction authority")
    if {str(turn.get("recording_family_id") or "") for turn in selected} != {interaction_family}:
        raise ContractError("selected timeline recording family does not match split authority")
    messages = []
    for turn_id, turn in zip(selected_turn_ids, selected):
        if turn.get("text_resolution_state") != TextResolutionState.RESOLVED.value:
            raise ContractError("selected turn text evidence is unresolved")
        try:
            TextAuthority(str(turn.get("text_authority")))
        except ValueError:
            raise ContractError("selected turn has no explicit text authority")
        resolution = turn.get("text_resolution_provenance") or {}
        unresolved_flags = MATERIALIZATION_BLOCKING_FLAGS
        active_disagreements = unresolved_flags & set(turn.get("disagreement_flags") or [])
        resolved_disagreements = set(resolution.get("resolved_disagreements") or [])
        if active_disagreements and (resolution.get("status") != "RESOLVED" or not active_disagreements.issubset(resolved_disagreements)):
            raise ContractError("unresolved transcript disagreement cannot enter materialization")
        text = str(turn.get("resolved_text") or "").strip()
        if not text:
            raise ContractError("selected turn has no text evidence; no synthetic prompt may be inserted")
        role = role_for_turn(turn)
        is_final_target = turn_id == final_target_turn_id
        supervise = is_final_target and membership != FinalViewMembership.EXCLUDED
        if is_final_target and role != "assistant":
            raise ContractError("supervised target must preserve the assistant role")
        messages.append({"role": role, "content": text, "supervise": supervise, "source_turn_ids": [turn_id]})
    expected_supervised = 0 if membership == FinalViewMembership.EXCLUDED else 1
    if sum(bool(message["supervise"]) for message in messages) != expected_supervised:
        raise ContractError("target supervision count does not match final view membership")
    target_turn_ids = [final_target_turn_id]
    history_turn_ids = [turn_id for turn_id in selected_turn_ids if turn_id != final_target_turn_id]
    row = {
        "schema_version": "1.0.0",
        "materialization_version": MATERIALIZATION_VERSION,
        "sample_id": str(interaction.get("sample_id")),
        "target_turn_id": final_target_turn_id,
        "recording_id": next(iter(recording_ids)),
        "canonical_recording_id": interaction.get("canonical_recording_id"),
        "recording_family_id": interaction.get("recording_family_id"),
        "messages": messages,
        "context_turn_ids": history_turn_ids,
        "target_turn_ids": target_turn_ids,
        "source_state": interaction.get("source_state", "SEMANTIC_VERIFIED_V2_3"),
        "source_sampling_eligibility": interaction.get("source_sampling_eligibility", interaction.get("training_candidate")),
        "legacy_training_candidate": interaction.get("training_candidate"),
        "legacy_field_semantics": "SOURCE_SAMPLING_ELIGIBILITY_ONLY",
        "final_view_membership": membership.value,
        "final_view_membership_authority": "FINAL_VIEW_MEMBERSHIP",
        "supervision_state": SupervisionState.FINAL_TARGET_ONLY.value if membership != FinalViewMembership.EXCLUDED else SupervisionState.NO_SUPERVISION.value,
        "semantic_truth_ref": interaction.get("semantic_truth_ref"),
        "semantic_truth_sha256": interaction.get("semantic_truth_sha256"),
        "split_authority_ref": interaction.get("split_authority_ref"),
        "split_authority_sha256": interaction.get("split_authority_sha256"),
        "synthetic_prompt": False,
    }
    row["interaction_dedup_key"] = canonical_sha256({"canonical_recording_id": row["canonical_recording_id"], "provenance": interaction.get("provenance"), "context_turn_ids": history_turn_ids, "target_turn_ids": target_turn_ids, "messages": messages})
    return row


def export_prompt_completion(row: Dict[str, Any]) -> Dict[str, Any]:
    """Fallback export for trainers without selective assistant-token masks."""
    messages = row.get("messages") or []
    supervised = [index for index, message in enumerate(messages) if message.get("role") == "assistant" and message.get("supervise")]
    if supervised != [len(messages) - 1]:
        raise ContractError("prompt/completion export requires exactly the final assistant message to be supervised")
    prompt = [{key: value for key, value in message.items() if key != "supervise"} for message in messages[:-1]]
    completion = messages[-1]["content"]
    return {
        "sample_id": row.get("sample_id"),
        "prompt_messages": prompt,
        "completion": completion,
        "completion_role": "assistant",
        "target_turn_id": row.get("target_turn_id"),
        "historical_assistant_loss_leakage": 0,
    }


def belongs_to_train(row: Dict[str, Any]) -> bool:
    if row.get("final_view_membership_authority") != "FINAL_VIEW_MEMBERSHIP":
        raise ContractError("final train membership authority is missing")
    return row.get("final_view_membership") == FinalViewMembership.IN_TRAIN.value
