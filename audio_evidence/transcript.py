from __future__ import annotations

import re
from enum import Enum
from typing import Optional


class TranscriptDisagreement(str, Enum):
    MATCH = "MATCH"
    MINOR_TEXT_CHANGE = "MINOR_TEXT_CHANGE"
    MISSING_OLD_SPEECH = "MISSING_OLD_SPEECH"
    MISSING_NEW_SPEECH = "MISSING_NEW_SPEECH"
    BOUNDARY_CHANGE = "BOUNDARY_CHANGE"
    SPEAKER_ASSIGNMENT_CHANGE = "SPEAKER_ASSIGNMENT_CHANGE"
    MAJOR_TRANSCRIPT_CONFLICT = "MAJOR_TRANSCRIPT_CONFLICT"


class TextAuthority(str, Enum):
    OLD_TRANSCRIPT = "OLD_TRANSCRIPT"
    NEW_ASR = "NEW_ASR"
    RECONCILED_TEXT = "RECONCILED_TEXT"
    ADJUDICATED_TEXT = "ADJUDICATED_TEXT"


class TextResolutionState(str, Enum):
    RESOLVED = "RESOLVED"
    UNRESOLVED = "UNRESOLVED"


UNRESOLVED_BY_DEFAULT = {
    TranscriptDisagreement.MINOR_TEXT_CHANGE,
    TranscriptDisagreement.MISSING_OLD_SPEECH,
    TranscriptDisagreement.MISSING_NEW_SPEECH,
    TranscriptDisagreement.BOUNDARY_CHANGE,
    TranscriptDisagreement.SPEAKER_ASSIGNMENT_CHANGE,
    TranscriptDisagreement.MAJOR_TRANSCRIPT_CONFLICT,
}

# Disagreement classes that materialization must see explicitly adjudicated.
# MINOR_TEXT_CHANGE and MISSING_NEW_SPEECH are adjudicable under the frozen
# semantic-target authority, so they are listed here only when a resolver left
# them unresolved.
MATERIALIZATION_BLOCKING_FLAGS = {
    "MAJOR_TRANSCRIPT_CONFLICT",
    "BOUNDARY_CHANGE",
    "SPEAKER_ASSIGNMENT_CHANGE",
    "MINOR_TEXT_CHANGE",
    "MISSING_OLD_SPEECH",
    "MISSING_NEW_SPEECH",
}

FROZEN_TARGET_RESOLVER_VERSION = "frozen-verified-target-text-authority-v1"

# One cache-key namespace for the resolver so downstream provenance checks can
# assert the authority model that produced a resolution.
FROZEN_TARGET_ADVISORY_FLAGS = ("MINOR_TEXT_CHANGE", "MISSING_NEW_SPEECH")


def automatic_text_resolution(old_text: Optional[str], new_text: Optional[str], disagreement: Optional[TranscriptDisagreement]):
    """Resolve only unambiguous evidence; everything else requires adjudication."""
    old_value, new_value = str(old_text or "").strip(), str(new_text or "").strip()
    if old_value and not new_value and disagreement is None:
        return {
            "text_resolution_state": TextResolutionState.RESOLVED.value,
            "text_authority": TextAuthority.OLD_TRANSCRIPT.value,
            "resolved_text": old_value,
            "text_resolution_provenance": {"status": "RESOLVED", "resolver": "EXISTING_TRANSCRIPT_AUTHORITY", "revision": "audio-evidence-v1"},
        }
    if old_value and new_value and disagreement == TranscriptDisagreement.MATCH:
        return {
            "text_resolution_state": TextResolutionState.RESOLVED.value,
            "text_authority": TextAuthority.NEW_ASR.value,
            "resolved_text": new_value,
            "text_resolution_provenance": {"status": "RESOLVED", "resolver": "EXACT_OLD_NEW_MATCH", "revision": "audio-evidence-v1"},
        }
    return {
        "text_resolution_state": TextResolutionState.UNRESOLVED.value,
        "text_authority": None,
        "resolved_text": None,
        "text_resolution_provenance": None,
    }


def _tokens(text: Optional[str]):
    return re.findall(r"[a-z0-9']+", str(text or "").lower())


def _unresolved(reason: str, disagreement: Optional[TranscriptDisagreement], old_value: str, new_value: str):
    return {
        "text_resolution_state": TextResolutionState.UNRESOLVED.value,
        "text_authority": None,
        "resolved_text": None,
        "text_resolution_provenance": {
            "status": "UNRESOLVED",
            "resolver": FROZEN_TARGET_RESOLVER_VERSION,
            "reason": reason,
            "disagreement": disagreement.value if disagreement is not None else None,
            "revision": "audio-evidence-v1",
            "evidence": {"old_transcript": old_value or None, "new_asr_hypothesis": new_value or None},
        },
    }


def resolve_frozen_target_text(old_text: Optional[str], new_text: Optional[str], disagreement: Optional[TranscriptDisagreement], *, no_new_asr_evidence: bool = False):
    """Resolve the text of an already semantic-verified target turn.

    A verified interaction already owns a frozen semantic truth, so its
    pre-existing transcript is the supervisory text authority while the new ASR
    hypothesis is independent audio-confirmation *evidence*, not a replacement:

    ``MATCH``                    -> RESOLVED, old or equivalent new text
    ``MINOR_TEXT_CHANGE``        -> RESOLVED, OLD_TRANSCRIPT authority, new ASR
                                    kept as explicit disagreement evidence
    ``MISSING_NEW_SPEECH``       -> RESOLVED, OLD_TRANSCRIPT authority (the new
                                    ASR missed speech; the verified target is
                                    never deleted because of that)
    ``MAJOR_TRANSCRIPT_CONFLICT``-> UNRESOLVED (quarantine)
    ``BOUNDARY_CHANGE``          -> UNRESOLVED (quarantine)
    ``SPEAKER_ASSIGNMENT_CHANGE``-> UNRESOLVED (quarantine)
    ``MISSING_OLD_SPEECH``       -> UNRESOLVED; a verified target with no old
                                    speech is a data error and must never spawn
                                    a new supervision target automatically.

    ``no_new_asr_evidence`` is only true when the ASR stage itself produced no
    hypothesis for the window (for example the window routed to target-speaker
    extraction).  A window hypothesis the aligner could not attribute to this
    target turn is a boundary disagreement, not absent evidence, and is
    therefore quarantined rather than silently resolved to the old text.

    The new hypothesis is never silently promoted over the frozen target text.
    """
    if no_new_asr_evidence:
        disagreement = None
    old_value, new_value = str(old_text or "").strip(), str(new_text or "").strip()
    if not old_value:
        return _unresolved("FROZEN_TARGET_OLD_TRANSCRIPT_MISSING", disagreement, old_value, new_value)
    if disagreement is None:
        if new_value:
            return _unresolved("FROZEN_TARGET_DISAGREEMENT_UNCLASSIFIED", disagreement, old_value, new_value)
        # The audio stage produced no hypothesis at all: the frozen target text
        # is the only evidence and remains authoritative.
        return {
            "text_resolution_state": TextResolutionState.RESOLVED.value,
            "text_authority": TextAuthority.OLD_TRANSCRIPT.value,
            "resolved_text": old_value,
            "text_resolution_provenance": {
                "status": "RESOLVED",
                "resolver": "FROZEN_VERIFIED_TARGET_NO_NEW_ASR_EVIDENCE",
                "revision": FROZEN_TARGET_RESOLVER_VERSION,
                "disagreement": None,
                "evidence": {"old_transcript": old_value, "new_asr_hypothesis": None},
            },
        }
    if disagreement == TranscriptDisagreement.MATCH:
        return {
            "text_resolution_state": TextResolutionState.RESOLVED.value,
            "text_authority": TextAuthority.NEW_ASR.value,
            "resolved_text": new_value,
            "text_resolution_provenance": {
                "status": "RESOLVED",
                "resolver": "FROZEN_VERIFIED_TARGET_EXACT_MATCH",
                "revision": FROZEN_TARGET_RESOLVER_VERSION,
                "disagreement": disagreement.value,
                "resolved_disagreements": [disagreement.value],
                "evidence": {"old_transcript": old_value, "new_asr_hypothesis": new_value},
            },
        }
    if disagreement in (TranscriptDisagreement.MINOR_TEXT_CHANGE, TranscriptDisagreement.MISSING_NEW_SPEECH):
        return {
            "text_resolution_state": TextResolutionState.RESOLVED.value,
            "text_authority": TextAuthority.OLD_TRANSCRIPT.value,
            "resolved_text": old_value,
            "text_resolution_provenance": {
                "status": "RESOLVED",
                "resolver": "FROZEN_VERIFIED_TARGET_OLD_TRANSCRIPT_AUTHORITY",
                "revision": FROZEN_TARGET_RESOLVER_VERSION,
                "disagreement": disagreement.value,
                "resolved_disagreements": [disagreement.value],
                "new_asr_role": "INDEPENDENT_AUDIO_CONFIRMATION_EVIDENCE",
                "evidence": {"old_transcript": old_value, "new_asr_hypothesis": new_value or None},
            },
        }
    return _unresolved("FROZEN_TARGET_" + disagreement.value, disagreement, old_value, new_value)


def detect_disagreement(old_text: Optional[str], new_text: Optional[str], *, boundary_changed: bool = False, speaker_assignment_changed: bool = False) -> TranscriptDisagreement:
    if speaker_assignment_changed:
        return TranscriptDisagreement.SPEAKER_ASSIGNMENT_CHANGE
    if boundary_changed:
        return TranscriptDisagreement.BOUNDARY_CHANGE
    old, new = _tokens(old_text), _tokens(new_text)
    if not old and new:
        return TranscriptDisagreement.MISSING_OLD_SPEECH
    if old and not new:
        return TranscriptDisagreement.MISSING_NEW_SPEECH
    if old == new:
        return TranscriptDisagreement.MATCH
    if not old and not new:
        return TranscriptDisagreement.MATCH
    overlap = len(set(old) & set(new)) / float(max(1, len(set(old) | set(new))))
    return TranscriptDisagreement.MINOR_TEXT_CHANGE if overlap >= 0.7 else TranscriptDisagreement.MAJOR_TRANSCRIPT_CONFLICT
