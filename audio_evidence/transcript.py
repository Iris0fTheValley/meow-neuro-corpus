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
