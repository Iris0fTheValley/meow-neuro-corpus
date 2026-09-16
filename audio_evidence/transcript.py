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
