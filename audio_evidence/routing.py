from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List

from .contracts import ActivitySegment, DiarizationTurn


class RouteReason(str, Enum):
    CLEAN_SINGLE_SPEAKER = "CLEAN_SINGLE_SPEAKER"
    TARGET_OVERLAP = "TARGET_OVERLAP"
    MULTI_SPEAKER_CONFLICT = "MULTI_SPEAKER_CONFLICT"
    DIARIZATION_CONFLICT = "DIARIZATION_CONFLICT"
    SEVERE_BACKGROUND_SPEECH = "SEVERE_BACKGROUND_SPEECH"
    EXPLICIT_SMOKE_TEST_OVERRIDE = "EXPLICIT_SMOKE_TEST_OVERRIDE"


@dataclass(frozen=True)
class RouteDecision:
    use_tse: bool
    reason_code: RouteReason
    detail: str

    def to_dict(self):
        return {"use_tse": self.use_tse, "reason_code": self.reason_code.value, "detail": self.detail}


class AmbiguityRouter:
    def decide(self, activity: List[ActivitySegment], diarization: List[DiarizationTurn], *, severe_background: bool = False, force_tse: bool = False) -> RouteDecision:
        if force_tse:
            return RouteDecision(True, RouteReason.EXPLICIT_SMOKE_TEST_OVERRIDE, "explicit architecture smoke-test override")
        if severe_background:
            return RouteDecision(True, RouteReason.SEVERE_BACKGROUND_SPEECH, "upstream evidence marked severe background speech")
        if any(turn.overlap for turn in diarization):
            return RouteDecision(True, RouteReason.TARGET_OVERLAP, "diarization reports overlap in the planned window")
        clusters = {turn.speaker_cluster for turn in diarization}
        if len(clusters) > 1 and activity:
            return RouteDecision(True, RouteReason.MULTI_SPEAKER_CONFLICT, "multiple generic clusters coexist with target activity")
        if not activity and len(clusters) > 1:
            return RouteDecision(True, RouteReason.DIARIZATION_CONFLICT, "target activity unavailable and multiple clusters are present")
        return RouteDecision(False, RouteReason.CLEAN_SINGLE_SPEAKER, "raw waveform is the default ASR path")
