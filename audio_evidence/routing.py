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
    @staticmethod
    def _overlaps(left, right) -> bool:
        return min(left.end, right.end) > max(left.start, right.start)

    def decide(self, activity: List[ActivitySegment], diarization: List[DiarizationTurn], *, severe_background: bool = False, speaker_ambiguity: bool = False, force_tse: bool = False) -> RouteDecision:
        if force_tse:
            return RouteDecision(True, RouteReason.EXPLICIT_SMOKE_TEST_OVERRIDE, "explicit architecture smoke-test override")
        if severe_background:
            return RouteDecision(True, RouteReason.SEVERE_BACKGROUND_SPEECH, "upstream evidence marked severe background speech")
        if speaker_ambiguity:
            return RouteDecision(True, RouteReason.DIARIZATION_CONFLICT, "upstream evidence marks target speaker assignment ambiguous")
        target_intervals = [item.interval for item in activity if item.target_probability > 0.0]
        if any(turn.overlap and any(self._overlaps(turn.interval, target) for target in target_intervals) for turn in diarization):
            return RouteDecision(True, RouteReason.TARGET_OVERLAP, "overlap evidence intersects target-speaker activity")
        for index, left in enumerate(diarization):
            for right in diarization[index + 1:]:
                if left.speaker_cluster == right.speaker_cluster or not self._overlaps(left.interval, right.interval):
                    continue
                if any(self._overlaps(left.interval, target) and self._overlaps(right.interval, target) for target in target_intervals):
                    return RouteDecision(True, RouteReason.MULTI_SPEAKER_CONFLICT, "simultaneous anonymous speakers intersect target activity")
        return RouteDecision(False, RouteReason.CLEAN_SINGLE_SPEAKER, "raw waveform is the default ASR path")
