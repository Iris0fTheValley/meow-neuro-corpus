from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from .contracts import ActivitySegment, ContractError, DiarizationTurn


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
    target_activity_policy: Optional[Dict[str, object]] = None

    def to_dict(self):
        return {"use_tse": self.use_tse, "reason_code": self.reason_code.value, "detail": self.detail, "target_activity_policy": self.target_activity_policy}


@dataclass(frozen=True)
class TargetActivityPolicy:
    threshold: float
    version: str

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold <= 1.0:
            raise ContractError("target activity threshold must be within (0, 1]")
        if not self.version:
            raise ContractError("target activity threshold version is required")

    def to_dict(self):
        return {"threshold": self.threshold, "version": self.version}


class AmbiguityRouter:
    def __init__(self, target_activity_policy: Optional[TargetActivityPolicy] = None) -> None:
        self.target_activity_policy = target_activity_policy

    @staticmethod
    def _overlaps(left, right) -> bool:
        return min(left.end, right.end) > max(left.start, right.start)

    def decide(self, activity: List[ActivitySegment], diarization: List[DiarizationTurn], *, severe_background: bool = False, speaker_ambiguity: bool = False, force_tse: bool = False) -> RouteDecision:
        policy = self.target_activity_policy.to_dict() if self.target_activity_policy else None
        if force_tse:
            return RouteDecision(True, RouteReason.EXPLICIT_SMOKE_TEST_OVERRIDE, "explicit architecture smoke-test override", policy)
        if severe_background:
            return RouteDecision(True, RouteReason.SEVERE_BACKGROUND_SPEECH, "upstream evidence marked severe background speech", policy)
        if speaker_ambiguity:
            return RouteDecision(True, RouteReason.DIARIZATION_CONFLICT, "upstream evidence marks target speaker assignment ambiguous", policy)
        if activity and self.target_activity_policy is None:
            raise ContractError("target activity evidence requires an explicit versioned routing threshold")
        target_intervals = [item.interval for item in activity if item.target_probability >= self.target_activity_policy.threshold] if self.target_activity_policy else []
        if any(turn.overlap and any(self._overlaps(turn.interval, target) for target in target_intervals) for turn in diarization):
            return RouteDecision(True, RouteReason.TARGET_OVERLAP, "overlap evidence intersects thresholded target-speaker activity", policy)
        for index, left in enumerate(diarization):
            for right in diarization[index + 1:]:
                if left.speaker_cluster == right.speaker_cluster or not self._overlaps(left.interval, right.interval):
                    continue
                if any(self._overlaps(left.interval, target) and self._overlaps(right.interval, target) for target in target_intervals):
                    return RouteDecision(True, RouteReason.MULTI_SPEAKER_CONFLICT, "simultaneous anonymous speakers intersect thresholded target activity", policy)
        return RouteDecision(False, RouteReason.CLEAN_SINGLE_SPEAKER, "raw waveform is the default ASR path", policy)
