from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


AUDIO_EVIDENCE_SCHEMA_VERSION = "1.0.0"
ENROLLMENT_BANK_SCHEMA_VERSION = "1.0.0"
PIPELINE_VERSION = "audio-reconstruction-pipeline-v1"
MATERIALIZATION_VERSION = "role-preserving-materialization-v1"


class ContractError(ValueError):
    pass


class Timebase(str, Enum):
    WINDOW_LOCAL_SECONDS = "WINDOW_LOCAL_SECONDS"
    RECORDING_SECONDS = "SECONDS_FROM_RECORDING_START"


class EvidenceAvailability(str, Enum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class Interval:
    start: float
    end: float
    timebase: Timebase = Timebase.RECORDING_SECONDS

    def __post_init__(self) -> None:
        if not isinstance(self.timebase, Timebase):
            raise ContractError("unknown timebase")
        if self.start < 0 or self.end <= self.start:
            raise ContractError("interval must be non-negative and strictly increasing")

    def to_dict(self) -> Dict[str, Any]:
        return {"start": self.start, "end": self.end, "timebase": self.timebase.value}


TIMEBASE_NORMALIZATION_VERSION = "window-local-to-recording-v1"


def interval_from_dict(value: Dict[str, Any]) -> Interval:
    try:
        timebase = Timebase(str(value["timebase"]))
        return Interval(float(value["start"]), float(value["end"]), timebase)
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError("interval has an unknown or incomplete timebase") from exc


def to_recording_interval(interval: Interval, planned_window: Interval) -> Interval:
    """Normalize a bounded-window interval into recording-global seconds."""
    if planned_window.timebase != Timebase.RECORDING_SECONDS:
        raise ContractError("planned window must use recording-global seconds")
    if interval.timebase == Timebase.WINDOW_LOCAL_SECONDS:
        duration = planned_window.end - planned_window.start
        if interval.end > duration:
            raise ContractError("window-local interval lies outside the planned bounded interval")
        normalized = Interval(
            planned_window.start + interval.start,
            planned_window.start + interval.end,
            Timebase.RECORDING_SECONDS,
        )
    elif interval.timebase == Timebase.RECORDING_SECONDS:
        normalized = interval
    else:  # Defensive for objects constructed outside the dataclass contract.
        raise ContractError("unknown timebase")
    if normalized.start < planned_window.start or normalized.end > planned_window.end:
        raise ContractError("recording-global interval lies outside the planned bounded interval")
    return normalized


@dataclass(frozen=True)
class ModelProvenance:
    component: str
    backend: str
    revision: str
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.component or not self.backend or not self.revision:
            raise ContractError("component, backend, and explicit revision are required")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActivitySegment:
    interval: Interval
    target_probability: float
    model: ModelProvenance
    enrollment_id: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.target_probability <= 1.0:
            raise ContractError("target probability outside [0, 1]")
        if not self.enrollment_id:
            raise ContractError("target activity requires enrollment provenance")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": self.interval.start,
            "end": self.interval.end,
            "timebase": self.interval.timebase.value,
            "target_probability": self.target_probability,
            "model": self.model.to_dict(),
            "enrollment_id": self.enrollment_id,
        }


@dataclass(frozen=True)
class DiarizationTurn:
    interval: Interval
    speaker_cluster: str
    overlap: bool
    model: ModelProvenance
    exclusive: bool = False

    def __post_init__(self) -> None:
        if not self.speaker_cluster:
            raise ContractError("diarization cluster is required")

    def to_dict(self) -> Dict[str, Any]:
        value = self.interval.to_dict()
        value.update({"speaker_cluster": self.speaker_cluster, "overlap": self.overlap, "exclusive": self.exclusive, "model": self.model.to_dict()})
        return value


@dataclass(frozen=True)
class TranscriptHypothesis:
    text: str
    model: ModelProvenance
    source_waveform: str
    enrollment_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.source_waveform:
            raise ContractError("ASR source waveform provenance is required")

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["model"] = self.model.to_dict()
        return value


@dataclass(frozen=True)
class AlignmentUnit:
    text: str
    interval: Interval
    confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        value = self.interval.to_dict()
        value.update({"text": self.text, "confidence": self.confidence})
        return value


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def require_fields(value: Dict[str, Any], names: List[str], label: str) -> None:
    missing = [name for name in names if value.get(name) in (None, "", [])]
    if missing:
        raise ContractError("%s missing required fields: %s" % (label, ", ".join(missing)))
