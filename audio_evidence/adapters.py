from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .contracts import ActivitySegment, AlignmentUnit, DiarizationTurn, ModelProvenance, TranscriptHypothesis
from .enrollment import EnrollmentReference


class AdapterUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class AudioInput:
    uri: str
    checksum: str
    recording_id: str
    interval: Any

    def __post_init__(self) -> None:
        from .contracts import ContractError, Interval
        if not self.uri or not self.recording_id or len(self.checksum) != 64:
            raise ContractError("audio input requires URI, recording id, and SHA-256")
        if not isinstance(self.interval, Interval):
            raise ContractError("audio input must carry its exact bounded interval")


class TargetActivityAdapter(ABC):
    provenance: ModelProvenance

    @abstractmethod
    def detect(self, audio: AudioInput, enrollment: EnrollmentReference) -> List[ActivitySegment]:
        raise NotImplementedError


class DiarizationAdapter(ABC):
    provenance: ModelProvenance

    @abstractmethod
    def diarize(self, audio: AudioInput) -> List[DiarizationTurn]:
        raise NotImplementedError


class TargetSpeakerExtractionAdapter(ABC):
    provenance: ModelProvenance

    @abstractmethod
    def extract(self, audio: AudioInput, enrollment: EnrollmentReference) -> AudioInput:
        raise NotImplementedError


class ASRAdapter(ABC):
    provenance: ModelProvenance

    @abstractmethod
    def transcribe(self, audio: AudioInput) -> TranscriptHypothesis:
        raise NotImplementedError


class ForcedAlignmentAdapter(ABC):
    provenance: ModelProvenance

    @abstractmethod
    def align(self, audio: AudioInput, text: str) -> List[AlignmentUnit]:
        raise NotImplementedError


class EnrollmentEmbeddingAdapter(ABC):
    """Produces a backend-native embedding only from a confirmed reference."""

    provenance: ModelProvenance

    @abstractmethod
    def embed(self, enrollment: EnrollmentReference) -> Dict[str, Any]:
        raise NotImplementedError


class CallableTargetActivityAdapter(TargetActivityAdapter):
    """Dependency boundary for nomo-pvad or another enrollment-conditioned backend."""

    def __init__(self, provenance: ModelProvenance, function: Callable[[AudioInput, EnrollmentReference], List[ActivitySegment]]) -> None:
        self.provenance, self._function = provenance, function

    def detect(self, audio: AudioInput, enrollment: EnrollmentReference) -> List[ActivitySegment]:
        return self._function(audio, enrollment)


class CallableDiarizationAdapter(DiarizationAdapter):
    """Dependency boundary for pyannote Community-1 or another clusterer."""

    def __init__(self, provenance: ModelProvenance, function: Callable[[AudioInput], List[DiarizationTurn]]) -> None:
        self.provenance, self._function = provenance, function

    def diarize(self, audio: AudioInput) -> List[DiarizationTurn]:
        return self._function(audio)


class CallableTSEAdapter(TargetSpeakerExtractionAdapter):
    """Keeps WeSep/REAL-TSE native enrollment handling behind its own adapter."""

    def __init__(self, provenance: ModelProvenance, function: Callable[[AudioInput, EnrollmentReference], AudioInput]) -> None:
        self.provenance, self._function = provenance, function

    def extract(self, audio: AudioInput, enrollment: EnrollmentReference) -> AudioInput:
        return self._function(audio, enrollment)


class CallableASRAdapter(ASRAdapter):
    def __init__(self, provenance: ModelProvenance, function: Callable[[AudioInput], TranscriptHypothesis]) -> None:
        self.provenance, self._function = provenance, function

    def transcribe(self, audio: AudioInput) -> TranscriptHypothesis:
        return self._function(audio)


class CallableForcedAlignmentAdapter(ForcedAlignmentAdapter):
    def __init__(self, provenance: ModelProvenance, function: Callable[[AudioInput, str], List[AlignmentUnit]]) -> None:
        self.provenance, self._function = provenance, function

    def align(self, audio: AudioInput, text: str) -> List[AlignmentUnit]:
        return self._function(audio, text)


class CallableEnrollmentEmbeddingAdapter(EnrollmentEmbeddingAdapter):
    def __init__(self, provenance: ModelProvenance, function: Callable[[EnrollmentReference], Dict[str, Any]]) -> None:
        self.provenance, self._function = provenance, function

    def embed(self, enrollment: EnrollmentReference) -> Dict[str, Any]:
        return self._function(enrollment)


class NomoPVADAdapter(CallableTargetActivityAdapter):
    """Thin nomo-pvad boundary; the callable owns its supported enrollment encoding."""

    def __init__(self, revision: str, function, parameters: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(ModelProvenance("target_activity", "nomo-pvad", revision, parameters or {}), function)


class PyannoteCommunity1Adapter(CallableDiarizationAdapter):
    def __init__(self, revision: str, function, parameters: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(ModelProvenance("diarization", "pyannote-community-1", revision, parameters or {}), function)


class WeSepAdapter(CallableTSEAdapter):
    """WeSep boundary. It must load/derive its native enrollment representation."""

    def __init__(self, revision: str, function, parameters: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(ModelProvenance("tse", "wesep", revision, parameters or {}), function)


class RealTSEAdapter(CallableTSEAdapter):
    """REAL-TSE boundary. No ERes2Net embedding compatibility is assumed."""

    def __init__(self, revision: str, function, parameters: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(ModelProvenance("tse", "real-tse", revision, parameters or {}), function)


class Qwen3ASRAdapter(CallableASRAdapter):
    def __init__(self, revision: str, function, parameters: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(ModelProvenance("asr", "qwen3-asr", revision, parameters or {}), function)


class Qwen3ForcedAlignerAdapter(CallableForcedAlignmentAdapter):
    def __init__(self, revision: str, function, parameters: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(ModelProvenance("alignment", "qwen3-forced-aligner", revision, parameters or {}), function)
