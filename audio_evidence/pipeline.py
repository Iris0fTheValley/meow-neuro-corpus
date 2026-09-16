from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

from .adapters import (
    ASRAdapter,
    AdapterUnavailable,
    AudioInput,
    DiarizationAdapter,
    ForcedAlignmentAdapter,
    TargetActivityAdapter,
    TargetSpeakerExtractionAdapter,
)
from .cache import CheckpointStore, EvidenceCache
from .contracts import AUDIO_EVIDENCE_SCHEMA_VERSION, PIPELINE_VERSION, ActivitySegment, AlignmentUnit, DiarizationTurn, Interval, ModelProvenance, Timebase, TranscriptHypothesis
from .enrollment import EnrollmentBank
from .routing import AmbiguityRouter
from .transcript import detect_disagreement


def _activity_from_dict(value: Dict[str, Any]) -> ActivitySegment:
    from .contracts import ModelProvenance
    model = value["model"]
    return ActivitySegment(Interval(float(value["start"]), float(value["end"]), Timebase(value["timebase"])), float(value["target_probability"]), ModelProvenance(**model), str(value["enrollment_id"]))


def _diarization_from_dict(value: Dict[str, Any]) -> DiarizationTurn:
    from .contracts import ModelProvenance
    model = value["model"]
    return DiarizationTurn(Interval(float(value["start"]), float(value["end"]), Timebase(value["timebase"])), str(value["speaker_cluster"]), bool(value["overlap"]), ModelProvenance(**model), bool(value.get("exclusive")))


class AudioEvidencePipeline:
    """Bounded per-window orchestration; semantic and identity authorities stay upstream."""

    def __init__(
        self,
        bank: EnrollmentBank,
        cache: EvidenceCache,
        checkpoint: CheckpointStore,
        *,
        activity: Optional[TargetActivityAdapter],
        diarization: Optional[DiarizationAdapter],
        asr: Optional[ASRAdapter],
        aligner: Optional[ForcedAlignmentAdapter],
        tse: Optional[TargetSpeakerExtractionAdapter] = None,
        router: Optional[AmbiguityRouter] = None,
        allow_synthetic: bool = False,
    ) -> None:
        self.bank, self.cache, self.checkpoint = bank, cache, checkpoint
        self.activity, self.diarization, self.asr, self.aligner, self.tse = activity, diarization, asr, aligner, tse
        self.router = router or AmbiguityRouter()
        self.allow_synthetic = allow_synthetic

    def _cached(self, stage: str, window: Dict[str, Any], revision: str, parameters: Dict[str, Any], enrollment_revision: str, function):
        key = self.cache.key(str(window["audio_checksum"]), dict(window["merged_interval"]), revision, parameters, enrollment_revision)
        payload, hit = self.cache.get_or_compute(stage, key, function)
        self.checkpoint.record(str(window["window_id"]), stage, key)
        return payload, hit

    def process_window(self, window: Dict[str, Any], enrollment_id: str, old_turns: List[Dict[str, Any]], *, force_tse: bool = False, severe_background: bool = False) -> Dict[str, Any]:
        enrollment = self.bank.confirmed(enrollment_id, allow_synthetic=self.allow_synthetic)
        enrollment_cache_revision = "%s:%s:%s:%s" % (self.bank.revision, enrollment.enrollment_id, enrollment.checksum, enrollment.embedding_revision)
        interval = window.get("merged_interval") or {}
        if interval.get("timebase") != Timebase.RECORDING_SECONDS.value:
            raise ValueError("unknown timebase")
        audio = AudioInput(str(window["source_audio"]), str(window["audio_checksum"]), str(window["recording_id"]))
        stage_status: Dict[str, str] = {}
        cache_hits: Dict[str, bool] = {}

        activity_values: List[Dict[str, Any]] = []
        if self.activity is not None:
            try:
                activity_values, cache_hits["target_activity"] = self._cached("target_activity", window, self.activity.provenance.revision, self.activity.provenance.parameters, enrollment_cache_revision, lambda: [item.to_dict() for item in self.activity.detect(audio, enrollment)])
                stage_status["target_activity"] = "AVAILABLE"
            except AdapterUnavailable:
                stage_status["target_activity"] = "UNAVAILABLE"
        else:
            stage_status["target_activity"] = "UNAVAILABLE"
        activity = [_activity_from_dict(item) for item in activity_values]

        diarization_values: List[Dict[str, Any]] = []
        if self.diarization is not None:
            try:
                diarization_values, cache_hits["diarization"] = self._cached("diarization", window, self.diarization.provenance.revision, self.diarization.provenance.parameters, enrollment_cache_revision, lambda: [item.to_dict() for item in self.diarization.diarize(audio)])
                stage_status["diarization"] = "AVAILABLE"
            except AdapterUnavailable:
                stage_status["diarization"] = "UNAVAILABLE"
        else:
            stage_status["diarization"] = "UNAVAILABLE"
        diarization = [_diarization_from_dict(item) for item in diarization_values]
        route = self.router.decide(activity, diarization, severe_background=severe_background, force_tse=force_tse)

        asr_audio = audio
        tse_source = None
        if route.use_tse:
            if self.tse is None:
                stage_status["tse"] = "UNRESOLVED"
            else:
                try:
                    extracted_value, cache_hits["tse"] = self._cached(
                        "tse", window, self.tse.provenance.revision, self.tse.provenance.parameters,
                        enrollment_cache_revision,
                        lambda: self._audio_to_dict(self.tse.extract(audio, enrollment)),
                    )
                    extracted = AudioInput(**extracted_value)
                    asr_audio = extracted
                    stage_status["tse"] = "AVAILABLE"
                    tse_source = {"source_audio_checksum": audio.checksum, "output_audio_checksum": extracted.checksum, "enrollment_id": enrollment.enrollment_id, "model": self.tse.provenance.to_dict()}
                except AdapterUnavailable:
                    stage_status["tse"] = "UNRESOLVED"
        else:
            stage_status["tse"] = "NOT_ROUTED"

        hypothesis = None
        if self.asr is not None and not (route.use_tse and stage_status["tse"] == "UNRESOLVED"):
            try:
                asr_parameters = dict(self.asr.provenance.parameters)
                asr_parameters["source_audio_checksum"] = asr_audio.checksum
                hypothesis_value, cache_hits["asr"] = self._cached(
                    "asr", window, self.asr.provenance.revision, asr_parameters, enrollment_cache_revision,
                    lambda: self.asr.transcribe(asr_audio).to_dict(),
                )
                model = hypothesis_value["model"]
                hypothesis = TranscriptHypothesis(
                    str(hypothesis_value["text"]), ModelProvenance(**model),
                    str(hypothesis_value["source_waveform"]), hypothesis_value.get("enrollment_id"),
                )
                stage_status["asr"] = "AVAILABLE"
            except AdapterUnavailable:
                stage_status["asr"] = "UNAVAILABLE"
        else:
            stage_status["asr"] = "UNAVAILABLE" if not route.use_tse else "UNRESOLVED"
        alignment: List[AlignmentUnit] = []
        if hypothesis is not None and self.aligner is not None:
            try:
                alignment_parameters = dict(self.aligner.provenance.parameters)
                alignment_parameters.update({"source_audio_checksum": asr_audio.checksum, "text_sha256": hashlib.sha256(hypothesis.text.encode("utf-8")).hexdigest()})
                alignment_values, cache_hits["alignment"] = self._cached(
                    "alignment", window, self.aligner.provenance.revision, alignment_parameters, enrollment_cache_revision,
                    lambda: [item.to_dict() for item in self.aligner.align(asr_audio, hypothesis.text)],
                )
                alignment = [AlignmentUnit(str(item["text"]), Interval(float(item["start"]), float(item["end"]), Timebase(item["timebase"])), item.get("confidence")) for item in alignment_values]
                stage_status["alignment"] = "AVAILABLE"
            except AdapterUnavailable:
                stage_status["alignment"] = "UNAVAILABLE"
        else:
            stage_status["alignment"] = "UNAVAILABLE"

        turns = []
        source_turns = old_turns or [{"audio_turn_id": "window:" + str(window["window_id"]), "start": interval["start"], "end": interval["end"], "old_transcript": None, "speaker_cluster": None}]
        multiple_source_turns = len(source_turns) > 1
        for index, old in enumerate(source_turns):
            turn_id = str(old.get("audio_turn_id") or old.get("turn_id") or (str(window["window_id"]) + ":" + str(index)))
            old_text = old.get("old_transcript", old.get("text"))
            turn_start = float(old.get("start", interval["start"]))
            turn_end = float(old.get("end", interval["end"]))
            turn_alignment = [item for item in alignment if turn_start <= (item.interval.start + item.interval.end) / 2.0 <= turn_end]
            if hypothesis is None:
                new_text = None
            elif not multiple_source_turns:
                new_text = hypothesis.text
            elif turn_alignment:
                new_text = " ".join(item.text for item in turn_alignment).strip()
            else:
                # A window-level hypothesis must not be copied into every old
                # turn. Until alignment resolves boundaries it remains window
                # evidence, not a silent per-turn transcript replacement.
                new_text = old.get("new_asr_hypothesis")
            overlapping_diarization = [item for item in diarization if min(turn_end, item.interval.end) > max(turn_start, item.interval.start)]
            speaker_cluster = old.get("speaker_cluster")
            if speaker_cluster is None and overlapping_diarization:
                speaker_cluster = max(overlapping_diarization, key=lambda item: min(turn_end, item.interval.end) - max(turn_start, item.interval.start)).speaker_cluster
            turn_activity = [item.to_dict() for item in activity if min(turn_end, item.interval.end) > max(turn_start, item.interval.start)]
            disagreement = []
            if new_text is not None:
                disagreement = [detect_disagreement(old_text, new_text).value]
            elif hypothesis is not None and multiple_source_turns:
                disagreement = ["BOUNDARY_CHANGE"]
            turns.append({
                "schema_version": AUDIO_EVIDENCE_SCHEMA_VERSION,
                "pipeline_version": PIPELINE_VERSION,
                "recording_id": window["recording_id"],
                "audio_turn_id": turn_id,
                "window_id": window["window_id"],
                "start": turn_start,
                "end": turn_end,
                "timebase": Timebase.RECORDING_SECONDS.value,
                "role": old.get("role"),
                "speaker_cluster": speaker_cluster,
                "target_activity_evidence": turn_activity,
                "identity_evidence": old.get("identity_evidence"),
                "identity": old.get("identity"),
                "old_transcript": old_text,
                "new_asr_hypothesis": new_text,
                "optional_tse_asr_hypothesis": new_text if tse_source and hypothesis else None,
                "alignment": [item.to_dict() for item in turn_alignment] if turn_alignment else None,
                "overlap_state": "OVERLAP" if any(item.overlap for item in overlapping_diarization) else "NO_OVERLAP_EVIDENCE",
                "source_audio": window["source_audio"],
                "source_interval": dict(interval),
                "asr_source": {"audio_checksum": asr_audio.checksum, "source_waveform": asr_audio.uri, "model": hypothesis.model.to_dict()} if hypothesis else None,
                "tse_source": tse_source,
                "model_provenance": {
                    "target_activity": self.activity.provenance.to_dict() if self.activity is not None else None,
                    "diarization": self.diarization.provenance.to_dict() if self.diarization is not None else None,
                    "asr": self.asr.provenance.to_dict() if self.asr is not None else None,
                    "alignment": self.aligner.provenance.to_dict() if self.aligner is not None else None,
                },
                "enrollment_provenance": {"bank_id": self.bank.bank_id, "bank_revision": self.bank.revision, "enrollment_id": enrollment.enrollment_id, "source_checksum": enrollment.checksum},
                "disagreement_flags": disagreement,
                "old_transcript_overwritten": False,
            })
        return {
            "window_id": window["window_id"], "route": route.to_dict(), "stage_status": stage_status,
            "cache_hits": cache_hits, "window_asr_hypothesis": hypothesis.to_dict() if hypothesis else None,
            "turns": turns,
        }

    @staticmethod
    def _audio_to_dict(audio: AudioInput) -> Dict[str, Any]:
        return {"uri": audio.uri, "checksum": audio.checksum, "recording_id": audio.recording_id}
