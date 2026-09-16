from __future__ import annotations

"""Raw-waveform production adapters used by the real audio reconstruction run.

The adapters deliberately keep all coordinates in the bounded recording window
until the v1 pipeline performs canonical timebase normalization.  No transcript
or frozen diarization artifact is read by these adapters.
"""

import hashlib
import subprocess
import tempfile
from pathlib import Path
from typing import Any

try:
    import imageio_ffmpeg
    _FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:  # system Python uses the repository's validated binary
    _FFMPEG = str(Path(r"J:\AI friend\MEOW Qwen3.5\.venv-data\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe"))
import numpy as np
import soundfile as sf
import torch
try:
    import torchaudio
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]
except Exception:
    torchaudio = None
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

from .adapters import (
    AdapterUnavailable,
    AudioInput,
    CallableASRAdapter,
    CallableDiarizationAdapter,
    CallableForcedAlignmentAdapter,
    CallableTargetActivityAdapter,
    Qwen3ASRAdapter,
    Qwen3ForcedAlignerAdapter,
)
from .contracts import ActivitySegment, AlignmentUnit, DiarizationTurn, Interval, ModelProvenance, Timebase, TranscriptHypothesis
from .enrollment import EnrollmentReference


def decode_waveform(audio: AudioInput | EnrollmentReference, source_audio: str | None = None) -> tuple[np.ndarray, int]:
    """Decode an exact source interval to mono 16 kHz float32 waveform."""
    path = Path(source_audio or getattr(audio, "uri", ""))
    interval = audio.interval if isinstance(audio, AudioInput) else audio.source_interval
    if not path.exists():
        raise AdapterUnavailable(f"source audio missing: {path}")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        wav = Path(handle.name)
    try:
        duration = float(interval.end - interval.start)
        command = [_FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{float(interval.start):.6f}", "-i", str(path), "-t", f"{duration:.6f}", "-vn", "-sn", "-dn", "-ac", "1", "-ar", "16000", "-f", "wav", str(wav)]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        samples, rate = sf.read(wav, dtype="float32", always_2d=False)
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if rate != 16000 or len(samples) < max(3200, int(duration * 16000 * 0.25)):
            raise AdapterUnavailable(f"decoded waveform is incomplete ({rate} Hz, {len(samples)} samples)")
        return samples, int(rate)
    except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
        raise AdapterUnavailable(f"waveform decode failed: {path}") from exc
    finally:
        wav.unlink(missing_ok=True)


class ECAPARawAdapters:
    """Enrollment-conditioned ECAPA activity plus frame diarization."""

    def __init__(self, model_dir: str, threshold: float = 0.62, threshold_version: str = "ecapa-target-activity-v1") -> None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(model_dir),
            run_opts={"device": device},
            local_strategy=LocalStrategy.COPY,
        )
        self.device = device
        self.threshold = float(threshold)
        self.threshold_version = threshold_version
        self._enrollment_cache: dict[str, np.ndarray] = {}
        params = {"target_activity_threshold": self.threshold, "target_activity_threshold_version": threshold_version, "frame_seconds": 1.5, "hop_seconds": 0.75, "waveform_sample_rate": 16000}
        self.activity = CallableTargetActivityAdapter(ModelProvenance("target_activity", "speechbrain-ecapa-voxceleb", "ecapa-target-activity-2026-09-17", params), self.detect)
        self.diarization = CallableDiarizationAdapter(ModelProvenance("diarization", "ecapa-frame-cluster", "ecapa-frame-diarization-2026-09-17", params), self.diarize)

    def _embed(self, clips: list[np.ndarray]) -> np.ndarray:
        if not clips:
            return np.empty((0, 192), dtype=np.float32)
        max_len = max(len(x) for x in clips)
        batch = np.zeros((len(clips), max_len), dtype=np.float32)
        for i, clip in enumerate(clips):
            batch[i, : len(clip)] = clip
        with torch.inference_mode():
            value = self.encoder.encode_batch(torch.from_numpy(batch)).squeeze(1).detach().cpu().numpy().astype(np.float32)
        value /= np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-8)
        return value

    def enrollment_embedding(self, reference: EnrollmentReference) -> np.ndarray:
        if reference.enrollment_id in self._enrollment_cache:
            return self._enrollment_cache[reference.enrollment_id]
        waveform, _ = decode_waveform(reference, reference.raw_audio_uri)
        # ECAPA is stable on 1.5–8 s clips; split long references and average.
        size, step = 24000, 12000
        clips = [waveform[i : i + size] for i in range(0, max(1, len(waveform) - 4000), step) if len(waveform[i : i + size]) >= 9600]
        if not clips:
            clips = [waveform]
        emb = self._embed(clips).mean(axis=0)
        emb /= max(float(np.linalg.norm(emb)), 1e-8)
        self._enrollment_cache[reference.enrollment_id] = emb
        return emb

    def _frames(self, waveform: np.ndarray, rate: int = 16000) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
        frame, hop = int(1.5 * rate), int(0.75 * rate)
        clips, spans = [], []
        if len(waveform) <= frame:
            return [np.pad(waveform, (0, frame - len(waveform)))], [(0.0, len(waveform) / rate)]
        for start in range(0, len(waveform) - frame + 1, hop):
            clip = waveform[start : start + frame]
            if float(np.sqrt(np.mean(clip * clip) + 1e-9)) < 0.003:
                continue
            clips.append(clip)
            spans.append((start / rate, min(len(waveform) / rate, (start + frame) / rate)))
        return clips, spans

    def _target_scores(self, audio: AudioInput, enrollment: EnrollmentReference) -> tuple[np.ndarray, list[tuple[float, float]]]:
        waveform, rate = decode_waveform(audio)
        clips, spans = self._frames(waveform, rate)
        if not clips:
            return np.empty(0), []
        ref = self.enrollment_embedding(enrollment)
        scores = self._embed(clips) @ ref
        return scores, spans

    def detect(self, audio: AudioInput, enrollment: EnrollmentReference) -> list[ActivitySegment]:
        scores, spans = self._target_scores(audio, enrollment)
        if not len(scores):
            return []
        active = scores >= self.threshold
        out: list[ActivitySegment] = []
        begin = None
        for i, yes in enumerate(active.tolist() + [False]):
            if yes and begin is None:
                begin = i
            elif not yes and begin is not None:
                s, _ = spans[begin]
                _, e = spans[i - 1]
                prob = float(np.clip((float(scores[begin:i].mean()) + 1.0) / 2.0, 0.0, 1.0))
                out.append(ActivitySegment(Interval(audio.interval.start + s, min(audio.interval.end, audio.interval.start + e), Timebase.RECORDING_SECONDS), prob, self.activity.provenance, enrollment.enrollment_id))
                begin = None
        return out

    def diarize(self, audio: AudioInput) -> list[DiarizationTurn]:
        waveform, rate = decode_waveform(audio)
        clips, spans = self._frames(waveform, rate)
        if not clips:
            return []
        emb = self._embed(clips)
        # Deterministic two-cluster frame clustering without using old labels.
        if len(emb) == 1:
            labels = np.zeros(1, dtype=int)
            centroids = emb
        else:
            c0, c1 = emb[0].copy(), emb[-1].copy()
            for _ in range(8):
                sims = np.stack([emb @ c0, emb @ c1], axis=1)
                labels = sims.argmax(axis=1)
                if np.any(labels == 0): c0 = emb[labels == 0].mean(axis=0); c0 /= max(float(np.linalg.norm(c0)), 1e-8)
                if np.any(labels == 1): c1 = emb[labels == 1].mean(axis=0); c1 /= max(float(np.linalg.norm(c1)), 1e-8)
            centroids = np.stack([c0, c1])
        out = []
        for i, ((s, e), label) in enumerate(zip(spans, labels.tolist())):
            sims = emb[i] @ centroids.T
            order = np.argsort(sims)[::-1]
            overlap = len(order) > 1 and float(sims[order[0]]) >= 0.48 and float(sims[order[1]]) >= 0.43 and float(sims[order[0]] - sims[order[1]]) < 0.08
            out.append(DiarizationTurn(Interval(audio.interval.start + s, min(audio.interval.end, audio.interval.start + e), Timebase.RECORDING_SECONDS), f"ECAPA_CLUSTER_{int(label):02d}", bool(overlap), self.diarization.provenance, exclusive=not overlap))
        return out


class QwenRawAdapters:
    def __init__(self, asr_dir: str, aligner_dir: str) -> None:
        from qwen_asr import Qwen3ASRModel
        self.asr_model = Qwen3ASRModel.from_pretrained(asr_dir, forced_aligner=aligner_dir, forced_aligner_kwargs={"device_map": "cuda:0" if torch.cuda.is_available() else "cpu"}, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, device_map="cuda:0" if torch.cuda.is_available() else "cpu", max_new_tokens=512)
        self.asr = Qwen3ASRAdapter("Qwen/Qwen3-ASR-1.7B@7278e1e70fe206f11671096ffdd38061171dd6e5", self.transcribe, {"return_time_stamps": True, "raw_waveform": True, "sample_rate": 16000})
        self.aligner = Qwen3ForcedAlignerAdapter("Qwen/Qwen3-ForcedAligner-0.6B@c7cbfc2048c462b0d63a45797104fc9db3ad62b7", self.align, {"raw_waveform": True, "sample_rate": 16000})
        self._result_cache: dict[str, Any] = {}

    def _run(self, audio: AudioInput):
        waveform, rate = decode_waveform(audio)
        with torch.inference_mode():
            result = self.asr_model.transcribe((waveform, rate), language="English", return_time_stamps=True)
        if not result:
            raise AdapterUnavailable("Qwen ASR returned no result")
        return result[0], waveform, rate

    def transcribe(self, audio: AudioInput) -> TranscriptHypothesis:
        result, _, _ = self._run(audio)
        self._result_cache[self._key(audio)] = result
        text = str(result.text or "").strip()
        if not text:
            raise AdapterUnavailable("Qwen ASR returned empty text")
        return TranscriptHypothesis(text, self.asr.provenance, audio.uri)

    def align(self, audio: AudioInput, text: str) -> list[AlignmentUnit]:
        result = self._result_cache.get(self._key(audio))
        if result is None:
            result, _, _ = self._run(audio)
        stamps = getattr(result, "time_stamps", None)
        if stamps is None:
            return []
        items = getattr(stamps, "items", stamps)
        out = []
        for item in items or []:
            token = str(getattr(item, "text", "") or "").strip()
            start = float(getattr(item, "start_time", 0.0))
            end = float(getattr(item, "end_time", 0.0))
            # Qwen aligner timestamps are seconds in current qwen-asr releases.
            if token and end > start:
                out.append(AlignmentUnit(token, Interval(audio.interval.start + start, min(audio.interval.end, audio.interval.start + end), Timebase.RECORDING_SECONDS), None))
        return out

    @staticmethod
    def _key(audio: AudioInput) -> str:
        return f"{audio.uri}|{audio.recording_id}|{audio.interval.start:.6f}|{audio.interval.end:.6f}"
