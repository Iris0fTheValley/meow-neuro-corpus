from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import threading
import time
from pathlib import Path

import imageio_ffmpeg
import torch
from faster_whisper import WhisperModel

from manifest_tools import ROOT, log_event, read_jsonl, safe_name, write_json, write_master


PIPELINE_VERSION = "faster-whisper-chunked-v2"


def update_performance_profile(result: dict, model: str, compute_type: str) -> None:
    path = ROOT / "checkpoints" / "gpu_performance_profile.json"
    payload = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    profile = payload.setdefault("scheduler_profile", {})
    history = payload.setdefault("asr_history", [])
    history.append({"source_id": result.get("source_id"), "audio_duration_seconds": result.get("audio_duration_seconds"), "real_time_factor": result.get("real_time_factor"), "model": model, "compute_type": compute_type, "observed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    del history[:-20]
    values = [float(x["real_time_factor"]) for x in history if x.get("real_time_factor") is not None]
    if values:
        profile.update({"asr_rtf": statistics.median(values), "asr_backend": "faster-whisper", "asr_model": model, "asr_compute_type": compute_type, "profile_status": "rolling_observed"})
    payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(path, payload)


def locate_audio(row: dict) -> Path | None:
    if row.get("audio_path"):
        path = ROOT / str(row["audio_path"])
        if path.exists():
            return path
    candidates = sorted((ROOT / "raw_audio").glob(f"{safe_name(row['source_id'])}.*"))
    return candidates[0] if candidates else None


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def gpu_snapshot() -> dict:
    try:
        raw = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], text=True, stderr=subprocess.DEVNULL, timeout=5).strip().splitlines()[0]
        util, memory = [int(x.strip()) for x in raw.split(",")[:2]]
        return {"utilization_gpu_percent": util, "memory_used_mib": memory}
    except Exception:
        return {}


class GpuSampler:
    def __init__(self) -> None:
        self.samples: list[dict] = []
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        def loop() -> None:
            while not self.stop.is_set():
                sample = gpu_snapshot()
                if sample:
                    self.samples.append(sample)
                self.stop.wait(0.5)
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def finish(self) -> dict:
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=3)
        if not self.samples:
            return {"sample_count": 0}
        return {
            "sample_count": len(self.samples),
            "gpu_utilization_peak_percent": max(x["utilization_gpu_percent"] for x in self.samples),
            "gpu_utilization_mean_percent": round(sum(x["utilization_gpu_percent"] for x in self.samples) / len(self.samples), 2),
            "vram_peak_mib": max(x["memory_used_mib"] for x in self.samples),
        }


def dedupe_segments(segments: list[dict]) -> list[dict]:
    ordered = sorted(segments, key=lambda s: (float(s.get("start", 0)), float(s.get("end", 0))))
    result: list[dict] = []
    for current in ordered:
        if not current.get("text", "").strip():
            continue
        if result:
            previous = result[-1]
            overlap = min(float(previous.get("end", 0)), float(current.get("end", 0))) - max(float(previous.get("start", 0)), float(current.get("start", 0)))
            same_text = normalize_text(previous.get("text")) == normalize_text(current.get("text"))
            near_start = abs(float(previous.get("start", 0)) - float(current.get("start", 0))) <= 1.5
            if overlap >= -0.25 and near_start and (same_text or normalize_text(previous.get("text")) in normalize_text(current.get("text"))):
                if len(str(current.get("text", ""))) > len(str(previous.get("text", ""))):
                    result[-1] = current
                continue
        result.append(current)
    return result


def decode_chunk(audio: Path, output: Path, start: float, duration: float) -> float:
    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg, "-y", "-ss", f"{start:.3f}", "-i", str(audio), "-t", f"{duration:.3f}",
        "-vn", "-sn", "-dn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(output),
    ]
    t0 = time.perf_counter()
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return time.perf_counter() - t0


def load_partial(partial_path: Path, checkpoint_path: Path) -> tuple[list[dict], float, list[dict]]:
    segments: list[dict] = []
    if partial_path.exists():
        for line in partial_path.read_text(encoding="utf-8").splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            segments.extend(payload.get("segments", []))
    checkpoint = {}
    if checkpoint_path.exists():
        try:
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            checkpoint = {}
    return dedupe_segments(segments), float(checkpoint.get("processed_until", 0.0)), checkpoint.get("chunks", [])


def append_jsonl(path: Path, payload: dict) -> float:
    t0 = time.perf_counter()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return time.perf_counter() - t0


def transcribe_source(model: WhisperModel, row: dict, args: argparse.Namespace, device: str, compute_type: str, model_load_seconds: float = 0.0) -> dict:
    source_id = str(row["source_id"])
    audio = locate_audio(row)
    if not audio:
        raise FileNotFoundError(f"audio asset not present: {source_id}")
    source_duration = float(row.get("duration") or 0.0)
    if source_duration <= 0:
        raise ValueError(f"missing duration: {source_id}")

    stem = f"{safe_name(source_id)}.faster-{safe_name(args.model)}-v2"
    partial_path = ROOT / "asr" / "partial" / f"{stem}.jsonl"
    checkpoint_path = ROOT / "asr" / "partial" / f"{stem}.checkpoint.json"
    output_path = ROOT / "asr" / f"{stem}.json"
    if args.force:
        partial_path.unlink(missing_ok=True)
        checkpoint_path.unlink(missing_ok=True)
    all_segments, processed_until, completed_chunks = load_partial(partial_path, checkpoint_path) if args.resume else ([], 0.0, [])
    chunk_seconds = max(300.0, float(args.chunk_seconds))
    overlap = max(0.5, min(10.0, float(args.overlap_seconds)))
    decode_seconds = 0.0
    transcription_seconds = 0.0
    serialization_seconds = 0.0
    chunk_count = 0
    language = "en"
    language_probability = None
    t_total = time.perf_counter()
    gpu_start = gpu_snapshot()
    gpu_sampler = GpuSampler(); gpu_sampler.start()
    try:
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass
    tmp_wav = ROOT / "tmp" / "asr_chunks" / f"{safe_name(source_id)}.wav"

    while processed_until < source_duration - 0.25:
        chunk_start = max(0.0, processed_until - overlap) if processed_until > 0 else 0.0
        chunk_end = min(source_duration, processed_until + chunk_seconds)
        chunk_duration = max(0.1, chunk_end - chunk_start)
        decode_seconds += decode_chunk(audio, tmp_wav, chunk_start, chunk_duration)
        t_transcribe = time.perf_counter()
        segments, info = model.transcribe(
            str(tmp_wav),
            language="en",
            vad_filter=not args.disable_vad,
            vad_parameters={
                "min_silence_duration_ms": int(args.vad_min_silence_duration_ms),
                "speech_pad_ms": int(args.vad_speech_pad_ms),
            },
            word_timestamps=True,
            condition_on_previous_text=True,
            beam_size=int(args.beam_size),
        )
        chunk_rows: list[dict] = []
        for segment in segments:
            words = []
            for word in (segment.words or []):
                words.append({
                    "start": float(word.start + chunk_start),
                    "end": float(word.end + chunk_start),
                    "word": word.word,
                    "probability": float(word.probability) if word.probability is not None else None,
                })
            chunk_rows.append({
                "id": int(segment.id) if segment.id is not None else None,
                "start": float(segment.start + chunk_start),
                "end": float(segment.end + chunk_start),
                "text": segment.text,
                "avg_logprob": float(segment.avg_logprob) if segment.avg_logprob is not None else None,
                "no_speech_prob": float(segment.no_speech_prob) if segment.no_speech_prob is not None else None,
                "compression_ratio": float(segment.compression_ratio) if segment.compression_ratio is not None else None,
                "words": words,
            })
        transcription_seconds += time.perf_counter() - t_transcribe
        language = getattr(info, "language", language) or language
        language_probability = getattr(info, "language_probability", language_probability)
        if processed_until > 0:
            chunk_rows = [s for s in chunk_rows if float(s["end"]) > processed_until + 0.15]
        all_segments = dedupe_segments(all_segments + chunk_rows)
        serialization_seconds += append_jsonl(partial_path, {
            "source_id": source_id,
            "chunk_start": chunk_start,
            "chunk_end": chunk_end,
            "processed_until": chunk_end,
            "segments": chunk_rows,
        })
        completed_chunks.append({"start": chunk_start, "end": chunk_end, "segments": len(chunk_rows)})
        processed_until = chunk_end
        chunk_count += 1
        write_json(checkpoint_path, {
            "schema_version": "0.1.0",
            "pipeline_version": PIPELINE_VERSION,
            "source_id": source_id,
            "processed_until": processed_until,
            "source_duration": source_duration,
            "chunks": completed_chunks,
            "status": "partial" if processed_until < source_duration - 0.25 else "complete",
        })
        tmp_wav.unlink(missing_ok=True)

    wall_time = time.perf_counter() - t_total
    gpu_end = gpu_snapshot()
    gpu_telemetry = gpu_sampler.finish()
    try:
        vram_peak = int(torch.cuda.max_memory_reserved() / (1024 * 1024)) if device == "cuda" else None
    except Exception:
        vram_peak = None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_payload = {
        "schema_version": "0.2.0",
        "pipeline_version": PIPELINE_VERSION,
        "source_id": source_id,
        "source_url": row.get("source_url"),
        "model": args.model,
        "backend": "faster-whisper",
        "device": device,
        "compute_type": compute_type,
        "language": language,
        "language_probability": language_probability,
        "audio_duration_seconds": source_duration,
        "segments": all_segments,
        "performance": {
            "audio_duration_seconds": source_duration,
            "wall_time_seconds": wall_time,
            "model_load_seconds": model_load_seconds,
            "decode_time_seconds": decode_seconds,
            "transcription_time_seconds": transcription_seconds,
            "serialization_time_seconds": serialization_seconds,
            "chunk_count": chunk_count,
            "chunk_seconds": chunk_seconds,
            "overlap_seconds": overlap,
            "real_time_factor": wall_time / source_duration,
            "audio_seconds_per_wall_second": source_duration / max(1e-9, wall_time),
            "gpu_snapshot_start": gpu_start,
            "gpu_snapshot_end": gpu_end,
            "gpu_telemetry": gpu_telemetry,
            "vram_peak_mib_torch_reserved": vram_peak,
        },
    }
    serialization_start = time.perf_counter()
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_payload["performance"]["serialization_time_seconds"] += time.perf_counter() - serialization_start
    row.update({
        "asr_status": "done",
        "asr_backend": "faster-whisper",
        "asr_model": args.model,
        "asr_compute_type": compute_type,
        "asr_pipeline_version": PIPELINE_VERSION,
        "asr_path": str(output_path.relative_to(ROOT)),
        "asr_segment_count": len(all_segments),
        "asr_device": device,
        "asr_performance": output_payload["performance"],
    })
    return {"source_id": source_id, "output": str(output_path.relative_to(ROOT)), **output_payload["performance"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium.en")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--source-id", action="append")
    parser.add_argument("--chunk-seconds", type=float, default=1200.0)
    parser.add_argument("--overlap-seconds", type=float, default=2.0)
    parser.add_argument("--vad-min-silence-duration-ms", type=int, default=500)
    parser.add_argument("--vad-speech-pad-ms", type=int, default=250)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--disable-vad", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    rows = read_jsonl(ROOT / "manifest" / "master_video_manifest.jsonl")
    selected = []
    for row in rows:
        if args.source_id and row.get("source_id") not in args.source_id:
            continue
        if not args.source_id and not args.all and len(selected) >= args.limit:
            break
        if not args.force and row.get("asr_status") == "done":
            continue
        selected.append(row)
    if not selected:
        print(json.dumps({"model": args.model, "processed": 0, "total": len(rows)}, ensure_ascii=False))
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = args.compute_type if device == "cuda" else "int8"
    model_load_start = time.perf_counter()
    try:
        model = WhisperModel(args.model, device=device, compute_type=compute_type)
    except Exception as exc:
        if device != "cuda":
            raise
        log_event("asr_faster_cuda_unavailable", model=args.model, error=repr(exc))
        device, compute_type = "cpu", "int8"
        model = WhisperModel(args.model, device=device, compute_type=compute_type)
    model_load_seconds = time.perf_counter() - model_load_start
    results = []
    for row in selected:
        try:
            result = transcribe_source(model, row, args, device, compute_type, model_load_seconds)
            result["model_load_seconds"] = model_load_seconds
            row.setdefault("asr_performance", {})["model_load_seconds"] = model_load_seconds
            update_performance_profile(result, args.model, compute_type)
            results.append(result)
            log_event("asr_faster_complete", source_id=row["source_id"], model=args.model, compute_type=compute_type, performance=result)
        except Exception as exc:
            row["asr_status"] = "retryable"
            row["asr_error"] = repr(exc)
            row["asr_fast_attempts"] = int(row.get("asr_fast_attempts") or 0) + 1
            log_event("asr_faster_failed", source_id=row.get("source_id"), model=args.model, error=repr(exc))
    write_master(rows)
    print(json.dumps({"model": args.model, "device": device, "compute_type": compute_type, "model_load_seconds": model_load_seconds, "processed": len(results), "failed": len(selected) - len(results), "results": results, "total": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
