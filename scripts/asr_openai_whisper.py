from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import torch
import whisper

from manifest_tools import ROOT, log_event, read_jsonl, safe_name, write_master


def locate_audio(row: dict) -> Path:
    if row.get("audio_path") and (ROOT / row["audio_path"]).exists():
        return ROOT / row["audio_path"]
    candidates = sorted((ROOT / "raw_audio").glob(f"{safe_name(row['source_id'])}.*"))
    if not candidates:
        raise FileNotFoundError(row["source_id"])
    return candidates[0]


def decode_audio(path: Path, start: float, duration: float | None) -> np.ndarray:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg, "-ss", str(start), "-i", str(path)]
    if duration:
        cmd += ["-t", str(duration)]
    cmd += ["-f", "s16le", "-ac", "1", "-ar", "16000", "pipe:1"]
    raw = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="small.en")
    parser.add_argument("--source-id", default="1CxTCk0I79w")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=60.0)
    args = parser.parse_args()
    rows = read_jsonl(ROOT / "manifest" / "master_video_manifest.jsonl")
    row = next(r for r in rows if r.get("source_id") == args.source_id)
    model_dir = ROOT / "models" / "openai_whisper"
    model_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    model = whisper.load_model(args.model, download_root=str(model_dir), device="cuda" if torch.cuda.is_available() else "cpu")
    audio = decode_audio(locate_audio(row), args.start, args.duration)
    result = model.transcribe(audio, language="en", fp16=torch.cuda.is_available(), temperature=0.0, condition_on_previous_text=True, word_timestamps=True)
    elapsed = time.perf_counter() - t0
    out = ROOT / "asr" / f"{safe_name(args.source_id)}.openai-{args.model}.json"
    payload = {"source_id": args.source_id, "source_url": row["source_url"], "model": args.model, "backend": "openai-whisper", "device": "cuda" if torch.cuda.is_available() else "cpu", "elapsed_seconds_including_model_load": elapsed, "segments": result.get("segments", [])}
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    row.setdefault("asr_benchmarks", {})[f"openai-{args.model}"] = {"path": str(out.relative_to(ROOT)), "elapsed_seconds_including_model_load": elapsed, "device": payload["device"], "segment_count": len(payload["segments"])}
    row.update({"asr_status": "done", "asr_model": f"openai-{args.model}", "asr_path": str(out.relative_to(ROOT)), "asr_segment_count": len(payload["segments"]), "asr_device": payload["device"]})
    write_master(rows)
    log_event("asr_openai_benchmark_complete", source_id=args.source_id, model=args.model, elapsed=elapsed, segments=len(payload["segments"]))
    print(json.dumps({"model": args.model, "elapsed_seconds_including_model_load": elapsed, "segments": len(payload["segments"]), "device": payload["device"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
