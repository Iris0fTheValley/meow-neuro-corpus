from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import soundfile as sf
import torch
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

from manifest_tools import ROOT, read_jsonl, safe_name, write_json
from diarize_production_ecapa import expand_speaker_units


DEFAULT_SOURCES = ["S8piSUUq43U", "MnsrybitPPo", "BV1K2ygYtEGY"]


def decode_window(audio: Path, wav: Path, start: float, duration: float) -> float:
    wav.parent.mkdir(parents=True, exist_ok=True)
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-ss", str(start), "-i", str(audio), "-t", str(duration), "-vn", "-sn", "-dn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)]
    t0 = time.perf_counter()
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return time.perf_counter() - t0


def locate(row: dict) -> Path:
    if row.get("audio_path") and (ROOT / str(row["audio_path"])).exists():
        return ROOT / str(row["audio_path"])
    candidates = sorted((ROOT / "raw_audio").glob(f"{safe_name(row['source_id'])}.*"))
    if not candidates:
        raise FileNotFoundError(row["source_id"])
    return candidates[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-id", action="append", dest="source_ids")
    ap.add_argument("--window-seconds", type=float, default=600.0)
    ap.add_argument("--start", type=float, default=600.0)
    args = ap.parse_args()
    wanted = args.source_ids or DEFAULT_SOURCES
    rows = {str(r.get("source_id")): r for r in read_jsonl(ROOT / "manifest" / "master_video_manifest.jsonl")}
    cases = []
    for source_id in wanted:
        row = rows.get(source_id)
        if not row or not row.get("asr_path"):
            raise FileNotFoundError(f"diarization benchmark ASR unavailable: {source_id}")
        asr = json.loads((ROOT / str(row["asr_path"])).read_text(encoding="utf-8"))
        start = min(float(args.start), max(0.0, float(row.get("duration") or 0.0) - args.window_seconds))
        segments = [unit for segment in asr.get("segments", []) for unit in expand_speaker_units(segment) if float(unit.get("end", 0)) > start and float(unit.get("start", 0)) < start + args.window_seconds and unit.get("text", "").strip() and float(unit.get("end", 0)) - float(unit.get("start", 0)) >= 0.6]
        cases.append({"source_id": source_id, "audio": locate(row), "start": start, "duration": args.window_seconds, "segments": segments})
    t_load = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    speechbrain_device = "cuda:0" if device == "cuda" else "cpu"
    encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "models" / "speechbrain_ecapa"), run_opts={"device": speechbrain_device}, local_strategy=LocalStrategy.COPY)
    model_load_seconds = time.perf_counter() - t_load
    output = []
    for case in cases:
        wav = ROOT / "tmp" / "diar_benchmark" / f"{safe_name(case['source_id'])}.wav"
        decode_seconds = decode_window(case["audio"], wav, case["start"], case["duration"])
        t_embed = time.perf_counter()
        embedding_count = 0
        batches = []
        with sf.SoundFile(wav) as source:
            rate = int(source.samplerate)
            pending = []
            for segment in case["segments"]:
                a = max(0, int((float(segment["start"]) - case["start"]) * rate))
                b = min(len(source), int((float(segment["end"]) - case["start"]) * rate))
                source.seek(a)
                clip = np.asarray(source.read(max(0, b - a), dtype="float32", always_2d=False), dtype="float32")
                if len(clip) < 0.6 * rate:
                    continue
                clip = clip[: 8 * 16000]
                padded = np.zeros(8 * 16000, dtype="float32"); padded[:len(clip)] = clip
                pending.append(padded)
                if len(pending) >= 32:
                    batches.append(np.asarray(pending)); embedding_count += len(pending); pending.clear()
            if pending:
                batches.append(np.asarray(pending)); embedding_count += len(pending)
        for batch in batches:
            with torch.inference_mode():
                encoder.encode_batch(torch.from_numpy(batch))
        embedding_seconds = time.perf_counter() - t_embed
        try:
            vram_peak = int(torch.cuda.max_memory_reserved() / (1024 * 1024)) if device == "cuda" else None
        except Exception:
            vram_peak = None
        wall = decode_seconds + embedding_seconds
        output.append({"source_id": case["source_id"], "audio_duration_seconds": case["duration"], "wall_time_seconds": wall, "real_time_factor": wall / case["duration"], "decode_time_seconds": decode_seconds, "embedding_time_seconds": embedding_seconds, "embedding_count": embedding_count, "vram_peak_mib_torch_reserved": vram_peak})
        wav.unlink(missing_ok=True)
    report_path = ROOT / "reports" / "gpu_pipeline_performance.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {"schema_version": "0.2.0"}
    median_rtf = statistics.median(float(x["real_time_factor"]) for x in output) if output else None
    report["diarization_benchmark"] = {"backend": "speechbrain-ecapa-voxceleb", "device": device, "model_load_seconds": model_load_seconds, "window_seconds": args.window_seconds, "runs": output, "median_rtf": median_rtf, "quality_note": "Uses the production random-access audio path and existing timestamped ASR segments; speaker labels remain anonymous."}
    profile = report.setdefault("scheduler_profile", {})
    if median_rtf is not None:
        profile["diarization_rtf"] = median_rtf
    write_json(report_path, report)
    write_json(ROOT / "reports" / "gpu_diarization_benchmark.json", report["diarization_benchmark"])
    summary_path = ROOT / "reports" / "gpu_pipeline_performance.md"
    if summary_path.exists():
        summary = summary_path.read_text(encoding="utf-8").split("\n## Diarization benchmark", 1)[0]
    else:
        summary = "# GPU/ASR pipeline performance\n"
    summary += "\n## Diarization benchmark\n\n"
    summary += f"- Backend: SpeechBrain ECAPA; model load: {model_load_seconds:.2f}s; median fixed-window RTF: {median_rtf:.4f}\n"
    summary += "- Production path uses bounded random-access reads and batch embeddings; labels remain anonymous until validated speaker mapping.\n"
    summary_path.write_text(summary, encoding="utf-8")
    print(json.dumps({"benchmark": "diarization_complete", "sources": len(output), "median_rtf": median_rtf, "report": "reports/gpu_pipeline_performance.json"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
