from __future__ import annotations

import argparse
import json
import re
import subprocess
import threading
import time
from difflib import SequenceMatcher
from pathlib import Path

import imageio_ffmpeg
import soundfile as sf
import torch

from manifest_tools import ROOT, read_jsonl, safe_name, write_json


DEFAULT_SOURCES = ["S8piSUUq43U", "MnsrybitPPo", "BV1K2ygYtEGY"]
DEFAULT_LABELS = {"S8piSUUq43U": "clean_speech", "MnsrybitPPo": "multi_speaker", "BV1K2ygYtEGY": "noisy_game_audio"}
PROPER_NAMES = ("neuro", "evil", "vedal", "onigiri", "nihmune", "minecraft", "hollow knight")


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", str(text or "").lower())


def normalize(text: str) -> str:
    return " ".join(tokens(text))


def gpu_snapshot() -> dict:
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        ).strip().splitlines()[0]
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
                    sample["time"] = time.time()
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
            "samples": self.samples,
        }


def decode_wav(audio: Path, output: Path, start: float, duration: float) -> float:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-ss", str(start), "-i", str(audio), "-t", str(duration), "-vn", "-sn", "-dn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(output)]
    t0 = time.perf_counter()
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return time.perf_counter() - t0


def locate_audio(row: dict) -> Path:
    if row.get("audio_path") and (ROOT / str(row["audio_path"])).exists():
        return ROOT / str(row["audio_path"])
    candidates = sorted((ROOT / "raw_audio").glob(f"{safe_name(row['source_id'])}.*"))
    if not candidates:
        raise FileNotFoundError(row["source_id"])
    return candidates[0]


def openai_run(model_name: str, cases: list[dict], out_dir: Path) -> tuple[list[dict], dict]:
    import whisper

    model_load_start = time.perf_counter()
    model = whisper.load_model(model_name, download_root=str(ROOT / "models" / "openai_whisper"), device="cuda" if torch.cuda.is_available() else "cpu")
    model_load_seconds = time.perf_counter() - model_load_start
    sampler = GpuSampler(); sampler.start()
    results = []
    for case in cases:
        wav = out_dir / f"{safe_name(case['source_id'])}.wav"
        decode_seconds = decode_wav(case["audio"], wav, case["start"], case["duration"])
        audio, _ = sf.read(wav, dtype="float32", always_2d=False)
        t0 = time.perf_counter()
        result = model.transcribe(audio, language="en", fp16=torch.cuda.is_available(), temperature=0.0, condition_on_previous_text=True, word_timestamps=True)
        transcribe_seconds = time.perf_counter() - t0
        segments = []
        for segment in result.get("segments", []):
            segments.append({"start": float(segment["start"] + case["start"]), "end": float(segment["end"] + case["start"]), "text": segment.get("text", ""), "words": [{"start": float(w["start"] + case["start"]), "end": float(w["end"] + case["start"]), "word": w.get("word"), "probability": w.get("probability")} for w in segment.get("words", [])]})
        results.append({"source_id": case["source_id"], "start": case["start"], "duration": case["duration"], "segments": segments, "decode_time_seconds": decode_seconds, "transcription_time_seconds": transcribe_seconds})
        wav.unlink(missing_ok=True)
    telemetry = sampler.finish()
    return results, {"backend": "openai-whisper", "model": model_name, "compute_type": "float16" if torch.cuda.is_available() else "cpu", "model_load_seconds": model_load_seconds, "telemetry": telemetry}


def faster_run(model_name: str, compute_type: str, cases: list[dict], out_dir: Path) -> tuple[list[dict], dict]:
    from faster_whisper import WhisperModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    actual_compute_type = compute_type if device == "cuda" else "int8"
    model_load_start = time.perf_counter()
    model = WhisperModel(model_name, device=device, compute_type=actual_compute_type)
    model_load_seconds = time.perf_counter() - model_load_start
    sampler = GpuSampler(); sampler.start()
    results = []
    for case in cases:
        wav = out_dir / f"{safe_name(case['source_id'])}.wav"
        decode_seconds = decode_wav(case["audio"], wav, case["start"], case["duration"])
        t0 = time.perf_counter()
        segments, info = model.transcribe(str(wav), language="en", vad_filter=True, vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 250}, word_timestamps=True, condition_on_previous_text=True, beam_size=5)
        segment_rows = []
        for segment in segments:
            segment_rows.append({"start": float(segment.start + case["start"]), "end": float(segment.end + case["start"]), "text": segment.text, "words": [{"start": float(w.start + case["start"]), "end": float(w.end + case["start"]), "word": w.word, "probability": w.probability} for w in (segment.words or [])]})
        transcribe_seconds = time.perf_counter() - t0
        results.append({"source_id": case["source_id"], "start": case["start"], "duration": case["duration"], "segments": segment_rows, "decode_time_seconds": decode_seconds, "transcription_time_seconds": transcribe_seconds, "language": getattr(info, "language", None)})
        wav.unlink(missing_ok=True)
    telemetry = sampler.finish()
    return results, {"backend": "faster-whisper", "model": model_name, "compute_type": actual_compute_type, "model_load_seconds": model_load_seconds, "telemetry": telemetry}


def quality_metrics(baseline: dict, candidate: dict) -> dict:
    base_segments = baseline.get("segments", [])
    cand_segments = candidate.get("segments", [])
    base_text = normalize(" ".join(s.get("text", "") for s in base_segments))
    cand_text = normalize(" ".join(s.get("text", "") for s in cand_segments))
    base_tokens, cand_tokens = base_text.split(), cand_text.split()
    matcher = SequenceMatcher(None, base_tokens, cand_tokens)
    opcodes = matcher.get_opcodes()
    missing = sum((i2 - i1) for tag, i1, i2, _, _ in opcodes if tag in {"delete", "replace"})
    extra = sum((j2 - j1) for tag, _, _, j1, j2 in opcodes if tag in {"insert", "replace"})
    base_buckets = {int(float(s.get("start", 0)) // 30) for s in base_segments if s.get("text", "").strip()}
    cand_buckets = {int(float(s.get("start", 0)) // 30) for s in cand_segments if s.get("text", "").strip()}
    words = [w for s in cand_segments for w in s.get("words", []) if w.get("start") is not None and w.get("end") is not None]
    short_base = sum(1 for s in base_segments if 0 < len(tokens(s.get("text", ""))) <= 3)
    short_cand = sum(1 for s in cand_segments if 0 < len(tokens(s.get("text", ""))) <= 3)
    base_starts = [float(a.get("start", 0)) for a in base_segments]
    drift = [min((abs(float(b.get("start", 0)) - start) for start in base_starts), default=0.0) for b in cand_segments]
    return {
        "normalised_edit_similarity": round(matcher.ratio(), 4),
        "missing_token_estimate": missing,
        "extra_token_estimate": extra,
        "baseline_segment_count": len(base_segments),
        "candidate_segment_count": len(cand_segments),
        "candidate_word_timestamp_count": len(words),
        "candidate_word_timestamp_fraction": round(len(words) / max(1, len(cand_tokens)), 4),
        "timestamp_start_drift_nearest_mean_seconds": round(sum(drift) / max(1, len(drift)), 3),
        "timeline_bucket_recall": round(len(base_buckets & cand_buckets) / max(1, len(base_buckets)), 4),
        "short_reply_count_baseline": short_base,
        "short_reply_count_candidate": short_cand,
        "proper_name_counts": {name: cand_text.count(name) for name in PROPER_NAMES},
        "quality_pass_proxy": bool(matcher.ratio() >= 0.85 and len(words) / max(1, len(cand_tokens)) >= 0.8 and len(base_buckets & cand_buckets) / max(1, len(base_buckets)) >= 0.9),
    }


def representative_segments(results: list[dict], limit: int = 20) -> list[dict]:
    flat = [dict(source_id=r["source_id"], **s) for r in results for s in r.get("segments", []) if s.get("text", "").strip()]
    if len(flat) <= limit:
        return flat
    indexes = [round(i * (len(flat) - 1) / (limit - 1)) for i in range(limit)]
    return [flat[i] for i in indexes]


def write_summary(report: dict) -> None:
    lines = ["# GPU/ASR pipeline performance", "", f"Created: {report.get('created_at')}", f"Sources: {len(report.get('sources', []))}; fixed window: {report.get('window_seconds')} seconds", "", f"Recommended candidate (proxy): `{report.get('recommended_candidate')}`", f"Status: {report.get('recommendation_status')}", "", "| Backend | Model load (s) | Window wall time (s) | RTF | Quality proxy |", "|---|---:|---:|---:|---|"]
    for run in report.get("runs", []):
        if "results" not in run:
            lines.append(f"| {run.get('id')} | — | — | — | error: {run.get('error')} |")
            continue
        audio = sum(float(x.get("duration", 0)) for x in run["results"])
        wall = sum(float(x.get("decode_time_seconds", 0)) + float(x.get("transcription_time_seconds", 0)) for x in run["results"])
        q = list(run.get("quality_comparison", {}).values())
        quality = "PASS" if q and all(x.get("quality_pass_proxy") for x in q) else ("FAIL/REVIEW" if q else "baseline")
        lines.append(f"| {run.get('id')} | {float(run.get('meta', {}).get('model_load_seconds', 0)):.2f} | {wall:.2f} | {wall / max(1, audio):.4f} | {quality} |")
    lines.extend(["", "## Notes", "", "- RTF is wall time divided by audio duration; model load is reported separately.", "- Each backend retains at least 20 evenly sampled representative segments when available.", "- Quality comparison is a screening proxy against the fixed-window OpenAI baseline and requires human/audio review before promotion.", "- Old artifacts are not overwritten; production fast artifacts use a versioned filename."])
    summary = report.get("summary") or {}
    lines.extend(["", "## Throughput estimate", "", f"- ASR speedup vs baseline: `{summary.get('asr_speedup_vs_baseline', {})}`", f"- Estimated backlog GPU hours: `{summary.get('backlog_estimated_gpu_hours_total')}` (benchmark RTF extrapolation; not a wall-clock guarantee)", f"- Estimated audio hours/day at sustained GPU saturation: `{summary.get('estimated_audio_hours_per_gpu_day')}`"])
    diar = report.get("diarization_benchmark")
    if diar:
        lines.extend(["", "## Diarization benchmark", "", f"- Backend: SpeechBrain ECAPA; model load: {float(diar.get('model_load_seconds', 0)):.2f}s; median fixed-window RTF: {float(diar.get('median_rtf', 0)):.4f}", "- Production path uses bounded random-access reads and batch embeddings; labels remain anonymous until validated speaker mapping."])
    write_json(ROOT / "reports" / "gpu_pipeline_performance_summary.json", {"markdown": "reports/gpu_pipeline_performance.md", "recommended_candidate": report.get("recommended_candidate")})
    (ROOT / "reports" / "gpu_pipeline_performance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-id", action="append", dest="source_ids")
    ap.add_argument("--window-seconds", type=float, default=600.0)
    ap.add_argument("--start", type=float, default=600.0)
    ap.add_argument("--no-distil", action="store_true")
    args = ap.parse_args()
    wanted = args.source_ids or DEFAULT_SOURCES
    rows = {str(r.get("source_id")): r for r in read_jsonl(ROOT / "manifest" / "master_video_manifest.jsonl")}
    cases = []
    for source_id in wanted:
        row = rows.get(source_id)
        if not row or not row.get("audio_path"):
            raise FileNotFoundError(f"benchmark source unavailable: {source_id}")
        start = min(float(args.start), max(0.0, float(row.get("duration") or 0.0) - args.window_seconds))
        cases.append({"source_id": source_id, "audio": locate_audio(row), "start": start, "duration": args.window_seconds, "label": DEFAULT_LABELS.get(source_id, "custom")})
    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "reports" / "gpu_benchmark_artifacts" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_results, baseline_meta = openai_run("medium.en", cases, out_dir)
    runs = [{"id": "openai-medium.en-float16", "meta": baseline_meta, "results": baseline_results, "representative_segments": representative_segments(baseline_results)}]
    candidates = [("medium.en", "float16", "faster-medium.en-float16") , ("medium.en", "int8_float16", "faster-medium.en-int8_float16")]
    if not args.no_distil:
        candidates.append(("distil-large-v3", "float16", "faster-distil-large-v3-float16"))
    for model_name, compute_type, run_id_name in candidates:
        try:
            result_rows, meta = faster_run(model_name, compute_type, cases, out_dir)
            runs.append({"id": run_id_name, "meta": meta, "results": result_rows, "representative_segments": representative_segments(result_rows)})
        except Exception as exc:
            runs.append({"id": run_id_name, "error": repr(exc), "meta": {"backend": "faster-whisper", "model": model_name, "compute_type": compute_type}})
    base_by_source = {r["source_id"]: r for r in baseline_results}
    for run in runs[1:]:
        if "results" in run:
            run["quality_comparison"] = {r["source_id"]: quality_metrics(base_by_source[r["source_id"]], r) for r in run["results"]}
    eligible = []
    for run in runs[1:]:
        if "results" not in run:
            continue
        q = list(run.get("quality_comparison", {}).values())
        wall = sum(float(x.get("decode_time_seconds", 0)) + float(x.get("transcription_time_seconds", 0)) for x in run["results"])
        audio = sum(float(x.get("duration", 0)) for x in run["results"])
        if q and all(x["quality_pass_proxy"] for x in q):
            eligible.append((wall / max(1.0, audio), run))
    recommendation = min(eligible, key=lambda x: x[0])[1]["id"] if eligible else "openai-medium.en-float16"
    selected_run = next((r for r in runs if r.get("id") == recommendation), runs[0])
    selected_results = selected_run.get("results", [])
    selected_audio = sum(float(x.get("duration", 0)) for x in selected_results)
    selected_wall = sum(float(x.get("decode_time_seconds", 0)) + float(x.get("transcription_time_seconds", 0)) for x in selected_results)
    baseline_audio = sum(float(x.get("duration", 0)) for x in baseline_results)
    baseline_wall = sum(float(x.get("decode_time_seconds", 0)) + float(x.get("transcription_time_seconds", 0)) for x in baseline_results)
    speedups = {}
    for run in runs:
        if "results" in run:
            audio = sum(float(x.get("duration", 0)) for x in run["results"])
            wall = sum(float(x.get("decode_time_seconds", 0)) + float(x.get("transcription_time_seconds", 0)) for x in run["results"])
            speedups[run["id"]] = baseline_wall / max(1e-9, wall) if run["id"] != "openai-medium.en-float16" else 1.0
    backlog_asr_seconds = sum(float(r.get("duration") or 0) for r in rows.values() if r.get("audio_download_status") == "done" and r.get("asr_status") not in {"done", "source_transcript"})
    backlog_diar_seconds = sum(float(r.get("duration") or 0) for r in rows.values() if r.get("audio_download_status") == "done" and r.get("asr_path") and r.get("diarization_status") != "done")
    report = {
        "schema_version": "0.2.0",
        "benchmark_type": "gpu_asr_backend_comparison",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "window_seconds": args.window_seconds,
        "sources": [{k: v for k, v in c.items() if k != "audio"} for c in cases],
        "runs": runs,
        "baseline_backend": "openai-medium.en-float16",
        "recommended_candidate": recommendation,
        "recommendation_status": "proxy_only_requires_human_readable_review",
        "scheduler_profile": {
            "asr_rtf": selected_wall / max(1.0, selected_audio),
            "asr_backend": selected_run.get("meta", {}).get("backend"),
            "asr_model": selected_run.get("meta", {}).get("model"),
            "asr_compute_type": selected_run.get("meta", {}).get("compute_type"),
            "profile_status": "benchmark_proxy_until_human_review",
        },
        "summary": {
            "asr_speedup_vs_baseline": speedups,
            "backlog_estimated_asr_gpu_hours": (backlog_asr_seconds * (selected_wall / max(1.0, selected_audio))) / 3600.0,
            "backlog_estimated_diarization_gpu_hours": None,
            "processed_audio_hours_per_benchmark_run": selected_audio / 3600.0,
            "sources_per_benchmark_run": len(cases),
        },
        "known_risks": ["OpenAI baseline uses the same fixed windows but backend decoders differ", "word timestamps are preserved but probability field semantics differ by backend", "proxy quality comparison does not replace human/audio review"],
        "artifact_directory": str(out_dir.relative_to(ROOT)),
    }
    write_json(ROOT / "reports" / "gpu_pipeline_performance.json", report)
    write_json(ROOT / "checkpoints" / "gpu_performance_profile.json", {"schema_version": "0.1.0", "source": "fixed-window-gpu-benchmark", "updated_at": report["created_at"], "scheduler_profile": report["scheduler_profile"]})
    write_summary(report)
    write_json(ROOT / "reports" / "gpu_benchmark_artifacts" / run_id / "benchmark.json", report)
    print(json.dumps({"benchmark": "complete", "sources": len(cases), "runs": [r["id"] for r in runs], "recommended_candidate": recommendation, "report": "reports/gpu_pipeline_performance.json"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
