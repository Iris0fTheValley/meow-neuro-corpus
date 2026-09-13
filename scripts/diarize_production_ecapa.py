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
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

from manifest_tools import ROOT, log_event, read_jsonl, safe_name, write_json, write_master


PIPELINE_VERSION = "ecapa-random-access-v2"


def update_performance_profile(performance: dict) -> None:
    path = ROOT / "checkpoints" / "gpu_performance_profile.json"
    payload = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    profile = payload.setdefault("scheduler_profile", {})
    history = payload.setdefault("diarization_history", [])
    history.append({"source_id": performance.get("source_id"), "audio_duration_seconds": performance.get("audio_duration_seconds"), "real_time_factor": performance.get("real_time_factor"), "observed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    del history[:-20]
    values = [float(x["real_time_factor"]) for x in history if x.get("real_time_factor") is not None]
    if values:
        profile.update({"diarization_rtf": statistics.median(values), "profile_status": "rolling_observed"})
    payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(path, payload)


def expand_speaker_units(segment: dict, max_seconds: float = 8.0, silence_gap: float = 0.8) -> list[dict]:
    """Use word timing gaps to protect speaker changes inside long ASR segments."""
    words = [w for w in (segment.get("words") or []) if w.get("start") is not None and w.get("end") is not None]
    if not words:
        return [segment]
    groups: list[list[dict]] = []
    current = [words[0]]
    for word in words[1:]:
        gap = float(word["start"]) - float(current[-1]["end"])
        span = float(word["end"]) - float(current[0]["start"])
        if gap >= silence_gap or span >= max_seconds:
            groups.append(current)
            current = [word]
        else:
            current.append(word)
    groups.append(current)
    units = []
    for index, group in enumerate(groups):
        unit = dict(segment)
        unit["id"] = f"{segment.get('id', 'segment')}:{index}"
        unit["start"] = float(group[0]["start"])
        unit["end"] = float(group[-1]["end"])
        unit["text"] = "".join(str(w.get("word") or "") for w in group).strip()
        unit["words"] = group
        if unit["end"] - unit["start"] >= 0.6 and unit["text"]:
            units.append(unit)
    return units or [segment]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-id", required=True)
    ap.add_argument("--clusters", type=int, default=2)
    args = ap.parse_args()
    rows = read_jsonl(ROOT / "manifest" / "master_video_manifest.jsonl")
    row = next(r for r in rows if r.get("source_id") == args.source_id)
    audio = ROOT / row["audio_path"]
    asr = json.loads((ROOT / row["asr_path"]).read_text(encoding="utf-8"))
    segments = [unit for segment in asr.get("segments", []) for unit in expand_speaker_units(segment) if unit.get("text", "").strip() and float(unit.get("end", 0)) - float(unit.get("start", 0)) >= 0.6]
    source_duration = float(row.get("duration") or 0.0)
    wav = ROOT / "tmp" / f"{safe_name(args.source_id)}.diarization.wav"
    t_total = time.perf_counter()
    decode_start = time.perf_counter()
    try:
        wav.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(audio), "-ac", "1", "-ar", "16000", "-f", "wav", str(wav)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        decode_seconds = time.perf_counter() - decode_start
        model_load_start = time.perf_counter()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        speechbrain_device = "cuda:0" if device == "cuda" else "cpu"
        encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "models" / "speechbrain_ecapa"), run_opts={"device": speechbrain_device}, local_strategy=LocalStrategy.COPY)
        model_load_seconds = time.perf_counter() - model_load_start
        try:
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        max_samples = 8 * 16000
        embeddings = []
        kept = []
        batch = []
        batch_kept = []

        def encode_pending() -> None:
            if not batch:
                return
            x = torch.from_numpy(np.asarray(batch))
            with torch.inference_mode():
                encoded = encoder.encode_batch(x).squeeze(1).detach().cpu().numpy()
            encoded /= np.maximum(np.linalg.norm(encoded, axis=1, keepdims=True), 1e-9)
            embeddings.append(encoded)
            kept.extend(batch_kept)
            batch.clear()
            batch_kept.clear()

        embedding_start = time.perf_counter()
        with sf.SoundFile(wav) as source:
            sample_rate = int(source.samplerate)
            for segment in segments:
                a = max(0, int(float(segment["start"]) * sample_rate))
                b = min(len(source), int(float(segment["end"]) * sample_rate))
                source.seek(a)
                clip = source.read(max(0, b - a), dtype="float32", always_2d=False)
                if len(clip) < 0.6 * sample_rate:
                    continue
                clip = np.asarray(clip[:max_samples], dtype="float32")
                padded = np.zeros(max_samples, dtype="float32")
                padded[:len(clip)] = clip
                batch.append(padded)
                batch_kept.append(segment)
                if len(batch) >= 32:
                    encode_pending()
            encode_pending()
        embedding_seconds = time.perf_counter() - embedding_start
        if not embeddings:
            raise ValueError(f"no usable speech segments for diarization: {args.source_id}")
        matrix = np.concatenate(embeddings, axis=0)
        clustering_start = time.perf_counter()
        actual_clusters = max(2, min(int(args.clusters), len(matrix)))
        labels = AgglomerativeClustering(n_clusters=actual_clusters, metric="cosine", linkage="average").fit_predict(matrix)
        centroids = np.vstack([matrix[labels == k].mean(axis=0) for k in range(actual_clusters)])
        centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-9)
        out_segments = []
        for segment, label, emb in zip(kept, labels, matrix):
            sims = centroids @ emb
            order = np.argsort(sims)[::-1]
            conf = float(max(0.0, min(1.0, (sims[order[0]] - sims[order[1]] + 1) / 2))) if len(order) > 1 else 1.0
            out_segments.append({"start": segment["start"], "end": segment["end"], "text": segment.get("text", ""), "cluster": f"SPEAKER_{int(label):02d}", "speaker_confidence": conf, "source_asr_segment_id": segment.get("id")})
        clustering_seconds = time.perf_counter() - clustering_start
        n_labels = len(set(labels))
        silhouette = float(silhouette_score(matrix, labels, metric="cosine")) if n_labels > 1 and len(matrix) > n_labels else None
        try:
            vram_peak = int(torch.cuda.max_memory_reserved() / (1024 * 1024)) if device == "cuda" else None
        except Exception:
            vram_peak = None
        wall_time = time.perf_counter() - t_total
        performance = {
            "source_id": args.source_id,
            "audio_duration_seconds": source_duration,
            "wall_time_seconds": wall_time,
            "real_time_factor": wall_time / source_duration if source_duration else None,
            "audio_seconds_per_wall_second": source_duration / max(1e-9, wall_time) if source_duration else None,
            "decode_time_seconds": decode_seconds,
            "model_load_seconds": model_load_seconds,
            "embedding_time_seconds": embedding_seconds,
            "clustering_time_seconds": clustering_seconds,
            "vram_peak_mib_torch_reserved": vram_peak,
            "segment_count": len(out_segments),
        }
        report = {
            "schema_version": "0.2.0",
            "pipeline_version": PIPELINE_VERSION,
            "source_id": args.source_id,
            "method": "SpeechBrain ECAPA embeddings + agglomerative clustering",
            "device": device,
            "segments": len(out_segments),
            "n_clusters": actual_clusters,
            "cluster_sizes": {f"SPEAKER_{k:02d}": int((labels == k).sum()) for k in range(actual_clusters)},
            "silhouette_cosine": silhouette,
            "speaker_mapping": "anonymous_only",
            "mapping_note": "Cluster labels are not asserted to be Neuro, Evil Neuro, Vedal, or guest without validated reference evidence.",
            "performance": performance,
            "segments_with_speaker": out_segments,
        }
        out = ROOT / "diarization" / f"{safe_name(args.source_id)}.ecapa-v2.json"
        write_json(out, report)
        row.update({"diarization_status": "done", "diarization_path": str(out.relative_to(ROOT)), "diarization_method": report["method"], "diarization_pipeline_version": PIPELINE_VERSION, "speaker_mapping_status": "attempted_anonymous", "speaker_mapping_note": report["mapping_note"], "diarization_performance": performance})
        update_performance_profile(performance)
        write_master(rows)
        log_event("diarization_production_complete", source_id=args.source_id, segments=len(out_segments), silhouette=silhouette, performance=performance)
        print(json.dumps({"source_id": args.source_id, "segments": len(out_segments), "cluster_sizes": report["cluster_sizes"], "silhouette_cosine": silhouette, "device": device, "performance": performance}, ensure_ascii=False))
    finally:
        wav.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
