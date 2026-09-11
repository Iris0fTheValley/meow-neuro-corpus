from __future__ import annotations

import argparse
import json
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


def audio_path(row: dict) -> Path:
    if row.get("audio_path") and (ROOT / row["audio_path"]).exists():
        return ROOT / row["audio_path"]
    candidates = sorted((ROOT / "raw_audio").glob(f"{safe_name(row['source_id'])}.*"))
    if not candidates:
        raise FileNotFoundError(row["source_id"])
    return candidates[0]


def make_wav(src: Path, out: Path, start: float, duration: float) -> None:
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-ss", str(start), "-t", str(duration), "-i", str(src), "-ac", "1", "-ar", "16000", str(out)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-id", default="1CxTCk0I79w")
    parser.add_argument("--max-segments", type=int, default=40)
    args = parser.parse_args()
    rows = read_jsonl(ROOT / "manifest" / "master_video_manifest.jsonl")
    row = next(r for r in rows if r.get("source_id") == args.source_id)
    if not row.get("asr_path"):
        raise SystemExit("run ASR benchmark first")
    asr = json.loads((ROOT / row["asr_path"]).read_text(encoding="utf-8"))
    segments = [s for s in asr.get("segments", []) if 2.0 <= (s["end"] - s["start"]) <= 12.0][: args.max_segments]
    wav_dir = ROOT / "tmp" / "diarization_benchmark" / safe_name(args.source_id)
    wav_dir.mkdir(parents=True, exist_ok=True)
    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(ROOT / "models" / "speechbrain_ecapa"),
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        local_strategy=LocalStrategy.COPY,
    )
    embeddings = []
    timings = []
    source = audio_path(row)
    for i, segment in enumerate(segments):
        wav = wav_dir / f"{i:04d}.wav"
        make_wav(source, wav, max(0, segment["start"] - 0.15), min(12.0, segment["end"] - segment["start"] + 0.3))
        signal, _ = sf.read(wav, dtype="float32")
        tensor = torch.from_numpy(signal).unsqueeze(0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            emb = encoder.encode_batch(tensor).squeeze().detach().cpu().numpy()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - t0)
        embeddings.append(emb / max(np.linalg.norm(emb), 1e-9))
    matrix = np.asarray(embeddings)
    results = []
    for k in (2, 3):
        if len(matrix) <= k:
            continue
        labels = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(matrix)
        score = float(silhouette_score(matrix, labels, metric="cosine"))
        results.append({"n_clusters": k, "silhouette_cosine": score, "cluster_sizes": {str(x): int((labels == x).sum()) for x in sorted(set(labels))}})
    report = {
        "source_id": args.source_id,
        "method": "SpeechBrain ECAPA-TDNN embeddings + agglomerative clustering",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "segments": len(segments),
        "mean_embedding_seconds": float(np.mean(timings)) if timings else None,
        "p95_embedding_seconds": float(np.percentile(timings, 95)) if timings else None,
        "results": results,
        "limitations": [
            "Silhouette is an unsupervised separation proxy, not speaker-label accuracy.",
            "The benchmark intentionally leaves clusters anonymous; Neuro/Evil/Vedal mapping requires high-confidence reference clips and manual/metadata validation.",
            "Music, TTS overlap, and synthetic voices may violate assumptions of ECAPA trained on human speech.",
        ],
    }
    per_source = ROOT / "reports" / f"diarization_benchmark_{safe_name(args.source_id)}.json"
    write_json(per_source, report)
    aggregate_path = ROOT / "reports" / "diarization_benchmark.json"
    aggregate = {"schema_version": "0.1.0", "benchmarks": {}}
    if aggregate_path.exists():
        try:
            aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    aggregate.setdefault("benchmarks", {})[args.source_id] = report
    write_json(aggregate_path, aggregate)
    row.update({"diarization_status": "benchmark_done", "diarization_method": report["method"], "diarization_benchmark_path": str(per_source.relative_to(ROOT)), "speaker_mapping_status": "attempted_anonymous", "speaker_mapping_note": "Anonymous clusters benchmarked; Neuro/Evil/Vedal mapping deliberately not forced."})
    write_master(rows)
    log_event("diarization_benchmark_complete", source_id=args.source_id, segments=len(segments), results=results)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
