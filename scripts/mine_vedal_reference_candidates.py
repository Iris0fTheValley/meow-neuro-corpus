from __future__ import annotations

import json
import re
import subprocess
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import soundfile as sf
import torch
import torchaudio

from manifest_tools import ROOT, read_jsonl, safe_name, write_json
from mine_speaker_reference_candidates import make_windows, text_is_usable, SINGING_HINT


def extract_wave(audio: Path, start: float, duration: float) -> np.ndarray | None:
    with tempfile.TemporaryDirectory(prefix="vedal_pitch_") as tmp:
        wav = Path(tmp) / "clip.wav"
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-ss", str(start), "-i", str(audio), "-t", str(duration), "-ac", "1", "-ar", "16000", "-f", "wav", str(wav)]
        try:
            subprocess.run(cmd, check=True, timeout=60, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            signal, _ = sf.read(wav, dtype="float32")
            return np.asarray(signal, dtype="float32")
        except (OSError, subprocess.SubprocessError, RuntimeError):
            return None


def median_pitch(signal: np.ndarray) -> float | None:
    if signal.size < 16000:
        return None
    wave = torchaudio.functional.detect_pitch_frequency(torch.from_numpy(signal).unsqueeze(0), 16000, frame_time=0.01).numpy().reshape(-1)
    voiced = wave[(wave >= 55) & (wave <= 350)]
    return float(np.median(voiced)) if voiced.size >= 3 else None


def known_hints(row: dict) -> set[str]:
    found = set()
    for value in row.get("participants") or []:
        low = str(value).lower()
        for key in ("neuro", "evil", "vedal"):
            if key in low:
                found.add(key.upper())
    return found


def main() -> None:
    rows = read_jsonl(ROOT / "manifest" / "master_video_manifest.jsonl")
    candidates = []
    source_reports = {}
    for row in rows:
        hints = known_hints(row)
        if "VEDAL" not in hints or not ({"NEURO", "EVIL"} & hints):
            continue
        if row.get("diarization_status") != "done" or not row.get("audio_path") or not row.get("diarization_path"):
            continue
        if SINGING_HINT.search(str(row.get("title") or "")):
            continue
        audio = ROOT / str(row["audio_path"])
        if not audio.exists():
            continue
        diar = json.loads((ROOT / str(row["diarization_path"])).read_text(encoding="utf-8"))
        segments = [s for s in diar.get("segments_with_speaker") or [] if text_is_usable(str(s.get("text") or "")) and float(s.get("speaker_confidence") or 0) >= 0.65]
        clusters = sorted({str(s.get("cluster")) for s in segments})
        if len(clusters) < 2 or len(clusters) > 3:
            continue
        cluster_pitch: dict[str, list[float]] = defaultdict(list)
        for cluster in clusters:
            selected = sorted([s for s in segments if str(s.get("cluster")) == cluster], key=lambda s: float(s.get("end") or 0) - float(s.get("start") or 0), reverse=True)[:8]
            for segment in selected:
                start = float(segment.get("start") or 0)
                duration = min(8.0, float(segment.get("end") or 0) - start)
                signal = extract_wave(audio, start, duration)
                pitch = median_pitch(signal if signal is not None else np.zeros(0, dtype="float32"))
                if pitch is not None:
                    cluster_pitch[cluster].append(pitch)
        medians = {cluster: float(np.median(values)) for cluster, values in cluster_pitch.items() if values}
        if len(medians) < 2:
            continue
        vedal_cluster = min(medians, key=medians.get)
        other_pitch = [value for cluster, value in medians.items() if cluster != vedal_cluster]
        pitch_gap = min(other_pitch) - medians[vedal_cluster]
        if pitch_gap < 35.0:
            continue
        vedal_segments = [s for s in segments if str(s.get("cluster")) == vedal_cluster]
        windows = make_windows(vedal_segments)
        if not windows:
            continue
        source_reports[str(row["source_id"])] = {
            "title": row.get("title"),
            "metadata_hints": sorted(hints),
            "cluster_pitch_hz": {k: round(v, 2) for k, v in medians.items()},
            "pitch_gap_hz": round(pitch_gap, 2),
            "vedal_cluster": vedal_cluster,
            "status": "provisional_pitch_based_candidate",
        }
        for index, window in enumerate(windows[:3]):
            rel = Path("speaker_refs") / "provisional_clips" / "VEDAL" / f"{safe_name(str(row['source_id']))}__{index:02d}.flac"
            output = ROOT / rel
            output.parent.mkdir(parents=True, exist_ok=True)
            # Reuse the project ffmpeg wrapper through the already validated
            # imageio-ffmpeg binary, keeping extraction local and auditable.
            with tempfile.TemporaryDirectory(prefix="vedal_extract_") as tmp:
                tmp_wav = Path(tmp) / "clip.wav"
                cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-ss", str(window["start"]), "-i", str(audio), "-t", str(window["duration"]), "-ac", "1", "-ar", "16000", "-f", "wav", str(tmp_wav)]
                try:
                    subprocess.run(cmd, check=True, timeout=120, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                    signal, rate = sf.read(tmp_wav, dtype="float32")
                    sf.write(output, signal, rate, format="FLAC")
                except (OSError, subprocess.SubprocessError):
                    continue
            if output.exists() and output.stat().st_size > 1024:
                candidates.append({
                    "candidate_id": f"vedal:{row['source_id']}:{index:02d}",
                    "identity_hint": "VEDAL",
                    "candidate_status": "provisional_pitch_based_not_trusted",
                    "source_id": row["source_id"],
                    "source_platform": row.get("source_platform"),
                    "source_url": row.get("source_url"),
                    "title": row.get("title"),
                    "audio_path": row["audio_path"],
                    "clip_path": str(rel),
                    "start": window["start"],
                    "end": window["end"],
                    "duration": window["duration"],
                    "cluster": window["cluster"],
                    "mean_speaker_confidence": window["mean_speaker_confidence"],
                    "text_preview": window["text"][:500],
                    "evidence": {"metadata_hints": sorted(hints), "acoustic_hint": "lowest median voiced F0 cluster with >=35 Hz gap", "identity_verification": "pending_cross_model_and_gold_validation", "trusted": False},
                })

    out = ROOT / "speaker_refs" / "vedal_provisional_reference_candidates.jsonl"
    out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in candidates), encoding="utf-8")
    summary = {
        "schema_version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PROVISIONAL_PITCH_BASED_PENDING_VALIDATION",
        "candidate_count": len(candidates),
        "source_count": len(source_reports),
        "trusted_clip_count": 0,
        "source_reports": source_reports,
    }
    write_json(ROOT / "speaker_refs" / "reference_bank_vedal_provisional.json", summary)
    print(json.dumps({"candidate_count": len(candidates), "source_count": len(source_reports), "sources": sorted(source_reports)}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
