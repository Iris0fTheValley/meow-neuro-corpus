from __future__ import annotations

import argparse
import json
from pathlib import Path

from faster_whisper import WhisperModel

from manifest_tools import ROOT, log_event, read_jsonl, safe_name, write_master


def locate_audio(row: dict) -> Path | None:
    if row.get("audio_path"):
        path = ROOT / row["audio_path"]
        if path.exists():
            return path
    candidates = sorted((ROOT / "raw_audio").glob(f"{safe_name(row['source_id'])}.*"))
    return candidates[0] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="distil-large-v3")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--source-id", action="append")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    out_dir = ROOT / "asr"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(ROOT / "manifest" / "master_video_manifest.jsonl")
    device = "cuda"
    compute_type = "float16"
    try:
        model = WhisperModel(args.model, device=device, compute_type=compute_type)
    except Exception as exc:
        log_event("asr_cuda_unavailable", error=repr(exc))
        device = "cpu"
        compute_type = "int8"
        model = WhisperModel(args.model, device=device, compute_type=compute_type)
    processed = 0
    for row in rows:
        if args.source_id and row.get("source_id") not in args.source_id:
            continue
        if not args.source_id and not args.all and processed >= args.limit:
            break
        if row.get("asr_status") == "done" and not args.force:
            continue
        audio = locate_audio(row)
        if not audio:
            row["asr_status"] = "blocked"
            row["asr_error"] = "audio asset not present"
            continue
        try:
            kwargs = {"language": "en", "vad_filter": True, "word_timestamps": True, "condition_on_previous_text": True}
            if args.start:
                kwargs["clip_timestamps"] = f"{args.start},{args.start + args.duration if args.duration else 1e12}"
            segments, info = model.transcribe(str(audio), **kwargs)
            seg_rows = []
            for segment in segments:
                seg_rows.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "avg_logprob": segment.avg_logprob,
                    "no_speech_prob": segment.no_speech_prob,
                    "compression_ratio": segment.compression_ratio,
                    "words": [
                        {"start": w.start, "end": w.end, "word": w.word, "probability": w.probability}
                        for w in (segment.words or [])
                    ],
                })
            output = out_dir / f"{safe_name(row['source_id'])}.{args.model.replace('/', '_')}.json"
            output.write_text(json.dumps({
                "source_id": row["source_id"], "source_url": row["source_url"], "model": args.model,
                "device": device, "compute_type": compute_type, "language": info.language,
                "language_probability": info.language_probability, "duration": info.duration,
                "segments": seg_rows,
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            row.update({"asr_status": "done", "asr_model": args.model, "asr_path": str(output.relative_to(ROOT)), "asr_segment_count": len(seg_rows), "asr_device": device})
            if args.force:
                row.setdefault("asr_benchmarks", {})[args.model] = {
                    "path": str(output.relative_to(ROOT)), "segment_count": len(seg_rows),
                    "device": device, "compute_type": compute_type, "language": info.language,
                    "language_probability": info.language_probability,
                }
            log_event("asr_complete", source_id=row["source_id"], model=args.model, device=device, segments=len(seg_rows))
        except Exception as exc:
            row["asr_status"] = "retryable"
            row["asr_error"] = repr(exc)
            log_event("asr_failed", source_id=row.get("source_id"), error=repr(exc))
        processed += 1
    write_master(rows)
    print(json.dumps({"model": args.model, "device": device, "processed": processed, "total": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
