from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from manifest_tools import ROOT, read_jsonl, safe_name, write_json
from build_auto_validation_anchors import diar_match, extract_clip, read_transcript


TTS_HINT = re.compile(r"\b(?:tts|text[- ]to[- ]speech|voice should be made|voice model|speech model)\b|语音合成|文字转语音", re.I)


def main() -> None:
    manifest = {str(row.get("source_id")): row for row in read_jsonl(ROOT / "manifest" / "master_video_manifest.jsonl") if row.get("source_id") and row.get("record_type") != "manifest_header"}
    rows = []
    source_counts = Counter()
    for source_id, source in manifest.items():
        audio = ROOT / str(source.get("audio_path") or "")
        diar_path = ROOT / str(source.get("diarization_path") or "")
        if not audio.exists() or not diar_path.exists():
            continue
        transcript = read_transcript(source_id)
        if not transcript:
            continue
        try:
            diar = json.loads(diar_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        segments = diar.get("segments_with_speaker") or []
        selected = []
        for index, segment in enumerate(transcript):
            text = str(segment.get("text") or "")
            if not TTS_HINT.search(text):
                continue
            start = float(segment.get("start") or 0.0)
            end = float(segment.get("end") or start)
            if end <= start:
                continue
            matched = diar_match(segments, start, end)
            if not matched or float(matched.get("speaker_confidence") or 0.0) < 0.65:
                continue
            clip_start = max(0.0, start - 1.0)
            clip_end = min(clip_start + 14.0, end + 1.0)
            if any(clip_start < old["end"] and old["start"] < clip_end for old in selected):
                continue
            selected.append({"index": index, "start": clip_start, "end": clip_end, "cluster": str(matched.get("cluster") or "UNKNOWN"), "confidence": float(matched.get("speaker_confidence") or 0.0), "text": text[:500]})
            if len(selected) >= 3:
                break
        for item in selected:
            rel = Path("speaker_refs") / "open_set_negative_clips" / "chat_tts" / f"{safe_name(source_id)}__tts_{item['index']:04d}.flac"
            if not extract_clip(audio, item["start"], item["end"], ROOT / rel):
                continue
            rows.append({
                "challenge_id": f"chat_tts:{source_id}:{item['index']:04d}",
                "source_id": source_id,
                "metadata_bucket": "chat_tts",
                "clip_path": str(rel),
                "start": item["start"],
                "end": item["end"],
                "cluster": item["cluster"],
                "speaker_confidence": round(item["confidence"], 6),
                "text_preview": item["text"],
                "gold_label": None,
                "gold_status": "UNLABELED_CHALLENGE_ONLY",
                "use_for_benchmark": False,
                "evidence": "ASR explicitly mentions TTS/voice generation; this identifies a challenge interval, not the speaker identity.",
            })
            source_counts[source_id] += 1
    output = ROOT / "speaker_refs" / "open_set_chat_tts_challenge.jsonl"
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    report = {
        "schema_version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "CHAT_TTS_CHALLENGE_READY_UNLABELED",
        "clip_count": len(rows),
        "source_count": len(source_counts),
        "source_clip_counts": dict(source_counts),
        "output": str(output.relative_to(ROOT)),
        "gold_label_count": 0,
        "benchmark_ready_count": 0,
        "policy": "TTS mention creates a challenge interval only; no clip is treated as OTHER or non-target without independent speaker evidence.",
    }
    write_json(ROOT / "reports" / "open_set_chat_tts_challenge.json", report)
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
