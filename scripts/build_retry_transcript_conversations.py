from __future__ import annotations

"""Build anonymous context windows from retry transcript fallbacks, separate from audio-mapped data."""

import json
from collections import defaultdict
from datetime import datetime, timezone

from manifest_tools import ROOT, read_jsonl, write_json


def main() -> None:
    grouped = defaultdict(list)
    for row in read_jsonl(ROOT / "retry_queue" / "transcript_fallbacks_normalized.jsonl"):
        grouped[str(row.get("source_id"))].append(row)
    output_path = ROOT / "conversations" / "retry_transcript_conversations_anonymous.jsonl"
    windows = []
    for source_id, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: (float(row.get("start") or 0), str(row.get("segment_id"))))
        for offset in range(0, max(0, len(rows) - 3), 4):
            selected = rows[offset : offset + 6]
            if len(selected) < 4:
                continue
            if float(selected[-1].get("end") or 0) - float(selected[0].get("start") or 0) > 180:
                continue
            windows.append({
                "conversation_id": f"retry:{source_id}:{offset:07d}",
                "source_video_id": source_id,
                "window_type": "4_8_turns",
                "status": "TRANSCRIPT_ONLY_ANONYMOUS_REVIEW",
                "speaker_mapping_status": "anonymous_transcript_only",
                "identity_space": "NEURO_FAMILY | VEDAL | OTHER | UNKNOWN",
                "training_candidate": False,
                "source_segment_ids": [row.get("segment_id") for row in selected],
                "turns": [
                    {
                        "turn_id": f"retry:{source_id}:{index:07d}",
                        "speaker": "UNKNOWN",
                        "speaker_identity": "UNKNOWN",
                        "speaker_confidence": None,
                        "start": row.get("start"),
                        "end": row.get("end"),
                        "text": row.get("text"),
                        "suspicious_transcription": False,
                        "source_provenance": row.get("provenance"),
                    }
                    for index, row in enumerate(selected)
                ],
            })
    output_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in windows), encoding="utf-8")
    report = {
        "schema_version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "TRANSCRIPT_ONLY_ANONYMOUS_CONVERSATIONS_QA_PENDING",
        "source_count": len(grouped),
        "window_count": len(windows),
        "output": str(output_path.relative_to(ROOT)),
        "training_candidate_count": 0,
        "identity_mapped_count": 0,
        "policy": "Transcript fallback windows are isolated from audio-mapped family conversations; all speakers remain UNKNOWN and training is closed until audio/speaker QA.",
    }
    write_json(ROOT / "reports" / "retry_transcript_conversation_progress.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
