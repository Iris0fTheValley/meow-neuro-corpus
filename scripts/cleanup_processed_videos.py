from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from manifest_tools import ROOT, log_event, read_jsonl, write_master


def eligible(row: dict) -> bool:
    video = row.get("video_path")
    audio = row.get("audio_path")
    quality = row.get("quality_metrics_path")
    if not video or not audio or not quality:
        return False
    if row.get("audio_download_status") != "done":
        return False
    if row.get("diarization_status") != "done" or row.get("conversation_status") != "done":
        return False
    return (ROOT / audio).exists() and (ROOT / quality).exists() and (ROOT / video).is_file()


def active_diarization_ids() -> set[str]:
    probe = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -match '^python(\\.exe)?$' -and "
        "$_.CommandLine -match 'diarize_production_ecapa.py' } | "
        "Select-Object -ExpandProperty CommandLine"
    )
    try:
        raw = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", probe],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except Exception:
        raw = ""
    return set(re.findall(r"--source-id(?:=|\s+)([A-Za-z0-9_-]+)", raw))


def cleanup_transient_audio() -> dict:
    """Remove regenerated WAV/part files only after durable outputs exist."""
    removed_wav = 0
    removed_bilibili = 0
    removed_bytes = 0
    active = active_diarization_ids()
    tmp = ROOT / "tmp"
    for wav in tmp.glob("*.diarization.wav"):
        source_id = wav.name[: -len(".diarization.wav")]
        output = ROOT / "diarization" / f"{source_id}.ecapa.json"
        if source_id in active or not output.is_file():
            continue
        size = wav.stat().st_size
        wav.unlink()
        removed_wav += 1
        removed_bytes += size
    rows = {row.get("source_id"): row for row in read_jsonl(ROOT / "manifest" / "master_video_manifest.jsonl")}
    for folder in tmp.glob("bilibili_*"):
        if not folder.is_dir():
            continue
        source_id = folder.name[len("bilibili_"):]
        row = rows.get(source_id) or {}
        audio = row.get("audio_path")
        if row.get("audio_download_status") != "done" or not audio or not (ROOT / audio).is_file():
            continue
        size = sum(x.stat().st_size for x in folder.rglob("*") if x.is_file())
        shutil.rmtree(folder)
        removed_bilibili += 1
        removed_bytes += size
    return {"removed_wav": removed_wav, "removed_bilibili_dirs": removed_bilibili, "removed_bytes": removed_bytes}


def main() -> None:
    rows = read_jsonl(ROOT / "manifest" / "master_video_manifest.jsonl")
    deleted = []
    skipped = 0
    now = datetime.now(timezone.utc).isoformat()
    for row in rows:
        if row.get("video_retention") == "deleted_after_processing":
            continue
        if not eligible(row):
            skipped += 1
            continue
        video = Path(ROOT / row["video_path"])
        original = row["video_path"]
        video.unlink()
        row["video_original_path"] = original
        row["video_path"] = None
        row["video_retention"] = "deleted_after_processing"
        row["video_deleted_at"] = now
        row["video_cleanup_note"] = "Audio, subtitles/ASR, diarization, conversation and quality metrics retained."
        deleted.append(row.get("source_id"))
        log_event("processed_video_deleted_audio_retained", source_id=row.get("source_id"), original_path=original)
    if deleted:
        write_master(rows)
    transient = cleanup_transient_audio()
    if transient["removed_wav"] or transient["removed_bilibili_dirs"]:
        log_event("transient_audio_cleanup", **transient)
    print(json.dumps({"deleted": deleted, "deleted_count": len(deleted), "skipped": skipped, "transient": transient}, ensure_ascii=False))


if __name__ == "__main__":
    main()
