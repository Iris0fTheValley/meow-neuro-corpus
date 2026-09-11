from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from manifest_tools import ROOT, log_event, read_jsonl, safe_name


DOWNLOAD_MIN_FREE_BYTES = 40 * 1024**3


def duplicate_key(row: dict) -> tuple[str, str, int] | None:
    """Collapse obvious mirror rows before acquisition, without title-only deletion."""
    duration = float(row.get("duration") or 0)
    title = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", str(row.get("title") or "").lower())
    title = re.sub(
        r"\b(?:20\d{2}[/-]\d{1,2}[/-]\d{1,2}|\d{1,2}[/-]\d{1,2}[/-]20\d{2}|\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+20\d{2})\b",
        " ",
        title,
    )
    title = re.sub(r"[^a-z0-9]+", " ", title).strip()
    date = str(row.get("stream_date_if_known") or row.get("upload_date") or "")
    if len(title) < 20 or duration < 600 or not date:
        return None
    return title, date, round(duration / 30)


def worker_python() -> str:
    candidate = ROOT.parent / ".venv-data" / "Scripts" / "python.exe"
    return str(candidate) if candidate.exists() else sys.executable


def choose(rows: list[dict], limit: int) -> list[dict]:
    candidates = []
    for row in rows:
        if row.get("source_platform") != "youtube":
            continue
        if row.get("download_status") in {"rejected", "duplicate", "failed_permanent"}:
            continue
        if row.get("processing_status") == "rejected_duplicate_candidate":
            continue
        if row.get("audio_download_status") in {"done", "blocked", "failed_permanent", "retryable", "running"}:
            continue
        if not row.get("source_url", "").startswith("https://www.youtube.com/"):
            continue
        duration = float(row.get("duration") or 0)
        transcript = bool(row.get("transcript_available") or row.get("transcript_raw_path"))
        title = str(row.get("title") or "").lower()
        participants = {str(x).lower() for x in (row.get("participants") or [])}
        score = 0.0
        score += 48 if transcript else 0
        score += 24 if duration >= 1200 else 0
        score += 12 if 1200 <= duration <= 14400 else 0
        score += 18 if "neuro" in title or "neuro" in participants else 0
        score += 18 if "evil" in title or "evil" in participants else 0
        score += 14 if "vedal" in title or "vedal" in participants else 0
        score += min(16, 4 * max(0, len(participants) - 1))
        score += 10 if any(k in title for k in ("collab", "with ", "dev stream", "just chatting", "interview")) else 0
        score -= 45 if duration and duration < 300 else 0
        score -= 22 if any(k in title for k in ("clip", "meme", "short", "karaoke", "song", "highlight")) else 0
        score -= 24 if row.get("duplicate_group") else 0
        score -= 35 if row.get("training_candidate") is False else 0
        score -= min(duration / 7200, 8)
        candidates.append((score, duration, row))
    candidates.sort(key=lambda item: (-item[0], item[1] or 10**9, item[2].get("source_id", "")))
    selected = []
    seen = set()
    for _, _, row in candidates:
        key = duplicate_key(row)
        if key is not None and key in seen:
            continue
        if key is not None:
            seen.add(key)
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-workers", type=int, default=2)
    args = parser.parse_args()
    free_bytes = shutil.disk_usage(str(ROOT)).free
    if free_bytes < DOWNLOAD_MIN_FREE_BYTES:
        print(json.dumps({"selected": [], "held": "disk_low", "free_bytes": free_bytes}, ensure_ascii=False))
        return
    rows = read_jsonl(ROOT / "manifest" / "master_video_manifest.jsonl")
    selected = choose(rows, max(1, args.max_workers))
    log_event(
        "download_selection",
        source_platform="youtube",
        policy="highest_value_audio_only",
        selected=[row["source_id"] for row in selected],
    )
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    children: list[tuple[str, subprocess.Popen]] = []
    for row in selected:
        source_id = str(row["source_id"])
        stem = safe_name(source_id)
        out = (log_dir / f"download_youtube_{stem}.out.log").open("a", encoding="utf-8")
        err = (log_dir / f"download_youtube_{stem}.err.log").open("a", encoding="utf-8")
        # YouTube IDs may legally begin with '-', so use the equals form to
        # prevent argparse from interpreting the ID as another option.
        command = [worker_python(), "neuro_corpus/scripts/acquire_assets.py", "--kind", "audio", f"--source-id={source_id}"]
        try:
            process = subprocess.Popen(
                command,
                cwd=str(ROOT.parent),
                stdout=out,
                stderr=err,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            children.append((source_id, process))
        finally:
            out.close()
            err.close()
    results = []
    for source_id, process in children:
        results.append({"source_id": source_id, "returncode": process.wait()})
    print(json.dumps({"selected": [row["source_id"] for row in selected], "results": results}, ensure_ascii=False))


if __name__ == "__main__":
    main()
