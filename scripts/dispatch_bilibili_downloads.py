from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from manifest_tools import ROOT, log_event, read_jsonl, safe_name, write_json


DOWNLOAD_MIN_FREE_BYTES = 40 * 1024**3


def worker_python() -> str:
    candidate = ROOT.parent / ".venv-data" / "Scripts" / "python.exe"
    return str(candidate) if candidate.exists() else sys.executable


def active_sources() -> set[str]:
    """Read local process metadata only; never inspect browser credentials."""
    try:
        import psutil
        processes = psutil.process_iter(["cmdline"])
        lines = [" ".join(p.info.get("cmdline") or []) for p in processes]
    except Exception:
        command = "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'acquire_bilibili_api.py' } | Select-Object -ExpandProperty CommandLine"
        raw = subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True, timeout=10).stdout
        lines = raw.splitlines()
    found: set[str] = set()
    for line in lines:
        if "acquire_bilibili_api.py" not in line:
            continue
        match = re.search(r"(?:--source-id[=\s]+)([A-Za-z0-9_-]+)", line)
        if match:
            found.add(match.group(1))
    return found


def score(row: dict) -> float:
    duration = float(row.get("duration") or 0)
    title = str(row.get("title") or "").lower()
    participants = {str(x).lower() for x in (row.get("participants") or [])}
    value = 0.0
    value += 48 if row.get("transcript_available") else 0
    value += 24 if duration >= 1200 else 0
    value += 12 if 1200 <= duration <= 14400 else 0
    value += 18 if "neuro" in title or "neuro" in participants else 0
    value += 18 if "evil" in title or "evil" in participants else 0
    value += 14 if "vedal" in title or "vedal" in participants else 0
    value += min(16, 4 * max(0, len(participants) - 1))
    value += 10 if any(k in title for k in ("联动", "录播", "直播", "collab", "with ")) else 0
    value -= 45 if duration and duration < 300 else 0
    value -= 22 if any(k in title for k in ("切片", "片段", "短", "karaoke", "song")) else 0
    value -= 24 if row.get("duplicate_group") else 0
    value -= 35 if row.get("training_candidate") is False else 0
    value -= min(duration / 7200, 8)
    return value


def duplicate_key(row: dict) -> tuple[str, str, int] | None:
    title = re.sub(r"[^\w]+", " ", str(row.get("title") or "").lower()).strip()
    duration = float(row.get("duration") or 0)
    date = str(row.get("stream_date_if_known") or row.get("upload_date") or "")
    if len(title) < 20 or duration < 600 or not date:
        return None
    return title, date, round(duration / 30)


def launch(source_id: str) -> dict:
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    out = (log_dir / f"acquire_{safe_name(source_id)}.out.log").open("a", encoding="utf-8")
    err = (log_dir / f"acquire_{safe_name(source_id)}.err.log").open("a", encoding="utf-8")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen(
        [worker_python(), "neuro_corpus/scripts/acquire_bilibili_api.py", "--source-id", source_id],
        cwd=str(ROOT.parent), stdout=out, stderr=err, creationflags=flags,
    )
    return {"source_id": source_id, "pid": proc.pid}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-workers", type=int, default=3)
    args = ap.parse_args()
    free_bytes = shutil.disk_usage(str(ROOT)).free
    if free_bytes < DOWNLOAD_MIN_FREE_BYTES:
        state = {
            "active_source_ids": sorted(active_sources()),
            "launched": [],
            "candidate_count": 0,
            "max_workers": args.max_workers,
            "held": "disk_low",
            "free_bytes": free_bytes,
        }
        write_json(ROOT / "checkpoints" / "network_coordinator_state.json", state)
        print(json.dumps(state, ensure_ascii=False))
        raise SystemExit(0)
    active = active_sources()
    rows = read_jsonl(ROOT / "manifest" / "master_video_manifest.jsonl")
    candidates = [
        r for r in rows
        if r.get("source_platform") == "bilibili"
        and r.get("audio_download_status") not in {"done", "blocked", "failed_permanent", "retryable", "running"}
        and r.get("source_id") not in active
    ]
    candidates.sort(key=lambda r: (-score(r), float(r.get("duration") or 1800), str(r.get("source_id"))))
    launches = []
    slots = max(0, args.max_workers - len(active))
    selected = []
    seen = set()
    for row in candidates:
        key = duplicate_key(row)
        if key is not None and key in seen:
            continue
        if key is not None:
            seen.add(key)
        selected.append(row)
        if len(selected) >= slots:
            break
    for row in selected:
        launches.append(launch(str(row["source_id"])))
    log_event(
        "download_selection",
        source_platform="bilibili",
        policy="highest_value_audio_only",
        selected=[row["source_id"] for row in selected],
    )
    state = {"active_source_ids": sorted(active), "launched": launches, "candidate_count": len(candidates), "max_workers": args.max_workers}
    write_json(ROOT / "checkpoints" / "network_coordinator_state.json", state)
    print(json.dumps(state, ensure_ascii=False))
