from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from manifest_tools import ROOT, read_jsonl, safe_name, write_json


GPU_SCRIPTS = ("asr_openai_whisper.py", "asr_faster_whisper.py", "diarize_production_ecapa.py", "diarize_ecapa_benchmark.py")


def worker_python() -> str:
    """Prefer the repository's CUDA-capable runtime over system Python."""
    candidate = ROOT.parent / ".venv-data" / "Scripts" / "python.exe"
    return str(candidate) if candidate.exists() else sys.executable


def gpu_job_active() -> list[dict]:
    active = []
    try:
        import psutil
        for proc in psutil.process_iter(["pid", "cmdline"]):
            cmd = " ".join(proc.info.get("cmdline") or [])
            if any(name in cmd for name in GPU_SCRIPTS):
                active.append({"pid": proc.info["pid"], "cmdline": cmd})
    except Exception:
        # Windows fallback for environments without psutil. The command line is
        # restricted to public local process metadata and contains no credentials.
        try:
            command = "$x=Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python(\.exe)?$' -and $_.CommandLine -match 'asr_openai_whisper.py|asr_faster_whisper.py|diarize_production_ecapa.py|diarize_ecapa_benchmark.py' } | Select-Object ProcessId,CommandLine; $x | ConvertTo-Json -Compress"
            raw = subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True, timeout=10, check=True).stdout.strip()
            if not raw:
                return []
            payload = json.loads(raw)
            if isinstance(payload, dict):
                payload = [payload]
            return [{"pid": x.get("ProcessId"), "cmdline": x.get("CommandLine", "")} for x in payload]
        except Exception:
            # Fail closed rather than risk launching a second GPU job.
            return [{"status": "unknown"}]
    return active


def score(row: dict) -> float:
    duration = float(row.get("duration") or 3600)
    participants = row.get("participants") or []
    # Completion bonus is intentionally strong; cost penalty prevents one long VOD
    # from starving short, already-acquired closure work.
    return (80.0 + 8.0 * bool(row.get("transcript_available")) + 4.0 * min(3, len(participants)) - duration / 1800.0)


def launch(args: list[str], source_id: str, stage: str) -> dict:
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    out = (log_dir / f"dispatch_{safe_name(source_id)}_{stage}.out.log").open("a", encoding="utf-8")
    err = (log_dir / f"dispatch_{safe_name(source_id)}_{stage}.err.log").open("a", encoding="utf-8")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen([worker_python(), *args], cwd=str(ROOT.parent), stdout=out, stderr=err, creationflags=flags)
    return {"pid": proc.pid, "stage": stage, "source_id": source_id, "args": args}


if __name__ == "__main__":
    state_path = ROOT / "checkpoints" / "coordinator_state.json"
    active = gpu_job_active()
    state = {"updated_at": datetime.now(timezone.utc).isoformat(), "active_gpu_processes": active}
    if active:
        state["action"] = "hold_gpu_slot"
        state["reason"] = "an existing ASR/diarization process owns the single GPU slot"
        write_json(state_path, state)
        print(json.dumps(state, ensure_ascii=False))
        raise SystemExit(0)

    rows = read_jsonl(ROOT / "manifest" / "master_video_manifest.jsonl")
    candidates = []
    for row in rows:
        if row.get("audio_download_status") != "done" or not row.get("audio_path"):
            continue
        if row.get("diarization_status") == "done":
            continue
        asr_path = row.get("asr_path")
        asr_ready = bool(asr_path) and (ROOT / str(asr_path)).exists()
        if row.get("asr_status") in {"done", "source_transcript"} and asr_ready:
            stage = "diarization"
            args = ["neuro_corpus/scripts/diarize_production_ecapa.py", f"--source-id={row['source_id']}", f"--clusters={3 if len(row.get('participants') or []) >= 3 else 2}"]
        else:
            stage = "asr"
            args = ["neuro_corpus/scripts/asr_openai_whisper.py", f"--source-id={row['source_id']}", "--model=medium.en", f"--duration={int(row.get('duration') or 3600)}"]
        candidates.append((score(row) + (18.0 if stage == "diarization" else 0.0), row, stage, args))
    if not candidates:
        state.update({"action": "no_gpu_runnable_work", "reason": "no acquired audio awaits ASR or diarization"})
    else:
        _, row, stage, args = sorted(candidates, key=lambda x: (-x[0], float(x[1].get("duration") or 3600), str(x[1].get("source_id"))))[0]
        state["launched"] = launch(args, row["source_id"], stage)
        state["action"] = "launched"
        state["candidate_count"] = len(candidates)
    write_json(state_path, state)
    print(json.dumps(state, ensure_ascii=False))
