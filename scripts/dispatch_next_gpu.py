from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from manifest_tools import AtomicDirectoryLock, ROOT, read_jsonl, safe_name, write_json


GPU_SCRIPTS = ("asr_openai_whisper.py", "asr_faster_whisper.py", "diarize_production_ecapa.py", "diarize_ecapa_benchmark.py", "benchmark_gpu_pipeline.py", "benchmark_diarization.py")


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


def load_profile() -> dict:
    for path in (ROOT / "checkpoints" / "gpu_performance_profile.json", ROOT / "reports" / "gpu_pipeline_performance.json"):
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
    return {}


def estimated_gpu_seconds(row: dict, stage: str, profile: dict) -> float:
    duration = float(row.get("duration") or 3600)
    if stage == "asr":
        rtf = float(profile.get("scheduler_profile", {}).get("asr_rtf", 0.10))
    else:
        rtf = float(profile.get("scheduler_profile", {}).get("diarization_rtf", 0.03))
    return max(1.0, duration * max(0.005, min(1.0, rtf)))


def score(row: dict, stage: str, profile: dict) -> float:
    cost = estimated_gpu_seconds(row, stage, profile)
    participants = row.get("participants") or []
    transcript_bonus = 35.0 if row.get("transcript_available") or row.get("asr_status") == "source_transcript" else 0.0
    closure_bonus = 130.0 if stage == "diarization" else 55.0
    artifact_bonus = 35.0 if stage == "diarization" else 0.0
    diversity_bonus = 5.0 * min(4, len(participants))
    short_job_bonus = 60.0 / (1.0 + cost / 900.0)
    # Stronger cost penalty prevents a single multi-hour VOD from monopolising
    # the slot while preserving a clear bonus for near-closure diarization.
    return closure_bonus + artifact_bonus + transcript_bonus + diversity_bonus + short_job_bonus - cost / 45.0


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
    pause_path = ROOT / "checkpoints" / "gpu_pause.flag"
    if pause_path.exists():
        state = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "action": "gpu_paused",
            "reason": "GPU dispatch paused by operator flag",
            "pause_flag": str(pause_path),
        }
        write_json(state_path, state)
        print(json.dumps(state, ensure_ascii=False))
        raise SystemExit(0)
    lock_path = ROOT / "checkpoints" / "gpu_dispatch.lockdir"
    try:
        with AtomicDirectoryLock(lock_path, timeout=1):
            active = gpu_job_active()
            state = {"updated_at": datetime.now(timezone.utc).isoformat(), "active_gpu_processes": active}
            if active:
                state["action"] = "hold_gpu_slot"
                state["reason"] = "an existing ASR/diarization process owns the single GPU slot"
                write_json(state_path, state)
                print(json.dumps(state, ensure_ascii=False))
                raise SystemExit(0)

            rows = read_jsonl(ROOT / "manifest" / "master_video_manifest.jsonl")
            profile = load_profile()
            candidates = []
            for row in rows:
                if row.get("audio_download_status") != "done" or not row.get("audio_path"):
                    continue
                audio_path = ROOT / str(row["audio_path"])
                if not audio_path.is_file() or audio_path.name.endswith(".part"):
                    # A stale manifest row must not monopolize the GPU slot.
                    # Leave it for alternate-source/download recovery instead.
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
                    if int(row.get("asr_fast_attempts") or 0) >= 2:
                        args = ["neuro_corpus/scripts/asr_openai_whisper.py", f"--source-id={row['source_id']}", "--model=medium.en", f"--duration={int(row.get('duration') or 3600)}"]
                    else:
                        args = ["neuro_corpus/scripts/asr_faster_whisper.py", f"--source-id={row['source_id']}", "--model=medium.en", "--compute-type=float16", "--chunk-seconds=1200", "--overlap-seconds=2", "--resume"]
                candidates.append((score(row, stage, profile), row, stage, args, estimated_gpu_seconds(row, stage, profile)))
            if not candidates:
                state.update({"action": "no_gpu_runnable_work", "reason": "no acquired audio awaits ASR or diarization"})
            else:
                _, row, stage, args, estimated_seconds = sorted(candidates, key=lambda x: (-x[0], x[4], str(x[1].get("source_id"))))[0]
                state["launched"] = launch(args, row["source_id"], stage)
                state["action"] = "launched"
                state["candidate_count"] = len(candidates)
                state["estimated_gpu_seconds"] = estimated_seconds
                state["scheduler_policy"] = "closure-aware cost-aware faster-whisper default"
            write_json(state_path, state)
            print(json.dumps(state, ensure_ascii=False))
    except TimeoutError:
        state = {"updated_at": datetime.now(timezone.utc).isoformat(), "action": "hold_dispatch_lock", "reason": "another bounded GPU dispatch is selecting or launching a job"}
        write_json(state_path, state)
        print(json.dumps(state, ensure_ascii=False))
