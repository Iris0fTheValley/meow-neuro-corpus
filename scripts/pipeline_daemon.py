from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

from manifest_tools import ROOT, AtomicDirectoryLock, log_event, write_json


CHILDREN: dict[str, list[subprocess.Popen]] = {}
DOWNLOAD_MIN_FREE_BYTES = 40 * 1024**3


def worker_python() -> str:
    candidate = ROOT.parent / ".venv-data" / "Scripts" / "python.exe"
    return str(candidate) if candidate.exists() else sys.executable


def active(pattern: str) -> bool:
    # The daemon owns its bounded producers. Track their Popen handles first;
    # this works even when the worker venv does not include psutil.
    for script, handles in list(CHILDREN.items()):
        alive = [handle for handle in handles if handle.poll() is None]
        CHILDREN[script] = alive
        if pattern in script and alive:
            return True
    # Match the script path, not an arbitrary substring. The supervising
    # Codex/PowerShell command line contains the task text and therefore may
    # mention every lane name even when no worker is running.
    token = f"neuro_corpus/scripts/{pattern}"
    try:
        import psutil
        return any(token in " ".join(p.info.get("cmdline") or []) for p in psutil.process_iter(["cmdline"]))
    except Exception:
        pass
    # Windows venv launchers can exit while their Python child remains alive,
    # so Popen handles alone are insufficient. Query public process metadata
    # (command lines only; never credentials) as a conservative fallback.
    if os.name == "nt":
        safe = token.replace("'", "''")
        probe = (
            "$p = Get-CimInstance Win32_Process | "
            f"Where-Object {{ $_.ProcessId -ne $PID -and $_.Name -match '^python(\\.exe)?$' -and $_.CommandLine -like '*{safe}*' }}; "
            "if ($p) { exit 0 } else { exit 1 }"
        )
        try:
            return subprocess.run(
                ["powershell.exe", "-NoProfile", "-Command", probe],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).returncode == 0
        except Exception:
            return False


def run(script: str, *args: str) -> None:
    command = [worker_python(), f"neuro_corpus/scripts/{script}", *args]
    try:
        result = subprocess.run(command, cwd=str(ROOT.parent), capture_output=True, text=True, timeout=120)
        if result.returncode:
            log_event("daemon_stage_failed", script=script, returncode=result.returncode, stderr=result.stderr[-1000:])
    except Exception as exc:
        log_event("daemon_stage_exception", script=script, error=repr(exc))


def launch_bounded(script: str, *args: str) -> None:
    """Start one independent network/CPU producer without blocking dispatch."""
    command = [worker_python(), f"neuro_corpus/scripts/{script}", *args]
    try:
        log_path = ROOT / "logs" / f"daemon_{script.replace('.py', '')}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(log_path, "a", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=str(ROOT.parent),
            stdout=handle,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        CHILDREN.setdefault(script, []).append(process)
        # The child owns the file descriptor after Popen; close our copy.
        handle.close()
        log_event("daemon_stage_launched", script=script, args=list(args))
    except Exception as exc:
        log_event("daemon_stage_launch_failed", script=script, error=repr(exc))


def downloads_allowed() -> bool:
    """Hold only new media acquisition when the workspace is near capacity."""
    try:
        free_bytes = os.statvfs(str(ROOT)).f_bavail * os.statvfs(str(ROOT)).f_frsize
    except AttributeError:
        try:
            import shutil
            free_bytes = shutil.disk_usage(str(ROOT)).free
        except Exception as exc:
            log_event("disk_guard_probe_failed", error=repr(exc))
            return True
    if free_bytes >= DOWNLOAD_MIN_FREE_BYTES:
        return True
    log_event(
        "downloads_held_disk_low",
        free_bytes=free_bytes,
        threshold_bytes=DOWNLOAD_MIN_FREE_BYTES,
    )
    return False


def cycle(last_maintenance: float) -> float:
    # Each dispatcher is bounded and independently fails closed. GPU ownership
    # is checked inside dispatch_next_gpu; Bilibili is capped at three sources.
    run("dispatch_next_gpu.py")
    if downloads_allowed():
        run("dispatch_bilibili_downloads.py", "--max-workers", "3")
        if not active("dispatch_youtube_downloads.py") and not active("acquire_assets.py"):
            launch_bounded("dispatch_youtube_downloads.py", "--max-workers", "2")
    # These lanes are intentionally independent of the exclusive GPU lane.
    # Keep them bounded to one metadata and one subtitle producer so a slow
    # public endpoint cannot starve local ASR/diarization or flood the network.
    if not active("fetch_metadata.py"):
        launch_bounded("fetch_metadata.py", "--limit", "20")
    if not active("fetch_subtitles.py"):
        launch_bounded("fetch_subtitles.py", "--limit", "20")
    if not active("dispatch_ladev_video.py"):
        launch_bounded("dispatch_ladev_video.py")
    if not active("process_completed_cpu.py") and not active("rebuild_diarized_conversations.py"):
        launch_bounded("process_completed_cpu.py", "--limit", "8")
    now = time.monotonic()
    if now - last_maintenance >= 120:
        for script in ("fingerprint_assets.py", "aggregate_forensic_stats.py", "build_closure_queue.py", "throughput_watchdog.py", "cleanup_processed_videos.py", "normalize_download_failures.py", "make_checkpoint.py"):
            run(script)
        return now
    return last_maintenance


if __name__ == "__main__":
    # v2 avoids an orphaned lease left by the earlier coordinator process. The
    # old lockdir is retained as an audit artifact; this path is now the sole
    # live coordinator lease for the patched daemon.
    lock_path = ROOT / "checkpoints" / "active_session_lease_v5.lockdir"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with AtomicDirectoryLock(lock_path, timeout=1):
        state_path = ROOT / "checkpoints" / "daemon_state.json"
        write_json(state_path, {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(), "policy": "single coordinator; bounded dispatch; no credential access"})
        last = 0.0
        while True:
            last = cycle(last)
            write_json(state_path, {"status": "running", "updated_at": datetime.now(timezone.utc).isoformat(), "policy": "single coordinator; bounded dispatch; no credential access"})
            time.sleep(15)
