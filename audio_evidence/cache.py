from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .contracts import ContractError, canonical_sha256


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes
            process_query_limited_information = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(process_query_limited_information, False, pid)
            if not handle:
                return False
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        except (AttributeError, OSError):
            # Fail safe: an uncertain owner is treated as alive, so its lock
            # is never reclaimed merely because liveness could not be proved.
            return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _remove_stale_lock(lock: Path, stale_after_seconds: float) -> bool:
    """Recover only an expired lock whose recorded owner is no longer alive."""
    try:
        metadata_path = lock / "owner.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        created_at = float(metadata.get("created_at", lock.stat().st_mtime))
        pid = int(metadata.get("pid", -1))
        expired = time.time() - created_at >= stale_after_seconds
        if not expired or _pid_is_alive(pid):
            return False
        if metadata_path.exists():
            metadata_path.unlink()
        lock.rmdir()
        return True
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _acquire_lock(lock: Path, stale_after_seconds: float, purpose: str, *, wait_timeout_seconds: float = 0.0, poll_seconds: float = 0.01) -> None:
    deadline = time.monotonic() + wait_timeout_seconds
    while True:
        try:
            lock.mkdir()
            (lock / "owner.json").write_text(json.dumps({"pid": os.getpid(), "created_at": time.time(), "purpose": purpose}, sort_keys=True), encoding="utf-8")
            return
        except FileExistsError:
            if _remove_stale_lock(lock, stale_after_seconds):
                continue
            if time.monotonic() >= deadline:
                raise ContractError("lock is active or cannot be recovered safely: %s" % lock)
            time.sleep(poll_seconds)


def _release_lock(lock: Path) -> None:
    metadata = lock / "owner.json"
    if metadata.exists():
        metadata.unlink()
    lock.rmdir()


class EvidenceCache:
    """Content-addressed, atomic cache with a per-key inter-process lock."""

    def __init__(self, root: Path, *, stale_lock_seconds: float = 3600.0, lock_wait_seconds: float = 30.0, lock_poll_seconds: float = 0.01) -> None:
        self.root = root
        self.stale_lock_seconds = stale_lock_seconds
        self.lock_wait_seconds = lock_wait_seconds
        self.lock_poll_seconds = lock_poll_seconds
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(audio_checksum: str, interval: Dict[str, Any], model_revision: str, parameters: Dict[str, Any], enrollment_revision: str) -> str:
        if not all((audio_checksum, interval, model_revision, enrollment_revision)):
            raise ContractError("cache key provenance is incomplete")
        return canonical_sha256({"audio_checksum": audio_checksum, "interval": interval, "model_revision": model_revision, "parameters": parameters, "enrollment_revision": enrollment_revision})

    def get(self, stage: str, key: str):
        path = self.root / stage / (key + ".json")
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return value.get("payload") if value.get("cache_key") == key and value.get("stage") == stage else None

    def put(self, stage: str, key: str, payload: Any) -> None:
        directory = self.root / stage
        directory.mkdir(parents=True, exist_ok=True)
        lock = directory / (key + ".lockdir")
        _acquire_lock(lock, self.stale_lock_seconds, "cache:%s:%s" % (stage, key), wait_timeout_seconds=self.lock_wait_seconds, poll_seconds=self.lock_poll_seconds)
        try:
            self._write_unlocked(stage, key, payload, directory)
        finally:
            _release_lock(lock)

    def _write_unlocked(self, stage: str, key: str, payload: Any, directory: Path) -> None:
        path = directory / (key + ".json")
        temporary = directory / (key + ".%s.%s.tmp" % (os.getpid(), id(payload)))
        try:
            temporary.write_text(json.dumps({"cache_key": key, "stage": stage, "payload": payload}, ensure_ascii=False, sort_keys=True), encoding="utf-8")
            os.replace(str(temporary), str(path))
        finally:
            if temporary.exists():
                temporary.unlink()

    def get_or_compute(self, stage: str, key: str, function: Callable[[], Any]):
        cached = self.get(stage, key)
        if cached is not None:
            return cached, True
        directory = self.root / stage
        directory.mkdir(parents=True, exist_ok=True)
        lock = directory / (key + ".lockdir")
        _acquire_lock(lock, self.stale_lock_seconds, "cache:%s:%s" % (stage, key), wait_timeout_seconds=self.lock_wait_seconds, poll_seconds=self.lock_poll_seconds)
        try:
            cached = self.get(stage, key)
            if cached is not None:
                return cached, True
            payload = function()
            self._write_unlocked(stage, key, payload, directory)
            return payload, False
        finally:
            _release_lock(lock)


class CheckpointStore:
    def __init__(self, path: Path, *, stale_lock_seconds: float = 3600.0) -> None:
        self.path = path
        self.stale_lock_seconds = stale_lock_seconds
        self._document: Dict[str, Any] = {}
        self._loaded = False
        self._dirty = False

    def load(self) -> Dict[str, Any]:
        # Memoized: a production run records >10k stage entries, and re-reading
        # the growing checkpoint file for every record would dominate wall time.
        if not self._loaded:
            if not self.path.exists():
                self._document = {"schema_version": "1.0.0", "completed": {}}
            else:
                self._document = json.loads(self.path.read_text(encoding="utf-8"))
            self._loaded = True
        return self._document

    def seed(self, document: Optional[Dict[str, Any]]) -> None:
        """Preload prior durable progress so redundant writes can be skipped."""
        if not document:
            return
        completed = self.load().setdefault("completed", {})
        for window_id, stages in (document.get("completed") or {}).items():
            completed.setdefault(str(window_id), {}).update(stages or {})

    def is_recorded(self, window_id: str, stage: str, cache_key: str) -> bool:
        return (self.load().get("completed", {}).get(str(window_id), {}) or {}).get(stage) == cache_key

    def record(self, window_id: str, stage: str, cache_key: str) -> None:
        if self.is_recorded(window_id, stage, cache_key):
            return
        self.load().setdefault("completed", {}).setdefault(str(window_id), {})[stage] = cache_key
        self._dirty = True
        self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock = self.path.with_suffix(self.path.suffix + ".lockdir")
        _acquire_lock(lock, self.stale_lock_seconds, "checkpoint:%s" % self.path.name)
        temporary = self.path.with_suffix(self.path.suffix + ".%s.tmp" % os.getpid())
        try:
            temporary.write_text(json.dumps(self._document, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
            os.replace(str(temporary), str(self.path))
            self._dirty = False
        finally:
            if temporary.exists():
                temporary.unlink()
            _release_lock(lock)

    def flush(self) -> None:
        if self._dirty:
            self._write()
