from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict

from .contracts import ContractError, canonical_sha256


class EvidenceCache:
    """Content-addressed, atomic cache with a per-key inter-process lock."""

    def __init__(self, root: Path) -> None:
        self.root = root
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
        try:
            lock.mkdir()
        except FileExistsError:
            raise ContractError("cache key is already being written by another worker")
        path = directory / (key + ".json")
        temporary = directory / (key + ".%s.tmp" % os.getpid())
        try:
            temporary.write_text(json.dumps({"cache_key": key, "stage": stage, "payload": payload}, ensure_ascii=False, sort_keys=True), encoding="utf-8")
            os.replace(str(temporary), str(path))
        finally:
            if temporary.exists():
                temporary.unlink()
            lock.rmdir()

    def get_or_compute(self, stage: str, key: str, function: Callable[[], Any]):
        cached = self.get(stage, key)
        if cached is not None:
            return cached, True
        payload = function()
        self.put(stage, key, payload)
        return payload, False


class CheckpointStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": "1.0.0", "completed": {}}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def record(self, window_id: str, stage: str, cache_key: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock = self.path.with_suffix(self.path.suffix + ".lockdir")
        try:
            lock.mkdir()
        except FileExistsError:
            raise ContractError("checkpoint is already being updated by another worker")
        temporary = self.path.with_suffix(self.path.suffix + ".%s.tmp" % os.getpid())
        try:
            value = self.load()
            value.setdefault("completed", {}).setdefault(window_id, {})[stage] = cache_key
            temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
            os.replace(str(temporary), str(self.path))
        finally:
            if temporary.exists():
                temporary.unlink()
            lock.rmdir()
