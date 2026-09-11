from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "manifest"
MASTER_PATH = MANIFEST_DIR / "master_video_manifest.jsonl"
EVENT_LOG = ROOT / "logs" / "pipeline_events.jsonl"


class AtomicDirectoryLock:
    """Small cross-process lock with no third-party dependency.

    mkdir is atomic on the local Windows filesystem.  The lock is deliberately
    scoped to the manifest writer; readers and all other pipeline stages remain
    independent.  A bounded timeout prevents a stale manifest lock from
    becoming a global scheduler stop.
    """

    def __init__(self, path: Path, timeout: float = 30.0) -> None:
        self.path = path
        self.timeout = timeout

    def __enter__(self) -> "AtomicDirectoryLock":
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                self.path.mkdir(parents=False)
                (self.path / "owner").write_text(f"pid={os.getpid()}\n", encoding="ascii")
                return self
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"manifest lock timeout: {self.path}")
                time.sleep(0.05)

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        try:
            (self.path / "owner").unlink(missing_ok=True)
            self.path.rmdir()
        except FileNotFoundError:
            pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("record_type") == "manifest_header":
            continue
        rows.append(row)
    return rows


def key(row: dict[str, Any]) -> str:
    platform = row.get("source_platform", "unknown")
    source_id = row.get("source_id") or row.get("source_url") or row.get("title", "")
    return f"{platform}:{source_id}"


def stable_id(source_url: str) -> str:
    return hashlib.sha1(source_url.encode("utf-8")).hexdigest()[:16]


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return value.strip("._")[:160] or "unknown"


_STATUS_RANKS = {
    "audio_download_status": {"pending": 0, "retryable": 1, "blocked": 2, "failed_permanent": 2, "done": 4},
    "video_download_status": {"pending": 0, "retryable": 1, "blocked": 2, "failed_permanent": 2, "done": 4},
    "metadata_status": {"pending": 0, "retryable": 1, "failed": 1, "blocked": 2, "done": 4},
    "subtitle_status": {"pending": 0, "retryable": 1, "blocked": 2, "not_available": 2, "done": 4},
    "asr_status": {"pending": 0, "retryable": 1, "failed": 1, "blocked": 2, "source_transcript": 3, "done": 4},
    "diarization_status": {"pending": 0, "retryable": 1, "failed": 1, "blocked": 2, "done": 4},
    "conversation_status": {"pending": 0, "retryable": 1, "failed": 1, "blocked": 2, "done": 4},
    "quality_status": {"pending": 0, "retryable": 1, "failed": 1, "done": 4},
    "speaker_mapping_status": {"pending": 0, "attempted_anonymous": 2, "attempted": 2, "validated": 4, "done": 4},
}


def merge_rows(*groups: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for group in groups:
        for incoming in group:
            k = key(incoming)
            if k not in merged:
                merged[k] = dict(incoming)
                continue
            current = merged[k]
            for field, value in incoming.items():
                if value not in (None, "", [], {}):
                    if field in {"participants", "subtitle_languages"} and isinstance(value, list):
                        current[field] = sorted(set(current.get(field, [])) | set(value))
                    elif field in _STATUS_RANKS and isinstance(value, str):
                        old = current.get(field)
                        ranks = _STATUS_RANKS[field]
                        if old in (None, "") or ranks.get(value, 0) >= ranks.get(str(old), 0):
                            current[field] = value
                    else:
                        current[field] = value
    return sorted(merged.values(), key=lambda r: tuple(str(r.get(k) or "") for k in ("source_platform", "upload_date", "source_id")))


def write_master(rows: Iterable[dict[str, Any]]) -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    incoming = list(rows)
    lock_path = MASTER_PATH.with_suffix(MASTER_PATH.suffix + ".lockdir")
    with AtomicDirectoryLock(lock_path, timeout=30):
        # Read only after acquiring the writer lock.  This prevents two workers
        # that loaded stale snapshots from erasing each other's stage fields.
        if MASTER_PATH.exists():
            incoming = merge_rows(read_jsonl(MASTER_PATH), incoming)
        header = {
            "record_type": "manifest_header",
            "schema_version": "0.1.0",
            "updated_at": utc_now(),
            "notes": "Public-source candidate registry. Raw assets and source attribution are never overwritten by normalization.",
        }
        temp_path = MASTER_PATH.with_suffix(MASTER_PATH.suffix + f".{os.getpid()}.tmp")
        with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(header, ensure_ascii=False) + "\n")
            for row in incoming:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(temp_path, MASTER_PATH)


def log_event(event: str, **fields: Any) -> None:
    EVENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": utc_now(), "event": event, **fields}
    with EVENT_LOG.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def sanitized_info(info: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "id", "title", "fulltitle", "description", "channel", "channel_id", "channel_url", "uploader", "uploader_id",
        "uploader_url", "upload_date", "timestamp", "release_timestamp", "duration", "duration_string", "view_count",
        "like_count", "comment_count", "categories", "tags", "chapters", "live_status", "is_live", "was_live",
        "availability", "webpage_url", "original_url", "extractor", "extractor_key", "thumbnail", "subtitles",
        "automatic_captions", "language", "language_preference", "age_limit", "playlist", "playlist_index",
    }
    return {k: info.get(k) for k in keep if k in info}
