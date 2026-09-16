from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from .contracts import ContractError, Interval, Timebase


WINDOW_POLICY_VERSION = "audio-window-planner-v1"


@dataclass(frozen=True)
class WindowRequest:
    sample_id: str
    target_turn_id: str
    recording_id: str
    canonical_recording_id: str
    recording_family_id: str
    source_audio: str
    audio_checksum: str
    requested: Interval


@dataclass(frozen=True)
class MergedWindow:
    window_id: str
    recording_id: str
    canonical_recording_id: str
    recording_family_id: str
    source_audio: str
    audio_checksum: str
    interval: Interval
    sample_ids: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_id": self.window_id,
            "recording_id": self.recording_id,
            "canonical_recording_id": self.canonical_recording_id,
            "recording_family_id": self.recording_family_id,
            "source_audio": self.source_audio,
            "audio_checksum": self.audio_checksum,
            "merged_interval": self.interval.to_dict(),
            "sample_ids": self.sample_ids,
            "policy_version": WINDOW_POLICY_VERSION,
        }


class AudioWindowPlanner:
    def __init__(self, padding_before: float = 0.0, padding_after: float = 0.0, merge_gap: float = 0.0) -> None:
        if min(padding_before, padding_after, merge_gap) < 0:
            raise ContractError("window planner parameters must be non-negative")
        self.padding_before = padding_before
        self.padding_after = padding_after
        self.merge_gap = merge_gap

    def request_from_interaction(self, row: Dict[str, Any]) -> WindowRequest:
        timestamps = row.get("timestamps") or {}
        try:
            start, end = float(timestamps["start"]), float(timestamps["end"])
        except (KeyError, TypeError, ValueError):
            raise ContractError("interaction has no usable target/context timestamps")
        timebase = timestamps.get("timebase", Timebase.RECORDING_SECONDS.value)
        try:
            interval = Interval(max(0.0, start - self.padding_before), end + self.padding_after, Timebase(timebase))
        except ValueError:
            raise ContractError("unknown timebase")
        target_ids = [str(value) for value in row.get("target_turn_ids") or []]
        required = ["sample_id", "recording_id", "canonical_recording_id", "recording_family_id", "source_audio", "source_audio_checksum"]
        if any(not row.get(name) for name in required) or not target_ids:
            raise ContractError("audio window request lacks stable linkage or source audio mapping")
        if len(str(row["source_audio_checksum"])) != 64:
            raise ContractError("source audio checksum must be SHA-256")
        return WindowRequest(
            sample_id=str(row["sample_id"]), target_turn_id=target_ids[-1], recording_id=str(row["recording_id"]),
            canonical_recording_id=str(row["canonical_recording_id"]), recording_family_id=str(row["recording_family_id"]),
            source_audio=str(row["source_audio"]), audio_checksum=str(row["source_audio_checksum"]), requested=interval,
        )

    def merge(self, requests: Iterable[WindowRequest]) -> Dict[str, Any]:
        ordered = sorted(requests, key=lambda item: (item.recording_id, item.source_audio, item.requested.start, item.requested.end, item.sample_id))
        windows: List[MergedWindow] = []
        mappings: List[Dict[str, Any]] = []
        group: List[WindowRequest] = []

        def flush(items: List[WindowRequest]) -> None:
            if not items:
                return
            first = items[0]
            if any((item.recording_id, item.canonical_recording_id, item.recording_family_id, item.source_audio, item.audio_checksum) != (first.recording_id, first.canonical_recording_id, first.recording_family_id, first.source_audio, first.audio_checksum) for item in items):
                raise ContractError("merged audio interval crossed a recording/provenance boundary")
            start = min(item.requested.start for item in items)
            end = max(item.requested.end for item in items)
            identity = "%s|%s|%s|%.6f|%.6f|%s" % (first.recording_id, first.source_audio, first.audio_checksum, start, end, WINDOW_POLICY_VERSION)
            window_id = "aw_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
            sample_ids = sorted({item.sample_id for item in items})
            windows.append(MergedWindow(window_id, first.recording_id, first.canonical_recording_id, first.recording_family_id, first.source_audio, first.audio_checksum, Interval(start, end), sample_ids))
            for item in items:
                mappings.append({"sample_id": item.sample_id, "target_turn_id": item.target_turn_id, "requested_interval": item.requested.to_dict(), "window_id": window_id})

        for request in ordered:
            if not group:
                group = [request]
                continue
            previous_end = max(item.requested.end for item in group)
            same_source = (request.recording_id, request.canonical_recording_id, request.recording_family_id, request.source_audio, request.audio_checksum) == (group[0].recording_id, group[0].canonical_recording_id, group[0].recording_family_id, group[0].source_audio, group[0].audio_checksum)
            if same_source and request.requested.start <= previous_end + self.merge_gap:
                group.append(request)
            else:
                flush(group)
                group = [request]
        flush(group)
        return {"schema_version": "1.0.0", "policy_version": WINDOW_POLICY_VERSION, "windows": [item.to_dict() for item in windows], "sample_window_mappings": sorted(mappings, key=lambda item: (item["sample_id"], item["window_id"]))}
