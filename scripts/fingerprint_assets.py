from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import av
import numpy as np
import imageio_ffmpeg

from manifest_tools import ROOT, log_event, read_jsonl, safe_name, write_json, write_master


def sha256(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def spectral_fingerprint(path: Path, points: list[float]) -> str | None:
    try:
        container = av.open(str(path))
        stream = next(s for s in container.streams if s.type == "audio")
        rate = 16000
        samples = []
        for point in points:
            container.seek(int(point * av.time_base), stream=stream, backward=True)
            frames = []
            for frame in container.decode(stream):
                arr = frame.to_ndarray()
                if arr.ndim == 2:
                    arr = arr.mean(axis=0)
                frames.append(arr.astype(np.float32))
                if sum(len(x) for x in frames) >= rate * 12:
                    break
            if frames:
                sample = np.concatenate(frames)[: rate * 12]
                spectrum = np.abs(np.fft.rfft(sample * np.hanning(len(sample))))
                bands = np.array_split(spectrum[1:], 32)
                values = np.array([float(np.mean(b)) for b in bands])
                threshold = float(np.median(values))
                samples.append("".join("1" if x >= threshold else "0" for x in values))
        container.close()
        return ":".join(samples) if samples else None
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    rows = read_jsonl(ROOT / "manifest" / "master_video_manifest.jsonl")
    exact: dict[str, list[str]] = defaultdict(list)
    fingerprint: dict[str, list[tuple[str, float | None]]] = defaultdict(list)
    processed = 0
    for row in rows:
        paths = []
        for field in ("audio_path", "video_path"):
            if row.get(field):
                p = ROOT / row[field]
                if p.exists():
                    paths.append((field, p))
        if not paths:
            continue
        for field, path in paths:
            hash_field = "content_hash" if field == "video_path" else "audio_content_hash"
            fp_field = "audio_fingerprint" if field == "audio_path" else "video_frame_fingerprint"
            if not row.get(hash_field):
                row[hash_field] = sha256(path)
            if field == "audio_path" and not row.get(fp_field):
                row[fp_field] = spectral_fingerprint(path, [0.0, max(0.0, float(row.get("duration") or 0) / 2.0)])
            exact[row[hash_field]].append(f"{row['source_platform']}:{row['source_id']}")
            if row.get(fp_field):
                fingerprint[row[fp_field]].append((f"{row['source_platform']}:{row['source_id']}", float(row.get("duration") or 0) or None))
        processed += 1
    groups = []
    seen = set()
    for mapping, relation in ((exact, "exact_content_hash"), (fingerprint, "audio_fingerprint")):
        for signature, members in mapping.items():
            if len(members) < 2:
                continue
            if relation == "audio_fingerprint":
                # The compact spectral signature is a candidate signal only.
                # Require near-equal duration before creating a cross-source group;
                # otherwise common music/silence can create false collisions.
                candidates = [m for m, _ in members]
                durations = [d for _, d in members if d]
                if len(durations) >= 2 and max(durations) / min(durations) > 1.05:
                    continue
                members = candidates
            key = tuple(sorted(members))
            if key in seen:
                continue
            seen.add(key)
            groups.append({"duplicate_group": f"dg_{len(groups)+1:05d}", "relation": relation, "members": sorted(members), "status": "review_required"})
    # Assign group IDs without removing alternate subtitle-bearing assets.
    for group in groups:
        for member in group["members"]:
            platform, source_id = member.split(":", 1)
            for row in rows:
                if row.get("source_platform") == platform and row.get("source_id") == source_id:
                    row["duplicate_group"] = group["duplicate_group"]
                    row["duplicate_relation"] = group["relation"]
    write_json(ROOT / "duplicate_groups.json", {"schema_version": "0.1.0", "groups": groups})
    write_master(rows)
    log_event("fingerprint_complete", assets=processed, duplicate_groups=len(groups))
    print(json.dumps({"assets": processed, "duplicate_groups": len(groups)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
