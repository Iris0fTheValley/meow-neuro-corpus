from __future__ import annotations

"""Build a conservative recording/content cluster index for split audits.

This is deliberately a leakage-audit layer, not an identity model.  It only
joins sources when there is strong content evidence: exact audio/content hash,
existing duplicate/parent-stream metadata, exact normalized transcript hash, or
a long exact title with near-identical duration.  No labels or promotion state
are changed by this script.
"""

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from manifest_tools import ROOT, read_json, read_jsonl, write_json


MANIFEST = ROOT / "manifest" / "master_video_manifest.jsonl"
OUT_JSON = ROOT / "reports" / "recording_content_cluster_audit.json"
OUT_JSONL = ROOT / "reports" / "recording_content_clusters.jsonl"


class UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def norm_text(value: Any) -> str:
    value = str(value or "").casefold()
    value = re.sub(r"https?://\S+", " ", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def text_fingerprint(path: Path) -> str | None:
    if not path.exists() or path.suffix.lower() not in {".json", ".jsonl"}:
        return None
    try:
        if path.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if isinstance(rows, dict):
        rows = rows.get("segments") or rows.get("results") or rows.get("items") or [rows]
    if not isinstance(rows, list):
        return None
    texts: list[str] = []
    for row in rows:
        if isinstance(row, dict):
            text = row.get("text") or row.get("transcript") or row.get("utterance")
            if text:
                texts.append(norm_text(text))
    joined = " ".join(texts).strip()
    if len(joined) < 80:
        return None
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def source_text_hash(row: dict[str, Any]) -> str | None:
    for key in ("asr_path", "normalized_subtitle_path", "transcript_path", "transcript_raw_path"):
        value = row.get(key)
        if not value:
            continue
        path = ROOT / str(value).replace("\\", "/")
        digest = text_fingerprint(path)
        if digest:
            return digest
    return None


def close_duration(left: Any, right: Any) -> bool:
    try:
        a, b = float(left), float(right)
    except (TypeError, ValueError):
        return False
    return abs(a - b) <= max(5.0, 0.01 * max(a, b, 1.0))


def main() -> None:
    rows = read_jsonl(MANIFEST)
    source_ids = [str(row.get("source_id") or "") for row in rows if row.get("source_id")]
    source_ids = sorted(set(source_ids))
    uf = UnionFind(source_ids)
    evidence: dict[str, set[str]] = defaultdict(set)
    key_to_sources: dict[tuple[str, str], list[str]] = defaultdict(list)
    prepared: dict[str, dict[str, Any]] = {}

    for row in rows:
        source_id = str(row.get("source_id") or "")
        if not source_id:
            continue
        title = norm_text(row.get("title"))
        prepared[source_id] = {
            "source_id": source_id,
            "title": row.get("title"),
            "source_url": row.get("source_url"),
            "source_platform": row.get("source_platform"),
            "duration": row.get("duration"),
            "audio_content_hash": row.get("audio_content_hash"),
            "audio_fingerprint": row.get("audio_fingerprint"),
            "duplicate_group": row.get("duplicate_group"),
            "duplicate_relation": row.get("duplicate_relation"),
            "parent_stream_if_known": row.get("parent_stream_if_known"),
            "title_key": title if len(title) >= 32 else None,
            "transcript_hash": source_text_hash(row),
        }
        for field in ("audio_content_hash", "audio_fingerprint", "duplicate_group", "parent_stream_if_known"):
            value = row.get(field)
            if value not in (None, "", [], {}):
                key_to_sources[(field, str(value))].append(source_id)
        digest = prepared[source_id]["transcript_hash"]
        if digest:
            key_to_sources[("transcript_hash", digest)].append(source_id)

    # Exact evidence is safe enough to union automatically.
    for (field, value), members in key_to_sources.items():
        unique = sorted(set(members))
        if len(unique) < 2:
            continue
        for member in unique[1:]:
            uf.union(unique[0], member)
            evidence[unique[0]].add(f"exact:{field}")

    # Conservative title-duration fallback.  It is recorded separately so it
    # can be reviewed and cannot silently act like an audio hash.
    title_groups: dict[str, list[str]] = defaultdict(list)
    for source_id, item in prepared.items():
        if item["title_key"]:
            title_groups[item["title_key"]].append(source_id)
    for title_key, members in title_groups.items():
        unique = sorted(set(members))
        for index, left in enumerate(unique):
            for right in unique[index + 1 :]:
                if close_duration(prepared[left]["duration"], prepared[right]["duration"]):
                    uf.union(left, right)
                    evidence[left].add("strict_title_duration")

    groups: dict[str, list[str]] = defaultdict(list)
    for source_id in source_ids:
        groups[uf.find(source_id)].append(source_id)

    cluster_rows: list[dict[str, Any]] = []
    source_to_cluster: dict[str, str] = {}
    for members in sorted(groups.values(), key=lambda group: (group[0], len(group))):
        members = sorted(members)
        cluster_id = "rc_" + hashlib.sha1("|".join(members).encode("utf-8")).hexdigest()[:16]
        reasons = sorted({reason for member in members for reason in evidence.get(member, set())})
        high_risk = any(reason in reasons for reason in {"exact:audio_content_hash", "exact:duplicate_group", "exact:transcript_hash", "exact:parent_stream_if_known"})
        for member in members:
            source_to_cluster[member] = cluster_id
        cluster_rows.append({
            "recording_cluster_id": cluster_id,
            "source_ids": members,
            "source_count": len(members),
            "cross_source": len(members) > 1,
            "high_risk_mirror_or_reupload": high_risk,
            "join_evidence": reasons,
            "sources": [prepared[member] for member in members],
        })

    def load_source_ids(path: Path) -> set[str]:
        values: set[str] = set()
        for row in read_jsonl(path):
            value = row.get("source_id")
            if value:
                values.add(str(value))
        return values

    anchor_sources = load_source_ids(ROOT / "speaker_refs" / "auto_validation_anchors.jsonl")
    calibration_sources = set()
    for name in ("open_set_calibration_family.jsonl", "open_set_calibration_vedal.jsonl", "open_set_calibration_other.jsonl"):
        calibration_sources |= load_source_ids(ROOT / "speaker_refs" / name)
    transfer = read_json(ROOT / "reports" / "identity_calibration_transfer_audit.json", {})
    validation_sources = {str(row.get("source_id")) for row in transfer.get("anchor_comparison", []) if row.get("source_id")}

    def cluster_set(sources: set[str]) -> set[str]:
        return {source_to_cluster[source] for source in sources if source in source_to_cluster}

    anchor_clusters = cluster_set(anchor_sources)
    validation_clusters = cluster_set(validation_sources)
    calibration_clusters = cluster_set(calibration_sources)
    overlap_anchor_validation = sorted(anchor_clusters & validation_clusters)
    overlap_anchor_calibration = sorted(anchor_clusters & calibration_clusters)
    overlap_calibration_validation = sorted(calibration_clusters & validation_clusters)

    report = {
        "schema_version": "0.1.0",
        "artifact_status": "current",
        "canonical_for": "recording_content_cluster_assignments_and_split_independence_audit",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": "working-tree",
        "source_manifest": str(MANIFEST.relative_to(ROOT)),
        "source_count": len(source_ids),
        "recording_cluster_count": len(cluster_rows),
        "cross_source_cluster_count": sum(row["cross_source"] for row in cluster_rows),
        "high_risk_mirror_or_reupload_cluster_count": sum(row["high_risk_mirror_or_reupload"] for row in cluster_rows),
        "join_evidence_counts": dict(Counter(reason for row in cluster_rows for reason in row["join_evidence"])),
        "split_audit": {
            "anchor_source_count": len(anchor_sources),
            "anchor_recording_cluster_count": len(anchor_clusters),
            "calibration_source_count": len(calibration_sources),
            "calibration_recording_cluster_count": len(calibration_clusters),
            "validation_source_count": len(validation_sources),
            "validation_recording_cluster_count": len(validation_clusters),
            "anchor_validation_overlap_clusters": overlap_anchor_validation,
            "anchor_calibration_overlap_clusters": overlap_anchor_calibration,
            "calibration_validation_overlap_clusters": overlap_calibration_validation,
            "anchor_recording_clusters_disjoint_from_validation": not overlap_anchor_validation,
            "anchor_recording_clusters_disjoint_from_calibration": not overlap_anchor_calibration,
            "calibration_recording_clusters_disjoint_from_validation": not overlap_calibration_validation,
            "policy": "A recording/content cluster is the unit of independence; source_id alone is insufficient.",
        },
        "policy": {
            "identity_labels_unchanged": True,
            "threshold_unchanged": True,
            "training_candidate_promotion": "forbidden",
            "unknown_and_ambiguous_clusters_retained": True,
        },
    }
    OUT_JSONL.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in cluster_rows), encoding="utf-8")
    report["output"] = str(OUT_JSONL.relative_to(ROOT))
    write_json(OUT_JSON, report)
    print(json.dumps({
        "status": "PASS" if report["split_audit"]["calibration_recording_clusters_disjoint_from_validation"] else "OVERLAP_REQUIRES_SPLIT_REBUILD",
        "source_count": len(source_ids),
        "recording_cluster_count": len(cluster_rows),
        "cross_source_cluster_count": report["cross_source_cluster_count"],
        "high_risk_mirror_or_reupload_cluster_count": report["high_risk_mirror_or_reupload_cluster_count"],
        "split_audit": report["split_audit"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
