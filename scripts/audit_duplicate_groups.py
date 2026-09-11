from __future__ import annotations

import difflib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from manifest_tools import ROOT, write_json


def transcript_sample(row: dict) -> str:
    path = ROOT / str(row.get("asr_path") or "")
    if not path.exists():
        return ""
    doc = json.loads(path.read_text(encoding="utf-8"))
    text = " ".join(str(x.get("text") or "") for x in doc.get("segments") or [])
    return re.sub(r"\s+", " ", text).strip().lower()[:20000]


def main() -> None:
    rows = {}
    for line in (ROOT / "manifest" / "master_video_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        rows[(row.get("source_platform"), row.get("source_id"))] = row
    groups_doc = json.loads((ROOT / "duplicate_groups.json").read_text(encoding="utf-8")) if (ROOT / "duplicate_groups.json").exists() else {"groups": []}
    reviewed = []
    for group in groups_doc.get("groups", []):
        members = [rows.get(tuple(member.split(":", 1))) for member in group.get("members", [])]
        members = [row for row in members if row]
        evidence = {
            "duplicate_group": group.get("duplicate_group"),
            "relation": group.get("relation"),
            "status": "review_required",
            "members": [row.get("source_id") for row in members],
            "titles": [row.get("title") for row in members],
            "durations": [row.get("duration") for row in members],
            "sha256_equal": len({row.get("audio_content_hash") for row in members}) <= 1,
            "transcript_sample_similarity": None,
            "decision": "retain_both_pending_manual_or_audio_alignment",
        }
        if len(members) == 2:
            first, second = (transcript_sample(row) for row in members)
            evidence["transcript_sample_similarity"] = round(difflib.SequenceMatcher(None, first, second, autojunk=False).ratio(), 6)
        reviewed.append(evidence)
    report = {
        "schema_version": "0.1.0",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": "REVIEW_PENDING_NO_DELETIONS",
        "group_count": len(reviewed),
        "groups": reviewed,
        "policy": "Fingerprint collisions are candidate signals; retain source provenance until transcript/audio alignment or human review confirms a duplicate.",
    }
    write_json(ROOT / "reports" / "duplicate_group_review.json", report)
    print(json.dumps({"status": report["status"], "group_count": report["group_count"], "groups": reviewed}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
