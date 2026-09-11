from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone

from manifest_tools import ROOT, read_jsonl, write_json


ALLOWED = {"NEURO_FAMILY", "VEDAL", "UNKNOWN"}


def main() -> None:
    mapping_path = ROOT / "identity_results" / "identity_mapping_family_proxy.jsonl"
    if not mapping_path.exists():
        raise SystemExit("family mapping is not ready")
    mapping = {f"{row['source_id']}:{row['cluster']}": row for row in read_jsonl(mapping_path)}
    source = ROOT / "conversations" / "natural_conversations_anonymous.jsonl"
    output = ROOT / "conversations" / "natural_conversations_family_mapped.jsonl"
    rows = []
    for row in read_jsonl(source):
        source_id = str(row.get("source_video_id"))
        turns = []
        counts = Counter()
        for turn in row.get("turns") or []:
            key = f"{source_id}:{turn.get('speaker')}"
            mapped = mapping.get(key)
            identity = str(mapped.get("identity") if mapped else "UNKNOWN")
            if identity not in ALLOWED:
                identity = "UNKNOWN"
            confidence = str(mapped.get("identity_confidence") if mapped else "unknown")
            turns.append({**turn, "speaker_identity": identity, "identity_confidence": confidence, "mapping_status": mapped.get("status") if mapped else "MISSING_MAPPING"})
            counts[identity] += 1
        mapped_turn_count = counts["NEURO_FAMILY"] + counts["VEDAL"]
        identity_status = "PROXY_MAPPED_OPEN_SET" if mapped_turn_count else "UNKNOWN_ONLY"
        rows.append({
            **row,
            "status": "FAMILY_MAPPED_PROXY_QA_PENDING",
            "training_candidate": False,
            "candidate_grade": "Q",
            "identity_mapping": "NEURO_FAMILY | VEDAL | UNKNOWN",
            "identity_confidence": "proxy_pending_qa",
            "speaker_identity_counts": dict(counts),
            "mapped_turn_count": mapped_turn_count,
            "unknown_turn_count": counts["UNKNOWN"],
            "identity_status": identity_status,
            "turns": turns,
        })
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    report = {
        "schema_version": "0.1.0",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": "FAMILY_MAPPED_PROXY_QA_PENDING",
        "source": str(source.relative_to(ROOT)),
        "output": str(output.relative_to(ROOT)),
        "record_count": len(rows),
        "training_candidate_count": 0,
        "identity_counts": dict(Counter({identity: sum((row.get("speaker_identity_counts") or {}).get(identity, 0) for row in rows) for identity in ("NEURO_FAMILY", "VEDAL", "UNKNOWN")})),
        "policy": "Family mapping is proxy-only; S/A candidates remain closed until conversation QA and negative open-set coverage are complete.",
    }
    write_json(ROOT / "reports" / "family_mapped_conversation_progress.json", report)
    print(json.dumps({"status": report["status"], "record_count": report["record_count"], "training_candidate_count": 0}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
