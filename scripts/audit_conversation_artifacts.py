from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from manifest_tools import ROOT, read_jsonl, write_json


def audit_layer(path: Path, layer: str) -> dict:
    errors: list[dict] = []
    ids: set[str] = set()
    rows = 0
    training_candidates = 0
    unknown_identity = 0
    type_counts: Counter[str] = Counter()
    if not path.exists():
        return {"layer": layer, "path": str(path.relative_to(ROOT)), "status": "MISSING", "rows": 0, "errors": [{"kind": "missing_file"}]}
    for line_no, row in enumerate(read_jsonl(path), 1):
        rows += 1
        conversation_id = str(row.get("conversation_id") or "")
        if not conversation_id:
            errors.append({"line": line_no, "kind": "missing_conversation_id"})
        elif conversation_id in ids:
            errors.append({"line": line_no, "kind": "duplicate_conversation_id", "id": conversation_id})
        ids.add(conversation_id)
        type_counts[str(row.get("window_type") or "missing")] += 1
        if row.get("training_candidate"):
            training_candidates += 1
            errors.append({"line": line_no, "kind": "training_candidate_true"})
        if row.get("identity_confidence") != "unknown":
            errors.append({"line": line_no, "kind": "identity_not_unknown", "value": row.get("identity_confidence")})
        else:
            unknown_identity += 1
        source_ids = list(row.get("source_segment_ids") or [])
        if len(source_ids) != len(set(source_ids)):
            errors.append({"line": line_no, "kind": "duplicate_source_segment_id_within_window"})
        if not source_ids:
            errors.append({"line": line_no, "kind": "empty_source_segment_ids"})
        if layer == "twitchtranscripts":
            for segment in row.get("segments") or []:
                if segment.get("speaker") != "UNKNOWN" or segment.get("speaker_identity") != "UNKNOWN":
                    errors.append({"line": line_no, "kind": "twitch_speaker_not_unknown"})
            if not row.get("provenance", {}).get("raw_preserved"):
                errors.append({"line": line_no, "kind": "raw_preservation_missing"})
    return {
        "layer": layer,
        "path": str(path.relative_to(ROOT)),
        "status": "PASS" if not errors else "FAIL",
        "rows": rows,
        "unique_conversation_ids": len(ids),
        "training_candidate_count": training_candidates,
        "unknown_identity_count": unknown_identity,
        "window_type_counts": dict(type_counts),
        "error_count": len(errors),
        "errors": errors[:100],
    }


def main() -> None:
    layers = [
        (ROOT / "conversations" / "natural_conversations_anonymous.jsonl", "natural_turns"),
        (ROOT / "conversations" / "twitchtranscript_conversations_anonymous.jsonl", "twitchtranscripts"),
    ]
    results = [audit_layer(path, name) for path, name in layers]
    report = {
        "schema_version": "0.1.0",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(x["status"] == "PASS" for x in results) else "FAIL",
        "training_candidate_count": sum(x.get("training_candidate_count", 0) for x in results),
        "layers": results,
        "policy": "Identity and training gates remain closed until trusted gold validation exists.",
    }
    write_json(ROOT / "reports" / "conversation_artifact_qa.json", report)
    print(json.dumps({
        "status": report["status"],
        "training_candidate_count": report["training_candidate_count"],
        "layers": [{"layer": x["layer"], "status": x["status"], "rows": x["rows"], "error_count": x["error_count"]} for x in results],
    }, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
