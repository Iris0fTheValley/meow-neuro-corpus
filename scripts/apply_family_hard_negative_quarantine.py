from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone

from manifest_tools import ROOT, read_jsonl, write_json


HARD_NEGATIVE_SOURCES = {"Ak7mzd3L7b0"}


def main() -> None:
    source = ROOT / "identity_results" / "identity_mapping_family_proxy.jsonl"
    if not source.exists():
        raise SystemExit("family proxy mapping is missing")
    original = list(read_jsonl(source))
    backup = ROOT / "identity_results" / "identity_mapping_family_proxy_pre_hard_negative.jsonl"
    backup.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in original), encoding="utf-8")
    changed = 0
    for row in original:
        if str(row.get("source_id")) not in HARD_NEGATIVE_SOURCES or row.get("identity") != "NEURO_FAMILY":
            continue
        row["identity"] = "UNKNOWN"
        row["identity_confidence"] = "unknown"
        row["status"] = "PROXY_QUARANTINED_HARD_NEGATIVE"
        row["best_score"] = None
        row["second_best_identity"] = None
        row["second_best_score"] = None
        row["margin"] = None
        row["matched_prototype"] = None
        row.setdefault("evidence", {}).update({
            "hard_negative_quarantine": True,
            "hard_negative_reason": "both ECAPA and x-vector misclassified known Vedal clips from this source as Evil/family",
            "gold_validated": False,
        })
        changed += 1
    source.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in original), encoding="utf-8")
    report = {
        "schema_version": "0.1.0",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": "HARD_NEGATIVE_QUARANTINE_APPLIED",
        "sources": sorted(HARD_NEGATIVE_SOURCES),
        "changed_family_to_unknown_count": changed,
        "backup": str(backup.relative_to(ROOT)),
        "policy": "Known non-target false positives are quarantined conservatively; raw audio and original proxy mapping remain preserved.",
    }
    write_json(ROOT / "reports" / "neuro_family_hard_negative_quarantine.json", report)
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
