from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone

from manifest_tools import ROOT, read_jsonl, write_json


def main() -> None:
    rows = list(read_jsonl(ROOT / "identity_results" / "identity_mapping_family_proxy.jsonl"))
    counts = Counter(str(row.get("identity")) for row in rows)
    statuses = Counter(str(row.get("status")) for row in rows)
    report = {
        "schema_version": "0.1.0",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PROXY_MAPPING_COMPLETE_GOLD_PENDING_HARD_NEGATIVES_APPLIED",
        "model": "speechbrain/spkrec-ecapa-voxceleb",
        "reference_bank": "speaker_refs/reference_bank_neuro_family.json",
        "source_count": len({str(row.get("source_id")) for row in rows}),
        "cluster_count": len(rows),
        "mapping_margin_threshold": 0.02,
        "identity_counts": dict(counts),
        "status_counts": dict(statuses),
        "gold_validated": False,
        "training_candidate_count": 0,
        "output": "identity_results/identity_mapping_family_proxy.jsonl",
        "policy": "Known hard-negative sources are quarantined to UNKNOWN; low-margin and unresolved clusters remain UNKNOWN.",
    }
    write_json(ROOT / "reports" / "identity_mapping_family_proxy.json", report)
    print(json.dumps({"status": report["status"], "cluster_count": len(rows), "identity_counts": dict(counts), "status_counts": dict(statuses)}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
