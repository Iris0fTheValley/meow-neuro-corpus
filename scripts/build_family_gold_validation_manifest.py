from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone

from manifest_tools import ROOT, read_jsonl, write_json


def main() -> None:
    family = list(read_jsonl(ROOT / "speaker_refs" / "neuro_family_reference_candidates.jsonl"))
    vedal = list(read_jsonl(ROOT / "speaker_refs" / "vedal_provisional_reference_candidates.jsonl"))
    rows = []
    for row in family:
        rows.append({
            "validation_id": f"gold_proxy:{row['candidate_id']}",
            "candidate_id": row["candidate_id"],
            "population": "NEURO_FAMILY",
            "gold_label": None,
            "provisional_subtype": row.get("subtype_hint_for_analysis_only"),
            "source_id": row.get("source_id"),
            "clip_path": row.get("clip_path"),
            "validation_status": "PENDING_HUMAN_GOLD_REVIEW",
        })
    for row in vedal:
        rows.append({
            "validation_id": f"gold_proxy:{row['candidate_id']}",
            "candidate_id": row["candidate_id"],
            "population": "VEDAL_NONTARGET_PROXY",
            "gold_label": None,
            "provisional_subtype": "VEDAL",
            "source_id": row.get("source_id"),
            "clip_path": row.get("clip_path"),
            "validation_status": "PENDING_HUMAN_GOLD_REVIEW",
        })
    output = ROOT / "speaker_refs" / "gold_validation_neuro_family.jsonl"
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    report = {
        "schema_version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PENDING_HUMAN_GOLD_REVIEW",
        "target_label_space": ["NEURO_FAMILY", "VEDAL", "OTHER", "UNKNOWN"],
        "population_counts": dict(Counter(row["population"] for row in rows)),
        "source_counts": {"NEURO_FAMILY": len({row["source_id"] for row in family}), "VEDAL_NONTARGET_PROXY": len({row["source_id"] for row in vedal})},
        "gold_labeled_count": 0,
        "other_unknown_gold_count": 0,
        "output": str(output.relative_to(ROOT)),
        "policy": "This is a fixed validation manifest, not a claim that metadata-derived candidates are gold; gold_label remains null until human/external validation.",
    }
    write_json(ROOT / "speaker_refs" / "gold_validation_neuro_family.json", report)
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
