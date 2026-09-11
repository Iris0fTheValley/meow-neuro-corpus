from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from manifest_tools import ROOT, write_json


def main() -> None:
    candidates_path = ROOT / "speaker_refs" / "provisional_reference_candidates.jsonl"
    benchmark_path = ROOT / "reports" / "speaker_reference_benchmark.json"
    candidates = [json.loads(x) for x in candidates_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    per_clip = {x["candidate_id"]: x for x in benchmark.get("per_clip", [])}
    by_source: dict[str, list[dict]] = defaultdict(list)
    for item in candidates:
        result = per_clip.get(item["candidate_id"])
        if result:
            by_source[str(item["source_id"])].append(result)

    # Keep only source-level consistent clips.  This is a conservative review
    # subset, not an identity promotion: the labels still originate from
    # metadata and the negative gold set is incomplete.
    source_decisions = {}
    keep_ids: set[str] = set()
    for source_id, results in by_source.items():
        agreement = sum(bool(x["correct_against_provisional_label"]) for x in results) / len(results)
        mean_margin = sum(float(x["margin"]) for x in results) / len(results)
        keep_source = agreement >= 0.75 and mean_margin >= 0.015
        source_decisions[source_id] = {
            "clips": len(results),
            "agreement": round(agreement, 6),
            "mean_margin": round(mean_margin, 6),
            "keep_for_next_validation": keep_source,
        }
        if keep_source:
            keep_ids.update(x["candidate_id"] for x in results if x["correct_against_provisional_label"])

    refined = []
    for item in candidates:
        if item["candidate_id"] not in keep_ids:
            continue
        item = dict(item)
        item["candidate_status"] = "provisional_high_consistency_subset"
        item["evidence"] = {**item.get("evidence", {}), "source_consistency_benchmark": "passed_proxy_gate", "trusted": False}
        refined.append(item)
    refined.sort(key=lambda x: (x["identity_hint"], x["source_id"], x["start"]))
    out = ROOT / "speaker_refs" / "reference_candidates_for_next_validation.jsonl"
    out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in refined), encoding="utf-8")
    summary = {
        "schema_version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PROVISIONAL_SUBSET_PENDING_INDEPENDENT_MODEL_AND_NEGATIVE_GOLD",
        "input_benchmark": "reports/speaker_reference_benchmark.json",
        "selection_policy": "source agreement >= 0.75 and mean leave-one-source-out margin >= 0.015; retain only clip-level provisional matches",
        "trusted_clip_count": 0,
        "candidate_count": len(refined),
        "by_identity": {identity: sum(x["identity_hint"] == identity for x in refined) for identity in ("NEURO", "EVIL", "VEDAL")},
        "source_decisions": source_decisions,
    }
    write_json(ROOT / "speaker_refs" / "reference_bank_refined.json", summary)
    print(json.dumps({"candidate_count": len(refined), "by_identity": summary["by_identity"], "kept_sources": [k for k, v in source_decisions.items() if v["keep_for_next_validation"]]}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
