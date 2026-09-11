from __future__ import annotations

import json
import argparse
from collections import Counter
from datetime import datetime, timezone

from manifest_tools import ROOT, write_json


def load_per_clip(path):
    report = json.loads(path.read_text(encoding="utf-8"))
    return {x["candidate_id"]: x for x in report.get("per_clip", [])}, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-manifest", default="speaker_refs/provisional_reference_candidates.jsonl")
    ap.add_argument("--ecapa-report", default="reports/speaker_reference_benchmark.json")
    ap.add_argument("--xvector-report", default="reports/speaker_reference_benchmark_xvector.json")
    ap.add_argument("--output-name", default="reference_candidates_dual_model_agreement.jsonl")
    ap.add_argument("--summary-name", default="reference_bank_dual_model.json")
    args = ap.parse_args()
    candidates_path = ROOT / args.candidate_manifest
    ecapa_path = ROOT / args.ecapa_report
    xvector_path = ROOT / args.xvector_report
    ecapa, ecapa_report = load_per_clip(ecapa_path)
    xvector, xvector_report = load_per_clip(xvector_path)
    candidates = [json.loads(x) for x in candidates_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    selected = []
    decisions = Counter()
    for item in candidates:
        a, b = ecapa.get(item["candidate_id"]), xvector.get(item["candidate_id"])
        if not a or not b:
            decisions["missing_model_result"] += 1
            continue
        label = item["identity_hint"]
        both_match = a["predicted_identity"] == label and b["predicted_identity"] == label
        margins_ok = float(a["margin"]) >= 0.01 and float(b["margin"]) >= 0.001
        if both_match and margins_ok:
            kept = dict(item)
            kept["candidate_status"] = "provisional_dual_model_agreement"
            kept["evidence"] = {
                **item.get("evidence", {}),
                "ecapa_prediction": a["predicted_identity"],
                "ecapa_margin": a["margin"],
                "xvector_prediction": b["predicted_identity"],
                "xvector_margin": b["margin"],
                "dual_model_agreement": True,
                "trusted": False,
            }
            selected.append(kept)
            decisions["selected"] += 1
        elif not both_match:
            decisions["model_disagreement_or_mismatch"] += 1
        else:
            decisions["low_margin"] += 1

    selected.sort(key=lambda x: (x["identity_hint"], x["source_id"], x["start"]))
    out = ROOT / "speaker_refs" / args.output_name
    out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in selected), encoding="utf-8")
    summary = {
        "schema_version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PROVISIONAL_DUAL_MODEL_AGREEMENT_PENDING_NEGATIVE_GOLD",
        "models": [ecapa_report.get("model"), xvector_report.get("model")],
        "selection_policy": "both models predict the metadata hint and ECAPA margin >= 0.01 and x-vector margin >= 0.001",
        "trusted_clip_count": 0,
        "candidate_count": len(selected),
        "by_identity": {identity: sum(x["identity_hint"] == identity for x in selected) for identity in ("NEURO", "EVIL", "VEDAL")},
        "decisions": dict(decisions),
    }
    write_json(ROOT / "speaker_refs" / args.summary_name, summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
