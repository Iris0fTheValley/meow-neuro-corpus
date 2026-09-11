from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone

from manifest_tools import ROOT, read_jsonl, write_json


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def main() -> None:
    ecapa = load("reports/open_set_auto_anchor_calibration.json")
    xvector = load("reports/open_set_auto_anchor_calibration_xvector.json")
    ecapa_rows = {str(row.get("anchor_id")): row for row in ecapa.get("per_clip") or []}
    xvector_rows = {str(row.get("anchor_id")): row for row in xvector.get("per_clip") or []}
    e_threshold = 0.135
    x_threshold = 0.0
    rows = []
    for anchor_id in sorted(set(ecapa_rows) & set(xvector_rows)):
        erow = ecapa_rows[anchor_id]
        xrow = xvector_rows[anchor_id]
        emargin = float(erow.get("family_vs_nontarget_margin") or -1.0)
        xmargin = float(xrow.get("family_vs_nontarget_margin") or -1.0)
        rows.append({
            "anchor_id": anchor_id,
            "source_id": erow.get("source_id"),
            "label": erow.get("label"),
            "ecapa_margin": emargin,
            "xvector_margin": xmargin,
            "ecapa_accept": emargin >= e_threshold,
            "xvector_accept": xmargin >= x_threshold,
            "ensemble_family_accept": emargin >= e_threshold and xmargin >= x_threshold,
        })
    family = [row for row in rows if row["label"] == "NEURO_FAMILY"]
    nontarget = [row for row in rows if row["label"] in {"VEDAL", "OTHER"}]
    challenge_summary = {}
    challenge_e = ROOT / "speaker_refs" / "open_set_challenge_scores_ecapa_calibrated.jsonl"
    challenge_x = ROOT / "speaker_refs" / "open_set_challenge_scores_xvector_calibrated.jsonl"
    if challenge_e.exists() and challenge_x.exists():
        e_rows = {str(row.get("challenge_id")): row for row in read_jsonl(challenge_e)}
        x_rows = {str(row.get("challenge_id")): row for row in read_jsonl(challenge_x)}
        joined = []
        for key in sorted(set(e_rows) & set(x_rows)):
            erow = e_rows[key]
            xrow = x_rows[key]
            ea = bool(float(erow.get("family_vs_nontarget_margin") or -1.0) >= e_threshold)
            xa = bool(float(xrow.get("family_vs_nontarget_margin") or -1.0) >= x_threshold)
            joined.append({**erow, "xvector_family_vs_nontarget_margin": xrow.get("family_vs_nontarget_margin"), "ensemble_family_accept": ea and xa})
        for bucket in sorted({str(row.get("metadata_bucket")) for row in joined}):
            bucket_rows = [row for row in joined if str(row.get("metadata_bucket")) == bucket]
            accepted = sum(bool(row["ensemble_family_accept"]) for row in bucket_rows)
            challenge_summary[bucket] = {"clip_count": len(bucket_rows), "ensemble_potential_family_accept_count": accepted, "ensemble_potential_family_accept_rate": round(accepted / max(1, len(bucket_rows)), 6)}
    report = {
        "schema_version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "ENSEMBLE_AUTO_ANCHOR_CALIBRATION_PROXY_NOT_GOLD",
        "models": [ecapa.get("model"), xvector.get("model")],
        "source_group_holdout": True,
        "ecapa_margin_threshold": e_threshold,
        "xvector_margin_threshold": x_threshold,
        "anchor_count": len(rows),
        "family_count": len(family),
        "family_ensemble_accept_count": sum(bool(row["ensemble_family_accept"]) for row in family),
        "family_recall_proxy": round(sum(bool(row["ensemble_family_accept"]) for row in family) / max(1, len(family)), 6),
        "nontarget_count": len(nontarget),
        "nontarget_ensemble_false_positive_count": sum(bool(row["ensemble_family_accept"]) for row in nontarget),
        "nontarget_ensemble_false_positive_rate": round(sum(bool(row["ensemble_family_accept"]) for row in nontarget) / max(1, len(nontarget)), 6),
        "label_counts": dict(Counter(row["label"] for row in rows)),
        "challenge_summary": challenge_summary,
        "promotion_decision": "DO_NOT_PROMOTE",
        "promotion_reason": "The ensemble is still calibrated on auto anchors, while challenge buckets are unlabeled and sparse. It is a safer mapping proxy, not final gold precision.",
        "per_clip": rows,
    }
    write_json(ROOT / "reports" / "open_set_ensemble_calibration.json", report)
    print(json.dumps({k: report[k] for k in ("status", "anchor_count", "family_recall_proxy", "nontarget_count", "nontarget_ensemble_false_positive_count", "nontarget_ensemble_false_positive_rate", "challenge_summary")}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
