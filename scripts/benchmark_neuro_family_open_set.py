from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone

from manifest_tools import ROOT, write_json


TARGET_HINTS = {"NEURO", "EVIL"}
THRESHOLDS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    report_path = ROOT / args.report
    source = json.loads(report_path.read_text(encoding="utf-8"))
    per_clip = source.get("per_clip") or []
    rows = []
    for row in per_clip:
        scores = row.get("scores") or {}
        family_score = max(float(scores.get("NEURO") or -1), float(scores.get("EVIL") or -1))
        vedal_score = float(scores.get("VEDAL") or -1)
        provisional = str(row.get("provisional_identity") or "UNKNOWN")
        population = "target_family" if provisional in TARGET_HINTS else "known_nontarget_vedal" if provisional == "VEDAL" else "unlabeled"
        rows.append({
            "candidate_id": row.get("candidate_id"),
            "source_id": row.get("source_id"),
            "provisional_identity": provisional,
            "population": population,
            "family_score": round(family_score, 6),
            "vedal_score": round(vedal_score, 6),
            "family_vs_vedal_margin": round(family_score - vedal_score, 6),
            "model_predicted_identity": row.get("predicted_identity"),
        })
    target = [row for row in rows if row["population"] == "target_family"]
    vedal = [row for row in rows if row["population"] == "known_nontarget_vedal"]
    thresholds = []
    for threshold in THRESHOLDS:
        target_accept = [row for row in target if row["family_vs_vedal_margin"] >= threshold]
        vedal_fp = [row for row in vedal if row["family_vs_vedal_margin"] >= threshold]
        thresholds.append({
            "family_margin_threshold": threshold,
            "target_family_clips": len(target),
            "target_family_accepted": len(target_accept),
            "target_family_recall_proxy": round(len(target_accept) / max(1, len(target)), 6),
            "vedal_nontarget_clips": len(vedal),
            "vedal_false_positive_count": len(vedal_fp),
            "vedal_false_positive_rate": round(len(vedal_fp) / max(1, len(vedal)), 6),
        })
    selected_threshold = 0.02
    accepted = [row for row in rows if row["population"] == "target_family" and row["family_vs_vedal_margin"] >= selected_threshold]
    report = {
        "schema_version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "OPEN_SET_PROXY_BENCHMARK",
        "model": source.get("model"),
        "source_report": args.report,
        "family_definition": "NEURO_FAMILY = provisional NEURO + provisional EVIL",
        "nontarget_definition": "VEDAL is the only safely labeled non-target proxy currently available",
        "cross_source_leave_one_source_out": source.get("cross_source_leave_one_source_out"),
        "clip_count": len(rows),
        "target_family_clip_count": len(target),
        "target_family_source_count": len({row["source_id"] for row in target}),
        "vedal_nontarget_clip_count": len(vedal),
        "vedal_nontarget_source_count": len({row["source_id"] for row in vedal}),
        "threshold_sweep": thresholds,
        "selected_mapping_margin_threshold": selected_threshold,
        "selected_target_family_accept_count": len(accepted),
        "selected_target_family_recall_proxy": round(len(accepted) / max(1, len(target)), 6),
        "selected_vedal_false_positive_count": sum(row["family_vs_vedal_margin"] >= selected_threshold for row in vedal),
        "negative_other": {
            "status": "INCOMPLETE",
            "missing": ["guest", "chat_tts", "male_guest", "female_guest", "game_voice", "singing"],
            "policy": "Do not interpret Vedal-only nontarget proxy as final open-set precision.",
        },
        "promotion_decision": "PROMOTE_NEURO_FAMILY_FOR_MAPPING_WITH_OPEN_SET_GATE",
        "promotion_reason": "Neuro/Evil subtype confusion is intentionally collapsed; family proxy is evaluated against Vedal with a positive margin gate. Non-target gold remains incomplete.",
        "per_clip": rows,
        "confusion_original_model": source.get("confusion", {}),
    }
    write_json(ROOT / args.output, report)
    print(json.dumps({
        "status": report["status"],
        "model": report["model"],
        "target_family_clips": len(target),
        "vedal_nontarget_clips": len(vedal),
        "selected_margin_threshold": selected_threshold,
        "target_family_recall_proxy": report["selected_target_family_recall_proxy"],
        "vedal_false_positive_count": report["selected_vedal_false_positive_count"],
        "vedal_false_positive_rate": round(report["selected_vedal_false_positive_count"] / max(1, len(vedal)), 6),
        "promotion_decision": report["promotion_decision"],
    }, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
