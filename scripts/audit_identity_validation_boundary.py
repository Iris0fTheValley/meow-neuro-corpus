from __future__ import annotations

"""Audit independence, validation coverage, and candidate fail-closed state.

This report is intentionally conservative.  It may report insufficient evidence
instead of manufacturing a positive validation metric from proxy mappings.
It never changes identity labels, thresholds, grades, or training flags.
"""

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from identity_validation_utils import bootstrap_mean_interval, wilson_interval
from manifest_tools import ROOT, read_json, read_jsonl, write_json


CURRENT_THRESHOLD = 0.163198
HISTORICAL_BENCHMARK_THRESHOLD = 0.1205
CLUSTER_REPORT = ROOT / "reports" / "recording_content_cluster_audit.json"
CLUSTER_ROWS = ROOT / "reports" / "recording_content_clusters.jsonl"
OUT = ROOT / "reports" / "identity_validation_boundary_audit.json"
SPLIT_OUT = ROOT / "reports" / "identity_validation_split_manifest.jsonl"


def source_ids(path: Path) -> set[str]:
    return {str(row["source_id"]) for row in read_jsonl(path) if row.get("source_id")}


def grouped_rate(rows: list[dict[str, Any]], accept_key: str = "accept") -> dict[str, Any]:
    if not rows:
        return {"row_count": 0, "row_accept_count": 0, "row_accept_rate": None, "row_wilson_95": [None, None], "source_balanced_rate": None, "source_balanced_bootstrap_95": [None, None], "recording_cluster_balanced_rate": None, "recording_cluster_balanced_bootstrap_95": [None, None]}
    row_accept = [1 if row.get(accept_key) else 0 for row in rows]
    by_source: dict[str, list[int]] = defaultdict(list)
    by_cluster: dict[str, list[int]] = defaultdict(list)
    for row, accepted in zip(rows, row_accept):
        by_source[str(row.get("source_id") or "unknown")].append(accepted)
        by_cluster[str(row.get("recording_cluster_id") or "unknown")].append(accepted)
    source_rates = [sum(values) / len(values) for values in by_source.values()]
    cluster_rates = [sum(values) / len(values) for values in by_cluster.values()]
    return {
        "row_count": len(rows),
        "row_accept_count": sum(row_accept),
        "row_accept_rate": round(sum(row_accept) / len(row_accept), 6),
        "row_wilson_95": [round(x, 6) if x is not None else None for x in wilson_interval(sum(row_accept), len(row_accept))],
        "source_count": len(source_rates),
        "source_balanced_rate": round(sum(source_rates) / len(source_rates), 6) if source_rates else None,
        "source_balanced_bootstrap_95": [round(x, 6) if x is not None else None for x in bootstrap_mean_interval(source_rates)],
        "recording_cluster_count": len(cluster_rates),
        "recording_cluster_balanced_rate": round(sum(cluster_rates) / len(cluster_rates), 6) if cluster_rates else None,
        "recording_cluster_balanced_bootstrap_95": [round(x, 6) if x is not None else None for x in bootstrap_mean_interval(cluster_rates)],
    }


def main() -> None:
    cluster_report = read_json(CLUSTER_REPORT, {})
    cluster_map = {
        row["source_id"]: row["recording_cluster_id"]
        for cluster in read_jsonl(CLUSTER_ROWS)
        for row in [{"source_id": source, "recording_cluster_id": cluster["recording_cluster_id"]} for source in cluster.get("source_ids", [])]
    }

    anchor_sources = source_ids(ROOT / "speaker_refs" / "auto_validation_anchors.jsonl")
    calibration_sources = set()
    for name in ("open_set_calibration_family.jsonl", "open_set_calibration_vedal.jsonl", "open_set_calibration_other.jsonl"):
        calibration_sources |= source_ids(ROOT / "speaker_refs" / name)
    protected_clusters = {cluster_map[source] for source in anchor_sources | calibration_sources if source in cluster_map}

    family_refs = read_jsonl(ROOT / "speaker_refs" / "neuro_family_reference_candidates.jsonl")
    independent_family_refs = [
        row for row in family_refs
        if cluster_map.get(str(row.get("source_id"))) not in protected_clusters
    ]
    # This pool is deliberately not promoted to validation: it is proxy-derived
    # and only exists to make the evidence gap machine-visible.
    fusion_rows = read_jsonl(ROOT / "identity_results" / "identity_mapping_multimodal_fusion_proxy.jsonl")
    proxy_family_pool = [
        row for row in fusion_rows
        if str(row.get("identity")) in {"NEURO_FAMILY_HIGH", "NEURO_FAMILY_MEDIUM"}
        and cluster_map.get(str(row.get("source_id"))) not in protected_clusters
    ]

    transfer = read_json(ROOT / "reports" / "identity_calibration_transfer_audit.json", {})
    validation_sources = {str(row.get("source_id")) for row in transfer.get("anchor_comparison", []) if row.get("source_id")}
    validation_clusters = {cluster_map[source] for source in validation_sources if source in cluster_map}

    # Existing stress scores are reported as observed stress acceptance, never
    # as false-positive rate because their labels remain UNLABELED_CHALLENGE_ONLY.
    negative_metrics: dict[str, Any] = {}
    for model_name, path in {
        "ecapa_voxceleb": ROOT / "speaker_refs" / "open_set_challenge_scores_ecapa_calibrated.jsonl",
        "xvector_voxceleb": ROOT / "speaker_refs" / "open_set_challenge_scores_xvector_calibrated.jsonl",
    }.items():
        rows = read_jsonl(path)
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            item = dict(row)
            item["recording_cluster_id"] = cluster_map.get(str(row.get("source_id")))
            item["accept"] = bool(row.get("potential_family_accept"))
            buckets[str(row.get("metadata_bucket") or "unknown")].append(item)
        negative_metrics[model_name] = {
            "score_artifact": str(path.relative_to(ROOT)),
            "label_status": "UNLABELED_CHALLENGE_ONLY",
            "bucket_metrics": {bucket: grouped_rate(bucket_rows) for bucket, bucket_rows in sorted(buckets.items())},
            "false_positive_rate_claim_allowed": False,
        }

    chat_rows = []
    chat_path = ROOT / "speaker_refs" / "identity_closure" / "chat_tts_stress_scores.jsonl"
    for row in read_jsonl(chat_path):
        item = dict(row)
        item["recording_cluster_id"] = cluster_map.get(str(row.get("source_id")))
        item["accept"] = float(row.get("family_margin", -999.0)) >= CURRENT_THRESHOLD
        chat_rows.append(item)
    negative_metrics["eres2netv2_chat_tts_current_threshold_readout"] = {
        "score_artifact": str(chat_path.relative_to(ROOT)),
        "threshold": CURRENT_THRESHOLD,
        "historical_proxy_threshold_not_used": HISTORICAL_BENCHMARK_THRESHOLD,
        "label_status": "UNLABELED_CHALLENGE_ONLY",
        "metrics": grouped_rate(chat_rows),
        "false_positive_rate_claim_allowed": False,
        "interpretation": "stress acceptance only; this is not a gold FPR and cannot promote or recalibrate family.",
    }

    split_audit = cluster_report.get("split_audit", {})
    positive_status = "PASS" if independent_family_refs else "INSUFFICIENT_EVIDENCE"
    candidate_eval = {
        "status": "NOT_EVALUATED_INSUFFICIENT_INDEPENDENT_POSITIVE" if not independent_family_refs else "PENDING_IMPLEMENTATION",
        "positive_validation_rows": len(independent_family_refs),
        "positive_validation_sources": len({str(row.get("source_id")) for row in independent_family_refs}),
        "positive_validation_recording_clusters": len({cluster_map.get(str(row.get("source_id"))) for row in independent_family_refs}),
        "proxy_family_pool_rows_not_used_as_validation": len(proxy_family_pool),
        "policy": "A proxy identity mapping cannot validate itself; no end-to-end family retention metric is claimed until an independent positive slice exists.",
    }

    split_rows = [
        {"split": "anchor_or_calibration_protected", "source_id": source, "recording_cluster_id": cluster_map.get(source), "eligible_for_independent_validation": False}
        for source in sorted(anchor_sources | calibration_sources)
    ]
    split_rows.extend(
        {"split": "transfer_validation_reference", "source_id": source, "recording_cluster_id": cluster_map.get(source), "eligible_for_independent_validation": cluster_map.get(source) not in protected_clusters}
        for source in sorted(validation_sources)
    )
    SPLIT_OUT.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in split_rows), encoding="utf-8")

    report = {
        "schema_version": "0.1.0",
        "artifact_status": "current",
        "canonical_for": "identity_validation_boundary_and_independence_audit",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": "working-tree",
        "current_operating_threshold": CURRENT_THRESHOLD,
        "historical_benchmark_threshold": HISTORICAL_BENCHMARK_THRESHOLD,
        "recording_content_cluster_report": str(CLUSTER_REPORT.relative_to(ROOT)),
        "split_independence": split_audit,
        "family_positive_validation": {
            "status": positive_status,
            "candidate_reference_rows": len(family_refs),
            "independent_rows_after_recording_cluster_exclusion": len(independent_family_refs),
            "independent_source_count": len({str(row.get("source_id")) for row in independent_family_refs}),
            "independent_recording_cluster_count": len({cluster_map.get(str(row.get("source_id"))) for row in independent_family_refs}),
            "row_balanced": None,
            "source_balanced": None,
            "recording_cluster_balanced": None,
            "confidence_interval": None,
            "evidence_sufficiency": False,
            "reason": "All existing provisional family reference clips fall inside anchor/calibration recording clusters; no independent positive gold/AUTO_TRUSTED slice is available.",
        },
        "hard_negative_validation": {
            "status": "STRESS_COVERAGE_EXPANDED_NO_GOLD_FPR",
            "taxonomy": ["chat_tts", "guest", "singing", "game_voice"],
            "metrics": negative_metrics,
            "policy": "Observed stress acceptance is not false-positive rate while gold_label is null; no 0 observed FP claim is made.",
        },
        "candidate_end_to_end_positive_slice": candidate_eval,
        "promotion_guard": {
            "training_candidate_allowed": False,
            "training_candidate_count": 0,
            "threshold_changed": False,
            "identity_mapping_changed": False,
            "human_gold_required": False,
        },
        "outputs": {"split_manifest": str(SPLIT_OUT.relative_to(ROOT))},
    }
    write_json(OUT, report)
    print(json.dumps({
        "status": report["family_positive_validation"]["status"],
        "recording_split_status": split_audit.get("calibration_recording_clusters_disjoint_from_validation"),
        "independent_positive_rows": len(independent_family_refs),
        "proxy_family_pool_not_used": len(proxy_family_pool),
        "training_candidate_count": 0,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
