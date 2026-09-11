from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from derive_candidate_counts import derive_candidate_counts
from manifest_tools import ROOT, read_json, write_json


OUTPUT = ROOT / "reports" / "identity_closure_lineage.json"


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    derived_candidates = derive_candidate_counts()
    historical_candidate_report = read_json(ROOT / "reports" / "family_training_candidate_progress.json", {})
    current = {
        "recording_content_clusters": "reports/recording_content_cluster_audit.json",
        "identity_validation_boundary": "reports/identity_validation_boundary_audit.json",
        "eres2netv2_cluster_remap": "identity_results/identity_mapping_eres2netv2_proxy.jsonl",
        "multimodal_fusion_mapping": "identity_results/identity_mapping_multimodal_fusion_proxy.jsonl",
        "post_fusion_candidate_regrade": "datasets/family_training_candidates_fusion_review.jsonl",
    }
    historical = {
        "historical_benchmark_threshold": "reports/identity_calibration_transfer_audit.json",
        "historical_pre_fusion_candidate_report": "reports/family_training_candidate_progress.json",
    }
    nodes = {}
    for name, relative in current.items():
        path = ROOT / relative
        nodes[name] = {
            "artifact_status": "current" if path.exists() else "missing",
            "canonical_for": name,
            "path": relative,
            "schema_version": "0.1.0",
            "generated_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat() if path.exists() else None,
            "code_commit": "working-tree",
            "input_artifact_hashes": {"self": sha256(path)},
        }
    for name, relative in historical.items():
        path = ROOT / relative
        nodes[name] = {
            "artifact_status": "historical" if path.exists() else "missing",
            "canonical_for": None,
            "path": relative,
            "schema_version": "0.1.0",
            "generated_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat() if path.exists() else None,
            "code_commit": "historical-unknown",
            "input_artifact_hashes": {"self": sha256(path)},
        }
    report = {
        "schema_version": "0.1.0",
        "artifact_status": "current",
        "canonical_for": "identity_closure_lineage_and_artifact_roles",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": "working-tree",
        "thresholds": {
            "current_operating": {"value": 0.163198, "status": "current", "source": "identity_results/identity_mapping_eres2netv2_proxy.jsonl"},
            "historical_benchmark": {"value": 0.1205, "status": "historical", "source": "reports/identity_calibration_transfer_audit.json"},
        },
        "candidate_counts": {
            "current_post_fusion_s_a": {"value": derived_candidates["s_a_review_candidate_count"], "status": "derived_from_row_level_artifact", "source": derived_candidates["source"]},
            "historical_pre_fusion_s_a": {"value": historical_candidate_report.get("s_a_review_candidate_count"), "status": "historical", "source": "reports/family_training_candidate_progress.json"},
            "training_candidate": {"value": derived_candidates["training_candidate_count"], "status": "closed_by_invariant"},
        },
        "nodes": nodes,
        "policy": "Historical artifacts are retained and explicitly non-canonical; downstream readiness must use current nodes and row-level derivation.",
    }
    write_json(OUTPUT, report)
    print({"status": report["artifact_status"], "output": str(OUTPUT.relative_to(ROOT))})


if __name__ == "__main__":
    main()
