from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone

from manifest_tools import ROOT, read_jsonl, write_json


def main() -> None:
    fusion = {f"{row.get('source_id')}:{row.get('cluster')}": row for row in read_jsonl(ROOT / "identity_results" / "identity_mapping_multimodal_fusion_proxy.jsonl")}
    eres = {f"{row.get('source_id')}:{row.get('cluster')}": row for row in read_jsonl(ROOT / "identity_results" / "identity_mapping_eres2netv2_proxy.jsonl")}
    matrix = Counter()
    family_rows = []
    missing = []
    for key, row in fusion.items():
        independent = eres.get(key)
        if independent is None:
            missing.append(key)
            continue
        matrix[f"{row.get('identity')} -> {independent.get('identity')}"] += 1
        if row.get("identity") in {"NEURO_FAMILY_HIGH", "NEURO_FAMILY_MEDIUM"}:
            family_rows.append((key, row, independent))
    report = {
        "schema_version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "FUSION_AUDIO_CONSENSUS_AUDIT_COMPLETE",
        "fusion_mapping_version": "MULTIMODAL_FUSION_PROXY_AUDIO_CONSENSUS_GATED",
        "independent_audio_model": "iic/speech_eres2netv2_sv_zh-cn_16k-common",
        "cluster_count": len(fusion),
        "joined_cluster_count": len(fusion) - len(missing),
        "missing_independent_audio_count": len(missing),
        "fusion_identity_counts": dict(Counter(row.get("identity") for row in fusion.values())),
        "cross_tab_fusion_to_eres2netv2": dict(matrix),
        "fusion_family_count": len(family_rows),
        "fusion_family_supported_by_eres2netv2": sum(row.get("identity") == "NEURO_FAMILY" for _, _, row in family_rows),
        "fusion_family_eres2netv2_unknown": sum(row.get("identity") == "UNKNOWN" for _, _, row in family_rows),
        "fusion_family_eres2netv2_non_target": sum(row.get("identity") in {"NON_TARGET_KNOWN", "NON_TARGET_GUEST"} for _, _, row in family_rows),
        "semantic_or_metadata_only_family_promotion_count": sum(row.get("identity") in {"NEURO_FAMILY_HIGH", "NEURO_FAMILY_MEDIUM"} and independent.get("identity") != "NEURO_FAMILY" for _, row, independent in family_rows),
        "promotion_decision": "DO_NOT_PROMOTE",
        "training_candidate_count": 0,
        "provenance": {
            "fusion_output": "identity_results/identity_mapping_multimodal_fusion_proxy.jsonl",
            "independent_audio_output": "identity_results/identity_mapping_eres2netv2_proxy.jsonl",
            "semantic_evidence_is_auxiliary": True,
            "metadata_and_role_cannot_override_audio_conflict": True,
            "human_gold_is_promotion_blocker": False,
        },
    }
    write_json(ROOT / "reports" / "identity_fusion_audio_consensus_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
