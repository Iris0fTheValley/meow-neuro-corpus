from __future__ import annotations

import json
from datetime import datetime, timezone

from manifest_tools import ROOT, read_jsonl, write_json


def main() -> None:
    candidates = {str(row["candidate_id"]): row for row in read_jsonl(ROOT / "speaker_refs" / "all_provisional_reference_candidates.jsonl") if row.get("identity_hint") in {"NEURO", "EVIL"}}
    ecapa = {str(row["candidate_id"]): row for row in json.loads((ROOT / "reports" / "speaker_reference_neuro_family_ecapa_open_set.json").read_text(encoding="utf-8")).get("per_clip", [])}
    xvector = {str(row["candidate_id"]): row for row in json.loads((ROOT / "reports" / "speaker_reference_neuro_family_xvector_open_set.json").read_text(encoding="utf-8")).get("per_clip", [])}
    selected = []
    for candidate_id, row in candidates.items():
        e = ecapa.get(candidate_id, {})
        x = xvector.get(candidate_id, {})
        if float(e.get("family_vs_vedal_margin") or -1) < 0.02 or float(x.get("family_vs_vedal_margin") or -1) < 0.02:
            continue
        selected.append({
            "candidate_id": candidate_id,
            "identity": "NEURO_FAMILY",
            "subtype_hint_for_analysis_only": row.get("identity_hint"),
            "source_id": row.get("source_id"),
            "source_platform": row.get("source_platform"),
            "source_url": row.get("source_url"),
            "title": row.get("title"),
            "clip_path": row.get("clip_path"),
            "start": row.get("start"),
            "end": row.get("end"),
            "duration": row.get("duration"),
            "cluster": row.get("cluster"),
            "dominant_cluster_share": row.get("dominant_cluster_share"),
            "mean_speaker_confidence": row.get("mean_speaker_confidence"),
            "text_preview": row.get("text_preview"),
            "verification": {
                "ecapa_family_vs_vedal_margin": e.get("family_vs_vedal_margin"),
                "xvector_family_vs_vedal_margin": x.get("family_vs_vedal_margin"),
                "dual_model_margin_gate": 0.02,
                "dual_model_gate_passed": True,
            },
            "status": "FAMILY_MAPPING_REFERENCE",
            "gold_validated": False,
        })
    selected.sort(key=lambda row: (str(row["source_id"]), str(row["candidate_id"])))
    output = ROOT / "speaker_refs" / "neuro_family_reference_candidates.jsonl"
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
    bank = {
        "schema_version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reference_bank_version": "neuro-family-2026-09-11-v1",
        "identity": "NEURO_FAMILY",
        "definition": "Neuro + Evil Neuro; subtype_hint is retained for analysis only",
        "status": "NEURO_FAMILY_PROMOTED_FOR_MAPPING_PROXY",
        "promotion_decision": "PROMOTE_NEURO_FAMILY_FOR_MAPPING_WITH_OPEN_SET_GATE",
        "mapping_ready_clip_count": len(selected),
        "gold_validated_trusted_count": 0,
        "source_count": len({row["source_id"] for row in selected}),
        "subtype_counts": {"NEURO": sum(row["subtype_hint_for_analysis_only"] == "NEURO" for row in selected), "EVIL": sum(row["subtype_hint_for_analysis_only"] == "EVIL" for row in selected)},
        "selection_policy": "Both ECAPA and x-vector family-vs-Vedal margin >= 0.02; target family clips are not rejected for Neuro/Evil subtype confusion.",
        "open_set_gate": {"family_margin_threshold": 0.02, "known_nontarget_proxy": "VEDAL", "known_nontarget_false_positive_count_ecapa": 2, "known_nontarget_false_positive_count_xvector": 2},
        "negative_gold_status": "INCOMPLETE_GUEST_TTS_SINGING_GAME_VOICE",
        "training_policy": "Use only after downstream conversation QA; this bank enables family mapping but does not by itself mark windows S/A.",
        "candidate_manifest": str(output.relative_to(ROOT)),
    }
    write_json(ROOT / "speaker_refs" / "reference_bank_neuro_family.json", bank)
    print(json.dumps({"status": bank["status"], "mapping_ready_clip_count": len(selected), "source_count": bank["source_count"], "subtype_counts": bank["subtype_counts"], "gold_validated_trusted_count": 0}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
