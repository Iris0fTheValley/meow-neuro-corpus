from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone

from manifest_tools import ROOT, read_jsonl, write_json


def load(path: str) -> dict:
    target = ROOT / path
    return json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}


def main() -> None:
    mapping_path = ROOT / "identity_results" / "identity_mapping_family_proxy.jsonl"
    rows = list(read_jsonl(mapping_path))
    anchors = [row for row in read_jsonl(ROOT / "speaker_refs" / "auto_validation_anchors.jsonl") if row.get("label") == "OTHER"]
    ecapa = {str(row.get("anchor_id")): row for row in (load("reports/open_set_auto_anchor_calibration.json").get("per_clip") or [])}
    xvector = {str(row.get("anchor_id")): row for row in (load("reports/open_set_auto_anchor_calibration_xvector.json").get("per_clip") or [])}
    targets = []
    for anchor in anchors:
        key = f"other:{anchor.get('anchor_id')}"
        erow = ecapa.get(key, {})
        xrow = xvector.get(key, {})
        emargin = float(erow.get("family_vs_nontarget_margin") or -1.0)
        xmargin = float(xrow.get("family_vs_nontarget_margin") or -1.0)
        # Require independent model disagreement in the direction of a strong
        # non-target anchor before quarantining a mapped cluster.
        if emargin >= 0.20 and xmargin < 0.0:
            targets.append({
                "source_id": str(anchor.get("source_id")),
                "cluster": str(anchor.get("cluster")),
                "anchor_id": anchor.get("anchor_id"),
                "text_preview": anchor.get("text_preview"),
                "ecapa_family_vs_nontarget_margin": emargin,
                "xvector_family_vs_nontarget_margin": xmargin,
                "reason": "metadata-confirmed OTHER self-identification is a cross-model family false positive",
            })
    backup = ROOT / "identity_results" / "identity_mapping_family_proxy_pre_auto_anchor_hard_negative.jsonl"
    if not backup.exists():
        backup.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    changed = 0
    target_keys = {(item["source_id"], item["cluster"]) for item in targets}
    output = []
    for row in rows:
        key = (str(row.get("source_id")), str(row.get("cluster")))
        if key in target_keys and row.get("identity") == "NEURO_FAMILY":
            changed += 1
            item = next(x for x in targets if (x["source_id"], x["cluster"]) == key)
            row = {**row, "identity": "UNKNOWN", "identity_confidence": "unknown", "status": "PROXY_QUARANTINED_HARD_NEGATIVE", "matched_prototype": None, "evidence": {**(row.get("evidence") or {}), "auto_anchor_hard_negative": item}}
        output.append(row)
    mapping_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output), encoding="utf-8")
    report = {
        "schema_version": "0.1.0",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": "AUTO_ANCHOR_HARD_NEGATIVE_QUARANTINE_APPLIED" if targets else "AUTO_ANCHOR_HARD_NEGATIVE_NONE",
        "target_count": len(targets),
        "changed_family_to_unknown_count": changed,
        "targets": targets,
        "backup": str(backup.relative_to(ROOT)),
        "policy": "Only metadata-confirmed OTHER self-identification with strong ECAPA family acceptance and X-vector non-target rejection is quarantined; no training promotion is performed.",
    }
    write_json(ROOT / "reports" / "auto_anchor_hard_negative_quarantine.json", report)
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
