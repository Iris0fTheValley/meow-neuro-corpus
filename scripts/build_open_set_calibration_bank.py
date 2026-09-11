from __future__ import annotations

"""Construct a clean, auditable calibration bank from automatic anchors.

Known vocal/music contamination is retained in the challenge set but removed
from the Vedal anchor bank. This prevents the 11.11% Vedal false-positive rate
from being hidden by simply relabeling it.
"""

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from manifest_tools import ROOT, read_jsonl, write_json


KNOWN_VOCAL_CONFOUNDS = {
    "Ak7mzd3L7b0": "candidate windows are lyric/music material; both ECAPA and x-vector produced family false positives",
}


def main() -> None:
    rows = []
    family_path = ROOT / "speaker_refs" / "neuro_family_reference_candidates.jsonl"
    vedal_path = ROOT / "speaker_refs" / "vedal_provisional_reference_candidates.jsonl"
    auto_path = ROOT / "speaker_refs" / "auto_validation_anchors.jsonl"
    for row in read_jsonl(family_path):
        if not (ROOT / str(row.get("clip_path") or "")).exists():
            continue
        rows.append({
            "anchor_id": f"family:{row['candidate_id']}",
            "label": "NEURO_FAMILY",
            "source_id": str(row.get("source_id")),
            "clip_path": row.get("clip_path"),
            "evidence_type": "metadata_acoustic_family_reference",
            "source_row": row,
        })
    excluded = []
    for row in read_jsonl(vedal_path):
        source_id = str(row.get("source_id"))
        if source_id in KNOWN_VOCAL_CONFOUNDS:
            excluded.append({
                "anchor_id": row.get("candidate_id"),
                "source_id": source_id,
                "clip_path": row.get("clip_path"),
                "reason": KNOWN_VOCAL_CONFOUNDS[source_id],
                "retained_as": "challenge_only",
            })
            continue
        if not (ROOT / str(row.get("clip_path") or "")).exists():
            continue
        rows.append({
            "anchor_id": f"vedal:{row['candidate_id']}",
            "label": "VEDAL",
            "source_id": source_id,
            "clip_path": row.get("clip_path"),
            "evidence_type": "metadata_pitch_provisional_cleaned",
            "source_row": row,
        })
    for row in read_jsonl(auto_path):
        if row.get("label") != "OTHER":
            continue
        if not (ROOT / str(row.get("clip_path") or "")).exists():
            continue
        rows.append({
            "anchor_id": f"other:{row['anchor_id']}",
            "label": "OTHER",
            "source_id": str(row.get("source_id")),
            "clip_path": row.get("clip_path"),
            "evidence_type": "metadata_self_identification_other",
            "source_row": row,
        })
    output = ROOT / "speaker_refs" / "open_set_calibration_candidates.jsonl"
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    family_simple = ROOT / "speaker_refs" / "open_set_calibration_family.jsonl"
    vedal_simple = ROOT / "speaker_refs" / "open_set_calibration_vedal.jsonl"
    other_simple = ROOT / "speaker_refs" / "open_set_calibration_other.jsonl"
    family_simple.write_text("".join(json.dumps({"candidate_id": row["anchor_id"], "identity_hint": "NEURO", "source_id": row["source_id"], "clip_path": row["clip_path"]}, ensure_ascii=False) + "\n" for row in rows if row["label"] == "NEURO_FAMILY"), encoding="utf-8")
    vedal_simple.write_text("".join(json.dumps({"candidate_id": row["anchor_id"], "identity_hint": "VEDAL", "source_id": row["source_id"], "clip_path": row["clip_path"]}, ensure_ascii=False) + "\n" for row in rows if row["label"] == "VEDAL"), encoding="utf-8")
    other_simple.write_text("".join(json.dumps({"candidate_id": row["anchor_id"], "identity_hint": "OTHER", "source_id": row["source_id"], "clip_path": row["clip_path"]}, ensure_ascii=False) + "\n" for row in rows if row["label"] == "OTHER"), encoding="utf-8")
    excluded_path = ROOT / "speaker_refs" / "open_set_calibration_excluded_confounds.jsonl"
    excluded_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in excluded), encoding="utf-8")
    report = {
        "schema_version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "CALIBRATION_BANK_READY_AUTO_ANCHOR_NOT_GOLD",
        "candidate_manifest": str(output.relative_to(ROOT)),
        "family_manifest": str(family_simple.relative_to(ROOT)),
        "vedal_manifest": str(vedal_simple.relative_to(ROOT)),
        "other_manifest": str(other_simple.relative_to(ROOT)),
        "excluded_confounds_manifest": str(excluded_path.relative_to(ROOT)),
        "label_counts": dict(Counter(row["label"] for row in rows)),
        "source_counts": dict(Counter(row["label"] + ":" + row["source_id"] for row in rows)),
        "source_counts_by_label": {label: len({row['source_id'] for row in rows if row['label'] == label}) for label in sorted({row['label'] for row in rows})},
        "excluded_confounds_count": len(excluded),
        "excluded_confounds": excluded,
        "policy": "Auto anchors calibrate the verifier only; no row is human gold or a training candidate.",
        "family_policy": "NEURO and EVIL are collapsed into NEURO_FAMILY.",
    }
    write_json(ROOT / "reports" / "open_set_calibration_bank.json", report)
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
