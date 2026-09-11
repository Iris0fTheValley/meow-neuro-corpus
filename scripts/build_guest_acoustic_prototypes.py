from __future__ import annotations

"""Materialize provisional cross-source guest acoustic prototypes."""

import json
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from manifest_tools import ROOT, read_jsonl, write_json


MODEL = "iic/speech_eres2netv2_sv_zh-cn_16k-common"
VERSION = "guest-negative-prototypes-eres2netv2-2026-09-11-v1"


def main() -> None:
    rows = []
    for row in read_jsonl(ROOT / "speaker_refs" / "identity_closure" / "recurring_guest_negative_bank.jsonl"):
        if (ROOT / str(row.get("clip_path") or "")).exists():
            rows.append(row)
    if not rows:
        raise SystemExit("no guest candidate clips")
    verifier = pipeline(task=Tasks.speaker_verification, model=MODEL, device="cpu")
    paths = [str(ROOT / row["clip_path"]) for row in rows]
    matrix = np.asarray(verifier(paths, output_emb=True)["embs"], dtype="float32")
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)
    label_indices = defaultdict(list)
    for index, row in enumerate(rows):
        label_indices[str(row["candidate_label"])].append(index)
    prototypes = {}; prototype_rows = []
    for label, indices in sorted(label_indices.items()):
        sources = sorted({str(rows[index]["source_id"]) for index in indices})
        if len(sources) < 2:
            continue
        vector = matrix[indices].mean(axis=0)
        vector /= max(float(np.linalg.norm(vector)), 1e-9)
        prototypes[label] = vector
        prototype_rows.append({"label": label, "candidate_count": len(indices), "source_count": len(sources), "source_ids": sources, "status": "PROVISIONAL_ACOUSTIC_PROTOTYPE", "trusted": False, "source_disjoint_validation_required": True})
    out_dir = ROOT / "speaker_refs" / "identity_closure"
    npz_path = out_dir / "recurring_guest_negative_prototypes_eres2netv2.npz"
    arrays = {"embeddings": matrix, "candidate_ids": np.asarray([str(row["candidate_id"]) for row in rows], dtype="U256"), "labels": np.asarray([str(row["candidate_label"]) for row in rows], dtype="U128"), "source_ids": np.asarray([str(row["source_id"]) for row in rows], dtype="U128")}
    for label, vector in prototypes.items():
        arrays[f"prototype__{label}"] = vector.astype("float32")
    np.savez_compressed(npz_path, **arrays)
    report = {"schema_version": "0.1.0", "version": VERSION, "created_at": datetime.now(timezone.utc).isoformat(), "status": "PROVISIONAL_ACOUSTIC_PROTOTYPES_READY", "model": MODEL, "candidate_count": len(rows), "prototype_count": len(prototypes), "prototypes": prototype_rows, "artifact": str(npz_path.relative_to(ROOT)), "gold_label_count": 0, "promotion_decision": "DO_NOT_PROMOTE", "policy": "Prototype vectors are derived from metadata/audio candidates only; they are negative evidence for review and must be source-disjoint from any validation source."}
    write_json(out_dir / "recurring_guest_negative_prototypes_report.json", report)
    print(json.dumps({"status": report["status"], "candidate_count": len(rows), "prototype_count": len(prototypes), "labels": sorted(prototypes), "promotion_decision": report["promotion_decision"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
