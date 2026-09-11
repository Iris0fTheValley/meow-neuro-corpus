# Identity validation reproducibility boundary

This stage is audit-only. It does not enter SFT and must not set any row's
`training_candidate` field to `true`.

Run the lightweight audit in this order from `neuro_corpus/scripts`:

```powershell
python build_recording_content_clusters.py
python audit_identity_validation_boundary.py
python build_identity_lineage.py
python make_readiness_report.py
python -m unittest discover -s ../tests -p "test_*.py"
```

The canonical ERes2NetV2 operating threshold is `0.163198`. The value `0.1205`
is retained only as a historical benchmark/calibration operating point. The
current post-fusion S/A count is derived from
`datasets/family_training_candidates_fusion_review.jsonl`; the historical
pre-fusion count remains labeled historical in lineage.

`reports/recording_content_cluster_audit.json` treats recording/content cluster
as the independence unit. Exact audio/content hashes, duplicate metadata,
parent-stream metadata, exact normalized transcript hashes, and a strict
title-duration fallback are recorded as join evidence. Source IDs alone are not
considered sufficient.

`reports/identity_validation_boundary_audit.json` reports positive-validation
coverage and hard-negative stress metrics with row/source/recording-cluster
balance and confidence intervals. Unlabeled challenge acceptance is not called
false-positive rate. If an independent family-positive slice is unavailable,
the report says `INSUFFICIENT_EVIDENCE` and readiness remains closed.

The minimal fixture and fail-closed invariants are in `tests/`. Full corpus,
model, audio, and cache artifacts are intentionally not required for CI.
