# M.E.O.W. Neuro Corpus v0.2 — v2.3.1 Production Artifacts

This dataset repository contains the frozen v2.3.1 production artifacts for
the M.E.O.W. Neuro/Evil Neuro corpus. It contains derived structured data,
quality evidence, provenance, split authority, validators, and reproducibility
metadata. Raw video/audio, media slices, model weights, credentials, and source
transcripts are not redistributed.

## Current release: v2.3.1

- Pipeline: `sft-interaction-closure-v2.3.1-2026-09-13`
- Context sufficiency: `context-sufficiency-v1-2026-09-16`
- Trajectory reconstruction: `trajectory-reconstruction-v1-2026-09-16`
- `ARCHITECTURE_FROZEN`: `true`
- `READY_FOR_FIRST_SFT`: `false` (training and LoRA sweeps are intentionally not run)
- Identity closure: unchanged and compatible; identity evidence remains separate from interaction semantics.

## Quality-closed corpus

The source verified pool contains 9,784 rows. Selected-context sufficiency
routes them into:

- 726 `SELF_CONTAINED` interactions;
- 9,030 `CONTEXT_INCOMPLETE` rows (1,827 bounded reconstruction requests,
  including 129 trajectory-eligible rows and 1,698 pending candidates);
- 28 `UNSUPPORTED_RELATION` rows, held out in quarantine.

The 129 trajectory rows are materialized only after reconstructed-context
verification and remain provenance-linked to canonical timeline turns. No
diagnostic-only turns are promoted into training context.

## Fixed train views

All views share the dataset-level recording-family split authority. Sampling
does not alter sealed assignment.

| View | Train | Validation | Sealed evaluation | Approx. train tokens |
|---|---:|---:|---:|---:|
| `natural_frequency` | 567 | 50 | 109 | 19,603 |
| `recording_balanced` | 567 | 50 | 109 | 19,603 |
| `high_precision` | 562 | 50 | 109 | 19,476 |
| `interaction_plus_trajectory` | 686 | 50 | 109 | 31,105 |

The four fixed recipes use `max_seq_length=2048`, no packing, response-only
loss, and seed `230916`. They are preparation artifacts, not completed model
training runs.

## Validation and semantic audit

All three interaction-view unified validators, the context-sufficiency
validator, and the trajectory validator pass. Target-turn reuse, prefix ladder,
and cross-split recording leakage are zero. Hard dedup is recording/provenance
aware, so repeated behavior across independent recordings is preserved.

A stratified semantic content audit read 115 final materialized samples across
five strata (24 per stratum). The model returned 108 `PASS` and 7 `FAIL`
verdicts. Direct review of the canonical selected turns and final materialized
text fail-closed one clear context-to-target mismatch; six behavioral or
ambiguous findings are retained with provenance rather than treating Neuro's
nonstandard behavior as an error. The audit used one semantic model and is not
claimed as independent-model verification.

Six strict protocol-invalid rows remain terminal-quarantined and excluded from
all pools and views. Three primary protocol-invalid rows are likewise excluded.

## Artifact layout

The current release is under `sft_v2_3/` and includes:

- context-sufficiency requests, decisions, materialization, and quarantine;
- reconstructed-context verification and trajectory pool/pending artifacts;
- interaction and interaction-plus-trajectory pools;
- recording repair, lineage, hard-dedup, and shared split authority;
- all four fixed views and view manifests;
- validator reports, semantic content audit, final sampling report, statistics,
  fixed recipes, release manifest, and reproducibility metadata.

`release_manifest_v2_3.json` is the checksum authority for the release files.
`production_final_report_v2_3.json` is the machine-readable final status.

## Reproduction

The companion GitHub repository contains the v2.3 reconstruction and validator
code. Reproduction commands and model/prompt metadata are recorded in
`sft_v2_3/reproducibility_v2_3.json`. All final rows retain source identifiers,
recording-family lineage, canonical turn IDs, timestamps, identity provenance,
transcript-quality evidence, semantic verification provenance, and pipeline
versions.

