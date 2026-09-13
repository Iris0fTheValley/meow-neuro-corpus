# M.E.O.W. Neuro Corpus v0.2 Derived Review Artifacts

This dataset repository contains derived, reviewable artifacts from the M.E.O.W.
Neuro/Evil Neuro corpus. Raw video and audio are not redistributed.

## v2.2 status

- `NEURO_FAMILY` means Neuro + Evil Neuro; subtype separation is not required.
- Structural candidates: 17,810.
- Model semantic accepted before the independent guard: 17,312.
- Semantic-verified candidates after the independent context/episode guard: 12,830.
- Recommended splits: 6,705 train, 1,043 validation, 1,043 sealed evaluation.
- Coverage: 64 sources and 27 underlying recording families.
- `training_candidate=true`: 0.
- `READY_FOR_FIRST_SFT`: `NO`.

The dataset is a candidate/review snapshot, not a training release. The
independent guard rejects missing observable triggers, likely hidden-trigger
context, and multi-segment episodes whose later fragment appears to be a new
speech act. `AUTO_TRUSTED`, `PROVISIONAL` and `HUMAN_VERIFIED` remain distinct
provenance levels; no HUMAN_VERIFIED rows are required for this review snapshot.

## Artifact layout

`sft_v2_2_semantic_verified/` contains the structural candidate pool, semantic
judge evidence, semantic-verified candidates, final candidate splits, identity
provenance, recording-overlap evidence, dedup audits, validation reports,
rejection funnels and stratified self-audit samples.

These are derived structured artifacts. Source URLs, timestamps, hashes and
provenance are preserved where available. Rights and redistribution status of
the underlying third-party media remain separate from this derived review set.

## Reproduction

Run the v2.2 scripts from the project root after reconstructing the canonical
timelines and identity mapping. The public GitHub repository contains the
reconstruction and validator code; this repository contains the larger JSONL
artifacts.
