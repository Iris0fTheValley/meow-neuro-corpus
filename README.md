# M.E.O.W. Neuro Corpus v0.2 — identity closure snapshot

This public snapshot contains the reproducibility layer for the M.E.O.W. v0.2 Neuro/Evil Neuro corpus. It intentionally contains code, schemas, lightweight reports, provenance and derived identity indices only. It does not redistribute raw VOD video/audio, model weights, caches, credentials, or third-party source transcripts.

## Current status

- Readiness: `NOT_READY_FOR_SFT`
- Identity scope: `NEURO_FAMILY` versus `NON_TARGET`; Neuro/Evil subtype separation is not required.
- ERes2NetV2 robust-median operating point: threshold `0.163198`; source-disjoint AUTO_TRUSTED family validation `12/16`; hard-negative false positives `0/29`.
- Formal cluster remap: 1,068 clusters; 197 ERes proxy family, 498 guest, 132 known non-target, 241 unknown.
- Multimodal fusion: 145 family recommendations, all with independent ERes support; semantic/metadata-only promotion `0`.
- Chat-TTS rejection layer: reject-only; 11 explicit TTS rejects, 78 quarantine/review, 979 no reject across 1,068 clusters. It cannot promote or recalibrate family identity.
- Candidate grades after identity closure: `S=7,951`, `A=287`; `training_candidate=true` remains `0`.
- `HUMAN_VERIFIED=0` is not a promotion prerequisite. Source-disjoint multi-evidence `AUTO_TRUSTED` anchors are allowed for calibration and validation, with provenance preserved.

## Reproduction boundary

Run from the corpus root with the project environment. The central identity path is:

1. `benchmark_identity_models.py` and `audit_calibration_transfer.py`
2. `remap_clusters_eres2netv2_proxy.py`
3. `build_chat_tts_rejection_layer.py`
4. `run_identity_multimodal_fusion.py`
5. `audit_fusion_audio_consensus.py` and `audit_family_clusters.py`
6. `regrade_candidates_multimodal_fusion.py`
7. `make_readiness_report.py` and `make_checkpoint.py`

All promotion decisions remain conservative until target-vs-non-target precision is validated on broader source-disjoint hard negatives, especially chat/donation TTS and high-pitch guest voices.

## Versioning and provenance

Reports record scorer/model, threshold, aggregation, reference-bank provenance, source splits, and promotion decisions. This snapshot is an identity-closure milestone and is **not ready for training**.

The raw-corpus recovery pass produced `production_v2_recovered` and a stricter `final_sft_candidate_v1` candidate: 5,516 rows across 20 sources and 12 recording clusters, split-disjoint by recording cluster. The candidate passed the independent structural validator with zero exact duplicates, response collisions, suspicious rows, or split leakage. It remains review-only: `READY_FOR_FIRST_SFT=NO` and `training_candidate=true` count `0`.

## SFT Semantic Closure v2

The current published reconstruction milestone is `sft_semantic_closure_v2`. It is built from raw timeline evidence and the verified identity layer rather than inheriting legacy conversation rows. Each sample represents one observable context anchor and one coherent Neuro response episode; target turns cannot be reused across samples, and bad/uncertain intermediate turns form hard boundaries.

- Candidate splits: `4,733` train / `1,558` validation / `1,499` sealed evaluation rows.
- Full post-clean train-cluster set: `8,865` rows in `train_clean_full.jsonl`.
- Coverage: `67` sources and `29` recording clusters; train/validation/sealed clusters are disjoint.
- Independent validator: PASS; anchor duplicates `0`, target-turn reuse `0`, prefix containment `0`, recording leakage `0`.
- Semantic QA: `6,724` accepted and `5,198` repaired candidates; raw reconstruction rejects are retained in the audit report.
- The dataset is still candidate-only: `READY_FOR_FIRST_SFT=NO` and `training_candidate=true` count `0`.

The v2 scripts and lightweight audit reports are published here for review. The structured JSONL artifacts are stored in the companion Hugging Face dataset repository. No raw media, model weights, caches, credentials, or source transcript files are included.

## Incremental SFT reconstruction v2.2

The latest canonical-timeline refresh is published as review-only derived
artifacts in the companion [Hugging Face dataset](https://huggingface.co/datasets/ID-BLUEBERRY/meow-neuro-corpus-v02-artifacts), under `sft_v2_2_semantic_verified/`.

- Structural candidates: `17,810`
- Model semantic accepted before independent guard: `17,312`
- Semantic-verified candidates after the independent context/episode guard: `12,830`
- Recommended train / validation / sealed evaluation: `6,705 / 1,043 / 1,043`
- Coverage: `64` sources and `27` underlying recording families
- Target-turn reuse, prefix ladder, and cross-split content leakage: `0`
- `training_candidate=true`: `0`
- Readiness: `NOT_READY_FOR_SFT`

The independent guard is intentionally conservative: it rejects candidate
contexts with no observable textual trigger and rejects multi-segment episodes
whose later fragment looks like a new speech act. The v2.2 artifacts remain
candidates for external review; they do not authorize SFT promotion.

The v2.2 incremental inventory and reconstruction scripts are in `scripts/`,
with lightweight reports in `reports/sft_v2_2/`. Large JSONL artifacts,
semantic-judge evidence, provenance and split manifests are stored on Hugging
Face. Raw video/audio, media slices, model weights, caches and credentials are
not published.
