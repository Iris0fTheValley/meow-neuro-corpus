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
