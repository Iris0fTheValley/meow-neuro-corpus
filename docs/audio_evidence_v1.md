# Speaker-Conditioned Audio Evidence v1

## Scope and compatibility

This is a sidecar architecture. It does not overwrite v2.3 artifacts, reinterpret semantic decisions, change identity fusion, or derive split lineage from new ASR text. Stable joins are `sample_id`, `target_turn_id`, `recording_id`, `canonical_recording_id`, and `recording_family_id`.

Versions: audio-evidence-schema `1.0.0`; audio-reconstruction-pipeline `v1`; enrollment-bank-schema `1.0.0`; role-preserving-materialization `v1`.

## Repository audit

The v2.3 structural and final materialization paths contain two assumptions that are bugs for multi-turn training:

- `sft_semantic_closure_v2_core._context_block`, `build_sft_v2_2_structural_candidates.context_episode`, and final interaction materialization stop when a historical target-family turn appears in context. That makes historical Neuro an incorrect recovery boundary.
- `sft_interaction_semantic_closure` rematerializes exactly one merged `user` message plus one `assistant` message. `validate_sft_v2_2` explicitly requires two messages and treats every assistant message as response text. This cannot represent a masked historical assistant.

The following are limitations rather than changes to semantic truth:

- `training_candidate=false` is source-state metadata while membership is determined by a downstream view. The old artifacts do not identify one field as final authority.
- Model-specific ASR/diarization scripts exist, but there was no common evidence interface, explicit missing-evidence state, or enrollment trust root.
- Existing dedup works on v2.3 two-message rows and therefore cannot distinguish repeated masked assistant history from repeated supervision.

The sidecar leaves those historical files untouched and supplies corrected contracts for new materialization.

## Enrollment trust root

`EnrollmentReference` records the exact source recording and interval, immutable source provenance, verification method and evidence, quality and overlap state, embedding producer/revision, raw audio URI, and SHA-256. `EnrollmentBank.add` accepts only `CONFIRMED_TARGET` references whose identity is `NEURO`, `EVIL_NEURO`, or `NEURO_FAMILY`.

Gold confirmation requires a human/known-official authority domain. Existing-identity confirmation requires all three independent domains: frozen existing identity authority, source provenance, and clean single-speaker evidence. Every evidence item has a structured decision and must be `SUPPORTS_CONFIRMATION`; rejected or inconclusive evidence cannot satisfy confirmation merely by naming an accepted authority domain. The reference must also have a clear speaker boundary, no identity conflict, no hard-negative collision, and `NO_OVERLAP_CONFIRMED`. No new threshold is invented; evidence points to the repository's frozen calibration artifacts and revisions.

The constructor rejects pVAD, new diarization, TSE, new ASR, semantic/persona content, or enrollment-similarity evidence as a confirmation authority. This makes circular bootstrap structurally invalid. Rejected, unverified, guest, unknown, contaminated, and unresolved-overlap references cannot enter the bank. Synthetic references are explicitly test-only, cannot carry a published embedding URI, and require an opt-in on the pipeline.

## Component boundaries

The interfaces in `audio_evidence.adapters` separate target activity, generic diarization, target speaker extraction, ASR, and forced alignment. Named thin boundaries are provided for nomo-pvad, pyannote Community-1, WeSep, REAL-TSE, Qwen3-ASR, and Qwen3-ForcedAligner. Concrete imports and native embeddings stay inside injected callables. TSE adapters do not assume that an ERes2NetV2 embedding matches the pretrained TSE embedding distribution.

Activity is evidence, not identity truth. Generic clusters are never renamed Neuro. The canonical timeline retains upstream `identity_evidence`; existing identity fusion remains the identity authority, and the referenced v2.3 semantic artifact remains semantic authority.

## Planning, routing, and evidence timeline

`AudioWindowPlanner` consumes only supplied interactions. It validates one recording-relative seconds timebase, requires source audio SHA-256 and lineage identifiers, deterministically merges overlapping/adjacent intervals within one recording/source/checksum, and emits reversible sample-to-window mappings. Its window artifact is the direct pipeline input. Every adapter receives an `AudioInput` containing the exact merged interval, so a source URI can never imply whole-recording inference. It has no corpus-wide default.

Adapters may report `WINDOW_LOCAL_SECONDS`, while canonical evidence always uses `SECONDS_FROM_RECORDING_START`. Before activity, diarization, or alignment evidence can reach routing or the timeline, the pipeline deterministically adds the planned window start, records the source timebase and conversion offset, and rejects unknown coordinates or intervals outside the exact bounded window. Thus a local `5–10s` result for a `1800–1860s` window becomes global `1805–1810s`.

`AmbiguityRouter` sends clean audio directly to ASR. Target activity is first filtered by an explicit, versioned threshold supplied in the pVAD adapter's model provenance; missing threshold configuration fails closed. The threshold is preserved in routing/model provenance and participates in the cache key. TSE is selected only when overlap intersects thresholded target activity, simultaneous anonymous speakers intersect that activity, speaker assignment is explicitly ambiguous, background speech is severe, or a smoke test explicitly overrides routing. Different speakers appearing sequentially in one window and low-probability background activity do not trigger TSE. This policy parameter is configuration, not a new identity calibration.

The timeline schema stores old transcript and new ASR separately, plus optional TSE ASR, alignment, activity, anonymous cluster, existing identity evidence, overlap, source interval/audio, model/enrollment provenance, and disagreement flags. It also records `text_resolution_state`, explicit `text_authority`, `resolved_text`, and resolution provenance. New ASR never mutates the old transcript, and unresolved boundary/speaker/major/missing-old-speech disagreements cannot enter materialization.

Text authority is turn-scoped. A turn already carrying a frozen v2.3 semantic-verified target resolves through `resolve_frozen_target_text`: its pre-existing transcript is the supervisory authority and the new ASR hypothesis is independent audio-confirmation evidence. `MATCH` resolves to the (equivalent) new text; `MINOR_TEXT_CHANGE` and `MISSING_NEW_SPEECH` resolve to the old transcript with the new ASR retained as explicit `INDEPENDENT_AUDIO_CONFIRMATION_EVIDENCE`; `MAJOR_TRANSCRIPT_CONFLICT`, `BOUNDARY_CHANGE`, and `SPEAKER_ASSIGNMENT_CHANGE` stay unresolved and quarantine; `MISSING_OLD_SPEECH` stays unresolved under reason `FROZEN_TARGET_OLD_TRANSCRIPT_MISSING` and never spawns a replacement supervision target. A window hypothesis the aligner cannot attribute to the target turn is a boundary conflict, not absent evidence. Every other turn—optional context included—keeps the conservative `automatic_text_resolution` policy, so an unresolved optional context turn is dropped rather than invalidating a resolved target.

Unavailable pVAD/diarization/ASR/aligner is explicit. A hard route without TSE becomes unresolved and is not silently sent through raw ASR. Missing alignment never manufactures word timestamps.

## Role, supervision, and authority contracts

`materialize_role_preserving` accepts a selected sequence of real timeline turns and a final target turn. Roles are preserved. Historical assistant messages have `supervise=false`; exactly the final assistant target has `supervise=true`. No text is summarized, rewritten, merged into a different role, or synthesized.

For trainers without selective assistant masking, `export_prompt_completion` exports every preceding role-preserving message as `prompt_messages` and only the final assistant text as `completion`. Historical assistant loss leakage is zero without inventing a fake user message.

Dedup and validation key target reuse on `(recording_id, target_turn_id)`. Reusing a masked historical assistant turn in later contexts is permitted. Prefix checks compare only supervised targets sharing the same interaction context; independent recordings with the same response remain distinct.

New rows retain `source_state`, `source_sampling_eligibility`, and `legacy_training_candidate`. `legacy_field_semantics=SOURCE_SAMPLING_ELIGIBILITY_ONLY` prevents that old flag from claiming final authority. `final_view_membership` is the sole membership decision and requires `final_view_membership_authority=FINAL_VIEW_MEMBERSHIP`. `supervision_state` is separate.

Every row carries immutable semantic and split authority references/hashes. Production validation requires the expected v2.3 hashes; omitting either produces `NOT_CHECKED` and blocks final pass. New ASR text is not an input to recording lineage or split assignment. Selected timeline recording, canonical recording, and recording family must all equal the interaction authority.

## Cache, checkpoint, and parallel safety

All expensive stages—target activity, diarization, TSE, ASR, and alignment—use a content-addressed key over audio SHA-256, exact interval/timebase, model revision, parameters, and enrollment-bank revision. Derived waveform/text checksums are included where relevant. Cache writes use a per-key lock directory and atomic replacement; workers never append to a shared JSONL. On a miss, a worker acquires the per-key lock, rechecks the cache, and computes only if it is still absent, so concurrent workers execute a given expensive key at most once. Locks contain owner PID/time metadata. An expired lock is recovered only when its owner is provably dead; uncertain or live owners remain fail-closed. Checkpoints use the same recovery rule. Revision, timebase-normalization policy, or routing-threshold changes produce a new key instead of trusting stale output.

`AudioEvidencePipeline.process_window(..., cache_only=True)` never computes an evidence stage: an absent key raises instead of silently degrading to `UNAVAILABLE`, so a cold cache cannot masquerade as an empty audio result. Cache hits no longer rewrite the checkpoint, which is what made a fully cache-warm 4,875-window re-drive cheap; `CheckpointStore` may also be seeded from a prior run's durable progress. Hard dedup and prefix-ladder removal return a single removal ledger (`audio_evidence.deduplication.deduplicate_materialized`) used by `dedup.jsonl`, the run manifest, and the finalizer alike.

## Production entry points

Plan a bounded input file:

```powershell
python scripts/run_audio_evidence_v1.py plan --interactions input.jsonl --output work/window_plan.json --merge-gap 0.25
```

Production code then instantiates `audio_evidence.pipeline.AudioEvidencePipeline` with environment-specific named adapters, an `EnrollmentBank` loaded only from confirmed references, `EvidenceCache`, and `CheckpointStore`. Process planned windows, persist timeline turns, call `materialize_role_preserving`, and gate the result with `validate_artifacts` using frozen semantic and split authority hashes.

The CLI implements bounded planning only. Bounded validation is the Python API `audio_evidence.validation.validate_artifacts`; no CLI validate command is advertised.

The repository production sidecar is executed with `scripts/run_audio_evidence_production.py`. It consumes the frozen verified interaction pool, existing GPU-produced ECAPA/Whisper canonical artifacts, and the confirmed enrollment bank, then writes a versioned run under `datasets/meow_v02_sft_v2_2_semantic_verified/audio_reconstruction_v1/`. `scripts/finalize_audio_evidence_production.py` materializes the four production views and reports.

The raw-waveform path is executed with `scripts/run_audio_evidence_real.py` (`MEOW_REAL_RUN_ID`, `MEOW_REAL_CACHE_RUN_ID`, `MEOW_REAL_CACHE_ONLY`) and finalized with `scripts/finalize_audio_evidence_real.py --run-id RUN`. `scripts/run_audio_evidence_completion.py` drives the cheap half of the chain only—text arbitration, materialization, hard dedup, prefix-ladder dedup, similarity audit, split inheritance, production views, mandatory validator, finalizer—against an existing evidence cache. `scripts/diagnose_audio_completion_gap.py` reproduces the plan, proves per-stage cache-key hits against a prior run's checkpoint, and reports quarantine reasons and target disagreement distribution without importing a model adapter.

There is intentionally no command that scans all interactions. A production Agent must pass an explicit bounded interaction list, pin every model revision, and stop if enrollment confirmation or authority hashes fail.

## Production completion run

The raw-waveform sidecar for the frozen v2.3 verified pool is complete. `audio-reconstruction-v1-real-completion-20260919` re-drove all 4,875 planned windows against the pinned evidence cache of the previous run with zero cache misses and zero expensive model stages executed:

- materialized `6,780`; `6,744` rows after similarity dedup; quarantine `3,003` (`TARGET_TEXT_UNRESOLVED` 2,976, `TARGET_AUDIO_EVIDENCE_UNAVAILABLE` 27)
- ten window failures (nine alignment, one ASR evidence unavailable) remain quarantined; their evidence was never produced by the original inference pass
- final supervised target `1` per row with assistant role preserved; masked historical assistant rows `4,364`
- removal ledger `37` (`prefix_ladder` 1, `similarity_mirror_or_reupload` 36) with 10 independent natural repeats kept
- train / validation / sealed evaluation membership `5,507 / 554 / 719`
- mandatory validator PASS (25/25 gates, 0 errors); target reuse, prefix ladder, historical assistant loss leakage, recording-family split leakage, semantic/split authority mutation, and unverified enrollment usage all `0`

## Verification performed

Tests use standard-library synthetic fixtures and mock model adapters. They cover confirmed/unverified/rejected/guest/contaminated/circular enrollment, timebase and interval failure, deterministic merging, cross-recording isolation, routing, transcript disagreement, frozen-target text authority, cache-only refusal to execute a model stage, dedup ledger completeness, multi-turn role preservation, assistant mask leakage, target uniqueness, train authority, cache invalidation/resume, clean and overlap end-to-end flows, and semantic/split immutability. The production completion run above executed the real raw-waveform sidecar over 4,875 windows by reusing its frozen evidence cache; no model stage was recomputed for it.
