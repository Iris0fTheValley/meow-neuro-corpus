# Interaction semantic closure v2.3

## Scope and authority

This stage begins after canonical timeline reconstruction, identity fusion, and
the v2.2 structural seed builder. It does not infer identity, change identity
thresholds, rewrite transcripts, or consume legacy sliding conversation
windows as a fact source.

The authority classes are explicit:

- **Hard invariants:** raw turn IDs must exist; context and target selections
  must be bounded and contiguous; bad boundaries cannot be crossed; target
  turns and context anchors cannot be reused in the verified pool; artifact
  schema versions must match.
- **Semantic evidence:** primary, strict, and optional adjudication model
  decisions determine observable interaction relation and episode boundary.
- **Risk heuristics:** lexical overlap, trigger/acknowledgement patterns,
  syntactic continuation, gaps, boundary flags, and ASR signals route and
  explain verification. They cannot accept or reject an interaction.
- **Distribution metadata:** small corpus-analysis fields support sampling and
  ablation. They cannot change identity, transcript truth, or quality status.

## Data flow

```text
canonical unique timeline + current identity mapping
    -> v2.2 structural single-turn seeds
    -> bounded context/target evidence envelope
    -> primary semantic judgement
    -> strict independent semantic pass for every proposed acceptance
    -> explicit conflict / ambiguity / rejection / invalid-evidence states
    -> optional targeted adjudication of conflicts
    -> recording-overlap edge aggregation and bridge quarantine
    -> provenance-aware hard dedup
    -> verified interaction pool
    -> recording-family split isolation
    -> selectable train view
```

The structural builder remains the high-recall search-space reducer. Its
continuation heuristic now emits routing evidence only. The semantic verifier
may select one exact context suffix/limited extension and one exact target
prefix from IDs present in the request. It cannot search the timeline freely.

## State machine

| State | Meaning | Pool eligible |
|---|---|---:|
| `PENDING_PRIMARY` | no valid primary evidence | no |
| `PENDING_STRICT` | primary proposed acceptance; a separate strict semantic pass is required | no |
| `VERIFIED` | primary and strict semantic passes agree, or a conflict was explicitly adjudicated as accept | yes |
| `REJECTED` | explicit observable-context rejection, or adjudicated reject | no |
| `AMBIGUOUS` | evidence is insufficient/unclear; no forced binary choice | no |
| `JUDGE_CONFLICT` | primary and strict disagree in outcome or selected boundary | no |
| `MISSING_INVALID_EVIDENCE` | malformed response, illegal turn selection, or incompatible evidence | no |

A primary `PASS` never becomes verified truth by itself. Conflicts are written
to a dedicated artifact and cannot enter the pool.

## Artifacts and versions

Interaction-closure artifacts use schema `2.0.0` and pipeline
`sft-interaction-closure-v2.3.1-2026-09-13`:

- `interaction_judge_requests_v2_3.jsonl`: immutable bounded evidence requests;
- `interaction_judge_{primary,strict,adjudication}_results_v2_3.jsonl`: append/resume stage evidence;
- `interaction_closure_decisions_v2_3.jsonl`: one resolved state per structural seed;
- `interaction_judge_conflicts_v2_3.jsonl` and `interaction_ambiguous_v2_3.jsonl`;
- `recording_overlap_edges_v2_3.jsonl`: pairwise evidence;
- `recording_family_repair_v2_3.json`: aggregate edge support, accepted merges, and quarantined bridges;
- `hard_dedup_audit_v2_3.json`;
- `verified_interaction_pool_v2_3.jsonl`: quality-closed facts before sampling;
- `views/<policy>/`: train view, validation, sealed evaluation, and view manifest.

Legacy v2.2 single-judge results have no schema `2.0.0`. The finalizer rejects
them instead of interpreting them under the new semantics.
The revised single-turn structural seed artifact is also schema `2.0.0` with
pipeline `sft-v2.2-structural-seed-v2-2026-09-13`; the prepare/finalize commands
fail closed on the older multi-turn seed semantics.

## Routine workflow

Run from the repository root:

```powershell
python scripts/build_sft_v2_2_structural_candidates.py
python scripts/run_sft_v2_2_semantic_judge.py prepare
python scripts/run_sft_v2_2_semantic_judge.py judge --stage primary
python scripts/run_sft_v2_2_semantic_judge.py judge --stage strict  # routed to primary proposed accepts
python scripts/finalize_sft_v2_2.py
python scripts/build_sft_v2_2_train_views.py --policy natural_frequency
python scripts/validate_sft_v2_2.py --views datasets/meow_v02_sft_v2_2_semantic_verified/views/natural_frequency
```

`judge` resumes by default. Use `--retry-invalid`, `--sample-ids`, or
`--sample-ids-file` for bounded reruns. `--fresh` is the only option that
discards a stage result file. Adjudication requires explicit sample IDs from
the conflict artifact.

Resume cache identity includes sample ID, request SHA-256, artifact schema,
pipeline, stage, prompt version, and the requested model. A stale but formerly
valid result is automatically rerun and appended; latest compatible evidence
logically supersedes it. Status reports stale/invalid evidence as
`rerun_required`, and strict routing consumes only compatible current primary
results.

Use `python scripts/run_sft_v2_2_semantic_judge.py status` to answer how many
requests remain missing or invalid by stage. The finalizer always writes the
closure state report before refusing an incomplete build.

## Dedup and selection semantics

Hard dedup removes identical interaction provenance, raw target/anchor reuse,
and exact/contained interactions across sources already joined into one
recording family (the clip/reupload case). Text-identical but distinct raw
episodes in one source are retained. Identical or similar behavior in
independent recording families is also preserved and may be annotated with a
`behavior_repeat_cluster` for later sampling.

Recording repair aggregates pairwise overlap support. A normal merge requires
two supporting sample pairs; an exceptionally long/high-similarity single edge
may merge. Component growth beyond the configured bound is quarantined for
audit rather than silently unioned. Eligible strong edges quarantined only for
component size additionally produce `split_exclusion_relations`; their
components must share a partition even though they do not share a recording
family ID. Weak/common-phrase quarantine produces no such constraint.

The verified pool has no train sampling decision. A dataset-level
`split_authority_v2_3.json` persists assignments by underlying canonical
recording member, not ephemeral family hash. Expanded families inherit any
historical sealed member. Merging historical train and sealed lineages creates
`SPLIT_LINEAGE_CONFLICT` and no view is written. All sampling policies share
this authority. The view builder first seals recording-family-disjoint
validation/evaluation partitions, then applies one of
`natural_frequency`, `recording_balanced`, or `high_precision` to train-only
rows. Sealed evaluation never participates in sampling or tuning.

After semantic boundary selection, messages, raw timeline indices, timestamps,
speaker/identity/confidence arrays, transcript/ASR snapshots, and boundary
provenance are all rematerialized from the canonical timeline. Selected target
continuations must independently pass speaker, identity, confidence, ASR,
text-quality, and hard-boundary gates. `SELF_CONTINUATION` remains a diagnostic
relation but is not accepted as an ordinary user-to-assistant training pair.

### Compatibility boundary

Interaction artifacts retain schema `2.0.0`, but pipeline version `v2.3.1`
intentionally makes prior `v2.3` judge results stale so they are rerun rather
than silently reused. Sampling/view artifacts move to schema `2.0.0` and use a
new dataset-level lineage authority. Older policy-local view manifests remain
historical outputs; they are not accepted or migrated implicitly as split
authority. Create `split_authority_v2_3.json` from the current verified pool and
recording-repair artifact before building any new view.
