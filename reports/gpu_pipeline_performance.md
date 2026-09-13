# GPU/ASR pipeline performance

Created: 2026-09-12T23:26:30Z
Sources: 3; fixed window: 600.0 seconds

Recommended candidate (proxy): `faster-medium.en-float16`
Status: proxy_only_requires_human_readable_review

| Backend | Model load (s) | Window wall time (s) | RTF | Quality proxy |
|---|---:|---:|---:|---|
| openai-medium.en-float16 | 5.61 | 98.15 | 0.0545 | baseline |
| faster-medium.en-float16 | 43.48 | 51.63 | 0.0287 | PASS |
| faster-medium.en-int8_float16 | 2.88 | 58.61 | 0.0326 | PASS |
| faster-distil-large-v3-float16 | 42.05 | 22.36 | 0.0124 | FAIL/REVIEW |

## Notes

- RTF is wall time divided by audio duration; model load is reported separately.
- Each backend retains at least 20 evenly sampled representative segments when available.
- Quality comparison is a screening proxy against the fixed-window OpenAI baseline and requires human/audio review before promotion.
- Old artifacts are not overwritten; production fast artifacts use a versioned filename.

## Throughput estimate

- ASR speedup vs baseline: `{'openai-medium.en-float16': 1.0, 'faster-medium.en-float16': 1.901, 'faster-medium.en-int8_float16': 1.675, 'faster-distil-large-v3-float16': 4.389}`
- Estimated backlog GPU hours: `2.33` (benchmark RTF extrapolation; not a wall-clock guarantee)
- Estimated audio hours/day at sustained GPU saturation: `836.74`

## Diarization benchmark

- Backend: SpeechBrain ECAPA; model load: 0.28s; median fixed-window RTF: 0.0019
- Production path uses bounded random-access reads and batch embeddings; labels remain anonymous until validated speaker mapping.
