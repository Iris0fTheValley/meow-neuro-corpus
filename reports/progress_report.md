# Neuro-sama / Evil Neuro corpus progress

Updated: 2026-09-12T22:50:22.643400+00:00

## Current state

- Coordinator policy: `CONTINUE_WORKING`
- Discovered manifest rows: **1242**
- Full-pipeline complete (checkpoint): **686**
- Audio acquired: **699**
- Metadata complete: **1234**
- ASR complete: **254**; source transcript: **492**
- Diarization complete: **686**
- Conversation complete: **689**
- Quality metrics present: **686**
- Runnable closure queue: **556**
- Pending/retryable audio acquisition: **61**
- Free disk at checkpoint: **36.26 GiB**

## Source coverage

- `bilibili`: 65
- `internet_archive`: 9
- `preservetube`: 3
- `twitch_archive`: 2
- `twitchtranscripts`: 2
- `youtube`: 1161

## Speaker-separation priority

- Explicit high-priority targets: **16**
- Explicit medium-priority targets: **16**

Recent/high-value targets are kept in the manifest with `speaker_separation_priority`, `speaker_separation_reason`, and `speaker_layout`; these fields prioritize work only and do not assert identity.

## Policy notes

- Media retention is audio-only after acquisition/processing; no raw video files are currently retained.
- New downloads remain held when the disk guard is below its safe threshold.
- Archive/transcript pages that fail informative public retries remain blocked and are not allowed to stall other queues.
- No synthetic dialogue is generated; transcript-only material remains attributed and excluded from high-weight training until audio/speaker validation.
