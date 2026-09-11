# Neuro-sama / Evil Neuro corpus progress

Updated: 2026-09-10

The Codex heartbeat `Neuro corpus checkpoint runner` is active at a low-frequency interval. It resumes from `checkpoints/latest.json`, stays quiet when state is unchanged, and is instructed to stop only at the repository-level stopping conditions.

## Discovery checkpoint

- 551 candidate videos are in `manifest/master_video_manifest.jsonl`.
- 547 are YouTube/API records and 4 are Bilibili discovery seeds.
- Source registry includes official channels, Neuro Archiver, unofficial VOD archives, collab archive, Twitch, Bilibili subtitle/recording leads, Library of Ladev, PreserveTube, community indexes, and newly discovered community archive/clip channels.
- Bilibili extraction is currently marked blocked where the public page returned HTTP 412; the queue continues through alternate sources.

## Tool and pipeline checkpoint

- `yt-dlp`, `imageio-ffmpeg`, `faster-whisper`, PyTorch CUDA 12.6, SpeechBrain, OpenAI Whisper, scikit-learn, and soundfile are installed in `.venv-data`.
- RTX 4090 CUDA is available to both OpenAI Whisper and SpeechBrain.
- OpenAI Whisper `medium.en` is the current English ASR candidate; source subtitles remain higher priority when present.
- `faster-whisper` installation is complete, but its first HF model download stalled at an incomplete blob through the current proxy; this is recorded as a non-blocking model-source issue because the OpenAI Whisper Azure model path is working.
- ECAPA embeddings plus anonymous agglomerative clustering are the current diarization benchmark fallback. The benchmark reports separation proxies, not speaker-label accuracy.

## Benchmark assets

- Audio acquired for `1CxTCk0I79w` (Neuro + Evil podcast/Just Chatting), `Ak7mzd3L7b0` (Evil dev stream), `29yeWXdUsqE` (Evil Just Chatting/art review), and `omVt_nBIbDE` (Neuro dev stream).
- One 480p-compatible MKV video is also acquired for `1CxTCk0I79w` for visual/audio traceability.
- ASR benchmark completed for the first three; 10 minutes of `1CxTCk0I79w` have been transcribed with `medium.en`.
- Library of Ladev full public transcript snapshots are preserved for `Ak7mzd3L7b0` and `omVt_nBIbDE`, producing 599 transcript-derived multi-turn windows with raw attribution.
- Anonymous multi-turn conversation windows were reconstructed and retained as raw/rejected-for-training until speaker mapping is validated.
- The current checkpoint contains 551 discovered rows, 206 transcript-available rows, 25 metadata-complete rows, 7 transcript assets, 5 acquired audio assets, 3 ASR benchmark rows, and 0 full-pipeline-complete videos. This is intentionally not being counted as the ten-video checkpoint yet.
- Five additional long-form public transcript assets were acquired from the Minecraft/Subathon archive index, adding 2,532 preserved multi-turn windows; aggregate preserved conversation records now total 3,144.
- A single GPU worker has been launched for a resumable full-audio `medium.en` ASR pass on `1CxTCk0I79w`; no second GPU job is being started concurrently. Its stdout/stderr are retained under `logs/` without credentials.
- That ASR pass completed with 1,529 segments in about 7.7 minutes. Production ECAPA diarization completed with 1,482 usable segments in two anonymous clusters (silhouette ≈0.548); the resulting 93 continuous windows remain excluded from training until reference validation.
- The latest checkpoint now counts 1 full-pipeline-complete source under the project definition: media/ASR, diarization, attempted speaker mapping, conversation reconstruction, and quality metrics are all present. Identity mapping is intentionally still unresolved.
- The sole GPU slot has been handed to the next high-value source, `29yeWXdUsqE` (Evil Just Chatting/art review), for full-audio `medium.en` ASR; diarization will start only after this worker exits.
- `29yeWXdUsqE` full ASR completed with 1,608 segments in about 7.3 minutes; the sole GPU slot is now running its production ECAPA diarization.
- `29yeWXdUsqE` diarization completed with 1,491 usable segments in two anonymous clusters (silhouette ≈0.531), yielding 93 rebuilt continuous windows. The checkpoint now counts 2 full-pipeline-complete sources; both remain training-excluded pending identity reference validation.
- `Ak7mzd3L7b0` now uses a preserved 5,825-segment Library of Ladev transcript adapter as its source transcript; production ECAPA diarization is running against the acquired 4-hour audio as the third bounded GPU job.
- `Ak7mzd3L7b0` diarization completed for 4,638 usable transcript/audio segments in two anonymous clusters (silhouette ≈0.462), yielding 290 continuous windows. The latest checkpoint then counted 3 full-pipeline-complete sources.
- `omVt_nBIbDE` (Neuro dev stream, 25 November 2024) has been promoted to the next source: its preserved 3,748-segment transcript is adapted and production ECAPA diarization is running against its acquired 3-hour audio.
- `omVt_nBIbDE` diarization completed for 3,321 usable transcript/audio segments in two anonymous clusters (silhouette ≈0.406), yielding 208 continuous windows. The checkpoint now counts 4 full-pipeline-complete sources and 3,216 deduplicated/rebuilt preserved conversation records.
- `pQnOBe5O1uE` (Minecraft Hardcore Part 6, 2025-12-07) has been promoted next using its 7,353-segment preserved transcript and acquired long-form audio; its production ECAPA job is the only active GPU task.
- `pQnOBe5O1uE` diarization completed for 7,265 usable segments in two anonymous clusters (silhouette ≈0.298), yielding 454 continuous windows. The latest checkpoint now counts 5 full-pipeline-complete sources; this is the midpoint to the first ten-video checkpoint.
- Network acquisition has started for the next transcript-rich long-form source, `VPOywcw8tXI` (Minecraft Hardcore Part 9); processing remains serialized and will begin only after the audio asset is complete.
- `VPOywcw8tXI` audio acquisition completed (about 576 MiB); its 7,701-segment transcript adapter is ready and the sole GPU worker is now running production ECAPA diarization.
- `VPOywcw8tXI` diarization completed for 7,601 usable segments in two anonymous clusters (silhouette ≈0.291), yielding 475 continuous windows. The latest checkpoint now counts 6 full-pipeline-complete sources.
- Audio acquisition is running for `VpuL3yKYU-w` (Minecraft Hardcore Part 7, 2025-12-07), the next transcript-rich source in the bounded queue.
- `VpuL3yKYU-w` audio acquisition completed (about 514 MiB); its 7,741-segment transcript adapter is ready and production ECAPA diarization is now the sole active GPU task.
- `VpuL3yKYU-w` diarization completed for 7,346 usable segments in two anonymous clusters (silhouette ≈0.284), yielding 459 continuous windows. The latest checkpoint now counts 7 full-pipeline-complete sources.
- A compact audio-fingerprint collision between two different Minecraft segments was investigated: SHA-256 and durations differ, so no duplicate group was retained. Fingerprint grouping now requires near-equal durations; current duplicate group count is zero.
- Audio acquisition has started for `8zyIE0q0STA` (2025 Neuro Subathon Part 65), the eighth transcript-rich source in the first checkpoint batch.
- `8zyIE0q0STA` audio acquisition completed (about 375 MiB); its 8,340-segment transcript adapter is ready and production ECAPA diarization is now the sole active GPU task.
- `8zyIE0q0STA` diarization completed for 7,267 usable segments in two anonymous clusters (silhouette ≈0.417), yielding 454 continuous windows. One cluster contains nearly all segments, so this source is retained but flagged for speaker-balance review. The checkpoint now counts 8 full-pipeline-complete sources.
- Audio acquisition has started for `Uzl3veJKQVY` (2025 Neuro Subathon Part 61), the ninth source in the first ten-video batch.
- `Uzl3veJKQVY` audio acquisition completed (about 380 MiB); its 9,318-segment transcript adapter is ready and production ECAPA diarization is now the sole active GPU task.
- `Uzl3veJKQVY` diarization completed for 7,460 usable segments in two anonymous clusters (silhouette ≈0.387), yielding 467 continuous windows. One cluster dominates, so the source is retained with a speaker-balance warning. The checkpoint now counts 9 full-pipeline-complete sources.
- Audio acquisition has started for `U4voGWGZKos` (2025 Neuro Subathon Part 53), the tenth source in the first formal checkpoint batch.
- `U4voGWGZKos` audio acquisition completed (about 373 MiB). Its search summary lacked a stored body, so the full 7,113-segment Library of Ladev transcript was fetched before starting production ECAPA diarization; the initial premature launch failed cleanly before GPU work.
- `U4voGWGZKos` diarization completed for 6,251 usable segments in two anonymous clusters (silhouette ≈0.360), yielding 391 continuous windows. Metadata was then verified. The first formal 10-video checkpoint is complete: 10/10 full-pipeline sources, 10 audio assets, 26 metadata-complete rows, 3,384 deduplicated/rebuilt conversation records, 0 retained duplicate groups, and about 225.2 GB free disk. All 10 remain identity-conservative and training-excluded pending reference validation.

## Known pending work

- Library of Ladev public transcript search pagination is now enabled (bounded to two pages per term); 206 manifest rows carry transcript availability metadata. Full transcript bodies remain source-attributed snapshots and are not treated as ASR ground truth.
- Historical note superseded by Runtime correction: acquisition and GPU stages are now dispatched continuously from the closure queue; the current active state is recorded in `checkpoints/coordinator_state.json` and the latest checkpoint.
- Add transcript-to-media alignment and raw/normalized subtitle preservation.
- Build reference embeddings for Neuro, Evil, Vedal, and guests; never force Neuro/Evil when evidence is insufficient.
- Historical first-batch objective completed; subsequent 10-video milestones are recorded as the corpus continues beyond 20 full-pipeline sources.

## Formal checkpoint 01 (10 full-pipeline videos)

- Completed sources: `1CxTCk0I79w`, `29yeWXdUsqE`, `Ak7mzd3L7b0`, `omVt_nBIbDE`, `pQnOBe5O1uE`, `VPOywcw8tXI`, `VpuL3yKYU-w`, `8zyIE0q0STA`, `Uzl3veJKQVY`, `U4voGWGZKos`.
- Quality caveat: anonymous ECAPA clusters are a separation proxy, not verified identity labels; Subathon sources with dominant clusters are flagged for balance review.
- Failure/blocked summary: one Bilibili HTTP 412 failure remains explicitly blocked; no credential bypass was attempted. One previously observed compact audio-fingerprint collision was rejected after SHA-256 and duration checks.
- Next batch policy: continue discovery and transcript/API expansion, then select new sources by context coverage and provenance rather than clip popularity.
- Second-batch discovery expanded the manifest from 551 to 664 records and transcript-available rows from 206 to 320 using additional public API queries (`Neuro-sama`, `Evil Neuro-sama`, `Vedal987`); no new source is promoted to processing until metadata/provenance and duplicate checks are applied.
- A bounded metadata pass processed 15 second-batch candidates; metadata-complete rows now total 41, with no new blocked error in this pass. Audio/GPU processing remains paused until the next candidates are ranked.
- One additional high-transcript candidate, `M015UjLTFYA`, was metadata-verified (12-hour Subathon segment); it is queued for later resource-aware selection rather than automatically downloaded.
- Public YouTube search expansion added 100 candidate records across collab, Evil dev-stream, interview, and Just Chatting queries; the manifest now contains 747 rows. The transcript-available count is 318 after merge reconciliation and will be audited before any second-batch promotion.
- Second-batch promotion selected `0skD_GzCLEg` (Neuro Archiver, Dev Stream, 30 October 2023; about 2.4 hours) as the next high-context source; bounded audio acquisition is running.
- `0skD_GzCLEg` audio acquisition completed (about 132 MiB); the sole GPU worker is now running full `medium.en` ASR. Its source has no trusted transcript body, so ASR will remain explicitly marked as local output.
- `0skD_GzCLEg` full ASR completed with 2,611 segments in about 7.5 minutes; production ECAPA diarization is now running as the sole GPU task. Audio asset count is 11.
- `0skD_GzCLEg` diarization completed for 2,164 usable segments in two anonymous clusters (silhouette ≈0.392), yielding 136 continuous windows. The latest checkpoint now counts 11 full-pipeline-complete sources, beginning formal checkpoint batch 02.
- Second-batch acquisition has started for `2QpbuPz8xXM` (Neuro-sama Unofficial VODs, 28 March 2023, about 2.5 hours), selected for source diversity and long-form context.
- `2QpbuPz8xXM` audio acquisition completed (about 122 MiB). Its archive search returned only two transcript matches and no body, so the sole GPU worker is now running full local `medium.en` ASR; no transcript quality is being implied.
- `2QpbuPz8xXM` full ASR completed with 644 segments in about 2.4 minutes; production ECAPA diarization is now running as the sole GPU task.
- `2QpbuPz8xXM` diarization completed for 538 usable segments in two anonymous clusters (silhouette ≈0.305), yielding 34 continuous windows; one cluster dominates and is flagged for balance review. The latest checkpoint now counts 12 full-pipeline-complete sources.
- Second-batch source `-UyVqjxe9ek` (official VOD, Chill Neuro-sama, 25 February) has entered bounded audio acquisition as a contrasting calm-state sample.
- `-UyVqjxe9ek` audio acquisition completed (about 141 MiB); full local `medium.en` ASR is now running as the sole GPU task. The leading-hyphen source ID is handled with explicit equals-form arguments.
- `-UyVqjxe9ek` full ASR completed with 1,309 segments in about 5.3 minutes; production ECAPA diarization is now running as the sole GPU task. The pre-diarization checkpoint records 13 audio assets and 12 completed sources.
- `-UyVqjxe9ek` diarization completed for 1,190 usable segments in two balanced anonymous clusters (silhouette ≈0.511), yielding 75 continuous windows. The latest checkpoint now counts 13 full-pipeline-complete sources; this calm-state source is suitable for later balance analysis.
- A further bounded metadata pass processed 15 second-batch candidates; metadata-complete rows now total 57. No new GPU job was started while candidate ranking continues.
- Second-batch source selection now promotes `-jwDs0n0R3o` (Neuro-Sama Chronicles, impromptu Neuro/collab interaction, about 29 minutes) for efficient multi-speaker context coverage; audio acquisition is running.
- `-jwDs0n0R3o` audio acquisition completed (about 28 MiB); its full local `medium.en` ASR is now running as the sole GPU task.
- `-jwDs0n0R3o` full ASR completed with 423 segments in about 1.5 minutes; production ECAPA diarization is now running as the sole GPU task.
- `-jwDs0n0R3o` diarization completed for 410 usable segments in two anonymous clusters (silhouette ≈0.331), yielding 26 continuous windows. The latest checkpoint now counts 14 full-pipeline-complete sources.
- Second-batch source `-IEJhV_pE4o` (Neuro × Lucy Pyre collab, 13 June 2025, about 2.9 hours) entered bounded audio acquisition to expand guest-interaction coverage.
- `-IEJhV_pE4o` audio acquisition completed (about 162 MiB); full local `medium.en` ASR is now running as the sole GPU task.
- `-IEJhV_pE4o` full ASR completed with 1,829 segments in about 9.2 minutes; production ECAPA diarization is running with three anonymous clusters to reflect the collab/guest setting.
- `-IEJhV_pE4o` diarization completed for 1,687 usable segments across three anonymous clusters (silhouette ≈0.352), yielding 106 continuous windows. The latest checkpoint now counts 15 full-pipeline-complete sources.
- Second-batch source `2HIZ6GzGSN8` (Neuro Sama Collab Archive, Experimental Twins/Cave/BBQ collab, about 2.7 hours) entered bounded audio acquisition for diverse multi-speaker coverage.
- `2HIZ6GzGSN8` audio acquisition completed (about 152 MiB); full local `medium.en` ASR is now running as the sole GPU task.
- `2HIZ6GzGSN8` full ASR completed with 2,002 segments in about 9.1 minutes; production ECAPA diarization is now running with three anonymous clusters for the multi-participant collab.
- `2HIZ6GzGSN8` diarization completed for 1,928 usable segments across three anonymous clusters (silhouette ≈0.459), yielding 121 continuous windows. The latest checkpoint now counts 16 full-pipeline-complete sources.
- Second-batch source `3QlnyZo39jY` (official VOD, Chill Neuro-sama, 7 March) entered bounded audio acquisition to extend calm-state coverage.
- `3QlnyZo39jY` audio acquisition completed (about 167 MiB); full local `medium.en` ASR is now running as the sole GPU task.
- `3QlnyZo39jY` full ASR completed with 847 segments in about 6.1 minutes; production ECAPA diarization is now running as the sole GPU task.
- Concurrency policy is now resource-aware: network acquisition may run in the background while GPU work runs; ASR and diarization may overlap only when measured free VRAM is sufficient, with manifest writes kept single-writer.
- With 20,728 MiB free VRAM observed, the next Evil Neuro source `3f9HLs85Ggs` entered background audio acquisition; no competing manifest writer or GPU job was started.
- `3f9HLs85Ggs` completed full `medium.en` ASR (950 segments) and ECAPA diarization (942 usable segments, two anonymous clusters, silhouette ≈0.608). A stale manifest snapshot briefly omitted its diarization pointer; the preserved diarization artifact was re-registered and 59 continuous windows rebuilt.
- `1cySjIF4DOg` audio was acquired in parallel while `3f9HLs85Ggs` was on GPU; its full `medium.en` ASR (786 segments) and three-cluster ECAPA diarization (silhouette ≈0.505) are complete, yielding 49 windows. The latest checkpoint now counts 19 full-pipeline-complete sources, 19 audio assets, zero duplicate groups, and about 221.8 GB free disk.
- Manifest writes now use a file lock; future parallel workers must still refresh rows before writing so stale snapshots cannot erase newly completed fields.
- User confirmed the pipeline must continue despite transient source outages. Added the public `vedal-subathon-info` event/timestamp index and Tinglo activity index to the source registry as discovery leads; no synthetic dialogue was generated.
- A proxy-assisted retry for `3mw92P7vv7A` also failed with the same SSL EOF pattern, so it remains pending for alternate-source recovery rather than being treated as a global stop.
- User clarified that idle queue state is not a stopping condition. Heartbeat policy was updated: each run must process bounded work or actively resolve blockers/expand public alternates; it may remain quiet only while bounded work is genuinely running.
- Expanded discovery from public event/timestamp pages and Reddit archive indexes; added the Bilibili long-recording lead `BV1m4u96dEen`, the community full-stream VOD lead, and the historical Neurosongsarchive lead. All remain provenance-first candidates pending public access verification.
- PreserveTube’s public channel index was expanded through its linked watch pages; three archive/clip records were added to the manifest with training disabled pending source-context recovery. The global candidate count is now 752.
- New Bilibili public leads `BV1X6X9BuEyg` (associated channel) and `BV1AJ9UBVEaE` (subtitle clip) were added with raw-source attribution and `training_candidate=false`; they are retained for context recovery rather than treated as replacements for full VODs.
- A bounded network audit confirmed the current workstation HTTPS path fails TLS handshakes for YouTube, Library of Ladev, PreserveTube, and direct curl alike, including the configured local proxy; no credentials or browser secrets were accessed. Web-indexed metadata remains usable for discovery, and affected download/API rows stay queued or blocked with alternate-source plans.
- PreserveTube public channel enumeration exposed additional archive-context titles and participant/date metadata; corresponding watch IDs were added as low-weight archive leads, not promoted to training data without full-context recovery.

## Runtime correction and checkpoint 02 continuation (2026-09-10)

- Throughput watchdog classified the prior 12-hour run as `THROUGHPUT_DEGRADED`: before this correction, approximately 19 full-pipeline sources had completed while CPU, network, and GPU capacity were frequently idle. The active scheduler now keeps the main session alive, pauses the periodic heartbeat, and treats GPU/network occupancy as local constraints rather than global blockers.
- A manifest file lock plus refresh-before-write merge was added to prevent stale worker snapshots from erasing completed artifact pointers. Closure queue generation now prioritizes near-complete records and records stage, completion bonus, transcript availability, estimated cost, and duplicate risk.
- Public discovery was expanded across YouTube fan archives, Bilibili subtitle/recording leads, Twitch public status, Library of Ladev API snapshots, GitHub repositories, Archive.vn indexes, and the public Vedal subathon timestamp index. Seven new public video leads were registered with attribution and conservative training flags.
- Bilibili public-page/API acquisition was validated as a usable alternate route despite the YouTube TLS failure. Five Bilibili audio assets are now present; three shorter subtitle/compilation leads were acquired in parallel while the long ASR ran. Signed media URLs were used only in memory and were never persisted or logged.
- `BV1m4u96dEen` (long Bilibili recording) completed local `medium.en` ASR in 446 seconds for 1,027 segments, then production ECAPA diarization in three anonymous clusters (989 usable segments; silhouette about 0.405). It yielded 62 continuous multi-turn windows and completed full-pipeline source 20.
- Current checkpoint 02 state: 756 discovered rows, 25 acquired audio assets, 20 full-pipeline-complete sources, 20 diarization-complete sources, 4,103 preserved conversation records, 81.27 processed media hours, zero retained duplicate groups, and about 221.2 GB free disk. Identity mapping remains conservative and anonymous where reference evidence is insufficient; training candidates remain excluded pending validated speaker mapping.
- At the time of this update, a single GPU diarization worker has already drained for `BV1m4u96dEen`; the next bounded GPU rotation is the newly acquired Bilibili candidates. Network discovery/acquisition and CPU indexing continue independently.
- `BV1DVoQY5EyJ` completed the next short-source rotation: 381 local ASR segments, three anonymous ECAPA clusters (314 usable segments; silhouette about 0.456), and 20 continuous windows. Bilibili public metadata capture was added so acquired Bilibili rows can satisfy the same metadata gate as YouTube rows.
- `BV12W8izqEUy` then completed a fast 152-second rotation with 37 ASR segments, two balanced anonymous clusters (33 usable segments; silhouette about 0.572), and 2 continuous windows. Full-pipeline count is now 22; recent 12-hour event rates have improved to about 1.92 ASR and 1.92 diarization completions/hour, but the watchdog remains degraded until sustained throughput recovers.
- Additional public Bilibili recording/subtitle groups were registered from search-indexed descriptions (`Neuro21烤肉组`, `牛肉烤吧`, `拿铁摩卡卡布奇诺`, and `NeuS放送委员会`). A GitHub public repository search was also executed and persisted; it returned no new repository candidates in this bounded pass.
- `BV18QdCYpESb` completed another rapid rotation: 12 ASR segments, three anonymous ECAPA clusters, and one preserved 12-turn window. Full-pipeline count is now 23; the 12-hour watchdog has recovered to `ACTIVE` under its conservative event threshold, while the historical degraded condition remains documented.
- `BV11r42147oq` completed the next historical rotation with 25 ASR/diarized segments and 2 continuous windows; full-pipeline count reached 24. The one-shot coordinator was then exercised successfully: it detected the empty GPU slot and automatically launched the next acquired candidate instead of waiting for heartbeat polling.
- `BV1X6X9BuEyg` completed its associated-channel short-source rotation with 9 ASR segments, six usable anonymous diarization segments across three clusters, and 1 preserved window. Full-pipeline count is now 25; the coordinator immediately claimed `BV1oWjj6UEMd` for the next ASR pass.
- `BV1oWjj6UEMd` completed with 9 ASR segments, eight usable three-cluster diarization segments, and 1 preserved window. Full-pipeline count is now 26; the CPU closure worker and GPU coordinator ran back-to-back, and the coordinator claimed `BV16LY1eKENj` for the next ASR pass.
- SozAI public transcript pages were expanded into 20 raw transcript snapshots with 1,055 transcript-only continuous windows. They are source-attributed, marked speaker-unvalidated, preserved in raw/cleaned/rejected outputs, and excluded from high-weight training until original audio and speaker evidence are available. This increased transcript-available rows to 392 without replacing any existing diarized records.
- `BV16LY1eKENj` completed the next Bilibili rotation with 115 ASR segments, 109 usable three-cluster diarization segments (silhouette about 0.488), and 7 continuous windows. Full-pipeline count is now 27; the coordinator immediately claimed `BV11L411r7DQ` for the next ASR pass.
- SozAI timestamp parsing was corrected to recover all HTML transcript entries rather than the first entry only. The resulting 1,055 transcript-only windows were rebuilt idempotently and remain excluded from training pending original-audio/speaker validation.
- `BV11L411r7DQ` completed with 69 ASR segments, anonymous diarization, and 4 continuous windows. Full-pipeline count is now 28; the coordinator immediately claimed `BV1LaaCeXEH6` (11-minute historical Bilibili recording) for the next ASR pass.
- `BV1LaaCeXEH6` completed with 131 ASR/diarized segments and 8 continuous windows; the CPU closure worker raised full-pipeline count to 29. The next single-GPU rotation claimed the long `BV17wyFYmEUV` recording while network workers acquired three shorter Bilibili leads in parallel.
- Runtime diagnosis found that system Python38 lacked the corpus media dependencies; dispatchers now prefer the verified `.venv-data` CUDA environment. A missing `filelock` dependency was removed from the critical path with an atomic directory lock and temp-file manifest replacement; status-field merging preserves completed stages under concurrent writes.
- Public discovery added 131 rows from the 2026 Vedal subathon index and 618 bounded Library of Ladev API results. TwitchTranscripts public channel/index pages were captured with 8,323 timestamped transcript entries across two VOD pages and 521 transcript-only context windows; speaker identity remains unknown and training is disabled pending audio alignment.
- Current active checkpoint after this throughput correction: 895 discovered rows, 34 audio assets, 29 full-pipeline-complete sources, 472 transcript-available manifest rows, 5,204 diarized/transcript conversation records before the Twitch snapshot append, zero duplicate groups, and roughly 220 GB free disk. Bilibili download dispatcher is bounded to three network workers and the GPU remains single-job.
- Formal checkpoint 03 (30 full-pipeline videos): `BV17wyFYmEUV` completed the long Evil Neuro karaoke recording with 1,240 ASR segments, 3 anonymous ECAPA clusters, 67 preserved windows, and an explicit low training-priority/singing-heavy provenance note. The latest checkpoint records 897 discovered rows, 34 acquired audio assets, 30 full-pipeline sources, 474 transcript-available rows, 10,963 conversation records, 86.49 processed media hours, and 0 duplicate groups. The next GPU job immediately claimed `BV1AJ9UBVEaE`; Bilibili download and transcript/API work continue in parallel.
- Hugging Face discovery inspected four public Neuro-related dataset leads through the Dataset Viewer API workflow and stored read-only snapshots under `sources/huggingface/`. They remain provenance-only/blocked or unvalidated dataset leads; no Q&A, translated, or synthetic records were mixed into the real-recording corpus.
- `BV1AJ9UBVEaE` and `BV1oh4y1F7jE` completed their short Bilibili rotations (anonymous ECAPA diarization and CPU conversation closure), raising full-pipeline count to 33. The Bilibili dispatcher kept the network lane active and acquired the 10-part `BV1uN411N7KR` recording without competing with the single long GPU job.
- A second SozAI public channel, Neuro-Sama Chronicles, was expanded: 4 transcript pages were fetched, 3 contained timestamped transcript data, and the source-attributed transcript builder added/rebuilt 23 relevant transcript-only sources. The corpus now preserves 15,148 conversation records, with speaker-unvalidated transcript records still excluded from training.
- Ladev transcript-first processing was widened by 36 additional public API bodies/adapters, bringing `asr_status=source_transcript` to 69 and retaining normalized/raw attribution. The forensic report now includes derived technical, emotional-intensity, state-transition, topic-switching, state-carry, lore, language-fingerprint, and visual-context metrics.
- Throughput correction follow-up: the coordinator now owns a single v5 lease, keeps the current long `BV12tstzNEcQ` ASR alive, and runs independent bounded metadata, subtitle, Ladev transcript-body, Bilibili (max three), CPU-closure, and single-GPU lanes. A Windows launcher/process-detection bug that caused duplicate metadata/subtitle workers was diagnosed and guarded; duplicate workers were not allowed to create a second GPU job.
- Public web discovery added six conservative Bilibili provenance leads (`BV1TbH3zUEnu`, `BV151gh6uE5C`, `BV17i4y1q7t1`, `BV1Zk4y147qt`, `BV1tL411i7nD`, `BV1Ld4y1H7wL`) from independent indexed pages. They remain `training_candidate=false` until parent-VOD/context recovery. The resumed bounded Ladev search captured 600 additional public API result references; the manifest reached 908 discovered rows and 41 acquired audio assets, with zero retained duplicate groups.
- YouTube bot-check/429 failures are now capped at an informative retry and then marked `blocked`, while alternate public sources remain eligible. No browser cookies, passwords, session tokens, or authorization headers were accessed.
- Formal checkpoint 04 (40 full-pipeline videos): the CPU closure lane completed seven acquired Bilibili sources, including the long `BV12tstzNEcQ`, after ASR/diarization. The latest checkpoint records 908 discovered rows, 41 acquired audio assets, 40 diarization-complete and 40 full-pipeline sources, 15,220 preserved conversation records / 302,613 turns, 0 duplicate groups, and about 218.7 GB free disk. The single GPU lane has rotated to `BV1uN411N7KR`; metadata, subtitle, Bilibili, Ladev, and CPU lanes remain independent.
- Post-checkpoint closure: `BV1uN411N7KR` completed CPU conversation reconstruction and quality scoring, bringing full-pipeline completion to 41 and preserved records to 15,455 / 307,299 turns. Public source expansion also registered the `Neuro-sama Stream Archive / vedalvods` channel and enumerated the 27 Apr 2023 Vedal + Neuro VOD lead `jTyo7Q46ny4`, which is now queued for metadata/subtitle acquisition.
- Transcript throughput continuation: 12 additional Library of Ladev public API bodies were fetched and adapted with raw/normalized attribution, raising `asr_status=source_transcript` to 81. Their 2,609 source-attributed records were rebuilt into the preserved conversation/rejection outputs; all remain excluded from training until acoustic speaker evidence is available. Aggregate totals are now 18,064 conversation records / 359,246 turns, with no duplicate audio groups.
- Archive-source continuation: public snapshots succeeded for TwitchNoSub, StreamRecorder, VOD Archive, vedal.ai, and the vedalvods YouTube channel. The two Twitch archive pages exposed 25 public VOD URL leads, which were added to `discovery_frontier.json` for later access/transcript checks; they are not treated as acquired media until provenance and accessibility are verified.
- Twitch acquisition continuation: the first two public archive VODs (`vedal987:2868847714`, about 2:52:50, and `vedal987:2866144769`) completed bounded audio-only acquisition without browser authentication material. The manifest now has 42 acquired audio assets; the single GPU coordinator immediately claimed `vedal987:2868847714` for `medium.en` ASR while the second asset remains queued for the next rotation.
- Twitch processing continuation: `vedal987:2868847714` completed `medium.en` ASR (1,423 segments; 638.7 seconds), three-cluster production diarization (1,326 usable segments; silhouette about 0.514), and CPU conversation closure (83 records / 1,326 turns). Full-pipeline completion is now 42 and processed media is about 98.4 hours; `vedal987:2866144769` is concurrently in the single GPU ASR lane.
- Second Twitch closure: `vedal987:2866144769` completed `medium.en` ASR, production diarization, and CPU conversation/quality reconstruction (105 records / 1,673 turns). Full-pipeline completion is now 43; both Twitch archive assets remain provenance-linked, fingerprinted, and free of retained duplicate groups.
- Ladev discovery round 3: the next bounded three-term search added 61 manifest references (970 discovered rows), and 12 new public transcript bodies/adapters contributed 2,659 source-attributed records. Aggregate preserved text now contains 20,911 conversation records / 415,962 turns; these transcript-only records remain speaker-unvalidated and excluded from training until audio alignment is available.
- Ladev transcript continuation: the following 12 public API bodies were fetched and adapted, adding 4,475 source-attributed conversation records. Aggregate preserved text now contains 25,386 records / 505,076 turns and `asr_status=source_transcript` is 105; raw attribution, rejection reasons, and speaker uncertainty remain intact.
- Ladev transcript continuation 2: another 12 public API bodies were fetched, adapted, and reconstructed, adding 3,008 source-attributed records. Aggregate preserved text now contains 28,394 records / 564,918 turns and `asr_status=source_transcript` is 117; all remain excluded from training pending audio/speaker validation.
- Ladev transcript continuation 3: a further 12 public API bodies were fetched, adapted, and reconstructed, adding 3,322 source-attributed records. Aggregate preserved text now contains 31,716 records / 630,924 turns and `asr_status=source_transcript` is 129; all remain excluded from training pending audio/speaker validation.
- Ladev transcript continuation 4: another 12 public API bodies were fetched, adapted, and reconstructed, adding 3,817 source-attributed records. Aggregate preserved text now contains 35,533 records / 706,812 turns and `asr_status=source_transcript` is 141; speaker validation remains a hard gate for training candidates.
### 2026-09-10 throughput correction and closure expansion

- Resumed from `checkpoints/latest.json`; the repository had 970 discovered rows, 43 acquired audio assets, and 43 conservative full-pipeline completions at resume.
- Diagnosed scheduler starvation: daemon lane detection matched task text/self-probing command lines and falsely treated metadata, subtitle, Ladev, CPU, and download workers as active. Tightened matching to real Python worker paths and excluded the PowerShell probe itself.
- Added a bounded YouTube audio dispatcher with two concurrent acquisition workers, while retaining the single-GPU rule and Bilibili three-worker cap. Four additional YouTube audio assets were acquired during this correction; production diarization and CPU conversation closure are flowing automatically.
- The closure queue now contains 42 `ready_for_audio`, 49 `ready_for_audio_with_transcript`, and 834 `ready_for_metadata` candidates; transcript-backed items receive a completion bonus and long VODs remain retained but cost-penalized.
- The latest Ladev body batch added 3,008 attributed transcript records; aggregate preserved corpus totals are being refreshed after the current acoustic closures. No synthetic dialogue or unvalidated speaker identity was added.
- Public discovery was expanded/verified through Neuro-sama Unofficial VODs, Neuro-Sama Collab Archive, N Archiver, PreserveTube, Twitch archive indexes, TwitchTranscripts, SozAI, Library of Ladev, GitHub, Hugging Face Viewer, and Bilibili public search leads.

### Formal checkpoint 05 — 2026-09-10 14:33 UTC

- 50 videos now satisfy the conservative full-pipeline predicate. Current snapshot: 970 manifest rows, 54 acquired audio assets, 52 completed diarization assets, 70 completed acoustic conversation reconstructions, 192 metadata-complete rows, and 238 source-transcript ASR rows.
- Preserved transcript corpus aggregate: 38,614 conversation records and 768,349 turns. Current derived statistics include 5,941,390 response-token observations, 16,332 topic-switch observations, and 7,921.5 state-carry observations; these remain descriptive and do not assert identity or psychology.
- Closure queue after checkpoint: 68 audio-only candidates, 70 transcript-backed audio candidates, 2 acoustic near-complete candidates, 2 diarization-ready candidates, and 778 metadata candidates. The active daemon is consuming these lanes concurrently with one GPU job at a time.
- No duplicate groups were introduced by the latest fingerprint pass; source attribution and anonymous cluster labels remain preserved, and training candidates remain conservative pending validated Neuro/Evil/Vedal mapping.
- Watchdog/resource sample after the correction: throughput state `ACTIVE`, 12-hour acquired-asset rate 4/hour, 1-hour acquired-asset rate 11/hour, GPU 94% utilization with 10.6/24.6 GB VRAM, CPU 59.1%, RAM 63.7%, and 195.7 GB free on J:. This confirms the prior scheduler starvation was not an unavoidable machine-wide wait.

### Formal checkpoint 06 — 2026-09-11 00:31 UTC

- Continuous daemon execution advanced the corpus to 93 conservative full-pipeline videos, 226 acquired audio assets, 93 completed diarization assets, 116 acoustic conversation reconstructions, 963 metadata-complete rows, and 511 Library of Ladev/source-transcript ASR rows.
- Preserved aggregate text now contains 40,271 conversation records and 801,848 turns across raw/cleaned/rejected outputs. Training candidates remain gated because speaker mapping is still anonymous or uncertain; no synthetic filler was introduced.
- Fingerprint review found two possible audio duplicate groups (`dg_00001`, `dg_00002`), both retained as `review_required` rather than deleted. Cross-platform/source attribution remains intact.
- Closure queue remains active with 400 audio-only, 337 transcript-backed audio, 122 diarization-ready, 11 ASR/transcript-ready, and 7 metadata-ready candidates. The daemon currently has concurrent YouTube downloads, CPU closure, and one coordinator-owned GPU lane; metadata/subtitle/Ladev producers continue independently.
- Disk free space is approximately 153.3 GB at this checkpoint. Heartbeat remains paused while the active coordinator lease is healthy.
- Post-checkpoint watchdog fix: one malformed `ready_for_diarization` candidate lacked `asr_path`; the dispatcher now verifies the transcript/ASR file before diarization and routes missing-file cases to the ASR lane. The failed attempt was recorded in its per-job log and no credentials or media were affected.

### Formal checkpoint 07 — 2026-09-11 00:38 UTC

- Continued autonomous throughput advanced the strict full-pipeline count to 97. The latest checkpoint records 244 acquired audio assets, 104 completed diarization assets, 122 acoustic conversation reconstructions, 963 metadata-complete rows, and 511 source-transcript ASR rows.
- The active manifest remains 971 rows; the checkpoint writer reports 970 because concurrent staged writes are still being reconciled by the coordinator. No acquired asset or processed row was discarded; fingerprinting continues to preserve cross-source duplicate relationships.
- Closure queue remains nonempty: 400 audio-only, 323 transcript-backed audio, 123 diarization-ready, 17 ASR/transcript-ready, 3 conversation-ready, and 7 metadata-ready candidates. The single coordinator is concurrently running two YouTube downloads, CPU closure, metadata/subtitle/Ladev producers, and one GPU diarization job.
- Disk free space is approximately 148.6 GB. All speaker identities remain conservative/anonymous or uncertain; no synthetic dialogue is present in the corpus outputs.

### Formal checkpoint 08 — 2026-09-11 00:40 UTC

- The latest checkpoint reached 100 strict full-pipeline videos, with 245 acquired audio assets and 105 completed diarization assets. Metadata remains complete for 963 rows; source-transcript ASR remains preserved for 511 rows.
- Watchdog reports `ACTIVE`, 747.9 processed media hours, a recent 1-hour acquisition rate of 117 assets/hour and 12-hour rate of 19.5 assets/hour. These are event-based rates and remain paired with the manifest as source of truth.
- Five audio sources are retryable and seven metadata sources are blocked after bounded attempts; they do not block other lanes. The archive retains their errors and alternate-source frontier work.
- Disk free space is approximately 148.6 GB. The main coordinator remains alive with no credential access, one-GPU ownership, bounded network concurrency, and active CPU/postprocess lanes.

### Formal checkpoint 09 — 2026-09-11 00:44 UTC

- Latest checkpoint reached 105 strict full-pipeline videos, 260 acquired audio assets, 114 diarization assets, and 129 completed acoustic conversation reconstructions. Metadata remains complete for 963 rows and source-transcript ASR remains preserved for 511 rows.
- Manifest currently contains 971 rows; free disk is approximately 143.9 GB. Fingerprint relationships remain retained for review rather than deleting possible cross-source duplicates.
- Active closure queue remains substantial: 400 audio-only, 307 transcript-backed audio, 126 diarization-ready, 18 ASR/transcript-ready, 7 conversation-ready, and 7 metadata-ready candidates. Downloads, CPU closure, and the single GPU lane continue concurrently under the live coordinator lease.

### Formal checkpoint 10 — 2026-09-11 00:51 UTC

- Latest checkpoint reached 112 strict full-pipeline videos, 274 acquired audio assets, 123 completed diarization assets, and 135 acoustic conversation reconstructions. Metadata remains complete for 963 rows; source-transcript ASR remains preserved for 511 rows.
- The 971-row manifest remains active with approximately 139.9 GB free disk at checkpoint time. Fingerprint review retains two possible audio duplicate groups for manual/source-level review; no asset was silently deleted.
- Closure queue remains active: 400 audio-only, 290 transcript-backed audio, 132 diarization-ready, 19 ASR/transcript-ready, 10 conversation-ready, and 7 metadata-ready. Network, CPU, and the single GPU worker continue independently under the coordinator lease.

### Formal checkpoint 11 — 2026-09-11 00:58 UTC

- Latest checkpoint reached 120 strict full-pipeline videos, 289 acquired audio assets, 133 completed diarization assets, and 143 acoustic conversation reconstructions. Metadata remains complete for 963 rows and source-transcript ASR remains preserved for 511 rows.
- The active manifest contains 971 rows and approximately 135.9 GB free disk. Two fingerprint duplicate groups remain retained for review; no cross-source asset was silently removed.
- Current closure queue: 400 audio-only, 274 transcript-backed audio, 134 diarization-ready, 22 ASR/transcript-ready, 13 conversation-ready, and 7 metadata-ready. The coordinator continues to rotate short/medium items for completion latency while retaining long VODs.

### Formal checkpoint 12 — 2026-09-11 01:04 UTC

- Latest checkpoint reached 124 strict full-pipeline videos, 294 acquired audio assets, 142 completed diarization assets, and 147 acoustic conversation reconstructions. Metadata remains complete for 963 rows and source-transcript ASR remains preserved for 511 rows.
- The 971-row manifest remains active with approximately 132.9 GB free disk. Fingerprinting currently reports three reviewable duplicate groups; all members are retained with provenance for later source-level resolution.
- Closure queue remains active: 400 audio-only, 269 transcript-backed audio, 131 diarization-ready, 22 ASR/transcript-ready, 17 conversation-ready, and 7 metadata-ready. The single GPU lane, two-download lane, CPU closure, and public metadata/subtitle/transcript producers are still running under one coordinator.

### Formal checkpoint 15 — 2026-09-11 01:21 UTC

- Latest checkpoint reached 137 strict full-pipeline videos, 320 acquired audio assets, 165 completed diarization assets, and 160 acoustic conversation reconstructions. Metadata remains complete for 963 rows and source-transcript ASR remains preserved for 511 rows.
- The active 971-row manifest has approximately 123.3 GB free disk. Three fingerprint duplicate groups remain retained for review with source provenance; no content was silently removed.
- Closure queue remains active: 400 audio-only, 244 transcript-backed audio, 127 diarization-ready, 27 ASR/transcript-ready, 28 conversation-ready, and 7 metadata-ready. The coordinator continues bounded YouTube acquisition, CPU closure, metadata/subtitle/Ladev producers, and one GPU diarization lane.

### Formal checkpoint 16 — 2026-09-11 01:26 UTC

- Latest checkpoint reached 140 strict full-pipeline videos, 328 acquired audio assets, 172 completed diarization assets, and 164 acoustic conversation reconstructions. Metadata remains complete for 963 rows and source-transcript ASR remains preserved for 511 rows.
- The 971-row manifest remains active with approximately 120.4 GB free disk. Three fingerprint duplicate groups remain retained for review and source-level resolution.
- Closure queue remains active: 400 audio-only, 238 transcript-backed audio, 129 diarization-ready, 27 ASR/transcript-ready, 29 conversation-ready, and 7 metadata-ready. The single coordinator continues bounded concurrent YouTube acquisition, CPU closure, metadata/subtitle/Ladev producers, and one GPU diarization job.

### Formal checkpoint 17 — 2026-09-11 01:32 UTC

- Latest checkpoint reached 144 strict full-pipeline videos, 335 acquired audio assets, 180 completed diarization assets, and 168 acoustic conversation reconstructions. Metadata remains complete for 963 rows and source-transcript ASR remains preserved for 511 rows.
- The 971-row manifest remains active with approximately 117.5 GB free disk. Three fingerprint duplicate groups remain retained for review with full source attribution.
- Closure queue remains active: 400 audio-only, 230 transcript-backed audio, 128 diarization-ready, 28 ASR/transcript-ready, 33 conversation-ready, and 7 metadata-ready. The live coordinator continues bounded concurrent YouTube acquisition, CPU closure, metadata/subtitle/Ladev producers, and one GPU diarization lane.
- Post-checkpoint retry diagnosis: two public YouTube IDs beginning with `-` were repeatedly rejected by argparse because the download lane passed them as separate option values. The dispatcher now uses `--source-id=<id>`; a direct bounded verification is running for those two public sources, with old failures retained only in per-job logs.

### Formal checkpoint 18 — 2026-09-11 01:41 UTC

- Latest checkpoint reached 150 strict full-pipeline videos, 337 acquired audio assets, and 189 completed diarization assets. The active manifest currently has 971 rows, 190 diarization rows, 174 conversation-complete rows, and 963 metadata-complete rows; the one-row live/checkpoint difference is due to concurrent atomic reconciliation.
- The negative-ID download fix was validated on two Library of Ladev-linked YouTube VODs; both now have durable local audio paths and public-source attribution. The previous argparse errors remain in logs only and are not retried.
- Fingerprint review continues to retain three duplicate groups rather than deleting source variants. Disk free space at checkpoint time was approximately 115.4 GB live / 124.1 GB in the checkpoint snapshot.
- Closure queue remains active and the coordinator continues one GPU diarization job, bounded YouTube downloads, CPU conversation closure, metadata/subtitle retrieval, and Library of Ladev transcript acquisition concurrently.

Post-checkpoint progress at 2026-09-11 01:45 UTC: strict full-pipeline reached 152, with 350 audio assets, 196 diarization assets, and 177 conversation-complete rows. The negative-ID downloader fix is stable; the main daemon continues concurrent GPU/download/CPU processing.

Post-checkpoint progress at 2026-09-11 01:53 UTC: strict full-pipeline reached 156, with 361 audio assets, 202 diarization assets, and 180 conversation-complete rows. The main coordinator remains healthy; the queue is still runnable and no session yield is being taken.

### Formal checkpoint 13 — 2026-09-11 01:08 UTC

- Latest checkpoint reached 128 strict full-pipeline videos, 303 acquired audio assets, 150 completed diarization assets, and 151 acoustic conversation reconstructions. Metadata remains complete for 963 rows and source-transcript ASR remains preserved for 511 rows.
- The 971-row manifest remains active with approximately 129.6 GB free disk. Three fingerprint duplicate groups are retained for review with their members and source attribution intact.
- Closure queue remains active: 400 audio-only, 262 transcript-backed audio, 130 diarization-ready, 23 ASR/transcript-ready, 20 conversation-ready, and 7 metadata-ready. The coordinator continues bounded concurrent downloads, CPU closure, metadata/subtitle/Ladev producers, and one GPU diarization job.

### Formal checkpoint 14 — 2026-09-11 01:15 UTC

- Latest checkpoint reached 132 strict full-pipeline videos, 311 acquired audio assets, 156 completed diarization assets, and 156 acoustic conversation reconstructions. Metadata remains complete for 963 rows and source-transcript ASR remains preserved for 511 rows.
- The 971-row manifest remains active with approximately 126.6 GB free disk. Three fingerprint duplicate groups remain reviewable and fully retained with provenance.
- Closure queue remains active: 400 audio-only, 252 transcript-backed audio, 129 diarization-ready, 26 ASR/transcript-ready, 25 conversation-ready, and 7 metadata-ready. The main coordinator continues bounded concurrent YouTube acquisition, CPU closure, public metadata/subtitle/Ladev work, and one GPU diarization job.
