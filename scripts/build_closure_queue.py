from __future__ import annotations

import json
from datetime import datetime, timezone

from manifest_tools import ROOT, read_jsonl, write_json


def stage(row: dict) -> str:
    if row.get('conversation_status') == 'done' and row.get('quality_metrics_path'):
        return 'complete'
    if row.get('diarization_status') == 'done' and row.get('asr_status') in {'done', 'source_transcript'}:
        return 'ready_for_conversation'
    asr_path = row.get('asr_path')
    asr_ready = bool(asr_path) and (ROOT / str(asr_path)).exists()
    if row.get('audio_download_status') == 'done' and row.get('asr_status') in {'done', 'source_transcript'} and asr_ready:
        return 'ready_for_diarization'
    if row.get('audio_download_status') == 'done':
        return 'ready_for_asr_or_transcript'
    if row.get('metadata_status') == 'done' and row.get('transcript_available'):
        return 'ready_for_audio_with_transcript'
    if row.get('metadata_status') == 'done':
        return 'ready_for_audio'
    return 'ready_for_metadata'


def priority(row: dict, current_stage: str) -> float:
    seconds = float(row.get('duration') or 3600)
    cost = max(0.25, seconds / 3600)
    score = 0.0
    score += {'ready_for_conversation': 100, 'ready_for_diarization': 85, 'ready_for_asr_or_transcript': 70,
              'ready_for_audio_with_transcript': 62, 'ready_for_audio': 48, 'ready_for_metadata': 20}.get(current_stage, 0)
    score += 18 if row.get('transcript_available') else 0
    score += 12 if len(row.get('participants') or []) >= 2 else 0
    score += 8 if any(x in (row.get('title') or '').lower() for x in ('evil', 'collab', 'vedal', 'neuro')) else 0
    score -= min(30, cost * 3)
    return round(score, 3)


def main() -> None:
    rows = read_jsonl(ROOT / 'manifest' / 'master_video_manifest.jsonl')
    queue = []
    for row in rows:
        current = stage(row)
        if current == 'complete':
            continue
        queue.append({
            'source_id': row.get('source_id'),
            'source_platform': row.get('source_platform'),
            'source_url': row.get('source_url'),
            'title': row.get('title'),
            'duration': row.get('duration'),
            'transcript_available': bool(row.get('transcript_available')),
            'participants': row.get('participants') or [],
            'stage': current,
            'priority': priority(row, current),
            'audio_download_status': row.get('audio_download_status'),
            'asr_status': row.get('asr_status'),
            'diarization_status': row.get('diarization_status'),
        })
    queue.sort(key=lambda x: (-x['priority'], x['duration'] or 999999))
    payload = {
        'schema_version': '0.1.0',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'policy': 'Closure bonus favors near-complete items; long streams are cost-penalized but retained.',
        'queue': queue,
        'stage_counts': {s: sum(1 for x in queue if x['stage'] == s) for s in sorted({x['stage'] for x in queue})},
    }
    write_json(ROOT / 'manifest' / 'closure_queue.json', payload)
    # The daemon is often launched under Windows PowerShell with a GBK
    # stdout encoding.  Queue titles may contain emoji or other characters
    # that GBK cannot represent; keep the on-disk JSON fully Unicode while
    # making the diagnostic stdout encoding-independent.
    print(json.dumps({'queued': len(queue), 'stage_counts': payload['stage_counts'], 'top': queue[:10]}, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
