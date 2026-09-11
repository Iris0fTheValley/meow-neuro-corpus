from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta, timezone

from manifest_tools import ROOT, read_json, read_jsonl, write_json


def main() -> None:
    now = datetime.now(timezone.utc)
    events = []
    path = ROOT / 'logs' / 'pipeline_events.jsonl'
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                item = json.loads(line)
                item['_time'] = datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00'))
                events.append(item)
            except (ValueError, KeyError, json.JSONDecodeError):
                continue
    rows = read_jsonl(ROOT / 'manifest' / 'master_video_manifest.jsonl')
    stages = Counter()
    for row in rows:
        for field in ('metadata_status', 'audio_download_status', 'subtitle_status', 'asr_status', 'diarization_status', 'conversation_status'):
            value = row.get(field)
            if value:
                stages[f'{field}:{value}'] += 1
    windows = {}
    for hours in (1, 3, 6, 12):
        cutoff = now - timedelta(hours=hours)
        recent = [e for e in events if e['_time'] >= cutoff]
        counts = Counter(e.get('event') for e in recent)
        windows[str(hours) + 'h'] = {
            'events': len(recent),
            'asset_complete': counts.get('asset_complete', 0),
            'asr_complete': counts.get('asr_openai_benchmark_complete', 0),
            'diarization_complete': counts.get('diarization_production_complete', 0),
            'fingerprint_runs': counts.get('fingerprint_complete', 0),
            'asset_complete_per_hour': round(counts.get('asset_complete', 0) / hours, 3),
            'asr_complete_per_hour': round(counts.get('asr_openai_benchmark_complete', 0) / hours, 3),
            'diarization_complete_per_hour': round(counts.get('diarization_production_complete', 0) / hours, 3),
        }
    payload = {
        'schema_version': '0.1.0',
        'created_at': now.isoformat(),
        # A sustained rate below two completed processing chains per hour is
        # considered degraded for this corpus; it should trigger scheduling
        # work, not a session exit. Diarization completion is the closest
        # event-level proxy for a closed chain because conversation rebuilds
        # were not logged by the original scripts.
        'throughput_state': 'THROUGHPUT_DEGRADED' if windows['12h']['diarization_complete'] < 24 else 'ACTIVE',
        'current_counts': {
            'videos_discovered': len(rows),
            'audio_acquired': sum(1 for r in rows if r.get('audio_download_status') == 'done'),
            'metadata_complete': sum(1 for r in rows if r.get('metadata_status') == 'done'),
            'asr_complete': sum(1 for r in rows if r.get('asr_status') in {'done', 'source_transcript'}),
            'diarization_complete': sum(1 for r in rows if r.get('diarization_status') == 'done'),
            'conversation_complete': sum(1 for r in rows if r.get('conversation_status') == 'done'),
            'full_pipeline_complete': sum(1 for r in rows if r.get('conversation_status') == 'done' and r.get('quality_metrics_path')),
            'processed_media_hours': round(sum((r.get('duration') or 0) for r in rows if r.get('audio_download_status') == 'done') / 3600, 3),
        },
        'stage_counts': dict(stages),
        'windows': windows,
        'notes': ['Event-based rates are conservative; manifest is the current source of truth.', 'GPU idle fraction requires sampled telemetry and is not inferred from one snapshot.'],
    }
    write_json(ROOT / 'reports' / 'throughput_metrics.json', payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
