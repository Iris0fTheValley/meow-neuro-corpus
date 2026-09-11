from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import imageio_ffmpeg
import requests

from manifest_tools import ROOT, read_jsonl, write_master, log_event, safe_name


UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131 Safari/537.36'
DOWNLOAD_MIN_FREE_BYTES = 40 * 1024**3


def page_data(session: requests.Session, url: str) -> dict:
    r = session.get(url, headers={'User-Agent': UA, 'Referer': 'https://www.bilibili.com/'}, timeout=30)
    r.raise_for_status()
    marker = 'window.__INITIAL_STATE__='
    pos = r.text.find(marker)
    if pos < 0:
        raise RuntimeError('Bilibili initial state not found')
    data, _ = json.JSONDecoder().raw_decode(r.text[pos + len(marker):])
    return data


def get_audio_url(session: requests.Session, bvid: str, cid: int) -> tuple[str, dict]:
    r = session.get(
        'https://api.bilibili.com/x/player/playurl',
        params={'fnval': 16, 'fnver': 0, 'fourk': 1, 'bvid': bvid, 'cid': cid},
        headers={'User-Agent': UA, 'Referer': f'https://www.bilibili.com/video/{bvid}/'},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    if data.get('code') != 0:
        raise RuntimeError(f"playurl code={data.get('code')} message={data.get('message')}")
    audio = data.get('data', {}).get('dash', {}).get('audio', [])
    if not audio:
        raise RuntimeError('no DASH audio returned')
    chosen = max(audio, key=lambda x: int(x.get('bandwidth') or 0))
    # Signed media URLs are kept in memory only and are never logged or saved.
    return chosen['baseUrl'], {'cid': cid, 'itag': chosen.get('id'), 'codec': chosen.get('codecs'), 'bandwidth': chosen.get('bandwidth')}


def download_part(args: tuple[str, str, Path, dict]) -> Path:
    bvid, referer, target, meta = args
    with requests.get(meta['url'], headers={'User-Agent': UA, 'Referer': referer}, stream=True, timeout=(30, 120)) as r:
        r.raise_for_status()
        with target.open('wb') as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
    if target.stat().st_size < 1024:
        raise RuntimeError(f'part too small: {target.stat().st_size}')
    return target


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-id', required=True)
    args = ap.parse_args()
    free_bytes = shutil.disk_usage(str(ROOT)).free
    if free_bytes < DOWNLOAD_MIN_FREE_BYTES:
        log_event(
            'asset_acquisition_held_disk_low',
            kind='audio',
            source_id=args.source_id,
            free_bytes=free_bytes,
            threshold_bytes=DOWNLOAD_MIN_FREE_BYTES,
            acquisition='bilibili_public_api',
        )
        print(json.dumps({'source_id': args.source_id, 'held': 'disk_low'}, ensure_ascii=False))
        return
    rows = read_jsonl(ROOT / 'manifest' / 'master_video_manifest.jsonl')
    row = next(r for r in rows if r.get('source_id') == args.source_id)
    bvid = args.source_id
    session = requests.Session()
    initial = page_data(session, row['source_url'])
    pages = initial.get('videoData', {}).get('pages', [])
    if not pages:
        raise RuntimeError('no Bilibili pages found')
    tmp = ROOT / 'tmp' / f'bilibili_{safe_name(bvid)}'
    tmp.mkdir(parents=True, exist_ok=True)
    jobs = []
    for idx, page in enumerate(pages, 1):
        url, meta = get_audio_url(session, bvid, int(page['cid']))
        meta['url'] = url
        jobs.append((bvid, row['source_url'], tmp / f'part{idx:02d}.m4s', meta))
    completed = []
    with ThreadPoolExecutor(max_workers=min(3, len(jobs))) as pool:
        futures = [pool.submit(download_part, job) for job in jobs]
        for future in as_completed(futures):
            completed.append(future.result())
    completed.sort()
    output = ROOT / 'raw_audio' / f'{safe_name(bvid)}.m4a'
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), '-y']
    for part in completed:
        cmd += ['-i', str(part)]
    labels = ''.join(f'[{i}:a]' for i in range(len(completed)))
    cmd += ['-filter_complex', f'{labels}concat=n={len(completed)}:v=0:a=1[out]', '-map', '[out]', '-c:a', 'aac', '-b:a', '160k', str(output)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not output.exists() or output.stat().st_size < 1024:
        raise RuntimeError('merged audio missing or too small')
    row.update({
        'audio_download_status': 'done',
        'audio_path': str(output.relative_to(ROOT)),
        'asset_bytes': output.stat().st_size,
        'duration': sum(int(p.get('duration') or 0) for p in pages),
        'bilibili_parts': [{'page': i + 1, 'cid': int(p['cid']), 'part': p.get('part'), 'duration': p.get('duration')} for i, p in enumerate(pages)],
        'processing_status': row.get('processing_status') or 'pending',
    })
    write_master(rows)
    log_event('asset_complete', kind='audio', source_id=bvid, path=row['audio_path'], bytes=row['asset_bytes'], acquisition='bilibili_public_api', parts=len(pages))
    print(json.dumps({'source_id': bvid, 'parts': len(pages), 'audio_path': row['audio_path'], 'bytes': row['asset_bytes'], 'duration': row['duration']}, ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        # Keep a failed public source from being relaunched forever. The
        # dispatcher will only reconsider explicitly queued retryable work;
        # after two informative attempts the source is blocked and an
        # alternate-source search can take over.
        source_id = None
        if '--source-id' in sys.argv:
            idx = sys.argv.index('--source-id')
            if idx + 1 < len(sys.argv):
                source_id = sys.argv[idx + 1]
        if source_id:
            rows = read_jsonl(ROOT / 'manifest' / 'master_video_manifest.jsonl')
            for row in rows:
                if row.get('source_id') == source_id:
                    attempts = int(row.get('audio_download_attempts') or 0) + 1
                    row['audio_download_attempts'] = attempts
                    row['audio_download_status'] = 'blocked' if attempts >= 2 else 'retryable'
                    row['audio_error'] = repr(exc)
                    break
            write_master(rows)
            log_event('audio_blocked' if attempts >= 2 else 'audio_failed', source_id=source_id, attempts=attempts, error=repr(exc), acquisition='bilibili_public_api')
        print(json.dumps({'source_id': source_id, 'error': repr(exc)}, ensure_ascii=False), file=sys.stderr)
        raise
