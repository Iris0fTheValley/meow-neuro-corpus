from __future__ import annotations

import html
import re

import requests

from manifest_tools import ROOT, merge_rows, read_jsonl, write_master, log_event


URL = 'https://pasu4.github.io/vedal-subathon-info/2026/overview.html'


def clean(value: str) -> str:
    return re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', '', value))).strip()


def main() -> None:
    response = requests.get(URL, headers={'User-Agent': 'Mozilla/5.0'}, timeout=30)
    response.raise_for_status()
    seeds = []
    for row_html in re.findall(r'<tr>(.*?)</tr>', response.text, re.S | re.I):
        # Only ingest the primary Streams table: its first cell is a dated
        # youtu.be link. Later tables aggregate links by participant/topic and
        # must never overwrite stream metadata in the manifest.
        first_cell = re.search(r'<td[^>]*>(.*?)</td>', row_html, re.S | re.I)
        first_link = re.search(r'href=["\']https://youtu\.be/([A-Za-z0-9_-]{11})["\']\s*>([^<]+)<', first_cell.group(1), re.S | re.I) if first_cell else None
        if not first_link or not re.search(r'\b20\d{2}\b', clean(first_link.group(2))):
            continue
        links = [first_link.group(1)]
        cells = [clean(x) for x in re.findall(r'<td[^>]*>(.*?)</td>', row_html, re.S | re.I)]
        if not links or len(cells) < 4:
            continue
        source_id = links[0]
        participants = [x.strip() for x in cells[3].split(',') if x.strip()]
        seeds.append({
            'source_platform': 'youtube',
            'source_id': source_id,
            'source_url': f'https://www.youtube.com/watch?v={source_id}',
            'title': cells[1],
            'stream_date_if_known': cells[0],
            'participants': participants,
            'uploader': 'vedal-subathon-info index',
            'discovery_method': 'public_event_index',
            'discovery_query': '2026 Content Overview',
            'discovery_source': URL,
            'download_status': 'pending',
            'processing_status': 'pending',
            'language': 'en',
            'subtitle_languages': [],
            'rights_note': 'Public event index lead; acquire from public original/archive and preserve source attribution.',
        })
    write_master(merge_rows(read_jsonl(ROOT / 'manifest' / 'master_video_manifest.jsonl'), seeds))
    log_event('discovery_complete', source_url=URL, count=len(seeds), method='requests_html_index')
    print({'index_rows': len(seeds), 'manifest_seeds_merged': len(seeds)})


if __name__ == '__main__':
    main()
