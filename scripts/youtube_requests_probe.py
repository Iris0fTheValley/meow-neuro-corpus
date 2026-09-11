from __future__ import annotations

import json
import re
import sys

import requests


def main() -> None:
    video_id = sys.argv[1]
    session = requests.Session()
    page = session.get(f'https://www.youtube.com/watch?v={video_id}', timeout=30)
    page.raise_for_status()
    key_match = re.search(r'INNERTUBE_API_KEY":"([^"]+)', page.text)
    if not key_match:
        raise RuntimeError('INNERTUBE_API_KEY not found')
    key = key_match.group(1)
    clients = [('WEB', '2.20260910.01.00'), ('ANDROID', '19.09.37'), ('TVHTML5', '7.20260910.18.00')]
    results = []
    for client_name, client_version in clients:
        body = {'context': {'client': {'clientName': client_name, 'clientVersion': client_version}}, 'videoId': video_id}
        response = session.post(f'https://www.youtube.com/youtubei/v1/player?key={key}', json=body, timeout=30)
        response.raise_for_status()
        data = response.json()
        formats = data.get('streamingData', {}).get('adaptiveFormats', [])
        audio = [x for x in formats if 'audio' in x.get('mimeType', '')]
        results.append({
            'client': client_name,
            'playability': data.get('playabilityStatus'),
            'audio_formats': [
                {'itag': x.get('itag'), 'mimeType': x.get('mimeType'), 'contentLength': x.get('contentLength'),
                 'has_url': bool(x.get('url')), 'has_signatureCipher': bool(x.get('signatureCipher'))}
                for x in audio
            ],
        })
    print(json.dumps({'video_id': video_id, 'clients': results}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
