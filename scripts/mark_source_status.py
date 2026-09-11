from __future__ import annotations

import argparse

from manifest_tools import ROOT, read_jsonl, write_master


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-id', required=True)
    ap.add_argument('--field', required=True)
    ap.add_argument('--value', required=True)
    ap.add_argument('--error', default='')
    args = ap.parse_args()
    rows = read_jsonl(ROOT / 'manifest' / 'master_video_manifest.jsonl')
    found = False
    for row in rows:
        if row.get('source_id') == args.source_id:
            row[args.field] = args.value
            if args.error:
                row[f'{args.field}_error'] = args.error
            found = True
            break
    if not found:
        raise SystemExit(f'unknown source_id: {args.source_id}')
    write_master(rows)
    print(f'{args.source_id}: {args.field}={args.value}')


if __name__ == '__main__':
    main()
