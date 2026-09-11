from __future__ import annotations
import argparse, json
from pathlib import Path
from manifest_tools import ROOT, read_jsonl, safe_name, write_master

def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument('--source-id',required=True); args=ap.parse_args()
    rows=read_jsonl(ROOT/'manifest/master_video_manifest.jsonl'); row=next(r for r in rows if r.get('source_id')==args.source_id)
    src=ROOT/'raw_subtitles'/'library_of_ladev'/f'{safe_name(args.source_id)}.json'
    payload=json.loads(src.read_text(encoding='utf-8')); subs=payload['payload']['data']['result'].get('subtitles',[])
    segments=[]
    for i,s in enumerate(subs):
        text=(s.get('text') or '').strip(); start=s.get('startTime'); end=s.get('endTime') or start
        if text and start is not None and end is not None:
            segments.append({'id':s.get('subtitleId',i),'start':float(start),'end':float(end),'text':text,'source_subtitle_id':s.get('subtitleId'),'source_attribution':'Library of Ladev public API'})
    out=ROOT/'asr'/f'{safe_name(args.source_id)}.ladev-transcript.json'; out.write_text(json.dumps({'source_id':args.source_id,'backend':'curated_source_transcript_adapter','model':'Library of Ladev public API','segments':segments},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    row.update({'asr_path':str(out.relative_to(ROOT)),'asr_status':'source_transcript','asr_model':'Library of Ladev public API','asr_segment_count':len(segments),'asr_source':'curated_transcript'})
    write_master(rows); print(json.dumps({'source_id':args.source_id,'segments':len(segments),'path':str(out.relative_to(ROOT))},ensure_ascii=False))
if __name__=='__main__': main()
