from __future__ import annotations

import argparse, json, re
from pathlib import Path
from manifest_tools import ROOT, log_event, read_jsonl, write_master

def feats(text: str) -> dict:
    low = text.lower(); tok = re.findall(r"\b[\w'’-]+\b", text)
    return {
        'response_length': len(tok), 'sentence_count': len(re.findall(r'[.!?]', text)) or int(bool(text.strip())),
        'self_reference': int(bool(re.search(r'\b(i|me|my|mine|myself|we|our|us)\b', low))),
        'uncertainty': int(bool(re.search(r'\b(maybe|perhaps|probably|i think|i guess|not sure|might)\b', low))),
        'repair': int(bool(re.search(r'\b(no,? no|i mean|wait|actually|or rather|sorry)\b', low))),
        'assertiveness': int(bool(re.search(r'\b(definitely|obviously|of course|must|never|always)\b', low))),
        'banter': int(bool(re.search(r'\b(lol|lmao|haha|chat|bro|dad|stupid|idiot)\b', low))),
        'absurd_shift': int(bool(re.search(r'\b(universe|alien|robot|drone|pizza|turtle|moon)\b', low))),
        'emotional_leak': int(bool(re.search(r'\b(love|hate|angry|sad|scared|cry|embarrass|happy)\b', low))),
        'identity_reference': int(bool(re.search(r'\b(neuro|evil|vedal|ai|artificial|daughter|sister)\b', low))),
    }

def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument('--source-id', required=True); args = ap.parse_args()
    rows = read_jsonl(ROOT/'manifest/master_video_manifest.jsonl'); row = next(r for r in rows if r.get('source_id') == args.source_id)
    asr = json.loads((ROOT/row['asr_path']).read_text(encoding='utf-8'))
    diar = json.loads((ROOT/row['diarization_path']).read_text(encoding='utf-8'))
    labels = {s.get('source_asr_segment_id'): s for s in diar.get('segments_with_speaker', [])}
    turns=[]
    for s in asr.get('segments', []):
        text=s.get('text','').strip(); d=labels.get(s.get('id'))
        if not text or not d: continue
        turns.append({'timestamp': {'start': s['start'], 'end': s['end']}, 'speaker': d['cluster'], 'speaker_confidence': d['speaker_confidence'], 'text': text, 'source': {'type':'local_asr_plus_ecapa','asr_path':row['asr_path'],'diarization_path':row['diarization_path'],'model':row.get('asr_model')}, 'turn_boundary':'asr_segment_with_diarization','interruption':False,'overlap':False,'uncertain_transcription':(s.get('avg_logprob') or -10) < -1.0,'raw_segment':s})
    records=[]
    for start in range(0, len(turns), 16):
        window=turns[start:start+20]
        if len(window)<4: continue
        joined=' '.join(t['text'] for t in window); f=feats(joined)
        records.append({'conversation_id':f"{row.get('source_platform', 'unknown')}:{args.source_id}:diarized:{start//16:05d}",'source_video_id':args.source_id,'source_url':row['source_url'],'stream_date_if_known':row.get('stream_date_if_known') or row.get('upload_date'),'participants':row.get('participants',[]),'persona_scope':row.get('neuro_or_evil','uncertain'),'turns':window,'quality':{**f,'multi_turn_value':min(1.0,len(window)/10.0),'asr_confidence':max(0.0,min(1.0,1.0+sum((t['raw_segment'].get('avg_logprob') or -2) for t in window)/len(window)/2)),'speaker_confidence':sum(t['speaker_confidence'] for t in window)/len(window)},'training_candidate':False,'candidate_reason':'anonymous diarization only; Neuro/Evil/Vedal mapping pending'})
    raw=ROOT/'conversations/conversations_raw.jsonl'; clean=ROOT/'conversations/conversations_cleaned.jsonl'; reject=ROOT/'datasets/rejected_segments.jsonl'
    for path in (raw,clean,reject):
        old=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()] if path.exists() else []
        kept=[x for x in old if x.get('source_video_id')!=args.source_id and x.get('record',{}).get('source_video_id')!=args.source_id]
        add=[{'conversation_id':r['conversation_id'],'reason':'speaker_mapping_pending','record':r} for r in records] if path==reject else records
        path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in kept+add),encoding='utf-8')
    qpath=ROOT/'reports'/f'quality_metrics_{args.source_id}.json'; qpath.write_text(json.dumps({'source_id':args.source_id,'records':len(records),'turns':len(turns),'policy':'Anonymous clusters are retained; no identity labels inferred.'},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    row.update({'conversation_status':'done','conversation_raw_count':len(records),'conversation_raw_path':'conversations/conversations_raw.jsonl','quality_metrics_path':str(qpath.relative_to(ROOT)),'training_candidates_status':'pending_speaker_mapping'})
    write_master(rows)
    transient = ROOT/'tmp'/f'{args.source_id}.diarization.wav'
    if transient.is_file():
        size = transient.stat().st_size
        transient.unlink()
        log_event('transient_audio_cleanup', source_id=args.source_id, removed_wav=1, removed_bilibili_dirs=0, removed_bytes=size)
    print(json.dumps({'source_id':args.source_id,'records':len(records),'turns':len(turns),'quality_metrics_path':str(qpath.relative_to(ROOT))},ensure_ascii=False))
if __name__=='__main__': main()
