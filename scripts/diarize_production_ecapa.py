from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import soundfile as sf
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

from manifest_tools import ROOT, read_jsonl, safe_name, write_json, write_master, log_event


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-id', required=True)
    ap.add_argument('--clusters', type=int, default=2)
    args = ap.parse_args()
    rows = read_jsonl(ROOT / 'manifest' / 'master_video_manifest.jsonl')
    row = next(r for r in rows if r.get('source_id') == args.source_id)
    audio = ROOT / row['audio_path']
    asr = json.loads((ROOT / row['asr_path']).read_text(encoding='utf-8'))
    segments = [s for s in asr.get('segments', []) if s.get('text', '').strip() and s['end'] - s['start'] >= 0.6]
    wav = ROOT / 'tmp' / f'{safe_name(args.source_id)}.diarization.wav'
    wav.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-i', str(audio), '-ac', '1', '-ar', '16000', '-f', 'wav', str(wav)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    signal, _ = sf.read(wav, dtype='float32')
    encoder = EncoderClassifier.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb', savedir=str(ROOT / 'models' / 'speechbrain_ecapa'), run_opts={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}, local_strategy=LocalStrategy.COPY)
    max_samples = 8 * 16000
    batch, kept = [], []
    for s in segments:
        a, b = max(0, int(s['start'] * 16000)), min(len(signal), int(s['end'] * 16000))
        clip = signal[a:b]
        if len(clip) < 0.6 * 16000: continue
        clip = clip[:max_samples]
        padded = np.zeros(max_samples, dtype='float32'); padded[:len(clip)] = clip
        batch.append(padded); kept.append(s)
    embeddings = []
    for i in range(0, len(batch), 32):
        x = torch.from_numpy(np.asarray(batch[i:i+32]))
        with torch.inference_mode():
            e = encoder.encode_batch(x).squeeze(1).detach().cpu().numpy()
        e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9)
        embeddings.append(e)
    matrix = np.concatenate(embeddings, axis=0)
    labels = AgglomerativeClustering(n_clusters=args.clusters, metric='cosine', linkage='average').fit_predict(matrix)
    centroids = np.vstack([matrix[labels == k].mean(axis=0) for k in range(args.clusters)])
    centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-9)
    out_segments = []
    for s, label, emb in zip(kept, labels, matrix):
        sims = centroids @ emb
        order = np.argsort(sims)[::-1]
        conf = float(max(0.0, min(1.0, (sims[order[0]] - sims[order[1]] + 1) / 2)))
        out_segments.append({'start': s['start'], 'end': s['end'], 'text': s.get('text', ''), 'cluster': f'SPEAKER_{int(label):02d}', 'speaker_confidence': conf, 'source_asr_segment_id': s.get('id')})
    n_labels = len(set(labels))
    silhouette = float(silhouette_score(matrix, labels, metric='cosine')) if n_labels > 1 and len(matrix) > n_labels else None
    report = {'schema_version': '0.1.0', 'source_id': args.source_id, 'method': 'SpeechBrain ECAPA embeddings + agglomerative clustering', 'device': 'cuda' if torch.cuda.is_available() else 'cpu', 'segments': len(out_segments), 'n_clusters': args.clusters, 'cluster_sizes': {f'SPEAKER_{k:02d}': int((labels == k).sum()) for k in range(args.clusters)}, 'silhouette_cosine': silhouette, 'speaker_mapping': 'anonymous_only', 'mapping_note': 'Cluster labels are not asserted to be Neuro, Evil Neuro, Vedal, or guest without validated reference evidence.', 'segments_with_speaker': out_segments}
    out = ROOT / 'diarization' / f'{safe_name(args.source_id)}.ecapa.json'
    write_json(out, report)
    row.update({'diarization_status': 'done', 'diarization_path': str(out.relative_to(ROOT)), 'diarization_method': report['method'], 'speaker_mapping_status': 'attempted_anonymous', 'speaker_mapping_note': report['mapping_note']})
    write_master(rows)
    log_event('diarization_production_complete', source_id=args.source_id, segments=len(out_segments), silhouette=report['silhouette_cosine'])
    print(json.dumps({k: report[k] for k in ('source_id','segments','cluster_sizes','silhouette_cosine','device')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
