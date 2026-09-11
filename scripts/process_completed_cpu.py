from __future__ import annotations

import argparse
import subprocess
import sys

from manifest_tools import ROOT, read_jsonl


def worker_python() -> str:
    candidate = ROOT.parent / ".venv-data" / "Scripts" / "python.exe"
    return str(candidate) if candidate.exists() else sys.executable


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=8)
    args = ap.parse_args()
    rows = read_jsonl(ROOT / "manifest" / "master_video_manifest.jsonl")
    candidates = [r for r in rows if r.get("asr_path") and r.get("diarization_path") and r.get("diarization_status") == "done" and r.get("conversation_status") != "done"]
    candidates.sort(key=lambda r: (float(r.get("duration") or 3600), str(r.get("source_id"))))
    completed = []
    for row in candidates[: max(0, args.limit)]:
        # Use the attached-value form so source IDs beginning with '-' are not
        # parsed as missing option arguments by argparse.
        subprocess.run([worker_python(), "neuro_corpus/scripts/rebuild_diarized_conversations.py", f"--source-id={row['source_id']}"], cwd=str(ROOT.parent), check=True)
        completed.append(row["source_id"])
    if completed:
        for script in ("aggregate_forensic_stats.py", "fingerprint_assets.py", "build_closure_queue.py", "throughput_watchdog.py", "make_checkpoint.py"):
            subprocess.run([worker_python(), f"neuro_corpus/scripts/{script}"], cwd=str(ROOT.parent), check=True)
    print({"candidates_seen": len(candidates), "completed": completed})
