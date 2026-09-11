from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from manifest_tools import ROOT, read_jsonl


def worker_python() -> str:
    candidate = ROOT.parent / ".venv-data" / "Scripts" / "python.exe"
    return str(candidate) if candidate.exists() else sys.executable


def main() -> None:
    limit = 12
    rows = read_jsonl(ROOT / "manifest" / "master_video_manifest.jsonl")
    candidates = []
    for row in rows:
        if row.get("transcript_source") != "Library of Ladev public API":
            continue
        if row.get("transcript_raw_path"):
            continue
        source_id = row.get("source_id")
        if source_id and source_id not in {r.stem for r in (ROOT / "raw_subtitles" / "library_of_ladev").glob("*.json")}:
            candidates.append(source_id)
    candidates = list(dict.fromkeys(candidates))[:limit]
    completed = []
    for source_id in candidates:
        subprocess.run(
            [worker_python(), "neuro_corpus/scripts/fetch_ladev_video.py", "--source-id", source_id],
            cwd=str(ROOT.parent),
            check=False,
        )
        path = ROOT / "raw_subtitles" / "library_of_ladev" / f"{source_id}.json"
        if path.exists():
            subprocess.run(
                [worker_python(), "neuro_corpus/scripts/transcript_asr_adapter.py", "--source-id", source_id],
                cwd=str(ROOT.parent),
                check=False,
            )
            completed.append(source_id)
    print(json.dumps({"candidate_count": len(candidates), "completed": completed}, ensure_ascii=False))


if __name__ == "__main__":
    main()
