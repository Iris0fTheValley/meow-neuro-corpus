from __future__ import annotations

"""Production completion runner for the raw-waveform audio reconstruction chain.

The expensive audio stages (ECAPA target activity, frame diarization, Qwen3-ASR,
Qwen3 forced alignment) for this corpus were already executed and are pinned in
the content-addressed evidence cache of the source run.  This completion runner
re-drives only the *cheap, deterministic* part of the chain:

    text arbitration -> materialization -> hard dedup -> prefix-ladder dedup
    -> similarity audit -> split inheritance -> production views
    -> mandatory validator -> finalizer

It writes a new versioned output directory and never overwrites the source run
or the frozen baseline.  ``MEOW_REAL_CACHE_ONLY=1`` makes every model stage
fail closed on a cache miss, so a cold cache surfaces as an explicit error
instead of silently re-running GPU inference.

Usage:
    python scripts/run_audio_evidence_completion.py [--run-id RUN] [--cache-run-id RUN]
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
CORPUS = ROOT.parent.parent
DATASET = CORPUS / "datasets" / "meow_v02_sft_v2_2_semantic_verified"
RUN_ROOT = DATASET / "audio_reconstruction_v1"

DEFAULT_RUN = "audio-reconstruction-v1-real-completion-20260919"
DEFAULT_CACHE_RUN = "audio-reconstruction-v1-real-context60-bounded120-20260918"
SOURCE_RUN = "audio-reconstruction-v1-real-20260917"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=os.environ.get("MEOW_REAL_RUN_ID", DEFAULT_RUN))
    parser.add_argument("--cache-run-id", default=os.environ.get("MEOW_REAL_CACHE_RUN_ID", DEFAULT_CACHE_RUN))
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--limit-windows", type=int, default=0, help="smoke-test only: stride-sample N windows")
    args = parser.parse_args()

    run = RUN_ROOT / args.run_id
    cache_run = RUN_ROOT / args.cache_run_id
    if not (cache_run / "cache").exists():
        print(json.dumps({"error": "evidence cache run not found", "cache_run": str(cache_run)}), flush=True)
        return 2
    for name in ("asr", "alignment", "target_activity", "diarization"):
        if not (cache_run / "cache" / name).exists():
            print(json.dumps({"error": "evidence cache stage missing", "stage": name}), flush=True)
            return 2
    run.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir) if args.log_dir else RUN_ROOT
    log_dir.mkdir(parents=True, exist_ok=True)

    environment = dict(os.environ)
    environment.update({
        "MEOW_REAL_RUN_ID": args.run_id,
        "MEOW_REAL_CACHE_RUN_ID": args.cache_run_id,
        "MEOW_REAL_CACHE_ONLY": "1",
        "PYTHONPATH": str(ROOT),
    })
    started = datetime.now(timezone.utc).isoformat()
    steps = []

    runner_log = log_dir / f"{args.run_id}.runner.log"
    print(json.dumps({"phase": "run", "run_id": args.run_id, "cache_run_id": args.cache_run_id, "cache_only": True, "limit_windows": args.limit_windows or None, "log": str(runner_log)}, ensure_ascii=False), flush=True)
    runner_command = [sys.executable, str(SCRIPTS / "run_audio_evidence_real.py")]
    if args.limit_windows:
        runner_command += ["--limit-windows", str(args.limit_windows)]
    with runner_log.open("w", encoding="utf-8") as handle:
        code = subprocess.call(runner_command, cwd=str(ROOT), env=environment, stdout=handle, stderr=subprocess.STDOUT)
    steps.append({"step": "run_audio_evidence_real", "exit_code": code, "log": str(runner_log), "limit_windows": args.limit_windows or None})
    if code != 0:
        print(json.dumps({"phase": "run_failed", "exit_code": code, "log": str(runner_log)}, ensure_ascii=False), flush=True)
        return code

    manifest_path = run / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    summary = {
        "run_id": args.run_id,
        "cache_run_id": args.cache_run_id,
        "started_at": started,
        "materialized": manifest.get("materialized"),
        "quarantine": manifest.get("quarantine"),
        "dedup_removed": manifest.get("dedup_removed"),
        "dedup_removed_by_reason": manifest.get("dedup_removed_by_reason"),
        "window_failures": manifest.get("window_failures"),
        "stage_cache_hits": manifest.get("stage_cache_hits"),
        "stage_cache_misses": manifest.get("stage_cache_misses"),
        "expensive_model_stages_executed": manifest.get("expensive_model_stages_executed"),
        "validator_pass": (manifest.get("validator") or {}).get("pass"),
        "steps": steps,
    }
    (run / "reports").mkdir(parents=True, exist_ok=True)
    (run / "reports" / "completion_run_state.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(json.dumps({"phase": "run_complete", "next": "python scripts/finalize_audio_evidence_real.py --run-id %s" % args.run_id}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
