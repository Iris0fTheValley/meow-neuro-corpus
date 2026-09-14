from __future__ import annotations

"""Execution-only Strict sharding coordinator.

This wrapper does not construct or alter semantic requests.  It uses the
existing judge runner with explicit sample-id files and independent result
artifacts, then appends validated shard results through one coordinator.
"""

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets" / "meow_v02_sft_v2_2_semantic_verified"
REQUESTS = OUT / "interaction_judge_requests_v2_3.jsonl"
CANONICAL = OUT / "interaction_judge_strict_results_v2_3.jsonl"
STATE = OUT / "strict_parallel_v2_3"
MODEL = "qwen3.8-27b-efficientthink-simpo-lynnstyle"
ENDPOINT = "http://127.0.0.1:1234/v1/completions"
WORKER = ROOT / "scripts" / "run_sft_v2_2_semantic_judge.py"
SHARD_COUNT = 4
MAX_ATTEMPTS = 3

sys.path.insert(0, str(ROOT / "scripts"))
import run_sft_v2_2_semantic_judge as workflow  # noqa: E402


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"JSONL corruption: {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise RuntimeError(f"JSONL row is not an object: {path}:{line_number}")
            rows.append(value)
    return rows


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def requests_by_id() -> dict[str, dict]:
    return {str(row.get("sample_id")): row for row in load_jsonl(REQUESTS)}


def required_ids(requests: dict[str, dict]) -> set[str]:
    primary = workflow.latest_results(workflow.result_path("primary"))
    return workflow.strict_required_sample_ids(
        list(requests.values()),
        primary,
        MODEL,
        set(workflow.load_terminal_quarantine()),
    )


def compatible_metadata(row: dict, request: dict) -> bool:
    expected = {
        "sample_id": str(request.get("sample_id")),
        "request_sha256": request.get("request_sha256"),
        "schema_version": workflow.closure.SCHEMA_VERSION,
        "pipeline_version": workflow.closure.PIPELINE_VERSION,
        "stage": "strict",
        "judge_prompt_version": workflow.PROMPT_VERSIONS["strict"],
        "judge_model": MODEL,
    }
    return all(
        (str(row.get(key)) if key == "sample_id" else row.get(key)) == value
        for key, value in expected.items()
    )


def pending_ids() -> list[str]:
    requests = requests_by_id()
    required = required_ids(requests)
    existing = workflow.latest_results(CANONICAL)
    return sorted(
        sample_id
        for sample_id in required
        if not compatible_metadata(existing.get(sample_id, {}), requests[sample_id])
        or not (existing.get(sample_id, {}).get("validated") or {}).get("valid")
    )


def create_plan(label: str, ids: list[str]) -> dict:
    plan_dir = STATE / label
    plan_dir.mkdir(parents=True, exist_ok=True)
    shards = []
    ownership = {}
    for index in range(SHARD_COUNT):
        shard_ids = ids[index::SHARD_COUNT]
        id_path = plan_dir / f"ids_{index}.txt"
        result_path = plan_dir / f"results_{index}.jsonl"
        run_path = plan_dir / f"run_{index}.json"
        id_path.write_text("".join(f"{sample_id}\n" for sample_id in shard_ids), encoding="utf-8")
        for sample_id in shard_ids:
            ownership.setdefault(sample_id, []).append(index)
        shards.append(
            {
                "index": index,
                "ids": str(id_path),
                "results": str(result_path),
                "run_report": str(run_path),
                "sample_count": len(shard_ids),
                "sample_ids": shard_ids,
            }
        )
    duplicate_owners = {sample_id: owners for sample_id, owners in ownership.items() if len(owners) != 1}
    if duplicate_owners:
        raise RuntimeError(f"deterministic shard ownership failure: {duplicate_owners}")
    plan = {
        "schema_version": "strict-parallel-v2.3",
        "label": label,
        "created_at": now(),
        "model": MODEL,
        "endpoint": ENDPOINT,
        "shard_count": SHARD_COUNT,
        "sample_count": len(ids),
        "sample_ids": ids,
        "shards": shards,
        "ownership_unique": len(ownership) == len(ids),
    }
    write_json(plan_dir / "plan.json", plan)
    return plan


def launch_and_wait(plan: dict) -> list[dict]:
    processes = []
    handles = []
    for shard in plan["shards"]:
        if not shard["sample_count"]:
            continue
        # Preserve resumable shard artifacts.  The judge runner already skips
        # compatible rows in an existing destination and reruns only missing,
        # stale, or invalid samples.  Deleting the file here would destroy the
        # only local resume point after a coordinator interruption.
        log_path = STATE / plan["label"] / f"worker_{shard['index']}.log"
        handle = log_path.open("w", encoding="utf-8")
        command = [
            sys.executable,
            str(WORKER),
            "judge",
            "--stage",
            "strict",
            "--requests",
            str(REQUESTS),
            "--model",
            MODEL,
            "--primary-model",
            MODEL,
            "--endpoint",
            ENDPOINT,
            "--sample-ids-file",
            shard["ids"],
            "--results",
            shard["results"],
            "--run-report",
            shard["run_report"],
            "--max-attempts",
            str(MAX_ATTEMPTS),
        ]
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
        processes.append((shard, process, command))
        handles.append(handle)
    write_json(
        STATE / plan["label"] / "launch.json",
        {
            "status": "RUNNING",
            "started_at": now(),
            "worker_count": len(processes),
            "workers": [
                {"index": shard["index"], "pid": process.pid, "command": command}
                for shard, process, command in processes
            ],
        },
    )
    outcomes = []
    try:
        for shard, process, _command in processes:
            return_code = process.wait()
            outcomes.append({"index": shard["index"], "pid": process.pid, "return_code": return_code})
    finally:
        for handle in handles:
            handle.close()
    write_json(
        STATE / plan["label"] / "launch.json",
        {"status": "COMPLETED", "completed_at": now(), "workers": outcomes},
    )
    return outcomes


def validate_plan(plan: dict) -> list[dict]:
    requests = requests_by_id()
    expected = set(plan["sample_ids"])
    owners: dict[str, set[int]] = {}
    rows_by_id: dict[str, dict] = {}
    shard_rows = []
    for shard in plan["shards"]:
        rows = load_jsonl(Path(shard["results"])) if shard["sample_count"] else []
        # A resumed shard may contain an earlier invalid attempt followed by
        # its corrected retry.  Validate the latest row per sample while
        # retaining the append-only history in the shard artifact.
        latest_rows: dict[str, dict] = {}
        for row in rows:
            sample_id = str(row.get("sample_id"))
            if sample_id not in expected:
                raise RuntimeError(f"shard {shard['index']} emitted unowned sample {sample_id}")
            owners.setdefault(sample_id, set()).add(shard["index"])
            if not compatible_metadata(row, requests[sample_id]):
                raise RuntimeError(f"metadata/hash mismatch for sample {sample_id}")
            latest_rows[sample_id] = row
        if len(latest_rows) != shard["sample_count"]:
            raise RuntimeError(
                f"shard {shard['index']} latest row count mismatch: expected "
                f"{shard['sample_count']}, got {len(latest_rows)}"
            )
        for sample_id, row in latest_rows.items():
            rows_by_id[sample_id] = row
            shard_rows.append((shard["index"], sample_id, row))
    duplicates = sorted(sample_id for sample_id, shard_indexes in owners.items() if len(shard_indexes) != 1)
    missing = sorted(expected - set(owners))
    if duplicates or missing:
        raise RuntimeError(f"shard result ownership failure: duplicates={duplicates[:10]} missing={missing[:10]}")
    write_json(
        STATE / plan["label"] / "validation.json",
        {
            "status": "PASS",
            "validated_at": now(),
            "expected_rows": len(expected),
            "actual_rows": len(rows_by_id),
            "ownership_unique": not duplicates and not missing,
            "metadata_hash_compatible": True,
            "jsonl_corruption": False,
        },
    )
    return [row for _index, _sample_id, row in sorted(shard_rows, key=lambda x: (x[0], x[1]))]


def merge_rows(label: str, rows: list[dict]) -> int:
    requests = requests_by_id()
    existing = workflow.latest_results(CANONICAL)
    append_rows = []
    for row in rows:
        sample_id = str(row.get("sample_id"))
        prior = existing.get(sample_id)
        if prior and compatible_metadata(prior, requests[sample_id]) and (prior.get("validated") or {}).get("valid"):
            continue
        append_rows.append(row)
        existing[sample_id] = row
    with CANONICAL.open("a", encoding="utf-8") as handle:
        for row in append_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
    write_json(
        STATE / label / "merge.json",
        {"status": "COMPLETED", "merged_at": now(), "appended_rows": len(append_rows)},
    )
    return len(append_rows)


def gpu_snapshot() -> dict:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return {"return_code": result.returncode, "output": result.stdout.strip()}
    except Exception as exc:
        return {"error": str(exc)}


def result_stats() -> dict:
    rows = load_jsonl(CANONICAL)
    latest = workflow.latest_results(CANONICAL)
    usage = [(row, row.get("raw", {}).get("usage") or {}) for row in rows]
    usage = [(row, value) for row, value in usage if value.get("total_tokens") is not None]
    timestamps = []
    for row in latest.values():
        try:
            timestamps.append(datetime.fromisoformat(str(row["completed_at"]).replace("Z", "+00:00")))
        except (KeyError, ValueError):
            pass
    span = (max(timestamps) - min(timestamps)).total_seconds() if len(timestamps) > 1 else 0
    attempts = [int(row.get("attempts") or 0) for row in latest.values()]
    valid = sum(bool((row.get("validated") or {}).get("valid")) for row in latest.values())
    total_tokens = sum(int(value.get("total_tokens") or 0) for _row, value in usage)
    return {
        "record_rows": len(rows),
        "unique_latest_rows": len(latest),
        "usage_rows": len(usage),
        "tokens_recorded": total_tokens,
        "valid_latest": valid,
        "invalid_latest": len(latest) - valid,
        "retry_rows_latest": sum(value > 1 for value in attempts),
        "effective_samples_per_15m": round(len(latest) * 900 / span, 2) if span else None,
        "gpu_snapshot": gpu_snapshot(),
        "captured_at": now(),
    }


def run() -> dict:
    STATE.mkdir(parents=True, exist_ok=True)
    report = {"status": "RUNNING", "started_at": now(), "model": MODEL, "shard_count": SHARD_COUNT}
    report["before"] = result_stats()
    write_json(STATE / "coordinator_status.json", report)

    first_pending = pending_ids()
    pilot_ids = first_pending[: min(8, len(first_pending))]
    if pilot_ids:
        pilot = create_plan("pilot", pilot_ids)
        outcomes = launch_and_wait(pilot)
        if any(item["return_code"] != 0 for item in outcomes):
            raise RuntimeError(f"pilot worker failure: {outcomes}")
        pilot_rows = validate_plan(pilot)
        report["pilot"] = {
            "sample_count": len(pilot_ids),
            "worker_count": SHARD_COUNT,
            "disjoint_ownership": True,
            "rows": len(pilot_rows),
            "merged": merge_rows("pilot", pilot_rows),
        }

    remaining = pending_ids()
    if remaining:
        full = create_plan("full", remaining)
        outcomes = launch_and_wait(full)
        report["full_worker_outcomes"] = outcomes
        if any(item["return_code"] != 0 for item in outcomes):
            raise RuntimeError(f"full worker failure: {outcomes}")
        full_rows = validate_plan(full)
        report["full"] = {
            "sample_count": len(remaining),
            "worker_count": SHARD_COUNT,
            "rows": len(full_rows),
            "merged": merge_rows("full", full_rows),
        }
    else:
        report["full"] = {"sample_count": 0, "worker_count": 0, "rows": 0, "merged": 0}

    report["after"] = result_stats()
    report["status"] = "COMPLETED"
    report["completed_at"] = now()
    write_json(STATE / "coordinator_status.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "status"])
    args = parser.parse_args()
    if args.command == "status":
        print(json.dumps(result_stats(), ensure_ascii=False, indent=2))
    else:
        try:
            print(json.dumps(run(), ensure_ascii=False, indent=2))
        except Exception as exc:
            failure = {"status": "FAILED", "error": str(exc), "failed_at": now(), "after": result_stats()}
            write_json(STATE / "coordinator_status.json", failure)
            raise


if __name__ == "__main__":
    main()


