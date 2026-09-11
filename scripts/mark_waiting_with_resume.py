from __future__ import annotations

import json
from datetime import datetime, timezone

from manifest_tools import ROOT, write_json


def main() -> None:
    path = ROOT / "checkpoints" / "latest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    gates = payload.setdefault("quality_gates", {})
    gates["state"] = "WAITING_WITH_RESUME"
    gates["wait_reason"] = "Immediate safe corpus work is complete for currently available audio; next progress depends on detached daemon outputs and human audio gold review."
    gates["resume_automation_id"] = "neuro-corpus-checkpoint-runner"
    gates["resume_interval"] = "5 minutes"
    gates["daemon_launcher_pid"] = 41532
    gates["daemon_worker_pid"] = 22824
    gates["daemon_state_path"] = "checkpoints/daemon_state.json"
    gates["daemon_lease_path"] = "checkpoints/active_session_lease_v5.lockdir/owner"
    gates["daemon_event_log"] = "logs/pipeline_events.jsonl"
    gates["daemon_stdout_log"] = "logs/pipeline_daemon.stdout.log"
    gates["daemon_stderr_log"] = "logs/pipeline_daemon.stderr.log"
    gates["human_gold_blockers"] = [
        "speaker_refs/gold_validation_neuro_family.json",
        "speaker_refs/open_set_negative_review_clips.jsonl",
    ]
    payload["created_at"] = datetime.now(timezone.utc).isoformat()
    write_json(path, payload)
    readiness_path = ROOT / "reports" / "meow_v02_data_readiness_interim.json"
    readiness = json.loads(readiness_path.read_text(encoding="utf-8")) if readiness_path.exists() else {}
    readiness["status"] = "WAITING_WITH_RESUME"
    readiness["wait_reason"] = gates["wait_reason"]
    readiness["resume_automation_id"] = gates["resume_automation_id"]
    readiness["resume_interval"] = gates["resume_interval"]
    readiness["daemon_launcher_pid"] = gates["daemon_launcher_pid"]
    readiness["daemon_worker_pid"] = gates["daemon_worker_pid"]
    write_json(readiness_path, readiness)
    print(json.dumps({"status": gates["state"], "resume_automation_id": gates["resume_automation_id"], "daemon_launcher_pid": gates["daemon_launcher_pid"], "daemon_worker_pid": gates["daemon_worker_pid"], "reason": gates["wait_reason"]}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
