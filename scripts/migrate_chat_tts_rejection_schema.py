from __future__ import annotations

"""Migrate the reject-only artifact without changing scores or decisions."""

import json
from pathlib import Path

from manifest_tools import ROOT


SOURCE = ROOT / "identity_results" / "chat_tts_rejection_scores.jsonl"
TEMP = SOURCE.with_suffix(".schema-migration.tmp")
REPORT = ROOT / "reports" / "chat_tts_rejection_layer.json"


def main() -> None:
    count = 0
    with SOURCE.open(encoding="utf-8") as source, TEMP.open("w", encoding="utf-8", newline="\n") as target:
        for line in source:
            row = json.loads(line)
            row.pop("rejection_is_promotion_only", None)
            row["rejection_only_layer"] = True
            row["can_promote_family"] = False
            row["can_recalibrate_family"] = False
            provenance = dict(row.get("provenance") or {})
            provenance["version"] = "chat-tts-rejection-2026-09-12-v2"
            row["provenance"] = provenance
            target.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    TEMP.replace(SOURCE)
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    report["schema_version"] = "0.2.0"
    report["version"] = "chat-tts-rejection-2026-09-12-v2"
    report["rejection_only_layer"] = True
    report["can_promote_family"] = False
    report["can_recalibrate_family"] = False
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "MIGRATED", "rows": count, "source": str(SOURCE.relative_to(ROOT)), "report": str(REPORT.relative_to(ROOT))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
