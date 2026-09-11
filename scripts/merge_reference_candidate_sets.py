from __future__ import annotations

import json
from datetime import datetime, timezone

from manifest_tools import ROOT, write_json


def main() -> None:
    paths = [
        ROOT / "speaker_refs" / "provisional_reference_candidates.jsonl",
        ROOT / "speaker_refs" / "vedal_provisional_reference_candidates.jsonl",
    ]
    rows = []
    seen = set()
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["candidate_id"] in seen:
                continue
            seen.add(row["candidate_id"])
            rows.append(row)
    rows.sort(key=lambda x: (x["identity_hint"], x["source_id"], x["start"]))
    out = ROOT / "speaker_refs" / "all_provisional_reference_candidates.jsonl"
    out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
    write_json(ROOT / "speaker_refs" / "reference_bank_all_provisional.json", {
        "schema_version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PROVISIONAL_PENDING_CROSS_MODEL_GOLD_VALIDATION",
        "candidate_count": len(rows),
        "by_identity": {identity: sum(x["identity_hint"] == identity for x in rows) for identity in ("NEURO", "EVIL", "VEDAL")},
        "source_count": len({x["source_id"] for x in rows}),
        "trusted_clip_count": 0,
        "inputs": [str(p.relative_to(ROOT)) for p in paths if p.exists()],
    })
    print(json.dumps({"candidate_count": len(rows), "by_identity": {identity: sum(x["identity_hint"] == identity for x in rows) for identity in ("NEURO", "EVIL", "VEDAL")}, "source_count": len({x["source_id"] for x in rows})}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
