from __future__ import annotations

import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets" / "meow_v02_sft_v2_2_semantic_verified"


def load(path: Path) -> list[dict]:
    result = []
    if not path.exists():
        return result
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except Exception:
                continue
            if isinstance(value, dict):
                result.append(value)
    return result


def text(row: dict) -> str:
    return "\n".join(f"{item.get('role')}: {item.get('content')}" for item in row.get("messages") or [])


def sample_rows(values: list[dict], count: int = 20) -> list[dict]:
    rng = random.Random(20260913)
    chosen = values if len(values) <= count else rng.sample(values, count)
    return [{
        "sample_id": row.get("sample_id"),
        "source_id": row.get("source_id"),
        "recording_family_id": row.get("recording_family_id"),
        "context_turn_ids": row.get("context_turn_ids"),
        "target_turn_ids": row.get("target_turn_ids"),
        "messages": row.get("messages"),
        "semantic_qa": row.get("semantic_qa"),
        "transcript_qa": row.get("transcript_qa"),
        "episode_reconstruction": row.get("episode_reconstruction"),
    } for row in chosen]


def main() -> None:
    train = load(OUT / "train.jsonl")
    validation = load(OUT / "validation.jsonl")
    sealed = load(OUT / "sealed_eval.jsonl")
    all_rows = train + validation + sealed
    audit = {
        "schema_version": "1.0.0",
        "pipeline_version": "sft-v2.2-sample-audit-2026-09-13",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample_policy": "deterministic seed 20260913; 20 per available slice; no text rewriting",
        "counts": {"train": len(train), "validation": len(validation), "sealed_eval": len(sealed)},
        "slices": {
            "multi_segment_episode": sample_rows([r for r in all_rows if len(r.get("target_turn_ids") or []) > 1]),
            "unknown_context": sample_rows([r for r in all_rows if "UNKNOWN" in ((r.get("speaker_evidence") or {}).get("context_identities") or [])]),
            "gameplay_or_event_source": sample_rows([r for r in all_rows if any(term in str(r.get("source_metadata", {}).get("title", "")).lower() for term in ("game", "gaming", "minecraft", "chess", "playing"))]),
            "semantic_accept_direct": sample_rows([r for r in all_rows if (r.get("semantic_qa") or {}).get("relation") == "DIRECT_RESPONSE"]),
            "semantic_accept_contextual": sample_rows([r for r in all_rows if (r.get("semantic_qa") or {}).get("relation") == "CONTEXTUAL_RESPONSE"]),
            "semantic_accept_game_state": sample_rows([r for r in all_rows if (r.get("semantic_qa") or {}).get("relation") == "GAME_STATE_RESPONSE"]),
        },
        "rejected_examples": sample_rows(load(OUT / "episode_reconstruction_audit_v2_2.jsonl"))[:20],
        "overlap_edge_examples": load(OUT / "recording_overlap_edges_v2_2.jsonl")[:20],
        "review_notes": [
            "This artifact is an audit sample, not a replacement for semantic QA.",
            "Review must distinguish Neuro's real abrupt/absurd style from wrong-context and ASR failures.",
            "No training_candidate flag is changed by this report.",
        ],
    }
    (OUT / "self_audit_samples_v2_2.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUT / 'self_audit_samples_v2_2.json'), "sample_slices": {key: len(value) for key, value in audit["slices"].items()}, "rejected_examples": len(audit["rejected_examples"]), "overlap_edges": len(audit["overlap_edge_examples"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
