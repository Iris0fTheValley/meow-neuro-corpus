from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from manifest_tools import ROOT


def derived_features(record: dict) -> dict[str, float]:
    turns = record.get("turns", [])
    text = " ".join(str(t.get("text", "")) for t in turns)
    low = text.lower()
    technical = bool(re.search(r"\b(code|coding|program|model|api|ai|stream|server|python|game|dev|developer|computer|gpu|twitch)\b", low))
    visual = bool(re.search(r"\b(look|see|screen|picture|image|drawing|art|watch|show|camera)\b", low))
    emotional_hits = len(re.findall(r"\b(love|hate|angry|sad|scared|cry|embarrass|happy|excited|lonely|afraid|feel|feeling)\b", low))
    topic_markers = len(re.findall(r"\b(anyway|speaking of|on another note|change of topic|what about|back to|actually)\b", low))
    carry_terms = {term for term in ("neuro", "evil", "vedal", "chat", "creator", "twin", "sister") if term in low}
    return {
        "technical_value": float(technical),
        "emotional_intensity": float(min(1.0, emotional_hits / 4.0)),
        "topic_switching": float(topic_markers),
        "state_transition_value": float(min(1.0, (topic_markers + int(bool(re.search(r"\bbut|however|wait|actually\b", low)))) / 3.0)),
        "state_carry": float(min(1.0, len(carry_terms) / 4.0)) if len(turns) >= 4 else 0.0,
        "lore_dependency": float(bool(re.search(r"\b(lore|remember|last time|before|promise|birthday|subathon|mars|creator)\b", low))),
        "visual_context_dependency": float(visual),
        "language_fingerprint_value": float(min(1.0, (sum(float(record.get("quality", {}).get(k, 0) or 0) for k in ("uncertainty", "repair", "banter", "absurd_shift")) / 4.0))),
        "context_dependency": float(max(record.get("quality", {}).get("context_dependency", 0) or 0, min(1.0, max(0, len(turns) - 3) / 10.0))),
    }


def main() -> None:
    path = ROOT / "conversations" / "conversations_raw.jsonl"
    stats = Counter()
    records = 0
    turns = 0
    for line in path.read_text(encoding="utf-8").splitlines() if path.exists() else []:
        if not line.strip():
            continue
        record = json.loads(line)
        records += 1
        turns += len(record.get("turns", []))
        for key, value in record.get("quality", {}).items():
            if isinstance(value, (int, float)):
                stats[key] += value
        for key, value in derived_features(record).items():
            stats[key] += value
    payload = {
        "schema_version": "0.1.0",
        "scope": "all preserved conversation records currently in conversations_raw.jsonl",
        "conversation_records": records,
        "turns": turns,
        "counts": dict(stats),
        "policy": "These are descriptive corpus statistics, not labels of the speaker's true psychology; unknown-speaker records remain excluded from training candidates until mapping is validated.",
    }
    (ROOT / "reports" / "forensic_style_stats.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
