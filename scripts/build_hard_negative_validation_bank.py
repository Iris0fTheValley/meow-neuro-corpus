from __future__ import annotations

"""Build a source-disjoint hard-negative bank with explicit evidence tiers.

Recurring named-guest candidates can be AUTO_TRUSTED for verifier calibration
when they have explicit participant metadata, cross-source recurrence, and are
independent of the family anchor sources.  Metadata-only singing/game/TTS
rows remain unlabeled stress clips and never become precision claims.
"""

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone

from manifest_tools import ROOT, read_jsonl, write_json


def main() -> None:
    closure = ROOT / "speaker_refs" / "identity_closure"
    family_anchors = [row for row in read_jsonl(ROOT / "speaker_refs" / "auto_validation_anchors.jsonl") if row.get("label") == "NEURO_FAMILY"]
    family_sources = {str(row.get("source_id")) for row in family_anchors}
    priors = {str(row.get("source_id")): row for row in read_jsonl(closure / "expected_participants.jsonl")}
    guest_rows = list(read_jsonl(closure / "recurring_guest_negative_bank.jsonl"))
    rows: list[dict] = []
    auto_counts = Counter()
    for row in guest_rows:
        source_id = str(row.get("source_id"))
        prior = priors.get(source_id, {})
        recurring = int(row.get("recurring_source_count") or 0)
        explicit_guests = [label for label in (prior.get("explicit_participant_labels") or []) if label not in {"NEURO_FAMILY", "VEDAL"}]
        source_disjoint = source_id not in family_sources
        auto_trusted = bool(source_disjoint and recurring >= 2 and explicit_guests and row.get("proxy_identity_at_mining") in {"UNKNOWN", "VEDAL"})
        tier = "AUTO_TRUSTED" if auto_trusted else "PROVISIONAL"
        rows.append({
            "record_id": row.get("candidate_id"),
            "source_id": source_id,
            "cluster": row.get("cluster"),
            "label": row.get("candidate_label"),
            "clip_path": row.get("clip_path"),
            "bucket": "named_guest",
            "evidence_tier": tier,
            "source_disjoint_to_family_bank": source_disjoint,
            "gold_validated": False,
            "provenance": {
                "explicit_participant_metadata": prior.get("explicit_participants", []),
                "explicit_guest_labels": explicit_guests,
                "recurring_source_count": recurring,
                "proxy_identity_at_mining": row.get("proxy_identity_at_mining"),
                "participant_prior_version": row.get("provenance", {}).get("participant_prior_version"),
                "not_derived_from_semantic_fingerprint": True,
                "auto_trusted_rule": "source-disjoint + explicit guest metadata + recurring across >=2 independent sources + non-family proxy identity",
            },
        })
        auto_counts[tier] += 1

    challenge_path = ROOT / "speaker_refs" / "open_set_challenge_set.jsonl"
    challenge_counts = Counter()
    challenge_sources: dict[str, set[str]] = defaultdict(set)
    for row in read_jsonl(challenge_path) if challenge_path.exists() else []:
        source_id = str(row.get("source_id"))
        source_disjoint = source_id not in family_sources
        if not source_disjoint:
            continue
        bucket = str(row.get("metadata_bucket") or "unknown")
        rows.append({
            "record_id": row.get("challenge_id"),
            "source_id": source_id,
            "cluster": row.get("cluster"),
            "label": None,
            "clip_path": row.get("clip_path"),
            "bucket": bucket,
            "evidence_tier": "UNLABELED_STRESS_ONLY",
            "source_disjoint_to_family_bank": True,
            "gold_validated": False,
            "provenance": {
                "metadata_bucket": bucket,
                "metadata_evidence_only": True,
                "challenge_source": row.get("evidence"),
                "policy": "bucket is not a speaker label; do not use for threshold fitting or precision claims",
            },
        })
        challenge_counts[bucket] += 1
        challenge_sources[bucket].add(source_id)

    output = closure / "hard_negative_validation_bank.jsonl"
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    report = {
        "schema_version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "SOURCE_DISJOINT_HARD_NEGATIVE_BANK_READY",
        "family_anchor_source_count": len(family_sources),
        "record_count": len(rows),
        "auto_trusted_named_guest_count": auto_counts["AUTO_TRUSTED"],
        "provisional_named_guest_count": auto_counts["PROVISIONAL"],
        "unlabeled_stress_clip_count": sum(challenge_counts.values()),
        "unlabeled_stress_bucket_counts": dict(challenge_counts),
        "unlabeled_stress_source_counts": {bucket: len(sources) for bucket, sources in challenge_sources.items()},
        "label_counts": dict(Counter(row["label"] for row in rows if row.get("label"))),
        "evidence_tier_counts": dict(Counter(row["evidence_tier"] for row in rows)),
        "output": str(output.relative_to(ROOT)),
        "gold_validated_count": 0,
        "human_gold_is_promotion_blocker": False,
        "policy": "AUTO_TRUSTED rows are source-disjoint, metadata-confirmed, recurring, and acoustically mined from a separate proxy; unlabeled challenge buckets stress the verifier but never become labels or final precision proof.",
    }
    write_json(closure / "hard_negative_validation_bank_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
