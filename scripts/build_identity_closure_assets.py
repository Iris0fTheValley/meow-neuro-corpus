from __future__ import annotations

"""Build auditable identity-closure assets without promoting proxy labels.

This stage is intentionally metadata-first.  It creates participant priors,
recurring guest negative *candidates*, an independent source-disjoint anchor
split, and a semantic style bank from the existing auto-anchor evidence.
Nothing produced here is human gold or a training label.
"""

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from manifest_tools import ROOT, read_jsonl, safe_name, write_json


VERSION = "identity-closure-2026-09-11-v1"
VALIDATION_SOURCE_IDS = {
    "8335sBz9aHU",  # explicit Evil-only source
    "SJxG6ASjZeY",  # explicit Evil-only source
    "GjIopQlnEUY",  # explicit Neuro-only source
    "FpMGqhh_yd8",  # explicit Neuro-only source
}
FAMILY_ALIASES = {
    "NEURO_FAMILY": ["neuro", "neuro-sama", "neuro sama", "evil", "evil neuro", "evil-neuro"],
}
VEDAL_ALIASES = ["vedal", "vedal987"]
GUEST_ALIASES = {
    "Camila": ["camila"],
    "MinikoMew": ["minikom ew", "minikom ew", "miniko", "miyune"],
    "Filian": ["filian"],
    "Nihmune": ["nihmune", "numi", "akumanihmune"],
    "CerberVT": ["cerbervt", "cerber"],
    "Layna Lazar": ["layna lazar", "layna"],
    "Koko D. Nuts": ["kokonuts", "koko d nuts", "koko"],
    "Toma": ["slice of toma", "toma"],
    "Anny": ["anny"],
    "OniGiri": ["onigiri", "oni giri", "giri"],
    "CottontailVA": ["cottontailva", "cottontail"],
    "CodeMiko": ["codemiko", "code miko"],
    "Lia": ["lia"],
    "Kiara": ["takanashi kiara", "kiara"],
    "fallenshadow": ["fallenshadow", "shondo"],
    "Zentreya": ["zentreya"],
    "Shylily": ["shylily"],
    "Snuffy": ["snuffy"],
    "Sinder": ["sinder"],
    "LucyPyre": ["lucypyre", "lucy pyre"],
    "HannahHyrule": ["hannahhyrule", "hannah hyrule"],
    "EllieMiniBot": ["ellieminibot", "ellie mini bot", "ellie"],
    "MageMimi": ["magemimi", "mage mimi", "mimi"],
    "MoniiBagel": ["moniibagel", "monii bagel"],
    "Michi Mochievee": ["michi mochievee", "michi"],
    "Crelly": ["crelly"],
    "Chrchie": ["chrchie"],
    "GEEGA": ["geega"],
    "MOTHERv3": ["motherv3", "mother v3"],
}
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "he", "her", "i",
    "if", "in", "is", "it", "me", "my", "of", "on", "or", "our", "so", "that", "the", "their",
    "there", "they", "this", "to", "us", "was", "we", "were", "will", "with", "you", "your",
}
DATE_TOKEN = re.compile(r"^(?:\d{1,2}[-/]\d{1,2}(?:#\d+)?|\d{4}[-/]\d{1,2}[-/]\d{1,2})$")


def norm(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s.-]+", " ", text)).strip()


def aliases() -> dict[str, list[str]]:
    out = {"NEURO_FAMILY": FAMILY_ALIASES["NEURO_FAMILY"], "VEDAL": VEDAL_ALIASES}
    out.update(GUEST_ALIASES)
    return out


ALIAS_TO_LABEL = {norm(alias): label for label, values in aliases().items() for alias in values}


def explicit_participant_labels(values: list[object]) -> tuple[set[str], list[str]]:
    labels: set[str] = set()
    clean: list[str] = []
    for value in values or []:
        raw = str(value).strip()
        n = norm(raw)
        if not n or DATE_TOKEN.match(n):
            continue
        clean.append(raw)
        if n in ALIAS_TO_LABEL:
            labels.add(ALIAS_TO_LABEL[n])
    return labels, clean


def text_hits(text: str) -> dict[str, list[str]]:
    low = norm(text)
    hits: dict[str, list[str]] = defaultdict(list)
    for label, values in aliases().items():
        for alias in values:
            a = norm(alias)
            if len(a) < 4 and a not in {"koko", "lia", "giri", "toma", "anny", "bao"}:
                continue
            pattern = rf"(?<![\w]){re.escape(a)}(?![\w])"
            if re.search(pattern, low):
                hits[label].append(alias)
    if re.search(r"(?<![\w])(neuro|evil)(?![\w])", low):
        hits.setdefault("NEURO_FAMILY", []).append("family-name-text")
    if re.search(r"(?<![\w])vedal(?:987)?(?![\w])", low):
        hits.setdefault("VEDAL", []).append("vedal-text")
    return dict(hits)


def source_prior(row: dict) -> dict:
    participants = row.get("participants") or []
    explicit, clean_participants = explicit_participant_labels(participants)
    text = " ".join(str(row.get(k) or "") for k in ("title", "description", "notes", "uploader", "discovery_query"))
    hits = text_hits(text)
    possible: list[str] = []
    if "NEURO_FAMILY" in explicit or "NEURO_FAMILY" in hits:
        possible.append("NEURO_FAMILY")
    if "VEDAL" in explicit or "VEDAL" in hits:
        possible.append("VEDAL")
    guest_labels = sorted(label for label in GUEST_ALIASES if label in explicit or label in hits)
    possible.extend(guest_labels)
    if any(label not in {"NEURO_FAMILY", "VEDAL"} for label in explicit):
        if not guest_labels:
            possible.append("OTHER_GUEST")
    participant_evidence = [{"raw": raw, "normalized": norm(raw), "label": ALIAS_TO_LABEL.get(norm(raw), "OTHER_GUEST")} for raw in clean_participants]
    raid_values = []
    for key in ("raid_target", "raid_targets", "raid", "raided_channel"):
        value = row.get(key)
        if value:
            raid_values.extend(value if isinstance(value, list) else [value])
    confidence = "high" if explicit else ("medium" if hits else "low")
    return {
        "source_id": str(row.get("source_id")),
        "source_url": row.get("source_url"),
        "title": row.get("title"),
        "possible": sorted(set(possible)),
        "explicit_participant_labels": sorted(explicit),
        "explicit_participants": clean_participants,
        "participant_evidence": participant_evidence,
        "text_alias_hits": hits,
        "confidence": confidence,
        "evidence": [
            "explicit participant metadata" if explicit else None,
            "title/description/notes alias match" if hits else None,
            "raid target retained separately and never used as participant prior" if raid_values else None,
        ],
        "raid_target_metadata_only": raid_values,
        "audio_available": bool(row.get("audio_path")),
        "diarization_available": bool(row.get("diarization_path")),
        "prior_policy": "participant metadata narrows identity space; it never overrides audio or forces a speaker label",
    } | {"evidence": [x for x in [
        "explicit participant metadata" if explicit else None,
        "title/description/notes alias match" if hits else None,
        "raid target retained separately and never used as participant prior" if raid_values else None,
    ] if x]}


def load_manifest() -> dict[str, dict]:
    return {str(row.get("source_id")): row for row in read_jsonl(ROOT / "manifest" / "master_video_manifest.jsonl") if row.get("source_id")}


def build_priors(manifest: dict[str, dict], out_dir: Path) -> list[dict]:
    rows = [source_prior(row) for row in manifest.values()]
    out = out_dir / "expected_participants.jsonl"
    out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in sorted(rows, key=lambda x: x["source_id"])), encoding="utf-8")
    write_json(out_dir / "expected_participants_report.json", {
        "schema_version": "0.1.0", "version": VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "EXPECTED_PARTICIPANT_PRIORS_READY", "source_count": len(rows),
        "sources_with_explicit_participants": sum(bool(r["explicit_participants"]) for r in rows),
        "possible_label_counts": dict(Counter(label for r in rows for label in r["possible"])),
        "known_guest_source_counts": dict(Counter(label for r in rows for label in r["possible"] if label in GUEST_ALIASES)),
        "raid_target_field_count": sum(bool(r["raid_target_metadata_only"]) for r in rows),
        "policy": "Raid targets are never promoted to stream participants; ambiguous sources keep a weak prior.",
    })
    return rows


def build_negative_bank(manifest: dict[str, dict], priors: dict[str, dict], out_dir: Path) -> list[dict]:
    mapping_path = ROOT / "identity_results" / "identity_mapping_family_proxy.jsonl"
    mapping = list(read_jsonl(mapping_path)) if mapping_path.exists() else []
    candidates: list[dict] = []
    for row in mapping:
        source_id = str(row.get("source_id"))
        prior = priors.get(source_id, {})
        named_guests = sorted(set(prior.get("explicit_participant_labels", [])) & set(GUEST_ALIASES))
        other_guest = "OTHER_GUEST" in prior.get("possible", [])
        if not named_guests and not other_guest:
            continue
        clip = ROOT / str(row.get("proxy_clip_path") or "")
        if not clip.exists():
            continue
        proxy_identity = str(row.get("identity") or "UNKNOWN")
        if proxy_identity == "NEURO_FAMILY":
            # A family proxy is not safe to relabel as a guest merely because a
            # guest is present in the same source.
            continue
        label = named_guests[0] if len(named_guests) == 1 else "OTHER_GUEST"
        candidates.append({
            "candidate_id": f"guestneg:{source_id}:{row.get('cluster')}:{label}",
            "source_id": source_id,
            "cluster": row.get("cluster"),
            "clip_path": str(clip.relative_to(ROOT)),
            "candidate_label": label,
            "explicit_guest_labels_in_source": named_guests,
            "proxy_identity_at_mining": proxy_identity,
            "proxy_margin": row.get("family_vs_nontarget_margin"),
            "status": "PROVISIONAL_METADATA_AUDIO_CANDIDATE",
            "trusted": False,
            "provenance": {
                "participant_prior_version": VERSION,
                "explicit_participant_metadata": prior.get("explicit_participants", []),
                "prior_evidence": prior.get("evidence", []),
                "leave_source_out_required": True,
                "not_derived_from_semantic_fingerprint": True,
            },
        })
    by_label = defaultdict(set)
    for row in candidates:
        by_label[row["candidate_label"]].add(row["source_id"])
    for row in candidates:
        row["recurring_source_count"] = len(by_label[row["candidate_label"]])
        row["prototype_status"] = "PROVISIONAL_ACOUSTIC_PROTOTYPE_CANDIDATE" if row["recurring_source_count"] >= 2 else "SINGLE_SOURCE_CANDIDATE"
    out = out_dir / "recurring_guest_negative_bank.jsonl"
    out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in sorted(candidates, key=lambda x: (x["candidate_label"], x["source_id"], str(x["cluster"])))), encoding="utf-8")
    named_counts = Counter(row["candidate_label"] for row in candidates)
    source_counts = {label: len(sources) for label, sources in by_label.items()}
    write_json(out_dir / "recurring_guest_negative_bank_report.json", {
        "schema_version": "0.1.0", "version": VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "RECURRING_GUEST_NEGATIVE_CANDIDATES_READY", "candidate_count": len(candidates),
        "candidate_label_counts": dict(named_counts), "candidate_source_counts": source_counts,
        "recurring_label_counts": {label: count for label, count in source_counts.items() if count >= 2},
        "known_guest_priority": sorted(GUEST_ALIASES), "other_guest_policy": "Use OTHER_GUEST when source metadata names multiple or unknown guests.",
        "gold_policy": "No candidate is gold; explicit metadata and proxy mapping are independent evidence but still require hold-out validation.",
    })
    return candidates


def style_features(text: str) -> dict:
    raw = str(text or "")
    low = raw.casefold()
    tokens = re.findall(r"[a-z]+(?:'[a-z]+)?", low)
    content = [token for token in tokens if token not in STOPWORDS]
    return {
        "word_count": len(tokens), "content_word_count": len(content), "char_count": len(raw),
        "first_person_count": len(re.findall(r"\b(?:i|i'm|i've|i'll|me|my|mine|we|us|our)\b", low)),
        "second_person_count": len(re.findall(r"\b(?:you|your|yours|u)\b", low)),
        "question_count": raw.count("?"),
        "hedge_count": len(re.findall(r"\b(?:maybe|perhaps|probably|i think|i guess|kind of|sort of|might|could)\b", low)),
        "assertion_count": len(re.findall(r"\b(?:obviously|definitely|clearly|must|will|is|are)\b", low)),
        "repair_false_start_count": len(re.findall(r"\b(?:i mean|sorry|wait|actually|no,|or rather|uh+|um+)\b", low)),
        "address_count": len(re.findall(r"\b(?:chat|guys|everyone|people|friend|buddy|sir|ma'am)\b", low)),
        "reaction_count": len(re.findall(r"\b(?:oh|wow|what|why|no|yes|yeah|haha|lol|yay|ugh|huh)\b", low)),
        "lexical_content_top": Counter(content).most_common(20),
    }


def anchor_context(anchor: dict, manifest: dict[str, dict]) -> str:
    source_id = str(anchor.get("source_id"))
    path = ROOT / "unique_timelines" / f"{safe_name(source_id)}.json"
    if not path.exists():
        return str(anchor.get("text_preview") or "")
    try:
        turns = json.loads(path.read_text(encoding="utf-8")).get("turns") or []
        start = float(anchor.get("start") or 0)
        index = min(range(len(turns)), key=lambda i: abs(float((turns[i].get("timestamp") or {}).get("start") or 0) - start)) if turns else 0
        nearby = turns[max(0, index - 4): index + 5]
        return " ".join(str(turn.get("text") or "") for turn in nearby)
    except Exception:
        return str(anchor.get("text_preview") or "")


def source_split(source_id: str, buckets: int = 5) -> str:
    return "validation" if source_id in VALIDATION_SOURCE_IDS else "bank_train"


def build_semantic_bank(manifest: dict[str, dict], out_dir: Path) -> None:
    anchors = list(read_jsonl(ROOT / "speaker_refs" / "auto_validation_anchors.jsonl")) if (ROOT / "speaker_refs" / "auto_validation_anchors.jsonl").exists() else []
    rows = []
    for anchor in anchors:
        context = anchor_context(anchor, manifest)
        rows.append({
            "anchor_id": anchor.get("anchor_id"), "source_id": str(anchor.get("source_id")), "label": anchor.get("label"),
            "split": source_split(str(anchor.get("source_id"))), "context_aware_text": context,
            "style_features": style_features(context), "semantic_status": "AUTO_TRUSTED_STYLE_ONLY_NOT_GOLD",
            "evidence": anchor.get("evidence", {}), "provenance": "independent metadata/acoustic auto anchor; no proxy identity mapping used",
        })
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_label[str(row["label"])].append(row)
    aggregate = {}
    for label, label_rows in by_label.items():
        numeric = [key for key, value in label_rows[0]["style_features"].items() if isinstance(value, (int, float))]
        aggregate[label] = {
            "anchor_count": len(label_rows), "source_count": len({row["source_id"] for row in label_rows}),
            "feature_means": {key: round(sum(float(row["style_features"][key]) for row in label_rows) / len(label_rows), 4) for key in numeric},
            "split_counts": dict(Counter(row["split"] for row in label_rows)),
        }
    out = out_dir / "semantic_identity_bank.json"
    write_json(out, {
        "schema_version": "0.1.0", "version": VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "SEMANTIC_IDENTITY_BANK_AUTO_TRUSTED_STYLE_ONLY", "embedding_model": None,
        "embedding_status": "not_installed_transformers_or_sentence_transformers; lexical/role features are retained without fabricating dense embeddings",
        "context_policy": "current anchor text plus up to four preceding and four following canonical turns",
        "labels": aggregate, "records": rows,
        "feature_schema": "lexical/discourse counts, repair/false-start, question/answer proxy, address/reaction counts, and content lexicon for audit only",
        "circularity_guard": "bank built only from independent auto-anchor metadata/acoustic evidence; proxy identity mapping is not an input",
        "promotion_policy": "auxiliary evidence only; semantic similarity cannot override strong acoustic conflict",
    })
    write_json(out_dir / "source_disjoint_validation_report.json", {
        "schema_version": "0.1.0", "version": VERSION, "status": "SOURCE_DISJOINT_SPLIT_READY",
        "records": len(rows), "source_count": len({row["source_id"] for row in rows}),
        "label_split_counts": {label: dict(Counter(row["split"] for row in label_rows)) for label, label_rows in by_label.items()},
        "validation_sources": sorted({row["source_id"] for row in rows if row["split"] == "validation"}),
        "validation_source_selection": "explicit metadata-confirmed Neuro/Evil sources selected before scoring; no query proxy identity used",
        "warnings": ["VEDAL and OTHER have too few independent sources for a stable hold-out precision claim"] if any(label in by_label and len({row['source_id'] for row in by_label[label]}) < 3 for label in ("VEDAL", "OTHER")) else [],
        "policy": "Any future threshold selection must fit on bank_train and report validation on disjoint source IDs.",
    })
    (out_dir / "semantic_identity_bank_records.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    manifest = load_manifest()
    out_dir = ROOT / "speaker_refs" / "identity_closure"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "collaborator_alias_map.json", {
        "schema_version": "0.1.0", "version": VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
        "family": FAMILY_ALIASES["NEURO_FAMILY"], "vedal": VEDAL_ALIASES, "known_guests": GUEST_ALIASES,
        "policy": "Alias matching creates priors/candidates only; it never assigns a speaker cluster by itself.",
    })
    prior_rows = build_priors(manifest, out_dir)
    priors = {row["source_id"]: row for row in prior_rows}
    candidates = build_negative_bank(manifest, priors, out_dir)
    build_semantic_bank(manifest, out_dir)
    print(json.dumps({
        "status": "IDENTITY_CLOSURE_ASSETS_READY", "version": VERSION,
        "source_count": len(manifest), "prior_count": len(prior_rows), "guest_negative_candidate_count": len(candidates),
        "known_guest_labels": sorted(GUEST_ALIASES), "training_candidate": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
