from __future__ import annotations

"""Audit the corpus tree and build public-review GitHub/HF snapshots.

The snapshot is intentionally conservative around source transcript
redistribution: structured records are copied to HF with transcript text
fields removed, while raw pages, source subtitles, media and model weights
are excluded and recorded in the manifest.
"""

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MEDIA_EXTENSIONS = {
    ".wav", ".pcm", ".mp3", ".flac", ".m4a", ".mp4", ".mkv", ".webm", ".avi", ".mov", ".ogg", ".opus",
}
WEIGHT_EXTENSIONS = {".pt", ".pth", ".ckpt", ".bin", ".safetensors", ".gguf"}
CACHE_EXTENSIONS = {".pyc", ".part", ".tmp", ".sample", ".pack", ".idx"}
TEXT_REDACTION_KEYS = {
    "text", "text_preview", "transcript", "transcript_text", "raw_transcript", "utterance", "subtitle", "subtitles",
}
SECRET_NAME_RE = re.compile(r"(?i)(^|[._-])(secret|token|credential|cookie|password|passwd|private[_-]?key)([._-]|$)|\.env")
SECRET_CONTENT_RE = re.compile(
    rb"(?i)(gho_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|BEGIN\s+(?:RSA|OPENSSH|EC|DSA)?\s*PRIVATE\s+KEY|authorization\s*:\s*bearer\s+[A-Za-z0-9._-]{20,}|(?:api[_-]?key|access[_-]?token|password)\s*[:=]\s*['\"][^'\"]{12,})"
)
STRUCTURED_REDACT_DIRS = {
    "asr", "conversations", "conversation_windows", "datasets", "forensic_annotations", "normalized_subtitles",
    "real_turns", "unique_timelines",
}
RAW_RISK_PARTS = {
    "raw_subtitles", "api_snapshots", "transcript_pages", "pages",
}
CACHE_DIR_NAMES = {
    ".git", ".cache", "__pycache__", "cache", "tmp", "temp", "node_modules", ".venv", "venv", "publish",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contains_secret(path: Path) -> bool:
    if SECRET_NAME_RE.search(path.name):
        return True
    if path.stat().st_size > 25 * 1024 * 1024:
        return False
    try:
        return bool(SECRET_CONTENT_RE.search(path.read_bytes()))
    except (OSError, UnicodeError):
        return False


def path_parts(rel: Path) -> set[str]:
    return {part.casefold() for part in rel.parts}


def is_code_or_doc(rel: Path) -> bool:
    suffixes = {".py", ".pyi", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".ejs", ".css", ".scss", ".html", ".md", ".txt", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".editorconfig", ".gitignore", ".json"}
    return rel.suffix.casefold() in suffixes


def is_redaction_candidate(rel: Path) -> bool:
    parts = path_parts(rel)
    return bool(parts & STRUCTURED_REDACT_DIRS) and rel.suffix.casefold() in {".json", ".jsonl"}


def classify(rel: Path, size: int) -> dict:
    parts = path_parts(rel)
    suffix = rel.suffix.casefold()

    if parts & CACHE_DIR_NAMES or suffix in CACHE_EXTENSIONS:
        return {"category": "excluded_cache", "destination": None, "reason": "cache_or_generated_workspace"}
    if suffix in MEDIA_EXTENSIONS:
        return {"category": "excluded_media", "destination": None, "reason": "raw_or_derived_media_asset"}
    if suffix in WEIGHT_EXTENSIONS or "models" in parts and suffix not in {".yaml", ".yml", ".json", ".md"}:
        return {"category": "excluded_cache", "destination": None, "reason": "model_weight_or_runtime_artifact"}
    if "sources" in parts and (parts & RAW_RISK_PARTS or suffix in {".html", ".vtt"}):
        return {"category": "excluded_redistribution_risk", "destination": None, "reason": "third_party_transcript_or_source_page"}
    if "raw_subtitles" in parts:
        return {"category": "excluded_redistribution_risk", "destination": None, "reason": "source_transcript_asset"}
    if rel.name.endswith(".metadata"):
        return {"category": "excluded_other", "destination": None, "reason": "download_sidecar_metadata"}
    if contains_secret(ROOT / rel):
        return {"category": "excluded_secret", "destination": None, "reason": "secret_name_or_secret_pattern_detected"}

    if "sources" in parts and "library-of-ladev" in parts:
        return {"category": "upload_to_github", "destination": "github", "remote_prefix": "third_party/library-of-ladev", "redact": False}
    if rel.parts and rel.parts[0].casefold() in {"scripts", "reports", "checkpoints"}:
        return {"category": "upload_to_github", "destination": "github", "redact": False}
    if rel.parts and rel.parts[0].casefold() in {"coverage", "quarantine"} and size <= 25 * 1024 * 1024:
        return {"category": "upload_to_github", "destination": "github", "redact": False}
    if len(rel.parts) == 1 and is_code_or_doc(rel) and size <= 25 * 1024 * 1024:
        return {"category": "upload_to_github", "destination": "github", "redact": False}
    if "models" in parts and suffix in {".yaml", ".yml", ".json", ".md"} and size <= 25 * 1024 * 1024:
        return {"category": "upload_to_github", "destination": "github", "redact": False}
    if "logs" in parts:
        if size == 0:
            return {"category": "excluded_other", "destination": None, "reason": "empty_operational_log"}
        return {"category": "upload_to_huggingface", "destination": "huggingface", "redact": False}
    if is_redaction_candidate(rel):
        return {"category": "upload_to_huggingface", "destination": "huggingface", "redact": True, "reason": "public_metadata_copy_text_fields_redacted"}
    if rel.parts and rel.parts[0].casefold() in {
        "identity_results", "speaker_refs", "diarization", "manifest", "retry_queue", "sources", "real_turns", "unique_timelines", "conversation_windows", "conversations", "datasets", "forensic_annotations", "normalized_subtitles", "asr",
    }:
        if suffix in {".json", ".jsonl", ".npz", ".yaml", ".yml", ".md", ".txt", ".tsv", ".csv"}:
            return {"category": "upload_to_huggingface", "destination": "huggingface", "redact": False}
    if is_code_or_doc(rel) and size <= 25 * 1024 * 1024:
        return {"category": "upload_to_github", "destination": "github", "redact": False}
    return {"category": "excluded_other", "destination": None, "reason": "not_in_public_review_allowlist"}


def redact(value):
    if isinstance(value, dict):
        result = {}
        removed = False
        for key, item in value.items():
            if str(key).casefold() in TEXT_REDACTION_KEYS:
                removed = True
                continue
            result[key] = redact(item)
        if removed:
            result["public_sync_text_redacted"] = True
        return result
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def redact_json_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.casefold() == ".jsonl":
        with source.open("r", encoding="utf-8", errors="replace") as src, target.open("w", encoding="utf-8") as dst:
            for line in src:
                try:
                    obj = json.loads(line)
                    obj = redact(obj)
                    dst.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
                except json.JSONDecodeError:
                    dst.write(line)
        return
    with source.open("r", encoding="utf-8", errors="replace") as src:
        obj = json.load(src)
    with target.open("w", encoding="utf-8") as dst:
        json.dump(redact(obj), dst, ensure_ascii=False, indent=2)
        dst.write("\n")


def copy_public_file(source: Path, target: Path, redact_text: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if redact_text and source.suffix.casefold() in {".json", ".jsonl"}:
        redact_json_file(source, target)
    else:
        shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", required=True)
    args = parser.parse_args()
    stage_root = Path(args.stage_root).resolve()
    github_stage = stage_root / "github"
    hf_stage = stage_root / "huggingface"
    github_stage.mkdir(parents=True, exist_ok=True)
    hf_stage.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    entries = []
    changed_during_copy = []

    files = sorted(path for path in ROOT.rglob("*") if path.is_file() and ".git" not in path.parts)
    for source in files:
        rel = source.relative_to(ROOT)
        try:
            before = source.stat()
        except OSError:
            continue
        decision = classify(rel, before.st_size)
        entry = {
            "local_path": str(rel).replace("\\", "/"),
            "destination": decision.get("destination"),
            "remote_path": None,
            "size": before.st_size,
            "sha256": None,
            "upload_status": "excluded",
            "excluded_reason": decision.get("reason"),
            "category": decision["category"],
            "redacted_copy": bool(decision.get("redact", False)),
        }
        if decision.get("destination"):
            prefix = decision.get("remote_prefix")
            remote_rel = Path(prefix) / rel if prefix else rel
            if decision.get("redact"):
                remote_rel = Path("redacted") / remote_rel
            stage = github_stage if decision["destination"] == "github" else hf_stage
            target = stage / remote_rel
            try:
                copy_public_file(source, target, bool(decision.get("redact", False)))
                after = source.stat()
                if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
                    changed_during_copy.append(str(rel).replace("\\", "/"))
                    entry["upload_status"] = "unstable_skipped"
                    entry["excluded_reason"] = "source_changed_during_snapshot"
                    target.unlink(missing_ok=True)
                else:
                    entry["remote_path"] = str(remote_rel).replace("\\", "/")
                    entry["sha256"] = sha256(source)
                    entry["staged_sha256"] = sha256(target)
                    entry["staged_size"] = target.stat().st_size
                    entry["upload_status"] = "staged"
                    entry["excluded_reason"] = None
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                entry["upload_status"] = "stage_error"
                entry["excluded_reason"] = f"stage_error:{type(exc).__name__}"
                if target.exists():
                    target.unlink()
        entries.append(entry)

    # Reuse the public-facing docs already reviewed for the existing repos.
    reviewed_github = ROOT / "publish" / "github" / "README.md"
    reviewed_gitignore = ROOT / "publish" / "github" / ".gitignore"
    reviewed_hf = ROOT / "publish" / "hf" / "README.md"
    for source, stage, remote in ((reviewed_github, github_stage, Path("README.md")), (reviewed_gitignore, github_stage, Path(".gitignore")), (reviewed_hf, hf_stage, Path("README.md"))):
        if source.exists():
            target = stage / remote
            shutil.copy2(source, target)

    counts = Counter(entry["category"] for entry in entries)
    public_manifest_entry = {
        "local_path": "reports/public_sync_manifest.json",
        "destination": "both",
        "remote_path": "reports/public_sync_manifest.json",
        "size": None,
        "sha256": None,
        "upload_status": "staged_after_manifest_write",
        "excluded_reason": None,
        "category": "upload_to_both",
        "redacted_copy": False,
        "self_referential_hash": True,
    }
    payload = {
        "schema_version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scan_started_at": started,
        "root": str(ROOT),
        "snapshot_stage_root": str(stage_root),
        "policy": {
            "raw_media_excluded": True,
            "model_weights_excluded": True,
            "secrets_and_credentials_excluded": True,
            "third_party_transcript_pages_excluded": True,
            "structured_transcript_text_redacted": True,
            "human_gold_is_not_required_for_snapshot": True,
        },
        "summary": {
            "scanned_file_count": len(entries) + 1,
            "scanned_bytes": sum(entry["size"] or 0 for entry in entries),
            "category_counts": dict(counts),
            "staged_github_count": sum(1 for entry in entries if entry["upload_status"] == "staged" and entry["destination"] == "github"),
            "staged_huggingface_count": sum(1 for entry in entries if entry["upload_status"] == "staged" and entry["destination"] == "huggingface"),
            "changed_during_snapshot_count": len(changed_during_copy),
            "secret_detected_count": counts.get("excluded_secret", 0),
            "media_excluded_count": counts.get("excluded_media", 0),
            "redistribution_risk_excluded_count": counts.get("excluded_redistribution_risk", 0),
        },
        "changed_during_snapshot": changed_during_copy,
        "entries": entries + [public_manifest_entry],
    }
    manifest = ROOT / "reports" / "public_sync_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for stage in (github_stage, hf_stage):
        target = stage / "reports" / "public_sync_manifest.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest, target)
    print(json.dumps({"status": "PUBLIC_SYNC_SNAPSHOT_READY", "stage_root": str(stage_root), "summary": payload["summary"], "manifest": str(manifest)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
