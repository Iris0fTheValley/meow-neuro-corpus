from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import requests

from manifest_tools import ROOT, write_json


DATASETS = [
    "neifuisan/Neuro-sama-QnA",
    "Braite/neuro.alpha",
    "CommentOut64/Neuro-sama-QnA-cleaned-translated",
    "wxltzy/Neuro-sama-QnA",
]
BASE = "https://datasets-server.huggingface.co"


def get(path: str, dataset: str) -> dict:
    response = requests.get(f"{BASE}/{path}", params={"dataset": dataset}, timeout=45)
    result = {"dataset": dataset, "endpoint": path, "http_status": response.status_code}
    try:
        result["payload"] = response.json()
    except ValueError:
        result["body_prefix"] = response.text[:500]
    return result


if __name__ == "__main__":
    out_dir = ROOT / "sources" / "huggingface"
    out_dir.mkdir(parents=True, exist_ok=True)
    registry_path = ROOT / "source_registry.json"
    frontier_path = ROOT / "discovery_frontier.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {"schema_version": "0.1.0", "sources": []}
    frontier = json.loads(frontier_path.read_text(encoding="utf-8")) if frontier_path.exists() else {"schema_version": "0.1.0", "frontier": []}
    results = []
    for dataset in DATASETS:
        safe = dataset.replace("/", "__")
        snapshot = {"dataset": dataset, "retrieved_at": datetime.now(timezone.utc).isoformat(), "calls": [get("is-valid", dataset), get("splits", dataset), get("size", dataset)]}
        write_json(out_dir / f"{safe}.json", snapshot)
        valid = bool(snapshot["calls"][0].get("payload", {}).get("valid"))
        if not any(x.get("source_id") == f"hf_{safe}" for x in registry.get("sources", [])):
            registry.setdefault("sources", []).append({"source_id": f"hf_{safe}", "platform": "huggingface", "kind": "derived_or_synthetic_dataset", "name": dataset, "url": f"https://huggingface.co/datasets/{dataset}", "uploader": dataset.split("/", 1)[0], "priority": "low", "discovery_status": "inspected" if valid else "blocked", "notes": "Public dataset lead; preserve provenance only. Exclude from real-recording training candidates unless each row is independently tied to a public source recording."})
        results.append({"dataset": dataset, "valid": valid})
    if not any(x.get("url") == "https://huggingface.co/datasets" for x in frontier.get("frontier", [])):
        frontier.setdefault("frontier", []).append({"type": "dataset_index", "platform": "huggingface", "url": "https://huggingface.co/datasets", "status": "expanded", "next_actions": ["inspect provenance only", "exclude synthetic/QnA derivatives from real dialogue", "follow linked public recordings if present"]})
    write_json(registry_path, registry)
    write_json(frontier_path, frontier)
    print(json.dumps({"datasets": results, "snapshot_dir": str(out_dir.relative_to(ROOT))}, ensure_ascii=False))
