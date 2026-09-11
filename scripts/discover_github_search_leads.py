from __future__ import annotations

import json
from datetime import datetime, timezone

import requests

from manifest_tools import ROOT, log_event


QUERIES = ["Neuro-sama transcript", "Neuro-sama VOD archive", "Evil Neuro subtitles"]
HEADERS = {"User-Agent": "neuro-public-corpus-discovery/0.1"}


if __name__ == "__main__":
    out = ROOT / "sources" / "github" / "search"
    out.mkdir(parents=True, exist_ok=True)
    registry_path = ROOT / "source_registry.json"
    frontier_path = ROOT / "discovery_frontier.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    frontier = json.loads(frontier_path.read_text(encoding="utf-8"))
    known = {str(x.get("source_id")) for x in registry.get("sources", [])}
    frontier_urls = {str(x.get("url")) for x in frontier.get("frontier", [])}
    report = []
    for query in QUERIES:
        response = requests.get("https://api.github.com/search/repositories", params={"q": query, "per_page": 30, "sort": "stars"}, headers=HEADERS, timeout=45)
        payload = response.json()
        (out / (query.replace(" ", "_") + ".json")).write_text(json.dumps({"query": query, "retrieved_at": datetime.now(timezone.utc).isoformat(), "status_code": response.status_code, "payload": payload}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        found = []
        for item in payload.get("items", [])[:30]:
            full_name = item.get("full_name")
            if not full_name:
                continue
            found.append(full_name)
            source_id = "github_repo_" + full_name.replace("/", "__")
            if source_id not in known:
                registry["sources"].append({"source_id": source_id, "platform": "github", "kind": "discovered_repository", "name": full_name, "url": item.get("html_url"), "uploader": item.get("owner", {}).get("login"), "priority": "medium", "discovery_status": "discovered_public_search", "notes": "Repository discovered by public GitHub search; inspect provenance before using any data."})
                known.add(source_id)
            url = item.get("html_url")
            if url and url not in frontier_urls:
                frontier["frontier"].append({"type": "repo", "platform": "github", "url": url, "status": "pending", "next_actions": ["inspect public repository provenance", "look for transcript/subtitle/VOD metadata", "exclude synthetic or recreation data"]})
                frontier_urls.add(url)
        report.append({"query": query, "status_code": response.status_code, "repositories": found})
        log_event("github_public_search_complete", query=query, status_code=response.status_code, results=len(found))
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    frontier["updated_at"] = datetime.now(timezone.utc).date().isoformat()
    frontier_path.write_text(json.dumps(frontier, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "report.json").write_text(json.dumps({"retrieved_at": datetime.now(timezone.utc).isoformat(), "queries": report}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"queries": len(report), "repositories": sum(len(x["repositories"]) for x in report)}, ensure_ascii=False))
