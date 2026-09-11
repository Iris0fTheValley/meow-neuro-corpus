from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import requests

from manifest_tools import ROOT, log_event


REPOS = ["michael620/library-of-ladev", "NeuroInfoAPI/Docs"]
HEADERS = {"User-Agent": "neuro-public-corpus-discovery/0.1"}


def get(url: str) -> requests.Response:
    response = requests.get(url, headers=HEADERS, timeout=45)
    response.raise_for_status()
    return response


if __name__ == "__main__":
    out_dir = ROOT / "sources" / "github"
    out_dir.mkdir(parents=True, exist_ok=True)
    report: list[dict] = []
    for repo in REPOS:
        slug = repo.replace("/", "__")
        item: dict = {"repo": repo, "url": f"https://github.com/{repo}", "retrieved_at": datetime.now(timezone.utc).isoformat()}
        try:
            meta = get(f"https://api.github.com/repos/{repo}").json()
            tree = get(f"https://api.github.com/repos/{repo}/git/trees/{meta['default_branch']}?recursive=1").json()
            paths = [x.get("path") for x in tree.get("tree", []) if x.get("path")]
            item.update({"default_branch": meta.get("default_branch"), "html_url": meta.get("html_url"), "description": meta.get("description"), "stars": meta.get("stargazers_count"), "paths": paths, "tree_truncated": tree.get("truncated", False)})
            (out_dir / f"{slug}.metadata.json").write_text(json.dumps(item, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            readme = requests.get(f"https://raw.githubusercontent.com/{repo}/{meta['default_branch']}/README.md", headers=HEADERS, timeout=45)
            if readme.ok:
                (out_dir / f"{slug}.README.md").write_text(readme.text, encoding="utf-8")
            log_event("github_source_discovered", repo=repo, paths=len(paths), tree_truncated=tree.get("truncated", False))
            report.append({"repo": repo, "status": "done", "paths": len(paths), "interesting_paths": [p for p in paths if any(k in p.lower() for k in ("metadata", "transcript", "vod", "api", "data"))][:80]})
        except Exception as exc:
            item["status"] = "blocked"
            item["error"] = repr(exc)
            (out_dir / f"{slug}.error.json").write_text(json.dumps(item, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            log_event("github_source_blocked", repo=repo, error=repr(exc))
            report.append({"repo": repo, "status": "blocked", "error": repr(exc)})
    (out_dir / "discovery_report.json").write_text(json.dumps({"retrieved_at": datetime.now(timezone.utc).isoformat(), "repositories": report}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
