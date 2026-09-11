from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import requests

from manifest_tools import ROOT, write_json


URLS = {
    "twitchnosub_vedal987": "https://twitchnosub.com/streamer/vedal987",
    "streamrecorder_vedal987": "https://streamrecorder.io/twitch/vedal987",
    "vodarchive_home": "https://vodarchive.com/",
    "vedal_ai_official": "https://vedal.ai/",
    "vedalvods_channel": "https://www.youtube.com/channel/UCVlZAhUVWiUJt7DxwKpKG9w",
}


def main() -> None:
    out = ROOT / "sources" / "archive_indexes"
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for name, url in URLS.items():
        item = {"name": name, "url": url, "retrieved_at": datetime.now(timezone.utc).isoformat()}
        try:
            response = requests.get(
                url,
                headers={"User-Agent": "neuro-corpus-public-discovery/0.1"},
                timeout=30,
                allow_redirects=True,
            )
            item.update({
                "status_code": response.status_code,
                "final_url": response.url,
                "content_type": response.headers.get("content-type"),
                "content_length": len(response.content),
                "access": "public_http_response" if response.ok else "blocked_or_error",
            })
        except Exception as exc:
            item.update({"status_code": None, "access": "blocked_or_error", "error": repr(exc)})
        write_json(out / f"{name}.json", item)
        results.append(item)
    write_json(out / "index.json", {"sources": results, "policy": "public page metadata only; no cookies, tokens, or authorization headers"})
    print(json.dumps({"probed": len(results), "ok": sum(1 for x in results if x.get("status_code", 0) < 400)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
