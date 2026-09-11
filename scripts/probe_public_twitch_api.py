from __future__ import annotations

import json
from datetime import datetime, timezone

import requests

from manifest_tools import ROOT, log_event


if __name__ == "__main__":
    url = "https://neuro.appstun.net/api/v2/twitch/stream"
    headers = {"User-Agent": "neuro-public-corpus-discovery/0.1"}
    out = ROOT / "sources" / "twitch" / "public_stream_status.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = requests.get(url, headers=headers, timeout=45)
        payload = {"retrieved_at": datetime.now(timezone.utc).isoformat(), "url": url, "status_code": response.status_code, "payload": response.json()}
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        log_event("public_twitch_status_complete", status_code=response.status_code, is_live=payload["payload"].get("data", {}).get("isLive"))
        print(json.dumps({"status_code": response.status_code, "is_live": payload["payload"].get("data", {}).get("isLive")}, ensure_ascii=False))
    except Exception as exc:
        payload = {"retrieved_at": datetime.now(timezone.utc).isoformat(), "url": url, "status": "blocked", "error": repr(exc)}
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        log_event("public_twitch_status_blocked", error=repr(exc))
        print(json.dumps(payload, ensure_ascii=False))
