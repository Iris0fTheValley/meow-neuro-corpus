from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import requests

from manifest_tools import ROOT, read_jsonl, safe_name, write_master, write_json, log_event


UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131 Safari/537.36"


def fetch(url: str) -> dict:
    response = requests.get(url, headers={"User-Agent": UA, "Referer": "https://www.bilibili.com/"}, timeout=45)
    response.raise_for_status()
    marker = "window.__INITIAL_STATE__="
    pos = response.text.find(marker)
    if pos < 0:
        raise RuntimeError("Bilibili initial state not found")
    data, _ = json.JSONDecoder().raw_decode(response.text[pos + len(marker):])
    return data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-id", action="append")
    args = ap.parse_args()
    rows = read_jsonl(ROOT / "manifest" / "master_video_manifest.jsonl")
    wanted = set(args.source_id or [])
    done = 0
    for row in rows:
        if row.get("source_platform") != "bilibili" or (wanted and row.get("source_id") not in wanted):
            continue
        if row.get("metadata_status") == "done" and not args.source_id:
            continue
        try:
            data = fetch(row["source_url"])
            video = data.get("videoData", {})
            owner = video.get("owner") or {}
            pages = video.get("pages") or []
            safe = {
                "source": "Bilibili public page",
                "source_url": row["source_url"],
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "bvid": video.get("bvid") or row["source_id"],
                "aid": video.get("aid"),
                "title": video.get("title") or row.get("title"),
                "desc": video.get("desc"),
                "pubdate": video.get("pubdate"),
                "duration": video.get("duration"),
                "owner": {k: owner.get(k) for k in ("mid", "name", "face") if owner.get(k) is not None},
                "pages": [{k: p.get(k) for k in ("page", "cid", "part", "duration") if p.get(k) is not None} for p in pages],
                "subtitle_list_present": bool((data.get("subtitle") or {}).get("list")),
            }
            path = ROOT / "manifest" / "metadata" / f"bilibili_{safe_name(row['source_id'])}.json"
            write_json(path, safe)
            row.update({"title": safe["title"], "uploader": owner.get("name") or row.get("uploader"), "duration": sum(int(p.get("duration") or 0) for p in pages) or safe.get("duration"), "metadata_status": "done", "metadata_path": str(path.relative_to(ROOT)), "metadata_source": "Bilibili public page", "metadata_error": None})
            log_event("metadata_complete", source_id=row["source_id"], source_url=row["source_url"], acquisition="bilibili_public_page")
            done += 1
        except Exception as exc:
            row["metadata_status"] = "retryable"
            row["metadata_error"] = repr(exc)
            log_event("metadata_failed", source_id=row.get("source_id"), source_url=row.get("source_url"), error=repr(exc), acquisition="bilibili_public_page")
    write_master(rows)
    print(json.dumps({"processed": done, "total_rows": len(rows)}, ensure_ascii=False))
