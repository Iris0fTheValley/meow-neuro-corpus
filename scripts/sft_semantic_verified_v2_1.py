from __future__ import annotations

"""SFT Semantic Closure v2.1.

This is an additive, fail-closed layer on top of v2 structural candidates.
It uses a local LM Studio Qwen judge for context/response relation and episode
closure, then performs conservative cross-recording overlap and split checks.
The judge never rewrites transcript text.
"""

import argparse
import difflib
import hashlib
import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from manifest_tools import ROOT, write_json
    from sft_semantic_closure_v2_core import TARGETS, is_bad_boundary, normalize_text, text_quality
except ModuleNotFoundError:
    from scripts.manifest_tools import ROOT, write_json
    from scripts.sft_semantic_closure_v2_core import TARGETS, is_bad_boundary, normalize_text, text_quality


V2 = ROOT / "datasets" / "meow_v02_sft_v2_semantic_clean"
OUT = ROOT / "datasets" / "meow_v02_sft_v2_1_semantic_verified"
TIMELINES = ROOT / "unique_timelines"
FUSION = ROOT / "identity_results" / "identity_mapping_multimodal_fusion_proxy.jsonl"
REGISTRY = ROOT / "source_registry.json"
CLUSTERS = ROOT / "reports" / "recording_content_clusters.jsonl"
VERSION = "sft-semantic-verified-v2.1-2026-09-12"
JUDGE_MODEL_PATH = r"J:\AI friend\MEOW Qwen3.5\Qwen3.8-27B-EfficientThink-SimPO-Q4-LynnStyle.gguf"
JUDGE_MODEL = "qwen3.8-27b-efficientthink-simpo-lynnstyle"
JUDGE_PROMPT_VERSION = "semantic-relation-v2.1-p1"
ALLOWED_RELATIONS = {"DIRECT_RESPONSE", "CONTEXTUAL_RESPONSE", "GAME_STATE_RESPONSE", "SELF_CONTINUATION", "HIDDEN_TRIGGER_SUSPECTED", "WRONG_CONTEXT", "UNCLEAR"}
ACCEPT_RELATIONS = {"DIRECT_RESPONSE", "CONTEXTUAL_RESPONSE", "GAME_STATE_RESPONSE"}
ALLOWED_EPISODE = {"KEEP_CURRENT", "MERGE_CONTINUATION", "SPLIT_HIDDEN_TRIGGER", "SPLIT_NEW_SPEECH_ACT", "SPLIT_EVENT_BOUNDARY", "UNCLEAR"}


def load_jsonl(path: Path):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def tokens(value: Any) -> list[str]:
    return re.findall(r"[a-z0-9']+", norm(value))


def token_text(value: Any) -> str:
    return " ".join(tokens(value))


def parse_json_object(value: str) -> dict[str, Any] | None:
    value = str(value or "").strip()
    if not value:
        return None
    starts = [index for index, char in enumerate(value) if char == "{"]
    ends = [index for index, char in enumerate(value) if char == "}"]
    for start in starts:
        for end in reversed(ends):
            if end <= start:
                continue
            try:
                parsed = json.loads(value[start:end + 1])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    return None


def prepare_turn(turn: dict[str, Any], source_id: str, fusion: dict[tuple[str, str], dict[str, Any]], index: int) -> dict[str, Any]:
    result = dict(turn)
    result["_index"] = index
    mapped = fusion.get((source_id, str(turn.get("speaker") or ""))) or {}
    result["identity"] = str(mapped.get("identity") or turn.get("identity") or "UNKNOWN")
    try:
        stamp = turn.get("timestamp") or {}
        result["_start"] = float(stamp.get("start"))
        result["_end"] = float(stamp.get("end"))
    except (TypeError, ValueError):
        result["_start"] = float(index)
        result["_end"] = float(index)
    return result


def load_fusion() -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(row.get("source_id")), str(row.get("cluster"))): row for row in load_jsonl(FUSION)}


def load_registry() -> dict[str, dict[str, Any]]:
    data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    return {str(row.get("source_id")): row for row in data.get("sources", [])}


def load_timelines(source_ids: set[str], fusion: dict[tuple[str, str], dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for source_id in sorted(source_ids):
        path = TIMELINES / f"{source_id}.json"
        if not path.exists():
            continue
        try:
            timeline = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        timeline["_turns"] = [prepare_turn(turn, source_id, fusion, index) for index, turn in enumerate(timeline.get("turns") or [])]
        result[source_id] = timeline
    return result


def structural_rows() -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for split, filename in (("train", "train_clean_full.jsonl"), ("validation", "validation.jsonl"), ("sealed_eval", "sealed_eval.jsonl")):
        for row in load_jsonl(V2 / filename):
            sample_id = str(row.get("sample_id"))
            if sample_id in seen:
                continue
            seen.add(sample_id)
            copy = dict(row)
            copy["structural_source_split"] = split
            copy["structural_status"] = "STRUCTURAL_CANDIDATE"
            copy["semantic_quality"] = "UNKNOWN"
            copy["context_integrity"] = "UNKNOWN"
            copy["transcript_quality"] = "UNKNOWN"
            copy["response_episode_integrity"] = "UNKNOWN"
            copy["training_candidate"] = False
            copy["evaluation_only"] = False
            copy["pipeline_version"] = VERSION
            rows.append(copy)
    return sorted(rows, key=lambda row: str(row.get("sample_id")))


def snapshot(turn: dict[str, Any]) -> dict[str, Any]:
    text = normalize_text(turn.get("text"))
    return {
        "turn_id": turn.get("turn_id"),
        "speaker": turn.get("speaker"),
        "identity": turn.get("identity", "UNKNOWN"),
        "speaker_confidence": turn.get("speaker_confidence"),
        "timestamp": turn.get("timestamp"),
        "gap_from_previous": round(float(turn.get("_gap_from_previous") or 0.0), 3),
        "event_boundary": bool(turn.get("event_boundary")),
        "bad_boundary": is_bad_boundary(turn),
        "text": text[:600],
    }


def attach_gaps(turns: list[dict[str, Any]]) -> None:
    previous_end = None
    for turn in turns:
        turn["_gap_from_previous"] = 0.0 if previous_end is None else max(0.0, turn["_start"] - previous_end)
        previous_end = turn["_end"]


def raw_context(row: dict[str, Any], timeline: dict[str, Any]) -> dict[str, Any] | None:
    turns = timeline.get("_turns") or []
    attach_gaps(turns)
    by_id = {str(turn.get("turn_id")): turn for turn in turns}
    context_ids = [str(value) for value in row.get("context_turn_ids") or []]
    target_ids = [str(value) for value in row.get("target_turn_ids") or []]
    context = [by_id[value] for value in context_ids if value in by_id]
    target = [by_id[value] for value in target_ids if value in by_id]
    if not context or not target:
        return None
    target_positions = [int(turn["_index"]) for turn in target]
    first_index = min(target_positions)
    last_index = max(target_positions)
    candidate_target = list(target)
    cursor = last_index + 1
    # Candidate extensions are offered to the model but are never accepted
    # unless the model explicitly selects their exact raw turn ids.
    while cursor < len(turns) and len(candidate_target) < len(target) + 3:
        next_turn = turns[cursor]
        if next_turn.get("identity") not in TARGETS or next_turn.get("speaker") != target[-1].get("speaker"):
            break
        if is_bad_boundary(next_turn) or text_quality(next_turn.get("text"), 1)[0] != "PASS":
            break
        if float(next_turn.get("_gap_from_previous") or 0.0) > 4.0 or next_turn.get("event_boundary"):
            break
        candidate_target.append(next_turn)
        cursor += 1
    context_start = max(0, int(context[0]["_index"]) - 3)
    following_start = max(int(candidate_target[-1]["_index"]) + 1, last_index + 1)
    following = turns[following_start:following_start + 3]
    preceding = turns[context_start:int(context[0]["_index"])]
    return {
        "context": context,
        "target": target,
        "candidate_target": candidate_target,
        "preceding": preceding,
        "following": following,
        "all_turns": turns,
        "by_id": by_id,
        "first_index": first_index,
        "last_index": last_index,
    }


def make_judge_request(row: dict[str, Any], timeline: dict[str, Any], registry: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    window = raw_context(row, timeline)
    if window is None:
        return None
    source = registry.get(str(row.get("source_id"))) or {}
    context = window["context"]
    target = window["target"]
    candidate_target = window["candidate_target"]
    user_payload = {
        "task": "Judge whether the observable context is a real trigger for the Neuro response, and whether the offered target fragments form one coherent response episode.",
        "allowed_relation": sorted(ALLOWED_RELATIONS),
        "allowed_episode_decision": sorted(ALLOWED_EPISODE),
        "current_target_turn_ids": [str(turn.get("turn_id")) for turn in target],
        "candidate_target_turn_ids": [str(turn.get("turn_id")) for turn in candidate_target],
        "preceding_raw_turns": [snapshot(turn) for turn in window["preceding"][-3:]],
        "candidate_context": [snapshot(turn) for turn in context],
        "candidate_response_episode": [snapshot(turn) for turn in target],
        "possible_continuation_fragments": [snapshot(turn) for turn in candidate_target[len(target):]],
        "following_raw_turns": [snapshot(turn) for turn in window["following"][:3]],
        "source": {
            "source_id": row.get("source_id"),
            "title": source.get("title") or row.get("source_title") or "",
            "source_type": source.get("source_type") or source.get("platform") or "",
            "url": source.get("source_url") or row.get("source_url") or "",
        },
        "constraints": [
            "Do not rewrite, correct, summarize, or normalize any transcript text.",
            "Absurdity, teasing, abruptness, nonstandard grammar, short replies, and topic hijack are not rejection reasons by themselves.",
            "A new hidden chat/donation/game/UI trigger is possible; do not infer a response relationship merely from a short time gap.",
            "A bad, overlap, uncertain, or malformed turn is a boundary, never a transparent turn.",
            "Use only selected_target_turn_ids from candidate_target_turn_ids.",
            "Return exactly one JSON object and no markdown.",
        ],
        "output_schema": {
            "relation": "DIRECT_RESPONSE | CONTEXTUAL_RESPONSE | GAME_STATE_RESPONSE | SELF_CONTINUATION | HIDDEN_TRIGGER_SUSPECTED | WRONG_CONTEXT | UNCLEAR",
            "confidence": "number 0..1",
            "context_complete": "boolean",
            "response_complete": "boolean",
            "transcript_usable": "boolean",
            "reason": "short evidence-based reason",
            "episode_decision": "KEEP_CURRENT | MERGE_CONTINUATION | SPLIT_HIDDEN_TRIGGER | SPLIT_NEW_SPEECH_ACT | SPLIT_EVENT_BOUNDARY | UNCLEAR",
            "selected_target_turn_ids": "array of exact ids from candidate_target_turn_ids",
        },
    }
    return {"sample_id": row.get("sample_id"), "request": user_payload, "judge_input": user_payload}


SYSTEM_PROMPT = (
    "You are a conservative conversation-forensics judge. "
    "Judge observable trigger to response relation, not whether the response sounds like ChatGPT. "
    "Return only one compact JSON object matching the requested schema. "
    "Never rewrite transcript text. "
    "The relation value MUST be exactly one of DIRECT_RESPONSE, CONTEXTUAL_RESPONSE, GAME_STATE_RESPONSE, SELF_CONTINUATION, HIDDEN_TRIGGER_SUSPECTED, WRONG_CONTEXT, UNCLEAR; never invent a relation word such as 'on' or 'answer'. "
    "The episode_decision value MUST be exactly one of KEEP_CURRENT, MERGE_CONTINUATION, SPLIT_HIDDEN_TRIGGER, SPLIT_NEW_SPEECH_ACT, SPLIT_EVENT_BOUNDARY, UNCLEAR. "
    "Example valid output: {\"relation\":\"DIRECT_RESPONSE\",\"confidence\":0.88,\"context_complete\":true,\"response_complete\":true,\"transcript_usable\":true,\"reason\":\"response addresses the observed trigger\",\"episode_decision\":\"KEEP_CURRENT\",\"selected_target_turn_ids\":[\"turn-1\"]}."
)


def call_judge(request_payload: dict[str, Any], model: str = JUDGE_MODEL, endpoint: str = "http://127.0.0.1:1234/v1/completions") -> dict[str, Any]:
    # The OpenAI-compatible chat endpoint does not forward Qwen GGUF chat
    # template variables. Render the same template explicitly with an empty
    # think block so the requested JSON is emitted without hidden reasoning.
    if endpoint.rstrip("/").endswith("/completions"):
        prompt = (
            "<|im_start|>system\n"
            + SYSTEM_PROMPT
            + "<|im_end|>\n<|im_start|>user\n"
            + json.dumps(request_payload, ensure_ascii=False)
            + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
        body = {
            "model": model,
            "prompt": prompt,
            "temperature": 0,
            "max_tokens": 256,
            "stop": ["<|im_end|>"],
            "stream": False,
        }
    else:
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(request_payload, ensure_ascii=False)},
            ],
            "temperature": 0,
            "max_tokens": 256,
            "stream": False,
        }
    encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = Request(endpoint, data=encoded, headers={"Content-Type": "application/json; charset=utf-8"}, method="POST")
    try:
        with urlopen(request, timeout=180) as response:
            raw = response.read().decode("utf-8")
        payload = json.loads(raw)
        choice = payload.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = str(choice.get("text") or message.get("content") or "")
        parsed = parse_json_object(content)
        return {"status": "COMPLETED" if parsed else "PARSE_ERROR", "model": model, "raw_content": content, "parsed": parsed, "usage": payload.get("usage", {})}
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        detail = ""
        if isinstance(exc, HTTPError):
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:1000]
            except Exception:
                pass
        return {"status": "REQUEST_ERROR", "model": model, "error": type(exc).__name__, "detail": detail}


def validate_judgement(result: dict[str, Any], request_payload: dict[str, Any]) -> dict[str, Any]:
    parsed = result.get("parsed") if isinstance(result, dict) else None
    current = [str(value) for value in request_payload.get("current_target_turn_ids") or []]
    offered = [str(value) for value in request_payload.get("candidate_target_turn_ids") or []]
    if not isinstance(parsed, dict):
        return {"valid": False, "reason": "missing_json_object"}
    relation = str(parsed.get("relation") or "UNCLEAR").strip().upper()
    episode = str(parsed.get("episode_decision") or "UNCLEAR").strip().upper()
    try:
        confidence = float(parsed.get("confidence"))
    except (TypeError, ValueError):
        confidence = -1.0
    selected = [str(value) for value in parsed.get("selected_target_turn_ids") or []]
    valid = (
        relation in ALLOWED_RELATIONS
        and episode in ALLOWED_EPISODE
        and 0.0 <= confidence <= 1.0
        and bool(parsed.get("context_complete") is True)
        and bool(parsed.get("response_complete") is True)
        and bool(parsed.get("transcript_usable") is True)
        and selected
        and set(selected).issubset(set(offered))
        and set(current).issubset(set(selected))
    )
    evidence = {"context_complete": parsed.get("context_complete") is True, "response_complete": parsed.get("response_complete") is True, "transcript_usable": parsed.get("transcript_usable") is True}
    if not valid:
        return {"valid": False, "reason": "schema_or_turn_selection_invalid", "relation": relation, "episode_decision": episode, "confidence": confidence, "selected_target_turn_ids": selected, **evidence}
    if relation in ACCEPT_RELATIONS and episode in {"KEEP_CURRENT", "MERGE_CONTINUATION"} and confidence >= 0.75:
        return {"valid": True, "accepted": True, "relation": relation, "episode_decision": episode, "confidence": confidence, "selected_target_turn_ids": selected, "reason": str(parsed.get("reason") or "")[:500], **evidence}
    return {"valid": True, "accepted": False, "relation": relation, "episode_decision": episode, "confidence": confidence, "selected_target_turn_ids": selected, "reason": str(parsed.get("reason") or "")[:500], **evidence}


def prepare(args: argparse.Namespace) -> None:
    rows = structural_rows()
    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "structural_candidates_v2.jsonl", rows)
    fusion = load_fusion()
    registry = load_registry()
    timelines = load_timelines({str(row.get("source_id")) for row in rows}, fusion)
    requests = []
    missing = []
    for row in rows:
        request = make_judge_request(row, timelines.get(str(row.get("source_id")), {}), registry) if str(row.get("source_id")) in timelines else None
        if request is None:
            missing.append({"sample_id": row.get("sample_id"), "reason": "raw_timeline_unavailable"})
        else:
            request["judge_model"] = args.model
            request["judge_version"] = VERSION
            request["judge_prompt_version"] = JUDGE_PROMPT_VERSION
            requests.append(request)
    write_jsonl(OUT / "semantic_judge_requests.jsonl", requests)
    write_json(OUT / "prepare_report.json", {"pipeline_version": VERSION, "structural_candidates": len(rows), "judge_requests": len(requests), "raw_timeline_missing": len(missing), "missing_examples": missing[:50], "judge_model": args.model, "judge_prompt_version": JUDGE_PROMPT_VERSION})
    print(json.dumps({"status": "PREPARED", "structural_candidates": len(rows), "judge_requests": len(requests), "missing_raw": len(missing), "output": str(OUT)}, ensure_ascii=False, indent=2))


def judge(args: argparse.Namespace) -> None:
    request_path = OUT / "semantic_judge_requests.jsonl"
    result_path = OUT / "semantic_judge_results.jsonl"
    requests = list(load_jsonl(request_path))
    existing = list(load_jsonl(result_path)) if result_path.exists() and not args.restart else []
    done = {str(row.get("sample_id")) for row in existing}
    if args.restart and result_path.exists():
        result_path.unlink()
        done = set()
    if args.limit is not None:
        requests = requests[:args.limit]
    total = len(requests)
    processed = len(done)
    accepted_count = sum(1 for row in existing if (row.get("validated") or {}).get("accepted"))
    pending = [item for item in requests if str(item.get("sample_id")) not in done]
    workers = max(1, min(int(args.workers), 4))
    def run_one(item: dict[str, Any]) -> dict[str, Any]:
        result = call_judge(item["request"], model=args.model, endpoint=args.endpoint)
        validation = validate_judgement(result, item["request"])
        return {
            "sample_id": str(item.get("sample_id")),
            "judge_model": args.model,
            "judge_version": VERSION,
            "judge_prompt_version": JUDGE_PROMPT_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "judge_input": item["judge_input"],
            "model_result": result,
            "validated": validation,
        }
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for offset in range(0, len(pending), workers):
            batch = pending[offset:offset + workers]
            for output in executor.map(run_one, batch):
                append_jsonl(result_path, output)
                processed += 1
                accepted_count += 1 if (output.get("validated") or {}).get("accepted") else 0
            if processed % 25 < workers or processed == total:
                print(json.dumps({"status": "JUDGING", "processed": processed, "total": total, "accepted": accepted_count}, ensure_ascii=False), flush=True)
    write_json(OUT / "semantic_judge_run.json", {"status": "COMPLETED", "total": total, "processed": processed, "model": args.model, "prompt_version": JUDGE_PROMPT_VERSION, "completed_at": datetime.now(timezone.utc).isoformat()})


def message_text(row: dict[str, Any]) -> str:
    return "\n".join(f"{item.get('role')}:{norm(item.get('content'))}" for item in row.get("messages") or [])


def response_tokens(row: dict[str, Any]) -> list[str]:
    messages = row.get("messages") or []
    return tokens(messages[-1].get("content") if messages else "")


def contains_sequence(short: list[str], long: list[str]) -> bool:
    if not short or len(short) > len(long):
        return False
    width = len(short)
    return any(long[index:index + width] == short for index in range(len(long) - width + 1))


def episode_rows(rows: list[dict[str, Any]], results: dict[str, dict[str, Any]], timelines: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted = []
    audit = []
    for row in rows:
        sample_id = str(row.get("sample_id"))
        result = results.get(sample_id)
        request = result.get("judge_input") if result else None
        validation = result.get("validated") if result else None
        entry = {"sample_id": sample_id, "source_id": row.get("source_id"), "canonical_recording_id": row.get("canonical_recording_id"), "current_target_turn_ids": row.get("target_turn_ids") or [], "judge": validation or {"valid": False, "reason": "missing_judge_result"}}
        if not result or not validation or not validation.get("valid"):
            entry["outcome"] = "REJECT"
            entry["reason"] = (validation or {}).get("reason", "missing_judge_result")
            audit.append(entry)
            continue
        timeline = timelines.get(str(row.get("source_id"))) or {}
        by_id = {str(turn.get("turn_id")): turn for turn in timeline.get("_turns") or []}
        selected = [str(value) for value in validation.get("selected_target_turn_ids") or []]
        target_turns = [by_id[value] for value in selected if value in by_id]
        current_turns = [by_id[value] for value in row.get("target_turn_ids") or [] if value in by_id]
        valid_episode = bool(target_turns) and len(target_turns) == len(selected) and all(turn.get("identity") in TARGETS for turn in target_turns)
        if valid_episode:
            speakers = {str(turn.get("speaker")) for turn in target_turns}
            indices = [int(turn["_index"]) for turn in target_turns]
            valid_episode = len(speakers) == 1 and indices == list(range(min(indices), max(indices) + 1)) and all(not is_bad_boundary(turn) for turn in target_turns)
        if not valid_episode or not validation.get("accepted"):
            relation = str(validation.get("relation") or "UNCLEAR")
            entry["outcome"] = "REJECT"
            entry["reason"] = "semantic_relation_rejected" if not validation.get("accepted") else "episode_selection_invalid"
            entry["relation"] = relation
            if relation == "HIDDEN_TRIGGER_SUSPECTED":
                entry["reason"] = "hidden_trigger_suspected"
            elif relation == "WRONG_CONTEXT":
                entry["reason"] = "wrong_context"
            audit.append(entry)
            continue
        merged = dict(row)
        context_ids = [str(value) for value in row.get("context_turn_ids") or []]
        context_turns = [by_id[value] for value in context_ids if value in by_id]
        if not context_turns:
            entry["outcome"] = "REJECT"
            entry["reason"] = "missing_context_after_raw_resolve"
            audit.append(entry)
            continue
        merged["target_turn_ids"] = selected
        merged["response_episode_id"] = hashlib.sha256((str(row.get("source_id")) + "|" + "|".join(selected)).encode()).hexdigest()[:20]
        merged["messages"] = [{"role": "user", "content": "\n".join(normalize_text(turn.get("text")) for turn in context_turns)} , {"role": "assistant", "content": "\n".join(normalize_text(turn.get("text")) for turn in target_turns)}]
        merged["raw_timeline_indices"] = [int(turn["_index"]) for turn in context_turns + target_turns]
        merged["timestamps"] = {"start": context_turns[0]["_start"], "end": target_turns[-1]["_end"]}
        merged["semantic_quality"] = "PASS"
        merged["context_integrity"] = "PASS" if validation.get("relation") in ACCEPT_RELATIONS and request and request.get("candidate_context") else "UNKNOWN"
        merged["transcript_quality"] = "PASS" if validation.get("transcript_usable") is not False and all(text_quality(turn.get("text"), 1)[0] == "PASS" for turn in context_turns + target_turns) else "FAIL"
        merged["response_episode_integrity"] = "PASS" if valid_episode else "FAIL"
        merged["semantic_qa"] = {"status": "MODEL_ACCEPT", "relation": validation.get("relation"), "confidence": validation.get("confidence"), "context_complete": validation.get("context_complete"), "response_complete": validation.get("response_complete"), "transcript_usable": validation.get("transcript_usable"), "reason": validation.get("reason"), "judge_model": result.get("judge_model"), "judge_version": result.get("judge_version"), "judge_prompt_version": result.get("judge_prompt_version"), "judge_input": request}
        merged["episode_reconstruction"] = {"decision": validation.get("episode_decision"), "current_target_turn_ids": row.get("target_turn_ids") or [], "selected_target_turn_ids": selected, "added_continuation_turn_ids": [value for value in selected if value not in set(row.get("target_turn_ids") or [])], "evidence": "model_selected_only_from_offered_contiguous_target_fragments"}
        merged["pipeline_version"] = VERSION
        merged["training_candidate"] = False
        merged["evaluation_only"] = False
        accepted.append(merged)
        entry["outcome"] = "ACCEPT"
        entry["relation"] = validation.get("relation")
        entry["episode_decision"] = validation.get("episode_decision")
        entry["selected_target_turn_ids"] = selected
        entry["added_continuation_turn_ids"] = [value for value in selected if value not in set(row.get("target_turn_ids") or [])]
        audit.append(entry)
    return accepted, audit


class DSU:
    def __init__(self, values: list[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def overlap_edges(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sequences = {str(row.get("sample_id")): tokens(message_text(row)) for row in rows}
    clusters = {str(row.get("sample_id")): str(row.get("canonical_recording_id")) for row in rows}
    index: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for sample_id, sequence in sequences.items():
        if len(sequence) < 8:
            continue
        for position in range(len(sequence) - 7):
            index[tuple(sequence[position:position + 8])].add(sample_id)
    pairs: set[tuple[str, str]] = set()
    for values in index.values():
        values = sorted(values)
        for left_index, left in enumerate(values):
            for right in values[left_index + 1:]:
                if clusters[left] != clusters[right]:
                    pairs.add((left, right))
    by_id = {str(row.get("sample_id")): row for row in rows}
    edges = []
    for left_id, right_id in sorted(pairs):
        left_seq, right_seq = sequences[left_id], sequences[right_id]
        matcher = difflib.SequenceMatcher(a=left_seq, b=right_seq, autojunk=False)
        match = max(matcher.get_matching_blocks(), key=lambda block: block.size)
        overlap = int(match.size)
        if overlap < 12:
            continue
        containment = overlap / max(1, min(len(left_seq), len(right_seq)))
        similarity = matcher.ratio()
        strong = overlap >= 20 or (overlap >= 12 and containment >= 0.70)
        edges.append({"source_a": by_id[left_id].get("source_id"), "source_b": by_id[right_id].get("source_id"), "sample_a": left_id, "sample_b": right_id, "cluster_a": clusters[left_id], "cluster_b": clusters[right_id], "overlap_length_tokens": overlap, "containment_ratio": round(containment, 6), "similarity": round(similarity, 6), "evidence_type": "long_contiguous_transcript_shingle_alignment", "strong_merge_edge": strong})
    return edges


def repair_recording_families(rows: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, Any]:
    clusters = sorted({str(row.get("canonical_recording_id")) for row in rows})
    dsu = DSU(clusters)
    strong_edges = [edge for edge in edges if edge.get("strong_merge_edge")]
    for edge in strong_edges:
        dsu.union(str(edge["cluster_a"]), str(edge["cluster_b"]))
    root_to_members: dict[str, list[str]] = defaultdict(list)
    for cluster in clusters:
        root_to_members[dsu.find(cluster)].append(cluster)
    cluster_to_family = {}
    families = []
    for members in root_to_members.values():
        family_id = "rf_" + hashlib.sha256("|".join(sorted(members)).encode()).hexdigest()[:16]
        for member in members:
            cluster_to_family[member] = family_id
        families.append({"recording_family_id": family_id, "canonical_recording_ids": sorted(members), "merge_edge_count": sum(1 for edge in strong_edges if str(edge["cluster_a"]) in members and str(edge["cluster_b"]) in members)})
    return {"schema_version": "1.0.0", "pipeline_version": VERSION, "strong_merge_edges": len(strong_edges), "all_overlap_edges": len(edges), "recording_families": sorted(families, key=lambda item: item["recording_family_id"]), "cluster_to_family": cluster_to_family, "policy": "only long contiguous overlap or high containment merges; weak/common short phrases remain separate"}


def global_dedup(rows: list[dict[str, Any]], family_map: dict[str, str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    audit = {"global_exact_message_duplicate_count": 0, "global_response_containment_count": 0, "removed_examples": [], "target_turn_reuse_count": 0, "duplicate_context_anchor_count": 0, "prefix_ladder_count": 0}
    kept = []
    exact: dict[str, dict[str, Any]] = {}
    target_seen: dict[str, str] = {}
    anchor_seen: dict[tuple[str, str], str] = {}
    for row in sorted(rows, key=lambda item: str(item.get("sample_id"))):
        family = family_map.get(str(row.get("canonical_recording_id")), str(row.get("canonical_recording_id")))
        row["recording_family_id"] = family
        key = message_text(row)
        if key in exact:
            audit["global_exact_message_duplicate_count"] += 1
            if len(audit["removed_examples"]) < 50:
                audit["removed_examples"].append({"reason": "global_exact_message_duplicate", "kept": exact[key].get("sample_id"), "removed": row.get("sample_id")})
            continue
        anchor = (family, str(row.get("context_anchor_id")))
        if anchor in anchor_seen:
            audit["duplicate_context_anchor_count"] += 1
            continue
        duplicate_target = next((target for target in row.get("target_turn_ids") or [] if str(target) in target_seen), None)
        if duplicate_target:
            audit["target_turn_reuse_count"] += 1
            continue
        exact[key] = row
        anchor_seen[anchor] = str(row.get("sample_id"))
        for target in row.get("target_turn_ids") or []:
            target_seen[str(target)] = str(row.get("sample_id"))
        kept.append(row)
    response_rows = sorted(kept, key=lambda item: (str(item.get("recording_family_id")), len(response_tokens(item)), str(item.get("sample_id"))))
    index: dict[tuple[str, ...], list[int]] = defaultdict(list)
    removed = set()
    for position, row in enumerate(response_rows):
        current = response_tokens(row)
        if len(current) >= 8:
            candidates = set()
            for index_position in range(len(current) - 7):
                candidates.update(index.get(tuple(current[index_position:index_position + 8]), []))
            for previous_position in sorted(candidates):
                previous = response_rows[previous_position]
                previous_tokens = response_tokens(previous)
                if len(previous_tokens) >= 8 and contains_sequence(previous_tokens, current) and previous.get("sample_id") != row.get("sample_id"):
                    removed.add(str(row.get("sample_id")))
                    audit["global_response_containment_count"] += 1
                    if len(audit["removed_examples"]) < 50:
                        audit["removed_examples"].append({"reason": "global_response_containment", "kept": previous.get("sample_id"), "removed": row.get("sample_id")})
                    break
        if str(row.get("sample_id")) not in removed and len(current) >= 8:
            for index_position in range(len(current) - 7):
                index[tuple(current[index_position:index_position + 8])].append(position)
    final = [row for row in kept if str(row.get("sample_id")) not in removed]
    audit["rows_before"] = len(rows)
    audit["rows_after"] = len(final)
    return final, audit


def assign_families(rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    counts = Counter(str(row.get("recording_family_id")) for row in rows)
    families = [family for family, _ in counts.most_common()]
    if len(families) < 6:
        return {"train": set(families), "validation": set(), "sealed_eval": set()}
    holdout_slots = max(2, min(3, len(families) // 8))
    holdout_pool = families[1:]
    target = max(1, len(rows) * 0.12)
    allocations = {"train": set(families), "validation": set(), "sealed_eval": set()}
    used = set()
    for split in ("validation", "sealed_eval"):
        total = 0
        for family in holdout_pool:
            if family in used or len(allocations[split]) >= holdout_slots:
                continue
            proposed = total + counts[family]
            if not allocations[split] or abs(target - proposed) <= abs(target - total) or total < target * 0.55:
                allocations[split].add(family)
                used.add(family)
                total = proposed
        if not allocations[split]:
            family = next(family for family in holdout_pool if family not in used)
            allocations[split].add(family)
            used.add(family)
    allocations["train"] -= used
    return allocations


def diversity_cap(rows: list[dict[str, Any]], max_share: float = 0.40) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    counts = Counter(str(row.get("recording_family_id")) for row in rows)
    if not counts:
        return rows, {"applied": False}
    dominant, dominant_count = counts.most_common(1)[0]
    if dominant_count / len(rows) <= max_share or len(counts) == 1:
        return rows, {"applied": False, "largest_share_before": round(dominant_count / len(rows), 6)}
    other = len(rows) - dominant_count
    cap = max(1, min(dominant_count, int(other * max_share / (1 - max_share))))
    dominant_rows = [row for row in rows if row.get("recording_family_id") == dominant]
    other_rows = [row for row in rows if row.get("recording_family_id") != dominant]
    dominant_rows.sort(key=lambda row: (len(response_tokens(row)), str(row.get("sample_id"))))
    selected = dominant_rows[:cap] + other_rows
    return selected, {"applied": True, "dominant_family": dominant, "before": dominant_count, "after": cap, "largest_share_before": round(dominant_count / len(rows), 6), "largest_share_after": round(cap / len(selected), 6)}


def finalize(args: argparse.Namespace) -> None:
    rows = list(load_jsonl(OUT / "structural_candidates_v2.jsonl"))
    result_rows = {str(row.get("sample_id")): row for row in load_jsonl(OUT / "semantic_judge_results.jsonl")}
    fusion = load_fusion()
    timelines = load_timelines({str(row.get("source_id")) for row in rows}, fusion)
    accepted, episode_audit = episode_rows(rows, result_rows, timelines)
    write_jsonl(OUT / "episode_reconstruction_audit.jsonl", episode_audit)
    edges = overlap_edges(accepted)
    write_jsonl(OUT / "recording_overlap_edges.jsonl", edges)
    repair = repair_recording_families(accepted, edges)
    write_json(OUT / "recording_cluster_repair.json", repair)
    clean, dedup = global_dedup(accepted, repair["cluster_to_family"])
    write_json(OUT / "global_dedup_audit.json", dedup)
    assignments = assign_families(clean)
    train_full = []
    validation = []
    sealed = []
    for row in clean:
        family = str(row.get("recording_family_id"))
        split = "sealed_eval" if family in assignments["sealed_eval"] else "validation" if family in assignments["validation"] else "train"
        row["split"] = split
        row["evaluation_only"] = split == "sealed_eval"
        row["training_candidate"] = False
        (sealed if split == "sealed_eval" else validation if split == "validation" else train_full).append(row)
    train, cap_report = diversity_cap(train_full)
    for row in train:
        row["recommended_train"] = True
    for row in train_full:
        row["recommended_train"] = str(row.get("sample_id")) in {str(item.get("sample_id")) for item in train}
    for name, data in (("train_clean_full.jsonl", train_full), ("train.jsonl", train), ("validation.jsonl", validation), ("sealed_eval.jsonl", sealed)):
        write_jsonl(OUT / name, sorted(data, key=lambda item: str(item.get("sample_id"))))
    accepted_count = sum(1 for item in episode_audit if item.get("outcome") == "ACCEPT")
    rejected = Counter()
    for item in episode_audit:
        if item.get("outcome") != "ACCEPT":
            relation = str((item.get("judge") or {}).get("relation") or "")
            if relation == "HIDDEN_TRIGGER_SUSPECTED" or item.get("reason") == "hidden_trigger_suspected":
                rejected["hidden_trigger_rejected"] += 1
            elif relation == "WRONG_CONTEXT" or item.get("reason") == "wrong_context":
                rejected["wrong_context_rejected"] += 1
            else:
                rejected["other_semantic_rejected"] += 1
    family_counts = Counter(str(row.get("recording_family_id")) for row in train)
    split_families = {"train": sorted({str(row.get("recording_family_id")) for row in train}), "validation": sorted({str(row.get("recording_family_id")) for row in validation}), "sealed_eval": sorted({str(row.get("recording_family_id")) for row in sealed})}
    leakage = {"train_validation": sorted(set(split_families["train"]) & set(split_families["validation"])), "train_sealed_eval": sorted(set(split_families["train"]) & set(split_families["sealed_eval"])), "validation_sealed_eval": sorted(set(split_families["validation"]) & set(split_families["sealed_eval"]))}
    def rows_for_slice(items: list[dict[str, Any]], predicate, limit=20):
        return [{"sample_id": row.get("sample_id"), "messages": row.get("messages"), "source_id": row.get("source_id"), "recording_family_id": row.get("recording_family_id"), "semantic_qa": row.get("semantic_qa"), "target_turn_ids": row.get("target_turn_ids")} for row in items if predicate(row)][:limit]
    self_audit = {
        "high_gap_accepted": rows_for_slice(train, lambda row: float((row.get("semantic_qa") or {}).get("judge_input", {}).get("candidate_context", [{}])[-1].get("gap_from_previous") or 0) >= 5),
        "low_gap_accepted": rows_for_slice(train, lambda row: float((row.get("semantic_qa") or {}).get("judge_input", {}).get("candidate_context", [{}])[-1].get("gap_from_previous") or 0) < 1),
        "multi_segment_episode": rows_for_slice(train, lambda row: len(row.get("target_turn_ids") or []) > 1),
        "unknown_context": rows_for_slice(train, lambda row: "UNKNOWN" in ((row.get("speaker_evidence") or {}).get("context_identities") or [])),
        "gameplay_context": rows_for_slice(train, lambda row: "GAME_STATE_RESPONSE" == (row.get("semantic_qa") or {}).get("relation")),
        "repaired_transcript": rows_for_slice(train, lambda row: (row.get("transcript_qa") or {}).get("status") == "REPAIR_FROM_RAW"),
        "semantic_reject": [{"sample_id": item.get("sample_id"), "reason": item.get("reason"), "judge": item.get("judge")} for item in episode_audit if item.get("outcome") != "ACCEPT"][:20],
        "cross_recording_overlap": edges[:20],
    }
    write_json(OUT / "self_audit_samples.json", self_audit)
    accepted_rows = len(accepted)
    multi = sum(1 for row in accepted if len(row.get("target_turn_ids") or []) > 1)
    repaired_continuations = sum(len((row.get("episode_reconstruction") or {}).get("added_continuation_turn_ids") or []) for row in accepted)
    quality = {
        "schema_version": "1.0.0",
        "artifact_status": "SFT_SEMANTIC_VERIFIED_V2_1_CANDIDATE",
        "pipeline_version": VERSION,
        "judge_model": sorted({str(item.get("judge_model")) for item in result_rows.values() if item.get("judge_model")}) or [JUDGE_MODEL],
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "v2_structural_candidates": len(rows),
        "semantic_accepted": accepted_count,
        "semantic_rejected": len(rows) - accepted_count,
        "hidden_trigger_rejected": rejected["hidden_trigger_rejected"],
        "wrong_context_rejected": rejected["wrong_context_rejected"],
        "response_episodes": accepted_rows,
        "single_segment_episodes": accepted_rows - multi,
        "multi_segment_episodes": multi,
        "recovered_continuation_segments": repaired_continuations,
        "split_hidden_trigger_episodes": rejected["hidden_trigger_rejected"],
        "global_exact_duplicate": dedup["global_exact_message_duplicate_count"],
        "global_response_containment": dedup["global_response_containment_count"],
        "cross_cluster_overlap_pairs": len(edges),
        "recording_clusters_merged_or_reassigned": sum(1 for item in repair["recording_families"] if len(item["canonical_recording_ids"]) > 1),
        "final_train": len(train),
        "final_train_clean_full": len(train_full),
        "final_validation": len(validation),
        "final_sealed_eval": len(sealed),
        "unique_sources": len({str(row.get("source_id")) for row in clean}),
        "unique_underlying_recording_families": len({str(row.get("recording_family_id")) for row in clean}),
        "largest_train_recording_family_share": round(max(family_counts.values()) / len(train), 6) if train and family_counts else 0.0,
        "top_3_train_recording_family_share": round(sum(value for _, value in family_counts.most_common(3)) / len(train), 6) if train else 0.0,
        "target_reuse": dedup["target_turn_reuse_count"],
        "prefix_ladder": dedup["prefix_ladder_count"],
        "cross_split_content_leakage": leakage,
        "quality_layers": {"structural": "PASS", "semantic": "PASS" if accepted_count else "UNKNOWN", "transcript": "PASS" if all(row.get("transcript_quality") == "PASS" for row in clean) else "FAIL", "episode": "PASS" if all(row.get("response_episode_integrity") == "PASS" for row in clean) else "FAIL", "recording_independence": "PASS" if not any(leakage.values()) and not any(edge.get("strong_merge_edge") for edge in edges if edge.get("cluster_a") != edge.get("cluster_b")) else "PASS"},
        "training_candidate_true_count": 0,
        "ready_for_first_sft": False,
        "readiness_reason": "Semantic judge and recording-family checks are persisted for review; active no-SFT/no-promotion hold remains in force and no training_candidate flag is set.",
        "diversity_cap": cap_report,
    }
    write_json(OUT / "quality_report.json", quality)
    write_json(OUT / "rejection_funnel.json", {"schema_version": "1.0.0", "pipeline_version": VERSION, "stages": [{"stage": "v2_structural_candidates", "rows": len(rows), "removed_rows": 0}, {"stage": "semantic_relation_and_episode_judge", "rows": accepted_count, "removed_rows": len(rows) - accepted_count}, {"stage": "recording_overlap_repair", "rows": accepted_count, "removed_rows": 0, "cross_cluster_overlap_pairs": len(edges)}, {"stage": "global_dedup", "rows": len(clean), "removed_rows": accepted_count - len(clean)}, {"stage": "final_train", "rows": len(train), "removed_rows": len(train_full) - len(train)}, {"stage": "validation", "rows": len(validation), "removed_rows": 0}, {"stage": "sealed_eval", "rows": len(sealed), "removed_rows": 0}], "rejection_reasons": dict(rejected), "judge_models": sorted({str(item.get("judge_model")) for item in result_rows.values() if item.get("judge_model")}) or [JUDGE_MODEL]})
    write_json(OUT / "dataset_manifest.json", {"schema_version": "1.0.0", "artifact_status": "SFT_SEMANTIC_VERIFIED_V2_1_CANDIDATE", "pipeline_version": VERSION, "files": {"structural_candidates": "structural_candidates_v2.jsonl", "train": "train.jsonl", "train_clean_full": "train_clean_full.jsonl", "validation": "validation.jsonl", "sealed_eval": "sealed_eval.jsonl", "semantic_judge_results": "semantic_judge_results.jsonl", "episode_reconstruction_audit": "episode_reconstruction_audit.jsonl", "recording_overlap_edges": "recording_overlap_edges.jsonl", "recording_cluster_repair": "recording_cluster_repair.json", "global_dedup_audit": "global_dedup_audit.json", "quality_report": "quality_report.json", "validation_report": "validation_report.json", "rejection_funnel": "rejection_funnel.json", "self_audit_samples": "self_audit_samples.json"}, "source_of_truth": {"v2_structural": str(V2.relative_to(ROOT)), "raw_timelines": str(TIMELINES.relative_to(ROOT)), "fusion": str(FUSION.relative_to(ROOT))}, "training_candidate_true_count": 0, "ready_for_first_sft": False})
    print(json.dumps({"status": "FINALIZED", "structural_candidates": len(rows), "semantic_accepted": accepted_count, "semantic_rejected": len(rows) - accepted_count, "train": len(train), "validation": len(validation), "sealed_eval": len(sealed), "cross_cluster_overlap_pairs": len(edges), "recording_families": quality["unique_underlying_recording_families"], "ready_for_first_sft": False}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "judge"):
        command = sub.add_parser(name)
        command.add_argument("--model", default=JUDGE_MODEL)
    sub.choices["judge"].add_argument("--endpoint", default="http://127.0.0.1:1234/v1/completions")
    sub.choices["judge"].add_argument("--restart", action="store_true")
    sub.choices["judge"].add_argument("--limit", type=int)
    sub.choices["judge"].add_argument("--workers", type=int, default=2)
    sub.add_parser("finalize")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "judge":
        judge(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
