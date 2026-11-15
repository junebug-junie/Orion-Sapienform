#!/usr/bin/env python3
"""Dry-run fold of Orion's page:juniper against live data. Writes NOTHING to the database.

Reads the same records the nightly fold reads (orion/memory/pages/juniper.py SQL), makes the one
LLM call through llm-gateway's OpenAI passthrough on the fold route (memory_distill: the agent
seat on gpu1, leased per call), runs the deterministic cite check, and prints the page and the
check. ``--out`` also saves prompt, raw answer, page and check as JSON (the eval reads it).

    .venv/bin/python scripts/core_page_juniper_dry_run.py --out /tmp/page_juniper_dry_run.json

DSN: $ORION_PAGES_DSN, else services/orion-durable-runs/.env POSTGRES_URI with the container host
swapped for the host port (localhost:55432).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from orion.memory.pages.juniper import (  # noqa: E402
    CURRENT_PAGE_SQL, EPISODE_MEMORY_SQL, MAX_VERSION_SQL, PAGE_JUNIPER, PERSON_FACT_SQL, PRODUCED_BY,
    build_fold_prompt, check_page, current_page_from_row, handle_map, sources_from_rows,
)


def default_dsn() -> str:
    if os.environ.get("ORION_PAGES_DSN"):
        return os.environ["ORION_PAGES_DSN"]
    for base in (REPO, Path("/mnt/scripts/Orion-Sapienform")):
        env = base / "services/orion-durable-runs/.env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("POSTGRES_URI="):
                    return line.split("=", 1)[1].strip().replace("@orion-athena-sql-db:5432", "@localhost:55432")
    raise SystemExit("no DSN: set ORION_PAGES_DSN")


def load(dsn: str):
    import psycopg
    from psycopg.rows import dict_row

    params = {"cid": PAGE_JUNIPER, "by": PRODUCED_BY}
    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        people = conn.execute(PERSON_FACT_SQL).fetchall()
        episodes = conn.execute(EPISODE_MEMORY_SQL).fetchall()
        current = conn.execute(CURRENT_PAGE_SQL, params).fetchone()
        top = conn.execute(MAX_VERSION_SQL, params).fetchone()
    return people, episodes, current, int((top or {}).get("v") or 0)


def call_gateway(url: str, route: str, prompt: str, *, max_tokens: int, timeout: float,
                 temperature: float = 0.3, queue_wait: float = 1800.0) -> dict:
    import httpx

    started = time.monotonic()
    # The HTTP passthrough waits at most ~60 s for a pool lease; a busy seat answers 503 (reason
    # "deadline"). Wait and ask again, up to ``queue_wait`` seconds in all, like the bus path's longer budget.
    while True:
        resp = httpx.post(f"{url.rstrip('/')}/v1/chat/completions", timeout=timeout, json={
            "model": route, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": temperature, "chat_template_kwargs": {"enable_thinking": False}})
        if resp.status_code == 503 and time.monotonic() - started < queue_wait:
            time.sleep(15.0)
            continue
        break
    resp.raise_for_status()
    body = resp.json()
    choice = (body.get("choices") or [{}])[0]
    return {"text": (choice.get("message") or {}).get("content") or "", "model": body.get("model"),
            "finish_reason": choice.get("finish_reason"), "usage": body.get("usage"),
            "latency_ms": int((time.monotonic() - started) * 1000)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gateway", default=os.environ.get("ORION_GATEWAY_URL", "http://localhost:8210"))
    ap.add_argument("--route", default="memory_distill")
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--attempts", type=int, default=1, help="re-ask on a failed check, like the nightly window")
    ap.add_argument("--out", help="save prompt, raw answer, page and check as JSON")
    args = ap.parse_args()
    if args.route.startswith(("fast", "quick", "metacog")):
        raise SystemExit("the page fold never runs on the fast or metacog lanes")

    people, episodes, current_row, top = load(default_dsn())
    sources = sources_from_rows(people, episodes)
    current = current_page_from_row(current_row)
    handles = handle_map(sources)
    prompt = build_fold_prompt(sources, current)
    print(f"sources={len(sources)} (person_fact={sum(s.kind == 'person_fact' for s in sources)}, "
          f"episode_memory={sum(s.kind == 'episode_memory' for s in sources)}) prompt_chars={len(prompt)} "
          f"current_version={current.version if current else None}", file=sys.stderr)
    tries = []
    for attempt in range(1, max(1, args.attempts) + 1):
        reply = call_gateway(args.gateway, args.route, prompt, max_tokens=args.max_tokens, timeout=args.timeout)
        check = check_page(reply["text"], handles)
        tries.append({"attempt": attempt, "reply": reply, "check": check.as_dict(), "errors": check.errors})
        print(f"attempt {attempt}: model={reply['model']} latency_ms={reply['latency_ms']} "
              f"finish={reply['finish_reason']} check={json.dumps(check.as_dict())}", file=sys.stderr)
        if check.ok:
            break
        for err in check.errors[:10]:
            print(f"  - {err}", file=sys.stderr)
    print("\n=== page:juniper (would be version %d; NOT written) ===\n" % (top + 1))
    print(check.clean_text)
    print("\n=== check ===")
    print(json.dumps({**check.as_dict(), "errors": check.errors}, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps({
            "would_be_version": top + 1, "sources": len(sources), "prompt": prompt, "attempts": tries,
            "page_text": check.clean_text, "stored_text": check.stored_text, "refs": check.refs,
            "check": {**check.as_dict(), "errors": check.errors}}, indent=2, default=str))
    return 0 if check.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
