#!/usr/bin/env python3
"""Smoke Hub presence → situation brief → chat grounding (session-scoped)."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from typing import Any

try:
    import requests
except ImportError:
    print("Install requests to run smoke_presence_grounding.py", file=sys.stderr)
    raise SystemExit(1)


def _headers(session_id: str) -> dict[str, str]:
    return {"X-Orion-Session-Id": session_id, "Content-Type": "application/json"}


def _print_step(label: str, payload: Any) -> None:
    print(f"\n=== {label} ===")
    print(json.dumps(payload, indent=2, sort_keys=True)[:4000])


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke Hub presence grounding")
    parser.add_argument("--hub-base", default="http://127.0.0.1:8080", help="Hub base URL")
    parser.add_argument("--skip-chat", action="store_true", help="Only exercise presence + situation brief APIs")
    args = parser.parse_args()
    base = args.hub_base.rstrip("/")
    session_id = f"smoke-presence-{uuid.uuid4().hex[:12]}"
    headers = _headers(session_id)

    r = requests.get(f"{base}/api/presence", headers=headers, timeout=15)
    r.raise_for_status()
    solo = r.json()
    _print_step("GET /api/presence (expect solo)", solo)
    assert str(solo.get("audience_mode") or "solo") == "solo", solo

    kid_payload = {
        "audience_mode": "kid_present",
        "requestor": {"display_name": "Juniper", "relationship_to_orion": "primary_operator"},
        "companions": [
            {"display_name": "Kid", "relationship": "child", "role": "listener", "age_band": "child"},
        ],
        "source": "hub_manual",
    }
    r = requests.post(f"{base}/api/presence", headers=headers, json=kid_payload, timeout=15)
    r.raise_for_status()
    posted = r.json()
    _print_step("POST /api/presence kid_present", posted)
    assert posted.get("audience_mode") == "kid_present"

    r = requests.get(f"{base}/api/presence", headers=headers, timeout=15)
    r.raise_for_status()
    got = r.json()
    _print_step("GET /api/presence (expect kid_present)", got)
    assert got.get("audience_mode") == "kid_present"

    r = requests.get(f"{base}/api/situation/brief", headers=headers, timeout=15)
    r.raise_for_status()
    brief = r.json()
    _print_step("GET /api/situation/brief", brief)
    presence = brief.get("presence") or {}
    assert presence.get("audience_mode") == "kid_present", brief

    if args.skip_chat:
        print("\nOK: presence + situation brief smoke passed (chat skipped).")
        return 0

    chat_body = {
        "messages": [{"role": "user", "content": "explain what a GPU is to the kid listening"}],
        "mode": "brain",
        "verbs": ["chat_quick"],
        "no_write": True,
        "disable_tts": True,
    }
    r = requests.post(f"{base}/api/chat", headers=headers, json=chat_body, timeout=120)
    r.raise_for_status()
    chat = r.json()
    routing = chat.get("routing_debug") or {}
    raw_meta = {}
    raw = chat.get("raw") if isinstance(chat.get("raw"), dict) else {}
    if isinstance(raw.get("metadata"), dict):
        raw_meta = raw["metadata"]
    debug = {
        "routing_debug_presence_context_present": routing.get("presence_context_present"),
        "routing_debug_audience_mode": routing.get("audience_mode"),
        "metadata_presence_context_present": raw_meta.get("presence_context_present"),
        "metadata_audience_mode": raw_meta.get("audience_mode"),
        "metadata_situation_fragment_present": raw_meta.get("situation_fragment_present"),
        "chat_stance_debug": chat.get("chat_stance_debug") or raw_meta.get("chat_stance_debug"),
    }
    _print_step("POST /api/chat debug slice", debug)
    assert routing.get("presence_context_present") is True, routing
    assert routing.get("audience_mode") == "kid_present", routing
    stance = debug.get("chat_stance_debug") if isinstance(debug.get("chat_stance_debug"), dict) else {}
    source = stance.get("source_inputs") if isinstance(stance.get("source_inputs"), dict) else {}
    situation = source.get("situation") if isinstance(source.get("situation"), dict) else {}
    presence = situation.get("presence") if isinstance(situation.get("presence"), dict) else {}
    if presence:
        assert presence.get("audience_mode") == "kid_present", presence
    print("\nOK: full presence grounding smoke finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
