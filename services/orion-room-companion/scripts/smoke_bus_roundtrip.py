#!/usr/bin/env python3
"""Live smoke: a room turn across the actual bus, end to end.

`smoke_room_turn.py` drives `run_turn` in-process, which proves the
subprocess, the credential and the session -- but deliberately never touches
Redis. This one proves the half that one skips: that a Hub-shaped
`RoomClaudeRequestV1` published on `orion:room:claude:request` is picked up by
a *running* companion container and comes back as a `RoomClaudeUtteranceV1` on
`orion:room:claude:utterance` with the correlation id intact.

That distinction matters for phase 2: Hub will only ever speak to this service
over these two channels, so a green in-process smoke says nothing about
whether the wiring Hub depends on actually works.

Run from the repo root, on the host (not inside the container):

    PYTHONPATH=. python services/orion-room-companion/scripts/smoke_bus_roundtrip.py

Requires the companion container to be up. Exits non-zero on timeout or on any
utterance that is not a real, metered, Claude-served reply.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import redis

from orion.schemas.room_claude import RoomClaudeRequestV1

DEFAULT_BUS_URL = "redis://100.92.216.81:6379/0"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bus-url", default=os.environ.get("ORION_BUS_URL", DEFAULT_BUS_URL))
    parser.add_argument("--request-channel", default="orion:room:claude:request")
    parser.add_argument("--utterance-channel", default="orion:room:claude:utterance")
    parser.add_argument("--room-id", default="bus-roundtrip")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    client = redis.Redis.from_url(args.bus_url)
    sub = client.pubsub()
    sub.subscribe(args.utterance_channel)
    # Drain the subscribe confirmation so it isn't mistaken for a reply.
    for _ in range(3):
        if sub.get_message(timeout=1) is None:
            break

    correlation_id = f"corr-roundtrip-{int(time.time())}"
    request = RoomClaudeRequestV1(
        room_id=args.room_id,
        invited_by="Juniper",
        correlation_id=correlation_id,
        prompt="One sentence: what's the difference between a companion and a tool?",
        social_memory_summary={"room": {"active_participants": ["Juniper", "Oríon"]}},
    )
    envelope = {
        "kind": "room.claude.request.v1",
        "source": {"name": "smoke-bus-roundtrip", "version": "0", "node": "athena"},
        "payload": request.model_dump(mode="json"),
    }
    subscribers = client.publish(args.request_channel, json.dumps(envelope))
    # This count includes PATTERN subscribers (orion-bus-mirror / orion-bus-tap
    # psubscribe broadly), so >1 here does NOT mean a second companion is
    # competing for the same request. The channel is declared single_consumer;
    # `PUBSUB NUMSUB` below is the number that actually speaks to that.
    exact = int(client.execute_command("PUBSUB", "NUMSUB", args.request_channel)[1])
    print(
        f"published {request.request_id} to {args.request_channel} "
        f"({subscribers} receiver(s) incl. pattern observers, {exact} exact subscriber(s))"
    )
    if subscribers == 0:
        print("FAIL: nothing is listening -- is the companion container up?")
        return 1
    if exact > 1:
        print(f"FAIL: {exact} exact subscribers on a single_consumer channel -- turns would be double-billed")
        return 1

    deadline = time.time() + args.timeout
    payload = None
    while time.time() < deadline:
        message = sub.get_message(timeout=5)
        if not message or message.get("type") != "message":
            continue
        body = json.loads(message["data"])
        candidate = body.get("payload", body)
        # Other rooms may be live on this channel; only ours counts.
        if candidate.get("request_id") == request.request_id:
            payload = candidate
            break

    if payload is None:
        print(f"FAIL: no utterance for {request.request_id} within {args.timeout}s")
        return 1

    print("\n=== utterance ===")
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    checks = {
        "utterance ok": payload.get("ok") is True,
        "text is non-empty": bool((payload.get("text") or "").strip()),
        "correlation id preserved": payload.get("correlation_id") == correlation_id,
        "responder is Claude": (payload.get("responder") or {}).get("participant_id") == "claude",
        "responder kind is peer_ai": (payload.get("responder") or {}).get("participant_kind") == "peer_ai",
        "cost was metered (> 0)": float(payload.get("cost_usd") or 0) > 0,
        "served by a claude model": "claude" in str(payload.get("model") or "").lower(),
        "claude session recorded": bool(payload.get("claude_session_id")),
    }
    print("\n=== verdict ===")
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")

    if not all(checks.values()):
        print("\nBUS ROUNDTRIP FAILED")
        return 1
    print("\nBUS ROUNDTRIP PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
