from __future__ import annotations

import argparse
import asyncio
import os
import time
from typing import Optional

import requests

from orion.core.bus.async_service import OrionBusAsync
from orion.schemas.chat_history import ChatHistoryTurnEnvelope, ChatHistoryTurnV1


def _endpoint() -> Optional[str]:
    if os.getenv("RECALL_RDF_ENDPOINT_URL"):
        return os.getenv("RECALL_RDF_ENDPOINT_URL")
    graphdb_url = os.getenv("GRAPHDB_URL")
    graphdb_repo = os.getenv("GRAPHDB_REPO")
    if graphdb_url and graphdb_repo:
        return f"{graphdb_url.rstrip('/')}/repositories/{graphdb_repo}"
    return None


def _sparql(session_id: str) -> str:
    return f"""
    PREFIX orion: <http://conjourney.net/orion#>
    SELECT ?turn ?prompt ?response ?timestamp
    WHERE {{
      ?turn a orion:ChatTurn ;
            orion:sessionId "{session_id}" ;
            orion:prompt ?prompt ;
            orion:response ?response .
      OPTIONAL {{ ?turn orion:timestamp ?timestamp }}
    }}
    ORDER BY DESC(?timestamp)
    LIMIT 5
    """


async def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test: chat.history -> rdf-writer -> GraphDB")
    parser.add_argument("--redis", default=os.getenv("ORION_BUS_URL", "redis://localhost:6379/0"))
    parser.add_argument("--session-id", default=f"smoke-chat-{int(time.time())}")
    parser.add_argument("--channel", default=os.getenv("CHANNEL_CHAT_HISTORY_TURN", "orion:chat:history:turn"))
    args = parser.parse_args()

    endpoint = _endpoint()
    if not endpoint:
        print("❌ Missing GraphDB endpoint (set RECALL_RDF_ENDPOINT_URL or GRAPHDB_URL/GRAPHDB_REPO).")
        return 2

    bus = OrionBusAsync(url=args.redis)
    await bus.connect()
    try:
        payload = ChatHistoryTurnV1(
            id=f"chat_turn_{int(time.time())}",
            correlation_id=f"smoke-{int(time.time())}",
            source="smoke",
            prompt="smoke chat prompt",
            response="smoke chat response",
            user_id="smoke-user",
            session_id=args.session_id,
        )
        env = ChatHistoryTurnEnvelope(payload=payload)
        await bus.publish(args.channel, env)
    finally:
        await bus.close()

    await asyncio.sleep(2.0)

    auth = (os.getenv("RECALL_RDF_USER") or os.getenv("GRAPHDB_USER") or "admin",
            os.getenv("RECALL_RDF_PASS") or os.getenv("GRAPHDB_PASS") or "admin")
    resp = requests.post(
        endpoint,
        data=_sparql(args.session_id),
        headers={
            "Content-Type": "application/sparql-query",
            "Accept": "application/sparql-results+json",
        },
        auth=auth,
        timeout=10.0,
    )
    if resp.status_code != 200:
        print(f"❌ GraphDB query failed: {resp.status_code} {resp.text[:200]}")
        return 1

    data = resp.json()
    bindings = data.get("results", {}).get("bindings", [])
    print("PASS" if bindings else "FAIL", bindings)
    return 0 if bindings else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
