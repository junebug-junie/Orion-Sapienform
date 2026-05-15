from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Optional

import requests

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from uuid import uuid4


def _query_url() -> Optional[str]:
    backend = (os.getenv("RDF_STORE_BACKEND") or "graphdb").strip().lower()
    if os.getenv("RDF_STORE_QUERY_URL"):
        return os.getenv("RDF_STORE_QUERY_URL")
    if backend == "fuseki":
        base = (os.getenv("RDF_STORE_BASE_URL") or "http://orion-athena-fuseki:3030").rstrip("/")
        ds = (os.getenv("RDF_STORE_DATASET") or "orion").strip("/")
        return f"{base}/{ds}/query"
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
    LIMIT 10
    """


async def main() -> int:
    redis_url = os.getenv("ORION_BUS_URL", "redis://localhost:6379/0")
    channel = os.getenv("CHANNEL_CHAT_HISTORY_TURN", "orion:chat:history:turn")
    session_id = "rdf-store-spike"

    endpoint = _query_url()
    if not endpoint:
        print("FAIL: missing query endpoint (set RDF_STORE_QUERY_URL or Fuseki/GraphDB vars).")
        return 2

    bus = OrionBusAsync(url=redis_url)
    await bus.connect()
    try:
        payload = {
            "id": "graphdb-replacement-smoke-001",
            "session_id": session_id,
            "prompt": "Can Orion write to the replacement RDF store?",
            "response": "Yes. This turn is stored outside GraphDB.",
            "timestamp": "2026-05-14T00:00:00Z",
            "correlation_id": "rdf-store-spike-001",
            "verb": "chat_general",
            "model": "smoke-model",
            "node": "athena",
        }
        env = BaseEnvelope(
            kind="chat.history",
            source=ServiceRef(name="smoke-chat-to-rdf-store", node=os.getenv("NODE_NAME"), version="0"),
            correlation_id=uuid4(),
            payload=payload,
        )
        await bus.publish(channel, env)
    finally:
        await bus.close()

    timeout = float(os.getenv("RDF_STORE_TIMEOUT_SEC", "10"))
    deadline = time.monotonic() + max(5.0, timeout * 3.0)
    auth_user = os.getenv("RDF_STORE_USER") or os.getenv("RECALL_RDF_USER") or os.getenv("GRAPHDB_USER") or "admin"
    auth_pass = os.getenv("RDF_STORE_PASS") or os.getenv("RECALL_RDF_PASS") or os.getenv("GRAPHDB_PASS") or "admin"
    auth = (auth_user, auth_pass)

    bindings: list[Any] = []
    while time.monotonic() < deadline:
        await asyncio.sleep(0.5)
        resp = requests.post(
            endpoint,
            data=_sparql(session_id),
            headers={
                "Content-Type": "application/sparql-query",
                "Accept": "application/sparql-results+json",
            },
            auth=auth,
            timeout=timeout,
        )
        if resp.status_code != 200:
            continue
        data = resp.json()
        bindings = data.get("results", {}).get("bindings", [])
        if bindings:
            break

    ok = False
    if bindings:
        b0 = bindings[0]
        prompt = b0.get("prompt", {}).get("value")
        response = b0.get("response", {}).get("value")
        ok = (
            prompt == "Can Orion write to the replacement RDF store?"
            and response == "Yes. This turn is stored outside GraphDB."
        )

    if ok:
        print("PASS")
        return 0

    print(
        "FAIL",
        {
            "backend": os.getenv("RDF_STORE_BACKEND"),
            "query_endpoint": endpoint,
            "dataset": os.getenv("RDF_STORE_DATASET"),
            "bindings_preview": bindings[:1],
        },
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
