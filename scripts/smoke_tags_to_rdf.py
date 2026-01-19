from __future__ import annotations

import argparse
import asyncio
import os
import time
from typing import Optional

import requests

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.telemetry.meta_tags import MetaTagsPayload


def _endpoint() -> Optional[str]:
    if os.getenv("RECALL_RDF_ENDPOINT_URL"):
        return os.getenv("RECALL_RDF_ENDPOINT_URL")
    graphdb_url = os.getenv("GRAPHDB_URL")
    graphdb_repo = os.getenv("GRAPHDB_REPO")
    if graphdb_url and graphdb_repo:
        return f"{graphdb_url.rstrip('/')}/repositories/{graphdb_repo}"
    return None


def _sparql(collapse_id: str) -> str:
    return f"""
    PREFIX orion: <http://conjourney.net/orion#>
    PREFIX cm: <http://orion.ai/collapse#>
    SELECT ?tag ?entity ?salience ?timestamp
    WHERE {{
      ?enrichment a orion:Enrichment ;
                  orion:collapseId "{collapse_id}" ;
                  orion:enriches ?event .
      OPTIONAL {{ ?event cm:hasTag ?tag }}
      OPTIONAL {{ ?event cm:hasEntity ?entity }}
      OPTIONAL {{ ?enrichment orion:salience ?salience }}
      OPTIONAL {{ ?enrichment orion:timestamp ?timestamp }}
    }}
    LIMIT 10
    """


async def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test: tags.enriched -> rdf-writer -> GraphDB")
    parser.add_argument("--redis", default=os.getenv("ORION_BUS_URL", "redis://localhost:6379/0"))
    parser.add_argument("--collapse-id", default=f"collapse_smoke_{int(time.time())}")
    parser.add_argument("--channel", default=os.getenv("CHANNEL_EVENTS_TAGGED", "orion:tags:enriched"))
    args = parser.parse_args()

    endpoint = _endpoint()
    if not endpoint:
        print("❌ Missing GraphDB endpoint (set RECALL_RDF_ENDPOINT_URL or GRAPHDB_URL/GRAPHDB_REPO).")
        return 2

    bus = OrionBusAsync(url=args.redis)
    await bus.connect()
    try:
        payload = MetaTagsPayload(
            service_name="smoke-tags",
            service_version="0.0.1",
            node="smoke",
            tags=["smoke-tag"],
            entities=["smoke-entity"],
            id=args.collapse_id,
            collapse_id=args.collapse_id,
            enrichment_type="tagging",
            salience=0.8,
            correlation_id=f"smoke-{int(time.time())}",
        )
        env = BaseEnvelope(
            kind="tags.enriched",
            source=ServiceRef(name="smoke-tags"),
            payload=payload.model_dump(mode="json"),
        )
        await bus.publish(args.channel, env)
    finally:
        await bus.close()

    await asyncio.sleep(2.0)

    auth = (os.getenv("RECALL_RDF_USER") or os.getenv("GRAPHDB_USER") or "admin",
            os.getenv("RECALL_RDF_PASS") or os.getenv("GRAPHDB_PASS") or "admin")
    resp = requests.post(
        endpoint,
        data=_sparql(args.collapse_id),
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
