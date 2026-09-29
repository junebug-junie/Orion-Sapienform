#!/usr/bin/env python3
"""Live smoke: ask orion-hub for reading_results over the real bus.

    ORION_BUS_URL=redis://100.92.216.81:6379/0 python scripts/smoke_introspect.py --limit 3
    ORION_BUS_URL=redis://100.92.216.81:6379/0 python scripts/smoke_introspect.py --query "graphics cards"

Read-only. Exit 0 = coherent answer, 1 = degenerate answer (a verified read with
no text, or an empty recent window), 2 = answer unknown.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

from orion.core.bus.async_service import OrionBusAsync
from orion.introspect.tools import IntrospectTools, IntrospectUnknownError
from orion.schemas.introspect import IntrospectToolBindingV1


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url")
    parser.add_argument("--query")
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()
    bus_url = os.environ.get("ORION_BUS_URL")
    if not bus_url:
        print("ORION_BUS_URL is required (redis://<tailscale-node-ip>:6379/0)", file=sys.stderr)
        return 2
    binding = IntrospectToolBindingV1(
        invocation_context="unified_chat", parent_run_id="smoke-introspect",
        parent_trace_id="smoke-introspect", memory_allowed=False,
    )
    if args.query:
        arguments = {"query": args.query, "limit": args.limit}
    elif args.url:
        arguments = {"url": args.url}
    else:
        arguments = {"limit": args.limit}
    bus = OrionBusAsync(bus_url)
    try:
        await bus.connect()
    except Exception as exc:
        print(f"UNKNOWN: bus unreachable ({type(exc).__name__})", file=sys.stderr)
        return 2
    try:
        result = await IntrospectTools(bus, binding).invoke("reading_results", arguments)
    except IntrospectUnknownError as exc:
        print(f"UNKNOWN: {exc}", file=sys.stderr)
        return 2
    finally:
        await bus.close()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    degenerate = [i["id"] for i in result["items"] if i["extra"].get("source_read") and not i["text"]]
    if degenerate:
        print(f"DEGENERATE: source_read=true with empty text: {degenerate}", file=sys.stderr)
        return 1
    unscored = [i["id"] for i in result["items"] if args.query and "similarity" not in i["extra"]]
    if unscored:
        print(f"DEGENERATE: query hits without similarity: {unscored}", file=sys.stderr)
        return 1
    if not args.url and not args.query and result["total_available"] == 0:
        print("DEGENERATE: recent window is empty; a responder that always answers [] passes nothing else",
              file=sys.stderr)
        return 1
    print(f"OK items={len(result['items'])} total_available={result['total_available']} as_of={result['as_of']}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
