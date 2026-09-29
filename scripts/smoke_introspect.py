#!/usr/bin/env python3
"""Live smoke over the real bus: reading_results via orion-hub, dreams via orion-dream.

    ORION_BUS_URL=redis://100.92.216.81:6379/0 python scripts/smoke_introspect.py --limit 3
    ORION_BUS_URL=redis://100.92.216.81:6379/0 python scripts/smoke_introspect.py --query "graphics cards"
    ORION_BUS_URL=redis://100.92.216.81:6379/0 python scripts/smoke_introspect.py --tool dreams --limit 3
    ORION_BUS_URL=redis://100.92.216.81:6379/0 python scripts/smoke_introspect.py --tool dreams --query "vision"
    ORION_BUS_URL=redis://100.92.216.81:6379/0 python scripts/smoke_introspect.py --tool dreams --dream-id dream:19

Read-only. Exit 0 = coherent answer, 1 = degenerate answer (a verified read with
no text, an empty recent window, or a named --dream-id that came back empty), 2 = answer
unknown. Bad argument combinations exit 2 via argparse.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url")
    parser.add_argument("--query")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--tool", choices=["reading_results", "dreams"], default="reading_results")
    parser.add_argument("--dream-id")
    return parser


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, dict]:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.tool == "dreams":
        if args.url:
            parser.error("--url is a reading_results argument; dreams takes --query, --dream-id, or --limit")
        if args.query and args.dream_id:
            parser.error("--query and --dream-id are mutually exclusive for --tool dreams")
    if args.query:
        arguments = {"query": args.query, "limit": args.limit}
    elif args.tool == "dreams" and args.dream_id:
        arguments = {"dream_id": args.dream_id}
    elif args.url:
        arguments = {"url": args.url}
    else:
        arguments = {"limit": args.limit}
    return args, arguments


def verdict(result: dict, args: argparse.Namespace) -> tuple[int, str]:
    degenerate = [i["id"] for i in result["items"] if i["extra"].get("source_read") and not i["text"]]
    if degenerate:
        return 1, f"DEGENERATE: source_read=true with empty text: {degenerate}"
    hollow = [i["id"] for i in result["items"] if args.tool == "dreams" and not i["text"]]
    if hollow:
        return 1, f"DEGENERATE: dream items with empty text: {hollow}"
    if args.tool == "dreams" and args.dream_id and not result["items"]:
        return 1, f"DEGENERATE: dream id not found: {args.dream_id} (a named, existing record must come back)"
    unscored = [i["id"] for i in result["items"] if args.query and "similarity" not in i["extra"]]
    if unscored:
        return 1, f"DEGENERATE: query hits without similarity: {unscored}"
    if not args.url and not args.query and not args.dream_id and result["total_available"] == 0:
        return 1, "DEGENERATE: recent window is empty; a responder that always answers [] passes nothing else"
    return 0, f"OK items={len(result['items'])} total_available={result['total_available']} as_of={result['as_of']}"


async def main() -> int:
    args, arguments = parse_args()
    # Deferred so --help and argument errors work without the bus/pydantic stack installed.
    from orion.core.bus.async_service import OrionBusAsync
    from orion.introspect.tools import IntrospectTools, IntrospectUnknownError
    from orion.schemas.introspect import IntrospectToolBindingV1

    bus_url = os.environ.get("ORION_BUS_URL")
    if not bus_url:
        print("ORION_BUS_URL is required (redis://<tailscale-node-ip>:6379/0)", file=sys.stderr)
        return 2
    binding = IntrospectToolBindingV1(
        invocation_context="unified_chat", parent_run_id="smoke-introspect",
        parent_trace_id="smoke-introspect", memory_allowed=False,
    )
    bus = OrionBusAsync(bus_url)
    try:
        await bus.connect()
    except Exception as exc:
        print(f"UNKNOWN: bus unreachable ({type(exc).__name__})", file=sys.stderr)
        return 2
    try:
        result = await IntrospectTools(bus, binding).invoke(args.tool, arguments)
    except IntrospectUnknownError as exc:
        print(f"UNKNOWN: {exc}", file=sys.stderr)
        return 2
    finally:
        await bus.close()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    code, message = verdict(result, args)
    print(message, file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
