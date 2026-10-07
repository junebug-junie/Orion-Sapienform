"""Read-only full-cache scan; prints counts and a coverage receipt, never graph text."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from orion.graph.falkor_client import RedisGraphQueryClient
from orion.substrate.falkor_store import FalkorSubstrateStore, FalkorSubstrateStoreConfig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uri", required=True)
    parser.add_argument("--graph", default="orion_substrate")
    parser.add_argument("--page-size", type=int, default=1000)
    args = parser.parse_args()
    client = RedisGraphQueryClient(uri=args.uri, graph_name=args.graph, read_only=True,
                                   socket_timeout=30, socket_connect_timeout=5)
    store = FalkorSubstrateStore(FalkorSubstrateStoreConfig(
        uri=args.uri, graph_name=args.graph, hydration_page_size=args.page_size), client=client)
    # Avoid silently retrying a failed construction scan via snapshot().
    receipt = store.last_scan_receipt
    print(json.dumps(asdict(receipt), indent=2))
    return 0 if receipt.complete and not receipt.stale else 1


if __name__ == "__main__":
    raise SystemExit(main())
