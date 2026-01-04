#!/usr/bin/env python3
import sys
import asyncio
from pathlib import Path

# Ensure repo root is in path so we can import orion
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orion.cognition.hub_gateway.bus_harness import main

if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main(sys.argv[1:])))
    except KeyboardInterrupt:
        sys.exit(130)
