#!/usr/bin/env python3
"""
Debug Recall Pipeline (Local)
-----------------------------
Runs the recall fusion logic directly without the Orion Bus.
This helps isolate if the crash is in the logic (Vector/SQL/RDF) or the Transport.

Usage:
    export PYTHONPATH=$PYTHONPATH:$(pwd)
    python scripts/debug_recall_pipeline.py "what happened regarding the database?"
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add root to path so we can import services
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# Add the service app directory to path so relative imports in worker.py work
SERVICE_APP = ROOT / "services" / "orion-recall" / "app"
sys.path.insert(0, str(SERVICE_APP))

# Mock settings if needed, or rely on .env
os.environ.setdefault("RECALL_ENABLE_VECTOR", "true")
os.environ.setdefault("RECALL_ENABLE_SQL_TIMELINE", "true")
os.environ.setdefault("LOG_LEVEL", "DEBUG")

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("debug-recall")

try:
    # Import the worker logic directly
    # Note: We import from the module path relative to root
    from services.orion_recall.app.worker import process_recall
    from orion.core.contracts.recall import RecallQueryV1
except ImportError as e:
    print("Import Error: Try running this from the repo root.")
    print(f"Details: {e}")
    # Fallback for complex pathing
    sys.path.append(str(ROOT / "services" / "orion-recall"))
    from app.worker import process_recall
    from orion.core.contracts.recall import RecallQueryV1

async def main():
    if len(sys.argv) < 2:
        query_text = "project timeline status"
    else:
        query_text = sys.argv[1]

    print(f"🔎 Running Process Recall for: '{query_text}'")
    
    q = RecallQueryV1(
        fragment=query_text,
        verb="debug",
        profile="reflect.v1", # Tries all backends
        session_id="debug-session",
        node_id="debug-cli"
    )

    try:
        # Calls the exact same function the service worker calls
        bundle, decision = await process_recall(q, corr_id="debug-local-run")
        
        print("\n✅ SUCCESS: Pipeline finished.")
        print(f"   Latency: {decision.latency_ms}ms")
        print(f"   Backends: {decision.backend_counts}")
        print(f"   Items Retrieved: {len(bundle.items)}")
        
        for item in bundle.items:
            print(f"   - [{item.source}] {item.snippet[:80]}...")

    except Exception as e:
        print("\n❌ CRASH DETECTED")
        print("This exception is likely what is killing your service thread:")
        print("-" * 40)
        import traceback
        traceback.print_exc()
        print("-" * 40)

if __name__ == "__main__":
    asyncio.run(main())
