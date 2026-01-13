#!/usr/bin/env python3
"""
Recall Service Test Harness
---------------------------
Sends a RecallQueryV1 via the Orion Bus and prints the returned MemoryBundle.
Usage:
    python scripts/test_recall_harness.py "What did I work on yesterday?" --profile deep.graph.v1

# Basic query using default profile
python scripts/test_recall_harness.py "project timeline"

# Query with specific profile and verb
python scripts/test_recall_harness.py "failed db connection" --profile deep.graph.v1 --verb investigate

# Point to specific redis
python scripts/test_recall_harness.py "hello" --redis redis://192.168.1.50:6379/0

"""


import asyncio
import argparse
import logging
import sys
import json
import time
from uuid import uuid4
from pathlib import Path

# -----------------------------------------------------------------------------
# Path Setup (ensure we can import orion.*)
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from orion.core.bus.async_service import OrionBusAsync
    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
    from orion.core.contracts.recall import RecallQueryV1, MemoryBundleV1
except ImportError as e:
    print(f"Error importing Orion packages: {e}")
    sys.exit(1)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
RECALL_REQUEST_CHANNEL = "orion-exec:request:RecallService"
RECALL_REPLY_CHANNEL_PREFIX = "orion-exec:result:RecallService"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("recall-harness")


async def main():
    parser = argparse.ArgumentParser(description="Test Orion Recall Service via Bus")
    parser.add_argument("query", type=str, help="The text fragment/query to recall")
    parser.add_argument("--profile", type=str, default="reflect.v1", help="Recall profile")
    parser.add_argument("--verb", type=str, default="chat", help="Contextual verb")
    parser.add_argument("--redis", type=str, default="redis://localhost:6379/0", help="Redis URL")
    parser.add_argument("--timeout", type=float, default=15.0, help="Timeout in seconds")
    args = parser.parse_args()

    # 1. Setup Bus
    bus = OrionBusAsync(args.redis)
    await bus.connect()
    
    correlation_id = str(uuid4())
    reply_channel = f"{RECALL_REPLY_CHANNEL_PREFIX}:{correlation_id}"

    # 2. Construct Query
    # We set 'reply_to' so the service knows where to send the answer.
    query_payload = RecallQueryV1(
        fragment=args.query,
        verb=args.verb,
        profile=args.profile,
        session_id="harness-session",
        node_id="harness-node",
        reply_to=reply_channel 
    )

    envelope = BaseEnvelope(
        kind="recall.query.v1",
        source=ServiceRef(name="recall-harness", version="0.1.0", node="cli"),
        correlation_id=correlation_id,
        payload=query_payload.model_dump(mode="json")
    )

    try:
        logger.info(f"Sending query: '{args.query}' (Profile: {args.profile})")
        logger.info(f"Waiting for reply on: {reply_channel}")

        # 3. Use built-in RPC helper
        # This subscribes to reply_channel, publishes the envelope, and awaits the first response.
        raw_msg = await bus.rpc_request(
            request_channel=RECALL_REQUEST_CHANNEL,
            envelope=envelope,
            reply_channel=reply_channel,
            timeout_sec=args.timeout
        )

        # 4. Decode Response
        # raw_msg is a Redis dict: {'type': 'message', 'data': b'...'}
        decoded = bus.codec.decode(raw_msg["data"])

        if not decoded.ok:
            logger.error(f"Decoding failed: {decoded.error}")
            return

        _process_reply_envelope(decoded.envelope)

    except asyncio.TimeoutError:
        logger.error(f"Timed out after {args.timeout}s waiting for response.")
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
    finally:
        await bus.close()


def _process_reply_envelope(env: BaseEnvelope):
    """Inspect the envelope payload and print the memory bundle."""
    payload = env.payload

    # Check for service-level errors
    if isinstance(payload, dict) and "error" in payload:
        logger.error(f"Service returned error: {payload['error']}")
        if "details" in payload:
            print(json.dumps(payload["details"], indent=2))
        return

    # Check for valid bundle
    if "bundle" not in payload:
        logger.warning("Received payload missing 'bundle' key.")
        print(json.dumps(payload, indent=2))
        return

    # Deserialize Bundle
    try:
        bundle = MemoryBundleV1.model_validate(payload["bundle"])
        _print_bundle(bundle)
    except Exception as e:
        logger.error(f"Failed to validate MemoryBundle schema: {e}")
        print(json.dumps(payload["bundle"], indent=2))


def _print_bundle(bundle: MemoryBundleV1):
    print("\n" + "="*60)
    print(f"📦 MEMORY BUNDLE RETRIEVED ({len(bundle.items)} items)")
    print(f"⚡ Latency: {bundle.stats.latency_ms}ms")
    print(f"📊 Backend Counts: {bundle.stats.backend_counts}")
    print("="*60 + "\n")

    if not bundle.items:
        print("  (No items found)")

    for idx, item in enumerate(bundle.items, 1):
        source_tag = item.source.upper()
        print(f"[{idx}] {source_tag} | Score: {item.score:.4f} | ID: {item.id}")
        
        if item.ts:
            print(f"     Time: {time.ctime(item.ts)}")
        if item.tags:
            print(f"     Tags: {', '.join(item.tags)}")
        
        # Helper to clean newlines for compact display
        clean_snippet = item.snippet.replace("\n", " ").strip()
        if len(clean_snippet) > 120:
            clean_snippet = clean_snippet[:120] + "..."
            
        print(f"     Text: {clean_snippet}")
        print("-" * 30)

    print("\n" + "="*60)
    print("RENDERED PROMPT CONTEXT (Preview):")
    print("-" * 20)
    print(bundle.rendered[:800] + ("..." if len(bundle.rendered) > 800 else ""))
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
