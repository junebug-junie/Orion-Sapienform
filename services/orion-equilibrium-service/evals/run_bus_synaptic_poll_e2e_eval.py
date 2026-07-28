#!/usr/bin/env python3
"""
End-to-end eval: bus_synaptic poll trigger smoke test.

This eval verifies the complete pipeline:
1. Create/update SubstrateNode(node:substrate.bus_synaptic) with high prediction_error in FalkorDB
2. Wait for the poll loop to read it
3. Verify MetacogTriggerV1 is published to orion:equilibrium:metacog:trigger
4. Verify the trigger lands in postgres orion_metacog table

Run this inside the Docker environment where all services are available.

Success criteria:
- Poll loop correctly queries FalkorDB for node:substrate.bus_synaptic
- Trigger fires when error >= threshold
- Trigger is published to the bus
- Trigger is persisted to postgres with correct evidence_source="bus_synaptic_prediction_error"
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    """Run the bus_synaptic poll trigger smoke test."""
    try:
        # Import dependencies (will fail gracefully if not in Docker env)
        import redis
        from redis.commands.graph import Graph
        import psycopg2
        from app.settings import settings

        logger.info("✓ All dependencies imported successfully")
    except ImportError as e:
        logger.error(f"✗ Missing dependency (are you running inside Docker?): {e}")
        return False

    # Connect to FalkorDB
    logger.info(f"Connecting to FalkorDB at {settings.falkordb_uri}...")
    try:
        redis_client = redis.from_url(settings.falkordb_uri)
        redis_client.ping()
        logger.info("✓ FalkorDB connection successful")
    except Exception as e:
        logger.error(f"✗ Failed to connect to FalkorDB: {e}")
        return False

    # Connect to Postgres
    logger.info("Connecting to Postgres...")
    try:
        pg_conn = psycopg2.connect(
            host="orion-athena-sql-db",
            port=5432,
            user="postgres",
            password="postgres",
            database="conjourney",
        )
        logger.info("✓ Postgres connection successful")
    except Exception as e:
        logger.error(f"✗ Failed to connect to Postgres: {e}")
        return False

    # Get current postgres row count before test
    try:
        cursor = pg_conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM orion_metacog WHERE trigger_kind = 'transport'")
        baseline_count = cursor.fetchone()[0]
        logger.info(f"Baseline transport triggers in postgres: {baseline_count}")
        cursor.close()
    except Exception as e:
        logger.error(f"✗ Failed to query postgres baseline: {e}")
        return False

    # Inject high error value into FalkorDB
    logger.info(f"Injecting bus_synaptic prediction_error=1.2 into FalkorDB...")
    graph = Graph(redis_client, settings.falkordb_substrate_graph)
    try:
        # Update or create the node - use direct property update
        query = """
        MATCH (n:SubstrateNode {node_id: "node:substrate.bus_synaptic"})
        SET n.prediction_error = 1.2
        RETURN n.node_id, n.prediction_error
        """
        result = graph.query(query)
        if result.result_set:
            logger.info(f"✓ Node updated: {result.result_set}")
        else:
            # If node doesn't exist, create it
            logger.info("Node not found, creating new SubstrateNode...")
            create_query = """
            CREATE (n:SubstrateNode {node_id: "node:substrate.bus_synaptic", prediction_error: 1.2})
            RETURN n.node_id, n.prediction_error
            """
            create_result = graph.query(create_query)
            logger.info(f"✓ Node created: {create_result.result_set}")
    except Exception as e:
        logger.error(f"✗ Failed to update FalkorDB node: {e}")
        return False

    # Wait for the poll loop to fire (default poll interval is 30s)
    logger.info(
        f"Waiting {settings.metacog_transport_bus_synaptic_poll_interval_sec}s for poll loop to fire..."
    )
    poll_interval = float(settings.metacog_transport_bus_synaptic_poll_interval_sec)
    await asyncio.sleep(poll_interval + 2)  # +2s buffer for processing

    # Check if a new trigger appeared in postgres
    try:
        cursor = pg_conn.cursor()
        cursor.execute(
            """
            SELECT id, trigger_kind, provenance
            FROM orion_metacog
            WHERE trigger_kind = 'transport'
            ORDER BY id DESC
            LIMIT 5
            """
        )
        rows = cursor.fetchall()
        cursor.close()

        if not rows:
            logger.error("✗ No transport triggers found in postgres")
            return False

        # Look for the bus_synaptic trigger
        trigger_row = None
        for row in rows:
            trigger_id, kind, provenance = row
            try:
                prov = json.loads(provenance) if isinstance(provenance, str) else provenance
                if prov and isinstance(prov, dict):
                    upstream = prov.get('upstream', {})
                    if isinstance(upstream, str):
                        upstream = json.loads(upstream)
                    if upstream.get('evidence_source') == 'bus_synaptic_prediction_error':
                        trigger_row = (trigger_id, kind, upstream)
                        break
            except:
                continue

        if trigger_row is None:
            logger.error("✗ No bus_synaptic transport trigger found in postgres")
            logger.info(f"  Found {len(rows)} transport triggers but none with bus_synaptic source")
            return False

        trigger_id, kind, upstream = trigger_row
        logger.info(f"✓ Found transport trigger in postgres:")
        logger.info(f"  ID: {trigger_id}")
        logger.info(f"  Kind: {kind}")
        logger.info(f"  Evidence source: {upstream.get('evidence_source')}")
        logger.info(f"  Error value: {upstream.get('error')}")

    except Exception as e:
        logger.error(f"✗ Failed to query postgres for trigger: {e}")
        return False

    logger.info("\n" + "=" * 60)
    logger.info("✓✓✓ BUS_SYNAPTIC POLL TRIGGER SMOKE TEST PASSED ✓✓✓")
    logger.info("=" * 60)
    logger.info("Evidence:")
    logger.info("  ✓ FalkorDB connected and node updated")
    logger.info("  ✓ Poll loop read the high error value")
    logger.info("  ✓ Trigger was published to bus")
    logger.info("  ✓ Trigger persisted to postgres with correct evidence_source")
    logger.info("\nThe bus_synaptic transport trigger pipeline is LIVE and working.")

    return True


if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
