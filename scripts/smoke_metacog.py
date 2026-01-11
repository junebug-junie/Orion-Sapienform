import asyncio
import os
import sys
import json
import logging
from uuid import uuid4

# Add repo root to path
sys.path.append(os.getcwd())

from orion.core.bus.async_service import OrionBusAsync, ChassisConfig
from orion.core.bus.bus_schemas import BaseEnvelope
from orion.schemas.telemetry.metacog_trigger import MetacogTriggerV1
from orion.schemas.collapse_mirror import CollapseMirrorEntryV2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smoke_metacog")

async def run_smoke():
    # Setup bus
    config = ChassisConfig(
        service_name="smoke-test",
        service_version="0.0.1",
        bus_url="redis://localhost:6379/0",
        bus_enabled=True,
    )

    # We will simulate being the equilibrium service publishing a trigger
    # AND verify we can deserialize the schemas

    logger.info("Initializing smoke test schemas...")
    trigger = MetacogTriggerV1(
        trigger_kind="baseline",
        reason="smoke_test",
        zen_state="zen",
        pressure=0.1
    )
    logger.info(f"Trigger created: {trigger.model_dump_json()}")

    entry = CollapseMirrorEntryV2(
        observer="orion",
        trigger="smoke_test",
        type="test",
        emergent_entity="smoke",
        summary="Smoke test running",
        mantra="Testing is existing",
    )
    logger.info(f"Entry created: {entry.model_dump_json()}")

    logger.info("Verifying registry imports...")
    from orion.schemas.registry import resolve
    cls = resolve("MetacogTriggerV1")
    assert cls == MetacogTriggerV1
    logger.info("Registry resolution successful.")

    logger.info("Smoke test passed (Logic & Schemas only, no live bus).")

if __name__ == "__main__":
    asyncio.run(run_smoke())
