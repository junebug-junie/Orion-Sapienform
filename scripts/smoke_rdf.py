# scripts/smoke_rdf.py
import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
import sys
import os

# Fix path to import from services/orion-rdf-writer which is not a proper python package name (dashes)
# We can dynamically load or just temporarily rename for import, or add to path.
# Actually, services.orion_rdf_writer is not valid because the directory is orion-rdf-writer.
# We must import from app directly after adding path.

sys.path.append(os.path.join(os.getcwd(), "services/orion-rdf-writer"))

from orion.schemas.rdf import RdfWriteRequest
from orion.schemas.telemetry.meta_tags import MetaTagsPayload
from orion.schemas.collapse_mirror import CollapseMirrorEntry
from app.rdf_builder import build_triples_from_envelope

# Setup Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smoke_rdf")

def test_schemas():
    logger.info("--- Testing Schemas ---")

    # 1. RdfWriteRequest
    req = RdfWriteRequest(
        id=str(uuid.uuid4()),
        source="smoke_test",
        triples="<http://s> <http://p> <http://o> .",
        graph="orion:smoke"
    )
    logger.info(f"RdfWriteRequest valid: {req.model_dump_json()}")

    # 2. MetaTagsPayload
    meta = MetaTagsPayload(
        id="evt_123",
        service_name="meta-tags",
        service_version="0.0.1",
        tags=["smoke", "test"],
        entities=[{"type": "TEST", "value": "Smoke"}]
    )
    logger.info(f"MetaTagsPayload valid: {meta.model_dump_json()}")

    # 3. CollapseMirrorEntry
    coll = CollapseMirrorEntry(
        observer="observer_1",
        trigger="trigger_val",
        observer_state=["idle"],
        field_resonance="high",
        type="test_event",
        emergent_entity="none",
        summary="Smoke test summary",
        mantra="test mantra"
    )
    coll.with_defaults()
    logger.info(f"CollapseMirrorEntry valid: {coll.model_dump_json()}")

    return req, meta, coll

def test_builder(req, meta, coll):
    logger.info("\n--- Testing RDF Builder Logic ---")

    # 1. Write Request
    nt, graph = build_triples_from_envelope("rdf.write.request", req)
    logger.info(f"[WriteRequest] Graph: {graph}, Triples: {nt.strip()}")
    assert graph == "orion:smoke"

    # 2. Meta Tags
    nt, graph = build_triples_from_envelope("telemetry.meta_tags", meta)
    logger.info(f"[MetaTags] Graph: {graph}, Triples (len): {len(nt)}")
    assert graph == "orion:enrichment"
    assert "http://orion.ai/collapse#hasTag" in nt

    # 3. Collapse
    nt, graph = build_triples_from_envelope("collapse.mirror.entry", coll)
    logger.info(f"[Collapse] Graph: {graph}, Triples (len): {len(nt)}")
    assert graph == "orion:collapse"
    assert "http://orion.ai/collapse#observer" in nt

def main():
    try:
        req, meta, coll = test_schemas()
        test_builder(req, meta, coll)
        logger.info("\n✅ SMOKE TEST PASSED: All schemas and builder logic verified.")
    except Exception as e:
        logger.error(f"\n❌ SMOKE TEST FAILED: {e}", exc_info=True)
        exit(1)

if __name__ == "__main__":
    main()
