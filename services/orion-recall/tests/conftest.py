from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RECALL_SERVICE_ROOT = ROOT / "services" / "orion-recall"
if str(RECALL_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(RECALL_SERVICE_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import pytest


@pytest.fixture(autouse=True)
def _reset_substrate_store_singleton():
    """app.substrate_store keeps a process-level store plus retry-backoff
    state; a failure recorded by one test must not suppress the next test's
    build (or leak a fake store into it)."""
    from app import substrate_store

    substrate_store._reset_for_tests()
    yield
    substrate_store._reset_for_tests()
