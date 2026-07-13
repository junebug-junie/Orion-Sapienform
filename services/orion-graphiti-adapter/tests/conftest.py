import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("POSTGRES_URI", "")
os.environ.setdefault("FALKORDB_ENABLED", "false")

SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SERVICE_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))


@pytest.fixture(autouse=True)
def _reset_graphiti_core_search_stack_cache():
    """app.backends.graphiti_core._search_stack_cache is process-local, module-level state
    (see graphiti_core.py's _get_search_stack). Without a reset, one test's cached
    driver/embedder/Graphiti stack (often built against a mocked sys.modules["graphiti_core"]
    that only exists inside that test's `with patch.dict(...)` block) could leak into a
    later test keyed by the same (falkordb_uri, graph_name, embed_url) tuple."""
    from app.backends import graphiti_core as core_backend

    core_backend._search_stack_cache.clear()
    yield
    core_backend._search_stack_cache.clear()
