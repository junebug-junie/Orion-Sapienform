"""The missing-input marker survives the real write path (review finding,
2026-10-21).

FalkorSubstrateStore.upsert_node caches the DECODED durable row, and the codec
promotes only an allowlist of metadata keys. A marker missing from that list
vanished even inside substrate-runtime's own process, so the history sampler
never saw it and placeholder 0.0 rows kept landing in
substrate_node_prediction_error_history. These tests go through the real
writer (``_write_prediction_error_node``), the real store and codec (against a
fake graph that applies the generated Cypher SET clauses), and the real
sampler -- no mocked writer, no hand-built metadata.
"""

from __future__ import annotations

import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSTRATE_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, SUBSTRATE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.worker import BiometricsSubstrateWorker  # noqa: E402

from orion.substrate.falkor_store import (  # noqa: E402
    NATIVE_NODE_RETURN_FIELDS,
    FalkorSubstrateStore,
    FalkorSubstrateStoreConfig,
)
from orion.substrate.prediction_error_freshness import missing_input_reason  # noqa: E402

ROOM = "node:substrate.perception"
_MATCH_SET = re.compile(r"ON MATCH SET (.*?) REMOVE", re.S)
_CREATE_SET = re.compile(r"ON CREATE SET (.*?) ON MATCH SET", re.S)


class _Graph:
    """Applies the MERGE's own SET clause (a null value removes the property,
    as in Cypher) and returns the stored row, like FalkorDB's RETURN."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}

    def graph_query(self, cypher: str, params: dict | None = None):
        params = params or {}
        if "MERGE (n:SubstrateNode" in cypher and "node_id" in params:
            node_id = params["node_id"]
            exists = node_id in self.nodes
            clause = (_MATCH_SET if exists else _CREATE_SET).search(cypher).group(1)
            props = self.nodes.setdefault(node_id, {"node_id": node_id})
            for key in re.findall(r"n\.(\w+) = \$\w+", clause):
                if params.get(key) is None:
                    props.pop(key, None)
                else:
                    props[key] = params[key]
            return [[props.get(f) for f in NATIVE_NODE_RETURN_FIELDS]]
        return []


def _worker(monkeypatch, store) -> BiometricsSubstrateWorker:
    monkeypatch.setenv("SUBSTRATE_WRITE_PREDICTION_ERROR_NODES", "true")
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    import app.settings as settings_mod

    settings_mod._settings = None
    worker = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    worker._settings = settings_mod.get_settings()
    worker._store = MagicMock()
    worker._store.fetch_prediction_error_history.return_value = []
    worker._get_substrate_graph_store = lambda **_kw: store
    return worker


def test_marker_and_staleness_survive_the_store_and_reach_the_sampler(monkeypatch) -> None:
    graph = _Graph()
    store = FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(uri="redis://localhost:6379"), client=graph, hydrate=False
    )
    worker = _worker(monkeypatch, store)
    now = datetime.now(timezone.utc)

    worker._write_prediction_error_node(
        node_id=ROOM, error=0.0, now=now - timedelta(seconds=30), reducer_key="perception",
        extra_channels={"embedding_staleness": 1.0}, missing_reason="stale",
    )
    node = store.get_node_by_id(ROOM)
    assert missing_input_reason(node) == "stale"
    assert node.metadata["embedding_staleness"] == 1.0
    worker._prediction_error_magnitudes(nodes=[node], now=now)
    worker._store.save_prediction_error_history_samples.assert_not_called()

    # The camera comes back: a real reading clears the marker on the same node.
    worker._write_prediction_error_node(
        node_id=ROOM, error=0.31, now=now, reducer_key="perception",
        extra_channels={"embedding_staleness": 0.0}, missing_reason=None,
    )
    node = store.get_node_by_id(ROOM)
    assert missing_input_reason(node) is None
    assert "prediction_error_missing_reason" not in graph.nodes[ROOM]
    worker._prediction_error_magnitudes(nodes=[node], now=now)
    saved = worker._store.save_prediction_error_history_samples.call_args.args[0]
    assert [(n, round(v, 6)) for n, _, v in saved] == [(ROOM, 0.31)]
