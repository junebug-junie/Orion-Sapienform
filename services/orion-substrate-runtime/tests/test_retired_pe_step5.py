"""PE suite rebuild step 5 (2026-11): node:substrate.bus_synaptic and the
prediction-error half of node:substrate.route are retired end to end.

Neither was a forecast (bus_synaptic: fraction of bus edges with |z| >= 3, a
lull-ending detector; route: categorical decision-mismatch rate). These gates
keep the producers gone and keep a lingering stored value (FalkorDB node,
7-day history rows) out of this service's prediction-error readers.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

_SVC = Path(__file__).resolve().parents[1]
_REPO = Path(__file__).resolve().parents[3]
for _p in (_REPO, _SVC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import app.settings as settings_mod  # noqa: E402
import app.worker as worker_module  # noqa: E402
from app.worker import BiometricsSubstrateWorker  # noqa: E402
from orion.substrate.attention_self_model import ACTIVE_INFERENCE_DOMAINS  # noqa: E402
from orion.substrate.prediction_error_freshness import (  # noqa: E402
    RETIRED_PREDICTION_ERROR_NODE_IDS,
)

NOW = datetime(2026, 11, 11, 12, 0, tzinfo=timezone.utc)


def test_bus_synaptic_producer_is_gone() -> None:
    for name in (
        "_bus_synaptic_tick",
        "_bus_synaptic_tick_loop",
        "_get_bus_synaptic_client",
        "_BUS_SYNAPTIC_EDGE_QUERIES",
    ):
        assert not hasattr(BiometricsSubstrateWorker, name), name
    assert not hasattr(worker_module, "bus_synaptic_prediction_error")
    assert not hasattr(worker_module, "route_prediction_error")
    fields = set(settings_mod.Settings.model_fields)
    assert not {f for f in fields if "bus_synaptic" in f or f == "falkordb_bus_graph"}


def test_worker_source_writes_neither_retired_node() -> None:
    source = (_SVC / "app" / "worker.py").read_text(encoding="utf-8")
    for node_id in RETIRED_PREDICTION_ERROR_NODE_IDS:
        assert f'node_id="{node_id}"' not in source, node_id
    assert "transport_thresholds" not in source


def test_active_inference_domains_match_the_worker_domain_map() -> None:
    assert set(ACTIVE_INFERENCE_DOMAINS) == {"execution", "biometrics", "chat"}
    assert set(worker_module._PREDICTION_ERROR_DOMAIN_NODE_IDS.values()) == set(
        ACTIVE_INFERENCE_DOMAINS
    )
    assert not RETIRED_PREDICTION_ERROR_NODE_IDS & set(worker_module._PREDICTION_ERROR_DOMAIN_NODE_IDS)


def _node(node_id: str, value: float, age_sec: float = 5.0) -> SimpleNamespace:
    return SimpleNamespace(
        node_id=node_id,
        metadata={"prediction_error": value},
        temporal=SimpleNamespace(observed_at=NOW - timedelta(seconds=age_sec)),
    )


def test_pe_magnitudes_skip_retired_nodes_and_their_history(monkeypatch) -> None:
    """A resurrected/lingering retired node gets no history row and no
    magnitude, and its 7-day history rows are dropped at seed."""
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    settings_mod._settings = None
    worker = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    worker._settings = settings_mod.get_settings()
    worker._store = MagicMock()
    worker._store.fetch_prediction_error_history.return_value = [
        ("node:substrate.bus_synaptic", NOW - timedelta(hours=1), 0.9),
        ("node:substrate.route", NOW - timedelta(hours=1), 0.5),
        ("node:substrate.chat", NOW - timedelta(hours=1), 0.1),
    ]
    worker._store.prune_prediction_error_history.return_value = 0

    nodes = [
        _node("node:substrate.bus_synaptic", 0.9),
        _node("node:substrate.route", 0.5),
        _node("node:substrate.chat", 0.2),
    ]
    magnitudes = worker._prediction_error_magnitudes(nodes=nodes, now=NOW)

    assert set(magnitudes) == {"node:substrate.chat"}
    assert set(worker._pe_history_cache) == {"node:substrate.chat"}
    saved = worker._store.save_prediction_error_history_samples.call_args.args[0]
    assert [n for n, _, _ in saved] == ["node:substrate.chat"]
    settings_mod._settings = None
