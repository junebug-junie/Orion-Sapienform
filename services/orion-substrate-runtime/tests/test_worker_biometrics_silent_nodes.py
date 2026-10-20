"""The biometrics tick leaves silent nodes (hecate in BMC-only standby) out of
node:substrate.biometrics' prediction error, and the kill switch puts them back.

The pure pieces are covered in tests/test_biometrics_silent_node_exclusion.py; this
checks the worker actually threads the learned set into the producer call.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSTRATE_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, SUBSTRATE_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import app.worker as worker_module
from app.worker import BiometricsSubstrateWorker
from orion.schemas.biometrics_projection import NodeBiometricsProjectionV1, NodeBiometricsStateV1
from orion.schemas.grammar import GrammarEventV1, GrammarProvenanceV1

T0 = datetime(2026, 10, 20, 12, 0, 0, tzinfo=timezone.utc)
HECATE_LAST_SEEN = datetime(2026, 10, 15, 6, 31, 3, tzinfo=timezone.utc)


def _proj(athena_ts: datetime) -> NodeBiometricsProjectionV1:
    return NodeBiometricsProjectionV1(
        projection_id="node_biometrics_projection",
        generated_at=athena_ts,
        nodes={
            "athena": NodeBiometricsStateV1(node_id="athena", last_seen_at=athena_ts, pressure_hints={"strain": 0.2}),
            "hecate": NodeBiometricsStateV1(
                node_id="hecate", last_seen_at=HECATE_LAST_SEEN, pressure_hints={"strain": 0.087, "stability": 0.9996}
            ),
        },
    )


def _worker(enabled: bool) -> BiometricsSubstrateWorker:
    worker = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    worker._settings = SimpleNamespace(
        biometrics_silent_node_exclusion=enabled,
        biometrics_silent_node_cadence_multiple=6.0,
    )
    return worker


@pytest.mark.parametrize("enabled,expected", [(True, {"hecate"}), (False, set())])
def test_silent_node_helper_and_kill_switch(enabled: bool, expected: set[str]) -> None:
    worker = _worker(enabled)
    out: frozenset[str] = frozenset()
    for i in range(6):
        ts = T0 + timedelta(seconds=31 * i)
        out = worker._silent_biometrics_nodes(_proj(ts), ts)
    assert set(out) == expected


def test_tick_passes_silent_nodes_to_the_producer(monkeypatch) -> None:
    worker = _worker(True)
    worker._settings.biometrics_grammar_batch_limit = 500
    for i in range(5):
        ts = T0 + timedelta(seconds=31 * i)
        worker._silent_biometrics_nodes(_proj(ts), ts)

    now = T0 + timedelta(seconds=31 * 5)
    worker._store = MagicMock()
    worker._store.fetch_biometrics_grammar_events.return_value = [
        GrammarEventV1(
            event_id="gev-1",
            event_kind="atom_emitted",
            trace_id="biometrics.node:athena:x",
            emitted_at=now,
            provenance=GrammarProvenanceV1(source_service="orion-biometrics"),
        )
    ]
    worker._store.load_node_biometrics.return_value = _proj(now)
    worker._absence_sweep = lambda _now: []
    worker._process_events_with_poison_isolation = lambda **kw: "gev-1"
    worker._write_prediction_error_node = MagicMock()

    class _FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):  # noqa: D401
            return now

    monkeypatch.setattr(worker_module, "datetime", _FixedDatetime)
    seen: dict = {}

    def fake_pe(prev, curr, *, exclude_nodes=()):
        seen["exclude"] = set(exclude_nodes)
        return 0.1

    monkeypatch.setattr(worker_module, "biometrics_prediction_error", fake_pe)
    worker._tick()
    assert seen["exclude"] == {"hecate"}
    receipt = worker._store.save_receipt.call_args[0][0]
    assert receipt.state_deltas[0].after["definition_version"] == 2
