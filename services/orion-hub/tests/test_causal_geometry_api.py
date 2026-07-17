from __future__ import annotations

import json
import os
import sys
import importlib.util
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi import HTTPException

os.environ.setdefault("CHANNEL_VOICE_TRANSCRIPT", "orion:voice:transcript")
os.environ.setdefault("CHANNEL_VOICE_LLM", "orion:voice:llm")
os.environ.setdefault("CHANNEL_VOICE_TTS", "orion:voice:tts")
os.environ.setdefault("CHANNEL_COLLAPSE_INTAKE", "orion:collapse:intake")
os.environ.setdefault("CHANNEL_COLLAPSE_TRIAGE", "orion:collapse:triage")

HUB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for candidate in (str(REPO_ROOT), str(HUB_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)
hub_scripts_pkg = HUB_ROOT / "scripts" / "__init__.py"
if (
    "scripts" not in sys.modules
    or not str(getattr(sys.modules.get("scripts"), "__file__", "")).startswith(str(HUB_ROOT))
):
    spec = importlib.util.spec_from_file_location(
        "scripts",
        str(hub_scripts_pkg),
        submodule_search_locations=[str(HUB_ROOT / "scripts")],
    )
    if spec is not None and spec.loader is not None:
        module = importlib.util.module_from_spec(spec)
        sys.modules["scripts"] = module
        spec.loader.exec_module(module)

from scripts import api_routes


def _make_proposal(proposal_id: str = "proposal-cg-1"):
    """Build a minimal valid MutationProposalV1 fixture for field_topology_weight_patch.

    Only used if orion.substrate.field_topology_learned_store is importable in this test
    environment; skipped gracefully otherwise (see module-level skip check below).
    """
    from orion.core.schemas.substrate_mutation import MutationPatchV1, MutationProposalV1

    patch = MutationPatchV1(
        mutation_class="field_topology_weight_patch",
        target_surface="field_topology",
        target_ref="edge-a->edge-b",
        patch={"edge_weight_delta": 0.05},
        rollback_payload={"edge_weight_delta": 0.0},
    )
    return MutationProposalV1(
        proposal_id=proposal_id,
        lane="operational",
        mutation_class="field_topology_weight_patch",
        risk_tier="low",
        target_surface="field_topology",
        anchor_scope="orion",
        subject_ref="edge-a->edge-b",
        rationale="observed causal correlation exceeds designed weight",
        expected_effect="nudge overlay toward observed geometry",
        evidence_refs=["causal_geometry_snapshot:snap-1"],
        source_signal_ids=["signal-1"],
        source_pressure_id="pressure-1",
        patch=patch,
    )


_FIELD_TOPOLOGY_STORE_IMPORTABLE = True
try:
    from orion.substrate.field_topology_learned_store import FieldTopologyLearnedWeightsStore  # noqa: F401
except Exception:
    _FIELD_TOPOLOGY_STORE_IMPORTABLE = False


requires_field_topology_store = pytest.mark.skipif(
    not _FIELD_TOPOLOGY_STORE_IMPORTABLE,
    reason="orion.substrate.field_topology_learned_store not landed yet in this environment (Rung 2B)",
)


# --- Graceful degradation + live Postgres read: snapshot / history ---
#
# These endpoints now read from the `causal_geometry_snapshots` table (see
# services/orion-hub/scripts/api_routes.py::_causal_geometry_snapshot_payload /
# _causal_geometry_history_payload) via the same `app.state.memory_pg_pool` asyncpg pool the
# memory-cards routes use. We fake the pool/connection rather than hitting real Postgres.

@pytest.fixture
def hub_main():
    """Fresh import of scripts.main, taken *inside* the test.

    tests/conftest.py's autouse `_hub_service_isolation` fixture purges `scripts`/`scripts.*`
    from sys.modules before every test to keep the Hub package resolving correctly across test
    files. A module-level `import scripts.main` here would bind to a module object that gets
    evicted before the test body runs, so patching its `app.state` would have no effect on the
    `app` object `_causal_geometry_snapshot_payload`'s `from .main import app` actually resolves
    at call time. Importing fresh inside the test (after the autouse fixture's reset) keeps both
    in sync via sys.modules["scripts.main"].
    """

    import scripts.main as _hub_main

    return _hub_main


class _FakeConn:
    def __init__(self, *, fetchrow_result=None, fetch_result=None, raise_exc: BaseException | None = None):
        self._fetchrow_result = fetchrow_result
        self._fetch_result = fetch_result if fetch_result is not None else []
        self._raise_exc = raise_exc

    async def fetchrow(self, *_args, **_kwargs):
        if self._raise_exc is not None:
            raise self._raise_exc
        return self._fetchrow_result

    async def fetch(self, *_args, **_kwargs):
        if self._raise_exc is not None:
            raise self._raise_exc
        return self._fetch_result


class _FakeAcquireCtx:
    def __init__(self, conn: _FakeConn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *_exc_info):
        return False


class _FakePool:
    def __init__(self, conn: _FakeConn):
        self._conn = conn

    def acquire(self):
        return _FakeAcquireCtx(self._conn)


def _sample_row(snapshot_id: str = "snap-1") -> dict:
    now = datetime(2026, 7, 16, 12, 0, 0, tzinfo=timezone.utc)
    return {
        "snapshot_id": snapshot_id,
        "generated_at": now,
        "window_start": now,
        "window_end": now,
        "designed_topology_version": "v3",
        "insufficient_data": False,
        "edges": json.dumps(
            [
                {
                    "source_id": "a",
                    "target_id": "b",
                    "lag_sec": 1.0,
                    "strength": 0.5,
                    "significance": 0.9,
                    "n_samples": 10,
                    "window_start": now.isoformat(),
                    "window_end": now.isoformat(),
                }
            ]
        ),
        "divergence": json.dumps(
            [
                {
                    "source_id": "a",
                    "target_id": "b",
                    "observed_strength": 0.5,
                    "designed_weight": 0.4,
                    "delta": 0.1,
                    "status": "aligned",
                }
            ]
        ),
        "notes": json.dumps(["note-1"]),
    }


@pytest.mark.asyncio
async def test_snapshot_endpoint_degrades_gracefully_when_pool_unavailable(monkeypatch, hub_main) -> None:
    monkeypatch.setattr(hub_main.app.state, "memory_pg_pool", None, raising=False)
    payload = await api_routes.api_causal_geometry_snapshot()
    assert "source" in payload
    assert "data" in payload
    assert payload["source"]["degraded"] is True
    assert payload["source"]["kind"] == "unavailable"
    assert payload["source"]["error"] == "memory_pg_pool_unavailable"
    assert payload["data"] is None


@pytest.mark.asyncio
async def test_history_endpoint_degrades_gracefully_when_pool_unavailable(monkeypatch, hub_main) -> None:
    monkeypatch.setattr(hub_main.app.state, "memory_pg_pool", None, raising=False)
    payload = await api_routes.api_causal_geometry_history(limit=20)
    assert "source" in payload
    assert "data" in payload
    assert payload["source"]["degraded"] is True
    assert payload["source"]["kind"] == "unavailable"
    assert payload["source"]["error"] == "memory_pg_pool_unavailable"
    assert payload["data"] == []


@pytest.mark.asyncio
async def test_history_endpoint_respects_limit_query_bounds(monkeypatch, hub_main) -> None:
    monkeypatch.setattr(hub_main.app.state, "memory_pg_pool", None, raising=False)
    payload = await api_routes.api_causal_geometry_history(limit=5)
    assert payload["source"]["query"]["limit"] == 5


@pytest.mark.asyncio
async def test_snapshot_endpoint_returns_real_row_from_pool(monkeypatch, hub_main) -> None:
    conn = _FakeConn(fetchrow_result=_sample_row("snap-live"))
    monkeypatch.setattr(hub_main.app.state, "memory_pg_pool", _FakePool(conn), raising=False)

    payload = await api_routes.api_causal_geometry_snapshot()
    assert payload["source"]["degraded"] is False
    assert payload["source"]["kind"] == "postgres"
    assert payload["data"]["snapshot_id"] == "snap-live"
    assert payload["data"]["edges"] == [
        {
            "source_id": "a",
            "target_id": "b",
            "lag_sec": 1.0,
            "strength": 0.5,
            "significance": 0.9,
            "n_samples": 10,
            "window_start": "2026-07-16T12:00:00+00:00",
            "window_end": "2026-07-16T12:00:00+00:00",
        }
    ]
    assert payload["data"]["divergence"][0]["status"] == "aligned"
    assert payload["data"]["notes"] == ["note-1"]
    assert payload["data"]["designed_topology_version"] == "v3"
    assert payload["data"]["insufficient_data"] is False


@pytest.mark.asyncio
async def test_history_endpoint_returns_real_rows_from_pool(monkeypatch, hub_main) -> None:
    rows = [_sample_row("snap-1"), _sample_row("snap-2")]
    conn = _FakeConn(fetch_result=rows)
    monkeypatch.setattr(hub_main.app.state, "memory_pg_pool", _FakePool(conn), raising=False)

    payload = await api_routes.api_causal_geometry_history(limit=20)
    assert payload["source"]["degraded"] is False
    assert payload["source"]["kind"] == "postgres"
    assert [item["snapshot_id"] for item in payload["data"]] == ["snap-1", "snap-2"]


@pytest.mark.asyncio
async def test_snapshot_endpoint_empty_table_is_degraded_with_honest_message(monkeypatch, hub_main) -> None:
    conn = _FakeConn(fetchrow_result=None)
    monkeypatch.setattr(hub_main.app.state, "memory_pg_pool", _FakePool(conn), raising=False)

    payload = await api_routes.api_causal_geometry_snapshot()
    assert payload["source"]["degraded"] is True
    assert payload["source"]["error"] == "no causal-geometry snapshot persisted yet"
    assert payload["data"] is None


@pytest.mark.asyncio
async def test_history_endpoint_empty_table_is_not_degraded_returns_empty_list(monkeypatch, hub_main) -> None:
    conn = _FakeConn(fetch_result=[])
    monkeypatch.setattr(hub_main.app.state, "memory_pg_pool", _FakePool(conn), raising=False)

    payload = await api_routes.api_causal_geometry_history(limit=20)
    assert payload["source"]["degraded"] is False
    assert payload["source"]["kind"] == "postgres"
    assert payload["data"] == []


@pytest.mark.asyncio
async def test_snapshot_endpoint_undefined_table_degrades_not_500(monkeypatch, hub_main) -> None:
    from asyncpg.exceptions import UndefinedTableError

    conn = _FakeConn(raise_exc=UndefinedTableError("relation \"causal_geometry_snapshots\" does not exist"))
    monkeypatch.setattr(hub_main.app.state, "memory_pg_pool", _FakePool(conn), raising=False)

    payload = await api_routes.api_causal_geometry_snapshot()
    assert payload["source"]["degraded"] is True
    assert payload["source"]["kind"] == "unavailable"
    assert "does not exist yet" in payload["source"]["error"]
    assert payload["data"] is None


@pytest.mark.asyncio
async def test_history_endpoint_undefined_table_degrades_not_500(monkeypatch, hub_main) -> None:
    from asyncpg.exceptions import UndefinedTableError

    conn = _FakeConn(raise_exc=UndefinedTableError("relation \"causal_geometry_snapshots\" does not exist"))
    monkeypatch.setattr(hub_main.app.state, "memory_pg_pool", _FakePool(conn), raising=False)

    payload = await api_routes.api_causal_geometry_history(limit=20)
    assert payload["source"]["degraded"] is True
    assert payload["source"]["kind"] == "unavailable"
    assert "does not exist yet" in payload["source"]["error"]
    assert payload["data"] == []


@pytest.mark.asyncio
async def test_snapshot_endpoint_unexpected_exception_degrades_never_raises(monkeypatch, hub_main) -> None:
    conn = _FakeConn(raise_exc=RuntimeError("connection reset by peer"))
    monkeypatch.setattr(hub_main.app.state, "memory_pg_pool", _FakePool(conn), raising=False)

    payload = await api_routes.api_causal_geometry_snapshot()
    assert payload["source"]["degraded"] is True
    assert payload["source"]["kind"] == "unavailable"
    assert payload["source"]["error"] == "connection reset by peer"
    assert payload["data"] is None


@pytest.mark.asyncio
async def test_history_endpoint_unexpected_exception_degrades_never_raises(monkeypatch, hub_main) -> None:
    conn = _FakeConn(raise_exc=RuntimeError("connection reset by peer"))
    monkeypatch.setattr(hub_main.app.state, "memory_pg_pool", _FakePool(conn), raising=False)

    payload = await api_routes.api_causal_geometry_history(limit=20)
    assert payload["source"]["degraded"] is True
    assert payload["source"]["kind"] == "unavailable"
    assert payload["source"]["error"] == "connection reset by peer"
    assert payload["data"] == []


@pytest.mark.asyncio
async def test_snapshot_endpoint_malformed_row_degrades_never_raises(monkeypatch, hub_main) -> None:
    """A row that fails to convert (e.g. an unexpected column type) must degrade, not 500.

    The row-to-dict transform happens after a successful query, so it must still be covered by
    the same never-500 contract -- not just the query call itself.
    """

    bad_row = _sample_row("snap-bad")
    bad_row["generated_at"] = "not-a-datetime"  # .isoformat() will raise AttributeError
    conn = _FakeConn(fetchrow_result=bad_row)
    monkeypatch.setattr(hub_main.app.state, "memory_pg_pool", _FakePool(conn), raising=False)

    payload = await api_routes.api_causal_geometry_snapshot()
    assert payload["source"]["degraded"] is True
    assert payload["source"]["kind"] == "unavailable"
    assert payload["data"] is None


@pytest.mark.asyncio
async def test_history_endpoint_malformed_row_degrades_never_raises(monkeypatch, hub_main) -> None:
    bad_row = _sample_row("snap-bad")
    bad_row["generated_at"] = "not-a-datetime"  # .isoformat() will raise AttributeError
    conn = _FakeConn(fetch_result=[bad_row])
    monkeypatch.setattr(hub_main.app.state, "memory_pg_pool", _FakePool(conn), raising=False)

    payload = await api_routes.api_causal_geometry_history(limit=20)
    assert payload["source"]["degraded"] is True
    assert payload["source"]["kind"] == "unavailable"
    assert payload["data"] == []


# --- Proposals: real backing store when importable, fallback store otherwise ---


def test_proposals_endpoint_never_500s_even_when_store_unavailable(monkeypatch) -> None:
    fake_store = api_routes._UnavailableCausalGeometryProposalStore(error="boom")
    monkeypatch.setattr(api_routes, "CAUSAL_GEOMETRY_PROPOSAL_STORE", fake_store)

    payload = api_routes.api_causal_geometry_proposals(limit=50)
    assert payload["source"]["degraded"] is True
    assert payload["source"]["kind"] == "unavailable"
    assert payload["source"]["error"] == "boom"
    assert payload["data"] == []


def test_proposals_endpoint_source_honesty_reflects_degraded_store(monkeypatch) -> None:
    class FakeDegradedStore:
        def source_kind(self) -> str:
            return "fallback"

        def degraded(self) -> bool:
            return True

        def last_error(self) -> str | None:
            return "sqlite_write_failed"

        def list_pending(self, limit: int = 50):
            return []

    monkeypatch.setattr(api_routes, "CAUSAL_GEOMETRY_PROPOSAL_STORE", FakeDegradedStore())
    payload = api_routes.api_causal_geometry_proposals(limit=10)
    assert payload["source"]["degraded"] is True
    assert payload["source"]["kind"] == "fallback"
    assert payload["source"]["error"] == "sqlite_write_failed"


def test_proposals_endpoint_returns_pending_items_from_healthy_store(monkeypatch) -> None:
    class FakeHealthyStore:
        def source_kind(self) -> str:
            return "memory"

        def degraded(self) -> bool:
            return False

        def last_error(self) -> str | None:
            return None

        def list_pending(self, limit: int = 50):
            return [{"proposal_id": "p-1"}, {"proposal_id": "p-2"}][:limit]

    monkeypatch.setattr(api_routes, "CAUSAL_GEOMETRY_PROPOSAL_STORE", FakeHealthyStore())
    payload = api_routes.api_causal_geometry_proposals(limit=1)
    assert payload["source"]["degraded"] is False
    assert payload["data"] == [{"proposal_id": "p-1"}]


def test_proposals_endpoint_dumps_pydantic_items(monkeypatch) -> None:
    class FakeItem:
        def model_dump(self, mode: str = "json"):
            return {"proposal_id": "p-model", "mode": mode}

    class FakeStoreWithModelItems:
        def source_kind(self) -> str:
            return "memory"

        def degraded(self) -> bool:
            return False

        def last_error(self) -> str | None:
            return None

        def list_pending(self, limit: int = 50):
            return [FakeItem()]

    monkeypatch.setattr(api_routes, "CAUSAL_GEOMETRY_PROPOSAL_STORE", FakeStoreWithModelItems())
    payload = api_routes.api_causal_geometry_proposals(limit=50)
    assert payload["data"] == [{"proposal_id": "p-model", "mode": "json"}]


# --- Adopt / reject: operator-invoked only, correctly call through to the store ---


def test_adopt_route_calls_through_to_store_with_operator_id(monkeypatch) -> None:
    observed: dict[str, object] = {}

    class FakeStore:
        def source_kind(self) -> str:
            return "memory"

        def degraded(self) -> bool:
            return False

        def last_error(self) -> str | None:
            return None

        def adopt(self, proposal_id: str, *, operator_id: str):
            observed["proposal_id"] = proposal_id
            observed["operator_id"] = operator_id
            return {"ok": True, "edge_id": "edge-a->edge-b", "adopted_delta": 0.05}

    monkeypatch.setattr(api_routes, "CAUSAL_GEOMETRY_PROPOSAL_STORE", FakeStore())
    payload = api_routes.api_causal_geometry_proposal_adopt(
        "proposal-1",
        api_routes.CausalGeometryProposalAdoptRequest(operator_id="juniper", rationale="looks right"),
    )
    assert observed["proposal_id"] == "proposal-1"
    assert observed["operator_id"] == "juniper"
    assert payload["data"]["ok"] is True
    assert payload["source"]["degraded"] is False


def test_adopt_route_raises_503_when_store_errors(monkeypatch) -> None:
    class FakeStore:
        def source_kind(self) -> str:
            return "memory"

        def degraded(self) -> bool:
            return False

        def last_error(self) -> str | None:
            return None

        def adopt(self, proposal_id: str, *, operator_id: str):
            raise RuntimeError("db unavailable")

    monkeypatch.setattr(api_routes, "CAUSAL_GEOMETRY_PROPOSAL_STORE", FakeStore())
    with pytest.raises(HTTPException) as exc:
        api_routes.api_causal_geometry_proposal_adopt(
            "proposal-1",
            api_routes.CausalGeometryProposalAdoptRequest(operator_id="juniper"),
        )
    assert exc.value.status_code == 503
    assert "causal_geometry_proposal_adopt_failed" in str(exc.value.detail)


def test_adopt_route_never_auto_invoked_requires_operator_id_field() -> None:
    """operator_id is a required field -- no code path can construct the request without one."""
    with pytest.raises(Exception):
        api_routes.CausalGeometryProposalAdoptRequest()  # type: ignore[call-arg]


def test_reject_route_calls_through_to_store_with_operator_id_and_reason(monkeypatch) -> None:
    observed: dict[str, object] = {}

    class FakeStore:
        def source_kind(self) -> str:
            return "memory"

        def degraded(self) -> bool:
            return False

        def last_error(self) -> str | None:
            return None

        def reject(self, proposal_id: str, *, operator_id: str, reason: str = ""):
            observed["proposal_id"] = proposal_id
            observed["operator_id"] = operator_id
            observed["reason"] = reason
            return None

        def status_for(self, proposal_id: str):
            return "rejected"

    monkeypatch.setattr(api_routes, "CAUSAL_GEOMETRY_PROPOSAL_STORE", FakeStore())
    payload = api_routes.api_causal_geometry_proposal_reject(
        "proposal-1",
        api_routes.CausalGeometryProposalRejectRequest(operator_id="juniper", reason="not a real edge"),
    )
    assert observed["proposal_id"] == "proposal-1"
    assert observed["operator_id"] == "juniper"
    assert observed["reason"] == "not a real edge"
    assert payload["data"]["status"] == "rejected"


def test_reject_route_raises_503_when_store_errors(monkeypatch) -> None:
    class FakeStore:
        def source_kind(self) -> str:
            return "memory"

        def degraded(self) -> bool:
            return False

        def last_error(self) -> str | None:
            return None

        def reject(self, proposal_id: str, *, operator_id: str, reason: str = ""):
            raise RuntimeError("db unavailable")

    monkeypatch.setattr(api_routes, "CAUSAL_GEOMETRY_PROPOSAL_STORE", FakeStore())
    with pytest.raises(HTTPException) as exc:
        api_routes.api_causal_geometry_proposal_reject(
            "proposal-1",
            api_routes.CausalGeometryProposalRejectRequest(operator_id="juniper"),
        )
    assert exc.value.status_code == 503
    assert "causal_geometry_proposal_reject_failed" in str(exc.value.detail)


# --- Real store integration (only runs when Rung 2B's module is importable) ---


@requires_field_topology_store
def test_proposals_lifecycle_against_real_field_topology_learned_store(monkeypatch) -> None:
    store = FieldTopologyLearnedWeightsStore()
    monkeypatch.setattr(api_routes, "CAUSAL_GEOMETRY_PROPOSAL_STORE", store)

    proposal = _make_proposal("proposal-real-1")
    store.propose(proposal)

    pending_payload = api_routes.api_causal_geometry_proposals(limit=50)
    assert pending_payload["source"]["degraded"] is False
    assert any(item["proposal_id"] == "proposal-real-1" for item in pending_payload["data"])

    adopt_payload = api_routes.api_causal_geometry_proposal_adopt(
        "proposal-real-1",
        api_routes.CausalGeometryProposalAdoptRequest(operator_id="juniper", rationale="approved"),
    )
    assert adopt_payload["data"]["ok"] is True

    # Adopted proposals drop out of the pending list.
    pending_after = api_routes.api_causal_geometry_proposals(limit=50)
    assert not any(item["proposal_id"] == "proposal-real-1" for item in pending_after["data"])


@requires_field_topology_store
def test_reject_lifecycle_against_real_field_topology_learned_store(monkeypatch) -> None:
    store = FieldTopologyLearnedWeightsStore()
    monkeypatch.setattr(api_routes, "CAUSAL_GEOMETRY_PROPOSAL_STORE", store)

    proposal = _make_proposal("proposal-real-2")
    store.propose(proposal)

    reject_payload = api_routes.api_causal_geometry_proposal_reject(
        "proposal-real-2",
        api_routes.CausalGeometryProposalRejectRequest(operator_id="juniper", reason="bad edge"),
    )
    assert reject_payload["data"]["status"] == "rejected"

    pending_after = api_routes.api_causal_geometry_proposals(limit=50)
    assert not any(item["proposal_id"] == "proposal-real-2" for item in pending_after["data"])
