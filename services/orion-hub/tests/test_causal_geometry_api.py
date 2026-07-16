from __future__ import annotations

import os
import sys
import importlib.util
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


# --- Graceful degradation: snapshot / history (no live persistence yet, Phase A) ---


def test_snapshot_endpoint_degrades_gracefully_never_500() -> None:
    payload = api_routes.api_causal_geometry_snapshot()
    assert "source" in payload
    assert "data" in payload
    assert payload["source"]["degraded"] is True
    assert payload["source"]["kind"] == "unavailable"
    assert payload["data"] is None


def test_history_endpoint_degrades_gracefully_never_500() -> None:
    payload = api_routes.api_causal_geometry_history(limit=20)
    assert "source" in payload
    assert "data" in payload
    assert payload["source"]["degraded"] is True
    assert payload["source"]["kind"] == "unavailable"
    assert payload["data"] == []


def test_history_endpoint_respects_limit_query_bounds() -> None:
    payload = api_routes.api_causal_geometry_history(limit=5)
    assert payload["source"]["query"]["limit"] == 5


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
