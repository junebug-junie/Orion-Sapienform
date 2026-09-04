from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[3]
HUB_ROOT = Path(__file__).resolve().parents[1]


def _ensure_hub_scripts_import_path() -> None:
    for key in list(sys.modules):
        if key == "scripts" or key.startswith("scripts."):
            del sys.modules[key]
    for p in (str(REPO_ROOT), str(HUB_ROOT)):
        try:
            sys.path.remove(p)
        except ValueError:
            pass
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(HUB_ROOT))


_ensure_hub_scripts_import_path()

from scripts import self_brain_routes  # noqa: E402


def _client() -> TestClient:
    self_brain_routes._region_provenance.cache_clear()  # force a fresh build per test
    app = FastAPI()
    app.include_router(self_brain_routes.router)
    return TestClient(app)


def test_region_provenance_covers_all_six_dimensions() -> None:
    resp = _client().get("/api/self-brain/region-provenance")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {
        "node_kind",
        "lane",
        "self_state",
        "lattice_layer",
        "honesty_metrics",
        "field_anomaly",
    }


def test_field_anomaly_provenance_names_field_digester() -> None:
    """The one dimension whose true producer differs from the other five --
    regression guard for that distinction actually reaching the frontend."""
    resp = _client().get("/api/self-brain/region-provenance")
    entry = resp.json()["field_anomaly"]
    assert entry["producer_service"] == "orion-field-digester"
    assert entry["upstream"] == [
        "metric://bus_channel/orion-field-digester/orion:field_channel:anomaly_score"
    ]


def test_other_dimensions_name_substrate_runtime() -> None:
    resp = _client().get("/api/self-brain/region-provenance")
    body = resp.json()
    for dim in ("node_kind", "lane", "self_state", "lattice_layer", "honesty_metrics"):
        assert body[dim]["producer_service"] == "orion-substrate-runtime"


def test_response_is_cached_after_first_build(monkeypatch) -> None:
    """Second call must not re-resolve the graph -- _region_provenance() is
    lru_cache(1), so orion.metrics.lineage.resolve_brain_regions() (the real
    work) must run at most once across repeated requests, even though the
    route now calls the (cached) wrapper unconditionally on every call."""
    from orion.metrics import lineage

    call_count = {"n": 0}
    real_resolve = lineage.resolve_brain_regions

    def _counting_resolve():
        call_count["n"] += 1
        return real_resolve()

    monkeypatch.setattr(lineage, "resolve_brain_regions", _counting_resolve)
    client = _client()
    client.get("/api/self-brain/region-provenance")
    client.get("/api/self-brain/region-provenance")
    assert call_count["n"] == 1


def test_cache_clear_forces_a_real_rebuild() -> None:
    """The other half of the lru_cache contract: cache_clear() must actually
    clear it, or _client()'s own per-test isolation (and this test file's
    other tests) would be silently sharing state."""
    self_brain_routes._region_provenance()
    info_before = self_brain_routes._region_provenance.cache_info()
    assert info_before.currsize == 1
    self_brain_routes._region_provenance.cache_clear()
    info_after = self_brain_routes._region_provenance.cache_info()
    assert info_after.currsize == 0
