from __future__ import annotations

from pathlib import Path

import pytest

from app.node_catalog import NodeCatalog

REPO_ROOT = Path(__file__).resolve().parents[3]
CATALOG_PATH = REPO_ROOT / "config" / "biometrics" / "node_catalog.yaml"


@pytest.fixture
def catalog() -> NodeCatalog:
    return NodeCatalog.load(CATALOG_PATH)


def test_resolves_circe_alias(catalog: NodeCatalog) -> None:
    p = catalog.resolve("circe.tail348bbe.ts.net")
    assert p.node_id == "circe"
    assert p.capabilities["local_llm_heavy"] is True
    assert p.known is True
    assert p.raw_node == "circe.tail348bbe.ts.net"


def test_retired_atlas_still_resolves_but_is_not_live(catalog: NodeCatalog) -> None:
    p = catalog.resolve("atlas.tail348bbe.ts.net")
    assert p.node_id == "atlas"
    assert p.retired is True
    assert p.expected_online is False
    assert "atlas" not in catalog.profiles


def test_resolves_prometheus_typo(catalog: NodeCatalog) -> None:
    p = catalog.resolve("prometheous")
    assert p.node_id == "prometheus"
    assert p.role == "observability"
    assert p.known is True


def test_unknown_node_gets_fallback(catalog: NodeCatalog) -> None:
    p = catalog.resolve("weirdbox")
    assert p.node_id == "weirdbox"
    assert p.known is False
    assert p.role == "unknown"


def test_hecate_expected_offline(catalog: NodeCatalog) -> None:
    p = catalog.resolve("hecate")
    assert p.expected_online is False
    assert p.node_id == "hecate"
