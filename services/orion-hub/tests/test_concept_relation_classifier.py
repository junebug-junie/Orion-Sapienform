"""Tests for the real LLM classifier wired into
``orion.substrate.relation_classification``'s ``RelationClassifier`` seam.

Covers ``services/orion-hub/scripts/concept_relation_classifier.py``. All bus
I/O is faked at the ``OrionBusAsync`` boundary -- no real LLM gateway, no
network, no event loop left dangling. Every test is fast and deterministic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
HUB_ROOT = Path(__file__).resolve().parents[1]


def _ensure_import_path() -> None:
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


_ensure_import_path()

from orion.core.schemas.cognitive_substrate import ConceptNodeV1, NodeRefV1, SubstrateEdgeV1  # noqa: E402
from orion.substrate.adapters._common import make_provenance, make_temporal  # noqa: E402

from scripts import concept_relation_classifier as crc  # noqa: E402


def _fake_settings() -> SimpleNamespace:
    return SimpleNamespace(
        ORION_BUS_URL="redis://fake:6379/0",
        SERVICE_NAME="hub",
        SERVICE_VERSION="0.0.0-test",
        NODE_NAME="test-node",
        CHANNEL_LLM_INTAKE="orion:exec:request:LLMGatewayService",
        LLM_GATEWAY_SERVICE_NAME="LLMGatewayService",
        HUB_LLM_GATEWAY_TIMEOUT_SEC=5.0,
    )


def _concept_node(node_id: str, *, label: str, definition: str = "") -> ConceptNodeV1:
    return ConceptNodeV1(
        node_id=node_id,
        anchor_scope="world",
        label=label,
        definition=definition or None,
        temporal=make_temporal(observed_at=None),
        provenance=make_provenance(source_kind="test", source_channel="test", producer="test"),
    )


def _edge(source_id: str, target_id: str, *, co_occurrence_count: int = 10) -> SubstrateEdgeV1:
    return SubstrateEdgeV1(
        source=NodeRefV1(node_id=source_id, node_kind="concept"),
        target=NodeRefV1(node_id=target_id, node_kind="concept"),
        predicate="co_occurs_with",
        temporal=make_temporal(observed_at=None),
        provenance=make_provenance(source_kind="test", source_channel="test", producer="test"),
        metadata={"co_occurrence_count": co_occurrence_count},
    )


class _FakeDecoded:
    def __init__(self, *, ok: bool, payload: Optional[dict] = None, error: Optional[str] = None):
        self.ok = ok
        self.error = error
        self.envelope = SimpleNamespace(payload=payload or {})


class _FakeCodec:
    def __init__(self, decode_fn):
        self._decode_fn = decode_fn

    def decode(self, data):
        return self._decode_fn(data)


class _FakeBus:
    """Stand-in for OrionBusAsync -- no network, fully scripted responses."""

    def __init__(self, url: str, *, rpc_fn=None, connect_error: Optional[Exception] = None, decode_fn=None):
        self.url = url
        self._rpc_fn = rpc_fn
        self._connect_error = connect_error
        self.codec = _FakeCodec(decode_fn or (lambda data: _FakeDecoded(ok=True, payload=data)))
        self.connected = False
        self.closed = False

    async def connect(self) -> None:
        if self._connect_error is not None:
            raise self._connect_error
        self.connected = True

    async def close(self) -> None:
        self.closed = True

    async def rpc_request(self, channel, envelope, *, reply_channel, timeout_sec):
        return await self._rpc_fn(channel, envelope, reply_channel=reply_channel, timeout_sec=timeout_sec)


def _install_fake_bus(monkeypatch: pytest.MonkeyPatch, *, rpc_fn=None, connect_error=None, decode_fn=None) -> None:
    def _factory(url: str, *_, **__):
        return _FakeBus(url, rpc_fn=rpc_fn, connect_error=connect_error, decode_fn=decode_fn)

    monkeypatch.setattr(crc, "OrionBusAsync", _factory)


def _content_msg(content: str) -> dict:
    return {"data": {"content": content}}


def _decode_content_passthrough(data: dict) -> _FakeDecoded:
    return _FakeDecoded(ok=True, payload={"content": data["content"]})


# --- clear supports / contradicts cases -------------------------------------


def test_classify_pair_supports_case_returns_predicate(monkeypatch: pytest.MonkeyPatch) -> None:
    async def rpc_fn(channel, envelope, *, reply_channel, timeout_sec):
        return _content_msg(json.dumps({"predicate": "supports"}))

    _install_fake_bus(monkeypatch, rpc_fn=rpc_fn, decode_fn=_decode_content_passthrough)

    node_a = _concept_node("sub-node-a", label="continuity")
    node_b = _concept_node("sub-node-b", label="memory persistence")
    edge = _edge(node_a.node_id, node_b.node_id)

    classifier = crc.build_llm_relation_classifier([(node_a, node_b, edge)], settings=_fake_settings())
    result = classifier(node_a, node_b, edge)
    assert result == "supports"


def test_classify_pair_contradicts_case_returns_predicate(monkeypatch: pytest.MonkeyPatch) -> None:
    async def rpc_fn(channel, envelope, *, reply_channel, timeout_sec):
        return _content_msg(json.dumps({"predicate": "contradicts"}))

    _install_fake_bus(monkeypatch, rpc_fn=rpc_fn, decode_fn=_decode_content_passthrough)

    node_a = _concept_node("sub-node-a", label="fixed identity")
    node_b = _concept_node("sub-node-b", label="fluid self-model")
    edge = _edge(node_a.node_id, node_b.node_id)

    classifier = crc.build_llm_relation_classifier([(node_a, node_b, edge)], settings=_fake_settings())
    result = classifier(node_a, node_b, edge)
    assert result == "contradicts"


def test_classify_pair_refines_case_returns_predicate(monkeypatch: pytest.MonkeyPatch) -> None:
    async def rpc_fn(channel, envelope, *, reply_channel, timeout_sec):
        return _content_msg(json.dumps({"predicate": "refines"}))

    _install_fake_bus(monkeypatch, rpc_fn=rpc_fn, decode_fn=_decode_content_passthrough)

    node_a = _concept_node("sub-node-a", label="narrow definition")
    node_b = _concept_node("sub-node-b", label="broad definition")
    edge = _edge(node_a.node_id, node_b.node_id)

    classifier = crc.build_llm_relation_classifier([(node_a, node_b, edge)], settings=_fake_settings())
    result = classifier(node_a, node_b, edge)
    assert result == "refines"


# --- ambiguous / unparseable degrades to None --------------------------------


def test_classify_pair_explicit_none_predicate_degrades_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    async def rpc_fn(channel, envelope, *, reply_channel, timeout_sec):
        return _content_msg(json.dumps({"predicate": "none"}))

    _install_fake_bus(monkeypatch, rpc_fn=rpc_fn, decode_fn=_decode_content_passthrough)

    node_a = _concept_node("sub-node-a", label="weather")
    node_b = _concept_node("sub-node-b", label="unrelated topic")
    edge = _edge(node_a.node_id, node_b.node_id)

    classifier = crc.build_llm_relation_classifier([(node_a, node_b, edge)], settings=_fake_settings())
    assert classifier(node_a, node_b, edge) is None


def test_classify_pair_unparseable_response_degrades_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    async def rpc_fn(channel, envelope, *, reply_channel, timeout_sec):
        # Prose, no JSON object at all.
        return _content_msg("I think these concepts are related somehow.")

    _install_fake_bus(monkeypatch, rpc_fn=rpc_fn, decode_fn=_decode_content_passthrough)

    node_a = _concept_node("sub-node-a", label="a")
    node_b = _concept_node("sub-node-b", label="b")
    edge = _edge(node_a.node_id, node_b.node_id)

    classifier = crc.build_llm_relation_classifier([(node_a, node_b, edge)], settings=_fake_settings())
    assert classifier(node_a, node_b, edge) is None


def test_classify_pair_invalid_predicate_value_degrades_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    async def rpc_fn(channel, envelope, *, reply_channel, timeout_sec):
        # Valid JSON, but "predicate" is not one of the allowed literals.
        return _content_msg(json.dumps({"predicate": "banana"}))

    _install_fake_bus(monkeypatch, rpc_fn=rpc_fn, decode_fn=_decode_content_passthrough)

    node_a = _concept_node("sub-node-a", label="a")
    node_b = _concept_node("sub-node-b", label="b")
    edge = _edge(node_a.node_id, node_b.node_id)

    classifier = crc.build_llm_relation_classifier([(node_a, node_b, edge)], settings=_fake_settings())
    assert classifier(node_a, node_b, edge) is None


def test_classify_pair_decode_failure_degrades_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    async def rpc_fn(channel, envelope, *, reply_channel, timeout_sec):
        return {"data": {}}

    def decode_fn(data):
        return _FakeDecoded(ok=False, error="bad codec frame")

    _install_fake_bus(monkeypatch, rpc_fn=rpc_fn, decode_fn=decode_fn)

    node_a = _concept_node("sub-node-a", label="a")
    node_b = _concept_node("sub-node-b", label="b")
    edge = _edge(node_a.node_id, node_b.node_id)

    classifier = crc.build_llm_relation_classifier([(node_a, node_b, edge)], settings=_fake_settings())
    assert classifier(node_a, node_b, edge) is None


# --- timeout / exception from the LLM call degrades to None, never raises ---


def test_classify_pair_rpc_timeout_degrades_to_none_without_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    async def rpc_fn(channel, envelope, *, reply_channel, timeout_sec):
        raise TimeoutError("RPC timeout waiting on reply channel")

    _install_fake_bus(monkeypatch, rpc_fn=rpc_fn, decode_fn=_decode_content_passthrough)

    node_a = _concept_node("sub-node-a", label="a")
    node_b = _concept_node("sub-node-b", label="b")
    edge = _edge(node_a.node_id, node_b.node_id)

    classifier = crc.build_llm_relation_classifier([(node_a, node_b, edge)], settings=_fake_settings())
    assert classifier(node_a, node_b, edge) is None


def test_classify_pair_rpc_exception_degrades_to_none_without_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    async def rpc_fn(channel, envelope, *, reply_channel, timeout_sec):
        raise RuntimeError("connection reset")

    _install_fake_bus(monkeypatch, rpc_fn=rpc_fn, decode_fn=_decode_content_passthrough)

    node_a = _concept_node("sub-node-a", label="a")
    node_b = _concept_node("sub-node-b", label="b")
    edge = _edge(node_a.node_id, node_b.node_id)

    classifier = crc.build_llm_relation_classifier([(node_a, node_b, edge)], settings=_fake_settings())
    assert classifier(node_a, node_b, edge) is None


def test_classify_batch_bus_connect_failure_degrades_all_pairs_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_bus(monkeypatch, connect_error=RuntimeError("redis unreachable"))

    node_a = _concept_node("sub-node-a", label="a")
    node_b = _concept_node("sub-node-b", label="b")
    node_c = _concept_node("sub-node-c", label="c")
    edge_ab = _edge(node_a.node_id, node_b.node_id)
    edge_bc = _edge(node_b.node_id, node_c.node_id)

    classifier = crc.build_llm_relation_classifier(
        [(node_a, node_b, edge_ab), (node_b, node_c, edge_bc)], settings=_fake_settings()
    )
    assert classifier(node_a, node_b, edge_ab) is None
    assert classifier(node_b, node_c, edge_bc) is None


# --- batch behavior: multiple pairs, one shared connection -------------------


def test_classify_batch_multiple_pairs_get_distinct_judgments(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    async def rpc_fn(channel, envelope, *, reply_channel, timeout_sec):
        calls.append(envelope.correlation_id)
        # Alternate response based on call order so each pair gets a distinct judgment.
        if len(calls) == 1:
            return _content_msg(json.dumps({"predicate": "supports"}))
        return _content_msg(json.dumps({"predicate": "none"}))

    _install_fake_bus(monkeypatch, rpc_fn=rpc_fn, decode_fn=_decode_content_passthrough)

    node_a = _concept_node("sub-node-a", label="a")
    node_b = _concept_node("sub-node-b", label="b")
    node_c = _concept_node("sub-node-c", label="c")
    edge_ab = _edge(node_a.node_id, node_b.node_id)
    edge_bc = _edge(node_b.node_id, node_c.node_id)

    classifier = crc.build_llm_relation_classifier(
        [(node_a, node_b, edge_ab), (node_b, node_c, edge_bc)], settings=_fake_settings()
    )
    assert classifier(node_a, node_b, edge_ab) == "supports"
    assert classifier(node_b, node_c, edge_bc) is None
    assert len(calls) == 2
