"""An eval's one-off gateway call on an operator route names its holder (leased_llm.gateway_call), so
gpu_pool_events says which eval ran. Orion's own routes never take a holder from a header."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app import passthrough_proxy


class _Req:
    def __init__(self, headers):
        self.headers = {k.lower(): v for k, v in headers.items()}


@pytest.fixture
def routes(monkeypatch):
    classes = {"operator_fast": "operator_fast", "quick": "fast", "operator": "operator"}

    def spec(route):
        if route not in classes:
            raise KeyError(route)
        return SimpleNamespace(work_class=classes[route])
    monkeypatch.setattr(passthrough_proxy, "route_spec", spec)


@pytest.mark.parametrize("route,header,want", [
    ("operator_fast", "operator:qwen35-fast-eval", "operator:qwen35-fast-eval"),
    ("operator", "operator:x", "operator:x"),
    ("quick", "operator:qwen35-fast-eval", "http:openai"),          # Orion's route keeps its holder
    ("operator_fast", "orion-mind", "http:openai"),                 # not operator-shaped
    ("operator_fast", "operator:" + "x" * 80, "http:openai"),       # too long
    ("operator_fast", "operator:a b", "http:openai"),
    ("unknown", "operator:x", "http:openai"),
    ("operator_fast", None, "http:openai"),
])
def test_holder_for(routes, route, header, want):
    req = _Req({passthrough_proxy.OPERATOR_HOLDER_HEADER: header} if header else {})
    assert passthrough_proxy.holder_for(req, route, "http:openai") == want
