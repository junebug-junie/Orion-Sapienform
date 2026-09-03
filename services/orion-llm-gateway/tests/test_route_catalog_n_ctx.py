"""`GET /routes` publishes the context window each worker is actually serving.

A route's usable window is a property of the running worker, not of anything a
caller can know statically: on 2026-09-03 `agent` was serving 32768 while
`chat`/`harness` served 131072, and the FCC harness budgeted every lane at one
env-wide 131072. Overrunning it does not surface as an HTTP error -- llama.cpp's
400 returns through FCC as a 200 whose assistant text is the error string -- so
the ceiling has to be read, not assumed.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from app import route_catalog
from app.llm_backend import RouteTarget


class _FakeResponse:
    def __init__(self, payload: Dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> Dict[str, Any]:
        return self._payload


class _FakeClient:
    """Minimal stand-in for httpx.AsyncClient's context-manager + get()."""

    def __init__(self, payload: Dict[str, Any], status_code: int = 200, **_: Any) -> None:
        self._payload = payload
        self._status = status_code

    async def __aenter__(self) -> "_FakeClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None

    async def get(self, _url: str) -> _FakeResponse:
        return _FakeResponse(self._payload, self._status)


def _models_payload(meta: Any) -> Dict[str, Any]:
    """The real /v1/models shape llama.cpp returns, trimmed to what is read."""
    entry: Dict[str, Any] = {"id": "/models/gguf/Qwen3.8-27B-UD-Q4_K_XL.gguf"}
    if meta is not None:
        entry["meta"] = meta
    return {"object": "list", "data": [entry]}


def _patch_client(monkeypatch: pytest.MonkeyPatch, payload: Dict[str, Any], status: int = 200) -> None:
    monkeypatch.setattr(
        route_catalog.httpx,
        "AsyncClient",
        lambda **kw: _FakeClient(payload, status, **kw),
    )


TARGET = RouteTarget(url="http://agent:8015", served_by="circe-worker-agent-1", backend="llamacpp")


@pytest.mark.asyncio
async def test_probe_reads_the_started_window_not_the_trained_maximum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`n_ctx` is what llama.cpp was launched with; `n_ctx_train` is the
    architecture's theoretical maximum and is ~8x larger on these builds
    (262144 vs 32768 live). Reading the wrong one licenses a prompt the worker
    rejects, which is the exact failure this field exists to prevent."""
    _patch_client(monkeypatch, _models_payload({"n_ctx": 32768, "n_ctx_train": 262144}))

    model, n_ctx = await route_catalog._probe_model(TARGET)

    assert n_ctx == 32768
    assert model == "/models/gguf/Qwen3.8-27B-UD-Q4_K_XL.gguf"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "meta",
    [
        None,                      # older llama.cpp: no meta block at all
        {},                        # meta present, n_ctx absent
        {"n_ctx": 0},              # nonsense value
        {"n_ctx": -1},
        {"n_ctx": "32768"},        # string, not int
        {"n_ctx": True},           # bool is an int subclass -- must not slip through
    ],
)
async def test_unreadable_window_is_None_never_a_guess(
    monkeypatch: pytest.MonkeyPatch, meta: Any
) -> None:
    """None means "no ceiling known", which consumers fall back on. A wrong
    number would be worse than no number: it would be trusted."""
    _patch_client(monkeypatch, _models_payload(meta))

    _model, n_ctx = await route_catalog._probe_model(TARGET)

    assert n_ctx is None


@pytest.mark.asyncio
async def test_a_failing_probe_reports_no_window(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_client(monkeypatch, {}, status=503)

    model, n_ctx = await route_catalog._probe_model(TARGET)

    assert (model, n_ctx) == (None, None)


@pytest.mark.asyncio
async def test_a_down_worker_reports_no_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """`_probe_backend` already blanks model/vision when health is down; the
    window has to be blanked with them or a stale ceiling outlives its worker."""
    _patch_client(monkeypatch, _models_payload({"n_ctx": 32768}))
    monkeypatch.setattr(route_catalog, "_probe_health", _fake_health("down"))
    monkeypatch.setattr(route_catalog, "_probe_vision", _fake_vision())

    status, _latency, model, vision, n_ctx = await route_catalog._probe_backend(TARGET)

    assert status == "down"
    assert (model, vision, n_ctx) == (None, None, None)


@pytest.mark.asyncio
async def test_an_up_worker_carries_its_window_into_the_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_client(monkeypatch, _models_payload({"n_ctx": 32768}))
    monkeypatch.setattr(route_catalog, "_probe_health", _fake_health("up"))
    monkeypatch.setattr(route_catalog, "_probe_vision", _fake_vision())

    probe = await route_catalog._probe_backend(TARGET)
    entry = route_catalog._entry_from_probe("agent", TARGET, *probe)

    assert entry.n_ctx == 32768
    # The serialized shape is the contract consumers read, not the dataclass.
    assert route_catalog._entry_to_dict(entry)["n_ctx"] == 32768


def test_an_unconfigured_route_serializes_a_null_window() -> None:
    entry = route_catalog.RouteHealthEntry(
        route_id="agent",
        served_by=None,
        backend=None,
        status="not_configured",
        latency_ms=None,
        last_checked_at=None,
    )

    assert route_catalog._entry_to_dict(entry)["n_ctx"] is None


def _fake_health(status: str):
    async def _probe(_target: Any) -> tuple[str, int]:
        return status, 12
    return _probe


def _fake_vision():
    async def _probe(_target: Any) -> bool:
        return False
    return _probe
