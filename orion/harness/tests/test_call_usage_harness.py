"""Per-call token ledger, harness side: a per-turn FCC spawn sends ``X-Orion-Correlation-Id`` on
every model call. The FCC adapter forwards it to llm-gateway (services/orion-fcc/tests/
test_correlation_forward.py), so the gateway's ledger row carries the turn id."""
from __future__ import annotations

from orion.harness.fcc_motor import _build_subprocess_env  # noqa: E402
from orion.schemas.gpu_pool import GpuLeaseRefV1  # noqa: E402


def test_spawn_env_carries_the_turn_correlation_id(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "X-Trace: t\nX-Orion-Correlation-Id: stale-turn")
    env = _build_subprocess_env(fcc_server_url="http://fcc:8082", auth_token="tok", correlation_id="turn-1")
    assert env["ANTHROPIC_CUSTOM_HEADERS"].splitlines() == ["X-Trace: t", "X-Orion-Correlation-Id: turn-1"]
    ref = GpuLeaseRefV1(lease_id="h", generation=1, role="agent", holder="durable-runs:r")
    env = _build_subprocess_env(fcc_server_url="http://fcc:8082", auth_token="tok", correlation_id="turn-2",
                                gpu_lease=ref.model_dump(mode="json"))
    assert "X-Orion-Correlation-Id: turn-2" in env["ANTHROPIC_CUSTOM_HEADERS"].splitlines()


def test_spawn_env_without_a_correlation_id_drops_an_inherited_one(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "X-Orion-Correlation-Id: stale-turn")
    assert "ANTHROPIC_CUSTOM_HEADERS" not in _build_subprocess_env(fcc_server_url="http://f", auth_token="t")
    # A header-injection attempt is refused, not forwarded.
    env = _build_subprocess_env(fcc_server_url="http://f", auth_token="t", correlation_id="a\nEvil: 1")
    assert "ANTHROPIC_CUSTOM_HEADERS" not in env
