"""Deterministic evals for Mind's drive_state_compact facet contract.

These measure the behavioral contract this seam must hold, not just unit
correctness of individual branches (that's test_mind_evidence_facets.py):
the timeout budget is actually honored under real wall-clock delay, the
recall prefetch and drive-state fetch genuinely run concurrently (not
serially), and a content-free drive_audits row never gets fabricated into a
facet Mind would treat as real signal.
"""
from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
import time
from pathlib import Path

import pytest

# Reuses services/orion-cortex-orch/tests/_orch_import_guard.py rather than a
# thinner reimplementation: this monorepo has three services (orion-mind,
# orion-cortex-exec, orion-cortex-orch) that each ship a package literally
# named `app`, which collide in a shared pytest session unless the foreign
# `app` module is purged from sys.modules and other services' paths are
# stripped from sys.path first -- a real, previously-hit collision class
# (see the guard's introducing commit and
# scripts/eval_spark_introspection_contract.sh's "Separate pytest processes
# avoid app package collisions between services" comment), not hypothetical.
_guard = Path(__file__).resolve().parents[1] / "tests" / "_orch_import_guard.py"
_spec = importlib.util.spec_from_file_location("_orch_guard_boot", _guard)
assert _spec and _spec.loader
_guard_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_guard_mod)


def _orch_prep() -> None:
    os.environ.setdefault("RECALL_PG_DSN", "")
    _guard_mod.ensure_orion_cortex_orch_app()
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def test_eval_timeout_budget_is_honored_under_real_delay() -> None:
    """A drive_audits query slower than MIND_DRIVE_STATE_FETCH_TIMEOUT_SEC must
    be cut off close to the configured timeout, not left to run to completion --
    this is the actual latency-safety contract the whole feature exists for."""
    _orch_prep()
    from app import mind_runtime as mod

    class _FakeSettings:
        recall_pg_dsn = "postgresql://x"
        mind_drive_state_fetch_timeout_sec = 0.05

    async def slow_query() -> dict:
        await asyncio.sleep(5.0)
        return {"dominant_drive": "coherence"}

    orig_settings = mod.get_settings
    orig_query = mod._query_latest_drive_audit_row
    mod.get_settings = lambda: _FakeSettings()  # type: ignore[assignment]
    mod._query_latest_drive_audit_row = slow_query  # type: ignore[assignment]
    try:
        t0 = time.perf_counter()
        compact, diag = asyncio.run(mod.fetch_drive_state_facet_for_mind("corr-eval-1"))
        elapsed = time.perf_counter() - t0
    finally:
        mod.get_settings = orig_settings  # type: ignore[assignment]
        mod._query_latest_drive_audit_row = orig_query  # type: ignore[assignment]

    assert compact is None
    assert diag["timed_out"] is True
    # Real wall-clock bound, not just the mocked diagnostics field: must return
    # near the 0.05s budget, nowhere near the 5s underlying delay.
    assert elapsed < 1.0


def test_eval_recall_and_drive_state_fetches_run_concurrently_not_serially() -> None:
    """prepare_plan_context_for_mind_projection must not pay both I/O delays
    serially -- this is the efficiency contract the asyncio.gather wiring
    exists to guarantee. Measures real wall-clock time, not call order."""
    _orch_prep()
    from unittest.mock import AsyncMock, patch

    from app import mind_runtime as mod
    from orion.core.bus.bus_schemas import ServiceRef
    from orion.schemas.cortex.contracts import CortexClientContext, CortexClientRequest, LLMMessage, RecallDirective
    from orion.schemas.cortex.schemas import ExecutionPlan, ExecutionStep, PlanExecutionRequest

    class _FakeSettings:
        recall_pg_dsn = "postgresql://x"
        mind_drive_state_fetch_timeout_sec = 0.3
        mind_recall_prefetch_enabled = True
        mind_recall_prefetch_timeout_sec = 0.3
        channel_recall_intake = "orion:recall:intake"
        channel_recall_reply_prefix = "orion:exec:result:RecallService"

    async def _slow_recall(*_args, **_kwargs):
        await asyncio.sleep(0.2)
        return {}, {"correlation_id": "corr-eval-2", "ok": True}

    async def _slow_drive_state(_correlation_id: str):
        await asyncio.sleep(0.2)
        return None, {"ok": True, "reason": "no_rows"}

    client_request = CortexClientRequest(
        verb="chat_general",
        mode="brain",
        recall=RecallDirective(enabled=True),
        context=CortexClientContext(
            messages=[LLMMessage(role="user", content="hi")],
            user_message="hi",
            metadata={},
        ),
    )
    plan_request = PlanExecutionRequest(
        plan=ExecutionPlan(
            verb_name="chat_general",
            steps=[ExecutionStep(verb_name="chat_general", step_name="noop", order=0, services=[])],
        ),
        context={"metadata": {}},
    )

    with (
        patch.object(mod, "get_settings", return_value=_FakeSettings()),
        patch.object(mod, "prefetch_recall_bundle_for_projection", new=AsyncMock(side_effect=_slow_recall)),
        patch.object(mod, "fetch_drive_state_facet_for_mind", new=_slow_drive_state),
    ):
        t0 = time.perf_counter()
        asyncio.run(
            mod.prepare_plan_context_for_mind_projection(
                object(),
                source=ServiceRef(name="cortex-orch", version="0.2.0", node="test"),
                client_request=client_request,
                plan_request=plan_request,
                correlation_id="corr-eval-2",
            )
        )
        elapsed = time.perf_counter() - t0

    # Serial would be ~0.4s (0.2 + 0.2); concurrent must land near max(0.2, 0.2).
    assert elapsed < 0.35, f"expected concurrent execution near 0.2s, took {elapsed:.3f}s"


def test_eval_content_free_row_never_fabricated_into_a_facet() -> None:
    """A drive_audits row that exists but carries no meaningful content (a
    routine quiet tick) must never be attached as if it were real signal --
    the empty-shell-cognition contract CLAUDE.md Section 0A requires."""
    _orch_prep()
    from app import mind_runtime as mod

    class _FakeSettings:
        recall_pg_dsn = "postgresql://x"
        mind_drive_state_fetch_timeout_sec = 0.4

    for content_free_row in (
        {"dominant_drive": None, "active_drives": [], "drive_pressures": {"coherence": 0.02}, "summary": None, "observed_at": None},
        {"dominant_drive": None, "active_drives": None, "drive_pressures": None, "summary": None, "observed_at": None},
        {"dominant_drive": "", "active_drives": [], "drive_pressures": {}, "summary": "", "observed_at": None},
    ):

        async def fake_query(_row=content_free_row) -> dict:
            return _row

        orig_settings = mod.get_settings
        orig_query = mod._query_latest_drive_audit_row
        mod.get_settings = lambda: _FakeSettings()  # type: ignore[assignment]
        mod._query_latest_drive_audit_row = fake_query  # type: ignore[assignment]
        try:
            compact, diag = asyncio.run(mod.fetch_drive_state_facet_for_mind("corr-eval-3"))
        finally:
            mod.get_settings = orig_settings  # type: ignore[assignment]
            mod._query_latest_drive_audit_row = orig_query  # type: ignore[assignment]

        assert compact is None, f"content-free row was fabricated into a facet: {content_free_row}"
        assert diag["ok"] is True
        assert diag["reason"] == "no_meaningful_content"
