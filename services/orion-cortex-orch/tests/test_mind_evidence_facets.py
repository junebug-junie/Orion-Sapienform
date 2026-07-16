from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_guard = Path(__file__).resolve().parent / "_orch_import_guard.py"
_spec = importlib.util.spec_from_file_location("_orch_guard_boot", _guard)
assert _spec and _spec.loader
_guard_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_guard_mod)

ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


def _orch_prep() -> None:
    _guard_mod.ensure_orion_cortex_orch_app()


def test_build_mind_run_request_attaches_evidence_facets() -> None:
    _orch_prep()
    from app.mind_runtime import build_mind_run_request
    from orion.schemas.cortex.contracts import CortexClientContext, CortexClientRequest, LLMMessage
    from orion.schemas.cortex.schemas import ExecutionPlan, ExecutionStep, PlanExecutionRequest

    pr = PlanExecutionRequest(
        plan=ExecutionPlan(
            verb_name="chat_general",
            steps=[ExecutionStep(verb_name="chat_general", step_name="noop", order=0, services=[])],
        ),
        context={
            "recall_bundle": {"fragments": [{"snippet": "memory", "source": "journal"}]},
            "chat_autonomy_state_v2": {
                "attention_items": [{"summary": "notice warmth"}],
                "candidate_impulses": [],
            },
            "social_turn_policy": {"mode": "warm"},
            "chat_situation_summary": {"headline": "evening"},
            "orion_identity_summary": ["Orion is a cognitive presence."],
            "metadata": {"mind_enabled": True},
        },
    )
    cr = CortexClientRequest(
        verb="chat_general",
        mode="brain",
        context=CortexClientContext(
            messages=[LLMMessage(role="user", content="hi")],
            user_message="hi",
            metadata={"mind_enabled": True},
        ),
    )
    req = build_mind_run_request(
        cr,
        pr,
        "550e8400-e29b-41d4-a716-446655440000",
        cognitive_projection_facet={
            "schema_version": "cognitive.projection.v1",
            "projection_id": "p1",
            "anchors": {
                "orion": {
                    "items": [{"label": "curiosity", "summary": "notice user tone", "salience": 0.5}]
                }
            },
            "item_count": 1,
        },
    )
    facets = (req.snapshot_inputs or {}).get("facets") or {}
    assert facets.get("recall_bundle", {}).get("fragments")
    assert facets.get("autonomy_compact", {}).get("attention_items")
    assert facets.get("social_compact", {}).get("social_turn_policy")
    assert facets.get("situation_compact", {}).get("chat_situation_summary")
    assert facets.get("identity_background", {}).get("background_identity") is True
    assert facets.get("cognitive_projection")


# --- drive_state_compact: DriveEngine facet (Mind previously only read the dead
# chat_autonomy_state_v2 / autonomy_state keys -- see
# orion/autonomy/drives_and_autonomy_retrospective.md section 8). These tests cover the
# new bounded, fail-open drive_audits Postgres read plus the sync glue that attaches it
# as a facet, and confirm the existing (still-dead-on-purpose) autonomy_compact path is
# completely unchanged. ---


def test_fetch_drive_state_facet_success() -> None:
    import asyncio

    _orch_prep()
    from datetime import datetime, timezone

    from app import mind_runtime as mod

    class _FakeSettings:
        recall_pg_dsn = "postgresql://x"
        mind_drive_state_fetch_timeout_sec = 0.4

    observed_at = datetime(2026, 7, 16, 12, 0, 0, tzinfo=timezone.utc)
    row = {
        "dominant_drive": "coherence",
        "active_drives": {"coherence": 0.7},
        "drive_pressures": {"coherence": 0.7, "continuity": 0.3},
        "summary": "Coherence pressure elevated.",
        "observed_at": observed_at,
    }

    async def fake_query() -> dict:
        return row

    orig_settings = mod.get_settings
    orig_query = mod._query_latest_drive_audit_row
    mod.get_settings = lambda: _FakeSettings()  # type: ignore[assignment]
    mod._query_latest_drive_audit_row = fake_query  # type: ignore[assignment]
    try:
        compact, diag = asyncio.run(mod.fetch_drive_state_facet_for_mind("corr-drive-1"))
    finally:
        mod.get_settings = orig_settings  # type: ignore[assignment]
        mod._query_latest_drive_audit_row = orig_query  # type: ignore[assignment]

    assert compact is not None
    assert compact["dominant_drive"] == "coherence"
    assert compact["active_drives"] == {"coherence": 0.7}
    assert compact["drive_pressures"]["continuity"] == 0.3
    assert compact["summary"] == "Coherence pressure elevated."
    assert compact["observed_at"] == observed_at.isoformat()
    assert diag["ok"] is True
    assert diag["reason"] == "success"


def test_fetch_drive_state_facet_timeout_degrades_gracefully() -> None:
    import asyncio

    _orch_prep()
    from app import mind_runtime as mod

    class _FakeSettings:
        recall_pg_dsn = "postgresql://x"
        mind_drive_state_fetch_timeout_sec = 0.01

    async def slow_query() -> dict:
        await asyncio.sleep(1.0)
        return {}

    orig_settings = mod.get_settings
    orig_query = mod._query_latest_drive_audit_row
    mod.get_settings = lambda: _FakeSettings()  # type: ignore[assignment]
    mod._query_latest_drive_audit_row = slow_query  # type: ignore[assignment]
    try:
        compact, diag = asyncio.run(mod.fetch_drive_state_facet_for_mind("corr-drive-2"))
    finally:
        mod.get_settings = orig_settings  # type: ignore[assignment]
        mod._query_latest_drive_audit_row = orig_query  # type: ignore[assignment]

    assert compact is None
    assert diag["ok"] is False
    assert diag["timed_out"] is True
    assert diag["reason"] == "timeout"


def test_fetch_drive_state_facet_no_rows_degrades_gracefully() -> None:
    import asyncio

    _orch_prep()
    from app import mind_runtime as mod

    class _FakeSettings:
        recall_pg_dsn = "postgresql://x"
        mind_drive_state_fetch_timeout_sec = 0.4

    async def empty_query() -> None:
        return None

    orig_settings = mod.get_settings
    orig_query = mod._query_latest_drive_audit_row
    mod.get_settings = lambda: _FakeSettings()  # type: ignore[assignment]
    mod._query_latest_drive_audit_row = empty_query  # type: ignore[assignment]
    try:
        compact, diag = asyncio.run(mod.fetch_drive_state_facet_for_mind("corr-drive-3"))
    finally:
        mod.get_settings = orig_settings  # type: ignore[assignment]
        mod._query_latest_drive_audit_row = orig_query  # type: ignore[assignment]

    assert compact is None
    assert diag["ok"] is True
    assert diag["reason"] == "no_rows"


def test_fetch_drive_state_facet_content_free_row_degrades_gracefully() -> None:
    """A row can exist (subject='orion' has been audited) while carrying no
    meaningful content -- a quiet tick where nothing crossed activation
    threshold writes dominant_drive=None, summary=None, active_drives=[] to
    drive_audits. That must fail open exactly like a missing row, not be
    attached to Mind's facets as if it were real signal (CLAUDE.md Section 0A,
    'no empty-shell cognition')."""
    import asyncio
    from datetime import datetime, timezone

    _orch_prep()
    from app import mind_runtime as mod

    class _FakeSettings:
        recall_pg_dsn = "postgresql://x"
        mind_drive_state_fetch_timeout_sec = 0.4

    observed_at = datetime(2026, 7, 16, 12, 0, 0, tzinfo=timezone.utc)
    row = {
        "dominant_drive": None,
        "active_drives": [],
        "drive_pressures": {"coherence": 0.02, "continuity": 0.01},
        "summary": None,
        "observed_at": observed_at,
    }

    async def fake_query() -> dict:
        return row

    orig_settings = mod.get_settings
    orig_query = mod._query_latest_drive_audit_row
    mod.get_settings = lambda: _FakeSettings()  # type: ignore[assignment]
    mod._query_latest_drive_audit_row = fake_query  # type: ignore[assignment]
    try:
        compact, diag = asyncio.run(mod.fetch_drive_state_facet_for_mind("corr-drive-6"))
    finally:
        mod.get_settings = orig_settings  # type: ignore[assignment]
        mod._query_latest_drive_audit_row = orig_query  # type: ignore[assignment]

    assert compact is None
    assert diag["ok"] is True
    assert diag["reason"] == "no_meaningful_content"


def test_fetch_drive_state_facet_db_error_degrades_gracefully() -> None:
    import asyncio

    _orch_prep()
    from app import mind_runtime as mod

    class _FakeSettings:
        recall_pg_dsn = "postgresql://x"
        mind_drive_state_fetch_timeout_sec = 0.4

    async def failing_query() -> dict:
        raise ConnectionRefusedError("db unreachable")

    orig_settings = mod.get_settings
    orig_query = mod._query_latest_drive_audit_row
    mod.get_settings = lambda: _FakeSettings()  # type: ignore[assignment]
    mod._query_latest_drive_audit_row = failing_query  # type: ignore[assignment]
    try:
        compact, diag = asyncio.run(mod.fetch_drive_state_facet_for_mind("corr-drive-4"))
    finally:
        mod.get_settings = orig_settings  # type: ignore[assignment]
        mod._query_latest_drive_audit_row = orig_query  # type: ignore[assignment]

    assert compact is None
    assert diag["ok"] is False
    assert diag["reason"] == "exception"
    assert diag["exception_type"] == "ConnectionRefusedError"


def test_fetch_drive_state_facet_no_dsn_configured_degrades_gracefully() -> None:
    """Unconfigured RECALL_PG_DSN (pool never resolves) must also fail open, not raise."""
    import asyncio

    _orch_prep()
    from app import mind_runtime as mod

    class _FakeSettings:
        recall_pg_dsn = ""
        mind_drive_state_fetch_timeout_sec = 0.4

    # fetch_drive_state_facet_for_mind reuses memory_extractor's shared pool
    # (not a pool of its own) -- reset that module's globals, not mind_runtime's.
    # _get_memory_pool() resolves its DSN via memory_extractor's own
    # `get_settings` binding, not mind_runtime's -- both must be patched.
    from app import memory_extractor as mem_mod

    orig_mind_settings = mod.get_settings
    orig_mem_settings = mem_mod.get_settings
    orig_pool = mem_mod._memory_pool
    orig_pool_failed = mem_mod._memory_pool_failed
    mod.get_settings = lambda: _FakeSettings()  # type: ignore[assignment]
    mem_mod.get_settings = lambda: _FakeSettings()  # type: ignore[assignment]
    mem_mod._memory_pool = None
    mem_mod._memory_pool_failed = False
    try:
        compact, diag = asyncio.run(mod.fetch_drive_state_facet_for_mind("corr-drive-5"))
    finally:
        mod.get_settings = orig_mind_settings  # type: ignore[assignment]
        mem_mod.get_settings = orig_mem_settings  # type: ignore[assignment]
        mem_mod._memory_pool = orig_pool
        mem_mod._memory_pool_failed = orig_pool_failed

    assert compact is None
    assert diag["ok"] is False


def test_build_mind_run_request_attaches_drive_state_facet_and_leaves_autonomy_compact_untouched() -> None:
    """(4) new facet populated correctly; (5)/(e) autonomy_compact regression guard."""
    _orch_prep()
    from app.mind_runtime import build_mind_run_request
    from orion.schemas.cortex.contracts import CortexClientContext, CortexClientRequest, LLMMessage
    from orion.schemas.cortex.schemas import ExecutionPlan, ExecutionStep, PlanExecutionRequest

    pr = PlanExecutionRequest(
        plan=ExecutionPlan(
            verb_name="chat_general",
            steps=[ExecutionStep(verb_name="chat_general", step_name="noop", order=0, services=[])],
        ),
        context={
            "chat_autonomy_state_v2": {
                "attention_items": [{"summary": "notice warmth"}],
                "candidate_impulses": [],
            },
            "drive_state_compact": {
                "dominant_drive": "coherence",
                "active_drives": {"coherence": 0.7},
                "drive_pressures": {"coherence": 0.7},
                "summary": "Coherence pressure elevated.",
                "observed_at": "2026-07-16T12:00:00+00:00",
            },
            "metadata": {"mind_enabled": True},
        },
    )
    cr = CortexClientRequest(
        verb="chat_general",
        mode="brain",
        context=CortexClientContext(
            messages=[LLMMessage(role="user", content="hi")],
            user_message="hi",
            metadata={"mind_enabled": True},
        ),
    )
    req = build_mind_run_request(cr, pr, "550e8400-e29b-41d4-a716-446655440001")
    facets = (req.snapshot_inputs or {}).get("facets") or {}
    assert facets.get("drive_state_compact", {}).get("dominant_drive") == "coherence"
    assert facets.get("drive_state_compact", {}).get("summary") == "Coherence pressure elevated."
    # Unchanged: still reads the (still-dead) chat_autonomy_state_v2 key, same shape as before.
    assert facets.get("autonomy_compact", {}).get("attention_items")


def test_build_mind_run_request_omits_drive_state_facet_when_absent() -> None:
    """No drive_state_compact in plan ctx (e.g. fetch failed open) -> no facet key at all."""
    _orch_prep()
    from app.mind_runtime import build_mind_run_request
    from orion.schemas.cortex.contracts import CortexClientContext, CortexClientRequest, LLMMessage
    from orion.schemas.cortex.schemas import ExecutionPlan, ExecutionStep, PlanExecutionRequest

    pr = PlanExecutionRequest(
        plan=ExecutionPlan(
            verb_name="chat_general",
            steps=[ExecutionStep(verb_name="chat_general", step_name="noop", order=0, services=[])],
        ),
        context={"metadata": {"mind_enabled": True}},
    )
    cr = CortexClientRequest(
        verb="chat_general",
        mode="brain",
        context=CortexClientContext(
            messages=[LLMMessage(role="user", content="hi")],
            user_message="hi",
            metadata={"mind_enabled": True},
        ),
    )
    req = build_mind_run_request(cr, pr, "550e8400-e29b-41d4-a716-446655440002")
    facets = (req.snapshot_inputs or {}).get("facets") or {}
    assert "drive_state_compact" not in facets


def test_prepare_plan_context_merges_drive_state_and_diag() -> None:
    """prepare_plan_context_for_mind_projection wires the bounded fetch into plan ctx/meta."""
    import asyncio
    from unittest.mock import AsyncMock, patch

    _orch_prep()
    from app import mind_runtime
    from orion.core.bus.bus_schemas import ServiceRef
    from orion.schemas.cortex.contracts import (
        CortexClientContext,
        CortexClientRequest,
        LLMMessage,
        RecallDirective,
    )
    from orion.schemas.cortex.schemas import ExecutionPlan, ExecutionStep, PlanExecutionRequest

    plan = PlanExecutionRequest(
        plan=ExecutionPlan(
            verb_name="chat_general",
            steps=[ExecutionStep(verb_name="chat_general", step_name="noop", order=0, services=[])],
        ),
        context={"metadata": {}},
    )
    client = CortexClientRequest(
        verb="chat_general",
        mode="brain",
        recall=RecallDirective(enabled=False),
        context=CortexClientContext(
            messages=[LLMMessage(role="user", content="hi")],
            user_message="hi",
            metadata={"mind_enabled": True},
        ),
    )
    drive_compact = {
        "dominant_drive": "capability",
        "active_drives": {"capability": 0.6},
        "drive_pressures": {"capability": 0.6},
        "summary": "Capability pressure rising.",
        "observed_at": "2026-07-16T12:00:00+00:00",
    }
    drive_diag = {"ok": True, "reason": "success", "correlation_id": "corr-drive-6"}
    with patch.object(
        mind_runtime,
        "fetch_drive_state_facet_for_mind",
        new=AsyncMock(return_value=(drive_compact, drive_diag)),
    ):
        asyncio.run(
            mind_runtime.prepare_plan_context_for_mind_projection(
                object(),
                source=ServiceRef(name="cortex-orch", version="0.2.0", node="test"),
                client_request=client,
                plan_request=plan,
                correlation_id="corr-drive-6",
            )
        )
    assert plan.context["drive_state_compact"]["dominant_drive"] == "capability"
    assert plan.context["metadata"]["mind_drive_state_fetch_diag"]["ok"] is True


def test_query_latest_drive_audit_row_uses_memory_extractors_pool_not_a_duplicate() -> None:
    """Regression guard: _query_latest_drive_audit_row must call
    memory_extractor's own _get_memory_pool(), not a second, independently
    maintained pool. A reverted implementation (mind_runtime keeping its own
    duplicate pool/lazy-init/fail-latch globals) would not call this mock at
    all and this test would fail with mock.assert_called_once() -- the
    earlier version of this fix shipped with no test able to tell shared-pool
    and duplicate-pool implementations apart."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock, patch

    _orch_prep()
    from app import mind_runtime as mod

    fake_pool = MagicMock()
    fake_pool.fetchrow = AsyncMock(return_value={"dominant_drive": "coherence"})

    with patch("app.memory_extractor._get_memory_pool", new=AsyncMock(return_value=fake_pool)) as mocked_get_pool:
        row = asyncio.run(mod._query_latest_drive_audit_row())

    mocked_get_pool.assert_called_once()
    fake_pool.fetchrow.assert_called_once()
    assert row == {"dominant_drive": "coherence"}
