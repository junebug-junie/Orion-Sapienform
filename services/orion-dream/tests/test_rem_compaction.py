"""Phase F — REM compaction narration (staged, applies nothing).

The load-bearing invariants: the delta is always a proposal, the assembly is
deterministic (code owns ops/metrics), a pass never raises, and Phase F performs
zero canonical writes (only the staging table).
"""
from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from orion.schemas.compaction import (
    MAX_CONSOLIDATE,
    MemoryCompactionDeltaV1,
)


def _req(theme="loop:ol-1", op_hint="consolidate", evidence=("ol-1", "ol-2"), rid="r-1"):
    return {
        "request_id": rid,
        "theme": theme,
        "op_hint": op_hint,
        "evidence_refs": list(evidence),
    }


# --- schema invariants --------------------------------------------------------

def test_delta_is_always_proposal_marked():
    d = MemoryCompactionDeltaV1(delta_id="d-1")
    assert d.proposal_marked is True
    # The literal type makes "applied fact" unrepresentable at the schema level.
    with pytest.raises(Exception):
        MemoryCompactionDeltaV1(delta_id="d-2", proposal_marked=False)


def test_consolidate_list_capped_by_schema():
    from orion.schemas.compaction import ConsolidateEntryV1

    with pytest.raises(Exception):
        MemoryCompactionDeltaV1(
            delta_id="d-3",
            consolidate=[ConsolidateEntryV1(gist_card="x") for _ in range(MAX_CONSOLIDATE + 1)],
        )


# --- deterministic assembly ---------------------------------------------------

def test_build_delta_consolidates_only_consolidate_hints():
    from app.rem_compaction import build_compaction_delta

    delta = build_compaction_delta(
        [_req(op_hint="consolidate", rid="a"), _req(op_hint="prune", rid="b")]
    )
    assert delta.metrics.cards_out == 1  # the prune-hint request is not fabricated here
    assert delta.source_request_ids == ["a"]
    # Phase F never invents downscale/prune ops from the awake path.
    assert delta.downscale == []
    assert delta.prune == []


def test_build_delta_is_deterministic_without_narrator():
    from app.rem_compaction import build_compaction_delta

    reqs = [_req(rid="a"), _req(theme="loop:ol-2", rid="b")]
    d1 = build_compaction_delta(reqs)
    d2 = build_compaction_delta(reqs)
    # Same requests → same gist cards + metrics (delta_id differs by design).
    assert [c.gist_card for c in d1.consolidate] == [c.gist_card for c in d2.consolidate]
    assert d1.metrics.cards_out == d2.metrics.cards_out == 2


def test_build_delta_narrator_failure_falls_back_to_floor():
    from app.rem_compaction import build_compaction_delta

    def boom(theme, evidence):
        raise RuntimeError("narrator exploded")

    delta = build_compaction_delta([_req()], narrator=boom)
    assert delta.metrics.cards_out == 1
    assert delta.consolidate[0].gist_card  # non-empty floor card, not a crash


def test_build_delta_empty_requests_is_empty_night():
    from app.rem_compaction import build_compaction_delta

    delta = build_compaction_delta([])
    assert delta.is_empty()
    assert delta.metrics.cards_out == 0


def test_build_delta_skips_themeless_requests():
    from app.rem_compaction import build_compaction_delta

    delta = build_compaction_delta([_req(theme="", rid="x")])
    assert delta.is_empty()


# --- orchestration ------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_disabled_returns_none(monkeypatch):
    from app import rem_compaction
    from app.settings import settings

    monkeypatch.setattr(settings, "ORION_DREAM_REM_ENABLED", False)
    out = await rem_compaction.run_rem_compaction_once(AsyncMock())
    assert out is None


@pytest.mark.asyncio
async def test_run_empty_queue_returns_none(monkeypatch):
    from app import rem_compaction
    from app.settings import settings

    monkeypatch.setattr(settings, "ORION_DREAM_REM_ENABLED", True)
    out = await rem_compaction.run_rem_compaction_once(
        AsyncMock(), request_loader=lambda n: [], delta_persister=lambda d: True
    )
    assert out is None  # nothing settled → honest empty, not empty-shell success


@pytest.mark.asyncio
async def test_run_publishes_and_persists_staged_proposal(monkeypatch):
    from app import rem_compaction
    from app.settings import settings

    monkeypatch.setattr(settings, "ORION_DREAM_REM_ENABLED", True)
    persisted = []
    bus = AsyncMock()
    out = await rem_compaction.run_rem_compaction_once(
        bus,
        request_loader=lambda n: [_req()],
        delta_persister=lambda d: persisted.append(d) or True,
    )
    assert out is not None
    assert out.proposal_marked is True
    assert persisted and persisted[0] is out
    bus.publish.assert_awaited_once()
    # Published on the delta channel, applied nothing.
    channel = bus.publish.await_args.args[0]
    assert channel == settings.CHANNEL_DREAM_COMPACTION_DELTA


@pytest.mark.asyncio
async def test_run_never_raises_on_loader_error(monkeypatch):
    from app import rem_compaction
    from app.settings import settings

    monkeypatch.setattr(settings, "ORION_DREAM_REM_ENABLED", True)

    def boom(n):
        raise RuntimeError("db down")

    out = await rem_compaction.run_rem_compaction_once(AsyncMock(), request_loader=boom)
    assert out is None  # degraded, not raised


# --- the "applies nothing" invariant, enforced structurally -------------------

def test_rem_store_only_writes_the_staging_table():
    """Phase F must perform zero canonical writes. Scan the store SQL for any
    INSERT/UPDATE/DELETE against a table other than the staging table."""
    src = Path(__file__).resolve().parents[1] / "app" / "rem_store.py"
    text = src.read_text(encoding="utf-8").lower()
    # No write verb may target a canonical memory table.
    forbidden = (
        "substrate_episode_summaries",
        "substrate_consolidation_frames",
        "substrate_memory",
        "collapse",
        "dream_compaction_request_queue",  # queue is read-only in Phase F
    )
    for verb in ("insert into", "update ", "delete from"):
        for table in forbidden:
            assert f"{verb}{table}" not in text.replace("  ", " "), (
                f"Phase F must not {verb.strip()} {table}"
            )
    # The one permitted write is the staging table.
    assert "insert into dream_compaction_delta" in text


def test_default_request_loader_accepts_positional_limit():
    """Regression: the default loader must match the positional RequestLoader
    alias so the real (non-injected) REM pass doesn't TypeError into a false
    empty night. Degrades to [] here (no DB), which is the point — it must not
    raise on a positional call."""
    from app.rem_store import load_pending_requests

    assert load_pending_requests(5) == []  # positional, best-effort → []


def test_delta_kind_resolves_via_runtime_registry():
    """Regression: the delta must be resolvable by the runtime `_REGISTRY`
    (resolve()), not just present in SCHEMA_REGISTRY — otherwise bus publish of
    dream.compaction.delta.v1 raises on every send."""
    from orion.schemas.registry import resolve

    model = resolve("MemoryCompactionDeltaV1")
    assert model is MemoryCompactionDeltaV1


def test_rem_modules_never_import_an_applier():
    """Phase F must not reach into any apply/mutation path (that is Phase G)."""
    for name in ("rem_compaction.py", "rem_store.py"):
        src = Path(__file__).resolve().parents[1] / "app" / name
        tree = ast.parse(src.read_text(encoding="utf-8"))
        mods: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods.append(node.module or "")
        for mod in mods:
            assert "appl" not in mod.lower(), f"{name} must not import an applier ({mod!r})"
