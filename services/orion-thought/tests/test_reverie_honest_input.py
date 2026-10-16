"""Reverie honest input + a chain that develops (2026-10-16).

Patch A: the prompt carries each loop's real measurement, the world-first
reason instead of chat-turn boilerplate, only fresh verdicts with who recorded
them, and says one thing won attention. A calm tick starts no chain.
Patch B: chain steps feed forward (earlier thoughts + next_focus) and a step
that returns no next_focus ends the chain.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from jinja2 import Environment

from orion.schemas.attention_frame import (
    AttentionBroadcastProjectionV1,
    AttentionFrameV1,
    OpenLoopV1,
    PredictionErrorMagnitudeV1,
)
from orion.schemas.reverie import SpontaneousThoughtV1
from orion.schemas.thought import CoalitionSnapshotV1

NOW = datetime(2026, 10, 16, 3, 0, tzinfo=timezone.utc)
_REPO = Path(__file__).resolve().parents[3]
_PROMPT = _REPO / "orion" / "cognition" / "prompts" / "reverie_narrate.j2"
GROUNDED = "The execution signal ol-1 is far above its own usual week and nothing has explained it yet."
BOILERPLATE = "novel or unresolved current-turn target with substrate pressure"


def _render(**ctx) -> str:
    return Environment().from_string(_PROMPT.read_text()).render(**ctx)


def _magnitude(**over):
    base = dict(value=0.38, age_sec=47.0, p50_7d=0.0, p90_7d=0.19, percentile_now=0.988,
                n_readings_7d=3000, band="high", trend="flat")
    base.update(over)
    return PredictionErrorMagnitudeV1(**base)


def _substrate_loop(oid="ol-1", *, reason="body unusual for itself (high), event 47s old (fade 1.00)",
                    magnitude=True):
    prov = {"signal_source": "substrate_broadcast"}
    if reason:
        prov["world_first_reason"] = reason
    return OpenLoopV1(
        id=oid, description="Execution prediction error", why_it_matters=BOILERPLATE,
        target_type="anomaly", provenance=prov, salience=0.8,
        magnitude=_magnitude() if magnitude else None,
    )


def _broadcast(loops=None, attended=("node:substrate.execution",), selected="ol-1"):
    return AttentionBroadcastProjectionV1(
        frame=AttentionFrameV1(open_loops=list(loops if loops is not None else [_substrate_loop()])),
        attended_node_ids=list(attended),
        selected_open_loop_id=selected,
        selected_description="Execution prediction error" if selected else None,
    )


# --- 1. measurement rendering -------------------------------------------------

def test_measurement_renders_real_numbers_in_plain_words():
    from app import reverie

    text = reverie.loop_measurement_text(_substrate_loop())
    assert text == "reading 0.38 (usual 0.00, 90th pct 0.19; top 1.2% of its own week, band high)"


def test_missing_measurement_renders_nothing():
    from app import reverie

    loop = _substrate_loop(magnitude=False)
    assert reverie.loop_measurement_text(loop) == ""
    entry = reverie._open_loops_for_prompt(_broadcast([loop]))[0]
    assert "reading" not in entry and "reading_age" not in entry
    out = _render(**reverie.build_reverie_context(_broadcast([loop])))
    assert "measured:" not in out


def test_missing_magnitude_fields_are_left_out_not_placeholdered():
    from app import reverie

    loop = _substrate_loop()
    loop.magnitude = _magnitude(p50_7d=None, p90_7d=None, percentile_now=None,
                                band="insufficient_history")
    text = reverie.loop_measurement_text(loop)
    assert text == "reading 0.38 (not enough history to say what is usual)"
    assert "None" not in text and "usual 0" not in text


def test_reading_age_only_when_no_reason_says_it():
    from app import reverie

    with_reason = reverie._open_loops_for_prompt(_broadcast())[0]
    assert "reading_age" not in with_reason  # the reason already says "event 47s old"
    no_reason = reverie._open_loops_for_prompt(_broadcast([_substrate_loop(reason=None)]))[0]
    assert no_reason["reading_age"] == "last reading 47s ago"


# --- 2. why_it_matters --------------------------------------------------------

def test_substrate_loop_why_is_the_world_first_reason_not_boilerplate():
    from app import reverie

    entry = reverie._open_loops_for_prompt(_broadcast())[0]
    assert entry["why_it_matters"] == "body unusual for itself (high), event 47s old (fade 1.00)"
    out = _render(**reverie.build_reverie_context(_broadcast()))
    assert BOILERPLATE not in out
    assert "why it surfaced: body unusual for itself (high)" in out
    assert "measured: reading 0.38" in out


def test_substrate_loop_with_no_reason_drops_the_false_boilerplate():
    from app import reverie

    entry = reverie._open_loops_for_prompt(_broadcast([_substrate_loop(reason=None)]))[0]
    assert entry["why_it_matters"] == ""


def test_non_substrate_loop_keeps_its_own_why():
    from app import reverie

    loop = OpenLoopV1(id="ol-1", description="d", why_it_matters="Juniper asked twice")
    entry = reverie._open_loops_for_prompt(_broadcast([loop]))[0]
    assert entry["why_it_matters"] == "Juniper asked twice"


# --- 3. outcome freshness + actor ---------------------------------------------

def test_stale_system_verdict_is_not_attached():
    from app import store

    rows = [{"loop_id": "ol-1", "verdict": "decayed_unattended", "note": "silent 24.1h",
             "actor": "system:implicit_decay", "created_at": NOW - timedelta(days=25)}]
    assert store.fresh_loop_outcomes(rows, now=NOW) == {}


def test_fresh_human_verdict_is_attached_and_labelled():
    from app import reverie, store

    rows = [{"loop_id": "ol-1", "verdict": "dismissed", "note": "known noise",
             "actor": "juniper", "created_at": NOW - timedelta(hours=3)}]
    outcomes = store.fresh_loop_outcomes(rows, now=NOW)
    assert outcomes["ol-1"]["human"] is True
    out = _render(**reverie.build_reverie_context(_broadcast(), loop_outcomes=outcomes))
    assert "earlier verdict: dismissed, 3 hours ago, recorded by Juniper" in out
    assert "(a person closing this loop) is ground truth" in out
    assert "is bookkeeping, not a" not in out  # no system verdict


def test_fresh_system_verdict_is_labelled_as_system_not_ground_truth():
    from app import reverie, store

    rows = [{"loop_id": "ol-1", "verdict": "decayed_unattended", "note": "silent 24.1h",
             "actor": "system:implicit_decay", "created_at": NOW - timedelta(hours=2)}]
    outcomes = store.fresh_loop_outcomes(rows, now=NOW)
    out = _render(**reverie.build_reverie_context(_broadcast(), loop_outcomes=outcomes))
    assert "recorded automatically by the system (system:implicit_decay)" in out
    assert "is bookkeeping, not a" in out
    assert "(a person closing this loop) is ground truth" not in out  # no human verdict


def test_freshness_window_mirrors_the_substrate_verdict_ttl():
    """orion-thought cannot import orion.substrate; pin the copied constant by text."""
    from app import store

    src = (_REPO / "orion" / "substrate" / "attention" / "verdicts.py").read_text()
    ttl = float(re.search(r"^VERDICT_EXCLUSION_TTL_HOURS = ([0-9.]+)", src, re.M).group(1))
    assert store.LOOP_OUTCOME_FRESH_HOURS == ttl


@pytest.mark.parametrize("actor,human", [
    ("juniper", True), ("system:implicit_decay", False), ("orion", False), ("", False),
])
def test_actor_classification(actor, human):
    from app import store

    assert store.loop_outcome_actor_is_human(actor) is human


# --- 4. prompt wording: one thing won -----------------------------------------

def test_coalition_branch_says_one_thing_won_not_a_coalition_of_concerns():
    from app import reverie

    out = _render(**reverie.build_reverie_context(_broadcast()))
    assert "One thing won your attention this tick: Execution prediction error [ol-1]" in out
    assert "coalition of concerns" not in out
    assert "ol-1 (won): Execution prediction error" in out


# --- 5. calm tick starts no chain ---------------------------------------------

@pytest.mark.asyncio
async def test_calm_tick_creates_no_chain_row_and_no_refractory(monkeypatch):
    from app import chain

    persisted, suppressed = [], []
    monkeypatch.setattr(chain, "persist_reverie_chain", lambda c: persisted.append(c))

    class _Store(chain.InMemoryRefractoryStore):
        def suppress(self, theme_key, until):
            suppressed.append(theme_key)

    step = AsyncMock()
    c = await chain.run_reverie_chain(
        AsyncMock(), step_fn=step, refractory_store=_Store(),
        broadcast_reader=lambda: _broadcast([], attended=(), selected=None),
        publish=True, now_fn=lambda: NOW,
    )
    assert c is None
    step.assert_not_awaited()
    assert persisted == [] and suppressed == []
