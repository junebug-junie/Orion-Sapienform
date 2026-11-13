"""Self-readings reach stance_react and reverie (self-calibration spec section C, patch 4)."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from jinja2 import Environment

import app.self_readings_cue as cue_mod
import orion.attention.self_signal_outcomes as sso_mod
from app.bus_listener import build_stance_react_context, run_stance_react
from app.reverie import build_reverie_context
from orion.schemas.attention_frame import AttentionBroadcastProjectionV1, AttentionFrameV1, OpenLoopV1
from orion.schemas.self_reading import SelfReadingsBlockV1
from orion.schemas.thought import HubAssociationBundleV1, StanceReactRequestV1

_REPO = Path(__file__).resolve().parents[3]
_STANCE = _REPO / "orion" / "cognition" / "prompts" / "stance_react.j2"
_REVERIE = _REPO / "orion" / "cognition" / "prompts" / "reverie_narrate.j2"
_spec = importlib.util.spec_from_file_location(
    "self_readings_history", _REPO / "tests" / "fixtures" / "self_readings_history.py"
)
fx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fx)  # type: ignore[union-attr]
_BLOCK = "- surprise in how my task execution is going: 10/10 for me (now 1.00; higher than 97% of my last 7 days) [pe:execution@2026-10-14T03:29:30Z]\nNothing else I can read is unusual right now."
_COLORING = {"attention_frontier": [{"label": "x", "summary": "y", "score": 0.7}], "reflective_themes": [],
             "curiosity_threads": [], "self_relevance": None, "juniper_relevance": None,
             "identity_salience": None}


def _request() -> StanceReactRequestV1:
    return StanceReactRequestV1(
        correlation_id="c-1", session_id="s-1", user_message="how are you?",
        association=HubAssociationBundleV1(correlation_id="c-1", broadcast=None, broadcast_stale=True,
                                           read_source="hub_sql_fallback"),
        repair_bundle=None, stance_inputs={"user_message": "how are you?"},
    )


def _main_template(path: Path) -> str | None:
    rel = path.relative_to(_REPO).as_posix()
    try:
        return subprocess.run(["git", "-C", str(_REPO), "show", f"origin/main:{rel}"], check=True,
                              capture_output=True, text=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _render(template: str, **ctx) -> str:
    return Environment().from_string(template).render(**ctx)


_STANCE_BASE = {"user_message": "hi", "stance_inputs": {"user_message": "hi"}, "association": {},
                "repair_bundle": None, "coalition_projection": None}


# --- flag off / no reading: today's prompt, byte for byte -----------------------------------

def test_context_without_readings_is_unchanged() -> None:
    base = build_stance_react_context(_request())
    assert build_stance_react_context(_request(), self_readings="") == base
    assert build_stance_react_context(_request(), self_readings=None, label_estimates=False) == base
    assert "self_readings" not in base and "self_readings_label_estimates" not in base


@pytest.mark.parametrize("ctx", [dict(_STANCE_BASE), dict(_STANCE_BASE, mind_coloring=_COLORING)])
def test_stance_prompt_byte_identical_to_main_when_off(ctx) -> None:
    old = _main_template(_STANCE)
    if old is None or "MY BODY RIGHT NOW" in old:
        pytest.skip("origin/main template unavailable or already carries the block")
    assert _render(_STANCE.read_text(), **ctx) == _render(old, **ctx)


def test_reverie_prompt_byte_identical_to_main_when_off() -> None:
    old = _main_template(_REVERIE)
    if old is None or "HOW MY BODY IS DOING" in old:
        pytest.skip("origin/main template unavailable or already carries the block")
    b = _broadcast()
    ctx = build_reverie_context(b)
    assert "self_readings" not in ctx
    assert _render(_REVERIE.read_text(), **ctx) == _render(old, **ctx)


def _strip_blocks(template: str) -> str:
    """The template as if this patch's blocks had never been added (no git needed: CI is shallow).
    Removes each ``{% if self_readings... %}`` block up to its MATCHING endif (blocks may nest)."""
    import re

    tag = re.compile(r"\{%-?\s*(if|endif)\b[^%]*%\}")
    opener = re.compile(r"\{% if self_readings(?:_label_estimates)? %\}")
    out = template
    while (m := opener.search(out)) is not None:
        depth, pos = 0, m.start()
        for t in tag.finditer(out, pos):
            depth += 1 if t.group(1) == "if" else -1
            if depth == 0:
                out = out[: m.start()] + out[t.end():]
                break
        else:
            raise AssertionError("unbalanced if/endif")
    return out


@pytest.mark.parametrize("ctx", [dict(_STANCE_BASE), dict(_STANCE_BASE, mind_coloring=_COLORING)])
def test_stance_blocks_leave_no_residue_when_off(ctx) -> None:
    text = _STANCE.read_text()
    assert _strip_blocks(text) != text
    assert _render(text, **ctx) == _render(_strip_blocks(text), **ctx)


def test_reverie_block_leaves_no_residue_when_off() -> None:
    text = _REVERIE.read_text()
    ctx = build_reverie_context(_broadcast())
    assert _strip_blocks(text) != text
    assert _render(text, **ctx) == _render(_strip_blocks(text), **ctx)


# --- flag on ---------------------------------------------------------------------------------

def test_context_carries_readings_and_estimate_label() -> None:
    ctx = build_stance_react_context(_request(), self_readings=_BLOCK, label_estimates=True)
    assert ctx["self_readings"] == _BLOCK
    assert "self_readings_label_estimates" not in ctx  # no Mind numbers to label
    ctx = build_stance_react_context(_request(), mind_coloring=_COLORING, self_readings=_BLOCK,
                                     label_estimates=True)
    assert ctx["self_readings_label_estimates"] is True


def test_stance_prompt_renders_block_and_labels_mind_numbers() -> None:
    out = _render(_STANCE.read_text(), **_STANCE_BASE, self_readings=_BLOCK, mind_coloring=_COLORING,
                  self_readings_label_estimates=True)
    assert "MY BODY RIGHT NOW" in out and "[pe:execution@2026-10-14T03:29:30Z]" in out
    assert "the task wins" in out and "not evidence_refs" in out
    assert "Oríon's own estimates from Mind, not measurements" in out


def _broadcast() -> AttentionBroadcastProjectionV1:
    loop = OpenLoopV1(id="ol-1", description="execution spiked", salience=0.7)
    return AttentionBroadcastProjectionV1(frame=AttentionFrameV1(open_loops=[loop]),
                                          selected_open_loop_id="ol-1", attended_node_ids=["node:substrate.execution"])


def test_reverie_coalition_branch_renders_block_concern_branch_never() -> None:
    ctx = build_reverie_context(_broadcast(), self_readings=_BLOCK)
    assert ctx["self_readings"] == _BLOCK
    out = _render(_REVERIE.read_text(), **ctx)
    assert "HOW MY BODY IS DOING" in out and "[pe:execution@" in out
    from orion.schemas.reverie import ConcernCardV1

    card = ConcernCardV1.from_harness_turn(
        coalition_ref="harness_closure:c", user_message_excerpt="Will the deploy slip?",
        stance_imperative="Name the tradeoff.", created_at=datetime(2026, 10, 14, tzinfo=timezone.utc))
    lifted = build_reverie_context(_broadcast(), concern_cards=[card], self_readings=_BLOCK)
    assert "self_readings" not in lifted


# --- the cue: fail-open, bounded --------------------------------------------------------------

def _fresh_cache(monkeypatch, associations=None):
    monkeypatch.setattr(cue_mod, "_cache", None)
    monkeypatch.setattr(cue_mod, "_assoc", None)
    monkeypatch.setattr(cue_mod, "fetch_associations", lambda since, holds: list(associations or []))


@pytest.mark.asyncio
async def test_disabled_never_reads(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(cue_mod, "fetch_since", lambda since: calls.append(since) or [])
    block = await cue_mod.build_self_readings_block("c-1", purpose="stance", enabled=False)
    assert block.skip_reason == "disabled" and block.text == "" and calls == []


@pytest.mark.asyncio
async def test_read_error_renders_nothing(monkeypatch) -> None:
    _fresh_cache(monkeypatch)

    def boom(since, holds):
        raise RuntimeError("db down")

    monkeypatch.setattr(cue_mod, "fetch_since", boom)
    block = await cue_mod.build_self_readings_block("c-1", purpose="stance", enabled=True, now=fx.NOW)
    assert block.text == "" and block.skip_reason == "fetch_failed"


@pytest.mark.asyncio
async def test_slow_read_is_bounded(monkeypatch) -> None:
    _fresh_cache(monkeypatch)
    import time

    monkeypatch.setattr(cue_mod, "fetch_since", lambda since: time.sleep(1.0) or [])
    monkeypatch.setattr(cue_mod.settings, "self_readings_fetch_timeout_sec", 0.05)
    t0 = asyncio.get_running_loop().time()
    block = await cue_mod.build_self_readings_block("c-1", purpose="stance", enabled=True, now=fx.NOW)
    assert block.skip_reason == "fetch_timeout" and asyncio.get_running_loop().time() - t0 < 0.8
    # Drain the timed-out worker thread so it cannot touch the next test's cache.
    assert await asyncio.to_thread(cue_mod._lock.acquire, True, 5.0)
    cue_mod._lock.release()


@pytest.mark.asyncio
async def test_real_rows_render_a_block(monkeypatch) -> None:
    _fresh_cache(monkeypatch)
    rows = fx.busy_body_rows()
    monkeypatch.setattr(cue_mod, "fetch_since", lambda since: [r for r in rows if r[1] >= since])
    monkeypatch.setattr(cue_mod.settings, "self_readings_fetch_timeout_sec", 5.0)
    block = await cue_mod.build_self_readings_block("c-1", purpose="stance", enabled=True, now=fx.NOW)
    assert "[pe:execution@2026-10-14T03:29:30Z]" in block.text
    assert len(block.text) <= 600


class _CapturingClient:
    def __init__(self) -> None:
        self.contexts: list[dict] = []

    async def execute_plan(self, **kwargs) -> dict:
        self.contexts.append(kwargs["req"].context)
        return {"final_text": json.dumps({
            "imperative": "Answer.", "tone": "plain", "strain_refs": [], "evidence_refs": ["hub:turn"],
            "stance_harness_slice": {"task_mode": "direct_response", "conversation_frame": "technical",
                                     "answer_strategy": "direct"}})}


@pytest.mark.asyncio
async def test_run_stance_react_passes_the_block(monkeypatch) -> None:
    async def fixed(corr, *, purpose, enabled, now=None):
        assert purpose == "stance" and enabled is True
        return SelfReadingsBlockV1(rendered_at=fx.NOW, text=_BLOCK, shown=["node:substrate.execution"])

    monkeypatch.setitem(run_stance_react.__globals__, "build_self_readings_block", fixed)
    client = _CapturingClient()
    await run_stance_react(_request(), bus=None, cortex_client=client)
    assert client.contexts[0]["self_readings"] == _BLOCK


@pytest.mark.asyncio
async def test_run_stance_react_flag_off_leaves_context_untouched(monkeypatch) -> None:
    settings_obj = run_stance_react.__globals__["settings"]
    monkeypatch.setattr(settings_obj, "self_readings_stance_enabled", False)
    client = _CapturingClient()
    await run_stance_react(_request(), bus=None, cortex_client=client)
    assert "self_readings" not in client.contexts[0]
    assert "self_readings_label_estimates" not in client.contexts[0]


def test_importing_the_cue_does_not_load_substrate() -> None:
    """The thin-import boundary (test_reverie_thin_import_boundary.py): the calibration modules
    load lazily, inside the worker thread."""
    import os
    import sys

    service_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(service_root), str(_REPO)]))
    code = ("import sys\nimport app.self_readings_cue, app.bus_listener, app.reverie\n"
            "sub=[m for m in sys.modules if m.startswith('orion.substrate')]\nassert not sub, sub\nprint('ok')\n")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                            cwd=str(service_root))
    assert result.returncode == 0, result.stderr


# --- "tends to" clauses (spec section E) ---------------------------------------------------------

_ASSOC_ROW = {
    "association_id": "ssa-0123456789", "computed_at": fx.NOW, "signal_id": "node:substrate.execution",
    "outcome": "rpc_timeout", "window_start": None, "window_end": None, "n_high": 42, "events_high": 13,
    "n_normal": 120, "events_normal": 12, "rate_high": 13 / 42, "rate_normal": 0.1, "ratio": 3.0952,
    "ci_low": 1.6, "ci_high": 6.0, "ci_method": "katz_log_rr_95", "min_hours": 24, "verdict": "tends_more",
    "excluded_reason": "",
    "clause": ("when this is high, my next hour had an RPC timeout 3.1x as often at equal activity (31% of 42 high hours "
               "vs 10% of 120 normal) [assoc:ssa-0123456789]"),
}


def _held(row: dict, n: int = 6, *, gap_at: int | None = None, flip_at: int | None = None) -> list[dict]:
    """``row`` as the newest of ``n`` hourly computations (older ones get their own ids), with an
    optional missing hour (``gap_at`` hours back) or a different verdict ``flip_at`` hours back."""
    from datetime import timedelta

    out = []
    for i in range(n):
        if i == gap_at:
            continue
        r = dict(row) if i == 0 else dict(row, association_id=f"ssa-{i:010d}",
                                          computed_at=row["computed_at"] - timedelta(hours=i))
        if i == flip_at:
            r.update(verdict="no_clear_difference", clause="")
        out.append(r)
    return out


async def _block_with(monkeypatch, *, enabled: bool, assoc_rows):
    _fresh_cache(monkeypatch, assoc_rows)
    if enabled:
        cue_mod.refresh_associations(fx.NOW)  # what startup warm / the background refresh do
    rows = [r for r in fx.busy_body_rows() if r[0] != "node:substrate.cabinet"]  # two readings: room
    monkeypatch.setattr(cue_mod, "fetch_since", lambda since: [r for r in rows if r[1] >= since])
    monkeypatch.setattr(cue_mod.settings, "self_readings_fetch_timeout_sec", 5.0)
    monkeypatch.setattr(cue_mod.settings, "self_signal_outcome_association_enabled", enabled)
    return await cue_mod.build_self_readings_block("c-1", purpose="stance", enabled=True, now=fx.NOW)


@pytest.mark.asyncio
async def test_flag_on_renders_the_cited_clause(monkeypatch) -> None:
    block = await _block_with(monkeypatch, enabled=True, assoc_rows=_held(_ASSOC_ROW))
    assert "[pe:execution@2026-10-14T03:29:30Z]; when this is high" in block.text
    assert "[assoc:ssa-0123456789]" in block.text


@pytest.mark.asyncio
@pytest.mark.parametrize("rows", [
    pytest.param(_held(_ASSOC_ROW, 5), id="five-of-six-holds"),
    pytest.param(_held(_ASSOC_ROW, 7, gap_at=3), id="missing-hour"),
    pytest.param(_held(_ASSOC_ROW, flip_at=2), id="verdict-flip"),
    pytest.param([_ASSOC_ROW], id="one-hour-only"),
])
async def test_a_clause_that_has_not_held_six_hours_renders_nothing(monkeypatch, rows) -> None:
    base = await _block_with(monkeypatch, enabled=True, assoc_rows=[])
    block = await _block_with(monkeypatch, enabled=True, assoc_rows=rows)
    assert block.text == base.text and "when this is high" not in block.text


@pytest.mark.asyncio
async def test_a_gap_older_than_the_run_does_not_matter(monkeypatch) -> None:
    block = await _block_with(monkeypatch, enabled=True, assoc_rows=_held(_ASSOC_ROW, 8, gap_at=6))
    assert "[assoc:ssa-0123456789]" in block.text


@pytest.mark.asyncio
async def test_a_stale_newest_row_renders_nothing_even_when_held(monkeypatch) -> None:
    from datetime import timedelta

    base = await _block_with(monkeypatch, enabled=True, assoc_rows=[])
    old = dict(_ASSOC_ROW, computed_at=fx.NOW - timedelta(hours=4))
    block = await _block_with(monkeypatch, enabled=True, assoc_rows=_held(old))
    assert block.text == base.text


def test_reader_asks_for_enough_history_for_the_holds_rule(monkeypatch) -> None:
    seen = []
    _fresh_cache(monkeypatch)
    monkeypatch.setattr(cue_mod, "fetch_associations", lambda since, holds: seen.append((since, holds)) or [])
    monkeypatch.setattr(cue_mod.settings, "self_signal_outcome_consecutive_holds", 6)
    cue_mod.refresh_associations(fx.NOW)
    (since, holds), = seen
    assert holds == 6
    # The newest row may be up to ASSOCIATION_MAX_AGE old; its run reaches 6 hours further back.
    assert fx.NOW - since >= sso_mod.ASSOCIATION_MAX_AGE + timedelta(hours=6)


@pytest.mark.asyncio
async def test_flag_off_is_byte_identical_and_never_reads_associations(monkeypatch) -> None:
    base = await _block_with(monkeypatch, enabled=True, assoc_rows=[])
    off = await _block_with(monkeypatch, enabled=False, assoc_rows=[_ASSOC_ROW])
    assert off.text == base.text and "when this is high" not in off.text
    monkeypatch.setattr(cue_mod.settings, "self_signal_outcome_association_enabled", False)
    calls = []
    monkeypatch.setattr(cue_mod, "_assoc", None)
    monkeypatch.setattr(cue_mod, "fetch_associations", lambda since, holds: calls.append(since) or [_ASSOC_ROW])
    assert cue_mod._associations(fx.NOW) is None
    assert cue_mod._assoc is None and calls == []  # flag off never reads, not even in background


@pytest.mark.asyncio
async def test_unreadable_associations_leave_readings_unchanged(monkeypatch) -> None:
    base = await _block_with(monkeypatch, enabled=True, assoc_rows=[])

    def boom(since, holds):
        raise RuntimeError("no such table: self_signal_outcome_association")

    _fresh_cache(monkeypatch)
    monkeypatch.setattr(cue_mod, "fetch_associations", boom)
    cue_mod.refresh_associations(fx.NOW)  # never raises
    rows = [r for r in fx.busy_body_rows() if r[0] != "node:substrate.cabinet"]
    monkeypatch.setattr(cue_mod, "fetch_since", lambda since: [r for r in rows if r[1] >= since])
    block = await cue_mod.build_self_readings_block("c-1", purpose="stance", enabled=True, now=fx.NOW)
    assert block.text == base.text


@pytest.mark.asyncio
async def test_a_row_without_a_clause_or_malformed_renders_nothing_extra(monkeypatch) -> None:
    base = await _block_with(monkeypatch, enabled=True, assoc_rows=[])
    quiet = dict(_ASSOC_ROW, verdict="below_min_n", clause="")
    broken = dict(_ASSOC_ROW, association_id="ssa-1111111111", verdict="nonsense")
    block = await _block_with(monkeypatch, enabled=True, assoc_rows=[quiet, broken])
    assert block.text == base.text


def test_association_query_is_fresh_bounded_and_one_row_per_pair_hour() -> None:
    from orion.attention.self_signal_outcomes import association_history_limit, association_history_sql

    sql = association_history_sql(":since", ":limit")
    assert "DISTINCT ON (signal_id, outcome, date_trunc('hour', computed_at AT TIME ZONE 'UTC'))" in sql
    assert "computed_at >= :since" in sql and sql.endswith("LIMIT :limit")
    assert association_history_limit(6) == 32 * 3 * 11
    assert sql.rstrip().endswith("ORDER BY computed_at DESC, signal_id, outcome LIMIT :limit")
    assert sso_mod.ASSOCIATION_MAX_AGE.total_seconds() == 3 * 3600


def test_flag_ships_on() -> None:
    from app.settings import ThoughtSettings

    assert ThoughtSettings(_env_file=None).self_signal_outcome_association_enabled is True
    env = (Path(__file__).resolve().parents[1] / ".env_example").read_text()
    assert "SELF_SIGNAL_OUTCOME_ASSOCIATION_ENABLED=true" in env
    assert ThoughtSettings(_env_file=None).self_signal_outcome_consecutive_holds == 6
    assert "SELF_SIGNAL_OUTCOME_CONSECUTIVE_HOLDS=6" in env


def test_a_stale_index_never_reads_on_the_callers_thread(monkeypatch) -> None:
    """The association query runs in a background thread: a slow table cannot cost a turn its
    self-readings (0.5 s budget) or hold the history lock."""
    import threading
    import time

    _fresh_cache(monkeypatch)
    monkeypatch.setattr(cue_mod.settings, "self_signal_outcome_association_enabled", True)
    release = threading.Event()
    monkeypatch.setattr(cue_mod, "fetch_associations", lambda since, holds: release.wait(5) and _held(_ASSOC_ROW))
    t0 = time.perf_counter()
    assert cue_mod._associations(fx.NOW) is None  # nothing loaded yet: no clause, no wait
    assert cue_mod._associations(fx.NOW) is None  # a refresh is in flight: not a second one
    assert time.perf_counter() - t0 < 0.2
    release.set()
    for _ in range(100):
        if cue_mod._assoc is not None:
            break
        time.sleep(0.02)
    got = cue_mod._associations(fx.NOW)
    assert got is not None and got["node:substrate.execution"].association_id == "ssa-0123456789"
