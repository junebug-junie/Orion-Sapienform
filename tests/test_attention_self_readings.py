"""Self-readings (self-calibration spec section C): build + render, pure."""

from __future__ import annotations

import importlib.util
from datetime import timedelta
from pathlib import Path

import pytest

import orion.attention.self_readings as sr
import orion.attention.world_first as wf
from orion.attention.pe_history_cache import NodePeHistoryCache

_spec = importlib.util.spec_from_file_location(
    "self_readings_history", Path(__file__).parent / "fixtures" / "self_readings_history.py"
)
fx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fx)  # type: ignore[union-attr]
NOW = fx.NOW


def _cache(rows, now=NOW) -> NodePeHistoryCache:
    cache = NodePeHistoryCache()
    cache.refresh(lambda since: [r for r in rows if r[1] >= since], now=now)
    return cache


def _by_id(readings):
    return {r.source_id: r for r in readings}


def test_prompt_reaching_nodes_come_from_the_glossary_and_exclude_world_and_placeholder() -> None:
    nodes = sr.prompt_reaching_nodes()
    assert nodes == sorted([
        "node:substrate.biometrics", "node:substrate.bus_synaptic", "node:substrate.cabinet",
        "node:substrate.chat", "node:substrate.codebase", "node:substrate.execution",
        "node:substrate.route", "node:substrate.strain",
    ])
    assert "node:substrate.perception" not in nodes  # the world, not the body
    assert "node:substrate.harness_closure" not in nodes  # placeholder


def test_calm_body_renders_nothing() -> None:
    block = sr.project_self_readings(_cache(fx.calm_rows()), now=NOW)
    assert block.text == ""
    assert block.skip_reason == "all_calm"
    assert all(not r.unusual for r in block.readings)
    assert {r.state for r in block.readings} == {"live"}


def test_missing_history_renders_nothing_and_is_not_calm() -> None:
    block = sr.project_self_readings(NodePeHistoryCache(), now=NOW)
    assert block.text == ""
    assert block.skip_reason == "no_history"  # never "all_calm" without a single live reading
    assert {r.state for r in block.readings} == {"no_reading"}
    assert all(r.value is None and r.score_0_10 is None for r in block.readings)


def test_stale_history_renders_nothing() -> None:
    later = NOW + timedelta(hours=2)  # every reading older than the 1800 s horizon
    block = sr.project_self_readings(_cache(fx.busy_body_rows()), now=later)
    assert block.text == ""
    assert {r.state for r in block.readings} == {"stale"}


def test_a_real_high_signal_renders_with_its_row_ref() -> None:
    block = sr.project_self_readings(_cache(fx.busy_body_rows()), now=NOW)
    assert block.skip_reason is None
    by = _by_id(block.readings)
    ex = by["node:substrate.execution"]
    assert ex.unusual and ex.state == "live" and ex.value == 1.0
    assert ex.ref == "pe:execution@2026-10-14T03:29:30Z"
    assert ex.source_ref.startswith("substrate_node_prediction_error_history:node:substrate.execution@")
    assert f"[{ex.ref}]" in block.text
    assert "node:substrate.execution" in block.shown
    assert block.text.endswith("Nothing else I can read is unusual right now.")


def test_trigger_reads_as_a_rate_never_a_magnitude() -> None:
    block = sr.project_self_readings(_cache(fx.busy_body_rows()), now=NOW)
    cab = _by_id(block.readings)["node:substrate.cabinet"]
    assert cab.value_kind == "trigger"
    assert cab.rate_window_sec == 3600.0
    assert cab.value == 2.0  # 2 onsets in the last hour: not 36 "on" checks, not the 0.8 magnitude
    assert cab.unusual
    assert "went off 2 times in the last hour" in cab.plain
    # The ref names the newest ON row, not the trailing 0.0 write.
    assert cab.ref == "pe:cabinet@2026-10-14T03:13:30Z"


def test_trigger_onsets_count_transitions_not_rows() -> None:
    t = [NOW - timedelta(seconds=30 * i) for i in range(10, 0, -1)]
    vals = [0, 1, 1, 1, 0, 0, 1, 1, 0, 1]
    assert sr.trigger_onsets(list(zip(t, vals))) == [t[1], t[6], t[9]]


def test_placeholder_never_rendered_as_a_measurement() -> None:
    rows = fx.busy_body_rows() + [("node:substrate.harness_closure", NOW - timedelta(seconds=5), 0.65)]
    readings = sr.build_self_readings(_cache(rows), now=NOW, nodes=["node:substrate.harness_closure"])
    assert readings == []
    block = sr.project_self_readings(_cache(rows), now=NOW)
    assert "harness" not in block.text and "0.65" not in block.text


def test_absent_signal_is_named_unreadable_not_calm() -> None:
    rows = [r for r in fx.busy_body_rows() if r[0] != "node:substrate.chat"]
    block = sr.project_self_readings(_cache(rows), now=NOW)
    chat = _by_id(block.readings)["node:substrate.chat"]
    assert chat.state == "no_reading" and not chat.unusual
    assert "Can't read right now: chat (no reading)." in block.text


def test_score_is_polarity_aware(monkeypatch) -> None:
    rows = fx.busy_body_rows()
    base = _by_id(sr.build_self_readings(_cache(rows), now=NOW))["node:substrate.bus_synaptic"]
    assert base.score_0_10 == 10 and base.unusual

    real = wf.node_prediction_error_semantics

    def flipped(node_id):
        vk, pol = real(node_id)
        return (vk, "higher_is_better") if node_id == "node:substrate.bus_synaptic" else (vk, pol)

    monkeypatch.setattr(wf, "node_prediction_error_semantics", flipped)
    monkeypatch.setattr(sr, "node_prediction_error_semantics", flipped)
    inv = _by_id(sr.build_self_readings(_cache(rows), now=NOW))["node:substrate.bus_synaptic"]
    # At the top of its week: great if higher is better, so its bad-direction score is 0.
    assert inv.score_0_10 == 0 and not inv.unusual


def test_caps_on_count_and_chars() -> None:
    readings = sr.build_self_readings(_cache(fx.busy_body_rows()), now=NOW)
    many = [r.model_copy(update={"source_id": f"node:substrate.x{i}", "plain": r.plain})
            for i in range(5) for r in readings if r.unusual][:5]
    block = sr.render_self_readings(many, now=NOW, max_shown=3)
    assert len(block.shown) == 3 and block.lines_dropped == 2
    assert "2 more unusual signals not shown." in block.text
    assert len(block.text) <= 600
    tight = sr.render_self_readings(many, now=NOW, max_chars=250)
    assert len(tight.text) <= 250
    assert all(line.startswith("- ") or "unusual" in line for line in tight.text.splitlines())
    nothing_fits = sr.render_self_readings(many, now=NOW, max_chars=40)
    assert nothing_fits.text == "" and nothing_fits.skip_reason == "render_failed"


def test_event_reading_past_its_orienting_window_is_not_news() -> None:
    later = NOW + timedelta(seconds=400)  # execution's spike is now 430 s old
    readings = _by_id(sr.build_self_readings(_cache(fx.busy_body_rows()), now=later))
    ex = readings["node:substrate.execution"]
    assert ex.state == "live" and not ex.unusual
    assert "orienting window" in ex.reason


def test_score_0_10_rounding() -> None:
    assert sr.score_0_10(None) is None
    assert sr.score_0_10(0.0) == 0
    assert sr.score_0_10(0.94) == 9
    assert sr.score_0_10(0.95) == 10
    assert sr.score_0_10(1.0) == 10


def test_unreadable_glossary_renders_nothing(monkeypatch) -> None:
    import orion.field.channel_glossary as cg

    monkeypatch.setattr(sr, "_NODES_CACHE", None)
    monkeypatch.setattr(cg, "load_glossary", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("x")))
    assert sr.prompt_reaching_nodes() == []
    block = sr.project_self_readings(_cache(fx.busy_body_rows()), now=NOW)
    assert block.text == "" and block.skip_reason == "no_history"


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_latest_value_is_no_reading_never_a_measurement(bad) -> None:
    rows = fx.busy_body_rows() + [("node:substrate.route", NOW - timedelta(seconds=3), bad)]
    block = sr.project_self_readings(_cache(rows), now=NOW)
    route = _by_id(block.readings)["node:substrate.route"]
    assert route.state == "no_reading" and route.value is None and not route.unusual
    assert "inf" not in block.text and "nan" not in block.text
    assert "route (no reading)" in block.text


def test_calm_with_unread_signals_is_not_logged_all_calm() -> None:
    rows = [r for r in fx.calm_rows() if r[0] in ("node:substrate.biometrics",)]
    block = sr.project_self_readings(_cache(rows), now=NOW)
    assert block.text == "" and block.skip_reason == "calm_partial"


def test_trigger_with_too_little_history_is_neither_calm_nor_unreadable() -> None:
    rows = [r for r in fx.busy_body_rows() if r[0] != "node:substrate.cabinet"]
    rows += [("node:substrate.cabinet", NOW - timedelta(minutes=5 * k), 0.0) for k in range(1, 300)]
    block = sr.project_self_readings(_cache(rows), now=NOW)
    cab = _by_id(block.readings)["node:substrate.cabinet"]
    assert cab.state == "insufficient_history" and not cab.unusual
    assert "cabinet" not in block.text


def test_event_signal_with_no_recent_event_is_not_called_unreadable() -> None:
    rows = [r for r in fx.busy_body_rows()
            if not (r[0] == "node:substrate.chat" and r[1] > NOW - timedelta(hours=1))]
    block = sr.project_self_readings(_cache(rows), now=NOW)
    assert _by_id(block.readings)["node:substrate.chat"].state == "stale"
    assert "No recent events: chat." in block.text
    assert "chat (stale)" not in block.text


def test_per_tick_signal_gone_stale_is_unreadable() -> None:
    rows = [r for r in fx.busy_body_rows()
            if not (r[0] == "node:substrate.biometrics" and r[1] > NOW - timedelta(hours=1))]
    block = sr.project_self_readings(_cache(rows), now=NOW)
    assert "Can't read right now: biometrics (stale)." in block.text


# --- what it tends to lead to (spec section E) --------------------------------------------------

from datetime import datetime, timezone  # noqa: E402

from orion.schemas.self_signal_outcome import SelfSignalOutcomeAssociationV1  # noqa: E402

_CLAUSE = ("when this is high, my next hour had a failed turn 3.1x as often at equal activity (31% of 42 high hours vs "
           "10% of 120 normal) [assoc:ssa-0123456789]")


def _assoc(signal: str, clause: str = _CLAUSE) -> SelfSignalOutcomeAssociationV1:
    return SelfSignalOutcomeAssociationV1(
        association_id="ssa-0123456789", computed_at=datetime(2026, 10, 14, 3, tzinfo=timezone.utc),
        signal_id=signal, outcome="turn_failed", n_high=42, events_high=13, n_normal=120, events_normal=12,
        rate_high=13 / 42, rate_normal=0.1, ratio=3.0952, ci_low=1.5, ci_high=6.3, min_hours=24,
        verdict="tends_more", clause=clause,
    )


def test_no_association_is_byte_identical_to_before() -> None:
    cache = _cache(fx.busy_body_rows())
    before = sr.project_self_readings(cache, now=NOW)
    assert sr.project_self_readings(cache, now=NOW, associations=None).text == before.text
    assert sr.project_self_readings(cache, now=NOW, associations={}).text == before.text
    assert before.text and "when this is high" not in before.text


def test_an_unusual_reading_gets_its_cited_clause() -> None:
    exe = "node:substrate.execution"
    readings = sr.build_self_readings(_cache(fx.busy_body_rows()), now=NOW, associations={exe: _assoc(exe)})
    # Two readings shown leave room for the clause under the 600-char cap (three do not: see below).
    block = sr.render_self_readings(readings, now=NOW, max_shown=2)
    line = next(ln for ln in block.text.splitlines() if "task execution" in ln)
    assert line.endswith("[pe:execution@2026-10-14T03:29:30Z]; " + _CLAUSE)
    r = _by_id(block.readings)[exe]
    assert r.tends_to == _CLAUSE and r.outcome_association_id == "ssa-0123456789"


def test_a_clause_never_attaches_to_a_calm_or_unread_signal() -> None:
    block = sr.project_self_readings(
        _cache(fx.calm_rows()), now=NOW,
        associations={n: _assoc(n) for n in sr.prompt_reaching_nodes()},
    )
    assert block.text == ""  # calm body: nothing rendered, clause or not


def test_the_clause_is_dropped_before_a_reading_would_be() -> None:
    nodes = ["node:substrate.execution", "node:substrate.bus_synaptic", "node:substrate.cabinet"]
    readings = sr.build_self_readings(_cache(fx.busy_body_rows()), now=NOW,
                                      associations={n: _assoc(n) for n in nodes})
    plain = sr.render_self_readings(sr.build_self_readings(_cache(fx.busy_body_rows()), now=NOW), now=NOW)
    block = sr.render_self_readings(readings, now=NOW)
    assert len(block.text) <= 600
    assert block.shown == plain.shown  # every reading still shown
    assert block.text == plain.text  # three readings fill the cap: every clause gave way
