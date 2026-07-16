"""Verdict-aware selection: resolved/dismissed loops must not win the rung-3
workspace competition.

Deferred non-goal of PR #1061 (see
docs/notes/2026-07-14-attention-salience-decay-bypass-investigation.md):
excluding `attention_loop_outcome`-verdicted loops from `build_open_loops`
entirely, not just down-weighting them. This suite covers both the pure
scoring-path exclusion (`build_open_loops(verdict_lookup=...)`) and the DB
lookup itself (`orion.substrate.attention.verdicts`), plus fail-open behavior
end-to-end through `build_substrate_attention_frame`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from orion.schemas.attention_frame import AttentionSignalV1
from orion.substrate.attention.common import compact, stable_id
from orion.substrate.attention.scoring import build_open_loops
from orion.substrate.attention import verdicts as verdicts_mod

_NOW = datetime(2026, 7, 16, 12, 0, 0, tzinfo=timezone.utc)


def _signal(text: str, *, salience: float = 0.9) -> AttentionSignalV1:
    return AttentionSignalV1(
        signal_id=f"sig-{text}",
        source="current_turn",
        target_text=text,
        target_type_hint="concept",
        signal_kind="test",
        salience=salience,
        confidence=0.9,
        evidence_refs=[f"ref-{text}"],
    )


def _loop_id(text: str) -> str:
    return stable_id("open-loop", compact(text, 120).lower())


def _build(signals, verdict_lookup=None):
    return build_open_loops(
        signals=signals,
        ctx={},
        inputs={},
        belief_lineage=[],
        direct_turn=False,
        generic_reversal=False,
        stale_thread_active=False,
        max_open=5,
        now=_NOW,
        verdict_lookup=verdict_lookup,
    )


# ---------------------------------------------------------------------------
# build_open_loops(verdict_lookup=...) -- pure, no DB
# ---------------------------------------------------------------------------


def test_loop_with_no_verdict_competes_normally():
    signals = [_signal("substrate:node:substrate.transport")]

    loops = _build(signals, verdict_lookup=lambda ids: set())

    assert len(loops) == 1
    assert loops[0].id == _loop_id("substrate:node:substrate.transport")


def test_loop_with_resolved_verdict_excluded_even_at_high_salience():
    high_salience_text = "substrate:node:substrate.transport"
    other_text = "a fresh unrelated concern"
    signals = [
        _signal(high_salience_text, salience=1.0),
        _signal(other_text, salience=0.21),
    ]
    excluded_id = _loop_id(high_salience_text)

    loops = _build(signals, verdict_lookup=lambda ids: {excluded_id})

    ids = {loop.id for loop in loops}
    assert excluded_id not in ids
    assert _loop_id(other_text) in ids
    assert len(loops) == 1


def test_loop_with_dismissed_verdict_excluded():
    text = "substrate:node:substrate.transport"
    excluded_id = _loop_id(text)

    loops = _build([_signal(text)], verdict_lookup=lambda ids: {excluded_id})

    assert loops == []


def test_excluded_loop_frees_its_slot_for_the_next_best_candidate():
    """Exclusion must be applied before the max_open cap, not after.

    With max_open=2 and 3 competing signals, excluding the top-ranked one
    must promote the 3rd-ranked signal into the field instead of just
    shrinking it to 1 -- attention_broadcast.py deliberately passes a wider
    signal pool than max_open specifically so this backfill can happen."""
    first, second, third = "first loop", "second loop", "third loop"
    signals = [_signal(first), _signal(second), _signal(third)]
    excluded_id = _loop_id(first)

    loops = build_open_loops(
        signals=signals,
        ctx={},
        inputs={},
        belief_lineage=[],
        direct_turn=False,
        generic_reversal=False,
        stale_thread_active=False,
        max_open=2,
        now=_NOW,
        verdict_lookup=lambda ids: {excluded_id},
    )

    ids = {loop.id for loop in loops}
    assert len(loops) == 2
    assert excluded_id not in ids
    assert ids == {_loop_id(second), _loop_id(third)}


def test_verdict_lookup_receives_only_candidate_loop_ids():
    """The lookup must be bounded to this tick's candidates -- never a
    whole-table scan -- and called at most once per build_open_loops call."""
    text = "substrate:node:substrate.transport"
    seen: list[list[str]] = []

    def _lookup(ids):
        seen.append(list(ids))
        return set()

    _build([_signal(text)], verdict_lookup=_lookup)

    assert len(seen) == 1
    assert seen[0] == [_loop_id(text)]


def test_no_verdict_lookup_preserves_prior_behavior():
    """verdict_lookup=None (the default, used by chat-scoped callers) must
    behave byte-identically to before this patch -- no exclusion at all."""
    text = "substrate:node:substrate.transport"

    loops = _build([_signal(text)])  # verdict_lookup defaults to None

    assert len(loops) == 1
    assert loops[0].id == _loop_id(text)


def test_verdict_lookup_exception_does_not_crash_build_open_loops():
    """A misbehaving verdict_lookup callable (raises instead of returning an
    empty set) must not propagate -- build_open_loops treats it as 'no
    verdicts known' and proceeds normally."""
    text = "substrate:node:substrate.transport"

    def _boom(ids):
        raise RuntimeError("simulated DB failure")

    loops = _build([_signal(text)], verdict_lookup=_boom)

    assert len(loops) == 1
    assert loops[0].id == _loop_id(text)


# ---------------------------------------------------------------------------
# orion.substrate.attention.verdicts -- the real DB lookup, mocked engine
# ---------------------------------------------------------------------------


def test_load_terminal_verdict_loop_ids_empty_input_short_circuits():
    assert verdicts_mod.load_terminal_verdict_loop_ids([]) == set()
    assert verdicts_mod.load_terminal_verdict_loop_ids(None) == set()  # type: ignore[arg-type]


def test_load_terminal_verdict_loop_ids_filters_to_terminal_verdicts(monkeypatch):
    rows = [
        {"loop_id": "open-loop-a", "verdict": "resolved"},
        {"loop_id": "open-loop-b", "verdict": "dismissed"},
        {"loop_id": "open-loop-c", "verdict": "decayed_unattended"},
    ]

    class _FakeResult:
        def mappings(self):
            return self

        def all(self):
            return rows

    class _FakeConn:
        def execute(self, stmt, params):
            return _FakeResult()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class _FakeEngine:
        def connect(self):
            return _FakeConn()

    monkeypatch.setattr(verdicts_mod, "_engine", lambda: _FakeEngine())

    result = verdicts_mod.load_terminal_verdict_loop_ids(
        ["open-loop-a", "open-loop-b", "open-loop-c"]
    )

    assert result == {"open-loop-a", "open-loop-b"}


def test_load_terminal_verdict_loop_ids_fails_open_on_db_error(monkeypatch):
    def _boom():
        raise RuntimeError("simulated connection failure")

    monkeypatch.setattr(verdicts_mod, "_engine", _boom)

    result = verdicts_mod.load_terminal_verdict_loop_ids(["open-loop-9d84d08cddf5"])

    assert result == set()


# ---------------------------------------------------------------------------
# End-to-end through build_substrate_attention_frame
# ---------------------------------------------------------------------------


def _node(node_id: str, label: str, pressure: float = 0.9) -> SimpleNamespace:
    return SimpleNamespace(
        node_id=node_id,
        label=label,
        metadata={"dynamic_pressure": pressure},
        signals=SimpleNamespace(confidence=0.8),
    )


def test_build_substrate_attention_frame_excludes_verdicted_loop(monkeypatch):
    import orion.substrate.attention_broadcast as attention_broadcast

    node = _node("node:transport", "substrate:node:substrate.transport", pressure=1.0)
    excluded_id = _loop_id("substrate:node:substrate.transport")

    monkeypatch.setattr(
        attention_broadcast, "load_terminal_verdict_loop_ids", lambda ids: {excluded_id}
    )

    frame = attention_broadcast.build_substrate_attention_frame(nodes=[node], now=_NOW)

    assert frame.open_loops == []
    # select_actions() always returns a placeholder action rather than None;
    # what matters is it has no open_loop_id to win with -- the excluded
    # loop cannot become `selected_action`.
    assert frame.selected_action is not None
    assert frame.selected_action.open_loop_id is None


def test_build_substrate_attention_frame_survives_verdict_lookup_db_failure(monkeypatch):
    """DB failure during the verdict lookup must not crash frame-building --
    the tick proceeds as if no verdicts exist (the loop stays eligible)."""
    import orion.substrate.attention_broadcast as attention_broadcast

    node = _node("node:transport", "substrate:node:substrate.transport", pressure=1.0)

    def _boom(ids):
        raise RuntimeError("simulated DB failure")

    monkeypatch.setattr(attention_broadcast, "load_terminal_verdict_loop_ids", _boom)

    frame = attention_broadcast.build_substrate_attention_frame(nodes=[node], now=_NOW)

    assert len(frame.open_loops) == 1
    assert frame.open_loops[0].id == _loop_id("substrate:node:substrate.transport")
