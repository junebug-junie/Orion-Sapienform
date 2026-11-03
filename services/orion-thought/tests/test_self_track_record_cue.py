"""The self-monitor track record reaches stance_react and reverie (spec step 3), fail-open."""
from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from jinja2 import Environment

import app.self_track_record_cue as cue_mod
from app.bus_listener import build_stance_react_context
from app.reverie import build_reverie_context
from orion.schemas.attention_frame import AttentionBroadcastProjectionV1, AttentionFrameV1, OpenLoopV1
from orion.schemas.self_monitor import FacultyRecordV1, SelfMonitorStateV1
from orion.schemas.thought import HubAssociationBundleV1, StanceReactRequestV1

_REPO = Path(__file__).resolve().parents[3]
_STANCE = _REPO / "orion" / "cognition" / "prompts" / "stance_react.j2"
_REVERIE = _REPO / "orion" / "cognition" / "prompts" / "reverie_narrate.j2"
NOW = datetime(2026, 11, 3, 4, 0, tzinfo=timezone.utc)
_LINE = "My walkway-grade journal runs failed to start 22 of 28 times this week."


def _state(computed_at: datetime = NOW) -> str:
    rec = FacultyRecordV1(
        faculty="act.actions.journal.make.walkway_grade", sources=["grammar_acts"], claim_kind="success",
        gradeability="ungradeable:too_few", window_start=NOW - timedelta(days=7), window_end=NOW, n_bets=28,
        n_graded=6, n_failed_to_execute=22, n_expired=0, weight_graded=4.0, reliability=1.0,
        execution_failure=0.74, execution_failure_lo=0.55, candidate_status="broken", status="broken",
        status_since=NOW, candidate_held=3,
    )
    return SelfMonitorStateV1(computed_at=computed_at, window_days=7.0, faculties=[rec]).model_dump_json()


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(cue_mod, "_cache", None)
    monkeypatch.setattr(cue_mod, "_refreshing", False)
    monkeypatch.setattr(cue_mod.settings, "self_track_record_cue_enabled", True)
    yield


def _load(monkeypatch, value=None, exc: Exception | None = None) -> None:
    def fetch():
        if exc is not None:
            raise exc
        return value

    monkeypatch.setattr(cue_mod, "fetch_raw", fetch)
    cue_mod.refresh()


@pytest.mark.asyncio
async def test_renders_from_fresh_state(monkeypatch) -> None:
    _load(monkeypatch, _state().encode())
    cue = await cue_mod.build_self_track_record_cue("c-1", purpose="stance", now=NOW)
    assert cue.text == _LINE and cue.skip_reason is None


@pytest.mark.asyncio
@pytest.mark.parametrize("value, reason", [(None, "absent"), (b"garbage", "unparseable")])
async def test_absent_or_garbage_renders_nothing(monkeypatch, value, reason) -> None:
    _load(monkeypatch, value)
    cue = await cue_mod.build_self_track_record_cue("c-1", purpose="stance", now=NOW)
    assert cue.text == "" and cue.skip_reason == reason


@pytest.mark.asyncio
async def test_stale_cached_value_blanks_at_render_time(monkeypatch) -> None:
    _load(monkeypatch, _state())
    later = NOW + timedelta(hours=3, seconds=1)
    cue = await cue_mod.build_self_track_record_cue("c-1", purpose="stance", now=later)
    assert cue.text == "" and cue.skip_reason == "stale"


@pytest.mark.asyncio
async def test_read_error_renders_nothing_then_keeps_last_good_value(monkeypatch) -> None:
    _load(monkeypatch, exc=ConnectionError("redis down"))
    cue = await cue_mod.build_self_track_record_cue("c-1", purpose="stance", now=NOW)
    assert cue.text == "" and cue.skip_reason == "fetch_failed"
    _load(monkeypatch, _state())
    _load(monkeypatch, exc=ConnectionError("redis down again"))
    cue = await cue_mod.build_self_track_record_cue("c-1", purpose="stance", now=NOW)
    assert cue.text == _LINE  # the age gate, not the read, decides when it goes away


@pytest.mark.asyncio
async def test_not_loaded_never_reads_on_caller_thread(monkeypatch) -> None:
    caller = threading.get_ident()
    seen: list[int] = []
    done = threading.Event()

    def fetch():
        seen.append(threading.get_ident())
        done.set()
        return _state()

    monkeypatch.setattr(cue_mod, "fetch_raw", fetch)
    cue = await cue_mod.build_self_track_record_cue("c-1", purpose="stance", now=NOW)
    assert cue.text == "" and cue.skip_reason == "not_loaded"
    assert done.wait(2.0) and seen and caller not in seen
    import time

    deadline = time.monotonic() + 2.0
    while cue_mod._refreshing and time.monotonic() < deadline:  # the thread writes the cache after fetch
        time.sleep(0.01)
    cue = await cue_mod.build_self_track_record_cue("c-1", purpose="stance", now=NOW)
    assert cue.text == _LINE


@pytest.mark.asyncio
async def test_disabled_never_reads(monkeypatch) -> None:
    monkeypatch.setattr(cue_mod.settings, "self_track_record_cue_enabled", False)
    monkeypatch.setattr(cue_mod, "fetch_raw", lambda: pytest.fail("read while disabled"))
    cue = await cue_mod.build_self_track_record_cue("c-1", purpose="stance", now=NOW)
    assert cue.text == "" and cue.skip_reason == "disabled"


@pytest.mark.asyncio
async def test_render_bug_is_named_not_raised(monkeypatch) -> None:
    _load(monkeypatch, _state())
    import orion.self_monitor.cue as pure

    monkeypatch.setattr(pure, "project_track_record_cue", lambda *a, **k: 1 / 0)
    cue = await cue_mod.build_self_track_record_cue("c-1", purpose="stance", now=NOW)
    assert cue.text == "" and cue.skip_reason == "render_failed"


# --- prompt wiring ------------------------------------------------------------------------------

def _request() -> StanceReactRequestV1:
    return StanceReactRequestV1(
        correlation_id="c-1", session_id="s-1", user_message="how are you?",
        association=HubAssociationBundleV1(correlation_id="c-1", broadcast=None, broadcast_stale=True,
                                           read_source="hub_sql_fallback"),
        repair_bundle=None, stance_inputs={"user_message": "how are you?"},
    )


def _broadcast() -> AttentionBroadcastProjectionV1:
    loop = OpenLoopV1(id="ol-1", description="execution spiked", salience=0.7)
    return AttentionBroadcastProjectionV1(frame=AttentionFrameV1(open_loops=[loop]),
                                          selected_open_loop_id="ol-1", attended_node_ids=["node:substrate.execution"])


def _render(path: Path, **ctx) -> str:
    return Environment().from_string(path.read_text()).render(**ctx)


_STANCE_BASE = {"user_message": "hi", "stance_inputs": {"user_message": "hi"}, "association": {},
                "repair_bundle": None, "coalition_projection": None}


def test_stance_context_and_prompt_carry_the_cue() -> None:
    base = build_stance_react_context(_request())
    assert "self_track_record" not in base
    assert build_stance_react_context(_request(), self_track_record="") == base
    ctx = build_stance_react_context(_request(), self_track_record=_LINE)
    assert ctx["self_track_record"] == _LINE
    out = _render(_STANCE, **_STANCE_BASE, self_track_record=_LINE)
    assert "MY TRACK RECORD" in out and _LINE in out and "the task wins" in out
    assert "MY TRACK RECORD" not in _render(_STANCE, **_STANCE_BASE)


def test_reverie_coalition_branch_only() -> None:
    ctx = build_reverie_context(_broadcast(), self_track_record=_LINE)
    out = _render(_REVERIE, **ctx)
    assert "MY TRACK RECORD" in out and _LINE in out
    assert "MY TRACK RECORD" not in _render(_REVERIE, **build_reverie_context(_broadcast()))
    from orion.schemas.reverie import ConcernCardV1

    card = ConcernCardV1.from_harness_turn(
        coalition_ref="harness_closure:c", user_message_excerpt="Will the deploy slip?",
        stance_imperative="Name the tradeoff.", created_at=NOW)
    assert "self_track_record" not in build_reverie_context(_broadcast(), concern_cards=[card],
                                                            self_track_record=_LINE)


def test_context_key_is_classified() -> None:
    from orion.schemas.context_provenance import classify

    assert classify("self_track_record") == "derived_summary"
