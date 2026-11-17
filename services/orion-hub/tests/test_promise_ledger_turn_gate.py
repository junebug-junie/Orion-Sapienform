"""Intention ledger, Orion's round-3 review (PR #3561/#3598 threads): which turns get the promise view.

Only a turn that gets the pre-turn view can open ``hub.promise`` acts, so the view's gate IS the
act-opening gate. It must match the rule the report's guard counts by: Juniper's own turn, sent from a
Hub browser surface (``surface_context`` on the payload). Live 2026-11-15: two coding-session notes were
posted to ``/api/chat`` into Juniper's session and persisted as ``hub_orion`` rows with no surface; after
the finder went live such a note would have opened acts graded against Juniper.

And every turn that read the view logs one ``promise_ledger`` line, including a harness timeout, a
harness error and an interrupted turn (before this, those ended silent after ``promise_view``).
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
HUB_ROOT = Path(__file__).resolve().parents[1]
for key in list(sys.modules):
    if key == "scripts" or key.startswith("scripts.") or key == "app" or key.startswith("app."):
        del sys.modules[key]
for candidate in (REPO_ROOT, HUB_ROOT):
    try:
        sys.path.remove(str(candidate))
    except ValueError:
        pass
for candidate in (REPO_ROOT, HUB_ROOT):
    sys.path.insert(0, str(candidate))

os.environ.setdefault("CHANNEL_VOICE_TRANSCRIPT", "orion:voice:transcript")
os.environ.setdefault("CHANNEL_VOICE_LLM", "orion:voice:llm")
os.environ.setdefault("CHANNEL_VOICE_TTS", "orion:voice:tts")
os.environ.setdefault("CHANNEL_COLLAPSE_INTAKE", "orion:collapse:intake")
os.environ.setdefault("CHANNEL_COLLAPSE_TRIAGE", "orion:collapse:triage")

from orion.grammar.promise_view import PromiseView  # noqa: E402
from orion.hub.turn_orchestrator import execute_unified_turn  # noqa: E402
from orion.regulation.juniper_turns import is_browser_surface_turn  # noqa: E402
from orion.schemas.harness_finalize import HarnessRunV1  # noqa: E402
from orion.schemas.thought import HubAssociationBundleV1, StanceHarnessSliceV1, ThoughtEventV1  # noqa: E402

CORR = "00000000-0000-4000-8000-000000000361"
BROWSER = {"surface_context": {"input_modality": "typed"}}


def _thought() -> ThoughtEventV1:
    return ThoughtEventV1(
        event_id="t-promise-gate", correlation_id=CORR, session_id="sess-1", created_at=datetime.now(timezone.utc),
        imperative="Answer directly.", tone="neutral", strain_refs=["n-1"], evidence_refs=["n-1"],
        disposition="proceed", disposition_reasons=[],
        stance_harness_slice=StanceHarnessSliceV1(task_mode="direct_response", conversation_frame="mixed",
                                                  answer_strategy="direct"))


def _run(**kw) -> HarnessRunV1:
    base = dict(correlation_id=CORR, final_text="hello", finalize_ran=True, step_count=1,
                compliance_verdict="completed", grounding_status="grounded")
    base.update(kw)
    return HarnessRunV1(**base)


async def _turn(*, payload, origin, harness_run, view=None):
    import scripts.harness_governor_client as harness_governor_client
    import scripts.thought_client as thought_client
    from orion.hub import promise_ledger

    view_read = AsyncMock(return_value=view if view is not None else PromiseView())
    run_mock = harness_run if isinstance(harness_run, AsyncMock) else AsyncMock(return_value=harness_run)
    with patch("orion.hub.turn_orchestrator.build_hub_association_bundle",
               return_value=HubAssociationBundleV1(correlation_id=CORR, broadcast=None, broadcast_stale=True,
                                                   read_source="felt_state_reader")), \
            patch.object(thought_client.ThoughtClient, "react",
                         AsyncMock(return_value=thought_client.ThoughtReactResult(thought=_thought()))), \
            patch.object(harness_governor_client.HarnessGovernorClient, "run", run_mock), \
            patch.object(promise_ledger, "promise_view_for_turn", view_read):
        await execute_unified_turn(bus=MagicMock(), correlation_id=CORR, session_id="sess-1", user_message="hi",
                                   payload=dict(payload), emit_observation_fn=lambda **_k: None,
                                   utterance_origin=origin)
    return view_read


@pytest.fixture(autouse=True)
def _flags_on(monkeypatch):
    from scripts.settings import settings

    monkeypatch.setattr(settings, "HUB_ACT_LEDGER_ENABLED", True, raising=False)
    monkeypatch.setattr(settings, "HUB_PROMISE_LEDGER_ENABLED", True, raising=False)


def test_browser_surface_rule_matches_the_ingress_predicate():
    assert is_browser_surface_turn(BROWSER)
    assert not is_browser_surface_turn({})
    assert not is_browser_surface_turn(None)
    assert not is_browser_surface_turn({"surface_context": {}})        # the ingress hop drops an empty one too
    assert not is_browser_surface_turn({"surface_context": "typed"})


@pytest.mark.asyncio
@pytest.mark.parametrize("payload,origin,reads", [
    (BROWSER, "juniper", True),            # Juniper typing or speaking in the browser
    ({}, "juniper", False),                # /api/chat: a coding session's note, the memory-scoreboard eval
    ({"no_write": True}, "juniper", False),
    (BROWSER, "orion", False),             # Orion's own turns: round-ups, intention.carry, curiosity
    ({}, "orion", False),
])
async def test_only_a_juniper_browser_turn_reads_the_view(payload, origin, reads):
    view_read = await _turn(payload=payload, origin=origin, harness_run=_run())
    assert view_read.await_count == (1 if reads else 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("harness_run,reason", [
    (None, "harness_timeout"),
    (_run(final_text="", finalize_ran=False), "harness_error"),
])
async def test_a_viewed_turn_that_ends_before_the_ledger_logs_one_skip_line(caplog, harness_run, reason):
    with caplog.at_level(logging.INFO):
        await _turn(payload=BROWSER, origin="juniper", harness_run=harness_run)
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("promise_ledger corr=")]
    assert lines == [f"promise_ledger corr={CORR} outcome=skipped reason={reason}"]


@pytest.mark.asyncio
async def test_an_interrupted_viewed_turn_logs_one_skip_line_and_still_raises(caplog):
    with caplog.at_level(logging.INFO), pytest.raises(RuntimeError, match="governor gone"):
        await _turn(payload=BROWSER, origin="juniper", harness_run=AsyncMock(side_effect=RuntimeError("governor gone")))
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("promise_ledger corr=")]
    assert lines == [f"promise_ledger corr={CORR} outcome=skipped reason=interrupted"]


@pytest.mark.asyncio
async def test_a_turn_without_a_view_logs_no_ledger_line(caplog):
    with caplog.at_level(logging.INFO):
        await _turn(payload={}, origin="orion", harness_run=None)
    assert not [r for r in caplog.records if r.getMessage().startswith("promise_ledger corr=")]
