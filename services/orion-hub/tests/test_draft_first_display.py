"""Draft-first display (spec L8, 2026-10-06): Hub side.

Covers the relay (governor draft preview -> per-turn queue), run_unified_turn
(draft frame forwarded before the final, final annotated with revised/reason,
flag off restores judge-first), and the browser seam (template loads
draft-revision.js, app.js routes the frames to it, the node suite passes).
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

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

import orion.hub.turn_orchestrator as orch  # noqa: E402
from orion.schemas.harness_finalize import HarnessRunDraftPreviewV1  # noqa: E402
from scripts.harness_step_relay import DRAFT_PREVIEW_ITEM_KIND, HarnessStepRelay  # noqa: E402

CORR = "00000000-0000-4000-8000-0000000008a8"


# ── relay ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_relay_queues_draft_preview_for_its_turn_only() -> None:
    relay = HarnessStepRelay(channel="orion:harness:run:step", draft_preview_channel="orion:harness:run:draft_preview")
    queue: asyncio.Queue = asyncio.Queue()
    relay.register_queue(CORR, queue)
    relay._dispatch_draft_preview(HarnessRunDraftPreviewV1(correlation_id=CORR, text="draft words"))
    relay._dispatch_draft_preview(HarnessRunDraftPreviewV1(correlation_id="other-corr", text="not ours"))
    item = queue.get_nowait()
    assert item == {"kind": DRAFT_PREVIEW_ITEM_KIND, "correlation_id": CORR, "text": "draft words"}
    assert queue.empty()


def test_relay_kind_constant_matches_orchestrator() -> None:
    assert DRAFT_PREVIEW_ITEM_KIND == orch.DRAFT_PREVIEW_ITEM_KIND


# ── run_unified_turn ─────────────────────────────────────────────────────


class _FakeRelay:
    def __init__(self) -> None:
        self.queues: dict[str, asyncio.Queue] = {}

    def register_queue(self, cid: str, q: asyncio.Queue) -> None:
        self.queues[cid] = q

    def unregister_queue(self, cid: str, q: asyncio.Queue) -> None:
        self.queues.pop(cid, None)

    def forget(self, cid: str) -> None:
        return None


class _FakeWs:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send_json(self, data: dict[str, Any]) -> None:
        self.sent.append(data)


def _final_frame(text: str, *, repair_reason: str | None = None) -> dict[str, Any]:
    return {
        "type": "final",
        "correlation_id": CORR,
        "mode": "orion",
        "llm_response": text,
        "finalize_ran": True,
        "finalize_changed": repair_reason is not None,
        "response_repair_ran": repair_reason is not None,
        "response_repair_reason": repair_reason,
    }


def _fake_execute(*, draft: str | None, frames: list[dict[str, Any]], seen: dict[str, Any]):
    async def _run(**kwargs: Any) -> list[dict[str, Any]]:
        seen.update(kwargs)
        queue = kwargs.get("harness_step_queue")
        if draft is not None and queue is not None:
            await queue.put({"kind": DRAFT_PREVIEW_ITEM_KIND, "correlation_id": CORR, "text": draft})
            await asyncio.sleep(0.15)  # let the drain task forward it, as during a real judge wait
        return [dict(f) for f in frames]

    return _run


async def _run_turn(*, flag: bool, draft: str | None, frames: list[dict[str, Any]]):
    from scripts.settings import settings as hub_settings

    ws = _FakeWs()
    seen: dict[str, Any] = {}
    with patch.object(hub_settings, "HUB_UNIFIED_DRAFT_FIRST_ENABLED", flag), patch.object(
        orch, "execute_unified_turn", _fake_execute(draft=draft, frames=frames, seen=seen)
    ), patch.object(orch, "publish_cockpit_frames", AsyncMock()):
        returned = await orch.run_unified_turn(
            ws,
            bus=object(),
            correlation_id=CORR,
            session_id="sess-1",
            user_message="hi",
            harness_step_relay=_FakeRelay(),
        )
    turn_frames = [f for f in ws.sent if f.get("type") in ("draft_preview", "final", "turn_error")]
    return ws, turn_frames, returned, seen


@pytest.mark.asyncio
async def test_draft_shown_first_then_revision_replaces_it_with_reason(caplog) -> None:
    caplog.set_level(logging.INFO, logger=orch.logger.name)
    _, turn_frames, returned, seen = await _run_turn(
        flag=True,
        draft="You -- all containers are up.",
        frames=[_final_frame("Juniper -- I checked two containers.", repair_reason="strain_unresolved")],
    )
    assert seen["draft_preview"] is True
    assert [f["type"] for f in turn_frames] == ["draft_preview", "final"]
    draft_frame, final = turn_frames
    assert draft_frame["draft_text"] == "You -- all containers are up."
    # Old browser JS renders llm_response/text; the draft must use neither.
    assert "llm_response" not in draft_frame and "text" not in draft_frame
    assert final["replaces_draft"] is True
    assert final["revised"] is True
    assert final["revised_reason"] == "strain_unresolved"
    # Persisted/returned frames still carry the final text only.
    assert returned[-1]["llm_response"] == "Juniper -- I checked two containers."
    logs = caplog.text
    assert f"unified_turn_first_visible corr={CORR} kind=draft_preview" in logs
    assert f"unified_turn_revision corr={CORR} reason=strain_unresolved" in logs
    assert f"unified_turn_final_visible corr={CORR}" in logs


@pytest.mark.asyncio
async def test_unrevised_draft_is_replaced_without_a_revised_mark() -> None:
    _, turn_frames, _, _ = await _run_turn(
        flag=True, draft="Same answer.", frames=[_final_frame("Same answer.")]
    )
    assert [f["type"] for f in turn_frames] == ["draft_preview", "final"]
    final = turn_frames[1]
    assert final["replaces_draft"] is True
    assert final["revised"] is False
    assert final["revised_reason"] is None


@pytest.mark.asyncio
async def test_text_change_without_repair_reason_is_still_named() -> None:
    _, turn_frames, _, _ = await _run_turn(
        flag=True, draft="Draft.", frames=[_final_frame("Different final.")]
    )
    assert turn_frames[1]["revised"] is True
    assert turn_frames[1]["revised_reason"] == "finalize_changed"


@pytest.mark.asyncio
async def test_flag_off_restores_judge_first(caplog) -> None:
    caplog.set_level(logging.INFO, logger=orch.logger.name)
    _, turn_frames, _, seen = await _run_turn(
        flag=False, draft="should never show", frames=[_final_frame("final only")]
    )
    assert seen["draft_preview"] is False
    assert [f["type"] for f in turn_frames] == ["final"]
    assert "replaces_draft" not in turn_frames[0]
    assert f"unified_turn_first_visible corr={CORR} kind=final" in caplog.text


@pytest.mark.asyncio
async def test_held_draft_turn_looks_exactly_like_today() -> None:
    # Sensitive turn: governor publishes nothing, so no draft frame arrives.
    _, turn_frames, _, seen = await _run_turn(flag=True, draft=None, frames=[_final_frame("judged")])
    assert seen["draft_preview"] is True
    assert [f["type"] for f in turn_frames] == ["final"]
    assert "replaces_draft" not in turn_frames[0]


@pytest.mark.asyncio
async def test_error_after_draft_tells_the_browser_a_draft_is_on_screen() -> None:
    err = {"type": "turn_error", "correlation_id": CORR, "phase": "finalize", "error": "x", "finalize_ran": False}
    _, turn_frames, _, _ = await _run_turn(flag=True, draft="draft", frames=[err])
    assert [f["type"] for f in turn_frames] == ["draft_preview", "turn_error"]
    assert turn_frames[1]["draft_shown"] is True


@pytest.mark.asyncio
async def test_hub_exception_after_draft_still_settles_the_bubble() -> None:
    """Review finding: an exception after the draft was shown left no frame
    naming the turn, so the browser bubble said "still being checked" forever."""
    from scripts.settings import settings as hub_settings

    async def _boom(**kwargs: Any) -> list[dict[str, Any]]:
        await kwargs["harness_step_queue"].put({"kind": DRAFT_PREVIEW_ITEM_KIND, "correlation_id": CORR, "text": "d"})
        await asyncio.sleep(0.15)
        raise RuntimeError("history publish failed")

    ws = _FakeWs()
    with patch.object(hub_settings, "HUB_UNIFIED_DRAFT_FIRST_ENABLED", True), patch.object(
        orch, "execute_unified_turn", _boom
    ), patch.object(orch, "publish_cockpit_frames", AsyncMock()):
        with pytest.raises(RuntimeError, match="history publish failed"):
            await orch.run_unified_turn(
                ws, bus=object(), correlation_id=CORR, session_id="s", user_message="hi",
                harness_step_relay=_FakeRelay(),
            )
    types = [f.get("type") for f in ws.sent if f.get("type")]
    assert types == ["draft_preview", "turn_error"]
    assert ws.sent[-1]["draft_shown"] is True
    assert ws.sent[-1]["correlation_id"] == CORR


@pytest.mark.asyncio
async def test_hub_exception_without_draft_sends_nothing_extra() -> None:
    from scripts.settings import settings as hub_settings

    async def _boom(**kwargs: Any) -> list[dict[str, Any]]:
        raise RuntimeError("boom")

    ws = _FakeWs()
    with patch.object(hub_settings, "HUB_UNIFIED_DRAFT_FIRST_ENABLED", True), patch.object(
        orch, "execute_unified_turn", _boom
    ), patch.object(orch, "publish_cockpit_frames", AsyncMock()):
        with pytest.raises(RuntimeError):
            await orch.run_unified_turn(
                ws, bus=object(), correlation_id=CORR, session_id="s", user_message="hi",
                harness_step_relay=_FakeRelay(),
            )
    assert ws.sent == []


def test_success_frames_carry_repair_reason() -> None:
    from orion.schemas.harness_finalize import HarnessRunV1

    run = HarnessRunV1(
        correlation_id=CORR,
        final_text="f",
        finalize_ran=True,
        response_repair_ran=True,
        response_repair_reason="misaligned",
        step_count=1,
        compliance_verdict="completed",
        grounding_status="grounded",
    )
    final = orch._success_frames(run, correlation_id=CORR)[-1]
    assert final["response_repair_reason"] == "misaligned"
    assert final["response_repair_ran"] is True


# ── browser seam ─────────────────────────────────────────────────────────


def test_template_loads_draft_revision_before_app_js() -> None:
    html = (HUB_ROOT / "templates" / "index.html").read_text()
    tag = '<script src="/static/js/draft-revision.js?v={{HUB_UI_ASSET_VERSION}}" defer></script>'
    assert tag in html
    assert html.index(tag) < html.index('/static/js/app.js?v=')
    assert (HUB_ROOT / "static" / "js" / "draft-revision.js").is_file()


def test_app_js_routes_draft_frames_to_the_module() -> None:
    js = (HUB_ROOT / "static" / "js" / "app.js").read_text()
    assert "d.type === 'draft_preview'" in js
    assert "OrionDraftRevision.showDraft(" in js or "draftApi.showDraft(" in js
    assert "window.OrionDraftRevision.settleWithFinal(conversationDiv, pendingDraftNode, finalNode, d)" in js
    assert "window.OrionDraftRevision.settleWithoutFinal(pendingDraftNode)" in js
    # appendMessage must hand back its node so the final can take the draft's slot.
    body = js.split("function appendMessage(sender, text, colorClass = 'text-white') {", 1)[1]
    body = body.split("\n  function collectConversationTurnsUpTo", 1)[0]
    assert "return div;" in body


def test_draft_revision_dom_cases_pass_under_node() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not on PATH -- draft-revision.test.js was NOT executed")
    result = subprocess.run(
        [node, "--test", str(HUB_ROOT / "static" / "js" / "draft-revision.test.js")],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "# fail 0" in result.stdout, result.stdout
    assert "# pass 7" in result.stdout, result.stdout
