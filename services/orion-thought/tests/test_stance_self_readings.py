"""Self-readings reach stance_react and reverie (self-calibration spec section C, patch 4)."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest
from jinja2 import Environment

import app.self_readings_cue as cue_mod
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
    """The template as if this patch's blocks had never been added (no git needed: CI is shallow)."""
    import re

    return re.sub(r"\{% if self_readings(?:_label_estimates)? %\}.*?\{% endif %\}", "", template,
                  flags=re.DOTALL)


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

def _fresh_cache(monkeypatch):
    monkeypatch.setattr(cue_mod, "_cache", None)


@pytest.mark.asyncio
async def test_disabled_never_reads(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(cue_mod, "fetch_since", lambda since: calls.append(since) or [])
    block = await cue_mod.build_self_readings_block("c-1", purpose="stance", enabled=False)
    assert block.skip_reason == "disabled" and block.text == "" and calls == []


@pytest.mark.asyncio
async def test_read_error_renders_nothing(monkeypatch) -> None:
    _fresh_cache(monkeypatch)

    def boom(since):
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
