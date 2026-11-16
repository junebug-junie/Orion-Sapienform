"""Reverie reads Orion's now and records what a published thought cited (integration patch 3, #3425).

One read per reverie chain (later steps reuse it), no ``since``; a failed read is a reverie without the section;
``experience:`` ids are never grounding (moved out of evidence_refs into cites); only shown ids are cited; only a
published thought's cites are posted, with citing_ref ``reverie_thought:<id>``."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlparse

import pytest
from jinja2 import Environment

from orion.schemas.attention_frame import AttentionBroadcastProjectionV1, AttentionFrameV1, OpenLoopV1
from orion.schemas.context_provenance import classify
from orion.schemas.reverie import ConcernCardV1

_REPO = Path(__file__).resolve().parents[3]
_PROMPT = _REPO / "orion" / "cognition" / "prompts" / "reverie_narrate.j2"
GROUNDED_TEXT = ("I keep returning to the unresolved thread ol-1 — the mail truck on the walkway "
                 "is the same one that stopped the deploy plan from feeling finished.")
NOW_BODY = {
    "shown_id": "shown-r1", "reader_organ": "reverie", "since": "2026-11-16T00:00:00Z",
    "as_of": "2026-11-16T01:00:00Z",
    "items": [
        {"event_id": "street_event:w1", "organ": "camera", "source_kind": "street_event",
         "label": "street event: a mail truck stopped", "occurred_at": "2026-11-16T00:40:00Z", "salience": 0.9},
        {"event_id": "reading_turn:act_1", "organ": "reading", "source_kind": "reading_turn",
         "label": "reading turn: stage1_completed: https://arxiv.org/abs/1", "occurred_at": "2026-11-16T00:10:00Z",
         "salience": None},
    ],
}


def _broadcast():
    return AttentionBroadcastProjectionV1(
        frame=AttentionFrameV1(open_loops=[OpenLoopV1(id="ol-1", description="d")]),
        attended_node_ids=["n-1"], selected_open_loop_id="ol-1")


@pytest.fixture()
def now_seam(monkeypatch):
    """The now on, its HTTP faked; returns the recorded GET urls and POST bodies."""
    import app.temporal_self_cue as cue
    from app.settings import settings

    calls = {"get": [], "post": [], "page": NOW_BODY}

    def get(url, timeout):
        calls["get"].append(url)
        page = calls["page"]
        if isinstance(page, BaseException):
            raise page
        return page

    def post(url, body, timeout):
        calls["post"].append((url, body))
        return {"shown_id": body["shown_id"], "citing_organ": "reverie", "uses": []}

    monkeypatch.setattr(settings, "reverie_now_enabled", True)
    monkeypatch.setattr(settings, "reverie_perception_enabled", False)
    monkeypatch.setattr(settings, "reverie_recent_arcs_enabled", False)
    monkeypatch.setattr(cue, "_now_http", lambda: {"get": get, "post": post})
    monkeypatch.setattr(cue, "_chain_now", None)
    return calls


async def _tick(answer, chain=None):
    from app import reverie

    cortex = AsyncMock()
    cortex.execute_plan = AsyncMock(return_value={"final_text": json.dumps(answer)})
    result = await reverie.run_reverie_once(AsyncMock(), broadcast_reader=_broadcast, cortex_client=cortex,
                                            chain_context=chain)
    return cortex.execute_plan.call_args.kwargs["req"].context, result


def test_organs_lived_key_is_classified_and_rendered_in_both_branches():
    from app import reverie

    assert classify("organs_lived") == "derived_summary"
    lines = "- [experience:street_event:w1] camera: street event: a mail truck stopped (20 min ago)"
    env = Environment()
    out = env.from_string(_PROMPT.read_text()).render(**reverie.build_reverie_context(_broadcast(),
                                                                                       organs_lived=lines))
    assert "WHAT YOUR OTHER ORGANS LIVED RECENTLY" in out and "[experience:street_event:w1]" in out
    assert '"cites"' in out and "NOT evidence_refs" in out
    card = ConcernCardV1(card_id="c1", coalition_ref="ol-1", human_text="deploy may slip", thread_label="deploy")
    ctx = reverie.build_reverie_context(_broadcast(), concern_cards=[card], organs_lived=lines)
    assert "WHAT YOUR OTHER ORGANS LIVED RECENTLY" in env.from_string(_PROMPT.read_text()).render(**ctx)
    assert "organs_lived" not in reverie.build_reverie_context(_broadcast())
    assert "WHAT YOUR OTHER ORGANS LIVED" not in env.from_string(_PROMPT.read_text()).render(
        **reverie.build_reverie_context(_broadcast()))


@pytest.mark.asyncio
async def test_a_tick_reads_the_now_without_since_and_shows_it(now_seam):
    ctx, thought = await _tick({"interpretation": GROUNDED_TEXT, "evidence_refs": ["ol-1"]})
    [url] = now_seam["get"]
    q = parse_qs(urlparse(url).query)
    assert urlparse(url).path == "/temporal-self/now" and q["organ"] == ["reverie"] and "since" not in q
    assert q["run_ref"][0].startswith("reverie_tick:")
    assert "[experience:street_event:w1] camera:" in ctx["organs_lived"]
    assert thought is not None and now_seam["post"] == []  # nothing cited, nothing posted


@pytest.mark.asyncio
async def test_one_read_per_chain_and_steps_reuse_it(now_seam):
    answer = {"interpretation": GROUNDED_TEXT, "evidence_refs": ["ol-1"]}
    await _tick(answer, chain=("chain-1", 0))
    await _tick(answer, chain=("chain-1", 1))
    assert len(now_seam["get"]) == 1
    assert parse_qs(urlparse(now_seam["get"][0]).query)["run_ref"] == ["reverie_chain:chain-1"]
    await _tick(answer, chain=("chain-2", 0))
    assert len(now_seam["get"]) == 2


@pytest.mark.asyncio
async def test_a_published_thought_posts_only_shown_cites_and_experience_refs_are_never_grounding(now_seam):
    ctx, thought = await _tick({"interpretation": GROUNDED_TEXT,
                                "evidence_refs": ["ol-1", "experience:reading_turn:act_1"],
                                "cites": ["experience:street_event:w1", "experience:never_shown:9"]})
    assert thought is not None and not thought.hollow
    assert thought.evidence_refs == ["ol-1"]
    [(url, body)] = now_seam["post"]
    assert url.endswith("/temporal-self/now/use") and body["shown_id"] == "shown-r1"
    assert body["citing_ref"] == f"reverie_thought:{thought.thought_id}"
    assert body["cites"] == ["street_event:w1", "reading_turn:act_1"]


@pytest.mark.asyncio
async def test_an_experience_ref_alone_is_hollow_and_nothing_is_posted(now_seam):
    _ctx, thought = await _tick({"interpretation": GROUNDED_TEXT, "evidence_refs": ["experience:street_event:w1"]})
    assert thought is None  # dropped as hollow: an experience id is never grounding
    assert now_seam["post"] == []


@pytest.mark.asyncio
async def test_a_failed_read_is_a_reverie_without_the_section(now_seam):
    now_seam["page"] = TimeoutError("slow")
    ctx, thought = await _tick({"interpretation": GROUNDED_TEXT, "evidence_refs": ["ol-1"],
                                "cites": ["experience:street_event:w1"]})
    assert thought is not None and "organs_lived" not in ctx and now_seam["post"] == []


@pytest.mark.asyncio
async def test_flag_off_never_reads(now_seam, monkeypatch):
    from app.settings import settings

    monkeypatch.setattr(settings, "reverie_now_enabled", False)
    ctx, thought = await _tick({"interpretation": GROUNDED_TEXT, "evidence_refs": ["ol-1"]})
    assert now_seam["get"] == [] and "organs_lived" not in ctx and thought is not None


def test_reverie_cites_keeps_shown_ids_in_order_and_caps_at_three():
    from app.temporal_self_cue import reverie_cites

    shown = ["a:1", "b:2", "c:3", "d:4"]
    cites, refs = reverie_cites(["experience:b:2", "[experience:a:1]", "x:9", "experience:b:2"],
                                ["ol-1", "[experience:c:3]"], shown)
    assert cites == ["b:2", "a:1", "c:3"] and refs == ["ol-1"]
    assert reverie_cites({"bad": 1}, [], shown) == ([], [])
    cites, _ = reverie_cites(["experience:a:1", "experience:b:2", "experience:c:3", "experience:d:4"], [], shown)
    assert len(cites) == 3


@pytest.mark.asyncio
async def test_a_raising_client_never_drops_the_tick(now_seam, monkeypatch):
    from orion.integration import now_client

    async def boom(*_a, **_k):
        raise RuntimeError("client bug")

    monkeypatch.setattr(now_client, "aread_now", boom)
    ctx, thought = await _tick({"interpretation": GROUNDED_TEXT, "evidence_refs": ["ol-1"]})
    assert thought is not None and "organs_lived" not in ctx
