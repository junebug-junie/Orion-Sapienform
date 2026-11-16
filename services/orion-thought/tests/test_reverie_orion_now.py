"""Integration patch 3: reverie reads Orion's now and records its cites.

Spec: docs/superpowers/specs/2026-11-15-orion-integration-and-a-life-design.md (A3, A4, rollback rule).
Covers: the cue built from a fake now (GET /temporal-self/now), cites limited to shown items and capped
at 3, the cites POSTed at the thought's close with the shown_id, the fallback to the closed-arcs block
when the now is off (404), unreachable or slow, one now per chain, and failures never failing a reverie.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
from jinja2 import Environment

from orion.schemas.attention_frame import AttentionBroadcastProjectionV1, AttentionFrameV1, OpenLoopV1
from orion.schemas.context_provenance import classify
from orion.schemas.reverie import MAX_EXPERIENCE_CITES
from orion.schemas.temporal_self import NOW_MAX_CITES

_REPO = Path(__file__).resolve().parents[3]
_PROMPT = _REPO / "orion" / "cognition" / "prompts" / "reverie_narrate.j2"
_FRAME = json.loads((_REPO / "orion/temporal_self/tests/patch4_rows/frame_2026-10-11_talking.json").read_text())
GROUNDED_TEXT = (
    "I keep returning to the unresolved thread ol-1 — the conflict about the "
    "deployment plan has not discharged and it is pulling attention."
)
SHOWN_ID = "shown-abc123"
EVENT_IDS = ["ev-cam-1", "ev-read-2", "ev-dream-3", "ev-body-4"]


def _fake_now(as_of: datetime | None = None) -> dict:
    as_of = as_of or datetime.now(timezone.utc)
    organs = [("camera", "vision_percept", "saw person, dog"), ("reading", "reading_turn", "reading turn: a paper on sleep"),
              ("dream", "dream_cycle", "dream cycle: 4 hypotheses from 9 replayed items"),
              ("body", "gpu_wait", "a background lease waited 900 ms for a GPU")]
    return {
        "schema_version": "temporal_self.now.v1",
        "shown_id": SHOWN_ID,
        "reader_organ": "reverie",
        "reader_run_ref": None,
        "since": (as_of - timedelta(hours=1)).isoformat(),
        "as_of": as_of.isoformat(),
        "items": [
            {"event_id": eid, "organ": organ, "source_kind": kind, "label": label,
             "occurred_at": (as_of - timedelta(minutes=10 * (i + 1))).isoformat(),
             "salience": 0.9 - 0.1 * i, "surprise_nats": None, "related_refs": []}
            for i, (eid, (organ, kind, label)) in enumerate(zip(EVENT_IDS, organs))
        ],
        "candidates": 9,
        "dropped_by_organ_cap": 0,
        "dropped_by_total_cap": 0,
        "unregistered_kinds": [],
    }


def _frame_now() -> dict:
    from orion.temporal_self.day import day_id_for

    frame = dict(_FRAME)
    now = datetime.now(timezone.utc)
    frame.update(as_of=now.isoformat(), day_id=day_id_for(now))
    return frame


def _broadcast():
    return AttentionBroadcastProjectionV1(
        frame=AttentionFrameV1(open_loops=[OpenLoopV1(id="ol-1", description="d")]),
        attended_node_ids=["n-1"],
        selected_open_loop_id="ol-1",
    )


class _Server:
    """A fake orion-durable-runs: the now, the frame (fallback input), and the use route."""

    def __init__(self, now_response=None, use_response=None):
        self.now_response = now_response
        self.use_response = use_response
        self.now_requests: list[httpx.Request] = []
        self.use_bodies: list[dict] = []
        self.frame_requests = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/temporal-self/now":
            self.now_requests.append(request)
            resp = self.now_response
            if isinstance(resp, Exception):
                raise resp
            return resp if resp is not None else httpx.Response(200, json=_fake_now())
        if path == "/temporal-self/now/use":
            body = json.loads(request.content)
            self.use_bodies.append(body)
            if isinstance(self.use_response, Exception):
                raise self.use_response
            if self.use_response is not None:
                return self.use_response
            return httpx.Response(200, json={
                "schema_version": "temporal_self.experience_use.v1", "shown_id": body["shown_id"],
                "citing_organ": "reverie", "rejected_not_shown": [], "rejected_over_cap": [],
                "uses": [{"use_id": f"u{i}", "event_id": e, "citing_organ": "reverie", "citing_ref": body["citing_ref"],
                          "shown_id": body["shown_id"], "cited_at": datetime.now(timezone.utc).isoformat()}
                         for i, e in enumerate(body["cites"])],
            })
        if path == "/temporal-self/frame":
            self.frame_requests += 1
            return httpx.Response(200, json=_frame_now())
        return httpx.Response(404)


@pytest.fixture
def server(monkeypatch):
    import app.reverie as reverie
    import app.temporal_self_cue as cue_mod
    from app.settings import settings

    srv = _Server()
    monkeypatch.setattr(cue_mod, "_transport", lambda: httpx.MockTransport(srv))
    monkeypatch.setattr(settings, "reverie_now_enabled", True)
    monkeypatch.setattr(settings, "reverie_recent_arcs_enabled", True)
    monkeypatch.setattr(settings, "reverie_perception_enabled", False)
    monkeypatch.setattr(reverie, "_CHAIN_DAY_CUES", {})
    return srv


# --- the cue -------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cue_built_from_fake_now(server):
    from app import reverie

    cue = await reverie.build_reverie_day_cue("c-1")
    assert cue.source == "now" and cue.shown_id == SHOWN_ID
    assert cue.shown_event_ids == tuple(EVENT_IDS)
    assert cue.recent_arcs == "" and cue.arc_ids == ()
    lines = cue.now_block.splitlines()
    assert len(lines) == 4
    assert lines[0].startswith("- [experience:ev-cam-1] saw person, dog (camera, 10 min ago)")
    assert "[experience:ev-read-2] reading turn: a paper on sleep (reading," in lines[1]
    # The read names the organ and no since= (the server's cursor: since reverie last looked).
    req = server.now_requests[0]
    assert req.url.params["organ"] == "reverie" and "since" not in req.url.params
    assert req.url.params["run_ref"] == "reverie_tick:c-1"
    assert server.frame_requests == 0  # the now replaces the arcs read


@pytest.mark.asyncio
async def test_empty_now_shows_nothing_and_does_not_fall_back(server):
    from app import reverie

    body = _fake_now()
    body["items"] = []
    server.now_response = httpx.Response(200, json=body)
    cue = await reverie.build_reverie_day_cue("c-1")
    assert cue.source == "now" and cue.now_block == "" and cue.shown_id == SHOWN_ID
    assert server.frame_requests == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("response,reason", [
    (httpx.Response(404, json={"detail": "off"}), "not_found"),
    (httpx.ConnectError("down"), "unreachable:ConnectError"),
    (httpx.Response(503), "http_503"),
    (httpx.Response(200, json={"garbage": True}), "invalid"),
])
async def test_now_unavailable_falls_back_to_arcs(server, caplog, response, reason):
    from app import reverie

    server.now_response = response
    caplog.set_level("INFO", logger="orion-thought.reverie")
    cue = await reverie.build_reverie_day_cue("c-1")
    assert cue.source == "arcs" and cue.fallback_reason == reason
    assert cue.shown_id is None and cue.now_block == ""
    assert cue.recent_arcs.startswith("Recently closed today") and len(cue.arc_ids) == 3
    assert server.frame_requests == 1
    assert any("reverie_now_fallback" in r.getMessage() and reason in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_slow_now_is_bounded_and_falls_back(server, monkeypatch):
    import app.temporal_self_cue as cue_mod
    from app import reverie
    from app.settings import settings

    async def slow(*a, **k):
        await asyncio.sleep(5)

    monkeypatch.setattr(cue_mod, "_get_now", slow)
    monkeypatch.setattr(settings, "reverie_now_fetch_timeout_sec", 0.05)
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    cue = await reverie.build_reverie_day_cue("c-1")
    assert loop.time() - t0 < 2.0
    assert cue.source == "arcs" and cue.fallback_reason == "timeout"


@pytest.mark.asyncio
async def test_flag_off_never_reads_the_now(server, monkeypatch):
    from app import reverie
    from app.settings import settings

    monkeypatch.setattr(settings, "reverie_now_enabled", False)
    cue = await reverie.build_reverie_day_cue("c-1")
    assert cue.source == "arcs" and cue.fallback_reason is None
    assert server.now_requests == [] and server.frame_requests == 1


@pytest.mark.asyncio
async def test_one_now_per_chain(server):
    from app import reverie

    first = await reverie.build_reverie_day_cue("c-1", chain_id="chain-1")
    second = await reverie.build_reverie_day_cue("c-2", chain_id="chain-1")
    assert first is second and len(server.now_requests) == 1
    assert server.now_requests[0].url.params["run_ref"] == "reverie_chain:chain-1"
    await reverie.build_reverie_day_cue("c-3", chain_id="chain-2")
    assert len(server.now_requests) == 2


# --- cites ---------------------------------------------------------------------------------------


def test_cite_cap_matches_the_now_contract():
    assert MAX_EXPERIENCE_CITES == NOW_MAX_CITES == 3


def test_shown_cites_keeps_only_shown_ids_capped_at_three():
    from app.temporal_self_cue import shown_cites

    raw = ["experience:ev-cam-1", "[experience:ev-read-2]", "ev-not-shown", "ev-cam-1",
           "experience:ev-dream-3", "ev-body-4", "", None]
    assert shown_cites(raw, EVENT_IDS) == ["ev-cam-1", "ev-read-2", "ev-dream-3"]
    assert shown_cites(["ev-x", "experience:ev-y"], EVENT_IDS) == []
    assert shown_cites(["ev-cam-1"], []) == []


def _parse(raw, now_ids=tuple(EVENT_IDS)):
    from app import reverie

    b = _broadcast()
    return reverie.parse_reverie_payload(raw, coalition=reverie.build_coalition_snapshot(b),
                                         correlation_id="c", broadcast=b, now_event_ids=now_ids)


def test_parse_reads_cites_field_only_shown_and_capped():
    thought = _parse({"interpretation": GROUNDED_TEXT, "evidence_refs": ["ol-1"],
                      "cites": ["experience:ev-read-2", "experience:ev-made-up", "ev-cam-1",
                                "experience:ev-dream-3", "experience:ev-body-4"]})
    assert thought.experience_cites == ["ev-read-2", "ev-cam-1", "ev-dream-3"]
    assert not thought.hollow and thought.evidence_refs == ["ol-1"]


def test_experience_id_in_evidence_refs_is_moved_never_grounding():
    thought = _parse({"interpretation": GROUNDED_TEXT, "evidence_refs": ["ol-1", "experience:ev-cam-1"]})
    assert thought.evidence_refs == ["ol-1"] and not thought.hollow
    assert thought.experience_cites == ["ev-cam-1"]
    # An experience id alone never anchors a thought.
    only = _parse({"interpretation": GROUNDED_TEXT, "evidence_refs": ["[experience:ev-cam-1]"]})
    assert only.hollow and only.hollow_reason == "unanchored_no_evidence"


def test_no_now_shown_means_no_cites():
    thought = _parse({"interpretation": GROUNDED_TEXT, "evidence_refs": ["ol-1"], "cites": ["ev-cam-1"]},
                     now_ids=())
    assert thought.experience_cites == []


def test_malformed_cites_field_is_ignored():
    thought = _parse({"interpretation": GROUNDED_TEXT, "evidence_refs": ["ol-1"], "cites": "ev-cam-1"})
    assert thought.experience_cites == []


# --- the prompt and context ----------------------------------------------------------------------


def _render(**ctx) -> str:
    return Environment().from_string(_PROMPT.read_text()).render(**ctx)


def test_context_and_prompt_carry_the_now_only_when_given():
    from app import reverie

    ctx = reverie.build_reverie_context(_broadcast())
    assert "orion_now" not in ctx
    out = _render(**ctx)
    assert "SINCE YOUR LAST REVERIE" not in out and "cites:" not in out

    block = "- [experience:ev-cam-1] saw person, dog (camera, 10 min ago)"
    ctx = reverie.build_reverie_context(_broadcast(), orion_now=block)
    assert ctx["orion_now"] == block
    out = _render(**ctx)
    assert "SINCE YOUR LAST REVERIE" in out and block in out
    assert "- cites: JSON array, at most 3" in out
    assert "[experience:...] ids are NOT evidence_refs" in out


def test_concern_card_mode_also_carries_the_now():
    from app import reverie
    from orion.schemas.reverie import ConcernCardV1

    card = ConcernCardV1(card_id="c1", coalition_ref="ol-1", human_text="deploy may slip", thread_label="deploy")
    ctx = reverie.build_reverie_context(_broadcast(), concern_cards=[card], orion_now="- [experience:e] x (camera, 1 min ago)")
    out = _render(**ctx)
    assert "SINCE YOUR LAST REVERIE" in out and "- cites: JSON array" in out


def test_orion_now_ctx_key_is_classified():
    assert classify("orion_now") == "derived_summary"


# --- the tick end to end -------------------------------------------------------------------------


async def _tick(cites=None, refs=("ol-1",), chain_context=None):
    from app import reverie

    payload = {"interpretation": GROUNDED_TEXT, "evidence_refs": list(refs)}
    if cites is not None:
        payload["cites"] = list(cites)
    cortex = AsyncMock()
    cortex.execute_plan = AsyncMock(return_value={"status": "success", "final_text": json.dumps(payload)})
    result = await reverie.run_reverie_once(AsyncMock(), broadcast_reader=_broadcast, cortex_client=cortex,
                                            chain_context=chain_context)
    ctx = cortex.execute_plan.call_args.kwargs["req"].context
    return ctx, result


@pytest.mark.asyncio
async def test_tick_shows_the_now_and_posts_cites_at_close(server, monkeypatch):
    import app.reverie as reverie

    monkeypatch.setattr(reverie, "persist_reverie_thought", lambda t: True)
    ctx, thought = await _tick(cites=["experience:ev-read-2", "experience:ev-unshown"])
    assert "[experience:ev-cam-1]" in ctx["orion_now"] and "recent_arcs" not in ctx
    assert thought is not None and thought.experience_cites == ["ev-read-2"]
    assert server.use_bodies == [{"shown_id": SHOWN_ID, "citing_ref": f"reverie_thought:{thought.thought_id}",
                                  "cites": ["ev-read-2"]}]


@pytest.mark.asyncio
async def test_tick_with_no_cites_posts_nothing(server, monkeypatch):
    import app.reverie as reverie

    monkeypatch.setattr(reverie, "persist_reverie_thought", lambda t: True)
    _, thought = await _tick(cites=[])
    assert thought is not None and thought.experience_cites == []
    assert server.use_bodies == []


@pytest.mark.asyncio
@pytest.mark.parametrize("use_response", [httpx.Response(500), httpx.ConnectError("down"),
                                          httpx.Response(404, json={"detail": "no shown row"})])
async def test_use_post_failure_never_fails_the_reverie(server, monkeypatch, use_response):
    import app.reverie as reverie

    monkeypatch.setattr(reverie, "persist_reverie_thought", lambda t: True)
    server.use_response = use_response
    _, thought = await _tick(cites=["ev-cam-1"])
    assert thought is not None and thought.experience_cites == ["ev-cam-1"]
    assert len(server.use_bodies) == 1


@pytest.mark.asyncio
async def test_tick_falls_back_to_arcs_when_the_now_is_off(server, monkeypatch):
    import app.reverie as reverie

    monkeypatch.setattr(reverie, "persist_reverie_thought", lambda t: True)
    server.now_response = httpx.Response(404)
    ctx, thought = await _tick(cites=["ev-cam-1"])
    assert "orion_now" not in ctx and ctx["recent_arcs"].startswith("Recently closed today")
    assert thought is not None and thought.experience_cites == []
    assert server.use_bodies == []


@pytest.mark.asyncio
async def test_chain_steps_share_one_shown_id(server, monkeypatch):
    import app.reverie as reverie

    monkeypatch.setattr(reverie, "persist_reverie_thought", lambda t: True)
    await _tick(cites=["ev-cam-1"], chain_context=("chain-9", 0))
    await _tick(cites=["ev-read-2"], chain_context=("chain-9", 1))
    assert len(server.now_requests) == 1
    assert [b["shown_id"] for b in server.use_bodies] == [SHOWN_ID, SHOWN_ID]
    assert [b["cites"] for b in server.use_bodies] == [["ev-cam-1"], ["ev-read-2"]]
