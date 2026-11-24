"""Prime reads Orion's now (integration patch 3, docs/superpowers/specs/2026-11-15-orion-integration-and-a-life-design.md).

Each thought reads GET /temporal-self/now?organ=prime&since=<last thought - 120 s> into WHAT YOUR ORGANS LIVED SINCE
YOUR LAST THOUGHT, keeps the shown_id, and at its close POSTs only the shown experiences it cited (at most 3) to
/temporal-self/now/use. The now off or failing leaves no section and never fails the thought. The frame no longer
reads Hub's curiosity dashboard. Fake bus, model and pages; no network."""

from __future__ import annotations

import asyncio
import inspect
import logging
import urllib.error
import urllib.parse
from datetime import timedelta

import pytest

from app.prime_loop import PrimeRuntime
from orion.prime.state import PrimeStateV1
from test_prime_loop import MOVE, NO_SEAT, T0, SeqLLM, runtime

SECTION = "WHAT YOUR ORGANS LIVED SINCE YOUR LAST THOUGHT"
EIDS = [f"curiosity_run:run{i}" if i % 2 else f"reading_turn:read{i}" for i in range(5)]


def now_page(n=5, shown_id="shown-1"):
    items = [{"event_id": EIDS[i], "organ": "curiosity" if i % 2 else "reading",
              "source_kind": "curiosity_run" if i % 2 else "reading_turn", "label": f"thing {i} happened",
              "occurred_at": (T0 - timedelta(minutes=10 + i)).isoformat(), "salience": 0.9 - i / 10,
              "surprise_nats": 2.5 if i == 0 else None, "related_refs": [f"act:act_{i}"]} for i in range(n)]
    return {"schema_version": "temporal_self.now.v1", "shown_id": shown_id, "reader_organ": "prime",
            "since": (T0 - timedelta(hours=1)).isoformat(), "as_of": T0.isoformat(), "items": items,
            "candidates": n, "dropped_by_organ_cap": 0, "dropped_by_total_cap": 0, "unregistered_kinds": []}


def answer(*cites, move=None):
    out = {"reading": "the reading was about heat", "concerns": {}, "move": move,
           "refs": [{"kind": "experience", "id": c} for c in cites]}
    return out


def _infer_close(rt):
    [close] = [a for k, a in rt.bus.acts() if k == "act_closed" and a["verb"] == "infer"]
    return close["outcome"]


class Posts:
    def __init__(self, reply=None, raises=None):
        self.calls, self.reply, self.raises = [], reply, raises

    def __call__(self, url, body):
        self.calls.append((url, body))
        if self.raises is not None:
            raise self.raises
        return self.reply if self.reply is not None else {
            "shown_id": body["shown_id"], "citing_organ": "prime",
            "uses": [{"event_id": c} for c in body["cites"]], "rejected_not_shown": [], "rejected_over_cap": []}


def test_the_frame_shows_the_now_with_ids_and_refs_and_reads_since_the_last_thought():
    posts = Posts()
    rt = runtime(answer=answer(), now_page=now_page(), post_json=posts)
    rt.state = PrimeStateV1(pending_changes=["something"], last_thought_at=T0 - timedelta(hours=2))
    asyncio.run(rt.tick())
    [call] = rt.llm.calls
    frame = call["user_prompt"]
    assert SECTION in frame
    section = frame.split(SECTION, 1)[1]
    for eid in EIDS:
        assert f"[experience:{eid}]" in section
    assert "thing 0 happened (salience 0.90; surprise 2.50 nats) refs act:act_0" in section
    assert "- [experience:curiosity_run:run1] curiosity 11 min ago: thing 1 happened" in section
    # The model is told it may cite, and how.
    assert '"kind": "experience"' in call["system_prompt"]
    [url] = [u for u in rt.urls if "/temporal-self/now?" in u]
    q = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
    assert q["organ"] == ["prime"]
    assert q["since"] == [(T0 - timedelta(hours=2, seconds=120)).isoformat()]
    assert q["run_ref"][0].startswith("prime_thought:")
    # Nothing cited: nothing posted (the shown row already says "read, cited nothing").
    assert posts.calls == []
    assert _infer_close(rt)["value"] == 1.0


def test_prime_sends_what_it_holds_and_the_frame_marks_an_item_about_a_held_concern():
    """#3783: the now read carries each held concern's event_ref, arc and own id; an item the server matched to one
    (``held_refs``) is marked with that concern's id."""
    from orion.prime.state import PrimeConcernV1

    seed = "world_pulse_read_seed:reading:50753c3f-d556-5a6f-bfa7-b02b95254905"
    held = PrimeConcernV1(concern_id="520a5a9f8d084495", text="Investigate the MemOnce GitHub read", event_ref=seed,
                          opened_at=T0 - timedelta(hours=3), last_evidence_at=T0 - timedelta(hours=3))
    arc = PrimeConcernV1(concern_id="00000000000000aa", text="the heat arc", arc_id="0011223344556677",
                         opened_at=T0 - timedelta(hours=1), last_evidence_at=T0 - timedelta(hours=1))
    waiting = PrimeConcernV1(concern_id="00000000000000bb", text="waits", event_ref="dream_day:2026-11-23#new1",
                             opened_at=T0, last_evidence_at=T0)
    page = now_page()
    page["items"][0]["held_refs"] = [seed]
    rt = runtime(answer=answer(), now_page=page)
    from orion.prime.thought import held_now_refs

    # A waiting concern has no slot: not held, not sent (checked on the state; a tick may give it a free slot).
    assert "dream_day:2026-11-23#new1" not in held_now_refs(PrimeStateV1(concerns=[held, arc], waiting=[waiting]))
    rt.state = PrimeStateV1(concerns=[held, arc], pending_changes=["something"],
                            last_thought_at=T0 - timedelta(hours=2))
    asyncio.run(rt.tick())
    [url] = _now_gets(rt)
    q = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
    assert q["held"] == [seed, "prime_concern:520a5a9f8d084495", "arc:0011223344556677",
                         "prime_concern:00000000000000aa"]
    section = rt.llm.calls[0]["user_prompt"].split(SECTION, 1)[1]
    assert f"[experience:{EIDS[0]}]" in section and "ABOUT YOUR CONCERN 520a5a9f8d084495" in section
    assert section.count("ABOUT YOUR CONCERN") == 1


def test_the_first_thought_ever_reads_without_since():
    rt = runtime(answer=answer(), now_page=now_page())
    asyncio.run(rt.tick())
    [url] = [u for u in rt.urls if "/temporal-self/now?" in u]
    assert "since=" not in url


def test_cites_are_posted_with_the_shown_id_only_shown_ones_and_at_most_three(caplog):
    caplog.set_level(logging.INFO)
    posts = Posts()
    cites = ["reading_turn:not-shown", EIDS[0], f"experience:{EIDS[1]}", EIDS[2], EIDS[3], EIDS[0]]
    rt = runtime(answer=answer(*cites), now_page=now_page(shown_id="sh-42"), post_json=posts)
    asyncio.run(rt.tick())
    [(url, body)] = posts.calls
    assert url == "http://dr:8121/temporal-self/now/use"
    assert body["shown_id"] == "sh-42"
    assert body["cites"] == [EIDS[0], EIDS[1], EIDS[2]]  # shown only, cite order, cap 3, dupes once
    thought_id = rt.state.last_thought_id
    assert body["citing_ref"] == f"prime_thought:{thought_id}"
    note = _infer_close(rt)["note"]
    assert "proposed 3 experience cite(s)" in note and "not shown in this frame" in note
    assert any(r.getMessage().startswith(f"prime_now_use thought={thought_id} shown_id=sh-42") and "stored=3" in
               r.getMessage() for r in caplog.records)
    assert any("now_use=stored 3/3" in r.getMessage() for r in caplog.records if "prime_thought " in r.getMessage())
    # The server's verdict is on the act note, next to the proposal.
    assert "proposed 3 experience cite(s), now_use stored 3/3" in note


def test_an_experience_ref_in_a_move_is_a_cite_and_never_reaches_curiosity():
    posts = Posts()
    move = dict(MOVE["move"], refs=[{"kind": "experience", "id": EIDS[1]}])
    rt = runtime(answer={**MOVE, "move": move}, now_page=now_page(), post_json=posts)
    asyncio.run(rt.tick())
    [(_channel, env)] = rt.bus.rpcs  # the question still went
    assert not env.payload["args"].get("refs")
    [(_url, body)] = posts.calls
    assert body["cites"] == [EIDS[1]]


def test_a_failed_use_post_never_fails_the_thought(caplog):
    caplog.set_level(logging.INFO)
    posts = Posts(raises=urllib.error.URLError("refused"))
    rt = runtime(answer=answer(EIDS[0]), now_page=now_page(), post_json=posts)
    asyncio.run(rt.tick())
    assert len(posts.calls) == 1
    assert _infer_close(rt)["value"] == 1.0
    assert any(r.getMessage().startswith("prime_now_use_failed") for r in caplog.records)
    assert rt.state.last_reading == "the reading was about heat"


@pytest.mark.parametrize("page", [
    urllib.error.HTTPError("u", 404, "off", {}, None),          # TEMPORAL_SELF_NOW_ENABLED=false
    urllib.error.HTTPError("u", 503, "tables missing", {}, None),
    urllib.error.URLError("connection refused"),               # durable-runs unreachable
    TimeoutError("timed out"),
    {"items": []},                                             # no shown_id: not a now
    ["not", "a", "now"],
])
def test_the_now_failing_leaves_no_section_and_the_thought_goes_on(page):
    posts = Posts()
    rt = runtime(answer=answer(EIDS[0]), now_page=page, post_json=posts)
    asyncio.run(rt.tick())
    [call] = rt.llm.calls
    assert SECTION not in call["user_prompt"] and '"kind": "experience"' not in call["system_prompt"]
    assert posts.calls == []  # nothing shown, nothing citable
    close = _infer_close(rt)
    assert close["value"] == 1.0 and "the now was not read this thought" in close["note"]


def test_an_empty_now_says_so_and_still_offers_nothing_to_cite():
    page = now_page(n=0)
    posts = Posts()
    rt = runtime(answer=answer(EIDS[0]), now_page=page, post_json=posts)
    asyncio.run(rt.tick())
    frame = rt.llm.calls[0]["user_prompt"]
    assert "(nothing new from the other organs since your last thought)" in frame
    assert posts.calls == []


def test_the_frame_no_longer_reads_the_curiosity_dashboard():
    rt = runtime(answer=MOVE, now_page=now_page())
    asyncio.run(rt.tick())
    assert rt.llm.calls  # a thought happened
    assert not any("/curiosity/api/runs" in u for u in rt.urls)
    # The cheap pending-focus check stays (patch 0, #3366).
    assert any(u.endswith("/curiosity/api/focus/pending") for u in rt.urls)
    src = inspect.getsource(PrimeRuntime._ask)
    assert "_read_curiosity" not in src and "_read_now" in src and "_read_pending_focus" in src


def _now_gets(rt):
    return [u for u in rt.urls if "/temporal-self/now?" in u]


def test_a_try_that_never_reached_the_model_reuses_its_now_instead_of_reading_again():
    posts = Posts()
    rt = runtime(answer=MOVE, now_page=now_page(shown_id="first"), post_json=posts)
    rt.llm = SeqLLM([NO_SEAT, (answer(EIDS[0]), None, {})])
    asyncio.run(rt.tick())
    assert len(_now_gets(rt)) == 1 and rt.bus.acts() == []
    rt.clock_box["now"] = T0 + timedelta(seconds=15)
    asyncio.run(rt.tick())
    assert len(_now_gets(rt)) == 1  # reused: no second experience_shown row the model never saw
    assert SECTION in rt.llm.calls[1]["user_prompt"]
    [(_url, body)] = posts.calls
    assert body["shown_id"] == "first"


def test_the_next_read_starts_where_the_seen_now_ended_and_a_failed_read_skips_nothing():
    page = now_page()
    page["as_of"] = (T0 - timedelta(seconds=118)).isoformat()
    rt = runtime(answer=answer(), now_page=page)
    asyncio.run(rt.tick())
    rt.clock_box["now"] = T0 + timedelta(hours=1)
    rt.state = rt.state.model_copy(update={"pending_changes": ["again"]})
    asyncio.run(rt.tick())
    q = urllib.parse.parse_qs(urllib.parse.urlsplit(_now_gets(rt)[-1]).query)
    assert q["since"] == [page["as_of"]]
    # A thought whose now read failed: the next read still starts at the last SEEN cut-off.
    rt2 = runtime(answer=answer(), now_page=urllib.error.URLError("refused"))
    rt2._now_seen_as_of = T0 - timedelta(hours=3)
    rt2.state = PrimeStateV1(pending_changes=["x"], last_thought_at=T0 - timedelta(hours=1))
    asyncio.run(rt2.tick())
    rt2.clock_box["now"] = T0 + timedelta(hours=1)
    rt2.state = rt2.state.model_copy(update={"pending_changes": ["again"]})
    asyncio.run(rt2.tick())
    for url in _now_gets(rt2):
        assert urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)["since"] == [(T0 - timedelta(hours=3)).isoformat()]


def test_the_real_use_client_has_its_own_short_timeout():
    rt = PrimeRuntime(settings=runtime(answer=None).s, bus=None, llm=None)
    assert rt._post_use.func.__name__ == "_post_json" and rt._post_use.keywords == {"timeout": 10.0}
