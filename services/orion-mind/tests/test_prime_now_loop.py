"""Prime reads Orion's now and records what it cited (integration patch 3, #3425).

The frame path reads the now (no ``since``: a real read moves Prime's cursor) and never Hub's curiosity dashboard;
a failed now read is a gap line, never a failed thought; the answer's cites are posted against the shown row only
when there are some; the top 2 items (never the chat organ's) are saved for the Prime block."""

from __future__ import annotations

import asyncio
import json
from urllib.parse import parse_qs, urlparse

from orion.prime.state import PRIME_NOW_SEEN_KEY, PrimeNowSeenV1
from test_prime_loop import MOVE, runtime

NOW_BODY = {
    "shown_id": "shown-1", "reader_organ": "prime", "since": "2026-10-30T06:00:00Z", "as_of": "2026-10-30T11:58:00Z",
    "candidates": 5,
    "items": [
        {"event_id": "street_event:w1", "organ": "camera", "source_kind": "street_event",
         "label": "street event: a mail truck stopped", "occurred_at": "2026-10-30T11:30:00Z", "salience": 0.9},
        {"event_id": "chat_turn:c1", "organ": "chat", "source_kind": "chat_turn", "label": "chat turn by juniper",
         "occurred_at": "2026-10-30T11:40:00Z", "salience": None},
        {"event_id": "reading_turn:act_1", "organ": "reading", "source_kind": "reading_turn",
         "label": "reading turn: stage1_completed: https://arxiv.org/abs/1", "occurred_at": "2026-10-30T10:00:00Z",
         "salience": None},
    ],
}


def now_runtime(*, answer, now_page=NOW_BODY, enabled=True):
    rt = runtime(answer=answer)
    rt.s.PRIME_NOW_ENABLED = enabled
    rt.now_urls, rt.posts = [], []

    def get(url, timeout):
        rt.now_urls.append(url)
        if isinstance(now_page, BaseException):
            raise now_page
        return now_page

    def post(url, body, timeout):
        rt.posts.append((url, body))
        shown = {i["event_id"] for i in NOW_BODY["items"]}
        uses = [{"use_id": f"u{n}", "event_id": c, "citing_organ": "prime", "citing_ref": body["citing_ref"],
                 "shown_id": body["shown_id"], "cited_at": "2026-10-30T12:00:00Z"}
                for n, c in enumerate(x.removeprefix("experience:") for x in body["cites"]) if c in shown]
        return {"shown_id": body["shown_id"], "citing_organ": "prime", "uses": uses,
                "rejected_not_shown": [c for c in body["cites"] if c.removeprefix("experience:") not in shown]}

    rt._now_http, rt._use_http = {"get": get}, {"post": post}
    return rt


def frame(rt):
    return rt.llm.calls[0]["user_prompt"]


def test_a_thought_reads_the_now_moving_its_cursor_and_shows_other_organs_with_refs():
    rt = now_runtime(answer={**MOVE, "cites": ["[experience:street_event:w1]"]})
    asyncio.run(rt.tick())
    [url] = rt.now_urls
    q = parse_qs(urlparse(url).query)
    assert urlparse(url).path == "/temporal-self/now" and q["organ"] == ["prime"] and "since" not in q
    assert q["run_ref"][0].startswith("prime_thought:")
    f = frame(rt)
    assert "WHAT YOUR ORGANS LIVED SINCE YOUR LAST THOUGHT" in f
    assert "- [experience:street_event:w1] camera: street event: a mail truck stopped" in f
    assert '"cites"' in rt.llm.calls[0]["system_prompt"]


def test_the_frame_path_never_reads_the_curiosity_dashboard():
    seen = []
    rt = now_runtime(answer=MOVE)
    inner = rt.get_json

    def spy(url):
        seen.append(url)
        return inner(url)

    rt.get_json = rt.get_json_long = spy
    rt._last_settle = rt.clock()  # no settling this tick: only the frame path runs
    asyncio.run(rt.tick())
    assert rt.llm.calls, "the thought ran"
    assert not [u for u in seen if "/curiosity/api/runs" in u]
    assert "- run " not in frame(rt).split("CURIOSITY:\n", 1)[1].split("\n\n", 1)[0]


def test_cites_are_posted_once_against_the_shown_row_with_the_thought_ref():
    rt = now_runtime(answer={**MOVE, "cites": ["experience:street_event:w1", "experience:not_shown:9"]})
    asyncio.run(rt.tick())
    [(url, body)] = rt.posts
    assert url == "http://dr:8121/temporal-self/now/use"
    assert body["shown_id"] == "shown-1" and body["citing_ref"].startswith("prime_thought:")
    assert body["cites"] == ["experience:street_event:w1", "experience:not_shown:9"]
    # the same thought id on the read and on the cite
    run_ref = parse_qs(urlparse(rt.now_urls[0]).query)["run_ref"][0]
    assert body["citing_ref"] == run_ref


def test_no_cites_means_no_post():
    rt = now_runtime(answer={**MOVE, "cites": []})
    asyncio.run(rt.tick())
    assert rt.now_urls and rt.posts == []


def test_a_malformed_cites_value_never_fails_the_thought():
    rt = now_runtime(answer={**MOVE, "cites": {"not": "a list"}})
    asyncio.run(rt.tick())
    assert len(rt.state.open_nominations) == 1 and rt.posts == []


def test_a_failed_now_read_is_a_gap_and_the_thought_still_happens():
    rt = now_runtime(answer={**MOVE, "cites": ["experience:street_event:w1"]}, now_page=TimeoutError("slow"))
    asyncio.run(rt.tick())
    assert "(unavailable: timeout)" in frame(rt)
    assert len(rt.state.open_nominations) == 1  # the move still went
    assert rt.posts == []  # nothing was shown, so nothing can be cited


def test_flag_off_means_no_read_and_no_section():
    rt = now_runtime(answer=MOVE, enabled=False)
    asyncio.run(rt.tick())
    assert rt.now_urls == [] and "WHAT YOUR ORGANS LIVED SINCE" not in frame(rt)
    assert "Orion's now is switched off for you" in frame(rt)


def test_the_top_two_items_but_never_the_chat_organ_are_saved_for_the_prime_block():
    rt = now_runtime(answer=MOVE)
    asyncio.run(rt.tick())
    seen = PrimeNowSeenV1.model_validate_json(rt.bus.redis.kv[PRIME_NOW_SEEN_KEY])
    assert seen.shown_id == "shown-1"
    assert [i.event_id for i in seen.items] == ["street_event:w1", "reading_turn:act_1"]
    assert json.loads(rt.bus.redis.kv[PRIME_NOW_SEEN_KEY])["schema_version"] == "prime.now_seen.v1"
