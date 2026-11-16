"""Curiosity reads Orion's now at kickoff and cites what its finding used (integration patch 3, #3425).

A real tick drives the whole path: the kickoff reads the now (no ``since``), offers it as candidates, keeps the shown
row in Redis, and at turn end posts only the shown items the finding names. A failed read is a kickoff without the
section. The read and the section are identical under every offer arm (#3183 is blinded)."""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from orion.curiosity.kickoff_prompt import build_kickoff_prompt
from orion.curiosity.study_material import StudyMaterial
from orion.curiosity.value import ARM_REPLICA_ORDER, ARM_UNCERTAINTY_ORDER, ARM_VALUE_ORDER, ArmDraw
from orion.integration import now_client
from scripts import curiosity_now
from scripts.curiosity_investigation import CuriosityInvestigation
from test_curiosity_investigation import NOW, _FakeBus, _loop

HEADING = "WHAT YOUR OTHER ORGANS LIVED SINCE CURIOSITY LAST LOOKED"
NOW_BODY = {
    "shown_id": "shown-c1", "reader_organ": "curiosity", "since": "2026-11-16T00:00:00Z",
    "as_of": "2026-11-16T01:00:00Z", "candidates": 9,
    "items": [
        {"event_id": "visual_run:v1", "organ": "painting", "source_kind": "visual_run",
         "label": "visual run: the walkway at dusk", "occurred_at": "2026-11-16T00:40:00Z", "salience": None},
        {"event_id": "prediction_surprise:14945", "organ": "prediction", "source_kind": "prediction_surprise",
         "label": "prediction surprise: cabinet forecast 0%, crossed", "occurred_at": "2026-11-16T00:10:00Z",
         "salience": 0.97},
    ],
}


@pytest.fixture()
def http(monkeypatch):
    calls = {"get": [], "post": [], "page": NOW_BODY}

    def get(url, timeout):
        calls["get"].append(url)
        if isinstance(calls["page"], BaseException):
            raise calls["page"]
        return calls["page"]

    def post(url, body, timeout):
        calls["post"].append((url, body))
        return {"shown_id": body["shown_id"], "citing_organ": "curiosity", "uses": [
            {"use_id": f"u{i}", "event_id": c, "citing_organ": "curiosity", "citing_ref": body["citing_ref"],
             "shown_id": body["shown_id"], "cited_at": "2026-11-16T01:00:00Z"} for i, c in enumerate(body["cites"])]}

    monkeypatch.setattr(now_client, "_http_get", get)
    monkeypatch.setattr(now_client, "_http_post", post)
    curiosity_now._LOCAL.clear()
    return calls


def _now_loop(bus, text="found it", **over):
    return _loop(bus, text=text, now_enabled=True, durable_runs_url="http://dr.test", **over)


def _section(prompt: str) -> str:
    if HEADING not in prompt:
        return ""
    return prompt[prompt.index(HEADING):].split("\n\n", 1)[0]


def test_a_tick_reads_the_now_offers_it_and_keeps_the_shown_row(http):
    bus = _FakeBus()
    loop = _now_loop(bus)
    assert asyncio.run(loop.tick()) is None
    [url] = http["get"]
    q = parse_qs(urlparse(url).query)
    assert urlparse(url).path == "/temporal-self/now" and q["organ"] == ["curiosity"] and "since" not in q
    run_id = q["run_ref"][0].removeprefix("curiosity_run:")
    sec = _section(loop.seen_prompt)
    assert "- [experience:visual_run:v1] painting: visual run: the walkway at dusk" in sec
    assert "[experience:prediction_surprise:14945] prediction:" in sec
    kept = json.loads(bus.redis.values[f"{curiosity_now.SHOWN_KEY_PREFIX}{run_id}"])
    assert kept == {"shown_id": "shown-c1", "shown_ids": ["visual_run:v1", "prediction_surprise:14945"]}
    assert http["post"] == []  # "found it" names nothing


def test_the_finding_cites_only_shown_items_it_names(http):
    bus = _FakeBus()
    text = ("I took up [experience:prediction_surprise:14945]: the forecast said 0% and it crossed. "
            "Unlike experience:vision_percept:zz, which I was never shown.")
    loop = _now_loop(bus, text=text)
    assert asyncio.run(loop.tick()) is None
    [(url, body)] = http["post"]
    run_id = parse_qs(urlparse(http["get"][0]).query)["run_ref"][0]
    assert url == "http://dr.test/temporal-self/now/use"
    assert body == {"shown_id": "shown-c1", "citing_ref": run_id, "cites": ["prediction_surprise:14945"]}


def test_a_failed_read_is_a_kickoff_without_the_section(http):
    http["page"] = TimeoutError("slow")
    bus = _FakeBus()
    loop = _now_loop(bus, text="experience:visual_run:v1")
    assert asyncio.run(loop.tick()) is None
    assert HEADING not in loop.seen_prompt
    assert not [k for k in bus.redis.values if k.startswith(curiosity_now.SHOWN_KEY_PREFIX)]
    assert http["post"] == []


def test_flag_off_never_reads(http):
    bus = _FakeBus()
    loop = _loop(bus, durable_runs_url="http://dr.test")  # now_enabled defaults off for bare constructions
    assert asyncio.run(loop.tick()) is None
    assert http["get"] == [] and HEADING not in loop.seen_prompt


def test_cites_survive_a_hub_restart_through_redis(http):
    bus = _FakeBus()
    lines = asyncio.run(curiosity_now.read_for_run("http://dr.test", "run1", enabled=True, redis=bus.redis))
    assert len(lines) == 2
    curiosity_now._LOCAL.clear()  # the restart: only Redis remembers
    stored = asyncio.run(curiosity_now.record_run_cites("http://dr.test", "run1", "used experience:visual_run:v1.",
                                                        enabled=True, redis=bus.redis))
    assert stored == 1 and http["post"][0][1]["cites"] == ["visual_run:v1"]


def test_a_run_whose_kickoff_read_nothing_posts_nothing(http):
    assert asyncio.run(curiosity_now.record_run_cites("http://dr.test", "never", "experience:visual_run:v1",
                                                      enabled=True, redis=_FakeBus().redis)) == 0
    assert http["post"] == []


@pytest.mark.parametrize("arm", [ARM_VALUE_ORDER, ARM_REPLICA_ORDER, ARM_UNCERTAINTY_ORDER])
def test_the_now_is_read_and_shown_identically_under_every_offer_arm(http, arm, monkeypatch):
    """#3183 is blinded until its review: the now must behave the same under every arm. Each arm's tick must send
    the same request (up to the run id) and render a byte-identical section that names no arm."""
    g = CuriosityInvestigation.tick.__globals__

    def fixed(run_id, **_kw):
        return ArmDraw(arm=arm, propensities={ARM_VALUE_ORDER: 0.3, ARM_REPLICA_ORDER: 0.3,
                                              ARM_UNCERTAINTY_ORDER: 0.4})

    async def no_plan(*, view, **_kw):
        class _P:
            pass
        p = _P()
        p.view = view
        return p

    monkeypatch.setitem(g, "draw_offer_arm", fixed)
    monkeypatch.setitem(g, "plan_offer", no_plan)
    bus = _FakeBus()
    loop = _now_loop(bus)
    assert asyncio.run(loop.tick()) is None
    [url] = http["get"]
    q = parse_qs(urlparse(url).query)
    assert sorted(q) == ["organ", "run_ref"] and q["organ"] == ["curiosity"]
    sec = _section(loop.seen_prompt)
    expected = "\n".join([HEADING + " (Orion's now; candidates, not orders):",
                          "- [experience:visual_run:v1] painting: visual run: the walkway at dusk (20 min ago)",
                          "- [experience:prediction_surprise:14945] prediction: prediction surprise: cabinet "
                          "forecast 0%, crossed (50 min ago; how unusual 0.97)"])
    assert sec.startswith(expected), sec
    lowered = sec.lower()
    for leak in (arm.lower(), "propensit", " arm", "replica", "uncertainty order", "value order"):
        assert leak not in lowered, leak


def test_curiosity_now_takes_no_arm_view_or_propensity():
    """Structural guard: the module and the loop's two seams never name the arm, a propensity or the view."""
    src = Path(curiosity_now.__file__).read_text()
    names = {n.id for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Name)} | {
        a.arg for n in ast.walk(ast.parse(src)) if isinstance(n, ast.arguments) for a in n.args + n.kwonlyargs}
    for bad in ("arm", "draw", "view", "value_arm_propensity", "propensities", "plan"):
        assert bad not in names, bad
    for fn in (CuriosityInvestigation._read_now_lines, CuriosityInvestigation._record_now_cites):
        params = set(inspect.signature(fn).parameters)
        assert params <= {"self", "run_id", "text"}, params


def test_kickoff_prompt_renders_the_section_with_or_without_a_graph_and_nothing_when_empty():
    lines = ["- [experience:visual_run:v1] painting: visual run (20 min ago)"]
    material = StudyMaterial(generated_at=NOW)
    for graph in (True, False):
        p = build_kickoff_prompt(material, run_id="r", graph_enabled=graph, now_lines=lines)
        assert HEADING in p and lines[0] in p and "experience:..." in p
    assert build_kickoff_prompt(material, run_id="r", now_lines=[]) == build_kickoff_prompt(material, run_id="r")
