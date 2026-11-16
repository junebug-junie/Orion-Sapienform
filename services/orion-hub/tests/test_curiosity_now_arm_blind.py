"""Curiosity's read of Orion's now is blind to the three-arm offer experiment (#3183, blinded until its review).

Integration patch 3 follow-on (#3425). A real investigate tick runs under each offer arm (value, replica,
uncertainty): the request it sends to durable-runs and the section its kickoff shows must be byte-identical, and
neither may name the arm. A structural test refuses any arm, view or propensity name in the read path, so a later
edit cannot quietly start passing the arm in."""

from __future__ import annotations

import ast
import asyncio
import inspect
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

import scripts.curiosity_now as curiosity_now_module
from orion.curiosity import now_section as now_section_module
from orion.curiosity.now_section import HEADER
from orion.curiosity.value import ARM_REPLICA_ORDER, ARM_UNCERTAINTY_ORDER, ARM_VALUE_ORDER, ArmDraw
from scripts.curiosity_investigation import CuriosityInvestigation
from scripts.curiosity_now import CuriosityNow
from test_curiosity_investigation import _FakeBus, _loop

ARMS = (ARM_VALUE_ORDER, ARM_REPLICA_ORDER, ARM_UNCERTAINTY_ORDER)
NOW_BODY = {
    "shown_id": "shown-arm", "reader_organ": "curiosity", "since": "2026-11-16T00:00:00Z",
    "as_of": "2026-11-16T01:00:00Z", "candidates": 7,
    "items": [
        {"event_id": "visual_run:v1", "organ": "painting", "source_kind": "visual_run",
         "label": "visual run: the walkway at dusk", "occurred_at": "2026-11-16T00:40:00Z", "salience": None},
        {"event_id": "prediction_surprise:14945", "organ": "prediction", "source_kind": "prediction_surprise",
         "label": "prediction surprise: cabinet forecast 0%, crossed", "occurred_at": "2026-11-16T00:10:00Z",
         "salience": 0.97},
    ],
}


def _section(prompt: str) -> str:
    """The now block of a kickoff prompt: from its header to the closing instruction paragraph."""
    if HEADER not in prompt:
        return ""
    tail = prompt[prompt.index(HEADER):]
    end = tail.find("Citing none is a real answer too.")
    return tail[: end + len("Citing none is a real answer too.")] if end >= 0 else tail


def _tick_under(arm: str, monkeypatch) -> tuple[list[str], str]:
    """One real investigate tick with the offer draw pinned to ``arm``. Returns (now request URLs, now section).
    Fails (not passes hollow) if the pin did not take: the arm the offer row recorded must be ``arm``, and the
    replica planner must run on the replica arm only."""
    urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        return httpx.Response(200, json=NOW_BODY)

    g = CuriosityInvestigation.tick.__globals__

    def pinned(run_id, **_kw):
        return ArmDraw(arm=arm, propensities={ARM_VALUE_ORDER: 0.3, ARM_REPLICA_ORDER: 0.3,
                                              ARM_UNCERTAINTY_ORDER: 0.4})

    class _Plan:
        def __init__(self, view):
            self.view = view

    planned: list[bool] = []

    async def no_plan(*, view, **_kw):
        planned.append(True)
        return _Plan(view)  # the replica arm's planner, without the bus round trip

    # The loop reads both names from its own module globals; setitem on a missing name would add a dead key and
    # leave every tick on whatever arm the real draw picks.
    for name in ("draw_offer_arm", "plan_offer"):
        assert name in g, f"{name} is no longer a global of curiosity_investigation: re-point this test"
    monkeypatch.setitem(g, "draw_offer_arm", pinned)
    monkeypatch.setitem(g, "plan_offer", no_plan)
    loop = _loop(_FakeBus())
    recorded: list[str] = []
    real_record = loop._record_offer

    async def spy_record(*args, **kwargs):
        recorded.append(kwargs.get("arm"))
        return await real_record(*args, **kwargs)

    loop._record_offer = spy_record
    loop._now_reader = CuriosityNow(
        enabled=True, durable_runs_url="http://dr.test", pool_provider=lambda: None,
        client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    assert asyncio.run(loop.tick()) is None
    assert recorded == [arm], f"the offer row recorded {recorded}, not the pinned arm {arm}"
    assert bool(planned) is (arm == ARM_REPLICA_ORDER)
    return [u for u in urls if urlparse(u).path == "/temporal-self/now"], _section(loop.seen_prompt)


def test_the_now_is_read_and_shown_identically_under_every_offer_arm(monkeypatch):
    seen = {arm: _tick_under(arm, monkeypatch) for arm in ARMS}
    sections = {arm: sec for arm, (_urls, sec) in seen.items()}
    for arm, (urls, sec) in seen.items():
        [url] = urls
        q = parse_qs(urlparse(url).query)
        # Only the organ and this run's own id: no arm, propensity or plan reaches durable-runs.
        assert sorted(q) == ["organ", "run_ref"] and q["organ"] == ["curiosity"], (arm, q)
        assert "[experience:visual_run:v1]" in sec and "[experience:prediction_surprise:14945]" in sec, arm
        low = sec.lower()
        for leak in (arm.lower(), "propensit", " arm ", "replica", "uncertainty order", "value order"):
            assert leak not in low, (arm, leak)
    assert len(set(sections.values())) == 1, "the now section differs between offer arms"


@pytest.mark.parametrize("module", [curiosity_now_module, now_section_module])
def test_the_now_read_path_names_no_arm_view_or_propensity(module):
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names |= {a.arg for n in ast.walk(tree) if isinstance(n, ast.arguments) for a in (*n.args, *n.kwonlyargs)}
    for bad in ("arm", "draw", "view", "propensities", "value_arm_propensity", "arm_propensities", "plan"):
        assert bad not in names, (module.__name__, bad)


def test_the_loops_now_seams_take_only_the_run_and_its_text():
    assert set(inspect.signature(CuriosityInvestigation._read_now).parameters) == {"self", "run_id"}
    assert set(inspect.signature(CuriosityInvestigation._schedule_now_cites).parameters) == {
        "self", "run_id", "source_tag", "text"}
