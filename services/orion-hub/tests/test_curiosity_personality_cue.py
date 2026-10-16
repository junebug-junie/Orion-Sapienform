"""The curiosity kickoff shows the personality read-back as its own block, fetched alongside the
today fact (gathered, not serial) and never at the cost of a failed or delayed kickoff."""

from __future__ import annotations

import ast
import asyncio
import inspect
import textwrap
from datetime import datetime, timedelta, timezone

import httpx

import scripts.curiosity_investigation as ci
from orion.core.bus.bus_schemas import ServiceRef
from orion.curiosity.kickoff_prompt import build_kickoff_prompt
from orion.curiosity.study_material import StudyMaterial
from orion.curiosity.worldview import RecentRun, WorldviewSnapshot
from orion.personality.cue import HEADER
from scripts.curiosity_investigation import CuriosityInvestigation



def frame(nodes: int = 2) -> dict:
    return {
        "generated_at": (NOW - timedelta(minutes=5)).isoformat(),
        "dispositions": [{
            "disposition_id": f"d{i}", "label": f"the walkway camera sees presence without a person {i}",
            "origin": "recurrence", "status": "active", "weight": 10.0 - i, "evidence_count": 7 + i,
            "lane_counts": {"reverie": 4, "dream": 2}, "first_seen_at": "2026-10-16T08:00:00+00:00",
            "last_reinforced_at": "2026-10-20T11:00:00+00:00",
        } for i in range(nodes)],
    }

SOURCE = ServiceRef(name="orion-hub", version="0.1.0", node="athena")
NOW = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)
VIEW = WorldviewSnapshot(recent_runs=[RecentRun(run_id="r1")])
CUE = HEADER + '\n- "the walkway camera sees presence without a person 0" (reverie, dreams; 7 times since Oct 16)'


def _loop(*, enabled=True, url="http://durable.test") -> CuriosityInvestigation:
    return CuriosityInvestigation(
        enabled=True, tick_interval_sec=60.0, min_cooldown_sec=60.0, daily_cap=3,
        timeout_sec=60.0, session_id="s", pool_provider=lambda: None, source_ref=SOURCE,
        reader=object(), durable_runs_url=url, personality_cue_enabled=enabled,
    )


def _material() -> StudyMaterial:
    return StudyMaterial(generated_at=NOW)


def test_kickoff_includes_the_block_after_the_today_line():
    prompt = build_kickoff_prompt(_material(), view=VIEW, run_id="r2",
                                  today_fact="TODAY SO FAR: 1 run.", personality_cue=CUE)
    assert CUE in prompt
    assert prompt.index("TODAY SO FAR: 1 run.") < prompt.index(HEADER)
    assert f"{CUE}\n\n" in prompt  # its own block


def test_kickoff_without_a_cue_is_unchanged():
    base = build_kickoff_prompt(_material(), view=VIEW, run_id="r2", today_fact="TODAY SO FAR: 1 run.")
    for absent in (None, "", "   "):
        assert build_kickoff_prompt(_material(), view=VIEW, run_id="r2",
                                    today_fact="TODAY SO FAR: 1 run.", personality_cue=absent) == base


def test_kickoff_without_a_graph_still_shows_the_block():
    prompt = build_kickoff_prompt(_material(), view=VIEW, run_id="r2", graph_enabled=False, personality_cue=CUE)
    assert CUE in prompt


def test_read_personality_cue_uses_durable_runs_url_and_flag(monkeypatch):
    real = httpx.AsyncClient
    seen = []

    def factory(*args, **kwargs):
        return real(transport=httpx.MockTransport(
            lambda r: (seen.append(str(r.url)), httpx.Response(200, json=frame()))[1]))

    monkeypatch.setattr(ci.httpx, "AsyncClient", factory)
    cue = asyncio.run(_loop()._read_personality_cue(NOW))
    assert seen == ["http://durable.test/personality/frame"] and cue.disposition_ids == ["d0", "d1"]
    off = asyncio.run(_loop(enabled=False)._read_personality_cue(NOW))
    assert off.skip_reason == "disabled" and len(seen) == 1


def test_bare_construction_is_off():
    loop = CuriosityInvestigation(
        enabled=True, tick_interval_sec=60.0, min_cooldown_sec=60.0, daily_cap=3,
        timeout_sec=60.0, session_id="s", pool_provider=lambda: None, source_ref=SOURCE, reader=object(),
    )
    assert loop.personality_cue_enabled is False


def _investigate_tree() -> ast.AST:
    return ast.parse(textwrap.dedent(inspect.getsource(CuriosityInvestigation._investigate)))


def test_investigate_gathers_both_durable_runs_reads():
    """Serial awaits would add the cue's deadline on top of the today fact's: one gather."""
    gathers = [n for n in ast.walk(_investigate_tree()) if isinstance(n, ast.Call)
               and isinstance(n.func, ast.Attribute) and n.func.attr == "gather"]
    reads = set()
    for g in gathers:
        for arg in g.args:
            if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Attribute):
                reads.add(arg.func.attr)
    assert {"_read_today_fact", "_read_personality_cue"} <= reads


def test_investigate_hands_the_cue_text_to_the_kickoff_prompt():
    calls = [n for n in ast.walk(_investigate_tree()) if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) == "build_kickoff_prompt"]
    assert len(calls) == 1
    kw = {k.arg: k.value for k in calls[0].keywords}
    assert "personality_cue" in kw and "personality_cue" in ast.unparse(kw["personality_cue"])


def test_both_reads_together_cost_one_deadline(monkeypatch):
    """Each read sleeps 0.3 s at the service; gathered they finish in about 0.3 s, not 0.6 s."""
    real = httpx.AsyncClient

    async def handler(request):
        await asyncio.sleep(0.3)
        if request.url.path == "/personality/frame":
            return httpx.Response(200, json=frame())
        return httpx.Response(404, json={})

    monkeypatch.setattr(ci.httpx, "AsyncClient",
                        lambda *a, **k: real(transport=httpx.MockTransport(handler)))
    loop = _loop()
    loop.temporal_self_fact_enabled = True

    async def run():
        ev = asyncio.get_running_loop()
        t = ev.time()
        fact, cue = await asyncio.gather(loop._read_today_fact(VIEW, NOW), loop._read_personality_cue(NOW))
        return fact, cue, ev.time() - t

    fact, cue, elapsed = asyncio.run(run())
    assert fact is None and cue.text and elapsed < 0.55
