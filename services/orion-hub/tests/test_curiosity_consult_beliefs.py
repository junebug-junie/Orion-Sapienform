"""Curiosity x consult outcomes: offered with their source beliefs at kickoff, read back at turn end
(recorded -> stamped, not recorded -> given back), given back when a run never reached its turn.

Lifecycle, not arithmetic: the same CuriosityInvestigation methods the live run calls, with the
consult store functions replaced by an in-memory table and the graph by a fake reader."""

from __future__ import annotations

import asyncio
import importlib
from typing import Any, Dict, List

from test_curiosity_investigation import _FakeBus, _FakeReader, _loop



class Store:
    def __init__(self) -> None:
        self.rows: Dict[str, Dict[str, Any]] = {
            "c1": dict(consult_id="c1", source_kind="curiosity_run", source_ref="curiosity_run:src", status="answered",
                       ending="answered", answer="by design", decision=None, question="why?",
                       orion_told_run_id=None, belief_offers=0, belief_recorded_at=None),
            "c2": dict(consult_id="c2", source_kind="stuck_prior", source_ref="prior:p-stuck", status="brief_sent",
                       ending="agreed_proposal", answer=None, decision="yes", question="fix?",
                       orion_told_run_id=None, belief_offers=0, belief_recorded_at=None),
        }
        self.stamped: List[tuple] = []
        self.released: List[tuple] = []

    def install(self, monkeypatch) -> None:
        # Looked up now, not at import: other Hub tests re-import ``scripts.*`` modules, and the
        # curiosity loop imports this module by name at call time.
        cl = importlib.import_module("scripts.consult_loop")
        for name in ("take_outcomes", "outcomes_offered_to", "stamp_belief", "release_outcomes"):
            monkeypatch.setattr(cl, name, getattr(self, name))

    async def take_outcomes(self, pool, *, run_id, limit=5):
        out = []
        for r in self.rows.values():
            if r["orion_told_run_id"] is None and r["belief_recorded_at"] is None:
                r.update(orion_told_run_id=run_id, belief_offers=r["belief_offers"] + 1)
                out.append(dict(r))
        return out

    async def outcomes_offered_to(self, pool, run_id):
        return [dict(r) for r in self.rows.values() if r["orion_told_run_id"] == run_id and not r["belief_recorded_at"]]

    async def stamp_belief(self, pool, consult_id, *, run_id, finding_id, prior_ids):
        self.stamped.append((consult_id, finding_id, tuple(prior_ids)))
        self.rows[consult_id]["belief_recorded_at"] = "now"

    async def release_outcomes(self, pool, run_id, *, consult_ids=None, refund):
        self.released.append((run_id, tuple(consult_ids) if consult_ids is not None else None, refund))
        for r in self.rows.values():
            if r["orion_told_run_id"] == run_id and not r["belief_recorded_at"] and (
                    consult_ids is None or r["consult_id"] in consult_ids):
                r["orion_told_run_id"] = None
                if refund:
                    r["belief_offers"] -= 1


PRIORS = [
    {"prior_id": "p-src", "claim": "The lease is held forever.", "confidence": 0.6, "status": "open", "times_tested": 1},
    {"prior_id": "p-stuck", "claim": "The meter cannot see.", "confidence": 0.5, "status": "open", "times_tested": 7},
]


def _world(monkeypatch, *, evidence=None, raises=False):
    store = Store()
    store.install(monkeypatch)
    loop = _loop(_FakeBus())
    loop.consult_provider = lambda: object()
    loop._reader = _FakeReader(raises=raises, answers={
        "p.last_run_id AS last_run_id": [{"prior_id": "p-src", "run_id": "src"}, {"prior_id": "p-other", "run_id": "x"}],
        "STARTS WITH 'consult:'": evidence or [],
        "p.claim AS claim": PRIORS,
    })
    return loop, store


class _View:
    is_unavailable = False


def test_kickoff_offers_each_outcome_with_the_beliefs_its_source_names(monkeypatch):
    loop, store = _world(monkeypatch)
    taken = asyncio.run(loop._take_consult_outcomes(_View(), "run-1"))
    by_id = {r["consult_id"]: r for r in taken}
    assert [p.prior_id for p in by_id["c1"]["belief_priors"]] == ["p-src"]
    assert [p.prior_id for p in by_id["c2"]["belief_priors"]] == ["p-stuck"]
    assert {r["belief_offers"] for r in store.rows.values()} == {1}


def test_turn_end_stamps_what_orion_recorded_and_gives_back_the_rest(monkeypatch):
    evidence = [{"finding_id": "consult:c1:answer", "prior_id": "p-src", "edge": "SUPPORTS", "run_id": "run-1"}]
    loop, store = _world(monkeypatch, evidence=evidence)
    asyncio.run(loop._take_consult_outcomes(_View(), "run-1"))
    asyncio.run(loop._stamp_consult_beliefs("run-1"))
    assert store.stamped == [("c1", "consult:c1:answer", ("p-src",))]
    assert store.released == [("run-1", ("c2",), False)], "not recorded: back to a later run, the offer counted"
    assert store.rows["c2"]["orion_told_run_id"] is None and store.rows["c2"]["belief_offers"] == 1


def test_an_unreadable_graph_gives_everything_back_counted(monkeypatch):
    loop, store = _world(monkeypatch)
    asyncio.run(loop._take_consult_outcomes(_View(), "run-1"))
    loop._reader.raises = True
    asyncio.run(loop._stamp_consult_beliefs("run-1"))
    assert store.stamped == [] and store.released == [("run-1", ("c1", "c2"), False)]


def test_a_failed_run_that_never_started_refunds_and_a_started_one_reads_back(monkeypatch):
    loop, store = _world(monkeypatch, evidence=[{"finding_id": "consult:c2:decision", "prior_id": "p-stuck"}])
    asyncio.run(loop._take_consult_outcomes(_View(), "run-1"))

    async def never_started(run_id):
        return False

    loop._run_started = never_started
    asyncio.run(loop._release_offers_after_failed_run("run-1"))
    assert store.released == [("run-1", None, True)]
    assert {r["belief_offers"] for r in store.rows.values()} == {0}

    asyncio.run(loop._take_consult_outcomes(_View(), "run-2"))

    async def unknown(run_id):
        return None

    loop._run_started = unknown
    asyncio.run(loop._release_offers_after_failed_run("run-2"))
    assert store.stamped == [("c2", "consult:c2:decision", ("p-stuck",))]
    assert store.released[-1] == ("run-2", ("c1",), False)


def test_no_outcomes_means_no_graph_read(monkeypatch):
    loop, store = _world(monkeypatch)
    asyncio.run(loop._stamp_consult_beliefs("run-none"))
    assert loop._reader.queries == []
