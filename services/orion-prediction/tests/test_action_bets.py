from __future__ import annotations

import copy
from datetime import timedelta

import pytest

import app.action_bets as ab
from app.action_bets import (BASELINE, MODEL, ActionBets, ConflictError, backoff, beta_eig, grade_bets, live_verdict,
                             model_p_from_counts)
from app.live import MIN_INDEPENDENT
from tests.helpers import T0

SETTLED = ("success", "failure")


class FakeStore:
    """In-memory twin of the PostgresStore action-bet methods (same row shapes)."""

    def __init__(self):
        self.rows: dict[str, dict] = {}
        self.count_calls = 0

    def action_bet_counts(self, domain, skills, places, body=None, combos=False):
        self.count_calls += 1
        agg: dict[tuple, dict] = {}

        def add(key, r, w=1.0):
            c = agg.setdefault(key, {"successes": 0, "failures": 0})
            c["successes" if r["result"] == "success" else "failures"] += w

        per_place: dict[tuple, int] = {}
        for r in self.rows.values():
            if r["result"] in SETTLED and r["place"] is not None:
                per_place[(r["domain"], r["place"])] = per_place.get((r["domain"], r["place"]), 0) + 1
        for r in self.rows.values():
            if r["result"] not in SETTLED:
                continue
            w = 1.0 if r["place"] is None else 1.0 / per_place[(r["domain"], r["place"])]   # one unit per place
            if skills is None or r["skill"] in skills:
                add(("skill", r["skill"], None, None, None), r, w)
            in_scope = (domain is None or r["domain"] == domain) and (body is None or r["body"] == body)
            if in_scope and (skills is None or r["skill"] in skills):
                add(("situation", r["skill"], r["domain"], tuple(sorted(r["context"].items())), None), r, w)
            if in_scope and r["place"] is not None and (places is None or r["place"] in places):
                add(("place", None, r["domain"], None, r["place"]), r)
                if combos:
                    agg.setdefault(("combo", r["skill"], r["domain"], tuple(sorted(r["context"].items())), r["place"]),
                                   {"successes": 0, "failures": 0})
        return [{"level": k[0], "skill": k[1], "domain": k[2], "context": dict(k[3]) if k[3] is not None else None,
                 "place": k[4], **c} for k, c in agg.items()]

    def get_action_bet(self, bet_id):
        r = self.rows.get(bet_id)
        return copy.deepcopy(r) if r else None

    def insert_action_bet(self, row):
        if row["bet_id"] in self.rows:
            return None
        full = {"settled_at": None, "result": None, "outcome": None, "detail": {}, **copy.deepcopy(row)}
        self.rows[row["bet_id"]] = full
        return copy.deepcopy(full)

    def settle_action_bet(self, bet_id, result, outcome, detail, now):
        r = self.rows.get(bet_id)
        if r is None or r["result"] is not None:
            return None
        r.update(result=result, outcome=outcome, detail=detail, settled_at=now)
        return copy.deepcopy(r)

    def expire_action_bets(self, placed_before, now, outcome):
        out = []
        for r in self.rows.values():
            if r["result"] is None and r["placed_at"] < placed_before:
                r.update(result="void", outcome=outcome, settled_at=now)
                out.append(r["bet_id"])
        return out

    def _match(self, r, domain=None, body=None, session_id=None):
        return ((domain is None or r["domain"] == domain) and (body is None or r["body"] == body)
                and (session_id is None or r["session_id"] == session_id))

    def settled_action_bets(self, domain=None, body=None, limit=5000):
        rows = [r for r in self.rows.values() if r["result"] in SETTLED and self._match(r, domain, body)]
        return copy.deepcopy(sorted(rows, key=lambda r: r["settled_at"], reverse=True)[:limit])

    def action_bets(self, domain=None, body=None, session_id=None, limit=20):
        rows = [r for r in self.rows.values() if self._match(r, domain, body, session_id)]
        rows.sort(key=lambda r: (r["placed_at"], r["bet_id"]), reverse=session_id is None)
        return copy.deepcopy(rows[:limit])

    def action_bet_status(self, domain, body, since, never_settled="never_settled"):
        rows = [r for r in self.rows.values() if self._match(r, domain, body)]
        recent = [r for r in rows if r["settled_at"] is not None and r["settled_at"] >= since]
        return {"open": sum(r["result"] is None for r in rows),
                "never_settled": sum(r["result"] == "void" and r["outcome"] == never_settled for r in recent),
                "void": sum(r["result"] == "void" for r in recent),
                "settled": sum(r["result"] in SETTLED for r in recent)}


class Clock:
    def __init__(self):
        self.now = T0

    def __call__(self):
        return self.now


def make(store=None, clock=None):
    clock = clock or Clock()
    return ActionBets(store or FakeStore(), clock, expire_every_sec=0.0, verdict_ttl_sec=0.0), clock


_n = [0]


def bet(svc, *, result=None, domain="retro:zelda", skill="traverse", context=None, place=None, session="s1",
        body="retro", bet_id=None):
    _n[0] += 1
    row, _ = svc.place({"bet_id": bet_id or f"b{_n[0]}", "domain": domain, "body": body, "skill": skill,
                        "context": context if context is not None else {"side": "open_unvisited"},
                        "place": place, "action": {"goal": "explore"}, "session_id": session})
    if result is not None:
        svc.settle(row["bet_id"], result, "completed" if result == "success" else "x", {})
    return row


# --- the pure model -------------------------------------------------------------------------------
def test_eig_matches_the_hand_checked_values():
    assert beta_eig(1, 1) == pytest.approx(0.1931, abs=1e-4)
    assert beta_eig(11, 11) == pytest.approx(0.0222, abs=1e-4)
    assert beta_eig(21, 3) == pytest.approx(0.020, abs=1e-3)


def test_no_data_at_all_is_the_prior_at_one_half():
    q = backoff(None, None, None, has_place=True)
    assert q["level"] == "prior" and q["p"] == 0.5 and q["p_baseline"] == 0.5
    assert q["interval80"] == pytest.approx((0.1, 0.9))
    assert q["eig_nats"] == pytest.approx(0.1931, abs=1e-4)
    assert [c["level"] for c in q["counts"]] == ["skill", "situation", "place"]


def test_no_place_data_equals_the_situation_mean():
    sk, sit = {"successes": 9, "failures": 5}, {"successes": 2, "failures": 0}
    with_place = backoff(sk, sit, None, has_place=True)
    without = backoff(sk, sit, None, has_place=False)
    assert with_place["level"] == "situation" and with_place["p"] == pytest.approx(without["p"])
    p_skill = 10 / 16
    assert without["p_baseline"] == pytest.approx(p_skill)
    assert without["p"] == pytest.approx((4 * p_skill + 2) / (4 + 2))


def test_place_shrinks_toward_the_situation_mean():
    sk, sit, pl = {"successes": 9, "failures": 5}, {"successes": 2, "failures": 0}, {"successes": 0, "failures": 5}
    q = backoff(sk, sit, pl, has_place=True)
    p_sit = backoff(sk, sit, None, has_place=False)["p"]
    assert q["level"] == "place" and q["p"] == pytest.approx(4 * p_sit / (4 + 5))
    assert q["p"] < p_sit and q["interval80"][0] < q["p"] < q["interval80"][1]
    # K = 4: 5 failures read p < 0.2 only when the situation prior is below ~0.45 (here 0.75 -> 0.33).


def test_model_p_is_recomputable_from_stored_counts():
    q = backoff({"successes": 3, "failures": 1}, {"successes": 1, "failures": 1}, {"successes": 0, "failures": 2},
                has_place=True)
    assert model_p_from_counts(q["counts"]) == pytest.approx(q["p"])


# --- place / settle / expire ----------------------------------------------------------------------
def test_place_computes_p_server_side_and_a_repeat_is_idempotent():
    svc, _ = make()
    for _ in range(3):
        bet(svc, result="success")
    bet(svc, result="failure")
    row, created = svc.place({"bet_id": "x", "domain": "retro:zelda", "body": "retro", "skill": "traverse",
                              "context": {"side": "open_unvisited"}, "action": {}, "p": 0.99})
    assert created and row["model"] == MODEL and row["level"] == "situation"
    assert row["p_baseline"] == pytest.approx(4 / 6) and row["p"] == pytest.approx((4 * 4 / 6 + 3) / 8)
    bet(svc, result="failure")                                  # the world moves on ...
    again, created2 = svc.place({"bet_id": "x", "domain": "retro:zelda", "body": "retro", "skill": "traverse",
                                 "context": {"side": "open_unvisited"}, "action": {}})
    assert not created2 and again == row                       # ... the stored bet does not


def test_settle_open_to_settled_only_and_conflict():
    svc, _ = make()
    b = bet(svc)
    s = svc.settle(b["bet_id"], "success", "completed", {"room": 119})
    assert s["result"] == "success" and s["detail"] == {"room": 119}
    assert svc.settle(b["bet_id"], "success", "completed", {})["result"] == "success"   # same result: 200
    with pytest.raises(ConflictError):
        svc.settle(b["bet_id"], "failure", "death", {})
    assert svc.settle("nope", "success", "completed", {}) is None


def test_bets_open_over_an_hour_expire_void_never_settled():
    svc, clock = make()
    old = bet(svc)
    clock.now = T0 + timedelta(minutes=30)
    young = bet(svc)
    clock.now = T0 + timedelta(minutes=61)
    assert svc.expire() == [old["bet_id"]]
    got = svc.store.get_action_bet(old["bet_id"])
    assert got["result"] == "void" and got["outcome"] == "never_settled"
    assert svc.store.get_action_bet(young["bet_id"])["result"] is None
    with pytest.raises(ConflictError):                          # a late settle cannot revive it
        svc.settle(old["bet_id"], "success", "completed", {})
    h = svc.health()
    assert h["open"] == 1 and h["never_settled_24h"] == 1 and h["served_model"] == MODEL


def test_quote_runs_expiry_cheaply_and_uses_one_count_query():
    svc, clock = make()
    b = bet(svc)
    clock.now = T0 + timedelta(hours=2)
    calls = svc.store.count_calls
    out = svc.quote("retro:zelda", [{"option_id": str(i), "skill": "traverse", "context": {}, "place": f"p{i}"}
                                    for i in range(20)])
    assert svc.store.count_calls == calls + 1 and len(out["quotes"]) == 20
    assert svc.store.get_action_bet(b["bet_id"])["result"] == "void"


# --- counts semantics -----------------------------------------------------------------------------
def test_void_is_excluded_from_counts():
    svc, _ = make()
    bet(svc, result="void")
    bet(svc, result="void")
    q = svc.quote("retro:zelda", [{"option_id": "a", "skill": "traverse", "context": {"side": "open_unvisited"}}])
    assert q["quotes"][0]["level"] == "prior" and q["quotes"][0]["p"] == 0.5


def test_skill_pools_across_domains_but_situation_and_place_do_not():
    svc, _ = make()
    ctx = {"side": "open_unvisited"}
    for _ in range(4):
        bet(svc, result="failure", domain="aitown", body="aitown", context=ctx, place="P")
    q = svc.quote("retro:zelda", [{"option_id": "a", "skill": "traverse", "context": ctx, "place": "P"}])["quotes"][0]
    counts = {c["level"]: (c["successes"], c["failures"]) for c in q["counts"]}
    # Four bets at one place count as ONE unit above the place level.
    assert counts == {"skill": (0, 1), "situation": (0, 0), "place": (0, 0)}
    assert q["level"] == "skill" and q["p"] == pytest.approx(1 / 3) and q["p_baseline"] == pytest.approx(1 / 3)


def test_situation_needs_the_exact_context():
    svc, _ = make()
    bet(svc, result="failure", context={"side": "closed", "threat": "none"})
    q = svc.quote("retro:zelda", [{"option_id": "a", "skill": "traverse", "context": {"side": "closed"}},
                                  {"option_id": "b", "skill": "traverse", "context": {"threat": "none", "side": "closed"}}])
    a, b = q["quotes"]
    assert a["level"] == "skill" and b["level"] == "situation"


# --- grading and the gate -------------------------------------------------------------------------
def test_grading_uses_grading_grade(monkeypatch):
    seen = []

    def fake_grade(probs, hits):
        seen.append((list(probs), list(hits)))
        return {"n": len(probs)}
    monkeypatch.setattr(ab, "grade", fake_grade)
    rows = [{"p": 0.8, "p_baseline": 0.6, "counts": [], "result": "success"},
            {"p": 0.3, "p_baseline": 0.6, "counts": [], "result": "failure"},
            {"p": 0.3, "p_baseline": 0.6, "counts": [], "result": "void"}]
    g = grade_bets(rows)
    assert seen[0] == ([0.8, 0.3], [True, False]) and seen[1] == ([0.6, 0.6], [True, False])
    assert g["served"] == {"n": 2} and g["reportable"] is False


def _verdict_rows(n_sessions, model_counts, p_baseline, result):
    return [{"bet_id": f"v{i}", "session_id": f"s{i}", "placed_at": T0 + timedelta(minutes=i),
             "p_baseline": p_baseline + 0.01 * (i % 3), "counts": model_counts, "result": result(i)}
            for i in range(n_sessions)]


SURE_YES = [{"level": "skill", "successes": 0, "failures": 0}, {"level": "situation", "successes": 40, "failures": 0}]


def test_live_verdict_one_pair_per_session_and_pending_below_twenty():
    rows = _verdict_rows(MIN_INDEPENDENT - 1, SURE_YES, 0.5, lambda i: "failure")
    rows += [dict(rows[0], bet_id="dup")]                       # same session: still one unit
    v = live_verdict(rows)
    assert v["n_independent"] == MIN_INDEPENDENT - 1 and v["n_scored"] == MIN_INDEPENDENT
    assert v["verdict"] == "pending"


def test_live_verdict_better_and_worse():
    worse = live_verdict(_verdict_rows(30, SURE_YES, 0.5, lambda i: "success" if i % 10 == 0 else "failure"))
    better = live_verdict(_verdict_rows(30, SURE_YES, 0.5, lambda i: "failure" if i % 10 == 0 else "success"))
    assert worse["verdict"] == "worse" and better["verdict"] == "better"


def test_the_gate_serves_the_baseline_when_the_model_is_live_worse():
    svc, clock = make()
    store = svc.store
    for r in _verdict_rows(30, SURE_YES, 0.5, lambda i: "success" if i % 10 == 0 else "failure"):
        store.rows[r["bet_id"]] = {**r, "domain": "retro:zelda", "body": "retro", "skill": "engage", "context": {},
                                   "place": None, "action": {}, "model": MODEL, "p": 0.97, "level": "situation",
                                   "chosen_rank": None, "settled_at": r["placed_at"], "outcome": "x", "detail": {}}
    assert svc.verdict()["verdict"] == "worse" and svc.served_model() == BASELINE
    for i in range(3):
        bet(svc, result="success", skill="traverse")
    q = svc.quote("retro:zelda", [{"option_id": "a", "skill": "traverse", "context": {"side": "open_unvisited"}}])
    o = q["quotes"][0]
    assert q["served_model"] == BASELINE and o["model"] == BASELINE and o["p"] == o["p_baseline"] == pytest.approx(0.8)
    row, _ = svc.place({"bet_id": "g", "domain": "retro:zelda", "body": "retro", "skill": "traverse",
                        "context": {"side": "open_unvisited"}, "action": {}})
    assert row["model"] == BASELINE and row["p"] == row["p_baseline"]
    assert model_p_from_counts(row["counts"]) != pytest.approx(row["p"])   # the shadow stays recomputable
    assert svc.health()["served_model"] == BASELINE


def test_live_view_lists_skills_situations_places_and_grading():
    svc, _ = make()
    for i in range(5):
        bet(svc, result="failure", place="zelda:0:119:S", context={"side": "closed"}, session=f"s{i}")
    bet(svc, result="success", place="zelda:0:119:W", context={"side": "open_unvisited"})
    bet(svc)                                                     # open
    out = svc.live("retro:zelda", None)
    assert [s["skill"] for s in out["skills"]] == ["traverse"]
    assert {(s["context"].get("side"), s["successes"], s["failures"]) for s in out["situations"]} == {
        ("closed", 0, 1), ("open_unvisited", 1, 0)}                 # 5 bets at one place = one unit
    south = next(p for p in out["places"] if p["place"] == "zelda:0:119:S")
    west = next(p for p in out["places"] if p["place"] == "zelda:0:119:W")
    assert south["mean"] < 0.2 < west["mean"]
    assert out["grading"]["served"]["n"] == 6 and out["grading"]["baseline"]["n"] == 6
    assert out["open"] == 1 and len(out["last_bets"]) == 7 and out["live_verdict"]["verdict"] == "pending"


# --- HTTP -------------------------------------------------------------------------------------------
@pytest.fixture
def client(monkeypatch):
    from fastapi.testclient import TestClient

    import app.main as main
    svc, clock = make()
    monkeypatch.setattr(main, "action_bets", svc)
    return TestClient(main.app), svc, clock      # no context manager: lifespan (bus, DB) does not run


def test_http_quote_place_settle_list(client):
    c, svc, clock = client
    opts = [{"option_id": "w", "skill": "traverse", "context": {"side": "open_unvisited"}, "place": "zelda:0:119:W"}]
    r = c.post("/action-bets/quote", json={"domain": "retro:zelda", "body": "retro", "options": opts})
    assert r.status_code == 200
    q = r.json()["quotes"][0]
    assert q["option_id"] == "w" and q["level"] == "prior" and q["model"] == MODEL and len(q["counts"]) == 3
    too_many = [dict(opts[0], option_id=str(i)) for i in range(21)]
    assert c.post("/action-bets/quote", json={"domain": "d", "body": "b", "options": too_many}).status_code == 422
    place = {"bet_id": "retro:zelda:s1:15:9", "domain": "retro:zelda", "body": "retro", "skill": "traverse",
             "context": {"side": "open_unvisited"}, "place": "zelda:0:119:W", "action": {"goal": "explore", "dir": "W"},
             "chosen_rank": 1, "session_id": "s1"}
    r1 = c.post("/action-bets", json=place)
    assert r1.status_code == 200 and r1.json()["p"] == 0.5 and r1.json()["result"] is None
    assert c.post("/action-bets", json={**place, "p": 0.9}).status_code == 422      # caller cannot send p
    assert c.post("/action-bets", json=place).json() == r1.json()
    s = c.post("/action-bets/retro:zelda:s1:15:9/settle", json={"result": "success", "outcome": "completed"})
    assert s.status_code == 200 and s.json()["result"] == "success"
    assert c.post("/action-bets/retro:zelda:s1:15:9/settle",
                  json={"result": "success", "outcome": "completed"}).status_code == 200
    conflict = c.post("/action-bets/retro:zelda:s1:15:9/settle", json={"result": "failure", "outcome": "death"})
    assert conflict.status_code == 409 and conflict.json()["detail"]["bet"]["result"] == "success"
    assert c.post("/action-bets/nope/settle", json={"result": "void", "outcome": "stopped:human"}).status_code == 404
    lst = c.get("/action-bets", params={"session_id": "s1"}).json()
    assert [b["bet_id"] for b in lst["bets"]] == ["retro:zelda:s1:15:9"]
    live = c.get("/action-bets/live", params={"domain": "retro:zelda"}).json()
    assert live["grading"]["served"]["n"] == 1 and live["last_bets"][0]["bet_id"] == "retro:zelda:s1:15:9"
    assert live["places"][0]["place"] == "zelda:0:119:W"


def test_one_stubborn_place_does_not_sink_the_whole_skill():
    # Live 2026-10-23: seven failed walks into room 119's south wall made untried exits read 0.20.
    svc, _ = make()
    for i in range(7):
        bet(svc, result="failure", place="zelda:0:119:S", context={"side": "unknown"}, session=f"s{i}")
    bet(svc, result="success", place="zelda:0:103:S", context={"side": "unknown"})
    q = svc.quote("retro:zelda", [{"option_id": "e", "skill": "traverse", "context": {"side": "unknown"},
                                   "place": "zelda:0:119:E"},
                                  {"option_id": "s", "skill": "traverse", "context": {"side": "unknown"},
                                   "place": "zelda:0:119:S"}])["quotes"]
    east, south = q
    assert east["level"] == "situation" and east["p"] == pytest.approx(0.5)   # 1 of 2 places, not 1 of 8
    assert south["level"] == "place" and south["p"] < 0.2


def test_backoff_p_matches_backoff_on_many_count_shapes():
    import itertools
    from app.action_bets import backoff, backoff_p
    vals = [None, {"successes": 0, "failures": 0}, {"successes": 1.5, "failures": 0.5}, {"successes": 0, "failures": 7}]
    for sk, si, pl in itertools.product(vals, vals, vals):
        for has_place in (True, False):
            assert backoff_p(sk, si, pl, has_place=has_place) == pytest.approx(
                backoff(sk, si, pl, has_place=has_place)["p"])



def test_bet_rows_drop_columns_this_code_does_not_know():
    # A later migration adds a column before the service rebuilds; ActionBetV1 forbids extras.
    from app.store import ACTION_BET_COLS, PostgresStore
    from orion.schemas.prediction import ActionBetV1
    row = {c: None for c in ACTION_BET_COLS}
    row.update(bet_id="b1", domain="retro:zelda", body="retro", skill="traverse", context={}, action={},
               model=MODEL, p=0.5, p_baseline=0.5, level="prior", counts=[], placed_at=T0, future_column={"x": 1})
    out = PostgresStore._bet(row)
    assert "future_column" not in out
    ActionBetV1.model_validate(out)
