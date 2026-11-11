"""Three-arm readout of the curiosity replica test (design docs/superpowers/specs/2026-11-04-curiosity-replica-design.md,
patch 5: sections 5, 6 and 8). Read-only.

Curiosity orders its offer three ways, drawn at random per run: by uncertainty (the old rule), by value (the second
rule), or by the replica's plan. This answers, from the live tables:

* Per arm: runs drawn, runs that finished, how many moved at least one belief, mean and median realized nats.
* The pre-registered test: a one-sided permutation test of the replica mean against EACH rule arm (10,000
  permutations). The replica wins only if both p-values are below 0.05. Requiring both is an intersection-union
  test, so no multiple-test correction applies. The test is read ONCE, at the stop: every arm has 250 completed
  runs, or 180 days have passed since the start, whichever comes first. Only runs drawn up to that fixed moment are
  in the test. Before the stop the readout shows counts only and says "no verdict"; the per-arm outcome numbers
  are hidden (``--unblind`` shows them, for checking the pipe, never for deciding).
* The harm look (every 30 days): is the replica arm's failed-turn rate or "tested nothing" rate worse than the
  pooled rule arms at p < 0.01 (one-sided Fisher exact test)? Read on the runs drawn before the latest 30-day mark,
  so it changes only at a look. A run with no outcome a day after its draw counts as failed. If so, pause the arm.
* Fallbacks over the last 14 days, by reason. If more than 30% of replica-arm runs fall back for ``unfunded``,
  that goes back to Juniper (the arm would lose by starvation, not by judgment).
* The replica's answers to Prime, from curiosity_replica_answers_readout.py (the 14-day refusal-rate alarm).
* E9, Juniper first (design acceptance checks): is Juniper's chat slower in the 14 days after the replica started
  than in the 14 days before? Two numbers per turn: the whole turn (her message in to the reply out), which also
  moves with what she asked, and the time her model calls waited for a seat on the chat lane, which is what the
  replica and Prime can cause. Only the seat wait raises the flag (one-sided permutation, p < 0.05). Juniper's turns
  are the ones orion/regulation/juniper_turns.py says are hers. Seat wait is recorded from 2026-10-29 only.
* The replica family's learned weight in the body economy (want_ledger), first and last value and its basis, so
  the 14-day review can see whether it drifted from its starting value. No pre-registered threshold: no flag.

Intention-to-treat: every replica-arm run counts as replica, whether its plan was used or it fell back. The
"as treated" split (plan used vs fell back) is shown for reading only, never for the decision.

The test starts at the first offer row whose recorded replica propensity is above 0. Rows from before that
(including 2026-11-04, when the replica arm existed with propensity 0) are a different randomization and are left
out (design section 5: compare only runs randomized at the same time).

    python scripts/analysis/curiosity_three_arm_readout.py
    python scripts/analysis/curiosity_three_arm_readout.py --json --days 14

POSTGRES_URI picks the database (from this host: postgresql://postgres:postgres@localhost:55432/conjourney).
Exit code 0, or 3 when something must go back to Juniper (harm pause, unfunded over 30%, refusal alarm, or the
answers could not be read), or 2 when Postgres could not be read.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pg_readonly import open_readonly_connection  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import curiosity_replica_answers_readout as answers_readout  # noqa: E402
from orion.curiosity.value import ARM_REPLICA_ORDER, ARM_UNCERTAINTY_ORDER, ARM_VALUE_ORDER  # noqa: E402
from orion.regulation.juniper_turns import JUNIPER_INGRESS_PREDICATE  # noqa: E402

DEFAULT_POSTGRES_URI = "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney"

# Pre-registered (design section 6). Changing any of these after the start invalidates the test.
STOP_RUNS_PER_ARM = 250
STOP_DAYS = 180
WIN_ALPHA = 0.05
PERMUTATIONS = 10_000
HARM_ALPHA = 0.01
HARM_LOOK_DAYS = 30
# Design section 8: more than this share of replica-arm runs unfunded -> back to Juniper.
UNFUNDED_ALARM_SHARE = 0.30
# E9: Juniper's chat is "slower" when her seat wait after the cut beats before at this level (one-sided).
E9_ALPHA = 0.05
E9_DAYS = 14
REPLICA_FAMILY = "curiosity.replica"
# Prime and the replica both call the model on this route (CURIOSITY_REPLICA_LLM_ROUTE=prime).
PRIME_ROUTE = "prime"
NOT_READ = object()  # build_report: this section was not asked for (distinct from None = could not be read)
RULE_ARMS = (ARM_VALUE_ORDER, ARM_UNCERTAINTY_ORDER)
ARMS = (ARM_REPLICA_ORDER, *RULE_ARMS)
NO_PLAN_ROW = "no_plan_row"


# --- pure ---------------------------------------------------------------------------


def replica_propensity(row: dict[str, Any]) -> float:
    props = row.get("arm_propensities")
    try:
        if isinstance(props, str):
            props = json.loads(props)
        return float((props or {}).get(ARM_REPLICA_ORDER) or 0.0)
    except (TypeError, ValueError, AttributeError):
        return 0.0


def three_arm_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Offer rows drawn while the replica arm was live (its recorded propensity above 0), oldest first."""
    live = [r for r in rows if r.get("arm") in ARMS and replica_propensity(r) > 0.0]
    return sorted(live, key=lambda r: r["decided_at"])


def completed(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Runs whose turn finished and produced text (turn_ok true): the primary outcome's population."""
    return [r for r in rows if r.get("turn_ok") is True]


def arm_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    done = completed(rows)
    nats = [float(r["realized_nats"]) for r in done if r.get("realized_nats") is not None]
    moved = sum(1 for r in done if (r.get("n_moved") or 0) > 0)
    return {
        "drawn": len(rows),
        "unfinished": sum(1 for r in rows if r.get("turn_ok") is None),
        "failed": sum(1 for r in rows if r.get("turn_ok") is False),
        "completed": len(done),
        "nats_unknown": len(done) - len(nats),
        "moved_a_belief": moved,
        "moved_share": moved / len(done) if done else None,
        "tested_nothing": sum(1 for r in done if (r.get("n_tested") or 0) == 0),
        "mean_nats": mean(nats) if nats else None,
        "median_nats": median(nats) if nats else None,
    }


def by_arm(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {arm: [r for r in rows if r.get("arm") == arm] for arm in ARMS}


def permutation_p_greater(a: Sequence[float], b: Sequence[float], *, permutations: int = PERMUTATIONS,
                          seed: int = 0) -> Optional[float]:
    """One-sided p that mean(a) > mean(b) by chance: shuffle the arm labels, count shuffles at least as extreme.
    No normal approximation (the data is mostly zeros with a long tail). The +1s keep p above 0."""
    if len(a) < 2 or len(b) < 2:
        return None
    pooled = list(a) + list(b)
    n_a, total = len(a), sum(pooled)
    observed = sum(a) / n_a - (total - sum(a)) / len(b)
    rng = random.Random(seed)
    hits = 0
    for _ in range(permutations):
        rng.shuffle(pooled)
        s = sum(pooled[:n_a])
        if s / n_a - (total - s) / len(b) >= observed - 1e-12:
            hits += 1
    return (hits + 1) / (permutations + 1)


def fisher_p_greater(a: int, b: int, c: int, d: int) -> float:
    """One-sided Fisher exact p that the first row's rate a/(a+b) exceeds the second's c/(c+d)."""
    row1, col1, n = a + b, a + c, a + b + c + d
    denom = math.comb(n, col1)
    return min(1.0, sum(math.comb(row1, x) * math.comb(n - row1, col1 - x)
                        for x in range(a, min(row1, col1) + 1)) / denom)


def stop_at(rows: Sequence[dict[str, Any]]) -> Optional[datetime]:
    """The one fixed moment the test stops: 180 days after the start, or the draw time of the run that gave the
    last arm its 250th completed run, whichever is earlier. Fixed so the verdict never depends on the read date."""
    if not rows:
        return None
    limit = rows[0]["decided_at"] + timedelta(days=STOP_DAYS)
    counts = {a: 0 for a in ARMS}
    for r in rows:  # oldest first
        if r["decided_at"] > limit:
            break
        if r.get("turn_ok") is True:
            counts[r["arm"]] += 1
            if all(n >= STOP_RUNS_PER_ARM for n in counts.values()):
                return r["decided_at"]
    return limit


def stop_status(rows: Sequence[dict[str, Any]], *, now: datetime) -> dict[str, Any]:
    if not rows:
        return {"start": None, "days": 0.0, "stop_at": None, "stopped": False, "reason": None,
                "completed_per_arm": {a: 0 for a in ARMS}}
    start, at = rows[0]["decided_at"], stop_at(rows)
    full = at < start + timedelta(days=STOP_DAYS)
    stopped = full or now >= at
    in_test = [r for r in rows if r["decided_at"] <= at] if stopped else list(rows)
    return {"start": start, "days": (now - start).total_seconds() / 86400.0, "stop_at": at, "stopped": stopped,
            "reason": (f"every arm has {STOP_RUNS_PER_ARM} completed runs" if full
                       else f"{STOP_DAYS} days have passed" if stopped else None),
            "completed_per_arm": {a: len(completed(rs)) for a, rs in by_arm(in_test).items()}}


def primary_test(rows: Sequence[dict[str, Any]], *, now: datetime,
                 permutations: int = PERMUTATIONS) -> dict[str, Any]:
    """The pre-registered decision. Not read before the stop, whatever the interim numbers look like. Only runs
    drawn up to the fixed stop moment are in it; runs drawn after are never counted."""
    stop = stop_status(rows, now=now)
    if not stop["stopped"]:
        return {"stop": stop, "read": False, "p": None, "replica_wins": None,
                "verdict": "RUNNING: no verdict before the stop rule is met"}
    arms = by_arm([r for r in rows if r["decided_at"] <= stop["stop_at"]])
    nats = {a: [float(r["realized_nats"]) for r in completed(rs) if r.get("realized_nats") is not None]
            for a, rs in arms.items()}
    p = {rule: permutation_p_greater(nats[ARM_REPLICA_ORDER], nats[rule], permutations=permutations)
         for rule in RULE_ARMS}
    if any(v is None for v in p.values()):
        return {"stop": stop, "read": True, "p": p, "replica_wins": None,
                "verdict": "INCONCLUSIVE: an arm has fewer than 2 scored runs at the stop -- bring it to Juniper"}
    wins = all(v < WIN_ALPHA for v in p.values())
    return {"stop": stop, "read": True, "p": p, "replica_wins": wins,
            "verdict": ("REPLICA WINS: it beat both rule arms (both p < 0.05)" if wins else
                        "REPLICA LOSES: it did not beat both rule arms -- retire it fully (design section 6)")}


def harm_check(rows: Sequence[dict[str, Any]], *, now: datetime) -> dict[str, Any]:
    """Replica arm vs the pooled rule arms, on failed turns and on completed turns that tested nothing. Read only
    at the scheduled looks (every 30 days from the start), on the runs drawn before the latest look, so reading the
    script daily does not turn one look into thirty. A run with no outcome a day after its draw counts as failed."""
    start = rows[0]["decided_at"] if rows else None
    if start is None:
        return {"look": None, "failed_turns": None, "tested_nothing": None, "pause": False,
                "next_scheduled_look": None, "verdict": "NO DATA"}
    looks_done = int((now - start).total_seconds() // (HARM_LOOK_DAYS * 86400))
    next_look = start + timedelta(days=HARM_LOOK_DAYS * (looks_done + 1))
    if looks_done == 0:
        return {"look": None, "failed_turns": None, "tested_nothing": None, "pause": False,
                "next_scheduled_look": next_look, "verdict": "no look yet"}
    look = start + timedelta(days=HARM_LOOK_DAYS * looks_done)
    arms = by_arm([r for r in rows if r["decided_at"] < look])
    rep = arms[ARM_REPLICA_ORDER]
    rules = [r for a in RULE_ARMS for r in arms[a]]

    def compare(bad, population) -> dict[str, Any]:
        rp, up = population(rep), population(rules)
        a, c = sum(1 for r in rp if bad(r)), sum(1 for r in up if bad(r))
        p = fisher_p_greater(a, len(rp) - a, c, len(up) - c) if rp and up else None
        return {"replica": f"{a}/{len(rp)}", "rules": f"{c}/{len(up)}", "p": p,
                "harm": p is not None and p < HARM_ALPHA}

    lost = now - timedelta(days=1)
    is_failed = lambda r: r.get("turn_ok") is False or (r.get("turn_ok") is None and r["decided_at"] < lost)  # noqa: E731
    finished = lambda rs: [r for r in rs if r.get("turn_ok") is not None or r["decided_at"] < lost]  # noqa: E731
    failed = compare(is_failed, finished)
    nothing = compare(lambda r: (r.get("n_tested") or 0) == 0, completed)
    pause = failed["harm"] or nothing["harm"]
    return {"look": look, "failed_turns": failed, "tested_nothing": nothing, "pause": pause,
            "next_scheduled_look": next_look,
            "verdict": "PAUSE the replica arm and bring it to Juniper" if pause else "no harm at p < 0.01"}


def fallback_report(rows: Sequence[dict[str, Any]], *, now: datetime, days: int = 14) -> dict[str, Any]:
    """Replica-arm runs drawn in the window, by what happened to their plan. A run with no plan row is
    counted apart: it fell back, but nothing recorded why."""
    since = now - timedelta(days=days)
    rep = [r for r in rows if r.get("arm") == ARM_REPLICA_ORDER and r["decided_at"] >= since]
    by_status: dict[str, int] = {}
    for r in rep:
        status = r.get("plan_status") or NO_PLAN_ROW
        by_status[status] = by_status.get(status, 0) + 1
    fell_back = sum(n for s, n in by_status.items() if s != "ok")
    unfunded = by_status.get("unfunded", 0)
    share = unfunded / len(rep) if rep else None
    alarm = share is not None and share > UNFUNDED_ALARM_SHARE
    return {"days": days, "replica_runs": len(rep), "by_status": dict(sorted(by_status.items())),
            "fell_back": fell_back, "fallback_share": fell_back / len(rep) if rep else None,
            "unfunded_share": share, "alarm": alarm,
            "verdict": (f"ALARM: {unfunded} of {len(rep)} replica-arm runs were unfunded (over "
                        f"{UNFUNDED_ALARM_SHARE:.0%}) -- bring the number to Juniper" if alarm else
                        "ok" if rep else "NO DATA: no replica-arm runs in the window")}


def as_treated(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Secondary, never used for the decision: replica-arm runs whose plan was used vs those that fell back."""
    rep = [r for r in rows if r.get("arm") == ARM_REPLICA_ORDER]
    return {"plan_used": arm_summary([r for r in rep if r.get("plan_status") == "ok"]),
            "fell_back": arm_summary([r for r in rep if r.get("plan_status") != "ok"])}


def _quantile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))]


def _e9_window(turns: Sequence[dict[str, Any]]) -> dict[str, Any]:
    turn_s = [(t["tf"] - t["t0"]).total_seconds() for t in turns if t.get("tf") is not None and t["tf"] >= t["t0"]]
    # A turn with no usage rows at all was not measured (before 2026-10-29, or the gateway missed it): not a 0 wait.
    measured = [t for t in turns if (t.get("calls") or 0) > 0]
    wait_s = [(t.get("chat_wait_ms") or 0) / 1000.0 for t in measured]
    return {"turns": len(turns), "finished": len(turn_s), "turn_s": turn_s, "wait_s": wait_s,
            "median_turn_s": _quantile(turn_s, 0.5), "p90_turn_s": _quantile(turn_s, 0.9),
            "wait_measured": len(measured), "median_wait_s": _quantile(wait_s, 0.5), "p90_wait_s": _quantile(wait_s, 0.9),
            "mean_wait_s": mean(wait_s) if wait_s else None}


def e9_report(turns: Sequence[dict[str, Any]], lane: Mapping[str, Mapping[str, Any]], *, cut: datetime,
              now: datetime, days: int = E9_DAYS, permutations: int = PERMUTATIONS) -> dict[str, Any]:
    """Juniper's turns in the ``days`` before ``cut`` against the ``days`` after (up to now). ``lane`` holds, per
    window, the chat lane's total model time and the part Prime and the replica used (route prime)."""
    lo, hi = cut - timedelta(days=days), min(cut + timedelta(days=days), now)
    before = _e9_window([t for t in turns if lo <= t["t0"] < cut])
    after = _e9_window([t for t in turns if cut <= t["t0"] < hi])
    p_turn = permutation_p_greater([math.log(max(x, 1e-3)) for x in after["turn_s"]],
                                   [math.log(max(x, 1e-3)) for x in before["turn_s"]], permutations=permutations)
    p_wait = permutation_p_greater(after["wait_s"], before["wait_s"], permutations=permutations)
    slower = p_wait is not None and p_wait < E9_ALPHA

    def share(w: str) -> Optional[float]:
        got = lane.get(w) or {}
        return (got.get("prime_ms") or 0) / got["chat_ms"] if got.get("chat_ms") else None

    for w in (before, after):
        w.pop("turn_s"), w.pop("wait_s")
    return {"cut": cut, "days_before": days, "days_after": round((hi - cut).total_seconds() / 86400.0, 1),
            "before": before, "after": after, "p_turn_slower": p_turn, "p_wait_slower": p_wait, "slower": slower,
            "prime_share_of_chat_lane": {"before": share("before"), "after": share("after")},
            "verdict": ("SLOWER: Juniper's calls waited longer for a chat-lane seat after the cut (p < 0.05) -- "
                        "bring it to Juniper" if slower else
                        "UNKNOWN: too few measured turns in a window" if p_wait is None else
                        "no seat-wait slowdown at p < 0.05")}


def weight_report(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The replica family's weight as the market used it, decision by decision (want_ledger), oldest first."""
    known = [r for r in rows if r.get("family_weight") is not None]
    if not known:
        return {"decisions": 0, "verdict": "NO DATA: the replica family has no market decisions"}
    first, last = known[0], known[-1]
    weights = [float(r["family_weight"]) for r in known]
    learned = sum(1 for r in known if r.get("weight_basis") not in (None, "prior"))
    return {"decisions": len(known), "first": weights[0], "first_at": first["decided_at"], "last": weights[-1],
            "last_at": last["decided_at"], "min": min(weights), "max": max(weights), "basis_now": last.get("weight_basis"),
            "learned_decisions": learned, "change": weights[-1] - weights[0],
            "verdict": ("still on its starting value (basis prior: fewer than 10 graded acts)"
                        if last.get("weight_basis") == "prior" else
                        f"learned (basis {last.get('weight_basis')}): moved {weights[-1] - weights[0]:+.3f} since the first decision")}


BLINDED = ("moved_a_belief", "moved_share", "mean_nats", "median_nats")


def blind(summary: dict[str, Any]) -> dict[str, Any]:
    """The outcome numbers hidden: before the stop the readout shows counts only (design section 6)."""
    return {k: v for k, v in summary.items() if k not in BLINDED}


def build_report(rows: Sequence[dict[str, Any]], answers: Optional[Sequence[dict[str, Any]]], *, now: datetime,
                 days: int = 14, permutations: int = PERMUTATIONS, unblind: bool = False,
                 e9: Any = NOT_READ, weight: Any = NOT_READ) -> dict[str, Any]:
    """``e9`` and ``weight`` are the already-built sections: NOT_READ leaves them out, None means they could not be
    read (which goes to Juniper, like unreadable answers)."""
    live = three_arm_rows(rows)
    primary = primary_test(live, now=now, permutations=permutations)
    show = unblind or primary["stop"]["stopped"]
    shown = (lambda x: x) if show else blind
    report = {
        "now": now,
        "blinded": not show,
        "arms": {a: shown(arm_summary(rs)) for a, rs in by_arm(live).items()},
        "primary": primary,
        "harm": harm_check(live, now=now),
        "fallbacks": fallback_report(live, now=now, days=days),
        "as_treated": {k: shown(v) for k, v in as_treated(live).items()},
        "answers": answers_readout.summarize(answers, today=now) if answers is not None else None,
    }
    for key, section in (("e9", e9), ("weight", weight)):
        if section is not NOT_READ:
            report[key] = section
    report["needs_juniper"] = [k for k, flag in (
        ("harm", report["harm"]["pause"]),
        ("unfunded", report["fallbacks"]["alarm"]),
        ("refusals", bool(report["answers"] and report["answers"]["alarm"])),
        ("answers_unknown", report["answers"] is None),
        ("chat_slower", bool(report.get("e9") and report["e9"]["slower"])),
        ("e9_unknown", "e9" in report and report["e9"] is None),
    ) if flag]
    return report


# --- I/O ----------------------------------------------------------------------------

PG_SQL = """
SELECT d.run_id, d.decided_at, d.arm, d.arm_propensities,
       o.turn_ok, o.realized_nats, o.n_tested, o.n_moved,
       p.status AS plan_status, p.fallback_reason
FROM curiosity_offer_decisions d
LEFT JOIN curiosity_run_outcomes o USING (run_id)
LEFT JOIN curiosity_offer_plans p USING (run_id)
WHERE d.arm_propensities IS NOT NULL
ORDER BY d.decided_at ASC
"""


def read_pg(dsn: str) -> Optional[list[dict[str, Any]]]:
    conn = open_readonly_connection(dsn, connect_timeout=10, statement_timeout_ms=60_000)
    if conn is None:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(PG_SQL)
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]
    finally:
        conn.close()


# cockpit_turn_sighting.ts is a UTC `timestamp without time zone`: compare it to naive UTC. A turn that starts in
# the window may finish up to a day later, so the hop scan runs a day past the window and t0 is filtered after.
E9_TURNS_SQL = f"""
WITH t AS (
  SELECT correlation_id,
         min(ts) FILTER (WHERE stage = 'ingress' AND seq = 0) AS t0,
         max(ts) FILTER (WHERE stage = 'finalize') AS tf,
         bool_or({JUNIPER_INGRESS_PREDICATE}) AS juniper
  FROM cockpit_turn_sighting
  WHERE ts >= %(lo_naive)s AND ts < %(hi_pad_naive)s
  GROUP BY 1
), w AS (
  SELECT correlation_id, count(*) AS calls,
         sum(wait_ms) FILTER (WHERE pool_role = 'chat') AS chat_wait_ms
  FROM llm_call_usage
  WHERE recorded_at >= %(lo)s AND recorded_at < %(hi_pad)s
  GROUP BY 1
)
SELECT t.t0, t.tf, w.calls, w.chat_wait_ms
FROM t LEFT JOIN w USING (correlation_id)
WHERE t.juniper AND t.t0 >= %(lo_naive)s AND t.t0 < %(hi_naive)s
"""

E9_LANE_SQL = """
SELECT CASE WHEN recorded_at < %(cut)s THEN 'before' ELSE 'after' END AS w,
       sum(duration_ms) FILTER (WHERE pool_role = 'chat') AS chat_ms,
       sum(duration_ms) FILTER (WHERE pool_role = 'chat' AND route = %(route)s) AS prime_ms
FROM llm_call_usage
WHERE recorded_at >= %(lo)s AND recorded_at < %(hi)s
GROUP BY 1
"""

WEIGHT_SQL = """
SELECT decided_at, (decision->>'family_weight')::float AS family_weight, decision->>'weight_basis' AS weight_basis
FROM want_ledger
WHERE family = %(family)s
ORDER BY decided_at ASC
"""


def _naive_utc(at: datetime) -> datetime:
    return at.astimezone(timezone.utc).replace(tzinfo=None)


def read_e9(dsn: str, *, cut: datetime, now: datetime, days: int = E9_DAYS) -> Optional[tuple[list, dict]]:
    lo, hi = cut - timedelta(days=days), min(cut + timedelta(days=days), now)
    params = {"lo": lo, "hi": hi, "hi_pad": hi + timedelta(days=1), "cut": cut, "route": PRIME_ROUTE,
              "lo_naive": _naive_utc(lo), "hi_naive": _naive_utc(hi), "hi_pad_naive": _naive_utc(hi + timedelta(days=1))}
    conn = open_readonly_connection(dsn, connect_timeout=10, statement_timeout_ms=60_000)
    if conn is None:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(E9_TURNS_SQL, params)
            turns = [{"t0": r[0].replace(tzinfo=timezone.utc),
                      "tf": r[1].replace(tzinfo=timezone.utc) if r[1] is not None else None,
                      "calls": r[2], "chat_wait_ms": r[3]} for r in cur.fetchall()]
            cur.execute(E9_LANE_SQL, params)
            lane = {r[0]: {"chat_ms": r[1], "prime_ms": r[2]} for r in cur.fetchall()}
        return turns, lane
    except Exception as exc:  # a read failure is "unknown", never a number
        print(f"UNKNOWN: E9 read failed: {exc}", file=sys.stderr)
        return None
    finally:
        conn.close()


def read_weights(dsn: str) -> Optional[list[dict[str, Any]]]:
    conn = open_readonly_connection(dsn, connect_timeout=10, statement_timeout_ms=60_000)
    if conn is None:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(WEIGHT_SQL, {"family": REPLICA_FAMILY})
            return [{"decided_at": r[0], "family_weight": r[1], "weight_basis": r[2]} for r in cur.fetchall()]
    except Exception as exc:
        print(f"UNKNOWN: weight read failed: {exc}", file=sys.stderr)
        return None
    finally:
        conn.close()


def _num(v: Optional[float], fmt: str = ".3f") -> str:
    return "-" if v is None else format(v, fmt)


def render(report: dict[str, Any]) -> str:
    names = {ARM_REPLICA_ORDER: "replica plan", ARM_VALUE_ORDER: "value rule", ARM_UNCERTAINTY_ORDER: "uncertainty rule"}
    prim, stop = report["primary"], report["primary"]["stop"]
    blinded = report["blinded"]
    lines = ["How each ordering did (completed runs only; a run 'moved' if it changed at least one belief)"
             + (". Outcome numbers are hidden until the stop (--unblind shows them; do not act on them):"
                if blinded else ":")]
    for arm, s in report["arms"].items():
        outcome = "" if blinded else (f" moved {s['moved_a_belief']} ({_num(s['moved_share'], '.0%')});"
                                      f" mean {_num(s['mean_nats'])} nats, median {_num(s['median_nats'])};")
        lines.append(f"  {names[arm]:<17} {s['completed']:>3} completed of {s['drawn']} drawn"
                     f" ({s['failed']} failed, {s['unfinished']} unfinished);{outcome} tested nothing {s['tested_nothing']}"
                     + (f"; nats unknown {s['nats_unknown']}" if s["nats_unknown"] else ""))
    if stop["start"] is None:
        lines.append("Stop rule: the test has not started (no run with a replica propensity above 0).")
    else:
        per_arm = ", ".join(f"{names[a]} {n}" for a, n in stop["completed_per_arm"].items())
        lines.append(f"Stop rule: started {stop['start']:%Y-%m-%d}, day {stop['days']:.0f} of {STOP_DAYS};"
                     f" completed runs toward {STOP_RUNS_PER_ARM} per arm: {per_arm}.")
    if prim["read"]:
        lines.append(f"  Stopped because {stop['reason']}. p (replica beats value) = {_num(prim['p'][ARM_VALUE_ORDER], '.4f')},"
                     f" p (replica beats uncertainty) = {_num(prim['p'][ARM_UNCERTAINTY_ORDER], '.4f')}.")
    lines.append(f"  {prim['verdict']}")
    h = report["harm"]
    if h["look"] is not None:
        lines.append(f"Harm look of {h['look']:%Y-%m-%d} (replica vs both rules pooled, one-sided, p < {HARM_ALPHA}):"
                     f" failed turns {h['failed_turns']['replica']} vs {h['failed_turns']['rules']}"
                     f" (p {_num(h['failed_turns']['p'])}); tested nothing {h['tested_nothing']['replica']} vs"
                     f" {h['tested_nothing']['rules']} (p {_num(h['tested_nothing']['p'])}).")
    else:
        lines.append("Harm look (every 30 days from the start):")
    lines.append(f"  {h['verdict']}" + (f"; next scheduled look {h['next_scheduled_look']:%Y-%m-%d}"
                                        if h["next_scheduled_look"] else ""))
    f = report["fallbacks"]
    lines.append(f"Fallbacks, last {f['days']} days: {f['fell_back']} of {f['replica_runs']} replica-arm runs did not"
                 f" use a plan; by plan status {f['by_status']}; unfunded share {_num(f['unfunded_share'], '.0%')}.")
    lines.append(f"  {f['verdict']}")
    t = report["as_treated"]
    if not blinded:
            lines.append(f"As treated (for reading only): plan used {t['plan_used']['completed']} runs, mean"
                     f" {_num(t['plan_used']['mean_nats'])} nats; fell back {t['fell_back']['completed']} runs, mean"
                     f" {_num(t['fell_back']['mean_nats'])} nats.")
    a = report["answers"]
    if a is None:
        lines.append("Answers to Prime: UNKNOWN (could not read them).")
    else:
        lines.append(f"Answers to Prime: {a['answers']} {a['by_stance_arm']}; {a['verdict']}")
    if "e9" in report:
        e = report["e9"]
        if e is None:
            lines.append("Juniper's chat speed (E9): UNKNOWN (could not read it).")
        else:
            b, af, sh = e["before"], e["after"], e["prime_share_of_chat_lane"]
            lines.append(f"Juniper's chat speed (E9), {e['days_before']} days before {e['cut']:%Y-%m-%d %H:%M} UTC vs"
                         f" {e['days_after']} days after:")
            for name, w in (("before", b), ("after", af)):
                lines.append(f"  {name:<6} {w['turns']} turns ({w['finished']} finished): whole turn median"
                             f" {_num(w['median_turn_s'], '.0f')} s, p90 {_num(w['p90_turn_s'], '.0f')} s; seat wait on the"
                             f" chat lane median {_num(w['median_wait_s'], '.1f')} s, p90 {_num(w['p90_wait_s'], '.1f')} s"
                             f" ({w['wait_measured']} turns measured)")
            lines.append(f"  Prime + replica used {_num(sh['before'], '.1%')} of the chat lane's model time before,"
                         f" {_num(sh['after'], '.1%')} after. p (whole turn slower) {_num(e['p_turn_slower'])}, which also"
                         f" moves with what was asked; p (seat wait slower) {_num(e['p_wait_slower'])}.")
            lines.append(f"  {e['verdict']}")
    if "weight" in report:
        w = report["weight"]
        if w is None:
            lines.append("Replica family weight: UNKNOWN (could not read it).")
        elif not w["decisions"]:
            lines.append(f"Replica family weight: {w['verdict']}")
        else:
            lines.append(f"Replica family weight in the body economy: {w['first']:.3f} on {w['first_at']:%Y-%m-%d} ->"
                         f" {w['last']:.3f} on {w['last_at']:%Y-%m-%d} (range {w['min']:.3f}-{w['max']:.3f},"
                         f" {w['decisions']} decisions); {w['verdict']}.")
    lines.append("Needs Juniper: " + (", ".join(report["needs_juniper"]) or "nothing"))
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--days", type=int, default=14, help="fallback report window")
    parser.add_argument("--unblind", action="store_true",
                        help="show per-arm outcome numbers before the stop (for checking the pipe, not for deciding)")
    parser.add_argument("--e9-cut", help="E9 cut, ISO time in UTC (default: the replica arm's start). "
                                         "2026-10-30 compares around Prime's start instead.")
    args = parser.parse_args(argv)
    dsn = os.environ.get("POSTGRES_URI") or DEFAULT_POSTGRES_URI
    rows = read_pg(dsn)
    if rows is None:
        print("UNKNOWN: Postgres unreadable", file=sys.stderr)
        return 2
    answer_rows = answers_readout.read_pg(dsn)
    answers = answers_readout.parse_rows(answer_rows) if answer_rows is not None else None
    now = datetime.now(timezone.utc)
    live = three_arm_rows(rows)
    cut = (datetime.fromisoformat(args.e9_cut).replace(tzinfo=timezone.utc) if args.e9_cut
           else live[0]["decided_at"] if live else None)
    e9: Any = None
    if cut is not None:
        got = read_e9(dsn, cut=cut, now=now)
        e9 = e9_report(*got, cut=cut, now=now) if got is not None else None
    weight_rows = read_weights(dsn)
    report = build_report(rows, answers, now=now, days=args.days, unblind=args.unblind, e9=e9,
                          weight=weight_report(weight_rows) if weight_rows is not None else None)
    print(json.dumps(report, indent=1, default=str) if args.json else render(report))
    return 3 if report["needs_juniper"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
