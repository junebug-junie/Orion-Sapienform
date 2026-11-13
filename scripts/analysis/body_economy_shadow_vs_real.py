#!/usr/bin/env python3
"""Shadow vs real (READ-ONLY): what the body economy would have funded, next to what really ran.

Body economy Patch C3a. Design: docs/superpowers/specs/2026-10-23-organism-body-economy-design.md,
Patch C row: "a shadow-vs-real disagreement table for 7 days" -- the evidence Patch D needs before
it lets the market replace an old veto.

Per want family over a window it reports:

* wants bid; shadow funded vs deferred;
* what really happened: did an act with the want's ``correlation_id`` (or its ``object_ref``) OPEN
  inside the want's window (``ran``), or not (``not run``: vetoed, refused or never started);
* the four-cell agreement table:

  ==================  ===========================  ===========================
                      really ran                   did not run
  ==================  ===========================  ===========================
  shadow funded       agree (funded & ran)         old veto STRICTER
  shadow deferred     old gate LOOSER              agree (deferred & not run)
  ==================  ===========================  ===========================

* the top binding constraints of the deferrals, and the old veto reasons where they are logged
  (curiosity ``scheduling_block_reason`` lines in the Hub's journald log; outreach's
  ``outreach_block_reason`` is not logged per tick -- reported as unlogged, never as zero).

Kept out of the table, each counted on its own: decisions the market could not make
(``physiology_unavailable``, ``unknown_family``, ``family_never_bids``: "no market"), and wants of
a family that has no act producer at all (no ``grammar_acts`` row ever: "real outcome unknown").

Two sources (``--source``):

* ``ledger``: the live ``want_ledger`` rows in the window (what the deployed bid sites decided).
* ``replay``: the last N days of REAL durable admissions (curiosity investigate, self-inquiry,
  self-sense eval, journal compose -- the runs ``app/run_bids.py`` bids for), re-decided by the live
  Choice code against the ``physiology_state`` row in force at each admission, the act history
  before it and one reservation book (opened at each act's open, released at each run's terminal),
  in time order (reserving from each act's open, as the live bid site does). This is the table the bid site would have produced had it been live; it sees only
  runs Hub already let through (a Hub veto never reaches admission -- the veto log covers those).

Writes nothing to any database (read-only psql session, ``_body_economy_data.run_query``).

    .venv/bin/python scripts/analysis/body_economy_shadow_vs_real.py --days 7 \
        --out-json /tmp/x.json --out-md /tmp/x.md [--source both] [--cache /tmp/x.cache.json]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT))

NO_MARKET = ("physiology_unavailable", "unknown_family", "family_never_bids")
# Patch D: binding bids are obeyed (a deferral is a hold that re-bids), so "deferred & ran" is not a
# disagreement for them. They are kept out of the table; body_economy_binding_replay.py covers them.
from orion.physiology.binding import BINDING_PRODUCERS  # noqa: E402
# Hand-run verification bids (deploy checks, probes, replays) are not producers' wants (the floor's own list).
from orion.physiology.choice import SYNTHETIC_PRODUCERS  # noqa: E402
OPEN_SLACK = timedelta(seconds=60)    # an act may be stamped a moment before the decision row
CELLS = ("funded_ran", "funded_not_run", "deferred_ran", "deferred_not_run")
CELL_LABEL = {"funded_ran": "funded & ran (agree)", "funded_not_run": "funded & not run (old veto stricter)",
              "deferred_ran": "deferred & ran (old gate looser)", "deferred_not_run": "deferred & not run (agree)"}
# Hub journald veto lines -> the want family whose run the veto stopped (curiosity_investigation.py).
VETO_FAMILY = {"curiosity_investigation": "durable.curiosity_investigate",
               "curiosity_self_inquiry": "durable.self_inquiry",
               "curiosity_self_sense_eval": "durable.self_sense_eval"}
UNLOGGED_VETOES = {"hub.outreach": "outreach_block_reason is evaluated every tick but not logged per tick "
                                   "(status() only): no veto history to count"}


def _ts(v: Any) -> Optional[datetime]:
    if v is None or v == "":
        return None
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    d = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


# --- the pure join --------------------------------------------------------------------------------

@dataclass(frozen=True)
class Decision:
    decision_id: str
    family: str
    producer: str
    decided_at: datetime
    outcome: str
    reason: str
    binding: Optional[str]
    correlation_id: Optional[str]
    object_ref: Optional[str]
    window_end: datetime


@dataclass(frozen=True)
class Act:
    actor: str
    opened_at: Optional[datetime]
    correlation_id: Optional[str]
    object_ref: Optional[str]


def window_end(decided_at: datetime, latest: Optional[datetime], opened_at: Optional[datetime],
               family_window_s: float) -> datetime:
    """The want's window: its own ``latest``, else from its opened_at (or the decision) + the family's
    deferrable window (families.yaml). Never before the decision itself."""
    if latest is not None:
        return max(latest, decided_at)
    base = opened_at or decided_at
    return max(base + timedelta(seconds=family_window_s), decided_at)


def classify(decisions: Iterable[Decision], acts: Iterable[Act]) -> list[dict]:
    """One row per decision: its cell, or why it is outside the table."""
    by_corr: dict[str, list[Act]] = defaultdict(list)
    by_ref: dict[str, list[Act]] = defaultdict(list)
    producing: set[str] = set()
    since: dict[str, datetime] = {}      # actor -> its first act open seen (the producer's start)
    for a in acts:
        producing.add(a.actor)
        if a.opened_at is None:
            continue
        if a.actor not in since or a.opened_at < since[a.actor]:
            since[a.actor] = a.opened_at
        if a.correlation_id:
            by_corr[a.correlation_id].append(a)
        if a.object_ref:
            by_ref[a.object_ref].append(a)
    rows = []
    for d in decisions:
        row = {"decision_id": d.decision_id, "family": d.family, "producer": d.producer,
               "decided_at": d.decided_at.isoformat(), "outcome": d.outcome, "reason": d.reason,
               "binding": d.binding, "window_end": d.window_end.isoformat(), "ran": None, "act_opened_at": None,
               "cell": None, "excluded": None}
        if d.producer in SYNTHETIC_PRODUCERS:
            row["excluded"] = "synthetic_probe"
        elif d.producer in BINDING_PRODUCERS:
            row["excluded"] = "binding_obeyed"
        elif d.reason in NO_MARKET:
            row["excluded"] = "no_market"
        elif d.family not in producing:
            row["excluded"] = "no_act_producer"
        elif d.family in since and d.decided_at < since[d.family] - OPEN_SLACK:
            # The family's act producer had not started yet: "no act" says nothing about the run.
            row["excluded"] = "before_act_producer"
        cands = by_corr.get(d.correlation_id or "", []) + by_ref.get(d.object_ref or "", [])
        hit = [a for a in cands if a.actor == d.family
               and d.decided_at - OPEN_SLACK <= a.opened_at <= d.window_end]
        if hit:
            first = min(a.opened_at for a in hit)
            row["ran"], row["act_opened_at"] = True, first.isoformat()
            row["wait_sec"] = round(max(0.0, (first - d.decided_at).total_seconds()), 1)
        elif row["excluded"] not in ("no_act_producer", "before_act_producer"):
            row["ran"] = False
        if row["excluded"] is None:
            row["cell"] = f"{'funded' if d.outcome == 'funded' else 'deferred'}_{'ran' if row['ran'] else 'not_run'}"
        rows.append(row)
    return rows


def summarize(rows: list[dict], vetoes: Optional[dict] = None) -> dict:
    """Per family and overall: bids, funded/deferred, ran/not run, the four cells, the top binding
    constraints, the no-market and no-producer counts, and the logged old veto reasons."""
    fams: dict[str, dict] = {}
    for r in rows:
        f = fams.setdefault(r["family"], {"bids": 0, "funded": 0, "deferred": 0, "ran": 0, "not_run": 0,
                                          "unknown_outcome": 0, **{c: 0 for c in CELLS}, "no_market": 0,
                                          "no_act_producer": 0, "before_act_producer": 0, "synthetic_probe": 0, "binding_obeyed": 0,
                                          "binding": Counter(),
                                          "reasons": Counter(), "deferred_ran_by_reason": Counter(), "waits": []})
        f["bids"] += 1
        f[r["outcome"]] += 1
        f["reasons"][r["reason"]] += 1
        f["ran" if r["ran"] else ("not_run" if r["ran"] is False else "unknown_outcome")] += 1
        if r["excluded"]:
            f[r["excluded"]] += 1
        else:
            f[r["cell"]] += 1
            if r["cell"] == "deferred_ran":
                f["deferred_ran_by_reason"][r["reason"]] += 1
            if r["ran"] and r.get("wait_sec") is not None:
                f["waits"].append(r["wait_sec"])
        if r["outcome"] == "deferred" and r["binding"]:
            f["binding"][r["binding"]] += 1
    out = {}
    for fam, f in sorted(fams.items()):
        in_table = sum(f[c] for c in CELLS)
        agree = f["funded_ran"] + f["deferred_not_run"]
        waits = sorted(f["waits"])
        out[fam] = {**{k: v for k, v in f.items() if k not in ("binding", "reasons", "deferred_ran_by_reason", "waits")},
                    "deferred_ran_by_reason": dict(f["deferred_ran_by_reason"]),
                    "median_wait_to_open_sec": waits[len(waits) // 2] if waits else None,
                    "in_table": in_table, "agreement": round(agree / in_table, 4) if in_table else None,
                    "top_binding": f["binding"].most_common(3), "reasons": dict(f["reasons"]),
                    "old_vetoes": (vetoes or {}).get(fam), "old_vetoes_note": UNLOGGED_VETOES.get(fam)}
    total = {c: sum(x[c] for x in out.values()) for c in (*CELLS, "bids", "funded", "deferred", "ran", "not_run",
                                                          "no_market", "no_act_producer", "before_act_producer",
                                                          "synthetic_probe", "binding_obeyed", "in_table")}
    dr = Counter()
    for x in out.values():
        dr.update(x["deferred_ran_by_reason"])
    total["deferred_ran_by_reason"] = dict(dr)
    agree = total["funded_ran"] + total["deferred_not_run"]
    total["agreement"] = round(agree / total["in_table"], 4) if total["in_table"] else None
    bind = Counter()
    for r in rows:
        if r["outcome"] == "deferred" and r["binding"]:
            bind[r["binding"]] += 1
    total["top_binding"] = bind.most_common(5)
    return {"families": out, "total": total}


# --- reads (read-only) -----------------------------------------------------------------------------

def _q(sql: str) -> Any:
    import _body_economy_data as data_mod

    return data_mod.run_query(sql)


def _iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch(start: datetime, end: datetime, *, history_days: int = 30) -> dict:
    """Every table this report reads, as JSON-safe rows (also the ``--cache`` format)."""
    s, e = _iso(start), _iso(end)
    hs = _iso(start - timedelta(days=history_days))
    tail = _iso(end + timedelta(days=1))
    return {
        "start": s, "end": e,
        "ledger": _q(f"""select coalesce(json_agg(r order by r.decided_at), '[]'::json) from (
            select decision_id, want_id, family, producer, tier, decided_at, outcome, reason, binding_constraint,
                   correlation_id, decision->'want'->>'object_ref' as object_ref,
                   decision->'want'->'window'->>'latest' as window_latest, decision->'want'->>'opened_at' as want_opened_at
              from want_ledger where decided_at >= '{s}' and decided_at < '{e}') r;"""),
        "ledger_funded_before": _q(f"""select coalesce(json_agg(r), '[]'::json) from (
            select decision_id, family, tier, decided_at, producer from want_ledger
             where outcome = 'funded' and decided_at >= '{_iso(start - timedelta(days=7))}' and decided_at < '{s}') r;"""),
        "acts": _q(f"""select coalesce(json_agg(r), '[]'::json) from (
            select act_id, actor, opened_at, closed_at, outcome_value, outcome_source, signal_id, cost_json,
                   correlation_id, object_ref
              from grammar_acts where coalesce(opened_at, closed_at, created_at) >= '{hs}'
               and coalesce(opened_at, closed_at, created_at) < '{tail}') r;"""),
        "act_actors": _q("select coalesce(json_agg(distinct actor), '[]'::json) from grammar_acts;"),
        "uses": _q(f"""select coalesce(json_agg(r), '[]'::json) from (
            select act_id, actor, started_at, link_status, used, window_closed
              from act_downstream_use where started_at >= '{hs}' and started_at < '{e}') r;"""),
        "admissions": _q(f"""select coalesce(json_agg(r order by r.created_at), '[]'::json) from (
            select run_id, request, created_at, terminal, updated_at
              from durable_admission_runs where created_at >= '{s}' and created_at < '{e}') r;"""),
        "states": _q(f"""select coalesce(json_agg(r order by r.generated_at), '[]'::json) from (
            select tick_id, generated_at, constraints, warnings from physiology_state
             where generated_at >= '{_iso(start - timedelta(minutes=10))}' and generated_at < '{e}') r;"""),
    }


def hub_vetoes(start: datetime, end: datetime, *, container: str = "orion-athena-hub") -> tuple[dict, Optional[str]]:
    """Curiosity veto reasons per family from the Hub's journald log, or ({}, why) when unreadable."""
    try:
        proc = subprocess.run(["journalctl", f"CONTAINER_NAME={container}", "--since", _iso(start).replace("T", " ")[:-1],
                               "--until", _iso(end).replace("T", " ")[:-1], "-o", "short-iso", "--no-pager"],
                              capture_output=True, text=True, timeout=300)
    except Exception as exc:  # noqa: BLE001
        return {}, f"journalctl unavailable: {type(exc).__name__}"
    if proc.returncode != 0:
        return {}, f"journalctl failed: {proc.stderr.strip()[:200]}"
    return parse_vetoes(proc.stdout.splitlines()), None


def parse_vetoes(lines: Iterable[str]) -> dict:
    out: dict[str, Counter] = defaultdict(Counter)
    for line in lines:
        # short-iso lines: "<ts> <host> <tag>[pid]: <message>"; the message holds the veto line.
        m = re.search(r"\b(curiosity_investigation|curiosity_self_inquiry|curiosity_self_sense_eval)_blocked reason=(\S+)", line)
        if m:
            out[VETO_FAMILY[m.group(1)]][m.group(2)] += 1
    return {f: dict(c.most_common()) for f, c in out.items()}


# --- the two sources -------------------------------------------------------------------------------

def _families():
    from orion.physiology.families import load_families

    fams, err = load_families(_ROOT / "config" / "physiology" / "families.yaml")
    if err:
        raise SystemExit(f"families.yaml: {err}")
    return fams


def _acts_for_join(ex: dict) -> list[Act]:
    return [Act(actor=str(r["actor"]), opened_at=_ts(r.get("opened_at")), correlation_id=r.get("correlation_id"),
                object_ref=r.get("object_ref")) for r in ex["acts"]]


def _producing(ex: dict) -> list[Act]:
    """Join acts plus one sentinel per actor that ever wrote an act (older than the history window)."""
    acts = _acts_for_join(ex)
    seen = {a.actor for a in acts}
    acts += [Act(actor=str(a), opened_at=None, correlation_id=None, object_ref=None)
             for a in ex.get("act_actors") or [] if a not in seen]
    return acts


def ledger_decisions(ex: dict, fams) -> list[Decision]:
    out = []
    for r in ex["ledger"]:
        cfg = fams.get(r["family"])
        decided = _ts(r["decided_at"])
        out.append(Decision(str(r["decision_id"]), str(r["family"]), str(r["producer"]), decided, str(r["outcome"]),
                            str(r["reason"]), r.get("binding_constraint"), r.get("correlation_id"), r.get("object_ref"),
                            window_end(decided, _ts(r.get("window_latest")), _ts(r.get("want_opened_at")),
                                       cfg.deferrable_window_s if cfg is not None else 0.0)))
    return out


def replay_admissions(ex: dict, fams) -> tuple[list[Decision], dict]:
    """Re-decide every real want-family admission in the window with the live Choice code."""
    sys.path.insert(0, str(_ROOT / "services" / "orion-durable-runs"))
    from app.run_bids import want_for_run  # the bid site's own want builder
    from app.run_acts import run_act_family

    from orion.physiology.appetite import ActRecord
    from orion.physiology.choice import FundedWant, ReservationBook, build_context, decide
    from orion.physiology.weights import UseRecord, weight_series
    from orion.schemas.physiology import PhysiologyStateV1

    acts = [ActRecord(act_id=str(r["act_id"]), family=str(r["actor"]), opened_at=_ts(r.get("opened_at")),
                      closed_at=_ts(r.get("closed_at")), outcome_value=r.get("outcome_value"),
                      outcome_source=r.get("outcome_source"), signal_id=r.get("signal_id"),
                      cost=r.get("cost_json") if isinstance(r.get("cost_json"), dict) else {}) for r in ex["acts"]]
    uses = [UseRecord(act_id=str(r["act_id"]), family=str(r["actor"]), started_at=_ts(r.get("started_at")),
                      link_status=str(r["link_status"]), used=r.get("used"), window_closed=bool(r.get("window_closed")))
            for r in ex["uses"]]
    states = [PhysiologyStateV1.model_validate({**s, "generated_at": s["generated_at"]}) for s in ex["states"]]
    start, end = _ts(ex["start"]), _ts(ex["end"])
    series = weight_series(fams, acts, uses, start=(start - timedelta(days=1)).date(), end=end.date())
    book = ReservationBook()
    book.seed_funded([FundedWant(str(r["decision_id"]), _ts(r["decided_at"]), str(r["family"]), int(r["tier"]))
                      for r in ex.get("ledger_funded_before") or [] if r.get("producer") not in SYNTHETIC_PRODUCERS], start)
    act_by_ref = {}
    for r in ex["acts"]:
        if r.get("object_ref") and r.get("opened_at"):
            act_by_ref.setdefault(r["object_ref"], r)
    events = []      # (t, order, kind, payload): bids before opens before closes at the same instant
    for adm in ex["admissions"]:
        req = adm["request"] if isinstance(adm["request"], dict) else json.loads(adm["request"])
        family = run_act_family(req)
        if family is None:
            continue
        t = _ts(adm["created_at"])
        events.append((t, 0, "bid", {"req": req, "family": family}))
        act = act_by_ref.get(f"durable_run:{adm['run_id']}")
        if act is not None:
            events.append((_ts(act["opened_at"]), 1, "open", adm["run_id"]))
        if adm.get("terminal") and adm.get("updated_at"):
            events.append((_ts(adm["updated_at"]), 2, "close", adm["run_id"]))
    events.sort(key=lambda e: (e[0], e[1]))
    out, si, stats = [], 0, Counter()
    current = None
    for t, _o, kind, p in events:
        if kind == "open":
            book.mark_opened(p, t)
            continue
        if kind == "close":
            book.release(p)
            continue
        while si < len(states) and states[si].generated_at <= t:
            current = states[si]
            si += 1
        cfg = fams.get(p["family"].actor)
        want = want_for_run(p["req"], cfg.tier if cfg is not None else 0)
        if want is None:
            stats["skipped_not_bidding"] += 1
            continue
        before = [a for a in acts if (a.closed_at or a.opened_at) is not None and (a.closed_at or a.opened_at) < t]
        ctx = build_context(fams, before, [u for u in uses if u.started_at and u.started_at < t], current, t,
                            weights=series.get(t.date(), {}))
        d = decide(want, current, ctx, book, t, reserve_at_open=True)   # as app/run_bids.py bids
        stats[d.reason] += 1
        out.append(Decision(d.decision_id, d.family, want.producer, t, d.outcome, d.reason, d.binding_constraint,
                            want.correlation_id, want.object_ref,
                            window_end(t, want.window.latest, want.opened_at, cfg.deferrable_window_s if cfg else 0.0)))
    return out, {"reasons": dict(stats), "states_read": len(states),
                 "admissions_in_window": len(ex["admissions"]), "want_family_bids": len(out)}


# --- render ---------------------------------------------------------------------------------------

def render_md(report: dict) -> str:
    L = [f"# Body economy: shadow vs real, {report['start'][:16]}Z .. {report['end'][:16]}Z", ""]
    if report.get("veto_note"):
        L += [f"Hub veto log: {report['veto_note']}", ""]
    for name, sec in report["sources"].items():
        tot = sec["summary"]["total"]
        L += [f"## Source: {name}", "", sec["note"], ""]
        if not tot["bids"]:
            L += ["No decisions in the window.", ""]
            continue
        agree_s = "n/a" if tot["agreement"] is None else f"{tot['agreement']:.0%}"
        L += [f"{tot['bids']} wants bid: {tot['funded']} funded, {tot['deferred']} deferred (shadow). "
              f"{tot['in_table']} in the table; {tot['no_market']} the market could not decide (no market), "
              f"{tot['no_act_producer']} of a family with no act producer (real outcome unknown), "
              f"{tot['before_act_producer']} decided before the family's act producer started (real outcome unknown), "
              f"{tot['synthetic_probe']} hand-run probes. "
              f"'Deferred & ran' by reason: {tot['deferred_ran_by_reason'] or 'none'} (a capacity deferral of a "
              f"run that later ran is the pool's own queue, not a veto disagreement). "
              f"Agreement {agree_s}. "
              f"Top binding constraints: {tot['top_binding'] or 'none'}.", "",
              "| family | bids | funded | deferred | ran | not run | funded & ran | funded & not run (old stricter) | "
              "deferred & ran (old looser) | deferred & not run | no market | no producer | median wait to open (s) | top binding | old veto reasons (logged) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for fam, x in sec["summary"]["families"].items():
            veto = x["old_vetoes"] if x["old_vetoes"] else (x["old_vetoes_note"] or "none logged")
            L.append(f"| {fam} | {x['bids']} | {x['funded']} | {x['deferred']} | {x['ran']} | {x['not_run']} | "
                     f"{x['funded_ran']} | {x['funded_not_run']} | {x['deferred_ran']} | {x['deferred_not_run']} | "
                     f"{x['no_market']} | {x['no_act_producer']} | {x['median_wait_to_open_sec']} | "
                     f"{x['top_binding'] or '-'} | {veto} |")
        L += [""]
    return "\n".join(L) + "\n"


def build_report(ex: dict, *, sources: Iterable[str], vetoes: Optional[dict], veto_note: Optional[str]) -> dict:
    fams = _families()
    join_acts = _producing(ex)
    report = {"start": ex["start"], "end": ex["end"], "veto_note": veto_note, "sources": {}}
    if "ledger" in sources:
        rows = classify(ledger_decisions(ex, fams), join_acts)
        report["sources"]["ledger"] = {
            "note": ("Live want_ledger rows (the deployed bid sites). shadow=true on every row: nobody obeyed them."),
            "summary": summarize(rows, vetoes), "rows": rows}
    if "replay" in sources:
        decs, meta = replay_admissions(ex, fams)
        rows = classify(decs, join_acts)
        report["sources"]["replay"] = {
            "note": ("Real durable admissions in the window, re-decided by the live Choice code against the "
                     "physiology_state in force at each admission (in time order, one reservation book held from "
                     "each act's open and released at each run's terminal). Every one of these runs had already "
                     "passed Hub's gates: a Hub veto never reaches admission, so the 'old veto stricter' cell here "
                     "is pool refusals / failures, and Hub's own vetoes are the logged reasons column. "
                     f"{meta}"),
            "summary": summarize(rows, vetoes), "rows": rows, "meta": meta}
    return report


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--end", default=None, help="ISO time; default now")
    ap.add_argument("--source", choices=("ledger", "replay", "both"), default="both")
    ap.add_argument("--veto-logs", choices=("journald", "none"), default="journald")
    ap.add_argument("--cache", default=None, help="read/write the extract as JSON (offline reruns)")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    a = ap.parse_args(argv)
    end = _ts(a.end) if a.end else datetime.now(timezone.utc).replace(microsecond=0)
    start = end - timedelta(days=a.days)
    if a.cache and Path(a.cache).exists():
        ex = json.loads(Path(a.cache).read_text())
    else:
        ex = fetch(start, end)
        if a.cache:
            Path(a.cache).write_text(json.dumps(ex, default=str))
    vetoes, note = ({}, "not read (--veto-logs none)") if a.veto_logs == "none" else hub_vetoes(_ts(ex["start"]), _ts(ex["end"]))
    sources = ("ledger", "replay") if a.source == "both" else (a.source,)
    report = build_report(ex, sources=sources, vetoes=vetoes, veto_note=note)
    Path(a.out_json).write_text(json.dumps(report, indent=1, default=str))
    md = render_md(report)
    Path(a.out_md).write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
