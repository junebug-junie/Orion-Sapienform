#!/usr/bin/env python3
"""Eval: the market's consent gate refuses what it must and changes nothing else (attention patch 3d).

Spec: docs/superpowers/specs/2026-11-07-attention-patch3-market-decides-design.md section 3, acceptance
checks 4 and 7 (the dream line). READ-ONLY.

Two lanes:

* ``--fixture`` (default; deterministic, no database). The shipped config/physiology/consent.yaml and
  families.yaml through the market's own path (``app.physiology_market.Market.bid``, fakes for Redis,
  Postgres and the bus):
  - ``irreversible`` (dream prune) is refused, even with a standing consent for downscale;
  - an unknown non-reversible act is refused; an unknown reversible act is allowed;
  - a missing attest is refused; ``downscale_only`` with both attests is allowed once a standing consent
    exists, and needs consent without one (the shipped state);
  - a ``consent`` deferral is never admitted by the binding rule, not at the window's end, not with no end,
    and a dead market (no body state, unreadable history) cannot turn it into "no answer";
  - with no consent request, every decision for the live want shapes (orion/physiology/tests/
    consent_fixture.py) is byte-identical to the one before this patch: no consent rules and no
    dream.compaction family on one side, both on the other, with uneven real-shaped weight evidence.
* ``--live``: over the last ``--days``, from Postgres (read-only): staged REM deltas
  (``dream_compaction_delta``) against consent asks (``want_ledger`` rows of family ``dream.compaction``).
  Gate: every ask carries a consent verdict, no ``dream.compaction`` decision is funded without ``allow``,
  and (after the dream deploy) at least one ask exists per staged delta. Before deploy it reports 0 asks:
  that is the UNVERIFIED state, not a pass.

    PYTHONPATH=. .venv/bin/python services/orion-durable-runs/evals/consent_gate_fixture.py [--live --days 3]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

SVC = Path(__file__).resolve().parents[1]
ROOT = SVC.parents[1]
sys.path[:0] = [str(ROOT), str(SVC)]

import yaml  # noqa: E402

from app.physiology_market import Market  # noqa: E402
from orion.physiology.binding import bind_verdict  # noqa: E402
from orion.physiology.consent import load_consent, parse_consent  # noqa: E402
from orion.physiology.families import load_families  # noqa: E402
from orion.physiology.tests.consent_fixture import live_wants  # noqa: E402
from orion.schemas.physiology import PhysiologyConstraintReadingV1 as R, PhysiologyStateV1  # noqa: E402
from orion.schemas.physiology_want import ConsentRequestV1, WantV1  # noqa: E402

CONTAINER = "orion-athena-sql-db"
NOW = datetime(2026, 11, 11, 8, 0, tzinfo=timezone.utc)
FAMS, _ = load_families(ROOT / "config/physiology/families.yaml")
SHIPPED, ERR = load_consent(ROOT / "config/physiology/consent.yaml")
_DOC = yaml.safe_load((ROOT / "config/physiology/consent.yaml").read_text())
GRANTED = parse_consent({**_DOC, "standing": {"dream-compaction-downscale": {"granted_by": "juniper",
                                                                             "granted_on": "2026-11-11"}}})


def _state():
    p = {"value": 0.0, "ceiling": 1.0, "bind_state": "slack", "horizon_sec": 3600.0, "why": ""}
    roles = ("agent", "agent-gpu2", "metacog", "diffusion", "chat", "fast")
    return PhysiologyStateV1(tick_id=f"physiology:{NOW.isoformat()}", generated_at=NOW, constraints=[
        R(constraint="heat", unit="c", level=28.0, budget=31.0, budget_kind="soft_line", price=p),
        R(constraint="attention:juniper", unit="messages", level=1.0, budget=4.0, budget_kind="daily_cap", price=p),
        *[R(constraint=f"lane:{r}", unit="slot_s", level=600.0, budget=3600.0, budget_kind="capacity",
            detail={"slots": 1, "window_sec": 3600.0}, price=p) for r in roles]])


def _acts():
    out = []
    for w in live_wants():
        for i in range(8):
            closed = NOW - timedelta(hours=2 + 3 * i)
            out.append({"act_id": f"{w.family}{i}", "actor": w.family, "opened_at": closed - timedelta(minutes=20),
                        "closed_at": closed, "outcome_value": 1.0, "outcome_source": "producer", "signal_id": None,
                        "cost_json": json.dumps({"seconds": 1200.0, "gpu_slot_s_by_lane": {"agent": 600.0}})})
    return out


def _uses():
    """Uneven downstream use (linked, window closed) so the tier's weight normalization really moves."""
    out = []
    for fam, rate in (("durable.curiosity_investigate", 0.9), ("thought.painting", 0.2)):
        for i in range(40):
            out.append({"act_id": f"{fam}u{i}", "actor": fam, "started_at": NOW - timedelta(days=1, hours=i),
                        "link_status": "linked", "used": (i % 10) < rate * 10, "window_closed": True})
    return out


class _Cur:
    def __init__(self, rows):
        self.rows = rows

    async def fetchall(self):
        return self.rows


class _Pool:
    def __init__(self, fail=False):
        self.fail, self.calls = fail, 0

    def connection(self):
        pool = self

        class _Conn:
            async def execute(self, sql, params=None):
                pool.calls += 1
                if pool.fail:
                    raise RuntimeError("db down")
                if "grammar_acts" in sql:
                    return _Cur(_acts())
                return _Cur(_uses() if "act_downstream_use" in sql else [])

        class _CM:
            async def __aenter__(self):
                return _Conn()

            async def __aexit__(self, *a):
                return False

        return _CM()


class _Redis:
    def __init__(self, raw):
        self.raw = raw

    async def get(self, key):
        return self.raw


def market(rules, *, raw=None, pool=None, families=None):
    async def publish(d):
        return None

    raw = _state().model_dump_json() if raw is None else raw
    return Market(families=families or FAMS, families_error=None, pool_getter=lambda: pool or _Pool(),
                  redis_getter=lambda: _Redis(raw if raw != "" else None), publish=publish, consent_rules=rules)


def cwant(verb, rev, attests, family="dream.compaction", i=0):
    wid = f"{family}:eval-{i}"
    return WantV1(want_id=wid, family=family, producer="orion-dream.compaction_applier.consent", verb=verb, tier=2,
                  consent_request=ConsentRequestV1(want_id=wid, family=family, verb=verb, payload_digest="sha256:0",
                                                   reversibility=rev, attests=list(attests)))


BOTH = ("snapshot_dir_writable", "downscale_only")
CASES = [
    # (name, rules, want, expected verdict, expected rule)
    ("prune is irreversible: refused", GRANTED, cwant("prune", "irreversible", BOTH), "refuse", "dream.compaction.prune"),
    ("unknown non-reversible act: refused", SHIPPED, cwant("compact", "snapshot_restorable", BOTH, "thought.painting"),
     "refuse", None),
    ("unknown reversible act: allowed", SHIPPED, cwant("make", "reversible", (), "thought.painting"), "allow", None),
    ("missing attest: refused", GRANTED, cwant("compact", "snapshot_restorable", ("downscale_only",)), "refuse",
     "dream.compaction.downscale_only"),
    ("downscale_only, both attests, standing consent: allowed", GRANTED, cwant("compact", "snapshot_restorable", BOTH),
     "allow", "dream.compaction.downscale_only"),
    ("downscale_only, shipped (no standing consent): needs consent", SHIPPED,
     cwant("compact", "snapshot_restorable", BOTH), "needs_consent", "dream.compaction.downscale_only"),
    ("no rules read: refused", None, cwant("compact", "snapshot_restorable", BOTH), "refuse", None),
]


def fixture() -> list[str]:
    fails: list[str] = []
    if ERR:
        return [f"shipped consent.yaml did not load: {ERR}"]
    for i, (name, rules, w, verdict, rule) in enumerate(CASES):
        d = asyncio.run(market(rules).bid(w, NOW + timedelta(seconds=i)))
        got = (d.consent.verdict if d.consent else None, d.consent.rule_id if d.consent else None)
        ok = got == (verdict, rule) and ((d.reason == "consent") == (verdict != "allow"))
        print(f"{'ok ' if ok else 'BAD'} {name}: verdict={got[0]} rule={got[1]} reason={d.reason}")
        if not ok:
            fails.append(f"{name}: got {got} reason={d.reason}")
        if verdict != "allow":
            for end in (NOW, NOW - timedelta(hours=1), None):
                if bind_verdict(d, NOW, end).admit:
                    fails.append(f"{name}: admitted by the binding rule at end={end}")
    # A dead market cannot turn a consent deferral into "no answer".
    for label, raw, pool in (("no body state", "", _Pool()), ("history unreadable", None, _Pool(fail=True))):
        d = asyncio.run(market(SHIPPED, raw=raw, pool=pool).bid(cwant("compact", "snapshot_restorable", BOTH), NOW))
        ok = d.reason == "consent" and not bind_verdict(d, NOW, NOW).admit and pool.calls == 0
        print(f"{'ok ' if ok else 'BAD'} dead market ({label}): reason={d.reason} reads={pool.calls}")
        if not ok:
            fails.append(f"dead market ({label}) let a consent want through: reason={d.reason}")
    # No consent request: byte-identical with and without the gate AND the new dream.compaction family
    # (a family in the weight pool moves every other family's weight; it ships `weighted: false`).
    from orion.physiology.families import FamiliesConfig

    before = FamiliesConfig(families={f: c for f, c in FAMS.families.items() if f != "dream.compaction"})
    same = 0
    for w in live_wants():
        a = asyncio.run(market(None, families=before).bid(w, NOW)).model_dump_json()
        b = asyncio.run(market(SHIPPED).bid(w, NOW)).model_dump_json()
        if a != b or '"consent"' in b:
            fails.append(f"{w.family}: decision changed by the consent gate with no consent request")
        else:
            same += 1
    print(f"{'ok ' if same == len(live_wants()) else 'BAD'} no consent request: {same}/{len(live_wants())} "
          "live want shapes decided byte-identically")
    return fails


def _psql(sql: str) -> list[list[str]]:
    out = subprocess.run(["docker", "exec", CONTAINER, "psql", "-U", "postgres", "-d", "conjourney", "-AtF", "\t", "-c",
                          sql], check=True, capture_output=True, text=True).stdout
    return [line.split("\t") for line in out.splitlines() if line]


def live(days: float) -> list[str]:
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    # Deltas are counted from the first ask on (deltas staged before the deploy were never asked about).
    first = _psql(f"SELECT coalesce(min(decided_at)::text, 'none') FROM want_ledger WHERE family = 'dream.compaction' "
                  f"AND decided_at >= '{since}'")[0][0]
    deltas = int(_psql(f"SELECT count(*) FROM dream_compaction_delta WHERE created_at >= '{since if first == "none" else first}'")[0][0])
    rows = _psql("SELECT outcome, reason, coalesce(decision->'consent'->>'verdict', '<none>'), "
                 "coalesce(decision->'consent'->>'rule_id', '<none>'), count(*) FROM want_ledger "
                 f"WHERE family = 'dream.compaction' AND decided_at >= '{since}' GROUP BY 1, 2, 3, 4 ORDER BY 5 DESC")
    asks = sum(int(r[4]) for r in rows)
    print(f"live {days:g} d: staged deltas={deltas} consent asks={asks}")
    for r in rows:
        print(f"  outcome={r[0]} reason={r[1]} verdict={r[2]} rule={r[3]} n={r[4]}")
    fails = []
    if any(r[2] == "<none>" for r in rows):
        fails.append("a dream.compaction decision carries no consent verdict")
    if any(r[0] == "funded" and r[2] != "allow" for r in rows):
        fails.append("a dream.compaction decision was funded without allow")
    if asks == 0:
        print("UNVERIFIED: no consent asks yet (orion-dream and orion-durable-runs not rebuilt, or no REM delta staged)")
    elif asks < deltas:
        fails.append(f"fewer consent asks ({asks}) than staged deltas ({deltas})")
    return fails


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--days", type=float, default=3.0)
    a = ap.parse_args(argv)
    fails = live(a.days) if a.live else fixture()
    for f in fails:
        print("FAIL", f)
    print("PASS" if not fails else f"FAIL ({len(fails)})")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
