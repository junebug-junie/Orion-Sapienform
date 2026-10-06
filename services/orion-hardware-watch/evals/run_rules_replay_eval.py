"""Replay orion-hardware-watch over REAL telemetry history, tick by tick, through the real Watcher.

Plan: docs/superpowers/plans/2026-09-29-urgent-curiosity-plan-4-5-hardware-watch-and-shedding.md
(task 6, acceptance 2-3). Spec: docs/superpowers/specs/2026-09-28-urgent-curiosity-and-hardware-watch-design.md
(acceptance 4).

The fixture (evals/fixtures/replay_history.csv.gz) is exported from production Postgres:
every home_cooling_sample row, and athena/circe temp_c_max + athena cabinet_temp_c +
any gpu{N}_temp_c from orion_biometrics_summary for the 14 days before the cutoff. It is committed
so this eval is deterministic and needs no database.

Two replays, both through ``app.watcher.Watcher`` with an in-memory store and a fake clock (the same
code production runs; only I/O is swapped):

- cooling + shed: from the first AC row to the cutoff (home_cooling_sample starts 2026-09-26).
- heat: the last 7 days (the first tick's baseline already has 7 days of history).

Plus rule-level statistics that do not depend on an incident being open: how often the shed rule
(and each of its legs) would be true at any 30 s tick, so the "sheds only with an open AC
incident" guarantee can be read against how warm the cabinet normally is.

Usage (from services/orion-hardware-watch):
  PYTHONPATH=../..:. python evals/run_rules_replay_eval.py            # replay + assertions
  PYTHONPATH=../..:. python evals/run_rules_replay_eval.py --json     # machine-readable report
  PYTHONPATH=../..:. python evals/run_rules_replay_eval.py --export   # refresh fixture (athena, docker)
"""
from __future__ import annotations

import argparse
import asyncio
import bisect
import csv
import gzip
import io
import json
import logging
import subprocess
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
SERVICE = HERE.parent
ROOT = SERVICE.parent.parent
for p in (str(ROOT), str(SERVICE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from orion.hardware_watch.rules import (  # noqa: E402
    CoolingPoint, ShedRuleConfig, TempPoint, cabinet_rise_c, shed_verdict,
)
from orion.schemas.curiosity_urgent import URGENT_REQUEST_CHANNEL  # noqa: E402
from orion.schemas.hardware_watch import HARDWARE_WATCH_INCIDENT_CHANNEL, HardwareWatchIncidentV1  # noqa: E402

from app.settings import Settings  # noqa: E402
from app.store import MemoryStore  # noqa: E402
from app.watcher import Watcher  # noqa: E402

FIXTURE = HERE / "fixtures" / "replay_history.csv.gz"
TICK = timedelta(seconds=30)
EXPORT_DAYS = 14
PSQL = ["docker", "exec", "-i", "orion-athena-sql-db", "psql", "-U", "postgres", "-d", "conjourney", "-At"]


# --- fixture -----------------------------------------------------------------------------------

def export(cutoff: datetime) -> None:
    since = (cutoff - timedelta(days=EXPORT_DAYS)).strftime("%Y-%m-%d %H:%M:%S")
    until = cutoff.strftime("%Y-%m-%d %H:%M:%S")
    cooling_sql = (
        "COPY (SELECT 'c', round(extract(epoch FROM ts)::numeric, 3), cooling_watts, stale, device_online, "
        f"controller_ready FROM home_cooling_sample WHERE ts < '{until}+00' ORDER BY ts) TO STDOUT WITH CSV")
    temp_sql = (
        "COPY (SELECT 't', node, k, round(extract(epoch FROM timestamp::timestamptz)::numeric, 3), "
        "round((measurements->>k)::numeric, 2) FROM orion_biometrics_summary, "
        "jsonb_object_keys(measurements) k "
        f"WHERE timestamp >= '{since}' AND timestamp < '{until}' AND node IN ('athena', 'circe') "
        "AND (k IN ('temp_c_max', 'cabinet_temp_c') OR k ~ '^gpu[0-9]+_temp_c$') "
        "ORDER BY node, k, timestamp) TO STDOUT WITH CSV")
    out = io.StringIO()
    out.write(f"# cutoff={cutoff.isoformat()} exported_at={datetime.now(timezone.utc).isoformat()}\n")
    for sql in (cooling_sql, temp_sql):
        res = subprocess.run(PSQL + ["-c", sql], capture_output=True, text=True, check=True)
        out.write(res.stdout)
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(FIXTURE, "wt", compresslevel=9) as fh:
        fh.write(out.getvalue())
    print(f"wrote {FIXTURE} ({FIXTURE.stat().st_size} bytes)")


def _ts(v: str) -> datetime:
    return datetime.fromtimestamp(float(v), tz=timezone.utc)


def _bool(v: str) -> bool | None:
    return None if v == "" else v == "t"


def load() -> tuple[datetime, list[CoolingPoint], dict[tuple[str, str], list[TempPoint]]]:
    cooling: list[CoolingPoint] = []
    temps: dict[tuple[str, str], list[TempPoint]] = {}
    cutoff = None
    with gzip.open(FIXTURE, "rt") as fh:
        first = fh.readline()
        cutoff = datetime.fromisoformat(first.split("cutoff=")[1].split()[0])
        for row in csv.reader(fh):
            if row[0] == "c":
                cooling.append(CoolingPoint(_ts(row[1]), float(row[2]) if row[2] else None, _bool(row[3]),
                                            bool(_bool(row[4])), bool(_bool(row[5]))))
            elif row[0] == "t" and row[4]:
                temps.setdefault((row[1], row[2]), []).append(TempPoint(_ts(row[3]), float(row[4])))
    cooling.sort(key=lambda p: p.ts)
    for pts in temps.values():
        pts.sort(key=lambda p: p.ts)
    return cutoff, cooling, temps


# --- replay harness ----------------------------------------------------------------------------

class ReplayStore(MemoryStore):
    """MemoryStore that sees only history up to the fake clock, sliced with bisect (fast)."""

    def __init__(self, clock, cooling, temps):
        super().__init__()
        self.clock = clock
        self.cooling = cooling
        self.temps = temps
        self._cts = [p.ts for p in cooling]
        self._tts = {k: [p.ts for p in v] for k, v in temps.items()}

    def cooling_points(self, since):
        lo = bisect.bisect_left(self._cts, since)
        hi = bisect.bisect_right(self._cts, self.clock())
        return self.cooling[lo:hi]

    def temp_points(self, node, key, since):
        ts = self._tts.get((node, key), [])
        lo = bisect.bisect_left(ts, since)
        hi = bisect.bisect_right(ts, self.clock())
        return self.temps.get((node, key), [])[lo:hi]

    def gpu_keys(self, node, since):
        now = self.clock()
        out = []
        for (n, k), ts in self._tts.items():
            if n == node and k.startswith("gpu") and k.endswith("_temp_c"):
                i = bisect.bisect_left(ts, since)
                if i < len(ts) and ts[i] <= now:
                    out.append(k)
        return sorted(out)


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


class Sink:
    def __init__(self):
        self.events: list[dict] = []
        self.urgent: list[dict] = []
        self.notices: list = []

    async def publish(self, channel, env):
        if channel == HARDWARE_WATCH_INCIDENT_CHANNEL:
            HardwareWatchIncidentV1.model_validate(env.payload)   # the contract the pool consumes
            self.events.append(env.payload)
        elif channel == URGENT_REQUEST_CHANNEL:
            self.urgent.append(env.payload)

    def notify(self, req):
        self.notices.append(req)
        return SimpleNamespace(ok=True, detail=None)


async def _inline(fn, *a, **kw):
    return fn(*a, **kw)


class NoCooling(Watcher):
    async def _cooling(self, now, open_rows, report):
        return None


def _settings(**kw) -> Settings:
    # The legacy (2026-09-30 fixture) checks are the v1 rules' own regression: they stay pinned to v1
    # while v1 remains the rollback path (spec "Rollback"). The v2 gate passes its own controller.
    base = dict(ORION_BUS_URL="redis://replay", POSTGRES_URI="replay", HARDWARE_WATCH_ENABLED=True,
                HARDWARE_WATCH_SHED_ENABLED=True, HARDWARE_WATCH_URGENT_ENABLED=True,
                HARDWARE_WATCH_HEAT_CONTROLLER="v1")
    base.update(kw)
    return Settings(**base)


def replay(cls, settings, cooling, temps, start, end):
    clock = Clock(start)
    store = ReplayStore(clock, cooling, temps)
    sink = Sink()
    w = cls(settings=settings, store=store, publish=sink.publish, notify=sink.notify, clock=clock, run_sync=_inline)
    shed_ticks = 0
    open_ticks = Counter()
    ticks = 0

    async def run():
        nonlocal shed_ticks, ticks
        while clock.t <= end:
            await w.tick()
            ticks += 1
            for r in store.open_incidents():
                open_ticks[r["rule"]] += 1
                if r["rule"] == "cooling" and r.get("shed_requested"):
                    shed_ticks += 1
            clock.t += TICK

    asyncio.run(run())
    return SimpleNamespace(store=store, sink=sink, ticks=ticks, shed_ticks=shed_ticks, open_ticks=open_ticks)


# --- report ------------------------------------------------------------------------------------

def _incidents(store) -> list[dict]:
    out = []
    for r in sorted(store.incidents.values(), key=lambda r: r["opened_at"]):
        dur = ((r["resolved_at"] or store.clock()) - r["opened_at"]).total_seconds()
        out.append({"rule": r["rule"], "subject": r["subject"], "open_reason": r["open_reason"],
                    "opened_at": r["opened_at"].isoformat(), "resolved_at": r["resolved_at"].isoformat()
                    if r["resolved_at"] else None, "minutes": round(dur / 60, 1),
                    "shed_reason": r.get("shed_reason"), "shed_requested": r.get("shed_requested")})
    return out


def shed_leg_stats(cabinet: list[TempPoint], start, end, cfg: ShedRuleConfig) -> dict:
    """At every 30 s tick in [start, end]: would the shed rule (and each leg) be true, ignoring
    whether an AC incident is open? This is what the cabinet looks like to the rule on a normal day."""
    ts = [p.ts for p in cabinet]
    t = start
    n = 0
    legs = Counter()
    reasons = Counter()
    rise_episodes_10 = 0
    prev10 = False
    while t <= end:
        lo = bisect.bisect_left(ts, t - timedelta(seconds=max(cfg.window_sec, cfg.max_age_sec) + 60))
        hi = bisect.bisect_right(ts, t)
        win = cabinet[lo:hi]
        v = shed_verdict(win, t, cfg)
        rise = cabinet_rise_c(win, t, cfg.window_sec)
        n += 1
        reasons[v.reason or "not_requested"] += 1
        if v.requested:
            legs["requested"] += 1
        if v.temp_c is not None and v.temp_c >= cfg.elevated_c:
            legs["elevated_ge_29_5"] += 1
        if v.temp_c is not None and v.temp_c >= 32.0:
            legs["hot_ge_32"] += 1
        r10 = rise is not None and rise >= cfg.rise_c
        if r10:
            legs["rise_ge_1_0"] += 1
        if rise is not None and rise >= 0.5:
            legs["rise_ge_0_5"] += 1
        if r10 and v.temp_c is not None and v.temp_c < cfg.elevated_c:
            legs["rise_ge_1_0_below_elevated"] += 1    # the only ticks where the rise leg is decisive
        if r10 and not prev10:
            rise_episodes_10 += 1
        prev10 = r10
        t += TICK
    days = (end - start).total_seconds() / 86400
    pct = {k: round(100 * c / n, 2) for k, c in legs.items()}
    return {"ticks": n, "days": round(days, 2), "pct_of_ticks": pct,
            "reasons_pct": {k: round(100 * c / n, 2) for k, c in reasons.items()},
            "rise_ge_1_0_episodes": rise_episodes_10,
            "rise_ge_1_0_episodes_per_day": round(rise_episodes_10 / days, 2) if days else None}


def cooling_facts(cooling: list[CoolingPoint]) -> dict:
    gaps = [(b.ts - a.ts).total_seconds() for a, b in zip(cooling, cooling[1:])]
    runs, cur = [], 1
    for a, b in zip(cooling, cooling[1:]):
        if b.watts == a.watts:
            cur += 1
        else:
            runs.append((cur, a.watts))
            cur = 1
    runs.append((cur, cooling[-1].watts))
    return {
        "rows": len(cooling), "first": cooling[0].ts.isoformat(), "last": cooling[-1].ts.isoformat(),
        "stale_true": sum(1 for p in cooling if p.stale is True),
        "stale_null_pre_2382": sum(1 for p in cooling if p.stale is None),
        "device_offline": sum(1 for p in cooling if not p.device_online),
        "controller_not_ready": sum(1 for p in cooling if not p.controller_ready),
        "watts_null": sum(1 for p in cooling if p.watts is None),
        "below_150w": sum(1 for p in cooling if p.watts is not None and p.watts < 150),
        "max_gap_sec": round(max(gaps), 1) if gaps else None,
        "gaps_over_300s": sum(1 for g in gaps if g > 300),
        "longest_identical_runs_rows": sorted(runs, reverse=True)[:3],
    }


INJECTIONS = {
    # mode: (expected open_reason, max seconds from the fault to the incident)
    "silence": ("no_samples", 300 + 30),       # orion-zwave / sql-writer stop writing rows
    "offline": ("device_offline", 300 + 30),   # the plug drops off Z-Wave (rows keep coming, not live)
    "stale": ("no_fresh_sample", 300 + 30),    # only cached, stale=true rows (post-#2382 producer)
    "zero": ("low_power", 180 + 30),           # compressor stops, plug reads ~0 W
    "frozen": ("frozen", 3600 + 30),           # the plug keeps repeating one wattage
}


def inject(cooling: list[CoolingPoint], at: datetime, mode: str) -> list[CoolingPoint]:
    """Real rows up to ``at``, then a synthetic fault on the same 5 s cadence for 70 min."""
    head = [p for p in cooling if p.ts < at]
    if mode == "silence":
        return head
    last = head[-1]
    tail, t = [], at
    while t <= at + timedelta(minutes=70):
        if mode == "offline":
            tail.append(CoolingPoint(t, last.watts, True, False, True))
        elif mode == "stale":
            tail.append(CoolingPoint(t, None, True, True, True))
        elif mode == "zero":
            tail.append(CoolingPoint(t, 0.0, False, True, True))
        else:  # frozen
            tail.append(CoolingPoint(t, last.watts, False, True, True))
        t += timedelta(seconds=5)
    return head + tail


def injection_stats(cooling, temps, cutoff) -> list[dict]:
    """Inject each fault at three real moments (post-#2382 rows) and time the incident."""
    s = _settings(HARDWARE_WATCH_HEAT_NODES="", HARDWARE_WATCH_GPU_NODES="")
    out = []
    for hours_before in (54, 30, 6):
        at = cutoff - timedelta(hours=hours_before)
        for mode, (want, max_sec) in INJECTIONS.items():
            res = replay(Watcher, s, inject(cooling, at, mode), temps, at - timedelta(minutes=70),
                         at + timedelta(minutes=66))
            incs = [i for i in _incidents(res.store) if i["rule"] == "cooling"]
            after = [i for i in incs if datetime.fromisoformat(i["opened_at"]) >= at]
            first = after[0] if after else None
            lat = (datetime.fromisoformat(first["opened_at"]) - at).total_seconds() if first else None
            out.append({"at": at.isoformat(), "mode": mode, "want": want, "got": first["open_reason"] if first else None,
                        "latency_sec": lat, "ok": bool(first) and first["open_reason"] == want and lat <= max_sec
                        and len(incs) == len(after) == 1,
                        "shed_reason": first["shed_reason"] if first else None})
    return out


# =============================================================================================
# Thermal controller v2: the 7-day hour-by-hour gate
# Spec: docs/superpowers/specs/2026-10-06-thermal-controller-redesign-design.md, Acceptance check 1,
# adapted to the APPROVED 34 C critical line (Decisions): the reflex sheds only at >= 34 C (or on
# sensor loss); 29.5-34 C is Orion's learned shed's band (D2/D8).
# =============================================================================================

V2_FIXTURE = HERE / "fixtures" / "thermal_v2_replay.csv.gz"
REFLEX_SHED_CHANNEL_FALLBACK = "orion:hardware:watch:reflex_shed"
CRITICAL_C = 34.0
ELEVATED_C = 29.5
EMAIL_WINDOW = timedelta(hours=6)
# The real week never reached 34 C (peak 33.5 C, 10-05 19:00). So the >= 34 C assertion is not
# vacuous, a second replay shifts 10-05's cabinet readings up by this much (peak 34.5 C).
HOT_DAY_SHIFT_C = 1.0


def _reflex_channel() -> str:
    try:
        from orion.schemas.hardware_watch import HARDWARE_WATCH_REFLEX_SHED_CHANNEL
        return HARDWARE_WATCH_REFLEX_SHED_CHANNEL
    except ImportError:   # pre-v2 code: no reflex channel exists, so the board only sees incidents
        return REFLEX_SHED_CHANNEL_FALLBACK


class PoolBoardSink(Sink):
    """Sink that also plays the pool's side of the shed board, the way services/orion-gpu-pool
    consumes it: v1 cooling incidents with ``shed.requested`` set ``cooling_incident``; v2 reflex
    signals set/clear their own reason. The board's own ``valid_until`` expiry applies."""

    def __init__(self, clock):
        super().__init__()
        from orion.gpu_pool.shed import ShedBoard, ShedSignal
        self._Sig = ShedSignal
        self.board = ShedBoard()
        self.clock = clock
        self.reflex: list[dict] = []
        self.reflex_channel = _reflex_channel()
        self.notice_at: dict[int, datetime] = {}

    def notify(self, req):
        self.notice_at[id(req)] = self.clock()
        return super().notify(req)

    async def publish(self, channel, env):
        await super().publish(channel, env)
        now = self.clock()
        p = env.payload
        if channel == HARDWARE_WATCH_INCIDENT_CHANNEL and p["rule"] == "cooling":
            shed = p.get("shed") or {}
            if p["status"] == "open" and shed.get("requested"):
                vu = datetime.fromisoformat(shed["valid_until"]) if shed.get("valid_until") else now + timedelta(seconds=300)
                self.board.set(self._Sig("cooling_incident", p["incident_id"], now, min(vu, now + timedelta(seconds=900))))
            else:
                self.board.clear("cooling_incident", p["incident_id"])
        elif channel == self.reflex_channel:
            from orion.schemas.hardware_watch import HardwareWatchReflexShedV1
            sig = HardwareWatchReflexShedV1.model_validate(p)   # the contract the pool consumes
            self.reflex.append(p)
            for reason in ("cabinet_hot", "cabinet_unknown"):
                if reason != sig.reason or not sig.active:
                    self.board.clear(reason, sig.source_id)
            if sig.active and sig.reason:
                self.board.set(self._Sig(sig.reason, sig.source_id, now,
                                         min(sig.valid_until, now + timedelta(seconds=900))))


def load_v2(path: Path = V2_FIXTURE):
    cooling: list[CoolingPoint] = []
    temps: dict[tuple[str, str], list[TempPoint]] = {}
    with gzip.open(path, "rt") as fh:
        head = fh.readline()
        since = datetime.fromisoformat(head.split("since=")[1].split()[0])
        cutoff = datetime.fromisoformat(head.split("cutoff=")[1].split()[0])
        for row in csv.reader(fh):
            if row[0] == "c":
                cooling.append(CoolingPoint(_ts(row[1]), float(row[2]) if row[2] else None, _bool(row[3]),
                                            bool(_bool(row[4])), bool(_bool(row[5]))))
            elif row[0] == "t" and row[4]:
                temps.setdefault((row[1], row[2]), []).append(TempPoint(_ts(row[3]), float(row[4])))
    cooling.sort(key=lambda p: p.ts)
    for pts in temps.values():
        pts.sort(key=lambda p: p.ts)
    return since, cutoff, cooling, temps


def _shift_day(temps, day: str, delta: float):
    out = dict(temps)
    out[("athena", "cabinet_temp_c")] = [
        TempPoint(p.ts, round(p.value + delta, 2)) if p.ts.strftime("%m-%d") == day else p
        for p in temps[("athena", "cabinet_temp_c")]]
    return out


def _health(w, store) -> dict:
    """What hardware-watch's /health would say right now (the learned shed reads this)."""
    last = w.last
    snap = getattr(w, "reflex_snapshot", None)
    return {"enabled": True, "last_tick_at": last.at.isoformat() if last.at else None,
            "last_tick_ok": last.ok, "open_incidents": [
                {"incident_id": r["incident_id"], "rule": r["rule"], "subject": r["subject"],
                 "open_reason": r["open_reason"], "shed_requested": r.get("shed_requested"),
                 "shed_reason": r.get("shed_reason")} for r in store.open_incidents()],
            **({"reflex_shed": snap()} if callable(snap) else {})}


def replay_v2(cooling, temps, start, end, **settings_kw):
    """One pass through the real Watcher, recording per tick: the pool board's blocked map, the
    cabinet reading, the learned shed's eligibility, and the open incidents."""
    from orion.autonomy.cabinet_heat import read_cabinet_heat
    from orion.autonomy.self_shed import HardwareWatchView, evaluate_shed_eligibility

    clock = Clock(start)
    store = ReplayStore(clock, cooling, temps)
    sink = PoolBoardSink(clock)
    s = _settings(**{"HARDWARE_WATCH_HEAT_CONTROLLER": "v2", **settings_kw})
    w = Watcher(settings=s, store=store, publish=sink.publish, notify=sink.notify, clock=clock, run_sync=_inline)
    ticks: list[dict] = []
    cab_key = ("athena", "cabinet_temp_c")

    async def run():
        while clock.t <= end:
            now = clock.t
            await w.tick()
            sink.board.prune(now)
            blocked = dict(sink.board.view(now, True).blocked)
            cab = store.temp_points(*cab_key, now - timedelta(minutes=30))
            reading = read_cabinet_heat(cab, now)
            hw = HardwareWatchView.from_health(_health(w, store), now=now)
            elig = evaluate_shed_eligibility(cabinet=reading, hardware_watch=hw, background_granted=1,
                                             background_queued=0, in_flight_episode_ids=[],
                                             holdback_fraction=0.5, now=now)
            ticks.append({"t": now, "blocked": blocked, "temp": reading.temp_c, "state": reading.thermal_state,
                          "eligible": elig["eligible"], "refusals": elig["refusals"],
                          "cooling_open": [r["open_reason"] for r in store.open_incidents() if r["rule"] == "cooling"]})
            clock.t += TICK

    asyncio.run(run())
    return SimpleNamespace(store=store, sink=sink, ticks=ticks, watcher=w)


def _hour(t: datetime) -> str:
    return t.strftime("%m-%d %H")


def thermal_v2_gate(path: Path = V2_FIXTURE) -> tuple[dict, list[str]]:
    since, cutoff, cooling, temps = load_v2(path)
    start = since + timedelta(minutes=70)   # the AC rule's lookback is filled
    res = replay_v2(cooling, temps, start, cutoff)
    fails: list[str] = []
    by_hour: dict[str, list[dict]] = {}
    for tk in res.ticks:
        by_hour.setdefault(_hour(tk["t"]), []).append(tk)
    cab = temps[("athena", "cabinet_temp_c")]
    hour_max: dict[str, float] = {}
    for p in cab:
        h = _hour(p.ts)
        hour_max[h] = max(hour_max.get(h, -1e9), p.value)
    incidents = _incidents(res.store)
    alerts = [n for n in res.sink.notices if n.severity == "critical"]

    # 1. 10-06 00:00-04:00: no shed, no cooling incident, no email.
    win = [tk for tk in res.ticks if datetime(2026, 10, 6, tzinfo=timezone.utc) <= tk["t"] < datetime(2026, 10, 6, 4, tzinfo=timezone.utc)]
    shed_win = [tk for tk in win if tk["blocked"]]
    open_win = [tk for tk in win if tk["cooling_open"]]
    mail_win = [n for n in alerts if n.context.get("rule") == "cooling" and
                _hour(_notice_at(res, n)) >= "10-06 00" and _hour(_notice_at(res, n)) < "10-06 04"]
    if shed_win:
        fails.append(f"cool night 10-06 00-04: shed active on {len(shed_win)} ticks (first {shed_win[0]['t'].isoformat()} "
                     f"{shed_win[0]['blocked']})")
    if open_win:
        fails.append(f"cool night 10-06 00-04: a cooling incident open on {len(open_win)} ticks "
                     f"({sorted(set(r for tk in open_win for r in tk['cooling_open']))})")
    if mail_win:
        fails.append(f"cool night 10-06 00-04: {len(mail_win)} cooling email(s)")

    # 2. 10-04/10-05 hour by hour: 29.5-34 C -> no reflex shed, learned shed eligible wherever the
    #    cabinet is elevated/hot; >= 34 C never happened on the real days (checked in 2b).
    hot_hours = []
    for h in sorted(by_hour):
        if not (h.startswith("10-04") or h.startswith("10-05")):
            continue
        mx = hour_max.get(h)
        if mx is None or mx < ELEVATED_C:
            continue
        tks = by_hour[h]
        if mx >= CRITICAL_C:
            hot_hours.append(h)
            continue
        reflex = [tk for tk in tks if tk["blocked"]]
        warm = [tk for tk in tks if tk["state"] in ("elevated", "hot")]
        not_elig = [tk for tk in warm if not tk["eligible"]]
        if reflex:
            fails.append(f"{h} (max {mx:.2f} C, below critical): reflex shed on {len(reflex)}/{len(tks)} ticks "
                         f"{reflex[0]['blocked']}")
        if not warm:
            fails.append(f"{h} (max {mx:.2f} C): cabinet never read elevated/hot")
        elif not_elig:
            why = Counter(r for tk in not_elig for r in tk["refusals"]).most_common(2)
            fails.append(f"{h} (max {mx:.2f} C): learned shed NOT eligible on {len(not_elig)}/{len(warm)} warm ticks {why}")
    if hot_hours:
        fails.append(f"real data reached >= {CRITICAL_C} C in {hot_hours}: re-check the gate's assumptions")

    # 2b. >= 34 C (10-05 shifted +1.0 C): every tick at >= 34 C has cabinet_hot within one tick, and it
    #     blocks background + system; it is gone once the cabinet is below the 33 C re-arm.
    shifted = _shift_day(temps, "10-05", HOT_DAY_SHIFT_C)
    d5 = datetime(2026, 10, 5, tzinfo=timezone.utc)
    hot = replay_v2(cooling, shifted, d5, d5 + timedelta(days=1))
    hot_ticks = [tk for tk in hot.ticks if tk["temp"] is not None and tk["temp"] >= CRITICAL_C]
    missed = [tk for i, tk in enumerate(hot.ticks) if tk in hot_ticks and
              hot.ticks[min(i + 1, len(hot.ticks) - 1)]["blocked"].get("system") != "cabinet_hot"]
    cool_shed = [tk for tk in hot.ticks if tk["temp"] is not None and tk["temp"] < CRITICAL_C - 1.0 and tk["blocked"]]
    shifted_hours = sorted({_hour(tk["t"]) for tk in hot_ticks})
    if not hot_ticks:
        fails.append("shifted 10-05 never reached 34 C (fixture changed?)")
    if missed:
        fails.append(f">= 34 C: cabinet_hot missing on {len(missed)}/{len(hot_ticks)} ticks (first {missed[0]['t'].isoformat()} "
                     f"{missed[0]['temp']} C blocked={missed[0]['blocked']})")
    if cool_shed:
        fails.append(f"shifted 10-05: reflex shed below the 33 C re-arm on {len(cool_shed)} ticks "
                     f"(first {cool_shed[0]['t'].isoformat()} {cool_shed[0]['temp']} C {cool_shed[0]['blocked']})")

    # 3. 10-03 outage (AC + cabinet silent 20:44-21:14 and 21:23-22:23): an alert fires; the shed is at
    #    most cabinet_unknown (background only); it lapses within 3 ticks of readings resuming.
    o0, o1 = datetime(2026, 10, 3, 20, 40, tzinfo=timezone.utc), datetime(2026, 10, 3, 23, 0, tzinfo=timezone.utc)
    out_alerts = [n for n in alerts if o0 <= _notice_at(res, n) <= o1]
    out_ticks = [tk for tk in res.ticks if o0 <= tk["t"] <= o1]
    too_much = [tk for tk in out_ticks if set(tk["blocked"]) - {"background"} or
                any(r != "cabinet_unknown" for r in tk["blocked"].values())]
    if not out_alerts:
        fails.append("10-03 outage: no alert")
    if too_much:
        fails.append(f"10-03 outage: shed beyond cabinet_unknown/background on {len(too_much)} ticks "
                     f"(first {too_much[0]['t'].isoformat()} {too_much[0]['blocked']})")
    if not any(tk["blocked"].get("background") == "cabinet_unknown" for tk in out_ticks):
        fails.append("10-03 outage: cabinet_unknown never shed background while the sensor was silent")
    for resume in (datetime(2026, 10, 3, 21, 14, 39, tzinfo=timezone.utc), datetime(2026, 10, 3, 22, 23, 48, tzinfo=timezone.utc)):
        after = [tk for tk in res.ticks if tk["t"] >= resume + 3 * TICK and tk["t"] <= resume + 10 * TICK]
        stuck = [tk for tk in after if tk["blocked"]]
        if stuck:
            fails.append(f"10-03 outage: shed still active {(stuck[0]['t'] - resume).total_seconds():.0f}s after readings "
                         f"resumed at {resume.strftime('%H:%M:%S')} ({stuck[0]['blocked']})")

    # 4. gpu_heat: zero incidents for 73-80 C (only the 85 C ceiling opens one).
    gpu_bad = [i for i in incidents if i["rule"] == "gpu_heat" and i["open_reason"] != "above_ceiling"]
    if gpu_bad:
        fails.append(f"gpu_heat: {len(gpu_bad)} incident(s) below the ceiling: "
                     f"{[(i['subject'], i['open_reason'], i['opened_at'][:16]) for i in gpu_bad[:4]]}")
    cpu_bad = [i for i in incidents if i["rule"] == "cpu_heat" and i["open_reason"] != "above_ceiling"]
    if cpu_bad:
        fails.append(f"cpu_heat: {len(cpu_bad)} incident(s) below the ceiling: "
                     f"{[(i['subject'], i['open_reason'], i['opened_at'][:16]) for i in cpu_bad[:4]]}")

    # 5. emails: at most one per rule+subject per 6 h (sliding).
    sent: dict[tuple, list[datetime]] = {}
    for n in alerts:
        sent.setdefault((n.context.get("rule"), n.context.get("subject")), []).append(_notice_at(res, n))
    for key, ts in sent.items():
        ts.sort()
        close = [(a, b) for a, b in zip(ts, ts[1:]) if b - a < EMAIL_WINDOW]
        if close:
            fails.append(f"emails {key}: {len(close)} pair(s) < 6 h apart (first {close[0][0].isoformat()} -> "
                         f"{close[0][1].isoformat()})")
    urgent = Counter((u.get("trigger"), u.get("subject")) for u in res.sink.urgent)

    n = len(res.ticks)
    share = lambda pred: round(100 * sum(1 for tk in res.ticks if pred(tk)) / n, 2)  # noqa: E731
    report = {
        "fixture": str(path.name), "since": since.isoformat(), "cutoff": cutoff.isoformat(), "ticks": n,
        "calibration_pct_of_ticks": {
            "temp_ge_29_5": share(lambda tk: tk["temp"] is not None and tk["temp"] >= ELEVATED_C),
            "temp_ge_32": share(lambda tk: tk["temp"] is not None and tk["temp"] >= 32.0),
            "temp_ge_34": share(lambda tk: tk["temp"] is not None and tk["temp"] >= CRITICAL_C),
            "reflex_shed_any": share(lambda tk: bool(tk["blocked"])),
            "learned_shed_eligible": share(lambda tk: tk["eligible"]),
        },
        "incidents": incidents,
        "alerts": [(n.context.get("rule"), n.context.get("subject"), _notice_at(res, n).isoformat()) for n in alerts],
        "urgent_requests": dict(Counter(f"{k[0]}:{k[1]}" for k in urgent.elements())),
        "reflex_signals": len(res.sink.reflex),
        "shifted_10_05": {"hot_ticks": len(hot_ticks), "hours_ge_34": shifted_hours,
                          "reflex_ticks": sum(1 for tk in hot.ticks if tk["blocked"])},
    }
    return report, fails


def _notice_at(res, n) -> datetime:
    """Notices carry no timestamp; the sink stamps them on arrival (PoolBoardSink.notify)."""
    return res.sink.notice_at[id(n)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true", help="refresh the fixture from production Postgres")
    ap.add_argument("--cutoff", help="export cutoff (UTC ISO); default now rounded down to the hour")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="show the watcher's own transition logs")
    ap.add_argument("--gate-only", action="store_true", help="only the thermal-v2 7-day gate")
    ap.add_argument("--legacy-only", action="store_true", help="only the 2026-09-30 v1-rules checks")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.ERROR)
    if not args.verbose:
        logging.getLogger("orion-hardware-watch").setLevel(logging.ERROR)
    if args.export:
        cutoff = (datetime.fromisoformat(args.cutoff) if args.cutoff
                  else datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0))
        export(cutoff.astimezone(timezone.utc))
        return 0

    gate_report, gate_fails = (None, []) if args.legacy_only else thermal_v2_gate()
    if args.gate_only:
        if args.json:
            print(json.dumps({"thermal_v2_gate": gate_report, "failures": gate_fails}, indent=2, default=str))
        else:
            _print_gate(gate_report, gate_fails)
        return 0 if not gate_fails else 1

    cutoff, cooling, temps = load()
    s = _settings(HARDWARE_WATCH_HEAT_NODES="", HARDWARE_WATCH_GPU_NODES="")
    ac = replay(Watcher, s, cooling, temps, cooling[0].ts, cutoff)
    heat_start = cutoff - timedelta(days=7)
    heat = replay(NoCooling, _settings(), cooling, temps, heat_start, cutoff)

    ac_inc = [i for i in _incidents(ac.store) if i["rule"] == "cooling"]
    heat_inc = [i for i in _incidents(heat.store) if i["rule"] != "cooling"]
    heat_days = (cutoff - heat_start).total_seconds() / 86400
    per_subject = Counter(f"{i['rule']}:{i['subject']}" for i in heat_inc)
    cabinet = temps.get(("athena", "cabinet_temp_c"), [])
    shed_stats = shed_leg_stats(cabinet, heat_start, cutoff, ShedRuleConfig())
    gpu_keys = sorted(k for (_, k) in temps if k.startswith("gpu"))
    first_real = next((p.ts for p in cooling if p.watts is not None and p.watts >= 500), None)

    report = {
        "cutoff": cutoff.isoformat(),
        "cooling_data": cooling_facts(cooling),
        "ac_replay": {"ticks": ac.ticks, "incidents": ac_inc, "alerts": len(ac.sink.notices),
                      "urgent_requests": len(ac.sink.urgent), "incident_events": len(ac.sink.events),
                      "shed_ticks": ac.shed_ticks,
                      "first_watts_ge_500": first_real.isoformat() if first_real else None},
        "heat_replay": {"days": round(heat_days, 2), "incidents": heat_inc,
                        "per_subject": dict(per_subject),
                        "per_subject_per_day": {k: round(v / heat_days, 2) for k, v in per_subject.items()},
                        "open_ticks_pct": {k: round(100 * v / heat.ticks, 2) for k, v in heat.open_ticks.items()},
                        "urgent_requests": len(heat.sink.urgent), "gpu_temp_keys_in_history": gpu_keys},
        "shed_rule_any_tick": shed_stats,
    }

    failures: list[str] = []
    # Acceptance 3: the AC rule fires on the real 33.11 W (kWh-counter bug) stretch, once, and nowhere else.
    if len(ac_inc) != 1:
        failures.append(f"expected exactly 1 AC incident, got {len(ac_inc)}")
    else:
        inc = ac_inc[0]
        opened = datetime.fromisoformat(inc["opened_at"])
        if inc["open_reason"] != "low_power":
            failures.append(f"AC incident opened as {inc['open_reason']}, expected low_power")
        if (opened - cooling[0].ts).total_seconds() > 210:
            failures.append(f"AC incident opened {(opened - cooling[0].ts).total_seconds():.0f}s after the first row (>210s)")
        if inc["resolved_at"] is None:
            failures.append("AC incident never resolved")
        elif first_real and (datetime.fromisoformat(inc["resolved_at"]) - first_real).total_seconds() < 600:
            failures.append("AC incident resolved < 10 min after real watts returned")
        if ac.sink.notices and ac.sink.notices[0].severity != "critical":
            failures.append("first AC notice is not critical")
    if len(ac.sink.urgent) != len(ac_inc):
        failures.append(f"urgent requests {len(ac.sink.urgent)} != AC incidents {len(ac_inc)}")
    # Shed only ever rides an OPEN cooling incident.
    for ev in ac.sink.events + heat.sink.events:
        shed = ev.get("shed") or {}
        if shed.get("requested") and (ev["rule"] != "cooling" or ev["status"] != "open"):
            failures.append(f"shed requested on {ev['rule']}/{ev['status']} event")
            break
    if heat.shed_ticks:
        failures.append("heat replay requested shedding (no cooling incident can be open there)")
    # Spec acceptance 4 (gate counts: p95 ~0.1/day athena, ~1.3/day circe). Loose bound: an
    # incident rate far above the gate means the rule or baseline is broken, not a hot week.
    for subj, rate in report["heat_replay"]["per_subject_per_day"].items():
        if subj.startswith("cpu_heat") and rate > 3.0:
            failures.append(f"{subj} opens {rate}/day (> 3/day): baseline or sustain broken")
    # GPU p95 arm must stay disarmed with < 3 days of GPU history.
    gpu_hist_days = 0.0
    for (n, k), pts in temps.items():
        if k.startswith("gpu") and pts:
            gpu_hist_days = max(gpu_hist_days, (cutoff - pts[0].ts).total_seconds() / 86400)
    if gpu_hist_days < 3 and any(i["open_reason"] == "above_p95" and i["rule"] == "gpu_heat" for i in heat_inc):
        failures.append("GPU p95 arm fired with < 3 days of GPU history")
    report["gpu_history_days"] = round(gpu_hist_days, 2)
    inj = injection_stats(cooling, temps, cutoff)
    report["fault_injection"] = inj
    for r in inj:
        if not r["ok"]:
            failures.append(f"injected {r['mode']} at {r['at']}: want {r['want']}, got {r['got']} after {r['latency_sec']}s")
    report["thermal_v2_gate"] = gate_report
    failures += [f"v2 gate: {f}" for f in gate_fails]
    report["failures"] = failures

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        cd = report["cooling_data"]
        print(f"cutoff {report['cutoff']}")
        print(f"AC data: {cd['rows']} rows {cd['first']} -> {cd['last']}; stale=true {cd['stale_true']}, "
              f"stale unknown (pre-#2382) {cd['stale_null_pre_2382']}, offline {cd['device_offline']}, "
              f"controller not ready {cd['controller_not_ready']}, null watts {cd['watts_null']}, "
              f"<150 W {cd['below_150w']}, max gap {cd['max_gap_sec']} s, gaps>300 s {cd['gaps_over_300s']}")
        print(f"AC replay: {len(ac_inc)} incident(s), {report['ac_replay']['alerts']} notices, "
              f"{report['ac_replay']['urgent_requests']} urgent requests, shed ticks {ac.shed_ticks}")
        for i in ac_inc:
            print(f"  {i['open_reason']} {i['opened_at']} -> {i['resolved_at']} ({i['minutes']} min) "
                  f"shed={i['shed_requested']}/{i['shed_reason']}")
        print(f"Heat replay ({report['heat_replay']['days']} d): {len(heat_inc)} incident(s); per day "
              f"{report['heat_replay']['per_subject_per_day']}; open time % {report['heat_replay']['open_ticks_pct']}; "
              f"gpu keys {gpu_keys or 'none'} ({report['gpu_history_days']} d)")
        for i in heat_inc:
            print(f"  {i['rule']}:{i['subject']} {i['open_reason']} {i['opened_at']} ({i['minutes']} min)")
        ss = report["shed_rule_any_tick"]
        print(f"Shed rule at any tick ({ss['days']} d, AC incident NOT required): {ss['pct_of_ticks']}; "
              f"reasons {ss['reasons_pct']}; rise>=1.0 episodes {ss['rise_ge_1_0_episodes']} "
              f"({ss['rise_ge_1_0_episodes_per_day']}/day)")
        print("Fault injection on real history (fault -> incident):")
        for r in report["fault_injection"]:
            print(f"  {r['at']} {r['mode']:8s} -> {r['got']} after {r['latency_sec']}s shed={r['shed_reason']} "
                  f"{'ok' if r['ok'] else 'FAIL'}")
        if gate_report is not None:
            _print_gate(gate_report, [])
        print("PASS" if not failures else "FAIL:\n  " + "\n  ".join(failures))
    return 0 if not failures else 1


def _print_gate(rep: dict | None, fails: list[str]) -> None:
    if rep is None:
        return
    print(f"Thermal v2 gate ({rep['fixture']}, {rep['since']} -> {rep['cutoff']}, {rep['ticks']} ticks):")
    print(f"  calibration % of ticks: {rep['calibration_pct_of_ticks']}")
    print(f"  incidents: {len(rep['incidents'])}; alerts: {rep['alerts']}; urgent: {rep['urgent_requests']}; "
          f"reflex signals sent: {rep['reflex_signals']}")
    for i in rep["incidents"]:
        print(f"    {i['rule']}:{i['subject']} {i['open_reason']} {i['opened_at'][:19]} ({i['minutes']} min)")
    print(f"  shifted 10-05 (+{HOT_DAY_SHIFT_C} C): {rep['shifted_10_05']}")
    if fails:
        print("FAIL:\n  " + "\n  ".join(fails))


if __name__ == "__main__":
    raise SystemExit(main())
