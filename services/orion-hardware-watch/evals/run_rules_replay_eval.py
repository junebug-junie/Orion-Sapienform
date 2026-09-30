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
    base = dict(ORION_BUS_URL="redis://replay", POSTGRES_URI="replay", HARDWARE_WATCH_ENABLED=True,
                HARDWARE_WATCH_SHED_ENABLED=True, HARDWARE_WATCH_URGENT_ENABLED=True)
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true", help="refresh the fixture from production Postgres")
    ap.add_argument("--cutoff", help="export cutoff (UTC ISO); default now rounded down to the hour")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="show the watcher's own transition logs")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.ERROR)
    if not args.verbose:
        logging.getLogger("orion-hardware-watch").setLevel(logging.ERROR)
    if args.export:
        cutoff = (datetime.fromisoformat(args.cutoff) if args.cutoff
                  else datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0))
        export(cutoff.astimezone(timezone.utc))
        return 0

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
        print("PASS" if not failures else "FAIL:\n  " + "\n  ".join(failures))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
