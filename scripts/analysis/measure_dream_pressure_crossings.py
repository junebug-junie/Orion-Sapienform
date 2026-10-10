#!/usr/bin/env python3
"""Read saved dream pressure alongside actual cycles; never run/recompute a dream.

JSONL input is the minimal projection printed by --print-sql. The store saves
pressure only on cycles in legacy exports. --with-checks includes the new real
check observations; missing checks never establish reset or recovery.

R2a (PR #2369 rev 4) counterfactual: --with-sources exports the four sleep
sources as key/weight/timestamp only (no text) plus chat timestamps; --replay
then re-runs the dream service's own candidate builders and compute_pressure at
every check over the window, as the sources stand now. It records when pressure
first crosses each candidate threshold after each real sleep, validates against
saved check observations, and simulates the schedule each threshold would give.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
from statistics import median
import bisect
import importlib.util
import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def utc(value):
    dt = datetime.fromisoformat(value) if isinstance(value, str) else value
    if dt.tzinfo is None:
        raise ValueError("timestamps must include a timezone")
    return dt.astimezone(timezone.utc)


def export_sql(start, end, *, with_checks=False, with_sources=False, lookback_hours=48.0):
    start, end = utc(start), utc(end)
    if end <= start:
        raise ValueError("end must follow start")
    # Sources reach back two lookbacks: the window (<= lookback) plus its prior.
    sources_sql = source_export_sql(
        (start - timedelta(hours=2 * lookback_hours)).isoformat(), end.isoformat()) if with_sources else ""
    checks_sql = f"""
SELECT jsonb_build_object('kind', 'pressure_check', 'observation', observation_json)
FROM dream_pressure_observation
WHERE observed_at >= '{start.isoformat()}'::timestamptz
  AND observed_at < '{end.isoformat()}'::timestamptz ORDER BY observed_at;
""" if with_checks else ""
    return f"""BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY;
SET LOCAL statement_timeout = '60s';
SET LOCAL TIME ZONE 'UTC';
SELECT row_to_json(r) FROM (
 SELECT cycle_id, started_at, ended_at, trigger, status,
        cycle_json->'pressure' AS reading
 FROM dream_cycle
 WHERE started_at < '{end.isoformat()}'::timestamptz
   AND (started_at >= '{start.isoformat()}'::timestamptz OR cycle_id = (
     SELECT cycle_id FROM dream_cycle WHERE started_at < '{start.isoformat()}'::timestamptz
     ORDER BY started_at DESC LIMIT 1) OR cycle_id = (
     SELECT cycle_id FROM dream_cycle WHERE started_at < '{start.isoformat()}'::timestamptz
       AND status <> 'failed' ORDER BY started_at DESC LIMIT 1))
 ORDER BY started_at, cycle_id
) r;
{checks_sql}{sources_sql}
ROLLBACK;"""


# Mirrors services/orion-dream/app/cycle_store.py SOURCE_QUERIES (key rule and
# filters). tests/test_regulation_measurements.py fails if those drift apart.
METACOG_KEY_RE = r"[0-9a-f]{8,}|-?[0-9]+(\.[0-9]+)?"


def source_export_sql(lo, hi):
    """Text-free source rows: md5 of each dedupe key, never theme or summary text."""
    return f"""SELECT row_to_json(r) FROM (
 SELECT 'source' AS kind, 'metacog' AS source_kind, ts AS at, id::text AS id, severity, trigger_kind,
        md5(lower(dedupe_key)) AS dedupe_key, has_text FROM (
   SELECT id, severity, trigger_kind, btrim(coalesce(summary, '')) <> '' AS has_text,
          COALESCE(regexp_replace(NULLIF(trigger_reason, ''), '{METACOG_KEY_RE}', '#', 'g'),
                   'id:' || id) AS dedupe_key,
          CASE WHEN timestamp ~ '^\\d{{4}}-\\d{{2}}-\\d{{2}}T' THEN CAST(timestamp AS timestamptz) END AS ts
     FROM orion_metacog WHERE severity IN ('degraded', 'critical')
 ) m WHERE ts > '{lo}'::timestamptz AND ts < '{hi}'::timestamptz
 UNION ALL
 SELECT 'source', 'compaction_request', created_at, request_id::text, NULL, NULL,
        md5(lower(theme)), btrim(coalesce(theme, '')) <> ''
   FROM dream_compaction_request_queue
  WHERE created_at > '{lo}'::timestamptz AND created_at < '{hi}'::timestamptz AND theme IS NOT NULL
 UNION ALL
 SELECT 'source', 'resonance', created_at, alert_id::text, violation_count::text, NULL,
        md5(lower(theme_key)), btrim(coalesce(theme_key, '')) <> ''
   FROM substrate_reverie_resonance_alert
  WHERE created_at > '{lo}'::timestamptz AND created_at < '{hi}'::timestamptz AND theme_key IS NOT NULL
 UNION ALL
 SELECT 'source', 'crystallization', h.created_at, c.crystallization_id::text, c.salience::text, NULL,
        md5(c.crystallization_id::text),
        btrim(coalesce(c.subject, '')) <> '' OR btrim(coalesce(c.summary, '')) <> ''
   FROM memory_crystallization_history h JOIN memory_crystallizations c USING (crystallization_id)
  WHERE h.op IN ('auto_activate', 'approve') AND c.status = 'active'
    AND h.created_at > '{lo}'::timestamptz AND h.created_at < '{hi}'::timestamptz
) r;
SELECT row_to_json(r) FROM (
 SELECT 'chat' AS kind, created_at AT TIME ZONE 'UTC' AS at,
        source = 'hub_orion' AND btrim(coalesce(prompt, '')) <> '' AS juniper
   FROM chat_history_log WHERE created_at < '{hi}'::timestamptz
    AND (created_at >= '{lo}'::timestamptz
         OR id IN (SELECT id FROM chat_history_log WHERE created_at < '{lo}'::timestamptz
                   ORDER BY created_at DESC LIMIT 1)
         OR id IN (SELECT id FROM chat_history_log WHERE created_at < '{lo}'::timestamptz
                   AND source = 'hub_orion' AND btrim(coalesce(prompt, '')) <> ''
                   ORDER BY created_at DESC LIMIT 1))
) r;
"""


def check_history(checks, cycles, start, end):
    """Use only real observations. Failed source reads break continuity."""
    checks = sorted((c for c in checks if start <= utc(c["observed_at"]) < end),
                    key=lambda c: utc(c["observed_at"]))
    if len({c["check_id"] for c in checks}) != len(checks):
        raise ValueError("duplicate pressure check")
    valid, invalid, gaps, zero, eligible_low = 0, 0, 0, 0, 0
    examples, discharged_windows, upward = [], set(), []
    previous = None
    for c in checks:
        p = c["reading"]
        if c["source_errors"]:
            invalid += 1
            previous = None
            discharged_windows.clear()
            continue
        value = float(p["pressure"])
        if not math.isfinite(value) or value < 0:
            raise ValueError("invalid check pressure")
        valid += 1
        zero += value == 0
        last_end = utc(c["last_attempt_end"]) if c["last_attempt_end"] else None
        timer_clear = last_end is None or (utc(c["observed_at"])-last_end).total_seconds() >= c["min_interval_hours"]*3600
        eligible_low += (not c["forced"] and timer_clear and p["idle_minutes"] is not None
                         and p["idle_minutes"] >= p["idle_required_minutes"] and value < p["threshold"])
        window = c["last_window_start"]
        if previous:
            gap = (utc(c["observed_at"])-utc(previous["observed_at"])).total_seconds()
            continuous = gap <= 2*max(c["check_interval_sec"], previous["check_interval_sec"])
            gaps += not continuous
            same_formula = c["formula"] == previous["formula"]
            if continuous and same_formula:
                old = previous["reading"]["pressure"]
                same_window = (window is not None and previous["last_window_start"] is not None
                               and utc(previous["last_window_start"]) == utc(window))
                if same_window and old < p["threshold"] <= value:
                    upward.append(dict(at=c["observed_at"], before=old, after=value, threshold=p["threshold"]))
                successful_sleep = any(
                    r["status"] in {"completed", "empty"} and window is not None
                    and utc(r["started_at"]) == utc(window) and r.get("ended_at")
                    and utc(previous["observed_at"]) <= utc(r["started_at"])
                    and utc(r["ended_at"]) <= utc(c["observed_at"]) for r in cycles)
                if value < old and successful_sleep:
                    discharged_windows.add(utc(window))
                    examples.append(dict(kind="fall_after_cycle", at=c["observed_at"], before=old, after=value))
                elif (value > old and window is not None and utc(window) in discharged_windows
                      and previous["last_window_start"] is not None
                      and utc(previous["last_window_start"]) == utc(window)):
                    examples.append(dict(kind="rise_after_observed_fall", at=c["observed_at"], before=old, after=value))
            else:
                discharged_windows.clear()
        previous = c
    return dict(samples=len(checks), valid_samples=valid, source_failed_samples=invalid,
                cadence_gaps=gaps, zero_samples=zero, idle_timer_clear_below_threshold=eligible_low,
                observed_fall_and_rise=any(e["kind"] == "rise_after_observed_fall" for e in examples),
                upward_threshold_crossings=upward[:20], upward_threshold_crossing_count=len(upward),
                examples=examples[:20])


def report(rows, start, end, *, interval_hours=6.0, check_seconds=600.0, legacy_cap=50):
    start, end = utc(start), utc(end)
    if end <= start or not math.isfinite(interval_hours) or interval_hours <= 0:
        raise ValueError("invalid window or minimum interval")
    if not math.isfinite(check_seconds) or check_seconds <= 0 or legacy_cap <= 0:
        raise ValueError("invalid cadence or legacy cap")
    checks = [r["observation"] for r in rows if r.get("kind") == "pressure_check"]
    rows = sorted((r for r in rows if r.get("kind") not in {"pressure_check", "source", "chat"}), key=lambda r: utc(r["started_at"]))
    if len({r["cycle_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate cycle_id")
    selected, previous_end = [], None
    for row in rows:
        at = utc(row["started_at"])
        ended = utc(row["ended_at"]) if row.get("ended_at") else None
        if ended is not None and ended < at:
            raise ValueError("cycle ends before it starts")
        if at >= end:
            break
        if at >= start:
            p = row["reading"]
            value, threshold = float(p["pressure"]), float(p["threshold"])
            if not all(math.isfinite(v) and v >= 0 for v in (value, threshold)):
                raise ValueError("invalid stored pressure")
            novelty = "new_counts" in p
            gap = (at-previous_end).total_seconds() if previous_end is not None else None
            delay = gap - interval_hours*3600 if gap is not None else None
            counts = p.get("counts", {})
            # The old four loaders each capped at 50; compaction weighs 0.5,
            # the other sources at most 1. Do not apply this ceiling to novelty.
            ceiling = legacy_cap * 3.5 if not novelty else None
            selected.append(dict(cycle_id=row["cycle_id"], started_at=at.isoformat(),
                ended_at=ended.isoformat() if ended else None, status=row["status"], trigger=row["trigger"],
                formula="novelty" if novelty else "legacy",
                pressure=value, threshold=threshold, threshold_ratio=value/threshold if threshold else None,
                counts=counts, new_counts=p.get("new_counts"), idle_minutes=p.get("idle_minutes"),
                hours_since_previous_attempt_end=gap/3600 if gap is not None else None,
                seconds_after_timer=delay,
                timer_aligned=(0 <= delay <= check_seconds) if delay is not None else None,
                legacy_ceiling=ceiling, fraction_of_legacy_ceiling=value/ceiling if ceiling else None,
                legacy_sources_at_cap=[k for k, n in counts.items() if n >= legacy_cap] if not novelty else None))
        if ended is not None:
            previous_end = max(previous_end, ended) if previous_end else ended
    versions = {}
    for version in ("legacy", "novelty"):
        values = [r for r in selected if r["formula"] == version]
        automatic = [r for r in values if r["trigger"] == "pressure" and r["seconds_after_timer"] is not None]
        pressures = [r["pressure"] for r in values]
        gaps = [r["hours_since_previous_attempt_end"] for r in automatic]
        versions[version] = dict(cycles=len(values), statuses=dict(Counter(r["status"] for r in values)),
            min_pressure=min(pressures) if pressures else None, max_pressure=max(pressures) if pressures else None,
            below_threshold_at_cycle=sum(r["pressure"] < r["threshold"] for r in values),
            comparable_automatic_cycles=len(automatic),
            comparable_automatic_below_threshold=sum(r["pressure"] < r["threshold"] for r in automatic),
            timer_aligned_cycles=sum(r["timer_aligned"] for r in automatic),
            median_gap_hours=median(gaps) if gaps else None)
    legacy = versions["legacy"]
    consistent = (legacy["comparable_automatic_cycles"] > 0
                  and legacy["timer_aligned_cycles"] == legacy["comparable_automatic_cycles"]
                  and legacy["comparable_automatic_below_threshold"] == 0)
    curve = check_history(checks, rows, start, end)
    return dict(start=start.isoformat(), end=end.isoformat(), cycles=selected, by_formula=versions,
        interval_hours_assumed=interval_hours, check_seconds_assumed=check_seconds,
        legacy_candidate_cap_assumed=legacy_cap,
        legacy_timing_consistent_with_timer=consistent,
        independent_check_samples=curve["samples"], check_history=curve,
        verdict=("Observed pressure fall after a completed cycle and a later rise; see check coverage."
                 if curve["observed_fall_and_rise"] else
                 "UNVERIFIED: available history has not established pressure falling and rising between dreams"),
        calibration="No automated threshold recommendation; inspect actual check coverage and source validity.",
        limits=["Do not interpolate missing checks or call cycle-to-cycle variation discharge/recovery.",
                "Manual and failed cycles remain visible; every attempt end affects the timer.",
                "new_counts distinguishes formula families, not every deployed revision.",
                "Timer/cadence and legacy cap are explicit assumptions; historical settings were not saved."])


def _service_replay():
    """The dream service's own pure module, loaded by path (no settings, no DB)."""
    name = "_orion_dream_replay"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, REPO / "services/orion-dream/app/replay.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def prepare_items(source_rows):
    """One (ts, key, kind, weight, rank) per row. weight is None when the service's
    own builder rejects the row (it still counts as a prior key, as in prior_keys).
    rank mirrors each SOURCE_QUERIES DISTINCT ON order: the row the service keeps."""
    svc = _service_replay()
    items = []
    for r in source_rows:
        kind = r["source_kind"]
        text = "x" if r.get("has_text") else ""
        row = dict(dedupe_key=r["dedupe_key"])
        rank = 0
        if kind == "metacog":
            row.update(id=r["id"], severity=r["severity"], summary=text, trigger_kind=r.get("trigger_kind"))
            rank = int(r["severity"] == "critical")
        elif kind == "compaction_request":
            row.update(request_id=r["id"], theme=text)
        elif kind == "resonance":
            row.update(alert_id=r["id"], theme_key=text, violation_count=int(r["severity"] or 0))
            rank = int(r["severity"] or 0)
        elif kind == "crystallization":
            row.update(crystallization_id=r["id"], subject=text, summary=text,
                       salience=float(r["severity"]) if r["severity"] is not None else None)
        else:
            raise ValueError(f"unknown source kind {kind!r}")
        key = svc.row_key(kind, row)
        if key is None:
            continue
        item = svc._BUILDERS[kind](row)
        items.append((utc(r["at"]), key, kind, item.weight if item is not None else None, rank))
    items.sort(key=lambda x: x[0])
    return items


class _Keyed:
    __slots__ = ("source_kind", "weight")

    def __init__(self, kind, weight):
        self.source_kind, self.weight = kind, weight


KEYS_PER_SOURCE = 5000  # services/orion-dream/app/cycle.py; live SQL keeps the newest this many


def pressure_at(items, times, now, since, lookback_hours=48.0):
    """compute_pressure exactly as read_pressure calls it: one kept row per key in
    (since, now) by the SQL's DISTINCT ON order, minus keys seen in the lookback before."""
    svc = _service_replay()
    def rows(lo, hi):
        return items[bisect.bisect_right(times, lo):bisect.bisect_left(times, hi)]
    kept = {}
    for ts, key, kind, weight, rank in rows(since, now):
        best = kept.get(key)
        if best is None or (rank, ts) >= (best[0], best[1]):
            kept[key] = (rank, ts, kind, weight)
    per_source = Counter(kind for _, _, kind, _ in kept.values())
    if any(n > KEYS_PER_SOURCE for n in per_source.values()):
        raise ValueError("a source exceeds KEYS_PER_SOURCE; the live LIMIT would drop keys this replay keeps")
    keyed = {k: _Keyed(kind, w) for k, (_, _, kind, w) in kept.items() if w is not None}
    seen = {key for _, key, _, _, _ in rows(since - timedelta(hours=lookback_hours), since)}
    return svc.compute_pressure(keyed, seen)


def _idle_minutes(chat_times, now):
    """Current IDLE_MINUTES_SQL: minutes since ANY chat row. None if none yet."""
    i = bisect.bisect_right(chat_times, now)
    return (now - chat_times[i - 1]).total_seconds() / 60 if i else None


def replay_novelty(rows, start, end, *, thresholds=(1, 2, 3, 4, 5, 8, 13, 21), check_seconds=600.0,
                   lookback_hours=48.0, interval_hours=6.0, idle_required=45.0, cycle_minutes=6.0,
                   formula_boundary=None):
    """R2a: replay the current (novelty) pressure at every check, as sources stand now.

    Real sleeps reset the window (sleep_windows); simulate() asks what each
    threshold would have scheduled instead. Limits: sources are read as they
    stand at export time, so rows later deleted or crystallizations later
    deactivated are invisible; idle uses every chat row like the live gate.
    """
    start, end = utc(start), utc(end)
    if end <= start or check_seconds <= 0:
        raise ValueError("invalid window or cadence")
    items = prepare_items([r for r in rows if r.get("kind") == "source"])
    times = [x[0] for x in items]
    chat_all = sorted(utc(r["at"]) for r in rows if r.get("kind") == "chat")
    chat_juniper = sorted(utc(r["at"]) for r in rows if r.get("kind") == "chat" and r.get("juniper") is True)
    cycles = sorted((r for r in rows if r.get("kind") not in {"source", "chat", "pressure_check"}),
                    key=lambda r: utc(r["started_at"]))
    good_starts = [utc(c["started_at"]) for c in cycles if c["status"] != "failed"]
    grid, at = [], start
    while at < end:
        grid.append(at)
        at += timedelta(seconds=check_seconds)

    def window_since(now, last):
        floor = now - timedelta(hours=lookback_hours)
        return max(floor, last) if last else floor

    # 1. Validation against what the live service saved at the same instants.
    validation = []
    for c in (r["observation"] for r in rows if r.get("kind") == "pressure_check"):
        now = utc(c["observed_at"])
        if not start <= now < end or c["source_errors"]:
            continue
        last = utc(c["last_window_start"]) if c["last_window_start"] else None
        value, _, _ = pressure_at(items, times, now, window_since(now, last), lookback_hours)
        validation.append(abs(value - float(c["reading"]["pressure"])))

    # 2. Real-sleep windows: when does replayed pressure first cross each threshold?
    windows = []
    for i, sleep_at in enumerate(good_starts):
        nxt = good_starts[i + 1] if i + 1 < len(good_starts) else end
        if nxt <= start or sleep_at >= end:
            continue
        series = [(t, pressure_at(items, times, t, window_since(t, sleep_at), lookback_hours)[0])
                  for t in grid if sleep_at < t < nxt]
        crossings = {}
        for th in thresholds:
            hit = next((t for t, v in series if v >= th), None)
            crossings[str(th)] = (hit - sleep_at).total_seconds() / 3600 if hit else None
        era = ("after_2557" if formula_boundary and sleep_at >= utc(formula_boundary) else "before_2557")
        windows.append(dict(sleep_started_at=sleep_at.isoformat(), next_sleep_at=nxt.isoformat(), era=era,
                            hours_to_next_sleep=(nxt - sleep_at).total_seconds() / 3600,
                            checks=len(series), min_pressure=min((v for _, v in series), default=None),
                            max_pressure=max((v for _, v in series), default=None),
                            zero_checks=sum(v == 0 for _, v in series),
                            hours_to_first_crossing=crossings))

    # 3. Simulate the schedule each threshold would have produced (current gate order).
    def simulate(threshold, chat_times):
        last_start = next((s for s in reversed(good_starts) if s <= start), None)
        last_end = last_start + timedelta(minutes=cycle_minutes) if last_start else None
        sleeps, held_low, held_not_idle, gaps = [], 0, 0, []
        for t in grid:
            if last_end and t - last_end < timedelta(hours=interval_hours):
                continue
            since = window_since(t, last_start)
            value, counts, _ = pressure_at(items, times, t, since, lookback_hours)
            idle = _idle_minutes(chat_times, t)
            is_idle = idle is not None and idle >= idle_required
            overdue = last_start is None or t - last_start >= timedelta(hours=lookback_hours)
            due = value >= threshold and is_idle
            backstop = not due and is_idle and bool(counts) and overdue
            if not (due or backstop):
                held_low += value < threshold and is_idle
                held_not_idle += not is_idle
                continue
            if last_start:
                gaps.append((t - last_start).total_seconds() / 3600)
            sleeps.append(dict(at=t.isoformat(), pressure=value, backstop=backstop))
            last_start, last_end = t, t + timedelta(minutes=cycle_minutes)
        clock = interval_hours + (cycle_minutes + check_seconds / 60) / 60
        return dict(threshold=threshold, sleeps=len(sleeps), backstop_sleeps=sum(s["backstop"] for s in sleeps),
                    idle_timer_clear_checks_held_below_threshold=held_low,
                    timer_clear_checks_held_not_idle=held_not_idle,
                    median_gap_hours=median(gaps) if gaps else None, max_gap_hours=max(gaps) if gaps else None,
                    sleeps_later_than_clock=sum(g > clock for g in gaps),
                    share_later_than_clock=(sum(g > clock for g in gaps) / len(gaps)) if gaps else None)

    sims = {"all_chat_idle": [simulate(th, chat_all) for th in thresholds],
            "juniper_only_idle": [simulate(th, chat_juniper) for th in thresholds]}
    per_era = {}
    for era in ("before_2557", "after_2557"):
        ws = [w for w in windows if w["era"] == era]
        per_era[era] = dict(
            windows=len(ws),
            windows_with_zero_checks=sum(w["zero_checks"] > 0 for w in ws),
            windows_never_reaching={str(th): sum(w["hours_to_first_crossing"][str(th)] is None for w in ws)
                                    for th in thresholds},
            median_hours_to_crossing={str(th): (median(v) if (v := [w["hours_to_first_crossing"][str(th)]
                                       for w in ws if w["hours_to_first_crossing"][str(th)] is not None]) else None)
                                      for th in thresholds})
    return dict(start=start.isoformat(), end=end.isoformat(), source_items=len(items),
                chat_rows=len(chat_all), juniper_turns=len(chat_juniper), checks=len(grid),
                formula_boundary=utc(formula_boundary).isoformat() if formula_boundary else None,
                validation=dict(saved_checks_compared=len(validation),
                                exact=sum(d < 1e-6 for d in validation),
                                max_abs_diff=max(validation, default=None)),
                by_era=per_era, sleep_windows=windows, simulated_schedules=sims,
                assumptions=[f"checks every {check_seconds:.0f}s from start; cycle lasts {cycle_minutes} min",
                             f"min interval {interval_hours} h, lookback/overdue {lookback_hours} h, idle {idle_required} min",
                             "current novelty formula applied to the whole window, including before #2557",
                             "sources as they stand at export: later deletes, deactivations and salience edits are invisible",
                             "simulated attempt end = simulated start + cycle_minutes; failed attempts are not simulated",
                             "only saved checks after #2557 validate the replay; earlier windows are counterfactual"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path, nargs="?")
    parser.add_argument("--start", type=utc, required=True)
    parser.add_argument("--end", type=utc, required=True)
    parser.add_argument("--print-sql", action="store_true")
    parser.add_argument("--with-checks", action="store_true", help="export new check history after its migration")
    parser.add_argument("--with-sources", action="store_true", help="export text-free source rows for --replay")
    parser.add_argument("--replay", action="store_true", help="R2a: replay current pressure at every check")
    parser.add_argument("--formula-boundary", type=utc, help="first novelty-formula sleep (#2557 live)")
    parser.add_argument("--interval-hours", type=float, default=6)
    parser.add_argument("--check-seconds", type=float, default=600)
    parser.add_argument("--legacy-cap", type=int, default=50)
    args = parser.parse_args()
    if args.print_sql:
        print(export_sql(args.start, args.end, with_checks=args.with_checks, with_sources=args.with_sources))
        return
    if not args.history:
        parser.error("history export required")
    rows = [json.loads(line) for line in args.history.read_text().splitlines() if line.strip()]
    if args.replay:
        print(json.dumps(replay_novelty(rows, args.start, args.end, interval_hours=args.interval_hours,
                                        check_seconds=args.check_seconds,
                                        formula_boundary=args.formula_boundary), indent=2))
        return
    print(json.dumps(report(rows, args.start, args.end, interval_hours=args.interval_hours,
                           check_seconds=args.check_seconds, legacy_cap=args.legacy_cap), indent=2))


if __name__ == "__main__":
    main()
