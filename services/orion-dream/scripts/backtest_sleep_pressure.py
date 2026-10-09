"""Backtest novelty sleep pressure against live history (design 2026-10-09).

Replays the last N days of the four pressure sources and reports how often
Orion would have slept, per threshold, if pressure counted only NEW distinct
items since the last sleep (not seen in the lookback before it). Read-only.

    docker exec orion-athena-sql-db psql -U postgres -d conjourney -AtF'|' -c "<EXPORT_SQL>" > ev.txt
    python3 services/orion-dream/scripts/backtest_sleep_pressure.py ev.txt

EXPORT_SQL is the four SELECTs in EXPORT_SQL below, run one after another
into the same file. Each line: source|epoch|key|weight.
"""
from __future__ import annotations

import sys
import time

KEY_NORMALIZE = r"[0-9a-f]{8,}|-?[0-9]+(\.[0-9]+)?"  # ids and numbers -> '#'

EXPORT_SQL = f"""
select 'metacog', extract(epoch from CAST(timestamp AS timestamptz)),
       regexp_replace(coalesce(trigger_reason,''),'{KEY_NORMALIZE}','#','g'),
       case severity when 'critical' then 1.0 else 0.6 end
  from orion_metacog where severity in ('degraded','critical')
   and timestamp ~ '^\\d{{4}}-\\d{{2}}-\\d{{2}}T' and CAST(timestamp AS timestamptz) > now()-interval '10 days';
select 'compaction', extract(epoch from created_at), theme, 0.5
  from dream_compaction_request_queue where created_at > now()-interval '10 days';
select 'resonance', extract(epoch from created_at), theme_key, least(1.0,0.3+0.1*greatest(0,violation_count))
  from substrate_reverie_resonance_alert where created_at > now()-interval '10 days';
select 'crystal', extract(epoch from h.created_at), h.crystallization_id::text, coalesce(c.salience,0.5)
  from memory_crystallization_history h join memory_crystallizations c using (crystallization_id)
 where op in ('auto_activate','approve') and h.created_at > now()-interval '10 days';
"""


def load(path: str):
    ev = []
    for line in open(path):
        source, t, key, w = line.rstrip("\n").split("|")
        ev.append((float(t), f"{source}:{key}", float(w)))
    ev.sort()
    return ev


def simulate(ev, threshold: float, *, days=7, min_hours=6.0, lookback_hours=48.0, step_sec=900, now=None):
    now = now or time.time()
    last = now - days * 86400
    t, sleeps = last, []
    while t < now:
        t += step_sec
        if t - last < min_hours * 3600:
            continue
        prior = {k for (tt, k, _) in ev if last - lookback_hours * 3600 <= tt < last}
        new: dict[str, float] = {}
        for tt, k, w in ev:
            if last <= tt < t and k not in prior:
                new[k] = max(new.get(k, 0.0), w)
        pressure = sum(new.values())
        if pressure >= threshold:
            sleeps.append((t, pressure, len(new)))
            last = t
    return sleeps


def main() -> None:
    ev = load(sys.argv[1])
    for threshold in (2, 3, 5, 8):
        sl = simulate(ev, threshold)
        gaps = sorted((b[0] - a[0]) / 3600 for a, b in zip(sl, sl[1:]))
        med = gaps[len(gaps) // 2] if gaps else 0
        print(f"threshold={threshold}: sleeps/7d={len(sl)} gap_h min/med/max="
              f"{min(gaps, default=0):.0f}/{med:.0f}/{max(gaps, default=0):.0f}")


if __name__ == "__main__":
    main()
