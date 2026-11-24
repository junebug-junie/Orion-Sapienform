## Summary

- The backfill's referent gate now uses a backfill-only floor for "a referent besides Juniper or Orion": 53%. Juniper approved it on 2026-11-24 (#3537).
- The any-referent floor stays at 97%. Live floors do not change (the live non-self floor stays 68%).
- Weeks below the floor are never passed silently. A week with 10 or more rows below the floor fails the gate. A smaller week below the floor is listed in the gate output and marked BELOW in the report table.
- `gate --other-min` can override the floor for one run, and the gate output records which floor it used.

## Outcome moved

The July-September backfill can pass its gate on a floor that fits that chat, with every low week still visible. Before, the only floor was the live 68%, which the backfill sample failed (57.5%) even though the referent step was working (100% of memories had a referent).

## Current architecture

`scripts/backfill_episode_memory_jul_sep.py gate` measured per-week referent coverage of the backfilled rows against the live band only (any >= 97%, besides Juniper/Orion >= 68%). Weeks under 10 rows were skipped without any flag.

## Architecture touched

- One operator script (`scripts/backfill_episode_memory_jul_sep.py`) and its tests. No service code, no schema, no bus, no env.

## Files changed

- `scripts/backfill_episode_memory_jul_sep.py`: `BACKFILL_OTHER_MIN = 0.53` with its basis; `gate_verdict(other_min=...)`; small weeks below the floor flagged; `gate --other-min`; report marks weeks below the floor.
- `orion/memory/episode/tests/test_backfill_jul_sep_script.py`: tests for the floor, the unchanged live floors, and the flagged small weeks.
- `docs/superpowers/pr-reports/2026-11-24-memory-backfill-floor-pr.md`: this report.

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed: the backfill gate's default floor for "besides Juniper/Orion" is 53% (backfill rows only).
- Compatibility notes: none; the script is run by hand.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed.
- skipped keys requiring operator action: none.

## Tests run

```text
pytest orion/memory/episode/tests -q   -> 412 passed, 34 skipped
```

## Evals run

```text
Basis for the floor (dry runs, nothing written):
  backfill sample, 15 episodes / 106 memories: besides Juniper/Orion 61/106 = 57.5%
    (95% range 48.0-66.5% rows, 41.4-73.4% episodes); any referent 106/106
  live October, 47 episodes / 239 rows: 178/239 = 74.5% (68.6-79.6% rows, 65.6-82.6% episodes)
  lowest live week 68.4% x backfill/October ratio 0.773 = 52.9% -> 53%
Full-run gate result: see the #3537 backfill report.
```

## Docker/build/smoke checks

```text
None needed. No service code changes.
```

## Review findings fixed

- Finding: (pending review subagent)
  - Fix:
  - Evidence:

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low
- Concern: a lower floor could hide a broken referent step.
- Mitigation: the any-referent floor (97%) is unchanged and is the check that catches a broken step; every week below either floor is listed.
- Issues opened: none. Issues closed: none.

## PR link

(set after creation)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
