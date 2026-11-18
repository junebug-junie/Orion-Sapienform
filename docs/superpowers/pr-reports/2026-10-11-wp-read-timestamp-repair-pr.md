## Summary

- One-off repair script `scripts/repair_wp_read_timestamps.py` (Juniper-approved 2026-10-11) that replaces model-written times on reading data with the server's own write time, finishing what #2599 started for new writes.
- Subcommands: `plan` (read-only snapshot), `apply` / `rollback` (compare-and-set: a value changes only if it still holds what the snapshot expected), `settle` (re-apply until it survives the decay tick), `verify`, `report`.
- Already run live: 244 wp-read concept nodes (`observed_at`) + 4 decay stamps in Falkor, 113 rows each in `journal_entries` and `journal_entry_index`. 7 nodes / 3 journal rows left alone (their seed rows no longer exist).

## Outcome moved

Before: 4 reading concepts dated in the future (recency 1.0, decay frozen), 52 dated after their write (worst +13.7 days), 114 more than an hour early. After: 0 / 0 / 0; the formerly-future nodes now read recency 0 and decay normally.

## Current architecture

wp-read nodes live only in FalkorDB `orion_substrate` (no Postgres copies). Reading journal rows live in `journal_entries` plus its copy `journal_entry_index`. Server times: `world_pulse_read_seed.handoff_at` (Stage 1), `stage2_completed_at` (Stage 2, status done).

## Architecture touched

None at runtime. New one-off script + test only.

## Files changed

- `scripts/repair_wp_read_timestamps.py`: the repair.
- `tests/scripts/test_repair_wp_read_timestamps.py`: matching rule, tolerance, unmatched handling, stage1/stage2 separation, UTC pinning, settle loop.
- `docs/superpowers/pr-reports/2026-10-11-wp-read-timestamp-repair-pr.md`: this report.

## Schema / bus / API changes

- Added / Removed / Renamed / Behavior changed: none.

## Env/config changes

- None. `.env_example` not touched; no sync needed.

## Tests run

```text
pytest tests/scripts/test_repair_wp_read_timestamps.py -q  -> 9 passed
```

## Evals run

```text
None: one-off data repair, no eval harness applies. Live verify (below) is the evidence.
```

## Docker/build/smoke checks

```text
No build. Live run 2026-10-11:
apply 00:48:46Z  474/474 applied, 0 errors -> reverted by decay tick (see review)
settle 00:52-00:56Z  round1 reapplied 238, rounds 2-3 reapplied 0, verify clean
verify: observed_at future 0, after write 0, >120s early 0, stamp future 0; journal still-off 0 (both tables)
Artifacts: /tmp/wp-read-timestamp-repair/{snapshot.json,progress.log,before_after.csv,report.md,rollback.sh}
```

## Review findings fixed

- Finding (critical): `SubstrateDynamicsEngine.tick` re-upserts `observed_at` from a pre-apply snapshot; 238/244 Falkor fixes reverted within 2 minutes while the single immediate verify read clean.
  - Fix: `settle` command re-applies (guarded) until two consecutive verifies, 75 s apart (> 2 tick periods), are clean with nothing re-applied.
  - Evidence: settle output above; spot-check after 00:56Z still holds.
- Finding (medium): verify was a single read 0.3 s after apply. Fix: settle's delayed verifies.
- Finding (low): verify crashed on a null `observed_at`. Fix: null-safe, counted as `falkor_observed_at_missing`.
- Finding (low): new values depended on the Postgres session timezone. Fix: `parse_ts` pins UTC; test added.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: medium
- Concern: the decay tick rewrites `observed_at` on every node it touches from its snapshot. Any out-of-band graph repair can be silently undone the same way.
- Mitigation: settle loop here; durable fix (tick should not persist `temporal.observed_at`) is a follow-up, not in this PR.
- Severity: low
- Concern: stored `handoff_json` / `stage2_result_json` still hold the model's `created_at`. Nothing replays them into nodes or journal today.

## PR link

(this PR)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
