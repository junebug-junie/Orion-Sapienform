## Summary

Attention patch 3e, "reading's energy, counted once". It follows patch 3a (#3176), which gave reads their GPU time but not their watt-hours (3a report, concern 2).

- Read acts (`hub.world_pulse_read`) now get their energy, money and heat: `wh_by_host`, `wh_gpu_by_host`, `usd`, `heat_c_by_host`. The same late fill that 3a added computes them over all of the read's turns, priced at the read's end.
- The hourly energy attribution moves in the same patch. A lease now counts as "metered" when ANY act is charged for its turn, not only when an act's own id equals the turn id. So a read's watt-hours move from "leased without an act" to "metered". They are not counted twice.
- A read's own pool holds are charged to the read on both sides, for energy only. They are not charged for GPU slot time.
- A backfill script moves both sides together for the last 30 days. It is a dry run by default and follows the CLAUDE.md section 14 protocol.
- A Postgres test sums the energy both ways (per-act total + hourly remainder == metered host energy): before 3e, after 3e, and in the double-count case that 3e prevents.

## Outcome moved

- **Each watt-hour is counted once.** On a real read's shape, per-act Wh + the hour's sponsored + unattributed Wh equals the hour's above-rest Wh within 0.05 Wh, both before and after 3e. With the read charged and the hourly side unchanged (the double count), the sum is over by exactly the read's Wh.
- **Live dry run (2026-11-11, 30 days, read-only):** see "Backfill" below. 708 reads gain energy, and 465 stay "unmeasured" (the card idle floor is missing before 10-30). Read Wh and the Wh moved in the hourly rows agree to 3%. Every changed hour still closes exactly on its stored above-rest energy.
- **Closure eval (live, read-only) on 11-09 and 11-10:** PASS. The acts' Wh vs the hourly metered Wh is 3,633 vs 3,780 and 3,804 vs 3,986 (within 5%). Coverage (metered / activity) on circe goes from about 12% to about 42%.
  - Before 3e, the ordinary (non-read) acts alone were already 16% under metered on 11-10. That gap was there before this patch and is not caused by it.

## Where the hourly attribution lived (before)

- `orion/grammar/act_energy_fill.py` `attribute_host_hour` splits each host-hour's chassis watts above rest into sponsored, metered (a lease whose turn has an act) and the remainder. The arithmetic is `orion/energy/act_energy.py` `attribute_host_window`.
- `services/orion-sql-writer/app/orion_energy_attribution.py` writes it to `orion_energy_attribution_hourly` every 10 min, for the last 3 hours.
- "Metered" was `turn_correlation_id IN grammar_acts.correlation_id`. A read's leases carry its durable turn ids, never the act's id (`request_id`). So every read lease counted as `leased_unmetered_wh`.
- Live check: 0 acts have a read turn id as their correlation_id, and 0 have a read run's correlation id.
- Consumer: the physiology report (`services/orion-durable-runs/app/physiology_report.py`): "work not yet metered", coverage, and the latest unattributed hour.

## Metric quality gate (reading's wh_by_host)

1. **Provenance.** It is the same function every other act's Wh comes from: `act_energy_cost`. It reads `orion_biometrics_summary` chassis, CPU and card watts, the pinned rest (`rest_body.yaml`) and `gpu_pool_events` terminal leases. The join is on the read's turn ids (`reading_cost_ids`) plus its hold ids (`reading_hold_ids`).
2. **Independence.** It is the same quantity the hourly row's metered Wh already measured, moved from one bucket to another. It is not a new signal. Reading's GPU slot-seconds (3a) are correlated with it, since both come from the same leases. Wh adds the card's draw and the whole machine's overhead.
3. **Theory.** Energy conservation. The chassis watts above rest are split by each card's above-idle share and then by lease. Each Wh goes to exactly one bucket.
4. **Live data.** It is not degenerate. Per read: median 58 Wh, p90 118 Wh, max 706 Wh. A read with no lease is written as "no lease found" (0 in 30 days). A missing card floor is written as "unmeasured" (465 reads before 10-30). The hourly split has the same hole there, so nothing is metered for those reads either. The rest point is a read that made no model call.
5. **Existing mechanism.** It uses the ActCostV2 fields and the existing fill. No new field, table or env key.
6. **Reversibility.** The backfill's `--restore` puts back both sides. A code revert brings back the old rule. The live writer then recomputes recent hours. Older hours need `--restore`.

## Architecture touched

- `orion/grammar` (sql-writer's act ledger): which turns count as metered, the per-act energy fill over a list of ids, and the late fill.
- `services/orion-sql-writer`: the hourly writer's docstring, the resolver loop docstring, the README, tests and the eval.
- There are no bus, schema, API or env changes.

## Files changed

- `orion/grammar/act_cost_ids.py`:
  - `reading_turn_owners`: the read turn ids with an act.
  - `turns_charged_to_acts`: the hourly "metered" rule.
  - `turns_with_own_act`: the old rule.
  - `reading_hold_ids`: the read's holds, used for energy only.
  - The known limits are documented.
- `orion/grammar/act_energy_fill.py`:
  - `act_energy_cost(correlation_ids=..., priced_at=...)`.
  - `attribute_host_hour` uses `turns_charged_to_acts`. `charge_reads=False` keeps the old rule, used only by the backfill and the tests.
- `orion/grammar/act_ledger.py`:
  - The late fill writes energy.
  - A third target arm selects reads that were GPU-filled before 3e.
  - The update guard is extended.
  - A fill that raises is retried up to 3 times.
  - `LateCostSummary.wh` reports the Wh filled.
- `orion/energy/act_energy.py`: the `LeaseSpan.metered` comment.
- `scripts/backfill_reading_act_energy.py` (new): the two-sided backfill. It has a dry run (the default), `--apply` and `--restore`.
- `services/orion-sql-writer/evals/act_energy_closure.py`: per-act agreement now groups a read's turns into one act.
- `services/orion-sql-writer/tests/test_reading_energy_closure_integration.py` (new): 11 Postgres tests (sum both ways, lifecycle, backfill).
- `services/orion-sql-writer/tests/test_orion_energy_attribution_pg.py`: the fixture gets the reading tables.
- `services/orion-sql-writer/tests/test_act_reading_cost_integration.py`: GPU-only tests turn energy off (there are no readings tables).
- `services/orion-sql-writer/tests/conftest.py`: the new module skips when there is no database.
- `services/orion-sql-writer/app/orion_energy_attribution.py`, `app/act_resolver_loop.py`, `README.md`: docs.
- `.github/workflows/orion-sql-writer-tests.yml`: the new test and the path filter for the script.

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed:
  - Read acts' `cost_json` gains `wh_by_host`, `wh_gpu_by_host`, `usd`, `heat_c_by_host` and `basis.wh`.
  - In `orion_energy_attribution_hourly`, a read's share moves from `leased_unmetered_wh`/`unattributed_wh` to `metered_wh`. `above_rest_wh` and `sponsored_wh` are unchanged.
- Compatibility notes: the field names are the same, so no reader needs a change.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed.
- Skipped keys requiring operator action: none.
- The existing flag `SQL_WRITER_ACT_ENERGY_COST_ENABLED` (default true) gates the read energy fill too.

## Tests run

```text
pytest -m integration (scratch Postgres 16):
  test_reading_energy_closure_integration.py test_act_reading_cost_integration.py
  test_act_ledger_integration.py test_act_workspace_binding_integration.py      55 passed
pytest test_orion_energy_attribution_pg.py test_orion_energy_rollup_pg.py         9 passed
pytest orion/grammar/tests services/orion-sql-writer/tests tests/test_act_ledger_contract.py
       orion/physiology/tests                                                     1547 passed, 214 skipped
Mutation check: with the hourly rule switched back to turns_with_own_act, 2 tests fail
  (sum-both-ways and the backfill test).
check_metric_lineage --gate, check_definition_drift --gate, check_inner_state_registry,
check_env_key_single_source, check_scripts_dir_no_stdlib_shadow                     all PASS
git diff --check                                                                    clean
```

The sum-both-ways test (`test_sum_both_ways_before_3e_regression_and_after`):
- Before 3e: acts + remainder - above_rest = 0 (within 0.05 Wh).
- After 3e: the same sum is 0; metered rose by exactly the read's Wh; leased_unmetered fell by it.
- The double count (read charged, old hourly rule): the sum is over by exactly the read's Wh.

## Evals run

```text
act_energy_closure.py --day 2026-11-09 (live, read-only): PASS  circe acts 3,633 Wh vs metered 3,780; coverage 0.42
act_energy_closure.py --day 2026-11-10 (live, read-only): PASS  circe acts 3,804 Wh vs metered 3,986; coverage 0.44
  before 3e on 11-10: metered 1,164 Wh; non-read acts 977 Wh (the 16% gap was already there)
```

## Backfill (dry run done; run --apply AFTER deploy)

Dry run, live, read-only, 30 days (2026-11-11 04:24Z). Files are in `/tmp/reading-energy-backfill/`: `report.md`, `before_after.csv`, `progress.log`, `snapshot-dry_run-*.jsonl`.

- Snapshot: 1,173 act rows and 332 hourly rows (845.8 KB). This is far under the 100k-row stop line.
- Read acts with no energy yet: **1,173**.
  - 708 gain energy, **47,555.5 Wh** in total (all on circe). Per read: median 57.9 Wh, p90 118.0 Wh, max 705.7 Wh.
  - 465 are unmeasured: circe had no card idle floor before 10-30 (`no_idle_floor` on gpu0 138, gpu3 136, gpu1 115, gpu2 76).
  - 0 have no lease.
- Hourly rows (circe, last 30 days, except the 4 hours the live writer owns): 332 examined, **197 changed**, 0 drifted, 0 already applied.
  - **49,036.9 Wh** move from leased-without-act (and unattributed) to metered.
  - Read Wh / moved Wh = 0.97. The gap comes from the window edges.
  - Every changed hour still closes on its stored above-rest energy: max error 0.0 Wh.
- Errors: 0. Anomalies: 466 (465 unmeasured, plus 1 read over 500 Wh: 705.7 Wh over 6 turns and 4 holds).
- Before/after examples:
  - Read `act_2592ee2a...`: wh_by_host null to circe 21.4 Wh.
  - Read `act_c2bd0d84...`: null to 81.2 Wh.
  - Hour circe 2026-10-31 16:00: metered 8.6 to 523.0 Wh, leased_unmetered 524.6 to 10.2 Wh, unattributed 764.5 to 250.1 Wh.

```bash
cd /mnt/scripts/Orion-Sapienform && .venv/bin/python scripts/backfill_reading_act_energy.py --apply
```

- Run it after the sql-writer restart. The live writer owns the last 4 hours and fixes them itself.
- Each act and each hour commits on its own. A re-run is safe: filled acts are not selected again, and hours already on the new rule are reported as already applied.
- Follow progress with `tail -f /tmp/reading-energy-backfill/progress.log`.
- Undo, right after a bad apply, after reverting the code: `.venv/bin/python scripts/backfill_reading_act_energy.py --restore`.

## Docker/build/smoke checks

```text
No image build: Python-only change in orion/ and sql-writer; no dependency or Dockerfile change.
Live path: UNVERIFIED until deploy. After the restart, check:
  docker logs orion-athena-sql-writer 2>&1 | grep ACT_COST_LATE        (wh > 0)
  psql: SELECT count(*) FROM grammar_acts WHERE actor='hub.world_pulse_read'
        AND closed_at > now()-interval '1 day' AND cost_json->'basis' ? 'wh';
  psql: SELECT hour_start, metered_wh, leased_unmetered_wh FROM orion_energy_attribution_hourly
        WHERE host='circe' ORDER BY hour_start DESC LIMIT 4;   (metered up, leased_unmetered down)
```

## Review findings fixed

The code review ran in a subagent on `git diff origin/main...HEAD`. It found no blocker.

- Finding (S1): the read's own pool hold shared every call's card 1/2, and nobody was charged for that share. Reads got about half their real energy.
  - Fix: `reading_hold_ids`. A read's run correlation ids (its holds' turn ids) are energy-only ids on both sides. GPU slot-seconds still leave them out.
  - Evidence: `test_the_reads_own_hold_is_charged_its_energy_not_its_slot_time`. The live 30-day read Wh went from 23.9k to 47.6k, and closure-eval agreement improved from 9% to 4%.
- Finding (S2): an energy fill that raised once was written off for good, while the hour still counted the read as metered.
  - Fix: it is retried up to `LATE_COST_MAX_FAILURES`, and only then written as unmeasured.
  - Evidence: `test_an_energy_fill_that_raises_is_retried_not_written_off`.
- Finding (S3): hours left alone for drift stayed double counted, and the report did not say how much.
  - Fix: the report now shows `wh_still_double_counted_in_drift_hours`. Live it is 0 (0 drifted hours).
- Finding (S4): the "second apply changes nothing" test passed because of drift rejection.
  - Fix: `recompute_hour` recognises a row already on the new rule. The test asserts 0 drift and 0 anomalies on the re-run.
  - Follow-on: after the dry run, hours with no read leases were also being labelled "already applied". That is fixed too: the label now requires the old and new rules to differ.
- Nits fixed:
  - `_READ_ACTS_SQL` filters `verb='look'`.
  - The known limits are documented: late leases, provisional fills, flag off.
  - The restore caveat is documented.
- Nits not taken:
  - Seed trace-id indexes. The scan is 38 ms over 2,083 rows. The hourly writer runs it about 3 times per 10 min, only for hosts with leases.
  - Delta-applying drifted hours. There are 0 live.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && scripts/rebuild_services_from_git_diff.sh
```

or by hand:

```bash
cd /mnt/scripts/Orion-Sapienform && scripts/safe_docker_build.sh orion-sql-writer up -d --build
```

Then the backfill: `cd /mnt/scripts/Orion-Sapienform && .venv/bin/python scripts/backfill_reading_act_energy.py --apply`

## Risks / concerns

- Severity: low.
  - Concern: the first tick after deploy selects every read GPU-filled in the last 96 h that has no energy, about 150 (limit 200). Each one reads the host series and floors once. Live dry run: about 0.1-3 s per read.
  - Mitigation: none needed. It happens once.
- Severity: low.
  - Concern: the physiology report's "work not yet metered" on circe drops sharply. Coverage goes from about 12% to about 42%, and Orion will read this in its body report.
  - Mitigation: this is the intended correction. The numbers were understated before.
- Severity: low.
  - Concern: energy is still lost (never double counted) for a read lease whose terminal event lands after the read's end + 120 s, and for a provisional fill that is never finalised. 465 old reads stay unmeasured because circe's card idle floors are missing before 10-30. The hourly split has the same hole there.
  - Mitigation: these limits are documented in `act_cost_ids.py`.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3186

🤖 Generated with [Claude Code](https://claude.com/claude-code)
