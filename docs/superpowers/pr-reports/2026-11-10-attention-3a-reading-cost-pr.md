## Summary

Attention patch 3a, "reading's real cost" (spec `docs/superpowers/specs/2026-11-07-attention-patch3-market-decides-design.md`, PR #3166, all defaults approved by Juniper 2026-11-10). No binding decision changes: reading still bids in shadow. Reading's shadow event-want decisions DO change (see risks).

- Reading's acts now get their real GPU lane cost. Before this, every read looked free: 0 of 1,117 read acts in 30 days had a cost.
- The fix sums the pool leases over every turn the read ran: the seed's two trace ids, plus each durable-run turn re-derived from its pool hold (this catches failed and retried attempts that the seed never records).
- A read's stage 2 runs after its act closes. So reading's cost is filled late, once the read has ended, by the sql-writer resolver tick. The close no longer writes a partial number.
- Reading declares its lanes in `families.yaml`: `[agent, agent-gpu2]`, `one_of`.
- A 30-day backfill script, dry run by default, follows the CLAUDE.md section 14 protocol.
- Replays C and D from the spec are committed as evals, with a fixture lane in CI.

## Outcome moved

- Dry run of the backfill on live data (2026-11-10 07:53Z, 30 days): **1,117 of 1,117 read acts gain a cost. 335.4 GPU slot-hours recovered** (agent-gpu2 190.3, agent 109.6, chat 30.4, metacog 4.2, fast 0.8). Per read: median 692 slot-s, p90 2,519, max 12,864. Errors 0. Anomalies 0.
- The old join gives 0 reads a cost. The seed's two trace ids alone reach 928 reads and 251.5 slot-hours. The other 189 reads failed stage 1 and only the re-derived turns reach them.
- The close-time window could only have seen 152.8 of the 335.4 slot-hours, because stage 2 runs after the close. That is why the fill is late.

### What the trace-id check against live Postgres found (before coding)

- The read act's `correlation_id` is the seed's `request_id`. No lease ever carries it: 0 of 1,117.
- The leases carry the turn id `uuid5("orion:durable:turn:{run_id}:{run correlation}:{hold lease}:{generation}")`. Verified on a live read: re-deriving it gives exactly the seed's `trace_id` (`fd92e090-...`) and `stage2_trace_id` (`a92fef07-...`).
- **The spec's "443 of 807 reads have no lease" was a data-retention artifact, not a join gap.** The spec's extract read `gpu_pool_leases`, which only keeps about 7 days (its extract began 10-31). `gpu_pool_events`, which the fill reads, goes back to 10-11. On it, every read with a recorded trace id has leases (928 of 928).
- Stage 2 runs after the act closes, since the act grades stage 1.

### Metric quality gate (reading's lane cost)

1. Provenance: `gpu_pool_events` terminal rows (`held_ms`, max per lease), written by the GPU pool. The join is on `turn_correlation_id` IN the read's turn ids (`orion/grammar/act_cost_ids.py`).
2. Independence: this is lane time. Reading's hunger, weight and salience are not computed from it (spec section 2, r = -0.02 against links).
3. Theory: opportunity cost. A slot-second that reading holds is one that curiosity cannot use. That is Choice's lane charge.
4. Live data: not degenerate. Median 692 s, p90 2,519 s, max 12,864 s, 0 zeros over 1,117 reads. The rest point is a read that made no model call. None did.
5. Existing mechanism: `ActCostV2.gpu_slot_s_by_lane` and the same fill. There is no new field.
6. Reversibility: `--restore` puts back the snapshot. The families.yaml lanes line is a one-line revert.

## Current architecture

- `orion/grammar/act_ledger.py` `fill_close_cost` charged an act for the leases whose `turn_correlation_id` equals the act's `correlation_id`, in the window from open minus 1 h to close plus 120 s.
- Hub's reading act (`services/orion-hub/scripts/act_emit.py`) uses the seed's `request_id` as its correlation_id and closes when stage 1 ends.
- The read's model calls run in durable-run turns (`services/orion-durable-runs/app/graph.py` `turn_correlation_id`).
- `families.yaml`: `hub.world_pulse_read` `lanes: []`, with the comment "non-GPU as far as the ledger can see".

## Architecture touched

- **sql-writer** (the act ledger in `orion/grammar`):
  - The cost join takes a list of ids.
  - A new late fill runs on the existing resolver tick.
- **durable-runs:** the turn-id rule moved to `orion/durable_runs/turn_identity.py`. Its behaviour is the same, and a test pins it to a live trace id.
- **config:** `config/physiology/families.yaml` (reading's lanes).
- There are no bus, schema or API changes.

## Files changed

- `orion/grammar/act_cost_ids.py` (new): the turn ids an act is charged for, and which actors are filled late.
- `orion/durable_runs/turn_identity.py` (new): the one rule for a durable turn's correlation id.
- `orion/grammar/act_ledger.py`:
  - the lease join over a list of ids;
  - the close leaves late actors unset;
  - `late_cost_targets`, `late_read_cost`, `apply_late_cost` and `refill_late_costs`.
- `services/orion-sql-writer/app/act_resolver_loop.py`: runs the late fill each tick and logs `[ACT_COST_LATE]`.
- `services/orion-durable-runs/app/graph.py`: `turn_correlation_id` calls the shared rule.
- `config/physiology/families.yaml`: reading's lanes `[agent, agent-gpu2]`, `one_of`, and the measured numbers.
- `scripts/backfill_reading_act_cost.py` (new): the 30-day backfill. It has a dry run (the default), `--apply` and `--restore`.
- `services/orion-sql-writer/tests/test_act_reading_cost_integration.py` (new): Postgres tests on a real read's trace shape, plus the backfill.
- `services/orion-sql-writer/tests/conftest.py`: the new module skips without a database (it errored 8 times before).
- `services/orion-durable-runs/tests/test_turn_identity_shared.py` (new): checks that the graph's turn id equals the shared rule and a live trace id.
- `services/orion-durable-runs/evals/reading_binding_replay.py` (new): replay C.
- `services/orion-gpu-pool/evals/value_queue_replay.py` (new): replay D.
- `.github/workflows/orion-sql-writer-tests.yml`, `orion-durable-runs-tests.yml`, `orion-gpu-pool-tests.yml`: the new tests, eval fixtures and path filters.
- `services/orion-sql-writer/README.md`: documents the late fill.

## Schema / bus / API changes

- Added: none. `ActCostV2` is unchanged.
- Removed: none.
- Renamed: none.
- Behavior changed:
  - A `hub.world_pulse_read` act no longer gets a GPU or energy fill at close. It was always empty, because the old join matched nothing.
  - Its `gpu_slot_s_by_lane` is filled after the read ends, and `basis["gpu"]` starts with `late fill`.
- Compatibility notes:
  - `grammar_acts.cost_json` is written once more per read, by an UPDATE after the close.
  - Energy (`wh_by_host`) is still not filled for reading. See the risks.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template changed).
- Skipped keys requiring operator action: none.

## Tests run

```text
git diff origin/main --check                                                         clean
pytest services/orion-sql-writer/tests/test_act_reading_cost_integration.py \
       test_act_ledger_integration.py test_act_workspace_binding_integration.py   42 passed (scratch Postgres 16)
pytest orion/grammar/tests services/orion-sql-writer/tests tests/test_act_ledger_contract.py \
       orion/physiology/tests                                                     1534 passed, 203 skipped
pytest services/orion-durable-runs/tests/test_turn_identity_shared.py \
       test_attempt_correlation.py test_reading_graph.py                          10 passed
Mutation check: with reading routed back to the old single-id join, 5 of the reading tests fail.
scripts/check_metric_lineage.py --gate, check_definition_drift.py --gate,
check_inner_state_registry.py, check_env_key_single_source.py,
check_scripts_dir_no_stdlib_shadow.py                                              all PASS
```

## Evals run

```text
Backfill dry run (live, 30 d): 1,117 targets, 1,117 gain a cost, 335.4 slot-hours, 0 errors, 0 anomalies.
Read ends: stage 2 done 919, stage 2 failed 9, stage 1 failed 187, stage 1 skipped 2.

reading_binding_replay.py (replay C) fixture: PASS
reading_binding_replay.py --live --days 14: PASS
  875 read starts, all with the 3a-measured cost
  verdicts: queued_at_pool 509, funded 366 (bid reasons: capacity_reserved 509, net_positive 366)
  held at all 0; held to the 2,900 s hold end 0; held > 1 h 0
  reading net p10 0.567, p50 0.764, p90 0.820
  reading share of Tier 2 funded wants (7 d) 5.7%; of Tier 2 acts started (7 d) 45.9%
value_queue_replay.py (replay D) fixture: 9/9 PASS
value_queue_replay.py --live --days 14 --aging 0.10: PASS
  2,650 background agent holds (from gpu_pool_events: gpu_pool_leases keeps ~7 days only)
  863 contested grants, 95 family changes (11.0% of contested, 3.6% of all), both directions
  aging 0: 124 changes (14.4%)
Differences from the spec's replay D (655 contested, 213 changes): 14 full days of holds; the 0.10
margin is enforced; only each family's earliest hold competes; upkeep keeps its exact place; holds
compete only with holds on the same lane.
```

## Docker/build/smoke checks

```text
No image build run: the change is Python in orion/ and service app code, with no dependency or Dockerfile change.
Live path: UNVERIFIED until deploy. After deploy, check:
  docker logs orion-athena-sql-writer 2>&1 | grep ACT_COST_LATE
  psql: SELECT count(*) FROM grammar_acts WHERE actor='hub.world_pulse_read' AND closed_at > now()-interval '1 day'
        AND cost_json->'gpu_slot_s_by_lane' IS NOT NULL;
```

## Backfill (run AFTER 3a is deployed)

```bash
cd /mnt/scripts/Orion-Sapienform && python3 scripts/backfill_reading_act_cost.py --apply
```

- The default is a dry run. The command uses the shared `.venv` python if system python lacks SQLAlchemy: `.venv/bin/python scripts/backfill_reading_act_cost.py --apply`.
- The snapshot, progress, report and before/after files go to `/tmp/reading-cost-backfill/`. Follow the progress with `tail -f /tmp/reading-cost-backfill/progress.log`.
- Undo: `.venv/bin/python scripts/backfill_reading_act_cost.py --restore`.

## Review findings fixed

Code review ran in a subagent on `git diff origin/main...HEAD`. No blockers. Six "should" findings:

- Finding: the 12 h fallback wrote a stage-1-only cost and then locked it. Live, 68 of 1,117 reads ended stage 2 more than 12 h after the close.
  - Fix: the fallback writes `late fill provisional`, and the fill runs again once the read ends. The live window went from 48 h to 96 h.
  - Evidence: `test_a_provisional_fill_is_completed_when_stage2_ends_14h_after_the_close`.
- Finding: the backfill snapshot was overwritten by a later run, so `--restore` restored nothing.
  - Fix: each run writes a new `snapshot-<mode>-<time>.jsonl`. Restore uses the earliest apply snapshot for each act.
  - Evidence: the backfill test now runs apply, a second apply and a dry run in one directory, then restores.
- Finding: replay C's gates could not fail. A hold ends at 0.81 h, and the acts-started share was counted from real acts.
  - Fix: a new gate on the share of reads held to the hold's end. Reading's acts-started share now uses the replayed admit times. The fixture also asserts net > 0.
  - Evidence: the live output above.
- Finding: replay D let a later hold jump an earlier hold of its own family, and it mixed lanes in one queue.
  - Fix: only each family's earliest hold competes, and holds compete only on the same role.
  - Evidence: a new fixture case (9/9), and the live re-run.
- Finding: an act whose fill always fails was retried every minute.
  - Fix: after 3 failures it is skipped for the rest of the process.
  - Evidence: `test_an_act_whose_fill_keeps_failing_is_skipped_after_three_ticks`.
- Finding: the report said nothing live changes. The lanes line DOES change reading's shadow event-want decisions, and those feed Prime's concerns.
  - Fix: stated under the risks below.
  - Evidence: replay C, 509 of 875 reading bids are `capacity_reserved`.
- Nits fixed:
  - A backfill write error is now counted inside the try block.
  - The dry run is tested through `main()`, with READ ONLY and rollback.
  - The families.yaml comment notes that `one_of` charges whole lane time.
- Nits not taken:
  - The target join index: the seed table has 2,024 rows.
  - A separate kill switch: no env key was added. The fill is one function call on the resolver tick, and a revert removes it.

## Restart required

```bash
# After merge, from the clean primary main checkout (post-pull rebuild picks these from the diff):
scripts/rebuild_services_from_git_diff.sh
# or by hand:
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-durable-runs up -d --build
# then, right away (see risks), the backfill:
.venv/bin/python scripts/backfill_reading_act_cost.py --apply
```

## Risks / concerns

- Severity: low.
  - Concern: reading's energy (`wh_by_host`) is still not charged. The hourly attribution (`attribute_host_hour`) counts a lease as metered only when an act's correlation_id is its turn id, so charging the read act too would count those watt-hours twice.
  - Mitigation: both sides must move together. That needs the same turn-id resolution inside the hourly attribution and its closure eval. That is a separate patch.
- Severity: low.
  - Concern: a turn replayed after a durable-runs restart (`turn_fence`) cannot be re-derived. It is only counted when it was the stage's last good turn (the seed's trace id). Live: 14 of 928 seed trace ids were not re-derivable, and all 14 were still counted through the seed.
- Severity: medium.
  - Concern: declaring reading's lanes changes a live behaviour. No binding family changes: no `hub.world_pulse_read` producer is in `BINDING_PRODUCERS`. But reading's after-the-read event wants (`orion-hub.reading_events`) are decided on those lanes too. In the last 7 days they were 263 funded and 2 deferred. Funded event wants become Prime's concerns (`PRIME_CONCERNS_FROM_EVENT_WANTS`, on by default). In replay C, 509 of 875 reading bids defer on capacity. So expect most reading event wants to defer, and **Prime to get far fewer reading concerns**. This lasts until 3b's value-ordered queue lands. Between the deploy and the backfill it is worse: with no lane history, a read is priced at one whole window.
  - Mitigation:
    - run the backfill right after the deploy;
    - the day after, check the reading verdicts in `want_ledger` (`family='hub.world_pulse_read'`);
    - roll back by reverting the families.yaml lanes line, which is one config line.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3176

🤖 Generated with [Claude Code](https://claude.com/claude-code)
