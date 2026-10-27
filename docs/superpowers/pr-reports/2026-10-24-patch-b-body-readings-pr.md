## Summary

Body economy Patch B1: fixes to the Body organ's readings, ahead of the shadow prices (B2).

- **The lane over-count was a bug.** Orion's body said the 1-slot agent lane was busy for 5,986 seconds in a 3,600-second hour. The cause: a lease the GPU pool moved from one seat to another was charged in full to its last seat. Each held span is now charged to the seat its own event names.
- **The rest of the over-count is real overlap, by design, and is now reported separately.** The pool lets a higher-priority one-off call run in the gap of an idle background hold. Both leases are live, but the slot is occupied once. `level` now counts at most `slots` leases at once. The raw sum is `detail.lease_s` and the difference is `detail.lent_overlap_s`.
- **Waiting self-work (hunger) now sits beside each lane.** Each lane shows how many of Orion's own deferrable holds are queued for a seat of that kind, and for how long. The source is `gpu_pool_state_history.deferrable_depth` (#2873). This is not strain and nothing prices it. It is the seed of Appetite.
- No schema change: everything new lives in the free-form `detail` map that `PhysiologyStateV1` already carries.

## Outcome moved

- `lane:<role>` readings can no longer go above the seat's capacity unless the seat has 0 live slots. In that case the reading is flagged `held_without_live_slot`.
- Live, before this patch: `lane:agent` was over capacity on 747 of 2,778 ticks, `lane:agent-gpu2` on 344, and `lane:diffusion` on 457. `lane:diffusion` held leases on a seat whose capacity read 0.
- The prices in B2 read a level that means "slot-seconds actually occupied". They no longer read a double count.

## Root cause (lane:agent 5,985.9 slot-s on 1 slot)

I took the worst tick, 2026-10-23 10:14:12, and rebuilt it from `gpu_pool_events` and `gpu_pool_leases`. The non-child agent leases in that window were:

| lease | what happened | held |
|---|---|---|
| f4126f56 | hold on agent, released 09:40:15.760 | 2,085.9 s |
| 720191b9 | hold on agent, released 10:14:07.759 | 2,031.9 s |
| 3e2aaebf | hold **aborted twice on agent-gpu2** (09:41:27, 10:05:16), then **re-granted on agent** at 10:14:07.766 and still open | 1,718.8 s + 755.1 s on agent-gpu2 |

- `_lease_intervals` kept **one role per lease**, taken from whichever row it saw last. That was the open row from `gpu_pool_leases` (role agent). So both agent-gpu2 spans landed on lane:agent: 1,635.3 s (clipped to the window) + 755.1 s = **2,390.4 s extra**.
- Running the old code on these exact rows gives **5,985.899**, which matches the live value to the millisecond.
- The same bug also dropped the re-grant's open span, because a lease with any close was treated as closed.
- **Verdict: a double count.** It is fixed: each close's span goes to its own event's role. An open (re-granted) lease adds `(granted_at, now)` on its current role beside its earlier spans.

I then rebuilt true occupancy for all 2,778 ticks, using every close span on its own role plus the leases open now. Charging spans per role explains 160 of the 747 agent over-capacity ticks. The rest (638 ticks) are **real concurrent leases**. On agent, every overlap pair is a background hold with a system- or interactive-priority request running in its gap:

| pair | count | seconds |
|---|---|---|
| request/system with hold/background | 51 | 5,821 |
| request/interactive with hold/background | 14 | 164 |

This is `orion/gpu_pool/scheduler.py` H3 gap lending (`free_for`, `charged`). **Verdict: real overlap by design.** It is capped at capacity and reported separately as `lent_overlap_s`.

## Current architecture

- `orion/physiology/body.py` is the pure Body reducer. `services/orion-durable-runs/app/physiology_store.py` does the reads. The `physiology` node of the temporal-self graph publishes `PhysiologyStateV1` to Redis `orion:physiology:latest` and to bus `orion:physiology:state`. sql-writer persists it to `physiology_state`.
- Lane level was the sum of lease-seconds per role, with one role per lease.

## Architecture touched

- The Body reducer only. The same node, the same transport, the same schema.
- The durable-runs pool-state RPC now asks for `include_config`, about 5 KB per tick. This gives the pool's own class-to-seats map, used to place waiting work.

## Files changed

- `orion/physiology/body.py`:
  - per-span roles;
  - a close that is newer than the open projection row wins;
  - `occupied_seconds` (cap at slots);
  - `deferrable_backlog` and `stale_backlog_hosts`;
  - hunger detail on each lane, placed by the class map;
  - new warnings `lane_leases_held_on_two_seats`, `lane_deferrable_unplaced`, `lane_backlog_hosts_stale`.
- `services/orion-durable-runs/app/physiology_store.py`:
  - `read_backlog`, which reads the newest snapshot per host and the start of the waiting run. The run restarts at an empty, NULL or post-gap row.
  - `class_roles_from_pool_state`.
- `services/orion-durable-runs/app/main.py`: passes the backlog to the reducer and fetches the pool config.
- `orion/physiology/tests/test_body.py`:
  - a regression test on the real 10-23 span shapes;
  - gap lending, 2-slot cap, 0-slot seat, same-instant ordering, and a lagging projection;
  - backlog placement, unknown-is-never-0, clock skew, and stale hosts.
- `services/orion-durable-runs/tests/test_physiology_node.py`: the backlog SQL on real Postgres, including NULL rows and a history hole, plus the class map parse.
- `services/orion-durable-runs/README.md`: lane semantics, the hunger fields, and known limits.
- `config/physiology/contract.yaml`: the source-module reason now names the backlog read.

## Schema / bus / API changes

- Added: none. The new keys are in `PhysiologyConstraintReadingV1.detail`, which already holds free-form scalars:
  - `lease_s`, `lent_overlap_s`, `held_without_live_slot`;
  - `deferrable_queued`, `deferrable_waiting_sec`, `deferrable_waiting_capped`, `deferrable_hosts_stale`, `deferrable_no_reading`.
- Removed: none. Renamed: none.
- Behaviour changed: `lane:<role>.level` now means slot-seconds occupied (at most `slots` at once) and is charged per seat.
- Compatibility: no reader of `physiology_state` decides anything yet (design: shadow until Patch D). sql-writer stores `detail` verbatim. Deploy order does not matter.

## Env/config changes

- Added, removed or renamed keys: none.
- `.env_example` updated: no. Local `.env` sync: not needed.
- Skipped keys requiring operator action: none.

## Tests run

```text
orion/physiology/tests                                  82 passed
services/orion-durable-runs/tests (isolated Postgres)   625 passed, 1 skipped
services/orion-durable-runs/tests/test_physiology_node.py + orion/physiology/tests: 96 passed
Old reducer on the 10-23 rows reproduces live 5,985.899 (scratch check). The new reducer reads 3,599.9 for agent and 2,390.4 for agent-gpu2.
make check-body-economy-contract                       OK (68 modules)
check_definition_drift --gate / check_metric_lineage --gate / check_sentience_instruments --static-only /
check_async_routes_not_blocking / check_env_key_single_source: PASS
git diff --check: clean
```

## Evals run

```text
No separate eval harness for the Body organ. The live-data reconstruction above is the evidence:
747 over-capacity agent ticks became 160 explained by per-seat charging, plus 638 explained by
H3 gap lending (every overlap pair is a higher-priority request in a background hold's gap).
```

## Docker/build/smoke checks

```text
Live read-only smoke (production DB, read-only transaction, 2026-10-27 03:25Z, stand-in capacities):
lane:agent 3,599.97 slot-s (lease_s 3,599.97, lent 0); deferrable_queued 5, waiting >= 6 h (capped).
The same 5 queued agent-class holds also show beside agent-gpu2 and chat (seats the agent class may use).
gpu_pool_state_history has no empty agent snapshot in the last 6 h, so the age is capped.
No Docker build run: requirements unchanged, and the image already copies orion/ and config/gpu_pool.yaml.
```

## Review findings fixed

Independent review subagent (orion-repo-agent, read-only).

- Finding: an open projection row still trusted after a newer close of the same lease pinned the lane at capacity. Every real lease then read as lending (should-fix).
  - Fix: a close at or after the open row's `granted_at` drops the open span.
  - Evidence: `test_lane_projection_still_granted_after_a_close_is_not_held_to_now`. The re-grant case still passes.
- Finding: hunger was matched to lanes by name, so `lane:agent-gpu2` read a false real 0 while agent-class holds waited for it (should-fix).
  - Fix: placed by the pool's own class-to-seats map (state RPC `include_config`). An unread map is `deferrable_no_reading: unread:gpu_pool_classes`.
  - Evidence: `test_backlog_sits_beside_every_seat_its_class_may_use`, `test_backlog_classes_add_on_a_shared_seat`.
- Finding: a hole in the snapshot history counted as still waiting (nit).
  - Fix: the waiting run restarts at a row more than 60 s after the previous one.
  - Evidence: the Postgres test, hecate case.
- Finding: a snapshot slightly in the future read as stale (nit).
  - Fix: 10 s skew allowance.
  - Evidence: `test_backlog_snapshot_slightly_ahead_of_our_clock_is_fresh`.
- Finding: a stale host's queue was silently dropped (nit).
  - Fix: `deferrable_hosts_stale` detail and a `lane_backlog_hosts_stale` warning.
  - Evidence: `test_backlog_stale_host_is_named_not_silently_dropped`.
- Finding: the newest-snapshot query scanned 6 h (nit).
  - Fix: bounded to the last 10 min.
- Finding: the cap uses today's slot count for the whole window (nit). Documented in the README, and listed under risks.

## Restart required

```bash
./scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

Run from the clean primary `main` checkout after merge.

## Risks / concerns

- Severity: low.
  - Concern: `lent_overlap_s` caps with the seat's slots now, not as they were earlier in the hour. A seat that is down now is uncapped for the whole window.
  - Mitigation: documented. It is only visible in the hour after a slot change.
- Severity: low.
  - Concern: the class map costs about 5 KB per state RPC every ~120 s.
  - Mitigation: the same RPC already runs. The map is optional, and its absence reads `no_reading`.
- Severity: info.
  - Concern: `deferrable_waiting_sec` is capped at 6 h. Live, it reads capped now.
  - Mitigation: `deferrable_waiting_capped` says so. The value means "at least".

## PR link

TBD

🤖 Generated with [Claude Code](https://claude.com/claude-code)
