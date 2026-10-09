## Summary

Two Orion runs can now work on gpu2 at the same time instead of one waiting behind the other.

- New per-role setting `max_holds` (how many durable runs a role may carry at once; default 1 = the
  old rule). The scheduler never lets it exceed the slots the worker actually reports, and says
  so when it has to cut it down (`/health` → `holds`, and one `gpu_pool_max_holds_clamped` log line).
- New per-role setting `reserve_one_off_slots` (slots kept free for one-off calls; default 0).
  Shipped at 0 on purpose -- see "The one-off reserve decision" below.
- Gap pinning: when a one-off call borrows a run's pause between calls, it is charged to ONE run.
  Before this, one borrowed gap on a 2-run seat blocked both runs' next calls.
- `agent-gpu2: max_holds: 2` (Bonsai, 2 slots × 131,072). gpu1's `agent` stays at 1 (1 slot).
- Evals: pool-day replay gains a 2-slot concurrency scenario with the measured Bonsai slowdown
  (46 tok/s alone → 26 tok/s each with both slots busy) and a scripted gap-pinning tick;
  `hold_fairness` check A reads the hold limit instead of assuming one.

## Outcome moved

Durable-run queueing on the agent cards (spec acceptance checks 4, 5, 9, 10 in replay; 6 measured).
Replay of the same traffic, max_holds 1 → 2 on agent-gpu2 (means over 3 seeds, one-off rate 15/h):

| | max_holds 1 (7.2) | max_holds 2 (7.3) |
|---|---:|---:|
| queued-hold wait p90 | 7,060 s | 5,685 s |
| mean runs waiting | 7.76 | 6.61 |
| single-slot work done on gpu2 per hour | ~2,900 s | ~3,840 s (+32%) |
| a run's call waiting (max) | 49 s | 86 s (bound: one slowed one-off, 108 s) |
| one gap stalling two runs | -- | 0 s |
| one-off call wait p90 | 12 s | 139 s |

At the live one-off rate (~1 per 50 min, 12 seeds, scratch run, not in CI): hold wait p90 6,638 →
5,947 s, mean waiting 9.32 → 8.23, gpu2 work/hour +39%, run-call wait p90 0 s in both; one-off wait
p90 per seed 0 s in every seed → 0-196 s.

## Current architecture

- `orion/gpu_pool/scheduler.py` `_Ctx.free_for`: `if lease.kind == "hold" and self.holds.get(role)
  and lease.priority != URGENT: return 0` -- one non-urgent hold per role, whatever the slots.
- Gap sharing (H3) counted a one-off in a gap only as "one more used slot". With two idle runs and
  one borrower, each run's next call saw `2 - 1 used - 1 other idle run = 0`: both stalled
  (reproduced in `test_one_interloper_never_stalls_both_runs` with pinning disabled).
- 7.2 is live: agent-gpu2 runs Ternary-Bonsai-2-27B with 2 slots × 131K; the second slot only took
  one-off calls.

## Architecture touched

- Pool core (`orion/gpu_pool/config.py`, `scheduler.py`), pool service health + log
  (`services/orion-gpu-pool/app/{runtime,main}.py`), pool config (`config/gpu_pool.yaml`), static
  gate (`scripts/check_gpu_pool_config.py`), two evals. No bus, schema, projection or env change.

## Files changed

- `orion/gpu_pool/config.py`: `RoleSpec.max_holds` (≥1, default 1), `reserve_one_off_slots` (≥0,
  default 0); a service role with more holds than declared slots is refused; `check_max_holds()`.
- `orion/gpu_pool/scheduler.py`: `hold_cap()` (configured limit, clamped to discovered slots, less the
  reserve but never below one hold) replaces the hard-coded one; `_Ctx.charged()` gap pinning;
  `idle_holds()` excludes a charged hold; `_Ctx.wanting` (runs with a call waiting this tick).
- `config/gpu_pool.yaml`: `agent-gpu2: max_holds: 2`; profile-rollback note.
- `config/llm_profiles.yaml`: comment on the Bonsai profile (no value change).
- `scripts/check_gpu_pool_config.py`: max_holds vs `n_parallel` of `launch.profiles[0]`.
- `services/orion-gpu-pool/app/runtime.py`, `main.py`: `hold_limits()`, `/health` `holds`, edge-triggered
  clamp log.
- `services/orion-gpu-pool/evals/run_pool_day_eval.py`: concurrency scenario + gap-pinning case.
- `services/orion-durable-runs/evals/hold_fairness.py`: check A = peak concurrent holds ≤ hold limit.
- `services/orion-gpu-pool/README.md`: holds per role, gap pinning, two holds on a swap seat.
- `scripts/sql/2026-10-09_gpu_pool_stage7_3_acceptance.sql`: the live acceptance queries below.
- Tests: `orion/gpu_pool/tests/test_stage7_3_max_holds.py` (new),
  `services/orion-gpu-pool/tests/test_stage7_3_hold_limits.py` (new),
  `orion/gpu_pool/tests/test_stage7_2_bonsai_seat.py` (the 7.2 "second run never holds" pin now
  asserts both the shipped 2 and the max_holds 1 rollback).

## Schema / bus / API changes

- Added: none (no event, channel or forbid-schema field). `GET /health` gains a `holds` block (free
  dict, not a contract schema).
- Behavior changed: a role may carry `max_holds` non-urgent holds; gap borrowers are pinned.
- Compatibility: `RoleSpec` is `extra="forbid"`. **Old code cannot read the new YAML** (verified:
  origin/main's parser on this branch's `gpu_pool.yaml` → `roles.agent-gpu2.max_holds Extra inputs
  are not permitted`). Images that COPY the YAML with their code are safe. The one live reader that
  mounts a checkout and parses it per request is circe's `orion-gpu-lane-controller` -- see deploy.
  Launch digests are unchanged (agent-gpu2 `387ba0235eaf…`, diffusion `1db8c5aef416…` on both),
  so pool and controller still agree across the deploy.

## Env/config changes

- Added keys: none. `.env_example` updated: no. Local `.env` sync: not needed (no template changed).
- Config keys (YAML, not env): `roles.<r>.max_holds`, `roles.<r>.reserve_one_off_slots`.

## The one-off reserve decision

Juniper: "decide after measurements; 7.3 makes it a setting." Measured (7 days to 2026-10-09):

- One-off agent calls (no run attached): 200 in 7 days (~29/day). Wait p90 pooled **49.5 s**; on
  agent-gpu2 **0.19 s** (n=46, it had a free second slot); on agent 92 s; on chat 39 s.
- Every system agent request (run calls + one-offs): p90 **0.29 s** agent, **0.24 s** agent-gpu2.
  (The brief's 3.3 s / 0.44 s and the spec's 2.1 s / 0.3 s are older windows; I could not reproduce
  3.3 s / 0.44 s exactly, so these are the baselines the acceptance SQL compares against.)

Default: **`reserve_one_off_slots: 0`**. Why:

1. On a 2-slot seat a reserve of 1 is exactly `max_holds: 1` -- it would cancel the approved change.
2. Check 6 as written ("system agent request wait p90") is dominated by run calls (~92% of the
   population). Gap pinning keeps those at p90 0 s in every replay, so check 6 should hold.
3. **The honest cost:** one-off calls that used to find gpu2's free slot will now wait for one of
   the two running calls to finish (~1-3 min at the shared 26 tok/s). Pooled one-off p90 is likely
   to move from 49.5 s toward the 2× line (99 s). Volume is small (~7/day landed on gpu2), but some
   of these are FCC `http:anthropic` turns.
4. Trigger to reverse: if check 6a's pooled one-off p90 over the first 7 days is above **99 s**,
   delete `max_holds: 2` (or set `reserve_one_off_slots: 1`, the same thing on 2 slots).

## Tests run

```text
python -m pytest orion/gpu_pool/tests -q                         420 passed
cd services/orion-gpu-pool && python -m pytest tests -q          155 passed, 15 skipped
cd services/orion-durable-runs && python -m pytest tests -q      371 passed, 1 skipped
cd services/orion-gpu-lane-controller && python -m pytest tests -q   79 passed
cd services/orion-llm-gateway && python -m pytest tests -q -k "pool or route"   102 passed
python scripts/check_gpu_pool_config.py                          ok (5 cards, 9 roles, 9 classes, 2 launch blocks)
Gap-pinning tests fail with pinning disabled (charged() -> empty): 4/4 fail, as they should.
```

## Evals run

```text
python services/orion-gpu-pool/evals/run_pool_day_eval.py        VERDICT: PASS (63 s; was 10 s)
  concurrency_scenario: max_holds_2 two-holds seconds 7,942-7,997 of 8,000 per seed; gap stall 0 s;
  gap_pinning_case granted {"A-call": "agent-gpu2"} (without pinning: {"one-off": ...}, neither run)
  max_holds_2_reserve_1 never carried two holds (reserve in force)
ORION_ADMISSION_TEST_DSN=<throwaway postgres:16> python services/orion-durable-runs/evals/hold_fairness.py
  PASS; agent_hold_limit {max_holds 1, slots 1, effective 1, peak_concurrent 1}
```

## Docker/build/smoke checks

```text
Not built or deployed (brief: don't deploy). Pre-deploy control from the acceptance SQL, 7 days:
check 4 non-urgent overlapping holds on agent-gpu2 = 0 pairs (two URGENT holds overlapped on
10-06 04:24 -- U3 stacking, already allowed, excluded by the query).
```

## Live acceptance (after deploy)

`docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v since="'<deploy UTC>'" < scripts/sql/2026-10-09_gpu_pool_stage7_3_acceptance.sql`

| Check | Query block | Before (7 d to 10-09) | Pass |
|---|---|---|---|
| 4 two runs at once on agent-gpu2 | check 4 | 0 pairs | ≥ 1 pair |
| 5 queued-hold wait p90 | check 5a | agent 5,656 s, agent-gpu2 4,581 s (spec 3-day: 8,724 s) | below 8,724 s on agent (7.3 + 7.5) |
| 5 mean waiting agent holds | check 5b | 1.44 (max 16) (spec: 2.6) | below 2.6 |
| 6 one-off wait p90 | check 6a | pooled 49.5 s, agent-gpu2 0.19 s | pooled ≤ 99 s |
| 6 all system agent requests p90 | check 6b | 0.29 s agent, 0.24 s agent-gpu2 | ≤ 2× |

Also: `curl -s localhost:8127/health | jq .holds` → `agent-gpu2 effective 2, reason null`
once the seat is loaded.

## Review findings fixed

Code-review subagent on the full diff: 1 blocker, 1 should-fix, 6 nits. All fixed.

- Finding (blocker): a failed probe reads a role as 0 slots; gap pinning then charged its idle hold
  as "lent", the seat looked empty, and a draining seat was unloaded under a live run -- a change from
  main even at max_holds 1.
  - Fix: `lent = min(used, used + idle - slots)` -- only a running call can be in a gap.
  - Evidence: `test_a_failed_probe_never_makes_a_held_seat_look_empty[1|2]` (plus an empty-seat control).
- Finding (should-fix): the charge ignored hold priority, so an urgent run could be charged for a
  system one-off that was only ever allowed into a background run's gap.
  - Fix: charge the lowest-priority idle hold first.
  - Evidence: `test_a_gap_is_charged_to_the_lowest_priority_run_never_an_urgent_one`.
- Finding (nit): docs said urgent holds don't count toward the limit; the code counts them (same as
  main at 1). Fix: wording in config, scheduler docstring, README; `test_an_urgent_hold_counts_toward_the_hold_limit`.
- Finding (nit): the config validator refused a reserve that `hold_cap` allows. Fix: validator branch removed.
- Finding (nit): an unload after a clamp logged "in force". Fix: `no_slots` keeps the last state said.
- Finding (nit): hold_fairness check A runs on a 1-slot fixture, so it cannot catch a broken
  max_holds > 1 path. Fix: labelled as the gpu1 one-hold invariant; the >1 path is gated by the
  pool-day concurrency scenario and the scheduler tests.
- Finding (nit): the eval measured "one gap stalled two runs" before the tick's grants. Fix: measured
  after. Eval numbers unchanged, still PASS.
- Finding (nit): the "identical at 1 slot" docstring was only true after the blocker fix. Fix: reworded.

## Restart required

In order, one line each:

```bash
# 1. athena -- the pool (primary checkout on main, after merge):
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-gpu-pool up -d --build
# 2. right after (or before) step 1 -- the #27148 leak probe on the Bonsai seat, from athena:
cd /mnt/scripts/Orion-Sapienform && python3 services/orion-llamacpp-host/scripts/probe_slot_bleed.py --url http://100.112.254.99:8016
```

circe needs **nothing** for 7.3: the launch digest did not move, and the controller keeps its
current checkout. **But** whenever circe's checkout is next pulled past this merge, rebuild the
controller in the same step, or every swap is refused (old parser, new YAML):

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-gpu-lane-controller up -d --build
```

Launch digest: unchanged, nothing to sync. durable-runs and the gateway carry their own YAML copy;
they pick the new keys up on their next rebuild and do not need one for 7.3.

Rollback (one line): delete `max_holds: 2` under `agent-gpu2` in `config/gpu_pool.yaml`, then redeploy
the pool. Running runs finish; the next tick grants no second hold (`test_rollback_to_one_hold_…`).

## Risks / concerns

- Severity: high. Concern: **llama.cpp #27148 (cross-conversation leak) is still untested on
  Bonsai, and 7.3 is the change that puts two Orion runs on it at once.** No probe evidence exists
  on athena (`/tmp/slot-bleed-probe/` absent). Mitigation: run `probe_slot_bleed.py` before or right
  after deploy; on LEAK, roll back `max_holds` and set the cache flags per D2 (Juniper's answer 2).
- Severity: medium. Concern: one-off agent calls on gpu2 lose their free slot (see the decision
  above). Mitigation: check 6a after 7 days, one-line reverse.
- Severity: medium. Concern: circe controller + forbid parser + per-request YAML read: a circe pull
  without a controller rebuild breaks swaps. Mitigation: the paired circe line above. Not gated.
- Severity: low. Concern: an owner reclaim or the 2.5 h `max_hold_sec` drain on gpu2 now takes back
  two runs (two 600 s graces, two of each run's 12 take-backs). Tested; by design.
- Severity: low. Concern: two runs decoding at once on gpu2 raise sustained power/heat; the thermal
  guard only gates loads. Watch gpu2 temperature in the first days.
- UNVERIFIED: all live effects. Nothing here was deployed; replay numbers are a model (fixed call
  shape, slowdown table from one measurement), not live traffic.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2553

🤖 Generated with [Claude Code](https://claude.com/claude-code)
