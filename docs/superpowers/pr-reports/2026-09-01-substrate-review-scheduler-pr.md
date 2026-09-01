# PR report: give the graph-review loop a clock

Branch: `feat/substrate-review-scheduler`
Status: see **Completion status** at the bottom.

## Summary

- Orion's substrate graph-review loop had no clock. Both halves — seed the queue,
  drain an item — were reachable only from HTTP endpoints, so nothing ever ran
  them unattended. `substrate_review_queue_item` held **zero rows and never had
  any**, so the review runtime selected nothing and emitted no telemetry.
- The *downstream* mutation scheduler does run autonomously (every 30s) against
  that telemetry, and had starved for **4620 consecutive cycles**.
- Adds `execute_substrate_review_scheduled_cycle()` plus a Hub asyncio loop:
  prune finished items, reseed when nothing usable is left, drain one due item.
- Adds `GraphReviewQueue.prune_finished()` / `usable_items()`. Without pruning the
  loop reaches a permanently un-drainable state ~42 minutes after first bootstrap.
- Stops the hub test suite writing into the live control plane, via an explicit
  `SUBSTRATE_CONTROL_PLANE_DETACHED` refusal. Clearing env keys was measured
  insufficient.

## Outcome moved

`consecutive_starved_cycles` on `/api/substrate/mutation-runtime/signal-intake`
had reached 4620 with `reason="zone_filter_rejected_all"`. The loop that feeds it
now has a scheduler. Live post-deploy confirmation is recorded in the
**Deploy verification** section.

## Current architecture (before this patch)

```
  [nothing]  ->  substrate_review_queue_item (0 rows, ever)
                        |
                        v
                 GraphReviewRuntimeExecutor  -- selects nothing, emits nothing
                        |
                        v
       mutation scheduler (autonomous, 30s)  -- 4620 starved cycles
```

`_bootstrap_substrate_review_frontier` (`api_routes.py`) and
`_execute_substrate_review_cycle` were called only from
`/api/substrate/review-runtime/{bootstrap,execute-once,execute-once-followup}`
and a debug pass. Nothing scheduled either.

### Correcting the previous diagnosis

An earlier finding (recorded 2026-08-31) blamed a disjoint 2-field filter: the
producer at `api_routes.py:2734` writes `invocation_surface="chat_reflective_lane"`
while the consumer demands `"operator_review"`. **That was wrong**, and acting on
it would have changed nothing.

Live `conjourney` Postgres, 2026-09-01 — `substrate_review_telemetry`, 1562 rows:

| rows | surface | zone | `selection_reason` | what they are |
|------|---------|------|--------------------|---------------|
| 1560 | `operator_review` | `concept_graph` | `test` | 780 `failed` + 780 `executed`, all `queue_item_id IS NULL`, zero pressure events |
| 2 | `chat_reflective_lane` | `autonomy_graph` | `producer_pressure_events:…` | real chat feedback, latest 2026-08-14 |

The 1560 are **test pollution**, not a competing producer. The filters were never
structurally disjoint: `review_bootstrap.py:44` seeds `hotspot_region ->
autonomy_graph` under `invocation_surface="operator_review"`, satisfying both.
They were simply never fed. Widening the surface constant would have admitted 2
rows from August and looked like a fix.

## Architecture touched

- `services/orion-hub` — new scheduled loop, tick function, cycle lock, three
  settings keys plus a control-plane detach switch.
- `orion/substrate/review_queue.py` — pruning and a usable-items view. This is
  the shared substrate package, not hub-local; the change is additive (two new
  methods, no signature or behaviour change to existing ones).
- No bus, schema-registry, or channel changes.

## Files changed

- `services/orion-hub/scripts/api_routes.py` — `execute_substrate_review_scheduled_cycle()`;
  `SUBSTRATE_REVIEW_LOCAL_CYCLE_LOCK` + `_review_cycle_lock_or_conflict()` applied to the
  tick and all three review-runtime endpoints; `SUBSTRATE_CONTROL_PLANE_DETACHED` honoured
  in `_resolve_control_plane_postgres_url()`.
- `services/orion-hub/scripts/main.py` — the loop, task global, shutdown cancellation;
  moved a stranded `substrate_decay_task = None` back into its own block.
- `services/orion-hub/app/settings.py` — 4 keys.
- `services/orion-hub/.env_example`, `docker-compose.yml` — same 4 keys, incl. the
  compose `environment:` allowlist (without which they are dead in the container).
- `orion/substrate/review_queue.py` — `prune_finished()`, `usable_items()`.
- `services/orion-hub/tests/conftest.py` — control-plane detach at `pytest_configure`.
- `services/orion-hub/tests/test_substrate_review_scheduler.py` — 9 tests (new).
- `services/orion-hub/tests/test_control_plane_isolation_guard.py` — 4 tests (new).
- `orion/substrate/tests/test_review_queue_pruning.py` — 8 tests (new).

## Schema / bus / API changes

- Added: none.
- Behaviour changed: `/api/substrate/review-runtime/{execute-once,execute-once-followup,bootstrap}`
  now return **409** when a cycle is already in flight. Previously they would run
  concurrently with each other and, after this patch, with the scheduled tick.
- Compatibility: the tick reuses `invocation_surface="operator_review"` rather
  than adding a third value to `GraphReviewRuntimeSurfaceV1`. That `Literal` is
  read by the consumer filter, the policy-profile rollout scope, and
  `review_runtime._select_item`'s zone gate; a new value would need all three
  changed together, with the stale-consumer hazard that implies.

## Env/config changes

- Added keys: `SUBSTRATE_REVIEW_SCHEDULER_ENABLED`,
  `SUBSTRATE_REVIEW_SCHEDULER_INTERVAL_SEC`,
  `SUBSTRATE_REVIEW_SCHEDULER_BOOTSTRAP_LIMIT`,
  `SUBSTRATE_REVIEW_SCHEDULER_PRUNE_AFTER_SEC`,
  `SUBSTRATE_CONTROL_PLANE_DETACHED`.
- `.env_example` updated: yes. Compose `environment:` allowlist updated: yes.
- Local `.env` synced: **by hand**. `scripts/sync_local_env_from_example.py` reads
  `.env_example` from the *primary* checkout, so worktree-added keys are invisible
  to it. Verified present in the live file.
- Interval is 420s, not a round number and 14x slower than the mutation
  scheduler's 30s, because a tick can run semantic graph queries.

## Tests run

```text
pytest orion/substrate/tests -q                     -> 646 passed
pytest services/orion-hub/tests -q  (branch)        -> 32 failed, 1910 passed, 5 skipped
pytest services/orion-hub/tests -q  (merge-base)    -> 32 failed, 1898 passed, 5 skipped
```

Set-difference of the two FAILED lists: **zero new failures**, one incidentally
fixed (`test_routing_dry_run_produces_trial_and_decision_without_side_effects`).
The 32 are pre-existing breakage on `main` (route-lane assertions, static JS-text
assertions, an unrelated `x_orion_operator_token` TypeError) and are not this
branch's to fix — but they mean **`main` currently ships a red hub suite**.

### Mutation testing

Every load-bearing claim was mutated against the real file, not a synthetic copy:

| mutation | result |
|---|---|
| `allow_followup=False` -> `True` | 1 test failed |
| `if queue_before == 0` -> `if True` | 2 tests failed |
| reseed gate back to `queue_total == 0` (the original B1 bug) | 1 test failed |
| `prune_finished` body -> no-op | 3 tests failed |
| prune cutoff removed (prune everything immediately) | 1 test failed |

## Evals run

```text
none
```

`services/orion-hub/evals/` exists but has no harness covering the review
runtime. Not added here. Recorded as a follow-up rather than claimed.

## Docker/build/smoke checks

See **Deploy verification**.

## Review findings fixed

Code review ran in a subagent at high effort and returned **BLOCKED**. Every
finding below was reproduced before being fixed.

- **B1 — the loop had a ~42-minute lifespan.**
  - Reseeding was gated on an *empty* queue. Nothing in `GraphReviewQueue` removes
    an item (the only deletion is the `max_items` eviction in `upsert`), and
    `list_eligible` filters suppressed items while `snapshot` returns them. Once
    the first generation suppresses — 6 executed cycles at the schema's
    `suppress_after_low_value_cycles=2` — the queue is permanently non-empty and
    permanently un-drainable. Not operator-recoverable either: `upsert` matches on
    region key and copies `suppression_state` forward, so re-running bootstrap
    resurrects the dead items.
  - Fix: `prune_finished(older_than_sec)` + `usable_items()`, gate on usable
    emptiness. The cutoff (21600s, matching the policy's own
    `slow_revisit_seconds`) is load-bearing: pruning a just-suppressed item would
    let the bootstrapper reseed the same region next tick and suppression would
    mean nothing.
  - Evidence: reviewer drove the real class to `{'suppressed': 3}` after 6 cycles.
    Covered by `test_review_queue_pruning.py`; mutation-killed twice.
  - **My own test asserted the bug was correct** — `total=5, due=0 ->
    idle_none_due` unconditionally, with nothing distinguishing "not due yet" from
    "never due again". Replaced.

- **B2 — my conftest turned 3 tests red while I reported them as pre-existing.**
  - `os.environ[key] = ""` still satisfies `os.environ.setdefault`, which
    `test_grammar_atlas_api.py:39` uses to install its own DSN. Fix: `pop`.
  - Evidence: that file alone, 3 failed / 6 passed -> **6 passed**.
  - I had called these pre-existing because the baseline subset I checked did not
    contain that file.

- **S1 — the commit was selling a guarantee the code did not provide.**
  - Claimed `allow_followup=False` was what kept the loop out of
    `self_relationship_graph`. Wrong three ways: `frontier_followup_executor` is
    never wired on this deployment; `frontier_curiosity.py:239` refuses that zone
    unconditionally anyway; `consolidation.py:279` echoes the request's own zone,
    so an autonomy_graph item cannot mint one.
  - The real guard is `review_schedule.py:84`, which returns `queue_item=None` —
    and `review_schedule.py:65` is the only `queue.upsert` caller in the repo.
    Conclusion held; stated mechanism was fiction. Now named and tested against a
    real scheduler.
  - This matters concretely: the tick runs under `operator_review`, which
    *satisfies* the zone gate at `review_runtime.py:226`. If such an item ever
    reached the queue by a future route, the loop would select it.

- **S2 — no lock; lost update reproduced.** `_persist` is a full-table swap and the
  operator endpoints are plain `def` (FastAPI threadpool), genuinely parallel with
  `to_thread`. Added `SUBSTRATE_REVIEW_LOCAL_CYCLE_LOCK`, mirroring
  `SUBSTRATE_AUTONOMY_LOCAL_CYCLE_LOCK`.

- **S3 — stranded task reset.** My edit moved `substrate_decay_task = None` into the
  new review block, leaking a cancelled Task on any box where the review scheduler
  is off, which is the default.

- **S4 — no durability gate.** A degraded control plane makes the queue pure process
  memory; the tick would log `items_enqueued=12` every interval against a queue
  that dies with the process. Now blocks with `queue_store_not_durable`.

- **S6 — relocated the test writes rather than stopping them.**
  `SubstratePolicyProfileStore` falls back to a hardcoded shared
  `/tmp/orion_substrate_policy.sqlite3`. Pinned per session.

- **S5 — no guard test.** Added; without one nothing fails if a later edit drops the
  fix, which is how the leak survived ~130 runs.

- **S7 — misattribution.** `test_phase20_policy_comparison.py` uses `tmp_path`
  sqlite throughout and never wrote to production. Sole culprit:
  `test_substrate_standalone_page.py`, 12 rows/run.

### Found after the review, by the guard test the review asked for

The env-only isolation **did not work**, and my "PROVEN" claim was an artifact of
pytest collection order.

`test_grammar_atlas_api.py:39` calls `os.environ.setdefault("DATABASE_URL", <live
conjourney DSN>)` at collection time — and that `setdefault` only succeeds
*because* `pytest_configure` had just popped the key. The fix created the
condition for its own defeat. Every `api_routes` import collected after it
(alphabetically, `test_substrate_standalone_page.py`) re-bound to production.

I had verified with `pytest test_substrate_standalone_page.py
test_grammar_atlas_api.py`; pytest collects in **command-line order**, so the
polluting module was imported before the `setdefault` ran. Alphabetical
full-suite order reverses it.

Measured: `substrate_review_telemetry` went **1562 -> 1598** across three
full-suite runs, 12 rows each, while the fix was believed to be holding.

Fix: `SUBSTRATE_CONTROL_PLANE_DETACHED`, read at the top of
`_resolve_control_plane_postgres_url()`. Nothing else in the repo reads that key,
so no module-level `setdefault` can satisfy it by accident — the property
env-clearing structurally could not have. The guard test now asserts
`_resolve_control_plane_postgres_url() is None` rather than the absence of keys
another module is entitled to reinstall.

The guard caught this within an hour of existing, by passing in isolation and
failing in the full suite.

## Restart required

```bash
cd <worktree>
scripts/safe_docker_build.sh orion-hub build
scripts/safe_docker_build.sh orion-hub up -d
```

## Risks / concerns

- **Severity: medium — Concern:** this makes an operator-only cognition loop
  self-starting. **Mitigation:** `SUBSTRATE_REVIEW_SCHEDULER_ENABLED=false` stops
  it with no other change; `settings.py` and compose both default it off.
- **Severity: medium — Concern:** the tick runs under `operator_review`, which
  satisfies the `self_relationship_graph` zone gate. Containment is
  `review_schedule.py:84` refusing to enqueue such an item at all.
  **Mitigation:** covered by a test against the real scheduler; if a future route
  ever enqueues that zone, this loop would select it and that test is where the
  breakage should surface.
- **Severity: low — Concern:** 15 undisposed `create_engine()` calls per executed
  tick (`review_queue.py:238,257,280`). Pre-existing, but this patch converts it
  from operator-triggered to a forever timer — relevant given PR #2010's
  connection-ceiling history. **Mitigation:** 420s interval; follow-up filed below.
- **Severity: low — Concern:** the loop sleeps before its first run, so evidence
  takes one interval to appear after a restart.

## Follow-ups not done here

- `main` ships 32 failing hub tests.
- 1596 `selection_reason="test"` rows remain in production
  `substrate_review_telemetry`. Snapshot at
  `/tmp/telemetry-test-row-purge/substrate_review_telemetry_full_backup.csv`.
  Not deleted — production `DELETE` needs Juniper's explicit approval.
- No eval harness covers the review runtime.
- `create_engine()` churn in `review_queue.py`.
- `frontier_followup_executor` is never wired; `/execute-once-followup` is
  therefore dead. Either wire it or delete it per the repo's "kill means kill" rule.
- `orion-athena-pageindex` is crash-looping (FastAPI `on_startup` incompatibility),
  unrelated to this branch.

## PR link

<pending>
