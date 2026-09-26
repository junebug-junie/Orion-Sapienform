# fix(durable-runs): a pool that cannot take a hold request no longer fails the run

## Summary

- When the GPU pool can't take a durable run's request for a GPU right now -- it is unreachable,
  running an older version that can't read the request, or mid config roll -- the run now keeps
  waiting and asks again later, instead of being failed for good.
- Only refusals that are really about the run itself still fail it: its own deadline passed, it
  needs more context than its class can ever give, the pool's backlog give-up expired, the request
  is too big, or a class/operator-only refusal that durable-runs' *own* copy of `gpu_pool.yaml`
  agrees with.
- Retries back off exponentially (15 s, 30 s, 60 s ... capped at 300 s, both configurable) and
  every refusal is a visible `run.waiting_resource` event carrying the pool's reason.
- Replies the pool could not even parse (`invalid:*`, `unknown_verb:*`) to heartbeat, status or
  release are treated like a missed RPC, so a skewed pool mid-run no longer kills a running turn
  or pretends a hold was released.
- Runbook fix: the cutover step 6 row-count guard always errored; it is now evaluated lazily.

## Outcome moved

Incident 2026-09-26 02:32 UTC: durable-runs 4.5 was deployed ~20 s before the upgraded pool. The
old pool answered every hold acquire
`unavailable reason=invalid:2 validation errors for GpuLeaseRequestV1 hold_lease_id Extra inputs
are not permitted` (its request model predates `hold_lease_id`/`hold_generation` and is
`extra="forbid"`). `AdmissionRuntime.register` read "unavailable with no lease_id" as "can never
be placed" and failed all 11 resumed runs (`gpu_pool_unavailable:invalid:...`). With this patch
the same sequence leaves all 11 waiting and they complete once the pool answers; the new eval
replays it (11 runs, 0 failed, 11 completed, 44 acquire RPCs in a 1.2 s skew window vs 660
without backoff).

## Current architecture

`resource_request` (`register`) asks the pool for a hold; `resource_wait` (`lease`) interrupts
until the pool grants it; `_hold_ready` decides when an interrupted run is re-driven (pool event
or one status read per `DURABLE_RUNS_HOLD_STATUS_POLL_SEC`). Before this patch:

- any `unavailable` reply without a `lease_id`, or `unknown_lease`, to an acquire failed the run;
- an acquire RPC exception kept waiting but retried every poll interval with no backoff and no
  event (only a log line);
- `lease()` failed the run on `unknown_class`/`operator_only_class` regardless of config;
- a skew reply to heartbeat raised `HoldLost` (turn stopped), to status ended the hold, and to
  release was recorded as `resource.lease_released`.

## Architecture touched

durable-runs only (no pool, schema or bus change). `app/pool_hold.py` gains the refusal
classifier; `app/admission_runtime.py` uses it in `register`, `lease`, `_beat`, `guard`,
`release`, `_end_hold` and `_hold_ready`. Backoff state lives inside the existing checkpointed
`hold` dict (`retry_at`, `refusals`), so no graph state-schema change and it survives a restart.

### Classification (what fails a run, and why)

| pool reason | verdict | why |
| --- | --- | --- |
| RPC timeout / exception | wait | pool unreachable; the acquire may have landed, re-asking with the same request id is idempotent |
| `invalid:*` | wait | the pool's request model rejected ours: version skew (the incident) |
| `unknown_verb:*` | wait | the pool predates the verb: version skew |
| `unknown_class:<c>`, `c` in our `gpu_pool.yaml` | wait | the pool loaded a different config (roll in progress) |
| `unknown_class:<c>`, `c` not in ours | **fail** | nobody knows the class; the run can never be placed |
| `operator_only_class`, our config also operator-only | **fail** | durable-runs is never an operator |
| `operator_only_class`, our config disagrees | wait | config skew |
| `deadline` | **fail** (`workflow_deadline`) | the hold's `deadline_at` is the run's own admission deadline |
| `min_ctx_exceeds_class` | **fail** | the run asks for more context than any role of its class serves (pool remembers restarting roles' ctx) |
| `backlog_max_age` | **fail** | the pool's configured give-up for a backlogged hold |
| `replay_payload_too_large` | **fail** | the request itself is over the cap |
| `unknown_lease` answer to an acquire, empty or unrecognised reason | wait | failing on a reason nobody classified is exactly this incident; the run stays visible and cancellable |
| `backlogged` | wait | spec Decision 1 rule 6 (unchanged; now tested) |

## Files changed

- `services/orion-durable-runs/app/pool_hold.py`: `refusal_is_terminal`, `is_pool_trouble`, documented reason sets.
- `services/orion-durable-runs/app/admission_runtime.py`: `_pool_refused` backoff + event; classifier in `register`/`lease`; skew handling in `_beat`/`guard`/`release`/`_end_hold`; backoff gate in `_hold_ready`.
- `services/orion-durable-runs/app/settings.py`, `.env_example`, `docker-compose.yml`: two new keys.
- `services/orion-durable-runs/README.md`: classification + deploy-order note.
- `services/orion-durable-runs/tests/test_pool_refusal_postgres.py`: incident regression tests (new).
- `services/orion-durable-runs/tests/test_admission_runtime_postgres.py`: unreachable-pool test updated for the new `hold` shape.
- `services/orion-durable-runs/evals/deploy_order_skew.py`: incident replay eval (new), wired into `.github/workflows/orion-durable-runs-tests.yml`.
- `docs/runbooks/2026-09-25-gpu-pool-stage4-cutover.md`: step 6 guard fix + note on the 2026-09-26 leftovers.

## Schema / bus / API changes

- Added: none. `run.waiting_resource` (existing event) now also carries `transient`, `reason`, `refusals`, `retry_at` in `detail` (free-form dict on `ResourceEventV1`).
- Removed / Renamed: none.
- Behavior changed: pool refusals that are not about the run no longer end it.
- Compatibility notes: a checkpoint written before this patch has no `retry_at`; it behaves as before on its first re-drive.

## Env/config changes

- Added keys: `DURABLE_RUNS_POOL_RETRY_BASE_SEC=15.0`, `DURABLE_RUNS_POOL_RETRY_MAX_SEC=300.0`
- Removed / Renamed keys: none
- `.env_example` updated: yes; compose `environment:` updated (parity check OK, 37 keys)
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes -- both keys present in `/mnt/scripts/Orion-Sapienform/services/orion-durable-runs/.env` (lines 92-93)
- skipped keys requiring operator action: none

## Tests run

```text
cd services/orion-durable-runs && ORION_ADMISSION_TEST_DSN=<throwaway postgres:16 on 127.0.0.1:55491> pytest tests -q
140 passed (117 before + 23 new)

New tests against origin/main's admission_runtime.py/settings.py (fail-before check):
FAILED test_version_skewed_pool_refusal_waits_backs_off_and_completes_when_the_pool_is_upgraded  (assert 'failed' is None)
FAILED test_rpc_timeout_backs_off_visibly_and_the_run_completes
FAILED test_a_pool_config_roll_unknown_class_waits_but_a_class_nobody_knows_fails
FAILED test_a_skewed_heartbeat_mid_turn_does_not_kill_the_turn
(backlogged / deadline spec-conformance tests pass on both: that behaviour was already right)

python scripts/check_env_template_parity.py            PASS
python scripts/check_service_env_compose_parity.py orion-durable-runs   OK
python scripts/check_metric_lineage.py --gate          PASS
python scripts/check_definition_drift.py --gate        PASS
git diff --check                                       clean
```

The incident refusal is reproduced faithfully: the test bus validates the real wire payload
against the current `GpuLeaseRequestV1` with `hold_lease_id`/`hold_generation` removed and
`extra="forbid"`, and replies exactly as `services/orion-gpu-pool/app/main.py` `_on_lease` does.

Runbook guard, checked on the throwaway postgres:16: the old form errors even on a matching count
(`invalid input syntax for type integer: "ok"` there; live it surfaced as `division by zero`);
the new form prints `ok:2` and commits on a match, and raises `division by zero` and rolls back on
a mismatch.

## Evals run

```text
python services/orion-durable-runs/evals/deploy_order_skew.py   PASS
  11 runs (6 curiosity, 5 self-sense), old pool for 1.2 s then upgraded:
  no_run_failed, all_completed, refusals_visible, no_retry_storm (44 <= 66), one_hold_per_run
python services/orion-durable-runs/evals/hold_fairness.py       PASS
```

## Docker/build/smoke checks

```text
Not deployed (per instruction). No docker build run; the change is pure Python in an existing
image layer plus two env keys with compose defaults.
```

## Deploy-order guard (considered, not added)

durable-runs does not refuse to start or probe the pool first. The pool has no version/capability
verb, and the acquire itself is the probe: with this patch an old, down, or mid-roll pool only
makes runs wait (visibly) and they proceed on the first normal answer. A boot gate would add a way
to hang durable-runs without removing any failure mode. The runbook's order (pool first) is still
the right order; this makes getting it wrong harmless.

## Recovery of the 11 failed runs

Read-only findings (`durable_admission_runs` / `durable_resource_events`): 6 `curiosity.investigate`
(`54537b5b5ccc` from 2026-09-21, the rest 2026-09-25 09:16-18:07) and 5 `self_sense_eval`
(2026-09-25 12:12 - 2026-09-26 00:17), all failed 02:32:18 with the `invalid:` error, none with a
`deadline_at`. Their 11 legacy demands are still there (9 `pending`, 2 `suspended`).

durable-runs has **no supported replay**: `POST /runs` is idempotent on `run_id` (the same id
returns the failed row; a different body under it is a 409), and `POST /runs/<id>/resume` on a
terminal run is a no-op. The only way is a new submission under a new `run_id`:

```bash
# NOT RUN. One run at a time; new run_id and correlation_id, same brief/admission.
RID=5827d47c8a14
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc \
  "SELECT request FROM durable_admission_runs WHERE run_id='$RID'" \
| jq --arg id "${RID}-replay1" --arg corr "$(uuidgen)" --arg now "$(date -u +%FT%TZ)" \
     '.run_id=$id | .correlation_id=$corr | .requested_at=$now' \
| curl -fsS -X POST http://localhost:8124/runs -H 'content-type: application/json' -d @-
```

Recommendation: **do not replay.** Both workflows recur on their own (live history: curiosity
roughly every 2 h, self-sense every 3 h). Right now Hub is blocking new curiosity runs with
`curiosity_investigation_blocked reason=daily_cap` (Hub log, every 5 min through 02:56 UTC), so
they resume when the cap resets; whether these failed runs counted toward the cap is UNVERIFIED.
A replay posted straight to durable-runs would bypass that cap. Self-sense runs have a separate
producer; none has been submitted since 00:17 UTC, and its next 3-hourly run (~03:17) is
UNVERIFIED as of writing. Also, the briefs were built from material that is now 3-110 hours old
(`54537b5b5ccc` is 4.5 days old and had been stuck since 2026-09-25 08:19), and a replayed run is
detached from whatever originated it (new run_id/correlation_id), so its result lands nowhere that
is waiting for it. Replaying all 11 would queue 11 stale GPU runs on the agent lane at once. Let
the next scheduled runs happen after this patch is deployed.

## Review findings fixed

Review ran in a subagent (read-only, with mutations on a scratch copy).

- Finding: `is_pool_trouble` required no lease_id, but the real pool echoes the asked-about
  lease_id on `unknown_verb:*`, so that arm never matched (unit test used a fake reply shape).
  - Fix: match on status + reason prefix only; test uses the real shape.
  - Evidence: `test_a_skew_refusal_to_a_lease_verb_says_nothing_about_the_lease`.
- Finding: a pool event arriving mid-backoff was discarded when the hold had no lease_id -- an
  acquire that timed out but landed and was granted sat idle until `retry_at`.
  - Fix: every hint wakes the run (the pool only emits events for a hold it created).
  - Evidence: `test_a_timed_out_acquire_that_landed_is_woken_by_its_grant_event_mid_backoff`; mutation fails it.
- Finding: refusal events in a later skew episode reused `pool_refused:<request_id>:1` and were
  dropped by `ON CONFLICT DO NOTHING` (silent).
  - Fix: event id also carries `retry_at`.
- Finding: backoff fields survived into a granted/requeued hold, so a later episode started at the old count.
  - Fix: `_without_backoff` strips them in `lease()` GRANTED and `release(keep_requeued)`.
- Finding: the `_beat` skew comment overstated what the TTL guarantees.
  - Fix: comment corrected (same exposure as an unanswered RPC, caught by the next readable beat or the tail guard).
- Finding: four skew branches (`lease`, `guard`, `_end_hold`, hints) were unpinned by tests.
  - Fix: `test_a_status_skew_while_waiting_keeps_the_hold_and_it_is_granted_later`,
    `test_a_skewed_release_is_retried_until_the_pool_can_read_it`,
    `test_a_skewed_boundary_heartbeat_keeps_the_lease`, plus the hint test above.
  - Evidence: removing each branch now fails exactly one test (5 mutations, 5 caught). The release
    test also caught a real bug: the new release-refused log line had 3 placeholders for 4 args.
- Finding: two tests raced wall-clock backoff against Postgres; "delay grows" was never asserted.
  - Fix: backoff tests run on a fake `rt.now`; the skew test asserts `retry_at` gaps of 20 s then 30 s (cap).
- Not changed (documented): failed releases retry every reconcile tick with no backoff (pre-existing
  behaviour for unreachable pools; idempotent, no double-release); a permanent unclassified refusal
  waits indefinitely (see Risks); a class with `on_unavailable: fail` can loop acquire/GONE without a
  pause (pre-existing, not touched here; no durable-run route maps to such a class today).

## Restart required

```bash
# From a worktree of merged main (the wrapper refuses the shared checkout):
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

## Risks / concerns

- Severity: low. Concern: a genuinely permanent refusal whose reason is not in the terminal list
  now waits forever (max one ask per 5 min, each visible) instead of failing. Mitigation: the run
  shows in `/runs/unfinished` with the reason on every event, is cancellable, and its admission
  `deadline_at` (when set) still ends it.
- Severity: low. Concern: backoff delays the first retry after the pool recovers by up to
  `DURABLE_RUNS_POOL_RETRY_MAX_SEC` (300 s). Mitigation: configurable; a pool event about the
  run's own lease bypasses the backoff.

## PR link

(filled in after push)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
