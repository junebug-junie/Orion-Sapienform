# GitHub compactor: stop the permanent "degraded" badge and the notification flood

## Summary

- Attention signals no longer page off `health == "degraded"` alone — a trailing-5-run property that outlives a failure by days and kept nagging on schedules that were succeeding.
- Failed scheduled dispatches now retry with bounded, exponential backoff instead of being rewound to a past `next_run_at` and re-claimed at scheduler-poll cadence forever.
- An over-budget compactor `card_summary` is now trimmed to its cap instead of raising and discarding an already-validated digest.
- Both compactor bootstraps move to `notify_on="failure"`, and reconcile that onto their own already-seeded live records (bootstrap is seed-once, so a changed default would otherwise never land).
- Review follow-ups in the same branch: hung dispatches are now reaped (retiring a `_claim_ttl` that was accepted and never read), the retry budget is persisted rather than derived from a truncated history, a failed dispatch can no longer revive a cancelled schedule, and the operator-facing `notify_on` update now writes the copy that actually governs notification.
- New tests cover each failure mode, including one that caught a real bug in this patch's own first implementation.

## Outcome moved

`github_compactor_pass` notification volume, measured from `notify_requests`:

| day | `orion.workflow.failed` | attention nags | workflow succeeded? |
|---|---|---|---|
| 2026-08-20 | 343 | 4 | no |
| 2026-08-22 | 40 | 5 | yes (after 40 retries) |
| 2026-08-27 | 2 | 2 | yes |
| 2026-08-28 | 0 | 8 | yes |
| 2026-08-29 | 0 | 8 | yes |
| 2026-08-30 | 3 | 8 | yes |
| 2026-08-31 | 0 | 8 | yes |
| 2026-09-01 | 0 | 8 | yes |

The attention nag (published twice per fire: `workflow.schedule.attention.v1` + `orion.chat.attention`, 6h cooldown) ran continuously from 2026-08-21 to 2026-09-01 — roughly 100 messages, every one of them about a job that was completing successfully.

After this patch, the same days produce: 0 messages while succeeding, one "recovered" transition when a failure clears, and at most 3 failure notifications for a genuinely broken day instead of 343.

## Current architecture (before this patch)

`orion-actions` owns `WorkflowScheduleStore`, a JSON-backed store at `/data/orion-actions/workflow_schedules.json`. A 45s scheduler loop in `main.py` calls `claim_due()`, dispatches each claimed slot to `orion-cortex-orch` over a blocking RPC, then calls `mark_dispatch_succeeded`/`mark_dispatch_failed`, then `evaluate_attention_signals()`.

Three properties combined into the observed behavior:

1. `_derive_analytics` scores `health` over the **last 5 runs**; any single failure in that window yields `degraded`.
2. `mark_dispatch_failed` rewound `next_run_at` to `claimed_for_run_at`, a timestamp already in the past.
3. `evaluate_attention_signals` paged whenever `health == "degraded"` and `recent_failure_count >= 2` — a guard that cannot discriminate, because retry bursts are always ≥ 2.

Separately, `assert_fields_within_budget` raised `compactor_output_over_budget:<field>` on over-long digest prose, which `workflow_runtime` converted into a `WorkflowExecutionError` — a hard failure feeding property 2.

## Architecture touched

- `orion-actions`: schedule store retry policy, attention condition, a new `set_notify_on` seam, bootstrap reconcile.
- `orion-cortex-orch`: both compactor digest lanes now repair-and-log instead of raise.
- Shared `orion/cognition/compactor`: the budget primitive changes from assert-shaped to fit-shaped.

## Files changed

- `orion/cognition/compactor/budget.py`: `assert_fields_within_budget` → `fit_fields_within_budget`, which trims at a word boundary and returns which fields it touched. The ellipsis is paid for out of the budget, so `len(value) <= max_chars` holds exactly.
- `orion/cognition/github_compactor/digest.py`, `orion/cognition/chat_history_compactor/digest.py`: `assert_*_within_budget` → `fit_*_within_budget`, returning `(digest, trimmed_field_names)`.
- `services/orion-cortex-orch/app/workflow_runtime.py`: both call sites repair instead of raising, and log `compactor_digest_trimmed_to_budget` with the correlation ID so the repair is inspectable rather than silent.
- `services/orion-actions/app/workflow_schedule_store.py`: bounded retry with exponential backoff, `_consecutive_failures`, the narrowed attention condition, and `set_notify_on`.
- `services/orion-actions/app/workflow_schedule_bootstrap.py`: `BOOTSTRAP_NOTIFY_ON = "failure"` plus `_reconcile_bootstrap_notify_on`.
- `services/orion-actions/app/settings.py`, `services/orion-actions/.env_example`, `services/orion-actions/app/main.py`: the two new retry keys.
- Tests/evals across both services updated to the new contract and extended.

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: `assert_fields_within_budget` → `fit_fields_within_budget`; `assert_digest_within_budget` → `fit_digest_within_budget`; `assert_chat_compactor_digest_within_budget` → `fit_chat_compactor_digest_within_budget`. All callers updated; no references to the old names remain outside explanatory comments.
- Behavior changed: an over-budget digest now persists a trimmed summary instead of failing the workflow. New store event kinds `schedule_retry_scheduled`, `schedule_retry_budget_exhausted`, `schedule_notify_on_updated` are appended to the existing event log (additive; the log is already a free-form `kind` + `extra` shape).
- Compatibility notes: no persisted-store migration. `_consecutive_failures` is derived from existing run history, so an already-running store needs no backfill.

## Env/config changes

- Added keys: `ACTIONS_WORKFLOW_SCHEDULE_MAX_DISPATCH_ATTEMPTS=3`, `ACTIONS_WORKFLOW_SCHEDULE_RETRY_BACKOFF_SECONDS=300`
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: yes
- local `.env` synced: yes — `python3 scripts/sync_local_env_from_example.py orion-actions --all-keys`, run from the worktree so the branch's `.env_example` is the one read; both keys confirmed present at `services/orion-actions/.env:146-147`
- skipped keys requiring operator action: none

## Notes on the metric quality gate

No new metric or telemetry channel is introduced. `_consecutive_failures` is a control input to a retry policy, not a signal wired into a model or cognition loop, and it is read only by the code that writes it.

The patch does *retire* a bad instrument rather than merely excluding it: `assert_fields_within_budget` is deleted, not left ticking behind a narrowed caller. No production or test path still calls it.

## Tests run

Services must be run separately — combining `orion-actions` and `orion-cortex-orch` in one pytest invocation hits a pre-existing `app` package collision (17 collection errors), which is why CI runs them with per-service `PYTHONPATH`.

```text
$ .venv/bin/python -m pytest services/orion-actions/tests -q
166 passed, 123 warnings in 7.87s

$ .venv/bin/python -m pytest services/orion-cortex-orch/tests/test_workflow_lane.py -q
65 passed, 25 warnings in 7.85s

$ .venv/bin/python -m pytest orion/cognition/compactor orion/cognition/github_compactor orion/cognition/chat_history_compactor -q
44 passed in 0.42s

$ .venv/bin/python -m pytest tests/test_github_compactor_memory_cards.py -q
1 passed in 0.48s
```

Full `services/orion-cortex-orch/tests`: **34 failures on this branch, the same 34 as merge-base `38f1494d8`** — zero new, zero fixed (two entries change name only, from `*_fails_without_persist` to `*_is_trimmed_and_persisted`).

One caveat worth recording, because the naive check was misleading: those two over-budget lane tests were *already* red in the full-suite run on main, from unrelated test-order pollution (`recall_pg_dsn_unavailable` short-circuits the persist). So a same-set comm-diff looked clean while this patch had genuinely broken them. Running them **in isolation** on both sides is what surfaced it — they pass isolated on main and now pass isolated here (`65 passed` for the whole file).

### CI static gates

Derived from `.github/workflows/orion-static-gates.yml` rather than memory. All 10 pass:

```text
check_metric_lineage.py --gate      check_compose_no_relative_mounts.py
check_definition_drift.py --gate    check_journal_dispatch_registry.py
check_inner_state_registry.py       check_daily_schedule_collisions.py
check_scripts_dir_no_stdlib_shadow  check_system_health_producers.py
check_service_hostname_refs.py      check_control_surface_store_parity.py
```

The three gates CLAUDE.md §17 names — `check_env_template_parity.py`, `check_schema_registry.py`, `check_bus_channels.py` — **do not exist in `scripts/`**. Not run, not claimed.

## Evals run

```text
$ .venv/bin/python -m pytest services/orion-cortex-orch/evals -q
5 passed in 0.30s
```

`test_eval_over_budget_digest_fails_loud` is replaced by `test_eval_over_budget_digest_is_repaired_not_discarded`, asserting the digest still yields a usable card.

**Eval gap:** `services/orion-actions` has **no `evals/` directory**. The scheduler-side changes (retry budget, attention threshold, reaper) are covered by unit tests only. Worth a follow-up eval harness measuring notification volume per schedule-health episode, since that is the quality this patch is actually moving and no test asserts it end-to-end.

### Mutation testing

Nine mutations against the new guards; **all nine killed**. Two survived the first pass and are now covered:

| mutation | result |
|---|---|
| remove the stale-claim reaper | 1 failed |
| reaper never expires (TTL → ∞) | 1 failed |
| drop the cancelled/paused guard | 1 failed |
| counter never persisted | 9 failed |
| success does not reset counter | 2 failed |
| `apply_management` skips the embedded copy | 1 failed |
| attention pages on first failure (`>= 1`) | 2 failed *(survived until a test was added)* |
| budget: remove the `limit < 2` branch | 1 failed |
| budget: `value or ""` → `value` | 1 failed *(survived until a test was added)* |

## Docker/build/smoke checks

```text
$ scripts/safe_docker_build.sh orion-actions config      -> rc=0
$ scripts/safe_docker_build.sh orion-cortex-orch config  -> rc=0
```

The worktree has no root `.env` (gitignored, lives only in the primary checkout), so the compose files were resolved by symlinking the live `.env` files in — the symlinks are themselves gitignored and are not part of the commit.

Settings load verified against the real `.env`, not just the template:

```text
max_dispatch_attempts = 3
retry_backoff_seconds = 300
store_path            = /data/orion-actions/workflow_schedules.json
```

Live store state confirming a clean deploy (read from the running container):

```text
chat_history_compactor_pass  consecutive_failures = 0  last_result = completed
github_compactor_pass        consecutive_failures = 0  last_result = completed
orphaned dispatched rows: 1   (2026-08-20T16:24:52Z, 13 days stale)
```

No service was rebuilt or restarted. This is **not deployed**.

## Review findings fixed

- **Finding: the attention narrowing opened a hole — a hung `dispatched` run is the newest run, so `most_recent_result_status` reads `"dispatched"` and a genuinely stuck schedule goes silent. Root cause: `_claim_ttl` was accepted by the constructor and never read anywhere in the service, so nothing ever reaped a hung dispatch.**
  - Fix: `_reap_stale_claims()` at the top of `claim_due` routes any `dispatched` run older than `_claim_ttl` through the normal failure path (`_mark_failed_locked`, error `claim_expired_after_<ttl>s`). This retires the dead `_claim_ttl` rather than working around it, and a hung RPC is exactly what a bounded retry is for.
  - Evidence: `test_hung_dispatch_is_reaped_and_still_pages` — silent inside the TTL, reaped past it, attention fires. Mutation: removing the reaper, or making the TTL never expire, each turns a test red.

- **Finding: the retry budget was derived from run history, but `_persist` truncates to `history_limit` while the in-memory list is not trimmed — so a restart hands back a spent budget. The live file is already sitting exactly at its 200-run cap.**
  - Fix: the count is persisted on the record (`metadata["consecutive_failures"]`), incremented on failure and cleared on success. No migration: an absent key reads 0 and self-corrects on the next outcome.
  - Evidence: `test_retry_budget_survives_run_history_truncation` drives 3 failures, floods the window with 25 unrelated schedules, reloads from disk, and asserts the budget is still spent. Reproduced the original defect first (3 → 0 across a restart at `history_limit=20`).

- **Finding: `apply_management`'s `update` wrote only the record field and `execution_policy`, leaving the authoritative embedded copy stale — so an operator changing `notify_on` saw no change in what notified them, and the new bootstrap reconcile would then overwrite their setting at the next restart.**
  - Fix: extracted `_apply_notify_on()` and routed both `set_notify_on` and the management update through it.
  - Evidence: `test_management_update_of_notify_on_reaches_the_authoritative_copy`. Mutation: reverting the management path to the record-field-only write turns it red.

- **Finding: a dispatch failure resurrects a cancelled or paused schedule — and this patch upgraded that from "re-armed at a past slot" to "explicitly scheduled to run again in 5 minutes".**
  - Fix: the retry branch is guarded with `schedule.state not in {"cancelled", "paused"}`.
  - Evidence: `test_a_failed_dispatch_does_not_revive_a_cancelled_schedule` — cancel mid-flight, fail the dispatch, assert still `cancelled` and never re-claimed.

- **Finding: the new condition pages on the *first* failure, where the old `recent_failure_count >= 2` did not — four messages for a blip the next retry fixes.**
  - Fix: the condition is now the retry budget running out (`consecutive_failures >= max_dispatch_attempts`), i.e. the point where the store has stopped retrying and only a human can move it. This also makes the `most_recent == "failed"` clause unnecessary.
  - Evidence: `test_a_blip_that_the_retry_fixes_never_pages` (zero signals for the whole episode) and `test_repeated_failures_still_page_once_the_budget_is_spent` (silent at 1 and 2, pages at 3).

- **Finding: `fit_fields_within_budget`'s "guaranteed `len(value) <= max_chars`" has counterexamples at `max_chars <= 1` — 989 violations in a 20k random sweep.**
  - Fix: a `limit < 2` branch takes a hard slice, since there is no room to spend a character on the ellipsis.
  - Evidence: re-swept 20,000 random inputs (multibyte, whitespace-dominant, unbroken tokens) → **0 violations**. `test_fit_fields_within_budget_holds_the_cap_below_ellipsis_width` pins limits 0/1/2.

- **Finding: two mutations survived — the `dispatched` skip in the failure counter, and `value or ""` for a `None` field.**
  - Fix: the counter is no longer derived from history, so the first mutation no longer exists as a code path; the orphan case it guarded is now covered by the reaper test. Added `test_fit_fields_within_budget_treats_none_as_empty` for the second.
  - Evidence: all 9 mutations killed on re-run.

- **Finding: two READMEs documented the removed fail-loud behavior.**
  - Fix: `orion/cognition/compactor/README.md:5` and `services/orion-cortex-orch/README.md:60` rewritten to describe repair-and-log, including that an over-budget digest no longer consumes the `quick` retry route.
  - Evidence: no reference to `assert_fields_within_budget` remains outside explanatory comments.

- **Finding (minor): `schedule_retry_scheduled` recorded `next_run_at` rather than the computed `retry_at`, so the event could claim a retry was scheduled when the `retry_at < next_run_at` guard declined it.**
  - Fix: the event now carries `retry_at` and an explicit `applied` boolean alongside the resulting `next_run_at`.

- **Finding (minor): `set_notify_on`'s `if embedded:` meant a record with no embedded policy could never report "unchanged", so every boot wrote an event, bumped `revision`, and persisted.**
  - Fix: the unchanged-check treats an absent embedded policy as nothing to disagree with.

### Accepted, not fixed

- **`model_copy(update=...)` converts `journal_title`/`journal_body` from `None` to `""` when any field is trimmed.** Behaviorally inert — both consumers (`workflow_runtime.py:2478`, `:2481`) already coerce with `or ""`.
- **The upstream digest failures are not fixed.** The two large bursts were `structured_output_rejected` (130 runs, 2026-08-20) and `invalid_json: 2 validation errors … journal_title/journal_body missing` (40 runs, 2026-08-22). This patch caps their blast radius at 3 LLM runs/day and stops them reaching Juniper as a flood, but the digest verb itself still fails on those days. See Risks.

## Restart required

```bash
sudo docker compose \
  --env-file .env \
  --env-file services/orion-actions/.env \
  -f services/orion-actions/docker-compose.yml \
  up -d --build orion-actions

sudo docker compose \
  --env-file .env \
  --env-file services/orion-cortex-orch/.env \
  -f services/orion-cortex-orch/docker-compose.yml \
  up -d --build orion-cortex-orch
```

`orion-actions` must restart for two reasons, not one: the new env keys are read at boot, **and** `_reconcile_bootstrap_notify_on` runs in the startup path — that is what flips the two live schedules from `notify_on="completion"` to `"failure"`.

Expected on the first `orion-actions` boot:
1. One `schedule_notify_on_updated` event per compactor schedule.
2. The 2026-08-20 orphaned `dispatched` row is reaped and recorded as `claim_expired_after_300s`.
3. `github_compactor_pass` currently sits at `condition=degraded, state=active`, so exactly one final `recovered` message — then silence.
4. The badge stays `degraded` for ~2 more daily runs until 08-30's failures age out of the 5-run window. That is honest, not a bug.

## Risks / concerns

- **Severity: low.** Concern: the reaper's first run marks the 13-day-old orphan as failed, which sets `consecutive_failures = 1` on `github_compactor_pass`. Mitigation: the budget is 3 and the next successful run clears it; it cannot page on its own.
- **Severity: low.** Concern: attention now waits for 3 consecutive failures on the `degraded` branch, so a schedule failing exactly once per day takes 3 days to page. Mitigation: the `failing` branch (2+ failures, zero successes) is untouched and still escalates immediately; and each failed daily run still sends its own `orion.workflow.failed` notification.
- **Severity: medium.** Concern: the underlying digest-verb failures are unfixed. On a bad LLM day the compactor still produces no digest — now failing quietly (3 runs) instead of loudly (343). Quiet failure is the point, but it does make a persistent breakage less visible than it was. Mitigation: the `schedule_retry_budget_exhausted` event and the attention signal both fire on exactly that case. Follow-up: give the GitHub lane the chat lane's `["chat", "quick"]` route fallback, and investigate `structured_output_rejected` at the gateway.
- **Severity: low.** Concern: `graphify-out/graph.json` was not regenerated. Regenerating the 43 MB artifact into a scheduler fix would dwarf the diff and risks the known destructive-update bug for no benefit here.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2031
