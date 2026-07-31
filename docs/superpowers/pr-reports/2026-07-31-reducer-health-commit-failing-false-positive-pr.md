# `cursor_commit_failing` was a ~20% false positive on a healthy system

Branch: `fix/reducer-health-commit-failing-false-positive`
Date: 2026-07-31
Status: **DONE**

Closes the loop `docs/superpowers/pr-reports/2026-07-13-substrate-health-recheck-debounce-pr.md` opened:
*"If this alert fires a third time, that would be strong evidence of a real, recurring issue (not
just alerting noise) worth a deeper dive."* It fired a third time. It was real, and it was in the
detector.

## Summary

- `reducer_cursor_commit_failing:*` paged CRITICAL three times — 2026-07-13 twice
  (`biometrics_grammar_consumer`), 2026-07-31 once (`execution_grammar_reducer`) — each
  "self-resolving within minutes with no reproducing evidence."
- There was nothing to reproduce. `ReducerHealthSnapshot.classify()` returned `cursor_commit_failing`
  for **any** `last_success_at > last_cursor_advance_at`, which is true on every healthy batch.
- Measured on the live, healthy system: **~22% of observations** for two reducers, with
  `last_error_at=None` on every single sample.
- Fix: the inversion must be accompanied by an actually-recorded error within
  `CURSOR_COMMIT_ERROR_GRACE_SEC` (60s).
- Container logs preserved to `/tmp/substrate-cursor-commit-3rd-fire/` **before** any recreate, per
  the standing instruction from the second investigation.

## The measurement

Two independent runs against live `/grammar/truth` on a fully healthy runtime — every reducer at
`last_error_at=None`, `blocked_failures=0` — computing the inversion from raw timestamps
independently of the deployed classifier:

```text
                        run 1 (40 samples)   run 2 (46 samples, review)
execution_trajectory          32.5%                  21.7%
transport_bus                   --                   21.7%
biometrics                    22.5%                  17.4%
route_grammar                 10.0%                   8.7%
chat_grammar                    --                    0.0%
```

Every inverted sample classified `cursor_commit_failing`. `chat_grammar` at 0% is consistent with it
being the traffic-gated idle lane (`grammar_truth.py`'s `_TRAFFIC_GATED_CURSORS`) — no batches, no
inversions — which also explains why the three fires hit biometrics and execution and never chat.

Confirmed behaviorally against `origin/main`:

```text
origin/main classify() on a healthy in-flight commit, zero errors:
  -> cursor_commit_failing
  last_error_at: None  blocked_failures: 0
```

## Where the window actually comes from

`_process_events_with_poison_isolation()` calls `record_success()` the moment `process_batch()`
returns. The caller then finishes its tick — projection reload, `save_execution_trajectory`,
`save_receipt`, `_write_prediction_error_node` (a **FalkorDB** write) — and only then reaches
`_advance_cursor()`.

The commit originally blamed the cursor commit's own SELECT+UPDATE. Review measured that at **~18ms**
(`cursor_positions[].updated_at` vs `last_cursor_advance_at`), which against a multi-second batch
interval would yield a ~0.3% inversion rate — two orders of magnitude below what is observed.
Back-solving from the measured rate puts the real window near ~1.6s, consistent with a graph write.
Corrected in the code comment, because someone trying to shorten the window would otherwise look in
the wrong file.

## What actually reaches this branch — the review's sharpest finding

**Not `_advance_cursor()`'s own failure paths, despite the name.** Both pass a real `event_id` to
`record_error()`, which sets `blocked_event_id`/`blocked_failures`, and the `blocked_on_event` branch
returns first. Verified by replaying the real call sequence against the live module:

```text
record_success + record_error(event_id="gev_missing")  -> blocked_on_event
record_success + record_error(event_id=None)           -> cursor_commit_failing
```

So a genuinely stuck cursor commit pages as `reducer_blocked:<cursor>` — and always has, before this
patch too. Detection is not lost; it lives elsewhere.

What this branch genuinely covers is the `event_id=None` path: the poll loops' own `record_error()`
after a tick raised **between** `record_success()` and the advance (a failed
`publish_accepted_events`, `save_execution_trajectory`, or `_write_prediction_error_node`). That is a
real condition and the reason the branch stays. `cursor_commit_failing` is a misnomer for it;
renaming is a contract change (the string reaches `degraded_reasons` and the alert text) and was
deliberately left alone.

## Why not the simpler ordering fix

Review noted a more direct option: move `record_success()` after `record_cursor_advance()`, or track
a `pending_advance_since` and classify on how long the advance has been outstanding. That measures
"the commit is not landing" directly — no dependence on a co-occurring error, no grace window, and it
would catch a hung `advance_fn` immediately rather than at the 120s heartbeat.

Not taken here, for two reasons worth stating rather than leaving implicit:

1. Moving `record_success()` changes what `last_success_at` **means** (from "batch processed" to
   "batch processed and committed"). That field is on the `/grammar/truth` contract and read by
   operators; redefining it inside an alerting bugfix is a wider blast radius than the bug warrants.
2. `pending_advance_since` is new state on a shared in-process snapshot, and the branch it would feed
   is — per the finding above — already mostly shadowed by `blocked_on_event`. Building new state to
   better serve a nearly-unreachable branch is the wrong order of operations.

The direct measurement is the better long-term shape. It belongs in a patch that also confronts the
`blocked_on_event` precedence and the misnomer, not bolted onto a false-positive fix.

## Metric quality gate (CLAUDE.md section 0A)

This changes the semantics of a live telemetry classification consumed by an alerting path, so the
gate applies.

1. **Provenance.** `last_success_at` / `last_cursor_advance_at` / `last_error_at` are set by
   `record_success()`, `record_cursor_advance()`, `record_error()` in `app/reducer_health.py`, called
   from `worker.py`'s poll loops and `_advance_cursor`. Traced to the call sites, not inferred.
2. **Independence.** The added term (`last_error_at`) is not a transform of the two it joins — it is
   set by a disjoint code path (error handlers) that the other two never touch.
3. **Theory anchor.** Named, not vibes: a cursor commit that is *in flight* records no error; one
   that is *failing* records one every poll. The predicate now distinguishes those two states
   instead of conflating them with "not yet committed."
4. **Live-data sanity.** Done twice, independently, above. Non-degenerate in both directions: the
   classification still moves (healthy/alive_behind/blocked_on_event all reachable) and the removed
   state was measured, not assumed.
5. **Existing mechanism.** `last_error_at` already existed and was already populated; nothing new was
   introduced to carry this signal.
6. **Reversibility.** One boolean conjunct and one module constant. Trivially removable; no schema,
   manifest, env key, or training default involved.

## Files changed

- `services/orion-substrate-runtime/app/reducer_health.py`: the predicate, the
  `CURSOR_COMMIT_ERROR_GRACE_SEC` constant, and a rewritten rationale.
- `services/orion-substrate-runtime/tests/test_reducer_health_commit_failing_false_positive.py`
  (new): 15 tests.
- `services/orion-substrate-runtime/README.md`, `app/settings.py`: both still told the wrong story
  ("one cursor-advance write losing a race with transient Postgres load"). Corrected — that
  explanation is how the wrong mental model survived two investigations.

## Schema / bus / API changes

None. `classification` values are unchanged; which state maps to `cursor_commit_failing` is narrowed.

## Env/config changes

None. No keys added, removed, or renamed.

`CURSOR_COMMIT_ERROR_GRACE_SEC` is a module constant, not an env key and not a parameter. It briefly
had a keyword parameter; review pointed out `grammar_truth.py` is the only production caller and
never passed it, so the parameter existed solely for two tests that tested the plumbing — the same
"knob with no operator use" smell as an env key, with an extra layer. Removed.

## Tests run

```text
services/orion-substrate-runtime$ pytest tests/test_reducer_health_commit_failing_false_positive.py -q
15 passed

services/orion-substrate-runtime$ pytest tests -q --ignore=tests/test_grammar_consumer_integration.py
23 FAILED/ERROR lines
```

Those 23 are **pre-existing**, established against a detached scratch worktree at `origin/main`:
identical sets, the only diff being one captured log line's line number (`worker.py:540` → `:555`,
from added comments). Review independently reproduced the same baseline via `git archive` and
identified the cause as a module-identity artifact (`app.reducer_health` imported twice under
different `sys.path` roots), unrelated to this change.

Red-before-green: `origin/main`'s `classify()` returns `cursor_commit_failing` for the healthy
in-flight state, reproduced directly (above).

## Evals run

```text
No eval harness exists for services/orion-substrate-runtime.
```

Flagged, not claimed. The behavior is directly observable in the live `/grammar/truth` classification
rate, which is the meaningful check and was measured twice.

## Review findings fixed

- **Finding (MAJOR): the stated mechanism was backwards.** The commit claimed `_advance_cursor()`'s
  `record_error()` calls are what keep the true positive alive; they are intercepted by
  `blocked_on_event` and never reach this branch.
  - Fix: rationale rewritten to name the `event_id=None` poll-loop path that genuinely reaches it,
    and to state the precedence explicitly so nobody "simplifies" the ordering.
  - Evidence: replayed the real call sequence against the live module — `event_id="gev_missing"` →
    `blocked_on_event`; `event_id=None` → `cursor_commit_failing`.
- **Finding (MAJOR): a test asserted a state its own docstring's mechanism cannot produce.**
  - Fix: re-docstringed to the real path, added a precondition assert on `blocked_event_id is None`,
    and added `test_advance_cursor_failure_is_blocked_on_event_not_commit_failing` pinning the
    precedence rule.
- **Finding: the headline test's "exact live window" was not inverted.** The quoted timestamps
  (`20:28:51.632147` / `20:28:52.043960`) have advance 411ms *later* — the recovered post-advance
  state, healthy under the old predicate too.
  - Fix: renamed, rebuilt on the pre-advance state, and the error recorded in the docstring.
- **Finding: the parametrization was inert.** `window_sec` was added to `PREVIOUS_ADVANCE_SEC`, so all
  four cases produced a ~20s inversion, and the predicate never inspects gap width anyway.
  - Fix: now varies the previous-advance staleness across live-measured batch gaps (4 / 13 / 28.1s,
    the observed max on route) plus an extreme case.
- **Finding: unused keyword parameter.** Removed (see Env/config).
- **Finding: `transport_bus` affected at 21.7% and never mentioned.** Added to the measurement table
  in code and here.
- **Finding: stale README and settings.py comments.** Both corrected.
- **Finding: boundary untested / `alive_behind` fallthrough untested.** Both added.
- **Finding: no PR report, metric gate unrecorded.** This document.

Verified clean by review, not re-litigated here: no path exists where a batch succeeds, the cursor
does not advance, and no error is recorded — `advance_fn` is an unconditional `INSERT ... ON CONFLICT
DO UPDATE` inside a transaction (commits or raises), the poison-isolation partial path re-raises
before `record_success`, and an empty batch returns before it.

## Restart required

```bash
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
```

## Risks / concerns

- **Severity: should-know. Wrong-reason false positive, bounded.** Any error recorded with
  `event_id=None` leaves `last_error_at` fresh for 60s, during which a benign inversion classifies
  `cursor_commit_failing`. Quantified with the live intervals (900s check, 15s recheck, ~20%
  inversion rate): **~0.3% per transient error**, versus the ~20% it replaces.
- **Severity: note. A hung `advance_fn` is now detected up to 120s later, under a different reason.**
  The poll loop suspends at `await asyncio.to_thread(...)`, so `record_tick` stops and
  `dead_no_heartbeat` fires at `reducer_heartbeat_stale_sec` (live: 120.0), surfacing as
  `reducer_heartbeat_stale:<cursor>`. Detection is not lost, only renamed and delayed.
- **Severity: note. Theoretical hole.** If `GrammarEventV1.event_id` were ever the empty string, the
  `if last_event_id:` guards would skip the advance with no error recorded — now classified healthy.
  Negligible in practice (`event_id` is the `grammar_events` key, every producer emits `gev_<hex>`),
  recorded rather than coded around.
- **Severity: note. `cursor_commit_failing` is a misnomer** for the condition it actually detects.
  Renaming touches `degraded_reasons` and alert text, so it is a contract change for its own patch.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1549
