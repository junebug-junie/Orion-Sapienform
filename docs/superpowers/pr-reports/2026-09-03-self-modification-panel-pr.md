# Show what Orion changed about itself, without a database query

Item 3 of `docs/plans/substrate/PR_self_modification_accountability_v1.md`. Follows PR #2050, which recorded the history and released the surface lock; this makes both visible.

## Summary

- Two rows on the hub's existing autonomy-readiness panel: the last change Orion made to its own routing threshold, and which mutation surfaces are locked.
- A lock past its rollback window renders `OVERDUE (window elapsed, still held)` — the 13-hour hold from 2026-09-02, made visible.
- A broken history renders `HISTORY UNREADABLE`, a broken lock read renders `LOCKS UNREADABLE`. Neither can be mistaken for a calm reading.
- Added to the existing panel rather than a new page: that surface already carries the scheduler gates and recent applies/rollbacks, and this belongs beside them.

## Outcome moved

Four questions that previously required `psql` now answer themselves on page load:

| question | before | after |
| --- | --- | --- |
| what did Orion last change about itself? | `psql` — and unanswerable before #2050 | `last self-change: 0.5 -> 0.58 by mutation_apply at …` |
| what is the value now? | `psql` | `routing threshold: 0.58 (source postgres)` |
| is a surface locked, and for how long? | `psql` | `routing held 13.0h since 2026-09-02T04:11:17Z OVERDUE …` |
| is settlement running? | not observable at all | the `OVERDUE` marker |

## Current architecture

`/api/substrate/autonomy-readiness` → `_autonomy_readiness_payload()` → rendered by `updateAutonomyReadinessPanel` in `services/orion-hub/static/js/app.js`, into `#autonomyReadinessOverview` (`templates/index.html:3295`). The panel already showed scheduler gates, surface counts, recall readiness and recent activity; it did not show anything about a change Orion had actually made.

## Files changed

- `services/orion-hub/scripts/api_routes.py`: `_self_modification_panel_payload()`, wired into the `routing` block.
- `services/orion-hub/static/js/app.js`: three rows and their branch logic.
- `orion/substrate/mutation_control_surface.py`: re-adds `chat_reflective_lane_threshold_history()`, dropped in #2050's review as a producer with no consumer. This is the consumer.
- `services/orion-hub/tests/test_self_modification_panel.py`: new, 9 tests.
- `docs/plans/substrate/PR_self_modification_accountability_v1.md`: acceptance check 4 status.

## Schema / bus / API changes

- Added: `routing.self_modification` on the autonomy-readiness response — `current`, `history`, `history_available`, `history_error`, `last_change`, `surface_holds`, `surface_holds_error`.
- Removed: `applied_patch` / `rollback_payload` were in the first commit's payload and are not in the final one; nothing rendered them (see Risks).
- Behaviour changed: none outside this endpoint.
- Compatibility: additive; existing keys untouched.

## Env/config changes

None.

## Tests run

```text
pytest orion/substrate/tests -q                                  -> 681 passed
pytest services/orion-hub/tests/test_self_modification_panel.py \
       services/orion-hub/tests/test_substrate_mutation_scheduler_runtime.py \
       services/orion-hub/tests/test_substrate_mutation_manual_route_routing.py \
       services/orion-hub/tests/test_autonomy_runtime_ui_panel.py \
       services/orion-cortex-orch/tests/test_control_surface_isolation_guard.py -q  -> 67 passed
node --check services/orion-hub/static/js/app.js                 -> OK
```

Mutation-checked against the real files, restored by file copy (never `git stash`, shared across worktrees here):

| Mutation | Caught |
| --- | --- |
| `held_for_sec` measured from `created_at` not `applied_at` | overdue test |
| `window_elapsed` hardcoded false (never flags a stuck surface) | overdue test |
| `last_change.previous_value` ← `new_value` | replaced-value test |
| `history_available` dropped | calm-vs-fault test |
| degraded store still reports `history_available` | unreadable-history test |
| **JS reads `snapshot.self_modification`** (wrong key path) | JS contract test |
| **JS checks the empty branch before the error branch** | JS contract test |

The last two are JS-only. Both survived every test in the repo before this PR's contract tests existed.

## Evals run

```text
none — a read-only presentation surface over state PR #2050 already tests
```

## Docker/build/smoke checks

```text
node --check on the changed static asset (above). No dependency, port, or boot-config surface touched.
```

## Review findings fixed

- **Finding (HIGH): a broken history table rendered identically to a calm one** — the exact property the first commit's message claimed it had. `RuntimeControlSurfaceStore.history()` swallows backend errors and returns `[]`, so `history_error` could never be set and `history_available` was always `True`. Reviewer reproduced it: dropping the table produced byte-identical output to a healthy empty store.
  - Fix: ask the store whether it is degraded instead of inferring health from an empty result; check the error branch first in the JS.
  - Evidence: `test_an_unreadable_history_does_not_read_as_an_empty_one` drops the real table; fails when the check is removed.
- **Finding (HIGH): `surface_holds_error` was written and read by nothing** — a failed lock read rendered `none held`, hiding exactly the lock this panel exists to surface.
  - Fix: set it from the mutation store's degraded state; render `LOCKS UNREADABLE`.
  - Evidence: covered by the JS contract test.
- **Finding (HIGH): the frontend half had zero coverage.** A wrong key path would render "unavailable" forever with every test green; the reviewer confirmed identical FAILED sets mutated vs baseline across all 81 files that read `app.js`.
  - Fix: two tests pinning the key path the JS reads, the branch order, and the payload nesting from the other side.
  - Evidence: both JS mutations now fail.
- **Finding (MEDIUM): the test named for the distinction never tested it** — it only built a healthy empty store, asserting a postcondition that could not fail.
  - Fix: split into a calm case and a real broken-table case.
- **Finding (MEDIUM): `data["current"]` cost two DB round trips and nothing rendered it.**
  - Fix: render it. Acceptance check 4 asked for the current value; the read was already being paid for and thrown away.
- **Finding (MEDIUM): the panel does not poll, so a server-computed elapsed freezes on an open dashboard.**
  - Fix: render `held_since` alongside the duration. An absolute timestamp does not go stale. Auto-refresh not added — see Risks.
- **Finding (LOW): `applied_patch`/`rollback_payload` were emitted and never rendered.** `applied_patch` is `dict[str, Any]` for any surface, and this router carries no auth dependency, so a future adoptable cognitive surface's patch body would ride out automatically.
  - Fix: dropped from the payload.
- **Finding (cosmetic): the flagship 13-hour case rendered as `780m`.** Fixed: seconds under 90s, minutes under 90m, hours above.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

Static assets are cache-busted (`templates/index.html:3885`), so no manual cache clear.

## Risks / concerns

- **Severity: low.** The panel still does not auto-refresh; `held_for_sec` is computed server-side at request time. Mitigated by rendering `held_since`, which cannot go stale, but a dashboard left open shows a frozen duration until reloaded. Adding a timer to this panel is a page-behaviour change and was left out of a presentation-only PR.
- **Severity: low.** `SUBSTRATE_MUTATION_STORE._adoptions` is read directly. Consistent with existing code in the same function, and no public by-id accessor exists. The store loads once at construction and is never refreshed, which is safe only because the hub runs a single uvicorn worker and the apply loop lives in-process. Adding a worker, or moving the apply loop to `orion-substrate-runtime`, would make this lock row silently stale until restart. Worth a follow-up guard.
- **Severity: low.** `store.degraded()` is sticky — `_last_error` is never cleared — so one transient backend error makes the panel report `HISTORY UNREADABLE` until the process restarts. Erring toward over-reporting a fault was deliberate; the opposite direction is the bug this PR fixes.

## Follow-ups

1. **Feed the monitor** — nothing supplies a post-adoption delta, so every change is kept because time passed, not because it helped. This panel now makes that visible; it does not fix it.
2. Item 4, real latitude: the threshold Orion can move is hardcoded `0.58`, and the confidence it is compared against is a keyword lookup table. `AUTO_ROUTER_LLM_ENABLED=false`.
3. A staleness guard on `_adoptions` if the hub ever runs more than one worker.

## PR link

<pending>
