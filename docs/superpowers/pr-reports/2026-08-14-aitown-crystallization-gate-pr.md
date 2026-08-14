# PR: gate ai-town out of the crystallization queue, purge the backlog, fix the review UX

## Summary

- **ai-town no longer reaches the human review queue.** `memory.turn.persisted.v1`
  now carries `source_platform`; a consolidation window whose turns *unanimously*
  come from an external platform auto-activates instead of queueing. Orion still
  forms, projects and recalls ai-town memory — Juniper is never asked to approve
  it turn-by-turn.
- **Backlog purged: governor queue 621 → 22.** 599 rows status-flipped
  proposed→rejected with a `memory_crystallization_history` row each. 0 errors,
  nothing deleted, fully reversible.
- **Review UX:** multi-select + bulk approve/reject, whole-row click to open,
  action buttons pinned to the top of the detail pane, per-turn evidence drop,
  and a fix for the stale-detail bug that made the item after a decision error out.
- **Analysis delivered:** no auto-gate can be built from the existing scores —
  `salience` is provably pinned at 1.0 on this whole path. Written up in
  `docs/superpowers/specs/2026-08-14-crystallization-queue-auto-gate-analysis.md`.
- **Two real bugs found and fixed along the way:** the Hub's cache-bust token
  hardcoded four filenames (so editing the queue's own JS produced no new `?v=`),
  and the queue UI never cleared its detail pane after a decision.

## Outcome moved

Human review workload on the crystallization queue: **621 items → 22**, and the
22 are all confirmed to contain at least one real turn with Juniper. Going
forward the gate keeps ai-town out at formation, so the queue stops refilling
with NPC dialogue.

## Current architecture (before this patch)

```
chat.history ──> sql-writer ──> chat_history_log (client_meta.external_room.platform)
                     │
                     └─> memory.turn.persisted.v1  ← platform dropped here
                              │
                       orion-memory-consolidation
                              │  _get_open_window()  ← ONE global open window
                              ├─ classify_turn / append_turn
                              └─ close ──> build_crystallization_from_window()
                                              kind = stance (dominant_shift=STANCE)
                                                 │
                                       resolve_formation_policy()
                                          stance ∈ GATED_KINDS ──> GOVERNOR_QUEUE
                                                 │
                                          Hub review queue (621 items)
```

The platform was recorded at the first hop and never propagated, so nothing
downstream could tell an NPC line from Juniper's.

## Architecture touched

| seam | change |
|---|---|
| `memory.turn.persisted.v1` | new optional `source_platform` field |
| sql-writer emit | stamps it from `client_meta.external_room.platform` (both the live payload path and the DB-row fallback path) |
| consolidation window row | per-turn `source_platform` in `turn_correlation_ids` |
| `provenance` | `source_platform`, set only on unanimous windows |
| `resolve_formation_policy` | new external-platform branch |
| Hub API | `POST /proposals/bulk`, `DELETE /{id}/evidence/{source_id}` |

## Files changed

- `orion/schemas/memory_consolidation.py`: `MemoryTurnPersistedV1.source_platform`
- `services/orion-sql-writer/app/worker.py`: `_chat_source_platform()` + stamps the field on both emit paths
- `services/orion-memory-consolidation/app/window_state.py`: persists it per turn
- `orion/memory/crystallization/intake_consolidation_window.py`: `_window_source_platform()` unanimity rule → provenance
- `orion/memory/crystallization/formation_policy.py`: the gate itself + `DEFAULT_AUTO_ACTIVATE_PLATFORMS`
- `orion/memory/crystallization/formation_executor.py`: forwards the platform set into its own policy re-resolution
- `orion/memory/crystallization/intake_pipeline.py`: threads the setting through
- `services/orion-memory-consolidation/app/settings.py`: `MEMORY_FORMATION_AUTO_ACTIVATE_PLATFORMS` + parsed property
- `services/orion-hub/scripts/crystallization_routes.py`: bulk decide + evidence delete
- `services/orion-hub/scripts/main.py`: cache-bust token globs `static/js/*.js`
- `services/orion-hub/static/js/memory-crystallization-ui.js`: checkboxes, bulk bar, row-click, top-pinned actions, detail-close, per-turn drop, platform badge
- `scripts/smoke_aitown_crystallization_gate.py`: read-only live replay of the gate
- `scripts/bulk_reject_aitown_proposals.py`: one-shot backlog purge with the §14 backfill protocol
- `docs/.../2026-08-14-crystallization-queue-auto-gate-analysis.md`: the auto-gate analysis

## Schema / bus / API changes

- **Added:** `MemoryTurnPersistedV1.source_platform: Optional[str] = None`
- **Added:** `POST /api/memory/crystallizations/proposals/bulk` → `{ids, action, reason}`
- **Added:** `DELETE /api/memory/crystallizations/{id}/evidence/{source_id}`
- **Added:** `provenance.source_platform` on consolidation-window crystallizations
- **Behavior changed:** a unanimous external-platform window now auto-activates
  instead of queueing
- **Compatibility:** the field is optional and defaults to `None` (= not
  external), so old rows and old producers behave exactly as before.
  **`MemoryTurnPersistedV1` is `extra="forbid"`**, so a consumer on old code
  hard-fails validation on the new field. **Deploy orion-memory-consolidation
  before orion-sql-writer.**

## Env/config changes

- Added key: `MEMORY_FORMATION_AUTO_ACTIVATE_PLATFORMS` (default `aitown`;
  empty string disables the gate)
- `.env_example` updated: yes (`services/orion-memory-consolidation/.env_example`)
- `docker-compose.yml` updated: yes
- local `.env` synced with `python3 scripts/sync_local_env_from_example.py`: yes —
  `orion-memory-consolidation: +MEMORY_FORMATION_AUTO_ACTIVATE_PLATFORMS='aitown'`,
  verified at `services/orion-memory-consolidation/.env:79`
- skipped keys requiring operator action: none

## Tests run

Final counts, after the review fixes below:

```text
services/orion-memory-consolidation/tests           97 passed          (71 on main; +26 new)
services/orion-sql-writer/tests                    307 passed, 10 failed, 3 skipped
                                                   (295 passed / identical 10 failures on main — pre-existing)
services/orion-hub/tests (full)                   1261 passed, 32 failed, 5 skipped
                                                   (1253/33 on main before this branch)
```

Hub failure sets were compared against a `main` baseline run rather than eyeballed.
The single worktree-only entry is `test_substrate_mutation_manual_route_routing.py`,
which failed a **different test from the same file on each of four runs**
(`..._changes_real_live_routing_surface`, `..._succeeds_for_auto_promote_and_can_rollback`,
`..._dry_run_produces_trial_and_decision`, then `..._succeeds_for_auto_promote...` again)
— order/state-dependent, and reverting this branch's three hub files in the same
worktree only moved the failure rather than clearing it.

One genuine regression was caught this way and fixed:
`test_main_mtime_token_includes_organ_signals_js` grepped `main.py` for a filename
literal that the cache-bust glob removed. Both it and its new counterpart are now
behavioral (touch the file → token must move; restore → token must return) and
were mutation-tested against the real `main.py`: reverting to a hardcoded
`["app.js", index.html]` list fails both, and only both.

New coverage, 63 tests:
- `test_external_platform_gate.py` — 23
- `test_window_state_source_platform.py` — 3
- `test_memory_turn_source_platform.py` — 12
- `test_crystallization_review_queue_ux.py` — 25

Covering:
- unanimity in both directions (all-external, all-direct, mixed, two platforms,
  empty, missing key, empty-string platform)
- policy routing (external auto-activates, direct/mixed/unlisted still queue,
  empty allowlist disables, duplicate still wins)
- kind scoping (only `stance` is bypassable; `contradiction`/`decision`/
  `attractor`/`failure_mode` still queue for an allowlisted platform)
- privacy precedence — with an explicit test pinning that these two guards are
  **unreachable** via this producer today, so the ordering is not mistaken for a
  live rail
- `formation_executor.auto_activate` honors a caller-supplied allowlist — it
  re-resolves policy independently, so an unforwarded set would be silently
  dropped on the deciding path
- sql-writer platform parsing (dict, JSON string, 8 malformed shapes) and the
  real emit path off a real `chat.history` envelope
- `window_state.append_turn` persists the platform on both the new-window and
  append branches
- bulk decide: all-succeed, partial failure, dedup, bad action, empty ids, size
  cap, a lower approve cap than reject, path not captured as an id
- evidence delete: correct row targeted, history written, `crys_` id normalized,
  404 unknown, 409 last turn, 409 on active
- UI: no refetch on checkbox toggle, select-all excludes undecidable rows,
  client chunks stay under the server caps, server error detail surfaced,
  `openDetail` rejection handled, actions above the detail body, and a
  mutation-tested cache-bust gate

## Evals run

```text
No eval harness exists for orion-memory-consolidation or orion-sql-writer
(services/*/evals/ absent for both — confirmed, not assumed).
```

The closest thing to an eval here is the live replay smoke below, which measures
the gate's real decision on the real corpus rather than on fixtures.

## Docker/build/smoke checks

Before the purge:

```text
$ python3 scripts/smoke_aitown_crystallization_gate.py
live proposed crystallizations: 621
  would AUTO-ACTIVATE (leave the queue): 599
  would STAY QUEUED (real review work):  22

resolved window platform (unanimous across all turns, else None):
        aitown: 599
          None: 22

why the survivors stayed queued:
         gated_kind:stance: 22
```

After the purge (and after the review fix to the smoke's failure condition — the
original `if total and not auto: fail` would have returned 1 on exactly this
state, which is what made it wrong):

```text
$ python3 scripts/smoke_aitown_crystallization_gate.py; echo $?
live proposed crystallizations: 23
  would AUTO-ACTIVATE (leave the queue): 0
  would STAY QUEUED (real review work):  23

resolved window platform (unanimous across all turns, else None):
          None: 23
0
```

23, not 22, because a real conversation landed during the session — the queue is
now doing what it should. `bulk_reject_aitown_proposals.py` re-run confirms
`proposed=23 external=0 keep=23 / nothing to do`.

Independently confirmed in SQL that all 22 survivors genuinely contain a
non-ai-town turn, and that zero are artifacts of a pruned `chat_history_log` row.

```text
$ python3 scripts/bulk_reject_aitown_proposals.py --apply
proposed=621 external=599 keep=22
snapshot rows=599 -> /tmp/aitown-crystallization-purge/snapshot.json
reject 599/599 (100.0%) errors=0
committed
after={'active': 589, 'proposed': 22, 'rejected': 609}
```

**No container was rebuilt.** Neither `orion-memory-consolidation` nor
`orion-sql-writer` bind-mounts `app/`, so both run pre-patch code right now. The
gate is `UNVERIFIED` on the live rail until the restart below — what *is* verified
is that the policy functions decide correctly on the real corpus.

## Review findings fixed

`/code-review high` against this branch returned 11 findings. All 11 were real;
10 are fixed in code, 1 is recorded as a documented open item.

- Finding (HIGH): `bulk_reject_aitown_proposals.py` ran 600+ statements in one
  `autocommit=False` transaction with a per-row `try/except`. psycopg2 puts the
  transaction into `INERROR` on the first failure, so every later `execute` raises
  `InFailedSqlTransaction` (caught and counted) and `commit()` is silently
  converted to ROLLBACK **without raising** — the run would log `committed`,
  write `verdict: APPLIED`, and have changed nothing.
  - Fix: `SAVEPOINT`/`ROLLBACK TO SAVEPOINT`/`RELEASE` per row, plus a post-commit
    read-back that compares the DB's actual `proposed` count against `len(keep)`
    and downgrades the verdict to `APPLY FAILED VERIFICATION` on mismatch.
  - Evidence: the real run had 0 errors so nothing was lost, but the script was
    wrong. Report now carries `post-apply read-back verified: True`.

- Finding (MEDIUM): the smoke's failure condition inverted once the gate worked —
  `if total and not auto: return 1` fails forever in steady state, and cannot
  distinguish "gate working" from "gate inert".
  - Fix: inverted the question — fail if any *queued* proposal resolves to an
    allowlisted platform (i.e. the gate failed to fire on a row it should have).
  - Evidence: live run now reports 23 proposed / 0 auto-activate / **exit 0**.
    The old condition would have returned 1 on this exact state.

- Finding (MEDIUM): the platform gate sat above the whole of `GATED_KINDS`, so an
  allowlisted platform could auto-activate `decision`/`contradiction`/`attractor`/
  `failure_mode` — latent today, but fictional NPC roleplay writing a
  `contradiction` straight to active would arrive silently with the next
  `_KIND_FOR_SHIFT` mapping.
  - Fix: `EXTERNAL_PLATFORM_BYPASSABLE_KINDS = {"stance"}` — the only gated kind
    this producer can emit.
  - Evidence: `test_other_gated_kinds_are_not_bypassed_by_the_platform_gate`.

- Finding (MEDIUM): the in-code comment presented the intimate/identity guards as
  live safety rails, but `build_crystallization_from_window` hardcodes
  `sensitivity="private"` and `scope=["memory_window:…"]`, so neither can fire for
  the only producer that sets `source_platform`. The two tests covering them pass
  only by hand-mutating the object.
  - Fix: no behavior change (the ordering is still correct defense-in-depth) —
    the comment and both test docstrings now say plainly that they are
    unreachable via this producer, and a new test pins that claim so it cannot
    quietly become false.
  - Evidence: `test_window_producer_cannot_actually_reach_those_guards`.

- Finding (MEDIUM): every checkbox tick fired an un-awaited full `loadInbox()`;
  three quick ticks launched three overlapping loads whose `innerHTML=""` and
  appends interleaved, producing duplicated/vanishing rows and stale checkbox state.
  - Fix: selection is local state — `refreshSelectionUi()` mutates the bulk bar
    and the existing boxes in place, no refetch.
  - Evidence: `test_ui_toggling_a_checkbox_does_not_refetch_the_queue`.

- Finding (MEDIUM): no client-side cap against the server's batch limit, and
  `apiFetch` discarded `err.body` — so "select all → Reject" on the 621-item
  backlog this feature was written for would fail with an opaque `HTTP 400`.
  - Fix: client chunks (200 reject / 25 approve) and error messages now include
    the server's `detail`.
  - Evidence: `test_ui_chunks_bulk_requests_under_the_server_caps`,
    `test_ui_surfaces_server_error_detail_not_just_the_status`.

- Finding (LOW): "select all" included retirement candidates (`status=active`),
  which the bulk endpoint always refuses — so every sweep reported `N failed`.
  - Fix: `isDecidable()`; undecidable rows get a spacer instead of a checkbox and
    are excluded from select-all.
  - Evidence: `test_ui_select_all_excludes_undecidable_rows`.

- Finding (LOW/MEDIUM): bulk approve ran the full single-item approve — including
  a chroma/card projection and a second write — up to 500 times in one request.
  - Fix: `BULK_APPROVE_MAX = 50` separate from `BULK_DECIDE_MAX = 500`.
  - Evidence: `test_approve_has_a_much_lower_cap_than_reject`.

- Finding (LOW): the new evidence-DELETE bound the raw path parameter to
  `$1::uuid`, skipping the `crys_<hex32>` → dashed-UUID normalization every other
  repository helper performs — a well-formed `crys_` id would clear the 404/409
  guards and then 503 on the cast.
  - Fix: extracted the (previously duplicated) normalization into
    `repository.normalize_crystallization_id()` and used it at all three sites.
  - Evidence: `test_drop_turn_normalizes_a_crys_prefixed_id`.

- Finding (LOW): `openDetail`'s rejections were unhandled from the row-click path,
  leaving the pane visible-but-empty with a dead id in its dataset.
  - Fix: `.catch()` → `closeDetail` + error status.
  - Evidence: `test_ui_handles_open_detail_rejection`.

- Finding (LOW, **not fixed — documented**): `close_current_window()` seeds the
  next window with the closing turn, so a direct turn from Juniper makes the
  following window permanently "mixed" and NPC dialogue right after she speaks
  still queues.
  - Why not fixed: it is the same global-window-cursor bug already listed as an
    open item; changing window lifecycle belongs in its own proposal.
  - Recorded in the analysis doc, because it fails safe and is otherwise
    indistinguishable from the gate working correctly.

Two findings from the review are **not** attributable to this branch and were
handled separately: the first `/code-review` invocation mis-scoped to the shared
checkout on `main` and reported a graphify destructive-update incident there
(`graph.json` 28,306 → 2,480 nodes, caused by `.git/hooks/post-checkout` firing on
this session's `git switch main`). Artifacts were restored from the hook's own
dated snapshot; `git status --short graphify-out/` is clean.

## Restart required

Order matters — `MemoryTurnPersistedV1` is `extra="forbid"`, so the consumer must
learn the field before the producer starts sending it.

```bash
# 0. MIGRATION FIRST, BEFORE ANY RESTART.
#    This branch adds `source_platform` to memory_consolidation_windows, and
#    _get_open_window() SELECTs on it. There is no auto-DDL anywhere in
#    services/orion-memory-consolidation/app/ -- this file is applied by hand.
#    Restarting without it means UndefinedColumnError on every
#    memory.turn.persisted.v1 and window formation stops completely.
#    (Already applied on athena, which is precisely why it is easy to forget.)
psql -h localhost -p 55432 -U postgres -d conjourney \
  -f services/orion-sql-db/manual_migration_memory_consolidation_v1.sql

# Optional but recommended: backfill the new column by the same unanimity rule
# the code uses, so historical rows are not all silently labelled "direct".
# The exact statement used on athena is in the PR body's Docker/smoke section.

# 1. CONSUMER FIRST
./scripts/safe_docker_build.sh orion-memory-consolidation up -d --build

# 2. then the producer
./scripts/safe_docker_build.sh orion-sql-writer up -d --build

# 3. Hub (new routes + cache-bust change; static/ and templates/ are bind-mounted
#    from the primary checkout, so the JS lands as soon as this branch is merged)
./scripts/safe_docker_build.sh orion-hub up -d --build

# verify the gate is live: a new ai-town window should no longer appear here
curl -fsS localhost:8080/api/memory/crystallizations/proposals | jq '.count'
```

## Risks / concerns

- Severity: **medium**
  Concern: deploy-order coupling. Starting sql-writer first makes every
  `memory.turn.persisted.v1` fail validation in the old consolidation consumer,
  stalling window formation until the consumer catches up.
  Mitigation: documented in the schema docstring, the commit message, and the
  restart block above. No silent-drop fallback was added, because `extra="forbid"`
  failing loudly is better than a field being silently ignored.

- Severity: **medium** — *superseded, now FIXED in `b6ab77068`.*
  Original concern: the global open-window cursor mixed platforms (26 windows
  historically), deferred as a separate proposal.
  What changed: it is fixed in this branch. `_get_open_window()` is partitioned
  on `source_platform`, which also removed two things not visible when the
  concern was written — `classify_turn()` was scoring novelty against an
  unrelated conversation, and `close_current_window()`'s closing-turn carry-over
  made the *next* window permanently mixed. This is a window-lifecycle change
  plus a schema column, an index and 7 new tests; review it as such rather than
  trusting this register's earlier "not fixed here".
  Remaining coarseness: partitioning is by platform, not `(platform, session_id)`
  — 302 ai-town sessions still share one rolling window. Harmless for the queue,
  but it blurs separate NPC conversations; recorded as a follow-up.

- Severity: **low**
  Concern: the backlog was *rejected* while future ai-town windows will be
  *auto-activated*. That asymmetry is deliberate (Juniper's explicit choice) but
  means the 599 purged windows are not recoverable as memory without the reversal
  SQL in the job report.
  Mitigation: nothing was deleted; `/tmp/aitown-crystallization-purge/snapshot.json`
  holds full pre-change row state and the report contains a one-statement undo.

- Severity: **low**
  Concern: `salience` is pinned at 1.0 for this entire path and is still rendered
  in the queue UI, where it reads as information.
  Mitigation: documented in the analysis doc; left in place rather than changing
  a scoring function in the same patch as a gate.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1678
