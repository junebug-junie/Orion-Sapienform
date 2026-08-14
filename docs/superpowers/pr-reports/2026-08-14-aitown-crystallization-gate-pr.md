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

```text
services/orion-memory-consolidation/tests           94 passed   (71 before; +23 new)
services/orion-sql-writer/tests                    307 passed, 10 failed, 3 skipped
                                                   (295 passed / same 10 failed on main — pre-existing)
services/orion-hub/tests/test_crystallization_routes_contract.py
services/orion-hub/tests/test_memory_crystallization_ui.py
services/orion-hub/tests/test_crystallization_review_queue_ux.py
services/orion-hub/tests/test_workflow_schedule_runtime_paths.py
                                                    39 passed
services/orion-hub/tests (full)                   1253 passed, 33 failed
                                                   (failure set identical to main — pre-existing)
```

New coverage, 41 tests:
- unanimity in both directions (all-external, all-direct, mixed, two platforms,
  empty, missing key, empty-string platform)
- policy routing (external auto-activates, direct/mixed/unlisted still queue,
  empty allowlist disables, duplicate still wins)
- privacy precedence (intimate and identity-scoped windows queue regardless of source)
- `formation_executor.auto_activate` honors a caller-supplied allowlist — it
  re-resolves policy independently, so an unforwarded set would be silently
  dropped on the deciding path
- sql-writer platform parsing (dict, JSON string, 8 malformed shapes) and the
  real emit path off a real `chat.history` envelope
- `window_state.append_turn` persists the platform on both the new-window and
  append branches
- bulk decide: all-succeed, partial failure, dedup, bad action, empty ids, size
  cap, path not captured as an id
- evidence delete: correct row targeted, history written, 404 unknown, 409 last
  turn, 409 on active
- UI wiring smokes incl. a regression test that the cache-bust token globs

## Evals run

```text
No eval harness exists for orion-memory-consolidation or orion-sql-writer
(services/*/evals/ absent for both — confirmed, not assumed).
```

The closest thing to an eval here is the live replay smoke below, which measures
the gate's real decision on the real corpus rather than on fixtures.

## Docker/build/smoke checks

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

- Finding:
  - Fix:
  - Evidence:

## Restart required

Order matters — `MemoryTurnPersistedV1` is `extra="forbid"`, so the consumer must
learn the field before the producer starts sending it.

```bash
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

- Severity: **medium**
  Concern: the global open-window cursor still mixes platforms (26 windows
  historically). Not fixed here — it changes window lifecycle semantics and wants
  its own proposal.
  Mitigation: the gate requires unanimity, so mixed windows keep reaching the
  human queue rather than auto-activating with Juniper's words inside them.

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

<to be filled after push>
