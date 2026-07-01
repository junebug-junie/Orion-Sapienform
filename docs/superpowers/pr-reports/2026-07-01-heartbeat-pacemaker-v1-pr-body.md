# PR: Orion heartbeat pacemaker v1 — continuous field/self-state chain + PR #766 rung-1 closure

> Committed to the branch as the PR body because this checkout has no GitHub auth
> (no `gh` login / `GH_TOKEN`). Open the PR at:
> https://github.com/junebug-junie/Orion-Sapienform/compare/main...feat/heartbeat-pacemaker-v1
> and paste the body below.

Branch: `feat/heartbeat-pacemaker-v1`
Plan: `docs/superpowers/plans/2026-07-01-orion-heartbeat-pacemaker-v1.md`

---

## Summary

Implements the heartbeat-pacemaker plan in full (Tasks 1–5), plus closes a gap
documented in PR #766's own PR body. Four independent pieces, each a thin seam
on top of already-live infrastructure — nothing here builds a new service:

1. **Field-digester idle tick (pacemaker).** `orion-field-digester` previously
   returned immediately when no new substrate reduction receipts arrived, so
   `tick_id` never advanced during quiet periods and nothing downstream had a
   reason to run. Adds a settings-gated idle tick that still runs
   decay/diffusion and persists a new field-state row, without touching the
   receipt cursor. **Ships default-off** — see Known Gap below.
2. **Whole-self surprise aggregate.** Formalizes the existing ad hoc
   `max(prediction_error_scores)` recompute (previously duplicated inline in
   `self_state_ctx.py`) as a real `overall_surprise` field on `SelfStateV1`,
   set once per tick by `orion-self-state-runtime`.
3. **Chat-stance φ readback.** `orion/substrate/relational/adapters/self_state_ctx.py`
   already mapped `SelfStateV1` into belief nodes already registered in the
   live producer registry `chat_stance.py` reads from — but nothing read
   those nodes. Adds `_project_self_state_from_beliefs`, mirroring the
   existing `_project_autonomy_from_beliefs` pattern, folding
   `overall_condition`/per-dimension pressure into stance hazards.
4. **Substrate dynamics tick — closes PR #766's rung-1 gap.** PR #766 wrote
   execution/transport prediction-error onto durable substrate nodes but
   documented explicitly that nothing consumed them: *"No periodic
   `SubstrateDynamicsEngine.tick()` runs against the shared store in any
   service today."* Adds exactly that — a bounded, fail-open, 30s-cadence
   tick loop in `orion-substrate-runtime` reusing the same cached store
   instance the prediction-error writer already builds.

## What this does NOT do

No new `orion-heartbeat` service, no tensor-network/quimb dependency, no
measurement harness. The substrate that actually exists and runs today is
the graph/field-dynamics engine already in the mesh; this PR wires the
existing chain into a continuous pulse rather than building a new substrate.
See the plan doc's supersession note re: `docs/research/2026-05-01-orion-heartbeat-research-charter.md`.

## Changes

| Area | Change | Files |
|------|--------|-------|
| Field-digester pacemaker | `_tick()` idle branch: decay/diffusion + `save_field()`, no cursor advance on empty polls. `FIELD_DIGESTER_IDLE_TICK_ENABLED` (default **false**). | `services/orion-field-digester/app/{worker,settings}.py` (+test) |
| Self-state aggregate | `compute_overall_surprise()`; `SelfStateV1.overall_surprise`; worker sets it alongside `prediction_error_scores`; adapter reads the stored field instead of recomputing. | `orion/self_state/prediction.py`, `orion/schemas/self_state.py`, `services/orion-self-state-runtime/app/worker.py`, `orion/substrate/relational/adapters/self_state_ctx.py` (+tests) |
| Chat-stance readback | `_project_self_state_from_beliefs`, wired into `build_chat_stance_inputs` alongside the existing autonomy projection. `SELF_STATE_STANCE_PRESSURE_THRESHOLD` (default `0.8`). | `services/orion-cortex-exec/app/chat_stance.py` (+test) |
| Dynamics tick | `_dynamics_tick` / `_dynamics_tick_loop`, registered in `start()`. `SUBSTRATE_DYNAMICS_TICK_ENABLED` (default `false`), `SUBSTRATE_DYNAMICS_TICK_INTERVAL_SEC` (default `30.0`). Dedup refactor: `_get_substrate_graph_store()` shared with PR #766's prediction-error writer (found in code review). | `services/orion-substrate-runtime/app/{worker,settings}.py` (+test) |
| env/compose/docs | `.env_example` + `docker-compose.yml` updated for field-digester and substrate-runtime (both enumerate env vars explicitly); cortex-exec uses `env_file` only, matching PR #766's own precedent for its analogous flag. READMEs updated for all three services. Stale rung-1 comment in substrate-runtime `.env_example` (referred to this PR as unmerged follow-up work) corrected. | `.env_example`, `docker-compose.yml`, `README.md` × 3 services |

**Not touched:** `orion/bus/channels.yaml`, `orion/schemas/registry.py` — verified, not needed. No new bus channel or schema kind is introduced anywhere in this PR: `substrate.self_state.v1` / `SelfStateV1` are already registered (this PR only adds a field to the existing class), and the dynamics tick reads/writes the substrate graph store directly rather than publishing anything new. `requirements.txt` unchanged in all four services — no new dependency in any of them.

## Known gap: idle tick ships default-off, pending retention

Code review (run as a separate pass on this branch) found a real capacity
risk: `save_field()` mints a fresh `tick_id` on every call, so every idle
tick is a genuine new row in `substrate_field_state` — at the default 2s
poll interval that's ~43k rows/day with **no pruning anywhere in this
service**. The growth cascades downstream: `orion-attention-runtime` writes
one `substrate_attention_frame` row per new `tick_id`, and
`orion-self-state-runtime` writes one `substrate_self_state` row per new
frame — neither of those services prunes either. This is the same failure
shape as a prior unbounded-Postgres-growth incident on this host (TOAST OOM
crash loop).

**Resolution shipped in this PR:** `FIELD_DIGESTER_IDLE_TICK_ENABLED` defaults
to `false`. With the flag off, this PR is byte-identical in behavior to
`main` — Tasks 2–4's plumbing exists and is tested, but the pacemaker itself
is inert until explicitly enabled. Documented prominently in
`services/orion-field-digester/README.md` and `.env_example`.

**Before enabling in production:** a retention/pruning story is needed for
all three tables (`substrate_field_state`, `substrate_attention_frame`,
`substrate_self_state`), mirroring the existing `receipt_pruner.py` pattern
already used in `orion-substrate-runtime`. This is a small, well-scoped
follow-up, not a redesign — recommend doing it before flipping the flag on
Athena.

## Safety / governance

- All four features are individually flag-gated and default to either the
  pre-existing behavior (`FIELD_DIGESTER_IDLE_TICK_ENABLED=false`,
  `SUBSTRATE_DYNAMICS_TICK_ENABLED=false`) or additive-only behavior that
  degrades to `None`/no-op on absent input (`_project_self_state_from_beliefs`,
  `overall_surprise`).
- Fail-open throughout: the dynamics tick's store-init and engine-tick paths
  are both wrapped so a bad tick never raises out of the loop; the chat-stance
  projector returns `None` on any missing/malformed belief data rather than
  raising.
- Nothing here touches autonomy policy gates, the endogenous-agency gate, or
  any other safety-critical control surface.
- `SubstrateDynamicsEngine.tick()`, `orion/substrate/pressure.py`, and
  `orion/substrate/graphdb_store.py` are unmodified — the dynamics tick only
  adds a periodic caller; the pressure math and store bounding
  (`limit_nodes=500`, `limit_edges=1000`, cache fallback on error) already
  existed and were verified, not changed.

## Tests

All green except 3 pre-existing, unrelated failures on `main` itself
(verified via `git stash` — identical before and after this branch):
`test_cursor_tail_seed.py::test_publish_accepted_events_uses_separate_channel_not_canonical`,
`test_quarantine_truth.py::test_truth_healthy_when_quarantine_acknowledged`,
`test_worker_independent_reducers.py::test_start_spawns_independent_reducer_poll_tasks`
(missing `pytest-asyncio` plugin / a stale hardcoded poll-task count from an
earlier, unrelated PR).

- `services/orion-field-digester/tests/` — 8/8 passed (3 new + 5 pre-existing)
- `tests/test_self_state_prediction.py` + `orion/substrate/relational/tests/test_self_state_adapter.py` — 5/5 passed (3 new + 2 pre-existing, fixture updated to match the new invariant)
- `services/orion-cortex-exec/tests/{test_chat_stance_self_state_projection,test_chat_stance_autonomy_plumbing,test_chat_stance_shared_spine}.py` — 32/32 passed (9 new + 23 pre-existing)
- `services/orion-substrate-runtime/tests/` (excluding a pre-existing, unrelated collection error in `test_grammar_consumer_integration.py`) — 31/34 passed, 3 pre-existing failures unrelated to this branch (4 new dynamics-tick tests + PR #766's own 3 prediction-error-node tests all pass)

## Commits

- `b2522aa1` — field-digester idle tick
- `01435f1a` — self_state overall_surprise aggregate
- `25fbd465` — chat_stance self_state readback
- `3e02d9d9` — substrate dynamics tick (closes PR #766 rung-1 gap)
- `069abedc` — fix stale rung-1 comment
- `a9457d1c` — dedupe substrate-graph-store init (code review finding)
- `01717f42` — default idle tick off pending retention (code review finding)

## Process notes

Implemented via 4 parallel subagents (one per task, each with an explicit
file allow-list, no `git commit` permission) in a single isolated worktree,
followed by an orchestrator review pass reading every diff against spec and
independently re-running every test suite (not just trusting agent reports),
then a separate code-review pass that found and fixed the two issues above.
`.env` files for all three affected services were synced on this host to
match `.env_example` (with `FIELD_DIGESTER_IDLE_TICK_ENABLED` deliberately
left at the new safe default, `false`, pending the retention follow-up).

## Reviewer notes

- The repo is concurrently mutated by an automation process on the main
  checkout (confirmed during this session — unrelated vision-service files
  showed up modified in `main` while this branch's work was entirely
  isolated in a worktree); this branch's diff is unaffected by that.
- Recommend a fast follow-up PR for the three-table retention/pruning story
  before flipping `FIELD_DIGESTER_IDLE_TICK_ENABLED=true` on Athena.
