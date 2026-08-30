## Summary

- Orion's routing threshold is a real, live knob that **nothing could actually turn.** orion-hub's mutation applier writes `routing.chat_reflective_lane_threshold`; `orion-cortex-orch/app/decision_router.py:356` reads it on every routing decision. They were on two different stores.
- Verified live before the fix: `hub source_kind=postgres threshold=0.5` / `cortex-orch source_kind=memory threshold=0.75`. Same code, same key, no symptom — `RuntimeControlSurfaceStore` fails **open** to a per-process in-memory dict that serves the compiled-in default and resets on restart.
- **Two independent causes, both fixed.** cortex-orch had no control-plane DB key; and once configured it *still* fell back with `last_error="No module named 'sqlalchemy'"`.
- Fixes the mechanism that let a pytest fixture write Orion's live routing threshold.
- Adds a gate for the class, because "configured correctly and still broken" is not something to rediscover.

This does **not** make Orion self-modify anything. It makes the knob reachable, which is the precondition.

## Outcome moved

A write on the applier's side is now read by the live routing decision. Proven end to end:

```text
hub writes 0.66  ->  cortex-orch reads 0.66  ->  restored to 0.75
```

That is the roadmap's Phase 1 falsifiability test ("does moving a knob observably change what Orion does?") passing for the first time.

## Current architecture

`orion/substrate/mutation_control_surface.py` is a shared mutable store resolved per process from `SUBSTRATE_CONTROL_PLANE_POSTGRES_URL` → `SUBSTRATE_POLICY_POSTGRES_URL` → `DATABASE_URL`, falling back to sqlite, then to memory. The fallback is silent by design, which is what hid this.

## Architecture touched

Config and dependency wiring plus one precedence rule. No cognition logic, no schema, no bus contract.

## Files changed

- `services/orion-cortex-orch/docker-compose.yml`: adds `SUBSTRATE_CONTROL_PLANE_POSTGRES_URL` (existing purpose-named key — no new name), on the Docker service hostname since cortex-orch is bridge-networked, not host like hub.
- `services/orion-cortex-orch/.env_example`: same key, documented.
- `services/orion-cortex-orch/requirements.txt`: `SQLAlchemy==2.0.43`, pinned to match hub.
- `orion/substrate/mutation_control_surface.py`: an explicitly-passed `sql_db_path` is no longer overridden by an ambient env Postgres URL.
- `scripts/check_control_surface_store_parity.py`: new gate.
- `orion/substrate/tests/test_control_surface_store_isolation.py`: new tests.

## Schema / bus / API changes

None.

## Env/config changes

- **Added key:** `SUBSTRATE_CONTROL_PLANE_POSTGRES_URL` (orion-cortex-orch).
- `.env_example` updated: yes.
- Local `.env` synced: **by hand.** `scripts/sync_local_env_from_example.py` reads `.env_example` from the primary checkout, so a key added in a worktree is invisible to it and it reports "no changes" — a false green. The key was added directly to `services/orion-cortex-orch/.env` and verified present in the running container.
- Skipped keys requiring operator action: none.

## Tests run

```text
orion/substrate/tests/                      628 passed
  incl. test_control_surface_store_isolation.py   3 passed
services/orion-cortex-orch/tests/           33 failed, 172 passed
```

Those 33 are **pre-existing**: identical count on `main` from the primary checkout (`33 failed, 172 passed`), concentrated in `test_workflow_lane.py`. Not touched here.

## Evals run

No eval harness for this seam. The live propagation smoke below is the behavioural check.

## Docker/build/smoke checks

```text
safe_docker_build.sh orion-cortex-orch build  -> Image Built
safe_docker_build.sh orion-cortex-orch up -d  -> Container Started

hub          source=postgres threshold=0.75
cortex-orch  source=postgres threshold=0.75    (was source=memory)

hub writes 0.66 -> cortex-orch reads 0.66 -> restore 0.75 -> cortex-orch reads 0.75
```

Cutover was a deliberate no-op: the stored value was reset from the test residue `0.5` to `0.75` — the value cortex-orch was already serving — *before* wiring it up, so enabling the shared store changed no behaviour.

## Review findings fixed

Review ran in a subagent; findings and fixes are recorded in the follow-up commit on this branch.

## Restart required

Already deployed. To reproduce:

```bash
bash scripts/safe_docker_build.sh orion-cortex-orch build
bash scripts/safe_docker_build.sh orion-cortex-orch up -d
```

## Risks / concerns

- **Severity: medium.** The compose default hardcodes `orion-athena-sql-db`. Correct on athena; **wrong on another host** (circe) if cortex-orch is ever deployed there. It is a `${VAR:-default}`, so an env override fixes it, but the default is host-specific.
- **Severity: low.** cortex-orch now opens a Postgres connection it did not before. It is lazy (only on control-surface access) and falls back safely, but it is a new dependency edge in the routing path.
- **Severity: informational.** I confirmed the router *reads* the knob on every decision and that changes propagate, but I could **not** confirm from logs how often `routing_threshold_gate` actually flips a decision at 0.75 — the confidence values are not logged. If it rarely fires, self-modifying this particular knob would be low-impact. Worth measuring before making it the first thing Orion tunes.
- **Severity: informational.** Test residue reached production once (4,925 updates by a fixture actor). Fixed at the source, but other stores in this repo may share the fail-open + ambient-env pattern.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2000
