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
orion/substrate/tests/                                631 passed
tests/scripts/                                        371 passed, 1 failed
services/orion-cortex-orch/tests/                     175 passed, 33 failed
  incl. test_control_surface_isolation_guard.py         3 passed
```

Pre-existing and confirmed identical on `main`: the 33 cortex-orch failures (concentrated in `test_workflow_lane.py`) and `tests/scripts/test_rebuild_affected_services.py::test_sample_pull_diff`. Not touched here.

**Scope note:** adding `orion/substrate/tests` to `testpaths` brings 628 previously-uncollected tests into the default run. All pass locally; flagged because it widens what CI executes.

## Evals run

No eval harness for this seam. The live propagation smoke below is the behavioural check.

## Docker/build/smoke checks

```text
safe_docker_build.sh orion-cortex-orch build  -> Image Built
safe_docker_build.sh orion-cortex-orch up -d  -> Container Started

hub             source=postgres threshold=0.75
cortex-orch     source=postgres threshold=0.75    (was source=memory)
field-digester  source=postgres threshold=0.75    (was source=memory)

hub writes 0.61 -> cortex-orch 0.61, field-digester 0.61 -> restored 0.75
```

Cutover was a deliberate no-op: the stored value was reset from the test residue `0.5` to `0.75` — the value cortex-orch was already serving — *before* wiring it up, so enabling the shared store changed no behaviour.

## Review findings fixed

Review found that the first commit made the blast radius **worse** before making it better, plus three holes in the gate meant to prevent exactly that.

- **Finding (MUST):** the patch *armed* an unisolated production writer. `services/orion-cortex-orch/tests/test_auto_router.py:79` sets the routing threshold to 0.95 through the module-global store, no isolation, no restore. `Dockerfile` COPYs `tests/` into the image, so `docker compose exec cortex-orch pytest tests/` writes it. Pre-patch this was harmless only because the container lacked sqlalchemy — the patch added sqlalchemy *and* a production URL, removing both accidental protections.
  - **Fix:** autouse fixture in the suite's `conftest.py` that strips the three resolver env keys and swaps in a tmp sqlite store. Autouse and unconditional, so the next writer cannot reintroduce it by forgetting to opt in.
  - **Evidence:** 3 guard tests; all go red when the fixture is switched to `autouse=False`.
- **Finding (MUST):** the gate could not see a third, misconfigured consumer — it discovered services by substring, missing `from orion.substrate import mutation_control_surface` and missing transitive reach entirely.
  - **Fix:** resolves the import graph with `ast`.
  - **Evidence:** immediately found `orion-field-digester` (reaches the surface via `worker` → `causal_geometry_producer` → `mutation_trials`), which was in the exact fail-open state the gate exists to catch. Now configured, deployed, verified `source=postgres`.
- **Finding (MUST):** the gate credited a key that is **empty**. It matched presence, never a non-empty value, and reported `orion-hub: SUBSTRATE_CONTROL_PLANE_POSTGRES_URL` for a key that is empty in hub's `.env_example` (hub works via `DATABASE_URL`). Root cause was subtler than it looks: `\s*` after the separator matches newlines, so `KEY=` followed by `OTHER=` captured the *next line* as this key's value.
  - **Fix:** horizontal-whitespace-only match, bare `${VAR}` passthroughs rejected, `.env_example` consulted for `env_file:` services.
  - **Evidence:** hub is now correctly credited to `DATABASE_URL`; 13 gate tests including one pinning the newline bug.
- **Finding (SHOULD):** I moved a per-call `create_engine()` onto the chat hot path — fresh pool, TCP connect and auth handshake on every routing decision, with **no connect timeout**, so an unreachable-but-not-refusing host would block routing on the OS TCP timeout.
  - **Fix:** engine built once and cached, `pool_pre_ping`, clamped `connect_timeout` (default 5s, `SUBSTRATE_CONTROL_SURFACE_CONNECT_TIMEOUT_SEC`).
- **Finding (SHOULD):** neither the gate nor the test ran anywhere. **Fix:** gate added to `orion-static-gates.yml` (now 10); `orion/substrate/tests` added to `pyproject` testpaths, which also collects 628 previously-uncollected passing tests.
- **Finding (NICE):** a provably dead clause (`and not self.postgres_url` — the `or` already short-circuits; a mutant deleting it survived every test) and two vacuous assertions, one of which passed under a full revert of the fix. **Fix:** clause removed, assertions made load-bearing.
- **Finding (NICE):** athena-pinned hostname default. **Fix:** `${PROJECT:-orion-athena}-sql-db`, verified via `docker compose config` that the nested interpolation resolves.

## Restart required

Already deployed. To reproduce:

```bash
bash scripts/safe_docker_build.sh orion-cortex-orch build
bash scripts/safe_docker_build.sh orion-cortex-orch up -d
```

## Risks / concerns

- **Severity: low** (was medium, fixed in review). The compose default now derives the hostname from `${PROJECT:-orion-athena}`, so a non-athena host resolves correctly. `.env_example` still ships the literal athena value with a comment explaining the override.
- **Severity: low.** cortex-orch now opens a Postgres connection it did not before. It is lazy (only on control-surface access) and falls back safely, but it is a new dependency edge in the routing path.
- **Severity: informational.** I confirmed the router *reads* the knob on every decision and that changes propagate, but I could **not** confirm from logs how often `routing_threshold_gate` actually flips a decision at 0.75 — the confidence values are not logged. If it rarely fires, self-modifying this particular knob would be low-impact. Worth measuring before making it the first thing Orion tunes.
- **Severity: informational.** Test residue reached production once (4,925 updates by a fixture actor). Fixed at the source, but other stores in this repo may share the fail-open + ambient-env pattern.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2000
