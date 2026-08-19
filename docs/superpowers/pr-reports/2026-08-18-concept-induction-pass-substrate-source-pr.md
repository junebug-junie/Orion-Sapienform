## Summary

- `concept_induction_pass` (the Hub chat workflow behind the Concept Induction Details Modal) was wired to two dead sources: `graph` pointed at Apache Fuseki (decommissioned 2026-07-23, hostname doesn't resolve), whose `fail_open_local` fallback landed on the local spaCy-extraction JSON store (writer disabled since 2026-07-11). A real replacement — the live FalkorDB substrate concept graph, already read by `chat_stance`'s `concept_induced` tier and rendered by Hub's Concept Atlas — existed but was never wired into this consumer.
- Adds a `substrate` `ConceptProfileRepository` backend (`orion/spark/concept_induction/substrate_repository.py`) that projects the live substrate concept region into the same `ConceptProfile` shape the other backends produce.
- Repoints `CONCEPT_PROFILE_BACKEND_CONCEPT_INDUCTION_PASS=substrate` and adds the FalkorDB connection env `orion-cortex-orch` needed but didn't have.
- Caught live during redeploy: `orion-cortex-orch`'s unpinned `redis>=5.0.4` had drifted to redis-py 8.x, which dropped `redis.commands.graph` entirely — pinned to `redis[hiredis]==5.2.1` (matches Hub's pin on the same graph).
- Code review (high effort, verified) caught a real masking bug: the substrate store builder never raises on misconfiguration, silently falling back to an empty in-memory store — fixed by detecting that fallback and reporting `unavailable` instead of `empty`, so a dropped env var can't silently masquerade as "genuinely zero concepts." Also extracted a shared `select_concept_nodes_by_anchor_scope()` helper to kill duplicated filtering logic between this and `concept_induction_ctx.py`.

## Outcome moved

`concept_induction_pass` now returns real, live concepts for orion/juniper/relationship instead of empty/dead-source output — live-verified inside the rebuilt `orion-athena-cortex-orch` container via the actual production settings loader (not mocked).

## Current architecture

Two independent concept-induction pipelines existed:
1. **Old (dead)**: `orion/spark/concept_induction`'s spaCy noun-chunk extraction → local JSON store or Fuseki SPARQL, read by `concept_induction_pass`.
2. **Live**: golden seed concepts + `orion-topic-foundry`-derived concepts → FalkorDB substrate concept region, read by `chat_stance.py`'s `concept_induced` tier and Hub's Concept Atlas — but not by `concept_induction_pass`.

## Architecture touched

- `orion/spark/concept_induction/` — new `substrate_repository.py` backend; `profile_repository.py`/`settings.py` accept `"substrate"` as a repository backend kind.
- `services/orion-cortex-orch/` — `workflow_runtime.py`/`concept_profile_config.py` accept `"substrate"`; env/compose wired to reach FalkorDB; `redis` pin fixed.
- `orion/substrate/` — new shared `select_concept_nodes_by_anchor_scope()` helper in `store.py`, exported from `orion/substrate/__init__.py`; `concept_induction_ctx.py` refactored to use it (behavior unchanged, verified by its existing 100-test suite).

## Files changed

- `orion/spark/concept_induction/substrate_repository.py` (new): the `substrate` repository backend.
- `orion/spark/concept_induction/profile_repository.py`: wires the new backend into `build_concept_profile_repository()`; `RepositoryBackendKind` gains `"substrate"`.
- `orion/spark/concept_induction/settings.py`: validators accept `"substrate"`.
- `orion/spark/concept_induction/tests/test_substrate_repository.py` (new): 9 unit tests.
- `orion/substrate/store.py`: new `select_concept_nodes_by_anchor_scope()` shared helper.
- `orion/substrate/__init__.py`: exports the new helper.
- `orion/substrate/relational/adapters/concept_induction_ctx.py`: refactored to use the shared helper (no behavior change).
- `services/orion-cortex-orch/app/workflow_runtime.py`: `ConceptProfileBackendKind` gains `"substrate"`; backend resolution and graph-unavailable-cutover logic extended to it.
- `services/orion-cortex-orch/app/concept_profile_config.py`: validators accept `"substrate"`.
- `services/orion-cortex-orch/.env_example`: `CONCEPT_PROFILE_BACKEND_CONCEPT_INDUCTION_PASS=substrate`; adds `SUBSTRATE_STORE_BACKEND`/`FALKORDB_URI`/`FALKORDB_SUBSTRATE_GRAPH`; documents why `local`/`graph` are dead.
- `services/orion-cortex-orch/docker-compose.yml`: passes through the 3 new env vars.
- `services/orion-cortex-orch/requirements.txt`: pins `redis[hiredis]==5.2.1`.
- `services/orion-cortex-orch/tests/test_concept_profile_config_adapter.py`, `tests/test_workflow_lane.py`: new backend-validation and end-to-end tests.

## Schema / bus / API changes

- Added: `"substrate"` as a valid `ConceptProfileRepositoryBackendKind` value (internal Python `Literal`, not a bus/schema contract).
- Removed: none.
- Renamed: none.
- Behavior changed: `concept_induction_pass` now reads real substrate concepts by default instead of empty/dead-source data.
- Compatibility notes: `local`/`graph`/`shadow` backends are untouched and still work as before for any other caller (`chat_stance.py`'s own direct use of `build_concept_profile_repository()` is unaffected — it doesn't pass a `backend_override` and its global `CONCEPT_PROFILE_REPOSITORY_BACKEND` stays `local`).

## Env/config changes

- Added keys (`services/orion-cortex-orch/.env_example`): `SUBSTRATE_STORE_BACKEND`, `FALKORDB_URI`, `FALKORDB_SUBSTRATE_GRAPH`.
- Removed keys: none.
- Renamed keys: none.
- Changed default: `CONCEPT_PROFILE_BACKEND_CONCEPT_INDUCTION_PASS` default value `graph` → `substrate`.
- `.env_example` updated: yes.
- local `.env` synced: yes, by hand (worktree-added `.env_example` keys are invisible to `scripts/sync_local_env_from_example.py`, which resolves `.env_example` from the primary checkout — see prior incident). Verified live with `scripts/safe_docker_build.sh orion-cortex-orch config`.
- skipped keys requiring operator action: none.

## Tests run

```
.venv/bin/python3 -m pytest orion/spark/concept_induction/tests -q
  → 85 passed, 1 pre-existing unrelated failure (test_falkor_materialization.py,
    confirmed present on main too)

.venv/bin/python3 -m pytest orion/substrate/relational/tests -q
  → 100 passed

.venv/bin/python3 -m pytest orion/substrate/tests -q
  → 542 passed

services/orion-cortex-orch: .venv/bin/python3 -m pytest tests/test_concept_profile_config_adapter.py tests/test_workflow_lane.py -q
  → 69 passed
```

(Note: running cortex-orch's *entire* `tests/` directory in one process shows pre-existing cross-file test-order pollution affecting ~30 unrelated tests, e.g. `test_chat_history_compactor_pass_*` — confirmed present on `main` at the same order of magnitude before this branch existed; not introduced by this patch, out of scope here.)

## Evals run

No eval harness exists for `orion-cortex-orch` or the `orion/spark/concept_induction` package.

## Docker/build/smoke checks

```
scripts/safe_docker_build.sh orion-cortex-orch config   → resolved env confirmed correct
scripts/safe_docker_build.sh orion-cortex-orch build    → clean build (redis[hiredis]==5.2.1 installed)
scripts/safe_docker_build.sh orion-cortex-orch up -d    → orion-athena-cortex-orch started clean, no errors in logs
curl http://localhost:8072/health                       → {"ok":true,...}
```

Live verification inside the running container, via the actual production settings loader (not mocked):

```
resolved backend: substrate
repo status: source_available=True
orion        available  concepts= ['Orion', 'substrate:node:substrate.harness_closure', ...]
juniper      available  concepts= ['Juniper']
relationship available  concepts= ['Orion-Juniper relationship']
```

Negative-path check (post-review-fix) with `SUBSTRATE_STORE_BACKEND` unset:

```
orion unavailable substrate_store_unavailable   # was previously "empty" before the fix
```

## Review findings fixed

- Finding: `SubstrateConceptProfileRepository` couldn't distinguish a misconfigured substrate store (silently falls back to an empty in-memory store) from a genuinely-empty live query, so it reported `availability="empty"` instead of `"unavailable"` and silently skipped the fail_open_local/fail_closed cutover.
  - Fix: `_get_store()` now detects an `InMemorySubstrateGraphStore` result and treats it as unavailable.
  - Evidence: new tests `test_lazy_get_store_treats_in_memory_fallback_as_unavailable_not_empty` / `test_status_reports_unavailable_when_store_resolves_to_in_memory`; live-verified both directions in the rebuilt container (see Docker checks above).
- Finding: unused module-level `_SUBJECTS` tuple (dead code).
  - Fix: removed.
- Finding: concept-node `node_kind`/`anchor_scope` filtering duplicated near-identically between `substrate_repository.py` and `concept_induction_ctx.py`.
  - Fix: extracted shared `orion.substrate.select_concept_nodes_by_anchor_scope()`, used by both; `concept_induction_ctx.py`'s existing 100-test suite confirms no behavior change.

## Restart required

```bash
scripts/safe_docker_build.sh orion-cortex-orch build
scripts/safe_docker_build.sh orion-cortex-orch up -d
```

Already done live on this session's Athena host as part of verification — `orion-athena-cortex-orch` is running the new image now.

## Risks / concerns

- Severity: low
- Concern: cortex-orch's full multi-file `tests/` directory run shows pre-existing test-order pollution (~30 unrelated tests fail together, including files this PR never touches). Confirmed present on `main` at similar magnitude before this branch existed.
- Mitigation: not fixed here (out of scope, pre-existing, unrelated to this patch's correctness). Verified this patch's actual test files pass cleanly both standalone and as a pair, and that the same pollution symptom exists on `main`.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1714
