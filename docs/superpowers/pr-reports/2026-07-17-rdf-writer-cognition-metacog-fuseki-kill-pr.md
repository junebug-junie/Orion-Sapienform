# PR report: stop writing cognition.trace / metacog.trace to Fuseki

## Summary

- Live investigation this session found `orion-rdf-writer` still committing ~750 real writes/6h to Fuseki for exactly two channels: `orion:metacog:trace` and `orion:cognition:trace`.
- Both are already durably persisted in Postgres via `orion-sql-writer`: `orion_metacognitive_trace` (already triple-written to sql-writer, vector-writer, and rdf-writer — confirmed via live logs) and `cognition_traces` (confirmed via direct Postgres query: 61,079 live rows, most recent ~1 minute old at check time). `channels.yaml` simply never declared `orion-sql-writer` as a consumer of `orion:cognition:trace` — a stale registry entry, not a missing migration.
- Consumer audit (done before any producer change, per the split-design doctrine's "producer + consumer + test in the same changeset" rule): confirmed the only other Fuseki reader of these two graphs, `orion-graph-compression`'s episodic federator, is already fail-open by construction and degrades correctly when these 2 of 9 federated graphs go quiet. No other real consumer exists — `orion-recall`'s graph-URI dict entries for both are dead, unreferenced constants.
- Removed both channels from `orion-rdf-writer`'s subscribe list, removed their dispatch branches and handler functions, fixed `channels.yaml`'s registry (added `orion-sql-writer` as a real consumer of `orion:cognition:trace`; dropped `orion-rdf-writer` from both).
- Code review caught and fixed a real env-parity gap: `CHANNEL_COGNITION_TRACE_PUB` was left dangling in `.env_example`/`docker-compose.yml` after the `Settings` field was removed.

## Outcome moved

`orion-rdf-writer` no longer writes redundant copies of cognition/metacognitive trace data to Fuseki — both already have a real, live, non-degrading durable home in Postgres. This is the second concrete "wrong-tool precedent" cleanup in the same arc as the drive-audit RDF kill (2026-07-15) and the Falkor substrate-runtime cutover (PR #1145/#1153, same session).

## Current architecture

Before this patch, `orion-rdf-writer` subscribed to `orion:metacog:trace` and `orion:cognition:trace`, built RDF triples for each (`_handle_metacognitive_trace`, `_handle_cognition_trace` in `rdf_builder.py`), and committed them to Fuseki graphs `orion:metacog`/`orion:cognition`. `orion-sql-writer` was *also* independently subscribed to both channels already, with real SQLAlchemy models and live data — making the RDF copy pure redundancy with no consumer that needed it (verified via a consumer audit, not assumed).

## Architecture touched

- `services/orion-rdf-writer/app/settings.py`: subscribe list, removed field.
- `services/orion-rdf-writer/app/rdf_builder.py`: dispatch + handler removal.
- `services/orion-rdf-writer/.env_example` / `docker-compose.yml`: dangling key cleanup (review finding).
- `orion/bus/channels.yaml`: registry accuracy for both channels.

## Files changed

- `services/orion-rdf-writer/app/settings.py`: removed `"orion:metacog:trace"` and `CHANNEL_COGNITION_TRACE_PUB` from `get_all_subscribe_channels()`; removed the now-unused `CHANNEL_COGNITION_TRACE_PUB` field; added an explanatory comment mirroring the existing `CHANNEL_MEMORY_DRIVES_AUDIT` precedent.
- `services/orion-rdf-writer/app/rdf_builder.py`: removed the `cognition.trace`/`metacognitive.trace.v1` dispatch branches, the `_handle_cognition_trace`/`_handle_metacognitive_trace` functions, and their now-unused schema imports.
- `services/orion-rdf-writer/.env_example`, `services/orion-rdf-writer/docker-compose.yml`: removed dangling `CHANNEL_COGNITION_TRACE_PUB` (review finding).
- `orion/bus/channels.yaml`: `orion:metacog:trace` drops `orion-rdf-writer` as a consumer; `orion:cognition:trace` gains `orion-sql-writer` (registry accuracy — it was already a real consumer) and never listed `orion-rdf-writer`.
- `services/orion-rdf-writer/tests/test_autonomy_materialization.py`: two new channel-not-subscribed regression tests (mirroring the existing drive-audit precedent test in the same file) and two new quiet-no-op dispatch tests.
- `services/orion-rdf-writer/tests/test_service_rdf_store_integration.py`: removed both kinds from the parametrized "writes for kind" test (that test fully mocks the builder, so this is cosmetic — real per-kind behavior is covered unmocked in `test_autonomy_materialization.py` instead; mirrors that `memory.drives.audit.v1` was never in this list either).

## Schema / bus / API changes

- Added: `orion-sql-writer` declared as a consumer of `orion:cognition:trace` in `channels.yaml` (registry accuracy — it was already consuming it in code).
- Removed: `orion-rdf-writer` as a consumer of `orion:metacog:trace` and (never-declared, now explicitly absent) `orion:cognition:trace`. `CHANNEL_COGNITION_TRACE_PUB` field removed from `orion-rdf-writer`'s `Settings` (unrelated same-named fields in `orion-spark-introspector`, `orion-equilibrium-service`, `orion-cortex-exec` are untouched — each owns its own independent field).
- Renamed: none.
- Behavior changed: `orion-rdf-writer` no longer writes `orion:cognition`/`orion:metacog` Fuseki graphs. `orion-graph-compression`'s episodic region summaries stop incorporating cognition/metacog content going forward (a deliberate, understood narrowing — that federator already degrades gracefully for graphs with no new triples; not touched in this patch).
- Compatibility notes: no schema changes. Postgres tables (`cognition_traces`, `orion_metacognitive_trace`) already existed and are untouched.

## Env/config changes

- Added keys: none.
- Removed keys: `CHANNEL_COGNITION_TRACE_PUB` from `services/orion-rdf-writer/.env_example`, `docker-compose.yml`, and the local `.env` (synced by hand in the primary checkout, not left as an operator TODO).
- Renamed keys: none.
- `.env_example` updated: yes (key removed, comment added explaining why, mirroring the `CHANNEL_MEMORY_DRIVES_AUDIT` precedent already in the same file).
- local `.env` synced: yes, directly edited `services/orion-rdf-writer/.env` in the primary checkout to remove the same key. `scripts/sync_local_env_from_example.py` run afterward — clean, no divergence reported for `orion-rdf-writer`.
- skipped keys requiring operator action: none.

## Tests run

```text
ORION_BUS_URL=redis://127.0.0.1:6379/0 PYTHONPATH=.:services/orion-rdf-writer \
  /tmp/orion-test-venv/bin/python3 -m pytest services/orion-rdf-writer/tests -q
→ 41 passed

git diff --check → clean
python3 scripts/sync_local_env_from_example.py → clean, no orion-rdf-writer divergence
scripts/check_service_env_compose_parity.py orion-rdf-writer → same pre-existing
  17-key gap (unrelated keys: PROJECT, NET, Fuseki JVM tuning, etc.) before and
  after this patch; no new gap introduced
```

`scripts/check_schema_registry.py` / `scripts/check_bus_channels.py` referenced in the standard gate list do not exist in this repo. `orion/bus/channels.yaml` was validated by direct YAML parse (262 entries parse cleanly) and manual review of both edited entries.

## Evals run

```text
No eval harness exists for rdf-writer's channel dispatch; focused deterministic
tests cover subscription list and dispatch behavior.
```

## Docker/build/smoke checks

```text
No Docker rebuild/restart performed for this PR — code change only, no live
service currently depends on this behavior changing until orion-rdf-writer is
rebuilt and restarted from this branch.
```

## Review findings fixed

- Finding (should-fix): `CHANNEL_COGNITION_TRACE_PUB` left dangling in `.env_example`/`docker-compose.yml` after the `Settings` field was removed — dead, misleading operator-facing config (harmless at runtime due to `extra="ignore"`, but a real env-parity gap per CLAUDE.md §7).
  - Fix: removed from both files with an explanatory comment; verified via `check_service_env_compose_parity.py` that no new parity gap was introduced.
  - Evidence: commit `fad4801d`.

Reviewer also confirmed: no dangling references to removed symbols anywhere in the repo, `channels.yaml`'s consumer lists are internally consistent for both channels, the new tests exercise real (unmocked) dispatch behavior rather than being tautological, `orion-sql-writer`'s consumption of both channels is confirmed live in code (not stale/aspirational), and unsubscribing introduces no other side effect (nothing depends on `orion:rdf:confirm`/`orion:rdf:error` firing for these two kinds specifically).

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-rdf-writer/.env \
  -f services/orion-rdf-writer/docker-compose.yml up -d --build
```

Not required for correctness before merge (no other service depends on this behavior changing), but needed to actually stop the live redundant Fuseki writes this session observed.

## Risks / concerns

- Severity: Low
- Concern: `orion-graph-compression`'s episodic region summaries will stop incorporating cognition/metacog trace content going forward (2 of its 9 federated graphs go quiet). Already assessed as a deliberate, understood, non-breaking scope narrowing — that federator is fail-open by construction and was not modified in this patch.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1155
