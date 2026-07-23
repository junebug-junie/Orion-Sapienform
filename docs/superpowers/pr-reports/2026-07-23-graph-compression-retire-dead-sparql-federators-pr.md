# PR report: retire graph-compression's two confirmed-dead SPARQL federators

## Summary

- Part of the same fresh Fuseki-dependency audit that led to the `orion-recall` and `orion-vision-scribe` fixes earlier today: `orion-graph-compression` runs three SPARQL "federator" classes feeding Leiden clustering. Two of the three (`SubstrateFederator`, `SelfStudyFederator`) were live-verified dead and retired outright; the third (`EpisodicFederator`) is deliberately untouched -- it's still genuinely load-bearing for collapse/social content with no Falkor equivalent.
- `SubstrateFederator`: direct SPARQL query against the live Fuseki container found `orion:substrate` frozen at 126 triples, with zero active writers anywhere in the repo (substrate-runtime moved Falkor-primary in PR #1153). `FalkorSubstrateFederator` is now the sole source for the `substrate` scope.
- `SelfStudyFederator`: direct SPARQL `COUNT` queries against all three of its source graphs (`orion:self`, `orion:self:induced`, `orion:self:reflective`) returned exactly 0 triples each, confirmed via raw JSON response. Zero producers found anywhere. Whole scope retired -- no Falkor migration needed since there was nothing to migrate.
- Review caught 4 additional stale references the initial diff missed (a shared pydantic schema `Literal`, an unused-but-misleading policy YAML block, a second un-swept scope-list fallback in `orion-recall`, a stale docker-compose inline default) -- all fixed in this same PR.

## Outcome moved

`orion-graph-compression` now depends on Fuseki for exactly one scope (`episodic`), down from three. Combined with the `orion-recall` and `orion-vision-scribe` fixes from earlier today, this narrows what's still genuinely blocking a full Fuseki decommission to: `EpisodicFederator`'s remaining SPARQL dependency (collapse/social, no Falkor writer yet) and the unmigrated `orion:chat:social:stored` data -- both already-known, separately-tracked gaps, not new ones.

## Current architecture

Before this patch: `worker.py::_process_scope` always ran the SPARQL federator for every scope, additionally (unioned, never swapped) running the Falkor equivalent if its flag was on -- a deliberate "verify live, never regress" posture while the Falkor side was unproven. That verification is now done for `substrate`; `self_study` never had a Falkor equivalent because it never had any real data.

## Architecture touched

- `services/orion-graph-compression/app/{worker.py,stale_listener.py}`, `app/federators/{self_study.py,substrate.py}` (deleted), `.env_example`, `docker-compose.yml`, `README.md`, `config/compression_policy.v1.yaml`, `tests/test_worker_degraded.py`
- `orion/schemas/graph_compression.py` (shared schema contract)
- `services/orion-recall/app/{worker.py,storage/graph_compression_adapter.py}`, `orion/recall/profiles/graph.compressions.{global,local,v1}.yaml` (a different service, touched because it hardcodes graph-compression's scope vocabulary)

## Files changed

- `services/orion-graph-compression/app/worker.py`: removed `SubstrateFederator`/`SelfStudyFederator` imports. `substrate` scope now: `triples = FalkorSubstrateFederator().fetch() if s.graph_compression_substrate_falkor_enabled else []` (no SPARQL call at all). `self_study` scope branch deleted entirely (falls to the existing `else: return`). Default `scopes_to_process` dropped `"self_study"`.
- `services/orion-graph-compression/app/federators/self_study.py`, `substrate.py`: deleted (zero remaining callers).
- `services/orion-graph-compression/app/stale_listener.py`: removed the three `orion:self*` graph→scope mappings and `"self_study"` from the "mark all scopes" fallback list. `orion:substrate`'s mapping left in place (harmless, scope still exists).
- `services/orion-graph-compression/.env_example`: `GRAPH_COMPRESSION_SUBSTRATE_FALKOR_ENABLED` checked-in default flipped `false` -> `true` (now the sole source, no SPARQL fallback left). `GRAPH_COMPRESSION_EPISODIC_FALKOR_ENABLED` untouched (stays `false`, genuinely still additive).
- `services/orion-graph-compression/docker-compose.yml`: inline fallback default for the same key updated to match (`:-true`), review finding -- was inconsistent with `.env_example` and a silent landmine for a deploy that skips env sync.
- `services/orion-graph-compression/README.md`: architecture diagram, Compression Scopes table (dropped `self_study` row, corrected `substrate`'s documented region kind from `contradiction` to `hotspot` -- a pre-existing doc/code mismatch caught while editing this table), rewrote the FalkorDB federators section.
- `services/orion-graph-compression/config/compression_policy.v1.yaml`: removed the unused-but-misleading `self_study` scope block (review finding -- confirmed via tracing `_load_policy()` that only the `clustering` key is ever read, so this was dead config actively claiming a retired scope was "enabled").
- `services/orion-graph-compression/tests/test_worker_degraded.py`: removed `SubstrateFederator`/`SelfStudyFederator` patches from every test; rewrote the hotspot-labeling test to exercise `FalkorSubstrateFederator` with the flag on; added `test_substrate_scope_produces_no_clusters_when_falkor_flag_off` (regression: flag off -> zero artifacts, no silent SPARQL fallback).
- `orion/schemas/graph_compression.py`: review finding -- `CompressionRegionV1.scope`/`.kind` `Literal`s still declared the retired `"self_study"`/`"self_study_cluster"` values; trimmed to match reality.
- `services/orion-recall/app/worker.py`, `app/storage/graph_compression_adapter.py`: two separate hardcoded default `compression_scopes` fallbacks (one fixed in the initial diff, a second identical copy caught by review) both dropped `"self_study"`.
- `orion/recall/profiles/graph.compressions.{global,local,v1}.yaml`: removed the explicit `- self_study` line from each profile.

## Schema / bus / API changes

- `CompressionRegionV1.scope`/`.kind` `Literal` types narrowed (removed `"self_study"`/`"self_study_cluster"`) -- a real, live pydantic validation contract (confirmed `region_builder.py::build_region` constructs this model on every write), not just a comment.
- No bus channel changes.

## Env/config changes

- Changed default: `GRAPH_COMPRESSION_SUBSTRATE_FALKOR_ENABLED` checked-in `.env_example`/docker-compose default `false` -> `true`. Pydantic code-level default stays `False` (safe fallback convention, unchanged).
- Local `.env` synced: primary-checkout's live `.env` already had this key `true` (confirmed before this patch), so no local edit needed.
- Skipped keys: none.

## Tests run

```text
ORION_BUS_URL=redis://127.0.0.1:6379/0 PYTHONPATH=.:services/orion-graph-compression venv/bin/python3 -m pytest services/orion-graph-compression/tests -q
-> 45 passed

ORION_BUS_URL=redis://127.0.0.1:6379/0 PYTHONPATH=.:services/orion-recall venv/bin/python3 -m pytest services/orion-recall/tests -q
-> 229 passed, 3 pre-existing unrelated failures (test_process_recall_active_turn_exclusion.py,
   test_recall_policy_harness.py, test_recall_vector_amputation.py -- confirmed identical on
   unmodified origin/main, an unrelated mock-signature issue in active-turn-exclusion tests)

git diff --check -> clean
```

## Evals run

No eval harness exists for this service; focused deterministic tests (including two new regression tests) cover the changed behavior.

## Docker/build/smoke checks

No container rebuild/restart performed as part of this PR. Live evidence backing the removal decision was gathered by directly SPARQL-querying the running `orion-athena-fuseki` container (see Summary).

## Review findings fixed

- Finding (should-fix, moderate): `orion/schemas/graph_compression.py`'s `CompressionRegionV1` still declared the retired scope/kind values as valid in a live pydantic `Literal` -- confirmed constructed on every real write, not dead code.
  - Fix: trimmed both `Literal`s.
  - Evidence: commit, `test_compression_schema.py` still passes.
- Finding (should-fix, low-moderate): `compression_policy.v1.yaml` still had a full `self_study` scope block claiming `enabled: true`.
  - Fix: removed; confirmed via tracing `_load_policy()` that only `clustering` is ever read, so this was pure doc-drift, not a live behavior change.
- Finding (should-fix, low): a second, identical copy of the `self_study`-including default-scopes fallback existed in `orion-recall/app/storage/graph_compression_adapter.py:107`, one file away from the one already fixed in `worker.py`.
  - Fix: same edit applied there.
- Finding (hygiene, low): `docker-compose.yml`'s inline fallback default (`:-false`) was inconsistent with the new `.env_example` default and the "sole source, no fallback" story.
  - Fix: updated to `:-true`.

## Restart required

No restart required to merge (default was already live `true` in the primary checkout). If `orion-athena-graph-compression` is redeployed from this branch/main, no behavior change is expected since the live env already matches.

```bash
docker compose --env-file .env --env-file services/orion-graph-compression/.env \
  -f services/orion-graph-compression/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: Low.
- Concern: none blocking. `EpisodicFederator`'s SPARQL dependency (collapse/social, no Falkor writer yet) remains open and unaffected by this PR -- tracked separately, not a new gap introduced here.

## PR link

(added after opening)
