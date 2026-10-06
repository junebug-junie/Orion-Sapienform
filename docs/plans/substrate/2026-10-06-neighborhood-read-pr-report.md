## Summary

- Add a bounded bidirectional semantic-neighborhood API with independent internal/boundary/neighbor budgets.
- Read Falkor and SPARQL directly without full hydration or stale-cache fallback; route only to the primary store.
- Allocate boundary edges by focal node, direction, and predicate; return complete endpoints and honest truncation/failure receipts.
- Add query-planner support, backend regressions, a hub-heavy eval, a read-only replay command, and CI.
- Deliver the design's recommended first patch. Assertion extraction/review/projection, complete hydration repair, and runtime-consumer migration remain subsequent patches.

## Outcome moved

The live eight-node focal set has 4 internal and 3,355 incoming boundary edges. With budgets 12/16/16, this API returns all 4 internal edges, 16 boundary edges, and 16 outside endpoints instead of letting boundary edges crowd out the internal graph. Final read-only replay took 1,140.07 ms, reported truncation, and had no dangling endpoints or degradation.

## Current architecture

Falkor's existing focal query reads its hydrated cache. Existing bounded queries mix incident/internal edges under one budget; callers can filter internal edges only after truncation. Existing APIs and consumers are unchanged by this patch.

## Architecture touched

Shared substrate store protocol, in-memory/Falkor/SPARQL/routed implementations, query coordinator, diagnostic CLI, tests/eval, and CI. No new service or runtime loop. Direct reads support `hydrate=False` and never mutate/warm the durable cache.

## Files changed

- `orion/substrate/neighborhood.py`: bounded request/result and shared allocation/integrity algorithm.
- `orion/substrate/neighborhood_backends.py`: native Cypher and SPARQL adapters.
- `orion/substrate/{store,falkor_store,graphdb_store,routed_store,query_planning}.py`: public API and planner integration.
- `orion/substrate/tests/test_neighborhood*.py`: budgets, fairness, scopes/states, endpoint integrity, failures, caps, backend parity.
- `orion/substrate/evals/{neighborhood_fixture,run_neighborhood_eval}.py`: hub-heavy replay across four budget combinations.
- `scripts/replay_substrate_neighborhood.py`: explicit-request read-only live replay.
- `.github/workflows/substrate-neighborhood.yml`: focused regression and eval gates.
- `docs/plans/substrate/2026-10-06-neighborhood-read-contract.md`: contract, limits, and runtime evidence; original approved design is included from its supplied commit.

## Schema / bus / API changes

- Added: `NeighborhoodRequestV1`, `NeighborhoodResultV1`, `store.read_neighborhood(request)`, coordinator `neighborhood` step.
- Removed / renamed: none.
- Behavior changed: opt-in direct neighborhood reads with separate edge budgets and complete endpoints.
- Compatibility notes: no persisted event/schema change; no registry/channel update needed. Existing APIs retain behavior. Default eligible node states are provisional/canonical; historical proposed nodes require explicit opt-in. Node state is not edge-acceptance authority. Anchor scope is not an ACL.

## Env/config changes

- Added / removed / renamed keys: none.
- `.env_example` updated: no.
- Local `.env` sync: not needed; no env template changed.
- Skipped keys: none. No bus access or bus URL change.

## Tests run

```text
Clean Python 3.12 venv with the exact CI dependency list:
python -m pytest -q orion/substrate/tests/test_neighborhood.py orion/substrate/tests/test_neighborhood_backends.py orion/substrate/tests/test_falkor_store.py orion/substrate/tests/test_graphdb_store.py orion/substrate/tests/test_routed_store.py orion/substrate/tests/test_phase16_query_planning.py
114 passed (4.48s). RDFLib emits deprecation warnings.
git diff --check: passed.
```

## Evals run

```text
python -m orion.substrate.evals.run_neighborhood_eval
PASS: four budget combinations over 4 internal / 3,357 boundary fixture edges.
Internal edges preserved; focal diversity; bounded neighbors; no dangling endpoints; explicit truncation.
```

## Docker/build/smoke checks

```text
python -m scripts.replay_substrate_neighborhood --request /tmp/reading-neighborhood/request.json --uri redis://localhost:6380
PASS via GRAPH.RO_QUERY against the running Falkor container.
Candidate: curiosity-e914a662ce620e47d36e1932.
Independent census: 4 internal, 3355 incoming, 0 outgoing.
Final result: 8 focal, 4 internal, 16 boundary, 16 neighbors, endpoint_integrity=true,
truncated=true, degraded=false, elapsed_ms=1140.07.
No container/service restart, deployment, production write, or graph mutation.
Evidence: /tmp/reading-neighborhood/{candidate,request,live-census,live-replay-final,eval}.json
```

## Review findings fixed

- Finding: a focal with both directions could receive two turns before another focal received one.
  - Fix: focal-first allocation, then rotation through that focal's directions and predicates.
  - Evidence: `test_focal_fairness_precedes_direction_and_predicate_diversity`; independent re-review confirmed resolution and no remaining material findings. Updated the tight-budget eval expectation to require both focal nodes before both directions of one focal.
- Review skill: `/home/athena/.claude/plugins/cache/claude-plugins-official/superpowers/6.4.1/skills/requesting-code-review/SKILL.md`, run in an independent read-only subagent.

## Restart required

No restart required. This patch provides an opt-in read API; existing runtime consumers have not been migrated.

## Risks / concerns

- Consistency: reads span multiple statements and are explicitly non-atomic. Detected missing/changed endpoints fail closed; undetected concurrent mutations may produce a mixed-time view.
- Continuations: not issued yet; supplied tokens return explicit restart-required status.
- Latency: about 1.14s on this live diagnostic; not a claim of suitability for every background tick. Bounded response/candidate volume does not guarantee bounded database work.
- Deferred work: assertion lifecycle/evidence retention/projection, full-cache correctness, persisted candidate migration, and Orion/Hub provenance interfaces are not implemented here. Their end-to-end path is UNVERIFIED.
- No new cognition metric is introduced. Receipt durations/counts are operational diagnostics only.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2497
