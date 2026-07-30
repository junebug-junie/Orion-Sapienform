# Concept-region reinforcement metadata-clobber protection

## Summary

- Fifth fix in a same-day incident chain (PR #1498, #1501, #1503, #1505) around `node:substrate.bus_synaptic`'s `prediction_error` getting durably frozen/nulled by unprotected writes.
- `reinforce_matched_concepts()` (`services/orion-recall/app/collectors/concept_region.py`) reads a concept node's full metadata via `get_node_by_id()`, boosts only `signals.activation.activation`, and re-persists the whole node — re-writing whatever stale copy of reducer-owned fields happened to be in that read.
- Lower frequency than the materializer/concept-induction paths (query-triggered per matched turn, not a tight loop), same bug class.
- Fix: `skip_metadata_keys=EXTERNALLY_OWNED_METADATA_KEYS`, same reused protection as the earlier PRs.

## Outcome moved

Closes another live instance of the same-day incident's root bug class, in orion-recall's concept-region reinforcement path.

## Current architecture

`reinforce_matched_concepts()` is called from `fetch_concept_region_fragment_and_reinforce()`, itself called from `orion-recall`'s worker for turns that mention a seeded concept label — boosting the matched concept node's activation toward 1.0 (a recall-relevance signal) and re-persisting via `store.upsert_node()`. Production resolves `store` to a real `FalkorSubstrateStore` (`SUBSTRATE_STORE_BACKEND=falkor` is this service's configured default).

## Architecture touched

`services/orion-recall` only. No contract/schema/bus changes.

## Files changed

- `services/orion-recall/app/collectors/concept_region.py`: `reinforce_matched_concepts()`'s `store.upsert_node()` call now passes `skip_metadata_keys=EXTERNALLY_OWNED_METADATA_KEYS`.
- `services/orion-recall/tests/test_concept_region_collector.py`: new test `test_reinforcement_protects_reducer_owned_metadata` spies on the store's `upsert_node` and asserts the kwarg is passed correctly.

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

```
$ /mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest services/orion-recall/tests/test_concept_region_collector.py -q
28 passed (27 baseline + 1 new), zero regressions.

$ /mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest services/orion-recall/tests -q
243 passed, 3 pre-existing failures — confirmed identical on main, zero regressions.
```

Red-before-green confirmed via `git show HEAD:<path>` (not `git stash`).

## Evals run

Not applicable.

## Docker/build/smoke checks

```
$ bash scripts/safe_docker_build.sh orion-recall up -d --build
... rebuilt and restarted ...
```

Live-verified: clean startup (bus connected, RPC listener ready). One pre-existing, unrelated log line at startup (`RDF endpoint check failed ... orion-athena-fuseki ... NameResolutionError`) — expected, Fuseki was fully decommissioned in an earlier, unrelated change; not caused by this patch.

## Review findings fixed

Code review pass (subagent) found **no blocking issues** in this diff's own files. Notes from that pass:

- Traced the real production call chain (`worker.py` → `fetch_concept_region_fragment_and_reinforce` → `get_substrate_store()` → `build_substrate_store_from_env()`) and confirmed `.env`/`.env_example`/`docker-compose.yml` all set `SUBSTRATE_STORE_BACKEND=falkor`, so production genuinely uses `FalkorSubstrateStore`, which already correctly supports `skip_metadata_keys`. Also checked every other concrete backend for completeness — all support it, no test-double regression risk like PR #1501 hit for a different service.
- Confirmed no other write call site in `orion-recall` touches substrate nodes — this was the only unprotected `upsert_node()` call in the service.
- **Flagged an inaccuracy in this diff's own code comment**: the comment lists `concept_induction`'s `materialize_concept_profile_to_falkor()` as already having this same protection "proven" — that's correct as of when this PR's branch was created, but the reviewing agent's worktree happened to predate PR #1503 (which fixes exactly that function) landing/being reviewed, so it read as unfixed from that vantage point. No code change needed here since PR #1503 already covers it; noting for the record so a future reader isn't confused by the apparent mismatch between the comment and an isolated worktree's view of `git log --all`.

## Restart required

```
bash scripts/safe_docker_build.sh orion-recall up -d --build
```

Already run during this session's live verification. No further restart needed unless this branch is rebuilt from a fresh checkout.

## Risks / concerns

- Severity: Low
- Concern: None material identified by review.
- Mitigation: N/A.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/new/fix/concept-region-reinforcement-metadata-protection
