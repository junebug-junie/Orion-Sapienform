## Summary

- Wires self_study Layer 2 (`induce_self_concepts`) to three new, additive concept sources: graphify community clustering, `orion.structural_mass` repo-wide structural delta, and `orion-self-study-enrichment`'s cached LLM enrichment summaries.
- None of the five existing hardcoded concept branches change; the new sources only add concepts.
- Adds a new `reflect_self_concepts()` finding that pairs a `structural_mass` concept with a `semantic_enrichment` concept when both are present in the same induction run (what moved vs. why).
- Fixes a real, reachable crash bug found in code review: `reflect_self_concepts()`'s internal idempotency re-check re-invokes `induce_self_concepts()` on the same snapshot, which — without memoization — silently dropped the structural delta concept on the second pass and raised `concept_idempotency_mismatch`.
- Also fixes an uncaught `AttributeError` on malformed enrichment-cache entries, and de-duplicates a 5x-copy-pasted section-name tuple into one shared constant.

This is the continuation of the self-study-enrichment service work (`services/orion-self-study-enrichment`, PRs #1574/#1578/#1586) — this PR is the consumer side of that service's cache, left as an explicit follow-up.

## Outcome moved

self_study's Layer 2 concept induction was previously five hardcoded branches with no path for new architectural signal to enter it. It can now surface real structural/semantic signal from graphify, structural_mass, and LLM-generated cluster summaries — verified against this repo's own live `graphify-out/graph.json` (165 real `graphify_community` concepts from live data, not mocked).

## Current architecture

- `services/orion-cortex-exec/app/self_study.py`: three-layer self-study (`build_self_snapshot` / `induce_self_concepts` / `reflect_self_concepts`), Phase 2A validation (`validate_phase2a_induction`) enforcing evidence-chain and same-snapshot-in → same-concept-ids-out idempotency.
- `services/orion-self-study-enrichment`: separate service, subscribes to `orion:self_study:enrichment:requested`, spawns a `claude -p` subprocess authenticated via the host's real Claude Code CLI session credential, caches content-hash-keyed LLM summaries on its own Docker volume.
- No prior wiring existed between these two — the enrichment service produced a cache nothing consumed.

## Architecture touched

- `services/orion-cortex-exec/app/self_study.py`: `_graphify_derived_concepts()`, `_structural_delta_concepts()`, `_semantic_enrichment_concepts()`, wired into `induce_self_concepts()`; new pairing finding in `reflect_self_concepts()`.
- `services/orion-cortex-exec/docker-compose.yml`: new read-only mount of `orion-self-study-enrichment`'s `self_study_enrichment_data` volume (external, owned by that service).
- `services/orion-cortex-exec/.env_example`: new `SELF_STUDY_ENRICHMENT_CACHE_MOUNT_DIR` key.
- `orion/schemas/self_study.py`: `SelfConceptKind` widened with `graphify_community` / `structural_mass` / `semantic_enrichment`.

## Files changed

- `services/orion-cortex-exec/app/self_study.py`: three new Layer 2 concept sources, new reflection-pairing finding, review-fix memoization/locking around the structural-delta global state, malformed-cache-entry guard, shared `_SNAPSHOT_SECTION_NAMES` constant.
- `orion/schemas/self_study.py`: `SelfConceptKind` literal widened; `inferred_from` field comment documents the two new tag values (already a free-form `list[str]`, no schema break).
- `services/orion-cortex-exec/docker-compose.yml`: read-only cross-service volume mount + env passthrough for the cache path.
- `services/orion-cortex-exec/.env_example`: new `SELF_STUDY_ENRICHMENT_CACHE_MOUNT_DIR` key, documented.
- `services/orion-cortex-exec/tests/test_self_study_pass1.py`: 11 new tests.

## Schema / bus / API changes

- Added: `SelfConceptKind` values `graphify_community`, `structural_mass`, `semantic_enrichment`.
- Added: `SelfInducedConceptV1.inferred_from` documented values `graphify_community`, `structural_mass_delta`, `semantic_enrichment` (field itself unchanged — free-form `list[str]`).
- Removed: none.
- Renamed: none.
- Behavior changed: `induce_self_concepts()` and `reflect_self_concepts()` can now produce additional concept/finding kinds; all pre-existing kinds are unaffected.
- Compatibility notes: additive only — no consumer of the old five concept kinds needs to change.

## Env/config changes

- Added keys: `services/orion-cortex-exec/.env_example`'s `SELF_STUDY_ENRICHMENT_CACHE_MOUNT_DIR=/mnt/self_study_enrichment_data/cache/self_study_enrichment`.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes.
- local `.env` synced: **not yet** — the real `.env` for `orion-cortex-exec` lives in the shared checkout (`/mnt/scripts/Orion-Sapienform/services/orion-cortex-exec/.env`), and this branch hasn't merged there. Per the established pattern from PR #1574→#1578→#1586, the env sync + redeploy happens in the shared checkout after merge. **Follow-up action required on merge**: run `python scripts/sync_local_env_from_example.py` for `orion-cortex-exec` from the shared checkout, or add the key manually.
- skipped keys requiring operator action: none beyond the sync above.

## Tests run

```text
/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest services/orion-cortex-exec/tests/test_self_study_pass1.py -q
42 passed, 13 warnings in 49.42s

/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest services/orion-cortex-exec/tests/test_self_study_consumer_wiring.py services/orion-cortex-exec/tests/test_self_study_harness.py services/orion-cortex-exec/tests/test_self_study_graphdb.py services/orion-cortex-exec/tests/test_self_study_policy.py -q
24 passed, 12 warnings in 115.53s
```

11 new tests: real (non-mocked) `graphify_community` concepts against this repo's own `graphify-out/graph.json`; covered-item skip; cold-start/unchanged-snapshot guards; direct memoization regression test; malformed-cache-entry handling (non-dict JSON, missing summary, invalid JSON); real cache-entry readback; the new structural+semantic reflection-pairing finding; and — critically — an end-to-end regression test (`test_reflect_self_concepts_does_not_crash_on_a_real_structural_delta`) that primes a real `GraphSnapshotStats` prior against the real on-disk graph to produce a genuine structural delta and confirms `reflect_self_concepts()` no longer raises.

## Evals run

No dedicated eval harness exists for `orion-cortex-exec`'s self_study module (unit tests are the only coverage lane) — not added here; flagged as a pre-existing gap, not new to this patch.

## Docker/build/smoke checks

Not run — no local Docker daemon access in this session for a full `docker compose up`. Config sanity was checked manually (grep-confirmed the new env key appears identically in both `.env_example` and `docker-compose.yml`; `check_service_env_compose_parity.py` itself fails on this repo's pre-existing `!override` YAML tag in the compose file, confirmed on `main` too, not introduced here).

## Review findings fixed

- Finding: `_structural_delta_concepts()` unconditionally mutates module-global state on every call; `reflect_self_concepts()`'s internal `validate_phase2a_induction()` re-invokes `induce_self_concepts()` on the same snapshot, so the second call sees its own first call's write as "unchanged" and silently drops the real delta concept, raising `concept_idempotency_mismatch` and crashing the reflect verb on a genuine repo change.
  - Fix: per-`snapshot_id` result memoization (bounded 8-entry `OrderedDict`) plus a `threading.Lock` around the global read-then-write.
  - Evidence: `test_reflect_self_concepts_does_not_crash_on_a_real_structural_delta` (primes a real `GraphSnapshotStats` prior, produces a genuine delta, confirms no raise) and `test_structural_delta_concepts_repeat_call_is_memoized_not_reobserved`.
- Finding: same global also had an unsynchronized read-then-write reachable from concurrent async bus-verb handlers, risking cross-request state bleed.
  - Fix: same lock covers this.
  - Evidence: code inspection — `_STRUCTURAL_DELTA_STATE_LOCK` now wraps the full read-modify-write section.
- Finding: `_semantic_enrichment_concepts()` crashed with `AttributeError` on a cache entry whose JSON parsed to a non-dict (e.g. a partial/crashed write by the other service).
  - Fix: added `isinstance(entry, dict)` guard before `.get()`.
  - Evidence: `test_semantic_enrichment_concepts_skips_malformed_entries`.
- Finding: the 7-item snapshot-section-name tuple was independently copy-pasted at 5 call sites (3 pre-existing, 2 new).
  - Fix: extracted to one shared `_SNAPSHOT_SECTION_NAMES` constant, all 5 sites updated.
  - Evidence: `grep -n _SNAPSHOT_SECTION_NAMES services/orion-cortex-exec/app/self_study.py` shows 1 definition + 5 uses, 0 remaining literal copies.
- Finding (partially addressed): `_structural_delta_concepts()`'s bare `except Exception` made "nothing cached yet" indistinguishable from a real degraded-read failure.
  - Fix: narrowed to `OSError, ValueError, TypeError` (the expected read/parse failure modes); unexpected exceptions now propagate instead of silently returning empty.
  - Evidence: code inspection.
- Not fixed (disclosed, not material correctness bugs): duplicate `graphify-out/graph.json` readers across `cortex-exec`/`orion-self-study-enrichment`/`orion-cocreation-signals` (existing-mechanism duplication, would need a shared `orion/` helper — bigger than this patch); repeated full-file JSON parse per request (latency/CPU cost, not correctness); `_enrichment_cluster_root()` intentionally duplicated from the other service's clustering rule (already documented in its own docstring per CLAUDE.md sec 5's cross-service boundary rule).

## Restart required

```bash
# After merge, from the shared checkout:
cd /mnt/scripts/Orion-Sapienform
git switch main && git pull --ff-only
python scripts/sync_local_env_from_example.py   # or manually add SELF_STUDY_ENRICHMENT_CACHE_MOUNT_DIR to services/orion-cortex-exec/.env
docker compose \
  --env-file .env \
  --env-file services/orion-cortex-exec/.env \
  -f services/orion-cortex-exec/docker-compose.yml \
  up -d --build
```

## Risks / concerns

- Severity: low
- Concern: `_structural_delta_concepts()`'s memoization cache is bounded to 8 entries but never explicitly evicts on process restart — acceptable given cortex-exec restarts periodically and 8 entries is small.
- Mitigation: none needed; documented in the code comment.

- Severity: low
- Concern: cross-service cache-format coupling — `_enrichment_cluster_root()` and the cache-entry JSON shape (`summary`/`touched_paths`) are duplicated/assumed from `orion-self-study-enrichment`'s real `app/cache.py`/`app/evidence.py`, not imported (correct per CLAUDE.md sec 5, but means the two can drift silently if that service's cache shape changes).
- Mitigation: docstring cross-references in place; a shared `orion/` schema for the cache entry shape would close this fully — flagged as a follow-up, not done here to keep this patch thin.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1596
