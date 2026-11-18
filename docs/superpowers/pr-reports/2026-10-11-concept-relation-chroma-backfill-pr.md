## Summary

- The vector database (orion-vector-db, Chroma) was running in memory: every restart wiped every collection. `IS_PERSISTENT=TRUE` was commented out in compose, and chroma 0.4.24 ignores `PERSIST_DIRECTORY` without it. Turned it on, with a gate test.
- Added `scripts/backfill_concept_relation_chroma.py`: a one-off (Juniper-approved 2026-10-11) re-projection of every active memory crystallization into `orion_memory_crystallizations`, the collection the concept-relation writer searches for candidates. It reuses the live projection call, so text, embedding endpoint/model, and bus path match exactly.
- Ran it live: 761/761 published, 0 errors. Concept-relation `/health` went from `degraded` (`chroma_collection_sparse:0`) to `ok` (count 761).

## Outcome moved

The concept-relation writer had nothing to compare against: 0 candidate docs vs 761 active crystallizations, and no decision since 2026-09-07. It now has a full candidate set. Once vector-db is redeployed it will also keep that set across restarts.

## Current architecture

The live path: `projector.project_crystallization` -> `chroma_publish.publish_crystallization_to_chroma` (HTTP embed, bge-large-en-v1.5) -> bus `orion:memory:vector:upsert` -> orion-vector-writer -> Chroma. Until PR #2600 the embed URL was wrong, so rows were skipped with `no_embedding` (421 rows had no chroma doc id at all). The 340 rows that once landed were later wiped by the in-memory Chroma.

## Architecture touched

- `services/orion-vector-db/docker-compose.yml`: `IS_PERSISTENT=TRUE`.
- New one-off script. Postgres is only read (`projection_refs` were not updated).

## Files changed

- `services/orion-vector-db/docker-compose.yml`: turns persistence on.
- `scripts/backfill_concept_relation_chroma.py`: the backfill (section 14 snapshot, progress, report).
- `tests/test_vector_db_compose_persistent.py`: fails if compose loses `IS_PERSISTENT` (verified it fails on the old compose).
- `tests/test_backfill_concept_relation_chroma.py`: covers idempotent skip, per-row failure isolation, doc-id parity with `build_chroma_upsert`, and progress fields.
- `services/orion-vector-db/README.MD` and `services/orion-memory-consolidation/README.md`: docs.

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none.
- Behavior changed: Chroma data now survives restarts. This applies to every collection on this Chroma (orion_main_store, orion_dreams, orion_curiosity, orion_reading_results, ...), not just this one. They will now grow across restarts, and there is no cap or cleanup.
- Compatibility notes: none.

## Env/config changes

- Added keys: none in `.env_example`. `IS_PERSISTENT` is a literal in the compose environment list.
- Local `.env` sync: not needed, because no template changed.

## Tests run

```text
pytest tests/test_backfill_concept_relation_chroma.py tests/test_vector_db_compose_persistent.py tests/test_compose_logging_journald.py tests/test_check_service_env_compose_parity.py -> all passed
scripts/check_compose_no_relative_mounts.py -> 0
```

## Evals run

```text
No eval harness for a one-off backfill. Live verification instead:
- snapshot: 761 targets, chroma_before count=0 (collection did not exist), 188 KB
- run: 761/761 published, 0 errors, 2.4 rows/s (0.2s throttle)
- chroma_after count=761; querying with a doc's own text returns itself at distance 0.0, then neighbors
- /health concept_relation: degraded (chroma_collection_sparse:0) at 00:48:24Z -> ok (count 761) at 00:58:24Z
Artifacts: /tmp/concept-relation-chroma-backfill/{targets.jsonl,chroma_before.json,chroma_after.json,before_after.csv,progress.log,report.md}
```

## Docker/build/smoke checks

```text
Not redeployed from this branch (production deploys run from the primary checkout on main).
Live evidence of the bug: in the vector-db container, Settings().is_persistent == False, and no chroma.sqlite3 exists anywhere.
```

## Review findings fixed

- Finding: the run order never said vector-db must be redeployed persistent first, or the backfill is lost.
  - Fix: added a step 0 to the docstring and README.
- Finding: `_chroma_ids` reported an unreachable Chroma as an empty collection.
  - Fix: heartbeat first. Only "does not exist" maps to empty, and any other error is raised. The count is now `len(ids)`.
  - Evidence: live check. A missing collection returns (0, empty, False), and a bad host raises.
- Finding: the doc-id parity test matched source text.
  - Fix: it now builds a real crystallization and compares `build_chroma_upsert().doc_id`.

## Restart required

After merge, from the primary checkout on main:

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-vector-db up -d
```

Then re-run the backfill, because the restart wipes the in-memory data one last time. Commands are in the script docstring. Other writers repopulate their own collections only as new events arrive.

## Risks / concerns

- Severity: high. Concern: the restart above wipes the current in-memory store (all collections) one final time. Mitigation: re-run the backfill for crystallizations. Other collections were already being wiped on every restart.
- Severity: medium. Concern: an older persistent Chroma from before the STORAGE_ROOT move (last write 2026-04-13) sits at `/mnt/storage-lukewarm/collapse-mirrors/chroma` (1.2 GB: orion_main_store 12,583, orion_chat_gpt 16,104, and others). This PR does not restore it. Mitigation: that is Juniper's decision.
- Severity: low. Concern: a new concept-relation decision row has not been observed yet. Decisions only fire when a new crystallization forms, about 2 per day. UNVERIFIED.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2609

🤖 Generated with [Claude Code](https://claude.com/claude-code)
