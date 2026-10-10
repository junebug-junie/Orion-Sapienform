## Summary

- The part of Orion that decides whether a new memory is the same as, refines, or contradicts an older one has written nothing since 2026-09-07 23:26. It was switched on, but it had no address for the embedding service or the vector store, so it quietly found zero matches every time.
- The service now ships with real addresses: embedding at `http://orion-athena-vector-host:8320/embedding`, Chroma (the vector database) at `orion-athena-vector-db:8000`. They are set in `.env_example`, `settings.py` and compose, and the feature is on by default.
- New boot check, re-run every 10 minutes: if the feature is on but a host is empty, a host is unreachable, or the vector collection is too sparse to supply candidates, the service logs `concept_relation_resolution_degraded` and `/health` returns `"degraded": true` with the reasons.
- `fetch_similar_candidates` now logs a warning once, instead of silently returning nothing, when the hosts are unset.

## Outcome moved

Before: this failure was a silent no-op for about a month. 86+ crystallizations formed with no relation decisions and no log line about it. After: the same misconfiguration is visible at boot and on `/health`, and the hosts are no longer empty by default.

A side effect: the same embed host also lets this service project new crystallizations into Chroma again. With the URL empty, `publish_crystallization_to_chroma` skipped every one with `no_embedding`.

## Current architecture

- **Service:** `services/orion-memory-consolidation` (container `orion-athena-memory-consolidation`, on network `app-net`).
- **Path:** `intake_pipeline` → `concept_relation.maybe_resolve…` → `candidate_retrieval.fetch_similar_candidates` (embeds over HTTP, then queries Chroma) → LLM judgment → a row in `memory_concept_relation_decisions`.

## Root cause (when and why the keys went empty)

- `git log -S` shows `CRYSTALLIZER_EMBED_HOST_URL` and `CHROMA_HOST` have shipped empty in `.env_example`, settings and compose since they were introduced (0bcd5d517 / aee5a871b, 2026-07-12). The empty value was on purpose ("degrades to a no-op").
- Live writes until 2026-09-07 therefore depended on hand-set values in the local `.env`.
- The last decision row is from 2026-09-07 23:26. Chroma projection coverage also collapsed that same week (19/19 active rows carried Chroma refs the week of 08-31, 9/25 the week of 09-07, about 0–4 a week after).
- A recorded session ran `sync_local_env_from_example.py --force` at 2026-09-08 03:33. That command resets every local value that differs from the example back to the example value, and the example value here was empty. This is the most likely cause, but I can't prove it, because `.env` is not versioned.
- Shipping real defaults removes the dependency on hand-set values.

## Architecture touched

- One service: `orion-memory-consolidation`.
- One shared helper: `orion/memory/crystallization/candidate_retrieval.py`. It only adds a log line, and its only caller is `concept_relation.py`.
- No bus or schema changes.

## Files changed

- `services/orion-memory-consolidation/.env_example`, `app/settings.py`, `docker-compose.yml`: real host defaults, flag on.
- `services/orion-memory-consolidation/app/concept_relation_readiness.py` (new): checks config, embeds a probe text, then runs a Chroma heartbeat and collection count (the two probes run concurrently). Includes the 10-minute re-check loop.
- `services/orion-memory-consolidation/app/main.py`: runs the check at boot, starts the loop, adds a `concept_relation` block and a `degraded` field to `/health`.
- `orion/memory/crystallization/candidate_retrieval.py`: one-time warning when unconfigured.
- `services/orion-memory-consolidation/README.md`: new defaults and readiness behavior; the table rendering is fixed.
- `services/orion-memory-consolidation/tests/test_concept_relation_readiness.py` (new): regression tests.

## Schema / bus / API changes

- Added: `/health` fields `concept_relation` (status, problems, checked_at, chroma_collection_count) and `degraded`.
- Removed / Renamed: none.
- Behavior changed: concept-relation resolution is on by default, and crystallization Chroma projection is re-enabled through the HTTP embed path.
- Compatibility notes: these fields are additive to a plain dict response.

## Env/config changes

- Added keys: none.
- Changed defaults: `CRYSTALLIZER_EMBED_HOST_URL` (empty → `http://orion-athena-vector-host:8320/embedding`), `CHROMA_HOST` (empty → `orion-athena-vector-db`), `CONCEPT_RELATION_RESOLUTION_ENABLED` (`false` → `true`).
- `.env_example` updated: yes.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: ran. It reported both host keys as "Diverged" (local value was empty) and correctly did not overwrite them. I did not use `--force`. Instead I edited exactly those two lines in the live `services/orion-memory-consolidation/.env` in the primary checkout. A re-run no longer lists any consolidation key as diverged, and `docker compose config` resolves all three values.
- Skipped keys requiring operator action: none. `ORION_BUS_URL` is untouched (`redis://100.92.216.81:6379/0`).

## Tests run

```text
.venv/bin/python -m pytest services/orion-memory-consolidation/tests -q        -> 279 passed, 11 skipped
.venv/bin/python -m pytest tests/test_memory_crystallization_concept_relation.py tests/test_memory_crystallization.py \
  tests/test_encode_reinforce_not_duplicate.py tests/test_check_concept_relation_digest_liveness.py \
  services/orion-hub/tests/test_concept_relation_classifier.py -q                -> 93 passed, 1 failed
  (test_memory_card_v1_unchanged_in_registry_gap fails identically on main; pre-existing, unrelated)
scripts/check_env_template_parity.py                                            -> PASS
scripts/check_env_key_single_source.py                                          -> OK
scripts/check_service_env_compose_parity.py orion-memory-consolidation          -> 14 pre-existing missing keys, none from this patch
```

## Evals run

```text
No eval harness covers this seam. The live smoke below stands in.
```

## Docker/build/smoke checks

I ran the new readiness module inside the live `orion-athena-memory-consolidation` container (read-only; no restart):

```text
real hosts     -> degraded ['chroma_collection_sparse:1'] count=1 (0.88s)  # hosts reachable, embed 200 / 1024-dim
current live   -> degraded ['embed_host_url_empty', 'chroma_host_empty']   # the bug, now loud
bogus hosts    -> degraded ['embed_host_unreachable:ConnectError', 'chroma_unreachable:ValueError']
```

From the container: `POST orion-athena-vector-host:8320/embedding` returned 200 with 1024 dimensions in 0.13s, and `orion-athena-vector-db:8000/api/v1/heartbeat` returned 200.

## Review findings fixed

- Finding: the Chroma collection `orion_memory_crystallizations` holds 1 document, against 760 active crystallizations (340 of which claim Chroma refs). Fixing the hosts alone would therefore report `ok` while still finding almost no candidates.
  - Fix: the readiness check now counts the collection and reports `chroma_collection_sparse:<n>` when it holds fewer documents than `CONCEPT_RELATION_CANDIDATE_LIMIT`.
  - Evidence: `test_sparse_collection_is_degraded`, plus the live smoke above.
- Finding: `/health` was a snapshot taken at boot.
  - Fix: `run_readiness_loop` re-probes every 10 minutes, and `checked_at` is exposed.
- Finding: the README paragraph split the env table, and the intro still said "off / empty".
  - Fix: moved the paragraph below the table and rewrote the intro.
- Finding: the probes ran one after the other.
  - Fix: they now run together with `asyncio.gather`.
- Finding: tests left module-global state behind, and the embed probe itself was untested.
  - Fix: added a fixture that resets the state, and `test_probe_embed_against_real_http_shape`, which uses `httpx.MockTransport`.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && docker compose --env-file .env --env-file services/orion-memory-consolidation/.env -f services/orion-memory-consolidation/docker-compose.yml up -d --build
```

Then check: `curl -s localhost:8635/health`. Expect `"degraded": true` with `chroma_collection_sparse:1` until Chroma is re-filled (see Risks). That reading is correct, not a regression.

## Risks / concerns

- Severity: high.
  - Concern: **a backfill is warranted, but it is a Chroma re-projection, not a re-run of relation decisions.** The vector collection holds 1 of about 760 active crystallizations. The vector-db container restarted about 18h ago, and Postgres still claims 340 Chroma refs, so the index was lost or never filled; I did not root-cause which. Until the active crystallizations are re-embedded and upserted, the writer will see almost no candidates. New rows will now project as they form, so coverage only rebuilds slowly.
  - Mitigation: `/health` now says this explicitly. Re-projection should be a separate, snapshotted job. I did not backfill old relation decisions, as instructed; replaying them would mean asking the LLM about 86 crystallizations after the fact, which is low value.
- Severity: medium.
  - Concern: `publish_crystallization_to_chroma` records a Chroma doc id as soon as the bus emit returns, without confirming that vector-writer actually stored it. That is how Postgres can claim 340 refs while Chroma holds 1. This is a separate silent-failure path and is not fixed here.
- Severity: low.
  - Concern: boot can wait up to about 5–10s if a host is hung. The probes have timeouts and never raise.
- Severity: low.
  - Concern: UNVERIFIED end to end. No row in `memory_concept_relation_decisions` will exist until the restart and the Chroma refill.

## PR link

(filled in after `gh pr create`)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
