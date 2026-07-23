# PR report: kill orion-vision-scribe's dead Fuseki RDF write

## Summary

- While auditing what still depends on Fuseki before considering the container/data-store safe to decommission, the live Fuseki container's own request log (checked directly, not assumed) showed a real, current write: `POST .../orion/data?graph=...orion/vision`, from `orion-vision-scribe`.
- Live audit (matching this repo's established "verify redundant, then kill" pattern used for every other successful Fuseki write removal): Postgres `vision_events` has 33,389 rows, latest row timestamped to the same second as the caught Fuseki write. Postgres's schema (`confidence`/`salience`/`evidence_refs`/`tags` as real structured columns) is strictly richer than the flat RDF literals (`hasNarrative`/`hasType`/`mentionsEntity` strings only). A repo-wide grep found zero readers of Fuseki's `orion:vision` graph anywhere.
- Removed the RDF write entirely (not migrated -- there was no real signal to preserve, same conclusion as cognition/metacog, identity/goals, and Hub's orionmem in prior PRs).

## Outcome moved

`orion-vision-scribe` no longer depends on Fuseki at all. This closes one of three open dependents identified in a fresh live-audit pass on `orion-recall`'s Falkor cutover (PR #1281/#1286) -- the only one of the three that was a hard write dependency (the other two, `orion-graph-compression`'s SPARQL federators and `orion:chat:social:stored`'s unmigrated SocialRoomTurn/SocialConceptEvidence data, are read-side/fail-open and remain separately open).

## Current architecture

Before this patch: `_write_to_sinks` dual-wrote every vision event -- Postgres (`vision.event.v1` bus message -> `orion-sql-writer` -> `vision_events` table) and Fuseki (`rdf.write.request` bus message -> `orion-rdf-writer` -> `orion:vision` RDF graph, built via `rdflib`). Both writes were independent, not transactional; `ack.ok` required both to succeed.

## Architecture touched

- `services/orion-vision-scribe/app/{main.py,settings.py,.env_example,docker-compose.yml,requirements.txt,README.md}`
- `scripts/vision_persistence_smoke.py`, `scripts/smoke_vision_persistence_live.sh` (shared smoke-test helper, also used by the pytest suite)
- `orion/bus/channels.yaml`
- `docs/vision_services.md`

## Files changed

- `services/orion-vision-scribe/app/main.py`: removed `_build_event_triples`, the `RdfWriteRequest` import + inline fallback stub, `rdflib` imports, the `ORION` namespace constant, and the RDF write branch inside `_write_to_sinks`. `ack.ok` now depends on SQL success alone (previously `sql_ok and rdf_ok` via the `errors` list) -- correct, since RDF failures were a false-negative source for the sink that actually mattered.
- `services/orion-vision-scribe/app/settings.py`, `.env_example`, `docker-compose.yml`: removed the now-dangling `CHANNEL_RDF_ENQUEUE` key.
- `services/orion-vision-scribe/requirements.txt`: removed `rdflib==7.0.0` (only used by the removed code).
- `services/orion-vision-scribe/README.md`: replaced the "RDF write path" section with a note explaining the removal and the live-verification evidence.
- `services/orion-vision-scribe/tests/test_write_to_sinks.py`: removed RDF-specific tests; added `test_write_to_sinks_makes_exactly_one_publish_call_no_rdf` and `test_write_to_sinks_reports_ok_when_sql_write_succeeds` (regression coverage for both real behavior changes).
- `services/orion-vision-scribe/tests/test_vision_persistence_smoke_helpers.py`: removed RDF-specific tests, trimmed `test_channel_constants_match_bus_catalog`, added `test_no_rdf_symbols_remain_on_smoke_module`.
- `scripts/vision_persistence_smoke.py`: removed `rdf_ntriple_markers`/`build_rdf_write_request`, `CHANNEL_RDF_ENQUEUE`/`CHANNEL_RDF_CONFIRM`/`KIND_RDF_WRITE_REQUEST`, `_wait_for_rdf_signals`, and all RDF logic from `run_contract_mode()`/`run_live_mode()` -- including the gate that would have made the live smoke test fail forever once the write it waited for stopped existing.
- `scripts/smoke_vision_persistence_live.sh`: updated header comments.
- `orion/bus/channels.yaml`: dropped `orion-vision-scribe` from `orion:rdf:enqueue`'s `producer_services`; dropped it from `orion:rdf:confirm`'s `consumer_services` too (review found this entry was already stale/aspirational -- no `CHANNEL_RDF_CONFIRM` setting ever existed in this service per git history, unrelated to this PR's own change but cleaned up alongside it since both concern the same removed RDF path).
- `docs/vision_services.md`: corrected the vision-scribe catalog line, which claimed "SQL, RDF, and Vector stores."

## Schema / bus / API changes

- Removed: `orion-vision-scribe` as a producer of `orion:rdf:enqueue`, and as a (stale, never-real) consumer of `orion:rdf:confirm`.
- Behavior changed: vision events are now Postgres-only. `ack.ok` reflects SQL-write success only.
- Compatibility notes: `orion-rdf-writer`'s generic `RdfWriteRequest` handler is untouched (it's a passthrough for any graph name, used by other producers) -- confirmed via grep, no vision-specific logic existed there to also remove.

## Env/config changes

- Removed keys: `CHANNEL_RDF_ENQUEUE` (`orion-vision-scribe`).
- `.env_example` updated: yes.
- Local `.env` synced: N/A -- this service declares `env_file:` in its own `.env_example`/compose wiring so no key existed to remove from a separate live `.env`; confirmed via `check_service_env_compose_parity.py orion-vision-scribe` -> N/A.
- Skipped keys requiring operator action: none.

## Tests run

```text
ORION_BUS_URL=redis://127.0.0.1:6379/0 PYTHONPATH=.:services/orion-vision-scribe venv/bin/python3 -m pytest services/orion-vision-scribe/tests -q
-> 8 passed

python3 scripts/check_service_env_compose_parity.py orion-vision-scribe -> N/A (env_file covers everything)
ORION_BUS_URL=redis://100.92.216.81:6379/0 python3 scripts/check_single_consumer_channels.py -> OK, 30 channels checked, 3 pre-existing warnings unrelated to this patch
python3 -c "import app.main" -> imports cleanly, RdfWriteRequest symbol gone
VISION_PERSISTENCE_SMOKE_MODE=contract python3 -m scripts.vision_persistence_smoke -> PASS (contract mode), reproduced by the review agent independently
git diff --check -> clean
```

## Evals run

No eval harness exists for this service; focused deterministic tests plus the existing contract-mode smoke test cover the changed behavior.

## Docker/build/smoke checks

No container rebuild/restart performed as part of this PR. Live Fuseki traffic (the write this PR removes) was directly observed in `orion-athena-fuseki`'s own container log during the investigation that led to this patch -- see Summary.

## Review findings fixed

- Finding (should-fix): `orion/bus/channels.yaml`'s `orion:rdf:confirm` entry still listed `orion-vision-scribe` as a consumer, undermining the diff's own "fully out of the RDF business" claim.
  - Fix: removed from `consumer_services`; added a comment noting (confirmed via git history) this was already a stale/aspirational entry, not a real subscription this PR broke.
  - Evidence: commit `7c4b9b1f`.
- Finding (should-fix): `docs/vision_services.md`'s service catalog still described vision-scribe as persisting to "SQL, RDF, and Vector stores."
  - Fix: corrected to describe the SQL-only reality and cite the removal.
  - Evidence: same commit.
- Informational (no fix needed, pre-existing and out of scope): the review also noted the catalog's "Vector stores" claim was already false before this PR (`CHANNEL_VECTOR_WRITE` is declared but never referenced in `main.py`, old or new code) -- not touched, unrelated to this PR's actual change.
- Informational (no fix needed): `ack.ok`'s new dependency on `sql_ok` alone was verified by the reviewer to be the correct, intended behavior (removes RDF flakiness as a false-negative source), not a hidden regression -- confirmed by reading the pre-diff file directly rather than trusting the PR's own framing.
- Informational (no fix needed): `run_live_mode`'s rewrite was hand-traced by the reviewer against the diff and confirmed to have no leftover references to removed queues/constants and no dead-end blocking wait.

## Restart required

No restart required to merge. `orion-athena-vision-scribe` should be rebuilt/redeployed once merged so the running container stops attempting the (already relatively harmless, fail-open) RDF write:

```bash
docker compose --env-file .env --env-file services/orion-vision-scribe/.env \
  -f services/orion-vision-scribe/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: Low.
- Concern: none identified that block merge. The two open, separately-tracked Fuseki dependents from the same audit (`orion-graph-compression`'s SPARQL federators, `orion:chat:social:stored`'s unmigrated data) are unaffected by this PR and remain open questions before the Fuseki container/data store itself can be safely removed.

## PR link

(added after opening)
