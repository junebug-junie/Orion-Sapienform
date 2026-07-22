# PR report: Cypher-native Falkor writer for collapse-triage enrichment

## Summary

- Phase 8 investigation traced what's still actually publishing to `orion-rdf-writer`'s `tags.enriched` subscription: only the generic `handle_triage_event` branch (Juniper-observer-gated `orion:collapse:triage` events) -- the chat/social-turn path stopped publishing to the bus entirely back on 2026-07-18. Live Fuseki check: 3 total `Enrichment` records across three weeks (2 real collapse-triage, 1 pre-cutover chat leftover) -- low volume but genuinely unique content, no Postgres/Falkor duplicate anywhere.
- Adds `write_collapse_triage_tags_to_falkor` (`app/falkor_recall_writer.py`), mirroring the existing chat-turn writer: `(:CollapseEvent {collapse_id})-[:HAS_TAG]->(:Tag)`, `-[:MENTIONS_ENTITY]->(:Entity)`, deliberately on the **same** `orion_recall` graph and Entity/Tag node universe as chat turns.
- Wired into `handle_triage_event`'s generic branch behind `RECALL_FALKOR_COLLAPSE_TRIAGE_ENABLED` (default `false` -- additive, not a swap; Fuseki remains the real persistence path for this workload until verified live).
- Code review caught a real bug in an unrelated same-day file (`orion-graph-compression`'s `episodic_falkor.py`, added earlier today): its coalesce() didn't include `collapse_id`, so once the new flag flips on, `:CollapseEvent`-sourced edges would silently resolve to a NULL node id and get dropped. Fixed with a regression test.
- Review also surfaced (documented, not fixed -- belongs to a different service) a real inconsistency in `orion-recall`'s entity-graph consumers once this flag is live: `fetch_entity_degrees` (unlabeled match, live in ranking) would pick up `:CollapseEvent` mentions, while `fetch_related_entities`/`fetch_bridging_turns`/`fetch_entity_mention_timeline`/`fetch_turns_mentioning_entities` (all hardcoded to `:ChatTurn`) would not.

## Outcome moved

The last remaining producer to `orion-rdf-writer`'s `tags.enriched` subscription now has a real Cypher-native alternative, ready to verify live and eventually cut Fuseki over for -- closing the loop on today's Phase 5/8 investigation.

## Current architecture

Before this patch: `orion-meta-tags`' generic collapse-triage branch had exactly one persistence path -- publish `tags.enriched` to the bus for `orion-rdf-writer` to materialize into Fuseki. No Falkor equivalent existed for this workload, unlike chat/social-turn tagging (Falkor-only since 2026-07-18).

## Architecture touched

- `services/orion-meta-tags/app/falkor_recall_writer.py`, `main.py`, `settings.py`, `.env_example`, `docker-compose.yml`, `README.md`.
- `services/orion-graph-compression/app/federators/episodic_falkor.py` (bug fix found via review, unrelated file added earlier today).

## Files changed

- `services/orion-meta-tags/app/falkor_recall_writer.py`: new `write_collapse_triage_tags_to_falkor`; comment on `write_entity_edges` acknowledging it's no longer the sole `MENTIONS_ENTITY` writer; docstring record-count clarification.
- `services/orion-meta-tags/app/main.py`: new async wrapper `_write_collapse_triage_tags_to_falkor_async`, wired into the generic triage branch before the existing bus publish.
- `services/orion-meta-tags/app/settings.py`, `.env_example`, `docker-compose.yml`: new `RECALL_FALKOR_COLLAPSE_TRIAGE_ENABLED` (default `false`).
- `services/orion-meta-tags/README.md`: new "Collapse-triage Falkor writer" section, including the disclosed `orion-recall` entity-consumer inconsistency.
- `services/orion-meta-tags/tests/test_falkor_recall_writer.py`, `test_chat_history_observer_gate.py`: unit tests for the new writer + async wrapper (flag-off, flag-on, write-failure-at-warning).
- `services/orion-graph-compression/app/federators/episodic_falkor.py`: added `collapse_id` to the identity coalesce, updated docstring.
- `services/orion-graph-compression/tests/test_federator_episodic_falkor.py`: regression tests proving CollapseEvent-sourced rows aren't dropped.

## Schema / bus / API changes

- Added: `(:CollapseEvent {collapse_id, ts, correlation_id?, sentiment?})` node shape in FalkorDB's `orion_recall` graph (dark until flag flips).
- Removed: none.
- Renamed: none.
- Behavior changed: none live yet -- flag defaults off.
- Compatibility notes: `orion:collapse:triage` intake and `orion:tags:enriched` publish are both unchanged; this is a pure addition alongside them.

## Env/config changes

- Added keys: `RECALL_FALKOR_COLLAPSE_TRIAGE_ENABLED` (orion-meta-tags), default `false`.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: no -- synced by hand directly in the primary checkout instead, since the generic sync script was found earlier today to revert in-flight unmerged changes when run against a still-stale primary `.env_example`.
- skipped keys requiring operator action: none.

## Tests run

```text
ORION_BUS_URL=redis://127.0.0.1:6379/0 PYTHONPATH=.:services/orion-meta-tags venv/bin/python3 -m pytest services/orion-meta-tags/tests -q
→ 26 passed

ORION_BUS_URL=redis://127.0.0.1:6379/0 PYTHONPATH=.:services/orion-graph-compression venv/bin/python3 -m pytest services/orion-graph-compression/tests -q
→ 44 passed

git diff --check → clean
scripts/check_service_env_compose_parity.py orion-meta-tags → same pre-existing gap (CHANNEL_EVENTS_CHAT_TURN, unrelated), no new gap
```

## Evals run

No eval harness exists for this service's tagging pipeline; focused deterministic tests cover the changed behavior.

## Docker/build/smoke checks

No Docker rebuild/restart performed. Flag defaults `false`, so no live behavior change until explicitly flipped after live verification (not done in this PR).

## Review findings fixed

- Finding (moderate): `episodic_falkor.py`'s coalesce() lacked `collapse_id`, meaning CollapseEvent-sourced graph-compression clustering input would silently drop to NULL once the new flag is enabled.
  - Fix: added `collapse_id` to both coalesce clauses, updated the federator's docstring (was stale as of this PR's own change), added a regression test.
  - Evidence: `test_fetch_does_not_drop_collapse_event_sourced_rows`, `test_fetch_query_coalesce_includes_collapse_id`.
- Finding (moderate, documented not fixed -- belongs to orion-recall, a different service): sharing the Entity/Tag namespace with chat turns does not mean all entity-graph consumers see CollapseEvent mentions consistently -- `fetch_entity_degrees` (live in ranking) would, the relatedness/bridging/timeline functions (hardcoded to `:ChatTurn`) would not.
  - Fix: documented explicitly in README.md's new section, not silently left for someone to discover after the flag is flipped.
- Finding (minor): `write_entity_edges`'s "single source of truth" docstring no longer covers all `MENTIONS_ENTITY` writers now that the collapse writer inlines its own Cypher.
  - Fix: comment added acknowledging the gap.
- Finding (trivial): docstring/README record-count wording mismatch (2 vs 3, both technically correct).
  - Fix: aligned wording.
- Finding (trivial): a test docstring overstated what it proves (normalization match vs. real node-identity resolution).
  - Fix: retitled and reworded.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-meta-tags/.env \
  -f services/orion-meta-tags/docker-compose.yml up -d --build

docker compose --env-file .env --env-file services/orion-graph-compression/.env \
  -f services/orion-graph-compression/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: Low.
- Concern: the disclosed `orion-recall` entity-consumer inconsistency means flipping `RECALL_FALKOR_COLLAPSE_TRIAGE_ENABLED` on later will change ranking behavior in a way that isn't obviously visible without reading this PR's README section first.
- Mitigation: flag defaults off; documented explicitly; flipping it is a separate, deliberate future decision, not part of this PR.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1271
