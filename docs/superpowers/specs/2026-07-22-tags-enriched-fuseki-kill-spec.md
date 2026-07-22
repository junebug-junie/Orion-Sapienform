# `orion:tags:enriched` Fuseki write retirement -- design spec

**Date:** 2026-07-22
**Status:** Approved for implementation (Juniper: "so we can kill both? if yes, commit a design spec before you execute")
**Mode:** Implementation (spec + kill shipped in the same PR, per explicit instruction)
**Extends:** `docs/superpowers/specs/2026-07-17-recall-rdf-writer-falkor-cutover-phase2-spec.md` (Phase 5 -- "retire the SPARQL producer path for tag/entity enrichment").

---

## Arsonist summary

`orion:tags:enriched` (`CHANNEL_EVENTS_TAGGED`) has exactly one remaining live producer -- `orion-meta-tags`' generic `handle_triage_event` branch (Juniper-observer-gated collapse-mirror events) -- and exactly one consumer, `orion-rdf-writer`, which materializes it into Fuseki's `orion:enrichment` graph as `Entity`/`Mention`/`Enrichment` triples. Every other content type that used to flow through this channel (`chat.history`, `social.turn.stored.v1` tag/entity data) already stopped publishing to it on 2026-07-18, when that data became Falkor-only.

As of today, the collapse-triage branch also has a working Falkor writer (`write_collapse_triage_tags_to_falkor`, PR #1271), a completed historical backfill (68/68 real Juniper-observed entries, PR #1273), and one confirmed real live event that landed correctly in both Fuseki and Falkor simultaneously. With Falkor now covering 100% of what this channel ever carried, the Fuseki side is pure redundancy -- the same "wrong-tool precedent" already established for `chat.history`, `cognition.trace`, and `metacog.trace`.

Kill both ends: `orion-meta-tags` stops publishing `tags.enriched`; `orion-rdf-writer` stops subscribing to it and drops the now-dead `Enrichment` materialization code.

---

## Current architecture

```text
orion-meta-tags (handle_triage_event, generic branch)
  -> write_collapse_triage_tags_to_falkor()  [Falkor, orion_recall graph]  -- KEEP
  -> bus.publish(CHANNEL_EVENTS_TAGGED, tags.enriched)                    -- KILL
       |
       v
orion-rdf-writer (subscribed to orion:tags:enriched)
  -> _build_enrichment_graph() -> Entity/Mention/Enrichment triples       -- KILL
       |
       v
Fuseki orion:enrichment graph (read by orion-graph-compression's SPARQL
EpisodicFederator, additively unioned with FalkorEpisodicFederator since
today -- graph-compression keeps working either way, see below)
```

`RECALL_FALKOR_COLLAPSE_TRIAGE_ENABLED` currently frames Falkor as "additive, not a swap" (PR #1271's own docstring/README) because Fuseki was still the only durable path at the time it was written. This spec supersedes that framing: once the bus publish is removed, Falkor becomes the sole persistence path for collapse-triage tag/entity data, the same transition chat/social-turn tagging already went through on 2026-07-18.

---

## Verified live before this spec (not assumed)

- **Producer audit**: grepped `orion-meta-tags/app/main.py` -- `CHANNEL_EVENTS_TAGGED` is published from exactly one call site, the generic branch. The chat/social branch's own bus publish was already removed 2026-07-18.
- **Consumer audit**: `orion-rdf-writer`'s live subscribe list (`get_all_subscribe_channels()`) still includes `orion:tags:enriched`; `_build_enrichment_graph` is still the only handler for `tags.enriched` envelopes.
- **No other reader depends on `orion:enrichment`'s continued growth**: `orion-graph-compression`'s `EpisodicFederator` (SPARQL) reads it, but is already additively unioned with `FalkorEpisodicFederator` (merged today, PR #1265) -- killing new writes to `orion:enrichment` does not remove graph-compression's ability to cluster this content, it just stops the SPARQL side from growing (the Falkor side keeps growing). `orion-cortex-orch`'s `concept_profile_config.py` (`RECALL_RDF_ENDPOINT_URL` alias) was checked and confirmed to target a *different* graph (`http://conjourney.net/graph/spark/concept-profile`, Spark ConceptProfile -- unrelated to chat/collapse tag enrichment).
- **End-to-end live proof**: a real Juniper-observed collapse-mirror event sent today (`collapse_11e07eaf1fbd478687a255a8f1e30a28`) landed correctly in Postgres (`collapse_mirror`, the real durable raw store -- untouched by this spec), Fuseki (`orion:enrichment`, about to be killed), and Falkor (`orion_recall`, `CollapseEvent` node with real extracted entities `orion`/`falkordb`) simultaneously. Falkor's copy is not a shadow -- it's the same content, verified queryable.
- **Historical backfill complete**: 68/68 real Juniper-observed `collapse_mirror` rows (2025-09-24 through 2026-07-08) already migrated to Falkor (PR #1273) -- killing the Fuseki write does not lose any historical data, since Falkor already has full parity plus the live event on top.

---

## Proposed changes (no schema/API changes -- this is a pure removal)

### `services/orion-meta-tags`

- `app/main.py`: remove the `await meta_tagger.bus.publish(settings.CHANNEL_EVENTS_TAGGED, out_env)` call and the `Enrichment`/`out_env` construction that only exists to feed it, from `handle_triage_event`'s generic branch. The branch keeps: the Juniper-observer gate, NER extraction, sentiment heuristic, and the `write_collapse_triage_tags_to_falkor` call.
- `app/settings.py`, `.env_example`, `docker-compose.yml`: `RECALL_FALKOR_COLLAPSE_TRIAGE_ENABLED` default flips `false` -> `true` (matching `RECALL_FALKOR_TAG_ENTITY_ENABLED`'s convention: once Falkor is the *only* path, disabling it means no persistence at all, not "skip a shadow write"). `CHANNEL_EVENTS_TAGGED` env var itself: check whether anything else in this service still references it before removing the field entirely (`preview_validate_only`/other routes do not touch it, but confirm during implementation, not assumed here).
- `README.md`: update "Collapse-triage Falkor writer" section to drop the "additive, not a swap" framing and fold in a "Historical note" matching the existing chat/social-turn section's shape.
- Tests: the three async tests added in PR #1271 (`test_collapse_triage_falkor_write_skipped_when_flag_disabled_by_default`, `..._writes_to_falkor_and_still_publishes_when_flag_enabled`, `..._write_failure_is_swallowed_at_warning_not_error`) all assert `bus.publish` behavior that no longer applies -- update to match the new sole-persistence-path shape (mirroring `test_chat_history_falkor_write_skipped_when_flag_disabled`'s pattern from the same file).

### `services/orion-rdf-writer`

- `app/settings.py`: remove `CHANNEL_EVENTS_TAGGED` from `get_all_subscribe_channels()`; remove the field with an explanatory comment (matching the existing dead-channel comment block already in this file for `chat.history`/`cognition.trace`/etc.).
- `app/rdf_builder.py`: remove the `tags.enriched` dispatch branch and the `_build_enrichment_graph` handler function (and any now-unused imports/helpers it alone used).
- `.env_example`, `docker-compose.yml`: remove the now-dangling `CHANNEL_EVENTS_TAGGED` key.
- `orion/bus/channels.yaml`: update `orion:tags:enriched`'s `consumer_services` to drop `orion-rdf-writer`.
- `README.md`: remove stale channel-table rows referencing this, matching the review-finding pattern from the earlier chat.history kill (PR #1164).
- Tests: add channel-not-subscribed + quiet-no-op dispatch regression tests, matching the exact pattern already used for every prior kill in this file (`test_drives_audit_channel_not_in_default_subscriptions`, `test_chat_history_channels_not_in_default_subscriptions`, `test_cognition_metacog_channels_not_in_default_subscriptions`).

---

## Files likely to touch

- `services/orion-meta-tags/app/main.py`, `settings.py`, `.env_example`, `docker-compose.yml`, `README.md`, `tests/test_chat_history_observer_gate.py`
- `services/orion-rdf-writer/app/settings.py`, `rdf_builder.py`, `.env_example`, `docker-compose.yml`, `README.md`, `tests/test_autonomy_materialization.py` (or wherever the existing channel-kill regression tests live)
- `orion/bus/channels.yaml`

---

## Non-goals

- Not touching `orion:rdf:enqueue`, `orion:rdf-collapse:enqueue`, `orion:collapse:intake`, `orion:chat:social:stored`, or `orion:core:events` -- `orion-rdf-writer` keeps all of these; it is not being decommissioned as a service, only this one channel/handler.
- Not touching the raw `collapse_mirror` Postgres table or the `orion:collapse:sql-write` channel that feeds it -- that remains the durable source-of-truth for the full CollapseMirrorEntryV2 payload, unrelated to the derived tag/entity enrichment this spec kills.
- Not decommissioning Fuseki itself, or purging the existing (now-frozen) `orion:enrichment` graph content -- matches the established precedent (Fuseki stays up read-only until Phase 10, a separate, later, explicit decision).
- Not touching `orion-graph-compression`'s SPARQL `EpisodicFederator` -- it keeps running, just stops seeing fresh `orion:enrichment` content going forward (the Falkor side already covers that gap additively).

---

## Acceptance checks

- [ ] `orion-meta-tags` no longer publishes to `orion:tags:enriched` for any kind.
- [ ] `orion-rdf-writer` no longer subscribes to `orion:tags:enriched`; dispatching that kind is a quiet no-op if it ever arrives.
- [ ] `RECALL_FALKOR_COLLAPSE_TRIAGE_ENABLED` defaults `true`; disabling it means no persistence for this data, documented as such.
- [ ] `channels.yaml` accurately reflects both changes.
- [ ] All existing + updated tests pass in both services.
- [ ] Live verification: send (or wait for) one real collapse-mirror event post-deploy, confirm it lands in Falkor and Postgres, confirm zero `tags.enriched` traffic in `orion-rdf-writer`'s logs.

---

## Recommended next patch

Implement exactly the changes above, in this same PR: producer kill (meta-tags), consumer kill (rdf-writer), contract fix (channels.yaml), tests, docs, live verification, PR report.
