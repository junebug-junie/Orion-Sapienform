# `orion:tags:enriched` Fuseki write retirement -- design spec

**Date:** 2026-07-22
**Status:** Approved for implementation, SCOPE CORRECTED mid-implementation (see below). Original ask (Juniper: "so we can kill both? if yes, commit a design spec before you execute") assumed `orion-rdf-writer` was the only consumer of this channel -- that assumption was wrong and is corrected here before any code shipped on the wrong premise.
**Mode:** Implementation (spec + kill shipped in the same PR, per explicit instruction)
**Extends:** `docs/superpowers/specs/2026-07-17-recall-rdf-writer-falkor-cutover-phase2-spec.md` (Phase 5 -- "retire the SPARQL producer path for tag/entity enrichment").

## Correction found during implementation

The spec below was written and committed first, as instructed, on the premise (stated in "Verified live before this spec" and confirmed in conversation) that `orion-rdf-writer` was the only consumer of `orion:tags:enriched`. While implementing the producer-side kill in `orion-meta-tags`, a grep across the repo (not just the two services named in the original ask) found `orion-sql-writer` is **also** subscribed to this exact channel (`services/orion-sql-writer/app/settings.py`'s `sql_writer_subscribe_channels`), materializing it into a Postgres `collapse_enrichment` table (`services/orion-sql-writer/app/models/collapse_enrichment.py`). Live-checked: 76 rows, latest timestamp same day as this correction -- not stale. Two real downstream readers exist: `services/orion-recall/app/storage/sql_adapter.py` and `services/orion-dream/app/aggregators_sql.py`, both with live `FROM collapse_enrichment` / equivalent queries.

This is a materially different situation from Fuseki's `orion:enrichment` graph, whose only reader (`orion-graph-compression`'s SPARQL `EpisodicFederator`) is already additively covered by the Falkor federator. `collapse_enrichment` has no such Falkor-side equivalent yet -- killing the bus publish would have silently broken two live services with no substitute in place, which is exactly the "no empty-shell cognition" / "runtime truth beats config truth" failure mode this repo's own doctrine exists to catch.

**Corrected scope: kill only `orion-rdf-writer`'s side.** `orion-meta-tags` keeps publishing to `orion:tags:enriched` unchanged -- `orion-sql-writer` needs it. Only `orion-rdf-writer`'s subscription to the channel and its `_build_enrichment_graph` Fuseki materialization are removed; that piece's redundancy (with the Falkor write) was independently verified and still holds. All producer-side edits made under the original (wrong) premise were reverted before this correction was written.

---

## Arsonist summary

*(Original framing below, kept for the record -- corrected by "Correction found during implementation" above: `orion-rdf-writer` is NOT the only consumer, so only its side is killed.)*

`orion:tags:enriched` (`CHANNEL_EVENTS_TAGGED`) has one producer -- `orion-meta-tags`' generic `handle_triage_event` branch (Juniper-observer-gated collapse-mirror events) -- and **two** consumers: `orion-rdf-writer` (materializes it into Fuseki's `orion:enrichment` graph as `Entity`/`Mention`/`Enrichment` triples) and `orion-sql-writer` (materializes it into Postgres `collapse_enrichment`, genuinely read by `orion-recall` and `orion-dream`). Every other content type that used to flow through this channel (`chat.history`, `social.turn.stored.v1` tag/entity data) already stopped publishing to it on 2026-07-18, when that data became Falkor-only.

As of today, the collapse-triage branch also has a working Falkor writer (`write_collapse_triage_tags_to_falkor`, PR #1271), a completed historical backfill (68/68 real Juniper-observed entries, PR #1273), and one confirmed real live event that landed correctly in Postgres, Fuseki, and Falkor simultaneously. Falkor now covers 100% of what this channel ever carried in terms of tag/entity graph presence -- but Postgres `collapse_enrichment` is a separately-shaped, separately-read materialization with no Falkor equivalent, so only the Fuseki side is pure redundancy (same "wrong-tool precedent" already established for `chat.history`, `cognition.trace`, and `metacog.trace`).

Kill only the Fuseki end: `orion-meta-tags` keeps publishing `tags.enriched` (unchanged); `orion-rdf-writer` stops subscribing to it and drops the now-dead `Enrichment` materialization code.

---

## Current architecture

```text
orion-meta-tags (handle_triage_event, generic branch)
  -> write_collapse_triage_tags_to_falkor()  [Falkor, orion_recall graph]  -- KEEP
  -> bus.publish(CHANNEL_EVENTS_TAGGED, tags.enriched)                    -- KEEP (sql-writer needs it)
       |
       +--> orion-sql-writer (subscribed to orion:tags:enriched)
       |      -> Postgres collapse_enrichment                            -- KEEP (real reads: orion-recall, orion-dream)
       |
       +--> orion-rdf-writer (subscribed to orion:tags:enriched)
              -> _build_enrichment_graph() -> Entity/Mention/Enrichment triples  -- KILL
                   |
                   v
              Fuseki orion:enrichment graph (read by orion-graph-compression's
              SPARQL EpisodicFederator, additively unioned with
              FalkorEpisodicFederator since today -- graph-compression keeps
              working either way, see below)
```

`RECALL_FALKOR_COLLAPSE_TRIAGE_ENABLED` currently frames Falkor as "additive, not a swap" (PR #1271's own docstring/README) because Fuseki was still one durable path at the time it was written. That framing is about the Falkor-vs-Fuseki write for tag/entity *graph* presence specifically -- it says nothing about Postgres `collapse_enrichment`, which is unaffected by this spec either way and remains the durable store `orion-recall`/`orion-dream` actually read from.

---

## Verified live before this spec (not assumed)

*(As with the Arsonist summary above, the producer/consumer audit below reflected the original wrong premise; see "Correction found during implementation" for the corrected consumer count. The rest of this section's findings still hold.)*

- **Producer audit**: grepped `orion-meta-tags/app/main.py` -- `CHANNEL_EVENTS_TAGGED` is published from exactly one call site, the generic branch. The chat/social branch's own bus publish was already removed 2026-07-18.
- **Consumer audit (corrected)**: `orion-rdf-writer`'s live subscribe list (`get_all_subscribe_channels()`) still includes `orion:tags:enriched`; `_build_enrichment_graph` is still the only handler for `tags.enriched` envelopes there. `orion-sql-writer`'s subscribe list also includes it (`sql_writer_subscribe_channels`), materializing to `collapse_enrichment` -- 76 live rows, latest timestamp same-day, genuinely read by `orion-recall`/`orion-dream`. This second consumer is why the kill is now `orion-rdf-writer`-only.
- **No other reader depends on `orion:enrichment`'s (Fuseki graph) continued growth**: `orion-graph-compression`'s `EpisodicFederator` (SPARQL) reads it, but is already additively unioned with `FalkorEpisodicFederator` (merged today, PR #1265) -- killing new writes to `orion:enrichment` does not remove graph-compression's ability to cluster this content, it just stops the SPARQL side from growing (the Falkor side keeps growing). `orion-cortex-orch`'s `concept_profile_config.py` (`RECALL_RDF_ENDPOINT_URL` alias) was checked and confirmed to target a *different* graph (`http://conjourney.net/graph/spark/concept-profile`, Spark ConceptProfile -- unrelated to chat/collapse tag enrichment).
- **End-to-end live proof**: a real Juniper-observed collapse-mirror event sent today (`collapse_11e07eaf1fbd478687a255a8f1e30a28`) landed correctly in Postgres (`collapse_mirror` raw store AND `collapse_enrichment` derived table -- both untouched by this spec), Fuseki (`orion:enrichment`, about to be killed), and Falkor (`orion_recall`, `CollapseEvent` node with real extracted entities `orion`/`falkordb`) simultaneously. Falkor's copy is not a shadow -- it's the same content, verified queryable.
- **Historical backfill complete**: 68/68 real Juniper-observed `collapse_mirror` rows (2025-09-24 through 2026-07-08) already migrated to Falkor (PR #1273) -- killing the Fuseki write does not lose any historical tag/entity-graph data, since Falkor already has full parity plus the live event on top. (Postgres `collapse_enrichment` has its own, separate, unaffected history.)

---

## Proposed changes (no schema/API changes -- this is a pure removal, scoped to `orion-rdf-writer` only)

### `services/orion-meta-tags`

**No changes.** The publish to `orion:tags:enriched` stays exactly as-is -- `orion-sql-writer` depends on it for Postgres `collapse_enrichment`, which `orion-recall` and `orion-dream` genuinely read. (`RECALL_FALKOR_COLLAPSE_TRIAGE_ENABLED`'s checked-in default is still `false` even though the live `.env` already has it flipped `true` from an earlier verification step in this same overall effort -- a real, pre-existing parity gap, but a separate, smaller housekeeping item from this spec's actual scope. Not fixed here to keep this PR's diff matched to what was actually verified and asked for; flagged as a follow-up in the PR report.)

### `services/orion-rdf-writer`

- `app/settings.py`: remove `CHANNEL_EVENTS_TAGGED` from `get_all_subscribe_channels()` and from the field list; add an explanatory comment (matching the existing dead-channel comment block already in this file for `chat.history`/`cognition.trace`/etc.) that explicitly says the publish itself is NOT dead -- `orion-sql-writer` still needs it.
- `app/rdf_builder.py`: remove the `tags.enriched`/`telemetry.meta_tags` dispatch branch and the `_build_enrichment_graph` handler function, plus now-dead helpers/imports it alone used (`_entity_uri`, `MetaTagsPayload`, `hashlib`, `RDFS`, `PROV`).
- `.env_example`, `docker-compose.yml`: remove the now-dangling `CHANNEL_EVENTS_TAGGED` key, with a comment explaining the publish survives elsewhere.
- `orion/bus/channels.yaml`: update `orion:tags:enriched`'s `consumer_services` to drop `orion-rdf-writer`, keeping `orion-sql-writer`.
- `README.md`: remove the stale channel-table row, replace with an inline note explaining what changed and why the publish itself is untouched.
- Tests (`tests/test_dead_channel_kill.py`): add a channel-not-subscribed regression test matching the file's existing pattern; replace the now-incorrect `test_chat_tagging_enrichment_falls_through_to_generic_event_uri` (which asserted the old dispatch branch still built a real graph) with a quiet-no-op test, since there is no more enrichment-specific handling left to "fall through" from.

---

## Files likely to touch

- `services/orion-rdf-writer/app/settings.py`, `rdf_builder.py`, `.env_example`, `docker-compose.yml`, `README.md`, `tests/test_dead_channel_kill.py`, `tests/test_service_rdf_store_integration.py`
- `orion/bus/channels.yaml`

---

## Non-goals

- **Not touching `services/orion-meta-tags` at all** -- corrected from the original draft, which proposed removing its bus publish. See "Correction found during implementation" above.
- Not touching `orion-sql-writer`'s subscription or `collapse_enrichment` materialization -- that is the real, live-read persistence path this correction exists to protect.
- Not touching `orion:rdf:enqueue`, `orion:rdf-collapse:enqueue`, `orion:collapse:intake`, `orion:chat:social:stored`, or `orion:core:events` -- `orion-rdf-writer` keeps all of these; it is not being decommissioned as a service, only this one channel/handler.
- Not touching the raw `collapse_mirror` Postgres table or the `orion:collapse:sql-write` channel that feeds it -- that remains the durable source-of-truth for the full CollapseMirrorEntryV2 payload, unrelated to the derived tag/entity enrichment this spec kills.
- Not decommissioning Fuseki itself, or purging the existing (now-frozen) `orion:enrichment` graph content -- matches the established precedent (Fuseki stays up read-only until Phase 10, a separate, later, explicit decision).
- Not touching `orion-graph-compression`'s SPARQL `EpisodicFederator` -- it keeps running, just stops seeing fresh `orion:enrichment` content going forward (the Falkor side already covers that gap additively).
- Not fixing `orion-meta-tags`' `RECALL_FALKOR_COLLAPSE_TRIAGE_ENABLED` checked-in-default-vs-live-.env parity gap noted above -- flagged as a follow-up, not blocking this PR.

---

## Acceptance checks

- [x] `orion-meta-tags` is unchanged and still publishes to `orion:tags:enriched`.
- [x] `orion-sql-writer`'s subscription and `collapse_enrichment` writes are unaffected.
- [x] `orion-rdf-writer` no longer subscribes to `orion:tags:enriched`; dispatching `tags.enriched`/`telemetry.meta_tags` is a quiet no-op if either ever arrives (e.g. a stale dead-letter replay).
- [x] `channels.yaml`'s `orion:tags:enriched` entry lists only `orion-sql-writer` as a consumer.
- [x] All existing + updated tests pass in `orion-rdf-writer`.
- [ ] Live verification post-deploy: confirm zero `tags.enriched`/`orion:enrichment` traffic in `orion-rdf-writer`'s logs, and confirm `orion-sql-writer` is still writing to `collapse_enrichment` normally (row count keeps advancing).

---

## Recommended next patch

Implement exactly the corrected changes above, in this same PR: consumer kill (rdf-writer only), contract fix (channels.yaml), tests, docs, live verification, PR report -- explicitly noting the scope correction and why the producer side was left alone.
