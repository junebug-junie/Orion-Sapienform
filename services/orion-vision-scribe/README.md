# Orion Vision Scribe
Records events.

## RDF write path: removed 2026-07-23

`_write_to_sinks` used to dual-write every event to Postgres (`vision_events`)
and Fuseki (`orion:vision` graph, via `orion.schemas.rdf.RdfWriteRequest`).
Live-verified pure redundancy before removal: Postgres `vision_events` gets
the same event at the same timestamp with a strictly richer schema
(`confidence`/`salience`/`evidence_refs`/`tags` as real structured columns,
not flat RDF literals), and nothing in the codebase ever reads the Fuseki
`orion:vision` graph back. Same pattern as every other Fuseki write this
migration killed outright rather than migrated (see
`docs/superpowers/specs/2026-07-17-recall-rdf-writer-falkor-cutover-phase2-spec.md`'s
PR #1155 precedent). SQL is now the sole sink.

## SQL write path

The SQL write side of `_write_to_sinks` publishes the `VisionEventBundleItem`
event directly (kind `vision.event.v1`) to `orion:vision:events:sql-write`,
consumed by `orion-sql-writer`'s `VisionEventSQL` model, which persists into
the `vision_events` table. This replaces the old, non-functional
`SqlWriteRequest`/`orion:collapse:sql-write` path, which used a schema shape
(`table`/`data`) that the real `orion.schemas.sql.schemas.SqlWriteRequest`
never accepted, and a channel that `orion-sql-writer` has no route for.
