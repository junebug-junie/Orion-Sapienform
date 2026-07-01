# Orion Vision Scribe
Records events.

## RDF write path

The RDF write side of `_write_to_sinks` builds real triples via `rdflib`
(`Graph` + `.serialize(format="nt")`) instead of hand-rolling a list of
tuples. The resulting N-Triples string is sent as `triples` on
`orion.schemas.rdf.RdfWriteRequest` (along with required `id`/`source`
fields), matching the shape the RDF writer's `_handle_write_request`
consumer expects.

## SQL write path

The SQL write side of `_write_to_sinks` publishes the `VisionEventBundleItem`
event directly (kind `vision.event.v1`) to `orion:vision:events:sql-write`,
consumed by `orion-sql-writer`'s `VisionEventSQL` model, which persists into
the `vision_events` table. This replaces the old, non-functional
`SqlWriteRequest`/`orion:collapse:sql-write` path, which used a schema shape
(`table`/`data`) that the real `orion.schemas.sql.schemas.SqlWriteRequest`
never accepted, and a channel that `orion-sql-writer` has no route for.
