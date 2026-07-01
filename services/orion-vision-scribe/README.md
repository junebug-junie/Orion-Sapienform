# Orion Vision Scribe
Records events.

## RDF write path

The RDF write side of `_write_to_sinks` builds real triples via `rdflib`
(`Graph` + `.serialize(format="nt")`) instead of hand-rolling a list of
tuples. The resulting N-Triples string is sent as `triples` on
`orion.schemas.rdf.RdfWriteRequest` (along with required `id`/`source`
fields), matching the shape the RDF writer's `_handle_write_request`
consumer expects.
