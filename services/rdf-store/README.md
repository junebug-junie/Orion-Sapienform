# RDF store (vendor-neutral)

This directory holds **operator-facing** layout and compose fragments for Orion’s RDF persistence. The canonical write path remains `services/orion-rdf-writer`: it builds triples from bus envelopes and posts them to the configured backend.

## What is *not* here

- No automatic migration of existing GraphDB graphs to another store.
- No changes to recall/substrate read paths or concept induction (see service READMEs for those contracts).

## Layout

| Path | Purpose |
|------|---------|
| `fuseki/` | Apache Jena Fuseki on `app-net`, durable host bind mount for TDB. |
| `graphdb/` | Pointer to the primary GraphDB stack and persistence conventions. |

## Performance intent

The writer uses an **async queue** (when enabled) so bus handlers are not blocked on remote HTTP. Tune `RDF_WRITE_*` and HTTP pool limits in `services/orion-rdf-writer/.env_example` rather than raising bus-side timeouts for slow stores.
