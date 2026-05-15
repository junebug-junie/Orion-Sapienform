# GraphDB (primary RDF store)

Orion’s default RDF writer backend targets **GraphDB** via the repository statements API.

- **Stack & ops:** see `services/graphdb/README.md` for the main GraphDB service, ports, and persistence layout.
- **Path parity:** keep GraphDB and Fuseki data roots on durable volumes (e.g. under `/mnt/storage-lukewarm/rdf-store/`) consistent with your backup policy.

This folder is a **neutral index** only; implementation lives under `services/graphdb/` and `services/orion-rdf-writer/`.
