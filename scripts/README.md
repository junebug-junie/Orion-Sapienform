# Smoke Tests

Run these scripts to validate bus publishing and GraphDB ingestion.

## Chat History → RDF
```bash
python scripts/smoke_chat_to_rdf.py --redis redis://localhost:6379/0
```
Expected output:
```
PASS [...]
```

## Tags Enrichment → RDF
```bash
python scripts/smoke_tags_to_rdf.py --redis redis://localhost:6379/0
```
Expected output:
```
PASS [...]
```

## Juniper Collapse Fanout → Triage + SQL
```bash
python scripts/smoke_juniper_collapse_fanout.py --redis redis://localhost:6379/0
```
Expected output:
```
PASS triage=ok sql-write=ok
```
