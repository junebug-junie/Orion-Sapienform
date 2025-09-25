# Orion GraphDB Service

This service integrates **GraphDB** into the Orion-Sapienform mesh for storing, reasoning, and querying collapse events in RDF.

---

## 🚀 Starting GraphDB

GraphDB runs inside Docker alongside Orion services.

From the `orion-gdb-client/` folder:

```bash
make up
```

Verify GraphDB is available at [http://localhost:7200](http://localhost:7200).

List repositories:

```bash
curl http://localhost:7200/rest/repositories
```

You should see the `collapse` repository.

---

## 📥 One-off Backfill (Manual Ingest)

You can manually inject historical collapse events into GraphDB.

### Option A: From a Turtle file

1. Place your Turtle file in `services/orion-gdb-client/oneoff/`
   Example: `collapse_entry.ttl`

2. POST it to GraphDB:

```bash
curl -X POST "http://localhost:7200/repositories/collapse/statements" \
  -H "Content-Type: text/turtle" \
  --data-binary @/mnt/services/Orion-Sapienform/services/orion-gdb-client/oneoff/collapse_entry.ttl
```

Expected result: `204 No Content`.

### Option B: Inline TTL (fast test)

```bash
curl -X POST "http://localhost:7200/repositories/collapse/statements" \
  -H "Content-Type: text/turtle" \
  --data-binary '
    @prefix cm: <http://orion.ai/collapse#> .

    cm:collapse_aeb1dfa237134be2956e51d99ec9759e
        cm:observer "Juniper" ;
        cm:trigger "testing end-to end Orion-Sapienform services" .
'
```

Expected result: `204 No Content`.

---

## 🔍 Verifying Data

### Option A: Workbench UI

1. Go to [http://localhost:7200](http://localhost:7200)
2. Select the `collapse` repository.
3. Open the **SPARQL** tab.
4. Run:

```sparql
PREFIX cm: <http://orion.ai/collapse#>

SELECT ?p ?o
WHERE {
  cm:collapse_aeb1dfa237134be2956e51d99ec9759e ?p ?o .
}
```

This lists all properties of the collapse entry.

### Option B: CLI SPARQL Query

```bash
curl -X POST http://localhost:7200/repositories/collapse \
  -H "Content-Type: application/sparql-query" \
  -d 'PREFIX cm: <http://orion.ai/collapse#>
      SELECT ?p ?o
      WHERE {
        cm:collapse_aeb1dfa237134be2956e51d99ec9759e ?p ?o .
      }'
```

### Option C: Dump All Triples

```sparql
SELECT ?s ?p ?o
WHERE { ?s ?p ?o }
LIMIT 20
```

This confirms if anything at all is in the repository.

---

## 🔒 SSH Tunnel from Carbon

If GraphDB runs on another host (e.g., Janus), tunnel it to Carbon:

```bash
ssh -L 7200:localhost:7200 janus@<JANUS_IP>
```

Now access [http://localhost:7200](http://localhost:7200) locally.

Run in background:

```bash
ssh -fN -L 7200:localhost:7200 janus@<JANUS_IP>
```

Close tunnel:

```bash
ps aux | grep ssh | grep 7200
kill <PID>
```

---

## 🧠 V2: Schema & Inference (Planned)

Future work will extend GraphDB beyond raw triples to support **structured reasoning**:

* **Schema (`graphdb_schema.ttl`)**: defines `cj:MemoryEvent`, `cj:thread_id`, `cj:generatedBy`, etc., so all collapse events use a shared ontology.
* **Rules (`graphdb_rules.txt`)**: enable inference such as causal lineage (A generated B, B generated C ⇒ A generated C), narrative threading (`sameThreadAs`), and introspection labeling.
* **RDFBuilder**: a Python helper to convert Collapse Mirror JSON into consistent triples using the schema.
* **TTL Logging & Audit**: write every event as `.ttl` snapshots, then diff & audit them for missing references or broken threads.

This layer will allow Orion to:

* Infer hidden links between collapse events.
* Align narrative sequences across memory threads.
* Label events as *introspective* vs *external*.
* Maintain an auditable RDF history for emergent reasoning.

---

## ✅ Quick Checklist

* [x] GraphDB running in Docker
* [x] `collapse` repository exists
* [x] TTL triples ingested (`204 No Content`)
* [x] SPARQL queries return results
* [x] Tunnel active if accessing from Carbon
* [ ] V2 schema & rules integrated for reasoning

---
