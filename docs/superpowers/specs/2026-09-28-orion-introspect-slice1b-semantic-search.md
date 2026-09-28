# orion-introspect slice 1b: semantic search over reading results

Status: approved design (Juniper, 2026-09-28: "hit it"). Builds on slice 1
(`2026-09-28-orion-introspect-mcp-design.md`, PR #2381).

## Why

Slice 1 can only look a reading up by exact URL, by request id, or "most recent".
Orion rarely has either key in hand; they remember *what a reading was about*.
Without search the tool reduces to scrolling a five-item window. Juniper: "it
should be semantically searchable for this to be a usable tool for orion" -- and
explicitly not a regex/keyword swamp.

## Design (vector-only, embed on landing)

1. **Write rail -- embed once.** A small loop in the Hub's `ReadingListener`
   (every `HUB_READING_SEARCH_INDEX_INTERVAL_SEC`, default 300) reads every
   *verified* reading (done, has a tool-trace read of its own source -- the same
   gate slice 1 uses), builds its index text (title + learned text, clipped to
   1,800 chars), hashes it, and compares against the `content_hash` stored in
   Chroma. Missing or changed rows (a Stage 2 summary replacing Stage 1 text)
   are embedded through orion-vector-host's pure HTTP `/embedding` endpoint and
   published as `VectorUpsertV1` on `orion:vector:semantic:upsert`, which
   orion-vector-writer already consumes. At most `HUB_READING_SEARCH_INDEX_BATCH`
   (default 10) per pass. The same loop is the backfill and the self-heal: a
   dropped write is simply "still missing" next pass.
   - Lives in the listener, not the Stage 2 tick, because Stage 2's loop does not
     run at all when Stage 2 is disabled.
   - HTTP `/embedding` never persists (verified: `services/orion-vector-host/app/main.py:752`);
     the bus `orion:embedding:generate` path does, so it is not used.
2. **Read rail -- embed only the query.** `reading_results(query=...)` embeds the
   query once, asks Chroma for the nearest 20, converts distance to cosine
   similarity, keeps hits at or above `HUB_READING_SEARCH_MIN_SIMILARITY`, then
   re-reads those rows from Postgres and re-applies the verified-reading gate.
   Chroma is an index, never the record. Each item carries `extra.similarity`.
   `total_available` counts gated hits. `since` narrows query mode as it does
   recent mode.
3. **Unknown, never empty.** Embedder down, Chroma down, collection not built,
   malformed reply, or search not configured -> tool error "answer unknown".
   `items=[]` only when the search ran and nothing cleared the floor.

Collection: `orion_reading_results`, id = `seed_id`, metadata = `content_hash`,
`url`, `request_id`, `occurred_at` (plus `embedding_model`/`embedding_dim` added
by the writer). Created by vector-writer's `get_or_create_collection`, so its
space is Chroma's default `l2` (squared); for bge's unit vectors
`cos = 1 - d/2`. A collection created with `hnsw:space: cosine|ip` uses `1 - d`.

Lesson carried from recall (`2026-07-21-memory-cards-self-authored-substrate-spec.md`):
the outage came from embedding every candidate per call. Here nothing but the
query is embedded at ask time.

## Metric quality gate: `similarity` and the relevance floor

1. **Provenance.** `similarity()` in `orion/world_pulse_read/search.py`, from
   Chroma 0.4.24 `/query` distances over `BAAI/bge-large-en-v1.5` 1024-dim
   L2-normalized vectors (vector-host `/embedding`, norm measured 1.0000).
2. **Independence.** Not combined with any other metric; it is the only ranking
   signal. The source-read gate is a hard filter, not a score.
3. **Theory anchor.** Cosine similarity of sentence embeddings trained for
   retrieval (bge is contrastively trained so relevant passage/query pairs score
   higher than irrelevant ones). Absolute values are not probabilities; the
   floor must be calibrated per corpus.
4. **Live sanity, including the rest point.** Probe 2026-09-28 against the 13
   verified readings (nearly all GPU/Nvidia coverage): related questions top out
   at 0.66-0.79 (graphics cards 0.742, AI chip export 0.704, blocking fetchers
   0.661); unrelated questions top out at 0.47-0.54 (cookie recipe 0.520, cat vet
   0.473, medieval poetry 0.486, Venezuela 0.536). **The rest point is not 0:**
   unrelated text still scores about 0.45-0.54 because bge's embedding space is
   anisotropic. A floor of 0.60 separates the two groups with about 0.06 of room
   on each side. The bge query instruction prefix gave mixed results and is not used.
   Caveat: a monotopic corpus. Recalibration is one command:
   `python services/orion-hub/evals/run_reading_search_calibration.py`.
5. **Existing mechanism.** No reading index exists (Chroma holds `orion_main_store`,
   `orion_chat`, `doc_semantic_drift`; no pgvector). `orion-recall` removed its
   per-call embedding cosine for cost, not quality. The crystallizer's
   embed-then-upsert pattern (`orion/memory/crystallization/chroma_publish.py`)
   is reused in shape; its chromadb-client query helper is not (the Hub has no
   chromadb dependency, and it returns `[]` on failure, which would read as "nothing").
6. **Reversibility.** One env key empties search (`HUB_READING_SEARCH_CHROMA_URL=`);
   the collection is disposable and rebuilt by the index loop; no Postgres
   schema change.

## Contract changes

- `ReadingResultArguments.query` and `ReadingToolRequestV1.query` (1-500 chars,
  stripped, not with `request_id`/`url`; omitted from the wire when null so a
  pre-1b Hub still accepts every non-query payload).
- `orion:vector:semantic:upsert` gains producer `orion-hub`.
- Hub env: `HUB_READING_SEARCH_CHROMA_URL`, `HUB_READING_SEARCH_EMBED_URL`,
  `HUB_READING_SEARCH_COLLECTION`, `HUB_READING_SEARCH_MIN_SIMILARITY`,
  `HUB_READING_SEARCH_INDEX_INTERVAL_SEC`, `HUB_READING_SEARCH_INDEX_BATCH`.

Known limit: `since` filters the 20 nearest hits after ranking, not the corpus
before it. With 13 verified readings the 20 nearest are the whole corpus; past
that, a narrow `since` window can come back empty while an in-window reading
exists further down. Pass `since` into Chroma's `where` (using `occurred_at`
metadata stored as epoch seconds) when the corpus grows past a few hundred.

## Non-goals

Full-text or hybrid (BM25) fusion; LLM reranking; other introspect domains;
deleting index entries for readings that later fail the gate (the read-side
re-gate already hides them).

## Acceptance checks

- Unit: contract validation; similarity conversion for l2/cosine; floor filter;
  Postgres re-gate drops hollow and unknown ids; every failure path raises
  `SearchUnavailableError`; index pass upserts only missing/changed verified rows
  and respects the batch.
- Real Postgres: index + search over done/hollow/queued rows.
- Live: after deploy, the Hub log shows `reading_search_index` passes with
  `pending` reaching 0; Chroma `orion_reading_results` count equals the verified
  count; the calibration eval passes; `scripts/smoke_introspect.py --query` returns a gated hit with `similarity`.
