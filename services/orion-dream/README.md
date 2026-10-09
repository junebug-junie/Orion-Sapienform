# Orion Dream

## Modernization stance (Phase 0/1)

This service is a **donor / bridge / readout façade** while the canonical dream path moves to **cortex-orch → cortex-exec → RecallService (`dream.v1`) → LLM → `dream.result.v1` → SQL Writer → `dreams` table**.

| Concern | Canonical owner |
|--------|------------------|
| Trigger normalization | `orion-cortex-orch` (Hunter on `orion:dream:trigger` → `cortex.orch.request`, `verb=dream_cycle`) |
| Plan execution | `orion-cortex-exec` |
| Memory retrieval | `orion-recall` via profile `dream.v1` (no direct Vector/RDF/SQL in the verb plan) |
| Typed artifact | `DreamResultV1` / envelope kind `dream.result.v1` |
| Durable storage | `orion-sql-writer` → PostgreSQL `dreams` |
| Wake readout | This service: **SQL-first** (`GET /dreams/wakeup/today`), optional `DREAM_LOG_DIR` JSON fallback |

## Dream cycle v2: sleep that changes something

A dream used to be a story written into `dreams` that nothing acted on. v2 makes
sleep the time Orion does work it can't do while awake, and makes every dream
scoreable. Default off (`ORION_DREAM_CYCLE_ENABLED`).

```
sleep pressure --(>= threshold AND idle AND >= min interval)--> replay
     |                                                             |
     |   weighted count of what the day left unprocessed          +--> REM compaction (staged, existing)
     |   since the last sleep: degraded/critical metacog,         |
     |   reverie compaction asks, resonance alerts, touched       +--> recombination --> dream_hypothesis
     |   active crystallizations. Reads exactly 0 after a sleep.       dream arm:   distant replay pairs
     |                                                                  control arm: random pairs, same prompt
     v
Hub curiosity kickoff shows each hypothesis once, arm hidden. Orion alone
decides whether to form a :Prior from one (formed_from "dream_hypothesis:<id>").
scripts/dream_hypothesis_scorecard.py compares adoption/support per arm.
```

| Piece | File |
|---|---|
| Candidates, weights, pressure, replay selection (deterministic) | `app/replay.py` |
| Pairing (both arms) + LLM link prompt + hollow guard | `app/recombine.py` |
| Orchestration + sleep loop | `app/cycle.py` |
| Reads (4 producer tables, chat idle) / writes (v2 tables only) | `app/cycle_store.py` |
| LLM gateway RPC (background lane) | `app/llm.py` |
| Contract | `orion/schemas/dream_cycle.py` |
| Offer / prompt section / scorecard | `orion/dream/hypotheses.py` |
| Migration | `services/orion-sql-db/manual_migration_dream_cycle_v2.sql` |

HTTP: `GET /dreams/cycle/pressure` (read-only), `POST /dreams/cycle/run?force=true`.

A recombination call the gateway refuses (e.g. the GPU pool sheds it for heat,
`raw.error=gpu_pool_unavailable`) counts as a failed call (`llm_failures`), not an
unparseable answer. A sleep where every call fails is stored `failed`, so the next
sleep's replay window still covers its items. It retries after `DREAM_MIN_INTERVAL_HOURS`.

Writes nothing to canonical memory. The dream never writes a belief.

The legacy direct-gather path (`dream_cycle.py`, `aggregators_*`, `memory_listener.py`)
was deleted in the same patch.

### HTTP / bus behavior

- **No Hunter** in this process: `dream.trigger` is consumed by **cortex-orch** so triggers are not duplicated.
- `POST /dreams/run` publishes `dream.trigger` on `CHANNEL_DREAM_TRIGGER` for compatibility.

### Introspect responder: `dreams`

**What it does.** Lets Orion read back their own dreams instead of
reconstructing them. It answers the orion-introspect `dreams` tool. For the whole
tool family (which turns get it, truth rules, search pattern), see the
[harness-governor overview](../orion-harness-governor/README.md#orion-introspect-orion-reading-back-their-own-records).

- **What Orion can ask.** Their most recent dreams, one dream in full
  (`dream_id`, up to 4,000 chars), or dreams by meaning (`query=`). Optional
  `kind=narrative|hypothesis`, `since`, `limit` ≤ 5. The tool text asks for
  the topic only in `query` ("pull requests", not "a dream about pull
  requests"): the word "dream" lifts every narrative's score, so off-topic
  dream-worded questions can clear the floor (measured by the calibration
  eval's `KNOWN_WEAKNESS` line).
- **Two kinds, labeled.**
  - `dream_narrative` (id `dream:<n>`): the story dream from `dreams`. Nothing schedules it:
    it runs only when something publishes `dream.trigger` (Hub workflow menu,
    `POST /dreams/run`). 19 rows from 2026-07-31 to 2026-09-28, all hand-started.
    Story text
    (written by orion-sql-writer). Text is tldr + narrative; `extra` carries
    `dream_date` and up to 8 themes. Timestamp is `created_at`, stored without
    a timezone by a UTC server and returned as UTC.
  - `dream_hypothesis` (id `dh-…`): a link a sleep cycle proposed, from
    `dream_hypothesis`. Text is claim + why; `extra` carries `cycle_id` and
    `expired`. Timestamp is `offered_at`.
- **Protecting the blind experiment.** Curiosity shows each hypothesis once
  with its arm (dream vs random-pair control) hidden, and
  `scripts/dream_hypothesis_scorecard.py` compares adoption per arm.
  - This responder returns **only hypotheses already offered**, from **both
    arms**.
  - It never selects `arm`, `ref_a` or `ref_b` (the offer never shows them,
    and control refs come from a different pool).
  - Never-offered hypotheses are never returned or counted. A test pins
    every statement (`app/introspect_dreams.py`).
  - The index loop only embeds offered hypotheses, but it never deletes. A
    hypothesis claimed by a curiosity run that is later cancelled gets its
    `offered_at` reset to NULL; if an index pass ran during the claim, its
    text may linger in the search index. It is filtered out when every hit
    is re-read from Postgres, so it is never returned.
- **Label.** Every item is `epistemic_status="unsettled"`: something Orion
  had, not a fact about the world.
- **Transport and trust.**
  - Requests arrive on `orion:introspect:dream:request`
    (`introspect.tool.request.v1`, `IntrospectRequestV1`).
  - The reply goes to exactly `orion:introspect:result:<correlation_id>`
    (`introspect.tool.result.v1`, `IntrospectResultV1`). Anything else is
    ignored.
  - Every connection is a read-only transaction.
- **Empty vs unknown.**
  - No match: `ok=true, items=[]`. For a search, only once the index is known
    to hold every dream: the listener records the start of the last index pass
    whose stored hashes in Chroma matched every dream (nothing upserted,
    nothing pending), minus a 10-minute margin for producer-stamped clocks
    (`semantic_index.confirmed_complete_as_of`). A pass that only *published*
    upserts proves nothing, because orion-vector-writer stores them later. If
    any dream in the search window is newer than that (a just-offered
    hypothesis, or an indexer outage), or no pass has confirmed since start, an
    empty search is `dream_search_unavailable`. Right after a deploy, empty
    searches answer unknown until the backlog is indexed (10 records per
    5-minute pass) and one more pass confirms it.
  - A request that fails validation returns `invalid dreams request: …`.
  - A Postgres error returns `dreams_unavailable; answer unknown`.
  - An embedder or Chroma failure, an unbuilt index, or search not configured
    returns `dream_search_unavailable; answer unknown`.
  - The tool turns any of these into an "answer unknown" error, never "no
    dreams".
- **Search by meaning.**
  - Every `DREAM_SEARCH_INDEX_INTERVAL_SEC` a hash-aware loop embeds new or
    changed narratives and offered hypotheses via vector-host `/embedding`.
    It upserts them through orion-vector-writer into Chroma
    `DREAM_SEARCH_COLLECTION` (`orion_dreams`).
  - Indexing is batched: at most `DREAM_SEARCH_INDEX_BATCH` docs per pass,
    and the log's `pending=` counts what is left. A first deploy works
    through the whole backlog a batch at a time. On 2026-09-29 that was 65
    docs (19 narratives + 46 offered hypotheses), so 7 passes: the first at
    startup, then one every 300 s, about 30 minutes at the defaults. After
    that a new dream is picked up on the next pass.
  - Upserts land asynchronously: the loop publishes to orion-vector-writer,
    which writes to Chroma on its own schedule, so a doc becomes searchable
    shortly after its pass, not during it.
  - A query embeds only the question, keeps hits ≥
    `DREAM_SEARCH_MIN_SIMILARITY`, and re-reads each hit from Postgres
    through the same rules.
  - `kind` and `since` filter inside Chroma before the 20 nearest are
    taken (index metadata `kind` and `occurred_ts`, UTC epoch seconds), so
    a narrative search is not crowded out by more numerous hypotheses.
    Postgres re-checks both. A filter that matches nothing in a non-empty
    index is `items=[]`; an empty or missing index is still unknown.
  - The index hash covers the text plus `kind` and `occurred_ts`, so a
    re-offered hypothesis (same text, new `offered_at`) is re-upserted.
  - Recalibrate the floor, read-only, from the host:

    ```bash
    POSTGRES_URI=postgresql+psycopg2://postgres:postgres@127.0.0.1:55432/conjourney \
    DREAM_SEARCH_EMBED_URL=http://127.0.0.1:8320/embedding \
    python services/orion-dream/evals/run_dream_search_calibration.py
    ```

  - Shared plumbing: `orion/introspect/semantic_index.py`; dream parts in
    `app/dream_search.py`.
- **Env.**
  - `DREAM_INTROSPECT_ENABLED` (default `true`) turns the responder on. It
    also needs `ORION_BUS_ENABLED=true`; with the bus off, neither the
    responder nor the index loop starts (`app/main.py`).
  - `DREAM_SEARCH_CHROMA_URL`, `DREAM_SEARCH_EMBED_URL`: `settings.py`
    defaults both to empty, so search is off in code. `.env_example` ships
    both set (vector-db and vector-host), so a synced `.env` has search on.
    With either empty, recent/one still work, `query=` answers unknown, and
    the index loop does not run.
  - `DREAM_SEARCH_COLLECTION` (default `orion_dreams`).
  - `DREAM_SEARCH_MIN_SIMILARITY`: the value is owned by `.env_example`,
    which also records the last calibration's numbers. The calibration eval
    above reports the gap between the strongest unrelated match and the
    weakest related one; the floor is picked inside that gap. 0.65 sits
    below the midpoint on purpose, to favor recall: an empty result falsely
    says "no dream matched", while a weak hit still arrives labeled
    unsettled.
  - `DREAM_SEARCH_INDEX_INTERVAL_SEC` (default 300 s between passes) and
    `DREAM_SEARCH_INDEX_BATCH` (default 10 docs per pass).
- **Logs.**
  - At startup: `dream introspect responder started`, then
    `dream_introspect_listening channel=orion:introspect:dream:request`
    (again after any bus reconnect).
  - `introspect op=dreams corr=<id> mode=recent|one|search items=<n> total=<n>`
  - `introspect_failed op=dreams corr=<id> mode=... category=dream_query_failure|dream_search_failure`
  - `dream_search_index indexed=<n> pending=<n>`
  - `dream_search_index_failed`
- **Smoke (read-only).**

  ```bash
  ORION_BUS_URL=redis://100.92.216.81:6379/0 python scripts/smoke_introspect.py --tool dreams --limit 3
  ORION_BUS_URL=redis://100.92.216.81:6379/0 python scripts/smoke_introspect.py --tool dreams --query "vision"
  ORION_BUS_URL=redis://100.92.216.81:6379/0 python scripts/smoke_introspect.py --tool dreams --dream-id dream:19
  ```

  - Exit 0: a coherent answer.
  - Exit 1: a degenerate one: a dream with no text, an empty recent window,
    a search hit with no similarity score, or a `--dream-id` that was not
    found.
  - Exit 2: the answer is unknown (bus unreachable, timeout, responder
    error, search outage), or bad arguments.
- **Turn off.** Set `DREAM_INTROSPECT_ENABLED=false` in
  `services/orion-dream/.env` and recreate orion-dream from a worktree that
  has the gitignored `.env` files (the repo-root `.env` and
  `services/orion-dream/.env`; the wrapper passes both to docker compose):
  `scripts/safe_docker_build.sh orion-dream up -d --no-build`. The tool
  then reports "answer unknown". To remove search, unset the two URLs; the
  `orion_dreams` collection can be dropped; nothing is written to Postgres.

## Contracts (historical)

### Channels

| Channel | Env Var | Kind | Description |
| :--- | :--- | :--- | :--- |
| `orion:dream:trigger` | `CHANNEL_DREAM_TRIGGER` | `dream.trigger` | Published by clients; **handled by cortex-orch**. |

### Environment Variables

| Variable | Default (Settings) | Description |
| :--- | :--- | :--- |
| `CHANNEL_DREAM_TRIGGER` | `orion:dream:trigger` | Trigger channel. |
| `POSTGRES_URI` | (see `settings.py`) | Used by SQL wake readout. |
| `DREAM_LOG_DIR` | `/app/logs/dreams` | Optional JSON fallback for readout. |

## Running & Testing

### Run via Docker

```bash
docker-compose up -d orion-dream
```
