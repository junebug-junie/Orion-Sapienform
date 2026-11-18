# orion-introspect slice 2: dreams (with search by meaning)

Parent design: `docs/superpowers/specs/2026-09-28-orion-introspect-mcp-design.md`.
Search pattern: `docs/superpowers/specs/2026-09-28-orion-introspect-slice1b-semantic-search.md`.

## Arsonist summary

Orion can't look at their own dreams. Asked "what did you dream last night?"
they either guess or say nothing. This slice adds a `dreams` tool to the
orion-introspect MCP server. `orion-dream` answers over the bus, reading
Postgres directly. It returns recent dreams, dreams found by meaning, or one
dream in full.

Two record kinds come back, labeled:

- **narrative dreams**: the nightly story, one row per night in `dreams`.
- **sleep-cycle hypotheses**: speculative links from `dream_hypothesis`.
  Only ones Orion has already been offered are returned, from both arms, and
  the arm is never exposed.

Both kinds are `epistemic_status="unsettled"`: a dream is something Orion had,
not a fact about the world.

The search plumbing (embed, nearest-match, index writes) moves out of
reading search into one shared module. Dreams, and later reveries and
curiosity, use it instead of copying it.

The truth eval ships as the next PR (§ Follow-up PR), not this one.

## Current architecture

- **Narrative dreams.** cortex-orch → cortex-exec (`dream_cycle` verb) →
  `dream.result.v1` → orion-sql-writer → `dreams`. 19 rows live, newest
  2026-09-28. `created_at` is `timestamp without time zone`, written by
  `now()` on a UTC server (`SHOW timezone` = `Etc/UTC`), so it is UTC.
  `orion-dream` only reads it (`app/sql_readout.py`).
- **Sleep cycles.** `orion-dream`'s own v2 loop (`app/cycle.py`) writes
  `dream_cycle`, `dream_replay_item` and `dream_hypothesis`. It runs about
  every 6 h; the latest was 2026-09-28 22:47 UTC.
  - 49 hypotheses total: 36 dream-arm (25 offered) and 13 control-arm
    (9 offered).
  - A 72 h TTL (`expires_at`) stops a hypothesis being offered; rows are
    never deleted.
- **The blind experiment** (`orion/dream/hypotheses.py`).
  - Hub's curiosity kickoff claims never-offered, unexpired hypotheses from
    both arms, shuffled and sorted by id so position carries no arm signal.
  - It shows `claim` + `why` once and stamps `offered_at`.
  - `scripts/dream_hypothesis_scorecard.py` compares adoption per arm.
  - A failed run releases its offers (`offered_at` back to NULL).
- **orion-dream runtime.**
  - On the `app-net` bridge network; reaches Chroma at
    `http://orion-athena-vector-db:8000` and the vector host at
    `http://orion-athena-vector-host:8320`. Both were verified from inside
    the container.
  - Sync SQLAlchemy + psycopg2, httpx present, `orion/` copied into the image.
  - `ORION_BUS_URL=redis://100.92.216.81:6379/0` from the root `.env`.
  - No bus subscriber today (dream triggers are consumed by cortex-orch).
- **Introspect today.**
  - `IntrospectOperation = Literal["reading_result"]`, one tool
    (`reading_results`) in `orion/introspect/tools.py`.
  - `IntrospectResultV1` / `IntrospectItemV1` are registered;
    `IntrospectRequestV1` does not exist yet.
  - The brief is in `orion/introspect/brief.py`.
  - The server is attached only when `HARNESS_FCC_INTROSPECT_ENABLED` is on
    (it is, live) and the turn has a reading binding and is not reading-only.
- **Reading search.** `orion/world_pulse_read/search.py` holds generic
  plumbing (`embed`, `_collection`, `nearest`, `similarity`,
  `content_hash`, `_stored_hashes`, the upsert publish) mixed with
  reading-specific SQL and gating.

## Decisions (from chat, 2026-09-28)

1. **Both kinds, labeled.** Narrative dreams and sleep-cycle hypotheses.
2. **Protect the blind experiment.**
   - Hypotheses are returned only when `offered_at IS NOT NULL`, from both
     arms.
   - No `arm`, no `ref_a`/`ref_b` (the offer never shows them, and control
     refs come from a different pool).
   - Item timestamp is `offered_at`.
   - Never-offered hypotheses are never returned or counted. Only offered
     ones are indexed, but the index never deletes: a hypothesis released
     by a cancelled curiosity run (`offered_at` reset to NULL) may linger
     in the index and is filtered out on the Postgres re-read.
3. **Search by meaning from the start,** same approach as readings.
4. **Extract, don't copy.** The shared search plumbing moves to
   `orion/introspect/semantic_index.py`; reading search keeps its behavior
   and its public names.
5. **Truth eval is its own PR,** right after this one. It is a hybrid: code
   checks what code can check, and an LLM judge (proven on hand-labeled
   examples with planted fabrications) checks claims against the tool results.

## Proposed schema / API changes

### `orion/schemas/introspect.py`

```python
IntrospectBusOperation = Literal["dreams"]
IntrospectOperation = Literal["reading_result", "dreams"]
FULL_TEXT_CAP = 4000          # single-record fetch; 1 item << 12,000-char MCP budget
THEME_CAP = 8                 # themes per narrative item
DREAM_ID_PATTERN = r"^(dream:[0-9]{1,12}|dh-[0-9a-f]{6,32})$"

class IntrospectRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation: IntrospectBusOperation
    binding: IntrospectToolBindingV1
    args: dict[str, Any]              # owner validates per operation

class DreamsArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str | None = Field(default=None, min_length=1, max_length=QUERY_CAP)
    dream_id: str | None = Field(default=None, pattern=DREAM_ID_PATTERN)
    kind: Literal["narrative", "hypothesis"] | None = None
    limit: int = Field(default=DEFAULT_LIMIT, ge=1, le=MAX_ITEMS)
    since: datetime | None = None
    # dream_id is exclusive with query/kind/since; since must be tz-aware;
    # a blank query is an error (normalize_query), not "recent".
```

### Items

| Field | Narrative | Hypothesis |
|---|---|---|
| `id` | `dream:<dreams.id>` | `hypothesis_id` (`dh-…`) |
| `kind` | `dream_narrative` | `dream_hypothesis` |
| `occurred_at` | `created_at` (UTC) | `offered_at` |
| `epistemic_status` | `unsettled` | `unsettled` |
| `text` | `tldr` + blank line + `narrative` | `claim` + `\nWhy: ` + `why` |
| `extra` | `dream_date`, `themes` (≤8, each ≤80 chars) | `cycle_id`, `expired` (bool) |

- **Text cap.** 900 chars (`DEFAULT_TEXT_CAP`) per item; `FULL_TEXT_CAP`
  (4000) when `dream_id` is given. `truncated` is set when cut.
- **Search mode.** `extra.similarity` (0–1, 3 dp) on every item.

### Modes

- **Recent** (no `query`, no `dream_id`).
  - Both kinds (or one via `kind`), merged newest first by `occurred_at`,
    optionally `>= since`.
  - `total_available` = count of eligible rows in the window.
- **Search** (`query`).
  - The query is embedded once and the 20 nearest are taken from
    `orion_dreams`, keeping those `>= DREAM_SEARCH_MIN_SIMILARITY`.
    `kind`/`since` filter inside Chroma first (metadata `kind`,
    `occurred_ts` UTC epoch seconds), so the more numerous hypotheses
    cannot crowd narratives out of the 20.
  - Each hit is re-read from Postgres and re-gated (hypotheses must still be
    offered; `kind`/`since` applied again), then trimmed to `limit`.
  - `total_available` = surviving hits.
- **One** (`dream_id`). The single row, or `ok=True, items=[],
  total_available=0` if it doesn't exist or is a never-offered hypothesis.
  Not-offered and missing look the same.

### Empty ≠ unknown

- **Empty.** Nothing in the window, no hit over the floor, or a
  `kind`/`since` filter that matches nothing in a non-empty index (checked
  with Chroma's count endpoint): `ok=True, items=[], total_available=0`.
- **Unknown.** A Postgres error, an embedder/Chroma failure, search not
  configured, a missing or empty collection, or a malformed request:
  `ok=False` with a safe error. The MCP tool then raises "answer unknown".
  The responder never turns an error into an empty list.

### Bus (`orion/bus/channels.yaml`)

| Channel | Kind | Schema | Producer | Consumer |
|---|---|---|---|---|
| `orion:introspect:dream:request` | `introspect.tool.request.v1` | `IntrospectRequestV1` | orion-harness-governor | orion-dream |
| `orion:introspect:result:*` | `introspect.tool.result.v1` | `IntrospectResultV1` | orion-dream | orion-harness-governor |

- The reply channel is `orion:introspect:result:<correlation_id>`.
- The responder ignores any envelope whose `reply_to` is not exactly that,
  or whose kind is wrong (same trust rule as the reading listener).
- `IntrospectRequestV1` is registered in `orion/schemas/registry.py`.

### MCP tool (`orion/introspect/tools.py`)

- `dreams` is listed alongside `reading_results`.
- `invoke("dreams", args)` validates `DreamsArguments` before transport
  (extra fields rejected) and sends `IntrospectRequestV1` over bus RPC with a
  15 s timeout.
- It validates the reply as `IntrospectResultV1` with `ok=True` and
  `operation="dreams"`; anything else raises `IntrospectUnknownError`.
- The description and `brief.py` gain matching lines. They say dreams are
  experiences, not facts; `items=[]` means none matched; an error means
  unknown.

## Shared search module: `orion/introspect/semantic_index.py`

Moved from `orion/world_pulse_read/search.py`, with no behavior change:

- `SearchConfig` (chroma_url, embed_url, collection, min_similarity,
  index_interval_sec, index_batch, `enabled`)
- `SearchUnavailableError`
- `IndexPass`
- `content_hash`
- `similarity`
- `embed(client, cfg, text, doc_prefix)`
- `nearest(client, cfg, vector, n)`
  - raises on a missing collection, an empty collection, or a malformed reply
- `stored_hashes(client, cfg, ids)`
- `publish_upsert(bus, source, cfg, doc_id, text, vector, model, meta)`

`orion/world_pulse_read/search.py` imports these and keeps its public names
(`ReadingSearchConfig = SearchConfig`, `SearchUnavailableError`, `embed`,
`nearest`, …), so Hub's listener and calibration eval don't change. The
existing reading-search tests are the proof and must pass unchanged.

## orion-dream changes

- `app/introspect_dreams.py`: SQL and item builders, sync, run via
  `asyncio.to_thread`.
  - Functions: `recent`, `by_ids`, `one`, and `index_rows` (the latest
    narratives plus offered hypotheses).
  - Hypothesis SQL always includes `offered_at IS NOT NULL`, and a test pins
    that.
  - Engine: one lazily-created engine on `POSTGRES_URI`,
    `pool_pre_ping=True`.
- `app/dream_search.py`: thin use of `semantic_index`.
  - `document_text` per kind; narrative index text = tldr + themes + the
    first part of the narrative, ≤1800 chars.
  - `index_missing` compares `content_hash` and embeds/upserts up to
    `index_batch` per pass. The Chroma metadata carries `kind`,
    `occurred_at`, `content_hash`.
  - `rank(query)`.
- `app/introspect_listener.py`.
  - Subscribes to `orion:introspect:dream:request`, validates, dispatches,
    and publishes the result.
  - Logs `introspect op=dreams corr=<id> mode=<recent|search|one>
    items=<n> total=<n>` on success, and `introspect_failed op=dreams
    corr=<id> category=<...>` on failure.
  - Runs the index loop every `DREAM_SEARCH_INDEX_INTERVAL_SEC`; a failed
    pass is logged and retried next interval.
  - Resubscribes after a bus drop.
- `app/main.py` lifespan starts and stops the listener when
  `DREAM_INTROSPECT_ENABLED`.

### Env (`services/orion-dream/.env_example`, compose, `settings.py`)

```text
DREAM_INTROSPECT_ENABLED=true
DREAM_SEARCH_CHROMA_URL=http://${PROJECT}-vector-db:8000
DREAM_SEARCH_EMBED_URL=http://${PROJECT}-vector-host:8320/embedding
DREAM_SEARCH_COLLECTION=orion_dreams
DREAM_SEARCH_MIN_SIMILARITY=0.65  # set by calibration; see below
DREAM_SEARCH_INDEX_INTERVAL_SEC=300
DREAM_SEARCH_INDEX_BATCH=10
```

An empty Chroma or embed URL leaves recent/one working and makes search
return unknown. Local `.env` is synced with
`scripts/sync_local_env_from_example.py`.

## Metric quality gate: dream similarity

1. **Provenance.** Cosine similarity between the vector host's bge-large
   (1024-dim) embedding of the query and of each indexed dream. Computed by
   `semantic_index.similarity` from Chroma's l2 distance on unit vectors
   (`cos = 1 − d/2`). Unit length verified: vector-host's `hf` backend
   L2-normalizes (`services/orion-vector-host/app/embedder.py`,
   `_embed_hf`), and live `/embedding` returned 1024-dim vectors of norm
   1.0 on 2026-09-29. That backend mean-pools token states instead of
   using bge's CLS pooling, which likely compresses scores into the
   0.5–0.9 band seen below.
2. **Independence.** It is the only relevance signal in this tool; there is
   no other metric in the same model.
3. **Theory anchor.** Dense retrieval: bge models are trained so that cosine
   similarity ranks passages by semantic relevance to a query. Same basis as
   reading search.
4. **Live sanity.** `services/orion-dream/evals/run_dream_search_calibration.py`
   runs hand-written questions against the live corpus. It builds each doc
   with the index's own code (`index_rows` + `document_text`, read-only
   transaction), embeds it through vector-host, and scores cosine locally
   (no Chroma, no bus). Some questions have a known target dream id; some
   have no matching dream (the negatives). It prints the similarity of each
   target and of the best negative, then picks the floor between them.
   - The gap must be non-degenerate: targets above negatives, not all
     scores bunched together.
   - If there is no gap, search ships with the floor set so only strong
     matches pass, and the PR says so.
   - The corpus is small (19 narratives, 49 offered hypotheses on
     2026-09-29); this is recorded, not hidden. It grows every ~6h sleep
     cycle, so these numbers drift; re-run the eval whenever the floor is
     recalibrated.
   - **Result, 2026-09-29 (68 docs).** 4 related questions, all best hits
     on the expected dream: related_min 0.733 (max 0.787). 3 unrelated:
     unrelated_max 0.603 ("medieval poetry" → `dh-d7a58be50c83`; min
     0.503). Gap 0.130, midpoint 0.668. Scores spread 0.50–0.79, not
     saturated or flat. Run at 0.65: exit 0.
   - **Why 0.65, not the midpoint.** 0.65 sits inside the gap, 0.083 below
     related_min and 0.047 above unrelated_max, and deliberately 0.018
     below the midpoint to favor recall. Under the tool's truth rules the
     two errors are not symmetric: an empty result falsely tells Orion "no
     dream matched", while a weak match arrives labeled `unsettled` with
     its similarity visible.
   - **Known weakness.** Off-topic questions that contain the word
     "dream" score above the floor against narratives. The dream framing
     lifts every narrative's score, so a query phrased that way can return
     a weak narrative match. Topic-only queries separate cleanly. The eval
     measures this in its own `KNOWN_WEAKNESS` section (never sets the
     floor or the exit code): 6 dream-worded related, 4 dream-worded
     unrelated questions. Same 2026-09-29 run (68 docs):
     framed_related_min 0.692, framed_unrelated_max 0.698 ("a dream about
     my grandmother" → dream:14), gap −0.006; all four framed negatives
     (0.660–0.698) clear the floor; "did you dream about pull requests?"
     ranks dream:19/12/16 above the correct dream:17 (0.610).
   - **Floor stays 0.65.** Clearing every framed negative needs a floor
     above 0.698. That would drop the second expected matches — topic-only
     dream:16 (0.661) for the vision question, dream-worded dream:15
     (0.636) for "did you dream about your eyes / vision?" — and the
     correct dream:17 (0.610) for "did you dream about pull requests?",
     which would then return nothing at all. The correct best hits survive
     (topic-only min 0.733, dream-worded correct min 0.716), so the cost is
     lost secondary and mis-ranked matches, traded for hiding weak
     `unsettled` ones — the wrong trade under the truth rules above.
   - **Mitigation (structural, not a word filter).** The `dreams` tool
     description and harness brief tell the model that every record is
     already a dream, so `query` names only the topic ("pull requests",
     not "a dream about pull requests"). No code strips words from
     queries.
5. **Existing mechanism.** Reading search, reused through the shared module.
6. **Reversibility.** An env knob and a Chroma collection. Nothing is written
   to Postgres. Dropping the collection plus unsetting the URL removes it.
   0.65 is tied to the current embedding: switching vector-host's backend
   (`VECTOR_HOST_EMBED_BACKEND`), model, or pooling (e.g. to bge's CLS
   pooling) shifts every score and invalidates the floor; re-index and
   re-run the calibration eval before trusting it.

## Files likely to touch

- `orion/introspect/semantic_index.py` (new)
- `orion/world_pulse_read/search.py` (imports shared plumbing)
- `orion/schemas/introspect.py`
- `orion/schemas/registry.py`
- `orion/bus/channels.yaml`
- `orion/introspect/tools.py`
- `orion/introspect/brief.py`
- `orion/introspect/tests/…`
- `services/orion-dream/app/{introspect_dreams,dream_search,introspect_listener,main,settings}.py`
- `services/orion-dream/{.env_example,docker-compose.yml}`
- `services/orion-dream/README.md`: new "### Introspect responder: `dreams`"
  section (see § Documentation)
- `services/orion-harness-governor/README.md`: flip the `dreams` row in the
  orion-introspect overview to Live, and update its search paragraph to point
  at `orion/introspect/semantic_index.py`
- `.github/workflows/orion-reading-tests.yml`: add `services/orion-dream/**`
  to the trigger paths, and the new orion-dream introspect tests to the run
- `services/orion-dream/tests/test_introspect_dreams.py`
- `services/orion-dream/tests/test_dream_search.py`
- `services/orion-dream/tests/test_introspect_listener.py`
- `services/orion-dream/evals/run_dream_search_calibration.py`
- `scripts/smoke_introspect.py` (`--tool dreams`)

## Documentation

The orion-introspect design overview lives in
`services/orion-harness-governor/README.md` ("orion-introspect: Orion reading
back their own records"). It covers which turns get the server, the truth
rules, the search pattern, how to verify it live and how to add a tool.
`orion/introspect/tests/test_readme_coverage.py` fails the build unless:

- every tool the server lists is a Live row in that table;
- each Live row's request channel is in `channels.yaml` with the answering
  service as a consumer;
- that service's README has an "### Introspect responder: `<tool>`" section
  naming the channel;
- every `orion:introspect:*:request` channel is a Live row.

This slice therefore adds `services/orion-dream/README.md` → "### Introspect
responder: `dreams`", written plain-English first. It covers:

- what Orion can ask (recent / one / by meaning), and the two record kinds
  with their fields;
- the blind-experiment rule: offered hypotheses only, both arms, no arm or
  refs, and why;
- the `unsettled` label, and empty vs unknown;
- the index loop, the `orion_dreams` collection, the calibrated floor and how
  to recalibrate;
- env keys, log lines, failure categories and the smoke command;
- how to turn it off (`DREAM_INTROSPECT_ENABLED=false`).

## Non-goals

- The truth eval (next PR).
- Reveries, curiosity and memory tools (later slices).
- Any write to dream tables, and any change to the offer or scorecard.
- Exposing the `arm`, replay items or linked memory refs.
- Attaching introspect to turns without a reading binding (unchanged
  attachment rule).
- Embed-on-landing. Narratives are written by sql-writer, and cycles run
  every 6 h, so a 5-minute index loop is enough.

## Acceptance checks

- `pytest services/orion-dream/tests -q` passes, including new tests for:
  - never-offered hypotheses excluded in recent, search and one mode
  - `arm`, `ref_a` and `ref_b` absent from every item
  - UTC `occurred_at` on narratives
  - empty vs unknown on each failure path
  - reply-channel trust
  - text caps
- Reading search tests (Hub + `orion/world_pulse_read`) pass unchanged after
  the extraction.
- `orion/introspect/tests` pass: the `dreams` spec is listed, bad args are
  rejected before transport, and timeout / malformed / mismatched replies
  raise unknown.
- `orion/introspect/tests/test_readme_coverage.py` passes with `dreams` Live
  (requires the docs PR `docs/introspect-readmes` on main first; this branch
  merges main before implementation).
- `python scripts/check_schema_registry.py`,
  `python scripts/check_bus_channels.py` and
  `python scripts/check_env_template_parity.py` pass.
- Live, after deploy:
  - `scripts/smoke_introspect.py --tool dreams` returns the newest narrative
    and offered hypotheses.
  - `--tool dreams --query "<theme of a real dream>"` returns that dream with
    similarity over the floor.
  - The orion-dream log shows `introspect op=dreams corr=…` lines.
  - Chroma `orion_dreams` holds 19 + 34 docs (or the live counts at deploy).
- The calibration eval output (targets vs negatives, chosen floor) is
  recorded in the PR.

## Follow-up PR: introspect truth eval

`orion/harness/evals/introspect_truth_live_eval.py` (opt-in live runner) plus
`orion/harness/evals/test_introspect_truth.py` (offline gate on fixtures).

- **Asking.** Real questions to real Orion through `POST /api/chat`, e.g.
  "what did you dream last night?", "did you dream about vision?", and
  reading equivalents. Each turn's exact introspect tool calls and results
  are pulled from the governor's Claude session log by correlation ID.
- **Code checks.**
  - Was the relevant tool called?
  - Did it error?
  - If every call errored, the turn is scored on whether the reply reports
    "unknown" (judged, see next).
- **LLM judge.**
  - Sees only the tool-result JSON and the reply. Lists each claim about
    Orion's dreams or readings as supported / unsupported / error-as-empty.
  - Must first pass a hand-labeled calibration set with planted
    fabrications, and catch all of them, before its scores count.
- **Hygiene.** Same cancel-on-error and one-runner-only rules as the stance
  live eval.

It gets its own short plan when started.

## Recommended next patch

This slice as one PR, from the plan
`docs/superpowers/plans/2026-09-28-orion-introspect-slice2-dreams.md`.
