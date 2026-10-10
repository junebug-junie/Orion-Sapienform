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
     |   NEW distinct things since the last sleep (absent from    +--> REM compaction (staged, existing)
     |   the 48 h before it): metacog event kinds, reverie        |
     |   compaction themes, resonance themes, newly activated     +--> recombination --> dream_hypothesis
     |   crystallizations. Reads exactly 0 after a sleep.              dream arm:   distant replay pairs
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

### What makes Orion tired

Pressure counts **things, not rows**, and only **new** things. Design and the
metric-gate record: `docs/superpowers/specs/2026-10-09-dream-sleep-pressure-novelty-design.md`.

- **One thing, one count.** Each source query returns one row per `dedupe_key`
  (`cycle_store.SOURCE_QUERIES`): metacog's `trigger_reason` with ids and numbers
  normalized (its `summary` is model prose, reworded every row), the reverie theme,
  or the crystallization id. 222 copies of one gateway timeout are one replay item.
- **Only new things add pressure.** A key also seen in the `DREAM_LOOKBACK_HOURS`
  before the window adds nothing. A chronic problem adds pressure once, the first
  time, and stays a replay candidate every window it recurs.
- **Recall is not new material.** Crystallizations count when they are activated
  (`memory_crystallization_history` `auto_activate`/`approve`), not when
  `updated_at` moves: every recall rewrites `updated_at` on ~100 of them.
- **Backstop.** If the window reaches `DREAM_LOOKBACK_HOURS` (its furthest reach)
  without crossing threshold, Orion sleeps anyway when idle and there is anything to
  replay (cycle note: `overdue`). Without it, a repetitive stretch with nothing new
  could hold pressure at 0 forever, and older material would fall out of view unreplayed.
- **No per-source cap.** Reads are one row per thing, so `DREAM_CANDIDATES_PER_SOURCE`
  (50) was retired: near real volume (~72 metacog kinds/48 h) it silently dropped keys.
  `DREAM_REPLAY_MAX` bounds a sleep.
- `SleepPressureV1.counts` = distinct things per source in the window;
  `new_counts` = the ones that drove `pressure`.
- Backtest on the real week before 2026-10-09 at threshold 3: 11 sleeps, gaps
  6-40.5 h (the old rule: 28 sleeps, every gap 6 h). Re-run with
  `pytest services/orion-dream/evals/test_sleep_pressure_backtest_eval.py -s`,
  or against fresh live data with `scripts/backtest_sleep_pressure.py`.

A recombination call the gateway refuses (e.g. the GPU pool sheds it for heat,
`raw.error=gpu_pool_unavailable`) counts as a failed call (`llm_failures`), not an
unparseable answer. A sleep where every call fails is stored `failed`, so the next
sleep's replay window still covers its items. It retries after `DREAM_MIN_INTERVAL_HOURS`.

Writes nothing to canonical memory. The dream never writes a belief.

The legacy direct-gather path (`dream_cycle.py`, `aggregators_*`, `memory_listener.py`)
was deleted in the same patch.

### The rest drive: tiredness other parts of Orion can read (2026-10-10)

Every check (and right after every sleep) the loop turns its pressure into a
`DriveReadingV1` (`orion/schemas/drive_reading.py`, built by
`orion/regulation/rest_drive.py::read_rest_drive` from the same values the sleep
gate uses) and writes it to Redis `orion:drive:rest:latest` with a 1800 s TTL.
States: `resting` (0.0), `building`, `due` (crossed the threshold, or the 48 h
overdue backstop), `refractory` (inside the 6 h minimum), `no_reading` (a source
read failed). Hub curiosity and outreach stretch their cooldowns only while it
reads `due`; anything else, or no key at all, changes nothing for them.
`source_ref` is the `dream_pressure_observation.check_id` of the same check (the
reading published right after a sleep is `dp-postsleep-*` and has no row).
`GET /dreams/cycle/pressure` shows the reading as `rest_drive`.
Off switch: `DREAM_REST_DRIVE_PUBLISH_ENABLED=false`. Eval:
`scripts/analysis/measure_rest_drive_easing.py`.

### Every sleep ends in a story

A completed sleep (not `failed`, not `empty`) starts one narrative dream, the kind
stored in the `dreams` table. Before 2026-10-09 nothing scheduled that dream: all 19
were started by hand, and Orion's journal noticed the silence. Now it runs on the
same tiredness gate as the sleep (`app/story.py`).

- The sleep publishes `dream.trigger` (`DreamInternalTriggerV1`) with `trigger_id`
  `sleep:<cycle_id>` and a `sleep` digest. The digest holds tiredness against the
  sleep line, whether this was an overdue (backstop) sleep, and `material`: the
  replayed items plus every item in the control pairs, in a seeded shuffle.
- cortex-orch runs the `dream_cycle` verb. `dream_cycle.j2` puts that material
  first and uses the recalled memories as texture. A hand-started dream (no `sleep`)
  gets the old memory-only prompt. Orch logs `sleep_material=<n>` on dispatch.
- The trigger, digest included, is saved with the dream in
  `dreams.metrics._dream_audit.trigger`, so each story names the sleep it came from.
- Blind experiment: the sleep's hypotheses are shown to Orion later with the arm
  hidden. Dream pairs come from the replay and control pairs from the whole pool.
  A story about the replay alone would make the dream-arm items familiar and bias
  the result, so the story gets both arms' items, unlabeled and unordered, and
  never the hypotheses themselves.
- No story for a sleep that was `failed`, `empty`, or not saved.
- Starting the story is best effort. If the publish fails, the sleep still counts.
- Off switch: `DREAM_STORY_AFTER_SLEEP_ENABLED=false`.
- With `DREAM_CARRY_ENABLED=true` (the default) the story is carried instead of
  published as one paragraph; see the next section. The story path above is what
  `DREAM_CARRY_ENABLED=false` returns to.

### Every sleep ends in a carried dream

Since 2026-10-10 the dream a sleep ends in is carried through words and pictures:
Orion writes a passage, it is painted, Orion looks at the painting, and the dream
continues from what was *seen*. Six hops (text, picture, text, picture, text,
picture), so the last picture's caption is the dream's last word. Design:
`docs/superpowers/specs/2026-10-10-dream-carry-through-design.md`.

- **Start.** A completed, saved sleep submits one `dream.carry` durable run
  through cortex-orch's durable ingress (`CHANNEL_CORTEX_REQUEST`,
  `metadata.durable_run`) instead of publishing `dream.trigger`. The brief is the
  same `sleep` digest the story would have had, and the run id is derived from
  `sleep:<cycle_id>`, so one sleep is one carry. The submit counts only when the
  receipt names this run, workflow and resource (`app/carry_submit.py`). A failed
  submit is logged (`dream_carry_submit_failed`) and never fails the sleep. It is
  tried once more (the run id dedupes, so a timeout after durable-runs already took
  the run cannot start a second dream); if that fails too, it falls back to the
  one-shot story (`dream_carry_fallback_story`) so the sleep keeps its dream.
- **Who does what.** orion-durable-runs runs the hops and checkpoints each one.
  orion-thought paints and captions the pictures. orion-dream answers the run's
  text and finish steps on `orion:dream:carry:step:request` (`app/carry_listener.py`,
  `app/carry.py`).
- **Text hop.** One gateway call on the `metacog_background` route, attached to
  the run's own LLM hold (`options.gpu_lease`). Hop 0 gets the sleep's material
  under the same blind-experiment rule as the story (both arms, unlabeled, never
  the hypotheses). Hops 2 and 4 get the previous passage and "the dream turned into
  a picture; looking at it you see: <caption>", and are told to follow the picture.
  The reply must be JSON `{"passage", "image_prompt"}`; the image prompt is clipped
  to 60 words (the painter's text encoder drops everything past 77 tokens). An
  empty, unparseable or refused reply, or a timeout, answers `retry`, never a
  blank hop. Refusals and timeouts retry until the deadline; a hop whose replies
  are unparseable 3 times (or whose handler crashes 3 times) answers `terminal`, so
  a model that keeps answering badly cannot burn the whole 4 h window.
- **Finish.** One `dream.result.v1` on `CHANNEL_DREAM_LOG`, so the carry lands in
  `dreams` like any other dream: `mode=carry`, `narrative` = the passages with each
  caption between them as `[picture] <caption>`, one `fragments` entry per hop
  (passage and image prompt, or sha256 and caption). The trigger (sleep digest,
  run id, `stopped_reason`) is in `metrics._dream_audit.trigger`. A carry that hit
  its deadline (`DREAM_CARRY_DEADLINE_SEC`, 4 h) publishes the hops it made with
  `stopped_reason`. A sleep's carry that made no hops falls back to the one-shot
  story (`dream.trigger` with the same digest) and answers `done` with dream id
  `story-fallback:<trigger_id>`, published once per run; a hand-started carry with
  no hops answers `terminal` and publishes nothing.
- **Replays.** The dream id is derived from the run id. A finish that was already
  published (remembered in-process, or found in `dreams` by that id) is answered
  `done` again without a second row.
- **By hand.** `POST /dreams/carry/run` starts a carry with no sleep behind it
  (trigger `manual:<uuid>`); it dreams from a free seed and returns `{run_id, status}`.
- Off switch: `DREAM_CARRY_ENABLED=false` stops new carries (sleeps go back to the
  one-shot story, and the endpoint refuses). The step responder keeps running so
  carries already in flight still finish.

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
| `orion:dream:trigger` | `CHANNEL_DREAM_TRIGGER` | `dream.trigger` | `DreamInternalTriggerV1`; published by `POST /dreams/run` and at the end of every completed sleep; **handled by cortex-orch**. |

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


## Pressure check history (2026-10-09)

Every successful scheduler pressure read now appends a `DreamPressureObservationV1`
(SQL-only) to `dream_pressure_observation`, including checks skipped for a recent
attempt, low pressure or activity. The recorded `reading` is the existing
`SleepPressureV1`, with its formula family, thresholds, prior cycle timestamps,
check cadence and source-read errors. Neither recording failure nor retention
changes a scheduling gate. Failed source/clock reads remain the old runtime
fallback but make the observation unusable as evidence of calm. HTTP pressure
reads do not create scheduler-check history.

Apply `services/orion-sql-db/manual_migration_regulation_history.sql` before
restarting dream. Recording failures log `dream_pressure_history_failed` and/or
`dream_pressure_history_write_failed`; they never stop a dream. The separate
one-connection history pool bounds connection/pool waits to two seconds,
statements to two seconds and lock waits to 500 ms. These waits add bounded
instrumentation latency, not a new gate. Retention deletes at most 1,000 rows
older than 30 days per check in a separate transaction; failure cannot undo the
new observation. No prompt, replay text or hypothesis body is copied.

The existing read-only report includes this history with `--with-checks`:

```bash
python scripts/analysis/measure_dream_pressure_crossings.py --print-sql --with-checks \
  --start 2026-10-09T06:00:00Z --end 2026-10-10T06:00:00Z
```

Execute its read-only SQL with `psql -XqAt -v ON_ERROR_STOP=1`, save JSONL, and
pass that export back to the same script without `--print-sql`. Old cycle-only
exports remain supported. The report only claims fall-and-rise when real checks
straddle a stored successful cycle and later rise without a source failure,
formula change or long sampling gap. A missing check is never interpolated.
