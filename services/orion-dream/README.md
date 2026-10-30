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

### Sleeps only when arousal reads idle (2026-10-12)

Temporal Self R3, reader 1. When a check would sleep (pressure due, or the overdue
backstop), the loop first reads durable-runs' `orion:regulation:latest`. `engaged`
(Juniper is talking) or `strained` (cabinet hot, or the GPU queue backed up for 5
minutes) holds the sleep and logs `dream_cycle held reason=arousal_<level>`. `idle`
lets it through. Arousal only vetoes: it never authorizes a sleep the
Juniper-filtered idle query refused, and a level that has held for
`DREAM_AROUSAL_MAX_HOLD_HOURS` (6) stops holding, so a fault-jammed GPU queue (which
reads strained) cannot cost Orion a day of sleep. A manual `/dreams/cycle/run` that
arousal held answers `{"status": "held", "reason": "arousal_<level>"}`. An absent, stale (> `ORION_REGULATION_READER_MAX_AGE_SEC`,
360 s), malformed or `unknown` state, a Redis error, or
`DREAM_IDLE_FROM_AROUSAL_ENABLED=false` leaves the idle query alone in charge, so a
durable-runs or Redis outage cannot stop Orion sleeping. Forced sleeps and the 6 h
minimum never read it. Eval: `scripts/analysis/measure_arousal_reader_replay.py`.

### Replay draws on closed arcs (2026-10-14)

Temporal Self rev 4, patch 7. Once a sleep goes ahead, it also reads the closed
Temporal Self arcs (`temporal_self_arc`) that ended since the last sleep began: a
concern held, a curiosity followed, a body state. `sleep` arcs are left out, and at
most 20 go in per sleep. Arcs about the same thing (same kind and subject) count
once, as the heaviest of them. Each becomes a replay item with `source_kind="arc"` and
`ref_id="arc:<arc_id>"`, so anything the dream produces from it can cite the arc.
Weight = `0.75 * (dwell + returns) / 2`, where `dwell = min(1, ln(1+min)/ln(61))`
(reaches 1 at an hour) and `returns = min(1, attention_returns/3)`. The 0.75 cap keeps
a critical metacog (1.0) and a 0.8-salience crystallization ahead of every arc. An arc with no dwell and no returns is dropped.
Arcs take at most a third of the replay (4 of 12), so the other sources, things the
day explicitly flagged, always keep at least two thirds; within that third an arc
heavier than an existing item does take its slot. The control arm's pool gets the
same top arcs, not all 20, so both arms see arcs at a similar rate. Arcs are replay material only:
they never add to sleep pressure, never count as a reason to sleep, and never touch
the rest drive. The cycle note records `arc candidates=N`. Off switch:
`DREAM_REPLAY_ARC_CANDIDATES_ENABLED=false`. A failed or slow read (2 s statement
timeout, 5 s overall) means no arcs, and the sleep runs exactly as before.

### Replay draws on graded walkway predictions (2026-10-15)

What Orion expected to see on the street and whether it came: met/missed rows of
`vision_percept_expectation` (the walkway rhythm reducer, orion-sql-writer) scored
since the last sleep began, one replay item per subject (`source_kind="walkway"`,
`ref_id="walkway:<expectation_id>"`), worded like "On the walkway I expected the black
dog around 07:40: came on Thu 22 Oct; did not come on Wed 21 Oct. Over the last 14
days it came 6 of 8 times." Open and unscorable rows are not results and are dropped.
Weight is a declared heuristic: 0.5 when the subject missed at least once in the
window (a prediction error), 0.3 when every one came; at most a sixth of the replay
(2 of 12). Like arcs: replay material only, read after the pressure read, never sleep
pressure, never starts a sleep, any read failure = the sleep as before. The cycle note
records `walkway candidates=N`. Empty for the camera's first week: the reducer only
emits an expectation with >= 5 days of support. Off switch:
`DREAM_REPLAY_WALKWAY_ENABLED=false` (or empty `DREAM_WALKWAY_STREAM_IDS`).

### Reverie themes read as titles

Compaction requests and resonance alerts name a reverie theme by its bare open-loop id
(`open-loop-968fda08910c`). The source queries also fetch the loop's latest human
description from `attention_salience_trace` (by the indexed `theme_key`), and replay
text uses it: "Harness closure prediction error: reverie_chain_max_steps". The id stays
the novelty key, ref and tag, so tiredness is unchanged when a title is reworded. No
title found -> the id, as before. The title is the one current when the window closed.

Cohort break: hypotheses made after this change pair titled reverie fragments, earlier
ones id-only. Both arms changed together (arm vs arm stays fair), but absolute adoption
rates may shift at the deploy date; split the dream scorecard there.

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

### Every sleep is one act in the act ledger (2026-10-23, body economy A5)

A sleep that starts replay is one `make` act (actor `dream.cycle`) on `orion:grammar:event`,
persisted by orion-sql-writer to `grammar_acts`. It opens once the cycle has passed its gates and
has material to replay, and closes once when the cycle is persisted: 1 for a completed, saved
sleep with at least one hypothesis, else 0. Any exception or cancellation after the open closes it
at 0 and is re-raised. A check that is not due, held, or `empty` opens nothing.

- Act key = the cycle id (stable act id). A restart mid-cycle leaves the act open; the writer
  expires it at 0 at its horizon: (DREAM_HYPOTHESES_PER_CYCLE + DREAM_CONTROL_PER_CYCLE) x
  DREAM_LLM_TIMEOUT_SEC + 120 s (480 s by default).
- GPU cost: during the sleep every recombination LLM call carries one correlation id derived from
  the cycle id, so the GPU pool's leases carry it and the ledger charges the cycle for them.
- Cost `seconds` = wall time from open to close.
- Flag: `DREAM_ACT_EMIT_ENABLED` (default on). Off = no act, and each LLM call keeps its own id.

### Body economy shadow bid (Patch C3b, 2026-10-27)

Since Patch D for dream (next section) this runs only with `ORION_PHYSIOLOGY_BID_DREAM_CYCLE=shadow`.

A check that would sleep asks the body economy "would you fund this sleep?", then ignores the
answer. The answer is only logged, next to what the cycle really did. Nothing changes what the
cycle does.

- Where: `run_cycle_once`, once the check would sleep (pressure due, the overdue backstop, or
  forced), before the arousal read. A check that is too soon or not due is not a want: no bid.
- Family `dream.cycle` (Tier 1), verb `make`. `want_id` = `act_emit.cycle_correlation_id(cycle_id)`,
  the sleep act's correlation id. The cycle id is now minted before the arousal read, so a held
  check has an id too; it opens no act.
- Real outcome: `ran` (the gates passed and the sleep's act is opened; a sleep that then fails
  still ran, and its act closes at 0), `blocked:arousal_<level>` (arousal held the sleep),
  `blocked:nothing_to_replay` (the `empty` cycle, which opens no act), or `blocked:cancelled`.
  With `DREAM_ACT_EMIT_ENABLED=false` a `ran` line has no act to join.
- Volume: every held check is a new want with a new `want_id` (a new cycle id). A long hold at
  the default 600 s check interval writes about 6 rows an hour; count holds per episode, not per row.
- Log line: `physiology_shadow_bid family=dream.cycle want_id=... shadow=funded|deferred|none ... real=...`.
  durable-runs writes the decision into `want_ledger`. A funded decision whose sleep was held
  releases its reservation.
- Switch: `DREAM_PHYSIOLOGY_SHADOW_BID_ENABLED` (default on). The bid is a background task; the
  cycle never waits on it. Code: `app/physiology_shadow.py`.
- Code: `app/act_emit.py`, `CycleActTrace` in `app/cycle.py`. Tests: `tests/test_dream_cycle_act.py`.

### The sleep obeys its bid (body economy Patch D, 2026-10-30)

Before this, the bid above was only logged. Now the sleep asks the body economy and obeys the answer.
This uses the same rule curiosity investigate obeys (`orion/physiology/binding.py`).

- Where: `run_cycle_once`, after every gate of the dream's own has passed (pressure or the overdue
  backstop, the 6 h refractory, arousal, something to replay), just before the sleep's act opens.
  A check that arousal holds, or a forced run, never bids.
- Funded: the sleep runs. Metacog lane full, but the sleep is worth its price: the sleep runs (its
  LLM calls wait in the GPU pool's queue). No answer (market down, stale body state, timeout 5 s):
  the sleep runs, as before.
- Deferred on price (Prime's slow charge counts here), no hunger, or a value at or below zero: the
  sleep is HELD. The dream logs why and bids again at the first check after the market's
  `retry_after_sec` (clamped 2 min .. 1 h). `POST /dreams/cycle/run` answers
  `{"status": "held", "reason": "economy_<reason>"}`.
- One want per stretch between sleeps: the first check that clears the gates opens it, and every
  re-bid and the eventual sleep's act share its cycle id. The want lives in process memory, so a
  restart opens a new one.
- Bounds. A held sleep is never dropped. It goes anyway at the earlier of its first bid + 6 h
  (`dream.cycle` `deferrable_window_s` in `config/physiology/families.yaml`) and the overdue line
  (the last good sleep + `DREAM_LOOKBACK_HOURS`). The 6 h counts from the first bid, not from the
  first check that wanted to sleep, so hours of arousal holds do not use it up. The 6 h is per
  process lifetime: a restart opens a new want, so across repeated restarts the bound is the 48 h
  overdue line. An overdue sleep never bids at all (`floor:overdue`). Delay costs no replay
  material: the window reaches back to the last good sleep.
- Two checks racing on one want (the loop and a manual run): the first admitted takes it; the other
  is refused (`economy_duplicate`) and opens nothing.
- Record: one log line per verdict, `physiology_bind family=dream.cycle ... verdict=... reason=...
  binding=... net=... why=... obeyed=true`. The decision is in `want_ledger` with producer
  `orion-dream.dream_cycle.binding`. A sleep that ran after a hold says so in its cycle note
  (`body economy: held N time(s) over M min ...`). The funded reservation is released when the act closes.
- Prime's `slow` receptor: `dream.cycle` declares `receptors: [slow]`, so Prime can put a bounded,
  temporary charge on these bids (at most half of the sleep's value). A hungry sleep outbids it.
  Only binding bids take the charge: with the flag at `shadow`, or on a forced run, the market
  neither applies it nor records it for Prime's grade.
  Quicken is not declared: dream has no price deferrals for a credit to lift.
- Switch: `ORION_PHYSIOLOGY_BID_DREAM_CYCLE=on` (default). `shadow` = the shadow bid above, exactly as
  before (instant rollback). Code: `app/physiology_bind.py`. Tests: `tests/test_physiology_bind.py`.
  Replay: `PYTHONPATH=. python services/orion-dream/evals/dream_binding_replay.py [--live --days 7]`.
- The old arousal hold stays until the binding bid has run `on` for 7 days with its checks green.

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

### Weekly/monthly Curiosity consolidation (durable)

`DREAM_CURIOSITY_COMPACTION_ENABLED=true` makes a **completed, persisted sleep cycle**
check closed Denver calendar weeks (Monday–Monday) and calendar months. This hook is
independent of the narrative story and `dream.carry`; it belongs to the broader sleep
arc. GitHub/chat compactors retain their independent daily actions schedules (06:10 /
06:00 Denver). REM's staged memory proposals remain a separate activity.

The sleep service snapshots Curiosity journal records and admits `curiosity.compaction`
through cortex. Durable-runs performs content-derived topic consolidation, reduces the
parts, then writes a distinct overall chronological synthesis. Every topic has dated
source references. Monthly input reuses wholly contained completed weekly topics,
then includes uncovered original write-ups (including late arrivals and month-edge days).
Source records are never deleted, retagged, or rewritten. Incomplete investigations
remain evidence of unfinished work, not claims of completion.

- First activation starts with the previous closed week/month, not all history.
  Thereafter each sleep admits the oldest pending/missing window per cadence.
- A pending window keeps an immutable input snapshot. Post-snapshot backdated inserts
  are not silently incorporated into that artifact; a future explicit rebuild is needed.
- A busy GPU pool is a durable wait. Each accepted topic/overall answer is saved before
  advancing the graph. Failed durable runs retry their saved progress on a later sleep
  using a fenced generation; operator-cancelled/abandoned windows are not resurrected.
  Retries are bounded: when 3 failed generations in a row end at the same saved cursor because
  the model's answer was rejected there (`MAX_STALLED_GENERATIONS`, `state.stall`; durable-runs
  writes `state.rejected_at`), the window is set `cancelled` and the next window can run. A GPU
  deadline, an outage or a code bug does not count. A generation that saves a batch resets the
  count. To re-open a given-up window, set its row back to `pending` and remove `state.stall`;
  the oldest pending window is resumed first, even after later windows exist.
- A source record longer than half the input budget is split into near-equal pieces at a
  word boundary (no tiny tail pieces).
- Empty windows are explicitly `empty`, with no model call or fabricated journal.
- Completed topic+overall summaries enter the existing journal/index path with stable
  IDs and no notification trigger. Sleep retries journal delivery until the SQL row exists.
- `GET /dreams/curiosity-compactions?limit=20` shows windows, durable run IDs/terminal
  state, phase, topics, overall prose and dated source references. Store failure is 503.
- No new metric is wired into a cognition loop. This patch creates inspectable derived
  artifacts, not a numerical claim about selfhood or growth.

Apply `services/orion-sql-db/manual_migration_curiosity_compaction.sql` and update the
DurableRunState/Request consumers before enabling. Deployment order and evidence:
`docs/superpowers/pr-reports/2026-10-14-curiosity-periodic-compaction-pr.md`.
Turning the flag off stops new admissions and delivery reconciliation. Use the existing
durable pause/cancel controls for work already admitted. Keep artifacts on rollback.
