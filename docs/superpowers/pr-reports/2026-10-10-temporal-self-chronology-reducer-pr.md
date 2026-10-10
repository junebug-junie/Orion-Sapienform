## Summary

This is patch 2 of Temporal Self ([PR #2369](https://github.com/junebug-junie/Orion-Sapienform/pull/2369), rev 4). It is the chronology lane: a pure reducer that turns rows Orion already writes into one account of its day, made of **arcs**. An arc is either a stretch of time Orion kept coming back to one subject, or one bounded process such as a conversation, a curiosity run or a sleep. Nothing runs the reducer yet. The durable-runs driver, migration and routes are patch 3. Nothing is deployed.

- `orion/schemas/temporal_self.py` adds the event, arc, frame, closed-day and state models. They are registered in `_REGISTRY` and checked through `resolve()`.
- `orion/temporal_self/` is the reducer itself, with no I/O and no LLM. It has eight arc lanes, one source adapter per live source, one shared "which day is it" helper, and a per-arc body summary.
- 73 gate tests cover each lane's rules, the source filters, midnight and timestamp casts, replay identity (including broadcast silence before midnight and an exact-watermark re-read), a restart, and frames with no winner.
- An arc-precision eval runs on real rows from 10-09 (read-only export, 980 KB gzipped, text-free). It covers evidence precision per kind, graded by an oracle that never uses the reducer's own extraction; process recall; replay identity; rest state; and six hand labels.
- K (ticks to open an arc) and R (the return window) were picked from a sweep over that real day. The metric gate was run for every number the frame exposes, and two body numbers failed it and were not built.

## Outcome moved

"Did Orion's morning change Orion's afternoon?" can now be answered from rows, on a real day, deterministically. Every arc carries the ids of the rows that built it, and the eval checks every one of those ids against the raw source columns.

### What the reducer finds on 10-09 (America/Denver)

Orion's day on 10-09 looked like this:

- **00:33** — Orion slept for 6 seconds and proposed 4 hypotheses. This was the first sleep driven by pressure rather than the clock (after #2557).
- **08:04–10:41** — Orion reached out four times on one chat session and got no reply. The reducer files these as Orion's own rows (context), not as a conversation.
- **08:05–16:04** — seven curiosity runs, back to back, each 28 to 67 minutes long. An eighth run had started at 13:56 the day before and finished at 10:31. Its 20.6-hour span is real in the source rows, and the reducer flags it.
- **15:00 and 18:08** — Juniper spoke twice ("howdy friend, and a Happy Friday to you"). These are two separate conversation arcs, because the turns were 3 h 8 min apart.
- **Workspace attention** (what the broadcast picked) moved between internal subjects:
  - `harness_closure`: 2.3 h in total, over 22 stretches;
  - `execution`: 2.2 h;
  - `chat`: 1.3 h.
  - The longest arc ran from 21:47 to 23:12 on `harness_closure`. `execution` interrupted it three times and attention came back each time. That count was checked by hand against the raw ticks.
- **Field attention** (which part of its own machinery was most salient) sat on `bus_synaptic` for 13.6 h of the day. One arc ran from 12:46 to 18:06 with 33 returns. This is the same hogging #2576 measured.
- **19:01** — Orion slept again, 18.5 h after the first sleep, and proposed 4 more hypotheses.
- **At midnight**, two concern threads were still open. Juniper first raised one of them on the evening of 10-08. On 10-09 Orion's own four outreach messages raised it again, once each, and Juniper raised it once more at 14:59. That makes 5 returns. Both lanes are linked by exact correlation ids, not by time.
- **Rest:** for 69% of the day no foreground arc was open, and 51% of broadcast ticks had no winner. The instrument can rest.

### Eval numbers (`python orion/temporal_self/evals/run_arc_precision_eval.py`, all gates green)

| Gate | Result |
|---|---|
| Evidence precision (every `evidence_ref` resolves to a raw row with the same subject) | **2,103 / 2,103 = 1.0**, 0 unresolved. By kind: attention 728/728, reverie 902/902, interoception 455/455, concern 6/6, curiosity 8/8, conversation 2/2, sleep 2/2 |
| Tick extraction (`tick_from_log_row` on the raw projection) vs the subject computed in SQL | 0 disagreements over 3,952 ticks |
| Context purity (only subject-less kinds in `context_event_ids`; no Juniper turn) | 0 violations |
| Process recall: sleep / curiosity / reverie / imagery | 2/2, 8/8, 184/184, 0/0, all exact |
| Each dream hypothesis attached to its own sleep arc by cycle id | exact for both sleeps |
| Replay identity: one pass vs 48 hourly chunks with `advance_clock` | byte-identical, both days |
| Rest reachable | foreground covers 30.6% of the day |
| Hand labels (`fixtures/day_2026-10-09.labels.json`, each checked against SQL) | 6/6 |

Arcs on 10-09: 49 attention, 44 interoception, 184 reverie, 8 curiosity, 2 conversation, 2 concern, 2 sleep.

### K and R from the live sweep (`--sweep`, 10-09)

| K | R | attention arcs (0 returns / ≥2) | interoception arcs (0 / ≥2) |
|---|---|---|---|
| 2 | 30 m | 50 (25 / 12) | 44 (9 / 28) |
| **3** | **30 m** | **49 (34 / 9)** | **44 (9 / 28)** |
| 3 | 10 m | 65 (54 / 2) | 165 (73 / 57) |
| 3 | 60 m | 30 (14 / 10) | 15 (4 / 11) |
| 3 | 180 m (spec default) | 13 (3 / 7) | 8 (0 / 7) |
| 5 | 30 m | 38 (32 / 4) | 44 (9 / 28) |

The chosen values are **K = 3 ticks and R = 30 min**:

- K = 3 is about 110 s at the live broadcast cadence. The 7-day tick gaps have a median of 37 s and a p99 of 64 s. Over 7 days, 603 streaks reached 3 ticks.
- At R = 180 min the 9 subjects collapse into about 13 arcs a day, and "zero returns" almost disappears. The spec wants zero to be common.
- R = 30 min keeps zero returns common, still produces real returns, and matches #2576's interoception probe (38 arcs in 21 h at R = 30).
- Conversations get their own **3 h** return window. They suspend after 45 min idle (the dream service's idle bar), so a 30-minute R would mean a conversation could never return.

## Current architecture

- Every source below writes timestamped rows. Nothing reads across them to say which stretch of Orion's day a row belonged to.
- `field_dominance_run` (S2, #2555) and the #2576 replays are on main.
- `orion/orion_day/` is the LLM-written daily letter. It gathers material; it does not build a chronology. Its America/Denver window helper is reused here, so the letter and the chronology agree about which day a row belongs to.
- The parallel regulation work (`orion/regulation/`, `orion/schemas/drive_reading.py`) is untouched.

## Architecture touched

- New package `orion/temporal_self/`: pure functions, no service.
- New schema module, plus registry entries.
- One honest exclusion pair in the inner-state gate.
- One path-filtered CI workflow.
- No bus channel, no env key, no service, no migration.

### Source liveness (live, 2026-10-10)

| Source | Status | Decision |
|---|---|---|
| `substrate_attention_broadcast_log` | 1,973 rows / 24 h | **bound** (attention lane). 44% of ticks select a loop, and every selected loop has `source_refs[0]` (6,544/6,544 over 7 days), so the spec's loop-id fallback is not needed |
| `field_dominance_run` | 414 / 24 h | **bound** (interoception) |
| `chat_history_log` | 6 / 24 h | **bound**, Juniper's turns only (see drift) |
| `curiosity_offer_decisions` ⋈ `curiosity_run_outcomes` | 10 / 24 h | **bound** |
| `substrate_reverie_chain` + thoughts | 285 / 24 h | **bound**. A chain with no thoughts (116 of 283 on 10-10) has no content and makes no arc |
| `reverie_visual_chain` / `_attempt` | 8 / 19 per 24 h | **bound** (imagery, constraints) |
| `dream_cycle`, `dream_hypothesis` | 1 / 4 per 24 h | **bound** |
| `gpu_pool_events` (admission-cue predicate) | 37 / 24 h | **bound** (constraint) |
| `orion_metacog` ⋈ `metacog_trigger` (degraded/critical) | ~300 / 24 h | **bound** (context) |
| `substrate_attention_schema` | 2,724 / 24 h | **bound** (per-arc lane counts) |
| `attention_salience_trace` (chat) / `attention_loop_outcome` | see drift | **bound** (concern) |
| `memory_consolidation_windows`, `episode_memory`, `substrate_action_outcomes`, `vision_events` (room, with entities) | 2 / 9 / 3 / ~10 per 24 h | **bound** (context / self-change) |
| `aitown_chat_history_log` | 0 rows since 09-17 | **dropped** (town lane, town filter test) |
| `vision_presence_transition` (S1) | table does not exist | **dropped** (company lane) |
| `vision_percept_expectation` | 0 rows | **dropped** |
| `vision_unresolved` | 1 row ever | **dropped** |
| FalkorDB `:PriorRevision`, peer-ask nodes | need a graph read | **deferred to patch 3** (a pure adapter is trivial once the driver reads them) |
| Situation revisions | Redis keeps only the latest; no history | **dropped** for replay |
| `substrate_attention_self_model` (self-prediction summary) | rows live, but acceptance check 9 requires reproducing the script's 66.0% on the *same* rows, which aged out (168 h) | **not bound** |

### Spec drift found live (the spec is stale against main and data)

1. **Chat-scope salience traces are not chat raises.** On 10-09 one loop got 381 `scope='chat'` traces, each with a distinct correlation id, spread across all 24 hours. Only 6 of them resolve to a chat turn. A concern raise now requires the trace's correlation id to exist in `chat_history_log`. Without that rule, concern dwell would measure the scorer's cadence. A SQL check of who wrote the raising turns found that on 10-09, 4 of the 5 raises came from Orion's own outreach. That is recorded here only; no field stores it, because nothing would read it yet.
2. **Orion's outreach looked like a conversation.** On 10-09 the only "conversation" session held four unsolicited outreach messages and no reply. The session named `orion_journal` is where Juniper actually talked. Conversation arcs now open only on Juniper's turns (non-empty prompt, the same rule as `measure_arousal_replay.py`). Orion's rows are context. This is the same failure class as spec danger mode 9.
3. **Curiosity attention rows carry a uuid5 `correlation_id` that matches no `run_id`** (0 of 41 over 3 days). Rule 8 cannot bind them by reference, so they bind to curiosity arcs by time. Reverie rows *do* bind by reference, through the thought's `correlation_id`. Thoughts link to chains by `thought_json.chain_id`, not by `correlation_id`.
4. **Visual attempt outcomes** add `abandoned`, the only deferral word seen in the last 7 days. Over 30 days, 12 `deferred_thermal` attempts exist.
5. **Dream hypotheses are written about 5 s before their cycle's `ended_at`.** The link is parked until the sleep arc exists. That bug was found by the eval and fixed here.
6. **S2 volume.** The spec expected about 2,900 runs a day. It is actually about 415 rows in 21 h, with a median of 28 ticks (as in #2576).
7. **`orion_biometrics_summary.timestamp`** is TEXT with a space separator, while `orion_metacog.timestamp` uses `T`. Comparing them as text silently dropped 3/4 of the cabinet rows in the first export. Both are now cast.

## Files changed

- `orion/schemas/temporal_self.py`: the stored models. Every enum value has a producer; dead-source values were removed and are listed in the module docstring.
- `orion/schemas/registry.py`: 10 `_REGISTRY` entries.
- `orion/temporal_self/day.py`: `day_id`, the half-open local day window (reusing `orion_day_window`), `day_phase` (pinned minute by minute against `TimeContextV1`'s helper), and the timestamp casts.
- `orion/temporal_self/sources.py`: one adapter per source kind, each documenting its time column, cast, subject and drop rule.
- `orion/temporal_self/broadcast.py`: `BroadcastTickView` extraction.
- `orion/temporal_self/arcs.py`: `fold`, `advance_clock`, `close_day`, `drain_closed_days`, the lane rules, and context binding.
- `orion/temporal_self/frame.py`: `build_frame`.
- `orion/temporal_self/body.py`: `summarize_body`.
- `orion/temporal_self/README.md`
- `orion/temporal_self/tests/`: 62 tests.
- `orion/temporal_self/evals/run_arc_precision_eval.py`, `export_fixture_day.sql` (a read-only `BEGIN READ ONLY … ROLLBACK`), `fixtures/day_2026-10-09.jsonl.gz`, and `fixtures/day_2026-10-09.labels.json`.
- `scripts/check_inner_state_registry.py`: two entries. `ArcAttentionSummaryV1` is a sub-object counting the already-registered `attention_schema.v1` rows. `TemporalSelfStateV1` is a name collision on "selfstate", not felt state.
- `.github/workflows/temporal-self-tests.yml`: gate tests plus the eval, path-filtered.

## Schema / bus / API changes

- **Added:**
  - the schemas `TemporalSelfEventV1`, `TemporalSelfArcV1`, `TemporalSelfFrameV1`, `TemporalSelfDayV1` and `TemporalSelfStateV1`;
  - the sub-objects `ArcSummaryV1`, `ArcAttentionSummaryV1`, `ArcBodySummaryV1`, `OpenThreadV1` and `ExpectationRefV1`.

  All are `extra="forbid"` and in `_REGISTRY`. None is a bus payload.
- **Removed / renamed:** none.
- **Behaviour changed:** none at runtime. The reducer has no caller yet.
- **Deviations from the spec's schema, each with its reason:**
  - **Fields added:**
    - `ended_at` on events (process rows are emitted once complete);
    - `segments` and `last_seen_at` on arcs (dwell and returns need them);
    - `source_gap_resumes` on arcs (a broadcast recorder outage is not a return);
    - `privacy_class` on arcs (outward boundary for concern and conversation arcs);
    - `warnings` and per-list `*_overflow` counters on arcs;
    - `expires_at` on expectations;
    - `active_by_kind`, `arcs_today_total`, `unbound_context_*`, `*_overflow`, `expectations_resolved_total` and `skipped_at_or_before_watermark` on the frame (caps and late rows never hide a drop).
  - **Not built:**
    - `ArcSelfModelSummaryV1` (see above);
    - the stance cue and curiosity facts projections (they land with their consumers in patch 4, so they are not ornaments here);
    - `fold_broadcast_ticks` / `fold_events` as separate calls (called one after the other, the second would skip everything at or before the first call's last key; ticks and events go through one `fold` per watermark);
    - `peak_pressure_max` and `cooling_switch_changes` (they failed the metric gate).
- **Compatibility:** new tables in patch 3 store these by `schema_version`.

## Env/config changes

- Added keys: none. `K`, `R`, the idle bar and the other values are `ReducerConfig` defaults. Patch 3 maps them to `TEMPORAL_SELF_*` keys.
- Removed / renamed keys: none.
- `.env_example` updated: no.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: not applicable, because no template changed.
- Skipped keys requiring operator action: none.

## Metric quality gate (CLAUDE.md 0A), every number the frame exposes

| Output | 1 Provenance | 2 Independence | 3 Theory anchor | 4 Live sanity (10-09) | Verdict |
|---|---|---|---|---|---|
| `attention_returns` (attention) | resumes counted in `_open_or_resume`, over `substrate_attention_broadcast_log` row order | no existing equivalent: `dwell_ticks` resets on restart and has no return notion | current concerns (Klinger 1975), interrupted tasks (Zeigarnik 1927) | 0 on 34/49 arcs, max 3; rest = 0 is common | **bound** |
| `attention_returns` (interoception) | the same rule over `field_dominance_run` rows | different id space from the workspace (0 overlap) | same | 0 on 9/44, ≥2 on 28/44, max 53 (the `bus_synaptic` hogging) | **bound**; high values are a true report of a hogging instrument, as #2576 found |
| `cumulative_dwell_sec` | the sum of segment spans; each span is a difference of source timestamps | within one uninterrupted stretch it is a monotone transform of `dwell_ticks`, so **redundant there**; across suspensions and restarts it is new | Event Segmentation Theory (Zacks et al. 2007) | attention 62 s–40 min (median 7 min); interoception 10 s–4.1 h; can be small; an outage is not counted | **bound** |
| conversation / concern dwell | the same | — | — | conversation 0 on 10-09 (Juniper's two turns were 3 h apart); concern is 0 by construction (a raise is a point) | **bound, but reads 0 today**. Not degenerate by construction for conversation; concern dwell is defined as 0 |
| `arcs_today_total`, `rows_by_lane` | row counts over one source each | counts of the sources' own records | none needed (counts) | 291 arcs on 10-09; substrate lane 81 rows in one 85-minute arc | **bound** |
| `chassis_watts_mean` | `orion_biometrics_cluster.chassis_watts` | an existing sensor at arc altitude | effort / metabolic cost (spec: regulators run on accumulated effort) | 1,093–2,007 W per arc, 106 distinct values; rest is the idle draw | **bound** |
| `cabinet_temp_c_min/max` | athena `orion_biometrics_summary.measurements.cabinet_temp_c` | the existing sensor | thermal load that already gates diffusion | 24.4–30.3 °C per arc, 73 distinct values over 105 arcs | **bound** |
| `ambient_spike_count` | `cabinet_ambient_spike` rows | the existing detector | — (a count) | 0 median, max 22, 9 distinct | **bound** |
| `thermal_refusals` | visual deferrals with `thermal_state='hot'` | the gate's own stamp | the thermal gate | 0 on 10-09; 12 `deferred_thermal` attempts in 30 days | **bound** (rare, rests at 0) |
| `peak_pressure_max` (spec) | `orion_biometrics_cluster.peak_pressure` | — | — | **fails**: `disk_capacity` pins the cluster peak at 0.813 on 78% of rows, and `power` saturates it at 1.0 (13%); the per-arc max had median 1.0 and floor 0.812, so it can never read calm | **not built** |
| `cooling_switch_changes` (spec) | `home_cooling_sample.switch_on` | — | — | **fails**: `switch_on` true on every sample since 09-26, 0 transitions in 30 days | **not built** |
| self-prediction accuracy (spec) | the calibration script's rule | — | — | **cannot be checked**: the 66.0% rows aged out | **not built** |
| `self_change_event_ids` | action outcomes, plus reverie verdicts other than `unscored` | — | — | first version: 251 of 256 entries were `unscored` (meaning "no verdict reached"), and the list was capped; fixed to 8 real changes, with an overflow counter | **bound after fix** |

Step 5, the existing-mechanism check: `orion_day` (an LLM letter, no arcs), `EpisodeSummaryV1` (15-minute counts, no subject), `recent_attention_cue` (ages of rows), the #2576 replays (offline, interoception only). None of them binds across processes or across a day.

Step 6, reversibility: no consumer, no table, no env key. Removing it means deleting the package and the registry lines.

## Tests run

```text
/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest -q orion/temporal_self/tests
73 passed in 3.8s
PYTHONPATH=. /tmp/orion-focus-runs-venv/bin/python -m pytest -q orion/temporal_self/tests   # CI-like: pytest + pydantic 2.10.3 only
73 passed
python scripts/check_metric_lineage.py --gate                         metric lineage gate: PASS
python -m pytest -q tests/test_inner_state_registry_gate.py           9 passed
python scripts/check_inner_state_registry.py                          inner_state_registry gate OK (20 entries checked)
python -m pytest -q <30 tests/ files that import the schema registry> 335 passed, 2 failed
  failing on clean main too (unrelated): test_autonomy_goals_bus_catalog::test_goal_proposal_schema_in_registry
  (GoalProposalV1.drive_origin), test_memory_crystallization::...registry_gap
git diff --check: clean
```

## Evals run

```text
python orion/temporal_self/evals/run_arc_precision_eval.py            passed: true (numbers above)
python orion/temporal_self/evals/run_arc_precision_eval.py --sweep    K/R table above
python orion/temporal_self/evals/run_arc_precision_eval.py --timeline worked example above
fixture export: docker exec -i orion-athena-sql-db psql -XqAt -v ON_ERROR_STOP=1 -U postgres -d conjourney \
  -v start="'2026-10-08T06:00:00Z'" -v end="'2026-10-10T06:00:00Z'" -v body_start="'2026-10-09T06:00:00Z'" \
  < orion/temporal_self/evals/export_fixture_day.sql | gzip -9n > .../day_2026-10-09.jsonl.gz
```

One export attempt was chosen as a deadlock victim. It was the read-only transaction (`AccessShareLock`) against a concurrent `AccessExclusiveLock` holder, probably a retention job. Postgres cancelled the read and the other process went on. Nothing was written, and a re-run succeeded.

## Docker/build/smoke checks

```text
None. No service, image or runtime path changed. Only read-only psql exports were run.
```

## Review findings fixed

A code-review subagent reviewed `feat/temporal-self-chronology-reducer` at 2daa0aedf plus the working copy, with probe scripts. It found 2 blockers, 8 should-fix and 6 nits.

- **Finding (blocker):** one-pass and chunked folds disagreed when the broadcast log went quiet before midnight. Only `advance_clock` suspended a silent attention arc, so one pass closed it `day_boundary` and the chunked fold closed it `return_window_expired`.
  - Fix: the silence rule now lives in `_expire`, which `fold`, `advance_clock` and the midnight roll all apply.
  - Evidence: `test_broadcast_silence_before_midnight_replays_identically`.
- **Finding (blocker):** `advance_clock(now == watermark)` moved `last_key` backwards, so a row stamped exactly `now` was folded twice.
  - Fix: `last_key` never moves backwards.
  - Evidence: `test_advance_at_the_exact_watermark_never_refolds`.
- **Finding:** calling `fold_broadcast_ticks` and then `fold_events` silently dropped the events.
  - Fix: both wrappers are removed. One `fold` per watermark is the documented contract.
- **Finding:** late or re-read rows were dropped with no trace.
  - Fix: `skipped_today` counts them, shown as `frame.skipped_at_or_before_watermark` with a warning.
  - Evidence: the updated refold and restart tests assert the count.
- **Finding:** timestamps not in UTC broke the sort key and `frame_id`.
  - Fix: events, ticks, `advance_clock` and `build_frame` all normalise to UTC.
  - Evidence: `test_non_utc_inputs_order_by_instant`.
- **Finding:** one piece of context was credited to two attention arcs while a new subject was building its K ticks.
  - Fix: an open attention arc reaches forward only while its subject won the latest tick. Flicker minutes belong to no arc and fall to the day.
  - Evidence: `test_context_while_another_subject_builds_is_not_credited_to_two_arcs`, `test_flicker_minutes_belong_to_no_attention_arc`.
- **Finding:** recorder outages counted as attention returns.
  - Fix: a resume after a gap increments `source_gap_resumes`, not `attention_returns`. A real interruption after the gap still counts.
  - Evidence: the updated gap and restart tests, plus `test_gap_then_another_subject_then_return_is_a_real_return`. The 10-09 numbers are unchanged.
- **Finding:** the reverie adapter emitted chains still running.
  - Fix: it requires `terminal_reason`.
  - Evidence: the adapter test.
- **Finding:** caps dropped data with no counter.
  - Fix: `expectation_overflow`, `constraint_overflow`, `interruptions_overflow` and `segments_merged` on arcs, and `correlation_overflow` in state with a frame warning. Segments are capped at 256.
  - Evidence: `test_expectation_overflow_is_counted_on_the_arc`, `test_segments_are_capped_and_merges_counted`.
- **Nits fixed:**
  - A concern continuation measures returns from the real last raise.
  - The closed day's final frame shows what was active at midnight.
  - Concern and conversation arcs carry `privacy_class=juniper_chat`.
  - The I/O ban in the purity test now also covers `os`, `subprocess`, `pathlib` and `open(`.
  - The CI push paths are widened.
  - Tick labels are clipped on construction.
  - Parked sleep links survive midnight.
  - `raised_by` was removed: nothing read it.
- **Eval independence:**
  - Ticks are now built by `tick_from_log_row` from the raw projection, graded against an SQL-computed `oracle_ref`.
  - The thought-to-chain oracle uses its own query.
  - Precision is reported per kind.
- **Declined: auto-suspending an interoception arc on silence.**
  - `field_dominance_run` writes a run only when it ends, so while a long run is in progress nothing is known yet.
  - Suspending on silence would turn an uninterrupted 30-minute run into a fake return.
  - An open interoception arc therefore means "the last completed focus". This is documented on `ReducerConfig.interoception_gap_sec`.

## Restart required

```text
No restart required.
```

## Risks / concerns

- **Medium: the eval covers one real day.** 10-09 had only two Juniper turns, so conversation returns and dwell read 0. The rules are pinned by fixtures, but a busy chat day has not been replayed.
- **Low: interoception "active" lags by one run.** S2 writes a run only when it ends, so the frame's interoception arc is the last completed focus, not the focus right now.
- **Medium: dwell for arcs crossing midnight.** A process arc that began before midnight belongs to the day it completed (`began_at` stays truthful). The pre-midnight context buffer is flushed at the day roll, so such an arc cannot bind context from before midnight.
- **Low: late rows.** `advance_clock(now)` declares nothing strictly before `now` will arrive. Patch 3's driver must read with a small lag, or a row committed late with an older timestamp is dropped by design.
- **Low: fixture size** is 980 KB gzipped. There is precedent (`services/orion-substrate-runtime/evals/fixtures/*.jsonl.gz`).

## What patch 3 needs

- A durable-runs self-driven thread that calls ONE `fold` per watermark with both ticks and events. It:
  - reads each bound source by keyset cursor, with a lag;
  - shapes rows with `sources.py` and `broadcast.py`;
  - calls `fold` → `advance_clock` → `build_frame`;
  - drains closed days.

  Its SQL must emit the flag columns the adapters expect (`has_prompt`, `unsolicited`, `chat_turn` EXISTS, the joined thought list, `trigger_timestamp`). `export_fixture_day.sql` is the working reference for every query.
- The `temporal_self_*` migration (events, arcs, day, projection, cursor) and the `GET /temporal-self/*` routes.
- Env keys mapped onto `ReducerConfig`:
  - `TEMPORAL_SELF_ARC_MIN_TICKS=3`
  - `TEMPORAL_SELF_RETURN_WINDOW_MIN=30` (changed from the spec's 180)
  - a conversation window of 180 min
  - `ORION_SITUATION_TIMEZONE`

  Flags ship on, and the local `.env` is synced.
- A per-closed-arc body read (cluster, cabinet, spikes) feeding `summarize_body`.
- A FalkorDB read for `:PriorRevision` and peer asks, if those are wanted (adapters still to write).
- Live proof: one arc with returns ≥ 2 whose evidence resolves in two tables, one closed day, and cursors advancing.

## PR link

(see below)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
