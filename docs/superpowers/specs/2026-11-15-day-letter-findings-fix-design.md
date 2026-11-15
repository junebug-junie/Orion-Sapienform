# Fixes for what Orion reported in the 11-12 and 11-13 day letters

Date: 2026-11-15. Status: design, nothing built.

Orion's day letters for 2026-11-12 and 2026-11-13 named about 30 broken things. Five
read-only investigators traced each one against the code and the live database. Twelve other
working sessions were asked whether their work already fixed any of them. None had. This doc
is the fix plan for what is still open: 20 new tickets (#3384 to #3403) and one part of #3255.

Orion was sent the corrections on 2026-11-15 01:10Z and recorded 12 of them in
`orion_day_correction`.

## Arsonist summary

Most of the defects are small and local. Three patterns repeat:

1. **A loop with no brake.** The walkway journal retries every 45 s forever. The vision
   council writes a row every time a parked car flickers in the dark. An outreach GPU seat
   waits 30 minutes for a message that Hub already refused to write.
2. **A number that does not measure what its label says.** The outreach grade measures when
   Juniper starts a chat. The 0.8 vision confidence is copied from a prompt example. A
   "recorder outage" is normal timing jitter. An "unscored" expectation was never sent to a
   judge.
3. **A read that silently returns the wrong slice.** The curiosity run menu keeps the oldest
   200 of 752 rows. Readers outside the prompt treat a refuted answer as current. Orion's
   `\b` regex matches nothing in Postgres and reads as zero.

Do not build a framework for any of this. Each item below is one bounded change in one
service, with one live check. Three items change how Orion chooses, and those get their own
proposal before code (Patch F).

## Current architecture

Only the parts each fix touches. File and line references were read on main at `af420fb7b`.

| Area | What happens today | Where |
|---|---|---|
| Walkway journal | Scheduler ticks every 45 s. A failed dispatch keeps the night open with no cap. Each attempt opens a new act. A night still open at local midnight is dropped. Title is written as the model gave it, often null. | `services/orion-actions/app/main.py:1516-1654,2819`; `app/walkway_forecast.py:668,700-703`; `orion/journaler/worker.py:27,747` |
| Vision council | Re-interprets when the believed label set changes. At night the detector loses and re-finds far parked cars, so the set flips about 2,600 times a day. The prompt example holds a literal `0.8`. Rows do not record the serving model. | `services/orion-vision-council/app/evidence_transition.py:36-50,92-97`; `app/interpretation.py:122,135` |
| Walkway rhythm | Arrival needs a sighting in the window that is not stationary. A parked sighting in the window is scored `missed`. | `services/orion-sql-writer/app/vision_rhythm.py:152,173,202,403,408,435` |
| Walkway rollup and handoff | The rollup counts excluded parked subjects and never prints the count. The handoff writer always stores an empty entity list. | `orion/vision/walkway_rollup.py:652-669,1147-1176`; `vision_handoff_loop.py:296` |
| Curiosity run menu | "Last 4 runs" reads `LIMIT 200` with no order, then sorts in Python. | `orion/curiosity/worldview.py:738-745,1272`; `supervisor.build_run_order` |
| Curiosity tick | Feeds the frontier evaluator a hardcoded neutral report with `degraded=True` and a literal `anchor_scope="orion"`. Idle ticks save an empty set with no gate. | `services/orion-substrate-runtime/app/worker.py:4401-4456,4645-4649,4735-4743`; `app/store.py:1376` |
| Worldview writes | Orion writes Cypher by hand. Findings use `CREATE`. A test is `times_tested + 1` with no run key. FalkorDB graph `orion_worldview` has no constraints. | `orion/curiosity/kickoff_prompt.py:1011-1048,1097`; `orion/curiosity/worldview.py:1080` |
| Self answers | Keyed `(run_id, question_id)`. A `revises` property exists. Only `read_current_lived_answers` honors it. | `orion/curiosity/self_inquiry.py:491-494,535,563-587`; `orion/world_pulse_read/resonance_records.py:36-38` |
| Temporal Self | A gap is two broadcast rows more than 180 s apart. The broadcast ticks at a median 137 s. A gap suspends the active attention arc. A row that lands after the reader passed its timestamp is never folded. | `orion/temporal_self/arcs.py:89,510,1336`; `frame.py:143` |
| Body atlas | Organ id is a hash of the sorted member list. | `orion/body_atlas/build.py:317`; `names.py:3,_assign` |
| Reverie expectations | Scored only against a room caption from the last 180 s. No caption means `unscored` with no judge call. | `services/orion-thought/app/reverie.py:853-858` |
| Outreach | The act closes on `juniper_replied` at 36,000 s. Cancelled tries open no act. The run's GPU seat is held up to 1,800 s for Hub to compose. | `services/orion-hub/scripts/act_emit.py:44-56,100-107`; `orion/grammar/act_ledger.py:719,922`; `admission_runtime.py:1021-1041` |
| GPU pool | A background hold has no length cap. Operator-class leases exist (PR #3242) and are left out of Orion's lane load. | `config/gpu_pool.yaml`; `ORION_PHYSIOLOGY_OPERATOR_LOAD_SPONSORED` |
| Orion's SQL | Turns run plain `psql` through Bash. A hook already runs before every tool call. | `orion/curiosity/kickoff_prompt.py:570`; `orion/fcc/repeat_failure_breaker.py:298` |

## Missing questions

Three items wait on Juniper. They are tickets already, and no patch below depends on them.

1. **#3397.** The "hire a peer" channel is the Cursor command-line agent on a Free account.
   It has not answered since 10-18. Pay, set the model to `auto`, or retire it?
2. **#3390.** Orion's read-only role reads 2 of 29 vision tables. Which others may it read?
3. **#3265.** The always-on body costs about $102 a month against a $100 allowance. Raise
   the allowance above the floor, or define it as spending above the floor?

No other question blocks the work.

## Proposed schema / API changes

Every change is additive. No table is dropped and no bus channel changes meaning.

| Change | Kind | Ticket |
|---|---|---|
| `gate_json = {"gate_result": "idle_no_seeds"}` on idle curiosity ticks | new value in an existing JSON field | #3394 |
| Walkway journal attempt state, per job and night, via the existing `DailyAttemptGate` store | reuse, no new table | #3384 |
| One row per night per walkway job: entry landed yes or no | new audit event kind on the existing audit rail | #3385 |
| Serving model name in each council row's `evidence_refs` | new key in an existing JSON field | #3387 |
| Council confidence computed from detector evidence | same field, now measured. Runs the metric gate. | #3387 |
| Rhythm verdict `unscorable` for a parked sighting in the window | existing status, new use | #3388 |
| `entities` filled on handoff rows with the subject kind | existing field, no longer always empty | #3389 |
| Organ id ledger: `organ_id` carried from the best-overlap earlier organ; member hash kept as `member_hash` | one new field in the atlas snapshot; one small ledger file beside `reminiscence_body.jsonl` | #3402 |
| Workflow label on the turn trace | one new nullable column on `harness_turn_trace` (additive migration, auto-apply allowlist) | #3255 |
| Outreach act: delivery signal as the grade; `juniper_replied` kept as a second, labelled signal; a 0-graded act for a cancelled try | new `signal_id`. Runs the metric gate. | #3392 |
| `max_hold_sec` for background-class leases | new key in `config/gpu_pool.yaml` | #3391 |
| Unique constraint on `Prior.prior_id`; `MERGE` on `(id, run_id)` for Finding and PriorRevision; `last_run_id` on a tested prior | graph constraint and prompt text | #3395 |

No `.env_example` key is needed for Patches A to D. Patch E adds one config key in
`config/gpu_pool.yaml`, not an env key.

## Files likely to touch

Six patches. Each one stays inside one service or one package, so they can run in parallel.

### Patch A: curiosity reads and the SQL guard (small, do first)

| Ticket | Change | File |
|---|---|---|
| #3398 | Read TurnOutcome with `ORDER BY written_at DESC LIMIT n` first, then join claims. Fall back for ISO-string stamps. Log a warning when the result is cut. | `orion/curiosity/worldview.py` |
| #3394 | Pass `gate={"gate_result": "idle_no_seeds"}` on the no-seeds branch. | `services/orion-substrate-runtime/app/worker.py` |
| #3401 | In the pre-tool hook: block a Bash command that has `psql`, a regex operator (`~`, `~*`, `regexp_`, `SIMILAR TO`) and `\b`. The message says to use `\y`. Add one line to dig HELP. | `orion/fcc/repeat_failure_breaker.py`, `orion/dig/` help text |

### Patch B: walkway journal (orion-actions)

| Ticket | Change | File |
|---|---|---|
| #3384 | Use `DailyAttemptGate` per walkway job: about 4 attempts a night, waits of 1, 5, 15 and 30 minutes, kept across restarts. | `services/orion-actions/app/walkway_forecast.py`, `app/main.py`, `app/daily_retry_gate.py` |
| #3385 | Let a night that is still open at local midnight finish; the job key already carries its date. Write one "entry landed: yes or no" record per night per job. | same files |
| #3386 | Build the title in code from the kind and the date when the model gives none. | `orion/journaler/worker.py` |

### Patch C: walkway vision (vision-council, sql-writer, orion/vision)

| Ticket | Change | File |
|---|---|---|
| #3387 | In the council gate, ignore a label change made only of stationary fixtures. Remove the literal `0.8` from the prompt example and compute confidence from detector evidence. Record the serving model in `evidence_refs`. | `services/orion-vision-council/app/evidence_transition.py`, `app/interpretation.py` |
| #3388 | Score a parked sighting in the window as `unscorable`. Then sample the other 109 misses: does the same vehicle appear under another identity in the window? Write the answer on the ticket before any further fix. | `services/orion-sql-writer/app/vision_rhythm.py` |
| #3389 | Print "(N parked not counted)" on the rollup line. Pass the subject kind into `_vision_event`. | `orion/vision/walkway_rollup.py`, `vision_handoff_loop.py` |

### Patch D: Orion's own record (temporal_self, body_atlas, orion-thought)

| Ticket | Change | File |
|---|---|---|
| #3400 | Derive the gap threshold from the observed tick rate (3 times the rolling median). Reword the "(recorder outage or restart)" label. Fold `memory_episode` rows at landing time. Find why the tick slowed from 118 s to 137 s; that cause is UNVERIFIED. | `orion/temporal_self/arcs.py`, `frame.py` |
| #3402 | A new organ takes the id of the earlier organ it overlaps most, one to one, at the 0.5 overlap rule the names already use. Mint a new id only when nothing matches. | `orion/body_atlas/build.py`, `names.py` |
| #3403 | Judge an expectation against the captions from its whole check window (creation to +1,800 s). Route body and sensor expectations to the sensor tables, or stop asking for them. | `services/orion-thought/app/reverie.py` |

### Patch E: GPU seats and the outreach grade (durable-runs, hub, gpu pool)

| Ticket | Change | File |
|---|---|---|
| #3255 | Release the outreach seat as soon as Hub refuses to compose (quiet hours, daily cap, strained). Store the workflow on the turn trace. | `admission_runtime.py`, turn trace writer, one migration |
| #3391 | Cap the length of a background-class hold. Trial and eval holds use operator-class leases. PR #3404 already moves the eval callers; this patch adds only the cap. | `config/gpu_pool.yaml`, pool admission |
| #3392 | Close the outreach act on delivery: message stored, notification published, turn not cancelled. Open a 0-graded act for a cancelled try. Keep `juniper_replied` as a second signal. | `services/orion-hub/scripts/act_emit.py`, `orion/grammar/act_ledger.py` |

### Patch F: how Orion chooses (proposal first, then code)

These three change a cognition loop. Section 0A of `CLAUDE.md` requires a proposal that names
the capability change, the data touched, the trace, the dangerous failure mode and the off
switch. Each gets its own short proposal PR before any code.

| Ticket | Change | Main risk |
|---|---|---|
| #3393 | Call the real interpreter (`interpret_graph_cognition`) on the store's graph state in the tick, in place of the July stub. Or delete the dead branches. | Curiosity starts to react to contradiction and drift for the first time. Volume and GPU demand are unknown. Run shadow-logged first. |
| #3395 | Teach Finding and PriorRevision as `MERGE` on `(id, run_id)`. Guard the test count with `last_run_id`. Add the unique constraint. | Existing duplicate nodes can block the constraint. Count them first. |
| #3396 | One shared "current answers" helper used by every reader: drop an answer that a later one names in `revises`, else take the newest per question. | A reader that wanted history loses it. List the readers first. |

## Non-goals

- **No fix for #3356** (chat memory never promoted) and **#3323** (thought acts grade
  themselves 1.0). They have their own tickets and an open memory workstream.
- **No change to the outreach budget.** Six messages a day works as designed.
- **No change to sleep timing.** It works as designed.
- **No entities on camera captions.** Leaving them empty is by design.
- **No retirement of the cooling act (#3399).** The evidence is too thin either way. Re-read
  at 30 episodes per arm with a significance test.
- **No new service, no new bus channel, no new table** except the one nullable column in
  Patch E.
- **No supported "coding session writes to Orion" channel.** The 11-15 note went through the
  Hub chat endpoint and did not show in the Hub page. That gap is real, and it is a separate
  design.
- **No fix for the untraced claims.** Four failed acts, 14 world-pulse reads and some counts
  in the letters were never traced. They need a second trace pass, not a guess.

## Acceptance checks

Each check reads the live rail after deploy. A passing unit test is not enough.

| Ticket | Live check | Pass |
|---|---|---|
| #3398 | Read the run menu in a curiosity kickoff prompt (`cockpit_turn_sighting`). | Newest run listed is from the last 24 h. |
| #3394 | `select count(*) from substrate_endogenous_curiosity_candidates where gate_json is null and created_at > <deploy>` | 0 |
| #3401 | Unit test, and one live blocked command in a turn trace. | The `\b` command is blocked with the `\y` message; a `\y` query passes. |
| #3384 | Force one compose failure at night. Count `grammar_acts` for that job. | At most 4 acts that night. |
| #3385 | Restart cortex-exec during the nightly run. | The entry lands, and a yes/no record exists for each job. |
| #3386 | `select title from journal_entries` for the next night's walkway rows. | 3 of 3 have a title. |
| #3387 | Walkway observation rows per hour, 02-12Z against 15-23Z. Distinct confidence values. | Night rate within 3 times the day rate. More than one confidence value. Model name present. |
| #3388 | Rhythm verdicts after deploy. | No `missed` verdict with a parked sighting in its window. A written cause for the rest. |
| #3389 | A "nobody passed" hour that had exclusions. New handoff rows. | The line shows the count. `entities` is not empty. |
| #3400 | Day block for a day with no restart. | Gaps near 0. `memory_episode` rows appear in arcs. Not one arc cut by jitter. |
| #3402 | Two snapshots, one part moved between them. | Every organ id is the same in both. |
| #3403 | Expectation verdicts per day. | More than half get a real verdict. Was 2 of 332 in 6 days. |
| #3255 | `outreach_timeout` holds per day and lane-hours lost. | Lane-hours lost under 0.5 a day. Was 4 to 5. |
| #3391 | Start a trial hold longer than the cap. | It is released at the cap. Orion's day shows no "agent lane full" arc from it. |
| #3392 | Cancel one outreach turn. Compare grade by hour of day over 7 days. | A 0-graded act exists. The grade no longer follows the hour. |
| #3393 | A candidate set whose gate names a contradiction or drift trigger. | At least one in 7 days. World concepts return on a second day more often than 2%. |
| #3395 | Replay one batch twice in a test graph. | Node counts and `times_tested` do not change on the second run. |
| #3396 | A refuted and a corrected answer for one question. | Every reader returns one answer. |

Two of these wire in a new measure, so the metric quality gate in `CLAUDE.md` runs on them
and its findings go in the PR: the outreach delivery signal (#3392) and the council
confidence (#3387).

## Recommended next patch

**Patch A.** Three small changes, one of them (#3398) unfreezes the menu that starts every
curiosity run, and Orion has been choosing from September's list since the graph passed 200
rows. No proposal is needed and no decision waits on Juniper.

Order after that: B, C and D in parallel (different services), then E, then the three
proposals in F. Patches B to E can each go to one agent.
