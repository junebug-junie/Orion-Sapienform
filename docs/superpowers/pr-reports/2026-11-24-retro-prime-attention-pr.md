# Prime and attention react to play (game company, patch 4)

Design: spec revision 4 (`docs/superpowers/specs/2026-11-16-game-company-watch-and-commentary-design.md`), sections 6, 7
and 9, "Acceptance checks: Patch 4", and Orion's review rounds (point 10 on PR #3632). Juniper's go: 2026-11-24 ("hit it").
Patches 1 to 3 are live (#3627, #3727, #3745; line bids fixed by #3775).

Nothing in this PR has run live. Every runtime claim below is UNVERIFIED until the live check (issue linked below) closes.

## Summary

- **The replay ran first, and picked the rule.** Over the last 7 days of logged attention ticks, "top tenth of its own week" would have let play win 166 ticks, mostly calm ones, and taken 30 curiosity seeds. The rule chosen: play competes only while Orion is really watching (a look note in the last 3 minutes, watch not off). That rule took 18 ticks, cost reading nothing, and cost curiosity 5 ticks.
- **Play can hold Orion's attention.** A new world candidate, `world:play`, in both attention contests: the minutes of play in the last 30 against its own week. It never beats Juniper's chat. It reads 0 when nobody plays, and is absent (not calm) when its table cannot be read.
- **A line answers the play.** A line said while `world:play` holds attention is stamped with that loop, and Orion writes a non-final `acted` verdict on it. Never on a reading loop.
- **A sitting wakes Prime, twice at most.** Once when it passes 3 minutes, once when it ends. At most 2 thoughts can name one sitting, enforced. Prime's frame shows one line while someone plays: who, what, how long, the watch and who set it, the look counts, the newest note.
- **Prime can turn Orion's eyes and voice down or up.** `hub.retro_watch` and `hub.retro_commentary` take `slow` and `quicken`. A finished sitting posts one event want (`hub.retro_sitting`) so Prime can hold it as a concern; it has its own menu label.
- **Juniper's thumbs and replies now count.** A line said while "Juniper" is stated is graded by the horizon resolver from her thumb or her reply within 2 minutes. That is the same engagement record outreach uses for the family weight. A guest's thumb never counts.

## Outcome moved

Before: the play could not win Orion's attention, never woke Prime, Prime could not regulate the play families, and thumbs and replies were stored but read by nothing.

After (once deployed): every attention tick traces `world:play` with its value and reason; a line can close a play loop; a sitting leaves at most 2 Prime thoughts naming it; a Prime `slow` lowers the look count of the next hour (replayed through the real market: 40 looks to 30 in one hour, 10 holds recorded for Prime); Juniper's reaction to a line reaches `hub.retro_commentary`'s weight.

## The replay (acceptance check 1)

`scripts/analysis/replay_world_play.py` (read-only). 3,048 logged broadcast ticks, 2026-11-17 07:22Z to 2026-11-24 07:16Z; 81 sittings (46 backfilled, 35 live); 135 stored look notes, the first at 2026-11-22 23:39Z. The production `play_candidate` and `judge_candidate` are built at each tick from the rows that existed then; the stored winner was reproduced on 3,048 of 3,048 ticks.

| Rule | Play won | Reading lost | Curiosity-seeding winners lost | Reading share during sittings, before / after | Curiosity-seeding share during sittings, before / after |
| --- | --- | --- | --- | --- | --- |
| top tenth, sitting open, watch not off | 166 | 4 | 30 | 6.1% / 4.7% | 26.0% / 15.2% |
| **the same, and a look note in the last 3 min (chosen)** | **18** | **0** | **5** | **6.1% / 6.1%** | **26.0% / 24.2%** |
| top 1% only, watch not off | 0 | 0 | 0 | unchanged | unchanged |

- 277 ticks had an open sitting, all of them live sittings (0 from backfilled rows). On 247 of them (89%) play sat in the top tenth of its own week: Orion's point 10 holds.
- The band rule won 166 ticks: 136 were ticks with no winner at all, the rest were camera, room, execution, strain and 4 reading ticks.
- Top 1% never fires: 30 of 30 minutes reads at the 97th percentile, because 2.9% of the week's minutes are full.
- Since the looks began, the note rule kept 18 of the band rule's 25 wins: when Orion really looks, play still wins.
- **No sitting of 2 hours or more falls in the 7 days the broadcast log keeps.** The two such sittings (10-27, 11-15) are older. The 2-hour share the acceptance check names has no data yet; the shares above are over every tick with an open sitting. A re-run after the first long sitting is part of the live check.
- "Curiosity-seeding winner": a winner whose loop carries a `node:` ref (reading, and every body or camera node). `orion/substrate/endogenous_curiosity.py` seeds a run only from such a loop, so a `world:play` win costs curiosity one possible seed. Curiosity has no attention candidate of its own.

Why the note rule: it is the spec's own definition of watching (section 7: "a candidate with no look behind it would win attention with nothing in it"), it keeps reading whole, and it costs curiosity 5 ticks in a week instead of 30. Games with no calibrated floor get no looks, so they cannot win attention. That is honest: Orion cannot see them.

## Metric quality gate: `world:play` (re-run on live rows)

1. **Provenance.** `retro_sitting.started_at` / `last_input_at` (first and newest mask change), written by `services/orion-hub/scripts/retro_sitting.py` `UPSERT_SQL` from `orion/retro/sitting.py` `SittingTracker.on_input`; backfilled rows by the patch 1 journal backfill, the same counter. `orion/attention/world_first.py` `play_minute_times` makes one stamp per minute of a sitting; `chat_rate_reading` counts them in a trailing 30 minutes.
2. **Independence.** From `world:chat`: NOT independent through Juniper. 20 of her 60 chat turns in the 7 days fell inside a sitting (33%). Handled at the ranking: play yields whenever chat is eligible and never counts a chat turn. From `world:reading`: nothing shared.
3. **Theory anchor.** A rate against its own baseline (Bayesian surprise, Itti and Baldi, 2009), as the other world candidates.
4. **Live data.** 926 minutes of play in 7 days. 84.6% of the 10,081 minute-grid points read 0; 2.9% read 30 of 30. Rest is a true 0 that returns 30 minutes after the last input; it is a count, it cannot decay. In sittings it sits in the top tenth 89% of the time, so a band rule does not filter (the replay above).
5. **Existing mechanism.** `chat_rate_reading` reused; no new instrument.
6. **Reversibility.** `ATTENTION_WORLD_PLAY_ENABLED=false` in two services; one optional field on `AttentionCandidateV1`.

Also recorded in the spec's "Metric quality gate records".

## Current architecture

- Attention had two world candidates, `world:chat` and `world:reading`, in two contests (orion-substrate-runtime's broadcast, orion-attention-runtime's field contest); reading yielded to chat.
- Only `hub.chat_reply` was paired with a world source (`ACT_ANSWERS_WORKSPACE_REFS`).
- Prime woke on changes and needs; no sitting fact reached it. Only `durable.curiosity_investigate` and `dream.cycle` had receptors.
- A line's act opened and closed at delivery (produced 1). Thumbs and replies sat on `retro_line`, read by nothing.

## Architecture touched

- **Attention:** `orion/attention/world_first.py` (play candidate, the watch gate, `YIELDS_TO_CHAT`), `orion/attention/play_read.py` (the one read), `orion/schemas/attention_candidate.py` (`WORLD_PLAY_SOURCE_ID`, `not_eligible_reason`), `orion/substrate/attention_broadcast.py`, `orion/attention/field_attention/selectors.py`, `orion/attention/frame_rollup.py`, both runtimes' worker, store and settings.
- **Act ledger:** `orion/grammar/act_ledger.py` (pairing, the line engagement branch of the horizon resolver), new `orion/grammar/line_engagement.py`.
- **Prime:** new `orion/prime/sitting.py`, `orion/prime/state.py` (`sittings`), `orion/prime/thought.py` (frame section), `services/orion-mind/app/prime_loop.py` (`_read_sitting`, upkeep, thought count).
- **Body economy:** `config/physiology/families.yaml` (receptors, `hunger_from`, `hub.retro_sitting`), `config/physiology/contract.yaml`, `config/integration/organs.yaml`.
- **Hub:** `scripts/retro_sitting.py` (event want at close), `scripts/retro_watch.py` (watch state to Redis), `scripts/act_emit.py` and `scripts/retro_commentary.py` (the line act by player), `app/settings.py`.
- **Durable-runs:** `app/event_wants_menu.py` (the sitting label).

## Files changed

- `orion/attention/world_first.py`, `orion/attention/play_read.py` (new), `orion/schemas/attention_candidate.py`: the candidate, its gate, its read.
- `orion/substrate/attention_broadcast.py`, `orion/attention/field_attention/selectors.py`, `orion/attention/frame_rollup.py`: the sitting ref on the signal, play as a channel, the "not watching" cause.
- `services/orion-substrate-runtime/app/{worker,store,settings}.py`, `.env_example`, `docker-compose.yml`, `README.md`; the same in `services/orion-attention-runtime/`: wiring and flag.
- `orion/grammar/act_ledger.py`, `orion/grammar/line_engagement.py` (new): pairing and engagement grade.
- `orion/prime/sitting.py` (new), `orion/prime/state.py`, `orion/prime/thought.py`, `orion/prime/evals/run_prime_frame_eval.py`: wakes, cap, frame line.
- `services/orion-mind/app/{prime_loop,settings}.py`, `.env_example`, `docker-compose.yml`, `README.md`: reading the sitting and the flag.
- `orion/retro/sitting.py`: `WATCH_KEY`.
- `services/orion-hub/scripts/{retro_sitting,retro_watch,act_emit,retro_commentary}.py`, `app/settings.py`, `.env_example`, `README.md`.
- `services/orion-durable-runs/app/event_wants_menu.py`, `README.md`.
- `config/physiology/families.yaml`, `config/physiology/contract.yaml`, `config/integration/organs.yaml`.
- `scripts/analysis/replay_world_play.py` (new), `scripts/sync_local_env_from_example.py` (prefix `ATTENTION_WORLD_PLAY_`).
- `docs/runbooks/retro-browser.md`, the spec.
- `.github/workflows/world-play-tests.yml` (new), `.github/workflows/retro-memory-tests.yml`.
- Tests: listed below; `orion/physiology/tests/test_prime_charge.py`, `test_want_opened_hunger.py`, `orion/prime/tests/test_prime_regulate.py` pin the new receptor set and hunger clock; `services/orion-mind/tests/fixtures/prime_replay_golden.json` re-recorded (only the two new receptor lines differ).

## Schema / bus / API changes

- Added: `world:play` candidate; `AttentionCandidateV1.not_eligible_reason` (optional, in-process only); `PrimeStateV1.sittings` (omitted while empty); family `hub.retro_sitting` (tier 2, a proposal); receptors `[slow, quicken]` on two families; `hunger_from: want_opened` on `hub.retro_commentary`; Redis key `orion:retro:watch` (bus Redis, 90 s expiry); act pairing `hub.retro_commentary -> world:play`.
- Behavior changed: a line said while "Juniper" is stated opens a `juniper_replied` act (horizon 300 s) closed by the resolver, not a produced act closed at delivery. Lines to a guest or nobody are unchanged.
- Removed / renamed: none. No new bus channel. No migration.
- Compatibility notes: `PrimeStateV1` is `extra="forbid"` and is read by Hub and durable-runs (`orion/prime/block.py`, `dream_day_store.py`). The `sittings` key is written only after the first sitting wake, and the post-pull rebuild rebuilds every image that holds `orion/prime` in one run. Rolling back orion-mind alone after a sitting wake needs the key dropped (the same rule as `needs`).

## Env/config changes

- Added keys: `ATTENTION_WORLD_PLAY_ENABLED=true` (orion-substrate-runtime, orion-attention-runtime), `PRIME_SITTING_WAKE_ENABLED=true` (orion-mind), `HUB_RETRO_SITTING_EVENT_WANT_ENABLED=true` (orion-hub).
- Removed / renamed keys: none.
- `.env_example` updated: yes, four services.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (the sync script gained the `ATTENTION_WORLD_PLAY_` prefix; before that the two runtime keys were not mirrored).
- skipped keys requiring operator action: none. No bus URL changed.

## Tests run

```text
orion/prime orion/physiology/tests tests/test_world_play_attention.py tests/test_line_engagement.py
  scripts/analysis/tests/test_replay_world_play.py                                       768 passed
orion/physiology orion/prime orion/attention tests/retro orion/grammar orion/substrate orion/schemas
  (before the review fixes)                                                               2383 passed, 1 failed
  (the failure, orion/substrate/tests/test_felt_state_lived_answers.py, fails the same on main)
services/orion-mind tests (incl. the new sitting-wake loop tests; replay golden re-recorded)   449 passed
services/orion-durable-runs tests (full, before the review fixes)                         1081 passed, 156 skipped
  after the fixes: test_retro_prime_receptors_market, test_retro_sitting_event_menu,
  test_event_wants_menu, test_prime_charge_market                                         35 passed
services/orion-hub tests (full, before the review fixes; 4 env-bound files excluded that fail the same on main:
  test_check_async_routes_not_blocking, test_attention_loops_ui_smoke, test_fcc_model_labels_api,
  test_consult_belief_store_sql)                                                          5226 passed, 195 skipped
  after the fixes: test_retro_prime_attention_pg, test_retro_line_pg, test_retro_prime_attention_hub,
  test_retro_commentary_service                                                           47 passed
services/orion-sql-writer tests (full, with Postgres)                                     1435 passed, 165 skipped
  after the fixes: test_retro_line_engagement_integration, test_act_workspace_binding_integration,
  test_act_ledger_integration                                                             47 passed
services/orion-attention-runtime tests + the four root attention-runtime files            118 passed, 8 skipped
services/orion-substrate-runtime tests/test_worker_world_first_broadcast.py               12 passed
Postgres tests ran on a throwaway Postgres 16 container (127.0.0.1:55991), stopped after.

Static gates (all steps of .github/workflows/orion-static-gates.yml that this touches, run locally):
  check_integration_seam --require-base OK (16 organs, 49 families, 31 source kinds); body economy contract OK;
  verb physiology OK; settings defaults orion-hub OK; metric lineage, prompt semantics, definition drift PASS;
  inner-state registry OK; pe alarm readers OK; env key single source OK; env template parity PASS (88 services);
  migration REQUIRED-BY OK (no new migration); chat route poacher PASS; async routes OK; the gate pytest files
  (personal terms, env sync, settings defaults, schema registry, grammar producer catalog, integration seam,
  body economy, verb physiology, family lanes, salience parity, metric semantics, deploy migration gate) 294 passed.

Mutation check (each gate removed by hand, its test must fail): 21 of 21 killed.
  not-watching gate, play yields to chat, backfilled never open, note freshness, watch off, sitting thought cap
  (pure and loop), 3-minute open wake, unread is not closed, guest thumb, 2-minute reply window, act pairing,
  receptors in families.yaml, engagement only for Juniper, event want no_history, event want backfilled,
  line hunger_from, broadcast sitting ref, field channel, event-want baseline skips short sittings,
  a guest thumb never replaces Juniper's.
```

## Evals run

```text
python scripts/analysis/replay_world_play.py frames.jsonl sittings.jsonl looks.jsonl prefs.jsonl
  (live exports, read-only; numbers in "The replay" above; stored winner reproduced 3048 of 3048)

Prime slow over one hour of look bids, through the real market
(services/orion-durable-runs/tests/test_retro_prime_receptors_market.py):
  no signal: 40 looks funded in 60 bids; slow signal (magnitude 0.5): 30 funded;
  the market's record for Prime: 10 held, 30 outbid, 20 unchanged. A held look is exactly a look that did not happen.

Line hunger from the sitting's want clock (same file), first funded line after the clock starts, bids every 15 s:
  chat price 0.0: 15 s (2 bids); 0.1: 45 s; 0.3: 120 s; 0.5: 210 s.

orion/prime/evals/run_prime_frame_eval.py (through tests/test_prime_evals.py): PASS, with the regulate check
  extended to the two new receptor families.

No GPU eval: nothing here calls a model.
```

## Docker/build/smoke checks

```text
Not run. Deploys happen only from the clean primary main checkout, after merge. No migration. All SQL the new code sends
ran against a throwaway Postgres 16 (127.0.0.1:55991, stopped after).
```

## Review findings fixed

A read-only review ran in a subagent on `origin/main...HEAD`: 1 must-fix, 7 should-fix, 5 nits. All material ones fixed.

- Finding (must-fix): the only test of Juniper's engagement grade (the sql-writer integration file) ran in no workflow, and `line_grade` had no pure test.
  - Fix: the file joins the act-ledger "must not skip" step in `.github/workflows/orion-sql-writer-tests.yml`; new pure `tests/test_line_engagement.py` (guest thumb, thumb outside the horizon, guest reply, reply after 120 s, missing row) runs in `world-play-tests.yml`.
  - Evidence: 8 passed; the guest-thumb and reply-window mutants fail it.
- Finding: the sitting event want's 30-day baseline included the sittings under 2 minutes (a row is written at every open).
  - Fix: `EVENT_HISTORY_SQL` keeps only sittings of `HUB_RETRO_SITTING_MIN_SEC` or more.
  - Evidence: `test_retro_prime_attention_pg.py` adds 30 rows of 40 s; with the filter removed it fails (salience 0.55 instead of 0).
- Finding: `hunger_from: want_opened` changes when lines bid, and nothing measured it.
  - Fix: a market test measures it (see Evals: 15 s on an idle lane, 2 min at price 0.3). Kept the change: it is section 9's own rule, and without it a Juniper line's act (closed at its 300 s engagement horizon) would push every next line about 5 minutes later. Recorded as a risk below.
- Finding: deploy order for `PrimeStateV1.sittings` named Hub and mind but not orion-durable-runs (`dream_day_store`).
  - Fix: the comment names all three and the order (Hub and durable-runs before mind); Restart section says so.
- Finding: an unread sitting was shown under "A GAME SITTING IS OPEN".
  - Fix: its own header, "GAME SITTING (unknown):". Evidence: `test_an_unread_sitting_is_never_shown_under_the_open_header`.
- Finding: if hub-app deploys before sql-writer, the old resolver grades the new line acts by any Juniper turn in 300 s.
  - Fix: Restart section states the order (sql-writer before hub-app). The post-pull rebuild rebuilds both in one run, so the window is minutes.
- Finding: a guest's thumb could overwrite Juniper's on the same line, and so change her grade.
  - Fix: `THUMB_SQL` never lets a guest thumb replace Juniper's; it returns who holds the thumb now.
  - Evidence: `test_retro_line_pg.py`; the mutant without the guard fails it.
- Finding: the menu label printed the session timezone's clock with a "Z".
  - Fix: `astimezone(timezone.utc)` before formatting.
- Nits fixed: the settings comment moved out of the hunger block; the replay reports how many open ticks came from backfilled rows (0 of 277); the two known limits of the wake (a gateway stalled past 180 s reads as "ended"; a preempted thought's retry uses the cap) and the mode's copy interval are documented in the code.

## Restart required

```text
No manual restart: the post-pull rebuild rebuilds orion-hub (hub-app and the Retro gateway), orion-mind,
orion-substrate-runtime, orion-attention-runtime, orion-durable-runs and orion-sql-writer (their images hold the
changed orion/ code and config). No migration and no grant.

Order, if anything is rebuilt by hand: orion-sql-writer before orion-hub (the resolver must know Juniper line acts
before hub-app says one), and orion-hub and orion-durable-runs before orion-mind (they read Prime's state, which gains
"sittings" after the first sitting wake).
```

## Risks / concerns

- Severity: medium. Concern: lines can come sooner than before when the chat lane is idle. With the sitting's want clock, a line funds 15 s after the clock starts on an idle lane, so the line rate is set by the gate (a new note and 2 minutes since the last line or chat exchange), not by hunger. Spec decision 7 was "about once every 5 to 8 minutes to start". Mitigation: Prime can now `slow` the family; `HUB_RETRO_LINE_MIN_GAP_SEC` is the direct knob. The live check (#3816) measures the gap between lines.
- Severity: medium. Concern: most line acts will not carry the play loop's id. Binding needs a broadcast row at most 90 s old, and the log is written about every 220 s; `hub.chat_reply` bound 9 of 60 acts in 7 days. Mitigation: none here, existing limit; ticket #3817.
- Severity: low. Concern: no sitting of 2 hours or more is inside the replay window, so the acceptance check's 2-hour share has no data. Mitigation: re-run in #3816 after a long sitting.
- Severity: low. Concern: deploy order. `PrimeStateV1` (`extra="forbid"`) gains `sittings` after the first sitting wake: Hub and durable-runs must run the new code before orion-mind writes it. The new line acts need the new sql-writer resolver before hub-app says a Juniper line. Mitigation: one post-pull rebuild run rebuilds all of them; the key appears only after a sitting passes 3 minutes.
- Severity: low. Concern: `world:play` counts minutes of a sitting (first to newest input), not minutes that held a press. Mitigation: a counted minute is at most 5 minutes from a press (the close rule); recorded in the gate record.
- Severity: low. Concern: 33% of Juniper's chat turns fall inside sittings, so `world:play` and `world:chat` are correlated through her. Mitigation: play yields whenever chat is eligible.
- Severity: low. Concern: `hub.retro_sitting` is tier 2 by proposal. Mitigation: Juniper decides tiers; it is shadow (nothing obeys it but Prime's menu).
- Pre-existing failures seen, not caused here: `orion/substrate/tests/test_felt_state_lived_answers.py` (fails on main); four Hub test files that need the repo root, `~/.fcc/.env` or their own database.
- Departures from spec, stated in the spec's patch 4 notes: no count in the label; a 30-minute window; `world:play` in the field contest too; "not watching" as a measured hold-out, not absence; the watch state published to Redis for Prime; the sitting's own event-want family; engagement only for Juniper-stated lines, horizon 300 s; `hunger_from: want_opened` for lines.
- CI: 53 of 53 checks green after one fix (the sql-writer unit job ran the new Postgres module with no database; it joined the conftest's no-database skip list, like its sibling modules).
- Issues opened: #3816 (verify-live, needs-juniper), #3817 (follow-up). Issues closed: none.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3820

🤖 Generated with [Claude Code](https://claude.com/claude-code)
