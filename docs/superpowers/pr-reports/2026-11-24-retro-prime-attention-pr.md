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

- 277 ticks had an open sitting. On 247 of them (89%) play sat in the top tenth of its own week: Orion's point 10 holds.
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
TESTS_PLACEHOLDER
```

## Evals run

```text
EVALS_PLACEHOLDER
```

## Docker/build/smoke checks

```text
Not run. Deploys happen only from the clean primary main checkout, after merge. No migration. All SQL the new code sends
ran against a throwaway Postgres 16 (127.0.0.1:55991, stopped after).
```

## Review findings fixed

REVIEW_PLACEHOLDER

## Restart required

```text
No manual restart: the post-pull rebuild rebuilds orion-hub (hub-app and the Retro gateway), orion-mind,
orion-substrate-runtime, orion-attention-runtime, orion-durable-runs and orion-sql-writer (their images hold the
changed orion/ code and config). No migration and no grant.
```

## Risks / concerns

RISKS_PLACEHOLDER

## PR link

PR_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
