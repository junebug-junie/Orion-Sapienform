# PR: memory confirmation loop -- "Orion is asking" about high-stakes memories

## Summary

When the shadow memory writer stores something sensitive (a "high-stakes" memory: health, family, how Juniper felt, a conclusion about who she is, or Orion's own conclusions about its machinery, its need for direction, or the relationship), Orion now asks her about it once, in the Hub's "Orion is asking" panel, and her answer changes the memory. Before this, the stakes label was written and nothing read it.

- **Asking.** orion-memory-consolidation opens one card per high-stakes memory, in plain English, quoting what Orion wrote down and saying why it is asking. At most 5 cards are open at once; the rest wait their turn. A card nobody answers closes after 7 days and the memory is marked "unconfirmed". Silence is never a yes.
- **Answering.** The panel moved to the top of the Hub home. Memory cards get **Confirm / Revise / Reject**. Revise opens a box with Orion's current wording and cannot be sent empty. An answer closes the card and writes the one resolution record (`attention_loop_outcome`) in a single database transaction, then announces it on the bus.
- **Applying.** orion-memory-consolidation applies the answer. Confirm: the memory becomes something they "worked out together" and is strengthened. Revise: her wording becomes a new confirmed memory, and the original is kept but superseded. Reject: the memory is marked rejected and excluded from recall. If the bus drops the announcement, the service picks the answer up from the table on its next tick (every 60 s).
- **Answering in chat: deferred.** See "Risks / concerns".
- **Side fix.** The walkway camera's daily ask limit (2 a day) used to count every row in the shared ask table. It now counts only camera asks, so memory cards cannot crowd out the camera's questions.

## Outcome moved

- The stakes labels now have a consumer. Each of the 7 high-stakes categories changes what the card says, and low-stakes memories are never asked about. Each card states which kind it is.
- The "Orion is asking" panel can now receive real cards from memory. It has had 0 rows since it shipped.
- Orion has a way to close a thought. Juniper's answer is recorded once (`attention_loop_outcome`, `loop_id = memory-confirm-<memory_id>`), and the card, the outcome and the memory event all join on that record's id.

## Current architecture

- The shadow distiller (orion-durable-runs, `orion/memory/episode/store.py`) writes `episode_memory` rows with `stakes`, `stakes_reason` and `confirmation_state='pending_confirmation'` for high-stakes rows. Nothing read these fields.
- `orion_ask` held only walkway-camera questions (orion-sql-writer). They were answered through `POST /api/asks/{id}/answer|dismiss` in the Vision panel.
- `AttentionLoopOutcomeV1` on `orion:attention:loop_outcome` was published by the Hub's Pending Attention panel, and no service consumed it (`consumer_services: []`).

## Architecture touched

- **Contract:** `orion:attention:loop_outcome` now lists orion-memory-consolidation as a consumer. The schema is unchanged; memory outcomes put `{resolution, ask_id, via, memory_id, prior_ids, related_loop_ids, node_ids}` in the existing free-form `features_at_close`. The metric definition lock was re-locked for that one routing change.
- **Shared library:** `orion/memory/episode/confirmation.py` holds the ids, card wording, opener, expirer, catch-up read and applier (asyncpg).
- **orion-memory-consolidation:** a ticker (expire, then catch up, then open) and a bus handler for the outcome channel.
- **orion-hub:** a new `POST /api/asks/{id}/resolve` route, a guard on `/answer` and `/dismiss` for memory cards, `memory_statement` added to `GET /api/asks`, the panel moved and given its new buttons.
- **orion-sql-writer:** the vision daily cap SQL.
- **Database:** indexes only (`manual_migration_memory_confirmation_v1.sql`). `confirmation_state` gains the value `unconfirmed`, and `status` gains `rejected`; neither column has a CHECK constraint.

## Files changed

- `orion/memory/episode/confirmation.py`: the loop, covering ids, wording, open/expire/catch-up/apply, and the `RECALLABLE_WHERE` predicate.
- `orion/memory/episode/report.py`: the daily report says each confirmation state in words.
- `orion/memory/episode/tests/test_confirmation.py`: tests for wording, source monitoring, ids, and that every category has its own consumer.
- `orion/memory/episode/tests/test_confirmation_pg.py`: real-Postgres tests of card creation, the cap, the expiry, each answer, the catch-up, the races, and the migration.
- `orion/schemas/ask.py`: docs for the new `source_kind`.
- `orion/bus/channels.yaml`, `config/metrics/metric_definitions.lock.json`: the new consumer, and the re-lock for it.
- `services/orion-memory-consolidation/app/confirmation_loop.py`, `app/main.py`, `app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: wiring, the flag, and the concept table.
- `services/orion-memory-consolidation/tests/test_confirmation_loop.py`: bus handler, flag-ON, and ticker tests.
- `services/orion-memory-consolidation/evals/run_memory_confirmation_replay_eval.py`, `evals/test_memory_confirmation_replay_eval.py`, `evals/results/2026-10-06-memory-confirmation-replay.json`: the replay eval (counts only).
- `services/orion-hub/scripts/ask_routes.py`: the resolve bridge, the guard, and `memory_statement`.
- `services/orion-hub/static/js/vision-asks.js`, `static/js/vision-asks.test.js`: Confirm / Revise / Reject.
- `services/orion-hub/templates/index.html`: the panel moved to the top of the Hub home, with a neutral subtitle.
- `services/orion-hub/app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: the flag and docs.
- `services/orion-hub/tests/test_ask_routes.py`, `tests/test_ask_resolve_pg.py`, `tests/test_orion_is_asking_browser_smoke.py`: route, end-to-end Postgres, and Chromium tests.
- `services/orion-sql-writer/app/vision_individuals.py`, `tests/test_vision_individuals.py`: the vision-only daily cap.
- `services/orion-sql-db/manual_migration_memory_confirmation_v1.sql` (+ `_rollback.sql`): indexes.
- `.github/workflows/orion-memory-episode-tests.yml`, `.github/workflows/schedule-browser-smoke.yml`: run the new Postgres and browser tests in CI.
- `docs/superpowers/specs/2026-09-30-memory-episode-redesign-design.md`: an "implemented, with these differences" note in section 3.

## Concept table

Every concept below has a producer, a consumer and a test. No stakes category was cut: the replay found 0 `health` and 0 `orion_machinery` cards in the sample, but both are in Juniper's rubric and both have their own card wording.

| Concept | Plain-English meaning | Producer | Consumer | Test |
|---|---|---|---|---|
| `stakes=high` | Orion should check with Juniper before keeping this as settled | distiller + `validate.resolve_stakes` | `confirmation.open_cards`: only high memories get a card | `test_high_stakes_memory_gets_one_card_and_low_gets_none` |
| `stakes_reason=health` | About her or her family's health | distiller (v3 rubric) | card line "It's about health, so I'd rather check than assume." | `test_every_high_stakes_category_has_its_own_card_wording` |
| `stakes_reason=family_relationships` | About her family and close relationships | distiller | card line "...the people close to you, so I want to get it right." | same |
| `stakes_reason=juniper_feelings` | About how she was feeling | distiller | card line "...I don't want to put words in your mouth." | same |
| `stakes_reason=identity_conclusion_about_juniper` | Orion's read on who she is, beyond her words | distiller | its own card line, plus the closer "Is that fair, and should I keep it?" | same + `test_direction_and_identity_cards_close_with_their_own_question` |
| `stakes_reason=orion_machinery` | Orion's conclusion about how it works | distiller | card line "...you can check it better than I can." | same |
| `stakes_reason=orion_asks_direction` | Orion needs her direction | distiller / validator | card line "I need your direction on this one.", plus the closer "Is that the right direction?" | same |
| `stakes_reason=orion_relationship` | About the two of them | distiller | card line "It's about us, so I don't want to decide it alone." | same |
| `stakes_reason=unjudged` | High stakes, but no category was given | validator | card line "I couldn't tell how personal this is..." | `test_uncategorized_high_stakes_says_so` |
| `stakes_reason=none` | Low stakes | distiller | never asked | `test_high_stakes_memory_gets_one_card_and_low_gets_none` |
| `confirmation_state=pending_confirmation` | Waiting to be asked or answered | validator | `open_cards`, `expire_cards`, daily report | `test_confirmation_pg.py` |
| `confirmation_loop_id` | Which card asks about this memory | `open_cards` | expiry join, the Hub's Revise prefill, `apply_outcome` | `test_high_stakes_memory_gets_one_card_and_low_gets_none` |
| `confirmation_state=unconfirmed` | Asked, no answer in 7 days; not a yes | `expire_cards` | daily report; a later answer is still accepted | `test_seven_day_expiry_marks_unconfirmed_never_confirmed_and_frees_the_slot` |
| `confirmation_state=confirmed` | Juniper said yes | `apply_outcome` | voice relabel, reinforcement, daily report | `test_confirm_relabels_voice_reinforces_and_records_the_outcome` |
| `confirmation_state=corrected` | Juniper reworded it; superseded | `apply_outcome` | supersede chain, daily report | `test_revise_supersedes_the_original_and_confirms_her_wording` |
| `confirmation_state=rejected` | Juniper said no; never recalled | `apply_outcome` | `RECALLABLE_WHERE`, daily report | `test_reject_excludes_the_memory_from_recall` |
| outcome row (`memory-confirm-*`) | Her answer, the one resolution record | Hub `/resolve` | bus handler + table catch-up | `test_ask_resolve_pg.py`, `test_catch_up_applies_an_outcome_whose_bus_event_was_lost` |

## Schema / bus / API changes

- Added:
  - `POST /api/asks/{ask_id}/resolve` `{resolution, note}`;
  - `memory_statement` on memory cards in `GET /api/asks`;
  - orion-memory-consolidation as a consumer of `orion:attention:loop_outcome`;
  - `orion_ask.source_kind='memory_confirmation'`, a free-text value, so no migration is needed;
  - `confirmation_state='unconfirmed'`, and `episode_memory.status='rejected'`.
- Removed: none.
- Renamed: none.
- Behavior changed:
  - `/answer` and `/dismiss` return 409 `ask_needs_resolution` for memory cards; vision cards are unchanged.
  - The vision daily ask cap counts only `vision_individual` asks.
  - The panel moved from inside Vision to the top of the Hub home.
- Compatibility notes:
  - `AttentionLoopOutcomeV1` is unchanged (`extra="forbid"` is not affected: the new keys live inside the free-form `features_at_close` dict).
  - Existing Pending Attention outcomes do not start with `memory-confirm-`, so the new consumer ignores them.

## Env/config changes

- Added keys:
  - orion-memory-consolidation: `MEMORY_CONFIRMATION_LOOP_ENABLED=true`, `MEMORY_CONFIRMATION_TICK_SEC=60`, `CHANNEL_ATTENTION_LOOP_OUTCOME=orion:attention:loop_outcome`.
  - orion-hub: `MEMORY_CONFIRMATION_LOOP_ENABLED=true`.
  - All ship ON, in `.env_example`, settings defaults and compose defaults.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes, both services.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py --all-keys orion-memory-consolidation orion-hub` (no `--force`): yes. It also added one pre-existing Hub key that was missing locally, `HECATE_BIOMETRICS_BASE_URL`, whose value equals the settings default, so nothing changes. Two keys that already differed locally were left alone (`CONCEPT_RELATION_RESOLUTION_ENABLED`, `HUB_ROOM_CLAUDE_ENABLED`).
- Skipped keys requiring operator action: none.

## Tests run

```text
# Shared library + orion-memory-consolidation (+ the throwaway Postgres at 127.0.0.1:55499)
pytest orion/memory/episode/tests/test_confirmation.py                   85 passed
pytest orion/memory/episode/tests/test_confirmation_pg.py                14 passed
pytest services/orion-memory-consolidation/tests/test_confirmation_loop.py  6 passed
pytest orion/memory/episode/tests services/orion-memory-consolidation/{tests,evals}
    -> 383 passed, 1 failed: test_episode_migrations_pg (pre-existing; local venv lacks psycopg; CI installs it)
# Hub
pytest services/orion-hub/tests/test_ask_routes.py                       37 passed
pytest services/orion-hub/tests/test_ask_resolve_pg.py                    5 passed (real route, real SQL)
pytest services/orion-hub/tests/test_orion_is_asking_browser_smoke.py     1 passed (Chromium)
node --test services/orion-hub/static/js/*.test.js                      252 tests, 230 pass, 0 fail, 22 skipped
# sql-writer
pytest services/orion-sql-writer/tests/test_vision_individuals.py        27 passed
# Mutation check: removing the advisory lock makes test_concurrent_openers_never_exceed_the_cap fail (3/3 runs).
```

Static gates (`.github/workflows/orion-static-gates.yml`): all pass. `check_definition_drift.py --gate` flagged the one intended routing change (the new consumer). It was re-locked with `--update`, the lock is committed, and the gate now passes.

## Evals run

```text
python services/orion-memory-consolidation/evals/run_memory_confirmation_replay_eval.py \
  --scratch-dsn <throwaway> --live-dsn <live, read-only transaction> \
  --summary-json services/orion-memory-consolidation/evals/results/2026-10-06-memory-confirmation-replay.json
pytest services/orion-memory-consolidation/evals/test_memory_confirmation_replay_eval.py   1 passed
```

The real card opener is replayed in a throwaway Postgres, with one tick per simulated day. The results are counts only; no memory text is printed or committed.

- **Live shadow memories (31 rows): 0 cards.** All 31 were written by the v2 distiller prompt, before the stakes rubric existed, so all are `low` with no category. The v3 prompt is in the running distiller, but no episode has been distilled since it deployed (the last run was 03:50 UTC today).
- **v3 relabel (the same kind of episodes under the v3 prompt, from the committed stakes summary): 15 cards out of 69 memories.**
  - By category: `juniper_feelings` 7, `orion_relationship` 3, `family_relationships` 2, `orion_asks_direction` 2, `identity_conclusion_about_juniper` 1, `health` 0, `orion_machinery` 0.
  - Never answered: 5 cards open on day 0, 5 more on day 7, 5 more on day 14, and all 15 end `unconfirmed`. Never more than 5 open at once.
  - Every card confirmed the day it opens: the queue drains in 3 days, all 15 end `confirmed`, and never more than 5 are open at once.
  - Framing: 10 cards say "You told me something...", and 5 say "this is my own take". The voices are **assumed** from the category for this input (Orion-self categories are given Orion's voice), so this split restates that assumption and does not measure anything.

## Docker/build/smoke checks

```text
docker compose --env-file <primary>/.env --env-file <primary>/services/<svc>/.env -f services/<svc>/docker-compose.yml config
  orion-memory-consolidation rc=0: MEMORY_CONFIRMATION_LOOP_ENABLED "true", MEMORY_CONFIRMATION_TICK_SEC "60",
                                  CHANNEL_ATTENTION_LOOP_OUTCOME orion:attention:loop_outcome
  orion-hub rc=0: MEMORY_CONFIRMATION_LOOP_ENABLED "true"
Import smoke: services/orion-memory-consolidation app.main imports; settings read the flag ON.
```

No image was built and nothing was deployed. Building from a worktree would retag the production image.

## Review findings fixed

No review subagent was run, per the task instructions. These were found by my own checks while building:

- Finding: the bus handler and the catch-up tick could race on one outcome. The loser would have logged a spurious "already resolved" event.
  - Fix: lock the memory row first, then check for an earlier apply.
  - Evidence: `test_bus_and_catch_up_racing_on_one_outcome_apply_it_once` (3 concurrent appliers give exactly one apply).
- Finding: the vision daily cap counted memory cards and would have silenced camera questions.
  - Fix: the cap counts `vision_individual` only.
  - Evidence: `test_ask_budget_counts_only_vision_asks`, plus the SQL run against a scratch database (1 of 2 rows counted).
- Finding: `/answer` and `/dismiss` could close a memory card with no outcome row, which would orphan the memory forever.
  - Fix: 409 `ask_needs_resolution`.
  - Evidence: `test_memory_card_cannot_be_answered_or_dismissed_without_an_outcome` and the Postgres version.
- Finding: the Revise box relied on the Tailwind `hidden` class and showed up without the stylesheet.
  - Fix: the DOM `hidden` attribute as well.
  - Evidence: the browser smoke asserts it is hidden until Revise is clicked.

## Restart required

Apply the migration, then deploy the Hub, then orion-memory-consolidation. All of this happens from the primary checkout on main after merge, not from this worktree.

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_memory_confirmation_v1.sql
```
```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```
```bash
scripts/safe_docker_build.sh orion-memory-consolidation up -d --build
```
```bash
scripts/safe_docker_build.sh orion-sql-writer up -d --build
```

**Deploy order:** migration, then Hub, then orion-memory-consolidation. orion-sql-writer can go at any point.

- The usual "consumer first" order means the service that reads the outcome event (orion-memory-consolidation) would go first. But that service also opens the cards, and an old Hub would show those cards with Answer/Dismiss and close them with no outcome.
- So the Hub goes first. Its new route has nothing to act on until cards exist. No outcome can be produced before orion-memory-consolidation is up, and if one were, the table catch-up would apply it.
- orion-sql-writer can go at any point. Until it is redeployed, memory cards count against the camera's 2-a-day ask limit.

## Risks / concerns

- Severity: medium. Concern: **no card will appear right after deploy.** Every live shadow memory predates the stakes rubric and is `low`. The first card needs a new episode distilled by the v3 prompt that it judges high. Mitigation: the v3 replay above shows what to expect, about 1 in 5 memories. The live path is **UNVERIFIED** until that happens.
- Severity: medium. Concern: **answering in chat is not built** (task item 4, deferred). A word list is banned, and doing it properly needs a model judgment on the turn that follows an asked card, plus a way to know that card was shown in chat at all, which needs the chat-stance seam (Stage 2 D/E's area). Mitigation: the panel path is complete. The spec's design (the distiller emits `confirms` / `corrects`, and persist writes the same outcome row with `via:"chat"`) fits the record this PR already uses: `apply_outcome` accepts any `memory-confirm-*` outcome, and the first answer wins.
- Severity: low. Concern: **no reader excludes rejected memories yet.** Nothing in recall or the reverie seed reads `episode_memory` today (checked on main and the 40 most recent remote branches). Mitigation: `confirmation.RECALLABLE_WHERE` is the predicate Stage 2 readers must use; it is tested and documented in the README.
- Severity: low. Concern: **differences from spec rev 3**, each noted in the spec:
  - an expired card leaves the memory `unconfirmed` rather than `pending_confirmation` (the task asked for this);
  - Revise writes her wording at once instead of waiting for the next distill;
  - Reject sets `status='rejected'`.
- Severity: low. Concern: card wording is a fixed frame around Orion's own statement, which is third person ("Juniper said..."). It is not rewritten into "you". Mitigation: this was deliberate, so there is no LLM rewrite that could drift from what was stored. The framing ("You told me something on Oct 3, and I wrote it down like this: ...") keeps it in Orion's voice.
- Severity: low. Concern: not built from Stage 3 here:
  - the attention-frame `related_loop_ids` extension;
  - the question/prior consumers;
  - outreach novelty;
  - the "Unconfirmed" render label (that belongs to Stage 2's voice renderer).
- UNVERIFIED:
  - the live bus delivery from Hub to orion-memory-consolidation; it is tested with a captured publish and a dead-bus catch-up, not on the live bus;
  - the panel at the top of the real Hub home in a browser; the smoke uses the real panel markup and JS, not the full page.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2517

🤖 Generated with [Claude Code](https://claude.com/claude-code)
