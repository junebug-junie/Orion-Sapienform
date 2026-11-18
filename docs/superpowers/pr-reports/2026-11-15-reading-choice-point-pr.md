## Summary

- After Orion reads a page, the second reading pass (Stage 2) now shows a numbered list of up to 12 links that page contained. Orion already picks links to read next (`need_stage1_urls`); now the links Orion left are on record too.
- After the pass, Hub publishes one `PersonalityChoiceV1` (lane `reading`) per pick on `orion:personality:choice`. An empty pick list is `declined_all` over every shown link. durable-runs already writes these rows. No new model call.
- The shown list is read back out of the prompt actually bound to the durable turn, never recomputed, so restarts and later polls record what Orion really saw.
- Reading choices feed tendencies and care, but are kept OUT of the family endorsement score (source (c), `ENDORSEMENT_EXCLUDED_LANES`). Source (c) now counts one decision per offer instead of one row per pick (review finding).
- Kill switch: `HUB_WORLD_PULSE_READ_CHOICE_ENABLED` (default `true`).

## Outcome moved

The reading row of personality rev 4's Move 1 table was "Orion already picks one link, UNVERIFIED whether the rest are saved". The answer was no: the alternatives were never saved. Now each Stage 2 pass with links on offer writes a chosen-over-declined record. A read-only replay of the last 7 days (`run_reading_choice_replay.py`) estimates **about 21 choice decisions a day** (15-31 on full days, about 53 rows a day because a pass usually picks 2-3 links). Spec acceptance M1 needs at least 20 rows a day from at least 3 choice points; reading alone clears the row count.

Real ordering ("market rank"): page order. That is the order of Stage 1's fetches, then the order each link appears in that fetch's retained text. Nothing scores the links. Replay rank spread of picks: ranks 1-12 all used (rank 1: 30, rank 2: 34, rank 3: 31, ..., rank 12: 9), plus 155 picks outside the list (98 met during Stage 2's own fetches, 76 parent-page links the list left out) and 20 declines.

## Current architecture

- The world-pulse digest queues every item with a URL; no model choice there (`orion/world_pulse_read/seeds.py`).
- Stage 1 reads one seed. Stage 2 (`services/orion-hub/scripts/world_pulse_read_stage2.py`) gets the handoff and returns JSON with `need_stage1_urls`. Each URL that a fetched page contained (`fetch_provenance.hop_allowlist`) re-enters Stage 1, up to `max_round_trips` per chain. Live 2026-11-03: the PBS tag page read led to Orion's PolitiFact hop (`reading:5889415d...`, `parent_request_id` set).
- The Stage 2 prompt never listed the candidate links, and nothing recorded which links Orion passed over.

## Architecture touched

- Hub reading Stage 2 (producer). Contract: existing `PersonalityChoiceV1` on the existing `orion:personality:choice` channel (Hub was already a listed producer).
- `orion/physiology/weights.py` source (c): reading is declinable; one decision counts once.
- `orion/world_pulse_read/reader_contract.py`: untrusted text cannot carry the new list markers.

## Files changed

- `orion/world_pulse_read/next_read_choice.py`: new. Candidate links (page order, label from markdown anchor text or the line, else URL words), prompt section render/parse, `reading_choices`, `publish_choice`.
- `services/orion-hub/scripts/world_pulse_read_stage2.py`: builds the list (minus pages Stage 1 fetched, assets, the site's top-level navigation, links already read; none when the chain is full), puts it in the prompt, keeps the bound prompt from `bind_turn`, publishes choices after `stage2_completed`.
- `orion/world_pulse_read/reader_contract.py`: `defuse_fence_markers` also defuses the list markers.
- `orion/physiology/weights.py`: `DECLINABLE_LANES` += `reading` (its choices are real: a decline is on offer); new `ENDORSEMENT_EXCLUDED_LANES = {reading}` keeps them out of source (c); `_endorsement` counts per decision.
- `services/orion-hub/app/settings.py`, `scripts/main.py`, `.env_example`, `README.md`: flag and docs.
- `orion/bus/channels.yaml`: producer note for the reading lane.
- `config/physiology/contract.yaml`: registers `next_read_choice.py` as a reader (imports only `endorsement_option_key`), so the body-economy contract gate passes.
- `docs/superpowers/specs/2026-10-16-orion-dynamic-personality-graph-design.md`: Reading row updated.
- Tests: `orion/world_pulse_read/tests/test_next_read_choice.py` (new), `services/orion-hub/tests/test_world_pulse_read_stage2.py`, `orion/physiology/tests/test_weights_endorsement.py`.
- Evals: `services/orion-hub/evals/run_reading_choice_replay.py` (new, live read-only), `services/orion-hub/evals/test_reading_choice_eval.py` (new, frozen PBS -> PolitiFact case).

## Schema / bus / API changes

- Added: none (lane `reading` already existed in `ChoiceLane`).
- Removed: none.
- Renamed: none.
- Behavior changed: Hub publishes `PersonalityChoiceV1` lane `reading` (feeds tendencies and care, not source (c)). Source (c) endorsement counts one decision per `act_id` (was one row per pick); for today's single-pick Prime/outreach rows the result is the same.
- Compatibility notes: a Stage 2 turn bound before deploy has no list in its prompt, so it records no choice.

## Env/config changes

- Added keys: `HUB_WORLD_PULSE_READ_CHOICE_ENABLED=true` (services/orion-hub).
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (`orion-hub: +HUB_WORLD_PULSE_READ_CHOICE_ENABLED`).
- skipped keys requiring operator action: none.

## Tests run

```text
pytest orion/world_pulse_read/tests orion/physiology/tests orion/personality          740 passed
pytest services/orion-hub/tests -k "reading or world_pulse or stage2" + evals/        468 passed, 89 skipped
pytest services/orion-durable-runs/tests                                              931 passed, 144 skipped
Mutation check: removing the bound-prompt capture line in _generate fails
test_stage2_choice_uses_the_bound_prompt_not_the_list_just_computed.
```

## Evals run

```text
pytest services/orion-hub/evals/test_reading_choice_eval.py     2 passed
PYTHONPATH=. python services/orion-hub/evals/run_reading_choice_replay.py --days 7   (read-only, live DB)
  stage2_passes=297; full days: decisions 31,16,18,18,24,15,26; per_day decisions=18.5 rows=53.1 (window incl. partial day)
  chain_full (no list) ~7-33/day; no_links_shown 0-6/day; links_shown mean 8.0
```

## Docker/build/smoke checks

```text
Not run. No dependency, port or compose change; Hub image already ships orion/.
Live path UNVERIFIED until Hub is rebuilt and a Stage 2 pass finishes.
```

## Review findings fixed

- Finding: multi-pick rows inflate the endorsement rate (3 picks + 1 decline = 0.75, not 0.5). MUST-FIX.
  - Fix: `_endorsement` counts each decision (act_id) once, chosen if any row chose the family, max care weight.
  - Evidence: `test_a_multi_pick_decision_counts_once_per_offer`.
- Finding: tests never ran the bound-prompt capture in `_generate`.
  - Fix: test runs the real `_stage2_pass`/`_generate` with stubbed `bind_turn`/`poll_turn` whose bound list differs from the one just computed.
  - Evidence: test passes; fails when the capture line is removed.
- Finding: the list was shown when the chain was already at `max_round_trips` (every pick refused).
  - Fix: no list, no choice when the chain count (same SQL as `enqueue_reading`'s cap) is at the cap; replay models it.
  - Evidence: `test_stage2_shows_no_list_when_the_chain_is_full`; replay `chain_full` column.
- Finding: site navigation links take the top ranks.
  - Fix: same-host links at path depth 0-1 are left out. Replay showed little change (unlisted picks 287 -> 284), so navigation was not the main cause of off-list picks.
  - Evidence: `test_the_read_sites_top_level_navigation_is_not_shown`.
- Finding: off-list parent links were described as "found this turn".
  - Fix: docstring/README say "outside the shown list"; `situation.unlisted_picks` counts them; replay splits them by source.
- NITs fixed: `html.unescape` before label matching; `seed_id[:250]` so pick ids never collide. Documented: exact-URL already-read filter; the list is recomputed on each poll tick (cheap; only the bound prompt's list is used).

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

Run from the clean primary `main` checkout after merge. durable-runs needs no change for the rows; `weights.py` reaches durable-runs' values market on its next rebuild: `scripts/safe_docker_build.sh orion-durable-runs up -d --build`. Deploy order: either order works; Hub first starts the rows, durable-runs then counts reading in source (c).

## Risks / concerns

- Severity: low (was medium). Concern: reading choices would have read about 86% "picked a link" and lifted `hub.world_pulse_read`'s weight with no comparison against other work (every option is one family, chosen inside a reading run). Mitigation: reading rows are excluded from source (c) (`ENDORSEMENT_EXCLUDED_LANES`, test `test_reading_rows_are_real_choices_but_never_move_reading_endorsement`); they still feed tendencies and care, which bind from deploy. A Prime pick of a reading concern still counts for the family. `HUB_WORLD_PULSE_READ_CHOICE_ENABLED=false` stops new rows.
- Severity: low. Concern: the list may nudge Orion toward more hops (each hop is an agent-GPU Stage 1 read, Wallet A gated). Mitigation: the chain cap and Wallet A are unchanged; replay shows 82% of passes already asked for hops.
- Severity: low. Concern: pub/sub delivery, so a choice published while durable-runs is down is lost. Mitigation: same as Prime/outreach.
- Live path: UNVERIFIED (no deploy in this PR). Check after deploy: `docker logs orion-athena-hub 2>&1 | grep reading_choice` and `SELECT count(*) FROM personality_choice WHERE lane='reading'`.
- Issues opened/closed: none.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3405

🤖 Generated with [Claude Code](https://claude.com/claude-code)
