# Personality loop wiring (rev 4): real option text, care reaches outreach, Prime choices count per family, quiet reducer ticks

Stacks on #3331 (outreach choice point) and #3332 (market values). Merge those first.

## Summary

- Curiosity choices now read as Orion's own words. Hub stores each offered prior's claim on the offer row
  (`curiosity_offer_decisions.offered[].claim`, JSON field, no migration). The durable-runs curiosity sync uses
  it as the option text. Old rows keep the de-slugged id.
- Outreach care is no longer always 0. The binding bid carries the shown topics' text (`object_text=`). When
  the want was funded only through care, the outreach choice is `care_promoted`.
- Prime choices count toward the right family. Each concern option is keyed `<want family>::<concern_id>`
  (`weights.endorsement_option_key`), from the funded event want the concern names. A concern with no event
  keeps its plain id (lane family `prime`).
- Prime choices carry `care_promoted`. `GET /physiology/event-wants` marks a listed want `care_promoted` when
  it is in the top `limit` only through care, or funded only through care. Prime copies the flag onto a
  choice of that concern.
- The reducer stops rewriting every node on every idle tick. A fade-only change is written once the weight
  moved 1% from the stored weight (about every 5 h at the 14-day half-life).

## Outcome moved

- Curiosity option text: "turn start context is stale snapshot" (id) becomes the prior's claim sentence,
  which the reducer embeds. Better directions for every new curiosity choice.
- Outreach care: from structurally 0 (`personality:care=0:reason=no_object_text`) to priced.
- Source (c): Prime picks of a reading concern count for `hub.world_pulse_read`, a camera ask for
  `vision.ask`, not all for `prime`.
- Runaway guard: Prime and outreach choices that only reached Orion through care count at half weight.
- Reducer writes: live replay of the 118 live choice nodes (read-only, `personality_disposition` dump at
  10:37Z): old rule writes 118 on the next idle tick, new rule writes 0; +1 h: 0; +5 h: 118 (the 1% fade).

## Current architecture

- Curiosity offer rows stored only prior ids; `orion/personality/choices.py` de-slugged them.
- Outreach bids had `object_ref="juniper"` and no `object_text`; the market's care term read no text.
- `orion/prime/choice.py` keyed options by concern id; `weights.option_family` mapped them all to `prime`.
- No producer set `PersonalityChoiceV1.care_promoted`.
- `orion/personality/graph.py` `_Tick.result` compared exact snapshots including `weight`, which fade moves
  every tick.

## Architecture touched

- orion-hub: curiosity offer log, outreach binder + tick + shadow bid, outreach choice builder.
- orion-durable-runs: event-wants menu (care_promoted), personality reducer write rule (shared `orion/personality`).
- orion-mind: Prime loop passes event families and care-promoted events to the choice builder.
- Shared: `FundedEventWantV1.care_promoted` (additive), `personality_values.care_funded`.

## Files changed

- `services/orion-hub/scripts/curiosity_offer_decisions.py`: `offered_rows` stores `claim` (<= 400 chars).
- `orion/personality/choices.py`: option text = stored claim, else readable id.
- `services/orion-hub/scripts/outreach_bind.py`: `bind(object_text=)`, sends it with the bid; `care_promoted` on the held want and in `record()`.
- `services/orion-hub/scripts/endogenous_outreach.py`: passes the topics' text to the bid; passes `care_promoted` to the choice.
- `services/orion-hub/scripts/outreach_choice.py`: `outreach_object_text`, `build_choice(care_promoted=)`.
- `services/orion-hub/scripts/physiology_shadow.py`: shadow bid carries `object_text`.
- `orion/physiology/personality_values.py`: `care_funded`.
- `orion/schemas/physiology_want.py`: `FundedEventWantV1.care_promoted`.
- `services/orion-durable-runs/app/event_wants_menu.py`: reads decision `reason`/`warnings`, computes `care_promoted`.
- `orion/prime/choice.py`, `services/orion-mind/app/prime_loop.py`: family keys, `care_promoted`.
- `orion/personality/graph.py`: `_must_write`.
- Tests: `orion/personality/tests/test_graph.py`, `test_choices.py`, `orion/prime/tests/test_prime_choice.py`,
  `orion/physiology/tests/test_personality_values.py`, `services/orion-durable-runs/tests/test_event_wants_menu.py`,
  `test_personality_naming.py` (fade assertions now compare the stored pair faded to now),
  `services/orion-hub/tests/test_outreach_bind.py`, `test_outreach_choice.py`, `test_curiosity_spend_log.py`,
  `services/orion-mind/tests/test_prime_choice_point.py`.
- READMEs: orion-durable-runs, orion-hub.

## Schema / bus / API changes

- Added: `FundedEventWantV1.care_promoted: bool = False` (HTTP contract of `GET /physiology/event-wants`, not a bus payload).
- Added: `curiosity_offer_decisions.offered[].claim` / `stale_offered[].claim` (JSON field, no DDL).
- Behavior changed: Prime `PersonalityChoiceV1` option keys are `<family>::<concern_id>` for event concerns.
- Compatibility: `FundedEventWantV1` is `extra="forbid"`. Deploy orion-mind (reader) before orion-durable-runs
  (writer); the other order makes Prime reject the list until mind is rebuilt. Old `personality_choice` rows keep
  their old keys and text (insert-only); source (c) maps the old plain Prime keys to `prime` as before.

## Env/config changes

- None. No `.env_example` changed; no sync needed.

## Tests run

```text
orion/personality orion/physiology/tests orion/prime: 688 passed
services/orion-durable-runs/tests: 908 passed, 136 skipped
services/orion-mind/tests: 367 passed
services/orion-hub/tests (full suite): see "Hub full suite" below
scripts/check_definition_drift.py --gate: PASS (0 changed)
```

## Evals run

```text
orion/prime/evals/run_prime_frame_eval.py: PASS
orion/prime/evals/run_prime_block_eval.py: PASS (0 failing)
orion/prime/evals/run_prime_wake_replay.py: PASS
Live replay (read-only) of personality_disposition, 118 choice nodes: idle-tick writes 118 -> 0
```

## Docker/build/smoke checks

```text
Not run: no Dockerfile, compose, requirements or env change. Images rebuild from source on deploy.
```

## Review findings fixed

See the PR conversation (filled after review).

## Restart required

Deploy order (from the primary main checkout, after #3331 and #3332):

```bash
scripts/safe_docker_build.sh orion-mind up -d --build
scripts/safe_docker_build.sh orion-durable-runs up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: low. Concern: readers of the raw `personality_disposition.weight` column (the market's tendency
  read, the frame) can be up to 1% staler than before. Mitigation: bounded by `history_min_rel_change`; the row
  stays exact as of its `updated_at`.
- Severity: low. Concern: outreach care embeds the joined topics, a blend, not one topic. Mitigation: shown
  topics are ranked by freshness, so care can only change whether the compose is funded, which is what the
  single bid prices.
- UNVERIFIED live: claim text on new curiosity rows, outreach `personality:care=` lines with a reason other
  than `no_object_text`, Prime family keys and `care_promoted` on live rows, and `nodes_written` near 0 on idle
  ticks. All need the deploy above.

## PR link

(filled after the PR opens)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
