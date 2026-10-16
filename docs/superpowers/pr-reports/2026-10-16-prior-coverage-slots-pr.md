## Summary

- Curiosity runs show 8 live priors. 2 of those slots now skip the uncertainty order and draw from the priors ranked past it (per-run rotation).
- The prompt lists the 2 drawn priors in their own block, "DRAWN AT RANDOM, not ranked", so the sentence "these are ORDERED" stays true.
- The offer log marks every offered prior `slot: order | coverage`, and `constants` records `prior_coverage_slots`.
- `HUB_CURIOSITY_PRIOR_COVERAGE_SLOTS=2` (0 = off).

## Outcome moved

Orion's Oct 13 letter: "continuity in my own record, especially the unasked half." Live data 2026-10-16: 72 of 119 live self-priors (60%) never appeared in any offer across 140 runs (`curiosity_offer_decisions`, 09-26..10-15); 105 of 162 live priors overall. 68 of the 72 sit at confidence ≥0.7 or ≤0.3. Cause: `select_priors` offers 8 per run ordered by closeness to 0.5, and `_rotation_key` only breaks exact ties, so a confident prior with fewer than 3 tests ranks past the sample forever.

## Current architecture

`read_snapshot` -> `select_priors` (`orion/curiosity/worldview.py`) -> `WorldviewSnapshot.live_priors` -> `_priors_section` (`orion/curiosity/kickoff_prompt.py`) and `offered_rows` -> `curiosity_offer_decisions` (Hub). Self-inquiry runs use the same Hub reader.

## Architecture touched

Hub curiosity loop only. No bus, no schema registry change. `curiosity_offer_decisions.offered` is JSONB, so the new `slot` key needs no migration; readers that ignore it are unaffected.

## Files changed

- `orion/curiosity/worldview.py`: `Prior.coverage` flag; `select_priors(coverage_slots=)`; `read_snapshot(coverage_slots=)`.
- `orion/curiosity/kickoff_prompt.py`: separate "DRAWN AT RANDOM" block.
- `services/orion-hub/scripts/curiosity_offer_decisions.py`: `slot` in offered rows.
- `services/orion-hub/scripts/curiosity_investigation.py`, `scripts/main.py`, `app/settings.py`: setting plumbed through; logged in `constants`.
- `services/orion-hub/.env_example`: new key.
- Tests: `tests/test_curiosity_worldview.py` (7), `services/orion-hub/tests/test_curiosity_spend_log.py` (1).

## Schema / bus / API changes

- Added: `slot` key inside `offered[]` rows; `prior_coverage_slots` inside `constants`.
- Compatibility: additive JSON keys. Rows before this patch have no `slot`; read them as `order`.

## Env/config changes

- Added: `HUB_CURIOSITY_PRIOR_COVERAGE_SLOTS=2` (Hub). `.env_example` updated. Local `.env` synced with `python3 scripts/sync_local_env_from_example.py` (verified key present). Parity PASS.
- Skipped keys: none.

## Proposal-mode answers (touches curiosity offer / self-model)

- Capability change: 2 of 8 offered slots are chosen by rotation, not by uncertainty.
- Data touched: read-only. Hub never writes the graph. No new stored state.
- Privacy: none new.
- Trace: `curiosity_offer_decisions.offered[].slot`.
- Danger: weaker order experiment. Mitigation: coverage picks are marked `slot=coverage`, so analysis can drop them; arm propensity still describes the 6 ordered slots.
- Roll back: `HUB_CURIOSITY_PRIOR_COVERAGE_SLOTS=0`, restart Hub.

## Tests run

```text
tests/test_curiosity_worldview.py + tests/test_curiosity_peer_kickoff.py + orion/curiosity: 282 passed
services/orion-hub tests/test_curiosity_spend_log.py + test_curiosity_investigation.py: 172 passed
services/orion-hub/evals/test_curiosity_value_order_eval.py: 3 passed
```

## Evals run

Replay of the new picker over the real 162 live priors (frozen copy of today's graph), seeds run-0..run-139:

```text
coverage_slots=0: live self-priors never offered = 105 of 119
coverage_slots=2: live self-priors never offered =  31 of 119
```

The live log shows 72 (not 105) for the old picker because the pool changed during those weeks. The replay is a frozen-pool estimate, not a forecast.

## Docker/build/smoke checks

UNVERIFIED live. After deploy: new `curiosity_offer_decisions` rows must show 2 `slot=coverage` entries in `offered`, and the kickoff prompt a "DRAWN AT RANDOM" block.

## Review findings fixed

Subagent review: no defects found. It checked every caller of `select_priors`/`read_snapshot`/`Prior` (`orion/situational/context.py` and the Hub value-order eval leave `coverage_slots` at 0), the prompt-drift parser (reads `preview()` lines, unchanged), edge cases (k > sample is capped, negatives clamp to 0, stale priors never drawn, short pools) and env parity. Forcing k=0 made 4 worldview tests fail.

- Noted, intended: the value-order experiment now has 6 ordered picks per run instead of 8. The other 2 are tagged `slot=coverage`; analysis should filter on `slot=order`.
- Noted, not changed: `coverage_slots >= sample` makes the whole list random. That is a config choice.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: low. The draw is random, not "longest unseen". It needs no new state. After 140 runs 31 of 119 stay unoffered in a frozen-pool replay.
- Follow-up: if the residual matters, track last-offered per prior (needs a read of the offer log) and draw the oldest instead.
