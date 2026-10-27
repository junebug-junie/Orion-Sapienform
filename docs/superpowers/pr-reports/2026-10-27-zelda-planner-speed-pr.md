## Summary

- **Short replies:** with a scored menu, Orion answers `{"option": n, "why": "..."}`.
  - Each menu line shows that option's goal, prediction and time limit.
  - The reply cap is 96 tokens on the first try and 320 on the retry.
  - The full goal shape is still accepted when no option fits.
- **Co-play priority:** co-play plans on the `metacog_turn` route, which is interactive priority and is never held by a heat shed. Solo leisure stays on `metacog_background`.
- **Faster polling:**
  - The play page polls every 250 ms while Orion holds the controller; otherwise it polls every 1 s.
  - Leisure polls every 0.25 s; after a transport failure it still waits 1 s.
- **Cache-friendly prompt:** fixed instructions come first, then `--- Current game state ---`, so llama.cpp can reuse the cached prefix.

## Outcome moved

Juniper: "Orion will be playing, run out into a crowded enemy field ... getting the shit kicked out of them, then they decide."

Live session 96cb54f4, before this patch:

- Link stood still **193 s** while waiting for decisions and acted **137 s** (58% frozen).
- Wait per decision was 5.8 s median and 12.8 s max.
- With 3 or more enemies present, 3 of the 8 waits took 11–13 s.

Where the model time goes (circe metacog llama.cpp log, Qwen3-8B Q5): about 0.35 s to read the prompt, and 2.7–4.0 s to write 156–205 reply tokens at 50–63 tok/s.

The same "south wall" decision, run 5 times on the live GPU with the cabinet idle:

| Code | Median | Choice |
|---|---|---|
| main (full JSON reply, background priority) | 1.79 s | exit E, 5 of 5 |
| this branch | **0.60 s** | exit E, 5 of 5 |

The first call of each run took about 5–6 s on both, which looks like a cold start.

The queueing behind background metacog work, which caused the 11–13 s live cases, is removed for co-play by interactive priority. That is **UNVERIFIED** live, because the queue was empty while measuring.

## Current architecture

- The planner called `metacog_background` for both co-play and leisure.
- The model wrote a full goal, prediction, rationale (up to 512 characters) and budget.
- The page polled once a second.
- The prompt began with the dynamic "why a decision now" line.

## Architecture touched

- `retro_planner`: `STATIC_RULES`, `REPLY_MENU`/`REPLY_FULL`, `choice_from_option`, the `route` parameter on `plan_zelda`/`_gateway_chat`, token caps, and `SHED_BACKOFF` keyed by route.
- `retro_experience`: `Option.default_plan` and `target_room`; the menu line shows the default bet; `for_planner` carries the prediction and budget.
- `retro_routes.plan_v2`: passes `COPLAY_ROUTE`.
- `retro_leisure`: `POLL_SEC` 0.25 and `FAILURE_POLL_SEC` 1.0.
- `retro-panel.js`: the poll rate; asset bumped to `v=10`.
- `config/gpu_pool.yaml`: comment only.

## Files changed

- `services/orion-hub/scripts/retro_planner.py`: prompt order, option replies, route, caps.
- `services/orion-hub/scripts/retro_experience.py`: default plans, menu line.
- `services/orion-hub/scripts/retro_routes.py`: co-play route.
- `services/orion-hub/scripts/retro_leisure.py`: poll spacing.
- `services/orion-hub/static/js/retro-panel.js`, `templates/retro.html`: 250 ms poll, `v=10`.
- `services/orion-hub/tests/retro/*.py`, `evals/retro_play_panel.mjs`: new and updated tests, asset version.
- `config/gpu_pool.yaml`: names the co-play caller of `metacog_turn`.

## Schema / bus / API changes

- No contract change.
- The planner's `planner_id` prefix now names the route (`llm-gateway:metacog_turn:<id>` for co-play). It is opaque everywhere it is stored.

## Env/config changes

None.

## Tests run

```text
PYTHONPATH=. pytest services/orion-hub/tests/retro -q             186 passed
PYTHONPATH=. pytest orion/gpu_pool/tests -q (with the above)      698 passed
```

## Evals run

```text
evals/retro_play_panel.py --output /tmp/orb9                       PASS
live planner timing, wall fixture x5, metacog_turn                 median 0.60 s (main: 1.79 s), same choice
```

## Docker/build/smoke checks

Not run. The change is Hub Python plus a static asset.

## Review findings fixed

- **Finding (M1):** at the 0.25 s poll, leisure gave up on Retro after about 1.25 s of transport errors.
  - Fix: failure branches sleep `FAILURE_POLL_SEC` = 1 s.
  - Evidence: `test_a_five_second_retro_outage_still_gives_up_after_about_five_seconds`. It fails when the failure branches use the 0.25 s poll (mutation checked).
- **Finding (M2):** the option's prediction was a hidden default; the menu line never showed it.
  - Fix: each line ends "Predicts <kind> [value], <n> s.", and the reply format says a prediction may be added.
- **Finding (L1):** fight-nearest with 2 or more enemies predicted `enemies_cleared`, which is almost always refuted.
  - Fix: it predicts `survive` unless exactly one enemy is present.
  - Evidence: `test_a_fight_with_several_enemies_predicts_survival_not_a_clear_room`.
- **Finding (L2):** the retry turn kept the 96-token cap, which could cut off a full-shape answer.
  - Fix: the retry gets 320.
- **Finding (L3):** `"option": "2"` was refused, `budget_seconds: null` failed, and a reply with both option and goal failed.
  - Fix: digit strings and whole floats are coerced, a null budget falls back, and the goal wins over the option.
  - Evidence: `test_option_picks_tolerate_loose_types_and_the_retry_gets_full_room`.
- **Not changed (L4):** a menu can still offer the move that just did nothing; the validator refuses it and the retry handles it. **L5:** the prefix-cache reuse is UNVERIFIED live.

## Restart required

```bash
git pull   # post-pull rebuild of orion-hub (Hub + retro gateway); reload the Retro page (retro-panel.js?v=10)
```

## Risks / concerns

- **Severity Medium:** co-play planning now runs under a heat shed.
  - Concern: one 96-token call per decision is added load while the cabinet is hot.
  - Mitigation: this is the same policy as chat and AI Town speech; leisure stays background; the 34 °C hardware reflex is unchanged.
- **Severity Low:** about 6 s on the first call of each test process.
  - Concern: it looks like a cold start, possibly the bus connect or the server.
  - Mitigation: none yet; check it in the Hub's long-lived process after deploy.
- **Not addressed:** Link still stands still between goals, now for about 1 s instead of about 6 s. The guard reflex and "one hit doesn't end a move" need Juniper's OK (they change the Zelda spec's no-reflex rule).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2905

🤖 Generated with [Claude Code](https://claude.com/claude-code)
