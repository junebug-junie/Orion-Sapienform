# PR report: Orion's curiosity loop moves off the worker Juniper types into

## Summary

- Curiosity investigations ran on the `harness` gateway route, which is
  **circe-worker-1** -- the same single-slot llama.cpp worker that serves
  `chat`. Confirmed live 2026-09-03 via `GET /routes`. Up to
  `HUB_CURIOSITY_INVESTIGATION_DAILY_CAP` (6) runs a day, each measured at
  657-1176s of FCC time, held that slot.
- Cause: the loop sent no model label, so it fell through to the unified turn's
  default (`MODEL_SONNET` -> `llamacpp/harness`). Nothing was choosing the
  chat worker; nothing was choosing anything.
- Fix: `HUB_CURIOSITY_INVESTIGATION_LLM_ROUTE=agent`, resolved through
  `orion/llm/routes.py::fcc_model_for_route` -- the route vocabulary's single
  owner -- into an explicit payload `fcc_model_label`. No new vocabulary, no
  second mapping.
- Uses **branch 1** of `_resolve_fcc_model_label` (explicit label wins
  outright), NOT branch 2 (Mode=Agent derives from `llm_route`). So `mode`
  stays `"orion"` and nothing downstream shifts: `chat_history_log` tags and
  `HarnessRunRequestV1.mode` are unchanged.
- **Found on the way:** the env sync silently skipped the new key.
  `HUB_CURIOSITY_` was never in `SYNC_PREFIXES`, so the run reported no
  changes and wrote nothing. Every `HUB_CURIOSITY_*` key in the live `.env`
  got there by hand. Prefix added; key then synced for real.
- **DOES NOT MERGE AS-IS.** See "Risks / concerns" -- the destination lane is
  ~1.9x slower and half the measured runs would exceed the governor's FCC
  ceiling. The routing is correct and tested; the deadline decision is
  Juniper's.

## Outcome moved

Orion's own time stops competing with Juniper's chat for the same GPU slot.

It does **not** yet reliably complete a turn there. Measured, not projected:
the 1.9x slowdown puts 3 of the 6 most recent completed investigations over
`HARNESS_FCC_TIMEOUT_SEC`.

## Current architecture (before this patch)

```
CuriosityInvestigation._generate
  -> execute_unified_turn(payload={"no_write": True, "source": ...})
     -> _resolve_fcc_model_label  -- no explicit label, mode "orion"
        -> DEFAULT_UNIFIED_TURN_FCC_MODEL_LABEL = "MODEL_SONNET"
           -> ~/.fcc/.env MODEL_SONNET=llamacpp/harness
              -> gateway route `harness` -> circe-worker-1 (35B), total_slots 1
                 ^^^ also serves `chat` -- Juniper's interactive lane
```

## Architecture touched

- `services/orion-hub/app/settings.py` -- the lane setting.
- `services/orion-hub/scripts/curiosity_investigation.py` -- route resolved
  once at construction; label placed on the turn payload.
- `services/orion-hub/scripts/main.py` -- wiring.
- `scripts/sync_local_env_from_example.py` -- prefix allow-list, and one
  credential moved into `NEVER_SYNC_KEYS` as a direct consequence.

Deliberately **not** touched: `endogenous_outreach.py`, which calls
`execute_unified_turn` the same way and has the same fall-through. Its turns
are short (~3 steps) so it is not the contention story, and moving it is a
separate decision.

## Files changed

- `services/orion-hub/app/settings.py`: `HUB_CURIOSITY_INVESTIGATION_LLM_ROUTE`,
  default `agent`. The comment records why `harness` and `chat` are one worker
  and deliberately carries **no literal** for `HARNESS_FCC_TIMEOUT_SEC` -- that
  key has a single-source gate and copies of it have drifted before.
- `services/orion-hub/scripts/curiosity_investigation.py`: `_turn_payload`
  helper; `llm_route` constructor param; resolution + boot log/warning;
  payload call site.
- `services/orion-hub/scripts/main.py`: passes the setting.
- `services/orion-hub/.env_example`: the new key.
- `scripts/sync_local_env_from_example.py`: `"HUB_CURIOSITY_"` in
  `SYNC_PREFIXES`; `"HUB_CURIOSITY_GRAPH_ORION_PASSWORD"` in
  `NEVER_SYNC_KEYS`.
- `services/orion-hub/tests/test_curiosity_investigation.py`: 10 new tests.

## Schema / bus / API changes

None. `HarnessRunRequestV1` is unchanged -- `fcc_model_label` is an existing
field and `label_to_claude_model_id` has accepted a `"<backend>/<route>"` spec
since PR #2062. No bus channel, no registry entry, no rolling-deploy concern.

## Env/config changes

- Added keys: `HUB_CURIOSITY_INVESTIGATION_LLM_ROUTE=agent` (orion-hub).
- Removed / renamed: none.
- `.env_example` updated: yes.
- local `.env` synced: yes, and only after fixing the sync itself --
  `orion-hub: +HUB_CURIOSITY_INVESTIGATION_LLM_ROUTE='agent'`, landing at
  `services/orion-hub/.env:639`.
- Skipped keys requiring operator action: none.

## Tests run

```text
PYTHONPATH=.:services/orion-hub pytest services/orion-hub/tests/test_curiosity_investigation.py -q
-> 117 passed

PYTHONPATH=. pytest tests/scripts/test_sync_local_env_from_example.py \
                    orion/harness/tests/test_compute_lane_model_and_ceiling.py -q
-> 46 passed

python scripts/check_env_key_single_source.py -> OK: 1 owned env key(s), no drifted copies.
python scripts/check_fcc_context_parity.py    -> ok motor_ctx=65536 profile_max_ctx=131072
```

Mutation-tested (each reverted after):

| mutation | caught by |
|---|---|
| `_turn_payload` drops the label | `test_the_lane_actually_reaches_the_unified_turn`, `test_the_payload_omits_the_label_entirely_when_there_is_no_override` |
| `.env_example` -> system-only `harness` | `test_the_shipped_env_example_names_a_route_that_actually_resolves` |
| `logger.warning` -> `pass` | `test_an_unusable_route_says_so_at_boot` |
| wiring line deleted from `main.py` | `test_the_wiring_line_is_present_in_main` |

## Evals run

No eval harness exists for the curiosity loop. Its real eval is the live
journal, which is what "Risks" is about. Not created here -- an eval whose
pass condition is "the turn finished" would go red for the deadline reason
below, not for a routing defect.

## Docker/build/smoke checks

None run. No image content changed; this is an env key plus Hub Python read at
boot.

## Review findings fixed

- Finding: `settings.py` comment claimed each run held the chat slot for
  `HUB_CURIOSITY_INVESTIGATION_TIMEOUT_SEC` (2700s). That is the Hub-side outer
  wait, not the binding ceiling, and it overstated the harm ~2.3x -- in a file
  that already warns against exactly this copy.
  - Fix: comment now cites measured occupancy (657-1176s) and names
    `HARNESS_FCC_TIMEOUT_SEC` as the real ceiling without copying its value.
  - Evidence: `check_env_key_single_source.py` still passes.
- Finding: the unusable-route warning -- the safety property both
  `.env_example` and `settings.py` advertise -- had zero coverage. Replacing
  `logger.warning` with `pass` left all tests green.
  - Fix: `test_an_unusable_route_says_so_at_boot` (caplog), and the
    empty-route test now asserts the absence its own docstring is about.
  - Evidence: mutation table above.
- Finding: deleting one line in `main.py` silently reverts the whole change
  and every test stays green -- the same silent fall-through this patch exists
  to end, one layer up.
  - Fix: `test_the_wiring_line_is_present_in_main`.
  - Evidence: mutation table above.
- Finding: adding `"HUB_CURIOSITY_"` to `SYNC_PREFIXES` newly brings
  `HUB_CURIOSITY_GRAPH_ORION_PASSWORD` into `--force`'s blast radius. Before,
  no prefix matched it, so the protection was an accident of the prefix list.
  `.env_example` ships it empty; flattening it costs Orion write access to its
  own worldview graph.
  - Fix: added to `NEVER_SYNC_KEYS`, beside the iLO credentials with the same
    rationale.
  - Evidence: all 22 `HUB_CURIOSITY_*` keys enumerated against the live `.env`;
    it was the only divergent one. It no longer appears in a `--dry-run` at all.
- Finding (reviewer, partially wrong -- corrected): claimed "five of the six"
  recent runs would exceed the ceiling.
  - Fix: recomputed from all six observed values rather than the three quoted.
    Actual is **3 of 6**. Direction holds, magnitude did not.
  - Evidence: see "Risks" below.

## Restart required

```bash
sudo systemctl restart orion-hub
```

(Hub reads `HUB_CURIOSITY_INVESTIGATION_LLM_ROUTE` at boot. Do not restart
until the deadline question below is settled -- restarting as-is moves
curiosity onto a lane where half its runs will die.)

## Risks / concerns

- Severity: **HIGH -- merge gate, not a follow-up.**
  Concern: the destination lane is ~1.9x slower at prompt processing (491 vs
  944 tok/s at an 18k prompt; the 35B is MoE with ~3B active per token, the 27B
  is dense on one V100). An agentic turn re-sends a growing prompt every step,
  so that ratio dominates. Applying it to the six most recent completed
  investigations:

  ```
  observed FCC sec   830  1166  1176   680   657  1028
  at 1.9x           1577  2215  2234  1292  1248  1953
  vs HARNESS_FCC_TIMEOUT_SEC=1600:  3 survive, 3 exceed
  ```

  Long turns already fail often on the *fast* lane -- 24 grounded vs 17
  `fcc_timeout` over 14 days for turns >=50 steps (41%). This would push that
  materially worse. And the failure is quiet: `_generate` returns empty,
  `_investigate` logs `curiosity_investigation_no_text` at INFO, no journal
  entry, no metric -- and the daily slot is consumed *before* the turn by
  design, so Orion burns all six slots and writes nothing.

  Mitigation: one of --
  (a) raise the deadline chain coherently. The chain is
  `FCC 1600 + voice 300 + reflect 180 + substrate 5 = 2085` under
  `HUB_HARNESS_GOVERNOR_RPC_TIMEOUT_SEC=2160`. Going to FCC 2400 needs RPC
  ~2960 and `HUB_CURIOSITY_INVESTIGATION_TIMEOUT_SEC` ~3200, all still under
  `HUB_HARNESS_GOVERNOR_RPC_MAX_WAIT_SEC=3600`. `HARNESS_FCC_TIMEOUT_SEC` is
  service-global, so this also raises the ceiling for interactive chat.
  (b) give the harness a per-request FCC deadline. This is the honest seam --
  an unattended 4-hourly loop and a turn Juniper is waiting on have genuinely
  different budgets -- but it is a `HarnessRunRequestV1` field, i.e. a contract
  change with producer/consumer/registry work. Out of scope here.
  (c) merge the routing and accept reduced journaling. Not recommended
  silently; the failure is invisible.

- Severity: MEDIUM.
  Concern: `agent` is not an always-on lane. `orion-gpu-lane-controller` flips
  GPU1 exclusively between `affect` and `agent`, so when affect holds it the
  route is down. Nothing here checks route status before dispatching, and
  `probe_route_runtime` fails open -- the turn dispatches into a dead route and
  burns a daily slot. `harness`/`chat` had no such property.
  Mitigation: the flip is operator-triggered today (no automated callers of
  `/v1/gpu-lane/flip` found outside that service), so this is a known hazard
  rather than a live one. A status pre-check belongs with whichever option
  above is chosen.

- Severity: LOW.
  Concern: `agent` is also `total_slots: 1`. A running investigation blocks
  Juniper's own Mode=Agent turns. The contention moved off the chat lane rather
  than disappearing.
  Mitigation: accepted -- chat is the lane that matters for responsiveness.

- Severity: LOW.
  Concern: the outreach *composition* turn goes through the same `_generate`,
  so prose Juniper reads is now written by the 27B. Composition is ~3 steps so
  the deadline risk does not apply, and the stance gate runs on the cortex path,
  untouched. It is a voice change, and it is traced via
  `offer_message(model=...)`.

## PR link

<pending>
