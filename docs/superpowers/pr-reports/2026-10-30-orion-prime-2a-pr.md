# feat(prime): Orion Prime 2A resident loop in orion-mind

## Summary

- orion-mind now keeps thinking between conversations. A cheap tick reads the body and Temporal Self every 15 s and notes what changed. When something changed, Prime bids in the body economy (family `prime`, Tier 1, lane `chat`) and obeys the answer. A funded thought is one model call on circe chat lane 0, at background priority.
- Each thought edits Prime's concerns (at most 3) and may make one move: a question for curiosity's next regular investigate run. The question goes through Hub's existing focus slot. Prime never replaces a focus that is already waiting, and the run is told the question was chosen "between conversations", not in a chat.
- 20 % of questions are held back (the narrator arm). They are written as acts and never sent. Both arms are graded the same way: did the curiosity run that took the question (or would have taken it) write something.
- Thoughts (`infer`) and questions (`ask`) are acts on the ledger, `actor = 'prime'`. State persists in Redis, so a restart resumes the same concerns.
- A new gateway route `prime`: the chat worker at `background` priority. Juniper's call always starts first, and the heat reflex sheds Prime.

## Outcome moved

Before this PR, Orion chose nothing between conversations: no process held a concern, and nothing outside a turn could ask a question. Now a resident loop forms concerns and asks curiosity questions. Every thought and question leaves an act row with an outcome. The holdback arm gives the first direct test of whether Prime causes anything or only narrates (design section 9).

## Current architecture

- orion-mind was request/response only: `run_mind()` ran when something called it.
- Curiosity's focus slot (`orion/curiosity/focus.py`) took one question, queued only from a chat turn (`queue_curiosity_focus`, invocation context `unified_chat`).
- The body economy priced wants in shadow. Every caller logged the decision and ignored it.

## Architecture touched

- **orion-mind**: new `app/prime_loop.py` (the I/O runtime), started in the FastAPI lifespan on its own bus connection. The connection retries, so a bus blip at boot delays Prime instead of disabling it. `mind_run` marks Juniper turns.
- **`orion/prime/`** (new, pure logic): the tick (`state.py`), the frame and answer handling (`thought.py`), the acts, the holdback draw and settlement (`acts.py`), and the evals.
- **Hub**: `curiosity_focus_listener.py` accepts invocation context `prime`: origin `self` only, and it never replaces a pending focus. `focus.py` gains `keep_pending`, plus wording that does not claim a Prime question came from a chat.
- **Contracts**: `IntrospectToolBindingV1.invocation_context` gains `"prime"`. channels.yaml: orion-mind becomes a producer on `orion:grammar:event` and on `orion:introspect:queue_curiosity_focus:request`, and a consumer of `orion:introspect:result:*`. `orion/grammar/acts.py` registers orion-mind as an act producer.
- **GPU pool / LLM routes**: `config/gpu_pool.yaml` route `prime: {class: chat, priority: background}`. `orion/llm/routes.py` lists it as accepted and background.
- **Body economy**: family `prime` (Tier 1, lane `chat`, window 1800 s) in `families.yaml`. `contract.yaml` registers `prime_loop.py` as a bidder that obeys its own bid.

## Departure from the design (Juniper decided, 10-30)

The spec's 2A effector was "Prime's act candidates go into the allocator". Live dispatch frames showed that reading, painting and retro leisure are already admitted on almost every tick: 3,519, 2,129 and 3,322 a day. Their own doers' caps hold them back, not the allocator. So a Prime candidate for any of them would change nothing, and the narrator test would read "no difference" because of how it was wired.

Juniper chose the curiosity question instead. It reuses Hub's existing focus slot rather than adding an allocator route. Prime does not add curiosity runs; it steers what the next run looks at. So it needs no bid of its own and no allocator slot.

Also deliberately not done in 2A: `prime.think` as a cortex verb. Prime calls the gateway directly, the same way `run_mind` does, and bids straight to `/physiology/bid`. A verb that nothing executes would be ceremony.

## Files changed

- `services/orion-mind/app/prime_loop.py`: resident loop (tick, bid, thought, nominate, settle, persist).
- `services/orion-mind/app/main.py`: start/stop Prime in the lifespan; mark Juniper turns.
- `services/orion-mind/app/llm_client.py`: `MindLLMClient(rpc_health=False)` for Prime. Its waits are by design, so they stay out of the RPC-health window.
- `services/orion-mind/app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: `PRIME_*` keys and documentation.
- `orion/prime/{state,thought,acts}.py`: pure logic. `orion/prime/evals/`: the frame eval (live fixture) and the narrator/cadence report.
- `orion/curiosity/focus.py`: `keep_pending`; Prime-aware wording (`is_prime_focus`).
- `services/orion-hub/scripts/curiosity_focus_listener.py`: accepts the `prime` context.
- `orion/schemas/introspect.py`: `invocation_context` gains `"prime"`.
- `orion/grammar/acts.py`: orion-mind is a registered act producer.
- `orion/bus/channels.yaml`, `config/metrics/metric_definitions.lock.json`: routing entries and the re-lock.
- `config/gpu_pool.yaml`, `orion/llm/routes.py`: the `prime` route.
- `config/physiology/families.yaml`, `config/physiology/contract.yaml`: the `prime` family and the reader registration.
- `orion/schema_skew_discovery.py`: note that orion-mind also writes `QueueCuriosityFocusArguments`.
- `scripts/sync_local_env_from_example.py`: `PRIME_` prefix and orion-mind in the default services. Without these, the 11 keys would never sync.
- `.github/workflows/orion-prime-tests.yml`: new CI.
- Tests: `orion/prime/tests/*`, `services/orion-mind/tests/test_prime_loop.py`, `test_rpc_health_publish.py`, `orion/curiosity/tests/test_focus_lifecycle.py`, `services/orion-hub/tests/test_curiosity_focus.py`, `orion/gpu_pool/tests/test_route_view.py`, `orion/introspect/tests/test_queue_curiosity_focus_tool.py`.

## Schema / bus / API changes

- **Added:**
  - `IntrospectToolBindingV1.invocation_context` value `"prime"`.
  - Gateway route `prime`.
  - Want family `prime`.
  - Act actor `prime` (`infer`, `ask`).
  - Redis key `orion:prime:state`.
- **Behavior changed:**
  - `queue_focus(keep_pending=True)` refuses whenever a focus is pending. Default is False, so nothing else changes.
  - A focus row whose `chat_correlation_id` starts with `prime:` is described to the run as chosen between conversations.
- **Compatibility:** consumer-first. Hub must run this build before orion-mind sends `"prime"`. An older Hub would reject it, and Prime records the question as refused, closed at 0. No migration.

## Env/config changes

- **Added keys (orion-mind):**
  - `PRIME_ENABLED=true`
  - `PRIME_TICK_SEC=15`
  - `PRIME_REFRACTORY_SEC=300`
  - `PRIME_SETTLE_INTERVAL_SEC=300`
  - `PRIME_LLM_ROUTE=prime`
  - `PRIME_BID_LANE=chat`
  - `PRIME_MAX_TOKENS=700`
  - `PRIME_LLM_TIMEOUT_SEC=60`
  - `PRIME_HOLDBACK_FRACTION=0.2`
  - `PRIME_DURABLE_RUNS_URL`
  - `PRIME_HUB_URL`
- **`.env_example` updated:** yes.
- **Local `.env` synced** with `python scripts/sync_local_env_from_example.py`: yes, all 11 keys added to `services/orion-mind/.env`. `PRIME_REFRACTORY_SEC` was then hand-set to 300, because the template moved after the first sync.
- **Skipped keys requiring operator action:** none.

## Metric quality gate (the question's grading signal)

1. **Provenance.** A question's outcome is the graded curiosity run's `outcome_kind`, as built in `orion/curiosity/run_story.py` and served by Hub's `GET /curiosity/api/runs`. Produced means `finished` or `reached_out_*`.
2. **Independence.** It is distinct from the run act's own `artifact_produced`, which only records that the run completed. 172 of 195 runs completed, but only 89 wrote something.
3. **Theory anchor.** A randomized holdback, so treated and control differ only in whether the question was sent. This is a standard A/B design. The difference is tested with Fisher's exact test, not by comparing averages.
4. **Live data.** 30 days of investigate runs (n=195): 89 wrote something, 83 wrote nothing, 19 died. The signal is not saturated, and either arm can show any value.
5. **Existing mechanism.** It reuses the focus slot, the run story and the act ledger. Nothing new is stored.
6. **Reversibility.** It lives only in act rows. `PRIME_ENABLED=false` stops it.

The body-change source in the tick reads the body's own `bind_state == "binding"` label. Live it is empty now, because every constraint reads slack or exogenous. It returns to empty whenever nothing binds, so it can rest.

## Tests run

```text
orion/prime                                         30 passed
services/orion-mind/tests                           138 passed
orion/curiosity + introspect + physiology + grammar + harness + schemas   1400 passed, 24 skipped
services/orion-hub/tests/test_curiosity*.py         534 passed, 3 skipped
orion/gpu_pool/tests + orion/llm/tests              539 passed
orion-static-gates.yml, all steps run locally       all PASS (definition drift PASS after the re-lock)
gpu-lane-controller-tests, orion-reading-tests, body-economy-analysis   PASS
tests/test_*{channel,schema,route,registry,...}     936 passed, 4 failed. The same 4 fail on clean origin/main
  (cortex_route_resolution, exec_result_channel_catalog, field_channel_glossary, single_consumer_channels_gate).
orion-gpu-pool-tests: 4 steps fail locally on missing deps. Identical on clean origin/main.
```

## Evals run

```text
PYTHONPATH=. python orion/prime/evals/run_prime_frame_eval.py   (live inputs captured 2026-10-30)
PASS grounded: 9 arc ids shown, unknown=[]
PASS bounded: 2999 chars (cap 12000)
PASS no_shells
PASS honest_gaps: every failed read is a named gap
PASS settles: control settles on 8ce4e099c883
PASS move_check_reads_live_shape

PYTHONPATH=. python orion/prime/evals/run_prime_report.py --dsn ...   runs live, empty (nothing deployed yet)

Live model smoke (metacog_background, Qwen3-8B; the prime route is not on the live gateway yet):
4 calls, all parsed, 3-7 s, ~200-240 completion tokens of 700. The runtime correctly dropped ids the
model invented (it used open-loop refs and arc ids as concern ids).
```

## Docker/build/smoke checks

```text
docker exec orion-mind: reached http://orion-athena-durable-runs:8121/physiology/state OK and
http://100.92.216.81:8080/curiosity/api/runs OK, which are the default PRIME_* URLs.
No image build run in this session. Runtime: UNVERIFIED until deployed (see Restart required).
```

## Review findings fixed

- **Finding:** a treated question whose focus was replaced, cleared or never written blocked every new question for about 49 h.
  - **Fix:** `treated_focus_gone` settles it after 2 consecutive reads that show it neither pending nor taken.
  - **Evidence:** `test_a_focus_replaced_in_hub_settles_after_two_reads_and_frees_prime`.
- **Finding:** "no bid during a Juniper turn" was described wrongly.
  - **Fix:** the docs now say the real mechanism. `run_mind` blocks the event loop, so no tick runs during a turn. What actually protects the turn is the overlap discard plus the background route. The concurrency of `run_mind` was left unchanged.
  - **Evidence:** README and module docstring.
- **Finding:** a frame build that raised made a hot bid loop and left acts open.
  - **Fix:** the refractory interval now starts when a thought opens. Its act always closes. Number casts are defensive.
  - **Evidence:** `test_a_crashing_thought_closes_its_act_and_still_waits_out_the_refractory`.
- **Finding:** Prime's background-route waits polluted orion-mind's RPC-health window.
  - **Fix:** `MindLLMClient(rpc_health=False)`.
  - **Evidence:** `test_prime_calls_never_land_in_the_published_window`.
- **Finding:** a concern on the arc that attention stays on faded after 24 h.
  - **Fix:** its evidence now refreshes every tick.
  - **Evidence:** `test_a_concern_on_the_arc_attention_stays_on_keeps_its_evidence`.
- **Finding:** a Prime call might occupy the chat slot after the client gave up.
  - **Fix:** none needed. The gateway already generates nothing for a call granted after its budget (`services/orion-llm-gateway/app/main.py`, "grant came after the caller's budget ran out").
  - **Remaining risk:** whether upstream generation stops when the gateway's read times out is UNVERIFIED.

## Restart required

Deploy from the primary main checkout after merge. The post-pull rebuild should pick these up. Order: hub, llm-gateway, gpu-pool, then durable-runs, then mind.

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only
```

If the post-pull rebuild skips any of them, rebuild that one by hand: `scripts/safe_docker_build.sh <service> up -d --build` for orion-hub, orion-llm-gateway, orion-gpu-pool, orion-durable-runs, and orion-mind last. On circe, pull so the lane controller loads the new route.

Live checks after deploy:
- `docker logs orion-mind | grep prime_started`
- `grammar_acts WHERE actor='prime'`
- `want_ledger WHERE family='prime'`
- the first `curiosity_focus_request` row with `chat_correlation_id LIKE 'prime:%'`

## Risks / concerns

- **Severity: medium.**
  - **Concern:** E9. Prime shares chat lane 0 (one slot) with Juniper. A Prime call already generating cannot be cancelled. Tier 1 bids are funded 351/352 today, so the 300 s floor, not the price, sets the rate: at most 12 thoughts an hour, about 3-7 s each on the smoke model.
  - **Mitigation:** background priority, 700-token cap, overlap discard. Measure chat latency with Prime on vs off in the first days (`PRIME_ENABLED`).
- **Severity: medium.**
  - **Concern:** Tier 1 for `prime` needs Juniper's sign-off (families.yaml rule; design open question 2).
  - **Mitigation:** flagged here. Changing to Tier 2 is a one-line edit.
- **Severity: low.**
  - **Concern:** the control arm's graded run can also take a chat-queued focus. That adds noise to the comparison, not bias.
  - **Mitigation:** focus use is rare (1 run in 30 days took one).
- **Severity: low.**
  - **Concern:** Temporal Self arcs often carry opaque labels (`open-loop-…`), so Prime's attention view is thin.
  - **Mitigation:** rung 1 (attention faces the world) is still open. This is a known gap, not something 2A creates.

## PR link

(filled after `gh pr create`)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
