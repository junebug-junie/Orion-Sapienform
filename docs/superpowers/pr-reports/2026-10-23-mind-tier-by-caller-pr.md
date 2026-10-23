# fix(mind): lift Mind to the interactive tier only on Juniper's marked turn

## Summary

- Mind now treats a run as "Juniper is waiting" only when the run says so: `utterance_origin="juniper"`. Unknown or `"orion"` means not live, so the run stays on the heat-sheddable `metacog` route (system tier). (`services/orion-mind/app/engine.py::carries_live_turn`)
- Before this patch the rule was inverted: anything not marked `"orion"` was lifted. World-pulse reads, outreach and the collapse-mirror reply set nothing, so they rode Juniper's never-shed tier.
- The brain-mode chat path with the Mind toggle (Hub -> cortex-orch -> Mind) never carried a mark. It now does: Hub's chat builder writes `metadata.utterance_origin="juniper"` for the websocket and HTTP chat call sites, and cortex-orch copies it onto the Mind request. Without this, the new rule would have silently demoted that Juniper path.
- A call-site table test pins the origin that every turn-building caller passes. A new caller fails the test until someone decides who is waiting on it. A new workflow runs the whole chain.
- The 60 s world-pulse stage-2 timeouts are explained (prompt too big for the metacog seat, see below). Not fixed here: it is not a timeout bug.

## Outcome moved

- Orion's own turns stop taking Juniper's tier. Live, since Juniper's last chat (2026-10-21 07:02 UTC) to 2026-10-23 18:40 UTC: 96 interactive `orion-mind` grants (1,230 s held) went to Orion-started turns. After deploy that number should be 0.
- Juniper's turns keep the interactive tier on every path, including the brain-mode Mind toggle path that had no mark.

## Current architecture

- `carries_live_turn(req)` returned `req.trigger == "user_turn" and req.utterance_origin != "orion"`. `trigger` defaults to `"user_turn"`; `utterance_origin` defaults to `None`.
- When true, every phase configured as `metacog` (semantic_synthesis, active_frontier_judge, stance_handoff) moves to `MIND_TURN_MODEL_ROUTE=metacog_turn` (`config/gpu_pool.yaml`: class metacog, priority interactive). All three phases lift together.
- Two producers build `MindRunRequestV1`:
  - orion-thought `build_light_mind_request`, from `stance_inputs.utterance_origin`, set by `execute_unified_turn(utterance_origin=...)`.
  - cortex-orch `build_mind_run_request`, run only when request metadata has `mind_enabled: true` (set by the Hub browser toggle). It never set `utterance_origin`.

## Juniper-path inventory (proof per path)

| Path | How it reaches Mind | Mark | Proof |
|---|---|---|---|
| Hub websocket chat, mode orion/agent | `run_unified_turn` -> `execute_unified_turn` -> orion-thought | `"juniper"` (already) | Call-site table test; `test_turn_orchestrator_utterance_origin.py` (stance_inputs); `test_mind_utterance_origin.py` (Mind request). Live: 38 of 38 `hub_orion` turns 10-14..10-21 have Mind grants, all `interactive`, 3 grants per turn (114 total). |
| Hub HTTP `/api/chat`, mode orion/agent | `handle_chat_request` -> `execute_unified_turn` | `"juniper"` (already) | Call-site table test. |
| Hub websocket chat, brain mode + Mind toggle (`hub_ws`, `app.js` / `thought-process.js` set `mind_enabled`) | `build_chat_request` -> cortex-gateway -> cortex-orch -> `/v1/mind/run` | **was none; now `"juniper"`** | Call-site table test; builder test; orch test. Live: 19 brain-mode Mind runs in 60 days (last 2026-10-21 05:54, corr `48039bf2-c438-5a62-8ae3-d13f25041fdc`, `interactive` only because unmarked was lifted). Under the new rule without this fix, it would have been shed. |
| Hub HTTP `/api/chat`, brain mode + Mind toggle | same as above | **was none; now `"juniper"`** | Call-site table test. |
| Hub UI's own social room (`hub_direct`) | brain mode via websocket | `"juniper"` kept | `test_hub_direct_social_room_is_still_juniper`. |
| Harness / Agent mode | same unified turn as orion mode | `"juniper"` | Same as first row (websocket `client_mode in ("orion","agent")`). |
| Agent-Claude mode | does not call Mind | n/a | `_run_agent_claude_turn_ws` has no Mind or thought call. |

Not Juniper, never marked: orion-social-room-bridge relays (`external_room.platform != "hub"`; `SOCIAL_BRIDGE_HUB_MODE=brain` live, no Mind toggle), `memory_graph_suggest`, `scenario_replay`. A client payload cannot claim the mark (`test_client_payload_cannot_claim_juniper`).

## Orion-started callers

| Caller | Session | Origin passed | Tier after deploy |
|---|---|---|---|
| curiosity / self-inquiry / self-sense-eval / urgent (`curiosity_investigation.py`) | `orion_curiosity`, `self-sense-eval` | `"orion"` (already) | system (already system live) |
| world-pulse read (`reading_turn_listener.py`) | `orion_world_pulse_read`, `..._stage2` | None | system (was interactive) |
| endogenous outreach (`endogenous_outreach.py`) | `orion_journal`, `orion_outreach` | None | system (was interactive) |
| collapse-mirror reply (`collapse_mirror_chat_reply.py`) | caller's session | None | system (was interactive) |

**Deviation from the brief:** the brief asked to mark these `"orion"`. I did not. In `execute_unified_turn`, `"orion"` is more than a label. It also turns on the curiosity role-teach splice: `ensure_orion_hire_depth` forces `expected_depth=deep`, and the motor prompt gets "Mind reads this sitting as deep work. Write hire_cursor and HelpRequest now". It also adds the stance note "self-authored investigation subject" (`mind_enrichment.ORIGIN_NOTES`). Both are wrong for a world-pulse read, an outreach message or a collapse-mirror reply. `orion/schemas/harness_finalize.py` already documents None for these callers. With the new default, None already gives the system tier. The call-site table test pins None, so nobody can lift them by accident.

## Live inversion (evidence)

`mind_runs` joined to `gpu_pool_events` on `turn_correlation_id`, last 72 h (to 2026-10-23 18:40 UTC):

| Session | Tier | Runs | Grants |
|---|---|---|---|
| orion_journal (Juniper, `hub_orion`) | interactive | 38 (8 days) | 114 |
| orion_journal (outreach, not in chat log) | interactive | 20 (8 days) | 58 |
| orion_world_pulse_read | interactive | 49 | 49 |
| orion_world_pulse_read_stage2 | interactive | 32 | 5 |
| orion_outreach | interactive | 2 | 6 |
| orion_curiosity | system | 6 | 18 |
| self-sense-eval | system | 100 | 296 |

## World-pulse stage-2 60 s timeouts: root cause

**The stage-2 prompt is too big for the metacog seat, so it waits for a big seat that is busy.**

- Stage-2's semantic_synthesis call needs 4,103 to 7,044 tokens of context (`gpu_pool_leases.min_ctx_tokens`, 32 runs). The metacog and fast seats hold 4,096 tokens per slot. Only the agent seats (131,072) fit.
- The agent seats are held for minutes by other work (durable-runs readings at background priority, cortex-exec agent calls of 90 to 160 s). Example, corr `6b82856e-87f1-51db-a969-226a4c3c6c08` 2026-10-23 18:30:49: admitted on metacog_turn, never granted, while cortex-exec and vision-council metacog leases were granted around it within 150 ms.
- The gateway caps the pool wait at the caller's own budget (`resolve_caller_budget_sec`, 60 s from `MIND_LLM_TIMEOUT_SEC`). So the gateway logs `gpu_pool_unavailable reason=deadline`, the pool records `cancelled`, and Mind logs `TimeoutError` within the same 100 ms. The timeouts are aligned; none of them is wrong.
- Counts (72 h): stage-2 27 of 32 leases cancelled, 4 granted on `agent`, 1 on `metacog` (min_ctx 3,384). Every other Mind session stays under 4,096 (max 3,772).

Not fixed here: there is no small, clearly correct change. Options, for whoever owns world-pulse reads:
1. Bound what stage-2 sends to Mind so semantic_synthesis fits in 4,096 (orion-thought `_MAX_USER_TEXT_CHARS` is 20,000 chars; the gateway estimate is chars/4 + up to 1,024 answer tokens). Cheapest.
2. Skip the Mind run on stage-2 reads. Its coloring cannot land if it never gets a seat.
3. Not recommended: a longer Mind timeout. It holds the stage-2 turn for minutes behind agent work.

After this patch these runs are `system` instead of `interactive`. That does not change the ctx fit.

## Fallback rate (corrects the 44/71 figure)

Last 72 h: 92 of 228 Mind runs returned `fallback_contract_only`. By cause:
- world-pulse read stage 1: 49 of 49, `semantic_synthesis_empty`. Granted on metacog, so the model answered, but the synthesis was empty. Separate bug, not investigated here. Possibly Qwen thinking eats the token budget, as seen before.
- world-pulse stage 2: 31 of 32 RPC timeouts (above), 1 `semantic_synthesis_empty`.
- self-sense-eval: 8 of 108, `empty_llm_response`. 7 of these have no pool row; they are probably gateway-local shed refusals (UNVERIFIED, see handoff point 3).
- orion_journal: 1 of 24, `json_parse_failed`.
- Juniper's chat turns: no fallback seen in the sample.

## Handoff back to the unified-turn self-prediction arc (#2869)

To: the session that wrote `docs/superpowers/specs/2026-10-23-unified-turn-self-prediction-design.md` ("Tiers") and board item `a83a796f`.

**1. What you asked, and the corrected finding.** You found that Mind's stance_handoff and frontier_judge ran at system priority on Juniper's turn path, so they could be heat-shed. That is not what happens. On Juniper's turn all three Mind steps move together to `metacog_turn` (interactive). Both Hub chat paths mark the turn `"juniper"`. Live: 38 of 38 Juniper turns 10-14..10-21 show 3 interactive Mind grants each. The real bug was the inverse. Orion-started turns also got Juniper's interactive tier, because "not marked" counted as "Juniper is waiting". Since 10-21 07:02, with no Juniper chat, Orion's own turns took 96 interactive Mind grants.

**2. What this PR changed.**
- The rule: a Mind run is on Juniper's tier only if it is marked. Field `MindRunRequestV1.utterance_origin`, value `"juniper"`. Unknown means not live.
- Unified turn callers set it with `execute_unified_turn(utterance_origin="juniper")`. Brain-mode chat callers set it with `build_chat_request(utterance_origin="juniper")`. Do not set it on anything Juniper did not type.
- Juniper paths with proof: see "Juniper-path inventory" above. One path was missing the mark (brain-mode chat with the Mind toggle) and now has it.
- Orion callers: none was changed to `"orion"`. See the deviation note above. Curiosity, self-inquiry and self-sense-eval already pass `"orion"`. Reading, outreach and collapse mirror pass nothing, which now means system.

**3. What it means for your design.**
- Tier per stance caller: Juniper chat -> interactive (never shed). Reading, self-sense eval, journal outreach, curiosity, outreach, collapse mirror -> system (shed by the heat reflex).
- An Orion-started Mind run can now be refused under the 34 °C reflex. Your self-prediction must treat a shed Mind run as **no reading**. Never treat it as calm or zero.
- Warning: a shed run does not say "shed" in `mind_runs`. The gateway answers a shed with empty text, so Mind records `mind_quality=fallback_contract_only` with `llm_fail_open:empty_llm_response`, the same as a model that really returned nothing. Live: the two `shed:cabinet_hot` refusals on 10-20 both read exactly that way.
- A shed can also leave no pool row. When the gateway is in shed backoff it refuses locally and does not ask the pool. 7 of the 8 self-sense-eval `empty_llm_response` runs in the last 72 h have no `gpu_pool_events` row at all. They are probably local sheds, but this is UNVERIFIED because the gateway restarted and its logs are gone. Read `gpu_shed_backoff_start/end` in the gateway log, or treat "empty reply and no grant" as "no reading".
- To read the tier of a run: join `gpu_pool_events` (holder `orion-mind`, column `priority`, and `reason` like `shed:%`) on `turn_correlation_id` = `mind_runs.correlation_id`. For Juniper's turns, `chat_history_log.correlation_id` is the same id.

**4. The 60 s world-pulse stage-2 timeout.** Cause: the stage-2 prompt needs 4.1k to 7k tokens, and only the busy 131k agent seats fit. The 60 s caller budget runs out in the queue. This PR does not fix it. Owner: the world-pulse read arc (stage-2 prompt size), or orion-thought if Mind input should be capped for all callers. Recommendation: option 1 above.

**5. Your live facts, updated.** "44/71 fallback, 16 timeouts" -> in the 72 h to 2026-10-23 18:40: 92 of 228 runs fell back. 49 are stage-1 empty synthesis, 31 are stage-2 queue timeouts, 8 are self-sense empty replies, 1 is a JSON parse failure. The timeouts were on `metacog_turn` only because of the inversion. After deploy they will be on `metacog` (system).

**6. Open items for you.**
- A live Juniper turn shows all three Mind steps at interactive. Verify after deploy: `select priority, count(*) from gpu_pool_events where holder='orion-mind' and event='granted' and turn_correlation_id='<corr>' group by 1` returns `interactive|3`. Cite a correlation id once she chats. UNVERIFIED until then.
- Orion's own turns are off interactive. Verify: the same query grouped by priority, for grants after deploy whose correlation id is not in `chat_history_log` with source `hub_orion`/`hub_ws`. Expect 0 interactive.
- A brain-mode chat with the Mind toggle lands interactive. Verify only if Juniper uses that mode. It is rare (19 runs in 60 days).
- Shed Mind runs read as "no reading" in the self-prediction design. Verify with a fixture in your eval.

**7. References.** This PR (link below). #2855 body-economy design, Organ 3b (tiers attach to callers). #2869 unified-turn self-prediction design. Board item `a83a796f-b09e-4cc8-808b-42160a30d6fb` (left open, pointer added: the timeout cause is reported, not fixed).

## Proposal-mode notes

- Capability change: Orion's own turns (reads, outreach, collapse mirror) can now be heat-shed. This is correct per the tiers design: only Juniper's waiting turn is protected.
- Data touched: none stored. One request metadata key (`utterance_origin`) on Hub -> cortex chat requests.
- Privacy boundary: unchanged.
- Trace that proves it: `gpu_pool_events.priority` per `turn_correlation_id` (queries above).
- Dangerous failure: a Juniper path missing the mark gets shed under heat, so her turn loses Mind coloring or waits. Guard: the call-site table test fails on any new unclassified caller. Every known Juniper path is pinned.
- Rollback: revert this PR. For an immediate partial rollback, `MIND_TURN_MODEL_ROUTE=` (empty) puts every run on system, which is not what you want for Juniper. Revert is the right rollback.

## Architecture touched

orion-mind (tier decision), cortex-orch (copy the mark), orion-hub (chat builder plus 2 call sites). No bus, schema or channel change: `MindRunRequestV1.utterance_origin` already existed.

## Files changed

- `services/orion-mind/app/engine.py`: `carries_live_turn` requires `"juniper"`; docstring lists the Juniper paths.
- `services/orion-mind/app/settings.py`, `services/orion-mind/.env_example`: comments only.
- `services/orion-mind/tests/test_mind_turn_route.py`: Juniper lifts all three phases; None and Orion stay; world-pulse regression; truth table.
- `services/orion-cortex-orch/app/mind_runtime.py`: copies `metadata.utterance_origin` onto the Mind request.
- `services/orion-cortex-orch/tests/test_mind_utterance_origin_tier.py`: new.
- `services/orion-hub/scripts/cortex_request_builder.py`: `utterance_origin` kwarg; writes the metadata mark; skips third-party social rooms.
- `services/orion-hub/scripts/websocket_handler.py`, `services/orion-hub/scripts/api_routes.py`: pass `"juniper"` for brain-mode chat; the HTTP unified branch drops the mark for an external social-room relay.
- `services/orion-hub/tests/test_mind_tier_by_caller.py`: new call-site table gate plus builder tests.
- `.github/workflows/mind-tier-by-caller-tests.yml`: new; runs the chain of tests.

## Schema / bus / API changes

- Added: `build_chat_request(..., utterance_origin=None)` kwarg; request metadata key `utterance_origin`.
- Removed / Renamed: none.
- Behavior changed: an unmarked Mind run is no longer lifted to interactive.
- Compatibility: the new metadata key is ignored by an old cortex-orch. An old Hub sends no mark, so a new Mind would shed the brain-mode toggle path. Deploy Hub and cortex-orch before Mind.

## Env/config changes

- Added / Removed / Renamed keys: none.
- `.env_example` updated: comment only (`services/orion-mind/.env_example`).
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes, no key changes.
- Skipped keys requiring operator action: none.

## Tests run

```text
services/orion-mind:        pytest tests -q                                   -> 122 passed
services/orion-mind:        pytest tests/test_mind_turn_route.py (CI style)   -> 12 passed
services/orion-cortex-orch: pytest tests/test_mind_utterance_origin_tier.py tests/test_mind_orch.py -> 25 passed
services/orion-cortex-orch: pytest tests -q -> 31 failed / 231 passed; main baseline 32 failed / 223 passed (pre-existing env failures, none in touched code)
services/orion-thought:     pytest tests/test_mind_utterance_origin.py        -> 4 passed
services/orion-hub:         pytest tests/test_mind_tier_by_caller.py tests/test_cortex_request_builder.py tests/test_turn_orchestrator_utterance_origin.py -> 62 passed, 1 failed (test_execute_unified_turn_uses_mind_appraisal_text_for_stance_not_harness, fails identically on main)
Hub after review fixes: tests/test_mind_tier_by_caller.py + test_cortex_request_builder.py 51 passed; tests/test_handle_chat_request_*.py + test_http_chat_spark_meta.py 36 passed, 1 failed (test_handle_chat_request_retries_memory_graph_suggest_on_timeout, fails identically on main).
Mutation check: removing the websocket "juniper" mark fails test_every_call_site_passes_its_expected_origin.
Static: check_body_economy_contract OK; check_chat_route_poachers PASS; check_env_template_parity PASS; check_async_routes_not_blocking OK; test_no_personal_terms + test_check_body_economy_contract 43 passed; git diff --check clean.
```

## Evals run

```text
No eval harness for Mind tier routing. The regression is pinned by tests plus the live SQL checks in the handoff section (run after deploy).
```

## Docker/build/smoke checks

```text
Not run: no Dockerfile, requirements or compose change. Live verification is the post-deploy SQL above (UNVERIFIED until deploy plus a Juniper turn).
```

## Review findings fixed

Review by an orion-repo-agent subagent (read-only). It found no Juniper path to Mind without the mark, and no correctness bug.

- Finding: the workflow's PR path filter named only specific files, so a PR adding a new caller in an unlisted file would not run the gate.
  - Fix: triggers now cover the gate's scan roots (`services/orion-hub/scripts/**`, `services/orion-hub/app/**`, `orion/**`) on PR and push.
  - Evidence: `.github/workflows/mind-tier-by-caller-tests.yml`.
- Finding: the HTTP unified branch (`mode` orion/agent) passed `"juniper"` unconditionally, so a social-room relay in that mode would carry Juniper's mark (pre-existing; live bridge mode is `brain`).
  - Fix: `utterance_origin=None if is_third_party_room(payload) else "juniper"`; the helper is now public in the builder.
  - Evidence: `test_unified_http_turn_from_external_room_is_not_juniper`; the call-site table pins the conditional expression.
- Finding: the call-site gate had blind spots: wildcard rows, `run_unified_turn` (hardcodes juniper) not scanned, direct `StanceReactRequestV1` construction not scanned, aliased imports invisible.
  - Fix: exact (file, function) rows only; `run_unified_turn` and `StanceReactRequestV1` added to the scan; a new test refuses aliased imports of the scanned names.
  - Evidence: `services/orion-hub/tests/test_mind_tier_by_caller.py`, 10 passed.
- Finding: one CI job installed four requirement files with conflicting pins (last wins).
  - Fix: one job per service.
- Finding (nit): long `.env_example` comment line.
  - Fix: re-wrapped.
- Not changed (reported only): `services/orion-hub/evals/run_self_sense_eval.py` posts `mode=orion` to `/api/chat`, so a manual eval run carries Juniper's mark (same as before). cortex-orch honors `metadata.utterance_origin` from any gateway caller; today only Hub reaches orch with `mind_enabled`.

## Restart required

Deploy order: orion-hub, then orion-cortex-orch, then orion-mind. From the clean primary `main` checkout after merge:

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-cortex-orch up -d --build
scripts/safe_docker_build.sh orion-mind up -d --build
```

## Risks / concerns

- Severity: medium. Concern: a Juniper path not in the inventory. Mitigation: the call-site gate covers every `execute_unified_turn` / `build_chat_request` caller under `orion/` and `services/orion-hub/`; a direct HTTP caller of `/v1/mind/run` other than cortex-orch does not exist today.
- Severity: low. Concern: stage-2 world-pulse Mind runs still time out. Mitigation: reported with options; out of scope.

## PR link

TBD
