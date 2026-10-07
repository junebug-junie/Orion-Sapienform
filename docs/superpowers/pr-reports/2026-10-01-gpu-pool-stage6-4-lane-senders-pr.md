# GPU pool stage 6.4: who still sends a lane (census before deleting lane routing)

## Summary

- The gateway now counts every LLM call that carries a lane, and every call that lane routing moves to a different route. For each one it records who sent it, which lane, the route it uses today, and the route it would get once lane routing is deleted. The numbers are at `GET /debug/lane-senders`, and each call also writes one `llm_gateway_lane_sender` log line, which survives a restart.
- Routing is unchanged. The counter is guarded, so a failure inside it cannot fail the call it is counting.
- A read-only census from code and live data answers the question: **deleting lane routing today is not a no-op.** About 3,070 background calls a day would move from the metacog class to the fast class (`quick`). Details below.
- The deletion is therefore not "remove `lane_routes.py`". The callers have to name their routes first, and then the 24 h window has to show zero calls moved. That plan is the follow-up section below.

## Outcome moved

Before this PR, nothing could say which calls lane routing actually changes. The gateway's existing log line normalises a missing lane to `chat` and does not name the sender. After it, one endpoint answers "would deleting `lane_routes.py` change anything?" as a number (`rerouted_total`), broken down by sender.

## Current architecture

`LLM_LANE_ROUTING_ENABLED=true` (live). `plan_llm_chat` (`services/orion-llm-gateway/app/llm_backend.py`) runs `lane_routes.resolve_llm_lane_route` on every bus chat call, except calls that carry a GPU pool hold (`options.gpu_lease`). It reads `options.llm_lane or options.execution_lane`.

`config/gpu_pool.yaml` has no `spark` or `background` route, so lane routing maps lanes to routes like this:

| Lane | Route lane routing picks | Route `_resolve_route` picks (after deletion) |
|---|---|---|
| `background` | `metacog` (metacog class, system priority) | the call's own route, `options.route`/`routing_key`, else `LLM_ROUTE_DEFAULT` (`quick`) |
| `spark` | `metacog` (spark falls back to background) | same as above |
| `agent` | `agent` | same as above |
| `chat`, invalid, or none | the call's route if the pool knows it, else `quick`; ignores `options.route`/`routing_key` when the call's route is empty | same as above; a route name the pool doesn't know is refused (`route_not_in_pool`) |

## Who sends a lane today

**From code:**

- **cortex-exec** is the main sender. Every LLM step spreads `lane_opts` from `app/llm_lane.py` into the gateway options (`executor.py` ~4400-4413), so `llm_lane` and `execution_lane` are always present.
  - The lane follows the container's `EXEC_LANE` and the verb. Anything on the `-background` container, or a `_BACKGROUND_VERBS` verb, gets `background`. Anything on `-spark`, or `introspect_spark`, gets `spark`.
  - It also has two side calls with a hardcoded lane: journal_pageindex (`executor.py` ~2406, background, no route) and the MetacogDraft for `log_orion_metacognition` (~3400, background, route `metacog_background`).
- **orion-dream** (`app/llm.py:22-29`) calls the gateway directly with lane `background` and route `metacog_background`.
- **These pass a lane only through cortex-exec:** harness finalize (`orion/harness/finalize.py:401/856`), thought stance (`bus_listener.py:214`), reverie (`reverie.py`), and the visual chain (`visual_chain.py`). Their leased calls carry `gpu_lease`, so the gateway skips lane routing for them.
- **Lane only in metadata, never in gateway options:** cortex-orch (`orchestrator.py:475/537`, `mind_runtime.py:197`).
- **No lane at all:** orion-mind (`metacog`), vision-council (`metacog_background`), topic-foundry (no route, so `quick`), memory-consolidation, hub, and cortex-orch's direct calls.
- **Nobody** sends `options.route` or `options.routing_key`. The quirk where lane routing ignores those fields has no live sender.

**From live data (read-only):**

- `docker logs orion-llm-gateway` only covers about 5 minutes, because the gateway restarted at 2026-10-01T01:24:47Z. In that window: 79 topic-foundry calls on chat → `quick` and 1 vision-council call on chat → `metacog_background`, with no route changes.
- For the 24 h view I used the `gpu_pool_leases` table, rows created in the last 24 h, holds excluded:

```text
cortex-exec|metacog|system|3078
cortex-exec|chat|interactive|1520
orion-topic-foundry|fast|system|1231
http:anthropic|agent|system|1152
orion-mind|metacog|system|529
cortex-exec|agent|system|309
vision-council|metacog|background|242
cortex-exec|fast|system|215
dream|metacog|system|16          <- dream asks for metacog_background; lane routing rewrites it to metacog
cortex-exec|fast|background|1    <- exec's quick -> quick_background demotion is almost never effective
```

- The census subagent joined cortex-exec leases to verbs via `cognition_traces`. Most of the 3,078 metacog/system leases are background-lane steps:
  - journal.compose: 1,919
  - reverie_narrate: 992
  - reverie_expectation_judge: 158
  - visual_context_interpret: 13
  - log_orion_metacognition: 9

## What deleting lane routing would do today

| Sender (24 h volume) | Lane | Today | After deletion |
|---|---|---|---|
| exec journal.compose + pageindex (~1,919) | background | `metacog` (metacog/system) | `quick` (fast/system), because its route is None and the default applies |
| exec reverie_narrate (~992) | background | `metacog` | `quick` (fast/system) |
| exec reverie_expectation_judge (~158) | background | `metacog` | `quick` (fast/system) |
| exec MetacogDraft (~9) | background | `metacog` (system) | `metacog_background` (metacog/background) |
| orion-dream (~16) | background | `metacog` (system) | `metacog_background` (metacog/background) |
| exec-spark / introspect_spark (0 in 24 h) | spark | `metacog` | that step's route, e.g. `quick_background` (fast/background) |
| visual_context_interpret, reverie with lift | background | `metacog` | `metacog` (no change) |
| all chat- and agent-lane callers, all calls under a hold | chat/agent | the call's route | same (no change) |

Net effect: about 3,070 calls a day would leave the metacog card for the fast card. That is the card topic-foundry and other system `quick` callers use. Dream and MetacogDraft would also drop from system to background priority. Deleting lane routing as things stand is a real routing change, not cleanup.

## Prepared follow-up: the deletion PR (do not start before the 24 h window)

**Gate:** deploy this PR, wait 24 h, then read `/debug/lane-senders`. Deletion can go ahead only if `uptime_sec >= 86400` and `rerouted_total == 0`, or every row with `rerouted: true` has an accepted answer. The census above says the window will **not** be zero today, so the work is two PRs.

1. **Make callers say what they mean** (routing stays identical, so the census should drop to zero):
   - cortex-exec: when the resolved lane is `background` or `spark` and the step has no explicit route override, set the body route to `metacog`. That is exactly what lane routing does today. Do it in `executor.py` next to `_apply_autonomous_background_route`. Optionally retire the dead `quick` → `quick_background` demotion, or make it real.
   - orion-dream and MetacogDraft: decide whether `metacog` at system priority (today's real behaviour) or `metacog_background` (what they ask for) is right. Then either change their route to `metacog`, or accept the priority drop and record that as the answer for their census row.
   - Deploy, re-run the 24 h window, and expect `rerouted_total == 0`, apart from rows answered on purpose.
2. **Delete lane routing:**
   - remove `services/orion-llm-gateway/app/lane_routes.py`
   - remove `tests/test_lane_routes.py` and `tests/test_llm_lane_run_llm_chat.py`
   - remove the lane branch in `plan_llm_chat`, with its `llm_gateway_lane_route`/`llm_gateway_lane_rejected` logs and the `llm_route_unavailable` error
   - remove the `llm_lane_default`/`llm_lane_routing_enabled` settings, the `LLM_LANE_ROUTING_ENABLED`/`LLM_LANE_DEFAULT` rows in `.env_example` and the README, and `LLM_LANE_*` from local `.env` (via `scripts/report_dead_env_keys.py`, once it exists)
   - remove `app/lane_senders.py`, `/debug/lane-senders`, and their tests
   - **Behaviour change in that PR:** the gateway would no longer change any call's route. One latent case remains: a call whose route the pool doesn't know (a typo) currently gets `quick`, and after deletion it is refused with `route_not_in_pool`. The census shows these as `route_without_lane_routing=rejected:<name>`, and none were seen live.
   - Lane metadata (`llm_lane`, `priority`, `allow_chat_fallback`) can keep flowing for logs. The gateway would just stop acting on it.

## Architecture touched

orion-llm-gateway only, in one function (`plan_llm_chat`). There is one new debug endpoint and no contract, bus, schema or env change.

## Files changed

- `services/orion-llm-gateway/app/lane_senders.py`: new. A bounded, locked in-process counter plus one log line per recorded call; it never raises.
- `services/orion-llm-gateway/app/llm_backend.py`: `plan_llm_chat` computes the route without lane routing and records each call on the applied, hold-skipped and disabled paths.
- `services/orion-llm-gateway/app/main.py`: `GET /debug/lane-senders`.
- `services/orion-llm-gateway/tests/test_lane_senders.py`: 26 tests.
- `services/orion-llm-gateway/README.md`: endpoint row.
- `docs/superpowers/pr-reports/2026-10-01-gpu-pool-stage6-4-lane-senders-pr.md`: this report.

## Schema / bus / API changes

- Added: `GET /debug/lane-senders` (debug HTTP only).
- Removed / Renamed: none.
- Behaviour changed: none. Routing is identical.
- Compatibility notes: none.

## Env/config changes

- Added / removed / renamed keys: none.
- `.env_example` updated: no. Local `.env` sync: not needed.
- Skipped keys requiring operator action: none.

## Tests run

```text
PYTHONPATH=$PWD .venv/bin/python -m pytest services/orion-llm-gateway/tests -q -p no:cacheprovider
378 passed
```

The tests include an oracle check. For a matrix of 13 call shapes (route or no route, `options.route`/`routing_key`, valid, invalid and blank lanes, a misspelled route), the census's predicted route must equal what `plan_llm_chat` actually picks with lane routing turned off.

## Evals run

No eval harness applies. The measurement is the 24 h live window after deploy (below).

## Docker/build/smoke checks

```text
Not deployed (per instruction). The live census is UNVERIFIED until deploy.
```

## Review findings fixed

A code-review subagent found no must-fix issues. Fixed:

- Finding: the counter ran unguarded on the request path.
  - Fix: `record()` and `lane_field()` swallow their own errors.
  - Evidence: `test_a_failing_counter_never_fails_the_call`.
- Finding: a route the pool doesn't know, after deletion, looked like "moves" when it really means "fails".
  - Fix: it is now recorded as `rejected:<name>`, and a call refused on both sides is not counted as a reroute.
  - Evidence: `test_unknown_route_rescued_...` and `test_both_sides_rejected_...`.
- Finding: the lane label used different logic from the router (`or` truthiness).
  - Fix: it now mirrors the router.
  - Evidence: `test_blank_llm_lane_is_labelled_like_the_router_reads_it`.
- Finding: reroutes could hide in the anonymous "other" overflow row.
  - Fix: overflow is split by `rerouted`.
  - Evidence: `test_rows_are_bounded`.
- Finding: no test proved the prediction matches real post-deletion routing.
  - Fix: parametrized oracle test.
  - Evidence: 13 cases.
- Finding: tests depended on the live `config/gpu_pool.yaml`.
  - Fix: the route table is pinned in a fixture.
- Finding: truncating route names before comparing could make two long names look equal.
  - Fix: names are compared in full and only truncated for display.
- Not changed: the snapshot returns a list rather than the sibling's dict, which reads better sorted by volume. INFO logging is kept, because the gateway runs uvicorn at `log_level="info"`.

## Restart required

Deploy, then measure for 24 h:

```bash
cd /mnt/scripts/Orion-Sapienform-<worktree-on-merged-main>
scripts/safe_docker_build.sh orion-llm-gateway up -d --build
# after >= 24 h of uptime:
curl -fsS http://localhost:8210/debug/lane-senders | python3 -m json.tool
docker logs --since 24h orion-llm-gateway 2>&1 | grep llm_gateway_lane_sender | grep 'rerouted=True' \
  | sed -E 's/corr=[^ ]+ //' | sort | uniq -c | sort -rn
```

## Risks / concerns

- Severity: low.
  - Concern: one extra INFO line per lane-bearing call, about 3-5k a day.
  - Mitigation: the existing `llm_gateway_lane_route` line already fires per call, and both go away with the deletion PR.
- Severity: low.
  - Concern: the in-process counter resets on restart.
  - Mitigation: the log line survives. Read `uptime_sec` before trusting `rerouted_total == 0`.
- Severity: medium (for the follow-up, not this PR).
  - Concern: deleting lane routing without step 1 moves about 3k calls a day onto the fast card.
  - Mitigation: the follow-up is gated on the census.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2448

🤖 Generated with [Claude Code](https://claude.com/claude-code)
