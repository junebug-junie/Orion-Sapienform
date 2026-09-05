# Incident: stance_react turns deferred, root cause is PR #2110, not the Flash-Next test session

Branch: `worktree-docs+attention-salience-gateway-starvation-incident`
Status: **BLOCKED** (root cause identified and evidenced; fix belongs to whoever owns PR #2110's attention-salience experiment, not this session)

## Summary

- Juniper reported `Turn deferred: stance_react_failed: stance_react exec
  result missing thought payload` recurring on 2026-09-05, after a multi-hour
  Qwen3.8-Flash-Next / Qwen3.8-27B-BF16 model comparison session (PRs #2099,
  #2100, #2108, #2112, #2113) had already been fully reverted and independently
  verified clean (chat/harness back on the 35B, agent back on the 27B,
  stance_react's own timeout back to 120s, all confirmed live from inside the
  running containers, not just from deploy logs).
- Initial hypotheses chased during live debugging (single-worker contention
  between chat_general and stance_react, gateway-vs-worktree `.env` drift,
  reverie's own recurring tick landing on the wrong route) were each real
  findings but did not explain why this started happening tonight specifically
  after "hundreds of successful turns" with no prior occurrence.
- The actual trigger: **PR #2110** (`chore(substrate): lower the attention
  salience gate 0.2 -> 0.05`), an unrelated change from a different initiative
  (the attention-override / self-model rebuild arc), merged **2026-09-05
  05:52:56Z**. Its own PR body measured a ~45x increase in how often multiple
  things compete for attention (0.8% of ticks at 0.2 -> 36.1% at 0.05, matched
  120-tick windows). `orion-topic-foundry`'s background LLM RPC-timeout rate
  measurably doubled starting **05:56Z** -- three minutes later -- and has
  stayed doubled since, confirmed against `docker logs -t` timestamps covering
  the full 10 hours available in the container's log buffer.

## Outcome moved

Correctly separated "this session's model-testing footprint" (confirmed fully
reverted) from "a live, ongoing, unrelated production issue" (still active,
not owned by this session). Documented the causal chain with timestamps and
code references so the next person does not have to re-derive it, and so
PR #2110's own author/owner has what they need to decide next steps.

## Current architecture

- `orion-substrate-runtime`'s `ORION_ATTENTION_BROADCAST_MIN_SALIENCE` gates
  which nodes on the live attention/coalition graph are salient enough to
  publish in the `AttentionBroadcastProjectionV1` broadcast
  (`services/orion-substrate-runtime/app/worker.py:2710`).
- `orion-thought`'s `reverie` loop consumes that exact broadcast as the input
  to `build_reverie_plan_request()` (`services/orion-thought/app/reverie.py:276`).
  When it finds "a current coalition" (more likely now that more nodes clear
  the lower gate), it dispatches a real, LLM-calling `reverie_narrate` plan
  instead of skipping the tick (`"reverie tick skipped: no current coalition"`
  is the skip-path log line, confirmed present in this session's own traces).
- Non-lift `reverie_narrate` ticks carry no explicit `llm_route` override
  (`services/orion-thought/app/reverie.py:287-293`), so they fall through
  `orion-cortex-exec`'s `_default_llm_route_for_step()` to the gateway's
  default route, `quick` -- the same lane `orion-topic-foundry` already uses
  constantly for its own background work.
- The gateway's background-priority admission path
  (`services/orion-llm-gateway/app/priority_admission.py`) polls each
  candidate upstream's `/slots` endpoint before dispatch, gated by
  `LLM_GATEWAY_BACKGROUND_MAX_WAIT_SEC` (30s) /
  `LLM_GATEWAY_BACKGROUND_POLL_INTERVAL_SEC` (0.5s). The synchronous variant
  (`wait_for_slack_sync`, used by `run_llm_chat`'s sync call path) is correctly
  dispatched off the event loop via `asyncio.to_thread`
  (`services/orion-llm-gateway/app/main.py:284`) -- not a blocking-event-loop
  bug, but it does consume a worker thread from the shared thread pool for the
  full wait.
- Live-confirmed on one specific failing call (corr=`c1a8f100...`): the
  interactive stance_react request reached the gateway and cortex-exec
  instantly, but the actual llama.cpp task for it did not launch on the
  (otherwise near-idle) chat/harness 35B worker until **18 minutes later** --
  the request was not slow because the model was busy, it was stuck inside
  the gateway process itself, most plausibly queued behind a saturated pool
  of worker threads all blocked in background-admission waits triggered by
  the reverie/topic-foundry volume increase above.

## Architecture NOT touched by this incident report

No code or config changed as part of this document. This is a findings-only
writeup. Specifically NOT reverted or modified here (out of scope, not this
session's to decide):

- `services/orion-substrate-runtime/.env` /
  `ORION_ATTENTION_BROADCAST_MIN_SALIENCE=0.05` (PR #2110's own live change).
- Any code in `orion-llm-gateway`, `orion-thought`, or `orion-cortex-exec`
  touching admission/routing/concurrency.

## Files changed

- `docs/superpowers/pr-reports/2026-09-05-stance-react-attention-salience-gateway-starvation-incident.md`:
  this report.

## Evidence log (exact commands and outputs this conclusion is built on)

1. **Confirmed the Flash-Next/BF16 test session's own revert was clean**,
   independent of this incident:
   - `GET /routes` (gateway, athena): `chat`/`harness` -> `Qwen3.6-35B-A3B-UD-Q5_K_M.gguf`,
     `agent` -> `Qwen3.8-27B-UD-Q4_K_XL.gguf`.
   - `docker exec orion-athena-thought python3 -c "from orion.cognition.plan_loader
     import build_plan_for_verb; print(build_plan_for_verb('stance_react',
     mode='brain').timeout_ms)"` -> `120000` (the pre-test value).
   - `git log --oneline --since="18 hours ago" -- services/orion-thought/app/reverie.py
     services/orion-thought/app/chain.py services/orion-thought/app/main.py
     orion/harness/ services/orion-cortex-exec/app/executor.py` -> only this
     session's own docs-only commit touches any of these files.

2. **Traced one specific failing interactive stance_react call end to end**
   (corr=`c1a8f100-1285-449e-a171-5e757f6ee7f2`):
   - `cortex-exec` -> gateway request received 07:01:28.407Z, route=chat,
     `payload_max_tokens=8000`, `timeouts=connect:10.0 read:115.0`.
   - Real completion (2127 tokens, `finish_reason=stop`, genuine 6385-char
     reasoning trace, not truncated) only published back at 07:20:25.439Z --
     ~19 minutes after the request, ~18 minutes after the timeout budget
     should have expired.
   - The chat/harness 35B worker's own `docker logs` (elapsed-time converted
     to wall clock via `docker inspect --format .State.StartedAt`) show only
     ~117 seconds of actual GPU work between 07:01:28 and 07:19:41 -- the
     worker was not the bottleneck.
   - `grep -c "GET.*\/slots" ` over the same window: 547 polls in ~19 minutes,
     traced to `services/orion-llm-gateway/app/priority_admission.py`'s
     background-admission slot-check, driven heavily by repeated
     `orion-topic-foundry` background requests observed in the same window.

3. **Established the baseline-vs-spike timing** via
   `docker logs -t orion-athena-topic-foundry --since 10h 2>&1 | grep -iE
   "\[rpc\] timeout waiting" | grep -oE "^[0-9-]+T[0-9:.]+" | cut -c1-16 |
   sort | uniq -c`: steady ~3/minute from 2026-09-04 23:43Z through
   2026-09-05 05:55Z, then 5-8/minute from 05:56Z onward, sustained through
   the end of the available log window (07:40Z).

4. **Correlated against the full merged-PR timeline** via
   `gh pr list --state merged --search "merged:>=2026-09-04T18:00:00" --json
   number,title,mergedAt`: PR #2110 merged 2026-09-05T05:52:56Z, three
   minutes before the observed rate change, and is the only merge in the
   surrounding window whose own stated purpose is to increase how often
   competing background activity fires.

5. **Confirmed the mechanism connecting PR #2110 to reverie's LLM-call
   volume** via direct code read: `services/orion-substrate-runtime/app/worker.py:2710`
   passes `min_salience=float(s.attention_broadcast_min_salience)` into the
   attention-broadcast publish path; `services/orion-thought/app/reverie.py`'s
   `build_reverie_plan_request()` takes that broadcast as its direct input and
   decides whether to skip the tick or fire a real LLM call based on whether
   it finds a coalition inside it.

## Risks / concerns

- Severity: should
- Concern: `orion-topic-foundry`'s background RPC-timeout rate was already
  non-zero (~3/minute) for the full 10 hours of log history checked, *before*
  PR #2110 merged -- meaning there is a real, separate, pre-existing baseline
  fragility in the shared background-admission path independent of this
  incident. PR #2110 made a pre-existing weakness bite harder; it did not
  create the weakness from nothing.
- Severity: should
- Concern: the gateway's background-admission thread-pool sizing is not
  currently instrumented -- this report infers thread-pool saturation as the
  most plausible mechanism for the 18-minute stall (consistent with every
  observed symptom: idle target worker, saturated `/slots`-polling volume,
  correct-but-thread-consuming async dispatch) but has not directly measured
  live thread-pool occupancy at the moment of a stall to prove it
  conclusively.
- Severity: note
- Concern: PR #2110 explicitly frames itself as a **deliberate, time-boxed
  experiment** ("Deliberately an experiment... for an hour of work instead of
  a week") to test a premise for a larger design (#2109). Its own body commits
  to adding a post-change measurement before merge; whether that measurement
  accounted for the gateway-contention side effect documented here is
  unknown to this report's author.

## Recommended next steps (for whoever owns PR #2110 / the attention-override arc, not this session)

- Decide whether `ORION_ATTENTION_BROADCAST_MIN_SALIENCE=0.05` stays, given
  this newly-documented side effect on shared background-lane contention.
- If it stays, the gateway's background-admission path likely needs either a
  larger thread-pool allocation, a hard cap on concurrent admission-checks,
  or route-level isolation so a flood of background traffic cannot delay an
  unrelated foreground request's ability to even begin dispatching.
- Independently worth a look regardless of PR #2110's fate: the pre-existing
  ~3/minute baseline topic-foundry RPC-timeout rate.

## PR link

<to be filled in after push>
