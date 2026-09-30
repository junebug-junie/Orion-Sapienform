# feat(llm-inference): per-role wait/model clocks + decode speed in the gateway inference report (GPU pool stage 6.2)

## Summary

- Every LLM call the gateway makes now has two separate clocks: how long it **waited for a GPU**
  (pool acquire -> grant) and how long the **model took** once it had one (grant -> reply, or the
  end of a stream). They are grouped by the pool role that granted the call (`chat`, `agent`,
  `agent-gpu2`, `metacog`, `fast`, or `ungranted`) instead of by machine.
- Each call also records llama.cpp's own **decode speed** (`timings.predicted_per_second`), which
  does not depend on how long the answer was.
- Stage 7 input (PR #2442) folded in so it is not built twice: **occupancy at grant** (how many of
  the gateway's calls were in flight on that role, this one included) with decode speed split into
  `solo` and `shared`; the pool's **slot count** per role (read once a minute); and llama.cpp's
  **prompt-cache counts** (`timings.prompt_n` processed vs `timings.cache_n` reused).
- The HTTP passthroughs (`/v1/messages`, `/v1/chat/completions`, what FCC/Claude Code uses) are now
  counted, **in the per-role clocks only**. They are kept out of the node counts behind the live
  field channel `inference_failure_pressure` on purpose (see Risks).
- Retires the old mixed `latency_p50_ms`/`latency_p95_ms` (one clock from before the lease). The
  projection model drops them on load, so the row the current reducer has already written still
  validates.

## Outcome moved

The question "is the chat worker slow, or is the line long?" (172 of 1,172 interactive chat grants
waited >60 s in the 24 h before the spec) becomes answerable from recorded data: `wait_p50/p95_ms`
vs `model_p50/p95_ms` per role, per minute. Nothing is wired into cognition yet; the numbers land
on the projection (debug surface) first, per spec gate G2.

## Current architecture

- `services/orion-llm-gateway/app/grammar_emit.py` counted bus-RPC calls per serving machine,
  one clock `dispatch_started` taken in `handle_chat` **before** `_dispatch_chat` acquired the
  pool lease (`main.py:~507`), emitted as `p50_ms`/`p95_ms` in the node atom.
- `passthrough_proxy.proxy_on_pool` recorded nothing.
- `orion/substrate/llm_inference_loop/extract.py` parsed `p50_ms`/`p95_ms` into
  `LlmInferenceNodeStateV1.latency_p50_ms/latency_p95_ms`. Nothing else read them
  (`rg latency_p50_ms`: only the schema and extract; not Hub, not scripts/analysis, not evals).

## Architecture touched

- orion-llm-gateway: `grammar_emit.py` (CallClock, ReplyTimings, per-role buckets, HTTP outcome
  classification), `main.py` (clock threaded through `_dispatch_chat`/`_dispatch_on_pool`,
  slots provider for the publisher), `passthrough_proxy.py` (one `_CallReport` per call, recorded
  exactly once on every exit path), `pool_placement.py` (occupancy counted at
  `PoolLease.acquire`/`release`; `role_slots()` from cached pool state).
- Contract: `orion/schemas/llm_inference_projection.py` (new `LlmInferenceRoleStateV1`,
  `by_role` on the node state, retired fields dropped on load), `orion/schemas/registry.py`.
- Consumer: `orion/substrate/llm_inference_loop/extract.py` (parses `roles=`).
- Wire: the node atom summary gains one kv token,
  `roles=<role>[calls:n|http_calls:n|served:n|...|decode_tps_p50:x|busy_max:n|slots:n|prompt_n:n|cache_n:n|...]...`,
  and loses `p50_ms`/`p95_ms`. Trace prefix, channel, atom roles unchanged. The per-role data rides
  inside the node atom (not separate atoms) so a reducer batch boundary can never split a role from
  its node.

## Files changed

- `services/orion-llm-gateway/app/grammar_emit.py`: two clocks, per-role buckets, timings, covariates.
- `services/orion-llm-gateway/app/main.py`: bus path clock; publisher gets `slots_provider`.
- `services/orion-llm-gateway/app/passthrough_proxy.py`: passthrough calls recorded (per-role only).
- `services/orion-llm-gateway/app/pool_placement.py`: role occupancy at the lease; `role_slots()`.
- `services/orion-llm-gateway/README.md`: inference lane section updated.
- `services/orion-llm-gateway/tests/test_inference_clocks.py` (new), `tests/test_grammar_emit.py`.
- `orion/schemas/llm_inference_projection.py`, `orion/schemas/registry.py`: contract.
- `orion/substrate/llm_inference_loop/extract.py`: consumer.
- `tests/test_llm_inference_substrate_reducer.py`: producer->consumer round trips, legacy row load.
- this report.

## Schema / bus / API changes

- Added: `LlmInferenceRoleStateV1` (`extra="forbid"`): `calls, http_calls, served, upstream_failed,
  refused, request_invalid, wait_p50_ms, wait_p95_ms, model_p50_ms, model_p95_ms, decode_tps_p50,
  decode_tps_samples, decode_tps_solo_p50, decode_tps_solo_samples, decode_tps_shared_p50,
  decode_tps_shared_samples, busy_p50, busy_max, slots, prompt_n, cache_n, cache_reports`.
  `LlmInferenceNodeStateV1.by_role: dict[str, LlmInferenceRoleStateV1]`. Registered.
- Removed: `LlmInferenceNodeStateV1.latency_p50_ms/latency_p95_ms`; node atom `p50_ms`/`p95_ms`.
- Behavior changed: HTTP passthrough calls are now reported (per-role only).
- Compatibility: `RETIRED_NODE_STATE_FIELDS` are dropped by a before-validator, so the live row
  (which still has `latency_p50_ms`) loads under `forbid` -- verified live:
  `check_substrate_projection_schema_drift.py` against the production DB: all 8 rows OK.
  The wire is tolerant in both orders: the old reducer ignores `roles=`; the new reducer reads a
  pre-6.2 atom as `by_role={}`. The consumer-first order below is so no minute of role data is lost,
  not to avoid a crash.
- No new bus channels, no registry channel entries, no tables, no migrations.

## Env/config changes

- Added keys: none. Removed: none. Renamed: none.
- `.env_example` updated: no. Local `.env` sync: not needed (no template changed).
- Skipped keys requiring operator action: none.

## Metric quality gate (CLAUDE.md 0A), items 1-3 now; item 4 is the 48 h checkpoint

| metric | 1. provenance (real code) | 2. independence | 3. theory anchor |
|---|---|---|---|
| `wait_ms` | `CallClock.waiting()` just before `pool_placement.lease_for_route` / `_acquire` (`main.py` `_dispatch_on_pool` loop; `passthrough_proxy.proxy_on_pool` loop) -> `granted()` when the lease yields. Summed across re-leases. | **Same fact** as the pool's `gpu_pool_wait` hop (`services/orion-gpu-pool/app/runtime.py`, `gpu_pool_events.waited_ms`) plus the bus round trip. Not new signal; kept only as the denominator for "line vs worker". Must never be wired alongside `gpu_pool_wait`. Overlaps the caller's `rpc_timeout_pressure` (a long wait is one cause of an RPC deadline) -- same causal chain, not independent. Also feeds `queue_contention_score`'s world via `gpu_pool_waiting` (queue depth) -- related, not identical (depth vs duration). | Queueing: time in queue is the pool's scheduling outcome, distinct from service time. |
| `model_ms` | `CallClock.granted()` -> `replied()` right after `_run_on_grant` returns (bus) or the upstream response / stream end (passthrough). Served calls only. | Disjoint from `wait_ms` by construction (tested). Not derived from `reasoning_load` (a run weight) or `inference_failure_pressure` (outcomes). Confounded by output length -- a 0.3 s classifier and a 120 s agent turn share a role -- so it is **not** a speed. | Service time of one call. Weak as a baseline because of length; see `decode_tps`. |
| `decode_tps` | llama.cpp reply `timings.predicted_per_second`: bus `result["raw"]["timings"]`, passthrough JSON body, or the last `predicted_per_second` in a stream's final 4 KB. Never derived from wall time. Shape verified on a live leased call through this gateway (b10398, fast role). | Not derived from `gpu_pressure` (GPU util), `reasoning_load`, or `inference_failure_pressure`. Depends on occupancy (next row), which is why it is banded. | llama.cpp's own per-token generation speed. At fixed model and fixed occupancy, a drop means the worker degraded (thermal, ctx growth, a non-gateway co-tenant). With N>1 slots llama.cpp decodes every busy slot in one batch, so sharing lowers per-call speed normally -- hence `solo` vs `shared`. |
| `busy_at_grant` | `pool_placement._occupancy_enter/_exit` at `PoolLease.acquire` success / `release` (idempotent). Covariate, not a signal (stage 7 C1). | Counts **this gateway's calls in flight on the role** -- deliberately not the pool's lease count, which includes idle durable-run holds that occupy a slot but decode nothing. The port gate (spec 6.6) makes the gateway the only LLM path to workers, so this is the concurrent-decode count. Not wired alone. | Batched decode splits compute across active slots. |
| `slots` | Pool state `roles[].slots` (`DiscoveredRoleV1`, from llama.cpp `/props total_slots`), fetched once per window by `pool_placement.role_slots()` via the existing cached state RPC; off the serving path. | Configuration fact, not a signal. | Denominator for occupancy. |
| `prompt_n` / `cache_n` | llama.cpp `timings.prompt_n` / `timings.cache_n`, summed per role over served calls reporting both. Live call: `cache_n 3 + prompt_n 11 = usage.prompt_tokens 14`. | Not in any existing metric. `prompt_tokens` (node-level) is their sum; the split is new. | KV prefix reuse: `cache_n/(cache_n+prompt_n)` is the reuse share (stage 7 D5 evidence). |

Items 5-6: existing mechanism is #2327's lane, extended not duplicated; the `llm:<role>#call`
rpc_health hops were dropped (spec G1). Reversible: projection-only fields, no field channel, no
training default; removal is a schema field drop with the same retired-field pattern.

## 48 h checkpoint (gate item 4) -- exact queries

Start the clock at the **gateway** deploy. The window must **not overlap a role's model or slot
change** (stage 7: Bonsai / 2-slot agent lanes). Run Q0 first; if it shows a change on a role,
restart that role's 48 h after the change, or split the window at it.

Run in `docker exec -i orion-athena-sql-db psql -U postgres -d conjourney` after
`\set deploy '<gateway deploy UTC timestamp>'`. Shared parsing CTE (paste as the head of Q1-Q5
in place of `<CTE>`):

```sql
with src as (
  select created_at, event_json->'atom'->>'summary' as s
  from grammar_events
  where trace_id like 'llm_gateway.inference:%'
    and event_json->'atom'->>'semantic_role' = 'llm_inference_window_observed'
    and created_at > :'deploy'::timestamptz
),
roles as (
  select src.created_at, m[1] as role, m[2] as body
  from src, regexp_matches(substring(src.s from 'roles=(\S+)'), '([a-z0-9_.-]+)\[([^\]]*)\]', 'g') as m
),
kv as (
  select created_at, role,
    (regexp_match(body, '(?:^|\|)calls:(\d+)'))[1]::int                        as calls,
    (regexp_match(body, '(?:^|\|)http_calls:(\d+)'))[1]::int                   as http_calls,
    (regexp_match(body, '(?:^|\|)served:(\d+)'))[1]::int                       as served,
    (regexp_match(body, '(?:^|\|)wait_p50_ms:(\d+)'))[1]::int                  as wait_p50_ms,
    (regexp_match(body, '(?:^|\|)wait_p95_ms:(\d+)'))[1]::int                  as wait_p95_ms,
    (regexp_match(body, '(?:^|\|)model_p50_ms:(\d+)'))[1]::int                 as model_p50_ms,
    (regexp_match(body, '(?:^|\|)model_p95_ms:(\d+)'))[1]::int                 as model_p95_ms,
    (regexp_match(body, '(?:^|\|)decode_tps_p50:([0-9.]+)'))[1]::float         as tps,
    (regexp_match(body, '(?:^|\|)decode_tps_solo_p50:([0-9.]+)'))[1]::float    as tps_solo,
    (regexp_match(body, '(?:^|\|)decode_tps_solo_n:(\d+)'))[1]::int            as tps_solo_n,
    (regexp_match(body, '(?:^|\|)decode_tps_shared_p50:([0-9.]+)'))[1]::float  as tps_shared,
    (regexp_match(body, '(?:^|\|)decode_tps_shared_n:(\d+)'))[1]::int          as tps_shared_n,
    (regexp_match(body, '(?:^|\|)busy_max:(\d+)'))[1]::int                     as busy_max,
    (regexp_match(body, '(?:^|\|)slots:(\d+)'))[1]::int                        as slots,
    (regexp_match(body, '(?:^|\|)prompt_n:(\d+)'))[1]::int                     as prompt_n,
    (regexp_match(body, '(?:^|\|)cache_n:(\d+)'))[1]::int                      as cache_n
  from roles
)
```

(The CTE was run against a synthetic new-format summary on the live Postgres; it parses every
field, and `calls` does not match `http_calls`.)

**Q0 -- no model or slot change inside the window (run first).**

```sql
-- model/profile changes seen by pool discovery, per role
select role, count(distinct detail->>'model_file') as models, count(distinct detail->>'profile_name') as profiles,
       min(created_at), max(created_at)
from gpu_pool_events
where event in ('discovery_confirmed', 'discovery_mismatch') and created_at > :'deploy'::timestamptz
group by role;
-- slot changes are silent in discovery events (stage 7 finding), so read the report's own stamp:
<CTE> select role, array_agg(distinct slots) as slot_values, min(created_at), max(created_at)
from kv where slots is not null group by role;
```

Pass: one model and one profile per role; one slot value per role. `agent-gpu2` load/unload swaps
of the same profile are not a model change.

**Q1 -- the metric exists and is not degenerate.**

```sql
<CTE> select role, count(*) as windows, sum(calls) as calls, sum(http_calls) as http_calls,
  count(*) filter (where wait_p50_ms is null and calls > 0)  as windows_no_wait,
  count(*) filter (where model_p50_ms is null and served > 0) as windows_served_no_model,
  count(*) filter (where tps is null and served > 0)          as windows_served_no_tps,
  count(distinct tps) as distinct_tps, min(tps), max(tps)
from kv group by role order by role;
```

Pass: every granted role with traffic has non-null `wait_p50_ms`, `model_p50_ms`, and (llama.cpp
roles) `tps`; `distinct_tps` > 1 (not flat); `http_calls` > 0 on the FCC role after an FCC turn
(acceptance check 3).

**Q2 -- the rest state: decode speed returns to a per-role, per-occupancy band at quiet times.**

```sql
<CTE> select role,
  percentile_cont(array[0.1, 0.5, 0.9]) within group (order by tps_solo)   as solo_p10_p50_p90,
  sum(tps_solo_n) as solo_samples,
  percentile_cont(array[0.1, 0.5, 0.9]) within group (order by tps_shared) as shared_p10_p50_p90,
  sum(tps_shared_n) as shared_samples,
  -- quiet hours (Juniper is MDT; 08:00-13:00 UTC = 02:00-07:00 MDT)
  percentile_cont(0.5) within group (order by tps_solo) filter (where extract(hour from created_at) between 8 and 12) as solo_quiet_p50
from kv group by role order by role;
```

Pass: `solo` is tighter than `shared` and higher; `solo_quiet_p50` is within the `solo` p10-p90
band (it can return to rest -- not a permanent floor). Sparse roles (`solo_samples` < ~30 over
48 h) are recorded as "not enough data", not as a band. Tiny generations are noisy
(`predicted_n` 2 gave 177 tok/s on the live probe); if the band is wide, that is the first
suspect -- do not add a length filter without a separate gate.

**Q3 -- line vs worker (the question 6.2 exists to answer).**

```sql
<CTE> select role, sum(calls) as calls,
  percentile_cont(array[0.5, 0.95]) within group (order by wait_p95_ms)  as wait_p95_dist,
  percentile_cont(array[0.5, 0.95]) within group (order by model_p95_ms) as model_p95_dist,
  count(*) filter (where wait_p95_ms > 60000) as windows_wait_over_60s,
  max(busy_max) as max_busy, max(slots) as slots
from kv group by role order by role;
```

**Q4 -- cross-check: gateway wait vs the pool's own `waited_ms`, same role and minute.**

```sql
<CTE> , pool as (
  select date_trunc('minute', created_at) as m, role,
         percentile_cont(0.5) within group (order by waited_ms) as pool_wait_p50
  from gpu_pool_events where event = 'granted' and created_at > :'deploy'::timestamptz
  group by 1, 2)
select kv.role, count(*) as minutes,
  percentile_cont(0.5) within group (order by abs(kv.wait_p50_ms - pool.pool_wait_p50)) as median_abs_diff_ms
from kv join pool on pool.role = kv.role and pool.m = date_trunc('minute', kv.created_at)
group by kv.role;
```

Pass: `median_abs_diff_ms` < 1000 (spec acceptance 2). Windows are the gateway's 60 s windows,
not wall-clock minutes, so this is approximate; a large gap is a bug to chase, a small one is noise.

**Q5 -- prompt-cache reuse per role (stage 7 D5 evidence).**

```sql
<CTE> select role, sum(prompt_n) as processed, sum(cache_n) as reused,
  round(sum(cache_n)::numeric / nullif(sum(cache_n) + sum(prompt_n), 0), 3) as reuse_share
from kv where prompt_n is not null group by role order by role;
```

Wiring any of these into `capability:llm_inference` is proposed only if Q0-Q2 pass, in a separate
PR (spec: "No field wiring in this stage"). FCC transport-baseline decision (spec Q2 / PR 6.7) uses
Q3's passthrough (`http_calls`) data.

## Tests run

```text
services/orion-llm-gateway: python -m pytest tests -q          -> 350 passed
root: python -m pytest tests/test_llm_inference_substrate_reducer.py -q -> 35 passed
root: tests/test_*registry*.py tests/test_llm_inference*.py tests/test_field_state_schemas.py
      tests/test_field_channel_glossary.py                     -> 92 passed
services/orion-field-digester tests (from root)                -> 257 passed, 6 skipped
services/orion-substrate-runtime tests (from root)             -> 16 failed, 370 passed;
      the identical 16 fail on origin/main in the same environment (diffed), pre-existing
      (e.g. test_worker_reducer.py::test_advance_cursor_records_commit_failure_when_created_at_missing).
scripts/check_substrate_projection_schema_drift.py --postgres-uri <live> -> all 8 rows OK
      (incl. substrate_llm_inference_projection, which still carries latency_p50_ms)
scripts/check_definition_drift.py --gate                      -> PASS, 0 changed definitions
      (no field channel or glossary meaning changed; no re-lock needed)
```

New tests cover: disjoint clocks, re-lease summing, never-granted calls (wait, no model), per-role
grouping and percentiles, decode_tps parsing (dict, stream tail, invalid -> None, never derived),
usage parsing (OpenAI + Anthropic), HTTP outcome classes, the bus path's wait vs model split under a
real queued grant, OpenAI + Anthropic + streamed passthroughs counted once, passthroughs kept out of
the failure population, occupancy seen by concurrent calls and never leaking (incl. a crashed
upstream), slots from pool state, publisher surviving a slots failure, producer->consumer round
trips, the live legacy row loading, and `forbid` still rejecting unknown fields.

## Evals run

```text
No new eval harness. services/orion-llm-gateway/evals/run_inference_outcome_eval.py replays
outcome classes from logs and does not read latency; unaffected. The live-data evaluation for
this metric is the 48 h checkpoint above (gate item 4), which cannot run before deploy.
```

## Docker/build/smoke checks

```text
Not deployed (per task). One read-only live probe: a 2-token /v1/chat/completions call through the
production gateway (pool-leased, fast role) to confirm llama.cpp's timings shape (cache_n/prompt_n).
Live schema drift check against production Postgres: OK.
```

## Review findings fixed

Code review ran in a subagent over `origin/main...HEAD`. No must-fix defects in the serving path.

- Finding: a refused call's wait (the "line is long" signal) never reached the projection -- no
  worker, so its node was `unrouted` and the reducer dropped it.
  - Fix: an ungranted call's role clock is filed under the pool's host node (`pool_placement.pool_node()`,
    config `host.name`); its node-level counts stay unattributed as before.
  - Evidence: `test_an_ungranted_wait_reaches_the_projection_under_the_pool_node` (reducer),
    `test_pool_refusal_records_the_wait_under_ungranted`, `test_passthrough_pool_refusal_is_counted_as_ungranted`.
- Finding: rolling the reducer back to main after this deploy crash-loops it (`by_role` under `forbid`).
  - Fix: rollback procedure added below (delete the singleton row; it self-heals).
  - Evidence: "Rollback" section.
- Finding: a grant landing in the same instant the HTTP client disconnects leaked the lease and the
  role occupancy count forever (every later call on that role read as "shared").
  - Fix: `passthrough_proxy._withdraw` releases a grant that completed while being withdrawn, on both
    the disconnect and the cancellation path.
  - Evidence: `test_a_grant_landing_as_the_client_leaves_is_released`, `test_client_gone_while_queued_is_recorded_once`.
- Finding: an HTTP-only minute emitted a `calls=0` node atom that folded a zero window and re-sent
  `inference_failure_pressure` -- a cadence change on the live channel.
  - Fix: the reducer skips the fold and the hint when `calls == 0`, keeping the last reading and span.
  - Evidence: `test_a_role_only_window_does_not_slide_or_resend_the_live_failure_reading`.
- Finding: passthrough exit paths untested.
  - Fix: tests for overflow -> re-lease (one `served`), overflow with nothing bigger (one
    `context_overflow`), lease revoked mid-call (one `gpu_pool_recalled`), client gone while queued
    (one `client_gone`, occupancy 0), stream breaking (one `upstream_error`).
- Finding (nit): HTTP token counts parsed and discarded. Fix: parse removed (`usage_tokens_from` deleted).
- Finding (nit): `slots` looked up by sanitized role key vs raw pool name. Fix: `role_slots` keys normalized with `role_key`.
- Finding (nit): `stream_owns_lease` set before the streaming response was built. Fix: set after construction.
- Not changed, documented: a re-leased call's model time sums under the final role; a streamed
  call's model time includes a slow client's read time (README "Known limits").
- Not changed (pre-existing, cosmetic): a `release()` that raises after a served reply is recorded
  as `gateway_exception`.

## Deploy order (consumer first)

1. **orion-substrate-runtime** (reducer: accepts `roles=`, drops the retired fields on load).
2. **orion-llm-gateway** (producer).

Either order is crash-safe (verified above), but gateway-first loses the role data until the
reducer lands. No migration. Field digester, Hub: no redeploy needed.

### Rollback

Gateway: redeploy the previous image; the new reducer reads a pre-6.2 atom as `by_role={}`.
substrate-runtime: the previous reducer's model forbids `by_role`, so once this reducer has written
the row, the old one fails every load. Roll back the gateway first, then run (operator action; the
row is a materialized cache rebuilt on the next tick, only the 10-minute failure span is lost):

```bash
docker exec orion-athena-sql-db psql -U postgres -d conjourney -c \
  "delete from substrate_llm_inference_projection where projection_id = 'active_llm_inference_projection'"
```

then redeploy the previous substrate-runtime image.

## Restart required

```bash
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
# confirm one tick loaded the old row: docker logs --tail=50 orion-athena-substrate-runtime | grep -i llm_inference
scripts/safe_docker_build.sh orion-llm-gateway up -d --build
# then, after ~2 minutes:
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc \
  "select projection_json->'nodes'->'llm_node:circe'->'by_role' from substrate_llm_inference_projection"
```

## Risks / concerns

- Severity: medium. Concern: passthrough calls are **not** in `inference_failure_pressure`'s
  population. The glossary says that channel is "calls the LLM gateway sent to this node's
  backends", which the passthroughs are, so today it undercounts. Widening it changes a live field
  channel's population (a metric-definition change) and was not in the approved scope.
  Mitigation: `http_calls` and per-role failure classes make the size of the gap visible at the
  checkpoint; a separate decision for Juniper.
- Severity: low. Concern: `busy` counts only this gateway's calls. Any non-gateway client on a
  worker port would be invisible. Mitigation: the port gate (spec 6.6) makes that a CI failure.
- Severity: low. Concern: `slots` is the value at window end, not at each grant. A slot change
  mid-window stamps the new value on the whole window. Mitigation: Q0 excludes change windows.
- Severity: low. Concern: when a call overflows its slot and is re-leased onto a bigger one, its
  `model_ms` includes the short overflowed attempt, and its `wait_ms` both waits. Mitigation:
  overflows are rare and rejected during prompt processing (fast); the role recorded is the final one.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2443

🤖 Generated with [Claude Code](https://claude.com/claude-code)
