# Pre-turn check queueing: why one of Juniper's turns waited 3 minutes

Status: design. Patch 1 ships separately (branch `fix/pre-turn-check-parallel`). Patch 2 is gated.
Evidence date: 2026-11-12 (live).

## Arsonist summary

Before every turn, Orion runs a short check. The check asks a small model one question: does this
conversation need repair? On 2026-11-12, one of Juniper's turns waited 3 minutes for that check.
The check itself took 2.4 seconds. The rest was a queue.

The queue had two causes. First, the check worker handled one request at a time. Nobody chose
that. It was a library default. Second, the requests ahead of Juniper's were world-pulse reads.
Their text was too long for the small model's 4k window. So they waited in the GPU pool until
their deadline ran out, and Juniper's check waited behind them.

Patch 1 makes the worker handle requests in parallel. That removes the line. It does not fix the
background checks that cannot fit. Patch 2 proposes that background turns run their check on the
GPU seat they already hold. That seat has a big window. But Patch 2 has open holes, so it waits
behind a measurement step and one live probe.

## Current architecture

**The incident.** Turn `aaee51e3-9f6b-484f-9ac7-6136b5d827a7` took about 4.5 min. 3:00 of that was
the check timing out at the Hub's limit. When the check finally ran, it took 2.4 s on the fast lane
(Qwen3-8B, circe :8013, `ctx_per_slot` 4096).

- Hub timeout: `PRE_TURN_APPRAISAL_TIMEOUT_MS=180000` (`services/orion-hub/.env_example:1386`).

**The serial worker.** The check runs in cortex-exec. Its bus worker is `pre_turn_appraisal_svc`
(`services/orion-cortex-exec/app/main.py:976`). It does not set `concurrent_handlers`, so the Rabbit
default (`False`) applies. Commit `8657c7879` added it with the default. It was not a design choice.
The neighbour `stance_prepare_svc` turns concurrency on for the same reason
(`concurrent_handlers=True`, `main.py:994`).

**The requests ahead.** A Hub restart let three world-pulse reads start at 02:59:52
(`HUB_WORLD_PULSE_READ_MAX_IN_FLIGHT=3`, `services/orion-hub/.env_example:535`). Their threads
are 21k-39k characters. The Hub caps the window at the last 8 messages
(`build_turn_window`, `orion/substrate/appraisal/turn_window.py:14`). It never shortens a message.
`build_repair_probe_prompt` (`orion/substrate/appraisal/paradigms/repair_pressure_v2.py:98`) puts
that window in the prompt with no character or token limit. The prompt cannot fit a 4k slot. The request waits in the GPU pool until
`reason=deadline`.

**Route.** The check always uses route `quick_turn` (`REPAIR_PRESSURE_PROBE_ROUTE`,
`services/orion-cortex-exec/app/settings.py:483`). That route is class `fast`, priority
`interactive` (`config/gpu_pool.yaml:191`). The `fast` class falls back through
`fast, metacog, agent, agent-gpu2, agent-deep, chat` (`config/gpu_pool.yaml:157`). So a long
prompt can, in principle, fall all the way to Juniper's chat card.

**Background turns hold a lease.** Durable and background turns carry `payload["gpu_lease"]`
(`orion/hub/turn_orchestrator.py:1066`, `:1194`). Their motor uses `GPU_LEASE_ROUTE="agent"`
(`orion/llm/resource_lease.py:22`). The check is sent at `turn_orchestrator.py:1111`. That is after
the Hub checks that a lease exists (`:1066`) and before it parses the lease (`:1194`). The check
ignores the lease: `_run_pre_turn_appraisal` (`:597`) takes no lease argument.

**Numbers (7 days, check failures by turn kind).**

| Turn kind | Failed / total | Note |
|---|---|---|
| `world_pulse_read_stage2` | 194 / 373 | avg ~100 s to fail |
| other long turns | 50 / 143 | |
| `world_pulse_read` | 8 / 418 | |
| short turns | 6 / 554 | |

**Where `quick_turn` checks landed (24 h):** fast 164, agent 18, agent-gpu2 7, none/deadline 33,
chat (gpu0) 0.

**Turn sources (24 h):** world_pulse_read 76, stage2 55, pr-field-notes 33, self-sense-eval 28,
curiosity 18, Juniper 13, consult 7, reminiscence 6, outreach 1. At most 4 turns in flight;
average 1.5.

**Chat lane.** circe :8011 runs Strata with `qwen3.8-flash-next`, `total_slots` 1, `n_ctx` 262144.
It has no `/apply-template` or `/completion`. `/v1/chat/completions` returns `logprobs` and
`top_logprobs` (tested: token `YES` at -0.018, 0.67 s). The gateway's native-completion logprob
path runs only for `backend == "llamacpp"` (`services/orion-llm-gateway/app/llm_backend.py:1003`).
Strata is in `OPENAI_COMPAT_BACKENDS` (`llm_backend.py:605`), so the chat path asks for logprobs and
reads them with `extract_llm_uncertainty_from_openai_response`. **UNVERIFIED:** whether the probe
gets real, varying scores from Strata. A hollow fallback (`_TEXT_FALLBACK_*`) happened before, on
2026-07-30.

**Schema.** `PreTurnAppraisalRequestV1` (`orion/schemas/pre_turn_appraisal.py`) is
`extra="forbid"`. A new field needs cortex-exec deployed before the Hub.

## Patch 1 (shipping now): parallel checks

Set `concurrent_handlers=True` on `pre_turn_appraisal_svc`. One stuck check then cannot block the
next one.

Impact review (done):

- The Hub is the only sender.
- The handler has no shared mutable state. Each request builds its own paradigm. `_BUS` is set once.
- Results go to per-turn reply channels.
- The equilibrium consumer (`services/orion-equilibrium-service/app/repair_pressure_metacog_gate.py`)
  is stateless per appraisal.
- cortex-exec's main exec worker already runs concurrently on the same bus
  (`EXEC_CONCURRENT_HANDLERS=true`).

What Patch 1 does not fix: long background checks still cannot fit 4k. They still fail at
deadline. Now they may also spill further down the fallback list, because more of them are in the
pool at once. That is the reason for the measurement step.

## Measurement step (between patches, 2 days)

Goal: count how many checks land on gpu0 (the `chat` role, Juniper's card) after Patch 1. Today
the count is 0. If it stays near 0, the flood that Patch 2 guards against is not real.

GPU pool leases for the check (from `gpu_pool_leases`):

```sql
-- docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "<query>"
SELECT date_trunc('hour', created_at) AS hour,
       coalesce(role, 'none') AS landed_on,
       reason,
       count(*)
FROM gpu_pool_leases
WHERE created_at > now() - interval '48 hours'
  AND holder = 'cortex-exec'
  AND work_class = 'fast'
  AND priority = 'interactive'
GROUP BY 1, 2, 3
ORDER BY 1, 2;
```

Caveat: route `quick_embodied` has the same class and priority (`config/gpu_pool.yaml:197`). The
lease row does not store the route. To keep only checks, join `turn_correlation_id` to the
correlation IDs in the check's own log lines. This join works only for turns with a UUID
correlation ID. For any other ID, cortex-exec uses a fresh `uuid4()` for the gateway call and logs
`repair_probe_turn_corr_not_uuid`.

```bash
docker logs orion-athena-cortex-exec --since 48h 2>&1 \
  | grep -E 'pre_turn_appraisal_paradigm_(result|failed)' \
  | grep -oE 'corr=[0-9a-f-]+' | sort -u | sed 's/corr=//' > /tmp/pre_turn_corrs.txt
```

Failure reasons from the same log:

```bash
docker logs orion-athena-cortex-exec --since 48h 2>&1 \
  | grep pre_turn_appraisal_paradigm_failed \
  | sed -E 's/.*reason=([^ ]+).*/\1/; t; s/.*/exception_no_reason/' | sort | uniq -c
```

Lines with no `reason=` come from the generic exception branch (for example a `TimeoutError` from
`asyncio.wait_for`). They are counted as `exception_no_reason`.

Decision rule: any `landed_on = 'chat'` row tied to a background turn means the flood is real.
Build Patch 2. Zero such rows over 2 days means Patch 2 is only about fitting long text. Then it
competes with simply trimming the prompt.

## Patch 2 (proposed, gated): background checks use their own seat

A turn that holds a GPU lease runs its check inside that leased seat. The check uses route `agent`
plus the lease reference. Admission already bounds those seats. So background checks add no new
GPU demand, and they get a window large enough for their text. Juniper's chat turns have no lease.
They keep `quick_turn`.

### Holes

| Hole | Why it matters | How to close it |
|---|---|---|
| Thinking model may emit a think tag as its first token | The probe scores the first token. A think tag gives a hollow score. The probe already sends `chat_template_kwargs={"enable_thinking": False}`. Nobody has checked that the agent-seat backend obeys it. | Prove it with the gate below. If the flag is ignored, score the first token after the tag. |
| Background and chat scores come from different models | Qwen3-8B vs the agent-seat model. Levels may not be comparable. Equilibrium reads them the same way. | Tag the model on the result. Compare distributions before letting equilibrium mix them. |
| Can the probe use the motor's seat? | UNVERIFIED. The gateway may refuse a second call on a held lease, or queue it behind the motor. | Read the gateway lease path. Run one live probe. |
| Two-service deploy order | `PreTurnAppraisalRequestV1` is `extra="forbid"`. A Hub that sends a new lease field to an old cortex-exec fails every check. | Deploy cortex-exec first. Hub second. Say so in the PR. |
| gpu0 flood is predicted, not observed | Today 0 checks landed on chat. The fallback list allows it. Nobody has seen it happen. | The measurement step. |

### Gate before building

One real probe through an agent seat, with a held lease, on a long world-pulse thread. Run it on at
least 5 different threads. Pass: real logprob scores that vary across threads, no `_TEXT_FALLBACK_*`
values, no think tag scored. Fail: do not build Patch 2. Trim the prompt instead.

## Proposed schema / API changes

- Patch 1: none.
- Patch 2: one optional field on `PreTurnAppraisalRequestV1`, e.g.
  `gpu_lease: GpuLeaseRefV1 | None = None`. cortex-exec uses it to pick route `agent` and pass the
  lease to the gateway. Absent field means today's behaviour (`quick_turn`). Consumer-first deploy.

## Files likely to touch

Patch 2 only (Patch 1 files belong to `fix/pre-turn-check-parallel`):

- `orion/schemas/pre_turn_appraisal.py`: optional lease field.
- `orion/hub/turn_orchestrator.py`: pass the turn's lease into the check request.
- `services/orion-cortex-exec/app/pre_turn_appraisal.py`: pick route and lease from the request.
- `orion/substrate/appraisal/paradigms/repair_pressure_v2.py`: think-tag handling if the gate shows it is needed, model tag on result.
- Tests in `services/orion-cortex-exec/tests/` and `services/orion-hub/tests/`.

## Missing questions

1. **Should Juniper's checks move to the chat lane?** This was Juniper's earlier instinct. Gain: one
   model for her checks and her replies, and a huge window. Cost: the chat lane has one slot. Her
   check and her motor would take turns on it. The check adds ~0.7 s, but if a background job
   ever holds that slot, her check waits. Strata logprob scores for the probe are also UNVERIFIED.
2. **Should the Hub's 180 s check timeout come down?** A 2.4 s check that waits 180 s is a stall,
   not a measurement. Something like 20-30 s would cap the damage. Cost: a slow-but-real check
   gets dropped instead.
3. **Should field-notes, self-sense, and reminiscence get explicit in-flight caps?** World-pulse
   reads have one (3). I did not find caps for these three. UNVERIFIED.

## Non-goals

- The separate ~5 min median motor latency since ~10-30. Different cause, different doc.
- Skipping the check for background turns. Juniper rejected that.
- Trimming the probe prompt. It is the fallback if the Patch 2 gate fails, not part of this plan.

## Acceptance checks

- Patch 1: two checks sent at once both start without waiting for each other (test). Live: no
  Juniper turn whose check waits more than ~10 s while a background check is stuck.
- Measurement: 48 h of landing counts by role, with the gpu0 count stated.
- Patch 2 gate: 5 live agent-seat probes with varying real scores, logged with correlation IDs.
- Patch 2: background check failure rate for `world_pulse_read_stage2` falls well below 194/373.
  Juniper's checks still land on `quick_turn`.

## Recommended next patch

Ship Patch 1. Run the 2-day measurement. Then run the one-probe gate. Build Patch 2 only if the
gate passes.
