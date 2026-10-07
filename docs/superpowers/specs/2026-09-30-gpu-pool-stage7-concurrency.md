# GPU pool stage 7: concurrency (two agent runs per card)

**Status:** design. No service code. Inputs: PR #2434 (Bonsai field notes, flash-attention A/B, and
the design `docs/superpowers/design/2026-09-30-agent-lanes-to-bonsai-design.md`), stages 1-6 specs,
and live reads on 2026-09-30 around 09:46 UTC.

**Binding scope (Juniper, 2026-09-30):** Bonsai goes on **gpu1 and gpu2**, the two agent cards.
gpu1 is the agent's home seat. gpu2 is the `agent-gpu2` swap seat, which shares its card with
diffusion and world. **gpu0 stays chat's card.** Bonsai does not take gpu0 or evict chat. PR #2434's
gpu0 default is not the target state (see "Prerequisite: undo #2434's gpu0 default").

## Arsonist summary

- **What we want:** each agent card serves two Orion runs at once instead of one. Today seven
  curiosity/reading runs sit in line for over an hour while each card works on exactly one.
- **The pool already counts in slots.** It reads a worker's slot count and per-slot context from
  llama.cpp, grants per slot, and the metacog and fast lanes already run 4 slots in production. A
  2-slot agent worker would be discovered and leased with **no pool change**.
- **But it would not run two Orion runs.** The scheduler allows **one durable-run hold per role**
  (`orion/gpu_pool/scheduler.py:276-277`). With two slots, the second slot only takes one-off
  requests. Of the three things we want from concurrency, this rule blocks the main one.
- **Two measured risks come first, before any scheduler change:**
  1. **Two at once can be slower than one.** With flash attention on and 4×65K slots, two
     concurrent Bonsai runs got 23.4 tok/s each (~47 total), below one run alone (~52). The
     proposed 2×131K layout with two active runs has **never been measured**.
  2. **llama.cpp's cross-slot prompt cache has an open bug that can paste one conversation's
     content into another slot** (llama.cpp #27148, open). It is on by default. Whether our builds
     have it is **UNVERIFIED**. The metacog and fast lanes already run 4 slots with the default on.
- **Plan:** measure first (7.1), then put Bonsai on gpu2 with two slots and today's one-hold rule
  (7.2). That alone frees a slot for one-off agent calls. Then let a role take as many holds as a
  config key allows, and fix a bookkeeping gap that would let one gap-sharing call stall two runs
  (7.3). Then telemetry (7.4), then gpu1 (7.5). Each step is one config line to roll back.

## What "concurrency" means here, concretely

A llama.cpp server started with `--parallel N` splits `--ctx-size` evenly into N **slots**. Each
slot is one conversation's working memory (its KV cache). The server decodes all busy slots
together in one batch. So:

- **Per-slot context = ctx_size / N.** Live `/props` confirms it: metacog runs `ctx_size 16384`,
  `--parallel 4`, and reports `n_ctx 4096` and `total_slots 4` (circe port 8012, 2026-09-30). The
  agent lanes run `--ctx-size 131072 --parallel 1` (live argv on both agent containers).
- **Two slots × 131K needs `ctx_size 262144`.** The Q4 27B cannot do that on a 32 GB V100: it
  uses 24.9 GB live at 1×131K, and the extra 131K of KV adds roughly 6-8 GB, past the profile's
  29.5 GiB cap. **Bonsai fits 2×131K in ~24.1 GB** (the same total cache as its measured 4×65K).
  So agent-lane concurrency is a Bonsai feature, unless a quantized KV cache is tested.
- **Throughput vs per-run speed** (PR #2434, tok/s, thinking off, 512 tokens):

  | Runs at once | Bonsai 4×32K, FA on | Bonsai 4×65K, FA on | Q4 27B 4×32K, FA on |
  |---:|---:|---:|---:|
  | 1 | 51.7 | 51.7 | 33.0 |
  | 2 | 42.8 each (~80) | **23.4 each (~47)** | 26.4 each (~51) |
  | 3 | 33.7 each (~91) | not measured | 22.1 each (~62) |
  | 4 | 26.8 each (~97) | 27.0 each (~99) | 18.0 each (~67) |

  At 61K of context, one Bonsai run decodes 32.4 tok/s with FA on; Q4 27B 25.4. **The 2×131K
  layout, with two active runs at depth, is unmeasured.** If it shows the 4×65K dip, each run gets
  ~23 tok/s: slower than today's Q4 alone (25-33), with ~47 total against Q4's 33. That is still
  more total work, but each individual run is slower.
- **Prompt cache.** Each slot keeps its own KV. llama.cpp chooses the slot, not the pool: it
  prefers an idle slot whose cached prompt is most similar. With default `--cache-ram 8192` MiB and
  `--cache-idle-slots`, an idle slot's state is saved to host RAM when a new task arrives and can
  be restored later. In #2434's four-loop test, 73-80% of prompt tokens came from cache and the
  slots did not evict each other. On both Qwen3.8 models, the previous assistant reply was
  re-processed every step (the template renders it differently as history).
- **VRAM is allocated at boot.** All slots' KV is reserved when the server starts, so memory does
  not grow with load. More concurrent decode does raise sustained power and heat.

## Current architecture (verified live, 2026-09-30 ~09:46 UTC)

- **Roles** (`GET /v1/pool`): chat 1×65536 (gpu0, 35B), agent 1×131072 (gpu1, Q4 27B), agent-gpu2
  1×131072 (gpu2, Q4 27B, swap seat, loaded 09:10), metacog 4×4096 and fast 4×4096 (gpu3),
  world `slots: 2` (service), diffusion evicted, experiment unloaded. gpu0 is `lent: true`.
- **Queue right now:** 7 background agent holds queued, the oldest for 62 minutes. Granted holds:
  one on agent, one on agent-gpu2, and one recalling on chat. Holders are durable runs (reading,
  reverie, curiosity, orion-day).
- **Last 3 days, grant waits** (`gpu_pool_events` `granted.waited_ms`):
  - background agent holds: p50 0.1 s / p90 **8,724 s** on agent; p50 1,224 s / p90 6,595 s on
    agent-gpu2 (the 1,200 s swap trigger); p50 101 s / p90 4,213 s on chat.
  - system agent requests: p90 2.1 s on agent, 0.3 s on agent-gpu2, 2.4 s on chat.
  - chat interactive: p50 6.7 s / p90 **90.3 s**.
  - Sampled every 10 min over 3 days: on average **2.6 agent holds waiting** (max 13) and 1.1
    granted.
- **Hold shape** (3 days): agent holds p50 2,225 s / p90 5,450 s held, ~11.5 calls each, median call
  ~181 s. A run keeps its seat busy ~72-78% of the time it holds it. The other ~25% is the "gap"
  that higher-priority calls share (Juniper, 2026-09-25).
- **Recalls** (3 days): 184 agent leases recalled off chat for `owner_waiting`, 102 metacog, 29 fast;
  17 `max_hold` + 7 `draining` on agent-gpu2.
- **Runs already make parallel calls sometimes:** 201 of 3,467 hold children (5.8%) were created
  while a sibling call of the same run was still in flight. They wait behind it (rule H2).
- **Slot/context trade-off is already live on gpu3:** metacog is 4×4K. In 3 days, 245 metacog
  requests needed more than 4K. 162 of them ran on **chat's card**, 28 on agent, and 53 were never
  placed.
- **Agent prompt sizes** (`min_ctx_tokens`, 3 days, 3,548 agent requests): p50 12K, p90 44K, max
  73K. 44 (1.2%) exceed 65K. Holds almost never carry a `min_ctx_tokens` (1 of 205).
- **Pool lock** (#2331): 221 lock holds over 250 ms in 24 h. Held time p50 294 ms / p90 699 ms, and
  waits up to 1.4 s, at ~10,000 leases a day.
- **circe cards now:** gpu1 24.9 GB used, 75 °C, 220 W. gpu2 25.7 GB, 72 °C.

## Breakage and impact inventory

Three verdicts: **works** (correct at N>1 as written), **degrades** (correct but worse, or needs
recalibration), **breaks** (wrong or blocks the goal). File:line against main `a005658db`.

**Checked against the real scheduler:** `schedule()` was replayed on main's code and
`config/gpu_pool.yaml`, with agent set to 2 slots × 131K.
- **Held run A plus a queued second run:** the second run is **not granted**. Control: alone on
  the role, it is granted.
- **Two held runs A and B, one system call using a gap, and both runs' next calls queued:**
  **neither call is granted**. Control: without the gap call, both are granted.

### Pool core

| Mechanism | Verdict | What happens at 2 slots |
|---|---|---|
| Discovery: `total_slots`, per-slot ctx (`orion/gpu_pool/discovery.py:97-102`) | **works** | Reads `/props` `total_slots` and `default_generation_settings.n_ctx`. Verified per-slot live (metacog 16384/4 = 4096). |
| Discovery change events (`services/orion-gpu-pool/app/runtime.py:334-337`, emit at `:543-547`) | **degrades** | Events fire only when a role's *status* changes. A profile that changes slots or ctx but keeps the same model file stays `confirmed`, so no event fires. `discovery_confirmed.detail` carries profile and file, not slots or ctx. A slot change is silent in history. |
| Announce vs live N (`orion/schemas/gpu_pool.py:367-380`, `discovery.py:113-119`) | **degrades** | The announce carries no slots. Only the model file is checked against the profile, so an `LLAMACPP_N_PARALLEL_OVERRIDE` that disagrees with the profile is never flagged. |
| `seen_ctx` / `min_ctx_exceeds_class` (`runtime.py:329-333`, `scheduler.py:845-862`) | **works at 2×131K; breaks at 4×65K** | Per-slot ctx is unchanged at 2×131K. At 4×65K on both agent cards, the class's largest known slot becomes 65K. 1.2% of agent requests would then be refused with `min_ctx_exceeds_class`, or spill to chat. |
| Placement / `fits` (`scheduler.py:294-312`) | **works** | Compares `min_ctx_tokens` to per-slot ctx. |
| Free slots (`scheduler.py:270-292`) | **works** | Free = `slots - used - idle holds it may not jump`. Slot-aware. |
| **One hold per role, H1** (`scheduler.py:276-277`) | **breaks the goal** | `if lease.kind == "hold" and self.holds.get(role) and lease.priority != URGENT: return 0`. A second durable run can never hold a 2-slot role. The 7 queued holds wait exactly as long as today. |
| One call per run, H2 (`scheduler.py:546-555`) | **works (keep)** | A run's calls stay serial in its one slot. The 5.8% overlapping children keep waiting for their sibling. See decision D3. |
| **Gap sharing with 2 holds, H3** (`scheduler.py:282-292`) | **degrades (once 2 holds are allowed; reachable today only with a stacked urgent hold)** | The pool does not record whose gap a gap-sharing call took. Two slots, two idle holds A and B, one interloper: A's next call sees `2 - 1 used - 1 (B idle) = 0`, and B's sees the same. **One interloper stalls both runs** until it ends. Today it stalls one. |
| Urgent stacking and pause, U1-U3 (`scheduler.py:276`, `:638-746`) | **works, better** | Urgent holds already stack, bounded by slots. With a free slot, urgent work is granted with no pause. With two background holds, U1 pauses the most recently granted one. |
| Owner recall (`scheduler.py:688-704`) | **works** | It counts per slot: `needed = len(starving) - already`. |
| Swap unload/idle (`scheduler.py:785-801`) | **degrades** | `busy_seat = occupancy > 0`. With two tenants, a seat is idle less often, so the `idle` unload and residents' reclaim come later. This is correct but shifts when diffusion gets gpu2 back. |
| `max_hold_sec: 9000` on agent-gpu2 (`config/gpu_pool.yaml:55`, `scheduler.py:476-482`, `:727-728`) | **degrades** | A drain recalls every hold on the seat. With two runs, one drain costs two take-backs, each with its 600 s grace (`hold_clawback_grace_sec`). The take-back cap is 12 per run (`services/orion-durable-runs/app/settings.py:75`). |
| Swap trigger `after_wait_sec: 1200` (`gpu_pool.yaml:60`) | **degrades (good)** | With 2 holds on gpu1, fewer runs wait 20 min, so gpu2 loads less often and diffusion/world keep gpu2 longer. This changes the balance Juniper set in stage 5. It is a real effect, not a bug. |
| `serialize_with` world↔diffusion (`gpu_pool.yaml:75-79`, `scheduler.py:345-352`) | **works** | Does not involve LLM roles. |
| Pool lock (#2331; `runtime.py:241-262`) | **degrades slightly** | More agent-card calls per hour means more lock traffic. The two agent roles carried ~1,150 of ~10,000 leases a day (3-day grants), so even doubling them adds ~10%. Watch `gpu_pool_slow_lock`. |
| DB events and prune (`services/orion-gpu-pool/app/store.py:192-210`) | **works** | Rows scale with calls, not slots: 86 MB since 09-24. |

### Swap seats and actuation

| Mechanism | Verdict | What happens |
|---|---|---|
| Profile sent on load (`orion/gpu_pool/config.py:335-341`) | **works, limits** | The pool always sends `launch.profiles[0]`. Choosing by ctx×slots/VRAM is not built. A 2-slot Bonsai profile first in the list is enough, and reverting is one line. |
| Env passed to compose (`services/orion-gpu-lane-controller/app/pool_fence.py:89-102`) | **works, limits** | Only `cuda_env` and `profile_var` are passed. N and the image come from the profile or compose, not the pool. The Bonsai image needs per-worker image selection (design doc step 1). |
| Unload idle check (`services/orion-gpu-lane-controller/app/launch_exec.py:171-174`) | **works** | Requires every slot `is_processing: false`. |
| Residency / cooldown (`gpu_pool.yaml:17,24`) | **works** | Card-level. |
| gpu2 VRAM with world co-resident | **works** | Bonsai at 2×131K is ~24.1 GB, against today's Q4 + world at 25.7 GB. Diffusion (24 GB) is still evicted. |
| experiment seat (`gpu_pool.yaml:88-90`) | **works** | Unaffected: operator-only, no launch block. |

### Gateway

| Mechanism | Verdict | What happens |
|---|---|---|
| One lease per call, attach under hold (`services/orion-llm-gateway/app/pool_placement.py:206-259`) | **works** | Two concurrent calls hold two leases. |
| Per-role executor (`pool_placement.py:398-409`; `LLM_GATEWAY_EXECUTOR_WORKERS_PER_ROLE=8`, `settings.py:99-100`) | **works** | 8 threads per role URL is more than 2 slots. |
| Overflow ladder (`services/orion-llm-gateway/app/main.py:380-426`) | **works at 131K/slot; breaks under a hold if ctx shrinks** | It re-leases with `ctx_per_slot + 1` only when `hold is None` (`main.py:398`). A held call that overflows fails, with no re-lease. Holds carry no `min_ctx_tokens`, so a hold could land on a smaller-slot role. Safe only while every agent role keeps 131K per slot. |
| `served_by` (`services/orion-gpu-pool/app/runtime.py:1305-1306`) | **works** | Per role (`circe-worker-agent-gpu2`). Nothing needs a slot index. |
| `/routes` compat `n_ctx` (`pool_placement.py:501`) | **works** | Per-slot. Stage 6.3/6.5 retire it anyway. |
| FCC ctx budget (`orion/harness/fcc_motor.py:555-563`, `:918-934`; `orion/fcc/context_budget.py:42-65`) | **works at 131K; silent shrink otherwise** | Reads per-slot `n_ctx`. A 4×65K profile would halve Claude Code's context with no warning. |
| Harness `fcc:<role>` keys (`services/orion-harness-governor/app/bus_listener.py:59-82`) | **degrades** | One latency population per role. Two turns sharing a card decode slower, which reads as a latency shift. |

### Durable runs

| Mechanism | Verdict | What happens |
|---|---|---|
| One hold per run (`services/orion-durable-runs/app/pool_hold.py:1-18, 166-172`) | **works (keep)** | Decision D3: a run never holds more than one slot. |
| Takeback cap / `taken_back` (`services/orion-durable-runs/app/admitted_graph.py:113-124`) | **works** | Per run. Fewer owner reclaims on chat means fewer take-backs overall; max_hold drains cost more (above). |
| Requeue after a failed turn, #2402 (`services/orion-durable-runs/app/admission_runtime.py:580-621`) | **works** | Per lease. |
| Urgent pause, #2385 (`admission_runtime.py:538-570`) | **works** | Per hold. |
| `hold_placement` min_ctx (`pool_hold.py:127-143`) | **gap (existing)** | It exists, but callers almost never set it (1 of 205 holds). It must be set before any agent role goes below 131K per slot. |

### Telemetry, field, cognition

| Mechanism | Verdict | What happens |
|---|---|---|
| `gpu_pool_waiting` / oldest wait (`services/orion-field-digester/app/store.py:483-548`; `orion/field/queue_contention.py:70-87`) | **works, recalibrates** | Counts waiting non-hold leases pool-wide. Waiting holds count in `durable_demand_pending` (12 h anchor). Both fall with concurrency. The 24 h EWMA baseline will re-learn a lower rest, so expect a transient "calmer" reading. This is true, not an artifact. |
| Stage 6.2 two clocks (spec on `docs/gpu-pool-stage6-design`, Decision 1, gate G2) | **breaks its own theory** | G2's anchor: "a drop in decode_tps at fixed model/slot means the worker degraded (thermal, a co-tenant, ctx growth)". At N=2 a co-tenant is normal. A per-role `decode_tps` rest band would be bimodal (alone vs shared) and read ordinary sharing as degradation. Its 48 h checkpoint is also invalidated if Bonsai lands mid-window. |
| Transport baseline `gpu_pool_wait` hop (`runtime.py:1404`; `services/orion-equilibrium-service/app/settings.py:230-237`) | **works** | Baselined, never triggers. |
| `fcc:` transport keys (6.7, deferred) | **degrades** | They absorb slot contention as "transport". This strengthens the case for 6.7's exclusion. |
| `bus_synaptic_prediction_error` (`orion/substrate/prediction_error.py:826-856`) | **works** | Built on bus inter-arrival gaps, not model latency. |
| `reasoning_load` → `node:circe` (`services/orion-field-digester/app/ingest/state_deltas.py:321-335`) | **works** | Per run, attributed by `served_by` node. Replace mode, last writer wins, the same as today across roles. |
| Orion's knowledge of its serving model (`orion/situational/context.py:646-651, 1976`; `fcc_served_model`) | **works** | Shows the model name (Bonsai will show as itself). Orion is not told a run shares its card; not needed now. |
| Hub pool panel (`services/orion-hub/static/js/gpu_pool.js:52-66, 272-275, 380`) | **works** | Already renders `busy/slots` and ctx per slot. It does not show holds against a hold limit (new in 7.3). |
| Grammar (`runtime.py:79-83`) | **works** | Grammar gets exceptions only. Volume grows with throughput, not slots. |

### Evals

| Eval | Verdict | Why |
|---|---|---|
| Pool day (`services/orion-gpu-pool/evals/run_pool_day_eval.py:95-98`, `:104`, `:176`) | **breaks as evidence** | Hardcodes agent roles at 1 slot. Service time (`CALL_SEC = (30, 90)`) ignores how many slots are busy, so a 2-slot replay would show a free doubling that the bench says is not free. |
| hold_fairness (`services/orion-durable-runs/evals/hold_fairness.py:6-10`, `:143`) | **breaks when H1 changes** | Check A asserts "at most one hold on the agent card at a time". |

## Downstream impacts beyond the pool

1. **Chat gets its card back more often (benefit).** Agent overflow spills onto gpu0 when it is
   lent: 2,290 system agent calls and 180 background holds ran on chat in 3 days. There were 184
   `owner_waiting` recalls of agent work, and chat's own p90 wait was 90 s. Two slots per agent
   card should cut both. It is a measurable acceptance check, not a promise.
2. **Cross-conversation bleed (llama.cpp #27148, open).** Under concurrent multi-slot load, the
   RAM prompt cache can restore an unrelated finished conversation into a fresh slot, and the API
   reports `cached_tokens: 0`. Reported on the Aug 2026 master branch; the fix PR is still open.
   Workarounds: `--cache-ram 0` or `--cache-idle-slots false`.
   - On agent lanes a bleed would mix Orion runs and Claude Code (FCC) turns. Those can carry
     Juniper's conversation, so this is a **privacy boundary**, not just a quality bug.
   - **metacog/fast already run 4 slots with the defaults.** Whether stock b10398 or Prism b10750
     has the bug is **UNVERIFIED**.
   - It is also in tension with the next item: the RAM cache is what protects a run's prefix
     today when a gap-sharing call takes its slot.
3. **Prefix-cache thrash.** The pool counts slots, but llama.cpp picks which slot a call lands in.
   If run A's next call lands in the slot that held run B, A re-reads its whole context. At 60K
   and ~380 tok/s prefill, that is ~2.5 min, and it evicts B unless the RAM cache saves it.
   `id_slot` can pin a request to a slot (documented for `/completion`; **UNVERIFIED** on
   `/v1/chat/completions` in our builds). Measure `timings.cache_n` per call before designing
   around it (7.4).
4. **Prism template re-read** (#2434: 23.5% of processed tokens were the model's own prior reply).
   Not caused by concurrency, and both Qwen3.8 models pay it. But at N=2, one slot's re-read
   prefill competes with the other slot's decode.
5. **Quality.** Hard depth test: Bonsai 19/20 against Q4's 20/20. One fabricated-looking story. At
   `medium` effort it once thought past an 8K cap. The live agent profile runs `reasoning_effort:
   xhigh` with `n_predict 16384`. At ~23-27 tok/s shared, one runaway thinker holds a slot for
   ~10 min and slows its co-tenant the whole time. The design doc's gpu2-first A/B (`turn_ok`, 5-point
   margin) is the gate.
6. **VRAM and thermal.** VRAM is fixed at boot (fine). gpu1 already sits at 75 °C and 220 W at one
   slot. Two slots decoding means a sustained ~100% duty cycle. The thermal guard only gates swap
   *loads* (`gpu_pool.yaml:60`); the `cooling_incident` shed (`orion/gpu_pool/shed.py`) is the only
   brake on running work. Check thermals in the 7.2 soak.
7. **Power intent.** Today only diffusion emits power intents (`services/orion-diffusion-host`).
   If LLM runs ever do, per-card incremental draw would be charged in full to each of two
   concurrent runs (double count). No action now; noted so it isn't built wrong later.
8. **Swap balance on gpu2.** Diffusion and world gain gpu2 time (above). The 1,200 s trigger was set
   when gpu1 took one run. It may be worth revisiting after 7.5, as a Juniper call.

## Proposed design

### Principle: the grant unit is already the slot. Change who may take slots, not what a slot is.

- **A pool slot is a count, not a llama.cpp slot index.** Keep it that way unless 7.4's
  `cache_n` data shows real thrash (then consider `id_slot` pinning, D5).
- **A durable run holds exactly one slot** (H2 unchanged). Its calls stay serial. Two runs on a
  2-slot card each hold one slot.
- **A new per-role config key, `max_holds`** (default 1, today's behaviour), replaces H1's
  hard-coded "one". The scheduler allows a non-urgent hold on a role while `holds < min(max_holds,
  slots)`. Urgent holds keep U3 (bounded by slots). Rollback = delete the key.
- **Pin a gap-sharing call to one hold.** When a strictly-higher-priority call uses an idle hold's
  gap, the scheduler charges it to one named hold: the most recently granted idle hold on that role
  whose priority it outranks. The other run's next call is then not blocked. The attribution lives
  in `_Ctx` for the tick, rebuilt from active leases each tick, so no schema change. This keeps
  "at most one interleaved call per run gap" true at N>1.
- **Profile choice stays "first entry".** Concurrency is a property of the profile
  (`n_parallel`, `ctx_size`). Choosing a profile by ctx×slots×VRAM is **not** built (D4).
- **Per-slot context stays 131K on agent roles.** Anything smaller first needs holds to carry
  `min_ctx_tokens` and the gateway to handle an overflow under a hold.

### Why not the alternatives

- **Slots as first-class indexed objects in the pool** (grant "slot 1 of agent-gpu2"): only
  useful with `id_slot` pinning. Wait for cache data.
- **Let a run hold 2 slots:** 5.8% of calls overlap a sibling. They wait for one call. Doubling a
  run's footprint to save that wait halves the number of runs that fit.
- **`--kv-unified` (one shared 262K pool):** flexible per-slot size, but discovery would then report
  a per-slot ctx that isn't a real per-conversation limit. Separate experiment.

## Decisions for Juniper (recommendations, not decisions)

- **D1. `max_holds` per agent role.**
  - Recommend `agent-gpu2: 2` first (7.3), then `agent: 2` after 7.5.
  - Reason: today system agent calls wait p90 2.1 s on agent and 0.3 s on agent-gpu2 while one hold
    sits there. Filling both slots with runs leaves one-off calls only the gaps (~25%).
  - Acceptance: the agent-class system-request wait p90 must stay within 2× today. Otherwise
    `max_holds: 1` on that role.
  - Alternative: `slots - 1` (always keep one slot for one-off calls). Safer for FCC turns, but
    gpu1 would then gain nothing for runs.
- **D2. #27148 mitigation.**
  - Recommend: run the bleed canary (7.1) on both builds first.
  - If it reproduces, set `--cache-idle-slots false` on multi-slot lanes (metacog/fast included)
    and measure the `cache_n` cost. If that cost is high, `--cache-ram 0` is not better; the
    remaining option is `id_slot` pinning (D5).
  - This one is a privacy call, so it is yours.
- **D3. A run never holds more than one slot.** Recommend yes (keep H2).
- **D4. Profile choice by ctx×slots.** Recommend: don't build. One profile per seat; roll back by
  reordering `launch.profiles`. Revisit only if a seat needs two layouts by time of day.
- **D5. `id_slot` pinning.** Recommend: not now. Decide on 7.4's `cache_n` data.
- **D6. gpu2 swap trigger (1,200 s)** after both cards run two slots. Recommend: leave it. Look again
  with a week of 7.5 data.
- **D7. Thinking effort on the Bonsai agent profile** (the design doc's open question 2).
  Recommend `xhigh` for parity, with `n_predict 16384` as the bound. Its cost now also lands on the
  co-tenant, so watch per-call decode tokens in the A/B.

## Stage-6 work that should change because of stage 7

- **6.2 (two clocks):** also record, per call:
  - the role's occupancy at grant (active leases on the role, from the pool state the gateway
    already has, or the grant reply);
  - llama.cpp `timings.prompt_n` and `cache_n`.

  Then G2's rest band is keyed by (role, occupancy), not role alone, and the cache data answers D5.
  Adding it now costs one field each. Adding it after 6.2 ships means a second rollout.
- **6.2's 48 h checkpoint** must not straddle a model or slot change on a role. Either finish it
  before 7.2 deploys, or restart the window afterwards.
- **6.7 (`fcc:` transport keys):** concurrency adds a second reason to exclude them.
- **6.3/6.5 (`/routes` retirement):** no change. `ctx_per_slot` from pool state is already
  per-slot.
- **6.1:** no change (#2441).

## Metric quality gate record

The only new measured values are two covariates on 6.2's existing samples. There is no new field
channel.

| # | Candidate | Verdict |
|---|---|---|
| C1 | occupancy at grant | **Covariate, not a signal.** Provenance: pool active leases per role at the grant tick. Independence: it is the pool's own count, so it is used only to band `decode_tps`, never wired alone. Theory: batched decode on one GPU splits compute across active slots (the #2434 table). Live data: metacog/fast run 4 slots today, so the band can be checked before the agent lanes change. Reversible: a projection field. |
| C2 | `cache_n / (cache_n + prompt_n)` per call | **Record only**, for D5. Provenance: llama.cpp `timings` in the reply. Rest state: 1.0 is reachable (a full cache hit) and so is 0.0 (a cold start). It must be read per role and per call kind, never averaged across roles. Not wired to the field. |

## Proposal-mode notes (autonomy/cognition-adjacent)

- **Capability change:** more Orion runs think at once. Each is individually a bit slower, and the
  queue is shorter.
- **Data touched:** none new. The pool's lease and event tables and existing harness traces.
- **Privacy boundary:** llama.cpp slots on one server share a process and a host-RAM prompt cache.
  #27148 is exactly a crossing of that boundary. Gate: D2 plus the canary.
- **Trace proving it worked:**
  - `gpu_pool_events` shows two concurrent `granted` holds on one role;
  - `discovery_confirmed` shows `slots: 2`;
  - the queued-hold p90 wait falls.
- **Dangerous failure mode:** content bleed between a private conversation and a run. Next: a
  quality drop that `turn_ok` does not catch.
- **Disable:** `max_holds` back to 1 (pool config only), or `launch.profiles` back to Q4.

## Prerequisite: undo #2434's gpu0 default

#2434 moves Bonsai's default card to gpu0 and documents stopping chat. Under Juniper's binding
scope, that is not the target state. Recommend fixing it **in #2434 before it merges** (smallest
change):

- `services/orion-llamacpp-bonsai-host/.env_example:26` and `docker-compose.yml:37`: restore
  `BONSAI_CUDA_VISIBLE_DEVICES` default 2, or drop the default so it must be set. Sync local `.env`
  on athena and circe.
- `config/llm_profiles.yaml` (`ternary-bonsai2-27b-pq2-v100-32gb-circe-np4`, ~line 2151 on that
  branch): `device_ids: [0]` → `[2]`, or empty (the actuator sets the card from `gpu_pool.yaml`).
- The README's "stop chat, start Bonsai on gpu0" swap: replace it with the gpu2 bake-off path (stop
  `atlas-agent-burst` via the pool: pause actuation or wait for idle unload).
- The design doc: drop any gpu0 wording. Its target cards are already gpu1/gpu2.

If #2434 merges as-is, 7.0 below does the same as a follow-up.

## Non-goals

- Chat concurrency on gpu0. The 35B uses 26.5 GB at 1×65K; a second slot does not fit.
- Moving chat, fast or metacog to Bonsai.
- Quantized KV (`q8_0`) or `--kv-unified`: separate experiments.
- Slot-indexed grants or `id_slot` pinning (D5, data first).
- Profile selection by ctx/VRAM (D4).
- Any change to the one-slot-per-run rule (D3).
- A reducer or field channel for occupancy (gate C1: covariate only).

## Acceptance checks (live)

1. **Bench (7.1):** a gpu2 bake-off at 2×131K, FA on, 1 and 2 concurrent runs at 14K/32K/61K/100K
   depth, written to a field note. Pass: 2 concurrent ≥ 1.3× the total tok/s of 1 run at every
   depth. Fail: stop at 7.2 with 1 slot, or test FA off at N=2.
2. **Bleed canary (7.1):** 2 (and 4) concurrent conversations, each with its own random nonce, 200
   turns, on stock b10398 (metacog port, off-hours) and on Prism b10750. Pass: no nonce appears in
   another conversation's output or reasoning.
3. **Discovery (7.2):** `GET /v1/pool` shows agent-gpu2 `slots: 2, ctx_per_slot: 131072`, and a
   `discovery_confirmed` event whose detail carries slots and ctx (7.4).
4. **Two runs at once (7.3):** `gpu_pool_events` shows two `granted` holds on agent-gpu2 whose held
   intervals overlap.
5. **Queue relief (7.3 + 7.5, 7 days vs the 3 days before):**
   - queued-hold wait p90 on agent falls below 8,724 s;
   - the average number of waiting agent holds falls below 2.6.
6. **No harm to one-off calls:** system agent request wait p90 stays within 2× baseline (2.1 s agent,
   0.3 s agent-gpu2).
7. **Chat benefit:** `owner_waiting` recalls of agent work on chat fall from 184 per 3 days, and chat
   interactive p90 wait falls below 90 s. (No regression is required; a fall is expected but not
   promised.)
8. **Quality:** the design doc's A/B. Bonsai `turn_ok` within 5 points of Q4, and a hand-read sample
   of curiosity findings.
9. **Gap pinning (7.3):** a unit test and a pool-day eval case. With 2 holds and 1 interloper, one
   run's next call is granted immediately.
10. **Rollback drill:** set `max_holds` to 1, and in the next tick no second hold is granted on that
    role. Revert `launch.profiles`, and the next load serves Q4.

## Stage-7 PR sequence

Each PR is its own branch off main and deployable alone. Deploy top to bottom.

| PR | What | Depends on | Services / deploy |
|---|---|---|---|
| **7.0** | `fix(bonsai-host)`: gpu0 default → gpu2 or unset; profile `device_ids`; README swap path. **Folds into #2434 if it hasn't merged.** | #2434 | none (manual-only service); local `.env` on athena + circe |
| **7.1** | `docs(bench)`: 2×131K concurrency at depth + #27148 bleed canary on both builds. Field note, bench scripts. No service code. | 7.0 (so the bake-off runs on gpu2, not chat's card) | circe manual bake-off (gpu2, pool actuation paused or seat idle) |
| **7.2** | `feat(llamacpp-host)`: the design doc's "recommended next patch": `Dockerfile.prism`, per-worker image for `atlas-agent-burst`, Bonsai agent profile `n_parallel: 2, ctx_size: 262144`, cache flags per D2, first in `agent-gpu2.launch.profiles`. **H1 unchanged**, so this adds one request slot, not a second run. | 7.1 pass; #2434 merged | llamacpp-host image on circe; `gpu_pool.yaml` (pool config reload) |
| **7.3** | `feat(gpu-pool)`: `max_holds` per role (default 1), gap pinning, scheduler tests, pool-day eval slots=2 variant with an occupancy slowdown model from the 7.1 table, hold_fairness check A reads `max_holds`. Then set `agent-gpu2: max_holds: 2`. | 7.2 live ≥ 24 h | orion-gpu-pool |
| **7.4** | `feat(gpu-pool, llm-gateway)`: discovery event on slot/ctx change with slots+ctx in the detail; pool state/Hub show `holds/max_holds` (forbid schemas: consumers first); occupancy-at-grant + `prompt_n`/`cache_n` in the gateway lane. **Fold the gateway part into 6.2 if 6.2 hasn't shipped.** | 6.2 (or merged into it) | orion-hub + orion-llm-gateway first, then orion-gpu-pool; substrate-runtime before gateway for lane fields |
| *(checkpoint)* | 7 days of 7.3 on gpu2: acceptance 4-8. No code. | 7.3, 7.4 | none |
| **7.5** | `feat(llamacpp-host)`: `atlas-agent` (gpu1) to the Bonsai profile via `ATLAS_AGENT_PROFILE_NAME`/image; `agent: max_holds: 2` per D1. | checkpoint pass | llamacpp-host on circe (gpu1 restart, a manual compose up); `gpu_pool.yaml` |
| **7.6** | `chore`: retire `services/orion-llamacpp-bonsai-host` (image now lives in llamacpp-host); keep the Q4 agent-flex profile as the rollback entry. | 7.5 stable | none |

No PR in this sequence adds a database migration. Schema and contract changes:

- **7.3:** a new `max_holds` key in `config/gpu_pool.yaml`, parsed in `orion/gpu_pool/config.py` and
  validated by `scripts/check_gpu_pool_config.py`.
- **7.4:** slots and ctx go into the `discovery_confirmed` event's `detail`. `GpuPoolEventV1.detail`
  is a free dict (`orion/schemas/gpu_pool.py:163`), so that needs no schema change.
- **7.4:** `holds`/`max_holds` per role is a real schema change. `DiscoveredRoleV1` and
  `GpuPoolStateV1` are `extra="forbid"` (`orion/schemas/gpu_pool.py:167, 232`), so it needs:
  - an optional field;
  - a registry check;
  - consumer-first deploy: every reader of `orion:gpu_pool:state` (Hub, gateway) before the pool.

## Juniper's answers (2026-10-01, "hit it" = take the recommendations)

1. **Keep one slot free for one-off calls?** Decide after 7.1's measurements. 7.3 makes it a setting either way.
2. **If the #27148 leak reproduces**, turn the idle-slot RAM prompt cache off on every multi-slot lane, metacog/fast included. Privacy over cache reuse.
3. **The #2434 gpu0 default** merged with #2434. It is fixed by 7.0 (Bonsai never on gpu0; gpu1 and gpu2 only).
4. **6.2's 48 h checkpoint finishes before 7.2 changes gpu2's model or slots.**

Also decided: HTTP passthrough calls stay out of `inference_failure_pressure` (6.2 ships record-only).

## Missing questions (Juniper only)

1. **D1:** two holds per agent card (runs first), or keep one slot always free for one-off calls
   (FCC/system first)?
2. **D2:** if the bleed canary reproduces, is losing some prompt-cache reuse acceptable, to turn
   the idle-slot RAM cache off on every multi-slot lane, metacog/fast included?
3. **7.0:** fix #2434's gpu0 default inside #2434 before merge (recommended), or merge and follow up?
4. **Order vs stage 6:** finish 6.2's 48 h checkpoint before 7.2 changes a role's model and slots
   (recommended), or restart the checkpoint after?

## #27148 probe on metacog/fast (2026-10-01)

**Did not reproduce.** 43 synthetic requests against circe metacog (:8012) and fast (:8013): 0 cross-conversation codewords, and server cache reuse (`timings.cache_n`) always equalled the true shared prefix (2,243 / 2,242 tokens in the shared-opening test).

- Setup tested: b10398, Qwen3-8B (dense), `--parallel 4`, 4,096 tokens per slot, default 8 GiB RAM cache with idle-slot publishing on, `kv_unified=false`. This is the same config class as the upstream report.
- The risky path did run: LRU slot picks over slots still holding unrelated conversations, and truly simultaneous pairs on metacog.
- **This does not clear the multi-slot targets.** Upstream reproduced the bug on Qwen3.6-35B-A3B, a hybrid model whose running state can't be partly rolled back. The dense 8B always truncated cleanly. So the 7.1 canary must run on the exact models stage 7 makes multi-slot (Bonsai / the 27B on gpu2), with prompts at ≥4.5K tokens and tool-call turns.
- Limits: small sample; prompts ≤3.2K tokens (slot size); no tool turns.
- Mitigation if it ever reproduces: `--cache-ram 0 --no-cache-idle-slots`. This needs two new profile fields plus `append_flag` lines in `services/orion-llamacpp-host/app/main.py`, because there is no extra-args passthrough today.
- Raw evidence: `/tmp/leak-probe-27148/` on athena.
