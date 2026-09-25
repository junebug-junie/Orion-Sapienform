# queue_contention_score — metric quality gate

**Date:** 2026-09-20
**Parent:** `docs/superpowers/specs/2026-09-20-hire-handoff-and-queue-pressure-design.md`
**Plan Task 5:** blocking before digester wire (Task 6)

## Verdict

**PASS to implement** with all three sources retained, with documented caveats:

- `durable_demand_pending` and `gateway_waiting` show real calm and real variation.
- `world_pulse_seed_pending` is chronically elevated (stock ~121 pending) — EWMA will re-baseline around that stockpile; the score then answers “worse than *this backlog’s* recent normal,” not “is the backlog zero.” That is still useful for hire weather. Do **not** drop the source; do **not** disclose raw counts.
- Not a rebadge of `gpu_pressure`, `sustained_load_pressure`, or `cortex_exec_step_load` (different producers, different theory).

## 1. Provenance

| Source key | Producer | Live read (2026-09-20) |
| --- | --- | --- |
| `world_pulse_seed_pending` | `SELECT count(*) FROM world_pulse_read_seed WHERE status='pending'` | **121** pending (also 103 skipped, 15 done, 3 failed) |
| `durable_demand_pending` | `SELECT count(*) FROM durable_resource_demands WHERE status='pending'` | **3** pending; 48 withdrawn since 2026-09-14; 1 granted today |
| `gateway_waiting` | LLM gateway `GET /admission` → sum `upstreams[*].waiting` | **0** waiting; inflight_sum **3** across 4 upstreams |

Digester writer (Task 6): `services/orion-field-digester/app/digestion/queue_contention.py` → `FieldStateV1.queue_contention_*` fields. Must be named again in the impl PR when the file lands.

## 2. Independence

| Existing metric | Chain to queue contention | Independent? |
| --- | --- | --- |
| `gpu_pressure` | Node biometrics / strain hints → field channel | **Yes** — different sensors; may correlate under shared circe load but not a monotonic transform of queue counts |
| `sustained_load_pressure` | Field channel regime (`loaded_steady`) over a window | **Yes** — level+dispersion on field channels, not SQL/admission queue depth |
| `cortex_exec_step_load` | Cortex-exec step telemetry | **Yes** — execution step load, not seed/durable/gateway waiting |

**Cross-source correlation (open, acceptable for `max()`):** seed backlog, durable GPU waits, and gateway waiting can all rise when the agent lane is overloaded. Spec already allows `max()` over correlated sources. No requirement that the three be orthogonal — only that the *score* not be a rename of an existing pressure channel.

## 3. Theory anchor

Measures **shared agent/curiosity capacity queue contention**: how backed up the reading-seed pipeline, durable GPU lease waits, and LLM gateway admission waiting are relative to each source’s own recent EWMA baseline. Used to inform Orion’s Cursor-vs-local hire decision. Not GPU thermals, not tension change-detection, not sustained field-channel overload.

## 4. Live-data sanity

### world_pulse_seed_pending

- Snapshot: 121 pending (chronic stockpile).
- 72h create cadence (rows with `created_at` in window): only 4 hour-buckets with activity; pending_created per active hour ~5–9 — backlog is not exploding hour-to-hour, it is **not draining to zero** either.
- Calm-at-zero: **not observed** in this snapshot. After EWMA warm-up, score≈0 means “at backlog normal,” not “empty queue.”
- **Keep** — still the dominant shared-capacity backlog Juniper named; EWMA relative signal is the product.

### durable_demand_pending

- Snapshot: 3 pending; daily withdrawn churn 2–11/day over 14 days; granted appears.
- Calm: pending often near 0 historically (withdrawn dominates). **Can return to calm.** Keep.

### gateway_waiting

- Snapshot: waiting_sum **0**, inflight 3 on one upstream.
- Admitted totals large (17k+ on busiest upstream) with waiting 0 now → **can be calm.** Keep. Sparse spikes expected; floor=1 in score math avoids divide-by-near-zero.

### Degenerate-source check

None dropped. Seed is chronically high but still varies enough for relative EWMA; durable and gateway show calm. Revisit after soak if seed sub-score never leaves 0 after warm-up while hire decisions still need seed weather — then consider rate-of-change, not raw stock (separate proposal).

## 5. Existing-mechanism check

`rg` on `orion/field` + digester (2026-09-20): `gpu_pressure`, `sustained_load_pressure`, `cortex_exec_step_load` exist; **no** `queue_contention*` producer. No existing FieldState field for seed/durable/gateway queue depths. Proceed with new instrument.

## 6. Reversibility

Additive `FieldStateV1` fields + digester module + inner-state registry + metric lock (+ optional glossary). Kill = delete producer **and** registry/lock/glossary together; Hub disclosure fails open if fields absent. No Hub Redis EWMA.

## Half-life / floor (provisional pending soak)

- Floor: `1.0` (plan).
- Half-life: ~24h (plan); digester tick ~2s → alpha derived in Task 6 settings.
- Recalibrate after first soak if seed warm-up hides real spikes.

## Gate status for Task 6

**UNBLOCKED** — implement Task 6 with all three source keys:
`world_pulse_seed_pending`, `durable_demand_pending`, `gateway_waiting`.

## Re-point: `gateway_waiting` → `gpu_pool_waiting` (2026-09-24, Juniper-approved)

The LLM gateway's in-process admission ledger (`GET /admission`, the `gateway_waiting` source) is
deleted by the GPU pool gateway cutover (stage 3 of `2026-09-24-gpu-pool-design.md`). The metric is
re-pointed, not left to go silent. The gate, re-run in full:

1. **Provenance.** `gpu_pool_waiting` = `SELECT count(*) FROM gpu_pool_leases WHERE status IN
   ('queued','backlogged')` (`services/orion-field-digester/app/store.py::count_gpu_pool_waiting`).
   The rows are the pool's fenced projection, written by the single-writer runtime
   (`services/orion-gpu-pool/app/runtime.py::_row`) on every lease transition. `retry_wait` is
   excluded (cooling down after a failure, not waiting for capacity), and so are `granted` and
   `recalling` (holding a GPU).
2. **Independence.** It replaces `gateway_waiting` (same concept, the retired producer). It does
   **not** double-count `durable_demand_pending`: a durable run waits in the old broker until it is
   granted, and only then takes a pool lease for the slot it uses. Its wait is counted in one place,
   never both. When durable runs move onto the pool (stage 4), `durable_demand_pending` retires and
   its wait moves into `gpu_pool_waiting`; that is the next re-point.
3. **Theory anchor.** Unchanged from this gate: backlog of work waiting for a GPU, relative to its own
   EWMA baseline, is the queueing-pressure signal the hire decision consumes. The pool's queue is
   where that waiting now physically happens, because every LLM call leases before it runs.
4. **Live data.**
   - **The retired source was degenerate.** `substrate_field_state` at 2026-09-24 23:29Z shows
     `queue_contention_ewma.gateway_waiting = 0.0` over **163,577** observations, and it was never the
     driver in 3 days (drivers: `world_pulse_seed_pending` ×66,458, avg score 0.14;
     `durable_demand_pending` ×54,581, max 6.38).
   - **Why it was zero.** The gateway's semaphore cap was 8 per upstream, far above the real slot
     counts (chat 1, agent 1, metacog/fast 4), so nothing ever queued there. Oversubscription
     happened inside llama.cpp instead, invisible to the metric.
   - **Why the new source can move.** The pool grants against discovered real slots, so the queue
     forms exactly where the GPU is full. Its rest state is a genuine 0 (empty queue). It is not a
     decay artifact: the value is a fresh count every tick, never carried forward.
   - **UNVERIFIED live** until stage 3 is deployed: after deploy, confirm the new source rises under
     load (e.g. a metacog burst beyond 4 slots, or chat while a turn runs) and returns to 0.
5. **Existing mechanism.** This is the same metric; only one source changes. `SOURCE_KEYS` stays at
   three.
6. **Reversibility.** Cheap. The key rename starts a fresh EWMA (the first tick scores 0 by
   construction). The retired `gateway_waiting` baseline is dropped from state rather than carried
   (`score_queue_contention` keeps only `SOURCE_KEYS`). Disclosure text is updated
   (`orion/curiosity/queue_contention_disclosure.py`).
