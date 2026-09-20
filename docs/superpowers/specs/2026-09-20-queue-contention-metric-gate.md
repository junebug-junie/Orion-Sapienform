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
