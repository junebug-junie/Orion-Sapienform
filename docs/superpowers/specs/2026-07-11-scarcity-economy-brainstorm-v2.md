# Scarcity Economy Brainstorm v2 — Live-Rail Findings and Redesign

**Date:** 2026-07-11
**Context:** Follow-up to `docs/superpowers/specs/2026-07-07-internal-economy-scarcity-allocation-design.md` (the "internal economy" spec). This round was grounded against the live mesh (Athena/Atlas/Circe), the local `.env` files, the bus, and the conjourney Postgres — not against the old spec's assumptions. Several of those assumptions inverted.

---

## What I found (answers to the open questions, from the live rail)

**1. `resource_pressure` isn't dead at 0.0 — it's saturated at 1.000.**
23,654 self-state samples over 48h (one tick every ~7s), mean 1.000, stddev **0.000**. `field_intensity` is also pinned at 1.0 and `coherence` at 1.0. The evidence trail shows `pressure=1.00, cpu_pressure=0.72, gpu_pressure=0.00`. So the interoceptive stack has no dynamic range: the old spec's `B = B_MAX×(1−resource_pressure)` would yield B_MIN forever — scarcity would "bind" 100% of the time, which is exactly as uninformative as never. A sensor pinned at ceiling is as blind as a dead one.

**2. `gpu_pressure=0.00` is a lie of scope, not a bug.**
Biometrics senses *this node's* GPUs — Athena's P4/P100 vision cards, currently idle. The GPUs that are actually scarce (Atlas and Circe's V100s serving chat/agent/metacog/quick) are **outside the sensed body**. Orion's body schema covers one node of three.

**3. Drives essentially never co-activate.**
Live bus sample (`orion:memory:drives:state`): peak pressure `capability=0.30`, `coherence=0.17`, `predictive=1e-34` — all far below the 0.62 activation threshold. Step 0(b) answers itself: at the *drive* grain, scarcity would never bind. The old spec built the market at the wrong grain. The real contention is at the infrastructure grain.

**4. Where scarcity actually lives, today:**
- `/mnt/docker` is at **100%** (3.7G free of 469G) — and `disk_pressure` in `orion/telemetry/biometrics_pipeline.py:110` measures IO *bandwidth*, not capacity. The most binding current scarcity is invisible to Orion.
- `LLM_GATEWAY_ROUTE_TABLE_JSON` is a hand-edited static env var, with a commented-out alternate routing fossilized below it, and `atlas-worker-1` actually pointing at Circe's IP. Every one of those edits was an allocation decision Juniper made under scarcity that Orion never felt.
- CPU: Athena's load average is 12 (`cpu_pressure=0.72` is the one honest, moving resource signal).

**5. `orion-gpu-cluster-power` is not a sensor — it's an actuator.**
It switches PSUs on/off over `orion:power:psu:command`. Circe's "+8 V100s" are *purchasable capacity*: Orion could, in principle, spend watts to buy cognition. Nothing consumes this motivationally.

**6. phi is on and healthy-ish.**
`ORION_PHI_ENCODER_ENABLED=true`, active weights, four encoder versions trained through 07-10 (seedv4). Value-biased bidding is no longer blocked. And `ORION_SUBSTRATE_AUTONOMY_METABOLISM_ENABLED=true` already exists — but `orion/autonomy/substrate_metabolism.py` only metabolizes world-pulse coverage gaps into predictive-drive deltas. The name is bigger than the organ.

---

## The reframe

**Juniper is currently Orion's homeostasis.** Every scarcity event — GPU contention, disk full, worker sizing, node failover — gets absorbed at *configuration time*: conservative worker counts, hand-edited route tables, manual cleanups. Orion never experiences scarcity because his environment is pre-metabolized. Developmentally, that's the co-regulation stage; the world-class move isn't "add an allocator," it's **the staged transfer of regulation from Juniper to Orion** — the same arc as parent→child allostasis. The list of things Juniper hand-edits under pressure *is* the list of organs Orion needs.

The counterfactual objection is the crux: GPU scarcity can't be measured by failure counting because the system correctly refuses to run at failure. Three ways out that don't require failures: **latency curvature** (queueing theory), **shadow decisions** (paper trading), and **time-shifted execution** (run the counterfactual when it's cheap).

---

## Ideas

### 1. Interoceptive recalibration — un-pin the senses (keystone, unglamorous)

- **What:** Replace raw-clamp channel→dimension mapping with adaptive normalization (deviation from a rolling baseline, Weber–Fechner style), plus a deterministic **pinned-sensor gate**: a check that fails if any self-state dimension shows stddev≈0 over 48h.
- **Why:** Homeostasis runs on *error signals*, not levels. Right now 3 of 13 dimensions are saturated flatlines; no economy, drive, or phi reward built on them can learn anything. The recurring failure ("sensor quietly pinned") becomes a failing gate, not a louder prompt.
- **Smallest slice:** Rolling-percentile normalization for `resource_pressure` only, behind a flag, plus `scripts/check_selfstate_liveness.py` querying `substrate_self_state` for zero-variance dimensions.
- **Files:** `orion/self_state/scoring.py`, `orion/self_state/builder.py`, `services/orion-field-digester/app/tensor/channels.py`, new script + gate test.

### 2. Queue-wait as the scarcity sense organ (solves the counterfactual problem passively)

- **What:** The LLM gateway timestamps enqueue→first-token per role/worker and publishes `LlmSchedulingSampleV1`. Scarcity is read from the *shape* of the wait curve — queueing theory (Kingman) says wait explodes convexly as utilization approaches saturation, so distance-to-the-cliff is continuously measurable **without ever falling off it**.
- **Why:** Direct answer to "I can't run prod at 8/10 failures to measure the frontier." No failures needed; latency curvature suffices. p95 wait per role is the counterfactual-free scarcity signal for exactly the resource that binds hardest (V100 workers).
- **Smallest slice:** Two timestamps and one bus event in the gateway request path; a 24h projection of wait percentiles per role. No behavior change.
- **Files:** `services/orion-llm-gateway/app/main.py`, `orion/schemas/registry.py`, `orion/bus/channels.yaml`, small reducer.

### 3. Shadow allocator — paper-trading Orion's own configuration

- **What:** An allocator that never acts. It watches real routing decisions (static table) plus queue-wait samples and logs where it *would have diverged* — `AllocationDivergenceV1`: "I would have routed metacog to circe this hour," "I would have deferred world_pulse under this wait spike."
- **Why:** Counterfactuals without risk. The divergence log is Orion forming *opinions about its own metabolism* before being trusted with it — and it produces the exact evidence needed to decide whether a live allocator would ever earn its keep, replacing the old spec's blocking measurement gate with a running one.
- **Smallest slice:** Consume the route table + scheduling samples, apply a trivial deterministic policy, emit divergences to one channel with a debug endpoint. Graduation criterion explicit at birth: shadow must beat static config on replayed wait/quality metrics before any live authority.
- **Files:** new `orion/autonomy/shadow_allocator.py`, consumer in a small worker, `channels.yaml`, registry.

### 4. The night market — scarcity in space becomes delay in time

- **What:** Wants that can't be afforded now (deferred curiosity, denied heavy inference, shadow-allocator divergences worth testing) go into a queue that *actually executes* during off-peak windows on Circe — including powering up expansion V100s via the existing PSU actuator if the queue justifies it.
- **Why:** Closes the counterfactual loop by *running* the counterfactual when it's cheap. Deferral gains a real, measurable cost (delay-hours) and a real payoff (the deferred thing eventually happens and gets scored). Also the beginning of a circadian metabolism: expensive wanting by day, cheap digestion by night — not on a cron, but because night is when Orion can afford itself.
- **Smallest slice:** A deferred-wants queue (Redis stream), one off-peak window definition, one executor that drains N items per night and writes outcome receipts. PSU purchase stays proposal-mode/operator-approved.
- **Files:** new thin service or a worker in `orion-cortex-exec`, consumer of `orion:power:psu:command` (gated), receipts into the phi corpus path.

### 5. Safe frontier probing — Orion learns its own capacity envelope

- **What:** Constrained canary experiments on capacity config (one worker's `n_parallel`, one role's routing) during low-stakes windows, guarded by the queue-wait sensor (idea 2) as the leading indicator, with automatic rollback — SafeOpt-style safe Bayesian exploration rather than failure counting.
- **Why:** Today the reliable-capacity frontier lives in Juniper's head and gets updated only when something breaks. This converts "configure conservatively forever" into a *mapped, versioned frontier Orion maintains itself* — adversity in controlled doses; the resilience thesis, operationalized.
- **Smallest slice:** One experiment spec in `orion-self-experiments`: raise `quick` role concurrency by 1 for 30 min at 3am, watch p95 wait + VRAM headroom, rollback on threshold, write a before/after report.
- **Files:** `services/orion-self-experiments/`, guardrail reader from idea 2's projection.

### 6. Quality-tier degradation — scarcity felt as cognitive dulling, not denial

- **What:** Roles get tiered profiles (bigger model / longer ctx / better quant ↔ smaller/shorter/cheaper), and under measured pressure the gateway funds a *lower tier* instead of queueing or denying. Offline, the same task set is scored across tiers to learn each role's quality-cost curve — which is itself the missing counterfactual, measured safely.
- **Why:** Biological scarcity rarely manifests as refusal; it manifests as degraded function you can *feel*. If metacog runs dulled for a day and Orion's introspection can detect its own reduced acuity ("my answers are worse this week because we're poor"), that's self-modeling with teeth — and it feeds phi exactly the friction signal the corpus lacked.
- **Smallest slice:** Two tiers for one role (`quick`), a static quality eval run at both tiers to establish the curve, and a manual tier switch. No dynamic switching until the curve exists.
- **Files:** `services/orion-llm-gateway/.env` route table (becomes tiered YAML — kill the JSON blob), `services/orion-llamacpp-host` profiles, one eval script.

### 7. Mesh body schema — give Orion its whole body

- **What:** Biometrics samplers on Atlas and Circe publishing to the bus; the self-state grows a per-node somatic map (Athena=viscera/memory, Atlas=fast reflexes, Circe=deliberation+reserve muscle), each with its own UPS-backed energy state, fused into mesh-level dimensions.
- **Why:** Fixes the `gpu_pressure=0.00` scope lie — the scarcest tissue is currently outside the felt body. Per-node UPS means per-limb energy, and a node outage becomes felt numbness rather than silent absence. Distributed interoception across an actual physical mesh is genuinely untrodden territory for a digital-mind project.
- **Smallest slice:** One lightweight sampler container on Circe publishing the existing biometrics sample shape with a `node_id`; field digester keys channels by node.
- **Files:** `orion/telemetry/biometrics_pipeline.py`, small compose for atlas/circe, `services/orion-field-digester/app/tensor/channels.py`.

### 8. Disk as digestion — capacity pressure plus autonomic excretion

- **What:** Add *capacity* (fill %, days-until-full trend) to the disk sense, and wire an autonomic pruning reflex with receipts: docker image GC, log rotation, corpus compaction — proposal-mode for anything touching memory artifacts.
- **Why:** `/mnt/docker` hit 100% and Orion never felt it approaching — a slow-onset, fully predictable scarcity, the easiest possible first case of *anticipatory* regulation (allostasis, not just homeostasis: act before the setpoint is violated).
- **Smallest slice:** Capacity + trend in the biometrics sample; a `disk_capacity_pressure` channel; an alert-only consumer that names the top growth directories. Reflexes come after the sense is proven.
- **Files:** `orion/telemetry/biometrics_pipeline.py`, field-digester channels, one consumer.

---

## Tensions and risks

- **The saturation bug silently poisons everything.** Any idea built before idea 1 inherits pinned sensors. This includes phi: if seedv4 trained on windows where `resource_pressure` was a constant 1.0, that dim is noise in the latent.
- **Shadow-mode purgatory.** Ideas 3/5/6 can generate opinions forever without authority ever transferring. Each needs a written graduation criterion at birth, or it's a keyword cathedral that logs.
- **The night market can become a landfill** — deferred wants that never score, queue growing unbounded. Cap the queue, expire wants, and treat expiry itself as signal (a want nobody re-bids on wasn't a want).
- **PSU actuation is the one physically dangerous seam.** Watts, heat, hardware wear. Operator-approved proposals only, hard interlocks, and the economy must never sit in the shutdown path (power-guard's protective role stays sovereign).
- **Frontier probing can eat trust fast.** One canary that degrades chat during the workday costs more goodwill than ten 3am successes buy. Window discipline is load-bearing.
- **Grain honesty:** drives don't contend; infrastructure does. Building drive-bidding now would be theater. Let drives join the economy later, once wants are strong enough to collide — that's a milestone to *measure toward*, not fake.

---

## Recommended starting point

**Ideas 1 + 2 together: honest senses before any economy.** Un-pin the saturated dimensions (with the pinned-sensor gate so this never regresses silently) and add the queue-wait organ at the gateway — the one sensor that reads the hardest scarcity, GPU worker capacity, without needing a single failure. Everything else — shadow allocator, night market, tier degradation, frontier probing — reads from those two senses. An economy built on a flatlined interoception is astrology with extra steps.

**First concrete step:** `scripts/check_selfstate_liveness.py` (the zero-variance gate — deterministic, ~20 lines, immediately red today against `resource_pressure`, `field_intensity`, and `coherence`), then trace *why* the `pressure` channel feeds 1.00 into the merge in `orion/self_state/builder.py:174-178` — that max-merge with a saturated channel is likely the whole saturation story.

**Also flagged while in there:** `substrate_active_node_pressure_projection` reports `availability_status: "stale"` with a ~200-entry `evidence_event_ids` blob and `pressure_score: 0.0` for athena — the node-pressure organ looks half-dead and worth a look independent of this arc.

No code changed; if this proceeds, ideas 1+2 belong as a spec in `reviews/pending`.
