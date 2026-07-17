# Field-digester decay/injection-interval mismatch: hold-by-default fix

Status: quick spec, implement same-session. User-approved fix direction (discussed and agreed
in a brainstorming session on `orion/autonomy/drives_and_autonomy_retrospective.md`, explicit
go-ahead given 2026-07-17).

## Current architecture

`services/orion-field-digester` runs a `perturb -> decay -> diffuse -> suppress` tick
(`app/tensor/update_rules.py:run_digestion_tick`, called every `RECEIPT_POLL_INTERVAL_SEC=2.0`
seconds from `app/worker.py`).

- `apply_perturbations()` (`app/digestion/perturbation.py`) applies a fresh biometrics/other
  reading via `mode="replace"`: `node_vec[channel] = max(0.0, min(1.0, p.intensity))` — a full
  overwrite, whenever new data lands. For biometrics-sourced channels this is roughly every
  15-30s (`orion-biometrics`' `CLUSTER_PUBLISH_INTERVAL=15` / `TELEMETRY_INTERVAL=30`).
- `apply_decay()` (`app/digestion/decay.py`) multiplies every channel in `NODE_DECAY_CHANNELS`
  (19 channels: hardware/biometrics — `staleness`, `cpu_pressure`, `memory_pressure`,
  `gpu_pressure`, `thermal_pressure`, `disk_pressure`; execution — `execution_load`,
  `execution_friction`, `reasoning_load`, `failure_pressure`,
  `egress_confidence_deficit`, `repair_pressure`, `conversation_load`; transport —
  `transport_pressure`, `contract_pressure`, `catalog_drift_pressure`,
  `observer_failure_pressure`, `reliability_pressure`, `field_coherence_warning`,
  `prediction_error`) by `BIOMETRICS_FIELD_DECAY_RATE=0.92` **every single 2s tick,
  unconditionally** — whether or not fresh data arrived that tick.
- `FieldStateV1.node_vectors: dict[str, dict[str, float]]` (`orion/schemas/field_state.py`) has
  no per-channel timestamp today. There is a *separate*, unrelated recency mechanism
  (`recent_perturbations` / `recent_perturbation_at`, fixed 2026-07-16) that tracks a
  60s-windowed list of perturbation *labels* for `self_state.recent_perturbation_count` — it is
  not per-(node, channel) and does not gate decay.

## The mechanism (confirmed root cause, `orion/autonomy/drives_and_autonomy_retrospective.md`
§5b)

`0.92^7 ≈ 0.56` — a channel loses ~44% of its value across the ~15s gap between two real
biometrics publishes (7-8 ticks at 2s each), then gets snapped straight back up to the current
real measurement on the next publish, with zero memory of the decayed trajectory in between.
This produces a mechanical sawtooth on every channel in `NODE_DECAY_CHANNELS`, **regardless of
whether the real underlying value is stable or bursty** — it is an artifact of
decay-every-tick-unconditionally plus reset-via-full-replace, not a reflection of genuine
volatility. It propagates into `self_state.coherence` / `agency_readiness` / `uncertainty`
(`orion/self_state/scoring.py`) and drives a large share of `tension.distress.v1` volume
(392/hr, the single largest tension-mint source as of 2026-07-17) into the drive economy
downstream.

The sampling interval itself (15-30s) is not the problem — it's a legitimate resource
tradeoff. The problem is that "no fresh data this tick" is currently treated as "trend toward
zero," when the honest interpretation is "hold the last known value; we simply haven't
re-measured yet."

## Proposed fix

**Hold by default, decay only once actually stale.** Track, per `(node_id, channel)`, the
timestamp of the last real write from `apply_perturbations()`. In `apply_decay()`, for every
channel in `NODE_DECAY_CHANNELS`, only apply this tick's multiplicative decay if the elapsed
time since that channel's last recorded update exceeds a staleness threshold. Otherwise, hold
the value unchanged this tick.

Concretely:

1. **New field** on `FieldStateV1` (`orion/schemas/field_state.py`): `node_vector_updated_at:
   dict[str, dict[str, datetime]] = Field(default_factory=dict)`, mirroring `node_vectors`'
   `node_id -> channel -> value` shape (`node_id -> channel -> last_updated_at`). Backward
   compatible the same way `recent_perturbation_at` was (2026-07-16 precedent): a FieldStateV1
   persisted before this fix loads with an empty dict, so every existing channel is "unknown
   freshness" until its next real perturbation.

2. **`apply_perturbations()`** (`app/digestion/perturbation.py`): whenever a channel is
   actually written (all three branches — `replace`, `availability`-min, additive-clamp),
   also record `state.node_vector_updated_at.setdefault(p.node_id, {})[p.channel] = ts` (the
   same `ts` already computed at the top of the function — `now` if passed, else
   `state.generated_at`). Do this unconditionally on every write, not just `replace`, so a
   channel that only ever receives additive/min perturbations still gets tracked.

3. **`apply_decay()`** (`app/digestion/decay.py`): change signature to
   `apply_decay(state: FieldStateV1, *, decay_rate: float, now: datetime,
   staleness_threshold_sec: float) -> None`. For each `(node_id, ch)` in `NODE_DECAY_CHANNELS`
   present in a node's vector: look up `state.node_vector_updated_at.get(node_id, {}).get(ch)`.
     - If **missing** (channel never perturbed, or persisted from before this fix): apply
       decay as today (safe default — matches legacy behavior, and for the common
       reconcile-seeded-0.0 case decaying 0.0 is a no-op anyway).
     - If **present** and `(now - last_updated_at).total_seconds() < staleness_threshold_sec`:
       **skip decay this tick** — hold the value.
     - If **present** and elapsed time `>= staleness_threshold_sec`: apply this tick's decay
       factor as today.
   Leave the existing `capability_vectors` / `CAPABILITY_DECAY_CHANNELS` loop and its
   `available_capacity` recompute completely untouched (decay.py's own comment already flags
   this loop as currently-dead-under-the-live-diffusion-model and says not to assume anything
   about it without checking `apply_diffusion()` first — out of scope here).

4. **Threading `now`**: `app/tensor/update_rules.py::run_digestion_tick()` already has access to
   `state` before calling `apply_decay`; pass `now=state.generated_at` (matching
   `apply_perturbations`' own default source of truth, and `worker.py`'s existing
   `state.generated_at = now` assignment immediately before the tick — this keeps
   `test_field_deterministic_replay`-style determinism: no wall-clock `datetime.now()` call
   introduced anywhere in this fix).

5. **New setting**: `field_decay_staleness_threshold_sec: float = Field(90.0, alias=
   "FIELD_DECAY_STALENESS_THRESHOLD_SEC")` in `app/settings.py`, added to `.env_example` with a
   comment explaining the choice (≈3x `TELEMETRY_INTERVAL` / 6x `CLUSTER_PUBLISH_INTERVAL`,
   real margin for jitter/backlog before a channel is treated as genuinely stale). Threaded
   into `worker.py`'s `run_digestion_tick(...)` call alongside the existing `decay_rate=`.

**Why one global threshold, not per-channel:** `NODE_DECAY_CHANNELS` spans biometrics
(15-30s natural cadence) and execution/transport channels (potentially different natural
cadences). A single conservative hold window is still strictly better than today's
unconditional-every-2s decay for *every* channel in the set — it cannot make staleness
detection worse, only more honest. A per-channel-category threshold map is a plausible future
refinement but is exactly the kind of extra knob this task should not invent without evidence
it's needed; ride the existing single-`decay_rate` seam instead.

## Non-goals

- **Not** touching `DriveEngine`'s fold-batch clamp collapse (`orion/spark/concept_induction/
  drives.py:70-91`) — a separate, still-open, still-unpatched finding (retrospective §5b/§6
  item 5). Explicitly deferred pending live data from *this* fix, per agreed sequencing: ship
  this, measure whether fold-batch volume drops enough on its own, decide on the DriveEngine
  fix from evidence rather than building both preemptively.
- Not touching `capability_vectors` / `CAPABILITY_DECAY_CHANNELS` (see point 3 above).
- Not introducing per-channel-category staleness thresholds (see rationale above).
- Not changing `RECEIPT_POLL_INTERVAL_SEC`, `BIOMETRICS_FIELD_DECAY_RATE`, or any
  biometrics-side publish cadence.

## Acceptance checks

1. Regression test reproducing **today's** sawtooth is characterized (either as a documented
   before/after comparison or a comment referencing the exact retrospective numbers): a
   channel receiving `apply_perturbations(replace, intensity=0.8)` once, then `apply_decay`
   called repeatedly at ~2s ticks with no further perturbation, held its value flat for ticks
   within the staleness window and only began decaying once genuinely stale.
2. A perturbation applied and immediately followed by `apply_decay` in the *same tick*
   (mirroring `run_digestion_tick`'s `perturb -> decay` order) does **not** decay the
   freshly-set value at all (elapsed time is 0s, well under any positive threshold).
3. A channel with no recorded `node_vector_updated_at` entry (simulating either a
   never-perturbed channel or a pre-fix persisted state) still decays every tick, unchanged
   from today's behavior — proves the migration path is safe.
4. A channel goes stale (no perturbation for longer than `staleness_threshold_sec`) and decay
   resumes applying normally from that point.
5. Full `services/orion-field-digester/tests` suite passes, including
   `test_perturbation_recency.py`, `test_field_node_biometrics_perturbations.py`,
   `test_field_chat_perturbations.py`, `test_worker.py` (decay/perturbation call-signature
   changes must not break existing callers/tests).
6. `.env_example` / `settings.py` parity: new key documented, no secrets, default matches the
   value used in code.
