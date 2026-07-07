# Homeostatic Drives — real tensions from the signal substrate (Implementation Plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make Drives mean something. Replace the tick-cadence pressure artifact with a cadence-invariant leaky integrator that rests at zero, and feed real deviation-triggered tensions from the existing `orion:signals:*` bus + failure + equilibrium traffic — with no keyword matching and no signal flood. **Ship enabled** (both flags default `true`), proven by live runtime truth (differentiated non-pinned pressures; the 55/s `scene_state` flood mints zero tensions).

**Spec:** `docs/superpowers/specs/2026-07-07-homeostatic-drives-real-tensions-design.md` (local, gitignored — read it first; it carries the full model, math, and ground truth).

**Owner directive:** Juniper has authorized flags ON (this is the proposal-mode approval). Flags default `true`; the sequence still builds → tests → evals → then live-enables + verifies, so "on" is proven, not assumed.

**Architecture (one paragraph):** A deviation gate (per-`signal_kind.dimension` EWMA baseline) converts non-stub `OrionSignalV1` + normalized failure/health events into `TensionEventV1` impulses via one structural YAML map (typed `signal_kind` → `drive_impacts`, zero free-text). A rate-limiter/dedup guards against storms; the existing `is_stub_signal` and `compute_tick_attribution` are reused unchanged. `DriveEngine.update` is rewritten as a wall-clock leaky integrator `p ← clamp01(p·e^(−Δt/τ) + impulse·(1−base))` (rests at 0, cadence-invariant, no fixed point). Decisions already flow from attribution, so `dominant_drive`/goals come alive automatically.

**Tech Stack:** Python 3.12, Pydantic v2, pytest, YAML, Redis bus (`OrionBusAsync`), stdlib `statistics`/`math`. No new deps.

**Worktree (this plan is being implemented here):**
```bash
# already created:
#   git worktree add ../Orion-Sapienform-homeostatic-drives -b feat/homeostatic-drives-real-tensions origin/main
cd /mnt/scripts/Orion-Sapienform-homeostatic-drives
```

---

## File map

| File | Action | Responsibility |
|------|--------|----------------|
| `orion/spark/concept_induction/drives.py` | Modify | Leaky-integrator `update`; remove `_soft_saturate` fixed point; wall-clock decay |
| `orion/autonomy/deviation_gate.py` | Create | EWMA baseline (μ,σ) per `signal_kind.dimension`; z-score; impulse |
| `orion/autonomy/signal_drive_map.py` | Create | Load + validate `signal_drive_map.yaml` |
| `config/autonomy/signal_drive_map.yaml` | Create | Structural typed `signal_kind`→`drive_impacts` contract |
| `orion/autonomy/signal_tension.py` | Create | `signal_to_tension` + failure/health normalizers |
| `orion/autonomy/tension_ratelimit.py` | Create | Per-`(organ,kind)` caps, dedup, source precedence |
| `orion/spark/concept_induction/bus_worker.py` | Modify | Subscribe new channels; wire gate→ratelimit→attribution→engine |
| `orion/core/schemas/drives.py` | Modify | Register `tension.signal.v1` / `.failure.v1` / `.health.v1` kinds |
| `orion/schemas/registry.py` | Modify | Register the three tension kinds |
| `orion/bus/channels.yaml` | Modify | Document new consumed channels |
| `orion/spark/concept_induction/settings.py` | Modify | New settings, flags **default true** |
| `services/orion-spark-concept-induction/.env_example` | Modify | New keys, flags **=true** |
| `services/orion-spark-concept-induction/.env` | Modify (sync) | `python scripts/sync_local_env_from_example.py` |
| `orion/autonomy/evals/run_homeostatic_drives_eval.py` | Create | Replay stream; assert liveness + flood→0 + rest-at-zero |
| tests under `orion/autonomy/tests/`, `orion/spark/concept_induction/tests/` | Create | Per §Tests in spec |

**Reused unchanged:** `orion/signals/stub_detection.py::is_stub_signal`, `orion/spark/concept_induction/drive_attribution.py`.

---

## Task 1 — Leaky-integrator pressure math (behavior-changing; own flag)

**Files:** `orion/spark/concept_induction/drives.py`, `orion/spark/concept_induction/tests/test_drives_leaky.py`, `settings.py`

- [ ] **Step 1:** Add `ORION_DRIVE_LEAKY_MATH_ENABLED` (default `true`). When true, `DriveEngine.update` uses `decay = exp(−Δt_wall/τ)`, `base = prev·decay`, `pressure = clamp01(base + impulse·(1−base))`; when false, the legacy `soft_saturate` path (preserve for rollback). Keep hysteresis (0.62/0.42) on the new pressure.
- [ ] **Step 2:** Ensure `Δt` is real wall-seconds from `previous_ts` (already `drive_state.updated_at`).
- [ ] **Step 3 (tests):** (a) rest-at-zero: any pressure + N no-impulse ticks → `< 1e-3`; (b) cadence-invariance: 100 ticks/s vs 1 tick/100s, same impulse wall-times → equal pressure within ε; (c) no-uniform-pin: distinct per-drive impulses → distinct pressures; (d) legacy flag path unchanged.

**Verify:** `pytest orion/spark/concept_induction/tests/test_drives_leaky.py -q` green; rest-at-zero + cadence-invariance proven.

## Task 2 — Deviation gate (EWMA baseline → impulse)

**Files:** `orion/autonomy/deviation_gate.py`, `orion/autonomy/tests/test_deviation_gate.py`

- [ ] **Step 1:** Pure `DeviationGate` holding `{(signal_kind,dimension): (μ,σ,warmup_count)}`; `observe(kind,dim,x,confidence,worse) -> impulse`. `z=(x−μ)/max(σ,σ_floor)`; `deviation=relu(direction·z − z_threshold)`; `impulse=k·deviation·confidence`. Update EWMA after computing (so first observations warm up, don't impulse).
- [ ] **Step 2 (tests):** steady input after warm-up → 0 impulse; a real drop/rise past threshold → sized impulse; cold-start warm-up mints nothing; `σ_floor` prevents blowup.

**Verify:** `pytest orion/autonomy/tests/test_deviation_gate.py -q` green; steady→0, deviation→impulse.

## Task 3 — Structural signal→drive map (starve the keyword cathedral)

**Files:** `config/autonomy/signal_drive_map.yaml`, `orion/autonomy/signal_drive_map.py`, `orion/autonomy/tests/test_signal_drive_map.py`

- [ ] **Step 1:** Author the YAML (spec §The unified model.3): `biometrics_state` (homeostasis↓/strain↑/thermal↑→capability,continuity), `mesh_health` (level↓→capability,continuity), `spark_signal` (coherence↓→coherence; valence↓→relational; novelty↑→predictive), `failure_event` (severity↑→capability,coherence).
- [ ] **Step 2:** Loader + validation: every drive ∈ `DRIVE_KEYS`, every `worse ∈ {up,down}`. Unmapped kind → returns nothing.
- [ ] **Step 3 (tests):** map validates; unmapped `signal_kind` → no mapping; **assert no code path reads signal text** (grep-guard test on the module).

**Verify:** `pytest orion/autonomy/tests/test_signal_drive_map.py -q` green.

## Task 4 — Signal→tension adapter (OrionSignal + failure + health)

**Files:** `orion/autonomy/signal_tension.py`, `orion/autonomy/tests/test_signal_tension.py`

- [ ] **Step 1:** `signal_to_tension(sig, gate, map) -> TensionEventV1 | None`: drop if `is_stub_signal(sig)`; for each mapped dimension, gate→impulse; sum impulses per drive → `drive_impacts`; `magnitude=Σimpulse`; `kind="tension.signal.v1"`. Degrade to `None` (never raise) on missing dims/unmapped kind.
- [ ] **Step 2:** `failure_to_signal(event) -> synthetic failure_event` (severity from event) and `equilibrium_to_deviation(snapshot, prev)` (edge-triggered `ok→down` only). Both feed the same `signal_to_tension` path with kinds `tension.failure.v1` / `tension.health.v1`.
- [ ] **Step 3 (tests):** stub → None; steady biometrics → None; homeostasis 0.82→0.55 → capability+continuity tension; `system:error` → failure tension; equilibrium `ok→down` → one tension, `down→down` → none.

**Verify:** `pytest orion/autonomy/tests/test_signal_tension.py -q` green.

## Task 5 — Rate-limit / dedup / precedence

**Files:** `orion/autonomy/tension_ratelimit.py`, `orion/autonomy/tests/test_tension_ratelimit.py`

- [ ] **Step 1:** Pure `bounded_tensions(candidates, now, state) -> kept`: per-`(organ_id,signal_kind)` cap `N`/`window` (default 3/60s), dedup by signature (reuse `recent_event_seen` pattern), source precedence (feedback-frame beats signal for same underlying event). Bounded state (cap map size).
- [ ] **Step 2 (tests):** 100-event storm/1s → ≤ cap; precedence drops the signal dup; state map stays bounded.

**Verify:** `pytest orion/autonomy/tests/test_tension_ratelimit.py -q` green.

## Task 6 — Bus wiring

**Files:** `orion/spark/concept_induction/bus_worker.py`, `orion/core/schemas/drives.py`, `orion/schemas/registry.py`, `orion/bus/channels.yaml`

- [ ] **Step 1:** Subscribe `orion:signals:*`, `orion:system:error`, `orion:grammar:event` (filter `exec_step_failed`), `orion:rdf:error`, `orion:vision:edge:error`, `orion:equilibrium:snapshot`.
- [ ] **Step 2:** Per event → adapter → `bounded_tensions` → merge into the tick's tension list → existing `compute_tick_attribution` → `DriveEngine.update`. Gate the whole path on `ORION_HOMEOSTATIC_DRIVES_ENABLED`.
- [ ] **Step 3:** Register the three tension kinds in schema + registry; document channels in `channels.yaml`.
- [ ] **Step 4 (tests):** worker consumes a captured signal → mints a tension → attribution non-zero → `dominant_drive` non-None (loop-liveness test).

**Verify:** `pytest orion/spark/concept_induction/tests -q` green (incl. liveness test); `python scripts/check_schema_registry.py`; `python scripts/check_bus_channels.py`.

## Task 7 — Env flags ON + sync

**Files:** `settings.py`, `services/orion-spark-concept-induction/.env_example`, `.env`

- [ ] **Step 1:** Add to `.env_example` with flags **enabled**:
  ```
  ORION_HOMEOSTATIC_DRIVES_ENABLED=true
  ORION_DRIVE_LEAKY_MATH_ENABLED=true
  DRIVE_DECAY_TAU_SEC=1800
  DEVIATION_EWMA_ALPHA=0.1
  DEVIATION_Z_THRESHOLD=1.5
  DEVIATION_SIGMA_FLOOR=0.02
  SIGNAL_TENSION_IMPULSE_K=0.25
  SIGNAL_TENSION_CAP_PER_WINDOW=3
  SIGNAL_TENSION_WINDOW_SEC=60
  ```
- [ ] **Step 2:** `settings.py` defaults mirror (`enabled=true`).
- [ ] **Step 3:** `python scripts/sync_local_env_from_example.py` — sync local `.env`. Report any skipped keys.
- [ ] **Step 4:** `python scripts/check_env_template_parity.py`.

**Verify:** parity check passes; `git check-ignore services/*/.env` confirms `.env` ignored; both flags present and `=true` in `.env`.

## Task 8 — Eval harness

**Files:** `orion/autonomy/evals/run_homeostatic_drives_eval.py`

- [ ] **Step 1:** Replay a captured/synthetic 1-hour stream (scene_state flood + a biometrics strain episode + an injected failure). Assert (a) tension rate rises from ~0.06% into a target band and tracks real deviations; (b) pressures differentiate + rest at zero in quiet spans; (c) scene_state flood contributes 0; (d) `dominant_drive` distribution reflects injected events (not alphabetical "autonomy", not constant None).
- [ ] **Step 2:** Report tension rate, per-drive pressure trajectories, suppression breakdown (stub|steady|cap|precedence).

**Verify:** `python orion/autonomy/evals/run_homeostatic_drives_eval.py` prints the four assertions PASS + the report.

## Task 9 — Review gate

- [ ] Run `/code-review` (high) in a subagent over the diff; fix material findings; re-run affected tests.
- [ ] `make agent-check SERVICE=orion-spark-concept-induction` (env parity, schema, channels, tests).

## Task 10 — Live enable + runtime-truth verification (the point)

**Files:** none (ops)

- [ ] **Step 1:** Print exact restart for Juniper (do NOT run sudo yourself):
  ```bash
  docker compose --env-file .env --env-file services/orion-spark-concept-induction/.env \
    -f services/orion-spark-concept-induction/docker-compose.yml up -d --build
  ```
- [ ] **Step 2:** After restart, tap live for 60s and assert the acceptance gates:
  - `drive_pressure_gauge` (or a live drive:audit tap) shows **differentiated, non-pinned** pressures that move with biometrics/health/failures and decay toward ~0 in a quiet window (NOT all six pinned at 0.7309).
  - The 55/s `scene_state` flood mints **0** tensions (suppression counter).
  - `dominant_drive` reflects real events (not constant "autonomy"/None).
- [ ] **Step 3:** Record the live before/after in the PR report (pinned-0.731 → differentiated).

**Verify:** live tap shows differentiated pressures + zero flood tensions + event-driven dominant.

---

## Rollout / rollback
- Two flags, independent. If the math swap misbehaves on `policy_act` predictive gating, set `ORION_DRIVE_LEAKY_MATH_ENABLED=false` (source adapter keeps running on legacy math — safe, decisions are attribution-based). Full disable: both `false` → byte-identical prior behavior.

## Restart required
```bash
docker compose --env-file .env --env-file services/orion-spark-concept-induction/.env \
  -f services/orion-spark-concept-induction/docker-compose.yml up -d --build
```

## Acceptance checks (whole plan)
- [ ] Live: pressures differentiated + non-pinned + rest toward zero in quiet spans.
- [ ] `scene_state` 55/s flood → 0 tensions.
- [ ] Real biometrics/health/failure deviation → one correctly-mapped tension each.
- [ ] `dominant_drive` reflects real events.
- [ ] Rest-at-zero + cadence-invariance unit tests pass; eval PASS.
- [ ] Both flags `=true` in `.env_example` + synced `.env`; parity/schema/channel checks green.
- [ ] Review ran; material findings fixed.

## Non-goals
- No 7th self-preservation drive (somatic → capability/continuity; gap documented).
- No changes to `orion/signals/*` producers or `drive_attribution.py`.
- No removal of the legacy `autonomy/reducer.py` keyword system (separate cleanup) — but do not extend it.
- No LLM in the tension/pressure path.
