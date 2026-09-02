# Phi v2: a real internal-state score built from live metrics

Status: DESIGN DRAFT — not implemented. Written after a full empirical audit of
the current live metric surface (substrate runtime, cocreation-signals,
heartbeat, biometrics, chat/relationship signals). No code changed yet.

**Update (2026-09-02):** this doc's live-metric audit (the "Signals confirmed
real, live, and independent" table below) was consumed by `mood_arc`'s `v4`
retrain — see `orion/mood_arc/README.md`'s v4 section — as the input feature
set for that module's existing unsupervised windowed-trajectory autoencoder,
deliberately narrowed to the subset dense enough for that module's ~60s
window (`action_warrant`, `heartbeat_mean_ratio`, per-domain
`prediction_error`; the field-digester-sourced signals in this table were
already flowing into that corpus and needed no new plumbing). **This doc's
actual point — a named, falsifiable predictive target (Missing Question 1:
"predict a near-future prediction-error spike") replacing pure
reconstruction-only training — is still fully open, not touched by that
retrain.** Consuming the audit is not the same as implementing this design;
the "Recommended next patch" below (retire the dead registry entries, rewrite
`fit_phi_encoder.py` against the real feature set and the named target) is
still exactly as unstarted as it was when this doc was written.

## Summary

Phi v1 (`phi_heuristic.valence` / `phi_intrinsic_reward.v1`) is dead — its sole
producer, `orion-spark-introspector`, was deleted outright on 2026-07-28. What
remains is an offline training script (`scripts/fit_phi_encoder.py`) hardcoded
to frozen `SelfStateV1`-era feature names nothing live still produces. Phi v2
is not a tune-up of that. It's a new score, built from scratch, on metrics we
actually pulled real numbers for.

## Current architecture

- No live phi producer exists today. `orion/inner_state_registry.py:301-383`
  still lists both old metrics pointing at the dead `producer_service` with no
  `RETIRED` marker — needs fixing regardless of what v2 becomes.
- `scripts/fit_phi_encoder.py` trains an autoencoder against hardcoded
  `SelfStateV1`-derived feature names. Nothing live imports it.
- A full live audit (four parallel investigations, live Postgres queries
  against `conjourney`, no code-only guessing) produced a real, evidence-
  checked inventory of what's actually running today. Results below.

### Signals confirmed real, live, and independent

| Signal | Source | Why it's good |
|---|---|---|
| `execution_pressure` | `substrate_field_state` | Real spread, 1.8e-15–0.375 |
| `reasoning_pressure` | `substrate_field_state` | Near-zero floor, real spikes to 0.43 |
| `reliability_pressure` | `substrate_field_state` | Bimodal — 59.7% decayed-near-0, 24.7% real elevation |
| `resource_pressure` | `substrate_field_state` | Real floor ~0.035, mean 0.325 |
| `deviation_pressure` | `substrate_field_state` | EWMA-baseline surprise on the pressure vector itself |
| `action_warrant` | `substrate_proposal_frames` | New, strong: Fisher-combined tail probability across the pressure z-scores, own metric-gate pass already in its own code (\|r\|≤0.151 against inputs, 9,941 distinct values over 7 days) |
| `cpu_pressure` / `gpu_pressure` / `disk_pressure` / `thermal_pressure` / `power_pressure` / `fan_pressure` / `memory_pressure` | field-digester `node_vectors` | Real host hardware load, live, non-degenerate |
| `cortex_exec_step_load`, `conversation_load`, `execution_friction`, `repair_pressure`, `compliance_deficit`, `egress_confidence_deficit` | field-digester `node_vectors` | Real cognitive/operational load signals |
| `heartbeat_mean_ratio` | orion-heartbeat H1 ensemble | Independent producer, not derived from anything else here |
| `execution` / `chat` / `biometrics` / `bus_synaptic` prediction-error domains | `substrate_attention_self_model` | Pairwise \|r\|≤0.05 — genuinely independent, theory-anchored (each domain fixed its own floor/decay bug already) |
| `swear_frequency` | `orion-cocreation-signals` affective_state producer, from Juniper's real Claude Code transcripts | The one real "about Juniper" signal — varies (max 0.0435) and genuinely returns to real zero on calm days, not an artifact floor |
| `git_delta` / `pr_lifecycle` / `graph_delta` scores | `substrate_codebase_delta_log` | Real measure of actual collaborative work happening on the codebase |
| `dev_economics` (session_count, total_tokens, total_estimated_cost_usd) | `dev_economics_ledger_log` | Real cost/effort ledger of the actual Juniper+Orion collaboration |
| `doc_semantic_drift` | `doc_semantic_drift_log` | Real embedding-distance measure of how much the docs are actually changing |

### Confirmed dead, degenerate, or duplicate — excluded

- `coherence`, `agency_readiness`, `field_intensity` — don't exist, 0 ticks since the 2026-07-22 SelfState burn.
- `continuity_pressure`, `introspection_pressure`, `social_pressure` — wired in code, never populate (0/102,160 ticks).
- `catalog_drift_pressure`, `contract_pressure`, `expected_offline_suppression`, `observer_failure_pressure`, `staleness`, `tool_failure_streak_pressure`, `turn_incompletion` — exact 0.0 forever.
- `overall_salience` (attention_frames) and `field_overall_salience` (attention_self_model) — pinned at exactly 1.0 across tens of thousands of rows. Saturated/dead.
- `confidence` (attention_self_model) — literal code-level copy of `coalition_stability_score`, not a second signal.
- `prediction_error_confidence` — `1 − mean(the domains)`, its own aggregate restated.
- `stream_backlog_health` = `delivery_confidence` — same value under two names.
- `candidate_ticks` = `dwell_ticks` in the coalition log — same value under two names.
- `tension_baseline_mu`/`tension_baseline_var` — upstream EWMA parameters that *compute* `deviation_pressure`, not a parallel signal.
- `l7_l11_ladder` throughput — a rate/liveness dead-man's-switch shape (~1700/hr, one real outage), not a graded pressure magnitude. Fine as a binary alive-gate, wrong shape as a continuous input.
- Old drives (`DriveStateV1`/`DriveAuditV1`) — fully retired, PR #1486, 2026-07-30.
- "Biometrics" (`orion_biometrics*`) sounding like Juniper's body — it's GPU/CPU telemetry from the compute hosts, not a person.
- Embodiment/AI Town world contact — confirmed dead right now (empty Convex URL in the live `.env`, tables frozen 3 weeks).
- Notification delivery tracking (`notify_attempts`) — zero rows ever, pure black box.

### One important operational fact, not a metric

`execution_allowed` in the policy-decision ladder stage has been hard `false`
across 100% of decisions for 5+ straight days. Orion is currently fully
advisory/read-only at the dispatch layer. Any closed-loop consumer proposed
below has to work within that reality — it can't gate real-world execution
that isn't happening anyway.

### Deferred, not forgotten

**PDU wall-outlet wattage (`pdu_watts`, circe + athena).** Real, physically-
grounded signal (electricity-meter truth, cross-validated against iLO to the
watt when it shipped) — but the fleet got re-cabled today (2026-08-21) and the
Postgres history has zero populated rows for it yet. Circe in particular would
get its first-ever real power reading from this (a genuine gap fill, not a
cross-check). **Action: revisit once real post-recabling history accumulates
— do not wire in on zero rows.** Tracked here as a follow-up, not scoped into
v1 of phi-v2.

## Missing questions

1. **What does phi v2 actually predict?** V1's failure mode across 4
   generations was training an autoencoder against its own reconstruction
   error — no external target, so every version was tuned by feel. V2 needs
   a real target named in the schema itself, not implied by a loss function.
   **Proposed target: near-future prediction-error.** Given the current state
   vector, predict whether `execution`/`chat`/`biometrics`/`bus_synaptic`
   prediction-error is about to spike in the next N ticks. This is concrete,
   checkable against data that already exists, and reframes "phi" as an
   anticipatory signal ("how much is Orion about to be surprised") rather
   than a vibes-based mood number.
2. **What's the closed-loop consumer?** Given `execution_allowed` is globally
   false right now, a real actuator can't be "gate real-world action." The
   most honest candidate today is `orion/substrate/endogenous_curiosity.py`'s
   candidate scoring — it's live, budget-constrained, and not blocked by the
   dispatch-gate — feeding phi's anticipated-surprise score into which
   curiosity candidates get budget. Needs a concrete design pass of its own
   once the target above is validated.
3. Does this get bus-published day one, or start as an offline batch score
   written to Postgres until the target is validated? Recommend the latter —
   no new always-on service before there's a real signal to publish.

## Proposed schema / API changes

- Extend `orion/schemas/telemetry/phi_encoder.py`'s `PhiEncoderManifestV1` in
  place (`encoder_version="v2"`), don't fork a new file.
  - `input_features`: the vetted list above (~18 features).
  - New field, not present in v1: `target_definition` — forces the objective
    function to be a reviewable, schema-visible fact, not an implicit
    property of whatever loss function the training script happens to use.
- `orion/inner_state_registry.py`: mark `phi_heuristic.valence` /
  `phi_intrinsic_reward.v1` **RETIRED** (dead since `442e51ee2`) in the same
  patch that registers v2. V2 gets its own registry identity — it does not
  inherit the old one's.
- No new bus channel yet — see "missing questions" above on staying offline
  until the target is validated.

## Files likely to touch

- `orion/inner_state_registry.py` — retire old entries, register new
- `orion/schemas/telemetry/phi_encoder.py` — manifest schema extension
- `scripts/fit_phi_encoder.py` — rewritten to pull the real ~18 features
  directly from Postgres and train against the real named target, replacing
  the hardcoded frozen `SelfState` names entirely
- `docs/superpowers/specs/2026-07-14-inner-state-signal-framework-working-doc.md`
  — living doc, update in place rather than fork a competing spec

## Non-goals

- No `SelfStateV1`/`InnerStateFeaturesV1` import anywhere in new code.
- No new always-on service before the target is validated offline.
- No PDU wattage input yet — deferred per above, revisit once real history exists.
- No `biometrics_cluster.v1` persistence work — separate, smaller task.
- Not gating real dispatch execution — `execution_allowed` is globally false; nothing here changes that.

## Acceptance checks

- Every input feature has a live, non-degenerate Postgres pull shown before
  it's in the set (done for all ~18 in this doc).
- A concrete, falsifiable target is named and written into the schema before
  any encoder architecture is coded.
- A named closed-loop consumer is designed alongside training, not bolted on after.
- Old registry entries carry `RETIRED` in the same patch that adds v2.
- `grep -r SelfStateV1`/`InnerStateFeaturesV1` on the new code returns nothing.

## Recommended next patch

1. Retire the two dead registry entries — cheap, zero risk, stops calling a
   dead thing live.
2. Rewrite `fit_phi_encoder.py` to pull the real ~18 features from Postgres
   over a real historical window, and hold out a real time window to test
   the "predict near-future prediction-error spikes" target before writing
   any encoder architecture.
3. If the target holds up on held-out data: design the endogenous-curiosity
   consumer wiring as its own follow-up patch, not bundled into this one.
4. PDU wattage: revisit as a follow-up once real post-recabling history exists.
