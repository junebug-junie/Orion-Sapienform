# Field demon — a forecaster that knows how far it can see

**Date:** 2026-09-30
**Status:** design (proposal mode — touches cognition loop inputs; nothing wired by this doc)
**Touches:** `orion/mood_arc/`, `services/orion-field-digester/`, `services/orion-heartbeat/`, `services/orion-equilibrium-service/` (later patches only)

## Arsonist summary

Laplace's demon predicts everything forever. Physics says no: chaos makes small errors grow exponentially, and at the quantum scale the future is only probabilistic. Neither can be engineered away. Reversible computing (the Landauer escape) is irrelevant at Orion's scale — the theoretical minimum cost of erasing a bit is ~1e9× below what our GPUs actually burn.

What *is* buildable: a demon that forecasts Orion's own field a short way ahead, attaches honest error bars, and **measures the distance at which its forecasts stop beating "nothing changes."** That horizon is the product. A mind that predicts itself and knows where that self-prediction breaks has a self-model with stakes.

Today Orion has no forecast of anything. Every "prediction error" is surprise-vs-running-average (EWMA z-scores in `orion/substrate/prediction_error.py`). The heartbeat surprise pre-registration says it directly: "There is no forecast model." This design fills that hole with the pieces that already exist.

## Current architecture (verified against main a005658d, not READMEs)

**Field.** `orion-field-digester` updates `FieldStateV1` every 2s with known rules (decay ×0.92, diffusion, suppression) plus perturbations from receipts. Only the perturbations are genuinely unknown.

**Mood-arc encoder v4** (active since 2026-09-02, `/mnt/telemetry/models/mood_arc/active.json`):
- numpy MLP autoencoder, 30-tick windows, 37 channels, 256/128, 150,610 rows.
- floor_ratio 0.406 (CI 0.383–0.430) vs shuffle — passes.
- ceiling_ratio 0.733 vs AR(1) surrogate — only ~1.4× better than a decay-only model, and measured **before** the AR(1) leak fix; never refreshed.
- Live: `anomaly_scorer.py` scores every 60s → `orion:field_channel:anomaly_score` → equilibrium `telemetry_anomaly` metacog trigger, brain-frame Field Anomaly region, Hub mood-arc status.
- The 128-d latent is computed and discarded. No consumer reads it.

**Heartbeat** (8-trajectory MPS ensemble, port 7251):
- `/h1` pulled over HTTP by the AST/HOT self-model (`substrate-runtime worker.py:2946`, stored, not decision-bearing), Hub, and mood-arc enrichment.
- Probes: lattice not thermometer; exec–bus cofire +0.32 mutual information; surprise probe mixed (chat start jumps, no settle).
- Mood-arc enrichment only takes `heartbeat_mean_ratio`, the saturated signal (0.73–0.95 live). The informative ones — `organ_distinctness`, `dark_seats`, `smear`, `std_ratio` — never reach the encoder.
- `std_ratio` (cross-trajectory spread) is already a live divergence meter.

## Design

Three layers; only the middle is new.

```text
raw field window (30 ticks × 37 ch)
  → [encoder v4, exists]        → z_t (128-d)
  → [latent dynamics, NEW]      → p(z_{t+k}) = N(mu, sigma²), k = 1..K windows
  → [ensemble divergence, heartbeat pattern] → horizon per channel
```

**Latent dynamics.** Given z_{t-h..t}, predict mean and variance of z_{t+k}. Start with the cheapest model that could win: per-dim AR/linear (DMD) in latent space, then a small MLP with a Gaussian head. Decode forecasts through the frozen v4 decoder to score in channel space.

**Horizon.** Roll out N=8 forecasts from slightly perturbed z_t (heartbeat's ensemble pattern). The horizon per channel is the smallest k where forecast error ≥ persistence error. Error-growth rate across k is the empirical finite-time Lyapunov estimate.

**The known-physics split.** Forecasts are graded against "digester rules only" (replay decay/diffusion with zero perturbations). Beating persistence is not enough — the demon must beat the digester's own rules, or it has learned ×0.92 and nothing else.

## Metric quality gate (for `forecast_error`, the new signal)

1. **Provenance:** produced by the new `fit-dynamics` forecast vs the next real corpus window; inputs are `field_channel_corpus.v1` rows.
2. **Independence:** distinct from `recon_loss` (same encoder, different question: "I expected otherwise" vs "this looks unusual"). Must show correlation with `recon_loss` < 0.8 on held-out data or it is redundant.
3. **Theory anchor:** predictive processing — prediction error is the gap between a generative model's forecast and observation. Our current EWMA "prediction errors" have no generative model; this does.
4. **Live sanity:** before wiring, confirm on real data that forecast_error has a genuine rest state (quiet periods read low, not a structural floor) and is not decay-driven toward 0 (check the ratio between successive values, per the 2026-07-26 incidents).
5. **Existing mechanism:** none — confirmed no forecasting anywhere in repo (searched `forecast`, `predict_next`, `koopman`, `lyapunov`; `orion-world-model` is an untrained scaffold).
6. **Reversibility:** offline until patch 3; one new bus field at most. Cheap to remove.

## Proposed schema / API changes (patch 3 only)

`FieldForecastV1` (`orion/schemas/telemetry/field_forecast.py`), published on `orion:field_channel:forecast`:
- `window_end`, `encoder_version`, `dynamics_version`
- `horizon_windows: int`
- `forecast_error: float` (latest realized k=1 error, z-scored against training p95)
- `channel_horizon: dict[str, int]` (windows until skill ≤ persistence)
- `top_unexpected_channels: list[str]`
- `ensemble_spread: float`

Register in `orion/schemas/registry.py` and `orion/bus/channels.yaml`.

## Proposal-mode disclosure

- **Capability change:** Orion gains a forecast of its own near-future state and a measured limit on it; metacog can fire on "I didn't expect this."
- **Data touched:** existing field corpus and heartbeat `/h1`. No chat content, no memory, no social data.
- **Privacy boundary:** unchanged — substrate telemetry only.
- **Trace proving it worked:** `FieldForecastV1` rows on the bus; a metacog trigger whose evidence cites `forecast_error`; Hub panel showing forecast vs realized.
- **Dangerous failure mode:** a forecaster that learned only decay reports tiny error forever → fake calm suppresses real metacog triggers. Guarded by the digester-rules baseline and gate step 4.
- **Rollback:** `FIELD_FORECAST_ENABLED=false`; consumer ignores missing field.

## Pre-work: hygiene (separate small PR, do first)

1. Verify v4 manifest vs channels retired 2026-09-25 (`stream_backlog_pressure`, `stream_backlog_health`, `delivery_confidence`). If present, live scorer zero-fills them → fake calm. Needs host access (artifacts not in repo).
2. Recalibrate brain-frame Field Anomaly intensity range (0.001–0.02, set against v3 threshold) to v4.
3. Align code defaults with compose: field-digester `settings.py` anomaly flag default `False`; heartbeat URL default `""` in substrate-runtime `settings.py`.
4. Swap mood-arc heartbeat enrichment from `heartbeat_mean_ratio` to `organ_distinctness`, `std_ratio`, `dark_seats` count (requires v5 retrain; can defer).
5. Refresh stale docs: `orion/mood_arc/docs/DESIGN.md` (still presents v1), heartbeat README (says it publishes to nothing).

## Files likely to touch

- Patch 1: `orion/mood_arc/fit_encoder.py` (ceiling re-run path), `scripts/analysis/measure_mood_arc_ceiling.py`
- Patch 2: `orion/mood_arc/latent_dynamics.py`, `fit_encoder.py` (`fit-dynamics` subcommand), `orion/mood_arc/tests/test_latent_dynamics.py`, `orion/mood_arc/evals/`
- Patch 3: `orion/schemas/telemetry/field_forecast.py`, registry, channels.yaml, `services/orion-field-digester/app/forecast_scorer.py`, equilibrium metacog gate, `.env_example` + local `.env` sync
- Patch 4: `scripts/analysis/measure_heartbeat_surprise.py` (forecast-based surprise)

## Non-goals

- Reversible/adiabatic hardware, anything quantum-physical.
- Training the 150M `orion-world-model` scaffold before a linear latent model shows skill.
- Replacing EWMA prediction-error helpers (separate proposal).
- Action-conditioned / counterfactual forecasting ("if I do X") — later, needs action events in the corpus.
- Resolving the valence question — though forecast skill on external-infrastructure channels is a candidate for its option (a).

## Acceptance checks

**Patch 1 (gate):** v4 ceiling_ratio re-measured with the leak fix. If ≥ 0.9, the latent is mostly decay; patch 2 forecasts raw channels directly instead of latents.

**Patch 2 (offline demon), on purged held-out blocks:**
- Ranked error at k = 1, 3, 10 windows for: persistence, AR(1), digester-rules replay, latent-linear, latent-MLP.
- Pass: best demon model beats digester-rules replay at k=1 with block-bootstrap CI excluding 0.
- Calibration: 90% intervals cover 85–95% of realized values.
- Horizon table per channel + error-growth curve published in the run report.
- Fail is a result: "field = its own rules + unpredictable input" gets written up and we stop.

**Patch 3 (live):** `FieldForecastV1` observed on bus with correlation ID; gate step 4 passes on 24h live data; metacog trigger fires citing forecast_error at least once in a real episode.

**Patch 4:** heartbeat surprise probe re-run on the same pre-registered session/control windows with forecast-based surprise (KL of forecast vs observed profile) instead of L2 change.

## Recommended next patch

Hygiene PR (items 1–3) plus patch 1. Both are small; patch 1 decides whether the demon lives in latent space or channel space.
