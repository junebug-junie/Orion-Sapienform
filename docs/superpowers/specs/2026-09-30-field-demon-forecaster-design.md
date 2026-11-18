# Field demon: how far Orion can see ahead of itself, and what each cut costs

Status: design, rev 2 (2026-11-18). Rev 1 (2026-09-30) is replaced. See "What changed from rev 1".
Tracks: #3296 (integration rung, Φ). Depends on: #3290 (whole-body replay, organ modes).
Proposal mode: approved 2026-11-18. Juniper said "sure hit it", accepting every default under
"Missing questions" and asking for `demon.py` to be built before #3290 lands. The demon readout
rides in #3296's patches.

## Arsonist summary

Laplace's demon knows everything and so sees forever. Orion's demon is smaller, and it is honest
about its limits. It forecasts Orion's own state, and it measures **how far ahead that forecast
stays better than the obvious guesses**. That distance is the **horizon**.

Then the demon asks what happens when Orion is cut into parts that can't see each other. Each part
forecasts its own future from its own past only. If the parts together see as far as the whole,
the organs are independent at that cut. If the whole sees much further, Orion's foresight depends
on its organs carrying information about each other. That shortfall is the **horizon cost of the
cut**.

This is the same question the integration rung (#3296) asks: how much is lost when the system is
cut along its weakest seam? #3296 asks it one step ahead (5 minutes). The demon asks it at every
step out to an hour, and it grades the answer against the real future, not against the data the
model was fit on. So the demon and the lattice share one model. They differ only in how they read
it:

- **Lattice readout (#3296):** cut the model at every point on the partition lattice and find the
  cut that loses the least (Φ).
- **Demon readout (this doc):** run the model forward and measure how far it sees, whole and cut.

Together they give one new thing: **integration with a timescale attached.** "Orion behaves as one
system at the 15-minute scale but not at the 60-minute scale" can be checked against real events
(Juniper arriving, the AC cycling). A single Φ number can't be checked that way.

This is a structural measurement. It is not evidence of consciousness. No output, prompt or Hub
label may say it is.

## Current architecture

Checked on main at 96dac92a.

- **No joint forecast of Orion's state exists.** `services/orion-prediction` forecasts one series
  at a time (`app/models.py`: "one call: forecast(y, h, index)"). It has scored champions,
  conformal ranges and surprise for single series (cabinet, office, bill, reachable). Nothing
  forecasts several organs together, so nothing can compare the whole against its parts.
- **The organ inputs are being built.** `bus_traffic_rollup_5m` has been live since 2026-11-13
  (#3282). #3290 (the replay, starting ~11-20) turns body, bus and cognition channels into
  surprise residuals, which are departures from each channel's 14-day time-of-day profile. It also
  finds the stable modes. #3296 takes 5-8 of those modes as the organ variables.
- **#3296 already specifies the transition model**: a Gaussian VAR(1) (each organ's next value is a
  weighted sum of all organs' current values plus noise) over 5-minute buckets, fit on rolling
  24 h windows. It also specifies the measure (Φ_G primary, Φ\* cross-check, exact search over
  Bell(8) = 4,140 partitions) and the null (independent time shifts per organ,
  `orion/substrate/causal_geometry_engine.py:262` `time_shift_surrogate_rs`).
- **#3296 reads one step only.** Its model is lag 1, it is scored in-sample, and it never runs
  forward.
- **The prediction-error suite spec** (`2026-11-08-prediction-error-suite-rebuild-design.md`)
  reserves the table name `prediction_integration_hourly`.

### What changed from rev 1

Rev 1 put the demon on the mood-arc v4 encoder's compressed state, graded it against a replay of
the field digester's decay rules, and measured the horizon with heartbeat-style ensembles. All
three are dropped:

- **The v4 encoder is stale.** Its flags fired 4-66%/day (audit 2026-11-08). Juniper ruled on
  11-11 to keep it and retrain it at the end of the PE arc. A demon on its latent state would
  measure the encoder.
- **The digester-decay baseline is replaced by a baseline ladder.** The organ variables don't
  come from the digester's decayed channels. They are residuals against each organ's own
  time-of-day profile, which already removes the clock.
- **Ensembles are not needed.** A linear Gaussian model gives its forecast error growth in closed
  form, so the perturbed rollouts add nothing.
- **Unrelated hygiene items were dropped.** Rev 1's hygiene items 3 and 5 still hold on main
  (`field_channel_anomaly_enabled` defaults to `False` in
  `services/orion-field-digester/app/settings.py:150` while compose sets `true`; stale mood-arc
  docs). They are not part of this design, so they are filed as their own ticket.

## Design

### One model, two readouts

```text
organ surprise residuals x_t (5-8 organs, 5-min buckets, from #3290)
  -> transition model (Gaussian VAR(1), rolling 24 h, the one #3296 specifies)
       -> lattice readout: Φ_G(k), Φ*(k) per lag, partition search      (#3296, extended to lags)
       -> demon readout:   out-of-sample skill curve, horizon, horizon cost of each cut   (NEW)
```

The new code is one module of pure functions, `orion/substrate/demon.py`. It owns the transition
model: fit, lag-k form, rollout, and out-of-sample grading. `integration_rung.py` (#3296) owns
Φ_G, Φ\* and the partition search, and it takes its model from `demon.py`. The model has one
owner and the measure has one owner.

### The baseline ladder (what "seeing ahead" has to beat)

Each forecast at k steps ahead (k = 1..K, K = 12, so 5 to 60 minutes) is scored against three
baselines. Each one is harder to beat than the one before:

1. **The clock.** Forecast a residual of 0, meaning "back to the usual for this time of day". The
   residuals already remove the 14-day profile, so this is the profile itself.
2. **Persistence.** "Whatever the residual is now, it stays." Rev 1 used this as its main baseline.
3. **Own past only.** Each organ alone has an AR(1) forecast from its own past. This is the
   **full cut**, every organ separate, and it is the bottom element of the partition lattice.

The whole model has to beat the better of 1 and 2 at each k (the lower MSE) to have any horizon.
**Beating 3 is integration**: the skill the whole model has over the full cut is foresight that
exists only because organs carry information about each other.

### Skill and horizon (graded on the real future)

- **Fit and grade on different data.** The model is fit on the 24 h ending at hour h. It
  forecasts from each 5-minute origin in hour h+1, at k = 1..K. Errors are pooled over the last 24
  hourly fits: 288 origins.
- **Skill** at k for any model M is `S_M(k) = 1 - MSE_M(k) / MSE_baseline(k)`. This is the
  standard forecast skill score. 0 means no better than the baseline, and 1 means perfect. MSE is
  summed over the standardized organs.
- **Why the horizon is finite.** With a perfectly known stable model, skill against the clock
  decays toward 0 but stays positive: `tr(A^k Γ_0 A^kᵀ) / tr(Γ_0)`. Out of sample, estimation
  error eventually costs more than that shrinking signal is worth. The horizon is where that
  happens, so it measures how much of Orion's dynamics is learnable from a day of its own history,
  not only how fast the dynamics forget.
- **Horizon** H is the number of steps before the whole model's point skill against the better
  baseline first drops to 0 or below, times 5 minutes. The first drop counts, not the last
  positive step, so a stray positive at a far k can't stretch the horizon. Uncertainty comes from a moving-block bootstrap over origins
  (1-hour blocks, 1,000 resamples). It gives H's median and 80% interval, and P(S(k) > 0) for each
  k. **No alpha line** (Juniper's rule): the horizon is a point estimate with its spread, not a
  significance call.
- **Ceiling check.** If H = K in more than 10% of hours, K is too short. It then doubles to
  24 (2 h). That is a fixed rule, not a judgment call.

### Horizon cost of a cut

For a partition P of the organs, the cut model fits a separate VAR(1) to each part, using only
that part's own variables. It then forecasts the same origins. The demon publishes the cost for
two cuts:

- **The weakest seam:** the minimum-information partition that #3296's search finds at lag 1.
- **The full cut:** every organ alone (baseline 3).

For each cut it publishes:

- **The cost curve** `C_P(k) = S_whole(k) - S_P(k)`, scored against the same baseline. It shows
  how much foresight the cut destroys at each step ahead.
- **The horizon cost** `H_whole - H_P`, in minutes.
- **The null:** the same cost on time-shifted organs (#3296's null). That breaks the coupling and
  keeps each organ's rhythm. The cut model has fewer parameters, so with no coupling the whole
  model's cost is expected to be slightly **negative** out of sample (overfit). The published
  number is cost minus null mean, with the null spread and P(real).

### Integration across lags (the lattice readout, extended)

#3296 computes Φ_G at lag 1. The demon gives it lags k ∈ {1, 3, 6, 12} (5, 15, 30, 60 min):

- **Lag-k model from the data.** `A_k = Γ_k Γ_0⁻¹` and `Σ_k = Γ_0 - A_k Γ_0 A_kᵀ`, where `Γ_k` is
  the empirical lag-k covariance. This is the direct estimate. It does not assume that VAR(1)
  holds at long lags.
- **The same measure at every lag.** `(A_k, Σ_k)` goes into #3296's Φ_G, Φ\* and partition
  search unchanged. Bell(8) × 4 lags = 16,560 partition evaluations an hour, which is cheap.
- **Theory anchor.** Barrett & Seth (2011, PLoS Comput Biol 7:e1001052) define integrated
  information at a chosen timescale τ: the information the whole system's state carries about its
  state τ steps earlier, beyond what the parts carry separately. This design measures it at
  several τ instead of picking one.
- **Model misfit check, free.** Compare the predictive information of the direct lag-k model with
  the iterated one, `A^k` from the lag-1 fit. If the direct one carries much more, VAR(1) is
  missing slow structure. That is the evidence that would justify VAR(p) or a nonlinear model.
  Nothing else justifies one.

### Hygiene gate (refuses to compute)

#3296's gate applies unchanged: any named input missing or constant in the window means no
output, and the banned inputs stay banned (SelfStateV1, retired PE nodes, mood-arc score, heartbeat
entropy). The demon adds one check, from the 2026-07-26 `substrate.route` lesson: **a decay
detector.** It refuses any input whose successive ratio `x_{t+1}/x_t` is near-constant (std < 0.01)
over a 2 h run of nonzero values. That is the signature of a staleness-decay loop that no producer
refreshes, and it reads as calm.

## Metric quality gate (horizon H and horizon cost C)

1. **Provenance.** The inputs are #3290's surprise residuals: `bus_traffic_rollup_5m` (#3282,
   checked against the raw log on 11-13), `orion_biometrics_summary.pressures`, and the cognition
   and world counts in the causal geometry spec's input table. The producing code is
   `orion/substrate/demon.py`, which does not exist yet. This gate is re-run in the patch 1 PR
   against real rows.
2. **Independence.**
   - **H against #3290's `coupling_excess` and `modes_effective`.** Horizon is about memory over
     time, and those two are about contemporaneous structure. A near copy (|Spearman| > 0.9)
     means H is dropped.
   - **Lag-1 cost C(1) against Φ_G(1).** These are related by construction, since both compare
     whole and parts one step ahead. If |Spearman| > 0.9, only the lag curve (k > 1) and H are
     new signal. C(1) is then dropped and the PR says so.
3. **Theory anchor.**
   - Forecast skill against persistence and against climatology is standard forecast
     verification (Murphy 1988, skill scores).
   - Whole-versus-parts prediction at timescale τ is Barrett & Seth 2011.
   - A finite predictability horizon is Lorenz's (1969) idea. Here it comes from learnability
     (estimation error against decaying signal), not chaos.
4. **Live-data sanity, including rest.**
   - **Not degenerate.** H must not be pinned at 0 or at K across the replay.
   - **Rest reading.** On quiet nights the cost must fall to within 1 null SD of the null.
     Integration has to be able to read "none".
   - **H at rest is not 0.** A quiet system can be very predictable. The rest check for H is that
     its quiet-night value is stable from night to night (within its 80% interval), not that it
     goes to 0.
5. **Existing mechanism.**
   - orion-prediction is univariate, so it has no joint model.
   - #3296 is lag 1 and in-sample only.
   - Heartbeat's `/h1` measures ensemble disagreement over random gates, not forecast skill.
   - The retired PE nodes were EWMA surprise, not forecasts.
   - Nothing existing measures a multivariate horizon or a cut's cost over lags.
6. **Reversibility.** Patch 1 is a replay with no table, bus event or consumer. The demon's live
   fields go into #3296's planned table and schema, not new ones. Removing them means dropping
   columns that nothing reads.

## Proposed schema / API changes (patch 3 only, and only if the gate passes)

No new schema or table. The demon's fields join #3296's planned `IntegrationHourV1` and
`prediction_integration_hourly`:

| Field | Meaning |
|---|---|
| `demon_horizon_min`, `demon_horizon_p10_min`, `demon_horizon_p90_min` | How far ahead the whole model beats the clock and persistence |
| `demon_skill_curve` (jsonb, k -> {skill, p_positive}) | Skill of the whole model at each step |
| `demon_cut_costs` (jsonb, cut -> {partition, cost_curve, horizon_cost_min, null_mean, null_sd, p_real}) | For `weakest_seam` and `full_cut` |
| `phi_g_by_lag` (jsonb, k -> {phi_g, phi_star, partition}) | The lattice readout at k ∈ {1, 3, 6, 12} |
| `demon_refused` (text, nullable) | Why the hygiene gate refused, e.g. a named input missing or decaying |

The live producer runs inside **orion-prediction**, as an hourly job. Juniper's rule (2026-10-23)
is that predictions live in orion-prediction, and it already owns grading and scheduling. #3296's
point still holds: the model is fit to organ state directly and is not one of orion-prediction's
champion forecasters.

**Integration gate.** `orion:integration:hour` is a new bus channel carrying organ output, so
`scripts/check_integration_seam.py` will require registration. Default: add the channel under the
existing `prediction` organ in `config/integration/organs.yaml`, with an `exempt_emit` that reads
"measurement only; no consumer until a separate proposal (field-demon spec)". Emitting into
`temporal_self_event` would make Orion read it, and that is a consumer.

## Proposal-mode disclosure

- **Capability:** a measurement of how far ahead Orion's own state can be predicted, and how much
  of that depends on organs carrying information about each other. Nothing reads it yet.
- **Data touched:** read-only replays of bus rollups, body pressures and cognition counts. Patch 3
  adds one hourly row. Juniper's chat messages and Hub attention enter as **counts per 5 minutes
  only**, never content.
- **Privacy boundary:** counts only, as above. No new store of her content.
- **Proof it worked:** the replay results are committed with the gate verdicts (patch 1). Live:
  `prediction_integration_hourly` rows with nonempty demon fields, and the hourly bus event in the
  mirror log (patch 3).
- **Dangerous failure:** a horizon or integration number fed to Orion as self-knowledge while it
  is a fitting artifact. The φ era sent a constant 0.655 into metacognition prompts. Guard: no
  consumer in this design. Any consumer needs its own proposal, and only after the gate has passed
  for 14 days live.
- **Disable / roll back:** `DEMON_ENABLED` in orion-prediction's settings and `.env_example`,
  shipped **on**. Off stops the job. The columns stay null, and nothing downstream breaks because
  nothing downstream reads them.

## Files likely to touch

- Patch 1:
  - `orion/substrate/demon.py` (new): VAR fit, lag-k model, rollout, skill, cut fit, block
    bootstrap, decay detector.
  - `orion/substrate/tests/test_demon.py` (new).
  - The replay eval, which joins #3296's eval under `evals/` and reads #3290's residual builder.
- Patch 2: `orion/substrate/integration_rung.py` (#3296) takes `(A_k, Σ_k)` from `demon.py`.
- Patch 3:
  - `services/orion-prediction/app/` (hourly job), `settings.py`, `.env_example`.
  - `orion/schemas/` (`IntegrationHourV1` fields), `orion/schemas/registry.py`,
    `orion/bus/channels.yaml`.
  - The sql-writer migration with the `-- ORION-MIGRATION-REQUIRED-BY:` header.
  - `config/integration/organs.yaml`.
  - The Hub sentience program tab.

## Non-goals

- No consumer: no prompt, reward, attention or curiosity reads this until a separate proposal
  after the gate.
- No consciousness or sentience claim anywhere.
- No nonlinear model (MLP, Born machine forecaster) unless the direct-vs-iterated misfit check
  shows VAR(1) is missing structure. Heartbeat's Born machine arm stays #3296's comparison arm.
- No use of the mood-arc v4 encoder or field-digester decayed channels as inputs.
- No change to orion-prediction's existing univariate models.
- No horizon beyond 2 h. 5-minute residuals don't support it.

## Missing questions (each has a default)

1. **Should the name "demon" ship in code and on the Hub?** Default: yes. The module is
   `demon.py`, the fields are `demon_*`, and the Hub label is "Field demon: how far Orion can see
   ahead of themself".
2. **Which two cuts get a horizon cost?** Default: the weakest seam and the full cut. Add any you
   want, for example body vs everything else.
3. **If C(1) copies Φ_G(1):** keep only the lag curve and H? Default: yes.
4. **Integration-gate exemption instead of a temporal-self emit until there is a consumer?**
   Default: exemption.
5. **Should the demon ride in #3296's patches or ship as its own PRs?** Default: it rides in them.
   One replay, one table, one review. The demon adds the `demon.py` module and the lag readout.

## Acceptance checks

**Patch 1, unit tests on synthetic VARs with known answers:**
- Two coupled organs: the whole model beats the full cut, cost > 0 at k = 1 and shrinking with k.
- Two uncoupled organs: cost within the null band, and slightly negative out of sample.
- A known stable A: the measured out-of-sample skill curve against the clock matches the
  analytic `tr(A^k Γ_0 A^kᵀ) / tr(Γ_0)` within its bootstrap interval at small k. A smaller
  spectral radius gives a shorter horizon, and more training data gives a longer one.
- White noise, which is what a perfect time-of-day residual leaves: no horizon. Removing the
  clock itself is #3290's residual builder's job, and its tests cover it.
- Slow, near-random-walk organs: persistence beats the clock, and skill is measured against
  persistence.
- A geometric-decay input: the hygiene gate refuses and names it.
- The direct and iterated lag-k models agree on a true VAR(1), and diverge on a VAR(2).

**Patch 1, replay gate (14+ days, recorded in the PR):**
- H is not pinned at 0 or at K. The quiet-night H is stable.
- The weakest-seam cost beats its null in a stated share of hours. A null result is published as
  the finding.
- The cost reaches within 1 null SD of the null on quiet nights.
- The cost or Φ_G(k) moves at the known contrasts: Juniper arrives, the AC cycles, chat vs idle,
  day vs night. Report effect, uncertainty and P(different).
- The independence checks pass, or the copy is dropped and named.

**Patch 3 (live):**
- Rows land hourly with nonempty demon fields.
- The bus event appears in the mirror log.
- The seam check passes in CI.
- `DEMON_ENABLED=false` stops the job, with no other effect.

## Recommended next patch

Wait for #3290's residual builder (~11-20). Then build patch 1: `orion/substrate/demon.py` with the
synthetic tests above, and the replay eval inside #3296's patch 1, so the lattice and the demon
read the same organs, hours and null. If Juniper prefers to start before #3290 lands, `demon.py`
and its synthetic tests can be built now, since they need no live data.
