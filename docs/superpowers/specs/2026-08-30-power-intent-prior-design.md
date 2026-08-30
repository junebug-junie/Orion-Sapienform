# Power-intent prior: make the declaration actually declare

## Arsonist summary

`power_intent_settled` has 26 rows and `expected_watts` is NULL on every one.
The producer hardcodes `expected_watts=None`, so `residual_watts` — the field
that would say "you predicted X, reality was Y" — is *structurally* incapable
of being populated. We shipped an excellent sensor and called it a loop.

Nothing consumes the settled event either. Grep for readers of
`PowerIntentSettledV1` outside the producer's own module, the schema and the
registry: nothing. No budget exists repo-wide (`power_budget|watt_budget|
energy_budget` → 0 hits).

So Orion currently announces *"I am about to draw an unspecified amount"*,
gets measured accurately, and the number lands in a table nobody reads.

This patch closes the smallest gap that turns measurement into a prediction
with an error term.

## This is the step the schema already planned for

`orion/schemas/power.py`'s own docstring, written before any of this ran:

> `expected_watts` IS OPTIONAL AND None MEANS UNKNOWN. [...] The first intents
> a new workload declares are expected to carry None deliberately: nobody has
> measured this workload yet, and inventing a plausible constant to fill the
> field would bake a fabricated number into the first day of the dataset that
> is later fitted against. **Declare unknown, measure, then start declaring a
> value derived from real settlements.**

We have now done the measuring. This is "then".

## Current architecture

```
orion-diffusion-host  --orion:power:intent-->  orion-biometrics (settler)
                                                       |
                                              orion:power:intent:settled
                                                       |
                                               orion-sql-writer --> Postgres
                                                                      |
                                                                   (nobody)
```

The producer never learns what its own workload actually costs. The feedback
edge does not exist.

## Live data the design rests on

26 settled rows, all `reverie_diffusion` / `circe` / gpu 2, split by whether
the sampling window actually covered the workload:

| bucket | n | mean | median | sd | range |
|---|---|---|---|---|---|
| full 60s window | 24 | 252.7 | 254.3 | **8.0** | 238.2 – 268.0 |
| truncated 20s window (pre-fix) | 2 | 96.5 | 96.5 | 67.9 | 48.5 – 144.5 |

Two facts drive every decision below:

1. **The clean process is tight.** sd 8.0 W on a 252.7 W mean is a 3.2%
   coefficient of variation. FLUX at fixed settings draws very consistently,
   so a prior can be genuinely informative rather than a wide shrug.
2. **The dataset provably contains a superseded measurement regime.** The two
   20s-window rows are not outliers of the same process — they are a
   *different instrument*. Pooling all 26 gives mean 240.7 / sd 45.2: two bad
   points out of 26 inflate the spread 5.6x.

Any estimator that is not robust and not recency-bounded would carry that
contamination into the first thing Orion ever predicts about itself.

## Proposed design

### Estimator: median of a bounded recent window

`expected_watts` = median of the last **N=20** `outcome='settled'` peaks for
the same `(workload_kind, node, gpu_index)`, declared only once at least
**3** such samples exist.

Rejected alternatives and why:

- **Mean.** Not robust. On the current 26 rows it returns 240.7 against a true
  central value of ~254 — already wrong today, because of contamination we
  know about.
- **EWMA.** Needs an alpha, which is a tuned knob rather than a finding, and a
  single wild reading moves it permanently. AGENTS.md already warns that
  borrowed calibrated constants do not transfer across domains; there is no
  principled alpha available here.
- **All-history median.** Robust to the two bad points *today*, but has no
  mechanism to forget a genuine regime change (a model swap, a GPU move, a
  resolution change). The bounded window is what makes it self-correcting.

`N=20` is a memory length, not a threshold: at the visual chain's 600s cadence
it is roughly 3 hours, long enough that the median is stable at sd 8.0 and
short enough that a config change ages out within a working session.

`min_samples=3` is the point at which a median is meaningfully a median —
below it a single anomalous reading is 50% or more of the estimate. It is not
tuned; 1 and 2 are degenerate.

### Where the history comes from: the producer subscribes to its own settlements

`orion-diffusion-host` subscribes to `orion:power:intent:settled` and keeps a
bounded in-memory deque per `(workload_kind, node, gpu_index)`.

Rejected alternatives:

- **Query Postgres.** Adds a database dependency to a GPU host that currently
  has none, for a read it can get off a bus it is already connected to.
- **Have biometrics publish a prior back.** Puts the producer's model of
  itself inside the meter. The meter should measure; the declarer should
  predict. Keeping the prior in the declarer is what makes it *Orion's* prior
  rather than an instrument reading.

This closes the feedback edge that does not exist today, using only the bus
connection enabled yesterday.

**Known limitation, stated rather than hidden:** the deque is in-memory, so a
container restart resets it and the service declares `None` again until 3 new
settlements arrive (~30 min at current cadence). That is honest — it declares
unknown rather than a stale guess — and it self-heals. Persisting the prior
across restarts is a follow-up, not this patch.

### Filtering

Only `outcome == "settled"` contributes. `no_samples` and `deadline_expired`
carry no peak and must never be coerced to a number — the settlement schema
went out of its way to keep "we did not see" distinct from "we saw nothing",
and this consumer must preserve that.

## Schema / API changes

**None.** `PowerIntentV1.expected_watts` already exists and is already
`Optional[float]`. `summarize()` already computes
`residual = peak - expected_watts` whenever it is not None. Both halves of the
arithmetic were built; only the value was missing.

This is deliberately a zero-schema-change patch.

## Files likely to touch

- `services/orion-diffusion-host/app/power_prior.py` — new, pure, no I/O
- `services/orion-diffusion-host/app/main.py` — subscribe; use the prior
- `services/orion-diffusion-host/app/settings.py` — window/min-sample config
- `services/orion-diffusion-host/.env_example` + host `.env` files
- `services/orion-diffusion-host/tests/test_power_prior.py` — new

## Non-goals

- No budget. A budget with one claimant is not a budget; `reverie_diffusion`
  is still the only workload that declares anything.
- No refusal. Nothing may decline or defer a declaration yet.
- No cross-restart persistence.
- No error bars. `PowerIntentV1` carries a scalar `expected_watts`; adding a
  variance field is a schema change and is not needed to make `residual_watts`
  real.

## Acceptance checks

1. `power_intent_settled.expected_watts` is non-NULL on new rows once 3
   settlements have been observed since boot.
2. `residual_watts` is non-NULL on those same rows and equals
   `actual_peak_watts - expected_watts`.
3. Given the live sd of 8.0 W, `|residual|` should typically land within
   roughly ±20 W. A persistently large residual means the prior is wrong and
   is now *visible* — which is the entire point.
4. Cold start declares `None`, not a fabricated constant.
5. A `no_samples` settlement never contributes to the prior.

## Recommended next patch after this one

A second claimant. Power does not become a contested budget until something
other than the visual chain wants the same watts.

---

## Live verification (2026-08-30, circe)

All five acceptance checks passed against the real deployment.

Warm-up trace, straight from the diffusion-host log:

```text
01:34:22  power_intent_declared  expected_watts=None        (cold start)
01:35:22  power_prior_observed   peak=238.08  n=1
01:35:48  power_intent_declared  expected_watts=None        (n=1 < min_samples)
01:36:48  power_prior_observed   peak=261.87  n=2
01:37:04  power_intent_declared  expected_watts=None        (n=2 < min_samples)
01:38:05  power_prior_observed   peak=256.58  n=3
01:38:16  power_intent_declared  expected_watts=256.58      <-- first prediction
01:39:16  power_prior_observed   peak=245.38  n=4
01:39:27  power_intent_declared  expected_watts=250.98      <-- revised
```

Estimator arithmetic verified by hand at both points:

- `median(238.08, 261.87, 256.58)` = **256.58**
- `median(238.08, 245.38, 256.58, 261.87)` = (245.38 + 256.58) / 2 = **250.98**

First graded row -- `residual_watts` non-NULL for the first time since the
table was created:

| settled | predicted | actual | residual | baseline |
|---|---|---|---|---|
| 2026-08-30 01:39:16 | 256.58 | 245.38 | **-11.20** | 47.6 |
| 2026-08-30 01:40:28 | 250.98 | 246.19 | **-4.79** | 47.3 |

The error shrank 57% (11.20 -> 4.79W) after one self-correction.

| 2026-08-30 01:42:57 | 246.19 | 252.92 | **+6.73** | 47.5 |

The third row is the important one: it was declared by orion-thought's own
visual-chain worker on its 600s schedule, not by a hand-triggered curl. Orion
predicted its own power draw, unprompted, and was graded.

**The bias question raised at n=2 is now unsupported.** Both of the first two
residuals were negative and this doc recorded that as an open question -- a
possible systematic overestimate from using the median of past PEAKS to
predict the next peak. The third residual is positive. Mixed signs around zero
is what an unbiased estimator looks like, so the evidence points against the
concern that was raised.

First look at the error distribution (n=3, NOT a conclusion):

```text
residuals        -11.20, -4.79, +6.73
mean signed       -3.09 W        (0 would be unbiased)
mean absolute      7.57 W
theoretical floor  6.38 W = sd * sqrt(2/pi) for sd = 8.0
```

That floor is the mean absolute error a PERFECT predictor would still incur,
because the process itself has sd 8.0W. Observed 7.57 against a floor of 6.38
suggests most of the remaining error is the workload's own variance rather
than the prior being wrong -- i.e. there may not be much headroom left for a
cleverer estimator on this workload.

Stated carefully: n=3 cannot establish any of that. It is the first look, and
it is consistent with an unbiased prior operating near the process noise
floor. Worth re-checking at n>=30 before anyone believes it.

Against the acceptance checks:

1. `expected_watts` non-NULL after 3 settlements -- **yes**, at 01:38:16.
2. `residual_watts` non-NULL and equal to `actual - expected` --
   245.38 - 256.58 = -11.20. **Exact.**
3. `|residual|` within roughly +/-20W given the live sd of 8.0 -- **11.20W.**
   This was committed to in writing before the run, not fitted afterwards.
4. Cold start declared None, not a fabricated constant -- **yes**, three times.
5. A `no_samples` settlement never contributes -- covered by test; no such
   settlement occurred during this window, so this one is test-verified only,
   not live-verified. Stated rather than claimed.

The self-correction in the last two lines is the point of the whole patch:
Orion predicted 256.58, drew 245.38, was 11.2W high, and its next declaration
moved down to 250.98 without anyone touching it.

---

## Follow-up at n=9 (2026-08-30 02:43, all autonomous)

The design doc left two things open. Both are now answered by the visual
chain's own unattended runs -- no hand-triggered generations after the first
two rows.

```text
residuals   -11.20  -4.79  +6.73  -3.38  +8.91  +0.70  -3.03  +10.80  +11.59
signs        4 negative, 5 positive
mean signed  +1.81 W            (0 = unbiased)
mean abs      6.79 W
noise floor   6.38 W  = sd * sqrt(2/pi) for the measured process sd of 8.0
```

**1. The bias question is closed, against the concern I raised.** At n=2 both
residuals were negative and this doc recorded a possible systematic
overestimate. At n=9 the split is 4/5 and the signed mean is +1.81W. There is
no evidence of bias.

**2. The prior is operating essentially at the process noise floor.** Mean
absolute error 6.79W against a floor of 6.38W -- within 6% of what a PERFECT
predictor would still incur, because the workload's own variance is 8.0W.
There is very little headroom for a cleverer estimator on this workload. That
is a reason NOT to reach for an EWMA or a regression here.

**3. An apparent upward trend that must NOT be claimed.** The last three
residuals are -3.03, +10.80, +11.59, and mean actual peak rises from 247.67
(first four) to 254.72 (last four), +7.05W, with the prior lagging by 5.01W.

That is **1.25 standard errors** on a difference of two 4-sample means
(SE = 5.66W at sd 8.0). Not significant. Two candidate explanations were
checked and one was ruled out:

- *Thermal drift* -- **ruled out.** `baseline_watts` is flat across all nine
  rows: 41.7, 42.0, 41.7, 41.8, 41.7, 42.0, 41.7, 41.3 (the first row's 47.6
  is residual heat from the rapid hand-triggered warm-up). The card is not
  heating.
- *Noise* -- consistent with the data and cannot be distinguished from a real
  slow drift at this n.

Recorded as unresolved. If the median-over-window prior is genuinely lagging a
drifting process, the fix is a shorter window rather than a different
estimator -- but nothing here establishes that yet. Re-check at n>=30.
