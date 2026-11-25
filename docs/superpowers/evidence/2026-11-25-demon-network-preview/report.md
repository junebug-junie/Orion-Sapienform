# PREVIEW: trained-network field demon on the 5 heartbeat seats (2026-11-25)

**Label: PREVIEW.** These are the per-service heartbeat seats, not the stable organ modes. The
data cover about 39 hours. Nothing here passes the spec's gate, and the data cannot support a
claim either way. Spec: `docs/superpowers/specs/2026-11-24-heartbeat-seats-field-demon-design.md`
(rev 3, PR #3837). Script: `scripts/demon_network_replay.py`. Numbers: `results.json` (seed 3837).

## Short answer

On the forecast test, the trained network does not beat the linear model at any lag. At 5
minutes it is about even with the simple guesses. The two models choose different weakest cuts at
every lag. The network's result also flips with the test window: on the lattice test window
(earlier rows, mostly night) it beats the linear model by 0.41 bits at 5 minutes, but on the
forecast test window (the last 12 hours) it loses by 0.04 bits. A result that flips with the
window is not a result.

One reading holds in both models: **splitting biometrics off from the pipeline (hub, exec, bus,
harness) costs foresight at 5 minutes, above the shifted null in both.** It is the only lead.

The cause of the rest is too little data. The network trained on 234-311 rows. A 10-site chain
has 59,049 possible states. The spec planned about 8,000 rows. Issue #3886 asks for a kept seat
rollup.

## What was read

- **Organs:** the 5 seats from `services/orion-heartbeat/app/substrate/routing.py` (after #3873):
  hub, biometrics, cortex-exec, bus, harness-governor.
- **Seat value per 5 minutes:** the summed gate level of the atoms the heartbeat would absorb
  (`route_atom`). For the event-driven seats (hub, exec, harness) this is the atom count times 0.5.
  For biometrics and bus it is the sum of their measured levels. Raw counts are useless for those
  two seats: they tick at a fixed rate (count spread 22-25% of the mean, mostly clock jitter).
- **Source:** `grammar_events`, about 172,000 atoms. The query runs in a read-only transaction
  through `docker exec orion-athena-sql-db psql`.
- **Time-of-day profile:** `tod_residuals` with at least 1 earlier day. The ledger keeps about 3
  days, so the first day only builds the profile. Later days rest on 1-2 earlier days, not 14.
- **Outage:** no biometrics or bus atoms at all from 11-24 15:55 to 19:05 UTC (39 buckets). The
  scored span is the longest unbroken stretch: 11-23 01:00 to 11-24 15:50 UTC, 467 rows (38.9 h).
- **Hygiene gate** (`demon.refusal_reason` on the raw values): passed. No seat is missing,
  constant or decaying. The script stops if this gate refuses.
- **Two splits.**
  - Forecast skill: train on the first 311 rows (26 h) and score the last 144 rows (12 h).
  - Cut lattice: it uses the first 351 rows, the same length a shifted null copy keeps (75%). It
    trains on 234 rows and scores 105 rows (origins 11-23 20:30 to 11-24 05:10 UTC, mostly night). Same
    length on both sides, so the observed cost and its null carry the same fitting noise.
- **Models:** the network is one trained chain per lag. Bond dimension 2 won at every lag. The
  linear model is the direct lag-k Gaussian, fed the same 3-level "now" as the network.

## 1. Forecast skill (last 12 hours)

Log-loss in bits per 5-minute row, summed over the 5 seats (lower is better). A uniform guess is
7.92 bits.

| k (steps) | network | linear | clock | persistence |
|---|---|---|---|---|
| 1 (5 min) | 7.561 | **7.519** | 7.915 | 7.617 |
| 3 (15 min) | 7.668 | **7.603** | 7.913 | 7.655 |
| 6 (30 min) | 7.916 | **7.781** | 7.911 | 7.819 |
| 12 (60 min) | 8.024 | **7.764** | 7.908 | 7.817 |

Network skill against the linear model (paired block bootstrap, 2,000 draws, blocks of 1 hour):

| k | skill | 10th-90th percentile | P(network better) |
|---|---|---|---|
| 1 | -0.6% | -4.0% to +1.7% | 0.30 |
| 3 | -0.9% | -4.4% to +2.0% | 0.31 |
| 6 | -1.7% | -5.9% to +1.2% | 0.19 |
| 12 | -3.4% | -7.2% to -1.5% | 0.03 |

The gate is "network beats linear at k=1 with P(real) at least 0.9". It reads P = 0.30. **Gate not
met.**

- Against the best of clock and persistence, the network gains +0.7% at 5 minutes (10th-90th
  -1.0% to +3.0%, P = 0.76).
- The linear model gains +1.3% over the same baselines at 5 minutes (-0.4% to +4.7%, P = 0.87).
- Past 15 minutes, the network is worse than the baselines.

## 2. Cut cost on all 52 partitions (lattice window)

Cut cost is the held-out log-loss when each part forecasts only from its own past, minus the
whole model's log-loss, in bits per row. Null: 20 time-shifted copies (each seat shifted at least
2 hours from every other). The chains are refit on each copy.

On this window the network forecasts well at 5 minutes: 7.21 bits against 7.62 for linear. Only 9
of the 51 cuts cost less than the whole at k=1. At 15-60 minutes, 48-50 of the 51 cuts cost less
than the whole, so the whole chain is overfit there. The null copies show the same overfit (null
means between -0.01 and -0.12 bits).

Weakest cut (minimum-information partition, Balduzzi-Tononi normalization). A negative cost counts
as 0. The null test compares the observed weakest cut with each null copy's own weakest cut.

| k | network weakest cut | cost (bits) | linear weakest cut | linear P(real) | agree |
|---|---|---|---|---|---|
| 1 | bus alone | -0.048 | exec+bus vs hub+biometrics+harness | 1.00 | no |
| 3 | hub / biometrics+exec / bus+harness | -0.144 | hub alone | 0.80 | no |
| 6 | harness alone | -0.150 | bus alone | 0.45 | no |
| 12 | hub alone | -0.203 | hub+harness vs biometrics+exec+bus | 1.00 | no |

- The network's weakest cut costs nothing at every lag (score 0). The network finds a split that
  loses no foresight, so it measures no integration on this data.
- The linear model's weakest cut costs a little more than its null's weakest cut at 5 and 60
  minutes (0.060 and 0.039 bits). That is an in-sample number and so is its null, so the bias
  cancels. But the linear cut changes at every lag.
- The two models never agree. The linear weakest cut ranks between 12th and 48th of 51 in the
  network's order.

## 3. The pipeline block (hub + exec + bus + harness | biometrics)

| k | network cost | network null (mean ± sd) | P(real) | linear cost | linear null | P(real) |
|---|---|---|---|---|---|---|
| 1 | +0.068 | -0.051 ± 0.062 | 1.00 | +0.135 | +0.030 | 1.00 |
| 3 | +0.033 | -0.071 ± 0.105 | 1.00 | +0.047 | +0.039 | 0.65 |
| 6 | -0.001 | -0.072 ± 0.112 | 0.85 | +0.026 | +0.039 | 0.25 |
| 12 | +0.029 | -0.056 ± 0.069 | 0.95 | +0.063 | +0.034 | 1.00 |

At 5 minutes, both models say that this split costs foresight, above every one of 20 null copies.
Body telemetry and the pipeline move together at the 5-minute scale. This is the only reading
that agrees across both models. It rests on 105 test rows from one night, so it is a lead, not a
finding.

## 4. Entanglement against mutual information per cut (k = 1 chain, lattice window)

| cut after | entanglement (bits) | mutual information (bits) |
|---|---|---|
| hub:now | 0.26 | 0.12 |
| hub:later | 0.22 | 0.10 |
| biometrics:now | 0.52 | 0.29 |
| biometrics:later | 0.50 | 0.28 |
| exec:now | 0.44 | 0.24 |
| exec:later | 0.37 | 0.19 |
| bus:now | 0.26 | 0.12 |
| bus:later | 0.18 | 0.08 |
| harness:now | 0.16 | 0.07 |

- Entanglement is above the classical mutual information at every cut, as it must be.
- Entanglement is about twice the classical shared information at every cut. Bond dimension 2
  allows at most 1 bit.
- The largest shared information sits at the biometrics and exec cuts. That fits the pipeline
  block reading.
- Bond dimension 2 was chosen at every lag. The held-out log-likelihood was worse at 8 and 16.
- Other lags: `results.json`, `tangling`.

## What this data can and cannot support

- **Can:**
  - The pipeline runs end to end on live data, and the hygiene gate passes on these seats.
  - The linear model has a small 5-minute foresight over the level baselines (P = 0.87).
  - One lead: biometrics is coupled to the pipeline at 5 minutes, in both models.
- **Cannot:**
  - Any claim that the network beats linear: its result flips between the two test windows.
  - Any weakest-cut claim, or any integration number.
  - Any reading of the entanglement profile as a property of Orion. It describes a chain fit on
    234 rows.
- **Why:**
  - Training is 234-311 rows.
  - The time-of-day profile rests on 1-2 days, not 14.
  - Each split has one test window of 9-12 hours.
- **Needed:**
  - About 20x more rows. `grammar_events` keeps about 3 days, so this needs a kept 5-minute seat
    rollup (#3886).
  - Then a re-run on the stable modes after the #3822 re-run (~12-08), as the spec says.

## Reproduce

```bash
/mnt/scripts/Orion-Sapienform/.venv/bin/python scripts/demon_network_replay.py \
  --out docs/superpowers/evidence/2026-11-25-demon-network-preview
```

It takes about 65 seconds on athena (single-threaded BLAS, which the script sets). The data window
moves, so a later run reads different hours.
