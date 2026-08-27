# Calibration finding: dollar-spend-per-window does not predict the limit

> **Status:** Measurement result. Closes the one gate item
> `2026-08-27-claude-quota-contested-scarcity-design.md` deliberately left open,
> **with a negative answer**, and changes that spec's recommended denominator.

## What was left open

The design spec passed items 1–6 of the metric-quality gate and recorded one
open item verbatim:

> `cost_usd` on a subscription token is notional (API-rate pricing over
> model-weighted tokens) and its monotonicity with the real rate-limit unit is
> **UNVERIFIED**. Until measured, the budget is honestly denominated in
> *dollars at API rates* rather than claimed to track *quota*.

It is now measured. **It does not hold.**

## Method

Ground truth for "the limit was actually reached" comes from the transcripts
themselves: `~/.claude/projects/**/*.jsonl` lines carrying `rate_limit_error`
or a usage-limit message. 66 such events total; 15 fall inside
`dev_economics_ledger_log`'s coverage (from 2026-08-12 06:48 UTC) with a full
5h window of ledger history behind them.

For each event, trailing spend was summed from real ledger ticks using
`orion.autonomy.quota_budget.sum_window` — the same code path the budget uses.

## Result

| rate-limit event (UTC) | 5h spend | 24h spend |
|---|---:|---:|
| 2026-08-19 02:25 | $107.01 | $319.34 |
| 2026-08-19 03:54 | $208.05 | $420.39 |
| 2026-08-19 04:28 | $215.04 | $427.37 |
| 2026-08-19 05:21 | $215.04 | $386.12 |
| 2026-08-20 01:46 | $144.37 | $485.34 |
| 2026-08-20 03:31 | $148.68 | $364.29 |
| 2026-08-20 05:38 | $154.01 | $422.81 |
| 2026-08-20 17:44 | **$85.39** | $342.56 |
| 2026-08-20 21:00 | $90.50 | $347.66 |
| 2026-08-25 22:52 | $152.60 | $315.34 |
| 2026-08-26 00:21 | $167.43 | $330.17 |
| 2026-08-26 05:19 | **$289.76** | $590.91 |
| 2026-08-27 00:17 | $171.21 | $619.94 |
| 2026-08-27 03:30 | $196.68 | $465.68 |
| 2026-08-27 04:55 | $150.17 | $456.82 |

**The limit fired anywhere between $85.39 and $289.76 of 5h spend — a 3.4x
spread.** And the highest 5h window observed across the whole period,
**$420.07, did not trip it** — higher than 14 of the 15 events that did.

A threshold cannot be drawn on this axis. There is no value that separates
limited from not-limited.

## Why the sweep looked so good, and why that was misleading

Replaying 826 decision points across 9 days at a $2.50 hypothetical ask:

| allowance / 5h | would-refuse rate | verdict |
|---:|---:|---|
| $150 | 19.4% | binds hard |
| $200 | 9.2% | binds |
| $300 | 1.7% | binds marginally |
| $450 | 0.0% | dead |
| $600 | 0.0% | dead |
| $800 | 0.0% | dead |

Read alone, "$150 refuses 19.4% of asks" looks like a working budget. It is
not — it is a number chosen low enough to bite, and the table above shows the
real constraint does not live at any particular point on this axis. Picking
$150 and shipping it would have been **inventing scarcity**, which is the
precise failure the whole arc exists to delete. The gate caught it.

## Most likely cause

Anthropic's subscription limits are **session-scoped 5-hour windows plus a
weekly limit**. `dev_economics_ledger_log` is **machine-wide**, summing every
concurrent Claude Code session on the host — which is the correct denominator
for the *contested-resource* argument (Orion and Juniper really do draw from
one pool) and the wrong one for *predicting a per-session ceiling*. One session
can exhaust its own window while machine-wide dollars look modest, and many
light sessions can run up large machine-wide dollars while no single one is
near its ceiling. That is exactly the 3.4x spread.

**Caveat, stated rather than buried:** `rate_limit_error` is a mixed
population. Some events are plausibly subscription usage limits; others may be
transient API-side throttling on a different path. The events labelled with an
explicit usage-limit message are the stronger signal. This does not change the
conclusion — a proxy that cannot separate the two is not a proxy — but it means
the spread is an upper bound on how bad the dollar axis is, not a precise
measurement of it.

## What survives, and what does not

**Does not survive:** dollars-per-rolling-window as the budget denominator. The
spec's proposed `ORION_QUOTA_ALLOWANCE_USD_PER_WINDOW` cannot be calibrated,
because there is no value to calibrate it to.

**Survives, untouched:** the contested-resource insight. Orion and Juniper draw
from one pool; a refusal costs somebody something. That argument never depended
on the dollar axis, only on the rivalry being real, and it is.

**Survives:** `orion/autonomy/quota_budget.py` itself. Its arithmetic, its
unknown-vs-zero handling, its fail-closed refusal and its undercount disclosure
are all denominator-agnostic. Only the units change.

## Better denominator: observe the limit, do not predict it

The rate-limit events are themselves the scarcity signal, and they are directly
observable in the same transcripts `claude_code_ingest.py` already reads.

```
was_rate_limited_recently(hours) -> bool | None
```

Compared to a dollar cap this is strictly better on every axis the spec cared
about:

- **No calibrated denominator.** Nothing to discover by hitting the limit,
  because hitting the limit *is* the reading.
- **No unverified proxy.** It measures the constraint itself rather than
  something hoped to track it.
- **Cannot be inflated into non-binding.** A $450 allowance silently stops
  refusing anything; "we were limited 20 minutes ago" cannot be tuned away.
- **Absence still reads as unknown.** No transcripts in the window means
  unknown, not "not limited" — the same property already built.

What it gives up is *anticipation*: it says "the pool is empty now", not "you
are on pace to empty it". That is a real loss and worth stating. The honest
answer is that the anticipatory version requires a per-session ceiling this
data cannot supply, so an accurate reactive signal beats a fabricated
predictive one.

## Recommended next patch

1. Add `was_rate_limited_recently()` to the dev-economics ingest, sourced from
   the transcript events above, with unknown-on-no-data.
2. Re-run the replay gate against **that** signal.
3. Leave `ORION_QUOTA_ALLOWANCE_USD_PER_WINDOW` unset and unbuilt. The reader
   stays useful as a spend *report*; it must not be presented as a quota gauge.

Do not wire the dollar budget into the allocator. It failed its own gate.
