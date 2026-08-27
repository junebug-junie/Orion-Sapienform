# Contested scarcity design — PR report

**Status:** DONE (design mode — spec only, no runtime change)

## Summary

- Adds `docs/superpowers/specs/2026-08-27-claude-quota-contested-scarcity-design.md`.
- Names why the existing allocator, which is good, still cannot produce an opportunity cost: **motor-seconds are scarce but uncontested.**
- Proposes Claude quota as the first resource with a **second claimant** (Juniper), making a refusal cost something to somebody.
- Resolves the two-currency problem without an exchange rate: **absolute floors compose across currencies; relative rankings cannot.**
- Records two live-data errors made while writing it, with the corrected numbers.
- No code, config, schema, or bus change. Nothing to deploy.

## Outcome moved

Nothing runtime. The deliverable is a decision: the 2026-07-07 internal-economy spec's gate ("build only if scarcity actually binds") is answerable for the first time, because a contested budget binds by construction where an invented one never did.

## Current architecture (as verified, not assumed)

| thing | state |
|---|---|
| `orion/autonomy/allocator.py` | **live**, wired at `services/orion-execution-dispatch-runtime/app/worker.py:17` |
| scoring | expected information gain in nats, Normal-Normal closed form, MC-verified |
| `min_nats_per_sec` | **absolute floor**, not a rank cut — this is what makes the two-currency fix work |
| `confidently_harmful` gate | **inert.** `contrast`/`contrast_sd` have no producer anywhere in the repo |
| `ORION_DISPATCH_MOTOR_BUDGET_ENFORCE` | `false` — allowance has never bound |
| trailing 24h | 25,152 dispatches, 83,787 motor-sec vs 129,600 allowance (65%) |
| `dev_economics_ledger_log` | **live**, 1,254 ticks since 2026-08-12 |
| remaining quota | **not readable** — no `claude usage` subcommand; `policy-limits.json` is enterprise policy config |

## Errors found and corrected while writing

Both were mine, both were caught by checking rather than by review.

- **Finding:** I read `dev_economics_ledger_log` with `max - min` per day and reported $27–48/day, 44–122M tokens/day.
  - **Cause:** rows are **per-tick deltas**, not cumulative totals. The producer docstring says so plainly: "the real *growth* in token/word/cost totals since the last check."
  - **Fix:** daily spend is `SUM`. Corrected to **$51–583/day, 0.2–1.2B tokens/day**; 2026-08-26 was **$582.88 / 1.23B tokens** — an order of magnitude above what I first reported, and recorded in the spec so the next reader does not repeat it.
  - **Evidence:** `SELECT sum(total_tokens), sum(total_estimated_cost_usd) ... GROUP BY date`.

- **Finding:** I called 2026-08-24's all-zero day "a genuine rest state" from the metric's own zero.
  - **Cause:** that is precisely the inference the `bus_synaptic_prediction_error` and `node:substrate.route` incidents exist to forbid — a zero can be calm, blind, or decayed, and they look identical from inside the metric.
  - **Fix:** cross-checked against host transcript mtimes — **zero `~/.claude/projects/*.jsonl` files modified** between 18:00 and 22:40 UTC on 2026-08-26, matching 18 consecutive all-zero ticks followed by 4 sessions at 22:47. The instrument was reading correctly.
  - **Evidence:** `find ~/.claude/projects -name '*.jsonl' -newermt ... | wc -l` → `0`.

- **Finding:** I told Juniper mid-session that the internal-economy allocator "was specced, never built."
  - **Fix:** false. `orion/autonomy/allocator.py` (15.8K) and `budget.py` (5.3K) exist and are live-wired, and are substantially better than the 2026-07-07 spec. Corrected in conversation and the spec is written against what is actually there.

## Metric quality gate (CLAUDE.md §0A)

Run in the spec for `quota_fraction_remaining`. Passes 1–6. **One item deliberately left open:** `cost_usd` on a subscription token is notional (API-rate pricing over model-weighted tokens), and its monotonicity with the real rate-limit unit is **UNVERIFIED**. Until measured, the budget is honestly denominated in *dollars at API rates* rather than claimed to track *quota*.

## Schema / bus / API changes

None in this PR. Proposed (not implemented): two optional `Candidate` fields, three `RefusalReason` values, `orion/autonomy/quota_budget.py`, channel `orion:autonomy:quota:allocation`.

## Env/config changes

None. Six keys proposed in the spec, none added.

## Tests / evals / Docker

Not applicable — no code changed. Acceptance checks are specified for the implementer, including that an empty allowance must leave `allocate()` byte-identical to today.

## Restart required

```text
No restart required.
```

## Risks / concerns

- **Severity: medium.** The cap **cannot be enforced.** Hub holds `/var/run/docker.sock`, so a Hub-resident agent is root-equivalent on the host. Resolved direction (Juniper, 2026-08-14) is advisory-plus-reconciliation. The spec keeps it advisory and says so; the risk is a future implementer rendering the number as a ceiling.
- **Severity: medium.** Cold-start scores a new action ~0.9905 nats — maximally informative by construction. Under a *contested* budget that means Orion's first act under scarcity spends Juniper's quota on the action with the least evidence. Mitigation specified (`COLD_CELL_MAX_FRACTION`), not built.
- **Severity: low.** Two independent greedy passes are not jointly optimal over a two-constraint problem. Accepted as cheaper than false precision.

## PR link

<filled on push>
