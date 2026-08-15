# PR report — feedback-credit integrity watch (R5 precondition detector)

Branch: `feat/credit-remnant-detector`

## Summary

- Watches whether a **feedback-credited dimension** could currently be fooled by a producer outage, on the hourly cron that already loads field history.
- Reuses `classify_producer_liveness()` verbatim rather than writing a second detector — the existing module had the right idea at the wrong layer.
- Adds a provenance signal to cover that classifier's declared blind spot.
- **Report-only, deliberately.** See "Why this does not gate".

## Outcome moved — and it reversed my own recommendation

I told Juniper not to build R5's guard: measured on provenance alone, the trap looked like 0.5% of ticks on one channel, never lasting more than a single 2-second tick. **That was wrong, and wrong because I used one signal that structurally could not see the failure.**

Run at the resolution the classifier can actually support, over 6,000 live ticks (3.4h):

| credited dimension | silent stretches | decay-only ticks | longest |
|---|---|---|---|
| `execution_pressure` | 37 | **20.1%** | 72s |
| `resource_pressure` | 15 | **9.1%** | 76s |
| `reliability_pressure` | 4 | **3.7%** | 115s |

These are 60–115 second stretches in which only the decay loop touched the dimension, against a **30-second** feedback window (`windows.field_after_window_sec`). Any feedback frame landing inside one credits the in-flight action for a decay artifact.

The whole-series verdict for all three is `refreshed`. That is the point: it is true about 3.4 hours and says nothing about any minute inside it.

So R5's trap is **not latent — it is routine**, and it can now be designed against 56 measured instances instead of a hypothesis.

## Two things measured that changed the design

**1. The shape signal cannot run at the window being protected.** `MIN_WINDOW_SAMPLES = 20` and `MIN_DECAY_FACTOR = 0.5` ("0.92/tick reaches this in ~28 ticks"), while 30s at the live 2.04s cadence is ~15 ticks. A 30s window fails the classifier twice over. Those constants are principled — a short window makes an ordinary quiet stretch look like an outage, which is the classifier's own stated reason for them — so lowering them would manufacture false positives, not resolution.

The module therefore runs shape at the span it can support (derived from the data's own median tick, reported as `shape_span_seconds`, currently 57s) and **states that it cannot see an outage shorter than that**. Provenance carries the short end, because it needs one tick rather than twenty.

**2. The policy names dimensions; the merge returns channels.** `resource_pressure` is fed by the channel `pressure`. The first version of this module looked policy names up directly in the channel dict and silently reported **0 samples for `resource_pressure`** — the one thing it exists to watch — while printing a clean report. `execution_pressure` and `reliability_pressure` spell the same in both namespaces, which was enough to hide it. An unmappable dimension is now a finding, not silence.

## Why this reuses rather than replaces

`orion/attention/tension/liveness.py` already documents this exact defect and classifies producer liveness. Checking before building (CLAUDE.md 0A) found it is:

- applied to **per-node raw** series, not the merged dimension the loop reads;
- used only by a one-off analysis script, wired to nothing scheduled;
- one verdict per series, so a 60s outage inside a 6h history is invisible.

The classifier is imported unchanged. What is new is the layer it is applied at, the rolling application, and the provenance signal.

## Files changed

- `orion/field/credit_integrity.py` (new)
- `scripts/check_merge_domination.py`: reports on the same state load; `--json` carries a `feedback_credit` block
- `tests/test_credit_integrity.py` (new): 11 tests

## Why this does not gate

At 20.1% / 9.1% / 3.7%, failing on presence would make the hourly cron **permanently red**, and a permanently red gate is ignored — which is worse than no gate. Gating needs a ratchet baseline like the merge-domination one beside it. The numbers to build that against now exist; this patch does not invent one.

Stated in the output itself, not just here, so nobody reads report-only as "nothing found".

## Schema / bus / API changes

None. Read-only over stored field state.

## Env/config changes

None. The window is read from `config/feedback/feedback_policy.v1.yaml` rather than declared, so widening `field_after_window_sec` widens the detector with it.

## Tests run

```
pytest tests/test_credit_integrity.py \
       orion/attention/tension/tests/test_liveness.py -q      22 passed
python scripts/check_metric_lineage.py --gate                 PASS
POSTGRES_URI=... python scripts/check_merge_domination.py --gate
                                                              PASS (exit 0)
```

The load-bearing test is `test_rolling_window_catches_what_whole_series_misses`: it asserts the whole-series verdict is `refreshed` as a **precondition**, then requires the rolling pass to find the outage. If it fails, this module has no reason to exist.

## Review findings fixed

Not yet reviewed — subagent review to follow before merge.

## Restart required

```text
No restart required.
```

Read-only analysis on an existing cron entry.

## Risks / concerns

- **Severity: medium.** `silent_producer` on a credited dimension is *correct behaviour* for a channel with nothing to report — nobody talking to Orion means `conversation_load` decays, and that is right. The defect is crediting a **decrease** during such a stretch, not the stretch itself. This detector deliberately does not judge which is which; that judgement is R5's guard.
- **Severity: low.** Shape resolution (57s) is coarser than the protected window (30s). Provenance covers the gap in principle, but only detects *cleared* provenance, not a producer writing a stale value.

## PR link

<pending>
