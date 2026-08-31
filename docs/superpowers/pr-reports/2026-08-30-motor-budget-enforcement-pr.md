# PR #2002 — Orion refuses to spend: motor budget enforcement, allocator selection, thermal gate

Merged as `d73533395` from `feat/orion-motor-budget-enforces`.

## Summary

- Flipped the motor-seconds budget and the value-of-information allocator from
  **preview** to **enforcing**. Before this PR both computed a verdict every tick
  and then dispatched everything anyway; the refusal existed only in a log line.
- Added `orion/autonomy/thermal_gate.py` — a pure hysteretic classifier that
  refuses GPU-heavy work when the office cabinet is too warm. Separate trip and
  re-arm thresholds (`32.0 °C` hot, `30.5 °C` re-arm) so a reading hovering on the
  boundary cannot chatter the gate on and off.
- Wired that gate into the reverie visual chain, which is the one action that
  actually spends sustained GPU watts on circe.
- Made the resulting silence *visible*: an all-refused tick now warns after a run
  of them rather than looking identical to an idle system, and every allocator
  refusal is stamped with a reason on the frame instead of being dropped.

## Outcome moved

**Orion can now refuse to act, and the refusal is real.** Prior to this PR the
budget was an observation: `motor_allocator_preview` logged what *would* have
happened and the dispatcher ignored it. The daily allowance (129,600 motor-seconds
= 36 h) was never a constraint on anything.

## Current architecture (before this patch)

`services/orion-execution-dispatch-runtime` ran the allocator in shadow mode.
`orion/autonomy/allocator.py` scored each candidate at
`0.5 * ln(1 + σ²/τ²)` nats and divided by expected cost in seconds to get a
value-per-motor-second rate; `allocate()` returned admitted and refused sets;
`worker.py` logged the preview and then dispatched the full candidate list.

The reverie visual chain ran on a fixed 600 s timer with no environmental gate.

## Architecture touched

| Surface | Change |
|---|---|
| `orion/autonomy/thermal_gate.py` | **New.** Pure function, no I/O, no service dependency. Classifies `normal` / `elevated` / `hot` with hysteresis; returns `degraded=True` and **fails open** on a missing or stale reading. |
| `orion/autonomy/allocator.py` | Refusal reasons carried out of `allocate()` so the caller can stamp them. |
| `services/orion-thought/app/visual_chain.py` | `read_cabinet_temp_c()`, `async evaluate_thermal_gate()`; gate evaluated **before** the single-flight lock so a refusal never holds it; refusal persisted with `terminal_reason="thermal_refused"`. |
| `services/orion-execution-dispatch-runtime/app/settings.py` | `orion_dispatch_motor_budget_enforce=True`, `orion_dispatch_allocator_enforce=True`, `orion_dispatch_all_refused_alert_ticks=20`. |
| `services/orion-execution-dispatch-runtime/app/worker.py` | Enforcement path; tripwire-probe exemption; cold-cost fallback; `motor_allocator_refused_everything` warning; refusals stamped with `ALLOCATOR_BLOCK_REASON`. |

## Design notes worth keeping

**The gate fails open, deliberately.** A thermal gate that fails *closed* on a
missing reading turns a sensor outage into a total loss of Orion's most
expressive action. It returns `degraded=True` so the refusal-that-didn't-happen
is inspectable, rather than silently reading "normal". This is the same rule as
[Never let an absent reading assert a cause] — a signal meaning "no reading" must
not render as physical fact in either direction.

**Hysteresis, not a threshold.** A single 32 °C cutoff on a trailing reading
oscillates: refuse, cool 0.1 °C, admit, heat, refuse. The re-arm at 30.5 °C means
the room has to actually recover before the action comes back.

**The probe exemption.** The dispatch tripwire probe is exempt from the budget.
It is the mechanism that proves the dispatch path is alive; letting the budget
starve it would mean a fully-exhausted budget looks identical to a dead
dispatcher.

## Env/config changes

- Added: `ORION_DISPATCH_MOTOR_BUDGET_ENFORCE`, `ORION_DISPATCH_ALLOCATOR_ENFORCE`,
  `ORION_DISPATCH_ALL_REFUSED_ALERT_TICKS`, thermal-gate keys on `orion-thought`.
- `.env_example` updated for both services; local `.env` synced.

## Tests run

```text
pytest tests/test_thermal_gate.py -q                                        # 113 lines new
pytest services/orion-thought/tests/test_visual_chain_thermal_gate.py -q    # 229 lines new
pytest services/orion-execution-dispatch-runtime/tests/test_allocator_enforcement.py -q
```

## Review findings fixed

- **Finding:** `allocation` was referenced on a path where it was never bound —
  an all-refused tick would have raised `UnboundLocalError` inside the dispatch loop.
  - **Fix:** bind `allocation = None` before the motor branch.
  - **Evidence:** commit `69eeaf355`.
- **Finding:** the enforcing path had a sealed exit — with the budget exhausted there
  was no route by which the tick could complete.
  - **Fix:** cold-cost fallback plus explicit skip accounting.
- **Finding:** the tripwire probe was blinded by the new gate, so the liveness
  signal would have gone dark exactly when it was most needed.
  - **Fix:** probe exemption ahead of the budget check.
- **Finding (live, not from review):** the thermal gate could not reach the sensor.
  It read `127.0.0.1:8080`, which is the *bridge container's own loopback*; the hub
  serving that sensor is `network_mode: host`.
  - **Fix:** tailscale node IP.
  - **Evidence:** commit `0a2249b00`. Config existing is not proof; the read had to
    be run from inside the container to find this.

## Risks / concerns

- **Severity: medium.** Two ceilings are now live at once — the pre-existing risk cap
  and the new motor budget. The motor budget's docstring says it supersedes the risk
  cap ("kill means kill"), but the old cap was not retired in this patch. Follow-up.
- **Severity: low.** `services/orion-thought/evals/test_visual_chain_honesty_eval.py`
  calls the live thermal gate unpatched, so it will fail on the weather if the office
  actually reaches 32 °C. Should read a fixture.
- **Severity: low.** Compose still defaults `ORION_DISPATCH_MOTOR_BUDGET_ENFORCE:-false`
  while `settings.py` defaults `True`. The compose edit was blocked mid-session and was
  not reapplied; the live `.env` sets it explicitly so production is correct, but the
  two defaults disagree.

## Status

DONE_WITH_CONCERNS — shipped, merged, live-verified; three follow-ups above.
