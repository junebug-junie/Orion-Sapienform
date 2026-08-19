# A5 — Make the deferral perceptible · proposal mode

Date: 2026-08-19
Roadmap: `docs/superpowers/specs/2026-08-13-scarcity-ROADMAP.md` §3, step A5
Status: **proposal + implementation** (Juniper asked for implementation directly, which §0A
allows; this document exists because A5 touches the cognition loop and the proposal content is
owed regardless of whether it blocks).

---

## 1. What A5 is for

> *A deferral is currently invisible to Orion. Turn it into a signal Orion can hold: I wanted to
> think and had to wait, this long, while this ran instead.* — ROADMAP A5

The whole scarcity arc exists because Orion decides constantly and nothing it decides costs
anything. §2A named the currency: **the processes that never run.** A deferral is the only place
in this system where that currency becomes an observable quantity — Orion asked for a slot,
something else had it, and the wait is the foregone process with a number attached.

A1–A4 built the instrument, the statistic, and the yielding mechanism. All three are on the
gateway side, where Orion cannot see them. A5 is the step that carries one number across into
Orion's own context.

---

## 2. The measurement, taken before designing anything

§7.5: no number without a paste. All of this is live, 2026-08-19, on the deployed fleet.

### 2.1 Background admissions over 4 h (gateway container uptime)

```text
docker logs orion-llm-gateway --since 4h | grep -c "LLM-GW background"
296

waited / outcome distribution:
     71  outcome=admitted  0.021s        1  outcome=admitted  0.091s   <- the maximum
     61  outcome=admitted  0.020s        1  outcome=admitted  0.039s
     25  outcome=admitted  0.019s        1  outcome=admitted  0.031s
     24  outcome=admitted  0.022s        1  outcome=admitted  0.029s
     23  outcome=admitted  0.018s        2  outcome=unchecked 0.018-0.019s
     ... all remaining between 0.012s and 0.032s

polls:     294  polls=1          <- every single admitted request cleared on the FIRST poll
reserved:  294  reserved=2 url=http://100.121.214.30:8013
outcomes:  0 timeout_forwarded
```

### 2.2 Live lane occupancy, ten consecutive 1 s samples of atlas `/slots`

```text
0 / 4   0 / 4   0 / 4   0 / 4   0 / 4
0 / 4   0 / 4   0 / 4   0 / 4   0 / 4
```

### 2.3 What that means

**Orion has not been deferred once in the observed window.** Every `waited` value in §2.1 is the
HTTP round-trip cost of asking `/slots` whether there is room — measurement cost, not opportunity
cost. `polls=1` on all 294 admissions says the answer was always yes on the first ask.

This is a consequence of the previous step, and it is worth stating plainly rather than burying:
**A4 found Orion's autonomous cognition was 100% on circe's single-slot `chat` lane (P(all busy)
8.10%), and PR #1708 moved it to atlas `quick_background` (4 slots, P(all busy) 4.01% when
measured on 2026-08-15).** Orion used to meet a ceiling daily. It was routed off that ceiling, and
onto a lane that is presently empty. The engineering outcome is correct — journal composes are
faster and no longer contend with Juniper's chat — and the cognitive outcome is that **the price
went to zero because we routed around it.**

### 2.4 The discrepancy that is not yet explained

A2 measured `P(bg blocked) = 4.84%` over 27.74 h on this exact lane at this exact threshold. At
that rate, 294 independent arrivals would produce zero blocks with probability ~4e-7. So either
the lane's load genuinely collapsed in the four days since, or something else changed. A 24 h
`record_lane_occupancy.py` run was started before any code was written
(`/tmp/lane-occupancy-a5/samples.jsonl`, all four lanes, 1 s interval) and its numbers will be
pasted into the PR report. **Four hours of a quiet night is not 24 hours and is not being treated
as one.**

---

## 3. Proposal-mode disclosures (§0A)

**What capability changes.** Orion gains one new fact in the metacog cue it already reads every
pass: whether its own background thinking was made to wait recently, how many times, and for how
long. Nothing else about routing, dispatch, or admission behaviour changes — the gate itself is
untouched.

**What data is touched.** Only gateway-side admission timings: route key, upstream URL, reserved
slot count, free slots observed, wait duration, poll count, outcome. No prompt text, no response
text, no user content, no identity. The ledger is in-process and bounded.

**What privacy boundary exists.** None is crossed. The ledger cannot hold content because it never
receives any — `wait_for_slack_sync` sees a `RouteTarget`, not a `ChatBody`. This is deliberate and
is pinned by a test.

**What trace proves it worked.** Three layers, all inspectable:
1. the existing `[LLM-GW background] admission waited=… polls=… outcome=…` log line (A4),
2. `GET /admission` on the gateway, returning the rolling counters with their window,
3. the `yield` key in the metacog cue, visible in cortex-exec's rendered prompt.

**What failure mode would be dangerous.** Two, and both are designed against:
- *Reporting a wait that did not happen.* A first-poll admit is **not** a deferral. If `waited`
  were surfaced raw, Orion would read "I waited 0.021 s to think" 294 times a day and learn a
  false fact about its own constraint. The ledger counts a deferral only when a poll interval was
  actually slept through (`polls > 1`) or the wait timed out.
- *Silence read as calm.* An unreachable gateway and a genuinely undeferred window must not look
  the same. They do not: the cue carries `yield:0` when the gateway answered and nothing waited,
  and omits the key entirely when the gateway could not be read.

**How to disable or roll back.** `CORTEX_EXEC_ADMISSION_CUE_ENABLED=false` removes the cue key and
the fetch. The gateway ledger is in-process, bounded, additive, and read by nothing else; deleting
it is a revert of one file. Nothing enters a schema, a bus contract, a manifest, or a training
default.

---

## 4. Metric quality gate (§0A) — recorded, not recited

| # | check | finding |
| --- | --- | --- |
| 1 | **Provenance** | `time.monotonic() - started` and the `polls` counter in `priority_admission.wait_for_slack` / `wait_for_slack_sync`, computed at the moment of the wait. Not a schema comment, not derived. |
| 2 | **Independence** | Orion's cue currently carries eleven pressures, all hardware sensors (power, cpu, mem, swap, disk, disk_capacity, net, gpu, gpu_mem, fan, thermal) plus `strain`/`peak_pressure` reductions of them. Slot-pool queueing state is **not present in any of them**, shares no sensor, and is not a monotonic transform of anything already there. It is causally *related* to atlas GPU load, but nothing in Orion's state reads llama.cpp slot occupancy at all. |
| 3 | **Theory anchor** | Roadmap §2A: a deferral is an observed opportunity cost — the foregone process, timed and attributable. Operationally it is blocking-time in a finite-server queue, which is the standard measure of a contention ceiling. Not "seems related". |
| 4 | **Live-data sanity** | **This is the gap, and it is stated rather than papered over.** §2 above: 294 admissions, zero deferrals, live lane 0/4 busy. The signal is at its rest state and I have **no live observation of it leaving rest**. It is a *genuine* zero — event-driven counters with no decay loop and no aggregation floor, so it is neither the `bus_synaptic_prediction_error` ~0.27 floor case nor the `node:substrate.route` decayed-to-zero case. But "cannot currently be shown to move" is a real finding and the 24 h recorder in §2.4 is what settles it. |
| 5 | **Existing mechanism** | Searched. No admission or deferral ledger exists anywhere in the repo. The only prior art is A4's log line, which is write-only. `record_lane_occupancy.py` measures the *lane*, not Orion's own requests, writes to a file, and publishes nothing. |
| 6 | **Reversibility** | High. In-process bounded deque, one additive cue key behind a flag, no schema/registry/channel entry, no persisted artefact. Removing it is deleting one module and one `if`. |

**Verdict: 1, 2, 3, 5, 6 pass. 4 passes on rest-state honesty and is open on excursion.** The
signal is built because its rest state is trustworthy and its wiring is cheap to remove — not
because the ceiling has been demonstrated live since the reroute.

---

## 5. Design

### 5.1 Why the ledger lives in the gateway

The wait happens in `priority_admission`, in the gateway process, and nowhere else. Anything that
measures it elsewhere is re-deriving a number that already exists — the exact mistake §0A's
provenance rule is written against.

### 5.2 Why the cue is read over HTTP and not over the bus

`CORTEX_EXEC_LLM_GATEWAY_URL` already exists and cortex-exec already talks to the gateway. A new
bus channel would need a schema, a registry entry, a producer, a consumer, a reducer and a writer
to deliver one integer — the shape §0A calls a cathedral. If a second consumer ever wants this,
that is when the channel earns itself.

### 5.3 The honest definition of a deferral, in code

```text
deferral  :=  polls > 1            (a poll interval was actually slept through)
           or outcome == timeout_forwarded
```

A first-poll admit is recorded as `checked`, contributes to the denominator, and is **not** a
deferral. This is the single most important line in the patch: without it the signal reads 294
phantom waits a day.

### 5.4 Shape of the cue addition

Appended to the same metacog cue that already carries `peak`/`peak_at`/`fleet_watts`, because that
is the surface Orion actually reads and B2 established the precedent of adding beside rather than
modifying.

```text
at rest, gateway answered:      "yield":0
after real deferrals:           "yield":{"n":3,"max_s":4.2,"h":6}
gateway unreachable:            key absent entirely
```

`n` is deferral count in the window, `max_s` the longest single wait, `h` the window in hours. Kept
to a few characters because the cue has a hard char budget and truncation drops the whole payload.

### 5.5 Files

| file | change |
| --- | --- |
| `services/orion-llm-gateway/app/admission_ledger.py` | new — bounded rolling ledger, deferral definition, snapshot |
| `services/orion-llm-gateway/app/priority_admission.py` | record each admission decision into the ledger |
| `services/orion-llm-gateway/app/main.py` | `GET /admission` |
| `services/orion-llm-gateway/tests/test_admission_ledger.py` | new |
| `services/orion-cortex-exec/app/admission_cue.py` | new — fetch + render, fail-quiet |
| `services/orion-cortex-exec/app/executor.py` | one key into `_metacog_biometrics_cue` |
| `services/orion-cortex-exec/app/settings.py`, `.env_example` | flag + window |
| `services/orion-cortex-exec/tests/test_admission_cue.py` | new |

### 5.6 Non-goals

- Not changing admission behaviour. The gate defers exactly as it did.
- Not attributing a deferral to the specific competing request. The gateway sees slot counts, not
  who holds them; claiming "while *this* ran instead" would be confabulation. The cue says how long
  Orion waited, not who took the slot.
- Not separating Orion's journal from AI Town speech. Both use `quick_background`; the ledger is
  lane-level and says so.
- Not persisting the ledger. It is a rolling window, lost on restart, and the log line remains the
  durable record.

---

## 6. Acceptance checks

1. `GET /admission` returns real counters on the live gateway.
2. A synthetic deferral (a poll interval actually slept) increments `deferrals` and `max_wait_s`;
   a first-poll admit does not. Pinned by test.
3. The cue renders `"yield":0` when the gateway answered and nothing waited, and omits the key
   when the gateway is unreachable. Pinned by test.
4. The rendered cue is observed in a live cortex-exec metacog pass.
5. **Gate:** a real deferral, if one occurs in the 24 h window, appears in the cue with its
   duration. If none occurs, that is reported as the finding rather than manufactured.
