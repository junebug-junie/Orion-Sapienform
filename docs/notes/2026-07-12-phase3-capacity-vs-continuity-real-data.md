# Phase 3 research — capacity-pressure vs. continuity-threat in real multi-node data — 2026-07-12

Context: the last open item in Phase 3 ("Mesh embodiment") of
`docs/superpowers/plans/2026-07-12-self-state-mesh-substrate-redesign.md` asks whether
design invariant 7 (`docs/superpowers/specs/2026-07-12-self-state-mesh-substrate-redesign-design.md:99-103`)
— "a difference in kind is not a difference in degree" between Athena losing power
(continuity threat) and Circe going quiet (expected, suppressed) — shows up as a real
pattern now that Atlas/Circe biometrics are live, or whether it's still purely theoretical.
Answered here from live `substrate_self_state`/`substrate_field_state`/`orion_biometrics`
data queried read-only via `docker exec orion-athena-sql-db psql` — not from design intent.

**Verdict: still purely theoretical. There is not yet enough real-world variation to say
anything beyond "revisit again once a real node-down event happens."** Both nodes have been
continuously up for the entire window in which node-attributed data exists. Do not build a
dedicated schema field on the strength of this session's data — there is nothing here that
distinguishes a resource squeeze from a continuity threat, because no continuity threat has
occurred. The one substantive finding is architectural, not empirical: `expected_offline_suppression`
already fires in real data, but it fires as a static per-node-config tag on every reading from
a node marked `expected_online: false`, not as a reaction to an actual offline→online
transition — so even a real Circe outage during this window would not yet be distinguishable
from Circe's ordinary "online but unexpected" steady state under the current implementation.

---

## Timeline: how narrow the real multi-node window actually is

This matters more than any single dimension value, so it's worth establishing first.

- `substrate_self_state` covers `2026-07-09 08:34:56` to `2026-07-12 08:42:08` (34,365+ ticks,
  roughly every 2-10s) — a ~3-day window, but this is the *substrate*'s whole lifetime, not
  the multi-node window.
- `orion_biometrics.node = 'circe'` has **55 rows total, ever** — `min(timestamp) =
  2026-07-12 08:10:03`, `max = 2026-07-12 08:39:08`. Circe biometrics did not exist before
  today; this is its first-ever reporting burst (~29 minutes as of the last query), not a
  resumption after any prior activity.
- `orion_biometrics.node = 'atlas'` has reported since `2025-11-12`, but the single largest
  gap in its entire history is **172 days, 10 hours** between `2026-01-20 20:48:49` and
  `2026-07-12 08:04:36` (today) — i.e. Atlas's biometrics agent was dormant for nearly six
  months and only just came back online today. (A second real gap, 37 days 22h, sits between
  mid-December 2025 and Jan 20 2026.) Both of these historical gaps entirely predate
  `substrate_self_state`'s existence (which starts 2026-07-09), so there is no self-state data
  to cross-reference them against — the self-state substrate simply didn't exist yet.
- Node-attributed evidence in `substrate_self_state.dimensions[*].reasons` (e.g. `"driven by
  X (node: circe)"`) first appears at `2026-07-12 08:16:18`, current as of `08:42:08` — a
  **~26-minute window**, all of it today, all of it after Atlas's 172-day-later restart and
  within ~6 minutes of Circe's very first biometrics reading.

So the entire observable "multi-node self-state" history is about 26 minutes long, and within
that 26 minutes:

- **Circe**: continuously present from its first reading (`08:10:03`) onward. The only gap in
  its 55-row series is `08:34:27 → 08:36:01` (94s, roughly 3x its ~31s poll interval) — normal
  jitter, not an outage. **Circe has not gone offline and come back during this window; it has
  been continuously online since it started reporting**, despite `config/biometrics/node_catalog.yaml`
  marking it `expected_online: false`.
- **Atlas**: continuously present since its `08:04:36` restart, with only the same order of
  poll jitter (max gap ~91s in the last 3 days, matching double its ~31s interval). No real
  gap.

Neither node has actually gone down during the only window in which their absence/presence
could be legible to `substrate_self_state` at all.

---

## Q1 — Does Circe/Atlas absence/presence visibly change `resource_pressure`/`reliability_pressure`/`coherence`, distinguishable from noise?

Checked the full 34,365-tick `substrate_self_state` history for these three dimensions:

| Dimension | Behavior across the whole 3-day window |
|---|---|
| `resource_pressure` | **Pinned at exactly `1.0` for 100% of ticks** (0 of 34,365 below 0.999). Saturated at ceiling the entire time — before, during, and after Atlas/Circe came online. |
| `coherence` | At or above `0.999` for 34,349 of 34,365 ticks; the 16 exceptions bottom out at `0.79`, and all of them (see below) predate the multi-node window and are unrelated to node presence. |
| `reliability_pressure` | An exponentially-decaying value asymptoting toward `0` for the entire window (e.g. `~8e-42` at 07-09, `~5e-136` at 07-12 08:40) — a smooth decay curve, not something that reacts to any event. Its one excursion to `1.0` (191 "strained" ticks, see below) is tagged `"reliability_pressure from field+attention channel synthesis"` — a generic, non-node-attributed reason. |

The 191 `overall_condition == "strained"` ticks (vs. 34,174 `"loaded"`) all fall in hour
buckets between `2026-07-09 16:00` and `2026-07-12 01:00` — **every one of them occurs before
`08:16` on 07-12**, i.e. before any node-attributed reasons exist in this system at all. Since
node attribution went live (`08:16` onward), `overall_condition` has been `"loaded"` for every
single tick — zero `"strained"` ticks overlap with the only window that has real multi-node
data. These strained episodes trace (via `substrate_field_state.node_vectors['node:athena']`)
to a recurring, decaying `contract_pressure`/`catalog_drift_pressure` blip (`0.92 → 0.8464 →
0.778688 → ...`, same geometric ratio every occurrence) that happens roughly every few hours
independent of Atlas/Circe — a generic periodic mechanism, not a node-continuity signal.

**Answer: no. Nothing in this data distinguishes Circe/Atlas absence or presence from noise,
because resource_pressure is permanently saturated, reliability_pressure is a smooth decay
curve untouched by any event, and the only real excursions (the "strained" episodes) both
predate the node-attribution window and trace to an unrelated periodic mechanism.**

---

## Q2 — Has Athena itself ever shown a real degradation traceable to a node-level cause?

Queried `substrate_field_state.field_json->'node_vectors'->'node:athena'` across all 35,545
ticks:

- `availability`: **`min == max == 0.8`** for the entire window (floating-point noise at the
  15th decimal only) — never moved.
- `failure_pressure` exceeded `0.05` in 242 ticks — but every one of these lines up exactly
  with the same "strained" episodes from Q1 (`0.92 → 0.8464 → 0.778688 → ...`, same hour
  buckets, same decay ratio), which is the generic periodic contract/catalog-drift blip, not a
  node-availability event. `reliability_pressure` never exceeded `0.05`; `staleness` never
  exceeded `0.05`. `availability` never dropped below `0.5`.
- `capability_provenance` in the latest tick shows Athena as the sole or dominant source for
  `capability:graph` (`pressure`), `capability:transport` (`confidence`, `contract_pressure`,
  `available_capacity`, `reliability_pressure`), and `capability:orchestration`
  (`pressure`, `execution_pressure`, `reasoning_pressure`) — all in a *good* direction in the
  most recent tick (`reliability_pressure` and `failure_pressure` both effectively `0`,
  `delivery_confidence = 1.0`, `bus_health = 1.0`).

**Answer: no. Athena-attributed evidence never appears in a bad direction anywhere in the
observed window.** Athena hosting hub/bus/postgres/self-state-runtime has never been stress-tested
by this data — there is no host-level Athena degradation event to classify as "continuity
threat" one way or the other.

---

## Q3 — Real gaps in `orion_biometrics` for atlas/circe, and did self-state register them?

Already covered under "Timeline" above, restated directly against the question:

- **Circe**: zero real gaps within its 55-row history (only normal ~31s-interval poll jitter,
  max observed gap 94s). Its entire history is 29 minutes long, so there is nothing to have
  registered as an outage yet.
- **Atlas**: one genuinely large real gap exists (172 days, `2026-01-20` → `2026-07-12`), but
  it entirely predates `substrate_self_state`'s existence (table starts `2026-07-09`). There is
  no self-state tick from that period to check — the runtime that would have registered it
  didn't exist yet. Within the period where both `orion_biometrics` and `substrate_self_state`
  coexist (`2026-07-09` onward), Atlas reported **zero** biometrics until its `08:04:36`
  restart today, and after that restart shows only normal poll jitter — no further gap to
  cross-reference.

**Answer: no gap exists in the window where both the source data and the self-state substrate
being asked to register it are both live.** The one real historical gap (Atlas's 172-day
dormancy) is invisible to this question by construction — the observability system that would
detect it wasn't running.

---

## Q4 — Does `expected_offline_suppression` (weight 0.30) ever fire as nonzero, or has it never fired?

**It does fire — but not the way the invariant's "detects an offline event and suppresses it"
framing implies.**

Traced the code (`services/orion-field-digester/app/ingest/state_deltas.py:80-88` and
`services/orion-field-digester/app/digestion/suppression.py:6-10`):

```python
# state_deltas.py — inside target_kind == "node_biometrics"
if after.get("expected_online") is False:
    out.append(Perturbation(node_id=node_id, channel="expected_offline_suppression",
                             intensity=1.0, label=delta.delta_id))
```

`expected_online` is not a live-detected transition — it's a static label copied from
`config/biometrics/node_catalog.yaml`'s per-node `expected_online` flag
(confirmed at `services/orion-biometrics/app/grammar_emit.py:47`: `"expected online" if
node_profile.expected_online else "expected offline"`). Since Circe's catalog entry is
`expected_online: false`, **every single Circe biometrics reading** — whether Circe is
actually up or down at that moment — carries this tag and emits
`expected_offline_suppression = 1.0`. `suppression.py` then floors that node's `availability`
to at least `0.85` and zeroes its `staleness` whenever this channel is `>= 1.0`.

Confirmed live: `substrate_field_state.node_vectors['node:circe']` shows
`expected_offline_suppression: 0.0` for every tick before real Circe telemetry existed
(default placeholder node from the field-topology config, `availability: 1.0` by default) —
then, from `2026-07-12 08:16:18` onward (right after Circe's first real biometrics reading),
`substrate_self_state`'s `coherence` dimension shows `"driven by expected_offline_suppression=1.00
(node: circe)"` in every one of the 270 ticks since. So the channel is genuinely nonzero in
real data, not dormant — but it has fired continuously since Circe started reporting, as a
blanket "don't penalize this node's presence-or-absence" flag, not as a reactive response to
an actual offline transition. There is a second code path (`state_deltas.py:47-55`, gated on
`delta.operation == "suppress"`, fed by a distinct `node_pressure_suppressed` role in
`orion/substrate/biometrics_loop/pressure_reducer.py`) that could in principle fire on an
actual detected suppression event rather than a static config tag — this was not separately
confirmed firing in the live data and would be the more relevant mechanism to examine if a
real Circe outage occurs.

**Answer: it fires (nonzero, continuously, since Circe's telemetry went live), but as an
always-on per-node-config dampener rather than a reaction to a real online/offline
transition — so its current behavior doesn't yet tell us anything about how the system would
handle an actual Circe outage, only that it pre-emptively immunizes Circe's node vector
against ever being read as unavailable.**

---

## Verdict table

| Question | Real signal found? | What it actually shows |
|---|---|---|
| Q1: resource_pressure/reliability_pressure/coherence track node presence? | No | resource_pressure saturated at ceiling for 100% of the 3-day window; reliability_pressure a smooth decay curve; the only real excursions (191 "strained" ticks) predate the node-attribution window and trace to an unrelated periodic contract/catalog-drift blip |
| Q1b: has Circe gone offline and back? | No | Continuously online for its entire 29-minute-old existence |
| Q2: Athena degradation traced to node-level cause? | No | `availability` constant at exactly `0.8` for the whole window; the only failure_pressure excursions are the same unrelated periodic blip, not a host event; capability_provenance shows Athena-attributed evidence only in good directions |
| Q3: real reporting gaps cross-referenced against self-state | No overlap exists | Circe's history (29 min) has no real gap; Atlas's one real gap (172 days) entirely predates the self-state substrate's existence, so nothing could have registered it |
| Q4: does expected_offline_suppression ever fire nonzero? | **Yes** | Fires continuously (270/270 ticks with the string) since Circe's telemetry began — but as a static per-node-config tag on every reading, not a reaction to a detected offline transition |

## Recommendation

**Do not build a dedicated continuity-threat schema field now.** The honest state of the
evidence is: no continuity-threat event has ever occurred in the only window where it could
be observed (~26 minutes of real multi-node self-state data, during which both Atlas and
Circe have been continuously up). Every dimension that would need to carry this distinction
(`resource_pressure`, `reliability_pressure`, `coherence`) is either permanently saturated or
smoothly decaying in ways unrelated to node presence in the available data, so there's nothing
to design a schema field *against* yet — a field built now would be speculative by the plan's
own standard for avoiding exactly that.

The one thing worth flagging as a real, already-shipped fact rather than a future decision:
`expected_offline_suppression` is currently implemented as a **static immunity tag keyed off
`config/biometrics/node_catalog.yaml`'s `expected_online` flag**, not as a detector of an
actual offline→online transition. This means that today, if Circe genuinely went down and
came back up during a self-state window, the current mechanism would very likely treat it
identically to Circe simply being present-but-unexpected — because the tag fires on every
Circe reading regardless of whether Circe is reachable, and there is no code path (confirmed)
that currently distinguishes "Circe is up, and its up-ness is unexpected" from "Circe just
went down, as expected." If and when a real Circe outage is observed, the follow-up question
isn't "do we need a new field" but "does the existing `expected_offline_suppression` channel,
node-attribution (`reasons`), and Athena's own node vector already make the difference
legible, or does the static-tag design mean a real outage would look identical to Circe's
current steady state?" That is a sharper, evidence-backed version of the original question,
answerable once real variation exists — not now.

**Sequencing suggestion**: leave design invariant 7 open exactly as the plan already frames
it, and re-run this same query set (`substrate_self_state` dimensions/reasons,
`substrate_field_state.node_vectors['node:athena']`/`['node:circe']`, `orion_biometrics` gaps)
the next time either (a) Atlas's biometrics agent goes dormant again for a real stretch (it
already has a precedent: 172 days), or (b) Circe's `orion-biometrics` container stops and
restarts for a real (non-jitter) window — whichever happens first, now that both nodes are
actually being monitored continuously going forward.
