# Heartbeat seats + field demon: fixed organs for Φ, a real forecast for the heartbeat

Status: design. Juniper asked for this on 2026-11-24. Related: #3296 (integration rung),
#3822 (slow test), #3795 (pipeline coupling floor).

## Arsonist summary

The field demon needs organs that hold still. The data-derived modes are real, but they drift
(#3822: day-to-day match 0.56-0.63, bar 0.7). The heartbeat already has fixed organs: 5 seats, one
per service. The service that emitted an event decides its seat, so a seat cannot drift.

The heartbeat needs a forecast. Its own surprise test says "There is no forecast model"
(`docs/research/preregistration/2026-09-20-heartbeat-surprise-run.md`). Its "surprise" is only how
much its picture changed since the last snapshot.

So the two fit:
- **Seats → demon:** forecast each seat, and measure the cut cost at each seat boundary. Φ then
  has something stable to cut between now, not after 12-08.
- **Demon → heartbeat:** each seat's forecast error is a surprise in the proper sense: what was
  expected, minus what happened.
- **Cross-check:** the heartbeat's neighbour test found that execution↔bus is really linked and
  biometrics↔execution is not (`2026-09-20-heartbeat-neighbor-links-v2-run.md`). The demon finds
  seat-to-seat links with different maths. If both methods agree, that is strong evidence. If they
  disagree, one of them is wrong.

**Snag (Juniper, 2026-11-24): the heartbeat runs on dead metrics.** It must be overhauled before
anything is built on its seats. Step 0 below is that overhaul.

## Current architecture

- `services/orion-heartbeat` subscribes to `orion:grammar:event`.
- It routes 5 producers to boundary seats 0-4 (`app/substrate/routing.py` `ORGAN_SITE_MAP`):
  hub (chat), biometrics, cortex-exec (execution), bus (transport), cortex-orch (route).
- Each `atom_emitted` event applies a gate to its seat. The atom type picks the gate kind.
  `confidence` / `salience` / `uncertainty` set its strength.
- A wall-clock loop relaxes each trajectory and applies a "reheat". The reheat is driven by the
  raw mean |gap z-score| of the `orion_bus_synapse` graph.
- The output is an H1 verdict (boundary/bulk entanglement ratio), published as a ledger-only
  trace. No consumer reads it.

### What is dead or near-dead today (live check, 2026-11-24 ~22:00 UTC)

Live data, last 6 h of `grammar_events`, `atom_emitted` only:

| Seat | Atom strengths that vary | Plain reading |
|---|---|---|
| biometrics | `signal` salience (824 values), `observation` salience (260) | carries real values |
| hub | a few values (4-14) | mostly fixed |
| cortex-exec | none: every atom type has 1-2 fixed values | only **when** it fires carries information |
| bus | `signal` uncertainty only; salience fixed at 0.5/0.6 | mostly fixed |
| cortex-orch | 40 events in 24 h | almost silent |

- **Fixed strengths.** For 3 of the 5 seats, the gate strength is a constant stamped by the
  emitter, not a measurement. Those seats carry timing only.
- **Reheat driver at its floor.** Live `raw_mean_abs_gap_zscore` is 0.53. The calm floor for
  |z| is about 0.80 (√(2/π)), so the reading sits below it, and `reheat_prob` is 0.0035. The
  10-02 check found the same thing: a near-constant hum, with busy edges and quiet edges not
  significantly different (shuffle p=0.16).
- **Narrow coverage.** 13 catalogued grammar producers are not routed to any seat (`/health`
  `catalog_producers_unrouted`), including mind, thought, dream, gpu-pool and actions. The 5 seats
  were "confirmed live" in July. They are not Orion's organs today.
- **Short history.** `grammar_events` keeps about 3 days (since 11-21).

Juniper may know of more dead inputs. Step 0 starts with a full audit, not this partial one.

## Missing questions

1. Which metrics does Juniper mean by "dead"? This doc found fixed atom strengths, a reheat stuck
   at its floor, and 13 unrouted producers. Are there others?
2. Should the seats stay per-service, or become per-organ, using the body atlas organ names
   (`project_body_atlas`)? Per-service is fixed but arbitrary. Per-organ is closer to "who is
   sitting where".
3. Is the MPS (tensor network) part of the heartbeat still wanted, given that its inputs are
   mostly timing? Or do the seats and their counts become the product, with the MPS kept only as
   one reader?

## Proposed schema / API changes

Step 0 (overhaul) and step 2 only. Step 1 changes nothing.

- **Step 0:**
  - Seat map: decide on question 2. Route every live producer, or name each one left out with a
    reason. Merge cortex-orch into execution unless it starts firing.
  - Gate strength: use a measured value where the emitter has one. Where it has none, gate on
    presence only, and say so. Do not use stamped constants.
  - Reheat: either drive it from a live signal that can read calm, or remove it. Removing it
    needs an answer to the 07-28 thermalization problem (see the README).
  - Retire every dead input completely: kill the producer path, not just the reader (CLAUDE.md
    metric gate, "retire the old one completely").
- **Step 2:** a durable 5-minute seat-count rollup. The likely home is the bus-history rollup
  from Patch 1 (#3282), with seat as a column. That keeps seat history past 3 days.

## Files likely to touch

- `services/orion-heartbeat/app/substrate/routing.py`, `service.py`, `bus_synaptic.py`, README,
  tests.
- Emitters with stamped strengths: `services/orion-cortex-exec/app/grammar_emit.py`,
  `services/orion-bus/app/grammar_emit.py`, `services/orion-hub/...` (step 0, only if real
  values exist there).
- `scripts/heartbeat_seats_demon_replay.py` (new, step 1).
- `orion/substrate/demon.py`: no change expected.
- Step 2: the bus-history rollup table and its reducer.

## Non-goals

- No change to how the demon itself works.
- No consumer of the heartbeat verdict. It stays ledger-only until it passes a test.
- No Φ verdict in this work. That is #3296's next step, once seats pass step 1.
- No new seats invented by hand. Seats come from emitter identity or the body atlas, not from a
  taxonomy.

## Acceptance checks

- **Step 0 (overhaul):**
  - Every input has a traced producer (file:line).
  - Every input varies on live data.
  - Every input can return to a calm reading on a quiet night.
  - The audit table is in the PR.
  - The 09-19 lattice-vs-thermometer and 09-20 neighbour-links probes are re-run. They still pass,
    or the PR says what changed.
- **Step 1 (read-only replay, 3+ days):**
  - Per-seat foresight (skill against the better of the clock and persistence) at 5-60 min.
  - Cut cost at each seat boundary, against the time-shift null.
  - Agreement table: demon links against heartbeat neighbour links. Report which seat pairs agree
    and which do not.
  - Pipeline check: report the cut cost of hub, exec, orch and bus as one block against the rest,
    so that pipeline coupling (#3795) is not read as integration.
- **Step 2:** the seat-count rollup has 7+ days of rows, matching `grammar_events` counts exactly
  over a 1-hour spot check.

## Recommended next patch

Step 0, the heartbeat overhaul, starting with the input audit. It is read-only to start, and it
answers missing question 1. Step 1 (the replay) can run on today's seats in parallel, labelled as a
preview, because it is read-only. It must not be read as a verdict until step 0 lands.
