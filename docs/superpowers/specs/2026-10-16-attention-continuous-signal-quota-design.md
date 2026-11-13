# Attention: how a constantly-reading body signal earns attention

Status: proposal (design mode + proposal mode). No runtime code changed.
Date: 2026-10-16. Data window: 2026-10-10 05:57 to 2026-10-16 02:30 UTC (11,552 broadcast frames,
every frame that carries a world-first trace).

## Arsonist summary

Two body signals read all the time: machine load change (`biometrics`) and the share of odd
message-bus links (`bus_synaptic`). The attention contest lets a body signal win when its current
reading is in the top 10% of its own week. A signal that reads every minute is in its own top 10%
about 10% of the time, whatever happens. So these two signals get a quota, not a verdict.

The data says the quota is noise, not news:

- A biometrics win lasts one reading. Its hot stretches have a median of 49 s and never last
  300 s (0 of 583 stretches).
- At a biometrics win, one GPU or power reading jumped by about 0.6. That same kind of jump
  happens in half of all biometrics readings (about 990 per day). GPU load levels at wins are the
  same as at any other time (circe GPU median 0.245 at wins, 0.245 overall).
- bus_synaptic wins at a median of 0.077. Its own producer documents that it cannot tell a
  few-organ fault from noise below about 0.15. Win share per hour does not track RPC timeouts
  (Spearman 0.09 over 140 hours).

The "two instruments disagree" finding is not two instruments. Self-readings (the body text
reverie and stance see) calls the same verdict function on the same history table. They disagree
only because they look at different moments of a one-reading flicker: biometrics won the
02:00:09 frame, and reverie's next tick at 02:01:11 found "no winner this tick (calm)".

Recommendation: a constantly-reading body signal earns attention only when it stays unusual
for longer than one event can explain: hot for more than 300 s within the last 600 s, and still
hot now. This is option (b), with the window taken from the existing episode gate, not a new
knob. One shared function decides "sustained" for the attention contest, self-readings, and the
metacog episode gate, so there is one verdict on the body. That is Juniper's lean (c) in
substance. Not the literal "compete while a metacog episode is open": that keeps bus_synaptic at
7.4% of ticks (see replay).

## Current architecture

- **Contest.** `orion/substrate/attention_broadcast.py:180` `build_substrate_attention_frame`
  calls `world_first_signals` (`:299`) when `ATTENTION_WORLD_FIRST_ENABLED`. Candidates are every
  `node:substrate.*` node with a `prediction_error` (`:342-372`) plus external candidates (chat).
  The field contest uses the same ranking (`orion/attention/field_attention/selectors.py:534`).
- **Eligibility.** `orion/attention/world_first.py:321` `judge_candidate`. A body (internal)
  candidate is eligible at band `high` or `unusual` in its bad direction (`:132`, `:371-376`). The band is
  the strict-below percentile against the node's own 7 days, cut at
  `DEFAULT_BAND_CUTS = (0.5, 0.9, 0.99)` (`orion/substrate/prediction_error_magnitude.py:46`).
  Event-written sources fade over `EVENT_ORIENTING_WINDOW_SEC = 300` (`world_first.py:121`).
  Per-tick sources (biometrics, bus_synaptic) get no time rule at all.
- **Producer.** `services/orion-substrate-runtime/app/worker.py:2933` `_attention_broadcast_tick`;
  magnitudes from the in-memory 7-day history (`:3068` `_prediction_error_magnitudes`).
- **biometrics PE.** `orion/substrate/prediction_error.py:567`: `min(1, mean|delta| / 0.30)` over
  the union of `pressure_hints` keys of every node in `substrate_node_biometrics_projection`,
  previous poll vs current poll (`worker.py:1277`). The keys include GPU util, power, fan, thermal,
  memory, disk, strain, stability, and on athena six `cabinet_*` room-sensor activities
  (proximity, vibration, ambient audio, EM, climate, UV).
- **bus_synaptic PE.** `prediction_error.py:826`: share of bus-synapse edges with `|z| >= 3`
  (FalkorDB `orion_bus_synapse`, EWMA per edge). Its docstring (`:888-900`) says the noise
  baseline max is 0.094 and only a broad mesh event (>= 0.15-0.20) is resolvable.
- **Self-readings.** `orion/attention/self_readings.py:254` `_reading` builds the same
  `node_candidate` and calls the same `judge_candidate`, over the same
  `substrate_node_prediction_error_history` rows (`services/orion-thought/app/self_readings_cue.py:36`).
  `all_calm` means no prompt-reaching signal is eligible right now (`self_readings.py:350-357`).
- **Existing persistence mechanism.** `orion/metacog/self_signal_episode.py:26-34,71-73`: an episode
  opens when a signal is hot for MORE than 300 s within 600 s, and closes after 600 s calm. The 300 s
  is `EVENT_ORIENTING_WINDOW_SEC`, pinned equal by a test. Its docstring already names the problem:
  "high is the top decile of each signal's OWN week and so is hot ~10% of the time by
  construction". `orion/attention/self_signal_outcomes.py:56-66` uses the same dwell for "high
  hours". Attention is the one reader that does not use it.
- **rest_range.** The glossary gives biometrics and bus_synaptic `rest_range: [0.0, 0.03]`
  (`config/field/field_channel_glossary.v1.yaml:382,625`). Its only runtime consumer is the immune
  sweep (`orion/regulation/immune.py:191`), which uses it to flag dead gauges, not to judge a
  single reading. See "Why not rest_range" below.

## What the data shows

### Continuous vs event-driven (live, history table and frames)

| Source | Zero readings (7 d) | Write cadence | Class | Hot-stretch median / p90 | Hot stretches > 300 s |
|---|---|---|---|---|---|
| biometrics | 0.0% | ~60 s, every poll | continuous | 49 s / 73 s | 0% (0/583) |
| bus_synaptic | 0.1% | ~60 s, every tick | continuous | 51 s / 139 s | 3% (17/642) |
| perception (camera) | 65% | every tick, 0.0 when calm | continuous, sparse (world) | 50 s / 87 s | 0% |
| cabinet | 98% | every 30 s | trigger | 606 s / 883 s | 70% |
| execution | 71% | per event | event | 54 s / 141 s | 2% |
| codebase | 84% | per event | event | 296 s / 341 s | 42% |
| chat (node) | 70% | per event | event | 222 s / 314 s | 26% |
| route | 85% | per event | event | 204 s / 344 s | 34% |
| world:chat | n/a | 15 min rate | external rate | 1141 s / 2380 s | 100% |

Bands seen in frames: biometrics `high` 5.7%, `unusual` 0.0%; bus_synaptic `high` 8.5%,
`unusual` 1.05%; perception `high` 9.1%, `unusual` 0.6%.

### biometrics: did anything real happen when it won?

Method: rebuild each poll's per-key change from `orion_biometrics_summary` (31 s cadence per node)
between consecutive biometrics readings. The rebuild is approximate (correlation 0.47 with the
stored value; the projection is fed from grammar events, not the summary table), so it is used
only to attribute which key moved, not to recompute the value.

- 516 biometrics wins. Stored value at wins: median 0.091 (mean key change 0.027). All
  readings: median 0.024, p90 0.068, max 0.361.
- Biggest single mover at a win: power 370, GPU util 117, thermal 22, cabinet room sensors 6.
- At 95% of wins, one key moved by >= 0.30 (median 0.59). Across ALL biometrics readings, 50%
  have a key move >= 0.30: about 988 a day. A GPU job starting or stopping is routine.
- Levels at wins are ordinary: circe GPU util median 0.245 (all 0.245), circe strain 0.203
  (all 0.179), athena thermal 0.257 (all 0.200).
- Hourly win share vs RPC timeouts: Spearman 0.07. Only 138 non-GPU-pool RPC timeouts in 7 days.

Verdict: when biometrics wins, a GPU or power reading stepped, as it does hundreds of times a
day. Nothing unusual happened to the body. A quiet body is the right reading.

### bus_synaptic: did anything real happen when it won?

- 942 wins. Value at wins: median 0.077, p90 0.109, max 0.255. All readings: median 0.032,
  p90 0.063, p99 0.104.
- 17 stretches stayed hot > 300 s. Their peak values: 0.079 to 0.157, and one at 0.255
  (10-15 22:18). Only that one is above the producer's own 0.15-0.20 "broad mesh event" line.
- Hourly win share vs RPC timeouts: Spearman 0.09. vs RPC call volume 0.01. vs chat activity
  -0.07. It does not track a measured bus failure or load.
- `self_signal_outcome_association` already marks bus_synaptic vs rpc_timeout
  `excluded_circular` (both read the same bus traffic), and every other pair is `below_min_n`.

Verdict: a single bus_synaptic win is noise by its producer's own documented limit. A sustained
stretch is at least a persistent mesh state. bus_synaptic is a transport (nervous-system)
signal, category `infra_transport` in the glossary, not a body-load signal.

### Independence check

- **biometrics vs self-readings:** not independent. Same table
  (`substrate_node_prediction_error_history`), same calibration (`NodePeHistoryCache`, 7 d), same
  verdict function (`judge_candidate`). At any one instant they must agree.
- **Why they looked different:** time. A biometrics hot stretch is one reading (median 49 s).
  Reverie runs later, often after refractory suppression, so it reads the next reading. Live
  example: biometrics won the 02:00:09 frame; reverie at 02:01:11 logged "reverie tick skipped: no
  winner this tick (calm)". After the 01:55 restart, self-readings showed `bus_synaptic` or
  `codebase` on all 14 renders. So it is not stuck on calm.
- **biometrics vs bus_synaptic:** independent upstreams (machine telemetry vs bus-mirror edge
  z-scores). Hourly win shares do not move together (Pearson -0.01 over 141 hours).
- **The 207/207 all_calm count from the 10-14 audit:** UNVERIFIED. The container restarted at
  2026-10-16 01:55 and the older logs are gone; self-readings text is not persisted in
  `substrate_reverie_thought`. The structural answer above stands either way.

## Replay (last 6 days of frames)

Script: `scripts/analysis/replay_attention_continuous_quota.py`. Each frame's trace already holds
every candidate's verdict, so the replay re-picks the winner without recomputing percentiles.
Status-quo picks match the recorded winner on 100.0% of frames. Dwell is time-weighted by the
gap between frames (median 49 s, capped at 120 s). The real patch would use readings.

Win share of all 11,552 frames:

| Source | Status quo | (a) continuous internal need "unusual" | (b) sustained 300 s in 600 s, hot now (RECOMMENDED) | (c-literal) compete while metacog episode open | (a+) / (b+) also perception |
|---|---|---|---|---|---|
| no winner | 64.5% | 74.7% | 73.9% | 68.1% | 82.1% / 81.6% |
| perception | 8.3% | 8.7% | 8.7% | 8.7% | 0.6% / 0.1% |
| bus_synaptic | 8.2% | 1.0% | 1.8% | 7.4% | 1.0% / 1.8% |
| biometrics | 4.5% | 0.0% | 0.1% (9 frames) | 0.4% | 0.0% / 0.1% |
| execution | 3.3% | 3.6% | 3.6% | 3.6% | 3.7% / 3.8% |
| world:chat | 3.2% | 3.2% | 3.2% | 3.2% | 3.3% / 3.3% |
| codebase | 2.7% | 3.0% | 3.0% | 2.9% | 3.3% / 3.3% |
| cabinet | 2.2% | 2.2% | 2.2% | 2.2% | 2.2% / 2.2% |
| chat (node) | 1.8% | 1.9% | 1.9% | 1.9% | 2.1% / 2.1% |
| route | 1.4% | 1.5% | 1.5% | 1.5% | 1.6% / 1.6% |

Where the frames that biometrics or bus_synaptic used to win go:

- (a): 88% no winner; then perception 3.7%, execution 3.1%, codebase 2.9%, route 1.2%, chat 1.0%.
- (b): 87% no winner; then perception 4.0%, execution 3.2%, codebase 3.0%, route 1.4%, chat 1.3%.
- (c-literal): 81% no winner; bus_synaptic keeps 4.5%, because 600 s of calm is needed to close.

Examples (b): 10-10 09:16, 09:29, 09:36 bus_synaptic -> execution; 10-10 11:29 bus_synaptic ->
perception.

### Why (b) and not the others

- **(a) "unusual" only** is still a quota, just a 1% one, and it still crowns one reading. It
  also moves a band cut, which is a knob. Biometrics never reached `unusual` in 6 days (its
  recent readings sit below the older part of its week: 0.01% of readings), so (a) silences it by
  accident of drift, not by a rule about what matters.
- **(c-literal)** gives attention the episode's exit hysteresis. A contest should drop a signal
  when it stops being hot. The calm gap is right for writing one journal entry per episode and
  wrong for picking this tick's winner. bus_synaptic would keep 7.4%.
- **(b)** asks one question with a theory anchor: "has this stayed unusual longer than one event
  can explain?" 300 s comes from 225 live onsets (`world_first.py:105-124`): one event's effect
  on the body is gone by 300 s. It is the number the episode gate and the outcome-association
  "high hour" already use. Same rule, three readers, one verdict.
- **(b+) perception** is out of scope. A camera surprise is the world, and orienting to a brief
  change is correct. Its 8.3% top-decile share is a separate question (see Missing questions).

### Why not rest_range

`rest_range` ([0, 0.03] for both) is a hand-set band, checked once against 24 h of data. Used
as a gate it is an absolute floor, which Juniper ruled out (a knob, not a finding). It also does
not fix the problem: 40% of biometrics readings and 55% of bus_synaptic readings over 7 days
sit above 0.03, so the band would still pass routine readings. The immune sweep can keep using it for what it was built for.

## Missing questions

1. Should perception (camera surprise, world) also need persistence, or a different rule? It
   wins 8.3% of frames on single readings. This doc says no: a brief change in the world is worth
   a glance. Juniper should confirm.
2. Should biometrics keep the six `cabinet_*` room-sensor keys? They measure the room (people
   walking past, sound), not machine strain. Removing them is a metric-definition change and needs
   approval or an adjacent metric.
3. atlas has been offline since 2026-08-20 but is still `expected_online` in the projection. Its
   8 keys are frozen and add 8 zero deltas to every biometrics mean. Retire it from the projection
   or mark it offline? Separate patch.

## Proposed schema / API changes

- `orion/schemas/attention_candidate.py` `AttentionCandidateV1` (extra="forbid"): add
  `sustain_window_sec: float | None` and `hot_dwell_sec: float | None`. Consumer-first: the
  judge reads them as optional (None = today's behaviour) before any producer sets them.
- `orion/attention/world_first.py`:
  - `SUSTAIN_DWELL_SEC = EVENT_ORIENTING_WINDOW_SEC` (300) and `SUSTAIN_WINDOW_SEC = 600`,
    imported by `orion/metacog/self_signal_episode.py` so the numbers cannot drift (today they are
    pinned by a test only).
  - New pure `hot_dwell_sec(history, *, now, window_sec, judge)`: seconds in the trailing window
    whose reading was eligible against its own 7 days, each reading counted until the next one.
  - `node_candidate` sets `sustain_window_sec` for internal per-tick level sources (glossary
    `sparsity: per_tick` and not event-written, not trigger).
  - `judge_candidate`: when `sustain_window_sec` is set and `hot_dwell_sec <= SUSTAIN_DWELL_SEC`,
    return ineligible with reason `body high for itself, not sustained (hot Ns of last 600s)`.
    Trace adds `hot_dwell_sec`.
- No bus channel change. No new table. Frame trace gains one field per candidate.
- Flag: `ATTENTION_SUSTAINED_BODY_ENABLED` (default true, ship-on rule), in
  orion-substrate-runtime and orion-attention-runtime settings and `.env_example`.

## Files likely to touch

- `orion/schemas/attention_candidate.py`
- `orion/attention/world_first.py`
- `orion/attention/self_readings.py` (pass history to the dwell; same verdict as the contest)
- `orion/metacog/self_signal_episode.py` (import the shared constants, no behaviour change)
- `orion/substrate/attention_broadcast.py` (pass per-node recent history into `node_candidate`)
- `services/orion-substrate-runtime/app/worker.py` (history deque already in
  `_prediction_error_magnitudes`; expose recent readings), `settings.py`, `.env_example`
- `services/orion-attention-runtime/` (field contest via `NodePeHistoryCache.history`),
  `settings.py`, `.env_example`
- `services/orion-attention-runtime/evals/replay_world_first.py` (add the sustained rule)
- `config/field/field_channel_glossary.v1.yaml` (no change expected; `sparsity` already marks
  per-tick)
- tests: `tests/test_world_first*.py`, `orion/attention/tests/`, `services/orion-thought/tests/test_stance_self_readings.py`

Not touched: `services/orion-thought/app/reverie.py`, `chain.py`, `reverie_narrate.j2` (another
agent is editing them).

## Non-goals

- No change to event-written sources (execution, chat, route, codebase) or their 300 s orienting
  window.
- No change to cabinet (trigger) or world:chat.
- No change to perception (world). Listed as a missing question.
- No change to band cuts, the 7-day window, or any producer formula.
- No absolute floor and no use of `rest_range` as a gate.
- No change to how metacog episodes open or close.

## Proposal mode

- **Capability change:** Orion stops attending to one-minute flickers in machine load and bus
  links. A body signal must stay unusual for over 5 minutes to win. Attention, self-readings and
  metacog episodes give the same answer about the body.
- **Data touched:** reads only `substrate_node_prediction_error_history` (already in memory in
  both runtimes). Writes one extra field to the existing frame trace in
  `substrate_attention_broadcast_log`. No new table.
- **Privacy boundary:** none crossed. Machine telemetry and bus statistics only.
- **Trace that proves it worked:** in `substrate_attention_broadcast_log`, biometrics and
  bus_synaptic candidates carry `hot_dwell_sec`; a `high` reading with dwell <= 300 has reason
  `not sustained`. Over 24 h after deploy: biometrics wins <= 0.5% of frames, bus_synaptic
  <= 3%, every body win has `hot_dwell_sec > 300`. Self-readings logs show the same signal as the
  frame at the same time.
- **Dangerous failure mode:** a real sudden body fault (GPU falls over, thermal spike) is
  silent for its first 5 minutes in attention. Mitigation: hard faults already have their own
  paths (cabinet trigger, arousal `cabinet_hot`, guardian checks, rpc timeout alerts), and a
  fault that lasts over 5 minutes still wins. A second failure mode: the dwell is computed over a
  stale or empty history and reads 0 forever. The trace field makes this visible, and a test
  pins "history missing = today's behaviour, not silence".
- **Disable / roll back:** `ATTENTION_SUSTAINED_BODY_ENABLED=false` restores today's eligibility
  exactly. The schema fields are optional, so leaving them in place is harmless.

## Metric quality gate: `hot_dwell_sec`

1. **Provenance.** Computed from the node's own rows in `substrate_node_prediction_error_history`,
   written by `worker.py:1277` (biometrics) and `_bus_synaptic_tick` (bus_synaptic). Each reading
   is judged by `judge_candidate` (`world_first.py:321`) against its trailing 7 days.
2. **Independence.** Not a new signal. It is a time filter on the same prediction error that
   already competes. It adds no new upstream and is not put next to the PE as a second vote.
3. **Theory anchor.** `EVENT_ORIENTING_WINDOW_SEC` = 300 s (`world_first.py:121`), derived from 225 isolated onsets:
   the body's response to one event is indistinguishable from zero after 240-300 s
   (`world_first.py:105-124`). "Unusual for longer than one event can explain" is the rule the
   episode gate (`self_signal_episode.py:26-34`) and outcome association already use.
4. **Live sanity.** Not degenerate in either direction. It reads 0 at rest (most frames: no hot
   readings in the window) and exceeds 300 s for real stretches: bus_synaptic 17 times in 6 days,
   biometrics briefly (9 frames). It can return to rest: every bus stretch ended.
5. **Existing mechanism.** `self_signal_episode` dwell and `self_signal_outcomes` high-hour
   already compute this from the same verdict. The patch shares one function instead of adding
   a fourth copy.
6. **Reversibility.** Cheap. One flag, two optional schema fields, one trace field. Nothing is
   persisted as a baseline or trained on.

## Acceptance checks

- `pytest` for world_first: a per-tick body signal hot for one reading is ineligible with reason
  `not sustained`; hot > 300 s in 600 s and hot now is eligible; event-written, trigger and
  external candidates are unchanged; `hot_dwell_sec=None` gives today's verdict.
- Self-readings test: same history and time gives the same `unusual` set as `judge_candidate`
  in the contest.
- Constants test: `self_signal_episode.DEFAULT_OPEN_DWELL_SEC is world_first.SUSTAIN_DWELL_SEC`.
- Eval: `services/orion-attention-runtime/evals/replay_world_first.py` with the rule reproduces
  this doc's (b) column within 1 percentage point on the same window.
- Live, 24 h after deploy: biometrics <= 0.5% of frames, bus_synaptic <= 3%, no body win with
  `hot_dwell_sec <= 300`.

## Recommended next patch

One PR, `feat/attention-sustained-body`: schema fields (consumer-first), `hot_dwell_sec` +
judge rule in `world_first.py`, shared constants into `self_signal_episode.py`, history passed
from both runtimes, self-readings on the same rule, flag ON, replay eval updated. Then separately
ask Juniper about the three missing questions (perception, cabinet keys in biometrics, atlas).
