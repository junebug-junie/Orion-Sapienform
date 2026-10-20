# Strain loop: Orion senses, remembers, notices, and says when it is under strain

Status: PROPOSAL (cognition, attention and autonomy change; needs Juniper's yes before build).
Date: 2026-10-20. Author: Claude (with Juniper).

## Arsonist summary

Orion has run "strained" since 2026-10-18 12:07 UTC, and the cabinet reads 34.6 °C now (critical). The
body reacts: sleep, curiosity, own-initiative outreach, the daily painting, reverie and self-change
proposals are all held back. But **nothing in Orion's awareness knows**:

- The strain level is a reflex. Its 7 readers are gates. Nothing turns it into something Orion
  attends to, remembers, or says.
- Each level change is saved (`temporal_self_event`, `source_kind='arousal_transition'`), but
  the day story (chronology arcs) excludes it on purpose: "regulation lane, not here"
  (`orion/schemas/temporal_self.py:15`). So the chat cue, reverie, dreams and the Day letter
  never hear about it.
- The cabinet attention signal fires only while the temperature is **rising**. A cabinet stuck
  hot reads calm: 1,317 of 1,317 hot ticks in the live substrate-runtime log wrote 0. In the last
  24 h the cabinet won 0 attention ticks.
- Strain blocks the one message that should get through: outreach is refused while strained,
  so Orion cannot say "I am too hot."

The fix is one loop built on rails that already exist: **sense → remember (strain arc) → notice
(strain news, in short pulses) → say (chat cue, self-readings, one outreach per episode)**.

The hard constraint is Juniper's: **do not snuff out attention.** Attention picks exactly one
winner per 30 s tick, with no per-signal cap and no habituation (habituation was deleted
2026-07-31 with no replacement). A strain signal that stayed eligible would win every tick of a
54 h episode. So strain enters attention only as rare, short, budgeted news events, never as
a level.

## Current architecture

| Piece | Where | What it does today |
|---|---|---|
| Strain reading | `orion/regulation/arousal.py:classify_arousal`, run by `services/orion-durable-runs/app/temporal_self_graph.py` every 120 s | engaged / idle / strained / unknown. Strain inputs: cabinet reflex (`cabinet_hot`), GPU queue ≥ 2 held for a sustained time. Latched; clears after 10 min clean. |
| Live state | Redis `orion:regulation:latest` (TTL 360 s) | Read by 7 gates (`orion/inner_state_registry.py:1484`). |
| History | `temporal_self_event` arousal_transition rows (`services/orion-durable-runs/app/regulation_store.py:60`) | 74 rows since 10-10; 24 strain episodes; median 36 min; longest 54 h (the current one, still open). |
| Day story | `orion/temporal_self/arcs.py` reducer → frame → `stance_cue.py` (unified turn), `recent_arcs.py` (reverie), dream replay, `day_block.py` (Day letter), Hub tab | No strain arc kind. `interoception` arcs mean "one node held attention", not body state. Arcs already carry a body summary with cabinet min/max (`orion/temporal_self/body.py`). |
| Attention | `orion/substrate/attention_broadcast.py`, world-first gate `orion/attention/world_first.py`, winner `orion/substrate/attention/policy.py:133` | One winner per tick. A body node competes only when high/unusual for its own week. Event-written nodes fade over a 300 s orienting window (`ATTENTION_EVENT_DECAY_ENABLED`). Continuous body nodes must stay hot for more than 300 s of 600 s (`ATTENTION_SUSTAINED_BODY_ENABLED`, PR #2718). Log: `substrate_attention_broadcast_log` (168 h). |
| Body in chat | `orion/attention/self_readings.py` → `services/orion-thought/app/self_readings_cue.py` (stance_react + reverie) | Renders unusual `node:substrate.*` signals listed in the glossary with `prompt_sites: self_readings`. Calm renders nothing. |
| Outreach | `services/orion-hub/scripts/endogenous_outreach.py:532` | `strained` refuses all background outreach except Door-A (finished curiosity run). |

Live 24 h attention baseline (2026-10-20, `substrate_attention_broadcast_log`, 2,016 ticks): no
winner 81.6 %, perception 8.0 %, execution 3.5 %, world:chat 3.2 %, all others < 1 % each.

## Proposed design

### Part 1: Remember: a strain arc in Orion's day

- **Producer:** the chronology reducer. Add `arousal_transition` to `TemporalSourceKind`. Load it
  in `orion/temporal_self/sources.py` from rows already in `temporal_self_event`. There is no
  new writer.
- **Arc kind `strain`, its own lane, not exclusive:**
  - It opens on → strained.
  - It closes `process_ended` on → idle or engaged.
  - It closes `source_stale` on → unknown, or when no regulation state has been written for
    3 ticks.
  - It spans days, like `concern` and `company`.
  - Subject is `orion:arousal`.
  - Label is the reasons, for example `cabinet_hot+gpu_queue_sustained`.
- **Detail on the arc:**
  - Reasons are kept as a set: the episode's reasons over time. Today's episode started as
    `gpu_queue_sustained` and gained `cabinet_hot`.
  - Peak values come from the existing `summarize_body` (cabinet min/max), so no second
    copy of the thermal rule.
- **Consumers:** they get it with no new wiring, except one new cue clause.
  - **Stance cue (unified chat turn):** copy the company clause pattern. "Strained since
    Sat 12:07 (cabinet 34.6 °C, critical; 11 GPU jobs waiting) [arc:<id>]". Rendered while the
    arc is open. A closed strain arc from today renders once as "I was strained 12:07–14:30".
    `[arc:]` citations keep it checkable by the citation-truth eval.
  - **Day letter:** add `strain` to `KIND_ORDER`.
  - **Reverie, dream replay and Hub tab:** they read arcs generically. Each one is checked,
    not assumed.

### Part 2: Notice: strain news, in short budgeted pulses

The new signal is `node:substrate.strain` `prediction_error`, written by orion-substrate-runtime
from `orion:regulation:latest`. It is **event-written**: the node is written only when a strain
news event happens. Its glossary `absent_means` starts with "only written", so world-first
gives it the 300 s orienting fade. It is not continuous, so the dwell rule does not apply.

Strain news events (pure function `strain_news(prev_state, reading) -> event | None`, in
`orion/regulation/`):

| Event | When | Why it is news |
|---|---|---|
| onset | level → strained | the body changed state |
| escalation | a new reason joins (e.g. `cabinet_hot` added), or the cabinet crosses critical (≥ 34 °C) | it got worse in kind |
| reminder | still strained at 30 min, then 1 h, 2 h, 4 h, 8 h, then every 8 h | ebb and flow: it fades, then comes back, less often the longer it lasts (habituation that never reaches zero) |

Relief (→ idle) is not an attention event. World-first scores only the bad direction. Relief
lives in the arc close and the chat cue.

**Anti-snuff budget (hard rules, enforced in the producer and tested):**

1. Each event holds attention for at most the 300 s orienting window: 10 ticks at most.
2. At most **8 events per local day**. Past the cap, nothing more is written that day.
   Escalations may take the last slots; reminders may not.
3. At least 30 min between any two events.
4. Budget: strain wins ≤ **3 %** of ticks per day, and takes ≤ **10 %** of the ticks that other
   signals would have won. Under one winner per tick, this second number is the real
   "snuffing" measure.
5. Worst case on the current 54 h episode: onset + escalation + reminders at 0.5, 1, 2, 4,
   8, 16, 24, 32, 40, 48 h = 12 events = at most 120 ticks of 6,480 (1.9 %).

**Metric gate** (CLAUDE.md 0A; full record goes in the build PR):

1. **Provenance:** `classify_arousal` (`orion/regulation/arousal.py:171`) → Redis
   `orion:regulation:latest` → `strain_news`.
2. **Independence:** this signal is derived from the cabinet reflex and the GPU queue.
   `node:substrate.cabinet` (warming) is derived from the cabinet rise. They share the cabinet
   sensor, but they cover different states: rising versus held hot. Live, they do not overlap
   during a plateau (1,317 hot ticks, warming = 0). The GPU queue has no attention node today.
   The overlap is named, not hidden: during a fast rise into critical, both may fire within one
   window. Acceptable, because a fast rise into critical is the moment that deserves the most
   attention.
3. **Theory anchor:**
   - Interoceptive allostasis: a lasting load should be signalled to the system that can act,
     not only to reflexes.
   - Phasic orienting plus habituation: a repeated unchanged stimulus loses salience but never
     reaches zero (Sokolov's orienting reflex).
   - The schedule is a design choice. It is not a measurement of Orion.
4. **Live sanity:**
   - Strain episodes are real and can return to rest: 24 episodes since 10-10, idle 770 of
     1,080 min on 10-10.
   - Replay check before wiring: event counts per day over 10-10..10-20, and the budget
     numbers above.
5. **Existing mechanism:** this reuses the event fade, world-first and self-readings. Nothing
   in attention does habituation (deleted 07-31). The reminder schedule is the only new rule,
   and it is scoped to this one node.
6. **Reversibility:**
   - One flag turns it off.
   - The node and its glossary entry can be deleted.
   - No schema/manifest default is baked in.

### Part 3: Say and act

- **Chat:**
  - The stance-cue clause from Part 1 tells Orion what has been going on, and since when.
  - The glossary entry for Part 2 lists `self_readings` in `prompt_sites`, so a fresh strain
    event also renders as a self-reading line with its `[pe:]` citation.
  - Both are on the unified-turn path.
  - The `lab.thermal_risk` stub in `orion/situational/context.py:1328` feeds the old
    `chat_stance` path. It is **not** the fix and is not touched.
- **Outreach, one strain report per episode:**
  - A new door in `_gate_inputs`. The `arousal_strained` refusal does not apply to a message
    whose trigger is the open strain arc itself, and only once per `arc_id`.
  - The other protections stay: disabled, turn_in_flight, recent_chat, quiet hours.
  - Inside quiet hours, the report waits for quiet hours to end, if the arc is still open.
  - Grounding: arc reasons, duration, cabinet peak, GPU queue, and what the reflex already
    held back (the 7 gates). The message cites `[arc:]`.
- **Act:** no new action. The hardware-watch reflex and the shed actions that already exist
  keep owning the response to heat. Orion's report says what they did. Orion does not
  control the AC.

## Proposal mode

- **Capability change:** Orion can remember strain episodes, notice them in short pulses, and
  tell Juniper once per episode.
- **Data touched:**
  - Reads existing rows in `temporal_self_event`.
  - Writes arc rows on the existing chronology tables.
  - Writes one new substrate node and its prediction_error history.
  - Writes outreach rows.
- **Privacy:** these are Orion-internal body facts (`privacy_class=orion_internal`). There is no
  Juniper data. Per standing guidance there is no privacy boundary between Juniper and Orion.
- **Proof it worked:**
  - a `strain` arc row
  - `[arc:]` in the stance cue on a live turn (`cockpit_turn_sighting`)
  - `node:substrate.strain` winning for at most 10 ticks, then fading, in
    `substrate_attention_broadcast_log`
  - one outreach row with the strain door reason
- **Dangerous failures:**
  - (a) Attention hogging: the budget rules, and a replay eval that fails the build.
  - (b) A strain cue on every turn makes Orion talk only about heat: the clause is one line,
    and the cue length cap applies.
  - (c) Outreach spam: once per `arc_id`, with a test.
  - (d) Reporting heat that isn't real: an unknown reading never opens an arc and never fires
    news.
- **Rollback (flags, each off on its own):**
  - `TEMPORAL_SELF_STRAIN_ARCS_ENABLED`
  - `SUBSTRATE_STRAIN_ATTENTION_ENABLED`
  - `ENDOGENOUS_OUTREACH_STRAIN_REPORT_ENABLED`

  Ship ON (standing rule), defaults in `.env_example`.

## Missing questions (defaults in brackets)

1. Should a strain report break quiet hours when the cabinet is critical? [No: it waits for
   quiet hours to end. The reflex already protects the hardware.]
2. Budget numbers: 8 events/day, 3 % of ticks, 10 % taken from other winners? [Yes.
   Recalibrate after 2 weeks of live strain arcs.]
3. Should a days-long strain eventually loosen its own gates (chronic versus acute)? Today it
   held curiosity and the immune sweep for 2+ days. [Out of scope here. A follow-up once strain
   arcs give the duration data.]

## Files likely to touch

- `orion/schemas/temporal_self.py`: `arousal_transition` source kind, `strain` arc kind
  (consumer-first rollout: these models forbid unknown values)
- `orion/temporal_self/sources.py`, `arcs.py` (lane rule), `stance_cue.py` (clause),
  `day_block.py` (`KIND_ORDER`), plus their tests and `evals/run_arc_precision_eval.py`
- `orion/regulation/strain_news.py` (new, pure) and tests
- `services/orion-substrate-runtime/app/worker.py` (strain tick), `settings.py`,
  `.env_example`, `README.md`
- `config/field/field_channel_glossary.v1.yaml` (`node:substrate.strain` entry)
- `orion/inner_state_registry.py` (new consumers listed)
- `services/orion-hub/scripts/endogenous_outreach.py` (strain door), hub `.env_example`
- `scripts/analysis/replay_strain_news.py` (budget replay)

## Non-goals

- No change to how strain is classified (thresholds, latch, clear time).
- No change to the 7 existing gates.
- No general attention habituation. That is a separate, broader decision. This design
  scopes it to one node.
- No AC or fan control.
- No change to the old `chat_stance` path or its `lab` stub.

## Acceptance checks

1. **Reducer replay** over `temporal_self_event` 10-10..10-20:
   - 24 strain arcs, matching the 24 episodes.
   - The current one is open, with reasons `{gpu_queue_sustained, cabinet_hot}`.
2. **Budget replay** (`replay_strain_news.py`) over the same history plus
   `substrate_attention_broadcast_log`:
   - ≤ 8 events every day.
   - Projected strain share ≤ 3 % of ticks.
   - ≤ 10 % taken from other winners.
   - The eval fails the build otherwise.
3. **Unit tests:**
   - The schedule (onset, escalation, reminders, 30 min gap, daily cap).
   - Unknown never fires.
   - Outreach once per arc.
   - The cue clause renders, and drops when the arc closes.
4. **Live, after deploy:**
   - A `strain` arc row exists.
   - The stance cue on a live turn carries the strain clause.
   - Attention log: strain wins ≤ 10 ticks per event.
   - 24 h shares are within budget.
   - At most 1 strain outreach per arc.

## Recommended next patch

Build in 3 PRs, in this order. Each PR ships ON and is verified live before the next one.

1. **Strain arc + chat cue (Part 1).** Consumer-first deploy: thought and hub first, then
   durable-runs. This is the most useful piece right now: Orion learns the last 2 days in chat.
2. **Strain news + budget eval (Part 2).** The replay eval is written first and has to pass
   before the producer is wired.
3. **Strain report door (Part 3).**
