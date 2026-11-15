# Attention gap: what Orion does vs what the broadcast says Orion attends to

Status: measurement (read-only), 2026-11-15. Feeds the next step of the 11-04 decision
"attention = the want-ledger competition" (`docs/superpowers/specs/2026-11-04-attention-as-want-competition-design.md`).

## Arsonist summary

Orion is busy with real work 88% of the time. The attention broadcast, which about 30 code paths
still read as "what Orion attends to", has a winner only a third of the time, and when it does the
winner is always a body or world sensor, never what Orion is working on. The 11-04 decision moved
attention to the want market, but the readers were never moved. They still read a sensor
alarm, and the alarm is quiet most of the day because the house is quiet most of the day.

## Measurements (live Postgres, 7 days to 2026-11-15)

**1. The broadcast mostly has no winner.** `substrate_attention_broadcast_log`, 5,025 ticks:
33.9% have a winner. Last 24 h: 438 `none`, 225 `watch`, 2 `defer` (66% none).

**2. Why: every candidate is a sensor, and the bar is "unusual for itself".** The world-first
candidate list (`frame.debug.world_first.candidates`) has exactly 13 sources, all body or world
gauges:

| source | eligible ticks / 664 (24 h) | usual band |
|---|---|---|
| node:prediction.room | 76 | quiet |
| node:prediction.body | 71 | quiet |
| node:substrate.perception | 31 | quiet |
| node:substrate.cabinet, .execution, world:chat | 26 each | quiet |
| world:reading | 17 | quiet |
| node:substrate.strain | 6 | quiet |
| node:substrate.biometrics, .perception_walkway | 5 each | quiet |
| node:substrate.street | 0 | quiet |
| node:prediction.chat_turn | 0 | insufficient_history |
| node:substrate.vision | 0 | insufficient_history, last read ~43 days ago |

No source is Orion's own work: not the curiosity question, not Prime's concern, not the reading, not
the chat topic. "No winner" is therefore an honest "nothing in my body or the world is unusual", not
"Orion is attending to nothing".

**3. Orion is doing something 88% of the time.** Act ledger (`grammar_acts`), sampled every 5 min
for 7 days: an act was in progress (from open to its producer's own close; resolver and
horizon closes excluded, so waiting for Juniper's reply does not count) at 88.3% of samples.
By actor: reading 43.7%, Prime 33.3%, self-inquiry 24.2%, curiosity investigate 23.5%, PR field
notes 15.5%, self-sense eval 14.8%, curiosity replica 10.5%, AI Town 8.3% (they overlap).

**4. During that work the broadcast looks elsewhere.** Temporal Self arcs (`temporal_self_arc`),
7 days: 70 curiosity arcs, 50.8 h. Broadcast ticks inside them: 1,466; with a winner 24.8% (below
the 33.9% baseline). Winners inside curiosity: substrate.execution 61, perception 58,
prediction.body 44, prediction.room 36, world:chat 34, world:reading 23, strain 20, none 1,102.
None of them is the run's subject.

**5. Live example (2026-11-13).** Curiosity run 8fae769c99a4 spent the morning on "my walkway vehicle
records", chose to look at the walkway camera twice (11:48Z, 11:58Z) and found a mismatch. At both
looks the broadcast was `none`, nothing attended, dwell 0.

## Who still reads the broadcast (non-test code)

- **Thought / chat coloring:** `services/orion-thought/app/broadcast_reader.py`, `chain.py`,
  `reverie.py`, `mind_enrichment.py` (unified-turn coloring); `orion/reverie/semantic_lift.py`.
- **Chronology:** `orion/temporal_self/broadcast.py` is "the attention-arc driver's only input",
  so Temporal Self's attention lane records 25 h of attention in 7 days.
- **Act binding:** `orion/grammar/act_ledger.py` `bind_act_to_workspace` (only `hub.chat_reply`).
- **Outreach / agency:** `services/orion-hub/scripts/tension_outreach_trigger.py`,
  `orion/autonomy/agency_episode*.py`, `orion/substrate/endogenous_curiosity.py` (flag off).
- **Reflection inputs:** `orion/cognition/projection_builder.py`, `orion/substrate/felt_state_reader.py`,
  `orion/hub/association.py`, `orion/substrate/system_one_appraisal.py`, `attention_self_model.py`.
- **Observability:** Hub substrate routes, `orion/metrics/liveness.py`, sql-writer `act_downstream_use`.

## What this means

Two things are both true and both useful, and the system has them merged into one word:

- **Alarm:** "is anything in my body or the world unusual?" The broadcast answers this well,
  and "no" most of the time is the right answer.
- **Focus:** "what am I doing and thinking about right now?" Only the act ledger and the want
  market know this. Nothing gives it to the readers above.

Every reader that wants Orion's focus but reads the alarm gets "nothing" two thirds of the day.

## Proposed next patch (design, needs Juniper's yes)

A read-only **current focus** view built from what already exists, no new producer:

- open acts (`grammar_acts`, opened, not yet closed by their producer) with actor, verb,
  `object_ref` and, for curiosity and Prime, the question text;
- the most recently funded want per family (`want_ledger`) for "what I chose to do next".

Then move readers one at a time from the broadcast's winner to this view (focus) while keeping
the broadcast for alarm. Order: Temporal Self's attention lane first (it is a pure recorder, so
the change is visible in the chronology and easy to check), then the unified-turn coloring.
Acceptance per reader: on a 7-day replay, the reader sees a non-empty focus at >= 80% of ticks
where an act is in progress, and the alarm still wins when a sensor is unusual.

Non-goals: no change to the want market's pricing; no new attention policy; the broadcast keeps
running as the alarm.

## Bugs found along the way

- `node:substrate.vision` has been `insufficient_history` for ~43 days (`age_sec` 3.77 M).
- `node:prediction.chat_turn` is `insufficient_history` (chat->chat_turn moved 11-13, #3281; may
  just need history to accrue).

## Queries (run with `docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "..."`)

```sql
-- 1. winner share
select round(100.0*avg((projection_json->>'selected_action_type'<>'none')::int),1), count(*)
from substrate_attention_broadcast_log where generated_at > now()-interval '7 days';

-- 2. candidate sources and eligibility (24 h)
with c as (select x from substrate_attention_broadcast_log l,
  jsonb_array_elements(l.projection_json->'frame'->'debug'->'world_first'->'candidates') x
  where l.generated_at > now()-interval '24 hours')
select x->>'source_id', count(*), sum((x->>'eligible')::bool::int),
       mode() within group (order by x->>'band') from c group by 1 order by 3 desc;

-- 3. time with real work in progress (producer closes only), 5-min samples
with g as (select generate_series(now()-interval '7 days', now(), interval '5 minutes') t),
w as (select actor, opened_at, closed_at from grammar_acts where opened_at > now()-interval '8 days'
      and closed_at is not null and outcome_source='producer' and actor <> 'orion.look')
select round(100.0*avg((exists(select 1 from w where w.opened_at<=g.t and w.closed_at>g.t))::int),1) from g;

-- 4. broadcast inside curiosity arcs
with c as (select began_at, coalesce(ended_at,(arc_json->>'last_seen_at')::timestamptz) e
           from temporal_self_arc where kind='curiosity' and began_at > now()-interval '7 days')
select coalesce(t.projection_json->'attended_node_ids'->>0,'(none)'), count(*)
from c join substrate_attention_broadcast_log t on t.generated_at between c.began_at and c.e
group by 1 order by 2 desc;
```
