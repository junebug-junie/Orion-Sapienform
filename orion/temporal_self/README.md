# Temporal Self reducer (patch 2)

A pure reducer that turns rows Orion already writes into one chronology of **arcs**: stretches
of the day Orion kept returning to one subject, or one bounded process. Spec:
`docs/superpowers/specs/2026-09-26-temporal-self-design.md` (PR #2369, rev 4). No I/O, no LLM,
no narrative. Patch 3 runs it live: the `chronicle` node of orion-durable-runs'
`temporal_self.update` thread (`services/orion-durable-runs/app/temporal_self_chronicle.py`, see
that service's README).

```python
from orion.temporal_self import ReducerConfig, fold, advance_clock, build_frame, drain_closed_days
from orion.temporal_self.sources import adapt          # row dicts -> TemporalSelfEventV1
from orion.temporal_self.broadcast import tick_from_log_row

state = fold(state, ticks, events, cfg)       # one merged, time-ordered fold
state = advance_clock(state, watermark, cfg)  # nothing strictly before watermark will arrive
frame = build_frame(state, watermark, cfg)
state, closed_days = drain_closed_days(state)
```

| Lane | Subject (exact ref) | Opens / returns / closes |
|---|---|---|
| attention | broadcast winner `source_refs[0]` | K=3 ticks open or resume; K no-winner ticks suspend; R=30 min |
| interoception | `field_dominance_run.target_id` | a run with `tick_count >= min_streak_at_run`; R=30 min |
| conversation | `chat_history_log.session_id`, Juniper's turns only | idle 45 min suspends; R=3 h |
| concern | `loop_id` of a chat-scope trace that resolves to a real chat turn | closes on its verdict; carried across midnight |
| curiosity / reverie / imagery / sleep | run id / chain id / visual chain id / cycle id | one completed process, born closed |

Subject-less events (metacog observations, GPU waits, Orion's own outreach rows, attention
rows, room percepts, memory episodes) bind by time to the arcs open then, or to the day.
Reverie attention rows bind to their chain by correlation id. Late verdicts attach by id and
never reopen an arc.

Gates: `pytest orion/temporal_self/tests` (rules, sources, day boundary, replay identity) and
`python orion/temporal_self/evals/run_arc_precision_eval.py` (real 10-09 rows; `--sweep` for K/R,
`--timeline` for the day). Re-export the fixture with `evals/export_fixture_day.sql` (read-only).
