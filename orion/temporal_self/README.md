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

## Readers (patch 4: Orion uses its own day)

Spec: `docs/superpowers/specs/2026-10-11-temporal-self-patch4-orion-uses-its-day.md` (PR #2616,
with Juniper's decisions). Both renderers are fixed templates over stored rows; no model writes them.

| Module | Reader | What it renders |
|---|---|---|
| `stance_cue.py` (`project_stance_cue`) | orion-thought, the unified turn's `stance_react` | <= 400 chars: phase and local time, what Orion is on, what it came from, Juniper's last turn, open threads, last sleep. Every clause cites `[arc:<id>]`. Frame missing / invalid / another local day / older than 900 s: `text == ""` with `skip_reason`. No open arc: a rest line. |
| `day_block.py` (`render_closed_day_block`) | Orion's Day letter (Hub gather -> digest) | <= 2,400 chars for one closed day: arc counts, Juniper's turns, top attention subjects, curiosity runs, sleep, images, expectations, refusals, threads, instrument warnings. Lines cite `[arc:<id>]`, whole-day counts `[day:<date>]`. First line: "as recorded at close". |
| `citations.py` | stance pipeline, durable-runs letter graph, Hub letter reread | The gate: unknown `arc:` ids removed and counted; a sentence naming a shown label or local clock time with no `[arc:]` counted and labelled `UNVERIFIED` (never blocked). |

Results land in `thought_decision.day_citations` (Hub turn trace) and
`orion_day_letter.sources.arc_citations`. Kill switches: `TEMPORAL_SELF_STANCE_CUE_ENABLED`
(orion-thought), `TEMPORAL_SELF_DAY_BLOCK_ENABLED` (orion-hub). Fixtures: `tests/patch4_rows/`
(rebuild with `tests/patch4_rows/build_fixtures.py`). Latency eval:
`services/orion-thought/evals/run_temporal_self_cue_latency_eval.py`.

### Thread arc ids and the truth eval (2026-10-12)

`OpenThreadV1.arc_id` names the concern arc a thread is, so the cue cites every open thread
(`2 open threads from conversation [arc:684d45c2010540ed] [arc:fd63c7ab0ad3ecbe].`) even when that
arc fell off `arcs_today`'s 64-arc cap; over 5 threads it says "the 5 most recent". The field is
optional: frames and closed days stored before it still load (the cue falls back to the old
subject_ref lookup and its "N with its record" wording). Rollout is consumer-first: orion-thought
and orion-hub before orion-durable-runs.

`citation_truth.py` + `evals/run_day_citation_truth_eval.py` (`make eval-temporal-self-truth`):
the gate proves a cited arc exists and was shown; this measures whether the sentence around it is
true to that arc. Read-only over `thought_decision.day_citations`, the stored reply
(`chat_history_log`, else `harness_turn_trace.run_artifact.final_text`) and `temporal_self_arc`.
Deterministic only (kind, attention subject, "N min ago" / "for N min" / local clocks, return and
open-thread counts), using the renderers' own phrasing; anything else is **undecidable** and
counted, never guessed. No LLM judge. Output: per sentence match / mismatch / undecidable with
reasons, and per ISO week `truth_rate = match / (match + mismatch)`. A measurement, not a gate.
Calibration tests run it over the renderers' own output (`tests/test_citation_truth.py`).
