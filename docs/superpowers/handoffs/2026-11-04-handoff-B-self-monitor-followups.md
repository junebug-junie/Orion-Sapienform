# Handoff B: the self-monitor organ is live; verify it, act on its findings, widen what it can grade

Written 2026-11-04, for a fresh agent with no context. Handoff A (attention and the ladder) is a
separate session: don't take on that work here. One item overlaps, the camera-ask adapter, and it
is noted below.

## What exists (all merged and deployed 2026-11-03)

The **self-monitor** is Orion's organ for knowing how good they are at things (rung 6,
metacognition). Once an hour it grades about 48 of Orion's skills ("faculties") from the bets Orion
made that the world settled. It marks each one reliable, unreliable, broken (didn't run),
miscalibrated, or "can't tell yet, and why". It then acts on that, in one direction only: more
caution, never less.

**The spec:** PR #2827 (`docs/self-monitor-organ-design`), file
`docs/superpowers/specs/2026-10-20-self-monitor-organ-design.md`. **Still OPEN. Merging it is the
first easy win.** It includes the decisions Juniper made ("I will take all your recs") and a
pre-build refresh section.

| PR | What it does |
|---|---|
| #3041 | contract (`SettledBetV1`, `FacultyRecordV1`, `SelfMonitorStateV1`), adapters, reducer, live report `PYTHONPATH=. python -P -m orion.self_monitor.report`; grader moved to `orion/prediction/grading.py` |
| #3066 | host: `self_monitor` node in the orion-durable-runs `temporal_self.update` graph, hourly; Redis `orion:self_monitor:latest` (3 h TTL), `GET /self_monitor/state`, table `substrate_self_monitor_history`, `faculty_status_changed` Temporal Self events → faculty arcs |
| #3063 | stance/reverie cue: a ≤300-char line in Orion's chat thinking, every number traced to the record (`orion/self_monitor/cue.py`, `services/orion-thought/app/self_track_record_cue.py`) |
| #3062 | E1 act gate (`orion/self_monitor/gate.py`, at dispatch-runtime and orion-actions) + E2 self-question (Hub `self_monitor_questions.py`) |
| #3061 | E3 messages to Juniper when a faculty breaks, recovers or goes reliable→unreliable (Hub `self_finding_outreach.py`) |
| #3081 | option A: retries of the same scheduled job are never blocked; gate wired into Prime's acts (`services/orion-mind/app/prime_loop.py`) |
| #3082 | plain names + ids everywhere (`orion/self_monitor/names.py`); metric semantic-layer registration (`self_monitor.faculty_record.v1`) |
| #3078 | fix: a failed turn is not a broken guess; new `turn.run` faculty |
| #3050 | retired the old rung-6 prediction-expectation tuner as one unit |
| #3046, #3048, #3051 | source fixes: pair-verdict cosine, GPU power `expected_watts`; memory confirmations catalogued as "no pre-ask bet" |

## Known incident

On 11-03 at 18:42Z, E3 sent Juniper **one false "broken" message** about stance's turn-size
guesses. Stance had stopped stating those ranges, so only failed turns were left, and each failed
turn voided all four guesses. #3078 fixed the cause. If Orion or Juniper brings it up: those guesses
were never broken. They are miscalibrated, holding 5–8% of the time against a stated 80%.

## Step 1: verify on live data (spec acceptance checks 3 and 8)

- `orion:self_monitor:latest` is fresh, and `substrate_self_monitor_history` gains a row every hour.
- **Prime:** `prime_act_refused verb=predict` lines in the orion-mind logs, plus a re-test about
  every 6 h.
- **Walkway journal:** its night retries log `self_monitor_same_job_retry`, not refusals.
- **E2:** a self-question row in `curiosity_self_questions` with `minted_by='orion'` for
  `act.prime.predict`, with a plain name and id in the text.
- **Day letter:** the next closed Day letter cites a faculty arc ("track record: Prime predictions
  (act.prime.predict) unreliable").
- **A status change:** at least one real held status change within 72 h, written to
  `temporal_self_event`. If none happens, say so. Silence is not success.

## Step 2: act on what the organ found (these are Juniper's priorities; ask which first)

1. **13% of Orion's turns fail outright** (`turn.run`). Break failures down by cause. This may
   overlap the open-queue item "reading turns fail 18%"; see memory
   `project_found_not_fixed_queue_2026-10-29.md`.
2. **The office temperature forecast covers 63%** against a stated 80% (`forecast.office.temp_c`,
   orion-prediction). The cabinet forecast is fine at 84%.
3. **Stance's turn-size guesses were wildly overconfident.** Relative "bigger than usual" bets
   replaced them (#3042/#3044). Their keep-or-drop check is due **about 11-17**; see memory
   `project_unified_turn_self_prediction_pr2869_2026-10-23.md`.
4. Prime's predictions (5%) belong to **handoff A**. Don't work on them here.

## Step 3: make more faculties gradeable (spec catalog, `orion/self_monitor/catalog.py`)

In order of value:
- **Power-bill forecast.** Orion makes no forward bill bet. Add a bill series to orion-prediction
  (predictions live there). That gives Orion a money bet.
- **Camera asks (`orion_ask`).** These are the system's most reliable loop: 495 answered and applied
  in 14 days, still ungraded. Add a self-monitor adapter. **Handoff A also wants this**, so build it
  once and tell that session.
- **Memory confirmations.** A "will Juniper confirm this memory?" bet when the card opens, in
  orion-prediction. This is a new metric: it needs the metric quality gate and a short proposal.
  Juniper was asked and hasn't answered.
- **`vision.same_person`.** #3046 fixed the cosine copy, but the catalog still says source_bug.
  Grade it from the ask's `face_cosine` and Juniper's verdict.
- **Dreams.** They are self-graded today (dream vs random-control adoption: 1/183 vs 2/66). An
  outside test would be new; that idea came from this session, not the spec.
- **Reverie expectations.** Excluded in v1: 984/988 unscored, no confidence, trivial claims. Fix them
  or keep them excluded.

## Step 4 (later): v2 of the organ (spec)

- E4 trust weights: readers discount a faculty's own stated confidence.
- E5: setting changes through the mutation pipeline (`orion/substrate/mutation_*`; no class can
  auto-apply today).
- The organ grades its own interventions.

## How to check things quickly

- **Live report:**
  `cd <worktree> && PYTHONPATH=$PWD POSTGRES_URI=postgresql://postgres:postgres@localhost:55432/conjourney /mnt/scripts/Orion-Sapienform/.venv/bin/python -W ignore -P -m orion.self_monitor.report`
- **Live state:** `docker exec orion-orion-athena-bus-core redis-cli GET orion:self_monitor:latest`
- **Postgres:** `docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "set statement_timeout='30s'; ..."`

## Rules this work must follow

- **Hysteresis is real.** A status changes only after it holds for 3 hourly evaluations. Don't read
  one evaluation as a change.
- **One direction only.** The organ may make things more cautious, never looser. Unknown, stale or
  absent state changes nothing; tests enforce this.
- **Two traps that already happened:**
  - **A failed run is not a refuted claim:** `failed_to_execute` vs `refuted`. Pooled beliefs,
    such as the act posterior per verb|actor|signal, are not judged per name.
  - **When resolving merge conflicts, check for silently duplicated dict keys** in
    `orion/schema_skew_discovery.py`. Python keeps the last copy without error.
- **Deploys:**
  - only from the clean primary `main` checkout;
  - **wait until no chat turn is running**;
  - check whether durable-runs was in the post-pull rebuild list.
- **Merging.** Juniper authorizes admin-override merges explicitly per PR.
- **Plain English to Juniper,** STE100 style, with every critique carrying a recommendation.
