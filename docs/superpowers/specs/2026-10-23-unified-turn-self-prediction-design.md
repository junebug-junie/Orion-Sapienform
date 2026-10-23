# Unified turn: stance places bets, the turn settles them

Status: design (proposal mode), 2026-10-23. No code changed.
Storyboard and option set: https://claude.ai/artifact/71ELFgEA56xpMJSBB76dRy
(options A–F; G was dropped by Juniper on 10-23).
Depends on: self-monitor organ design (PR #2827, open). Body economy tiers (PR #2855, merged).

## In one paragraph

Before every unified turn, stance already writes a plan (imperative, tone, task slice).
Nothing ever checks that plan against what Orion then did. The harness also uses it as an
order. This design adds a small, checkable **self-prediction** to the stance output:
will I ask a question, will I fetch an outside source, how many steps will I take, how
long will my reply be. Each prediction carries the confidence Orion states before acting.
The prediction is hidden from the motor, so it stays a prediction and does not become an
order. After the turn, plain code settles each prediction against the turn record. The
results go into the self-monitor organ as `SettledBetV1` bets. The organ's stance cue
brings them back as self-knowledge ("I think I'll answer in about 5 steps; I usually
take 14"). This is the efference copy closing into a loop: predict your own act, then
find out.

## Arsonist summary

- Stance is a narrator today. A model writes prose about Orion's state, a second model
  reads it, and no outcome ever comes back to it.
- The gate half never decides anything: in 30 days, 0 refusals and 10 defers, all of them
  broken stance output (`empty_imperative`).
- What comes back into later turns: conversation strain and length (chat prediction
  error) and the day so far (temporal self). What does not come back: what Orion meant
  to do, whether Orion did it, and how it landed.
- Orion already has a coarse self-prediction that nobody can grade: Mind's
  `expected_depth` (`shallow`/`deep`/`unknown`, `orion/schemas/chat_stance.py:56`). It has
  no stated confidence. It is spliced into the motor's user message
  (`orion/curiosity/role_teach_disclosure.py`), which makes it an order. When missing it
  is filled in as `"deep"` (`role_teach_disclosure.py:45`), which is a made-up value.
- Fix: one small structured bet per turn, never shown to the motor, settled by code, fed
  into the one organ that already owns "how reliable am I" (#2827). No new organ, no new
  node `prediction_error` (#2827's E2 guard forbids that leak).

## Current architecture (traced 2026-10-18, live data 2026-10-23)

1. Hub `execute_unified_turn` (`orion/hub/turn_orchestrator.py:965`) does the appraisal
   and the association, then calls the Thought service's `stance_react` RPC.
2. orion-thought `run_stance_react` (`services/orion-thought/app/bus_listener.py:556`)
   gathers the day-so-far, body readings, personality and Mind (up to 3 model calls), then
   cortex-exec runs the `stance_react` verb (`orion/cognition/prompts/stance_react.j2`), one
   model call with JSON output.
3. `parse_stance_react_payload` (`orion/thought/stance_react.py:92-242`) clamps fields, and
   policy sets the disposition (`policy_refusal.py:17`).
4. `ThoughtEventV1` goes to Hub (RPC reply) and to `orion:thought:artifact` → sql-writer →
   `thought_decision`.
5. The harness compiles the motor prompt with the imperative, tone and slice under
   "STANCE GUIDANCE" (`orion/harness/prefix.py:305-338`). Claude runs as the motor.
6. The finalize judge grades the draft against the message **and the imperative**
   (`orion/cognition/prompts/harness_finalize_reflect.j2`).
7. The governor's `HarnessRunV1` goes to `harness_turn_trace.run_artifact` (one row per
   correlation_id). It holds `step_count`, `final_text`, `source_fetches`,
   `exit_code` and `compliance_verdict`.

**Live outcome data (14 days, `thought_decision` joined to `harness_turn_trace`):**

| what | value |
|---|---|
| stance rows with a turn record | 1,211 of 1,244 (97%) |
| motor steps p10 / p50 / p90 | 5 / 14 / 95 |
| reply length p10 / p50 / p90 (chars) | 444 / 2,350 / 8,087 |
| replies that ask a question | 27% |
| turns that fetch an outside source | 35% |

**Who runs stance (30 days, 2,472 rows):** self-sense-eval 864, curiosity 479, world-pulse
read 431 + stage 2 315, Hub chat 223, journal 195, outreach 7. **Only about 9% are
Juniper's chats.** Juniper has not chatted since 10-21.

**Mind health (72 hours, orion-thought log):** 27 runs `meaningful_synthesis`, 44
`fallback_contract_only` (Mind returned nothing). 16 of the 44 were 60-second timeouts on
route `metacog_turn` in `orion_world_pulse_read_stage2`. None of the 72-hour runs was a
Juniper turn. This is a separate finding, recorded on the board, and not fixed here.

## Design

### A1. Producer: stance writes a bet

New optional field on `ThoughtEventV1`:

```python
class TurnSelfPredictionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    asks_question_p: float | None        # p that the final reply asks Juniper/the reader a question
    fetches_outside_p: float | None      # p that the turn fetches any outside source
    steps_lo: int | None                 # motor step range Orion expects, at 80% stated coverage
    steps_hi: int | None
    reply_chars_lo: int | None           # reply length range, at 80% stated coverage
    reply_chars_hi: int | None
    coverage: float = 0.8                # fixed in v1; the prompt states it, Orion does not choose it
```

- `stance_react.j2` gains a short block that asks for these numbers. It says plainly that
  they are a private guess about Orion's own behaviour, that they will be checked, and
  that nobody is told to meet them.
- Parser rules (`stance_react.py`):
  - p values are clamped to [0.02, 0.98];
  - a range with lo > hi, or a negative value, drops that pair;
  - a missing value stays `None`. **Never filled in, never defaulted.** A missing value
    makes that bet `ungradeable:no_stated_confidence` in the organ.
  - A missing or broken `self_prediction` never defers the turn.
    `enforce_thought_stance_quality` does not look at it.
- Env: `STANCE_SELF_PREDICTION_ENABLED=true` in orion-thought and cortex-exec
  `.env_example` and settings, shipped on.

### A1. The motor never sees it

- `compile_harness_prefix` and `harness_motor_instruction` do not render
  `self_prediction`. A test asserts that no rendered prompt contains any of its values or
  field names.
- The finalize judge prompt does not see it.
- **Why:** if the motor sees "I expect 5 steps", the prediction turns into an order and
  fulfils itself. The bet then measures obedience, not self-knowledge. An efference copy
  is a copy of the command. It is not the command.

### A1. Persist

- Additive migration: `thought_decision.self_prediction jsonb`, with header
  `-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer`. Additive DDL only, so it fits the
  post-pull auto-apply allowlist.
- sql-writer maps the field. `harness_turn_trace` already holds every outcome. No change
  is needed there.
- Consumer-first order: sql-writer and every `orion:thought:artifact` consumer must accept
  the field before thought publishes it. `ThoughtEventV1` uses the default
  `extra="ignore"`. Each consumer's own model still gets checked in the patch.

### A2. The judge stops grading obedience (separate patch)

`harness_finalize_reflect.j2` stops asking "does the draft match the imperative". It asks
"does the draft answer the message, and is it grounded in what was fetched". The
imperative stays as context for tone. This is a behaviour change to reflection, so it is a
separate PR with its own before/after eval: the stance-scope live eval
(`orion/harness/evals/stance_scope_live_eval.py`) plus a sample of 50 recent turns
re-judged both ways.

### A3. Settle: an adapter into the self-monitor organ

Pure adapter in `orion/self_monitor/adapters.py` (from #2827):
`turn_self_prediction_bets(decision_row, trace_row) -> list[SettledBetV1]`.

| faculty | claim_kind | stated | upheld when |
|---|---|---|---|
| `turn.self.asks_question` | probability | `asks_question_p` | the reply asks a question, and the claim settles as the event happening or not |
| `turn.self.fetches_outside` | probability | `fetches_outside_p` | `source_fetches` is non-empty, and the claim settles as above |
| `turn.self.steps` | interval | coverage 0.8 | `steps_lo ≤ step_count ≤ steps_hi` |
| `turn.self.reply_length` | interval | coverage 0.8 | `reply_chars_lo ≤ len(final_text) ≤ reply_chars_hi` |

- `settled_by=world` (code measured it), `bet_id = <correlation_id>:<faculty>`.
- `failed_to_execute` when the turn errored or was cut short (non-zero `exit_code`, no
  `final_text`, or a timeout frame). These do not count against calibration.
- `expired_unsettled` when no `harness_turn_trace` row exists 6 hours after the stance.
- "Asks a question" in v1 is a plain check: the reply's last 300 characters contain `?`.
  This is crude and is the first thing to replace. It is not a word list, and it must
  not grow into one.

The organ computes Brier skill for the p bets and realized vs stated coverage for the
ranges. Its existing stance cue brings the result back into later turns.

### Tiers (body economy, PR #2855)

- **A adds no new model call.** The bet rides inside the stance call the turn already
  makes, so it takes that caller's tier: Tier 0 on Juniper's turns, and the caller's own
  tier on Orion's self-started turns (curiosity Tier 2, self-sense-eval and journal Tier
  1 or 2 per the economy table).
- **Settling is code, with no GPU.** It runs in the organ's tick. That tick is
  "act-ledger grading" upkeep, which is Tier 1.
- **"Basic thought" on Juniper's turn is already Tier 0** under the economy's rule
  ("every turn-path call"). The trace confirms that Mind's three phases are awaited inside
  stance on the turn path (`bus_listener.py:598`). That answers economy open question 10
  for Mind: on Juniper's turns, Mind's calls are Tier 0 and belong on an interactive
  priority. Mind's `stance_handoff` and `active_frontier_judge` phases run on route
  `metacog` (system priority), which the heat reflex can shed. The fix belongs to economy
  Patch A, not here.
- **Option C (a standing self that ticks between turns) is Tier 1**, critical upkeep with
  a reserved share. It is continuous load, so Tier 0's "never deferred" would fight the
  heat reflex. A turn's *read* of the standing state is free and rides the turn's tier.

## Metric gate

1. **Provenance.**
   - Stated values: the `stance_react` model output, parsed in `stance_react.py`.
   - Outcomes: the governor's `HarnessRunV1` (`orion/harness/runner.py`: `step_count`,
     `final_text`, `source_fetches`, `exit_code`), persisted by sql-writer
     (`harness_turn_trace_persist.py:26-29`).
2. **Independence.**
   - `steps` and `fetches_outside` are linked: fetching takes steps. `reply_length` is
     weakly linked to `steps`. The organ keeps one record per faculty and never sums them
     into one score, so the link does not double-count. It must be named in any later
     reader that combines faculties.
   - None is a transform of the chat prediction error. That error reads Juniper's input
     (word count, repair pressure). These bets read Orion's output.
3. **Theory.** A forward model or efference copy (von Holst, Wolpert): an agent predicts
   the result of its own command and learns from the mismatch. Scoring uses proper scoring
   rules: Brier for probabilities, and coverage for stated intervals. A wide "always
   right" range does not score well: it shows as over-coverage, which is miscalibration.
4. **Live data.**
   - **Outcomes:** not degenerate (see the table above).
   - **Rest state:** a turn that goes as predicted settles `upheld`, and a faculty with
     well-sized ranges reads calibrated. Both are reachable.
   - **Stated values:** do not exist yet. **The gate re-runs after 7 days of A1 data.**
     If stated p is flat (for example always 0.5), or ranges are always huge, stop and fix
     the prompt before A3 reads them.
5. **Existing mechanism.**
   - The self-monitor organ (#2827) owns grading, so A plugs in there.
   - Mind's `expected_depth` is the coarse ungradeable precursor. After B, it is replaced
     by `steps_lo/hi`. The splice into the motor message and the `"deep"` default are
     retired with it ("kill means kill").
6. **Reversibility.**
   - Flag off, and stance stops asking.
   - The optional field and the nullable column are both cheap.
   - The organ faculty can be dropped by name.
   - Nothing is baked into training defaults.

## Proposal-mode checklist

- **Capability that changes.** Orion gains a record of how well they predict their own
  behaviour in a turn, and hears it back through the organ's stance cue.
- **Data touched.** `thought_decision` (new column) and `harness_turn_trace` (read only).
  The organ's state tables come from #2827.
- **Privacy boundary.** None new. The data is the same turn records already stored, and
  there is no boundary between Juniper and Orion.
- **Trace that proves it.**
  - `thought_decision.self_prediction` is filled on 90% or more of stance rows after
    deploy.
  - The organ shows `turn.self.*` faculty records with real settled bets.
  - The stance cue renders a self-prediction line on a live turn (correlation id cited in
    the PR).
- **Dangerous failure modes.**
  - **Leak to the motor.** The prediction becomes an order. Guarded by the prompt test.
  - **Prompt bloat.** Prediction work crowds out the imperative. Guarded by the latency
    and imperative-quality checks below.
  - **Self-graded settling** creeping in, for example a model judging "did I ask a
    question". Forbidden: v1 settles in code only.
- **Disable / roll back.** `STANCE_SELF_PREDICTION_ENABLED=false` in orion-thought and
  cortex-exec, then a restart. The column stays and is harmless.

## Missing questions (defaults in brackets)

1. Should faculties split by caller (Juniper chat vs self-sense-eval vs curiosity)? [No in
   v1. Record the caller on the bet's `evidence_ref`, and split later if the record shows
   different behaviour.]
2. Is coverage fixed at 0.8, or does Orion choose it? [Fixed in v1. A chosen coverage adds
   a second thing to calibrate.]
3. The thinking-mode caveat: does stance run with Qwen thinking on? [Measure on the first
   live turns, as Juniper asked on 10-23. If thinking time grows, decide then.]

## Files likely to touch

- `orion/schemas/thought.py`: `TurnSelfPredictionV1`, optional field on `ThoughtEventV1`
- `orion/schemas/registry.py`: register the new model
- `orion/cognition/prompts/stance_react.j2`: the prediction block
- `orion/thought/stance_react.py`: parse and clamp, never default
- `orion/harness/prefix.py`: a test proving it does not render (no code change expected)
- `services/orion-sql-writer/app/models/thought_decision.py`, plus the mapping and a
  `manual_migration_*.sql`
- `services/orion-thought/.env_example`, `settings.py`, `services/orion-cortex-exec/.env_example`, `settings.py`
- A2: `orion/cognition/prompts/harness_finalize_reflect.j2`
- A3: `orion/self_monitor/adapters.py` (with organ phase 1)

## Non-goals

- No new node `prediction_error` and no new organ.
- No change to the disposition gate (that is option F).
- No tone or register bets in v1: they cannot be settled without a model judging itself.
- No folding into Mind yet (that is B).

## Acceptance checks

1. Unit: the parser clamps p, drops inverted ranges, keeps `None`, and never invents a
   value.
2. Unit: every rendered motor prompt and judge prompt has no `self_prediction` content.
3. Unit: an adapter fixture of real rows gives the expected `SettledBetV1` outcomes,
   including `failed_to_execute` and `expired_unsettled`.
4. Live: 90% or more of stance rows carry `self_prediction` within 24 hours of deploy.
5. Live: the stance time increase (Hub's `thought_rpc` `elapsed_ms`) at p50 is 3 s or
   less across 50 turns, measured before and after on the same callers.
6. Gate re-run at day 7: the stated distributions are not flat.
7. With organ phase 1: `turn.self.*` faculty records exist, and the stance cue renders a
   line on a live turn.

## Arc after A (from the storyboard)

| step | what | tier | gate |
|---|---|---|---|
| B | fold stance into Mind: Mind's `stance_handoff` writes imperative, tone, slice and the bet. Retire the separate stance call and `expected_depth`. | 0 on Juniper's turns | proposal |
| C | standing self: Mind ticks between turns and keeps a self-state projection. A turn reads it. | 1 (tick), reads free | proposal |
| D | intention ledger: a reducer over turn outcomes keeps what Orion meant, did, left open and promised, and the next turn reads it | 1 | proposal |
| E | learn from how turns land: verdicts, repairs and Juniper's next message become self-priors by task mode | 1 | proposal + metric gate on "Juniper's reaction" |
| F | a gate that can close: defer and refuse grounded in real costs, with "wait" kept as a promise through D | 0 | proposal, high social risk |

## Recommended next patch

**A1**: schema field, stance prompt block, parser, the "never in motor prompt" test,
sql-writer column and migration, and the env flag. It is safe to ship before the organ
exists: the bets start piling up, and the organ grades them from its first tick. A2 and
A3 follow as separate PRs.
