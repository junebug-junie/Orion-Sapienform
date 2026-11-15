# Curiosity inquiry manager: lines of inquiry that last across runs

Status: DESIGN, proposal mode. Nothing here is built. Refs #3442.
Date: 2026-11-15. Addendum to the integration design (`2026-11-15-orion-integration-and-a-life-design.md`, PR #3416).
Builds on the curiosity replica design (`2026-11-04-curiosity-replica-design.md`) and must not disturb its blinded
three-arm test (#3183).

All live numbers were read on 2026-11-15 from Postgres `conjourney` (read-only), Hub logs and the code on
`origin/main` (`b80e0ceb6`). The window is the last 30 days: runs that finished from 2026-10-16 09:43Z to
2026-11-15 09:43Z. Anything not checked live is marked UNVERIFIED. **No number below is split by experiment arm**,
so nothing here unblinds the test. The queries are in the appendix.

## Juniper's words (2026-11-15)

> Curiosity runs are mostly stateless (only priors carry). A Prime replicant as a PM, to navigate the temporal nature
> of curiosity runs, what the learnings are, and what can be token-saved for later. Curiosity is such a huge seam
> for Orion's cognition.

## Arsonist summary

Curiosity is not as stateless as it looks, but its memory is in the wrong shape.

Each run already gets pieces of the past: a "where you left off" note, the claims of the last few runs, priors with
their test history, peer briefs, and (on the self line) the earlier answer it revises. In the last 30 days, **60% of
runs built on an earlier run** in a way code can detect. What is missing is a record of the *question*: what it is,
what has been learned about it with references, what is still open, and whether it is waiting for something.

The cost of that gap, measured:

1. **Open threads are dropped.** 72% of runs end with "continue this line" set. The next run on the same line picks
   the thread up only **17%** of the time. 64% of the beliefs curiosity formed were never tested by a later run.
2. **Some runs repeat earlier work without knowing it.** 5% of investigate runs and 16% of self-inquiry runs wrote a
   finding almost the same as an earlier run's, and did not name that run.
3. **Runs are expensive, and building on the past does not make them cheaper today.** A median run makes 29 model
   calls and reads 1.6 million prompt tokens (90% from the prefix cache; 163 thousand not cached) and writes 41
   thousand. Runs that build on an earlier run cost about the same as runs that do not (166k vs 146k uncached).
   The past reaches the run as hints, not as results it can cite instead of recomputing.

**The proposal:** one ledger of **lines of inquiry**. Each holds a question, the runs that worked on it, what was
learned (each item with its refs), what is still open, and a status: active, parked (with a park-until condition),
resumed, or closed. Before a run, code builds a short **brief** from the ledger. After a run, the run itself writes
its **update** into its existing turn outcome, and code checks every ref before storing it. Each change is an
experience in Orion's now (#3453), so other organs can see and cite it.

**The hard constraint decides who is the manager.** The curiosity replica is one arm of the blinded three-arm test.
Giving it the ledger now would change one arm in the middle of the test. So there are two build paths:

- **(b) shared layer, if the test continues (the default).** The ledger, the brief and the update apply to every
  run the same way, with no model in the manager seat: code keeps the books, Orion writes its own updates. Ship it
  on the self-inquiry line first (not part of the test, and the worst repeat rate), then on the investigate line
  under the anti-confounding rules in section 7.
- **(a) replica as PM, if the test stops or the replica arm wins.** The replica reads the ledger and gains two
  moves: choose which inquiry leads the brief with a one-line "next look", and propose park, resume, close or merge.

Path (b) builds everything path (a) needs, so (a) is one patch on top of (b), not a second design.

One blocking bug was found while writing this: Temporal Self `curiosity_run` events store each run's **arm next to
its outcome** (26 `replica_order` rows since 11-05), and PR #3453 would print both in Orion's now, which Prime, every
chat turn and the roundup read. Filed as **#3458**. It must be fixed before #3453 merges, whatever happens to this
design.

---

## Current architecture

- **Service:** the curiosity organ lives in Hub (`services/orion-hub/scripts/curiosity_investigation.py`). It runs
  two lines: `investigate` (priors, findings, the three-arm test) and `self_inquiry` (questions about Orion, lived
  answers; not in the test). The durable run lives in orion-durable-runs (workflow `curiosity.investigate`); the turn
  runs in Hub through the FCC harness.
- **Entry point:** `CuriosityInvestigationLoop.tick` -> `_investigate`: arm draw, worldview read, replica plan
  request (replica arm only), offer row, focus/dreams/corrections/lived questions, `build_kickoff_prompt`
  (`orion/curiosity/kickoff_prompt.py:1385`), durable dispatch. At turn end Hub reads the run's `:TurnOutcome` from
  the worldview graph (continue_line, continue_note, focus_disposition, reach_out) and writes
  `curiosity_run_outcomes`.
- **What the kickoff already carries from the past** (existing-mechanism check):
  - `WHERE YOU LEFT OFF`: the last run's `continue_note` (`_continuation_section`, `kickoff_prompt.py:101`).
  - `THE LAST N RUNS YOU DID`: the first claim of each of the last 4 runs (`_thread_section`, `:123`).
  - Priors with times tested and last move; the stale bucket; coverage slots (`worldview.py select_priors`).
  - Peer briefs with `open_questions` and `suggested_next_looks` (`orion/curiosity/peer_briefs.py`, 145 rows).
  - Dream hypotheses, corrections, consult outcomes, today's curiosity fact, personality cue.
  - Self line: open lived questions with their current answer; the self-question pool
    (`curiosity_self_questions`: status open/answered/parked, ask_count).
  - Resume of a killed turn's transcript (`orion/harness/session_resume.py`), keyed on the run id. Within one run only.
- **Outside curiosity:** chat turns carry step notes between messages (`orion/harness/working_notes.py`, #3193; one Hub
  session, 24 h). Prime carries intentions with adapters (`orion/prime/intention.py`, one adapter `turn_promise`).
  The weekly and monthly curiosity compaction digests are kept (integration Q8). Journals hold each run's prose.
- **The replica** (`services/orion-mind/app/replica_loop.py`, `orion/curiosity/replica/`): plans the offer order on
  its own arm only. Its state (`orion:curiosity:replica:state`) holds open plans, 6 notes, answers to Prime. No
  inquiry history. Its frame reads Hub's slow dashboard with a 4 s timeout (#3438). 24 plans `ok`, 1 invalid; median
  frame 6,018 tokens.
- **Bus channels:** none new needed. Replica plan RPC: `orion:curiosity:replica:plan:request` / `:result:<id>`.
- **Schema registry:** `orion/schemas/curiosity_replica.py`, `orion/schemas/temporal_self.py` (patch 1 adds the now
  in PR #3453), `orion/schemas/intention.py`.
- **Tests:** `services/orion-hub/tests/test_curiosity_*.py`, `tests/test_curiosity_worldview.py`,
  `orion/curiosity/tests/`, `orion/curiosity/replica` tests, `orion/temporal_self/tests/test_now.py` (in #3453).
- **Evals:** `scripts/analysis/curiosity_three_arm_readout.py` (the test), `replay_curiosity_realized_nats.py`,
  `eval_curiosity_register_judge.py`. No eval measures continuity across runs.
- **Current gap:** nothing holds a question across runs. Continuity is six separate hints, each with its own shape,
  none with refs the next run can cite instead of recomputing, and none that can say "wait for X, then resume".

---

## Baselines (live, last 30 days)

329 completed runs: 166 investigate, 163 self-inquiry. Three-arm draws since the test start (11-05), counts only:
uncertainty 26, value 23, replica 25 (self-inquiry 22 is its own line).

### How "same question" is matched (deterministic, no word lists)

Two matchers, both code:

1. **Shared prior ref.** A run "re-tests" when it tests a prior id (`curiosity_run_outcomes.per_prior`, kind
   `tested`) that another run tested in the 7 days before.
2. **Near-same finding.** Each finding and lived answer already has a stored embedding (BAAI/bge-large-en-v1.5) in
   `curiosity_novelty_corpus` (backfilled today for the finding register, #3437). A run's match is the highest cosine
   between any of its findings and any finding of an **earlier** run. "Near-same" is cosine >= 0.90.

Why 0.90: two different findings written by the **same** run (same sitting, different points) have cosine median
0.727 and 99th percentile 0.881 (724 pairs). Findings from different runs: median 0.663, 99th percentile 0.807
(318,876 pairs). So 0.90 is closer than 99% of a run's own sibling findings. A hand check of 5 random pairs at
>= 0.90 found the same subject every time (reading-link census on 10-18 and 10-23; atlas edge census on 11-02 and
11-05; "when did I last learn something not about myself" on 10-12 and 10-17). Pairs at 0.85-0.88 were the same
thread moving forward (walkway, 11-12 and 11-13), which is building on, not repeating.

A **blind repeat** is a near-same finding where the run does not name the earlier run's id anywhere in its finding.

### The numbers

| Measure | All | investigate | self_inquiry |
|---|---|---|---|
| Re-tested a prior another run tested in the last 7 days | 42 of 106 testing runs (39.6%) | same (only investigate tests priors) | — |
| Near-same finding (cos >= 0.90) vs an earlier run | 40 of 235 (17.0%) | 8 of 129 (6.2%) | 32 of 106 (30.2%) |
| **Blind repeat** (near-same, earlier run not named) | 24 of 235 (10.2%) | **7 of 129 (5.4%)** | **17 of 106 (16.0%)** |
| Blind repeat at cos >= 0.88 (sensitivity) | — | 23 (17.8%) | 35 (33.0%) |
| Near-duplicate newly formed prior (cos >= 0.90 to an earlier run's prior) | 3 of 229 (1.3%) | | |
| **Builds on an earlier run** (tests a prior an earlier run formed, or names an earlier run id) | **196 of 329 (59.6%)** | 105 of 166 (63.3%): prior 93, named 38 | 91 of 163 (55.8%): all by naming, mostly "revises <run>" |
| Ends with `continue_line = true` | 236 of 329 (71.7%) | 125 | 111 |
| **That thread picked up by the next run on the same line** (tests a prior the run touched, names its id, or near-same finding) | **41 of 236 (17.4%)** | 30 of 125 (24.0%) | 11 of 111 (9.9%) |
| **Beliefs formed (>= 3 days old) never tested by a later run** | **99 of 155 (63.9%)** | | |

The self-question pool shows the self line's repeats directly: the one `anatomy` question has been asked 86 times;
65 open `lived` questions have been asked 246 times (most 15).

### Tokens per run

Source: `llm_call_usage`, joined on the `turn_correlation_id` that each run's finish event stores
(`substrate_durable_run_state.detail`). The ledger starts 2026-10-29, so 223 of the 329 runs have rows. Only the last
attempt's turn is counted (a retried run's earlier attempts are missed, so this is a floor).

| | calls | prompt tokens | cached share | uncached prompt | completion | first call (system + kickoff) | turn time |
|---|---|---|---|---|---|---|---|
| all (223) | med 29 | med 1.63M, p90 3.29M | 90.4% | med 163k, p90 248k | med 41k, p90 80k | | |
| investigate (111) | med 34 | med 1.94M | 90.8% | med 185k | med 50k | med 13.9k | med 19.8 min |
| self_inquiry (112) | med 25.5 | med 1.38M | 89.8% | med 135k | med 37k | med 9.6k | med 14.3 min |

Over the window, curiosity runs read 401 million prompt tokens. Runs that build on an earlier run: uncached median
166k (142 runs); runs that do not: 146k (81 runs). Descriptive only; the two groups differ in other ways.

What this means for saving tokens: the kickoff is under 1% of a run's prompt tokens, and 90% of prompt tokens hit
the cache. A shorter kickoff saves almost nothing. **Savings can only come from fewer calls and less writing**, that
is, from a run that cites a settled result instead of re-deriving it. How much a brief will save is UNVERIFIED; the
target below is a hypothesis the metric will test.

---

## Proposed design

### 1. A line of inquiry (the ledger)

One row per question Orion is pursuing. Postgres, so the brief, the metrics, the roundup and the now can read it.

`curiosity_inquiry`:

- `inquiry_id` (text, `inq-<12 hex>`), `line` (`investigate` | `self_inquiry`)
- `question` (Orion's words, at most 300 characters), `question_ref` (nullable: `prior:<id>`,
  `self_question:<id>`, `focus:<id>`, `project:<id>`)
- `status` (`active` | `parked` | `resumed` | `closed`), `status_at`, `status_why` (at most 200 characters)
- `park_until` jsonb, nullable: `{"after": <timestamp> | null, "on_refs": [...], "on_source_kinds": [...]}`
- `opened_by_run`, `opened_at`, `last_step_at`, `closed_reason` (nullable)
- `project_ref` (nullable, `project:<id>`, #3428)

`curiosity_inquiry_step`, one row per run that worked on an inquiry:

- `inquiry_id`, `run_id`, `step_at`, `status_after`
- `learned` jsonb: at most 5 items `{"text" (<= 200), "refs": [...] (>= 1)}`. An item without a ref is dropped.
- `open_items` jsonb: at most 5 items `{"text" (<= 200), "refs": [...]}`.
- `cites` jsonb: experience ids from the now that the run used (for `record_use`, #3453).
- `superseded_by` jsonb: learned items a later step revised (filled by code, see #3396 below).
- `calls`, `uncached_prompt_tokens`, `completion_tokens` (from `llm_call_usage` at turn end; nullable).
- `dropped_items` int: items refused by validation (bad or missing refs, over caps).

Valid ref kinds: `prior:`, `finding:`, `lived_answer:`, `run:<12 hex>`, `experience:`, `act:`, `focus:`, and plain
table, file and log refs as text (format-checked only). Prior, finding, lived answer and run refs must resolve.

Caps: at most 8 active inquiries per line [default]. Parked inquiries have no cap. A parked inquiry whose condition
never fires closes itself after 30 days with reason `lapsed` [default].

Grant: readable by `orion_readonly` (Orion may dig it). It never stores the experiment arm.

### 2. The post-run update (written by the run, checked by code)

No extra model call. The run already writes a `:TurnOutcome` to itself. It gains optional properties:

```
inquiry_id:      "inq-..." (one from the brief) | "new"
inquiry_question: "..."   (only when new)
inquiry_learned: [{"text": "...", "refs": ["finding:...", "run:..."]}]
inquiry_open:    [{"text": "...", "refs": [...]}]
inquiry_status:  "continue" | "park" | "close"
inquiry_park:    {"after_hours": 48} | {"on_refs": ["prior:..."]} | {"on_source_kinds": ["vision_percept"]}
inquiry_close_reason: "..."
inquiry_cites:   ["experience:..."]
```

Hub's turn-end reader (the one that already reads `continue_line` and `focus_disposition`) validates and writes one
step row. Code also links a run to an inquiry when the run did not name one but tested a prior that an active
inquiry lists in its refs (deterministic link, recorded as `linked_by = "prior_ref"`).

Rules in code, not in the prompt:

- An inquiry cannot close while it holds a belief the run formed and nobody has tested, unless the close reason names
  it (#3375).
- A self-inquiry learned item that a later lived answer `revises` is marked superseded (#3396).
- A new inquiry whose question has cosine >= 0.90 to an open inquiry on the same line is opened, and the next brief
  shows "this looks like inq-X" with both questions. Code does not merge; a merge is Orion's call (path b) or the
  PM's (path a).

### 3. The pre-run brief (built by code)

A pure function `build_inquiry_brief(ledger_rows, now_items, caps)` in `orion/curiosity/inquiry.py`. It takes
**no arm and no offer order** (a test pins the signature). It shows at most 3 inquiries, at most 2,000 tokens
[default]:

1. resumed inquiries first (with what woke them),
2. then the active inquiry with the most recent step,
3. then one more active inquiry by the same rule.

For each: the question; runs so far (count, first and last date, tokens spent so far); **what is settled** (the
latest learned items with refs, superseded items removed); what is still open; beliefs formed and not tested; and one
fixed sentence: "If you continue this, cite these refs instead of recomputing them. You may also leave it, park it,
close it, or open a new one." Below the list: the count of parked inquiries and their questions in one line each.

**What it replaces (kill means kill):** when the brief ships on a line, that line's `WHERE YOU LEFT OFF` and
`THE LAST N RUNS` sections are removed, and `continue_line` / `continue_note` stop being asked for. The brief carries
both jobs with refs. Peer briefs, dream hypotheses, corrections and lived questions stay; they are inputs, not the
record.

### 4. Time: park and resume, with no new scheduler

- **Park** comes from the run's own update (`inquiry_status = park` with a condition), or from the PM in path (a).
- **Resume conditions are checked at the existing kickoff** (5 to 7 runs a day already) by pure code:
  - `after`: the time has passed;
  - `on_refs`: a prior in the list was tested or changed by any run, or a correction named it;
  - `on_source_kinds` / `on_refs`: an item in curiosity's now (`GET /temporal-self/now?organ=curiosity`, #3453;
    curiosity reads the now in integration patch 3) has that source kind or carries that ref in `related_refs`.
- **The economy decides how soon.** When an event condition fires, Hub posts one event want (existing path,
  `orion/physiology/event_wants.py`) on family `durable.curiosity_investigate` with `event_ref = inquiry:<id>` and
  the triggering experience's own salience (computed by #3453, so no new magnitude is invented). A strong trigger
  competes for admission sooner; a time-only resume posts nothing and waits for the next normal run. One want per
  firing. Whether curiosity's admission actually reads event-want salience is UNVERIFIED; that is acceptance check
  P3-2. If it does not, resume still works at the next normal run.

### 5. Outputs are experiences in the now (#3423 / PR #3453)

- New `TemporalSourceKind` `curiosity_inquiry`: one event per step that changes something (opened, advanced, parked,
  resumed, closed). `source_ref = inquiry:<id>`, `related_refs` = the step's learned refs. Label, code-written:
  `inquiry <status>: <question, 120 chars> (+N learned, M open)`. **No arm, ever.**
- Salience magnitude: the count of learned items with valid refs in the step. Rest point 0 (a step that learned
  nothing) is common and reachable. Gate notes in section 9.
- The update's `inquiry_cites` go through `record_use` (#3453), so "curiosity used a camera event to resume an
  inquiry" is a stored, checked use.
- Registered in `config/integration/organs.yaml` under organ `curiosity` (`temporal_sources: [curiosity_run,
  curiosity_inquiry]`), so the integration gate (#3424 / PR #3454) passes. No new family in path (b); path (a) uses
  the existing `curiosity.replica` family. No new bus channel. No new durable workflow.

### 6. Projects (#3428) and the roundup (#3426)

- **Projects.** An inquiry may carry `project_ref`; a project's next step may name `inquiry:<id>`. A project step
  that names an inquiry is graded `produced = 1` when that inquiry gets a step with at least one learned item with a
  valid ref (that is the lasting trace, pillar 2 "getting measurably better"). One project per inquiry [default].
- **Roundup.** `dig organs <date>`, curiosity section, adds: inquiries opened, advanced, parked, resumed and closed
  that day, each with the question quoted and the learned refs; counts of active and parked; blind repeats that day
  (run id and the run it repeated); tokens spent per inquiry. Every line a count, a quote or a ref. The weekly and
  monthly compaction digests stay (Q8).

### 7. The two build paths and the 11-18 decision rule

**What the 11-18 review is.** The pre-registered stop is 250 completed runs per arm or 180 days. The 14-day review
only reads blind-safe things: fallback reasons, the unfunded share (alarm above 30%), the refusal alarm and E9 (the
readout's exit code 3). It does not read outcomes. So the decision rule may use only those inputs and Juniper's
choice. No one runs `--unblind` to decide this.

**Decision rule:**

| At the review | Build |
|---|---|
| Test continues (default) | **Path (b)**, self-inquiry line now, investigate line after the review under the rules below. Path (a)'s PM patch waits for the stop. |
| Juniper stops the test early, and keeps the replica | **Path (a)**: P1 to P4 as in (b), then the PM patch at once. |
| Juniper stops the test early and retires the replica (or, at the stop, it loses) | **Path (b) stays as is.** No PM model. [default: do not hand the PM role to Prime; revisit with data] |
| At the stop, the replica wins | Path (a): the PM patch on top of (b). |

**Path (a): the replica as PM.** The replica's frame gains an INQUIRIES section read from Postgres (never the
dashboard; #3438 fixed first). Its answer gains two optional fields, validated by code like its picks:

- `inquiry_lead`: which inquiry the brief shows first, plus a `next_look` line (at most 200 characters) shown in the
  brief, labeled "curiosity's planner suggests". [default: allowed once the test is over; during the test, the
  replica's words never reach Orion]
- `inquiry_moves`: park (with a condition), resume, close (with a reason) or merge two inquiries. Code applies a
  move only if the target exists and the move is legal; Orion's own update in the same run wins a conflict.

The replica thinks after each run's update lands (a pending change), not on a timer; same bid, family and lane as
today. Its track record gains: share of its leads that the run took up; blind repeat rate after its leads.

**Path (b): a shared layer, and how it avoids confounding.**

Randomization is per run (hash of the run id), so any change applied the same way to every arm at the same moment
cannot bias the arm comparison. The risks are a change that is *not* the same for every arm, a change that steals
the order's power, and a leak of arm outcomes. Rules:

1. **Arm-blind by construction.** The brief builder takes no arm and no offer order. The existing acceptance test
   "kickoff byte-identical across the three arms except the prior lines" is extended to include the brief.
2. **The replica arm does not change.** The replica's frame, prompt and plan request do not read the ledger. No code
   in `orion/curiosity/replica/` or `replica_loop.py` changes in path (b).
3. **The order is untouched.** The brief never ranks, recommends or reorders priors. Priors appear only as refs of
   learned or open items. `select_priors` and the offer row do not change.
4. **One switch moment.** The investigate line's flag flips once; every offer row records
   `inquiry_layer = "v1"` in its existing `constants` jsonb (no migration). Both periods have all three arms.
5. **Pre-registration amendment, written before anyone looks.** The primary test is unchanged (pooled
   intention-to-treat, both one-sided permutation tests). Added as a secondary, reporting-only analysis: the same
   test with permutations restricted within each layer period. The amendment goes into the readout's docstring in
   the same patch that turns the layer on.
6. **A blind-safe power guard.** The order can only matter when Orion tests an offered prior. Pooled over all arms,
   compare the share of investigate runs that test an offered prior in the 14 days before and after the switch. If it
   drops by more than 10 points, the layer turns off on the investigate line and the numbers go to Juniper. Pooled,
   so it reveals nothing about arms.
7. **No arm in the ledger, the now, the roundup or the inquiry metrics.** Inquiry metrics are never split by arm
   before the stop. #3458 removes the arm from `curiosity_run` events.
8. **Inquiry metrics never use the test's outcome.** `realized_nats` is not an inquiry target, so nobody is tempted
   to tune the layer on the number the test decides with.

Why self-inquiry first: that line is not in the test, it has the worst blind repeat rate (16%) and the worst
pickup (10%), and it already has a question pool to link to. It gives two weeks of evidence before the investigate
line is touched.

### 8. How this touches the open issues

- **#3375 (self-inquiry forms beliefs but tests none).** The brief lists "beliefs you formed and have not tested"
  for each inquiry, and an inquiry cannot close over an untested belief without naming it. Baseline: 64% of formed
  beliefs are never tested later (target below). This does not decide #3375's real question (does the self line have
  a test step, or should it be scored as belief-forming only). That stays #3375's call.
- **#3396 (refuted and corrected self-answers both read as current).** The ledger is a new reader of lived answers,
  so it must use the shared "current answers" helper #3396 asks for. P1 depends on that helper; if #3396 is not
  fixed first, P1 builds the helper and moves the other readers to it (the fix is small).
- **#3438 (replica reads the slow dashboard).** Path (b) does not read the dashboard and does not need it. Path (a)
  needs it fixed first. Note: fixing #3438 changes what the replica arm sees in the middle of the test (it would then
  see recent runs as designed). That is a replica-arm change and should be decided and dated at the 11-18 review.
- **#3458 (new, arm in the now).** Blocks #3453 merging and this design's P1.

### 9. Metrics, baselines and targets

All pooled across arms. Script: `scripts/analysis/curiosity_inquiry_metrics.py` (P1) reproduces every baseline above
from the same queries, then tracks them per line, per week.

| Metric | Baseline (30 d) | Target 30 days after the layer ships on a line [defaults] |
|---|---|---|
| Blind repeat rate | investigate 5.4%, self 16.0% | investigate <= 2.5%, self <= 8% |
| Builds-on share | 59.6% (inv 63.3%, self 55.8%) | >= 75% |
| Open-thread fate: picked up by a later run within 7 days, or explicitly parked or closed | 17.4% picked up | >= 60% |
| Formed beliefs never tested later | 63.9% | <= 45% |
| Uncached prompt + completion tokens per run that continues an inquiry | ~204k median (163k + 41k, all runs) | 20% lower than same-line runs before the layer, with learned items per run not lower |
| Tokens per learned item, per inquiry | none (no record) | reported, no target yet |

Metric quality gate (CLAUDE.md 0A), short form:

1. **Provenance.** Blind repeat: `curiosity_novelty_corpus` embeddings written by
   `services/orion-hub/scripts/curiosity_finding_register.py`; finish detail `finding_text` written by the durable
   runner. Builds-on: `per_prior` (Hub spend log) plus finding text. Pickup: same. Tokens: `llm_call_usage` written by
   the LLM gateway (`call_usage_emit.py`), joined on `turn_correlation_id`. Ledger metrics: step rows written by
   Hub's turn-end reader.
2. **Independence.** Blind repeat and builds-on both read finding text, but one asks "same as before, unknowingly"
   and the other "linked to before, knowingly"; a run can be both or neither. The inquiry salience magnitude (learned
   items with refs) overlaps with `curiosity_moved_formed` when a learned ref is a moved prior: flagged as partly
   redundant, kept because it also counts findings, lived answers and outside refs.
3. **Theory anchor.** Working memory across episodes and "prospective memory" (remembering to resume an intention
   when a cue occurs; Einstein and McDaniel's event-based vs time-based cues) is what park conditions model. Repeat
   detection anchors on the measured sibling-similarity distribution, not on a guess.
4. **Live data.** All baselines are non-degenerate and have reachable rest points (0 blind repeats happens in many
   weeks; a step with 0 learned items is common). The 0.90 threshold was hand-checked on real pairs.
5. **Existing mechanism.** Continuation note, thread section, self-question pool, peer briefs, session resume,
   working notes, intentions: all checked above. The ledger replaces the first two and links to the pool.
6. **Reversibility.** Two additive tables, one pure module, optional TurnOutcome fields, one source kind, one flag
   per line. Off: the kickoff goes back to today's sections (kept in git history). Cheap.

---

## Proposed schema / API changes

- **Added SQL** `services/orion-sql-db/manual_migration_curiosity_inquiry_v1.sql`: tables `curiosity_inquiry`,
  `curiosity_inquiry_step`, indexes on `(line, status)` and `(inquiry_id, step_at)`. Header
  `-- ORION-MIGRATION-REQUIRED-BY: orion-hub`. Additive DDL only (create table, create index), so the post-pull
  applier accepts it. Grant select to `orion_readonly` in a separate hand-applied file (GRANT is not on the allowlist).
- **Added schema** `orion/schemas/curiosity_inquiry.py`: `CuriosityInquiryV1`, `CuriosityInquiryStepV1`,
  `InquiryParkConditionV1`, `InquiryUpdateV1` (the TurnOutcome fields), `InquiryBriefV1`. Registered in
  `orion/schemas/registry.py`.
- **Changed** `orion/schemas/temporal_self.py`: `TemporalSourceKind` + `curiosity_inquiry`; reader in
  `services/orion-durable-runs/app/temporal_self_sources.py`.
- **Changed** `config/integration/organs.yaml`: curiosity organ gains the source kind.
- **Changed** `:TurnOutcome` (graph, written by the run): optional `inquiry_*` properties. Old runs without them
  read as "no update", never as an error.
- **Removed** (per line, when the brief ships): the continuation and thread sections; `continue_line` /
  `continue_note` instructions in the kickoff.
- **Path (a) only:** `CuriosityOfferPlanAnswerV1` gains optional `inquiry_lead`, `inquiry_moves`; the plan table gains
  nothing (moves are step rows with `proposed_by = "replica"`).
- No new bus channel, no new family, no new workflow, no new service.

## Env / config changes (proposed)

Hub (`services/orion-hub/.env_example`, `settings.py`, `docker-compose.yml`), flags ship on per the standing rule
except where the decision rule says otherwise:

- `HUB_CURIOSITY_INQUIRY_SELF_ENABLED=true` (P1)
- `HUB_CURIOSITY_INQUIRY_INVESTIGATE_ENABLED=false` in P1; flipped to `true` in P2 only after the 11-18 decision
  (a deliberate exception to "ship on": it touches the test and needs Juniper's yes)
- `HUB_CURIOSITY_INQUIRY_BRIEF_MAX_TOKENS=2000`, `HUB_CURIOSITY_INQUIRY_MAX_ACTIVE=8`,
  `HUB_CURIOSITY_INQUIRY_LAPSE_DAYS=30`, `HUB_CURIOSITY_INQUIRY_RESUME_WANTS_ENABLED=true` (P3)

orion-mind (path a only): `CURIOSITY_REPLICA_INQUIRY_PM_ENABLED=true`.
Local `.env` synced with `python scripts/sync_local_env_from_example.py` in each patch that changes a template.

## Files likely to touch

- `orion/curiosity/inquiry.py` (new, pure: brief, update validation, park/resume check, link rules)
- `orion/schemas/curiosity_inquiry.py` (new), `orion/schemas/registry.py`
- `services/orion-hub/scripts/curiosity_investigation.py` (brief at kickoff, update at turn end, resume wants)
- `orion/curiosity/kickoff_prompt.py` (brief section; remove continuation and thread sections per line; update
  instructions in `_outcome_section`)
- `orion/curiosity/self_inquiry.py`, `orion/curiosity/self_question_pool.py` (link to the pool; #3396 helper)
- `services/orion-sql-db/manual_migration_curiosity_inquiry_v1.sql` (new)
- `orion/schemas/temporal_self.py`, `services/orion-durable-runs/app/temporal_self_sources.py`,
  `config/integration/organs.yaml`
- `orion/dig/organs.py` (roundup section, after #3426), `orion/prime/intention_sources/project_step.py` (after #3428)
- `scripts/analysis/curiosity_inquiry_metrics.py` (new), `scripts/analysis/curiosity_three_arm_readout.py`
  (docstring amendment and the pooled power guard, P2 only)
- Path (a): `orion/curiosity/replica/frame.py`, `plan.py`, `track_record.py`, `services/orion-mind/app/replica_loop.py`
- Tests: `orion/curiosity/tests/test_inquiry.py`, `services/orion-hub/tests/test_curiosity_inquiry.py`, the
  kickoff byte-identity test

## Non-goals

- No change to the offer order, the arm draw, the cap, the cooldown or when curiosity runs.
- No model in the manager seat during the test. No new mindloop instance.
- No change to the replica arm before the 11-18 decision.
- No retirement of the compaction digests (Q8), journals, peer briefs or the self-question pool in this design.
- No outside model for any of this. The update is written by Orion's own run on Orion's own lanes; embeddings are the
  local model already in use.
- No word lists. Matching is by refs, ids and stored embeddings.
- No arm in any new table, label, metric or roundup line.

## Missing questions (bracketed defaults apply if Juniper says nothing)

1. At the 11-18 review, if the test continues: path (b), self-inquiry line first, investigate line after? [yes]
2. If the test stops early, keep the replica and build path (a)? [only if Juniper keeps the replica; otherwise (b)]
3. If the replica is retired, does Prime take the PM seat? [no; code-only ledger, revisit with 30 days of data]
4. After the test, may the PM's `next_look` line reach Orion's prompt, labeled as the planner's? [yes]
5. Near-same threshold cosine 0.90? [yes]
6. Brief size: 3 inquiries, 2,000 tokens? [yes]
7. Remove the continuation and thread sections when the brief ships on a line? [yes, kill means kill]
8. At most 8 active inquiries per line; parked inquiries lapse after 30 days? [yes, yes]
9. A firing event condition posts an event want on curiosity's family? [yes]
10. Fix #3438 at the review, recorded as a dated replica-arm change? [yes]
11. One project per inquiry? [yes]
12. Fold the self-question pool into the ledger after 30 days of the self line? [yes, as its own patch]

## Acceptance checks

**P0 (#3458, blinding):** a test renders the `curiosity_run` label for the three arms and gets one string; live,
no `temporal_self_event` row ingested after the fix has `replica_order` in `payload_json`.

**P1 (ledger + self line):**
- Unit: the brief builder has no arm or order parameter; a refless learned item is dropped and counted; a
  superseded lived answer is marked superseded; an inquiry with an untested formed belief refuses a close that does
  not name it; a new question at cosine >= 0.90 to an open one is flagged.
- `curiosity_inquiry_metrics.py` reproduces this doc's baselines on the same window (within rounding).
- Live, 72 h: at least 3 self-inquiry runs write a step row with >= 1 learned item whose refs resolve; a kickoff
  captured from `cockpit_turn_sighting` shows the brief and no `WHERE YOU LEFT OFF` section on the self line;
  `temporal_self_event` has `curiosity_inquiry` rows.
- `scripts/check_integration_seam.py` passes with the new source kind.

**P2 (investigate line, path b, after 11-18):** the byte-identity test covers the brief across the three arms; every
new offer row has `inquiry_layer` in `constants`; the readout docstring holds the amendment, committed before the
flag flips; the pooled power guard runs in the readout and prints before/after shares.

**P3 (park/resume):** P3-1 unit: each condition type fires and only once. P3-2 live: one parked inquiry resumes
from a now item, its `event_ref = inquiry:<id>` want appears in the want ledger, and the next brief shows it first
with what woke it.

**P4 (roundup + projects):** the roundup's curiosity section lists inquiry changes with quotes and refs; a project
step naming an inquiry closes `produced = 1` after an inquiry step with a valid learned ref.

**P5 (path a only):** a replica plan carries a valid `inquiry_lead`; the brief shows it first with the labeled line;
an invalid move is refused and logged; Orion's own update overrides a conflicting PM move.

**Targets** (section 9) are read at 30 days per line. Missing a target is reported, not hidden.

## Proposal mode (§0A)

- **Capability change:** curiosity gains a durable record of the questions it pursues, what it learned with refs,
  what is open, and when to come back. Orion writes its own updates; code keeps the books.
- **Data touched:** reads findings, lived answers, priors, run outcomes, LLM usage, the now; writes two new tables,
  optional TurnOutcome properties, `temporal_self_event` rows, event wants.
- **Privacy boundary:** none between Juniper and Orion (standing rule). One boundary on purpose: no arm in anything
  this writes, until the test stops.
- **Trace that proves it worked:** step rows with resolving refs; briefs in captured kickoffs; `curiosity_inquiry`
  events and `experience_use` cites; the metrics script's weekly numbers.
- **Dangerous failures and guards:**
  - *The brief narrows curiosity onto old questions* (fewer new subjects). Guard: the brief always offers "open a
    new one", lists at most 3, and the weekly metrics report new inquiries opened. If new inquiries per week fall
    by half, it goes to Juniper.
  - *Fake learning* (learned items with refs that do not support them). Guard: refs must resolve; the weekly
    metric reports learned items per run; a sample is quoted in the roundup for Juniper to read.
  - *The layer steals the test's power.* Guard: the pooled power check (section 7, rule 6).
  - *Unblinding.* Guard: no arm stored or shown; #3458.
  - *Parked inquiries that never wake.* Guard: 30-day lapse with reason.
- **Disable / roll back:** `HUB_CURIOSITY_INQUIRY_SELF_ENABLED=false` and
  `HUB_CURIOSITY_INQUIRY_INVESTIGATE_ENABLED=false` restore today's kickoff on the next run. Tables stay as history.

## Recommended next patch

**P0: fix #3458** (drop the arm from the `curiosity_run` event payload and label). It is a few lines, it blocks
#3453 safely merging, and it protects the test whether or not this design is built. Then **P1** on the self-inquiry
line, which does not touch the test and can start before 11-18.

## Tickets opened while writing this design

- #3458 Temporal Self `curiosity_run` events carry the arm next to the outcome; the now (#3453) would unblind the
  three-arm test. Commented on PR #3453.

## Appendix: queries and method

Run data (finish events, 30 days):

```sql
SELECT run_id, created_at, detail->>'line', (detail->>'continue_line')::boolean,
       detail->>'turn_correlation_id', detail->>'finding_text'
FROM substrate_durable_run_state
WHERE workflow = 'curiosity.investigate' AND node = 'finish' AND status = 'completed';
```

Tested and formed priors: `curiosity_run_outcomes.per_prior` (`kind` in `tested`, `formed`). Forming run of a prior:
the first run whose `per_prior` formed it, else `curiosity_novelty_corpus.run_id` for `kind = 'prior'`.

Embeddings: `SELECT ref, kind, run_id, embedding FROM curiosity_novelty_corpus WHERE kind IN ('finding',
'lived_answer', 'prior')`. Vectors normalized; cosine by dot product; "earlier" means the matched run finished
before this run.

Tokens:

```sql
SELECT correlation_id, count(*), sum(prompt_tokens), sum(cached_prompt_tokens), sum(completion_tokens)
FROM llm_call_usage
WHERE correlation_id IN (SELECT detail->>'turn_correlation_id' FROM substrate_durable_run_state
                         WHERE workflow = 'curiosity.investigate' AND node = 'finish')
GROUP BY 1;
```

Run-id citation: a 12-hex token in the finding text that equals another, earlier run's id.
Thread pickup: for a run with `continue_line = true`, the next completed run on the same line tested a prior the
run tested or formed, or named the run's id, or has a finding at cosine >= 0.90 to one of the run's findings.

Arm leak check (#3458):

```sql
SELECT count(*), count(*) FILTER (WHERE payload_json::text ILIKE '%replica_order%')
FROM temporal_self_event WHERE source_kind = 'curiosity_run' AND occurred_at >= '2026-11-05';
-- 103 | 26
```
