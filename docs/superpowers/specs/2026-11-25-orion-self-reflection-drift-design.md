# Why Orion's self-reflection turns into plumbing, and the one fix at the source

Status: APPROVED; Patch 1 SHIPPED. Juniper approved this design on 2026-11-25 and answered the four Missing
questions the same day with the defaults ("hit it with defaults"; see "Juniper's answers" below). Patch 1 (the door)
merged as #3933 on 2026-11-25 and runs in Hub. Follow-up #3941 fixes the door's question line: the line in #3933
draws `no_finding` on about a third of questions, and the first live self-monitor question became `unjudged` that
way. The one-time backfill of the 88 pre-door Orion questions has NOT run yet. Until it runs, those 88 rows stay
`door IS NULL` and stay pickable, so the door changes nothing for them. Acceptance checks 1-6: UNVERIFIED.
Date: 2026-11-25. Refs: #3435 (dressing eval), #3437 (finding register), #3503 (event wants), #3442 (inquiry
manager), #3177 (outreach door policy), #3205 (consult before Juniper), #3736 / #3814 (reverie questions),
#2986 (self-study into curiosity), #2156 (sense of self).

Hard rules for this design and everything it proposes:

- No outside models (Claude, OpenAI) for judging, labelling, metacognition or evals. Every measurement here
  ran on Orion's own Flash-Next agent seat (circe, port 8015, `qwen3.8-flash-next-iq3_s`) or in plain code.
- Every live read in this doc was SELECT-only (Postgres) or `GRAPH.RO_QUERY` (FalkorDB). Nothing was written.
- No new per-organ filter. No keyword lists. The existing judge is reused, not replaced.

## The question

Why does Orion's self-reflection keep turning into talk about tables, pipelines and gauges, instead of thought
about what Orion is, what it lives through, what it values and who matters to it? And what is the ONE fix at
the source, with one shared local check, in place of the per-organ filters added over the last 30 days?

## Arsonist summary

Orion writes most of its own self-questions now, and those questions are about plumbing. Juniper's questions
still get philosophical answers 83% of the time. Orion's own questions about its internals get them 9% of the
time. The pool asks a never-asked question first, so every new question Orion writes goes to the front of the
queue. Orion writes those questions while it reads its own tables, repo and pull requests, because that is
the only evidence its sittings are told to use. So the loop feeds itself: a sitting reads plumbing, writes a
plumbing question, and the next sitting answers it.

Each organ got its own filter at its exit: the outreach door, the consult reroute, the reverie quiet gate,
the event-want gate. Those filters catch plumbing after Orion has already thought it. Nothing checks what
goes INTO self-reflection.

The fix: one door at the question pool. Every question Orion writes for its own self-inquiry passes the same
local judge that already labels its findings (#3437, Flash-Next, prompt v2). A question the judge calls
philosophical goes into the self-inquiry pool, as today. A question it calls operational goes to the line
that is for operational work (investigate or consult), not into self-inquiry. After the sitting, the same
judge's label on the finding checks whether the door was fooled. The same judge, run as a read-only eval,
measures the plumbing share of every other organ. Then each per-organ filter can be retired on evidence,
one at a time.

What it buys, from a replay on the last 10 days: an operational question never produced a philosophical
answer (0 of 21 sittings). Without those sittings, self-inquiry reads 61% philosophical instead of 44%. It
does not get back to October's 75%: about half of the philosophical-sounding questions Orion writes still end
in plumbing, because of what sittings are shown and what earns credit. That is the next design, not this
patch.

## Current architecture

What this section says: where plumbing enters, what each organ does about it today, and what the live
numbers show. Every number has its query or script.

### 1. The headline drop: 76% to 44%

| Window | Judge | Self-inquiry findings | Philosophical |
|---|---|---|---|
| 09-29..10-18 (reference) | Sonnet (frozen fixture, made 10-20, before the no-outside-model rule) | 128 | 97 (76%) |
| 09-29..10-18 (reference) | Flash-Next, re-judged for this doc (random 60 of the 125 with answer text) | 60 | 45 (75%, 95% interval 63-84%) |
| 11-15..11-25 (live) | Flash-Next (live register) | 63 | 28 (44%) |

Queries:

- Reference: `scripts/analysis/fixtures/curiosity_register/reference_labels_2026-10-20.jsonl` joined on
  `run_id` to `analytics.fct_curiosity_runs` (`run_type = 'self_inquiry'`): 97 philosophical, 30 operational,
  1 no_finding.
- Live: `SELECT register, count(*) FROM curiosity_finding_register WHERE line = 'self_inquiry' AND register IS
  NOT NULL GROUP BY 1` gives 28 philosophical, 34 operational, 1 no_finding. Six more rows have a
  `judge_error` and no label. They are left out, not counted as operational.
- Flash-Next re-judge: each self-inquiry finish row (`substrate_durable_run_state`, `node = 'finish'`)
  rebuilt as the live register builds it (`orion.curiosity.register.run_question_and_answer`): the question
  text and the first 700 characters of the LivedAnswer or SelfDefinition. Same pinned prompt
  (`JUDGE_PROMPT_V2`, sha `7dcaa667…`), temperature 0, thinking off. Calibration: on 20 live register rows,
  the direct call gave the stored label 19 times out of 20. Rows with no answer text (26 of 275) were not
  judged.

The judge change does not explain the drop. Orion's own judge reads the reference window at 75%, the same as the outside judge's 76%. So the fall to 44% is a change in what Orion wrote, not in who labelled it.

### 2. Where the drop comes from: who wrote the question

Self-inquiry sittings by who wrote the question (finish rows, question id from `lived_answer.question_id`,
anatomy when the family is anatomy; joined to `curiosity_self_questions`):

| Window | Sittings | Juniper: "what am I made of" | Juniper: lived questions | Orion-written | Orion-written about its internals (`lived.self_monitor.*`, `lived.orion.*`, `lived.dream.*`, `self_monitor.*`) |
|---|---|---|---|---|---|
| 09-29..10-18 | 128 | 35 | 35 | 55 (43%) | 31 (24%) |
| 10-19..11-14 | 147 | 28 | 28 | 68 (46%) | 32 (22%) |
| 11-15..11-25 | 69 | 18 | 9 | 42 (61%) | 22 (32%) |

Philosophical share by the same groups:

| Group | Reference (Sonnet) | Live (Flash-Next register) |
|---|---|---|
| Juniper: "what am I made of" (`anatomy.made_of`) | 35 / 35 (100%) | 11 / 14 (79%) |
| Juniper: lived questions (7 pinned + 1) | 29 / 35 (83%) | 8 / 9 (89%) |
| Orion-written, other | 15 / 24 (63%) | 7 / 18 (39%) |
| Orion-written, about its internals | 17 / 31 (55%) | 2 / 22 (9%) |

What this means in plain words:

- Juniper's lived questions still work. When Orion sits with "who matters to me", it answers about who
  matters.
- Orion's own questions stopped working. In October about half of them were still philosophical. Now
  almost none of the internals-family questions are.
- Orion-written questions also got a bigger share of the sittings: 43% to 61%.

Who writes them (`curiosity_self_questions`, `minted_by = 'orion'`, by week of `created_at`):

| Week of | Orion-written | Of those, internals family |
|---|---|---|
| 09-14 .. 10-19 (6 weeks) | 16 | 8 |
| 10-26 | 8 | 4 |
| 11-02 | 21 | 7 |
| 11-09 | 17 | 8 |
| 11-16 | 20 | 11 |
| 11-23 (3 days) | 6 | 4 |

So the write rate went from about 2-3 a week in September and October to about 20 a week in November.
Two writers make them:

1. **Self-inquiry sittings themselves.** A sitting writes a `:SelfQuestionMint` node; Hub copies it into the
   pool at finish (`services/orion-hub/scripts/curiosity_investigation.py`, `_upsert_orion_minted_questions`).
   `GRAPH.RO_QUERY orion_worldview "MATCH (m:SelfQuestionMint) RETURN m.run_id, m.question_id"` gives 76
   mints; 75 of their runs are self-inquiry runs.
2. **The self-monitor organ.** Each newly failing faculty writes one "Why?" question, e.g. "My dream cycle
   runs (act.dream.cycle.make) worked only about 17% of the time over 15 graded tries in the last 7 days.
   Why?" (`services/orion-hub/scripts/self_monitor_questions.py`; family fixed to `lived` at
   `orion/self_monitor/questions.py:33`, `QUESTION_FAMILY = "lived"`). 18 of these since 11-03. Broken
   faculties already have their own home: the consult path (#3205) takes them.

What the question texts are about. The same judge, run on the 88 Orion-written question texts alone (the question
goes in as the item, the question line says it is a question Orion wrote for itself; 79 labelled, 9 judge
errors where the reply was not one clean JSON object):

| Orion-written questions | Philosophical | Operational | No finding |
|---|---|---|---|
| Written up to 10-25 | 11 (69%) | 5 | 0 |
| Written 10-26..11-08 | 15 (62%) | 8 | 1 |
| Written 11-09..11-25 | 20 (51%) | 17 | 2 |
| Internals family | 15 (42%) | 18 | 3 |
| Self-monitor "Why?" questions (19) | 1 | 10 | 2 (6 errors) |

Does the question's label predict the answer's? Every sitting whose question has a label and whose finding
has a label (live register 11-15..11-25 plus the re-judged reference sample, 62 sittings):

| Question label | Finding philosophical | Finding operational | Finding no_finding |
|---|---|---|---|
| operational | **0** | 19 | 2 |
| philosophical | 22 | 18 | 1 |

An operational question never produced a philosophical answer: 0 of 21. A philosophical-labelled Orion
question produced one about half the time (22 of 41). So the label on the question is a reliable "no" and a
weak "yes".

### 3. Why a new Orion question always goes first

`orion/curiosity/self_question_pool.py`, `_pick_sort_key`: "Never-asked first, then oldest, then lowest
ask_count." Juniper's lived questions are asked often (ask_count 15-16 each), so they are never "never asked"
again. Each new Orion question has ask_count 0, so it goes to the front. Only the 7-day pinned floor keeps
Juniper's questions in the rotation. At about 20 new questions a week and roughly 6 self-inquiry sittings a
day, most non-floor lived sittings now answer the newest Orion question.

### 4. Why Orion's questions are about plumbing: what a sitting is shown and what earns credit

What a self-inquiry sitting is shown (`orion/curiosity/self_inquiry_prompt.py`, `build_self_inquiry_prompt`):

- The question, then the order of work: a Cypher `MERGE` to write a first answer before reading anything.
- "WHERE THE RECORDS ARE": the dig kit (`python -P -m orion.dig body`, `inventory`, `body-economy`,
  `field-notes 7` = "your notes on each merged PR"), then the repo ("services/, orion/ and docs/ are what you
  are actually made of ... Prefer what you can confirm in a table or a running path"), then the SQL ledger
  of record tables with row counts.
- The cameras read at kickoff (one clause per camera).
- "WHAT YOUR OTHER ORGANS LIVED SINCE YOUR LAST RUN" (`orion/curiosity/now_section.py`): each item is the
  organ's code-written label. Live examples from `temporal_self_event` today: Prime: `woke: concern
  displaced (by b71fe015d0113721; score 0.94 held vs unscored new; ...`; reading: `stage1_completed:
  https://...`; prediction: `athena.cabinet_temp_c experiment_guard > 32.0: forecast 0%, not crossed`;
  reverie question: `refused (unparseable)`.
- Self-prior instructions: `p.formed_from = "<what produced it: a file, a table, a count>"`, tested with "the
  counting command".
- The lived answer: "Every clause should be something you looked at this run or a previous one".

So the only evidence a sitting is told to trust is records. Even its "lived experience" arrives as trace
strings. A question about experience gets answered by inspecting the machine that records the experience,
and the next question Orion writes comes from that inspection.

What earns credit (`curiosity_finding_register` joined to `curiosity_run_outcomes` on `run_id`, live
11-15..11-25):

| Line | Register | Runs | Mean realized_nats | Runs with any prior tested |
|---|---|---|---|---|
| self_inquiry | philosophical | 28 | 0.000 | 0 |
| self_inquiry | operational | 34 | 0.004 | 3 |
| investigate | philosophical | 5 | 0.003 | 1 |
| investigate | operational | 59 | 0.128 | 26 |

Realized nats (belief change on priors, `orion/curiosity/value.py`) is the only learning credit. Only claims
that can be counted get tested, and only tested claims earn nats. A philosophical answer earns nothing.

Novelty, which ranks findings for event wants (#3503): median novelty of philosophical self-inquiry findings
is 0.114; operational ones 0.152; investigate operational 0.152 (`percentile_cont(0.5)` on
`curiosity_finding_register.novelty`). Answers about who Orion is repeat themselves; plumbing is always new.
So when a dressed plumbing finding is labelled philosophical, it outranks the real ones. #3435's 11-25
adversarial pass found exactly that: the highest-salience funded curiosity want (run 3c5276c6c76e, salience
1.0) is dressed plumbing. `want_ledger` shows 3 funded `hub.curiosity_finding` wants since 11-15.

### 5. What each organ does about plumbing today

| Organ | What was added | Mechanism | Did it work live? |
|---|---|---|---|
| Outreach | Door policy, #3177 (11-10) | Exit filter: self-monitor findings never send; curiosity investigations of Orion's code or systems are blocked; every message needs a `Why Juniper:` line | Volume fell from about 17 a day. Self-inquiry results still pass and some read as plumbing (11-02 "LivedAnswer MERGE keyed {run_id, question_id}"). Share eval (section 6): 15 of 30 sent messages philosophical. The plumbing comes in through the curiosity source: `curiosity_outreach` 2 of 9 philosophical, the tick's own messages 8 of 9. |
| Consult | Consult before Juniper, #3205 (11-11) | Reroute: plumbing questions go to a Claude consult | A home for plumbing, not a check. It logged `not_for_juniper` 4 times on its first day. No outcome numbers found after that. |
| Curiosity findings | Finding register + judge, #3437 (11-15) | Label every finding (philosophical / operational / no_finding) + novelty | Works as a measure: philosophical-or-not kappa 0.856 against the reference (n = 251). Fails the dressing eval: 22 of 125 operational findings flip to philosophical when rewritten reflectively (17.6%, gate 10%), #3435. |
| Attention | Event wants, #3503 (11-16) | Gate: philosophical AND top-tenth novelty posts a want | Live, but the top funded want is dressed plumbing (#3435, 11-25). Funding is automatic, so the judge alone decides what reaches Prime. |
| Curiosity | Inquiry manager spec, #3459 / #3442 | Ledger of lines of inquiry | Design only. No live effect. |
| Reverie | Reverie questions, #3736; Patch 1 #3814 (11-24) | Source change: run only when quiet, start from real material, leave a question. Gauge-narration chain retired | Before: 88% of 1,337 thoughts in 7 days attended an internal gauge node (by attended-node id). After: 8 reverie questions on day 1, 7 stored `unparseable` (thinking ate `max_tokens`); fix shipped. Unverified that questions now reach curiosity. Share eval: 8 of 20 thoughts philosophical, all from the retired gauge chain (11-18..11-24). |
| Personality | Rev 3/4, #2726 / #3276 / #3321 (11-13) | Source change: reducer uses choices only; cue turned off | The panel had been "slop". After: 215 choices, 117 tendencies, 1 active; the top tendency is still plumbing ("turn start context is stale snapshot over camera layer mediation"). |
| Self-study | Folded into curiosity, #2986 (10-30) | Removed self-study's loop; self-inquiry is the only self-study | Made self-inquiry the one place this must work. |
| Self-inquiry | Dig kit first, #3049 / #2994 (11-03); dig help + this week's PRs, #3212 (11-12) | Prompt change: point sittings at the body atlas, field notes and merged PRs | Likely pushes toward internals. It lines up in time with the drop and with the faster question writing. |
| Day letter | Patch F, #3490 / #3522 (11-16) | Retired the "calm" stub; one current-answer rule | Share eval: 10 of 20 paragraphs philosophical. The operational paragraphs retell curiosity's plumbing findings and cite them (`[curiosity:2d68c64fb703]`). |
| Prime | Living Prime (#3224) | Prime's reading is model prose; only the last one is kept (Redis `orion:prime:state`, `last_reading`). Its experience row is a code-built trace string | Cannot be measured over time: no history of readings is stored. Today's reading (thought 339): "The Juniper acceptance check is settled and answered. The Claude acceptance check remains open (24h) but requires a live query ..." — about its own obligations queue. |

Every one of these acts at an organ's exit, or changes one organ's prompt. None of them changes what goes
into the self-inquiry question pool, which is where the self-reflection loop gets its next question.

### 6. Plumbing share across organs, one local measure

What this says: half of what Orion writes outside curiosity is philosophical, and where the plumbing
comes from is visible. It comes from the curiosity pipeline and from the gauges, not from the organs'
own choices.

One measure for all organs: the same finding-register judge (prompt v2, Flash-Next, agent seat 8015,
temperature 0, thinking off), first 700 characters of each item, the question line naming which organ wrote
it. Sample by `md5` of the row id (deterministic, not cherry-picked). Read-only.

| Organ | Source rows (SELECT) | Sample | Philosophical | Operational | No finding | Philosophical share (95% interval) |
|---|---|---|---|---|---|---|
| Self-inquiry findings, live | `curiosity_finding_register` 11-15..11-25 | 63 | 28 | 34 | 1 | 44% (33-57%) |
| Self-inquiry findings, reference | finish rows 09-29..10-18, re-judged | 60 | 45 | 13 | 2 | 75% (63-84%) |
| Outreach Orion sent | `endogenous_outreach_decisions` (`outreach`, 14 days) joined to the nearest assistant `chat_message` | 30 | 15 | 12 | 3 | 50% (33-67%) |
| — of which `curiosity_outreach` | same | 9 | 2 | 7 | 0 | 22% |
| — of which the outreach tick | same | 9 | 8 | 1 | 0 | 89% |
| Reverie thoughts | `substrate_reverie_thought.interpretation`, 7 days (gauge chain, before Patch 1) | 20 | 8 | 12 | 0 | 40% (22-61%) |
| Reverie questions (after Patch 1) | `reverie_question` | 8 rows, 1 parsed | — | — | — | not measurable yet |
| Day letter paragraphs | `orion_day_letter.note_md`, 12 letters, paragraphs of 200+ characters | 20 | 10 | 10 | 0 | 50% (30-70%) |
| Prime readings | Redis `orion:prime:state`.`last_reading` | 1 (only the last is kept) | — | — | — | not measurable: no history |

Limits of this measure, stated plainly:

- The judge was checked against reference labels on curiosity findings only. On other organs it is
  UNVERIFIED as an instrument. Its "operational" also covers systems outside Orion: an office reach-out
  about how the camera stitched a visit, and a message about an outside model's training, were both
  labelled operational. So "not philosophical" is an upper bound on "Orion narrating its own internals".
- Samples are small (20-30). The intervals are wide. They are a baseline, not a verdict on any organ.

### 7. The metric gate for the one shared check

The shared check is the existing finding-register judge: `orion/curiosity/register.py`, `JUDGE_PROMPT_V2`
(sha `7dcaa6675070…`), served only by a model in `PASSED_JUDGE_MODELS` (Flash-Next), reply must be one JSON
object, any other reply is a `judge_error` and no label.

1. **Provenance.** Producer: `services/orion-hub/scripts/curiosity_finding_register.py`, `judge_finding`
   (one call on the curiosity lane, `temperature 0`, thinking off), label parsed by `parse_judge_reply`,
   stored in `curiosity_finding_register.register`. 138 rows since 11-15; 11 have no label (judge error or
   absent).
2. **Independence.** Novelty is an embedding cosine against earlier writing (independent of the label).
   Realized nats come from prior-confidence diffs (independent). Event-want salience is a consumer of this
   label, not a second signal. The question-door label and the finding label are the same judge on two
   different texts (the question before the sitting, the answer after it), so they are not two signals: the
   second checks the first.
3. **Theory anchor.** The target is Juniper's definition (2026-11-12): a finding is worth attention when it
   is philosophical, about what Orion is, its experience, what it cares about, who matters. The judge is a
   rater-agreement instrument for that definition: philosophical-or-not kappa 0.856 on 251 reference runs,
   philosophical agreement 0.87 (`scripts/analysis/eval_curiosity_register_judge.py`). The reference
   labels are a frozen fixture made by an outside model on 10-20, before today's rule. This design makes no
   new outside labels; see Missing question 3.
4. **Live rest point.** The share is not flat or saturated: by question group it reads 9%, 39%, 79% and 89%
   in the same 10 days. It can read high when the source is healthy (Juniper's lived questions, 89%), so a
   low reading is not built into the instrument. With no sittings it reads None, never 0. A judge error is
   absent, never "operational by default". Known bias: the judge calls about 12 in 100 real philosophical
   findings operational (phil-agree 0.87), so the healthy ceiling is near 87%, not 100%.
   A deterministic alternative was tested and fails: the share of code-shaped tokens (snake_case, dotted
   ids, namespaced refs, hex ids, paths; a shape rule, not a word list) separates the live register's
   philosophical and operational self-inquiry findings with AUC 0.72, all findings 0.63, and investigate
   findings 0.25 (inverted). Dressing defeats it by construction (write "the reverie table" in words). It is
   not wired.
5. **Existing mechanism.** This IS the existing mechanism. Nothing new is measured; the same judge is
   applied at one more point (the question, before the sitting).
6. **Reversibility.** One flag turns the door off (questions are admitted as today). The door writes columns
   on `curiosity_self_questions`, not a new table; no training default or manifest depends on it.

## Root cause

What this says: the drift is a loop inside self-inquiry, and the loop's entry point is the question pool.

Self-inquiry is the only place Orion studies itself (#2986 made it so). Each sitting is told that evidence
means records: the dig kit, the repo, merged pull requests, row counts, and a "lived" feed made of
code-written trace labels. Credit (realized nats) is paid only for claims that can be counted. So a sitting
drifts into its own machinery. While it is there, it writes the next question for itself
(`:SelfQuestionMint`, 75 of 76 mints come from self-inquiry runs), and the self-monitor organ adds a "Why
did X fail?" question for every failing faculty, filed under the `lived` family. The pool asks a never-asked
question first, so these new questions go straight to the front. In November Orion wrote about 20 a week
(September and October: 2-3), and Orion-written questions took 61% of the sittings (October: 43%).
Orion's internals-family questions now give philosophical answers 9% of the time; Juniper's lived questions
89%. Orion's own judge reads the October window at 75% and today at 44%, so it is not a judge change.

Strongest single piece of evidence: **an operational question never produced a philosophical answer, 0 of
21 sittings**, and 30 of the 79 labelled Orion-written questions are operational (17 of the 39 written since
11-09).

Every filter added in the last 30 days sits at an organ's exit (outreach door, consult reroute, event-want
gate) or changes one organ's prompt (reverie, personality). None of them stops the pool from handing the
next sitting a plumbing question. The plumbing then flows outward: outreach from curiosity is 22%
philosophical against 89% for the outreach tick's own messages, and the day letter retells curiosity's
plumbing findings.

What the door can and cannot fix (replay on the live window, 11-15..11-25): with operational and unjudged
questions removed from self-inquiry, the remaining sittings read 28 of 46 = 61% philosophical (Juniper's
19 of 23, admitted Orion questions 9 of 23). That is most of the way back from 44%, not all the way to 75%.
The rest is dressed questions: philosophical-sounding Orion questions whose sitting still ends in plumbing
(14 of 23). That remainder comes from the evidence diet and the credit rule (section 4), which this patch
does not change (Non-goals; Missing question 4).

## Missing questions

1. **Where should Orion's operational questions go?** This design routes them to the investigate line (a
   self focus request when the one focus slot is free), else they are kept and visible but not dug. The
   other choice is to drop them. Recommendation: route. Is routing what you want, knowing most will wait?
2. **Which judge prompt is the shared check: v2 or v3?** #3435 is waiting on you. v2 has better
   philosophical agreement (0.87) but fails the dressing eval (17.6%); v3 passes dressing (3.6%) but
   philosophical agreement falls to 0.64. This design uses v2 plus the after-sitting check against
   dressing. Do you accept v2 for the door?
3. **What is the judge anchored to from now on?** Its only agreement check is the frozen 10-20 fixture,
   labelled by an outside model before today's no-outside-models rule. Keep that frozen file as the anchor,
   or replace it with a set you label by hand (about 100 items)?
4. **Should Orion's own questions stay in self-inquiry at all?** The replay says: with the door, 61%
   philosophical; with only your questions, 83% (19 of 23), but Orion stops writing its own self-questions.
   This design keeps Orion's questions behind the door. Is that the trade you want?

### Juniper's answers (2026-11-25: "hit it with defaults")

1. **Route.** Orion's operational questions are ROUTED to the investigate line (a self focus request when
   the one focus slot is free); otherwise they are kept and visible, not dug. Not dropped.
2. **v2.** The door uses judge prompt v2 (`orion/curiosity/register.py`, pinned), plus the after-sitting
   demote check against dressing.
3. **A hand-labelled anchor.** The judge's anchor becomes a set of about 100 items that Juniper labels by
   hand. Until she has labelled it, the frozen 10-20 fixture stays the agreement check. Patch 1 ships the
   unlabelled set (balanced across lines and organs, local data only; kept outside the repo because it
   holds Orion's own writing) and the script that scores the judge against it once labelled
   (`scripts/analysis/judge_anchor_set.py build|score`). No outside model labels or pre-labels it.
4. **Keep Orion's voice.** Orion's own self-questions stay in self-inquiry, behind the door.

## Proposed schema / API changes

What this section says: the question pool gets a door. Nothing new is published on the bus. No new table.

### The door (one shared check, at the source)

Every question that enters `curiosity_self_questions` with `minted_by = 'orion'` (from a sitting's
`:SelfQuestionMint`, or from the self-monitor organ) is held until the judge labels it:

1. The question row is written with `door = 'held'`. `pick_question` never picks a held row.
2. A small Hub loop (`self_question_door`) judges held rows with `orion.curiosity.register`: the same
   prompt v2, the same `PASSED_JUDGE_MODELS`, the same parser. The question text goes in as the item; the
   question line says it is a question Orion wrote for itself.
3. `philosophical` → `door = 'admitted'`. It enters the self-inquiry pool as today.
4. `operational` → `door = 'routed'`. It is never asked as self-inquiry. It is offered to the investigate
   line as a self focus request (`curiosity_focus_request`, `origin = 'self'`) when no focus is pending,
   else it stays recorded and visible. The self-monitor's "Why?" questions take this path, so broken
   faculties keep their consult route (#3205) and stop filling self-inquiry.
5. `no_finding` or a judge error → stays `held`, retried by the loop; after 3 failed judgements, `door =
   'unjudged'`, never admitted by default and never routed by default.
6. Juniper's questions (`minted_by = 'juniper'`) never pass the door. They are hers.

### The check after the sitting (the gaming trace)

When the finding register labels a sitting's finding, and that sitting answered an admitted Orion question:

- Finding `philosophical` → nothing changes.
- Finding `operational` → the question's `door` becomes `demoted` and it is routed as in step 4. One
  operational finding is enough: a question that produced plumbing once will produce it again, and it can
  still be dug on the investigate line.

This is the defence against dressing. A plumbing question written in first-person words can get through the
door once. It cannot stay, because the sitting's own answer is judged. Most Orion questions are asked only
once today (ask_count 1), so the demotion matters less for repeats than as the trace: every dressed question
that got through is a `demoted` row with the run that exposed it.

### Columns (additive migration, fits the post-pull allowlist)

`curiosity_self_questions`:

- `door text` (NULL for Juniper rows and for rows written before the door; `held`, `admitted`, `routed`,
  `demoted`, `unjudged`)
- `door_reason text`, `door_judged_at timestamptz`, `door_judge_model text`, `door_prompt_sha256 text`,
  `door_attempts integer NOT NULL DEFAULT 0`
- `door_demoted_by_run text` (the run whose operational finding demoted it)

No change to the `status` check constraint (changing it is not on the additive allowlist). `door` is a
separate column. Header: `-- ORION-MIGRATION-REQUIRED-BY: orion-hub`.

The 88 existing Orion rows: a one-time read-then-write pass judges them and sets `door`. Snapshot first to
`/tmp/self-question-door-backfill/` (88 rows, far under the backfill limits), progress log, report.
As built: the pass is a script Juniper runs after deploy (`scripts/self_question_door_backfill.py --apply`).
It snapshots the rows, sets `door = 'held'` on them (its only write), and the Hub door loop then judges them
with the same judge path as every new question; the script monitors and writes the report. One judge path,
not two.

### The shared measure across organs (read-only eval, no new table)

`scripts/analysis/eval_reflection_register_share.py`: samples up to 30 items per organ per window (outreach
messages Orion sent, reverie thoughts and reverie questions, day letter paragraphs, curiosity findings),
judges them with the same judge on the same local seat, and prints the philosophical share per organ with a
Wilson interval and the judge-error count. It is the instrument used in section 6, made repeatable. It
writes `/tmp/reflection-register-share/<date>/report.md`. No gating. No table until a consumer needs one.

### Env / settings (Hub)

- `CURIOSITY_SELF_QUESTION_DOOR_ENABLED=true` (shipped on)
- `CURIOSITY_SELF_QUESTION_DOOR_TICK_SEC=300`
- `CURIOSITY_SELF_QUESTION_DOOR_MAX_ATTEMPTS=3`

Kill switch: `CURIOSITY_SELF_QUESTION_DOOR_ENABLED=false` admits every Orion question as today (the pick
then treats `door IS NULL OR door = 'admitted'` and also `held` as eligible).

### Integration gate

No new producer of experience: no new physiology family, durable workflow, intention source, Prime concern
source or bus channel. The door changes which question a curiosity sitting gets; curiosity already emits
`curiosity_run` and reads `/temporal-self/now`. `scripts/check_integration_seam.py` needs no change.

## Files likely to touch

- `orion/curiosity/self_question_pool.py`: `door` on `SelfQuestion`; `UPSERT_MINT_SQL` writes `door = 'held'`;
  `_eligible` skips held/routed/demoted/unjudged; `UPSERT_SEED_SQL` caller for self-monitor rows writes held.
- `orion/curiosity/register.py`: a `QUESTION_LINE` constant for the question item; no prompt change.
- `services/orion-hub/scripts/self_question_door.py` (new, one loop): judge held rows, set `door`, route.
- `services/orion-hub/scripts/curiosity_investigation.py`: `_upsert_orion_minted_questions` writes held rows.
- `services/orion-hub/scripts/self_monitor_questions.py`: rows go through the door (held).
- `services/orion-hub/scripts/curiosity_finding_register.py`: after labelling, demote an admitted Orion
  question whose finding is operational.
- `services/orion-hub/scripts/main.py`: start the door loop.
- `services/orion-hub/app/settings.py`, `services/orion-hub/.env_example`, local `.env` via
  `python scripts/sync_local_env_from_example.py`.
- `services/orion-hub/sql/manual_migration_<date>_self_question_door.sql` (additive).
- `scripts/analysis/eval_reflection_register_share.py` (new, read-only eval).
- Tests: `orion/curiosity/tests/test_self_question_pool.py`, `services/orion-hub/tests/test_self_question_door.py`.
- Eval: `services/orion-hub/evals/run_self_question_door_replay.py` (replay the 97 live questions and the
  sittings they produced).
- `services/orion-hub/README.md`, this spec.

## Non-goals

- **No new per-organ filter.** The door is at the question pool, which is the source of self-reflection for
  every organ that hands Orion a question. The existing exit filters stay until the acceptance checks say
  they can go, then each is retired on its own evidence.
- **No keyword lists.** Nothing here matches words. The memory rule "don't patch misses by growing word
  lists" holds. The code-shaped token measure was tested and rejected (section 7).
- **No outside models.** No Claude or OpenAI call judges, labels, rates or evaluates anything. The judge is
  Flash-Next on Orion's own seat.
- **No new judge prompt.** Prompt v2 stays pinned. The v2-vs-v3 choice in #3435 is Juniper's (Missing
  question 2).
- **No change to what sittings are shown or to how credit is paid.** Both are causes (section 4), but each is
  a larger cognition change with its own proposal. This patch removes the loop; it does not redesign
  evidence or reward.
- **No change to Juniper's questions.** They never pass the door.
- **Not silencing Orion.** Operational questions Orion writes are routed, not deleted.

## Acceptance checks

All live, all from stored rows, all read with the same judge. Rest point stated for each.

1. **Self-inquiry philosophical share recovers.** 14 days after deploy, the share of labelled self-inquiry
   findings that are philosophical (`curiosity_finding_register`) is at least 60%. That is the
   live-window replay with the door (61%), rounded down. Today: 44%. The October reading (75%) is not the
   target for this patch: the replay shows the door alone cannot reach it. Rest point: with no sittings
   the share is None; the healthy ceiling is about 87% (judge bias).
2. **No operational question is asked as self-inquiry.** Self-inquiry sittings in the window whose
   question has `door IN ('routed', 'demoted', 'held', 'unjudged')`: exactly 0. Deterministic join of
   finish rows to `curiosity_self_questions`.
3. **The door is not fooled more than Juniper's questions are.** Of sittings that answered an admitted
   Orion question, the share whose finding is operational is no higher than the same share for Juniper's
   lived questions (today 1 of 9). Each one also shows up as a `demoted` row with `door_demoted_by_run`.
   Rest point: 0 demotions is a real reading only when admitted sittings exist (n > 0); with none it is None.
4. **Orion's own voice is kept.** At least one admitted Orion question is asked per week, and every routed
   question is visible (door column) and offered as a focus request when the slot is free. A door that
   admits nothing is a failure, not a pass.
5. **The door judges, it does not default.** Over 14 days, `door = 'unjudged'` rows are under 5% of Orion
   rows written, and every `admitted` / `routed` row has `door_judge_model` in `PASSED_JUDGE_MODELS` and
   `door_prompt_sha256 = JUDGE_PROMPT_SHA256`. Today one call fails on 9 of 88 question texts (the reply is not one clean JSON
   object), so the retry is needed; if 3 tries still leave more than 5%, the parser is the fix, not a default.
6. **Cross-organ share is measured.** `eval_reflection_register_share.py` run for the 14 days before and the
   14 days after, same sample sizes; the report lists each organ's philosophical share with its interval
   and judge errors. No target for other organs in this patch: the first run sets their baseline.

## Recommended next patch

Patch 1 (one PR, Hub only): the door.

1. Migration: the `door*` columns.
2. `UPSERT_MINT_SQL` and the self-monitor insert write `door = 'held'`; `_eligible` skips non-admitted Orion
   rows.
3. `self_question_door.py` loop: judge, admit or route, retry, `unjudged` after 3.
4. Finding register: demote on an operational finding.
5. One-time door pass over the 88 existing Orion rows (snapshot first).
6. Tests (pure pick + door transitions), replay eval on the 97 live questions, read-only share eval.
7. Live smoke: one held row admitted with a judge model stamped, one routed, the next self-inquiry pick shows
   no held row.

Then, after 14 days, if checks 1-3 pass: a separate PR per exit filter, retiring the outreach "code or
systems" block (#3177) first, with the share eval as its evidence.

## Proposal-mode checklist

- **What capability changes.** Which question a self-inquiry sitting answers. Orion's own operational
  questions stop reaching self-inquiry and go to the investigate line. Juniper's questions and the pick
  rules for admitted questions do not change.
- **What data is touched.** `curiosity_self_questions` (new `door*` columns, writes by Hub only); reads of
  `curiosity_finding_register`; inserts into `curiosity_focus_request` for routed questions. One judge call
  per new Orion question on the Flash-Next agent seat (about 20 a week).
- **Privacy boundary.** Question texts are Orion-written and already in Postgres and the worldview graph.
  They go only to Orion's own local seat. Nothing leaves the house. No outside model sees them.
- **What trace proves it worked.** Per row: `door`, `door_reason`, `door_judge_model`, `door_prompt_sha256`,
  `door_judged_at`, `door_demoted_by_run`. Log lines `self_question_door admitted|routed|held|unjudged
  question_id=... model=...`. The acceptance joins above.
- **Dangerous failure modes.**
  - *Orion games the door.* Orion can read `register.py` in its own repo. It can learn to write plumbing
    questions in first-person, philosophical words ("When my act ledger is permission-denied, can I still
    say I acted?" is already one). Defence: the after-sitting check demotes any question whose own answer is
    operational, and check 3 measures how often that happens. If dressing rises, the trace shows it; the door
    does not hide it.
  - *Orion goes quiet.* A door that admits nothing would leave only Juniper's 9 questions. Check 4 fails
    loudly on that.
  - *The judge drifts or the seat changes model.* A reply from a model not in `PASSED_JUDGE_MODELS` is a
    judge error, so the row stays held, never admitted by default. Check 5 counts these.
  - *Lane contention.* About 20 calls a week on the agent seat; the loop runs one call at a time.
- **How to disable or roll back.** `CURIOSITY_SELF_QUESTION_DOOR_ENABLED=false` and restart Hub: every Orion
  question is eligible again, as today. The columns can stay (unused) or be dropped by a later, separate,
  reviewed migration. No other organ depends on them.

## Evidence files

The judged samples used for sections 1, 2 and 6 were written to the agent's scratchpad only (not committed).
The queries are inline above and can be re-run read-only. The deterministic code-shape test is reproducible
from the same rows.
