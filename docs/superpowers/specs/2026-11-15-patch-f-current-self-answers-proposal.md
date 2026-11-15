# Patch F proposal: one rule for "Orion's current answer" (#3396)

Status: proposal, no code. Refs #3396. Parent spec:
`docs/superpowers/specs/2026-11-15-day-letter-findings-fix-design.md` (Patch F).
All references read on `origin/main` at `b6eeb36f0`. Live reads taken 2026-11-15 23:58Z.

## Recommendation

**Add one small shared helper: the newest answer per question. Use it in the one reader that
is wrong today. Do not use `revises` to decide what is current. Leave the history readers
alone.**

Orion answers standing questions about their own life ("lived" questions). Each answer is a
node in Orion's graph (`LivedAnswer` in `orion_worldview`), keyed on the run and the
question. A later run writes a new node. The old one stays.

## What I measured

- 273 answers for 69 questions. 39 questions have more than one answer. One has 14. So 204
  of 273 answers (75%) are old ones.
- 208 answers carry a `revises` value. 3 of them name something that is not a run (for
  example the text `none`). 28 do not name the answer just before them.
- **The spec's rule gives the same result as "newest per question" on all 69 questions.**
  The rule was: drop an answer that a later one names in `revises`, else take the newest. A
  later answer can never name the newest one, so `revises` never changes the outcome. It is
  hand-filled lineage, and it is good for display only.
- **The ticket's summary is not quite right.** `read_current_lived_answers`
  (`orion/curiosity/self_inquiry.py:560`) does not read `revises` either. It is plain
  newest-per-question. No reader uses `revises` to filter.

## Every reader

| Reader | What it returns | Wants | Verdict |
|---|---|---|---|
| Curiosity prompts: `read_current_lived_answers`, called from `services/orion-hub/scripts/curiosity_investigation.py:3513, :3757` | newest per question, from the graph | current | Correct. Move its rule into the helper. |
| **Reading resonance pool**: `orion/world_pulse_read/resonance_records.py:36-79` | every answer as its own match target | current | **Wrong. Fix this one.** |
| Novelty corpus: `orion/curiosity/register.py:75` | every answer | history | Leave. It asks "has Orion said this before?". With the helper, a repeat of an old answer would score as new. |
| Run story and footprint: `orion/curiosity/atlas.py:248`, `read_lived_answers_for_run` (`self_inquiry.py:600`), `read_lived_answer` (`:460`) | the answers one run wrote | history | Leave. They show what a run did. |
| Scoring script: `scripts/analysis/score_curiosity_findings.py:108` | every answer, by run | history | Leave. |
| Backfill scripts in `services/orion-hub/scripts/` | every answer | history | Leave. One-shot tools. |
| Postgres mirror readers (`self_concept_history` rows `self:lived:*`): the self panel (`orion/curiosity/self_panel.py:136`), the chat context lane (`orion/substrate/felt_state_reader.py:225-260`), the self-sense eval (`orion/evals/self_sense_runner.py:64`), the reverie self-study seed (`services/orion-thought/app/store.py:1005`) | newest row per question, from the mirror | current | Correct by their own rule. See "two rules" below. |

**The one wrong reader.** When Orion reads world news, a step looks for places where the
reading touches something Orion already holds. It offers every answer, old and new, as a
candidate. In the stored packets from 11-11 to 11-14 (`reading_resonance`, 109 rows), 32
candidates were lived answers. **24 of the 32 were already replaced when the packet was
built.** So Orion was asked "does this article bear on what you think?" about views Orion
had dropped.

```sql
select c->>'ref_id', r.request_json->'brief'->'packet'->>'built_at'
from reading_resonance r,
     jsonb_array_elements(r.request_json->'brief'->'packet'->'candidates') c
where c->>'ref_kind' = 'lived_answer';
-- compare each ref (question@run) with the newest LivedAnswer for that question at built_at
```

**Two rules for "current" exist today.** The graph readers take the newest answer. The mirror
skips an answer that has no evidence, so its readers take the newest answer with evidence.
They agree on 60 of 69 questions. They differ on 5, and 4 of the 5 are this case. 4 questions
have no mirror row at all. 57 of 273 answers have no evidence. For 8 questions the newest
answer has none.

## The six points

- **Capability that changes:** reading resonance compares an article with what Orion thinks
  now, not with every view Orion ever held. The older answer still rides along as "the
  previous answer", as it does today.
- **Data touched:** read paths only. No node, row or mirror entry is written, changed or
  deleted. Old answers stay in the graph.
- **Privacy boundary:** none between Juniper and Orion. Lived answers can name other people.
  This patch sends fewer answers to the resonance step, never more, and nothing leaves the
  host that did not before.
- **Trace that proves it worked:** new `reading_resonance` packets. Every candidate with
  `ref_kind = lived_answer` is the newest answer for its question at `built_at`. Baseline: 8
  of 32.
- **Dangerous failure mode:** the helper is put into a history reader. The novelty corpus
  then forgets 204 old answers, and Orion's repeats score as new thoughts. A second risk: a
  run that is cut short leaves an answer with no evidence, and that draft replaces a good
  answer everywhere. Question 1 below decides this.
- **Disable or roll back:** revert the commit. No data to restore.

## Smallest first patch

1. One pure function in `orion/curiosity/self_inquiry.py`: answers in, newest per question
   out, with the tie-break that `read_current_lived_answers` already uses (`written_at`, then
   `run_id`).
2. `read_current_lived_answers` calls it. No behavior change there.
3. `resonance_records.py` calls it before it builds the `lived_answer` match targets. The
   `previous_answer` neighbor still reads the full list.
4. A comment on each history reader that says it wants history on purpose.

## Non-goals

- No delete or "superseded" mark on old answers in the graph.
- No use of `revises` as a filter, and no repair of the 28 stale `revises` values.
- No change to the mirror, to the history readers, or to how answers are written.
- No merge of the graph rule and the mirror rule beyond what question 1 decides.

## Acceptance checks (corrected)

The spec said "every reader returns one answer". Four readers want history on purpose, so:

| Check | Pass |
|---|---|
| Unit test: one question with a refuted answer and a later corrected one. | The helper, the prompt reader and the resonance pool return the corrected answer only. The novelty corpus and the run story still return both. |
| Unit test: two answers with the same `written_at`. | The result is the same on every call. |
| Live, 3 days after deploy: the packet query above. | Every lived-answer candidate is the newest for its question at `built_at`. Was 8 of 32. |

## Question for Juniper

1. Can an answer with no evidence count as Orion's current answer outside the curiosity
   prompt, yes or no? I recommend no: the resonance pool takes the newest answer that has
   evidence, and falls back to the newest of any kind. That matches what chat and the self
   panel already show. The curiosity prompt keeps "newest of any kind", because it uses the
   answer to fill in `revises`.

## UNVERIFIED

- Whether any of the 24 stale candidates led to a wrong resonance verdict. I counted what was
  offered, not what the model then said.
- The one question where graph and mirror differ and the newest graph answer has evidence
  (`lived.self_monitor.act.actions.journal.make.walkway_day.broken.20261113T0757Z`). I did
  not trace why the mirror lags.
- Why 4 questions have no mirror row.
