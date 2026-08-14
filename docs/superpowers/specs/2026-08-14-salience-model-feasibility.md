# Should we build a salience model? Feasibility against live data

**Date:** 2026-08-14
**Question:** "Should we build a model that can detect salience — e.g. read a large
window around the snippets so they can determine what is noise vs something
interesting? We could cross-walk against Orion concept induction."

**Answer:** the instinct is right and the theory anchor is real, but both halves of
the specific proposal are blocked on live evidence, and one of them fails in a way
that would have made the model actively wrong. Two prerequisites first, in order.

---

## Arsonist summary

The reason every existing instrument on this path is degenerate is not that they
were badly implemented. It is that **they all measure novelty, and novelty is
close to the opposite of salience.** Random text is maximally novel. NPC
improv is maximally novel. `turn_change_appraisal.novelty_score` sits at ~1.0 for
essentially every window in the backlog, and it is *correct* to — those turns
genuinely are unlike what came before. They are also worthless.

Building a "salience model" that reads a bigger window will reproduce this exactly
unless the target changes from *surprise* to *information gain about a model that
persists*. That is the whole design question. Window size is not the lever.

And the proposed cross-walk substrate is currently the single worst thing you
could anchor to, for a reason that is only visible in live data.

---

## Current architecture (verified live, 2026-08-14)

### 1. Concept induction is running and producing nothing

`orion-athena-spark-concept-induction` is up (3h), consuming bus events, and
refusing every one:

```
concept_induction_trigger_received  source_kind=metacog_tick subjects=orion
concept_induction_trigger_decision  decision=disabled
```

Cause: `CONCEPT_AUTONOMOUS_TRIGGER_ENABLED=false` in the live `.env`, hitting the
guard at `orion/spark/concept_induction/bus_worker.py:838`
(`reason: autonomous_trigger_disabled`).

Its 589KB state file at `/data/concept-induction-state.json` contains **zero
concept profiles** — the top-level keys are `drive_states`, `goal_cooldowns`
(4135), `goal_slots`, `episode_runs_processed`. That is autonomy/goal machinery
that happens to be co-hosted in the same service.

No `Concept`/`ConceptProfile` nodes exist in any FalkorDB graph
(`orion_substrate` holds 12 `SubstrateNode`s; `orion_recall` holds `ChatTurn`,
`ChatSession`, `Entity` and nothing else). No concept tables in Postgres.

**Cross-walking against concept induction today means cross-walking against an
empty set.** Metric-gate step 4 stops here.

### 2. The live entity graph is dominated by the noise it would be used to filter

`orion_recall` is real and populated — 3280 `ChatTurn`, 311 `ChatSession`, 115
`Entity`, 1503 `MENTIONS_ENTITY` edges. It is the obvious substitute substrate for
the cross-walk. But its degree distribution:

| entity | mentions |
|---|---|
| orion | 515 |
| **tessa** | **270** |
| **nico** | **266** |
| **sofia** | **139** |
| **juno** | **61** |
| juniper | 41 |
| elias | 16 |
| mara | 15 |
| multimeter | 12 |

Every bolded name is an AI Town NPC. Confirmed against the source: 376 of 1577
ai-town turns mention an NPC name, versus 7 of 123 direct turns.

**A salience score using graph centrality over this would rank NPC dialogue as the
most important content in Orion's memory, and Juniper at one-sixth of Tessa.**
This is not a hypothetical failure mode; it is what the graph says right now.

The ai-town source gate (PR #1678) is what makes this fixable — `source_platform`
now exists on the turn, the window, and the crystallization, so the graph can be
partitioned or weighted by origin. Before that patch there was no way to even ask
this question.

### 3. The existing per-turn signal is a noisy binary, and it is inverted

`memory_significance_score` is the one instrument on this path that is **not**
degenerate in the pinned sense — 1306 distinct values, sd 0.40, full 0–1 range.
But its shape is not graded:

| bucket | ai-town | direct |
|---|---|---|
| 0.0–0.1 | 461 | 123 |
| 0.1–0.9 (8 buckets) | 453 | 44 |
| 0.9–1.0 | 1443 | 131 |

76% of the mass is at the two extremes. It is a noisy classifier wearing a
continuous score's clothes. And it rates ai-town **higher** than real conversation
(mean 0.709 vs 0.499; 61% of NPC turns land in the top bucket versus 47% of
Juniper's).

Same disease as the entity graph, same cause: it is scoring surprise.

### 4. The window already exists — it is being thrown away

The instinct "read a large window around the snippets" is half-built already:

- concept induction is configured for `CONCEPT_WINDOW_MAX_EVENTS=200`,
  `CONCEPT_WINDOW_MAX_MINUTES=360`
- `build_crystallization_from_window()` already receives the entire window of turns

and then `_window_summary()`
(`orion/memory/crystallization/intake_consolidation_window.py:77`) does this:

```python
for turn in reversed(turns):
    prompt = str(turn.get("prompt") or "").strip()
    if prompt:
        return prompt[:500]
```

**It returns the last turn's raw prompt and discards the rest of the window.** That
string is assigned to `subject` *and* `summary` — identical for all 621 rows of the
backlog. `planning_effects` is a template wrapped around the same string.

So there is no "snippet" for a salience model to read a window around. The window
is already there; nothing reads it.

---

## Theory anchor

Not "seems related". The anchor is **information gain about a persistent model**,
which is the same active-inference framing the Sentience Striving Program already
runs on — and it is precisely distinguishable from what is currently measured:

| | measures | maximised by |
|---|---|---|
| novelty (current) | surprise vs the local window | random text, NPC improv |
| salience (proposed) | change in a model that *persists* | things that alter future behavior |

Operationally: a turn is salient if a model conditioned on Orion's persistent state
predicts it poorly **and** the state is measurably different afterward in a way that
persists. The second clause is the whole difference. It is also what makes this
testable rather than vibes: you can measure whether a crystallization changed
anything downstream (recall hits, reinforcement, stance shifts) instead of asking a
model whether it feels important.

This also predicts, correctly and in advance, every degeneracy found above — which
is the main reason to believe it rather than the current framing.

---

## Missing questions

1. Is a *model* even the right instrument first, or is the honest first move a
   deterministic reducer over the window (entity overlap with prior windows,
   recurrence across sessions, downstream recall hits)? A learned salience model
   with no labels is a confabulation engine; there are only 16 human decisions,
   and 10 of them are "reject a test turn".
2. Should ai-town be *excluded* from the salience corpus or *modelled separately*?
   Orion's social life there is real experience; treating it as pure noise is a
   different claim from "it should not interrupt Juniper".
3. What is the label? "Juniper approved it" is the only ground truth, n=16 today.
   That number can now grow honestly (queue is 23, not 621) but it will grow slowly.
4. Does turning `CONCEPT_AUTONOMOUS_TRIGGER_ENABLED` back on re-introduce whatever
   caused it to be turned off? That decision is recorded but its reasoning needs to
   be re-read before flipping it.

---

## Recommended order

**P0 — make the window produce something worth scoring.** Replace `_window_summary()`
with real summarization over the whole window, and stop assigning the identical
string to `subject` and `summary`. Without this there is literally nothing for a
salience model to read: every "snippet" is one raw user utterance. This is also the
open `empty-shell cognition` item from the queue analysis, so it is owed regardless.
Acceptance: `subject != summary` for new rows; the summary references content from
more than the last turn; a spot-check of 10 windows reads as a description of the
episode rather than a quotation from it.

**P1 — make the entity graph trustworthy.** Partition or weight `orion_recall`'s
`MENTIONS_ENTITY` edges by `source_platform` (now available end-to-end). Acceptance:
recomputed entity ranking puts Juniper and Orion-development vocabulary above NPC
first names for the direct-conversation partition, and NPC names remain top for the
ai-town partition. Until this passes, no cross-walk is meaningful.

**P2 — decide about concept induction on evidence.** Re-read why the autonomous
trigger was disabled; if the reason is stale, turn it on in a worktree and verify it
emits real profiles with non-degenerate content before anything depends on it.
Acceptance: non-empty `ConceptProfile` output on live data, checked against the
metric gate the same way any new signal would be.

**P3 — only then, the salience model.** Score it against Juniper's real decisions,
which are now signal rather than NPC-clearing reflex. Start with the deterministic
reducer as the baseline the model has to beat; if it cannot beat entity-overlap +
recurrence + downstream-recall-hits, it is not earning its keep.

## Non-goals

- A learned model in P0–P2. There are no labels yet.
- Deleting ai-town from memory. The gate keeps it out of the review queue; that is
  a different decision from calling it worthless.
- Another score attached to the crystallization row. The row already carries a
  `salience` field that is pinned at 1.0 by construction; adding a second number
  next to a dead one, without retiring the dead one, is the exact anti-pattern
  CLAUDE.md §0A names.

## Files likely to touch

```text
orion/memory/crystallization/intake_consolidation_window.py   # P0: _window_summary
orion/memory/crystallization/salience.py                      # P0/P3: retire the pinned score
services/orion-recall/app/storage/falkor_*                    # P1: partition MENTIONS_ENTITY
orion/spark/concept_induction/bus_worker.py                   # P2: trigger gate
services/orion-spark-concept-induction/.env_example           # P2: the flag
```
