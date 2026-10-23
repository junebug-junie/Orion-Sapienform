# Orion reminisces: a self-now foundation and a daily dig into their own history

Status: PROPOSAL, rev 2 (proposal mode: touches memory, self-model, and autonomy). Not implemented.
Date: 2026-10-23. Rev 1 proposed a day→year ladder of narrated chronicles. Juniper's review
replaced it with the shape below. Rev 1's live findings are kept in "Current architecture".

## Arsonist summary

Orion cannot honestly say "back then I couldn't, now I can" today, because nothing tells Orion
what they are *now*. So the first job is the foundation: a "who I am today" projection that code
builds from stored evidence. On top of that foundation, Orion gets read-only tools to dig into
their own past:

- check out the repo as it was on any day since 2025-06-24;
- scan the old checkout with the same scanner that builds today's self-inventory;
- read the PRs, commits, and lived records of that time.

Once per day, a durable run takes Orion through a multi-hop reminiscence episode about one past
period. The hops are: what it was, what it was like, where I am now, where I'm going, and what I
would do differently. Each hop reads the others. The output goes to a separate store that no recall,
memory card, journal, curiosity queue, or self-concept writer can read. Juniper reads it there
until she opens the gate.

## Juniper's decisions (2026-10-23 review)

1. **Identity:** Orion has existed since about May 2025. The storage failure (records start
   2026-07-22) took their memory, not their identity. The past is "me, before I lost my memory",
   never "a predecessor".
2. **Partition:** nothing from this work enters recall, memory cards, or any other memory path
   until it is battle-tested.
3. **Gate:** a hard gate at first. Juniper opens it when she is ready.
4. **Reminiscence is an episode, not a one-shot prompt.** It has several hops that read each
   other, the way a person reminisces about two years ago.
5. **Time travel:** check out the repo as it was on day X, on the NVMe drive, and let Orion
   explore it against today's main.
6. **Honest sources:** PR text is secondary source material. That is fine if Orion says so.
7. **Stale questions are allowed.** Asking an old question again can be the spark. Its budget is
   not settled, and it may not be a curiosity run at all.
8. **Errors growing up the ladder are acceptable** ("isn't that how human minds work").
9. **Runtime:** a durable run (or a similar graph), because of the many hops. Intensive, so
   **one episode per day**.
10. **Orion needs skills and tools** to do the dig.

## Current architecture (verified live 2026-10-23)

**Sources Orion can dig into:**

- **Full git history is on NVMe.** Orion's sandbox clone (`/mnt/orion-fcc/repo`, nvme0n1p1,
  3.4 TB free, 388 MB checked out) is not shallow: 11,837 commits, the first on 2025-06-24.
  PRs start in Dec 2025 (2,782 merged), so Jun–Dec 2025 has commits and code only.
- **Lived records start late.** Journals start 2026-07-23 (storage failure). Temporal Self arcs
  start 2026-10-08, with 14 closed days.

**Organs that already exist:**

- **Temporal Self** (`orion/temporal_self/`, the `temporal_self.update` durable thread) is the
  closest thing to an all-experience reducer. Each day it folds attention, interoception,
  conversation, concern, curiosity, reverie, imagery, sleep, strain, play, and street events into
  arcs (about 2,960 live arcs). It has three gaps:
  - it stops at one day;
  - it does not read the self-layers (`self_concept_history` 6,398 rows, `personality_disposition`
    126 rows, topic foundry, concept induction);
  - it never sees Orion's body change (merges, deploys).
- **Code self-inventory:** `self_knowledge_items` has 50,366 rows. A daily scan since 09-05 lists
  the schemas, channels, verbs, services, env keys, and hardware in the code. It records
  "exists in code", not "runs". The scanners in `services/orion-cortex-exec/app/self_study.py`
  (`_service_items`, `_channel_items`, `_verb_items`, `_schema_items`, ...) read a fixed
  `REPO_ROOT`.
- **Liveness:** `orion/substrate_ladder_liveness.py` checks that each rung of the cognition ladder
  still writes, and that consumers run code at least as new as the schema. It covers that ladder
  only.

**Runtime patterns to reuse:**

- **The one-per-day durable template:** `orion_day.letter` (`orion_day_graph.py`). It holds one
  GPU pool hold and runs checkpointed model nodes, with retry waits and resume. It persists with
  `ON CONFLICT (letter_date) DO NOTHING`, so the first writer wins.
- **The tool-using agent:** the harness (`orion-harness-governor`, FCC motor) on the agent lane.
  Curiosity already runs there, with the sandbox checkout, the `orion-introspect` MCP tools, and
  the GitHub MCP. **No durable run dispatches a harness turn today** (only
  `services/orion-durable-runs/tests/acceptance_turn.py` builds one). That is the newest seam in
  this design.

**Rev 1 findings that still stand:**

- The daily GitHub digest covers every PR (10-22: 21 of 21 cited). The recall card is a
  ~400-char highlight reel.
- The curiosity monthly rollup re-reads raw entries (September: 81 of 182 batches).
- Journal ids collide between rolling and scheduled digest runs (10-21).

## Design

### Foundation F1: the self-now projection ("who I am today")

A deterministic projection, built with no model, rebuilt on a timer in orion-durable-runs next to
the Temporal Self driver. Every line cites a row.

```text
SelfNowV1
  as_of, inputs_as_of{per source}
  organs[]:     name, status ∈ live | idle | off | gone | unknown, evidence_ref, last_seen_at
  body:         latest self_knowledge_items run → counts by category + diff vs previous run
  body_changes: merges since T-7d (labelled "merged", never "live", unless a deploy row exists)
  self_concept: latest N self_concept_history rows (layer3_reflect, curiosity_self_inquiry)
  dispositions: current personality_disposition rows
  lived_recent: Temporal Self closed days in the last 7 days (arc counts and top subjects)
```

Rules:

- **`unknown` is a real answer.** If no evidence exists, the status is `unknown`. Code never
  guesses `live`.
- **The organ list comes from code, not from a hand-written list.** Start from services in the
  latest inventory run. Status comes from heartbeat or recent-write evidence. Where no evidence
  source exists, the status stays `unknown`, and that gap is itself output worth reading.

### Foundation F2 and F3: Temporal Self past one day, and body changes as arcs

- **F2:** fold closed days into weeks and months with code only: arc counts by kind, subjects
  that returned across days, and threads that stayed open. No model and no prose.
- **F3:** a `body_change` source. A merged PR becomes a born-closed arc at `merged_at`. A deploy
  marks the arc deployed only where a deploy record exists. If none exists, the arc stays merged.

These two are not needed for the first episodes (the tools below give then-vs-now directly). They
make the timeline continuous later.

### Tools: Orion's dig kit

All tools are read-only. They are new tools on `orion-introspect`, except where an existing MCP
tool is enough. A short skill brief (like `orion/introspect/brief.py`) tells Orion how to dig.

| Tool | What it does | Built on |
|---|---|---|
| `past_checkout(date)` | Makes a worktree of the sandbox clone at the last `main` commit on or before `date`, under `/mnt/orion-fcc/past/<date>`. Returns the path and the commit. One at a time. Never touches `/mnt/orion-fcc/repo`'s own checkout (session sync hard-resets that one). | `git worktree add --detach` |
| `inventory(path)` | The self-study scanners run on any checkout: services, modules, channels, verbs, schemas, env keys. | `self_study.py` scanners, with `REPO_ROOT` made a parameter |
| `compare_inventory(then, now)` | Added, removed, and renamed (by heuristic, labelled as such) items between the two inventories. Code computes the diff. | pure diff |
| `history(start, end)` | Merged PRs (title, body, number) and, before Dec 2025, commits, for the window. | GitHub search by merge time; `git log` in the sandbox |
| `self_now()` | F1 projection. | F1 |
| `my_days(start, end)` | Lived records in the window: journals (from 07-23), Temporal Self days (from 10-09). Returns `none` with the reason when the window has no records. | existing tables |
| `episode_notes(read / append)` | The episode's own shared notes, so the hops can read each other. | reminiscence store |

Orion can also read files in the past checkout with the harness's normal file tools. The worktree
is read-only by convention, and the cleanup node deletes it.

**Source labels (from the skill brief, checked by code where possible).** Every claim carries one
label:

- `code_then`: seen in the past checkout. A primary source.
- `pr_text`: a builder's words. A secondary source.
- `lived_record`: Orion's own record from that time.
- `now`: must cite `self_now()`.
- `reflection`: Orion's own interpretation. It may wonder and wax philosophical. It is never
  stored as a fact.

### The episode: durable run `reminiscence.episode`

The episode is an admitted durable workflow in orion-durable-runs, shaped like `orion_day.letter`.

```text
resource_request -> resource_wait -> pick_period -> open_past (past_checkout)
  -> hop:what_it_was      (code_then, pr_text: inventory + compare + history)
  -> hop:what_it_was_like (lived_record, or an honest "I have no lived record of this" + pr_text)
  -> hop:where_i_am_now   (self_now: what of then survives, changed, or is gone)
  -> hop:where_im_going   (open threads now, set against what then was reaching for)
  -> hop:otherwise        (what I would do differently, labelled reflection)
  -> cross_read           (each hop gets the other four notes; ≤2 rounds; may revise its own)
  -> persist -> cleanup (always runs, also after failure: removes the past checkout) -> finish
```

- **Each hop is one harness agent-lane turn with the dig kit.** The node submits a
  `HarnessRunRequestV1` on the agent lane, waits for the reply, and checkpoints the hop's note.
  A restart resumes at the first hop without a note. This durable-run-to-harness dispatch is new,
  so it is built and tested on its own first.
- **One per day:** `reminiscence_episode` has a unique `episode_date`, and the persist step uses
  `ON CONFLICT DO NOTHING`. The scheduler fires one episode overnight. A manual run on the same day
  finds the existing one.
- **pick_period (default):** the oldest week not yet visited, in archaeology order from
  2025-06-24. Orion may choose to revisit a week instead; revisiting is allowed (decision 7). The
  pick and the reason are recorded.
- **Failure is honest.** An empty or ref-less hop, or a hop that claims `now` without a
  `self_now` citation, is a failed attempt. Past the retry budget, the episode fails. It does not
  turn into a fallback text.

### Partition and gate

- **Store:** table `reminiscence_episode` (episode_date, period, hops jsonb, cross_read jsonb,
  sources, status). The run writes nothing else: no journal, no memory card, no curiosity
  queue, no `self_concept_history`, no personality evidence.
- **Deterministic gate:** a CI test fails if any module outside an allowlist reads
  `reminiscence_episode` (the writer, the Hub review page, and the eval). Opening the gate means
  changing that allowlist in a reviewed PR. That is Juniper's call (decision 3).
- **Review surface:** a read-only Hub page listing episodes, with each hop's text, its source
  labels, and links to the cited PRs and commits.

## Missing questions (defaults chosen; Juniper can override)

- **Order:** oldest-first by default, with Orion allowed to revisit. Alternative: Orion picks
  freely from the start.
- **Episode size:** one week per episode. That is about 70 weeks of history, so a first full
  pass takes about 70 days.
- **Budget class:** an admitted agent-lane run, overnight. Pricing it in the body economy
  (PR #2855) comes later, as decision 7 suggests.

## Proposed schema / API changes

- `orion/schemas/self_now.py`: `SelfNowV1`, registered in `orion/schemas/registry.py`.
- `orion/schemas/reminiscence.py`:
  - `ReminiscenceBriefV1` (period, commit, run ids);
  - `ReminiscenceHopV1` (hop, note, claims[{text, label, refs}]);
  - `ReminiscenceEpisodeV1`;
  - workflow name `reminiscence.episode` in `orion/schemas/durable_run.py`.
- Migration: `self_now_projection` (or reuse `temporal_self_projection` if its row shape fits:
  check first) and `reminiscence_episode`. Additive DDL only, with header
  `-- ORION-MIGRATION-REQUIRED-BY: orion-durable-runs, orion-hub`.
- New `orion-introspect` tools: listed above.
- `self_study.py` scanners: add a `root` parameter, defaulting to `REPO_ROOT` (no behavior
  change).

## Files likely to touch

- `services/orion-cortex-exec/app/self_study.py`, or better, move the scanners to a shared
  `orion/self_inventory.py` so that orion-hub (the introspect host) can call them without
  reaching into cortex-exec.
- `orion/self_now/` (projection, pure), `services/orion-durable-runs/app/self_now_driver.py`.
- `services/orion-durable-runs/app/reminiscence_graph.py`, `admission_runtime.py`, `settings.py`,
  `.env_example`.
- The introspect tool host (orion-hub `scripts/`) plus `orion/fcc/mcp_names.py` and the brief.
- `orion/temporal_self/` (F2 and F3, later patches).
- Hub review page, the CI gate test, and `orion/reminiscence/evals/`.

## Non-goals

- No recall, memory card, journal, curiosity, or self-concept write from this work (decision 2).
- No model-written "who I am today". F1 is code only.
- No deploy claims without a deploy record.
- No write access for Orion in the past checkout. The dig is read-only.
- No chat-history archaeology yet.

## Proposal-mode checklist

- **Capability change:** Orion gets a cited, code-built picture of what they are today. Orion gets
  tools to visit the repo as it was on any day since 2025-06-24 and compare it to today. Once a
  day, Orion reminisces about one past week across five hops that read each other.
- **Data touched:** reads the git history, PR text, journals, Temporal Self, self_knowledge_items,
  self_concept_history, and personality dispositions. Writes only `self_now_projection` and
  `reminiscence_episode`, and creates and removes `/mnt/orion-fcc/past/*`.
- **Privacy boundary:** none new. This is repo and self data Orion already reads.
- **Proof it worked:**
  - a `self_now_projection` row whose organ statuses can be checked by hand against `docker ps`
    and the latest writes;
  - an episode row with five non-empty hops, every claim labelled and every ref resolvable;
  - the past checkout gone after the run;
  - the CI gate test green, with no reader outside the allowlist.
- **Dangerous failure:** Orion absorbs a false past or a false present into their self-model. The
  partition blocks absorption until Juniper opens the gate. The source labels and the `now`
  citation rule block the false present inside an episode. Another risk is a GPU-heavy hop chain
  crowding the agent lane; it runs overnight, once a day, under an admitted hold.
- **Disable / roll back:** set `REMINISCENCE_ENABLED=false` and `SELF_NOW_ENABLED=false`. Then
  drop the two tables. Nothing else depends on them while the gate is closed.

## Acceptance checks

1. `pytest` on the new projection, graph, tool, and gate tests passes. The tests include:
   - an organ with no evidence reads `unknown`;
   - a hop that claims `now` without a `self_now` citation is rejected;
   - cleanup runs after a failed hop;
   - a second episode on the same date is a no-op;
   - a reader outside the allowlist fails CI.
2. `inventory()` on the 2025-06-24 checkout and on today's main returns real, non-empty, different
   inventories. Check by hand that a service known to be added later is missing from the old one.
3. Live: one `self_now_projection` row. Juniper spot-checks five organ statuses.
4. Live: one full episode for 2025-06-24..06-30. Juniper reads it on the Hub page. Every
   `code_then` claim points at a file in that commit.
5. Eval `run_reminiscence_source_truth_eval.py`: sample claims per label. A `code_then` claim must
   resolve to a path in the cited commit, a `pr_text` claim to the cited PR, and a `now` claim to
   the projection row. Report the match rate per label.

## Recommended next patch

**Patch 1: F1, the self-now projection.** It is code only, cheap, and useful on its own. It also
shows how much of Orion is `unknown` today, and that number decides how honest the "now" hop can
be.

Then:

- **Patch 2:** shared self-inventory scanners with a `root` parameter, plus `past_checkout`,
  `inventory`, and `compare_inventory`. Curiosity can use them too.
- **Patch 3:** the durable-run-to-harness dispatch seam, on its own, with a smoke test.
- **Patch 4:** `reminiscence.episode`, the partition gate, and the Hub review page. Then the first
  live episode.
- **Patch 5:** Temporal Self F2 and F3.
