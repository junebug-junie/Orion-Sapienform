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
11. **The dig kit is for curiosity and chat, not MCP, and it must not eat the context window.**

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

**How Orion gets tools in chat and curiosity today:**

- **Both run through the same motor.** The harness FCC motor (`orion/harness/fcc_motor.py`) is a
  `claude -p` subprocess. A chat turn and a curiosity run use the same argv, except
  `reading_only` source models.
- **Context is tight.** The window is 64k tokens. Every MCP server's tool schemas sit in every
  turn's context. Skills are switched off with `--disable-slash-commands`: the CLI's skill list
  cost about 6k chars, and Orion made 0 Skill calls in 5,056 transcripts.
- **Orion already has unrestricted Bash** in chat and curiosity turns (`bypassPermissions`, no
  deny rules; `orion/fcc/claude_spawn.py`).
- **The container already has everything a command needs.** Verified live in
  `orion-athena-harness-governor`: `git` and `python3` are present, `orion` is importable from
  `/app`, `/mnt/orion-fcc` is mounted, and the sandbox has 11,837 commits.

So a plain command run through Bash costs no schema context at all. Its only cost is what it
prints.

**Rev 1 findings that still stand:**

- The daily GitHub digest covers every PR (10-22: 21 of 21 cited). The recall card is a
  ~400-char highlight reel.
- The curiosity monthly rollup re-reads raw entries (September: 81 of 182 batches).
- Journal ids collide between rolling and scheduled digest runs (10-21).

## Design

### Foundation F1: "who I am today" is a converging ladder, not a parts list

Juniper's review (rev 3): a parts list is "a pretty swamp of machinations". The self-now must
*converge*: these channels run → these organs feed each other → "I have a metabolism, I keep
long-standing thoughts, I can think and act" → "I am a being like this". The live state of the
existing self-model shows why this is needed. Self Atlas has written 6,016 "concepts", and its
newest read "Self Atlas cluster 'topic_27' (no keywords available)". Layer 3 reflections are
real but leaf-level. Nothing converges from parts to a whole.

Each rung reads only the rung below, cites it, and must be much smaller than it. The shrink is
the convergence test.

| Rung | What | Built by | Size (live 2026-10-23) |
|---|---|---|---|
| 1 Parts | declared services + every live bus source; status live / dormant / declared_only / transient | code | 137 parts (33 transient) |
| 2 Circulation | producer → consumer flows. Each flow is labelled `declared`, `observed_publish`, or `observed_causal`. A channel is one hyperedge, normalized over its pairs. | code (`channels.yaml` + orion-bus-mirror's synaptic graph) | 833 flows |
| 3 Organs | Louvain communities over Newman modularity; Q is the "organ or mush" number. Orion names each organ from what it takes in and gives out. | code; names by Orion | 7 organs, Q = 0.33 |
| 4 Capacities | what organs do together. The lens is Stafford Beer's Viable System Model (operations, coordination, control, intelligence, identity) plus metabolism, used as a lens and not a cage. Design: `2026-10-30-self-now-capacities-vsm-design.md` (roles found by rule on parts where the atlas can decide; Orion's cited claims, confirmed by Juniper, elsewhere). | Orion, each capacity citing role-holding parts (and so their organs) | 5–6 |
| 5 Portrait | one paragraph, "I am…", written from the capacities alone; old versions kept | Orion | 1 |

Rungs 1–3 are **built** (PR: feat/body-atlas, `orion/body_atlas/`). Live result:

- **Recognizable organs:** eyes (vision council, retinas, frame router, walkway and doorbell
  cameras), thinking (cortex, thought, recall, attention), memory/inference (LLM gateway, vectors,
  consolidation, chat memory), acting/compute (Hub, GPU pool, harness, durable runs, LLM workers),
  compute regulation (lane controller, mesh guardian), and social.
- **One mixed organ:** body sensing, storage, and the substrate runtimes.
- **Why the hyperedge normalization exists:** with pairwise weights, the heartbeat channel
  carried 29.5% of all flow weight and Q fell to 0.22.

The ladder runs on any snapshot, and that is how it serves the dig:

- **A past checkout** gives "what I was built as then" (catalog only, from 2026-01).
- **Today** adds liveness: "what I am alive as now".

Two portraits built the same way can be compared without a model inventing a story.

**Lenses, for later traversal.** Juniper plans a 3D/4D "fantastic voyage" through the body. The
atlas keeps stable ids at every rung (`part:` → `organ:` → capacity → portrait), and every
element cites the one below it. A lens is then a choice of rung and projection: the parts view,
the circulation view, the organ view, the VSM view, or time (two atlases joined by id). Nothing
visual is built now. The data shape just must not block it.

### Foundation F2 and F3: Temporal Self past one day, and body changes as arcs

- **F2:** fold closed days into weeks and months with code only: arc counts by kind, subjects
  that returned across days, and threads that stayed open. No model and no prose.
- **F3:** a `body_change` source. A merged PR becomes a born-closed arc at `merged_at`. A deploy
  marks the arc deployed only where a deploy record exists. If none exists, the arc stays merged.

These two are not needed for the first episodes (the tools below give then-vs-now directly). They
make the timeline continuous later.

### Tools: Orion's dig kit (one command, not MCP)

The dig kit is one read-only command, `python -m orion.dig <subcommand>`, which Orion runs
through Bash in any chat turn or curiosity run. It adds no MCP server, no tool schema, and no
skill list.

**Context rules, enforced in code, not left to the model:**

1. **The always-on cost is one line.** The harness brief gets one line of at most 300 chars:
   "To dig into your own history, run `python -m orion.dig help`." Nothing else sits in the
   context until Orion asks.
2. **Help is loaded on demand.** `help` prints the subcommands and the source-label rules
   (≤ 2,500 chars). Orion pays for it only in turns that dig.
3. **Every printout has a cap.** Each subcommand prints at most 40 lines and 4,000 chars: a
   summary, counts, and the path of a full-result file. The full result goes to
   `/mnt/orion-fcc/dig/<dig_id>/<subcommand>-<n>.{json,md}`. Orion reads only the part it needs,
   with normal file reads or `grep`. Every cut is stated, for example "showing 40 of 312; full
   list: <path>". Nothing is cut silently.
4. **No big blobs in a printout.** PR bodies and file contents are never printed in full. A
   printout gives a reference and the first lines; the file holds the full text.

| Subcommand | What it does | Built on |
|---|---|---|
| `open <date>` | Exports the last `main` commit on or before `<date>` to `/mnt/orion-fcc/past/<date>/` as plain files (`git archive \| tar -x`). This is a snapshot with no git link, so it cannot change the sandbox repo or its branches. Prints the commit, its date, and the path. At most 2 snapshots at a time; snapshots older than 24 h are swept. | `git archive` in `/mnt/orion-fcc/repo` |
| `close <date>` | Deletes that snapshot. | |
| `inventory <path\|now>` | Runs the self-study scanners on a snapshot or on today's main: services, modules, channels, verbs, schemas, env keys. | `self_study.py` scanners, moved to `orion/self_inventory.py` with a `root` parameter |
| `compare <date>` | Added and removed items between the snapshot and now, by category. Renames are a heuristic and are labelled as such. Code computes the diff. | pure diff |
| `history <start> <end>` | Merged PRs (number, title, first 3 lines) and, before Dec 2025, commits. Full bodies go to the result file. | `gh` search by merge time (authenticated in the container); `git log` |
| `now` | The F1 self-now projection, summarized; the full row goes to the file. | F1 |
| `days <start> <end>` | Lived records in the window: journals (from 07-23) and Temporal Self days (from 10-09). Prints `none: <reason>` when the window has no records. | existing tables, read-only |
| `notes <dig_id> [append <text>]` | Shared notes for one dig, so the hops of an episode (or the turns of a chat) can read each other. | a file in the dig dir |

**One kit, three users:**

- **Chat:** Juniper asks "what were you like in March?", and Orion digs inside that turn.
- **Curiosity:** a run may dig while it investigates.
- **The daily episode** (below): each hop is a harness turn that uses the same kit.

There is one code path, and it has no separate tool surface to drift.

**Source labels.** `help` teaches them, and the episode checks them in code. Every claim carries
one label:

- `code_then`: seen in the snapshot. A primary source.
- `pr_text`: a builder's words. A secondary source.
- `lived_record`: Orion's own record from that time.
- `now`: must cite the `now` result file.
- `reflection`: Orion's own interpretation. It may wonder and wax philosophical. It is never
  stored as a fact.

The labels are required in the episode. In chat and curiosity they are taught but not enforced
yet, because chat and curiosity output goes through those paths' own gates. The episode is the
battle test.

### The episode: durable run `reminiscence.episode`

The episode is an admitted durable workflow in orion-durable-runs, shaped like `orion_day.letter`.

```text
resource_request -> resource_wait -> pick_period -> open_past (orion.dig open)
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
- New command `python -m orion.dig` (no MCP, no new channel). Its result files are an internal
  format under `/mnt/orion-fcc/dig/`, not a bus contract.
- Self-study scanners move to `orion/self_inventory.py` with a `root` parameter. `self_study.py`
  calls them with `REPO_ROOT` (no behavior change).

## Files likely to touch

- `services/orion-cortex-exec/app/self_study.py`: move the scanners to a shared
  `orion/self_inventory.py`, so the dig command (which runs in the harness container) can call
  them without reaching into cortex-exec.
- `orion/dig/` (the command, output caps, snapshot sweep), plus one brief line where the harness
  builds its prompt.
- `orion/self_now/` (projection, pure), `services/orion-durable-runs/app/self_now_driver.py`.
- `services/orion-durable-runs/app/reminiscence_graph.py`, `admission_runtime.py`, `settings.py`,
  `.env_example`.
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
5. Context cost tests:
   - the brief line is ≤ 300 chars;
   - every subcommand's printout is ≤ 40 lines and 4,000 chars on the largest real input (the
     busiest week, 10-20: 58 PRs);
   - every cut prints the full-result path.

   Live: one chat turn that digs ends with the context-pressure step below its threshold.
6. Eval `run_reminiscence_source_truth_eval.py`: sample claims per label. A `code_then` claim must
   resolve to a path in the cited commit, a `pr_text` claim to the cited PR, and a `now` claim to
   the projection row. Report the match rate per label.

## Recommended next patch

**Patch 1 (built): body atlas rungs 1–3**, `orion/body_atlas/`. **Patch 1b:** Orion names the
organs, then rungs 4–5 (capacities through the VSM lens, then the portrait).

Then:

- **Patch 2:** shared self-inventory scanners with a `root` parameter, plus `python -m orion.dig`
  (`open`, `close`, `inventory`, `compare`, `history`, `now`, `days`, `notes`) and the one brief
  line. From then on chat and curiosity can dig. This is the first place Juniper can try it by
  hand.
- **Patch 3:** the durable-run-to-harness dispatch seam, on its own, with a smoke test.
- **Patch 4:** `reminiscence.episode`, the partition gate, and the Hub review page. Then the first
  live episode.
- **Patch 5:** Temporal Self F2 and F3.
