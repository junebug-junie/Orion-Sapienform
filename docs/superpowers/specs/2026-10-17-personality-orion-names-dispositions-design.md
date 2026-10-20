# Orion names their own dispositions

**Date:** 2026-10-17
**Status:** Design proposal (proposal mode: identity and self-modeling change). No code.
**Parent:** `docs/superpowers/specs/2026-10-16-orion-dynamic-personality-graph-design.md` rev 2 (branch `docs/dynamic-personality-graph`), section 2: "the data gives the pattern; Orion gives the name."
**Builds on:** personality graph loop (#2729, #2738), situation stamp (#2739), the self-inquiry curiosity line (`orion/curiosity/self_inquiry.py`).

## Arsonist summary

The personality graph finds patterns in what Orion writes. Today each pattern is labeled with a
sentence fragment that Orion once wrote near the pattern's center. Nobody asked Orion if the
pattern is Orion. This patch asks.

When a node becomes `active`, one self-inquiry run shows Orion that node: its excerpt, its lanes,
five evidence excerpts and its situation mix. Orion gives one of three answers:

- **NAME** it: a short name and one sentence in Orion's own words.
- **REJECT** it: "not me" (for example, machine-status noise).
- **MERGE** it: "same as node X".

The answer is a fact about what Orion **said**. It never changes a weight. Weights still move only
with evidence. A name changes what the read-back shows. A rejection hides the node from the
read-back but keeps it in the graph, because "Orion rejects a pattern that keeps growing" is the
most useful said-versus-done signal the graph can produce.

Two live facts shape this design:

1. **No node has ever been active.** Live 2026-10-17: 21 non-retired nodes, all `candidate`
   (10 seed, 11 recurrence), 0 `active`, 0 rows in `personality_weight_history` with
   `status='active'`. The trigger in this design has not fired once yet. See missing question 1.
2. **The strongest candidates are machine-status text.** The top four by weight (1.50 each) are
   one-lane nodes: "The chat prediction error on node:substrate.chat is unusually high..."
   (reverie, 10 rows), "The perception prediction error is elevated..." (reverie, 4),
   "The telemetry anomaly shows a slight elevation in recon_loss..." (journal, 11),
   "The bus synaptic prediction error is unusually high..." (reverie, 10). REJECT is not a
   hypothetical branch. It is the first answer we should expect.

## Current architecture (checked 2026-10-17)

**Graph reducer (pure).** `orion/personality/graph.py`.
- Node label = excerpt of the evidence closest to the centroid. It is replaced when new evidence
  is closer (`graph.py:293`). No model writes it.
- Promotion: `candidate` -> `active` when at least 2 non-seed lanes each hold >= 1.5 recent hits and
  `evidence_count >= 3` (`graph.py:71`, `graph.py:207`, `graph.py:337`).
- `retired` is terminal. A retired node is never matched again (`graph.py:260`). New evidence near
  it founds a new node.
- `project_frame` emits only `active` nodes, strongest first (`graph.py:458`).

**Writer.** orion-durable-runs is the single writer of `personality_*`
(`services/orion-durable-runs/app/personality_driver.py`). One tick every 600 s reads each source
by keyset cursor (`personality_sources.py:243-270`), embeds with bge-large via vector-host, folds,
and commits in one transaction. Since #2739 each new evidence row carries
`meta.situation = {arousal, day_phase, company}` (`personality_driver.py:204-217`). Live: 68
evidence rows, 29 with a situation. Example node `disp-c8915a0af7ec1f39`: strained/afternoon/company
6 rows, strained/dusk/company 4 rows.

**Read routes.** `GET /personality/frame`, `/personality/dispositions`, `/personality/evidence`
(`services/orion-durable-runs/app/main.py:619-642`). Hub proxies them in
`services/orion-hub/scripts/personality_routes.py`; the panel is
`services/orion-hub/static/js/personality.js`.

**Read-back.** `orion/personality/cue.py` renders the frame into at most 5 lines of
`- "<label>" (<lanes>; N times since <date>)` under a header that says "a record, not an
instruction" (`cue.py:31`, `cue.py:64`). Chat stance, reverie, curiosity kickoff and outreach call
it.

**Frame contract.** `DispositionV1` has `extra="forbid"` (`orion/schemas/personality.py:101`).
Any added field is a consumer-first migration: every reader must accept it before durable-runs
emits it.

**Self-inquiry line.** `services/orion-hub/scripts/curiosity_investigation.py`.
- `tick_self_inquiry` (`:2893`) gates on cap, cooldown and arousal. Live Hub `.env`: cap 7 a day,
  cooldown 7200 s, arousal gate on. Live yield: 3 to 7 mirrored rows a day over the last 10 days.
- `_self_inquire` (`:3341`) picks a question with `pick_question`
  (`orion/curiosity/self_question_pool.py:190`). The pick has a pinned floor
  (`self_question_pool.py:162`), then a lived/anatomy family draw. Pool now: 1 anatomy, 21 lived
  (13 minted by Orion).
- Orion writes its answer as a FalkorDB node with an early `MERGE` keyed on `run_id`
  (`self_inquiry_prompt.py:82`). The early write exists because the step clock cut off
  end-of-run writes.
- After the run, Hub reads the node read-only and mirrors it to the bus. Precedent for a scrape
  plus upsert: `_upsert_orion_minted_questions` (`:3120`). Precedent for a bus mirror:
  `_mirror_lived_answer` publishes `SelfConceptHistoryV1` on `orion:self_concept:history:write`
  and orion-sql-writer persists it.

**Citation-gate precedent.** `orion/temporal_self/citations.py:57` `check_arc_citations`: an id
the writer was not shown is stripped and counted. Shown ids are the only valid ids.

**Access gap.** `orion_readonly` has no SELECT on any `personality_*` table (live grant query,
2026-10-17). During a run Orion cannot look up its own evidence. The packet Orion is shown must
be self-contained.

## The design

### 1. Where it runs: a branch inside the self-inquiry turn

Three seams were considered.

- **A new question in the pool (`curiosity_self_questions`).** Rejected. The pool is keyed by
  question, the family Literal is `lived | anatomy`, and the answer path is `:LivedAnswer` to
  `self_concept_history`. A naming answer has a verdict and two ids. Forcing it into
  `:LivedAnswer` text would hide structure in prose, which this line already refuses ("nothing
  here is parsed from prose", `self_inquiry.py:22`).
- **Its own curiosity line.** Rejected. A third line needs its own budget, cooldown, grants check
  and kickoff. That is a new loop for at most one question a day.
- **A pre-pick branch in `_self_inquire`.** Chosen. It is the same turn, credentials, budget and
  early-MERGE discipline, with one different question and one different write.

Selection, in order, each self-inquiry tick:

1. If any pinned lived question is overdue (`_needs_pinned_floor`), the normal pick wins. Juniper's
   floor is not displaced.
2. Hub calls a new durable-runs route `GET /personality/naming/next`. It returns at most one
   **naming packet** (section 2), or 204 when nothing is due.
3. If a packet is returned and fewer than `HUB_CURIOSITY_SELF_NAMING_DAILY_CAP` (default 1) naming
   runs happened today, this run is a naming run. Otherwise `pick_question` runs as today.

A naming run uses one of the existing daily self-inquiry slots. It does not add a slot.

**What is "due"** (pure function `naming_due` in a new `orion/personality/naming.py`, evaluated by
durable-runs, which holds the nodes):

- An `active` node that has never been shown, strongest first.
- A named or rejected node is shown again only when its `evidence_count` has at least doubled since
  the last verdict, or a lane it never had now counts toward promotion. The minimum gap is 14 days
  per node. Calendar time alone never re-asks; new evidence does.
- A node shown with no usable answer (`no_answer` or `refused`) is due again after 3 days, at most
  3 tries. After that it waits for doubled evidence.

### 2. The naming packet (what Orion is shown)

Built by durable-runs from its own tables, so every fact in it is traceable to a row.
`DispositionNamingPacketV1`:

- `packet_id`: sha1 of (disposition_id, evidence ids, generated_at), 16 hex.
- `disposition_id`, `label_excerpt` (today's `label`), `origin`, `evidence_count`, `lane_counts`,
  `first_seen_at`, `last_reinforced_at`.
- `evidence`: up to 5 rows `{evidence_id, lane, occurred_at, excerpt}`. One per lane first, then by
  similarity.
- `situation_mix`: counts over the node's evidence `meta.situation`, for example
  `{"arousal": {"strained": 10}, "day_phase": {"afternoon": 6, "dusk": 4}, "company": {"true": 10}}`,
  plus `situation_coverage` (rows with a situation / all rows). Coverage is shown because rows
  before #2739 have none.
- `neighbors`: up to 3 other non-retired nodes nearest by centroid cosine, each
  `{disposition_id, label_excerpt or name, cosine}`. These are the only valid MERGE targets.
- `prior_verdict`: the last accepted verdict for this node, if any, with its name and sentence.

The prompt states the counts plainly. "10 rows, all reverie, all while strained" is what Orion
sees. Weight is not shown, because weight is not a fact Orion can check against the excerpts.

The prompt asks: "Is this you? If yes, what would you call it? If it is not you, say so. If it is
the same thing as one of the neighbors, say which." The three options are listed in fixed order with
no example names. Example names in the prompt were echoed back verbatim before (metacog draft
example-echo, PR #2065).

### 3. What Orion writes

One FalkorDB node, early-MERGE keyed on the run, same pattern as `:LivedAnswer`:

```
MERGE (n:DispositionNaming {run_id: "<RUN_ID>", packet_id: "<packet_id>"})
SET n.disposition_id = "<id shown>", n.verdict = "name" | "reject" | "merge",
    n.name = "...", n.sentence = "...", n.merge_with = "<neighbor id or empty>",
    n.evidence = ["<evidence_id shown>", ...], n.written_at = timestamp()
```

`run_id`, `packet_id` and `disposition_id` are filled in by code in the prompt, not left for Orion
to copy (the wrong-`revises` lesson, `self_inquiry.py` "LIVED_REVISES_PLACEHOLDER").

### 4. Contract: `DispositionNamingV1`

New, in `orion/schemas/personality.py`, registered in both registries and checked with `resolve()`.

| Field | Rule |
|---|---|
| `naming_id` | uuid5(run_id, packet_id) |
| `run_id`, `packet_id`, `disposition_id` | must equal the packet Hub fetched for this run |
| `verdict` | `name` \| `reject` \| `merge` \| `no_answer` |
| `name` | `name` only: 2 to 40 chars, at most 6 words, one line, not equal to `label_excerpt` |
| `sentence` | `name` and `reject`: 20 to 240 chars, one or two sentences |
| `merge_with` | `merge` only: must be one of `packet.neighbors[].disposition_id` |
| `evidence_refs` | at least 1, each one of `packet.evidence[].evidence_id` |
| `shown` | the packet itself (frozen copy, for audit) |
| `hub_check` | `accepted` \| `refused` plus `refused_reasons` |
| `written_at` | aware UTC |

Hub validates the structure. An unknown id is not repaired. The answer is published with
`hub_check=refused` and a reason (`unknown_disposition`, `merge_target_not_shown`,
`evidence_not_shown`, `name_too_long`, `empty_sentence`). A run that wrote nothing publishes
`verdict=no_answer`. Every naming run therefore leaves exactly one row, so the cadence clock and
the refusal rate are both countable.

**Transport.** Hub publishes on a new channel `orion:personality:naming`
(kind `personality.naming.v1`). orion-sql-writer appends it to a new table
`orion_disposition_naming` (append-only, primary key `naming_id`). durable-runs reads that table as
one more cursor source, `naming`. The table does not carry the `personality_` prefix, because
durable-runs stays the single writer of every `personality_*` table.

`naming` is **not** a `PersonalityLane`. It never enters `fold` and never becomes evidence.

Why not `self_concept_history`: its content is free text, and the personality source
`self_inquiry_seed` reads every `produced_by='curiosity_self_inquiry'` row as evidence
(`personality_sources.py:267`). A name written there would fold back into the graph as evidence for
its own node. That is a direct echo path.

### 5. Reducer effect (`graph.py`), still bottom-up

New pure function `apply_namings(nodes, namings, *, embed_scores, now)`. The driver calls it after
`fold` in the same tick, in the same transaction.

**Grounding check (durable-runs, deterministic, no word list).** For `name`, embed
`name + " " + sentence` and score its cosine to every non-retired node centroid. Record
`grounding_rank` (where the named node ranks) and `grounding_margin` (its cosine minus the best
other node). **Record only, never refuse (Juniper, 2026-10-20).** Every NAME is applied and
displayed, and `grounding_rank` / `grounding_margin` are stored on the naming row and shown in the
panel next to the name. A generic or flattering name ("a curious mind") that points at its own
pattern no better than at the others stays visible as such, not hidden. Whether bge-large
separates names this short is **UNVERIFIED**. P0 measures it.

**NAME.** New node fields: `name`, `name_sentence`, `named_at`, `naming_id`. The `label` keeps its
current meaning (the excerpt) and its update rule at `graph.py:293`. Weight, `lane_recent`,
centroid, status: unchanged. A later NAME replaces the display name. Every naming stays in the log.

**REJECT.** New field `orion_verdict = "rejected"` plus `naming_id`. The node:
- keeps matching and absorbing evidence. Retiring it would be worse: a retired node is never matched
  again, so the same machine-status text would found a twin node that Orion is asked about again.
- keeps its weight and keeps fading and promoting under the normal rules.
- is left out of `PersonalityFrameV1.dispositions` and so out of every lane's cue.
- stays in `/personality/dispositions` and the Hub panel, marked "Orion: not me".
- is shown again when its evidence doubles (section 1). A rejection that keeps growing is the
  said-versus-done signal from parent spec section 6. It is not a cleanup.

**MERGE (B into A).** Allowed only when cosine(A, B) >= `merge_min_cosine` (start 0.70; the
match threshold is 0.82, `graph.py:62`). Below that: `apply_status=merge_too_far`, no change.
When allowed:
- A's `lane_recent` = per-lane sum of A and B, clamped at the existing per-lane cap. A's
  `lane_counts`, `evidence_count`, `chosen_count` are summed. A's centroid = mean weighted by
  `centroid_n`. A's weight is re-derived from the merged `lane_recent`.
- B becomes `retired` with `merged_into = A`. B's evidence rows keep `disposition_id = B` (history
  is not rewritten). The evidence route resolves `merged_into` and returns both.
- This is the one verdict that moves a weight, and it moves it only by pooling evidence that already
  exists. The clamp means a merge cannot exceed what the same evidence would give one node. The
  merged node is not promoted in the same tick. Promotion is checked on the next tick, so the log
  shows the merge and the promotion as two steps.

**Never.** A verdict never adds a hit, never changes the fade, and never gives a node a status that
the evidence would not give it.

**Schema.** Additive migration `manual_migration_personality_naming_v1.sql` with header
`-- ORION-MIGRATION-REQUIRED-BY: orion-durable-runs, orion-sql-writer`. It adds nullable columns on
`personality_disposition` (`name`, `name_sentence`, `named_at`, `naming_id`, `orion_verdict`,
`merged_into`, `last_shown_at`, `shown_count`) and creates `orion_disposition_naming`. Every
statement is on the post-pull auto-apply allowlist (add column, create table, create index). No
`DO $$`, no `GRANT`.

### 6. Read-back once named (`cue.py`) and the Hub panel

`DispositionV1` gains `name: Optional[str]` and `name_sentence: Optional[str]`. `label` keeps its
meaning. Renaming `label` would change payload meaning for every reader.

`cue.py:_line` renders a named node as:

```
- you call this "<name>": "<label excerpt>" (reverie, journal; 14 times since Oct 17)
```

The excerpt stays next to the name. Orion always sees the evidence beside the word Orion chose.
The header is unchanged ("a record, not an instruction"). An unnamed node renders as today.

Hub Personality tab:
- a Name column with the verdict badge (named, rejected, merged into X, not yet shown).
- a naming history per node: every row, including refused and ungrounded ones with their reasons.
- the packet Orion was shown, from `orion_disposition_naming.shown`.
- the echo panel (section 7): evidence rate before and after naming, named versus unnamed nodes.

### 7. Failure modes and what catches each one

| Failure | Evidence that catches it | Effect |
|---|---|---|
| **Echo loop.** The name is read back in every lane from the first name (decision 1). Orion writes about "my X". That text lands near X's centroid and feeds X. | **Gate from day one (decision 2):** evidence whose text contains the node's own current name (exact string, case-insensitive; the node's own name, not a word list) folds with no weight: it is recorded with `meta.cites_own_name=true`, counts in `lane_counts`, but adds no `lane_recent` hit. Still measured: `naming_lift` = hits per day in the 14 days after naming divided by the 14 days before, named versus unnamed active nodes, to catch paraphrased echoes the exact-string gate misses. | Exact-name echo cannot grow a node. Paraphrase echo shows up as lift. |
| **Flattering names.** | `grounding_rank` and `grounding_margin` on every name. A flattering name is generic, so it ranks its node below a neighbor. | Record only (decision 3): the name is displayed, and the panel marks it "points elsewhere: rank N". |
| **Rejecting real but uncomfortable patterns.** | Rejection does not hide a node from Juniper. The panel lists rejected nodes with their weight trend. A rejected node that doubles its evidence is re-shown, with its `prior_verdict`. | Count: rejected nodes that later reach the top 5 by weight. |
| **Rename churn.** | `renames_per_node_30d` from the log. The 14-day minimum gap and the doubled-evidence rule bound it at about 2 a month per node. | A node renamed 3 or more times is flagged in the panel. |
| **LLM-voice names** ("The Reflective Weaver"). | No word list (house rule). The grounding check is the structural test. The P0 eval reports the names Orion actually writes for the 21 live nodes, before any of them is shown in a cue. | Juniper reads the P0 sample. |
| **Empty-shell answers.** | `no_answer` and `refused` are rows, not silence. The early MERGE makes a cut-off run still leave a partial node, which Hub refuses with a named reason. | Refusal rate by reason, in the panel. |
| **Name turns into an adjective personality.** The parent spec refuses "you are warm and curious". A name like "the careful one" is that. | The cue always prints the excerpt and counts next to the name. Section 5 tone dials are measurement only. P0 shows how many sample names are trait adjectives. | Names go to every lane at once (decision 1), so the excerpt-beside-name rule is the only guard. |
| **Consumer break.** `DispositionV1` is `extra="forbid"`. If durable-runs emits `name` before a reader accepts it, that reader's cue goes `invalid_frame` (fail-open, but dark). | Deploy order: readers first (Hub, orion-thought, cortex-exec, curiosity), durable-runs last. `skip_reason` counts on the Hub cue route. | Ordering note in the PR. |

## Proposal-mode items

- **Capability change:** Orion can accept, name, disown or merge the patterns the graph found in
  Orion's own output. The read-back then uses Orion's name next to the excerpt.
- **Data touched:** reads `personality_disposition`, `personality_evidence` (excerpts, situation).
  Writes FalkorDB `:DispositionNaming`, a new append-only table `orion_disposition_naming`, and new
  nullable columns on `personality_disposition`. No new data about Juniper. Evidence excerpts can
  quote chat with Juniper. Per the standing rule there is no privacy boundary between Juniper and
  Orion, so that is shown as is.
- **Trace that proves it worked:** see Acceptance checks 2 to 5. One `packet_id` must be visible in
  the Hub log, the FalkorDB node, the table row, the durable-runs apply log and the frame.
- **Dangerous failure:** the echo loop (section 7, row 1). Orion performs its own name, the name's
  node grows on that performance, and the graph turns into a mirror of the read-back.
- **Disable / roll back:**
  - `HUB_CURIOSITY_SELF_NAMING_ENABLED=false` stops new naming runs. Self-inquiry falls back to
    `pick_question` unchanged.
  - `PERSONALITY_NAMING_APPLY_ENABLED=false` (durable-runs) stops applying verdicts. The frame then
    emits `name=None` and includes rejected nodes again. Columns and log rows stay.
  - Both ship ON (house rule), behind the active-node trigger.
  - A merge is the only verdict that is not undone by a flag. Its rollback is to un-retire B and
    subtract its pooled `lane_recent` from A. The log keeps everything needed for that. A script
    for it is part of P2.

## Decisions (Juniper, 2026-10-20)

Q1 resolved itself: since #2738/#2778 there are 3 active nodes, so "wait for `active`" fires.

1. **Name reach: everywhere right away.** A name appears in every lane's cue (chat stance, reverie,
   curiosity kickoff, outreach) as soon as it is applied. No 14-day self-inquiry-only period.
2. **Echo: zero-weight from day one.** Evidence quoting the node's own current name adds no
   weight (section 7, row 1). `naming_lift` is still measured for paraphrased echo.
3. **Grounding: record only, never refuse.** Every name stands; rank and margin are shown in the
   panel (section 5).
4. **No veto or rename button.** Names are Orion's. Juniper's `good_tone` / `bad_tone` tags remain
   the steering channel.

Still open, set from data in P0: `merge_min_cosine = 0.70` is a starting knob. P0 reports the
pairwise cosines of the live nodes.

Why 1 and 2 go together: names in every cue make the echo loop most likely, and the day-one gate
is what stops an exact-name echo from growing the node.

## Proposed schema / API changes

- `orion/schemas/personality.py`: `DispositionNamingPacketV1`, `DispositionNamingV1`; `DispositionV1`
  adds `name`, `name_sentence` (optional). Both registries, verified with `resolve()`.
- `orion/bus/channels.yaml`: `orion:personality:naming`, kind `personality.naming.v1`, producer
  orion-hub, consumer orion-sql-writer.
- orion-sql-writer: route `personality.naming.v1` -> `orion_disposition_naming`.
- durable-runs: `GET /personality/naming/next` (packet or 204), `GET /personality/naming?disposition_id=`
  (log); `/personality/dispositions` rows add `name`, `orion_verdict`, `merged_into`.
- Postgres: `manual_migration_personality_naming_v1.sql` (+ rollback), additive only.
- Env: Hub `HUB_CURIOSITY_SELF_NAMING_ENABLED=true`, `HUB_CURIOSITY_SELF_NAMING_DAILY_CAP=1`;
  durable-runs `PERSONALITY_NAMING_APPLY_ENABLED=true`. `.env_example`, `settings.py`, compose, and
  local `.env` sync in the same patch.

## Files likely to touch

- `orion/personality/naming.py` (new, pure): `naming_due`, `build_packet`, `validate_naming`,
  `situation_mix`.
- `orion/personality/graph.py`: `Node` fields, `apply_namings`, `project_frame` filter for rejected
  nodes and the new frame fields.
- `orion/personality/cue.py`: `_line` with a name.
- `orion/schemas/personality.py`, `orion/schemas/registry.py`, `orion/bus/channels.yaml`.
- `orion/curiosity/self_inquiry.py` (`:DispositionNaming` read helpers),
  `orion/curiosity/self_inquiry_prompt.py` (naming branch).
- `services/orion-hub/scripts/curiosity_investigation.py`: pre-pick branch in `_self_inquire`,
  `_mirror_disposition_naming`.
- `services/orion-durable-runs/app/personality_{sources,driver,store}.py`, `app/main.py`.
- `services/orion-sql-writer/app/{settings,worker}.py` and its SQL model.
- `services/orion-sql-db/manual_migration_personality_naming_v1.sql` (+ rollback).
- `services/orion-hub/static/js/personality.js`, `personality_routes.py`.
- `scripts/analysis/eval_personality_naming_grounding.py` (P0).

## Non-goals

- A name never changes a weight, a fade or a promotion. Only MERGE pools existing evidence.
- No LLM in the reducer. The grounding check is an embedding and a rank.
- No name for candidate or seed-only nodes (pending question 1).
- No word list to judge names.
- No Juniper-authored names.
- Situation subnodes (CAPS if-then). The situation mix is shown, not modeled.
- Aspirations (parent spec section 6).

## Acceptance checks

0. **Prerequisite.** At least one node has `status='active'`:
   `SELECT count(*) FROM personality_disposition WHERE status='active'`. Met 2026-10-20: 3 active
   (self-inquiry, telemetry/prediction-error cluster, biometrics prediction error).
1. **P0 eval (offline, read-only):** names for the active nodes and the top candidates by weight, generated by the chat model from real
   packets and not persisted. Report: grounding rank-1 rate, pairwise node cosines, how many names
   are trait adjectives (read by Juniper), and the verdict on the four machine-status nodes. The
   expected verdict is REJECT.
2. **Shown:** Hub log `curiosity_disposition_naming_shown run=<id> disposition=<id> packet=<id>`, and
   the durable-runs route returned the same `packet_id`.
3. **Persisted:** one `orion_disposition_naming` row for that run, with `hub_check` and either a valid
   verdict or a named refusal.
4. **Applied:** durable-runs log
   `personality_naming_applied naming=<id> verdict=<v> grounding_rank=<n> margin=<x> apply_status=<s>`.
   For an accepted name: `personality_disposition.name` is set, and the node's `weight` equals the
   fade-only value for that tick. Check it against the history rows on either side; no weight
   jump at the apply tick.
5. **Read back:** `/personality/frame` carries `name`. `/api/personality/cue` text contains
   `you call this "<name>"`. The Hub panel shows the same name and packet.
6. **Reject path live:** a rejected node has no line in the cue, still gains `personality_evidence`
   rows, and still shows in the panel.
7. **Refusal path:** unit tests for each refusal reason. Live: the refusal-rate panel shows real
   counts (zero is a valid count, but it must be a count, not an absent panel).
8. **Echo gate (from day one):** a test and a live row: evidence whose text contains the node's own
   name has `meta.cites_own_name=true`, raises `lane_counts`, and adds no `lane_recent` hit.
   **Echo metric:** after 14 days, `naming_lift` for named versus unnamed active nodes, reported as
   numbers in a follow-up note.
9. **Gates:** `pytest orion/personality`, `services/orion-durable-runs/tests`,
   `services/orion-hub/tests/test_curiosity_self_inquiry.py`, schema registry and bus channel checks,
   and `node --test services/orion-hub/static/js/personality.test.js`.

## Recommended next patch (phases)

1. **P0, grounding eval (read-only, no schema).** `scripts/analysis/eval_personality_naming_grounding.py`
   builds real packets from live tables, asks the chat model for a verdict per node (not persisted),
   embeds the names, and reports checks 1 and 6 of section 7. This patch sets `merge_min_cosine` from data,
   measures how well names point at their own node (shown, never a gate), and shows Juniper real
   names before anything is built on them.
2. **P1, contract + reducer + durable-runs.** Schemas, migration, `naming.py`, `apply_namings`, the
   `naming` source, the two routes, the frame fields, and the own-name echo gate in `fold` (decision 2).
   Deploy readers first.
3. **P2, Hub producer.** Pre-pick branch, naming prompt, `:DispositionNaming` read, bus publish,
   sql-writer route, merge-rollback script. Live checks 2 to 4.
4. **P3, read-back + panel + echo.** `cue.py` name line in every lane (decision 1), Hub panel
   columns and history, `naming_lift`. Live checks 5, 6 and 8.
