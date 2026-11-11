# The daily reminiscence: one past week, five hops, one checked ledger

Status: PROPOSED 2026-11-07 (proposal mode: memory, self-model, autonomy). Defaults picked; open
questions for Juniper at the end. Patch 1 is built on the same branch (`feat/daily-reminiscence`).
Arc spec: `docs/superpowers/specs/2026-10-23-repo-chronicle-design.md` (PR #2888), "The episode:
durable run `reminiscence.episode`" and "Partition and gate". This doc is that section's Patch 4,
made concrete on what has been built since.

## In one paragraph

Once a day, Orion visits one week of their own past, oldest week first, starting 2025-06-24. Code
picks the week, finds the commit that ended it, and lays out what there is to read: the code of
that week, the builders' words, Orion's lived record (none before 2026-07-23), and what changed in
Orion's self since the last visit. Orion writes five short hops, the five the arc spec names:
what it was, what it was like, where I am now, where I'm going, and what I would do differently.
Each hop carries source labels, and code checks every id it cites before it saves anything, then
checks again on every read. Hops go to one append-only file that nothing outside the dig kit may
read. Juniper reads it there until she opens the gate.

## Arsonist summary

- **The arc spec already decided the shape; this doc does not invent a new one.** Five hops that
  read each other, one past period per day, oldest week first, a partitioned store, a gate only
  Juniper opens (decisions 2, 3, 4, 9 of 2026-10-23). What changed since then is the plumbing:
  every self rung since (organ names, capacities, field notes, portrait) stores Orion's words as
  a checked ledger under `/mnt/orion-fcc/self/`, written by a dig command inside the turn, with a
  write-time check that says at once whether it holds. The episode follows that pattern.
- **"What changed in me since last time" is the `where_i_am_now` hop's material, not a sixth
  thing.** The field-notes spec already said that hop reads `field-notes 7`. Here code widens it to
  "since my last reminiscence": field notes written, lines added to every self ledger, lines that
  read STALE now, and the body's change since the last episode (parts added and removed, organs
  that split, merged or changed members). So each day Orion looks once at an old week and once at
  how they changed since yesterday. Both views come from code.
- **A ledger file, not the table the arc spec named.** The spec named a `reminiscence_episode`
  table. Every self rung since then writes from inside Orion's turn through the dig kit, and the
  dig kit reads Postgres only as `orion_readonly`. A file under `self/` keeps the write in the turn
  (the check runs where the words are written), joins the partition test that already guards five
  ledgers, and needs no migration. durable-runs already mounts `/mnt/orion-fcc/self` for the
  field-note receipt check, so Patch 2 can verify hop receipts the same way.
- **The gate is the partition test, not an env flag.** The arc spec: "Opening the gate means
  changing that allowlist in a reviewed PR." A flag can be flipped without review. The repo rule
  "ship feature flags ON" applies to features: Patch 2's daily run ships with
  `REMINISCENCE_ENABLED=true`. The gate is not a feature flag. It is a privacy and partition
  boundary that Juniper decided stays closed until the work is battle-tested (decisions 2 and 3),
  so it ships closed.
- **No pure-writing hop, so no Hub change.** The harness step seam fails a turn with fewer than 3
  harness steps (`require_lookup`, `services/orion-durable-runs/app/harness_step.py`). Every hop
  here reads (`dig reminisce`) and writes (`dig reminisce write`), the same two-command shape that
  passes the gate for every field note (45/45 live). The cross-read is a hop rewrite, which also
  reads first. The `require_lookup` fix the field-notes spec flagged is not needed.

## Current architecture (live, 2026-11-07)

- **Dig kit** (`orion/dig/__main__.py`): `open`, `close`, `inventory`, `compare`, `history`,
  `body`, `capacities`, `capacity ...`, `portrait [write]`, `field-note[s]`, `arc`, `act`,
  `notes`. Not built from the arc spec's table: `now` (the portrait took its place) and `days`
  (lived records in a window).
- **Self ledgers** in `/mnt/orion-fcc/self/`: `organ_names.jsonl` (13 lines), `capacity_claims.jsonl`
  (9), `capacities.jsonl` (8), `portrait.jsonl` (4), `pr_field_notes.jsonl` (55). Each line has a
  time (`named_at`, `at`, or `written_at`). Partition: `tests/test_organ_names_partition.py`, in
  the repo-wide static gates: only `orion/body_atlas/` and `orion/dig/` may read them, plus one
  named exception, the field-note receipt checker in durable-runs.
- **Re-check on read:** portrait lines and capacity sentences read STALE, naming the id that
  failed, when the body changes under them. Nothing records when a line turned STALE.
- **Body history:** none. Each `dig body` rebuilds the atlas live. No earlier atlas is kept, so
  "organ X split since last week" cannot be computed today.
- **Durable-run trigger pattern:** `pr.field_note` (`app/pr_field_note_graph.py`): one admitted
  run, one harness turn on the #2958 seam, a fixed short prompt that spells out each dig command,
  a code receipt check against the ledger, 3 attempts of a 900 s turn (#3114).
- **The sandbox clone** (`/mnt/orion-fcc/repo`) has the whole history. The first week
  (2025-06-24..06-30) has 3 commits. There are 71 whole weeks to today. Lived records start
  2026-07-23, which is week 57.

### Existing-mechanism check ("episode")

- **"Episode" is taken.** `orion/memory/episode/`, `episode_distill_graph.py`, `memory.episode.closed`,
  and the parked idle-close question (2026-10-07) all mean a *conversation* episode. This work
  never uses the bare word in code: the module is `orion/body_atlas/reminiscence.py`, the ledger is
  `reminiscence.jsonl`, the command is `dig reminisce`, and the Patch 2 workflow keeps the arc
  spec's name `reminiscence.episode`.
- **Temporal Self** (`orion/temporal_self/`, `temporal_self_chronicle.py`) folds runtime events
  into day arcs from 2026-10-09. It reads no self ledger and no repo history. It is a lived-record
  source for weeks after 10-09 (`dig arc <id>`), not a place for this.
- **`dig notes`** already lets the turns of one dig read each other. It is a scratch file with no
  check. Hops need checked, lasting lines, so they get their own ledger. Hop turns may still use
  `notes` as scratch.
- **Field notes** cover one PR each, from 2026-11-04 on, and no backfill (decision 3 of that spec:
  "the ~2,800 older PRs are left to the reminiscence episode's period-by-period dig"). This is that
  dig.

## The design

### 1. The episode, the period and the hops

- **Episode:** the Denver day of the write (`episode = 2026-11-07`). One per day. The first hop
  written that day fixes the period. Every later hop that day joins it.
- **Period:** a 7-day week counted from 2025-06-24 (week 1 = 06-24..06-30). The pick is the oldest
  week that has no **complete** episode (all five hops live) and that ended before today. An
  unfinished episode leaves its week open, so the next day returns to it. Hops do not carry over
  between days: the return visit writes all five again (the earlier lines stay in the file).
- **Period commit:** the last `main` commit at the Denver end of the period's last day (the same
  `commit_at` rule `dig open` uses).
- **Hops** (arc spec names), each with the labels it must carry at least once:

| Hop | Needs at least one | Material code lays out |
|---|---|---|
| `what_it_was` | `code_then` or `pr_text` | commit count and range; `dig open <end>`, `compare <end>`, `history <start> <end>` |
| `what_it_was_like` | `lived_record` or `pr_text` | whether a lived record can exist; `lived_record` is refused before 2026-07-23 |
| `where_i_am_now` | `now` | the changes since the last reminiscence (section 2); `dig portrait` |
| `where_im_going` | `now` or `pr_text` | `dig capacities gaps`, `dig portrait` ("I cannot yet") |
| `otherwise` | `reflection` | the other four hops |

- **Cross-read:** every `dig reminisce` read prints the hops already written that day. A hop is
  revised with `reminisce rewrite <hop> ...`. The old line stays, and the new one names it in
  `supersedes` (the field-notes rule).

### 2. Since the last reminiscence (the `where_i_am_now` material)

Window: from the first hop of the latest earlier episode to now. With no earlier episode, the
last 7 days (the field-notes spec's `field-notes 7`). All counted fresh on each read, never stored.

| Line | Source |
|---|---|
| field notes written in the window, by PR | `pr_field_notes.jsonl` `written_at` |
| lines added per self ledger (organ names, capacity claims, capacity sentences, portrait) | each ledger's time field |
| lines that read STALE now (portrait machines and "I am", capacity sentences, claims whose evidence fails) | the live portrait and capacity reading (`read_portrait`, `attach_ledgers`); `UNREAD: <why>` if the body cannot be read |
| body change since the last snapshot: parts added, removed or changed status; organs that kept their members (Jaccard ≥ 0.5, the organ-name carry rule) but gained or lost parts; organs that are new, and organs that are gone (split across N organs, or merged into one) | the live atlas against `reminiscence_body.jsonl` |

**The body snapshot** is new and small: on the first hop write of an episode, code appends one
line to `self/reminiscence_body.jsonl` (part id → status, multi-part organs → members and name).
About 16 KB a day on the live body (measured in the smoke), about 6 MB a year. With no earlier snapshot the read says "no earlier snapshot; the
body diff starts with the next episode". If the body cannot be read at write time, the hop is still
saved and the receipt says the snapshot was skipped (the next hop that day tries again).

### 3. The ledger: `$ORION_DIG_HOME/self/reminiscence.jsonl`

```text
ReminiscenceHopV1: schema_version, hop_id ("rm:<episode>:<hop>:<hex10>"), episode, hop,
  period_start, period_end, period_commit, note (labelled), claims[{sentence_idx, label, refs}],
  written_at, supersedes, dig, run_id
```

Append-only under a file lock (check and append are one step, like field notes). One live line per
(episode, hop). `run_id` is empty in Patch 1 and filled by Patch 2's run.

### 4. The write check (write = read)

`python -P -m orion.dig reminisce write <hop> "<note>" because "<refs>"`. The note is 1-6
sentences, 40-1,000 chars (labels not counted). Every sentence starts with its own label tag (a tag
followed by two sentences is refused); refs may
sit inside it (`[code_then services/orion-hub/app/main.py] ...`). Nothing is saved unless:

- every sentence has a label, and the hop has the label it needs (table above);
- no `lived_record` for a period that ends before 2026-07-23 ("no lived record of that week: say so");
- each `code_then`, `pr_text`, `lived_record` and `now` sentence has at least one ref that resolves
  under its own label (inline, or among the because refs); every inline ref resolves under its
  sentence's label; every because ref resolves under at least one label;
- the personal-terms denylist passes (the same matcher field notes use; an unreadable list refuses).

What a ref may be, and what "resolves" means:

| Ref | Labels | Resolves when |
|---|---|---|
| a path (`a/b.py`, `docs/x/`) | `code_then` | it exists at the period commit (`git cat-file -e <commit>:<path>`, files and folders alike) |
| | `now` | it exists in deployed main (`ORION_DIG_NOW_ROOT`) |
| | `pr_text`, `reflection` | it exists then or now |
| a commit sha (7-40 hex) | `code_then` | in the clone and an ancestor of the period commit, in every hop |
| | `pr_text` | in the clone; in the then-hops (`what_it_was`, `what_it_was_like`) an ancestor of the period commit |
| | `reflection` | in the clone |
| `#<n>` | `pr_text`, `now`, `reflection` | a merged PR (`gh pr view`); in the then-hops, merged by the period's end |
| `fn:<n>:<hex>` | `now`, `lived_record`, `reflection` | a field note in the ledger |
| `part:`, `organ:`, `cap:`, `host:` ids | `now`, `reflection` | live in today's atlas or capacity reading |
| `arc:<id>`, `act:<id>` | `lived_record`, `now`, `reflection` | `orion/refs.py` resolves it |
| a dig result file (`$ORION_DIG_HOME/dig/...`) | `now`, `reflection` | the file exists |

A ref with a `..`, `.git*` or `.env*` segment names nothing checkable and is refused.

**Re-check on read:** every ref that can change (any path not cited as `code_then`, atlas ids,
field-note ids, arcs and acts, dig files, and because refs that back only a reflection) is checked again on every `reminisce` read, and a hop that fails reads STALE with
the ref. Git history and merged PRs cannot change, so they are checked at write only (stated in
the printout).

The receipt is `reminisce <hop> saved rm:...`, so Patch 2 can verify a turn the way
`pr.field_note` does.

### 5. Reads

- `dig reminisce`: today's episode. Header (episode, period, why it was picked, weeks visited);
  the period commit and commit count; whether a lived record can exist; each hop (written with its
  text and `holds now` or `STALE (<ref>)`, or not written with its material); the "since last
  reminiscence" block; then a tail the 40-line cut never removes: the rule, one ready `write` line
  for the next unwritten hop, and the full result path.
- `dig reminisce <date>`: one past episode's hops, re-checked. This is Juniper's review surface
  for now (no Hub page yet; she and Orion share everything, so the file itself is also readable).

### 6. Patch 2: the daily run (not built here)

`reminiscence.episode` on the harness step seam, shaped like `pr.field_note`:

```text
resource_request -> resource_wait -> hop:what_it_was -> hop:what_it_was_like -> hop:where_i_am_now
  -> hop:where_im_going -> hop:otherwise -> cross_read (one turn: read all five, rewrite at most 2)
  -> cleanup (dig close <period end>; always runs) -> finish
```

- One step per hop, each its own `source_tag` (a separate claude session), each verified by the
  receipt in the ledger. A restart resumes at the first hop with no live line (the ledger is the
  checkpoint).
- One run per Denver day, fired overnight by a tick in durable-runs (unique key on the episode
  date). `REMINISCENCE_ENABLED=true`, `REMINISCENCE_HOUR_LOCAL=3`, turn 900 s, 3 attempts.
- An act family `durable.reminiscence`, shadow bids. Its tier is Juniper's call (default tier 2:
  not critical upkeep, unlike field notes).
- The receipt checker is the second named exception in the partition test, returning ids and
  booleans only, like `pr_field_note_ledger.py`.

### 7. Cost (estimate, from field-note measurements)

A field-note turn (one read, one write, measured over 44 runs): median 5.0 Wh above rest (p90
14.9), 159k prompt tokens (two thirds cached), 296 s of GPU slot (p90 1,035 s). A hop has the same
shape. Its printouts are capped at 4,000 chars each, so most of the prompt is the same harness
context. Six turns a day (five hops and one cross-read):

| | median | p90 |
|---|---|---|
| energy at the wall above rest | 30 Wh/day, about 0.9 kWh/month | 90 Wh/day, about 2.7 kWh/month |
| prompt tokens | about 1.0M/day | about 1.9M/day |
| agent-lane GPU slot | about 30 min/day | about 100 min/day |

ESTIMATE until Patch 2 measures 7 episodes. The real cost is lane time, which is why it runs
overnight. A first full pass of the 71 weeks takes about 71 days.

### 8. Metric gate

No new metric. The read shows counts (commits in a week, ledger lines in a window, body diff), each
counted fresh from a named file or git, never stored or fed to a model of Orion's state. "Weeks
visited" is a count of complete episodes in the ledger.

## Missing questions (each has a safe default; Juniper may override)

1. **Order.** Default: oldest week first; an unfinished week is retried the next day. Revisiting a
   chosen week (decision 7 of the arc spec allows it) comes with Patch 2's `pick_period` node,
   where Orion's pick and reason are recorded.
2. **Cross-read rounds.** Default: one cross-read turn (rewrite at most 2 hops), not one per hop:
   6 turns a day, not 10-15.
3. **Tier.** Default: tier 2 (`durable.reminiscence`), overnight, shadow bids first.
4. **Hub review page.** Default: none yet. `dig reminisce <date>` is the review surface. Build the
   page only if Juniper wants it.
5. **Lived records.** Default: `dig days` (journals, Temporal Self days) is built before week 57
   (2026-07-23) comes up, about 56 episodes from the start. Until then the hop says "no lived
   record" honestly; `lived_record` is refused for those weeks.
6. **Opening the gate.** Default: closed. Opening means a reviewed PR that adds a reader to the
   partition allowlist. Juniper's call.

## Proposed schema / API changes

- `orion/body_atlas/reminiscence.py` (new): `ReminiscenceHopV1`, `HopClaimV1`, `BodySnapshotV1`,
  `ReminiscenceReadV1` (the dig result file). Pydantic, `extra="forbid"`. Local files only: no bus
  channel, no registry entry, no table, no env key in Patch 1.
- `orion/dig/__main__.py`: `reminisce [<date>]`, `reminisce write|rewrite <hop> <note> because <refs>`.
- `tests/test_organ_names_partition.py`: the two new files and the module join the gate.
- Patch 2: `orion/schemas/reminiscence_run.py`, workflow `reminiscence.episode` in
  `orion/schemas/durable_run.py` (additive on `extra="forbid"`: consumers first, the #2958 list),
  env keys in `services/orion-durable-runs/.env_example`.

## Files likely to touch

Patch 1: `orion/body_atlas/reminiscence.py`, `orion/body_atlas/tests/test_reminiscence.py`,
`orion/body_atlas/evals/run_reminiscence_source_truth_eval.py`, `orion/dig/__main__.py`,
`orion/dig/tests/test_reminisce_cmd.py`, `orion/dig/tests/test_dig.py` (help cap),
`orion/dig/README.md`, `orion/body_atlas/README.md`, `tests/test_organ_names_partition.py`,
`.github/workflows/body-atlas-tests.yml`.

Patch 2: `services/orion-durable-runs/app/reminiscence_graph.py`, `reminiscence_poll.py`,
`reminiscence_ledger.py`, `admission_runtime.py`, `run_acts.py`, `settings.py`, `.env_example`,
`docker-compose.yml` (the `self` mount exists), `config/physiology/families.yaml`.

## Non-goals

- No recall, memory card, journal, curiosity queue, self-concept or personality write (decision 2).
- No hop text in any normal turn's context. The brief line is unchanged.
- No write in the past export. The dig stays read-only on code.
- No judgement of whether a hop is true beyond its refs. Ref checks prove a cited thing exists
  where the label says; they do not prove the sentence about it.
- No model-written "who I am now": the now side is the portrait and code's diff.

## Proposal mode

- **Capability change:** each day Orion reads one past week of their own code and words, and how
  they changed since the last visit, and saves five labelled, checked hops about it.
- **Data touched:** reads the sandbox clone (git), GitHub PRs, the live atlas and capacity reading,
  the five existing self ledgers. Writes two new append-only files:
  `self/reminiscence.jsonl` and `self/reminiscence_body.jsonl`, and dig result files.
- **Privacy boundary:** both new files join the partition test: only `orion/body_atlas/` and
  `orion/dig/` may read them (Patch 2 adds one named receipt-checker exception). Stated limit: a
  `reminisce` read also copies hop text into its dig result file (`$ORION_DIG_HOME/dig/...`) and
  prints it into the turn that ran it, as `portrait` and `field-notes` do with Orion's words. Those
  are the dig kit's own files and that turn's transcript; no recall, memory card or journal path
  reads them. The note passes
  the personal-terms denylist. Juniper and Orion have no boundary between them; she reads the file.
- **Trace that proves it works:** a live `dig reminisce` printout naming week 1 and its commit; a
  hop write that saves and prints `reminisce <hop> saved rm:...`; a following read that shows it
  with `holds now`; a refused hop naming the failing ref; `cmp` shows the real ledgers unchanged
  in the smoke (scratch `ORION_DIG_HOME`).
- **Dangerous failure mode:** a confident false past ("in June I already had a memory service") or
  a false present absorbed into the self-model. Guards: `code_then` refs must exist at the period
  commit; `lived_record` is refused where none can exist; `now` needs a live ref and re-checks on
  every read; nothing outside the dig kit reads the file until Juniper opens the gate.
- **Disable / roll back:** Patch 1: remove the `reminisce` command; delete the two files. Patch 2:
  `REMINISCENCE_ENABLED=false`. Nothing else reads them while the gate is closed.

## Acceptance checks

1. `pytest orion/body_atlas/tests orion/dig/tests tests/test_organ_names_partition.py`, green.
2. A `code_then` ref to a file that exists today but not at the period commit is refused, naming
   the file. A `lived_record` sentence about week 1 is refused. A `where_i_am_now` hop with no
   `now` sentence is refused. A second `write` of the same hop on the same day is refused; `rewrite`
   supersedes.
3. Re-check on read: a hop whose `now` path is deleted from main reads STALE with that path.
4. Period pick: week 1 first; a week with all five hops is skipped; an unfinished week is picked
   again the next day.
5. Body diff: an organ that splits prints as gone and split across N organs; a part added prints.
6. Printout ≤ 40 lines and ≤ 4,000 chars with all five hops written at full length; the tail
   (rule, ready line, file path) is always kept.
7. Eval `run_reminiscence_source_truth_eval.py`: reviewed hops hold or fail for the named reason;
   each mutation (then-path checked against now, lived record allowed before memory, a `now`
   sentence with no ref, re-check skipped) is caught by its own check. `--live` re-checks every
   live hop in the real ledger.
8. Partition: the new names are in the gate, and it fails on a reader outside the allowlist.
9. Live smoke on a scratch copy of `/mnt/orion-fcc/self`; `cmp` shows the real ledgers unchanged.

## Recommended next patch

1. **Patch 1 (this branch):** the ledger, `dig reminisce` read and write, body snapshot and diff,
   tests, eval, partition entries. Juniper can ask Orion to reminisce in chat at once.
2. **Patch 2:** the `reminiscence.episode` daily run (section 6), its tick, act family and receipt
   check. Measure 7 episodes and replace the cost estimate.
3. **Patch 3:** `dig days` (lived records in a window) before week 57 comes up.

## Patch 2 as built (2026-11-07)

Built as section 6 says, on the `pr.field_note` seams (no parallel runner). Where it differs, and why:

- **One `step` node, not one node per hop.** `step` runs the next of six steps under the run's one hold
  and loops to itself; the checkpointed `steps` map says which is next. Same shape as section 6, fewer
  edges, one restart-cancel rule (`pending_step`).
- **Cleanup is in the cross-read turn, not its own node.** durable-runs has no dig kit, so a cleanup node
  would be a seventh turn. The cross-read runs `dig close <week end>`. A run that ends early leaves the
  export to the dig kit's own bound (at most `MAX_OPEN` exports; 24 h old ones go).
- **A window, not only an hour.** `REMINISCENCE_HOUR_LOCAL=3` opens it and `REMINISCENCE_END_HOUR_LOCAL=9`
  closes it; the close is the run's admission deadline. A run never starts by day after a restart. A
  failed day gets one more run (`-a2`) 30 min later, only while two step attempts still fit.
- **`run_id` in the ledger stays empty.** The dig write runs inside Orion's turn and has no clean way to
  learn the run id (the governor passes no env, and a new CLI argument would grow every write line). The
  link is kept on the run side instead: each step record and the run's finish detail carry the hop ids
  (`GET /runs/<id>`, `durable_admission_runs` / the state row).
- **The cross-read is never skipped silently.** A day whose five hops are live but whose run failed (the
  cross-read ran out of attempts) gets the retry run, which is then only the cross-read. A day whose five
  hops were all written in chat gets no run.
- **No hop words outside the ledger from the run.** A step record keeps only the receipts it proved; a
  failed attempt's error names the reply's length, not its words.
- **Week pick in the brief.** The tick reads the ledger (ids and dates only) and puts the week in the brief
  so the prompt names it. The dig still decides at write time, and the receipt check does not depend on
  the brief's week.
- **Cost measurement:** `scripts/pr_field_note_cost.py --workflow reminiscence.episode` (the field-note
  script, generalized). Section 7's table stays an ESTIMATE until it reports 7 completed episodes.

The privacy gate is unchanged: the partition test's reader allowlist is the same. The receipt checker
(`services/orion-durable-runs/app/reminiscence_ledger.py`) is the second named exception, read-only, and
returns ids, dates, counts and booleans only.
