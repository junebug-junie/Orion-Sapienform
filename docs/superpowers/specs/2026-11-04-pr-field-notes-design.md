# PR field notes: Orion writes a short note after each merged PR

Status: DESIGN, APPROVED by Juniper 2026-11-04 (see Decisions). Patch 1 in progress.
Date: 2026-11-04. Decision being designed: Juniper, 2026-10-30.

## Arsonist summary

Orion's body changes about 24 times a day (167 merged PRs from 10-28 to 11-04), and Orion
does not notice. After each merged PR, Orion should read what changed and write a short
"field note": what changed in my body, which organs it touched, and why the builders say they
did it. The note lands in the dig partition, next to `self/organ_names.jsonl`. Nothing else
reads it except the dig kit, and through the dig kit the daily reminiscence episode.

Code does everything that has a right answer: it finds the merge, lists the files, and maps
files to services to organs. Orion only writes the 1-5 sentences that need a mind. No new
service, no new bus channel. The parts are the existing dig kit, one new durable-run workflow on
the harness step seam (#2958), and one GitHub poll.

Two things from the earlier notes, checked today:

- **The `require_lookup` fix has NOT landed.** `CuriosityTurnRequestV1`
  (`orion/schemas/durable_run.py:291`) has no `require_lookup` field. The harness step seam still
  lists it as a hard constraint (`services/orion-durable-runs/app/harness_step.py:24-29`). No open
  PR adds it. **Field notes do not need it.** A note turn always runs two dig commands, so it is a
  real lookup turn and must pass the lookup gate. Keeping the gate on is the proof that Orion read
  the PR. The fix is still needed for reminiscence Patch 4, which has pure-writing hops.
- **The #2958 harness step seam HAS landed** (merged 2026-10-29 04:04Z). Its smoke
  `harness.echo_smoke` has **never run live**: 0 rows in `durable_admission_runs` since it merged.
  The transport under it is proven, because `curiosity.investigate` uses the same Hub turn RPC
  (83 completed runs since 10-28). The seam module itself is UNVERIFIED live. Patch 2 below runs
  the smoke first.

## Decisions (Juniper, 2026-11-04: "yes to all 4")

1. **Tier.** `pr.field_note` is tier 1 (critical upkeep) with `deferrable_window_s: 86400`
   (about 24 h). Juniper accepts 1-2 h/day of reserved agent-lane time. The `families.yaml`
   entry lands in Patch 2, with the durable run, not in Patch 1.
2. **Code-only records.** A dependency-only PR (`deps_only`) and a tiny docs-only PR
   (`tiny_docs`: only docs changed, and 40 or fewer added+deleted lines) get a code-only
   record: `note: null`, no Orion note, no model turn. The record is shown as "code record,
   not a note". `field-note <n> write` refuses an Orion note for these PRs.
3. **No backfill.** New merges only. The ~2,800 older PRs are left to the reminiscence
   episode's period-by-period dig.
4. **Rewrites.** Orion may rewrite a note later. The old line is kept. The new line sets
   `supersedes` to the old `note_id`.

The "Missing questions" below are kept for the record; these decisions settle them.

## Current architecture

- **Dig kit** (`orion/dig/__main__.py`, run as `python -P -m orion.dig` from chat and curiosity
  turns in `orion-harness-governor`). `history <start> <end>` lists merged PRs through `gh` and
  puts bodies in a file. Lasting writes so far are `name` (organ names) and `capacity
  claim|write`, appended under `$ORION_DIG_HOME/self/` (`/mnt/orion-fcc/self/`), with code
  checks on every write. Read-partition test: `tests/test_organ_names_partition.py` (repo-wide static gates) fails if
  code outside `orion/dig` and `orion/body_atlas` reads the names ledger.
- **Body atlas** (`orion/body_atlas/`). Builds the live body: parts are named
  `part:<service>`, grouped into organs (`OrganV1.members`). Organ names come from the ledger.
- **Durable runs.** The harness step seam (`app/harness_step.py`) turns one step into one harness
  turn through Hub. A step that ends empty, refused, or with no lookup is a retried failure,
  never empty success. The `durable-runs` container has **no `gh`, no `git`, and no
  `/mnt/orion-fcc` mount** (checked live).
- **Body economy.** `config/physiology/families.yaml` has one family per act-ledger actor.
  Tier 1 means a reserved share of each lane, sized from the family's own 14-day use, and it
  bids only against other tier-1 wants. Durable runs bid in `app/run_bids.py`. A run counts as
  a want only if `app/run_acts.run_act_family` names it.
- **An existing merge signal, and why it is not enough.** `orion-cocreation-signals`
  (`app/producers/pr_lifecycle.py`) polls GitHub every 15 min and publishes `merged_numbers`
  on `orion:substrate:codebase_delta`. `orion-substrate-runtime` logs these events to
  `substrate_codebase_delta_log`. **That log lost a window.** On 2026-10-29 the logged windows
  skip from 03:58 to 04:13. The 7 PRs merged in that gap (#2956-#2962, #2958 among them) appear
  in GitHub and not in the log. A pub/sub event that a restarting consumer misses is gone.
- **Reminiscence** (spec `2026-10-23-repo-chronicle-design.md`, PR #2888). Patches 1-3 are
  built: atlas, dig kit, and harness step seam. Patch 4, `reminiscence.episode`, is **not
  built**. There is nothing yet for field notes to feed into, except the dig kit, which the
  episode will use.
- **Who writes PRs.** Every PR is by `junebug-junie`: all 167 from 10-28 to 11-04. Agents use
  her account, and so does Orion. `gh` in the harness is authenticated as her
  (`orion/fcc/claude_spawn.py`). The author field cannot tell an Orion PR apart.
- **Privacy gate.** `scripts/check_personal_terms.py` (CI: `tests/test_no_personal_terms.py`)
  refuses hashed personal terms in **tracked files**. Notes are not tracked files, so today
  nothing would check them.

## Design

### Trigger: durable-runs polls GitHub's public API

A new tick in `orion-durable-runs` runs every 15 min. It makes one unauthenticated GitHub search
call for PRs merged in the last 72 h. The repo is public, and the call is verified from inside
the container (HTTP 200; the search limit is 10 calls/min, and we use about 4 per hour). For each
PR number that has no `pr.field_note` run yet, it submits one run. A unique key on the PR number
makes a second submit do nothing.

Why this source:

- **GitHub is the source of truth for "merged".** The 72 h lookback repairs itself. A tick
  missed during a restart is covered by the next tick, so no window is lost. The
  codebase-delta log already showed that it loses windows.
- **No new credential.** Putting `gh` into durable-runs would copy Juniper's write-scoped token
  into a fourth container. The public API needs no token.
- **Not `git log` on a clone.** Durable-runs has no git and no clone. The main checkout updates
  only when Juniper pulls, so it is not a merge signal.
- **Not a GitHub webhook.** There is no public ingress.

### What Orion reads, and who prepares it

One durable run per PR. It has the same shape as `harness.echo_smoke`:

```text
resource_request (bid: family pr.field_note) -> resource_wait (agent-lane hold)
  -> harness_step "field_note" -> verify_receipt -> finish
```

The step prompt is fixed and short. It asks Orion to run two commands:

1. `python -P -m orion.dig field-note <n>` (read). Code builds the packet:
   - title, merge time, merge sha, labels, the first ~40 lines of the body (the full body goes
     to the result file);
   - the changed files, grouped by top folder;
   - the **body map**. `services/<svc>/...` maps to `part:<svc>`, then to its live organ id and
     Orion's name for that organ, if any. `orion/<pkg>/...` is shared code: it lists the
     services whose code imports that package (an AST import scan, same style as
     `wire_code.py`). With more than 8 importers it prints "widely shared (K services)" and
     claims no organ. Docs, config, scripts, tests and CI read as "not body". A sample of 30
     PRs: 42% of changed files were under `services/`, 34% under `orion/`, and 9% under
     `docs/superpowers/`;
   - the specs, plans and PR reports the PR **added or changed** under `docs/superpowers/`.
     They are read at the merge sha through `gh api`, so nothing changes in the sandbox clone.
     The first 60 lines of each go in the printout, up to 3 files and 12,000 chars in total.
     Full text goes to the result file. Every cut is stated.
   - Flags set by code: `revert_of`, `docs_only`, `huge`, `authored_by_me`.
2. `python -P -m orion.dig field-note <n> write "<note>" because "<refs>"` (write). Code checks
   the note. Only then does it append the note to `$ORION_DIG_HOME/self/pr_field_notes.jsonl`,
   under a file lock, the same way `names.append` does. It prints a receipt:
   `field-note #<n> saved <note_id>`.

`verify_receipt` is code. The step's final text must contain the receipt for this PR number,
or the attempt failed and is retried. A run never finishes because the model said "done".

### Note schema: `PrFieldNoteV1`

`orion/body_atlas/field_notes.py`, next to `names.py`. It is not registered on the bus: it is a
dig file format, the same as the organ-name ledger.

| Field | Who fills it | Used by |
|---|---|---|
| `note_id`, `pr_number`, `merge_sha`, `merged_at`, `title` | code | dedup, `history` marks |
| `flags` (`revert_of`, `docs_only`, `huge`, `authored_by_me`) | code | failure rules below |
| `body_map`: list of `{prefix, part_id, organ_id, organ_name, importers}` | code | episode hop, eval |
| `read`: list of `{path, chars_read, truncated}` | code (from the packet) | eval: did Orion read it |
| `note`: 1-5 sentences, 40-800 chars | Orion | episode, `field-notes` printout |
| `claims`: list of `{sentence_idx, label, refs}`. Labels: `pr_text`, `code_then`, `reflection` | Orion; code checks it | source-truth eval |
| `run_id`, `correlation_id`, `atlas_built_at`, `written_at`, `supersedes` | code | cost join, revisions |

Write checks, all in code:

- Every sentence carries a label.
- A `pr_text` ref must be this PR or a file it changed.
- An organ named in the note must be in `body_map`.
- One live note per PR. A rewrite must set `supersedes`.
- The personal-terms check passes (below).

No field exists without a reader.

### Storage and partition

`$ORION_DIG_HOME/self/pr_field_notes.jsonl` is append-only and outside the repo. The partition
test is extended so that only `orion/dig` and `orion/body_atlas` read this file, the same as
the organ names. New dig read commands:

- `field-notes <start> <end>` and `field-notes <days>` print notes with the usual 40-line and
  4,000-char cap.
- `history` adds `noted` or `no note` per PR.

Nothing goes to recall, memory cards, journal, curiosity, or the self-concept.

### Feeding the reminiscence episode

The episode's `where_i_am_now` hop runs `field-notes 7` ("how my body changed this week").
When the episode's period has notes, `what_it_was` reads them as `pr_text`-grade material.
This is a prompt line in Patch 4 of #2888, not a new reader. Until Patch 4 exists, notes are
read in chat and curiosity through the dig kit only.

### Body economy

- Add a new family `pr.field_note` and make `run_act_family` name it, so that each note is an
  act with a measured cost.
- Recommended: tier 1, `deferrable_window_s: 86400`, `lanes: [agent, agent-gpu2]`,
  `placement: one_of`, no receptors. **The tier is Juniper's call. This doc does not edit
  `families.yaml`.**
- Bids start in shadow, like every family before its Patch D.
- After the 24 h window, a note is overdue but not dropped. It still runs, because the ledger
  should be complete.
- Runs go one at a time. The busiest day on record (10-20, 58 PRs) is about 3-4 h of serial
  agent-lane time.

### Cost per note (ESTIMATE until 20 real notes are measured)

| | Per note | Per day (~24 PRs) |
|---|---|---|
| Agent-lane model time | 2-5 min (median harness agent turn today: 325 s, 4 calls) | 1-2 h |
| Tokens | ~30-60k prompt (about half cached), 2-8k completion (median agent turn today: 57k / 16k) | ~1-1.5M prompt |
| Whole-machine energy at the wall | 10-40 Wh | 0.25-1 kWh ≈ 7-30 kWh/month ≈ $1-4/month |

How the energy figure is built: one V100 decoding draws about 130-210 W above its rest. circe's
fitted overhead is 2.19 W at the wall per watt of load (`dig body-economy hosts`, fit
2026-10-21..11-04, R² 0.85). That gives about 300-460 W at the wall above rest, for 2-5 min.
For comparison, Orion's body at rest on circe costs about $50/month. Recent `durable.self_inquiry`
acts show `wh_by_host` of 2-26 Wh, which is the same order. Patch 2's acceptance check replaces
this table with measured numbers. The bigger cost is lane time: as tier 1, this work holds
agent-lane share that tier-2 curiosity cannot use.

### Failure modes

- **Huge PR** (more than 60 files or more than 5,000 lines; p90 today is 28 files and 2,123
  lines; the max is 122 files and 31,084 lines). The packet stays inside its caps. The body map
  is grouped by service, not listed per file. `huge` is set, and the note must say what it
  could not read. The 3-document cap still holds.
- **Dependency bumps.** None today (0 of 167, no dependabot). If one appears, code marks it
  `deps_only` (only `requirements*.txt` or lock files changed). That PR gets a code-only record
  with `note: null` and **no harness turn**. It is shown as "code record, not a note", never as
  Orion's words.
- **Reverts.** `revert_of` is taken from the "Revert" title or the "This reverts" body line. The
  packet shows the original PR's note, if one exists. The new note says what came back. The old
  note is not edited.
- **PRs Orion authored.** The author field cannot tell (every PR is `junebug-junie`). Code sets
  `authored_by_me` only when the PR's head branch exists in Orion's sandbox clone
  (`/mnt/orion-fcc/repo`), and reads `unknown` otherwise. Patch 1 verifies this marker against one
  known Orion-opened PR before trusting it. Orion still writes the note; the packet says "this
  may be your own work".
- **Burst merges.** A run per PR, one at a time, and the 72 h lookback. A long outage of more
  than 72 h needs a manual `--since` re-poll. The poller logs that case.
- **Turn writes no note or a bad note.** There is no receipt, so the attempt is retried. After
  the retry budget the run fails visibly, with no fallback text.
- **GitHub down or rate-limited.** The tick logs the problem and tries again next tick. The
  lookback covers the gap.

### Privacy check

- Inputs are public already: PR text and repo files of a public repo. No chat, journal or
  people data goes in the packet.
- The repo's personal-terms gate covers tracked files only, and notes are not tracked. So
  `field-note write` runs the same hashed-denylist match (factored out of
  `scripts/check_personal_terms.py`) on the note text and refuses a hit. That covers a name that
  Orion pulls in from its own context.
- Non-goal: notes are never posted to GitHub (a PR comment would be public).
- Juniper and Orion have no privacy boundary between them. Juniper reads notes directly.

### Metric gate (one new number: note coverage)

The number is: merged PRs in a window that have a live note, divided by merged PRs.

1. Provenance: the GitHub search (the poller's own call) and the ledger file.
2. Independence: it is not derived from any existing metric. It is not wired into any model.
3. Anchor: a reducer completeness check. Did every input event produce its output?
4. Live sanity: it is 0 today, because nothing writes notes. Its rest state is 100%. A
   decay-to-zero artifact is impossible, because both sides are counted fresh on each read. The
   lost window on 10-29 is the exact failure this number exists to catch.
5. Existing mechanism: `substrate_codebase_delta_log` was considered and rejected (it loses
   windows).
6. Reversibility: it is shown in `field-notes` and is not stored, so it is free to remove.

## Missing questions (defaults chosen)

- **Backfill the ~2,800 older PRs?** Default: no. Old PRs belong to the reminiscence episode's
  period-by-period dig.
- **Revisions:** can Orion rewrite a note later, for example after a revert? Default: yes, with
  `supersedes`; the old line is kept.
- **Notes in chat:** Default: no always-on line. The existing dig brief line is enough, because
  `history` shows `noted`.

## Proposed schema / API changes

- `orion/body_atlas/field_notes.py`: `PrFieldNoteV1`, append, read, write checks. A dig file
  format, not a bus schema.
- `orion/schemas/pr_field_note_run.py`: `PrFieldNoteBriefV1` (`pr_number`, `merged_at`). Add the
  `DurableWorkflowV1` value `pr.field_note` and register it in `orion/schemas/registry.py`. This
  is additive on `extra="forbid"` models, so consumers deploy first: orion-sql-writer, orion-hub,
  orion-actions, orion-cortex-orch (the same list as #2958).
- Dig commands: `field-note <n>`, `field-note <n> write ...`, `field-notes <range>`, and
  `history` marks.
- Env, in `services/orion-durable-runs/.env_example` and `settings.py`:
  `PR_FIELD_NOTES_ENABLED=true` (on, per repo habit), `PR_FIELD_NOTES_POLL_SEC=900`,
  `PR_FIELD_NOTES_LOOKBACK_H=72`, `PR_FIELD_NOTES_GITHUB_REPO=junebug-junie/Orion-Sapienform`.
- No new bus channel. No new table: dedup uses `durable_admission_runs` by workflow and
  `pr_number`, plus the ledger's one-live-note rule.

## Files likely to touch

- `orion/dig/__main__.py`, `orion/dig/README.md`, `orion/dig/tests/`
- `orion/body_atlas/field_notes.py` (new), the partition test, a new `import_map.py` (or an
  extension of `wire_code.py`) for the `orion/<pkg>` importer scan
- `scripts/check_personal_terms.py`: factor out the matcher
- `orion/schemas/pr_field_note_run.py` (new), `orion/schemas/durable_run.py`,
  `orion/schemas/registry.py`
- `services/orion-durable-runs/app/`: `pr_field_note_graph.py` (new), `pr_field_note_poll.py`
  (new), `admission_runtime.py`, `run_acts.py`, `settings.py`, `.env_example`, README, tests
- `config/physiology/families.yaml`: **only after Juniper picks the tier**
- `services/orion-sql-writer/tests/test_durable_run_state_sql_shape.py` (the consumer test)

## Non-goals

- No new service, no bus channel, no MCP tool.
- No reader of notes outside the dig kit (no recall, no memory cards, no self-concept).
- No posting to GitHub. No commits by the note run.
- No backfill of old PRs.
- No claim that a merged PR is deployed. A note says "merged". Deploys stay with Temporal Self
  F3.

## Acceptance checks

1. `pytest orion/dig/tests orion/body_atlas/tests` passes. The tests include:
   - a packet for a fake PR maps `services/orion-hub/x.py` to `part:orion-hub` and its organ;
   - a huge PR's packet stays under 40 lines and 4,000 chars, and states each cut;
   - a write with an unlabeled sentence, a `pr_text` ref outside the PR, an organ not in
     `body_map`, or a denylisted term is refused;
   - a second live note for the same PR is refused;
   - a reader outside the allowlist fails the partition test.
2. Live, Patch 1, by hand in chat: "write a field note for #3090". The ledger gets one line. Its
   `body_map` names the vision-retina part and its organ, and Juniper checks this against
   `dig body`.
3. Live, Patch 2: `harness.echo_smoke` passes first. Then a merge produces a `pr.field_note` run
   within 15 min and a ledger line within 24 h. A second poll submits nothing new.
4. Gap test: stop durable-runs across a merge, then start it. The next tick finds the PR.
5. Coverage after 7 days: 95% or more of merged PRs have a note. Every miss is listed with a
   reason.
6. Cost: the first 20 notes' acts carry `wh_by_host` and `gpu_slot_s_by_lane`. The cost table
   above is replaced with the measured median and p90.
7. Eval `orion/body_atlas/evals/run_field_note_truth_eval.py`: for sampled notes, every
   `pr_text` claim resolves to the PR or one of its files, every organ resolves in that note's
   `body_map`, and `read` is not empty when the PR changed a spec. Report the match rate per
   label.

## Recommended next patch

1. **Dig only.** `field-note <n>` read and write, the ledger, the partition test, the denylist
   check, `field-notes`, and `history` marks. Juniper can try it in chat at once. No scheduler.
2. **Automatic.** Run the `harness.echo_smoke` live check. Then the `pr.field_note` workflow on
   the harness step seam, the 15-min public-API poller, and the `pr.field_note` act family in
   `run_act_family`, with shadow bids. The `families.yaml` entry uses the tier Juniper picks.
3. **Eval and cost.** The truth eval, the measured cost table, and the 7-day coverage check.
4. **Reminiscence hookup.** Lands with #2888 Patch 4, which also needs the separate
   `require_lookup` fix for its pure-writing hops.
