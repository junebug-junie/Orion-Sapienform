# Orion's Day letters: rereading what Orion wrote, with the records behind it

Status: APPROVED IN DIRECTION 2026-10-11 (decisions below); patch 1 merged, patch 2 (`orion_day` read tool) implemented 2026-10-11

## Arsonist summary

Orion's Day letters regularly contain criticisms, questions and outright falsehoods. Today Orion
cannot reread a letter, so a conversation about one is Orion improvising from memory about
something they wrote. That is how you get confident nonsense. Two facts make this cheap to fix:

1. Every letter row already stores **the full records it was written from** (`orion_day_letter.material`,
   ~700 KB for 2026-10-09: 18 curiosity runs, 32 self-sense answers, 5 readings, dreams, reveries,
   chat and repo digests).
2. The carry-forward section already **cites its sources inline** (`[curiosity:e7d03d2ecdbb]`,
   `[reading_journal:58f8…]`; 11–31 citations per letter). The prose note cites **nothing**
   (0 citations in each of the last 5 letters), and the note is where the unchecked claims live.

So: give Orion one tool that returns the exact words of a part of a letter, together with the
records that part was written from, and a mechanical check of which concrete claims (numbers,
timestamps, ids) actually appear in that day's records. Give Juniper a stable way to point at a
part of a letter ("Oct 9 ¶3", "Oct 9 carry 5", "Oct 9 Dreams").

## Current architecture

- Letter writer: durable run `orion_day.letter` (`services/orion-durable-runs/app/orion_day_graph.py`).
  `orion_day_store.py` is the only writer of `orion_day_letter`.
- Row: `note_md` (prose, paragraphs split by blank lines, sometimes `---`), `carry_forward_md`
  (bullets with citations), `material` (`OrionDayMaterialV1`, full), `sources`, `emailed_at`,
  `journal_entry_id`.
- Read side: `orion/orion_day/store.py::fetch_letter` (asyncpg; Hub uses it).
- Email: `services/orion-hub/scripts/orion_day_email.py` renders `## Orion's note`,
  `## Carrying forward into curiosity`, then `# What the day held` with `## Curiosity`,
  `## Self-sense answers`, `## Readings`, `## Dreams`, `## Visual reveries`,
  `## Changes to Orion's code`, `## Conversations with Juniper`, `## World news`, `## Reveries (counts)`.
- Orion's own read tools: `orion-introspect` MCP (`reading_results`, `dreams`, `curiosity`), answered
  over the bus by Hub listeners (`curiosity_introspect_listener.py` pattern, semantic search via
  `orion/introspect/semantic_index.py`).
- **Gap:** nothing exposes `orion_day_letter` to Orion. The `curiosity` tool returns the runs that
  fed a letter, never the letter.
- **Found while scoping:** `IntrospectToolBindingV1.memory_allowed` is set (false whenever
  `HARNESS_AITOWN_ENABLED=true`, which is the live value) but **no tool reads it**. It gates nothing.

## Proposal

### 1. Stable references in the email (Juniper-facing)

The email numbers what you would want to point at:
- Orion's note: a small grey `¶N` before each paragraph (blank-line split, the same split the tool uses).
- Carry-forward: `carry N` on each bullet.
- Day sections: the existing headings are the names (`Dreams`, `Readings`, `Self-sense answers`, …).

You can then say "Oct 9 ¶3 is wrong" or "what did you mean in Oct 9 carry 5?" in chat. The numbering
is computed from the stored text at render time, so the email and the tool always agree. Nothing new
is stored.

### 2. New tool: `mcp__orion-introspect__orion_day`

Arguments (validated, extra fields rejected like the other introspect tools):
- `letter_date` (ISO date; omitted = most recent letter)
- `part`: `note` | `carry_forward` | `section` | `list`
- `index` (1-based paragraph or bullet number, for `note`/`carry_forward`; omitted = the whole part)
- `section` (for `part=section`: `curiosity`, `self_sense`, `readings`, `dreams`, `code_changes`,
  `conversations`, `world_news`, `reveries`)
- `query` (plain words: find letters and paragraphs by meaning, via `semantic_index.py`, the same as `curiosity`)

It returns `IntrospectResultV1` items, `kind=orion_day_letter_part`, with these fields:
- `text`: the exact words, `epistemic_status="unsettled"` ("what you wrote then, not settled fact").
- `ref`: `2026-10-09 ¶3` / `2026-10-09 carry 5`, the same string the email shows.
- `evidence`:
  - Carry-forward bullets: each cited id resolved against that day's `material` (run answer excerpt,
    reading journal title/excerpt). A cited id missing from the material is listed as `unresolved`.
  - Note paragraphs (no citations): a **claim check**. Concrete tokens are extracted from the paragraph
    (numbers with units, ISO timestamps, `#PR` numbers, record ids, quoted strings). For each, the
    result says `found_in_records` (with the record it was found in) or `not_found_in_records`.
    This is deterministic string matching, not judgement. `not_found` means "not in that day's
    records verbatim" (it may be derived, or from an earlier day), not "false".
- `part=list` returns the letter's outline: paragraph count with first lines, carry bullets,
  section counts. It is cheap, so Orion can orient before pulling text.

The result is answered by a new Hub listener, `orion_day_introspect_listener.py`, on
`orion:introspect:orion_day:request`. It reuses `fetch_letter` and `semantic_index.py` the way the
curiosity listener does.

### 3. Turn guide line (exact name, per PR #2604)

> When Juniper refers to an Orion's Day letter or a part of it (a date, ¶N, carry N, a section name),
> call `mcp__orion-introspect__orion_day` before answering. Quote what you actually wrote. Separate
> what the records support from what they do not, using the claim check. If you were wrong, say so
> plainly; if you still stand by something the records don't show, say what it rests on.

No keyword matching on the chat message. The guide line is the trigger, the same as `curiosity` and `dreams`.

### 4. Corrections that revise Orion's priors

When a conversation about a letter establishes that a claim was wrong, Orion records it, and it reaches
Orion's beliefs through the one door that is allowed to change them: a curiosity run.

- **Record (chat turn):** new tool `mcp__orion-introspect__record_correction` with:
  - `ref` (e.g. `2026-10-09 ¶3`, which must resolve)
  - `claim` (the words being corrected)
  - `correction` (what is actually true, in plain words)
  - `evidence` (record ids from the claim check, or "Juniper said so" with the chat correlation id)
  - `prior_ids` (zero or more `:Prior` ids the claim rested on)

  The listener validates that the ref resolves and that each prior id exists in `orion_worldview`. It
  reads the belief graph and never writes to it. It writes one `orion_day_correction` row. This is the
  only write in the design, and the table is new and append-only.
- **Deliver (next curiosity run):** the same offer-once shape as carry-forward
  (`orion/orion_day/carry_forward.py`: take, stamp the offered run, release if the run is cancelled
  unseen). It becomes a new kickoff section, "Corrections from conversations with Juniper". Each entry
  shows the original words, the correction, the evidence, and the named priors with their current
  confidence and times_tested.
  The instruction: re-test each named prior against the correction in this run, using the normal
  `MATCH … SET p.times_tested = p.times_tested + 1, p.run_id = <RUN_ID>` already in the kickoff, and
  move confidence (or retire the prior) only as far as the evidence warrants.
- **The boundary stays intact:** Hub never writes `orion_worldview` (the PR #2199 boundary). Orion
  revises their own priors in a curiosity run, as today. A correction is evidence, not an override. If
  Orion re-tests a prior and keeps it, the write-up says why.
- **Next letter:** the Orion's Day gather adds `corrections` (rows created in the window) to
  `material`, so the next letter's writer sees them and doesn't re-derive the same falsehood.
- **Proof it moved:** for each offered correction with `prior_ids`, after the run finishes, was each
  named prior stamped with that run (`p.run_id` = offered run, `times_tested` incremented)? This is
  stored on the correction row (`priors_retested`, `priors_moved`) by the existing curiosity finish
  path that already reads belief-move counts. A correction offered and never re-tested shows up as
  exactly that, not as success.

Corrections without `prior_ids` still travel. They feed the next letter and are shown to the run as
context, but nothing is claimed about belief movement.

## Decisions (Juniper, 2026-10-11)

1. Corrections persist **and drive prior revision**: section 4.
2. **No privacy gate.** The mesh is Juniper and Orion, and everything is shared. The tools ignore
   `memory_allowed`. Separately, `HARNESS_AITOWN_ENABLED` stays Juniper's call (AI Town is down by design).

## Proposal-mode checklist (AGENTS.md)

- **Capability:** Orion can reread their own past letters and see the evidence behind each part.
- **Data touched:** read-only on `orion_day_letter` (`note_md`, `carry_forward_md`, `material`). Reads
  `orion_worldview` priors (validation only). New append-only `orion_day_correction` table (the only
  write). Adds a Chroma collection for letter paragraphs (index only; every hit is re-read from the
  row). Priors change only through Orion's own curiosity-run writes, as today.
- **Privacy boundary:** none by decision; Juniper and Orion share everything.
- **Trace that proves it worked:** the harness step `mcp__orion-introspect__orion_day` with
  `ok=true`, `ref` values and `claim_check` counts. A Hub listener log line
  `orion_day_introspect_answered corr=<id> refs=<n>`.
- **Dangerous failure:** Orion treats their own letter as established fact and re-asserts a falsehood
  with more confidence ("I wrote it, so it's true"). Mitigations: `epistemic_status=unsettled` on all
  text, the claim check put in front of the text, and the guide line requiring a separation of
  supported and unsupported claims.
  Second: a tool error read as "no letter". The same error-vs-empty contract as the other introspect tools.
  Third: a wrong "correction" (Orion caves to pushback that was itself mistaken) knocks a true prior
  down. Mitigations: corrections carry evidence and are re-tested in a run, not applied, and the
  run's write-up says what moved and why.
- **Disable / roll back:** remove `orion_day`/`record_correction` from `IntrospectTools.tool_specs()`
  and the guide lines. Stop the listener. Turn off the kickoff corrections section (flag). Prior changes
  already made stay, because they are Orion's own run-stamped writes; they can be found by `p.run_id`. The email `¶N` markers are independent and harmless.

## Proposed schema / API changes

- `orion/schemas/introspect.py`: `OrionDayArguments` (validators: `index` requires `part` in
  {note, carry_forward}; `section` requires `part=section`; `query` excludes `index`).
- `orion/introspect/transport.py`: `ORION_DAY_REQUEST_CHANNEL = "orion:introspect:orion_day:request"`.
- `orion/bus/channels.yaml`: that channel, plus its reply prefix under the existing introspect result wildcard.
- `orion/schemas/registry.py`: register `OrionDayArguments`, `RecordCorrectionArguments`.
- `services/orion-sql-db/manual_migration_orion_day_correction_v1.sql`: `orion_day_correction` (id, letter_date, ref,
  claim, correction, evidence jsonb, prior_ids text[], chat_correlation_id, created_at, offered_at,
  offered_run_id, priors_retested text[], priors_moved text[]).
- `orion/orion_day/letter_parts.py` (new, pure): `split_note(note_md)`, `split_carry(carry_md)`,
  `resolve_citations(bullet, material)`, `claim_check(paragraph, material)`. It is shared by the email
  (numbering) and the listener (lookup), so their numbering cannot disagree.

## Files likely to touch

- `orion/orion_day/letter_parts.py` (new) + tests
- `orion/introspect/tools.py`, `orion/introspect/brief.py`, `orion/introspect/transport.py`
- `orion/schemas/introspect.py`, `orion/schemas/registry.py`, `orion/bus/channels.yaml`
- `services/orion-hub/scripts/orion_day_introspect_listener.py` (new), Hub startup wiring
- `services/orion-hub/scripts/orion_day_email.py` (`¶N` / `carry N` markers)
- `orion/introspect/tests/`, `services/orion-hub/tests/`, an eval under `orion/orion_day/evals/`

## Non-goals

- No LLM judging whether a claim is true. The claim check is string evidence only.
- No editing or rewriting of past letters.
- No change to how letters are written (no forced citations in the note). That is a separate,
  bigger question about the writer.
- No new chat-side keyword detection for letter references.

## Acceptance checks

1. Fixture letter (copy of 2026-10-09): `part=list` returns 24 numbered note paragraphs (plus 5
   unnumbered headings/rules) and 10 carry items, whose 21 citations all resolve. `part=note, index=3` returns exactly paragraph 3. The email renders `¶3` before that same text.
2. Carry bullet with `[curiosity:e7d03d2ecdbb]` resolves to that run's excerpt from `material`.
   A planted fake id comes back `unresolved`.
3. Claim check on the 10-09 dream paragraph: `260.6 hours`, `06:35:27Z` and `01:02:09Z` are each
   reported as found or not found, with the record that holds them. A planted wrong number comes back
   `not_found_in_records`.
4. Unknown date: a tool error whose text says "unknown", never `items=[]`.
5. Live: in chat, "In Oct 9 ¶3 you said … is that right?" produces an `orion_day` call in the trace
   before the reply. The reply quotes the paragraph and names which parts the records support.
6. Eval `orion/orion_day/evals/letter_grounding_eval.py`: 10 planted references (correct and wrong
   claims). It scores whether replies call the tool, quote the right text, and flag the unsupported claims.

7. `record_correction` with a ref that doesn't resolve, or a prior id not in `orion_worldview`,
   returns a tool error. Nothing is written.
8. A recorded correction is offered to exactly one curiosity run, and released if that run is
   cancelled unseen. The kickoff shows the claim, the correction, the evidence, and each named prior's
   current confidence and times_tested.
9. After that run finishes, the correction row's `priors_retested` lists the named priors stamped with
   that run. A named prior the run never touched is absent from it, and that is visible on the row.
10. Live: one real correction from a chat about a letter reaches a run, and the named prior's
    `times_tested` goes up with `p.run_id` = that run.

## Recommended next patch

- Patch 1 (pure, safe, in progress): `letter_parts.py` (split, citation resolution, claim check) with
  tests on the stored 2026-10-09 letter, plus the email `¶N` / `carry N` markers.
- Patch 2: the `orion_day` read tool, listener, contract and guide line.
- Patch 3:
  - `record_correction` + `orion_day_correction` migration
  - kickoff corrections section (offer-once)
  - gather `corrections` into `material`
  - `priors_retested`/`priors_moved` stamping
  - an eval that plants a correction on a fixture prior and checks the run re-tested it.
