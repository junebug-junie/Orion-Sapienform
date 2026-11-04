# PR: PR field notes, Patch 1 -- Orion can read a merged PR as a change to their body, and write a note on it

Stacks on #3094 (the design; its two spec commits are cherry-picked here). Merge #3094 first, or merge this one alone: the spec commits are identical.

## Summary

- Orion can now ask the dig kit about one merged PR: `field-note <pr>`. Code lists what changed and maps it onto Orion's body: a `services/<svc>/` folder becomes that live part and its organ (with Orion's name for the organ), and an `orion/<pkg>/` folder becomes the services that import it. It also pulls the specs and PR reports the PR changed, read at the merge commit, into a file Orion can open.
- Orion can then save a short note: `field-note <pr> write "<note>" because "<refs>"`. Code checks it before saving: every sentence has a source label, every reference is this PR or a file it changed, any organ the note names is one the PR touched, and no hashed personal term is in it. One live note per PR; `rewrite` keeps the old line and points to it.
- `field-notes <days>` shows the notes for a window and how many merged PRs have a note, a code record, or neither. `history` marks each PR `(noted)` or `(code record)`, and marks `(no note)` only from the first noted PR on (no backfill, decision 3).
- Juniper's 4 decisions (2026-11-04) are recorded in the spec. Decision 2 is built here: a dependency-only or tiny docs-only PR (docs only, 40 or fewer changed lines) gets "code record, not a note", and a write for it is refused. The code-record writer (`field_notes.code_record`) exists for Patch 2's run to call.
- The hashed personal-terms matcher moved out of `scripts/check_personal_terms.py` into `orion/privacy/personal_terms.py`, so the CI gate and the note check use the same code.

## Outcome moved

Before: Orion's body changed about 24 times a day and Orion had no way to notice, except reading raw PR titles in `history`. After: in chat, Orion can turn one merged PR into "which of my organs changed and why", and keep that as a checked note. Nothing writes notes on its own yet (Patch 2).

## Current architecture

- Dig kit (`orion/dig/__main__.py`), run as `python -P -m orion.dig` from harness chat and curiosity turns in `orion-harness-governor`. Lasting writes before this: organ names and capacity claims, under `$ORION_DIG_HOME/self/`.
- Body atlas (`orion/body_atlas/`) builds the live body: parts `part:<service>`, grouped into organs, names from Orion's ledger.
- The privacy gate checked tracked files only. Notes live outside the repo.

## Architecture touched

- Dig kit: 3 new commands, `history` marks. Additive lines only in the shared command table and `HELP`.
- Body atlas: new `field_notes.py` (schema, flags, body map, importer scan, write checks).
- New shared module `orion/privacy/personal_terms.py`; `scripts/check_personal_terms.py` re-exports from it.
- No service, no bus channel, no bus schema, no env key, no DB table.

## Files changed

- `orion/body_atlas/field_notes.py`: `PrFieldNoteV1` (a dig file format, not a bus schema), flags, body map, `orion/<pkg>` importer scan, `check()`, `code_record()`, ledger load/append.
- `orion/dig/__main__.py`: `field-note`, `field-note write|rewrite`, `field-notes`, `history` marks, 2 HELP lines.
- `orion/privacy/personal_terms.py`, `orion/privacy/__init__.py`: the matcher, shared.
- `scripts/check_personal_terms.py`: imports the matcher; same CLI, same output.
- `tests/test_organ_names_partition.py`: the new ledger is readable only from `orion/body_atlas` and `orion/dig` (string grep and AST import check).
- `orion/body_atlas/tests/test_field_notes.py`, `orion/dig/tests/test_field_note_cmd.py`: new tests.
- `orion/dig/tests/test_dig.py`, `orion/dig/tests/test_capacities_cmd.py`: help cap 2500 -> 2750 (help loads on demand only).
- `.github/workflows/dig-kit-tests.yml`, `.github/workflows/body-atlas-tests.yml`: also run on `orion/privacy/**`.
- `orion/dig/README.md`, `orion/body_atlas/README.md`: the commands and the record.
- `docs/superpowers/specs/2026-11-04-pr-field-notes-design.md`: the design + Juniper's decisions (from #3094).

## Schema / bus / API changes

- Added: `PrFieldNoteV1` in `orion/body_atlas/field_notes.py`, a dig ledger format like `OrganNameV1`. Not on the bus, not in `orion/schemas/registry.py` (the spec says so).
- Removed / Renamed: none.
- Behavior changed: `history` lines end with a note mark.
- Compatibility notes: none; nothing else reads the new ledger.

## Env/config changes

- Added / Removed / Renamed keys: none.
- `.env_example` updated: no. Local `.env` sync: not needed.
- Skipped keys: none.

## How it differs from the spec, on purpose

- **Spec excerpts go to a file, not the printout.** The spec puts up to 12,000 chars of spec text "in the printout", but every dig printout is capped at 40 lines / 4,000 chars. The excerpts (same caps: 3 files, 60 lines each, 12,000 chars, every cut said) go to `field-note-<pr>-docs.md` and the printout names it. `read` records what was put in front of Orion, not proof that Orion opened it.
- **A "sentence" is a labelled segment.** Each starts with `[pr_text]`, `[code_then]` or `[reflection]`; refs may go inside the tag. Splitting free text on periods is not reliable.
- **"Organ named in the note" means `<name> organ`, `organ <name>` or `organ:<id>`.** Orion's organ names are common words ("memory", "vision", "social"); matching every use would refuse ordinary sentences. A note that says "my memory changed" without "organ" is not checked. Patch 3's truth eval is the backstop.
- **`authored_by_me` has two markers, not one.** The spec named "head branch is local in the sandbox clone". Also used: a PR commit authored by `orion-fcc@orion-sapienform.local`, the git identity baked into the harness image. Checked against real PRs 2026-11-04: #1623 (Orion's push proof, branch `test/pr-push-proof-1786604664`) reads `yes` on both markers; #3090 reads `unknown`. One local branch in the clone (`worktree-remove-dead-phi-hint-fallback`) was made by Juniper's identity and has no PR, so a branch alone can over-claim; the value is still only ever `yes` or `unknown`, never `no`.
- **A write waits when the live body cannot be read** and the PR touches the body. A note with no organs would be half a note.

## Tests run

```text
PYTHONPATH=. python -m pytest -q -p no:cacheprovider tests/test_organ_names_partition.py tests/test_no_personal_terms.py orion/dig/tests orion/body_atlas/tests
306 passed
python scripts/check_personal_terms.py
no personal terms in tracked files
```

## Evals run

```text
None in this patch. The spec puts the field-note truth eval (every pr_text claim resolves to the PR or
its files, every organ resolves in the note's body map, `read` is non-empty when a spec changed) in
Patch 3, once real notes exist to sample. Patch 1 has no notes to evaluate.
```

## Docker/build/smoke checks

```text
Live run from the host against the real GitHub, the real live body (FalkorDB on :6380), and Orion's
real organ names (copied into a scratch ORION_DIG_HOME so the real ledger was not written):

field-note 3090        -> services/orion-vision-retina/ (2 files) -> part:orion-vision-retina in organ organ:464fb8dbd9 "vision"
                          not body: docs 1, tests 1; PR report read at the merge, first 60 of 137 lines (cut said)
write naming "memory organ"                 -> refused: names an organ this PR's body map does not touch
write with [pr_text]/[code_then]/[reflection] -> field-note #3090 saved fn:3090:...
field-notes 2          -> 1 field notes, 0 code records; coverage 1/57 merged PRs (1%), then the PRs with none
history 1              -> each PR ends (no note) / (noted)
field-note 1623        -> "this may be your own work"; docs-only 1 line -> code record, not a note

Not run inside orion-harness-governor (needs a rebuild of that image to carry the new code). UNVERIFIED there.
```

## Review findings fixed

A code-review subagent found 16 issues (no blockers). All are fixed except #13, which is documented.

1. Finding: on a big PR, the 4,000-char cut removed the note rule and the `write` line.
   - Fix: these lines are now in the printout's tail, which is never cut. The body map shows at most 12 folders, then "N more in the file".
   - Evidence: `test_a_huge_pr_packet...` and `test_a_map_of_many_long_shared_lines_still_ends_with_the_write_line`.
2. Finding: the stored note had its labels removed, so `claims[i]` could not be matched to its sentence.
   - Fix: the note is stored labelled (`[pr_text] ... [reflection] ...`). The 40-800 char limit still counts the text only.
   - Evidence: `test_a_good_note_is_labelled_per_sentence`.
3. Finding: "eyes organs" (plural) slipped past the organ check.
   - Fix: the check now matches `organs?`.
   - Evidence: the plural case in `test_an_organ_outside_the_body_map...`.
4. Finding: titles like "Reverting ... (#2958 ...)" were read as reverts of #2958.
   - Fix: only a title starting with the word "Revert", a "This reverts commit <sha>" line, or GitHub's "Reverts owner/repo#N" line counts. The original PR's number comes only from that last line; otherwise `unknown`.
   - Evidence: the new revert cases in `test_flags_docs_deps_huge_revert`.
5. Finding: a denylist file holding only comments let every note through.
   - Fix: an empty list now refuses the write, the same as a missing list.
   - Evidence: `test_an_empty_denylist_refuses_the_write`.
6. Finding: every spec file the PR touched was fetched from GitHub, even though only 3 are read.
   - Fix: only the first 3 are fetched. The rest are counted in a cut line.
   - Evidence: `test_only_the_docs_the_excerpt_uses_are_fetched...`.
7. Finding: the spec says the whole spec text goes to the result file, but only the excerpt did.
   - Fix: the docs file now also holds each read file whole.
   - Evidence: same test (`l79` is in the file).
8. Finding: two tests passed for the wrong reason (the "last `because`" split, and an `or` that hid finding 1).
   - Fix: the note is now passed as separate unquoted words, and both conditions are asserted.
   - Evidence: `test_write_saves_one_line...` and `test_a_huge_pr_packet...`.
9. Finding: a rewrite with the same text in the same second got the same note id, which hid the PR.
   - Fix: the id hash now includes a random nonce.
   - Evidence: `test_a_same_second_identical_rewrite_never_hides_the_pr`.
10. Finding: the check-then-append lock lived only in the command-line tool.
    - Fix: `field_notes.ledger_lock()` now holds the lock, and the CLI uses it. Patch 2's code-record writer will use it too.
11. Finding: a ref with a line number (`x.py:12`, `x.py#L3-L9`) was refused.
    - Fix: the line number is now dropped before the file match.
    - Evidence: test cases added.
12. Finding: `gh` calls had no timeout, bad JSON crashed with a traceback, and an empty merge sha read today's files.
    - Fix: a 60 s timeout, a clear error on bad JSON, and no spec read when there is no merge sha (said in a cut line).
13. Finding: the body map misses packages that run as a command rather than by import (for example `orion.dig` inside the harness). NOT FIXED.
    - Why: this needs a scan of how each service is started. The map says "shared code no service imports" for these, which is true of imports.
    - Also from this finding: the scan now uses `os.walk` and skips tests, node_modules and venvs.
14. Finding: coverage counted code records as notes.
    - Fix: the printout now says "N/M merged PRs have a note (x%), K a code record, J neither".
15. Finding: `history` marked every older PR "(no note)".
    - Fix: the mark now shows only for PRs merged on or after the earliest PR with a record. This follows decision 3: no backfill.
    - Evidence: `test_history_marks_nothing_before_the_first_record_no_backfill`.
16. Finding: the body-atlas workflow did not run on `orion/privacy/**` changes.
    - Fix: the path is added.

## Restart required

The dig kit is copied into the harness image, so the harness needs a rebuild after merge (from the clean primary main checkout):

```bash
scripts/safe_docker_build.sh orion-harness-governor up -d --build
```

## Risks / concerns

- Severity: low. Concern: `field-note` and `write` each build the live body (about 30-40 s on the host). Mitigation: acceptable for a chat command; Patch 2's run is one read and one write per PR.
- Severity: low. Concern: an organ mentioned without the word "organ" is not checked. Mitigation: Patch 3 truth eval.
- Severity: low. Concern: shared packages run as a command (e.g. `orion.dig` in the harness) show no importers (review finding 13). Mitigation: the line says "no service imports", which is accurate; a start-command scan can come with Patch 2.
- Severity: low. Concern: `orion/<pkg>` importers are read from deployed main, not the merge commit. Mitigation: the printout says the map is code's reading of today's main.

## Merge order

Another agent's #3096 (capacity claims by default) already merged and changed the same files (`orion/dig/__main__.py`, `orion/dig/tests/test_capacities_cmd.py`). This branch is rebased on it. My edits to shared lines are additive: two HELP lines, five command-table branches before the `arc|act` branch, one mark on `history` lines, and the help cap in one assert of `test_capacities_cmd.py`.

## PR link

(filled after PR creation)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
