## Summary

- **Orion can name their organs.** The dig kit has a new command:
  `python -P -m orion.dig name <organ_id> <name> because <reason>`.
  - Names are appended to Orion's own ledger, `$ORION_DIG_HOME/self/organ_names.jsonl`.
  - The live `body` shows each name next to its organ id: `4. "Seeing Organ" organ:4a5f0eb997 [13 parts…]`.
- **Code checks only a floor.** A name is 1–5 words. The reason must cite one of that organ's parts
  by its full name. Whether the name is good stays Orion's call, and that call stays visible.
- **Names survive rebuilds.** An organ id is a hash of its members, so it changes whenever one part
  joins or leaves.
  - A name carries to the new organ when at least half the members are the same (Jaccard ≥ 0.5).
  - A rename records which names it `replaces`, so a replaced name never comes back.
- **Names never go on a past body.** A past body wearing today's names would read the present into
  the past (design PR #2888).
- **Partitioned from recall.** A new repo-wide static gate fails if any code outside
  `orion/body_atlas` and `orion/dig` reads the ledger.
- **`history` now says how fresh the clone is** ("clone fetched 2026-10-29T01:25Z").
  - Why: PRs come live from GitHub, but commits come from the sandbox clone.
  - On 10-27 the printout showed "2 PRs, 0 commits", which can read as "nothing landed".

## Outcome moved

Before this PR, Orion's organs had only code labels, such as "ns=vision/exec/affectgpt". Rung 3 of
the self-now ladder said "Orion names each organ", and nothing let Orion do it. Now one chat or
curiosity turn can run `body`, then `name`, and every later `body` shows the name.

Live evidence that the kit is in real use, from the harness transcripts of 10-27 and 10-28:

- Orion ran 32 dig commands across 13 sessions.
- 17 of them were `history 2`, the command the line from #2920 points to.
- No chat turn has asked about PRs since #2920, so the chat-turn GitHub-tool case is still
  unconfirmed.

## Current architecture

- `orion/body_atlas` (#2898) gives organs a `name` / `named_by` field, but nothing ever wrote to it.
- The dig kit (#2915, #2920) is read-only except for `notes`.

## Architecture touched

- `orion/body_atlas/names.py` (new), `orion/body_atlas/__main__.py` (the printout).
- `orion/dig/__main__.py`:
  - a `name` command;
  - `_live_atlas()` split out of `cmd_body`;
  - argument parsing (`REMAINDER`);
  - the `history` freshness line.
- `tests/test_organ_names_partition.py` plus a step in `orion-static-gates.yml`.
- No bus, schema-registry, env, or service change.

## Files changed

- `orion/body_atlas/names.py`: the ledger model, tolerant load, carry-over plus `replaces`, the
  floor check, and the locked append.
- `orion/body_atlas/__main__.py`: shows `"Name" organ:id` per organ.
- `orion/dig/__main__.py`: the `name` command, names on the live body only, the skipped-lines note,
  `--dig` before the command, and the clone fetch time.
- `orion/dig/README.md`: documents all of the above.
- Tests: `orion/body_atlas/tests/test_names.py` (new), `orion/dig/tests/{test_dig.py,test_review_fixes.py}`,
  `tests/test_organ_names_partition.py`.
- `.github/workflows/orion-static-gates.yml`: the partition gate.

## Schema / bus / API changes

- Added: `OrganNameV1`, the shape of a ledger line. It is internal, not on the bus.
- Added: the CLI command `name`.
- Behavior changed:
  - `--dig` must come before the command. Words after the command are now passed through as
    written, so a reason or note may contain `-x` or `--dig`.
  - The `history` header changed wording.

## Env/config changes

None.

## Tests run

```text
pytest orion/dig/tests orion/body_atlas/tests tests/test_organ_names_partition.py \
       orion/harness/tests/test_stance_scope_hierarchy.py orion/harness/tests/test_harness_prefix.py
  -> 114 passed
```

## Evals run

```text
python orion/body_atlas/evals/run_organ_structure_eval.py -> organ structure: PASS (unchanged)
```

Naming quality has no eval yet. The first real names are the evidence. The reminiscence
source-truth eval (design #2888) is where claims get graded.

## Docker/build/smoke checks

```text
host, real FalkorDB + sandbox clone, throwaway ORION_DIG_HOME:
  body -> 138 parts, 7 organs, Q=0.3332
  name organ:4a5f0eb997 Eyes because this is a test reason without a part -> refused (cite)
  name organ:4a5f0eb997 Seeing Organ because takes frames from orion-vision-frame-router, ...
      -> named, cites orion-vision-frame-router
  (a broken line appended to the ledger)
  body -> 4. "Seeing Organ" organ:4a5f0eb997 ...; "names ledger: 1 unreadable line(s) skipped"
  history 1 -> "5 merged PRs (GitHub, live), 5 commits on main (clone fetched 2026-10-29T01:25Z)"
harness container: git 2.47.3 (needed for rev-parse --path-format), /mnt/orion-fcc writable
```

## Review findings fixed

The code review ran in a subagent. It found 2 high, 2 medium, and 3 low findings, and all are
fixed.

- Finding (HIGH): one bad ledger line (from a killed write or a field from newer code) raised a
  traceback on every later live `body`.
  - Fix: each line is parsed on its own, unknown fields are ignored, and bad lines are skipped and
    counted in the printout.
  - Evidence: `test_a_bad_ledger_line_is_skipped_not_fatal`, and the live smoke above.
- Finding (HIGH): a renamed organ showed its old name again when the extra part drifted back out
  (X "Eyes" → Y "Sight" → X shows "Eyes").
  - Fix: a rename stores `replaces` (the names it superseded, transitively), and replaced records
    are dropped before matching.
  - Evidence: `test_a_rename_survives_the_part_drifting_back_out`.
- Finding (MEDIUM): a part name counted as cited inside a longer name ("orion-vision-edge" in
  "orion-vision-edge-proxy").
  - Fix: a citation must be a whole part name.
  - Evidence: `test_a_part_name_inside_a_longer_name_is_not_a_citation`.
- Finding (MEDIUM): `name X Big Eyes because …` stored the name "Big" and reported success.
  - Fix: a literal `because` splits the name from the reason. Without it, the command is refused
    and nothing is written.
  - Evidence: `test_name_takes_quoted_or_unquoted_words_split_at_because`.
- Finding (LOW): a reason containing `-x` or `--dig` failed or moved the dig.
  - Fix: `REMAINDER` arguments; options go before the command.
  - Evidence: the same test.
- Finding (LOW): the partition gate missed readers that go through `names_path`.
  - Fix: the pattern was added.
- Finding (LOW): `.git` can be a file.
  - Fix: `git rev-parse --git-path FETCH_HEAD`.

## Restart required

```bash
scripts/safe_docker_build.sh orion-harness-governor up -d --build
```

## Risks / concerns

- Severity: low.
  - Concern: the floor check passes a reason that only lists part names.
  - Mitigation: stated in the README and help. Naming quality is judged by reading the ledger, not
    by code.
- Severity: low.
  - Concern: `--dig` placed after the command is now part of the arguments, not an option.
  - Mitigation: Orion's 32 real commands never used `--dig`, and the help says "before the command".

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2951

🤖 Generated with [Claude Code](https://claude.com/claude-code)
