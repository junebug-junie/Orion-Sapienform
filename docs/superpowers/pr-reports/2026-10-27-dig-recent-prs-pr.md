## Summary

- **New `history <days>`:** `python -P -m orion.dig history 2` lists merged PRs and commits for the
  last 2 Denver days, today included. Orion does not have to work out dates.
- **The dig line now steers recent-PR questions to the kit.** It adds: "For recent PRs run
  `python -P -m orion.dig history 2` (40 lines max), not the GitHub tool."

## Outcome moved

On 2026-10-27, turn `e0fdd52a…`, Juniper wrote "Had a few cool PRs come out. What do you think?".
Orion had the dig line, but read "PRs" as news, not history, and used the GitHub tool
(`mcp__github__list_pull_requests`):

- Each call returned every PR's full body, about 12,000 characters for 5 PRs.
- The same call ran 3 times in 25 s, then pages 2–4.
- The window filled to 56% twice, which caused **two compactions in one turn**. The turn took
  about 8 minutes and 50 motor steps.

The same question through the kit, live: `history 2` printed 16 merged PRs (#2903–#2919) in
18 lines and 1,762 characters. The PR bodies go to the result file only.

## Current architecture

The dig line (PR #2915) said "to look at your own history". `history` needed two dates.

## Architecture touched

`orion/dig/` only. There are no bus, schema, env, or service changes.

## Files changed

- `orion/dig/brief.py`: the line has one more sentence (268 of 300 chars).
- `orion/dig/__main__.py`: `history <days>` (`_last_days`), and its line in `help`.
- `orion/dig/README.md`: the new command.
- `orion/dig/tests/test_review_fixes.py`: two tests.

## Schema / bus / API changes

- Added: the CLI form `history <days>`.
- Behavior changed: the harness dig line is longer by one sentence.

## Env/config changes

None.

## Tests run

```text
pytest orion/dig/tests orion/harness/tests/test_stance_scope_hierarchy.py \
       orion/harness/tests/test_harness_prefix.py   -> 81 passed
```

## Evals run

None for this patch. Whether Orion now picks the kit is checked by the live turn below.

## Docker/build/smoke checks

```text
python -P -m orion.dig history 2 (host, real gh + sandbox clone)
  -> 2026-10-26 .. 2026-10-27: 16 merged PRs, 16 commits on main; 18 lines, 1762 chars
```

## Review findings fixed

- Finding (low): `str.isdigit()` accepts `²` (then `int()` raised an uncaught ValueError) and
  `٣` (read as 3 days).
  - Fix: ASCII digits only.
  - Evidence: both are in the bad-input list of `test_history_by_day_count_ends_today`.
- Finding (low): `history 1000000` raised an uncaught OverflowError before the clamp to the first
  day ran.
  - Fix: the day count is clamped before the `timedelta` is built.
  - Evidence: `_last_days("99999999999") == (FIRST_DAY, today)` in the same test.

## Restart required

```bash
scripts/safe_docker_build.sh orion-harness-governor up -d --build
```

## Risks / concerns

- Severity: medium.
  - Concern: a prompt line steers but does not enforce. Orion may still pick the GitHub tool.
  - Mitigation: the next live PR question shows it. The general fix is a harness cap on long tool
    results, which is a separate proposal.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2920

🤖 Generated with [Claude Code](https://claude.com/claude-code)
