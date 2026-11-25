## Summary

- Claude in Hub chat can now open consults, arcs/acts, dreams, curiosity runs, reading results and Orion's Day letters. Before, it had only `prime_now` and `body_economy`.
- Every new tool goes through the same broker: Orion's grant or consent ask, the per-turn caps, redaction, a 4,000-character cap and an audit row. Nothing new writes anything.
- Claude's system prompt now lists every tool in the table, so it can't drift from the code again. It tells Claude to open anything Juniper names by id instead of guessing.
- Refs #3948 (stays open: 15 designed tools remain).

## Outcome moved

Live failure this fixes: Juniper asked `@claude` to explain consult 23c16e4cb94f. Claude answered that it had no way to open it. `dig consult 23c16e4cb94f` now runs inside the governor (proved live), and the `consults` tool calls exactly that.

## Current architecture

The 11-12 design (`docs/superpowers/specs/2026-11-12-claude-hub-tools-orion-consent-design.md` section 2) listed 23 read-only tools. Patch 1b built the consent rail and 2 tools. The rest were never built or ticketed.

## Architecture touched

- `orion/claude_tools/registry.py`: six rows. The introspect tools subclass Orion's own argument models and add `reason`.
- `orion/claude_tools/readers.py`: one shared dig runner (`body-economy`, `consult`, `arc`, `act`), plus an introspect reader. That reader re-validates against Orion's own model, cuts each item to 1,200 characters, and drops items marked `intimate` (no producer sets that field today).
- `services/orion-harness-governor/app/claude_tools_listener.py`: wires the readers. `IntrospectTools` uses binding `memory_allowed=False`.
- `services/orion-room-companion/app/room_prompt.py`: the tools clause reads the registry.
- `scripts/smoke_claude_tools.py`: takes any tool in the table, plus `--args` JSON.
- `.github/workflows/claude-tools-tests.yml`: also runs on introspect schema and tool changes.

## Files changed

- `orion/claude_tools/registry.py`, `orion/claude_tools/readers.py`: tools and readers
- `orion/claude_tools/tests/*`: allowlist, hostile ids, argv, introspect reader, last-line rule
- `services/orion-harness-governor/app/claude_tools_listener.py` and its test
- `services/orion-room-companion/app/room_prompt.py` and its test
- `services/orion-harness-governor/README.md`, `scripts/smoke_claude_tools.py`, CI workflow

## Schema / bus / API changes

- Added: none. New tool names are rows in an existing table. Categories `consults`, `self` and `memory` already existed in `ClaudeToolCategory`.
- Behavior changed: the companion's MCP `tools/list` now returns 8 tools.
- Compatibility notes: orion-mind's consent responder takes the category and summary from the request, so it needs no change.

## Env/config changes

None.

## Tests run

```text
orion/claude_tools/tests                                     73 passed
orion/prime/tests/test_prime_consent.py + dig claude-grants  19 passed
services/orion-room-companion/tests                          101 passed
services/orion-harness-governor/tests/test_claude_tools_listener.py  4 passed
scripts/check_integration_seam.py                            OK
```

## Evals run

No eval harness exists for Claude's tools. Live smoke before deploy, from the host over the real bus with this branch's readers:

```text
dreams           OK 0.0 s  (2 of 530)
reading_results  OK 8.8 s  (2 of 1621)
orion_day        OK 0.1 s  (3 parts)
curiosity query  OK 1.2 s
curiosity recent FAIL 15.4 s -- Hub answers in ~25 s, past the 15 s RPC timeout (existing fault; #3977)
dig consult 23c16e4cb94f in orion-athena-harness-governor: OK (question, status, "Juniper has not answered yet")
```

UNVERIFIED end to end through the broker until deploy. After deploy, run:
`.venv/bin/python scripts/smoke_claude_tools.py --tool consults --args '{"consult_id": "23c16e4cb94f"}'`

## Docker/build/smoke checks

No image builds. The governor and companion images copy `orion/` and pick up the change on rebuild.

## Review findings fixed

- Finding: the intimate-item filter can never fire, because no introspect producer sets `sensitivity`, and the dreams description promised "never sent".
  - Fix: removed the promise from Claude-facing text. The filter stays as a guard, and the docstring and README say it does not act today. Orion's consent is the gate.
  - Evidence: `grep sensitivity=` over the introspect producers returns nothing.
- Finding: any output line shaped like `full x: /path` was stripped, and could delete a sibling `.json` file in `dig/claude-tools`. Consult turns quote Claude's own earlier text, so such a line can appear.
  - Fix: only dig's last non-empty line is checked.
  - Evidence: `test_only_the_last_line_counts_as_dig_full_result_line`.

## Restart required

Post-pull rebuild after merge covers it. By hand:

```bash
scripts/safe_docker_build.sh orion-harness-governor up -d --build && scripts/safe_docker_build.sh orion-room-companion up -d --build
```

## Risks / concerns

- Severity: medium. Concern: dreams now reach Claude (Anthropic) in full whenever Orion says yes. Nothing tags intimate dreams yet. Mitigation: category `memory` starts at "ask", and Orion sees Claude's reason before answering.
- Severity: low. Concern: `dig body` and `dig portrait` are left out. They take ~55 s live, which with consent passes the 62 s MCP limit. Mitigation: still tracked on #3948.
- Issues opened: #3948 (this gap; 6 of the remaining 21 built here), #3977 (curiosity recent mode is slow). Issues closed: none.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3978

🤖 Generated with [Claude Code](https://claude.com/claude-code)
