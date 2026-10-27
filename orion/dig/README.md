# Dig kit

Orion's read-only tools for their own history and body: `python -m orion.dig help`. Design: PR
#2888 (`docs/superpowers/specs/2026-10-23-repo-chronicle-design.md`, "Tools").

**Not MCP, on purpose.** Chat turns and curiosity runs both go through the harness (`claude -p`,
64k window). An MCP tool's schema costs context on every turn, and Orion already has Bash there.
So the only always-on cost is one line in the turn rules
(`orion/dig/brief.py`, `HARNESS_FCC_DIG_BRIEF_ENABLED`, never on reading-only turns). `help`
loads the rest on demand.

**Context rules:**

- Every command prints at most 40 lines and 4,000 chars.
- The full result goes to `$ORION_DIG_HOME/dig/<dig>/<command>-<n>.json`, and the printout
  always ends with that path.
- A cut is always stated.

| Command | What |
|---|---|
| `open <date>` | Exports the last `main` commit of that Denver day to `past/<date>/` as plain files (`git archive`, no `.git`). At most 2 open; exports older than 24 h are swept. Removal only touches folders under `past/` that carry the dig marker. |
| `close <date>`, `list` | |
| `inventory <date\|now>` | Services, modules, channels, verbs, schemas, from `orion/self_inventory.py` (the same scanner self-study uses). |
| `compare <date> [<date\|now>]` | Added / removed / kept per category. |
| `history <start> <end>` | Merged PRs (`gh pr list --search merged:…`) and commits. PR bodies go to the file only. |
| `body [<date>]` | Body atlas (`orion/body_atlas`). No date: the live bus. A date: that export's catalog only (the catalog starts 2026-01). |
| `notes [append <text>]` | Shared notes for one `--dig`, so episode hops or turns can read each other. |

**Honesty rules in the output:**

- `help` teaches the source labels: `code_then`, `pr_text`, `lived_record`, `now`, `reflection`.
- A checkout that predates today's layout is called **different, not empty**, and its top level is
  shown. Live example: 2025-08-01 was `conjourney-api/` + `emergence/`, with no `services/` yet.

**Paths** (harness defaults):

- `ORION_DIG_HOME=/mnt/orion-fcc`
- `ORION_DIG_REPO=/mnt/orion-fcc/repo` (the full-history sandbox clone)
- `ORION_DIG_NOW_ROOT=/mnt/scripts/Orion-Sapienform` (deployed main, mounted read-only)

`body` with no date needs `FALKORDB_URI` (in `services/orion-harness-governor/.env_example`).

Live check, 2026-10-27, in orion-athena-harness-governor:

- 2026-02-01 → now: services +58 −4 =45. The removed ones were graphdb, orion-gdb-client,
  orion-rdf-writer, and orion-topic-rail.
- The body on 2026-02-01 was 34 parts in 6 organs (Q = 0.47). Now it is 137 parts in 7 organs
  (Q = 0.33).

Tests: `pytest orion/dig/tests` (a temporary git repo with dated commits; the Denver end-of-day
rule is tested exactly).
