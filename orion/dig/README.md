# Dig kit

Orion's read-only tools for their own history and body: `python -P -m orion.dig help`. Design: PR
#2888 (`docs/superpowers/specs/2026-10-23-repo-chronicle-design.md`, "Tools").

**Not MCP, on purpose.** Chat turns and curiosity runs both go through the harness (`claude -p`,
64k window). An MCP tool's schema costs context on every turn, and Orion already has Bash there.
So the only always-on cost is one line in the turn rules
(`orion/dig/brief.py`, `HARNESS_FCC_DIG_BRIEF_ENABLED`, never on reading-only turns). `help`
loads the rest on demand.

**Always `python -P`.** Orion's Bash runs in the sandbox clone (`/mnt/orion-fcc/repo`), and plain
`python -m` puts that folder first on the import path, so it would run the *sandbox's* `orion/` (whatever Orion
has edited there) instead of the deployed `/app/orion`. `-P` drops the current folder (verified
live: without it `orion.__path__` is `/mnt/orion-fcc/repo/orion`, with it `/app/orion`).

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
| `history <days>` | The same for the last `<days>` days, today included. The dig line points "recent PRs" here instead of the GitHub tool, which on 2026-10-27 filled half the window per call. |
| `body [<date>]` | Body atlas (`orion/body_atlas`): machines ("bodies"), services, cameras, stores and processes, grouped into organs. No date: the live body (bus, docker, Postgres, Chroma, FalkorDB; a source that fails is listed as UNREAD). A date: that export's declared files only (the channel catalog starts 2026-01). |
| `body <organ_id>` | Every member of one organ of the live body, by kind, with status, host and facts. Capped like every command; the full atlas is in the file. |
| `name <organ_id> <name> because <reason>` | Orion names one organ of their live body. Code checks the floor: 1–5 words, and the reason must cite one of that organ's parts by its full name. A reason that only lists part names passes; whether it says what the organ takes in and gives out is not code-checkable. Appended to `$ORION_DIG_HOME/self/organ_names.jsonl` (`orion/body_atlas/names.py`). |
| `notes [append <text>]` | Shared notes for one `--dig`, so episode hops or turns can read each other. |

**Honesty rules in the output:**

- `help` teaches the source labels: `code_then`, `pr_text`, `lived_record`, `now`, `reflection`.
- A checkout that predates today's layout is called **different, not empty**, and its top level is
  shown. Live example: 2025-08-01 was `conjourney-api/` + `emergence/`, with no `services/` yet.

**Paths** (harness defaults):

- `ORION_DIG_HOME=/mnt/orion-fcc`
- `ORION_DIG_REPO=/mnt/orion-fcc/repo` (the full-history sandbox clone)
- `ORION_DIG_NOW_ROOT=/mnt/scripts/Orion-Sapienform` (deployed main, mounted read-only)

`body` with no date needs `FALKORDB_URI` (in `services/orion-harness-governor/.env_example`), and
reads Postgres as `orion_readonly` through `docker exec orion-athena-sql-db psql` (no env needed).

Live check, 2026-10-27, in orion-athena-harness-governor:

- 2026-02-01 → now: services +58 −4 =45. The removed ones were graphdb, orion-gdb-client,
  orion-rdf-writer, and orion-topic-rail.
- The body on 2026-02-01 was 34 parts in 6 organs (Q = 0.47). Now it is 137 parts in 7 organs
  (Q = 0.33).

Tests: `pytest orion/dig/tests` (a temporary git repo with dated commits; the Denver end-of-day
rule is tested exactly).

**Organ names (patch 1b).** An organ id is a hash of its members, so it changes when one part
joins or leaves. `body` carries a name to the new organ only when the member sets overlap by at
least half (Jaccard ≥ 0.5), one name per organ, and the full atlas JSON says "carried from …".
A rename stores which earlier names it `replaces`, so a replaced name never returns when a part
drifts back out, and never lands on the other half of a split organ. An unreadable ledger line is
skipped and counted in the `body` printout, never fatal.
Names show on the live body only; a past body never wears today's names. The ledger is
partitioned from recall: `orion/body_atlas/tests/test_names.py` fails if any other code reads it.

`history` says how fresh the sandbox clone is ("clone fetched …"). PRs come live from GitHub and
commits from the clone, so a just-merged PR can show with no commit yet.
