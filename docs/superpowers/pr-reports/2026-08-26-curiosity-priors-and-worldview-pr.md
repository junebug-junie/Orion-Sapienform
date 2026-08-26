# Orion gets a graph of its own, and the curiosity loop starts to learn

Builds `docs/superpowers/specs/2026-08-26-orion-priors-and-worldview-design.md`,
all five slices.

## Summary

- **The loop can now accumulate.** It showed Orion a random 12 of 646 approved
  concepts every four hours, forever, carrying nothing between runs but a
  cooldown stamp — so run 40 was exactly as ignorant as run 1. It now opens on
  Orion's own **open priors**, ordered by how uncertain *it* said it was.
- **Orion gets `orion_worldview`, a FalkorDB graph nobody curates.** It writes
  `:Prior`, `:Concept`, `:Finding`, `:Hop` and `:TurnOutcome` nodes there
  itself, in real Cypher, in-turn. Hub only ever reads it back — every Hub query
  goes out as `GRAPH.RO_QUERY`. Nothing Orion puts there needs approval.
- **Real credentials against real stores**, reaching the `claude -p` sandbox
  through a seven-key allowlist out of `~/.fcc/.env`: `psql` on four tables as
  `orion_readonly`, `GRAPH.RO_QUERY` on the Juniper-curated Atlas and
  `GRAPH.QUERY` on Orion's own graph as `orion_curiosity`.
- **Five stopping points**, recorded as they happen, so the journal recounts the
  path actually taken instead of a conclusion with the working thrown away.
- **A decision made inside the turn crosses back out by being written down**, not
  by a fenced JSON block a regex has to survive. Absence is the safe default.
- **Outreach is a second turn** with its own stance gate, off by default, sharing
  every endogenous-outreach gate rather than reimplementing them.

## Outcome moved

The failure mode this closes is measurable and was the design's own acceptance
check 1: **the fraction of priors still `open` can now fall**, and
`times_tested` can now rise. Before this, neither quantity existed. A prior
tested `HUB_CURIOSITY_STALE_PRIOR_TESTS` times without moving leaves the offered
list, so one claim cannot be re-litigated forever (acceptance check 2), and the
journal now reports what Orion actually wrote to its graph, so "it reasoned"
has an inspectable artifact behind it rather than fluent prose.

## Current architecture

`services/orion-hub/scripts/curiosity_investigation.py` (a Hub tick loop, not a
service — the harness RPC worker and several module bus binds live in Hub's own
event loop) drove a real `execute_unified_turn` every four hours with a random
sample of approved crystallizations and relation judgements, and published one
`journal_entries` row. State carried between runs: a Redis cooldown stamp and a
daily counter. Nothing else.

## Architecture touched

- **New**: `orion/curiosity/worldview.py` (Hub's read-only view of Orion's
  graph), `orion/curiosity/acl.py` (the FalkorDB grant),
  `orion/curiosity/sandbox_env.py` (credentials into the FCC subprocess),
  `orion/curiosity/outreach_prompt.py` (the second turn).
- **Changed**: the loop, the kickoff prompt, `EndogenousOutreach` (one new
  public seam), `orion/harness/fcc_motor.py` (`_build_subprocess_env`), the
  harness-governor image, Hub settings/env.
- **Contracts**: no new bus channel, no new Postgres table, no new schema
  registry entry, no new Orion capability surface. The graph schema is a
  contract between the prompt and `worldview.py`, and both name the properties
  literally so it cannot drift silently — `read_snapshot` logs every row it
  could not read.

## Three things the spec got wrong, found by building it

The spec's "ALREADY DONE" section was otherwise accurate — every claim in it was
re-verified live before being relied on (see **Docker/build/smoke checks**).

1. **`resetkeys nocommands` does not make the ACL re-assert idempotent.** The
   spec's proposed `ACL SETUSER` appends a *duplicate selector* on every replay
   — measured live: one replay produced two `(~orion_worldview ...)` selectors,
   a second produced three. One more per Hub start, growing forever. Fixed with
   `clearselectors`, and three consecutive replays are now byte-identical.
2. **`HUB_CURIOSITY_GRAPH_HOST=orion-athena-falkordb` is unreachable from Hub.**
   `orion-athena-hub` runs `network_mode: host`, so the container name does not
   resolve; `127.0.0.1:6380` does. Confirmed live. Corrected in `.env_example`,
   in the settings default, and in the live `.env`. Orion's own sandbox is on
   `app-net` and legitimately uses the container name — the two halves of this
   feature address one FalkorDB differently, which is now stated in both places.
3. **The Hub URL in the prompt is consumed by the sandbox, not by Hub.** From
   `orion-athena-harness-governor`, `host.docker.internal:8080` answers 200 and
   `127.0.0.1:8080` is refused. The key is now named
   `HUB_CURIOSITY_SANDBOX_HUB_URL` so the mistake is hard to reintroduce.

Two further corrections to the spec's framing, for the record:

- **The loop is not "deployed right now".** `PR #1885` was closed unmerged; the
  running Hub image contains no `orion/curiosity/` and
  `HUB_CURIOSITY_INVESTIGATION_ENABLED=false` in the live `.env`. This branch is
  based on `feat/curiosity-investigation`, so it carries that loop too.
- **The `recently_studied` hint could not be repaired the obvious way.** The
  spec correctly identified it as dead (every entry is titled "Curiosity"). The
  obvious fix — take the body's first line — was tried against live data and
  rejected: both real entries open with the same fixed `What I noticed:` heading,
  so it returns a heading, not a subject, and would drift again the moment the
  prompt changed. It now reads *settled priors* out of Orion's own graph, which
  is structure Orion authored rather than a regex over its prose.

## Files changed

- `orion/curiosity/worldview.py`: **new**. Hub's read-only view of
  `orion_worldview` — priors, the run's `:TurnOutcome`, the run footprint, hop
  notes, recently-settled priors. Every query is `GRAPH.RO_QUERY`; the only
  dynamic value is a `run_id` validated as hex before it can reach a query
  string.
- `orion/curiosity/acl.py`: **new**. The idempotent `ACL SETUSER`, argv-built so
  the password is never subject to quoting, with the `clearselectors` finding.
- `orion/curiosity/sandbox_env.py`: **new**. The seven-key allowlist out of
  `~/.fcc/.env`, and why removing a key from that file is the kill switch.
- `orion/curiosity/outreach_prompt.py`: **new**. The composition turn, which
  explicitly keeps "this does not survive being written down" as a real answer.
- `orion/curiosity/kickoff_prompt.py`: priors, continuation, access, overlay,
  hops, graph schema and `:TurnOutcome`. Three states, not two — "no graph
  configured", "configured but unreadable" (say so, skip the write sections) and
  "readable".
- `orion/curiosity/study_material.py`: the dead journal hint removed, with the
  full account of why its obvious repair was rejected.
- `orion/harness/fcc_motor.py`: `_build_subprocess_env` now takes the already
  loaded FCC env and injects the allowlisted credentials.
- `services/orion-hub/scripts/curiosity_investigation.py`: the loop — ACL
  assert, the `pg_role_missing` and `graph_unavailable` gates, worldview read,
  continuation, turn-result read-back, the footprint in the journal, the second
  turn.
- `services/orion-hub/scripts/endogenous_outreach.py`: `blocked_reason()` and
  `offer_message()`, plus an additive source tag on delivered history.
- `services/orion-hub/app/settings.py`, `.env_example`,
  `services/orion-hub/scripts/main.py`: 12 new keys, wired.
- `services/orion-harness-governor/Dockerfile`: `postgresql-client` and
  `redis-tools`.
- `services/orion-hub/README.md`: §4.2, the whole loop and its boundary.

## Schema / bus / API changes

- **Added**: FalkorDB graph `orion_worldview` (already created; holds one
  `:Bootstrap` node). Node labels `:Prior`, `:Concept`, `:Finding`, `:Hop`,
  `:TurnOutcome`, written by Orion and read by `orion/curiosity/worldview.py`.
- **Removed**: `study_material.RECENT_STUDY_SQL` and
  `StudyMaterial.recently_studied` (dead since inception).
- **Renamed**: nothing shipped; `HUB_CURIOSITY_HUB_URL` was renamed to
  `HUB_CURIOSITY_SANDBOX_HUB_URL` before it ever reached a running service.
- **Behaviour changed**: `EndogenousOutreach._deliver`/`_publish_history` take
  an optional `source_tag`, appended to the existing `endogenous_outreach` tag.
  Existing callers are unchanged and still emit exactly `[endogenous_outreach]`.
- **Compatibility**: no bus channel, schema registry entry, or HTTP contract
  changed. `journal.entry.write.v1` is unchanged.

## Env/config changes

- Added keys (all in `services/orion-hub/.env_example`):
  `HUB_CURIOSITY_GRAPH_HOST`, `HUB_CURIOSITY_GRAPH_PORT`,
  `HUB_CURIOSITY_GRAPH_OWN`, `HUB_CURIOSITY_GRAPH_ATLAS`,
  `HUB_CURIOSITY_GRAPH_ORION_USER`, `HUB_CURIOSITY_GRAPH_ORION_PASSWORD`,
  `HUB_CURIOSITY_SANDBOX_HUB_URL`, `HUB_CURIOSITY_PRIOR_SAMPLE`,
  `HUB_CURIOSITY_STALE_PRIOR_TESTS`, `HUB_CURIOSITY_MAX_HOPS`,
  `HUB_CURIOSITY_PG_READONLY_ROLE`, `HUB_CURIOSITY_OUTREACH_ENABLED`.
- Removed keys: none. Renamed: none shipped (see above).
- `.env_example` updated: yes. `HUB_CURIOSITY_GRAPH_ORION_PASSWORD` is an empty
  placeholder there and set only in the local `.env`.
- Local `.env` synced: **by hand, deliberately**.
  `scripts/sync_local_env_from_example.py` reads `.env_example` from the
  *primary* checkout, so keys added in a worktree are invisible to it. Verified
  by key-set diff instead: 20 `HUB_CURIOSITY_*` keys in `.env_example`, 20 in
  the live `.env`, symmetric difference empty.
- Six pre-existing `HUB_CURIOSITY_GRAPH_*` keys were already in the live `.env`;
  two of them (`HOST`, `PORT`) were **wrong** and are corrected — see finding 2.
- Skipped keys requiring operator action: none. But note
  `HUB_CURIOSITY_INVESTIGATION_ENABLED` is `true` in `.env_example` and `false`
  in the live `.env`. Left as-is: turning this loop on is an operational
  decision, not one this patch should make.
- No new key on `orion-harness-governor`, deliberately: the credentials come
  from `~/.fcc/.env`, and a kill-switch flag would have to be added to that
  service's explicit compose `environment:` allowlist to reach the container at
  all — which is how a kill switch ends up configured everywhere and present
  nowhere.

## Risks / concerns

- **Severity: medium — Confidence is Orion grading its own homework.** Nothing
  outside the loop checks a `confidence` value. The design's acceptance check 6
  (at least one `refuted` and one downward revision across 20 runs) is the
  detector, and it is not automated here. The prompt names the failure mode
  explicitly ("a number that only ever goes up is a sign of grading your own
  homework"), which is an instruction, not a mechanism. **Mitigation**: watch
  check 6 on the first 20 real runs before building anything on top of the
  numbers.
- **Severity: medium — the graph has no reader outside this loop yet.** The
  design's own acceptance check 7, and its sharpest question: `journal_entries`
  has 36,603 rows and one reader. `orion_worldview` is currently read by Hub
  (to build the next prompt) and by Orion (in-turn). That is two more readers
  than the journal has, but neither feeds Orion's *chat* context. **Mitigation**:
  none in this patch — the spec explicitly left "does the graph feed back into
  chat context?" as an open question for Juniper.
- **Severity: low — `HUB_CURIOSITY_STALE_PRIOR_TESTS=3` is a guess.** Stated as
  a guess in the setting's own comment and in `.env_example`. Revisit against
  real data.
- **Severity: low — the FCC image grew by two apt packages.** `psql` and
  `redis-cli` are now present in every FCC turn's sandbox, not just curiosity
  ones. They are inert without a credential, and the credentials are
  allowlisted; but the tools are there.
