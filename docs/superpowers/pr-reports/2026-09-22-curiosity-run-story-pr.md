# PR report — Curiosity tab redesign, Patch 1: read model + endpoints + page rewrite

## Summary

- New `orion/curiosity/run_story.py`: a pure cross-store join that assembles
  one curiosity run into a clock-ordered story — lifecycle transitions,
  Orion's own graph nodes (role choice, hops, findings, revisions, outcome,
  self-writes), the journal, the outreach decision, any reply, hop readings,
  and self-sense scores.
- `orion/curiosity/atlas.py` drops the unbounded `runs[]`/growth read and adds
  windowed, run-id-scoped graph readers (`run_nodes_cypher`,
  `run_ids_since_cypher`, `prior_claims_cypher`), parameterized through
  FalkorDB's `CYPHER k=v` prefix rather than string interpolation; hops now
  select `written_at` and sort by clock, fixing the retried-run
  1,1,2,2-interleaved bug the design doc named.
- New `services/orion-hub/scripts/curiosity_run_store.py` and two new
  endpoints, `GET /curiosity/api/runs` and `GET /curiosity/api/run/{run_id}`,
  wired into `curiosity_routes.py`; the dead park/pin self-question routes
  were removed (nothing called them).
- `curiosity_atlas.html` rewritten: budget tiles per line, a 14-day sittings
  strip (chip per run, colour = line, glyph = outcome), a run-story timeline
  with a relative clock, the self card, a live-priors list with a closed
  toggle/filter/time-axis sparkline, contractor briefs behind a disclosure.
  Hash-routed (`#run=<id>`), polls every 60s, re-renders a section only when
  its payload's content hash changed, and an open story survives a poll.
- A sibling arc (PR #2288) root-caused that since 2026-09-14 curiosity runs
  go through a resource-admission path that writes every transition to
  `durable_resource_events`/`durable_admission_runs` and only copies the
  terminal `completed` row into the older `substrate_durable_run_state`
  bridge table (mislabelling its workflow). `run_story.py` reads the
  admission path when a run has rows there and falls back to the bridge
  table only for pre-09-14 runs.
- A second sibling, PR #2290 ("record every curiosity outreach decision;
  stamp Juniper's reply"), is **open, not yet merged**. It will add the
  reach-out truth write side: every curiosity outreach decision — including a
  pre-check block that today only logs a line and writes nothing — gets a row
  keyed by the deterministic `uuid5(NAMESPACE_URL, f"curiosity_outreach:{run_id}")`,
  and Juniper's reply gets stamped `client_meta.in_reply_to` on the next chat
  turn in that session. This patch's read side is already written against
  that exact contract (`result_json->>'source' = 'curiosity_outreach'`,
  `result_json.run_id`, the `in_reply_to` stamp) so no read-side change is
  needed once #2290 merges; until then it degrades to `not_recorded` for
  every run, which is what the live verification below shows.

## Outcome moved

Before: the tab could not answer "what happened in this run", "did Orion
reach me and did I answer", or "what does each of the three curiosity lines
even mean" — it rendered an unbounded, unordered graph inventory that
re-rewrote itself every 60 seconds and lost every open state doing it. After:
a bounded 14-day strip opens a real per-run story with a clock, the three
lines are named in plain words everywhere they appear, and the reach-out
tally is now honest — "N wanted, M sent, here is what blocked the rest" —
instead of showing only the intent half of the story.

## Current architecture

Before this patch: `curiosity_atlas.html` (965 lines) fetched one JSON blob
(`GET /curiosity/api/atlas`) every 60 seconds and rewrote eight sections from
`innerHTML`, including an unbounded runs ledger, an unbounded stacked
node-count "growth" chart, and a 2000-row "every prior" table. The read model
(`orion/curiosity/atlas.py`) assembled `runs[]` from five unbounded Cypher
reads (hard `LIMIT`s of 2000–5000 rows) with hops sorted by `n` alone, so a
retried run rendered its two attempts' hop notes interleaved. Three curiosity
lines (`investigate`, `self_inquiry`, `self_sense_eval`) shared one ledger
with no plain-language distinction, and a fourth workflow
(`self_study.reflect`) was structurally able to leak into it. Reach-outs
showed `:TurnOutcome.reach_out` (Orion's intent) with no link to whether a
message was ever delivered.

## Architecture touched

- `orion/curiosity/run_story.py` (new)
- `orion/curiosity/atlas.py` (modified: run-node readers added, `runs[]`
  and growth reads removed, hop `written_at` fix, `last_tested_at_ms` and
  `written_at` added to trajectory points)
- `services/orion-hub/scripts/curiosity_run_store.py` (new)
- `services/orion-hub/scripts/curiosity_routes.py` (two new GET routes;
  schedule endpoint reports all three lines; park/pin routes removed)
- `services/orion-hub/templates/curiosity_atlas.html` (full rewrite)
- `orion/curiosity/README.md`, `services/orion-hub/README.md` (docs)

## Files changed

- `orion/curiosity/run_story.py`: new — the cross-store join, dataclasses,
  `build_stories`/`summaries`/`reach_out_totals`, payload builders.
- `orion/curiosity/atlas.py`: run-node Cypher readers bounded by window or
  run-id list; `assemble_runs`/`AtlasRun`/growth Cypher removed; hop clock
  fix; trajectory points gain `written_at`.
- `services/orion-hub/scripts/curiosity_run_store.py`: new — SQL readers for
  the bridge lifecycle table, the admission path, journals, outreach
  decisions, chat replies, hop readings, self-sense scores, and the
  orchestration that turns them into the two endpoints' payloads.
- `services/orion-hub/scripts/curiosity_routes.py`: `GET /curiosity/api/runs`,
  `GET /curiosity/api/run/{run_id}` added; `_read_schedule` now reports all
  three lines' budgets from the same Redis keys the loop writes; dead
  `POST /curiosity/api/self-questions/{id}/pin|park` routes removed (grepped
  the whole repo — no template, no JS, no other test called them).
  `_wrote_on` replaced by `_runs_on_local_date` (per-line, keyed on
  `started_at`/`finished_at` from the new run summaries rather than
  `total_added`, which no longer exists).
- `services/orion-hub/templates/curiosity_atlas.html`: full rewrite.
- `tests/test_curiosity_run_story.py` (new), `tests/test_curiosity_atlas.py`
  (updated for the new run-node readers), `tests/test_curiosity_atlas_template.py`
  (rewritten against the new page),
  `services/orion-hub/tests/test_curiosity_routes_runs.py` (new).
- `orion/curiosity/README.md`, `services/orion-hub/README.md`: pin/park
  route removal documented; new §4.2.3 Curiosity tab section.

## Schema / bus / API changes

- Added: `GET /curiosity/api/runs?days=14&line=all|investigate|self_inquiry|self_sense_eval`,
  `GET /curiosity/api/run/{run_id}`.
- Removed: `POST /curiosity/api/self-questions/{id}/park`,
  `POST /curiosity/api/self-questions/{id}/pin` (dead — confirmed no caller
  anywhere in the repo). `GET /curiosity/api/atlas`'s `runs[]` and growth
  fields.
- Behavior changed: `GET /curiosity/api/atlas` no longer includes `runs[]` or
  growth data (the two new endpoints own it); its `priors[]` gains
  `last_tested_at_ms` and trajectory points gain `written_at`; the `schedule`
  object now carries a `lines` map for all three lines in addition to its
  existing top-level (investigate-only) fields, kept for back-compat.
- Compatibility notes: no bus channel, schema registry, or `.env_example`
  change (verified: `git diff main..HEAD -- orion/bus/channels.yaml
  orion/schemas/registry.py .env_example 'services/*/.env_example'` is
  empty). Read-only throughout — Hub still never writes to Orion's graph.

## Env/config changes

None. No `.env_example` touched; `scripts/check_env_template_parity.py`
reports the same 23 pre-existing, unrelated warnings this branch did not
introduce (services missing newer template keys entirely — not a regression
from this patch) and PASS overall.

## Tests run

```text
$ python -m pytest tests/test_curiosity_atlas.py tests/test_curiosity_atlas_template.py \
    tests/test_curiosity_atlas_peer_briefs.py tests/test_curiosity_run_story.py -q
90 passed in 2.27s

$ python -m pytest services/orion-hub/tests/test_curiosity_self_panel_route.py \
    services/orion-hub/tests/test_curiosity_routes_runs.py -q
19 passed in 1.15s

$ python -m pytest services/orion-hub/tests/test_curiosity_self_question_persistence.py -q
9 passed in 1.03s   # confirms the removed pin/park HTTP routes did not
                     # break the still-live PARK_SQL/PIN_SQL column helpers

$ python scripts/check_env_template_parity.py
env template parity: PASS (88 service(s) compared)
```

`scripts/check_schema_registry.py` and `scripts/check_bus_channels.py` do not
exist in this repo (checked `ls scripts/`; nearest equivalents are
`check_inner_state_registry.py`, `check_journal_dispatch_registry.py`,
`check_substrate_projection_schema_drift.py`, none of which this patch's
surface touches) — matches the design doc's own non-goal ("Not touched:
`orion/bus/channels.yaml`, `orion/schemas/registry.py`, any `.env_example`").

Root and Hub curiosity test suites were run as two separate `pytest`
invocations, matching CI (`.github/workflows/orion-reading-tests.yml`):
running them together fails collection because the repo has two same-named
`scripts` packages (`scripts/` at root and `services/orion-hub/scripts/`)
that resolve differently depending on which one reaches `sys.path` first —
a pre-existing repo characteristic, not something this branch caused.

## Evals run

No eval harness exists for `orion/curiosity/` or the Hub curiosity surface
(`orion/curiosity/evals/` and `services/orion-hub/evals/test_curiosity*`
do not exist). Follow-up: none proposed here — this patch is read-only
rendering of existing data, and its correctness is covered by the join
tests against fixtures that mirror every store's real row shape plus the
live-payload verification below, not by a quality eval.

## Docker/build/smoke checks

Not run — no Docker compose, dependency, port, health-check, or worker
change (design non-goal). Per CLAUDE.md §8, Docker is only required when
runtime behavior at boot changes; this patch is a read model, two GET
routes, and a template.

## Live payload verification (runtime evidence)

Ran the read model directly against the live stores (Postgres via
`localhost:55432`, FalkorDB via `localhost:6380`, no Hub process, no writes)
from `services/orion-hub` with both packages on `sys.path`:

```python
runs = await read_runs_payload(pool=pool, reader=reader, days=14, line="all")
story = await read_run_payload(pool=pool, reader=reader, run_id="446ddd7165d5")
story2 = await read_run_payload(pool=pool, reader=reader, run_id="9a5dfe992452")
```

`GET /curiosity/api/runs?days=14` (137 runs in the live 14-day window):

```json
{
 "available": true, "window_days": 14,
 "stores": {"postgres": "ok", "graph": "ok"},
 "reach_outs": {"wanted": 8, "sent": 0, "blocked_by": {},
                "top_block_reason": null, "not_recorded": 8},
 "totals": {"investigate": 75, "self_inquiry": 50, "self_sense_eval": 12}
}
```

Acceptance check 1 (design doc): every run carries a `plain_line_label`
from exactly `{World question, Self question, Self-sense check}` — verified,
`Counter({'World question': 75, 'Self question': 50, 'Self-sense check': 12})`
sums to all 137 rows. `reflect rows live: 0` (queried
`substrate_durable_run_state` directly) and zero `self_study.reflect` rows
leaked through the admission path either — none of the 137 summaries carry
that workflow.

Acceptance check 2: `GET /curiosity/api/run/446ddd7165d5` — first timeline
item is `accepted` (the run's admission), hops ordered `[1, 2]` by
`written_at`, last items are `lease_released`/`completed`. Matches the
coordinator's example exactly: accepted `08:25:27.855Z`, admitted
`+423:39` (7h03m39s wait, `wait_sec: 25419.0`) = `15:29:07Z`, completed
`16:02:18.493Z`.

Acceptance check 3: for reach-out-wanted runs (`9a5dfe992452` and 7 others
in the 14-day window), `reach_out.decision` reads `not_recorded` — no
`endogenous_outreach_decisions` row exists yet with
`correlation_id = uuid5(NAMESPACE_URL, "curiosity_outreach:<run_id>")` for
any of them (confirmed directly against Postgres: zero rows for all 8 keys).
This is honest, not a bug in the read: PR #2290's write side has not merged, no
run has been *newly* blocked since it deployed, and the 63 `curiosity_outreach`
-tagged rows that do exist in the table are a one-time historical backfill
(all dated `2026-09-19 08:21:39`, all with empty `correlation_id` and no
`result_json.run_id`) that predates and does not match the per-run key.

Run `446ddd7165d5` full timeline (abridged):

```text
+0:00    lifecycle   accepted
+0:00    lifecycle   waiting (agent lane)
+423:39  lifecycle   admitted (agent, waited 423:39)
+423:39  lifecycle   resumed / running (run_started)
+435:18  role_choice local_crawl -- "queue pressure was high"
+447:12  hop 1       "Window read since 2026-09-20T00:00Z: 3577 journal rows..."
+447:12  hop 2       "juniper_primary is not new: 65 rows from 2026-07-24..."
   ?     finding     "Whole-table journal author column holds exactly 3 values..."
+447:12  revision    0.60 -> 0.68, revised -> supported
+449:34  outcome     continue_line=True, reach_out=False
+456:49..+456:50  lifecycle running (harness_turn, read_turn_result, publish_attention_row, journal)
+456:50  journal     curiosity-investigation:446ddd7165d5
+456:50  lifecycle   lease_released / completed
```

`prior_touched` correctly names the who-matters prior and its 0.60→0.68
move; `journal_body` is the full 3,707-char write-up.

## Findings surfaced while verifying (not fixed here — out of scope)

- **`self_sense_eval_log` stopped receiving rows after 2026-09-21 18:08:03
  UTC**, while the self-sense line's admission-path runs kept completing
  every ~3h since (5 runs: `20260921T205245Z-2822b8` through
  `20260922T090939Z-427741`, all `terminal=completed`, zero
  `self_sense_eval_log` rows each — confirmed by direct query). This tab
  will show all five as `wrote_nothing` on the strip, which is correct
  behavior for what the stores actually say — and is exactly the kind of
  silent gap this redesign exists to surface (the design doc's own
  "priors=0/0 for four hours" precedent). Severity: should. Not investigated
  further or fixed here — separate follow-up, the scoring writer
  (`orion/evals/self_sense_runner.py` or its caller) needs a look.
- Confirms the design doc's own finding: curiosity outreach has never been
  delivered (`sent: 0` of 8 wanted in the live 14-day window, same as the
  design doc's headline number).

## Review findings fixed

Reviewed by a subagent against `main..HEAD` (12 files, 3876+/1467-).
Independently re-verified findings myself before fixing (checked PR #2290's
actual GitHub state via `gh pr view`, not just trusted the report).

- Finding (must-fix): comments in `curiosity_run_store.py` (lines 62, 78)
  and `services/orion-hub/README.md` claimed PR #2290 was "merged" / "live" /
  read as already-resolved. Independently confirmed via `gh pr view 2290`:
  `"state":"OPEN","mergedAt":null`. The claim was aspirational, not runtime
  fact, and would have misled the next reader into thinking outreach
  decisions are already being recorded.
  - Fix: reworded every mention (`curiosity_run_store.py`'s two comment
    blocks, `services/orion-hub/README.md`'s reach-out-honesty paragraph,
    `test_curiosity_routes_runs.py`'s docstring, this report's Summary and
    Risks sections) to state #2290 is open/not-yet-merged, and to say
    explicitly that the read side is written against its contract now so no
    change is needed when it lands.
  - Evidence: `gh pr view 2290 --json state,mergedAt` →
    `{"mergedAt":null,"state":"OPEN"}`; `grep -rn "PR #2290" <touched files>`
    shows no remaining "merged" claim.
- Finding (should-fix): `_line_for`'s last-resort fallback (checking whether
  a revised prior's own `line == "self"` when nothing else names the line)
  had zero test coverage — every self-inquiry fixture in the suite set
  either the finish-row `line` or the journal title, never exercising this
  branch.
  - Fix: added `test_the_last_resort_line_fallback_reads_the_revised_priors_own_line_field`
    and `test_the_last_resort_fallback_defaults_to_investigate_when_nothing_names_a_line`
    to `tests/test_curiosity_run_story.py`.
  - Evidence: both new tests pass; see Tests run below.
- Findings checked and confirmed clean (no fix needed): SQL injection
  (asyncpg `$1`/`$2` binding throughout, tested with a literal `' OR 1=1`
  run id), Cypher injection (`valid_run_id`/`_RUN_ID_RE` allow-list gates
  every id before the `CYPHER` param prefix; `prior_claims_cypher`'s
  free-text path uses `json.dumps` for FalkorDB's parameter lexer, not
  string splicing), the never-500/`available: false` contract on both new
  routes, template XSS escaping (verified by the reviewer running the actual
  page script under Node against an `<img onerror=...>` payload), the
  hash-routing/poll-diffing logic, and that no dead reference to the removed
  `assemble_runs`/`AtlasRun`/growth Cypher/park-pin routes remains anywhere
  in the tree.

## Restart required

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml \
  up -d --build orion-hub
```

No other service needs a restart. This patch does not depend on PR #1b
(`finish_detail.harness_elapsed_sec`/`turn_correlation_id` on
`orion-durable-runs`) — `run.harness` reads `None` until that lands, printed
on the page as "harness timing: not recorded for this run".

## Risks / concerns

- Severity: should. Concern: the `self_sense_eval_log` writer gap above is
  now visible to Juniper for the first time; the tab surfacing it is the
  point, but it will read as a new problem the day this ships rather than
  a pre-existing one just made visible. Mitigation: named explicitly in this
  report and worth a one-line heads-up when this deploys.
- Severity: should. Concern: PR #2290 (the write side this patch's reach-out
  rendering is built for) is still open. Until it merges and deploys, every
  run in the sittings strip and every run story will show `reach_out.decision
  = "not_recorded"` for a wanted reach-out — correct given what the stores
  hold today, but it means the "blocked: `<gate>`" and "reply: ..." UI paths
  ship untested against real data in this patch (they are unit-tested against
  fixtures; see Review findings fixed). Mitigation: none needed for this PR;
  flagged so whoever merges #2290 knows to spot-check the tab against a real
  blocked/sent run afterward.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2291
