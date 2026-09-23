# PR report — Curiosity run story: an explicit reply box

## Summary

- New `POST /curiosity/api/run/{run_id}/reply`: Juniper can answer one of
  Orion's curiosity reach-outs from the run's own story page, with an
  EXPLICIT link back to that exact outreach. This is design doc "Missing
  question 1, option (b)" (`docs/curiosity-tab-redesign` branch,
  `2026-09-22-curiosity-tab-redesign-design.md`), the half PR #2290's chat
  reply left for later.
- `orion/curiosity/run_story.py`: `ReachOut.can_reply` — true only for a
  confirmed `sent` Door-A decision — so the page knows when to show the box.
  This is display-only; the route re-checks the live decision row itself.
- `services/orion-hub/scripts/curiosity_run_store.py`:
  `fetch_sent_outreach_session()` resolves the sent decision and the real
  delivery session from `chat_history_log`, raising (not degrading) on a
  DB failure since this sits on a write path.
- `services/orion-hub/scripts/api_routes.py`: `handle_chat_request()` gained
  an optional `client_meta` kwarg threaded into `execute_unified_turn()` —
  the same seam `websocket_handler.py` already uses for the heuristic reply
  stamp, now reusable for an explicit one.
- `services/orion-hub/templates/curiosity_atlas.html`: a reply box appears
  only when `reach_out.can_reply` is true; never renders the reply
  optimistically — shows "Sent…" and re-fetches the story later so the real
  reply (once persisted) appears from a later poll.

## Outcome moved

Before: Juniper's only way to answer a curiosity reach-out was an ordinary
chat message, linked back to the outreach by a 12-hour time-adjacency guess
(PR #2290) — correct most of the time, but a guess. After: she can click
reply on the exact run whose reach-out she's answering, from the run story
itself, and the link is a fact (`client_meta.in_reply_to`), not an inference.
The two mechanisms coexist: the heuristic still covers ordinary chat, the
explicit link covers this one new button.

## Current architecture

`orion/curiosity/run_story.py`'s `ReachOut` already computed `decision` (one
of `sent | blocked:<gate> | composed_empty | passed | not_recorded`) and any
matched `reply` from `chat_history_log.client_meta->>'in_reply_to'`, keyed on
`outreach_key(run_id) = uuid5(NAMESPACE_URL, f"curiosity_outreach:{run_id}")`.
`services/orion-hub/scripts/curiosity_routes.py` served three GETs (`/atlas`,
`/runs`, `/run/{id}`) plus two existing POSTs (`run-now`,
`self-inquiry/run-now`) that only ask a loop to take a turn sooner. Chat
turns went through `handle_chat_request()` in `api_routes.py`, which for the
unified (`mode in ("orion", "agent")`) lane called
`orion.hub.turn_orchestrator.execute_unified_turn()` — already accepting a
`client_meta` kwarg for the WebSocket path's heuristic reply stamp
(`websocket_handler.py`'s `reply_stamp_for_session()`), but the HTTP path
never passed one through.

## Architecture touched

- `orion/curiosity/run_story.py`: `ReachOut.can_reply` property + payload key.
- `services/orion-hub/scripts/curiosity_run_store.py`: new write-path reader.
- `services/orion-hub/scripts/api_routes.py`: `handle_chat_request()` gains
  `client_meta`, threaded only into the unified-turn branch; the legacy
  non-unified chat path is untouched.
- `services/orion-hub/scripts/curiosity_routes.py`: new route + a
  `_handle_chat_request()` indirection.
- `services/orion-hub/templates/curiosity_atlas.html`: reply box + JS.
- `services/orion-hub/README.md`: documents the new endpoint; also corrects
  a stale "PR #2290 open" claim (it merged 2026-09-22, confirmed via
  `gh pr view 2290`).
- No bus, schema, registry, or `.env_example` change.

## Files changed

- `orion/curiosity/run_story.py`: `ReachOut.can_reply` (true iff
  `decision == "sent"`); included in `_reach_payload`.
- `services/orion-hub/scripts/curiosity_run_store.py`:
  `fetch_sent_outreach_session(pool, run_id)` — one query for the sent
  decision, one for the real delivery session, both parameterized.
- `services/orion-hub/scripts/api_routes.py`: `handle_chat_request(...,
  client_meta=None)`, threaded into `execute_unified_turn(client_meta=...)`.
- `services/orion-hub/scripts/curiosity_routes.py`: `POST
  /api/run/{run_id}/reply`; `_handle_chat_request()` module-level
  indirection so tests can patch it reliably (this repo has two same-named
  `scripts` packages — root and per-service — and a bare
  `from .api_routes import handle_chat_request` at call time was found
  during development to resolve inconsistently against them).
- `services/orion-hub/templates/curiosity_atlas.html`: reply textarea +
  button (event-delegated via the existing `data-*` click-listener pattern),
  `sendReply()`, CSS.
- `services/orion-hub/README.md`: new endpoint documented; stale PR #2290
  status corrected.
- Tests: `services/orion-hub/tests/test_curiosity_run_reply_route.py` (new,
  17 tests); `tests/test_curiosity_run_story.py` (`can_reply` cases);
  `tests/test_curiosity_atlas.py` (the write-route inventory guard test,
  updated with justification); `tests/test_curiosity_atlas_template.py`
  (reply box presence/absence + wiring); `services/orion-hub/tests/
  test_curiosity_routes_runs.py` (one payload-equality assertion updated for
  the new `can_reply` field).

## Schema / bus / API changes

- Added: `POST /curiosity/api/run/{run_id}/reply` — body `{"text": str}`
  (rejects empty/whitespace and >4000 chars, `400`); `409
  {"ok": false, "reason": "no_sent_outreach_to_reply_to"}` for any run whose
  Door-A outreach was not confirmed `sent`; `500 {"ok": false, "reason":
  "session_unresolvable"}` if the sent decision has no matching
  `chat_history_log` row; a downstream chat-turn failure returns `200
  {"ok": false, "reason": ...}`, never a bare 500; success returns `200
  {"ok": true, "correlation_id": ..., "session_id": ...}`.
- Removed: none.
- Renamed: none.
- Behavior changed: `handle_chat_request()` accepts an optional `client_meta`
  kwarg (default `None`, backward compatible — every existing caller is
  unaffected); `ReachOut`/`run_to_payload` gain `can_reply` (additive JSON
  key).
- Compatibility notes: purely additive. No `.env_example`, bus channel, or
  schema registry change (`orion/bus/channels.yaml`, `orion/schemas/
  registry.py` untouched — confirmed via `git diff main...HEAD --stat` above).

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced: not needed.
- skipped keys requiring operator action: none.

## Tests run

```text
cd services/orion-hub
.venv python -m pytest tests/test_curiosity_run_reply_route.py -q
17 passed

cd (repo root)
python -m pytest tests/test_curiosity_run_story.py tests/test_curiosity_atlas.py \
  tests/test_curiosity_atlas_template.py tests/test_curiosity_atlas_peer_briefs.py -q
109 passed

cd services/orion-hub
python -m pytest tests -q -k "curiosity"
277 passed, 2520 deselected

python -m pytest tests -q -k "chat or api_routes or handle_chat or unified_turn or websocket"
339 passed, 2 skipped, 2454 deselected
  2 pre-existing failures, both confirmed present on main too (unrelated to
  this branch):
  - tests/test_turn_orchestrator_utterance_origin.py::
    test_execute_unified_turn_uses_mind_appraisal_text_for_stance_not_harness
  - tests/test_handle_chat_request_turn_effect.py::
    test_handle_chat_request_retries_memory_graph_suggest_on_timeout

python scripts/check_env_template_parity.py
env template parity: PASS (88 service(s) compared) -- same 23 pre-existing,
unrelated warnings as PR #2291.
```

`scripts/check_schema_registry.py` and `scripts/check_bus_channels.py` do
not exist in this repo (per CLAUDE.md's own note and prior PR reports).

## Evals run

No eval harness exists for `orion/curiosity/` or the Hub curiosity surface
(same gap noted in PR #2291's report — this patch is a single write route
plus a display flag, and its correctness is covered by unit tests against
fixtures, not a quality eval). No new eval added; not claimed as covered.

## Docker/build/smoke checks

Not run — instructed not to `docker compose up` the hub. Deterministic
checks only (tests, env parity). Restart command listed below for when
Juniper deploys this.

## Live verification (read-only)

Queried live Postgres (`postgresql://postgres:postgres@127.0.0.1:55432/
conjourney`, read-only, no writes) for a run that would resolve
`can_reply: true`:

```sql
SELECT count(*) FROM endogenous_outreach_decisions
WHERE result_json->>'source' = 'curiosity_outreach' AND reason = 'sent'
  AND coalesce(result_json->>'run_id','') <> '';
-- 0
```

**Zero real curiosity Door-A sends with a run id exist yet.** The 4 rows
that do match `reason='sent' AND source='curiosity_outreach'` today all carry
`correlation_id` values of literally `corr-1` or `corr-door-a` and an empty
`result_json->>'run_id'` — these are test fixtures written by another
session's test run against the same shared dev Postgres (PR #2298 just
merged the Door-A share path, and a concurrent test suite run left rows
behind), not real production sends. This is expected, not a gap: `#2298`
just landed and no run has been *newly* sent through it yet. Per
instructions, I did **not** manually insert a live-looking fixture into the
real database to force a false positive, and did **not** send an actual
reply to a real run (that would really message Orion). The endpoint's happy
path is verified only against the unit-test fixtures in
`test_curiosity_run_reply_route.py` and `test_curiosity_run_store.py`-style
mocks — never against a genuine live sent outreach.

## Review findings fixed

Code-review subagent ran against `main...HEAD` (read-only), covering gating
logic, `client_meta` reaching `chat_history_log`, SQL injection, the
never-500 contract, the `_handle_chat_request` indirection, and frontend
XSS/double-submit. Verdict: no `must` or `should` findings.

- Finding (nit): the store docstring and one test name
  (`..._never_a_bare_500`) claimed the route "never" 500s, but a DB failure
  on the outreach lookup correctly does return 500 (an infra failure, not a
  chat-turn refusal) — only a downstream chat-turn failure is guaranteed
  `ok:false` + 200. The docstring and test name overstated the guarantee.
  - Fix: reworded `fetch_sent_outreach_session`'s and the route's docstrings
    to name the actual split (infra failures 500, chat-turn failures don't);
    renamed the test to `test_reply_route_surfaces_a_lookup_failure_as_ok_false_with_a_500`.
  - Evidence: `services/orion-hub/scripts/curiosity_run_store.py`,
    `services/orion-hub/scripts/curiosity_routes.py`,
    `services/orion-hub/tests/test_curiosity_run_reply_route.py`; re-ran the
    17-test file after the rename, still green.
- Checked and confirmed clean (no fix needed): gating is server-side
  re-checked per request (a stale client-side `can_reply` cannot force a
  reply through — the 409 path re-verifies the live decision row);
  `client_meta` traced end to end from the route through
  `execute_unified_turn` to the same persistence path PR #2290 already
  verified live; both new SQL queries are `$1`-parameterized with `run_id`
  validated via `valid_run_id()` before it ever reaches a query; the
  `_handle_chat_request` indirection is a real fix for a real
  import-resolution ambiguity in this repo's two same-named `scripts`
  packages, not a workaround papering over a design flaw; frontend reply
  text is set/read via `.value` (never `innerHTML`), `run_id` is escaped via
  `esc()` in every HTML-attribute position and via a separate `cssEsc()` in
  the one CSS-attribute-selector lookup (correct escaping for that distinct
  context); double-submit is prevented by disabling the button immediately
  on send, re-enabled only on failure (success leaves it disabled until the
  5s-later re-render); the reply is never rendered optimistically — only a
  "Sent…" status string, never the reply text itself, before the backend
  confirms.

## Restart required

```bash
# From a worktree, not the shared checkout:
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: should
  - Concern: the happy path (a confirmed `sent` outreach → successful
    reply) is verified only against unit-test fixtures, never against a
    real live sent outreach, because none exists yet (see Live verification
    above).
  - Mitigation: the store query, the gating (409/500 paths), and the exact
    `client_meta` shape are all covered by tests against realistic fixture
    rows; `client_meta` threading through `execute_unified_turn` reuses an
    already-verified-live mechanism (PR #2290). Whoever deploys this should
    spot-check the reply box against the first real sent curiosity outreach.
- Severity: info
  - Concern: `client_meta` only threads through the unified-turn HTTP
    branch of `handle_chat_request()` (`mode in ("orion", "agent")`); if
    `ORION_UNIFIED_TURN_ENABLED` is ever false, a reply posted through this
    route falls through to the legacy chat path and loses the explicit
    stamp (though it would still send and land in history, just without
    `in_reply_to`).
  - Mitigation: this mirrors the existing scope of PR #2290's own
    `client_meta` threading (also unified-turn only); `ORION_UNIFIED_TURN_ENABLED`
    is the live default. No action needed unless that default changes.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2299

🤖 Generated with [Claude Code](https://claude.com/claude-code)
