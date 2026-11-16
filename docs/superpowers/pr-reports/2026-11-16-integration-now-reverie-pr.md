## Summary

Integration patch 3, reverie slice. Reveries now see what Orion's other organs lived, and say which of those things they used.

- Each reverie chain reads Orion's now once: `GET /temporal-self/now?organ=reverie`. This shows the few most salient experiences that other organs recorded since reverie last looked: cameras, readings, curiosity, dreams, body, predictions. This read replaces the closed-arcs Temporal Self block.
- The prompt shows one line per experience with an `[experience:<id>]` token. The model names what it used in a structured `cites` field. The field is kept only for shown ids, at most 3 per shown now across the whole chain, and stored as `SpontaneousThoughtV1.experience_cites`.
- When the thought closes, its cites are posted to `POST /temporal-self/now/use` with the `shown_id`. The server checks the cites again and writes `experience_use` rows.
- Fallback (the spec's rollback rule): if the now is off (404), unreachable, slow or malformed, the reverie gets the old closed-arcs block. The log line is `reverie_now_fallback reason=...`. No failure in this path can fail a reverie.
- `config/integration/organs.yaml`: the reverie entry now names `reads_now: services/orion-thought/app/temporal_self_cue.py`, and its read exemption is removed. No other organ entry was touched.

## Outcome moved

Before this patch, reverie could not use a camera event, a reading, or a dream unless one happened to be the thing its attention settled on. Now a reverie can be about one of those by evidence, and every use leaves a checked `experience_use` row (`citing_organ = reverie`). That row is the integration metric's raw input.

## Current architecture

- orion-thought's reverie (`app/reverie.py` `run_reverie_once`, driven per step by `app/chain.py`) fetched the Temporal Self day frame (`GET /temporal-self/frame`). From it, reverie rendered the last 3 closed arcs as background (`build_recent_arcs_text`).
- Nothing reverie produced was recorded as a use of another organ's output.
- Patch 1 (#3453) added the now endpoints in orion-durable-runs. Patch 2 (#3454) added the organ registry and the CI gate, with reverie exempt from reading.

## Architecture touched

- orion-thought: the reverie cue and tick (`app/temporal_self_cue.py`, `app/reverie.py`), settings and env.
- Shared prompt `orion/cognition/prompts/reverie_narrate.j2`, rendered by orion-cortex-exec.
- Shared schema `orion/schemas/reverie.py`: one additive field.
- The ctx provenance registry gets one new key: `orion_now`.
- The organ registry gets the reverie entry change.
- No bus channel changes. No changes to orion-durable-runs.

## Files changed

- `services/orion-thought/app/temporal_self_cue.py`: adds the now read (`fetch_now_bounded`), the block renderer, the shown-cites filter, and the use post (`post_now_use`). This is the `reads_now` module the integration gate checks.
- `services/orion-thought/app/reverie.py`: adds `build_reverie_day_cue`, which reads the now with the arcs block fetched alongside as fallback, keeps one read per chain, and re-renders ages each step. Also adds `record_reverie_now_use`, the per-shown cite budget, and cites parsing in `parse_reverie_payload`.
- `orion/schemas/reverie.py`: adds `experience_cites` (max 3) and `MAX_EXPERIENCE_CITES`.
- `orion/cognition/prompts/reverie_narrate.j2`: adds the "WHAT THE REST OF YOU LIVED SINCE YOUR LAST REVERIE" section and the `cites` field, in both the concern-card branch and the coalition branch.
- `orion/schemas/context_provenance.py`: classifies `orion_now` as `derived_summary`.
- `config/integration/organs.yaml`: reverie `reads_now` is set and its `exempt_read` is removed.
- `services/orion-thought/app/settings.py`, `services/orion-thought/.env_example`: 2 new keys.
- `services/orion-thought/tests/test_reverie_orion_now.py`: 30 tests (new file).
- `.github/workflows/temporal-self-readers-tests.yml`: runs the new tests and the recent-arcs tests.

## Schema / bus / API changes

- Added: `SpontaneousThoughtV1.experience_cites: list[str]` (default `[]`, max 3).
- Removed: none.
- Renamed: none.
- Behavior changed: the reverie prompt gets `orion_now` instead of `recent_arcs` whenever the now is available.
- Compatibility notes:
  - The new field is additive. The model keeps pydantic's default `extra=ignore`, so old consumers and old stored rows still parse.
  - The field persists inside `thought_json`. No migration is needed.

## Env/config changes

- Added keys: `ORION_REVERIE_NOW_ENABLED=true`, `ORION_REVERIE_NOW_FETCH_TIMEOUT_SEC=2.0` (orion-thought).
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes. Both keys are in the primary checkout's `services/orion-thought/.env`.
- Skipped keys requiring operator action: none.

## Tests run

```text
services/orion-thought: pytest tests                                  889 passed, 32 skipped
services/orion-thought: pytest tests/test_reverie_orion_now.py tests/test_reverie_recent_arcs.py   all passed
orion/schemas/tests/test_context_provenance.py                        6 passed
orion/temporal_self/tests                                             passed
scripts/check_env_template_parity.py                                  OK
scripts/check_integration_seam.py      no reverie findings; main's own 2 failures (session_roundup)
                                       are pre-existing, fixed by open PR #3493 (issue #3491)
```

The new tests cover these cases:

- The cue is built from a fake now.
- An empty now shows no block and does not fall back.
- The fallback happens on 404, connect error, 503, a malformed body, and a slow read. The fallback costs max(now timeout, arcs timeout), not their sum.
- With the flag off, the now is never read.
- A chain makes one read and every step uses the same shown_id. Ages are re-rendered at each step.
- Cites are kept only for shown ids, at most 3. An experience id in `evidence_refs` is moved to the cites and never counts as grounding.
- The chain shares one cite budget, and only cites the server stored use it up.
- `record_use` is posted with the `shown_id` and `citing_ref`. When there are no cites, nothing is posted.
- A failed post (500, connect error, 404) never fails the reverie.
- The prompt renders the now block in both branches.

## Evals run

```text
None. The spec's ablation eval (a weekly replay with a local model that removes a cited item and
checks whether the output changes) is not built in this slice. Tracked in the spec's patch 3
acceptance checks.
```

## Docker/build/smoke checks

```text
No Docker build was run from this worktree. Deployment and live verification are deliberately left
out of this task ("Do not merge or deploy").
```

## Review findings fixed

- Finding (should-fix): the steps of a chain share one shown_id, and the server caps cites at 3 per shown row across calls. A later step's cites were rejected but still kept on its thought.
  - Fix: a per-shown-now record of the cites the server stored. Later steps get only the remaining budget and never re-cite an id already stored.
  - Evidence: `test_chain_cite_budget_is_shared_across_steps`, `test_chain_budget_counts_only_cites_the_server_stored`.
- Finding (should-fix): the fallback ran after the now fetch, so the worst case was the now timeout plus the arcs timeout (about 3 s).
  - Fix: the arcs block is fetched alongside the now and cancelled when the now arrives.
  - Evidence: `test_fallback_costs_the_slower_fetch_not_the_sum`.
- Finding (nit): "N min ago" went stale on later chain steps.
  - Fix: the cue keeps the read and re-renders it each step.
  - Evidence: `test_chain_steps_rerender_ages`.
- Finding (nit): a stray experience ref was dropped without notice in fallback mode.
  - Fix: the log line now records whether a now was shown.
- Finding (should-fix, design): the server moves reverie's cursor on every read, even when the read then goes unused (hollow thought, GPU refusal, client timeout).
  - Not fixed here. The cursor rule lives in orion-durable-runs (patch 1) and applies to every reader. Filed as #3498.

## Restart required

The prompt template is baked into orion-cortex-exec's image, so both services need a rebuild after merge, from the clean primary `main` checkout:

```bash
scripts/safe_docker_build.sh orion-thought up -d --build && scripts/safe_docker_build.sh orion-cortex-exec up -d --build
```

## Risks / concerns

- Severity: medium.
  - Concern: the live path is UNVERIFIED. No live `experience_use` row with `citing_organ = reverie` exists yet, and no live reverie has been quoted citing a camera or reading item. Both are patch 3 acceptance checks and need a deploy and 48 h.
  - Mitigation: after deploy, check the `reverie_now` and `reverie_now_use` log lines in orion-thought. Then check `experience_use` rows with `citing_organ = 'reverie'`.
- Severity: low.
  - Concern: the now endpoint's latency on the live database is not measured. The 2.0 s timeout is a guess.
  - Mitigation: a slow read falls back to the arcs block. `reverie_now_fallback reason=timeout` lines would show it.
- Severity: low.
  - Concern: each reverie read moves reverie's cursor even when the read goes unused (#3498).
- Severity: low.
  - Concern: the deploy order of the two services does not matter. If orion-thought is rebuilt first, the old prompt ignores `orion_now` and no cites come back. If cortex-exec is rebuilt first, the new prompt renders without the block.
- Issues opened: #3498. Issues closed: none.

## PR link

(see the PR)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
