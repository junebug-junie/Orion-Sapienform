## Summary

- The Hub Personality tab gets a **"When" column**: the mix of situations each node's evidence happened in, for example "idle 70% · night 55% · alone 80% (n=20)". It is observability only. Weights, matching, the frame and anything Orion reads are unchanged.
- durable-runs `GET /personality/dispositions` gains a per-node `situation_mix`, computed with one grouped SQL query over `personality_evidence.meta->'situation'`.
- **Live bug fixed:** situations were stamped before Temporal Self had caught up. Rows at 00:24 on 10-17 read "with company, strained". The real situation was "alone, engaged": the person had left at 00:20:40, and arousal had changed. Reads now stay behind the chronicle's watermark, minus the presence grace. A row's situation is therefore final when it is stamped.

## Outcome moved

Juniper can watch CAPS if-then signatures ("this theme comes up when idle and alone") form in the panel before anything reads them back. The recorded situation data is now accurate, not optimistic.

## Current architecture

Since #2739, each evidence row records `meta.situation`. Before this PR, the reducer read rows 120 s behind now, while:
- the Temporal Self chronicle folds about 5 min behind now and backdates a company arc's end to the last sighting;
- vision-window confirms a departure only after `WINDOW_PRESENCE_GRACE_SEC=120`;
- regulation writes arousal transitions up to 120 s late.

## Architecture touched

- **orion-durable-runs.**
  - `PersonalityStore.dispositions()`: `situation_mix`.
  - `PersonalityStore.load_situation_inputs`: `company_known_until`.
  - `chronicle_watermark()` is new.
  - The driver's read bound is now `min(now - lag, watermark - COMPANY_CLOSE_GRACE)` when the chronicle is fresh (30 min or less behind). A stalled chronicle does not stall personality: company is stamped unknown and a warning is emitted.
- **Shared pure module.** `orion/personality/situation.py`: `COMPANY_CLOSE_GRACE` (180 s), and company reads `None` past the known-until time.
- **orion-hub.** Panel JS "When" column. The company share is computed over rows where company is known.

## Files changed

- `orion/personality/situation.py` and its tests: known-until, the grace constant
- `services/orion-durable-runs/app/personality_store.py`: `situation_mix`, watermark, known-until
- `services/orion-durable-runs/app/personality_driver.py`: the read bound and the lagging warning
- `services/orion-durable-runs/tests/test_personality_loop.py`, `test_personality_postgres.py`: mix tests, the watermark bound, a stalled chronicle, and the backdated-close regression
- `services/orion-hub/static/js/personality.js`, `personality.test.js`: the "When" column
- `services/orion-hub/tests/test_personality_routes.py`: the field passes through unchanged
- `services/orion-durable-runs/README.md`, `services/orion-hub/README.md`: notes
- `docs/superpowers/pr-reports/2026-10-17-personality-situation-panel-pr.md`: this report

## Schema / bus / API changes

- Added: `situation_mix` on `/personality/dispositions` rows (additive JSON field)
- Removed: none
- Renamed: none
- Behavior changed: evidence is folded about 8 min after it happens instead of about 2 min (chronicle lag plus grace). Situations are final when stamped.
- Compatibility notes: no bus, env or migration change. `meta.situation.company` may be `null` when the chronicle is stalled.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
orion/personality/tests -> 111 passed (also in a venv with only pytest + pydantic)
services/orion-durable-runs: pytest tests (throwaway Postgres 16) -> 533 passed, 1 skipped
services/orion-hub: test_personality_routes.py test_hub_panel_reachability.py -> 34 passed, 4 skipped (browser smoke needs playwright)
node --test services/orion-hub/static/js/personality.test.js -> 17 passed
```

## Evals run

```text
Live read-only checks:
- situation_mix over the 22 live nodes: 32 of 71 evidence rows carry a situation; top node "strained 100% · afternoon 60% · with company 100% (n=10)".
- After the chronicle caught up, the 00:24 row's true situation reads {"arousal": "engaged", "day_phase": "dusk", "company": false}; it was stamped {"strained", "afternoon", true}.
- Chronicle watermark lag measured at about 5 min; recent -> absent at 113 s (00:20:40 -> 00:22:33).
```

## Docker/build/smoke checks

```text
Not deployed. UNVERIFIED live until durable-runs and hub are rebuilt.
```

## Review findings fixed

- Finding: bounding by the watermark alone did not make company final. The chronicle backdates an arc's end to the last sighting and only learns of the departure when the presence grace (120 s) runs out.
  - Fix: reads and company trust stay `COMPANY_CLOSE_GRACE` = 180 s behind the watermark.
  - Evidence: `test_a_backdated_company_close_is_folded_before_rows_near_it_are_stamped`.
- Finding: the panel's company share counted unknown (null) rows in its denominator.
  - Fix: the share is computed over known rows.
  - Evidence: a node test.
- Finding: I first described the arc end as written about 5 min late. That is the chronicle's read lag, not the backdating.
  - Fix: corrected in code comments.

## Restart required

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build && scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: low. Concern: about 30 evidence rows stamped between 2026-10-16 23:36 and 10-17 00:26 keep the early, possibly wrong, situations, because evidence is append-only. Mitigation: they are a small fraction and will be outweighed. A one-off re-stamp is possible if wanted.
- Severity: low. Concern: a camera producer that dies silently is closed by the chronicle's stale rule up to 1500 s later, also backdated. That case is not covered by the 180 s grace. Mitigation: documented. It is rare (the producer writes keepalives).
- Severity: low. Concern: `COMPANY_CLOSE_GRACE` mirrors vision-window's `WINDOW_PRESENCE_GRACE_SEC` by hand. Mitigation: a comment names the source key. If that grace grows, this must grow too.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2743

🤖 Generated with [Claude Code](https://claude.com/claude-code)
