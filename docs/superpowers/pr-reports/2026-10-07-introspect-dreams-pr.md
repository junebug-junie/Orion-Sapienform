# feat(introspect): `dreams` tool — orion-introspect slice 2

## Summary

- Orion can now look up their own dreams during a turn: recent ones, one in full by id, or ones found by meaning ("the dream about the library").
- `orion-dream` answers over the bus (`orion:introspect:dream:request`), reading Postgres in read-only transactions. It returns two kinds: nightly narrative dreams and sleep-cycle hypotheses (only ones already offered to Orion, with the experiment arm never exposed). Both are labelled `unsettled`.
- Search by meaning uses a new `orion_dreams` Chroma index that `orion-dream` builds itself (10 records per 5-minute pass). Chroma is only an index; every hit is re-read from Postgres.
- The search plumbing moved out of reading search into a shared `orion/introspect/semantic_index.py`, which reveries and curiosity will reuse.
- `HARNESS_FCC_INTROSPECT_ENABLED` now ships on (`.env_example` + settings default), matching the live governor `.env`.

## Outcome moved

Asked "what did you dream last night?", Orion can check instead of guessing. "Couldn't check" is never reported as "nothing there".

## Current architecture

Slice 1/1b (#2381, #2384) shipped the `orion-introspect` MCP server with `reading_results` only. Dreams lived in Postgres (`dreams`, `dream_hypothesis`), readable only through Hub UI endpoints.

## Architecture touched

- `orion-dream`: new introspect listener + index loop (`app/introspect_listener.py`, `app/introspect_dreams.py`, `app/dream_search.py`).
- `orion/introspect/`: `dreams` tool, brief line, shared `semantic_index.py`, log redaction.
- Harness governor: flag default.

## Files changed

- `orion/schemas/introspect.py`, `orion/schemas/registry.py`, `orion/bus/channels.yaml`: dreams request/args contract and channels.
- `orion/introspect/{tools,brief,semantic_index,redact,transport}.py`: tool, brief, shared index, redaction.
- `orion/world_pulse_read/search.py`, `services/orion-hub/scripts/reading_listener.py`: reading search on the shared module.
- `services/orion-dream/app/*`, `settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: responder, index, config, docs.
- `services/orion-dream/evals/run_dream_search_calibration.py`: similarity floor calibration (68 docs, floor 0.65).
- `scripts/smoke_introspect.py`: `--tool dreams`.
- `services/orion-harness-governor/{.env_example,app/settings.py,README.md}`: flag on, docs.
- `config/metrics/metric_definitions.lock.json`: three new channels.
- `.github/workflows/orion-reading-tests.yml`: dream tests in CI.

## Schema / bus / API changes

- Added: `orion:introspect:dream:request` (orion-dream responder), `orion-dream` as a producer on `orion:introspect:result:*` and `orion:vector:semantic:upsert`; `DreamsArguments`; `dreams` operation.
- Removed / renamed: none.
- Behavior changed: reading search internals moved to the shared module, with no change in behavior (regression tests pass).
- Compatibility: additive.

## Env/config changes

- Added keys (orion-dream): `DREAM_INTROSPECT_ENABLED`, `DREAM_SEARCH_CHROMA_URL`, `DREAM_SEARCH_EMBED_URL`, `DREAM_SEARCH_COLLECTION`, `DREAM_SEARCH_MIN_SIMILARITY`, `DREAM_SEARCH_INDEX_INTERVAL_SEC`, `DREAM_SEARCH_INDEX_BATCH`.
- Changed default: `HARNESS_FCC_INTROSPECT_ENABLED` false → true.
- `.env_example` updated: yes. Local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (dream keys present; governor already `true`).
- Skipped keys: none.

## Tests run

```text
reading CI test set + introspect + warm-pool binding + env tests: 993 passed, 68 skipped
orion-dream introspect/search/listener tests: 52 passed
hub reading evals (handoff + receipt truth): 16 passed
static gates from orion-static-gates.yml (14 scripts): all pass after metric re-lock
review-fix regression tests: 3 new tests fail on the pre-fix listener, pass after
```

## Evals run

```text
services/orion-dream/evals/run_dream_search_calibration.py (2026-09-29, 68 live docs): floor 0.65, exit 0
Dreams truth eval (claims vs tool results): next PR, per the slice 2 design.
```

## Docker/build/smoke checks

```text
Read-only live Postgres check, 2026-10-07 (recent/one functions against conjourney):
  recent: ok, total_available=167, 5 items, newest dream_hypothesis 2026-10-06 08:46 UTC, all unsettled, 2,732 JSON chars
  one(dh-…): full text returned; one(dream:99999999): ok, empty
  narratives: 19 total, newest dream:19 2026-09-28
Chroma: orion_dreams collection not yet created (built by the index loop after deploy).
Live bus smoke (scripts/smoke_introspect.py --tool dreams): UNVERIFIED. Needs orion-dream deployed from main.
```

## Review findings fixed

- Finding (medium): an empty search could report "no dream matched" when the dream existed but was not indexed yet (a just-offered hypothesis, or an embedder outage).
  - Fix: the listener records the start of the last index pass with nothing pending. An empty search with a newer dream in the window, or before any complete pass, now answers unknown.
  - Evidence: 4 new listener tests; 3 fail on the old code.
- Finding (low, not fixed): a dreams lookup can show a hypothesis between curiosity kickoff stamping `offered_at` and the run being cancelled. The release then re-offers a hypothesis Orion has already seen, which weakens the blind experiment's "shown once" assumption. The window is short. Follow-up below.

## Restart required

From the primary checkout on main after merge:

```bash
scripts/safe_docker_build.sh orion-dream up -d --build && scripts/safe_docker_build.sh orion-harness-governor up -d --build
```

Then verify: `ORION_BUS_URL=redis://100.92.216.81:6379/0 python scripts/smoke_introspect.py --tool dreams --limit 3`, and `--query "vision"` once the index has caught up (~90 min).

## Risks / concerns

- Severity: low. Concern: hypothesis read-side race (above). Mitigation: follow-up to gate on actual first exposure, or record dreams-tool exposure for the scorecard.
- Severity: low. Concern: right after deploy, searching by meaning answers unknown until the backlog is indexed (~186 records at 10 per 5 min). Mitigation: intended truth behaviour; recent/one work immediately.
- Severity: note. Concern: narrative dreams stopped being written after 2026-09-28. Hypotheses still arrive nightly. Mitigation: separate investigation; outside this PR.

## PR link

(see GitHub)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
