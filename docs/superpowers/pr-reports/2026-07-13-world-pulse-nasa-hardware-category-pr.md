## Summary

- Root-caused a silent regression in World Pulse's "curiosity autonomy": `hardware_compute_gpu` (a `recommended` digest section) has permanently shown `status="covered"` since 2026-07-09 20:17 UTC, which suppressed the live-search gap-fill (`build_curiosity_followups`) that's supposed to fire when a section has zero accepted articles.
- Confirmed via the `world_pulse_article`/`world_pulse_digest` tables in the `conjourney` Postgres DB that every article ever counted toward `hardware_compute_gpu` coverage was a `nasa_news` press release (launches, contract awards, Mars missions) — never actual GPU/hardware/compute content.
- `nasa_news` was the only source in `config/world_pulse/sources.yaml` tagged with `hardware_compute_gpu` (alongside its correct `science_climate_energy` tag). Since NASA publishes near-daily, `_compute_coverage()`'s rule — a section is `covered` if `digest_count > 0 or accepted_articles > 0`, with no topical check — has marked the section covered on every run since, killing a gap-fill that fired ~22 times historically before that.
- Fix: drop the `hardware_compute_gpu` tag from `nasa_news`. Bare fix per explicit decision — no replacement source added. `hardware_compute_gpu` now has zero passive sources and relies entirely on the curiosity gap-fill firing on every run where it's uncovered.
- Added a regression test that loads the real `sources.yaml` and asserts no source tagged `hardware_compute_gpu` matches the known off-topic `nasa_news` source_id/domain. Documented the zero-source tradeoff in `docs/world_pulse_dev.md` so nobody "fixes" the resulting gap by re-tagging an off-topic source.

## Outcome moved

`hardware_compute_gpu` stops being permanently (and silently) marked covered by off-topic NASA press releases. The section will now correctly show `missing`/`no_articles` when it has no genuine hardware/GPU content, which re-enables the curiosity live-search gap-fill (`build_curiosity_followups`) and its downstream fold into the daily journal ("Orion went looking...", via `merge_world_pulse_curiosity_into_draft` in `orion/journaler/worker.py`).

## Current architecture

`services/orion-world-pulse`'s pipeline (`app/services/pipeline.py`) computes per-section coverage in `_compute_coverage()`, then calls `build_curiosity_followups()` (`app/services/curiosity.py`) for any section not `status="covered"`, up to `world_pulse_curiosity_max_sections`. Findings are attached to `digest.curiosity_followups`, rendered in the plaintext/hub digest (`app/services/renderers.py`), and — for the journal path — either included by the LLM compose step or deterministically appended by `merge_world_pulse_curiosity_into_draft` if the LLM omitted them. None of that machinery was broken; it simply never had a genuine gap to fill because `hardware_compute_gpu` looked "covered" every run.

## Architecture touched

`config/world_pulse/sources.yaml` only (source-category data). No schema, bus, Docker, or env changes.

## Files changed

- `config/world_pulse/sources.yaml`: `nasa_news.categories` changed from `["science_climate_energy", "hardware_compute_gpu"]` to `["science_climate_energy"]`.
- `docs/world_pulse_dev.md`: added a "Known zero-source sections" note explaining `hardware_compute_gpu` has zero passive sources as of 2026-07-13, why, and that this is an accepted, intentional tradeoff — not a bug to "fix" by re-tagging an off-topic source.
- `services/orion-world-pulse/tests/test_source_registry_config.py` (new): loads the real `sources.yaml` via `load_source_registry()` and asserts (1) `nasa_news` no longer carries `hardware_compute_gpu`, (2) no source tagged `hardware_compute_gpu` matches the known off-topic `nasa_news` source_id/domain.

## Schema / bus / API changes

None.

## Env/config changes

None. Pure source-registry YAML data change — no `.env`/`.env_example` keys added, removed, or renamed.

## Tests run

```text
/mnt/scripts/Orion-Sapienform/.venv-world-pulse/bin/python -m pytest services/orion-world-pulse/tests -q
72 passed, 15 warnings in 2.65s   (70 pre-existing + 2 new)
```

Verified the new tests are a real regression guard by temporarily reverting the `sources.yaml` change and re-running: both new tests failed as expected. Re-applied the fix; full suite green.

## Evals run

None applicable — this is a source-registry data fix, not a model or ranking change. No eval harness exists for `orion-world-pulse` source-category correctness; a topical-relevance eval (e.g. flagging sources whose categories don't match their actual content) would be a reasonable follow-up but is out of scope for this bare fix.

## Docker/build/smoke checks

Not rebuilt/deployed this session. `config/world_pulse/sources.yaml` is baked into the `orion-world-pulse` image at build time (`COPY config/world_pulse /app/config/world_pulse` in the Dockerfile), so the fix has no runtime effect until the image is rebuilt and the container restarted — see Restart required below.

## Review findings fixed

Diff-scoped correctness review performed: traced every `hardware_compute_gpu` reference repo-wide via `rg` to confirm no other coupling to `nasa_news` exists. All other hits are synthetic test fixtures (`orion/autonomy`, `orion/spark/concept_induction`, `orion/substrate` tests) and unrelated design docs — none reference the real source registry. No material findings.

## Restart required

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-world-pulse/.env \
  -f services/orion-world-pulse/docker-compose.yml \
  up -d --build

curl -fsS http://localhost:8628/health
```

Required for this fix to take effect on the live system — `sources.yaml` is read at image-build time, not at runtime, so the currently-running container will keep mistagging `nasa_news` until rebuilt.

## Risks / concerns

- Severity: low
- Concern: `hardware_compute_gpu` now has zero passive sources and depends entirely on the curiosity live-search gap-fill firing every run it's uncovered — this means a live web-fetch call (via `resolve_fetch_backend()`) on every digest run instead of only when a stable feed goes quiet, with associated cost/rate-limit exposure on whatever fetch backend is configured (Firecrawl per `.fcc` mount).
- Mitigation: this is the user's explicit choice (declined the alternative of adding a real GPU/hardware RSS source); the fetch is capability-gated (`web.fetch.readonly`, `budget_per_cycle: 2` in `config/autonomy/capability_policy.v1.yaml`) and degrades gracefully to an empty followup on any fetch/mapping failure (never fails the run). Documented in `docs/world_pulse_dev.md` as an accepted tradeoff.
- Severity: low
- Concern: every journal/digest email sent between 2026-07-09 20:17 UTC and this fix's deployment went out without the GPU gap-fill it should have had — unrecoverable, content already sent.
- Mitigation: none possible retroactively; going forward, once deployed, the next run with a genuine `hardware_compute_gpu` gap will trigger curiosity correctly. Recommend a live smoke run post-deploy to confirm end-to-end (coverage → gate → fetch → journal merge) before considering this closed.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/new/fix/world-pulse-nasa-hardware-category
