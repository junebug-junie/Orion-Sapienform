## Summary

- Connect completed readings to a durable, background judgment over a frozen evidence packet.
- Record accepted connections for the daily letter and the existing outreach path, with fresh citation checks.
- Reopen at most two challenged beliefs for a later curiosity test; offer at most four unscored follow-up seeds.
- Preserve the existing attention budget, outreach gates, and reading pipeline. Automatic resonance sharing defaults off because the historical quality replay failed.
- Add restart/persistence tests using isolated Postgres, offline replay checks, and explicit failure visibility.

## Outcome moved

The October 13 design now has a production integration: completed read → immutable request → durable admission → grounded judgment → stored result → letter/outreach/curiosity consumers. This patch builds that path; its live execution is **UNVERIFIED**. Juniper asked that no containers be touched during an active Athena chat.

The old replay produced 11 accepted share decisions. Agent review found at most two potentially defensible, both with caveats: below the design's 70% quality target. This is not a measurement of Juniper's preferences. No labels are required from Juniper to finish the implementation.

## Current architecture

Patch A already provided packet contracts, centered/hub-corrected retrieval, the judge prompt, deterministic quote/citation checks, and a read-only replay. Hub already owned Stage 2, finding outreach, and the daily letter. Durable-runs already owned background resource admission and checkpoint recovery. Substrate already stored unscored concept-expansion candidates. The missing pieces were production request/result persistence and consumers.

## Architecture touched

- Hub: reconcile recently completed reads outside the Stage 2 tick; index worldview text through the existing semantic index seam; freeze packet, retained text, retrieval basis, and admission request.
- Durable-runs: register `reading.resonance`, request a background hold, call the gateway once per successful attempt, validate, persist, release, and emit lifecycle stages. Transport failures use existing bounded retries. Invalid model output is a terminal rejected receipt, never a request for a more appealing answer.
- Letter/outreach: rehydrate citations from current authoritative records; omit a connection if any citation has disappeared. Missing stores produce a named letter gap. Share candidates still pass the attention economy and human contact gates.
- Curiosity: accepted strain evidence reopens at most two priors, including stale ones. A later prior test clears that opportunity; resonance never rewrites the prior. Original replica coverage is preserved.
- Substrate: accepted follow-ups with real graph anchors become at most four unscored seeds. Stored candidate receipts prevent reminting within the bounded retention window.

Adaptations from October: this uses the current durable gateway call pattern, also used by memory distillation, instead of a cortex verb that could introduce new recall into frozen evidence. A registered typed result in a shared receipt table replaces the proposed lossy finding event. No new service or attention score is introduced.

## Files changed

- `orion/schemas/reading_resonance.py`, `durable_run.py`, `reading.py`, `reading_finding.py`, `registry.py`: request/result, lifecycle, and consumer contracts.
- `orion/world_pulse_read/resonance_*.py`: source hydration, frozen submission, result resolution, and strain evidence.
- `services/orion-durable-runs/app/reading_resonance_*.py`, `admission_runtime.py`, `runner.py`: durable judgment and persistence.
- `services/orion-hub/scripts/world_pulse_read_stage2.py`, `main.py`, `reading_finding_outreach.py`, `curiosity_investigation.py`: producers and consumers.
- `orion/curiosity/worldview.py`, `orion/orion_day/`, `orion/substrate/resonance_seeds.py`, substrate store/worker: downstream effects.
- `services/orion-sql-db/manual_migration_reading_resonance.sql`: immutable first request/result and outreach demotion receipt.
- Hub settings/template, service READMEs, workflows and tests: operator contract and acceptance checks.
- `services/orion-hub/evals/recheck_reading_resonance_archive.py`: reproduce deterministic checks on saved packets without services or model calls.

## Schema / bus / API changes

- Added: `ReadingResonanceRunBriefV1`, `ReadingResonanceResultV1`, optional landed concept IDs, `reading_resonance` table, derived worldview vector collection.
- Removed: none.
- Renamed: none.
- Behavior changed: `reading.lifecycle.v1` permits `resonance_started/completed/failed`; durable-runs joins Hub as a producer. `ReadingFindingV1` can carry resonance evidence.
- Compatibility notes: new fields default safely; only updated durable-runs accepts the new workflow. Apply the additive migration before the three consumers. The migration dependency gate enforces that order. The metric definition lock changes the lifecycle channel's canonical producer key because producer names are sorted; the channel is not removed.

## Env/config changes

- Added keys: `HUB_READING_RESONANCE_ENABLED=true`, `HUB_READING_RESONANCE_SHARE_ENABLED=false`, `HUB_READING_RESONANCE_LANE=agent`, `HUB_READING_RESONANCE_MAX_TOKENS=900`, `HUB_READING_RESONANCE_DAILY_MAX=12`.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: Hub; compose already loads this env file, so no extra passthrough or dependency change is needed.
- local `.env` synced with `python scripts/sync_local_env_from_example.py orion-hub`: yes, ignored primary local file; bus kept at `redis://100.92.216.81:6379/0`. Athena's live configuration was not changed.
- skipped keys requiring operator action: no new key skipped. Protected existing host/secret keys, including `PUBLISH_CORTEX_EXEC_GRAMMAR`, were preserved. Local parity reports 283 pre-existing missing/protected keys; all five new keys are present.

## Tests run

```text
PYTHONPATH=. python -m pytest -q orion/world_pulse_read/tests orion/orion_day/tests \
  services/orion-hub/tests/test_reading_resonance_outreach.py \
  services/orion-hub/tests/test_reading_finding_outreach.py \
  services/orion-hub/evals/test_reading_resonance_eval.py
361 passed

TEST_POSTGRES_DSN=postgresql://juniper@127.0.0.1:55439/postgres PYTHONPATH=. python -m pytest -q \
  services/orion-durable-runs/tests/test_reading_resonance_graph.py \
  services/orion-durable-runs/tests/test_reading_resonance_pg.py \
  services/orion-durable-runs/tests/test_journal_compose_graph.py \
  services/orion-durable-runs/tests/test_episode_distill_graph.py
27 passed, with the service's pinned LangGraph 1.2.11

Replica plan/contracts/answers plus resonance outreach: 58 passed.
Existing memory-distillation gateway regression: 4 passed.
```

The database was a disposable local Postgres process, not a container or production database. Tests cover immutable first binding, persistence retry without regeneration, restart after resource wait, rejection/none, grounded findings, dedupe, local-day bounds, strain clearing, seed eligibility and anti-starvation. CI installs `asyncpg` as a test-only dependency.

## Evals run

```text
Frozen CoreWeave quote/citation/packet eval: included in the 361 tests above, PASS.
PYTHONPATH=. python services/orion-hub/evals/recheck_reading_resonance_archive.py /tmp/orion-resonance-review
29 saved answers: 23 structurally accepted; 11 share, 11 note, 1 none.
6 rejected: 2 unparseable, 1 quote, 1 relation, 1 relation+quote, 1 summary length.
Agent qualitative review of all 29: historical share-quality gate FAIL.
```

The packets were built on 2026-10-14 and copied from Athena's host files. Private prompts, citations, outputs and per-item agent notes remain outside the public repository at `/tmp/orion-resonance-review/`. Checksums bind the assessment to the saved answers. Structural acceptance is not semantic quality. Common errors: treating one contract as evidence about another, inventing a causal relationship from timing, treating open artifacts as independent verification, and deriving a compute ratio from one cost. Instructions now address these failures. The revised prompt has not been judged live and its quality is **UNVERIFIED**. Automatic sharing remains disabled.

### Metric quality gate

1. **Provenance:** existing `resonance_similarity.center`, `dot`, `hub_from_sims`, and `rank_score` consume vector-host embeddings. Production `resonance_retrieval.score_pool` calls those functions; `prepare_brief` freezes the corpus mean, history count, floor and hub basis alongside the request. `resonance_records.worldview_pool` and `atlas_hydrate` read the actual graph records. Judge confidence is retained as model output only, never a routing weight.
2. **Independence:** raw cosine, centered cosine, and hub-adjusted rank share the same vectors and are explicitly redundant transformations, not three independent signals. Stored lifecycle stages describe workflow progress, not psychological state.
3. **Theory:** reuse the existing mean-centering correction (Mu & Viswanath, All-but-the-Top) and CSLS-style hub correction (Conneau et al.). They rank retrieval candidates; neither demonstrates a meaningful connection. Only explicit cited evidence can support the judgment.
4. **Real-data/rest check:** the October 13 persisted replay diagnosis reported unrelated raw cosine median 0.66, centered 0.078, related p10 0.246, unrelated p95 0.310; the unrelated tail overlaps the 0.25 offer floor. The old correction has a real near-zero unrelated baseline, but it is not a calibrated interest detector. The 29 real saved answers were rechecked here and failed semantic share quality. Current corpus calibration and the new live retrieval path are **UNVERIFIED**; rerun the live replay before activation of sharing. Do not treat a missing store as zero connection strength.
5. **Existing mechanism:** reuse Patch A scoring and packet builder, semantic indexing, durable resource admission, finding outreach, prior selection, and substrate candidate storage. No replacement metric or producer is introduced.
6. **Reversibility:** scores only retrieve candidates; packets preserve the basis for audit. `HUB_READING_RESONANCE_ENABLED=false` stops new requests. In-flight durable runs finish their frozen request. `HUB_READING_RESONANCE_SHARE_ENABLED=false` stops this outreach source while preserving ordinary reading outreach. Receipts and indexes remain inspectable; no belief confidence or graph truth is automatically rewritten.

## Docker/build/smoke checks

```text
No container calls, builds, restarts, deployments, or live model calls: deferred at Juniper's request.
Live path: UNVERIFIED.
Metric lineage, inner-state registry, env-key single-source, Hub settings defaults, env parity: PASS.
Definition drift lock regenerated for the lifecycle producer addition; gate PASS.
scripts/safe_graphify_update.sh: unavailable, PermissionError on /mnt/storage-warm.
graphify prs --worktrees / --conflicts: CLI not installed on this host.
No graph was deleted or replaced.
```

## Review findings fixed

- CPU-heavy similarity work blocked Hub's event loop: moved it to a worker thread and reused normalized historical vectors.
- A held resonance could block unrelated outreach: filter only the held finding and continue to later reads; regression tests cover mixed bundles.
- Ineligible rows could starve graph seeds: require anchors before the SQL limit; real Postgres test proves the eligible row survives.
- Missing result stores could look like no connection: named error source and visible letter text, tested.
- Strain ordering could displace replica coverage: carry the strain map through resampling, tested.
- Retrieval could offer self/root siblings and duplicate citations could evade dedupe: exclude all root sources, cap only eligible hydrated records, hash a citation set, tested.
- UTC midnight could reset a per-day gate early: use the configured local timezone, tested.
- Lifecycle completion and landed graph anchors were missing: wire actual stages and stored IDs.
- Clean CI lacked `asyncpg`: add it to test-only requirements. Broad collection also exposed a top-level Hub import; use the repository's test isolation fixture before import.

Final independent review: no findings; 71 focused tests passed.

## Restart required

Deferred. These are deployment instructions only, for after Juniper's live chat and a separately authorized rollout, from clean primary `main` synchronized with `origin/main`:

```bash
python3 scripts/sync_local_env_from_example.py orion-hub
python3 scripts/apply_post_pull_migrations.py services/orion-sql-db/manual_migration_reading_resonance.sql
scripts/safe_docker_build.sh orion-durable-runs up -d --build
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

Keep sharing disabled. Verify a new read's submitted request, durable completion, stored result, letter finding, strain opportunity and seed receipt with its run ID. Re-run retrieval calibration and a fresh judged replay before considering sharing. SQL-writer accepts the existing reading lifecycle schema; its image should be included in rollout if local lifecycle validation uses an older schema.

## Risks / concerns

- Severity: medium. Historical model shares often overclaim. Mitigation: default sharing off, stronger instructions, transparent agent assessment; new quality remains unverified.
- Severity: medium. Current graph/vector/SQL hydration and lifecycle flow are not live-tested. Mitigation: frozen requests, explicit unavailable failures, migration gating, deployment smoke required.
- Severity: low. Graph refresh unavailable on this host. Mitigation: preserve existing graphs; refresh through the safe wrapper on the host with storage mounted.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3208
