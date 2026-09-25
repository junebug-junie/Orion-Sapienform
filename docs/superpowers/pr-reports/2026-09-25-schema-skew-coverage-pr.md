## Summary

- The schema check inside the ladder watchdog used to watch one schema, `FieldStateV1`. It now finds, from the code itself, every strict schema one service writes and a different service reads back. Today that is 84 schema files and 303 strict models across 81 services, including the bus envelope every service sends and receives.
- "Strict" means `extra="forbid"`: the reader rejects any field it doesn't know. That rule caused the 09-20 outage. Discovery is an AST scan (it parses the source without running it). It counts a service as a **reader** when it calls `X.model_validate` (or `parse_obj`, or `X(**data)`), and as a **writer** when it builds `X(field=...)`. Shared-library code counts for the services that actually call that function. `channels.yaml` producers count as writers. A nested model goes wherever its outer model goes.
- A strict model that someone reads but nobody is found writing must be listed in a small declared table (`DECLARED_WRITERS`, 22 entries). The list gives either the real writer or the reason there is none, such as a YAML file or LLM output. A gate test fails when a new one shows up undeclared, and another fails when an entry goes stale.
- The live check now reads each running container's own copies of those files with **one `docker exec` per container**. It parses them on the host and compares **fields**, not bytes. A field the writer sends that a strict reader lacks is red, which is exactly the `extra_forbidden` break, whatever the image ages say. So is a field the reader requires that the writer lacks. A comment-only change is no longer red. A non-strict reader that would silently drop a new field shows up as a warning, never a card.
- Replay proof uses the real file bytes from both sides of the breaking commits. For 09-20 (`586faf93b`), the diff names the five `queue_contention_*` fields the readers rejected. For a second real pair (`c95c8360c`, 2026-08-21: `expected_signal`/`expected_direction` added to `ProposalCandidateV1`, which proposal-runtime writes and execution-dispatch-runtime validates), a stale dispatch container goes red with those field names.

## Outcome moved

The failure is a writer shipping a new field on a strict schema while a different service still runs the old copy and rejects every row. Before this patch that was caught for 1 schema file (5 reader services). Now it is caught for 84 files, 272 distinct (schema file, reader service) pairs, and 5,980 (file, writer, reader) combinations. Live run today: **GREEN, 11.5s wall**. That is about 6.5s of code scan plus about 5s of parallel docker reads across 69 containers.

## Current architecture

- `orion/substrate_ladder_liveness.py` had one hand-declared `StrictSchema` (FieldStateV1 → orion-field-digester). Its readers came from an import scan in which any transitive import of the schema module counted, so policy-runtime was flagged although it never validates FieldStateV1.
- `scripts/check_substrate_ladder_liveness.py` hashed the schema file in each container (one exec per container per schema). It went red when the hashes differed and the reader image was older. Sidecars sharing a compose file (redis, grafana, otel) had no `orion` package, so their unreadable hash fell into the timestamp fallback. With broader coverage that would have produced false reds; they are now skipped.

## Architecture touched

- New stdlib-only module `orion/schema_skew_discovery.py`: repo index, discovery, declared table, host-side shape parser and field comparison.
- `orion/substrate_ladder_liveness.py`: `StrictSchema` now describes one (file, writer) pair with the models each reader reads. `evaluate_skew` compares fields first and falls back to bytes and time. New statuses: `drops_fields` and `no_writer`, neither red. `STRICT_SCHEMAS` is now a pin list that discovery must keep finding. The import-scan `schema_consumer_services` is removed.
- `scripts/check_substrate_ladder_liveness.py`: `check_skew` with one parallel read per container, one `git cat-file --batch` for the main-branch hashes, commit times fetched only when the fallback needs them, `--list-candidates` (the old `--list-consumers` still works as an alias), `--include-loose/--no-include-loose`, and `--docker-workers`. By default the human output hides ok, not_running and no_writer rows.
- Debounce keys for skew are now `skew:<schema file>:<container>`, so a card already delivered for FieldStateV1 re-arms once under the new key. A key counts as recovered only when no writer comparison for it is red that tick.

## Files changed

- `orion/schema_skew_discovery.py`: discovery, `DECLARED_WRITERS`, `shapes_from_sources`, `compare_models`.
- `orion/substrate_ladder_liveness.py`: schema pairs, field-level verdict, key and green semantics.
- `scripts/check_substrate_ladder_liveness.py`: batched live read, sidecar skip, CLI flags.
- `tests/scripts/test_schema_skew_discovery.py`: real-repo gate (nothing unresolved, no stale declarations, bounded coverage), synthetic discovery (inheritance across files, nested models, library-function attribution, re-exports, `**` splat, channels producers, wildcard `*` ignored), shape parser, and an end-to-end `check_skew` with docker faked (one read per container, sidecar skipped).
- `tests/scripts/test_substrate_ladder_liveness.py`: real-bytes replays of 09-20 and of the proposal-frame pair, compatible-bytes case, reader ahead of writer, loose drop, writer not running, multi-writer key semantics, and pin checks. The existing placeholder-hash 09-20 fixture still replays red, including on the timestamp fallback.
- `tests/fixtures/schema_skew/*.py.txt`: `git show` of `field_state.py` around `586faf93b` and `proposal_frame.py` around `c95c8360c`.
- `tests/scripts/conftest.py`: one shared real-repo discovery per session (~6s).
- `.github/workflows/orion-static-gates.yml`: runs the new test file.
- `Makefile`, `scripts/README.md`: describe the new check.

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none. The check only reads.
- Behavior changed: the watchdog can now raise cards for any strict cross-service schema, not just FieldStateV1. Skew card keys changed format (see above).
- Compatibility notes: `--list-consumers` still works.

## Env/config changes

- Added keys: none. Removed keys: none. Renamed keys: none.
- `.env_example` updated: no.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed, since no template changed.
- Skipped keys requiring operator action: none.

## Tests run

```text
.venv/bin/python -m pytest tests/scripts/test_substrate_ladder_liveness.py tests/scripts/test_schema_skew_discovery.py -q
59 passed in 8.67s
.venv/bin/python -m pytest tests/scripts/ -q
1 failed, 460 passed -- the failure is test_rebuild_affected_services.py::test_sample_pull_diff,
which fails identically on a clean origin/main tree (unrelated, pre-existing).
python scripts/check_definition_drift.py --gate  -> PASS
```

## Evals run

```text
No eval harness for this watchdog. The replay tests over the two real breaking commits are its
eval: both go red with the exact rejected field names.
```

## Docker/build/smoke checks

```text
make substrate-ladder-check   (read-only: Postgres, docker ps/inspect/exec, git)
exit 0, GREEN, wall 11.47s
skew rows: drops_fields=6, no_writer=1049, not_running=943, ok=5635, producer_differs_from_main=6
7 containers skipped as having no orion package: pageindex, bus-core, bus-exporter,
signal-gateway redis/otel-collector/grafana/tempo
```

Nothing red today. The FieldStateV1 readers (attention, proposal, feedback, hub) all match the digester field-for-field, so feedback-runtime has been redeployed since the 09-25 finding.

Warnings (not red, no card), all real stale copies:

- `orion/schemas/vision.py`: orion-vision-scribe (image 2026-08-30) drops `VisionEventBundleItem.stream_id` from vision-edge and vision-council. orion-vision-host-qwen (image 2026-09-21, **deployed from the worktree `Orion-Sapienform-vision-host-qwen-athena-t10`**, not main) drops `VisionObject.embedding/embedding_ref/thumb_ref/zone` from vision-edge and vision-window.
- `orion/schemas/telemetry/biometrics.py`: orion-state-service (image 2026-08-30) drops `BiometricsClusterV1.measurements_by_node` from biometrics and cortex-orch.
- Writers whose copy differs from main (reported, not red): spark-concept-induction (`autonomy/models.py`), agent-council (`bus_schemas.py`), graph-compression (`substrate_mutation.py`), embodiment (`journaler/schemas.py`), state-service (`biometrics.py`), vision-host (`vision.py`).

## Review findings fixed

REVIEW_PLACEHOLDER

## Restart required

```text
No restart required. The cron entry runs the script from the checkout, so it picks up this code on the next pull.
```

Redeploys the warnings point at (not done here; for Juniper, from a worktree on current main):

```bash
scripts/safe_docker_build.sh orion-vision-scribe up -d --build
scripts/safe_docker_build.sh orion-state-service up -d --build
# vision-host-qwen runs services/orion-vision-host/docker-compose.circe-qwen.yml from a worktree;
# rebuild it from main with that same compose file, e.g.:
scripts/safe_docker_build.sh orion-vision-host -f services/orion-vision-host/docker-compose.circe-qwen.yml up -d --build
```

## Risks / concerns

- Severity: medium. Concern: a wider net means more possible cards, and card keys changed format (a FieldStateV1 card already delivered would re-send once). Mitigation: red needs a real field mismatch between two running copies. Byte-only differences and loose drops never page.
- Severity: low. Concern: discovery cannot see writers that validate and then persist (for example memory-consolidation's drafts) or that serve over HTTP. Mitigation: the `DECLARED_WRITERS` table and its gate test. Two such writers were found and declared.
- Severity: low. Concern: field comparison resolves inherited fields only through direct imports among the fetched files. A base class reached only through a re-export is left out on both sides equally, so a change there reads as compatible. Mitigation: the bytes are still compared (`schema_sha256`) for producer-vs-main reporting.
- Severity: low. Concern: a writer that dumps with `exclude_none`/`exclude_unset` may never actually send a new optional field, so red there can come early. Mitigation: early is the safe direction for a forbid reader.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
