## Summary

- Orion can now read an internal text document by absolute path, for example a markdown spec. It works from the Reading tab and from chat (`recommend_reading`).
- Hub reads the file once, at acceptance. It stores the exact text by hash in `reading_document_snapshot` and pins the source as `file:///abs/path?sha256=<hex>`.
- Stage 1 hands those exact bytes to the reader. Hub itself writes the proof that the document was read (`orion_document_snapshot`), and only after checking the text is in the bound prompt, so a model tool call cannot forge it.
- An unchanged file comes back as `already_read`, and an edited file is a new read. A status lookup by the bare path finds any version.
- Only allowlisted folders, text types and sizes up to 48 KB are read. Secrets, `.git`, symlinks that point out of the folders, and swaps between the check and the open are all refused. Larger files are refused, never truncated.

## Outcome moved

The Reading pipeline could only read public web pages. Now Orion can read their own design docs through the same evidence-gated path. The eval shows 279 of 282 real specs (98.9%) are readable under the defaults.

## Current architecture

`ReadingRequestedV1.url` was an `HttpUrl`, and `validate_source_url` refused anything that was not public http(s). Stage 1 told the reader to fetch the URL, and read evidence was the reader's own fetch tool calls. Dedup was keyed by the seed `url` column.

## Architecture touched

- Shared library: `orion/world_pulse_read` (the new `documents.py`, plus `queue`, `read_evidence`, `tools`, `operator`, `urls`, `verify`), `orion/schemas/reading.py`, `orion/harness/reading_receipts.py`, `orion/introspect/tools.py`.
- Hub: the Stage 1 pipeline, the reading listener, routes, settings, `main.py`, and the Reading tab template and JS.
- Postgres: a new `reading_document_snapshot` table (created at startup by `ensure_seed_queue_schema`, with a manual migration mirror).

## Files changed

- `orion/world_pulse_read/documents.py`: new. Document policy, pure ref parsing, safe capture, snapshot storage.
- `orion/world_pulse_read/queue.py`: `accept_source` (capture or pinned-provenance check); `reading_status` matches any version for a bare path.
- `orion/world_pulse_read/read_evidence.py`: document seeds accept only Hub's snapshot evidence.
- `orion/schemas/reading.py`: `url` accepts `file:///`; optional `content_sha256` on evidence, omitted when unset.
- `orion/world_pulse_read/{tools,operator,urls,verify}.py`, `orion/harness/reading_receipts.py`, `orion/introspect/tools.py`: switched to `normalize_reading_source`; the tool description covers document paths.
- `services/orion-hub/scripts/world_pulse_read_pipeline.py`: document Stage 1 prompt, bound-prompt check, Hub evidence.
- `services/orion-hub/scripts/{reading_listener,world_pulse_read_routes,main}.py`, `app/settings.py`: pass the document policy through; return policy codes.
- `services/orion-hub/{templates/reading.html,static/js/reading.js}`: path input, refusal messages, document labels, snapshot badge.
- `services/orion-hub/.env_example`, `README.md`: 3 new keys, plus docs.
- `services/orion-sql-db/manual_migration_reading_document_snapshot_v1.sql`: migration mirror.
- Tests and eval: `orion/world_pulse_read/tests/test_documents.py`, Hub reading tests, `services/orion-hub/evals/test_reading_document_eval.py`.
- `docs/superpowers/specs/2026-09-28-reading-internal-documents-design.md`: design, privacy boundary, rollback.

## Schema / bus / API changes

- Added: `SourceFetchEvidenceV1.content_sha256` (optional; omitted from JSON when unset). The `reading_document_snapshot` table.
- Removed: none.
- Renamed: none.
- Behavior changed: `ReadingRequestedV1.url` also accepts `file:///abs/path[?sha256=]`. New refusal codes `document_*` and `no_read_evidence:document_not_in_prompt`.
- Compatibility notes: web payloads serialize byte-identically (tested). No channel changes. Stage 2 re-entry stays web-only.

## Env/config changes

- Added keys: `HUB_READING_DOCUMENT_ROOTS`, `HUB_READING_DOCUMENT_EXTENSIONS`, `HUB_READING_DOCUMENT_MAX_BYTES` (orion-hub).
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes (`services/orion-hub/.env_example`).
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (the 3 keys are in the live Hub `.env`). The script exits 2 only because of values that had already diverged in other services.
- skipped keys requiring operator action: none.

## Tests run

```text
pytest orion/world_pulse_read orion/introspect orion/harness services/orion-hub/{tests,evals} -k "read or pulse or introspect or receipt or document or settings or main"
  677 passed, 67 skipped, 1 failed (test_fcc_model_labels_reads_fixture_env -- fails on main too)
RUN_READING_POSTGRES=1 pytest services/orion-hub/tests/test_reading_postgres.py   44 passed
node --test services/orion-hub/static/js/reading.test.js                          12 passed
scripts/check_env_template_parity.py PASS; check_env_key_single_source OK;
check_async_routes_not_blocking, check_metric_lineage --gate, check_definition_drift --gate,
check_chat_route_poachers, check_compose_no_relative_mounts, check_scripts_dir_no_stdlib_shadow: PASS
tests/test_agent_trace_schema_registry.py 2 passed
```

Also failing on main and unrelated: `test_agent_trace_debug_panel.py::test_memory_and_autonomy_modals_coordinate_scroll_lock_and_visibility`.

## Evals run

```text
pytest services/orion-hub/evals/test_reading_document_eval.py -s
reading_document_eval specs=282 accepted=279 share=0.989 refusals={'document_too_large': 3} longest_prompt_bytes=50424
```

## Docker/build/smoke checks

```text
Not run. The default roots are both mounted into Hub already
(docker-compose.yml: repo at /mnt/scripts/Orion-Sapienform:ro, sandbox at /mnt/orion-fcc).
Live path UNVERIFIED until Hub is rebuilt and one document read lands.
scripts/safe_graphify_update.sh: OK, node count 77966 -> 90228.
```

## Review findings fixed

- Finding: a `file://...?sha256=` ref skipped every path check. A known hash could claim a verified read of any path, even with reading turned off.
  - Fix: pinned refs pass `check_document_path` (roots, denylist, extension, on/off switch). They are accepted only if Hub captured those bytes from that exact path (`PINNED_SNAPSHOT_SQL`: the snapshot's `first_source`, or an existing seed row).
  - Evidence: `test_a_pinned_document_ref_still_needs_policy_and_provenance`, plus the real-Postgres case.
- Finding: checks ran on the path before `open()`, so a symlink or named-pipe swap in a writable root could read outside the roots or hang the worker.
  - Fix: open with `O_NOFOLLOW|O_NONBLOCK`, `fstat` for a regular file, and require the opened file's real path to equal the checked path (`document_changed_during_read`).
  - Evidence: `test_open_judges_the_file_it_opened_not_the_path_it_checked` (FIFO, final symlink, directory symlink).
- Finding: a lookup by a symlinked path or a `//`-prefixed path returned `not_found`.
  - Fix: `normalize_document_ref` resolves symlinks, and parsing collapses a leading `//`.
  - Evidence: `test_lookups_normalize_to_the_stored_form`.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

The snapshot table is created on Hub startup. To apply it by hand instead:
`psql "$DSN" -f services/orion-sql-db/manual_migration_reading_document_snapshot_v1.sql`.

## Risks / concerns

- Severity: low
  - Concern: document reads spend the same Wallet A read slots as operator reads.
  - Mitigation: intended; reads are paced by GPU admission.
- Severity: low
  - Concern: live path not yet verified.
  - Mitigation: after the rebuild, paste a spec path into the Reading tab and confirm the "Hub document snapshot" badge. A second submit should return `already_read`.
- Rollback: set `HUB_READING_DOCUMENT_ROOTS=` (empty) and recreate Hub.

## PR link

(filled in on creation)
