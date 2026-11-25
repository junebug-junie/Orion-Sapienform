## Summary

- New ChatGPT exports split conversations into `conversations-000.json` ... `conversations-NNN.json`. The importer only knew the single `conversations.json`.
- The zip matcher used `endswith("conversations.json")`, so it picked `sectioned_conversations.json` (an empty list) and reported success with 0 conversations.
- The importer now matches exact file names and merges shards in number order, from one shard folder.
- If an export has a single `conversations.json`, it still wins.

## Outcome moved

The 2026-11-24 export (930 conversations) now loads. Before this patch it loaded 0 rows and reported success.

## Current architecture

`scripts/import_chatgpt_export.py` -> `orion.importers.chatgpt_export.resolve_export_path` -> one `conversations.json` -> bus fanout -> sql-writer writes the `chat_gpt_*` tables.

## Architecture touched

Only `resolve_export_path` and its helpers. It still returns one `conversations.json` path, so callers do not change.

## Files changed

- `orion/importers/chatgpt_export.py`: exact-name match, shard detection, number-order merge.
- `tests/test_chatgpt_export_resolve.py`: 7 regression tests.

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

```text
pytest tests/test_chatgpt_export_resolve.py services/orion-sql-writer/tests/test_chatgpt_import_layers.py -q
13 passed
```

## Evals run

No eval harness for the importer. This is a file-discovery fix. The live load below is the proof.

## Docker/build/smoke checks

Live load of the real 2026-11-24 export (`--force-full`, no embeddings). Dry run: conversations_total=930, messages_published=28164, turns_published=14082. Rows landed in `chat_gpt_message`, `chat_gpt_log` and `chat_gpt_conversation`.

## Review findings fixed

- Finding: shards sorted as text (`-10` before `-9`).
  - Fix: sort by the shard number.
  - Evidence: `test_shards_sort_by_number_not_text`.
- Finding: a zip with shards in two folders would merge both sets.
  - Fix: use only the shallowest folder that holds shards.
  - Evidence: `test_zip_in_top_folder_uses_one_shard_set`.
- Finding: no test where a single file and shards are both present.
  - Fix: added `test_single_file_wins_over_shards`.
- Finding: the docstring said "file", but a single shard file is not accepted.
  - Fix: the docstring now names the accepted inputs.
- Not changed: for a sharded export, the import record holds the merged temp file's path, size and hash. Zip exports already worked this way, and re-runs still give the same `import_run_id`.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low
- Concern: all shards are held in memory while they merge (about 190 MB of JSON). The old single file was also loaded whole.
- Mitigation: none needed at this size.
- Issues opened/closed: none.

## PR link

See PR.
