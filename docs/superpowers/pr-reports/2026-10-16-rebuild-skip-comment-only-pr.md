## Summary

- Post-pull rebuild planner now ignores `.py` files whose AST is unchanged between base and HEAD (comment/whitespace-only edits).
- Docstring edits still count (`__doc__` can be read at runtime). Unparseable or unreadable files still count.

## Outcome moved

Pulling #2724 + #2725 planned 68 services, all from a 4-line comment in `orion/schemas/grammar.py` (reached `orion.core.bus.async_service` through the import graph, added 2026-10-15 in 2f7b6cd00/dd2faddba). Replaying that pull now plans 23 (the real `pre_turn_appraisal.py` change).

## Files changed

- `scripts/rebuild_affected_services.py`: `changed_files` uses `--name-status`; `_same_python_ast` drops comment-only `M` files.
- `tests/scripts/test_rebuild_affected_services.py`: 4 tests (comment-only skipped, real change kept, docstring kept, unparseable kept).

## Tests run

```text
pytest tests/scripts/test_rebuild_affected_services.py tests/scripts/test_rebuild_services_from_git_diff.py -q
78 passed
```

## Evals / Docker

No eval harness for this script; no runtime/Docker change. No env, schema, or bus changes. No restart required.

## Risks / concerns

- Severity: low. Mitigation: any doubt keeps the file as changed.
- Follow-up: `pre_turn_appraisal.py` still fans out to 23 services (real change, not addressed).
