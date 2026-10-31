## Summary

- The post-pull rebuild planner now skips a `.yaml`/`.yml` file whose change is comments, whitespace or quoting only, the same way it already skipped comment-only `.py` edits.
- "Same" means same documents as ordered, typed JSON: `1` vs `true` vs `1.0`, key order, and an added duplicate key all still count as changes.
- Any doubt (parse error, custom tag, no PyYAML on the host python) keeps the file counted as changed.

## Outcome moved

A comment added to `orion/cognition/verbs/stance_react.yaml` (PR #3027) rebuilt 13 services on athena (2026-10-31 `git pull`). Replaying that pull with this patch plans 2: orion-thought (code changed) and orion-cortex-exec (its image copies its own tests/ folder, so a new test file does change it).

## Current architecture

`scripts/rebuild_affected_services.py` `changed_files()` lists files from `git diff --name-status base..HEAD`, skipping `M` `.py` files whose AST is unchanged (`_same_python_ast`). Every other changed file fed the path/COPY/mount rules; `orion/` is COPYed into most images.

## Architecture touched

- `scripts/rebuild_affected_services.py`: `_base_and_head_blobs`, `_same_yaml_data`; `changed_files` YAML branch.

## Files changed

- `scripts/rebuild_affected_services.py`: YAML comment-only skip.
- `tests/scripts/test_rebuild_affected_services.py`: 6 new tests.

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

```text
tests/scripts/test_rebuild_affected_services.py tests/scripts/test_rebuild_services_from_git_diff.py: 147 passed
Replay `--list-only --base 7458776dd` (the pull that rebuilt 13): orion-cortex-exec, orion-thought
  (same under /usr/bin/python3, which the post-merge hook uses)
```

## Evals run

No eval harness for the rebuild planner; the replay above is the behavioural check.

## Docker/build/smoke checks

None needed: host-side script, no service image changes.

## Review findings fixed

- Finding: `==` on parsed data calls `1 == True == 1.0` and ignores key order, so real value changes would be skipped.
  - Fix: compare `json.dumps(docs, sort_keys=False)` per side.
  - Evidence: `test_yaml_type_and_order_changes_are_real`.
- Finding: unguarded `import yaml` would crash the whole planner on a host python without PyYAML.
  - Fix: import inside the try; ImportError keeps the file counted.
  - Evidence: `test_missing_yaml_package_keeps_the_file`.
- Finding: `safe_load` silently accepts duplicate keys, which compose rejects.
  - Fix: loader that raises on a duplicate key -> file counted as changed.
  - Evidence: `test_yml_suffix_and_duplicate_keys`.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low
- Concern: a consumer that reads a YAML file as raw text (not parsed) would miss a comment-only edit. Review found none tracked.
- Mitigation: comment-only means no data change to any YAML parser; block-scalar comments are string content and still count.
