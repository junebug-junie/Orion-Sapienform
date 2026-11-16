# Every prediction-error reader says how it treats alarms (#3307, #3513)

## Summary

- Cabinet warming, strain news and street events are **alarms**: their value means "it fired", not "how big a surprise". They still travel on the node `prediction_error` field, the history table and the field channel of the same name.
- A new CI gate, `scripts/check_pe_alarm_readers.py`, fails on any Python file that touches `prediction_error` unless the file either skips alarms in code (`is_alarm_node` / `ALARM_NODE_IDS`) or is registered with a class and a reason in `config/substrate/prediction_error_readers.yaml`. 64 files are accounted for, each classified by reading its code.
- Two latent readers that ranked alarms as surprise sizes are fixed:
  - **The field merge** (`orion/field/pressure.py`, and its mirror in `commensurability.py`) max-merged the field `prediction_error` channel across nodes, so a firing alarm (1.0) would win it. Alarm nodes now never compete for that channel, and they never empty it either.
  - **`magnitude_scorer`** ranked any ids it was handed. It now skips alarms.
- `ALARM_NODE_IDS` / `is_alarm_node` move to a stdlib-only module, `orion/prediction/alarm_nodes.py`, re-exported from `prediction_error_freshness`. This lets the light field module ask the question without importing the heavy `orion.substrate` package.

## Outcome moved

- A new reader that forgets alarms is a CI failure, not a review hope. (Ticket option (a); option (b), moving alarms off the field, is not needed for safety now.)
- Evidence that the gap was latent, not live: over the full 3-day retention of `substrate_field_state` (119k ticks), no alarm node ever appears in the field. Over the last 24 h, 38,008 ticks: 0 where an alarm won `prediction_error`. Both fixes guard the day alarms do reach the field. They change nothing that runs today.

## Current architecture

- Alarms were re-filed as alarms in #3314.
- Readers asked `is_alarm_node()` by convention, with no check.
- The field merge and the society-of-mind scorer did not ask.

## Architecture touched

- `orion/field` (merge rule)
- `orion/prediction` (new alarm-id module)
- `orion/substrate/prediction_error_freshness.py` (re-export)
- `orion/attention/field_attention/candidate_society_of_mind.py`
- the static gates workflow
- one new config registry

No bus, schema or env changes.

## Files changed

- `scripts/check_pe_alarm_readers.py`:
  - The gate. TOUCH matches the quoted key anywhere, `.prediction_error`, or the history table.
  - The auto-pass is judged on code tokens only, and fails closed on files that don't parse.
  - Stale or redundant registry entries fail.
  - Excluded paths must be narrow, existing directories.
- `config/substrate/prediction_error_readers.yaml`: 51 registered files (`alarm_aware` / `single_node` / `not_a_reader`) and one exclusion (`scripts/analysis/`, offline). The other 13 pass automatically.
- `.github/workflows/orion-static-gates.yml`: runs the gate and its tests.
- `orion/field/pressure.py`:
  - `skips_merge` drops alarm nodes from the `prediction_error` merge.
  - `_stale_node_channels` ignores those pairs, so a fresh alarm can't arm the exclusion of stale real sources.
- `orion/field/commensurability.py`: the mirror uses the same `skips_merge`.
- `orion/prediction/alarm_nodes.py`: new, stdlib only.
- `orion/substrate/prediction_error_freshness.py`: re-exports the ids and points at the gate.
- `orion/attention/field_attention/candidate_society_of_mind.py`: `magnitude_scorer` skips alarms.
- Tests:
  - `tests/test_check_pe_alarm_readers.py` (12): every key spelling, comment-only mentions, unparseable files, stale entries, path rules, and the checked-in repo passing.
  - `tests/test_field_pressure_provenance.py`: an alarm never wins the merged channel, merge plus mirror.
  - `tests/test_stale_source_merge.py`: a fresh alarm does not empty the channel.
  - `tests/test_attention_candidate_society_of_mind.py`: the scorer skips alarms.

## Schema / bus / API changes

- None. Behaviour changed: the field `prediction_error` channel's merge ignores alarm nodes, with no live effect today (above).

## Env/config changes

- No env keys. New config file `config/substrate/prediction_error_readers.yaml` (read by CI only).

## Tests run

```text
python3 scripts/check_pe_alarm_readers.py -> OK (64 files touch prediction_error, all accounted for)
pytest tests/test_check_pe_alarm_readers.py tests/test_field_pressure_provenance.py tests/test_merge_commensurability.py
       tests/test_stale_source_merge.py tests/test_pe_alarm_refile.py tests/test_field_channel_glossary.py
       tests/test_attention_candidate_society_of_mind.py -> 105 passed
21 root test files touching orion.field / orion.prediction / prediction_error_freshness -> 449 passed; the only failure,
  test_retired_pe_step5_readers (no sqlalchemy in the image), is the same on main
services/orion-field-digester tests -> the same 5 failures as main (test_worker_anomaly_wiring, ...), none new
mutation checks: a removed registry entry fails; the old fail-open tokenize fails the new test
```

## Evals run

```text
No eval harness for a static gate. Live evidence instead (read-only SQL on substrate_field_state, above).
```

## Docker/build/smoke checks

```text
The field digester and Hub images copy all of orion/, so the new orion/prediction/alarm_nodes.py ships. No dependency changes.
```

## Review findings fixed

Round 1 (the gate):

- Finding: a real cross-node reader of alarms that the gate could not see. `orion/field/pressure.py` max-merged the field `prediction_error` channel, so a firing alarm would win it, and the registry's reason for `state_deltas.py` was false.
  - Fix: `skips_merge` in the merge and its mirror; reason corrected.
  - Evidence: `test_an_alarm_never_wins_the_merged_prediction_error_channel`.
- Finding: the TOUCH regex missed `getattr`, `in`-membership, SQL `->>`, constants and f-strings.
  - Fix: match the quoted key anywhere; 16 more files classified.
  - Evidence: `test_every_spelling_of_the_key_counts_as_touching`.
- Finding: the auto-pass matched comments and docstrings.
  - Fix: code tokens only.
  - Evidence: `test_a_mention_in_a_comment_or_docstring_is_not_a_filter`.
- Finding: excluded paths used a raw prefix match.
  - Fix: a trailing `/` is required, a whole scan root is refused, the path must exist.
  - Evidence: `test_excluded_paths_must_be_narrow_existing_directories`.

Round 2 (the field merge):

- Finding: a fresh alarm still counted as the fresh source in `_stale_node_channels`. With every real prediction-error node stale, the merged channel vanished, a made-up absence.
  - Fix: skipped pairs never touch the staleness sets.
  - Evidence: `test_a_fresh_alarm_neither_wins_nor_empties_the_prediction_error_channel`.
- Finding: an unparseable file auto-passed on a comment.
  - Fix: `_code_only` fails closed.
  - Evidence: `test_an_unparseable_file_never_auto_passes_on_a_comment`; the old behaviour fails it.
- Checked with no finding: every other field consumer (dimension maps, `winning_write_time`, Hub glossary routes) goes through the same merge; the re-export is not monkeypatched anywhere; 3.12 f-string tokens are handled.

## Restart required

```text
No manual restart. The post-pull rebuild picks up orion-field-digester / Hub / others as usual; no behaviour change is live today.
```

## Risks / concerns

- Severity: low. Concern: the gate over-includes. Any file that names the key in a comment must be classified. Mitigation: the classification is one line, and over-inclusion is the safe direction.
- Severity: low. Concern: a file that calls `is_alarm_node` only for a label but still ranks alarms auto-passes. Mitigation: documented in the gate's docstring ("the author decided", not a proof); reviewers see the call.
- Issues: Fixes #3307. Fixes #3513 (the field-channel readers are now under the gate, and the one real merge reader is fixed).

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
