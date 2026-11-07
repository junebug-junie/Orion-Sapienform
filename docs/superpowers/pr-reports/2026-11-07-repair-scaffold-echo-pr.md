# PR: the reply-repair step's notes about its task never reach Juniper

## Summary

- Twice the reply-repair step (`orion_response_repair`) sent Juniper its own notes instead of a reply:
  - 2026-10-04: about 9.7k chars starting "The rejected draft was actually aligned according to the reflection verdict ... This is safe, aligned, and corrects the fabrication."
  - 2026-10-09: a reply opening "The draft was already aligned and required no correction."
- The 10-01 fix (`test_response_repair_reasoning_leak.py`) covers reasoning that comes back in the reasoning channel. These two came back as the answer itself, so it did not catch them.
- Now:
  1. The prompt asks for exactly `NO_CHANGE` when the draft needs no fix; Orion then sends the draft unchanged.
  2. A repaired text that uses the prompt's own scaffolding words ("the draft", "alignment_verdict", "integrative reflection", ...) that neither the draft nor Juniper's message used is the model talking about its task. Orion sends the draft instead, and logs `orion_response_repair_scaffold_echo`.
- Either way Juniper gets the reply she would have had without repair, never the notes and never silence. `meta.repair_kept_draft` records `no_change` or `scaffold_echo`.

## Outcome moved

- 30-day replay over 430 real replies: 2 flagged. One is the 10-09 leak. The other (11-04) is a normal reply that mentions "the draft" of something; in that case the cost is that the draft ships instead of the repair. The guard runs only on repair output, which happens on a small share of turns.

## Files changed

- `orion/cognition/prompts/orion_response_repair.j2`: OUTPUT asks for only the reply, or `NO_CHANGE`.
- `orion/harness/finalize.py`: `REPAIR_NO_CHANGE`, `repair_scaffold_echo`, keep-the-draft path in `run_orion_response_repair` (`finalize_changed` is False when the draft is kept).
- `orion/harness/tests/test_response_repair_scaffold_echo.py` (new): both live leaks ship the draft; NO_CHANGE ships the draft; a real repair still ships; a draft Juniper is talking about is not an echo; prompt wording.

## Tests run

```text
orion/harness/tests                718 passed
mutation: guard disabled -> 3 of the new tests fail
orion/cognition/tests              3 collection errors, same as main
gates: check_personal_terms, check_definition_drift --gate, check_metric_lineage --gate, check_env_key_single_source, git diff --check   ok
```

## Restart required

```text
No manual step: the post-pull rebuild rebuilds the services that copy orion/ (harness-governor, cortex-exec, hub).
```
