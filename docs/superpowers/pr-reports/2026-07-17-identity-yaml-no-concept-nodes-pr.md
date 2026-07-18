# Fix: identity_yaml adapter no longer reifies policy prose as concept nodes

PR: https://github.com/junebug-junie/Orion-Sapienform/pull/1165
Branch: `fix/identity-yaml-no-concept-nodes`

Supersedes: PR #1162 (`fix/concept-induction-pg-chat-source`, closed — built on a wrong
root-cause diagnosis, see below).

## What actually happened this session

Juniper reported chat-stance artifacts bleeding into "the new orion concept induction" —
concrete example: "speak as an ongoing presence rather than customer support software"
appearing as if it were a graph concept.

**First pass (wrong):** traced the bus-envelope → concept-induction extraction path
(`orion-spark-concept-induction`) and shipped PR #1162 to read chat-history text from
Postgres instead of the bus envelope. Juniper corrected this directly: "nowhere in
chathistory log doe the prompts contain any of the stance shit." Checked live data (should
have done this first): recent `chat_history_log` rows are clean (verified via direct
`psql`); the old April rows that did match were a different, already-fixed bug (raw
`<think>` reasoning trace bleeding into the `response` column before `thought_process`
was split into its own column). PR #1162 closed.

**Second pass (correct):** queried the live FalkorDB `orion_substrate` graph directly
(`redis-cli -h localhost -p 6380 GRAPH.QUERY orion_substrate ...`) instead of guessing from
code, and found the actual node:

```
label: "speak as a real ongoing presence rather than customer-support software"
node_kind: concept
provenance_producer: identity_yaml_adapter
provenance_source_kind: identity_yaml
promotion_state: canonical
observed_at: 2026-07-17T18:29:03Z
```

Traced to `orion/substrate/relational/adapters/identity_yaml.py:map_identity_yaml_to_substrate`,
which turned every line of `ctx["orion_identity_summary"]` into a `ConceptNodeV1`. That
ctx key is built by `orion/cognition/personality/identity_context.py:build_identity_context`
as a flat merge of `orion_identity.yaml`'s `nature`, `core_drives`, `self_permissions`, and
`anti_patterns` blocks — confirmed the exact source line in
`orion/cognition/personality/orion_identity.yaml`:

```yaml
core_drives:
  - "speak as a real ongoing presence rather than customer-support software"
```

An operator-authored behavioral directive, mechanically reified as a graph "Concept" node
with equal standing to genuine self-model facts.

## Fix

Removed the `ConceptNodeV1` loop from `map_identity_yaml_to_substrate` entirely, per
Juniper's explicit scope ("I only want the chat history going in, not this other
bullshit" — concepts in the graph should only come from real induction, never from
operator config). The `StateSnapshotNodeV1` (the thing `chat_stance.py`'s stance-brief
compilation actually reads) is unchanged, so prompt assembly is unaffected. Confirmed via
subagent review: no other consumer depends on identity_yaml producing concept-kind nodes
specifically.

## Live cleanup (pending)

14 already-materialized bad nodes remain in the live `orion_substrate` FalkorDB graph
(10 from the real `orion_identity.yaml`, 4 from `chat_stance.py`'s
`FALLBACK_ORION_IDENTITY_SUMMARY` constants, all written 2026-07-17). The code fix only
stops new ones. Snapshot taken to `/tmp/identity-yaml-concept-cleanup/snapshot_before_delete.txt`
before any deletion; deletion itself needs an explicit go-ahead (destructive action on
shared live infra) — asked, not yet executed as of this report.

## Pre-existing flake found, not fixed

`services/orion-cortex-exec/tests/test_chat_stance_brief.py::test_build_chat_stance_inputs_falls_back_when_identity_missing`
fails when run after other tests in specific worktree environments. Confirmed via
`git stash` in the same worktree that this is unrelated to this fix (reverting the fix
in-place makes it *worse* — 3 failures instead of 1 — while the primary checkout stays
clean at 0). Looks like `CognitiveUnificationLayer`'s `ThreadPoolExecutor`-based fan-out
(`orion/substrate/relational/layer.py`) interacting with the module-level
`_UNIFICATION_LAYER` singleton in `chat_stance.py`. Not investigated further — flagging
for a separate session.

## Status

DONE_WITH_CONCERNS — code fix complete, reviewed, tested (127 passed, scoped to
`orion/substrate/relational/tests`), pushed, PR open. Concerns: (1) live graph cleanup of
the 14 existing bad nodes still needs explicit go-ahead and execution, (2) pre-existing
test flake found but not fixed (out of scope, documented).
