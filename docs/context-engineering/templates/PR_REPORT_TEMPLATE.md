# PR Report: <service> substrate trace adoption

## Branch

`<branch>`

## Summary

Describe what changed in one paragraph.

## Service role classification

| Port | Role | Trace required | Notes |
|---|---|---:|---|
| <port> | <role> | yes/no | <notes> |

## Semantic roles emitted

| Semantic role | When emitted | Layer | Evidence refs |
|---|---|---|---|
| <role> | <condition> | <layer> | <refs> |

## Architecture

```text
<service native transition>
  -> substrate trace event (GrammarEventV1)
  -> grammar_events
  -> optional reducer
  -> optional state delta / field / frame / feedback / consolidation
```

## Files changed

| Path | Change |
|---|---|
| <path> | <change> |

## Redaction guarantees

This PR does not emit:

- raw model prompts;
- raw model completions;
- credential material;
- full payload blobs;
- full private payloads;
- database connection material.

It emits ids, refs, bounded summaries, and redacted error classes.

## Tests

```bash
<commands>
```

Results:

```text
<results>
```

## Live proof

SQL/API/log proof:

```text
<proof>
```

Status:

```text
unit_tested / integration_tested / live_smoked / observed_in_sql / observed_in_hub / not_verified_live
```

## Downstream stages

| Layer | Status | Notes |
|---|---|---|
| 1 Trace emission | implemented/planned/not applicable | |
| 2 Trace persistence | implemented/planned/not applicable | |
| 3 Reducer | implemented/planned/not applicable | |
| 4 Field digestion | implemented/planned/not applicable | |
| 5 Attention | implemented/planned/not applicable | |
| 6 Self-state | implemented/planned/not applicable | |
| 7 Proposal | implemented/planned/not applicable | |
| 8 Policy | implemented/planned/not applicable | |
| 9 Dispatch | implemented/planned/not applicable | |
| 10 Feedback | implemented/planned/not applicable | |
| 11 Consolidation | implemented/planned/not applicable | |

## Known gaps / follow-ups

- <gap>

## Non-goals respected

- No central emitter service.
- No behavior change when publishing disabled.
- No raw payload mirroring.
- No downstream mutation unless explicitly implemented and tested.
