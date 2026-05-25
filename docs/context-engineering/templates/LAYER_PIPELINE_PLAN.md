# Layer Pipeline Plan: <service>

## Goal

Move `<service>` substrate-relevant transitions through the appropriate parts of the 1–11 substrate.

## Layer 1: Service-owned substrate trace emission

Status: planned / implemented / not applicable

Files:

```text
services/<service>/app/grammar_emit.py
services/<service>/app/grammar_publish.py
```

Semantic roles:

- <role>

## Layer 2: Trace substrate persistence

Status: planned / implemented / not applicable

Expected table:

```text
grammar_events
```

Proof query:

```sql
select * from grammar_events where source_service = '<service>' order by created_at desc limit 10;
```

## Layer 3: Reducer

Needed? yes/no

If yes:

```text
orion/substrate/<domain>/
```

Reducer output:

```text
StateDeltaV1
ReductionReceiptV1
```

## Layer 4: Field digestion

Needed? yes/no

Pressure hints:

```yaml
example_pressure: 0.0
```

Field channels:

```yaml
example_pressure: example_field_channel
```

## Layer 5: Attention

Does this service affect attention? yes/no

Expected attention targets:

- <target>

## Layer 6: Self-state

Does this service affect self-state? yes/no

Expected dimensions or labels:

- <dimension_or_label>

## Layer 7: Proposal

Could this service create proposal pressure? yes/no

Expected proposal types:

- <proposal>

## Layer 8: Policy

Does this service require policy gating? yes/no

Policy gates:

- read_only
- operator_review
- execution_policy

## Layer 9: Dispatch

Can this service be an effector or dispatch target? yes/no

Default stance:

```text
dry_run unless explicitly approved
```

## Layer 10: Feedback

What consequences or absences should be captured?

- <feedback observation>

## Layer 11: Consolidation

What repeated patterns should become motifs or expectations?

- <motif>

## Out of scope

- <explicit non-goal>

## Live proof checklist

- [ ] Layer 1 trace observed
- [ ] Layer 3 receipt observed, if applicable
- [ ] Layer 4 field perturbation observed, if applicable
- [ ] Layer 5 attention observed, if applicable
- [ ] Layer 6 self-state observed, if applicable
- [ ] Layer 7 proposal observed, if applicable
- [ ] Layer 8 policy decision observed, if applicable
- [ ] Layer 9 dispatch frame observed, if applicable
- [ ] Layer 10 feedback observed, if applicable
- [ ] Layer 11 motif observed, if applicable
