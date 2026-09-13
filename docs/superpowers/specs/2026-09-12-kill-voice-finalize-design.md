# Kill mandatory voice finalize — design

**Date:** 2026-09-12  
**Status:** implemented  
**Incident seed:** corr `18b2540e-8499-40df-817b-4986fea84915` — motor draft `hey. i'm here. what's on your mind?` rewritten by 5c into assumption / next-concrete-move scaffold despite `aligned` + task-tracking hazard.

## Arsonist summary

Mandatory `orion_voice_finalize` (5c) is redundant style laundering. The FCC motor already receives Orion identity, stance, hazards, repair overlays, and conversation context. 5c mostly re-asks another LLM to “refine voice and rhythm,” which has caused concrete harm (aligned relational drafts destroyed; reading JSON rewritten to prose, requiring a special-case bypass).

**Kill voice finalization as a normal stage.** Keep 5a/5b as the post-draft evaluator. Run a *minimal* response-repair LLM only when that evaluator explicitly rejects the draft. Exact draft passthrough when the draft is accepted.

## Current architecture

```text
FCC motor → draft
  → 5a substrate finalize appraisal
  → 5b integrative reflection (optional 5b-prime tool retry)
  → 5c orion_voice_finalize on nearly every ordinary turn
  → 6b outcome + closure
```

Exceptions today:

- `preserve_structured_output` / reading_only: skip prose 5c; canonicalize JSON
- 5c failure after appraisal+reflection: fail-closed artifacts (do not silently ship)

Known choke points:

- `orion/harness/finalize.py::run_harness_finalize_chain` always calls `run_orion_voice_finalize` when not structured
- `orion/harness/prefix.py::_format_grounding_self_block` **withholds** `response_policy_summary` from the motor (“reserved for voice finalize”)
- `HarnessRepairOverlayV1.finalize_overlay` is a second-pass-only instruction string compiled in `orion/harness/repair.py`
- `surprise_resolved` incorrectly requires `finalize_changed or surprise_level < epsilon` (`emit_turn_outcome_molecule`)

## Problem

1. Motor already has enough speech policy to speak as Orion.
2. 5c has no unique job on `aligned` turns except polish risk.
3. `finalize_changed` only proves string mutation, not improvement.
4. Overlay-as-repair-trigger would recreate a second writer even when 5b says the motor already obeyed.
5. Surprise/closure semantics currently treat “no rewrite” as unresolved surprise when surprise_level is high — wrong once passthrough is the happy path.

## Target architecture

```text
FCC motor (includes all speech / response policy + former finalize_overlay content)
        ↓
5a appraisal
        ↓
5b reflection / optional 5b-prime
        ↓
structured output
    → validate and canonicalize; no prose repair

aligned AND NOT strain_unresolved
    → exact draft passthrough (no repair LLM)

misaligned OR uncertain OR strain_unresolved
    → minimal orion_response_repair
        ↓
6b outcome + closure
```

## Repair gate (deterministic)

```python
needs_repair = (
    reflection.alignment_verdict in {"misaligned", "uncertain"}
    or reflection.strain_unresolved
)
```

**Not** in the gate:

- Non-default repair overlay mode alone
- `finalize_changed` history
- High `surprise_level` alone (5b decides whether the draft handled it)

If `needs_repair` is false: `final_text = draft_text` byte-identical (after any reading-receipt grounding that already applies to both paths), `finalize_changed=false`, `response_repair_ran=false`.

## Overlay handling

Today the motor already receives `mode`, `prefix_overlay`, and `rule_lines` via the harness prefix. If it follows them and 5b returns `aligned`, a second LLM is redundant.

**Change:**

1. Move any uniquely useful content that today lives only in `finalize_overlay` into the **motor** prefix (alongside existing overlay fields).
2. Delete `finalize_overlay` as a second-pass-only concept (schema field may remain temporarily unused then removed in the same PR if cheap; prefer delete over leave-dead).
3. Overlay must **not** independently force `needs_repair`.

## Surprise resolution

Replace the mutation-coupled formula:

```python
# REMOVE dependence on finalize_changed / surprise epsilon for resolution
surprise_resolved = (
    not repair_failed  # aka not finalize_failed for the repair/passthrough path
    and reflection.alignment_verdict == "aligned"
    and not reflection.strain_unresolved
)
```

Reflection is the evidence that surprise was resolved — not string mutation. An aligned high-surprise draft returned unchanged must yield `surprise_resolved=true` so closure and N+1 strain are not contaminated.

## Verb: semantic deletion, not a moustache rename

### Remove

- Cognition verb `orion_voice_finalize` from registry and route configuration
- Prompt `orion_voice_finalize.j2` (or replace entirely; do not leave as alias)
- Any runtime ability to invoke the old verb

### Add

- Verb `orion_response_repair`
- Prompt `orion_response_repair.j2` with **only**:

  - original user message
  - draft
  - alignment verdict and notes
  - unresolved strain
  - factual/tool receipts needed for preservation
  - instruction: make the **smallest necessary correction**; do not polish, restyle, or rewrite an already-acceptable answer

### Explicitly absent from the repair prompt

- “refine voice and rhythm”
- general style teaching / Orion voice curriculum
- duplicated identity / relationship grounding (motor already had it)
- default polishing
- any instruction to rewrite an already-aligned answer

### Historical compatibility (readers only)

```text
historical phase / grounding_status containing orion_voice_finalize
  → interpret as response_repair failure (or legacy voice-pass failure)
```

No callable old-verb alias. No dual registration.

## Observability

Keep `finalize_ran=true` meaning **5a/5b finalization completed** (appraisal + reflection path ran). Document that meaning change explicitly wherever operator docs or Hub UI copy imply “voice pass ran.”

Add on `HarnessRunV1` (and outcome molecule if consumers need it):

| Field | Meaning |
|-------|---------|
| `response_repair_ran: bool` | Repair LLM was invoked |
| `response_repair_reason: str \| None` | Why: e.g. `misaligned`, `uncertain`, `strain_unresolved`, or `null` when skipped |
| `finalize_changed: bool` | Draft text ≠ final text (still useful; no longer the surprise proxy) |

Passthrough: `finalize_ran=true`, `response_repair_ran=false`, `finalize_changed=false`.  
Repair that returns identical text: `response_repair_ran=true`, `finalize_changed=false`.  
Those two cases must be distinguishable.

Log line for skip: `response_repair_skipped reason=aligned` (or equivalent).

## Fail-closed

If `needs_repair` and the repair call fails (timeout, error-shaped text, missing final_text): **do not publish the known-bad draft**. Keep existing failure artifact / outcome / closure / turn-error behavior (`HarnessFinalizeFailedError` path).

Structured-output path unchanged: validate/canonicalize; never run prose repair.

## Motor speech policy

In `orion/harness/prefix.py::_format_grounding_self_block` (or successor):

- Include `response_policy_summary` in the motor grounding block
- Remove the comment that reserves policy for voice finalize
- Ensure repair-overlay instructions that were finalize-only are represented on the motor path

Update grounding evals that assert policy is voice-only.

## Schema / API / bus

| Surface | Change |
|---------|--------|
| `orion/cognition/verbs/` | Remove `orion_voice_finalize.yaml`; add `orion_response_repair.yaml` |
| Prompt registry | New repair prompt; delete old voice prompt |
| Cortex-exec routes | Agent-lane routing for `orion_response_repair` only; drop old verb |
| `HarnessRunV1` | Add `response_repair_ran`, `response_repair_reason`; document `finalize_ran` |
| `HarnessTurnOutcomeMoleculeV1` | Same two repair fields on the published outcome (trace/Hub parity); fix `surprise_resolved` producer |
| `HarnessRepairOverlayV1` | Remove or stop producing `finalize_overlay` |
| Historical readers | Map old phase string → repair failure synonym |

Contract registry / channel docs updated only if payload shapes change for published molecules.

## Files likely to touch

- `orion/harness/finalize.py` — gate, passthrough, surprise_resolved, repair invoke
- `orion/harness/prefix.py` — motor response policy + overlay content
- `orion/harness/repair.py` — stop compiling finalize_overlay
- `orion/schemas/harness_finalize.py` — run/outcome/overlay fields
- `orion/cognition/verbs/orion_response_repair.yaml` (new)
- `orion/cognition/prompts/orion_response_repair.j2` (new)
- delete `orion_voice_finalize` verb + prompt
- `services/orion-cortex-exec/app/executor.py` (+ route/lane tests)
- `orion/hub/turn_orchestrator.py` — status string readers for legacy phase
- harness tests: finalize chain, voice-changes, grounding consumers, surprise/outcome, layer-attribution
- READMEs: harness-governor, unified-turn notes as needed

## Tests (acceptance)

Gate tests (deterministic):

1. **Aligned casual passthrough** — fixture reflection `aligned`, `strain_unresolved=false`: cortex repair never called; `final_text == draft_text`; `response_repair_ran=false`.
2. **Misaligned runs repair** — repair client invoked once; fail-closed if it raises.
3. **Uncertain runs repair**.
4. **Strain unresolved runs repair** even if verdict were somehow aligned (gate uses OR).
5. **Overlay non-default does not force repair** when aligned + strain resolved.
6. **Structured reading** still skips prose repair.
7. **surprise_resolved** true for aligned passthrough with high surprise_level and `finalize_changed=false`.
8. **Motor prefix** contains `response_policy_summary` when present on capsule.
9. **Registry** — `orion_voice_finalize` not resolvable; `orion_response_repair` is.
10. Update/remove tests that stub 5c to prove “wiring by different text” as if that were quality.

## Non-goals

- Deleting 5a / 5b / 5b-prime / outcome / closure
- Historical draft-vs-final quality audit as a merge gate (optional follow-up script)
- Changing which physical model serves the agent lane
- Keyword detectors on user text to choose repair
- Prompt-only “please don’t polish” without the deterministic skip gate

## Rollback

- Feature flag optional but not required if gate tests are solid: `HARNESS_RESPONSE_REPAIR_ENABLED` default on; off restores… **no**, we are not restoring mandatory voice. Rollback = revert the PR.
- If repair volume spikes, investigate 5b false misalignments — do not reintroduce aligned polishing.

## Risks

| Severity | Risk | Mitigation |
|----------|------|------------|
| High | Motor without response_policy regresses voice before repair can help | Ship policy-into-motor in same PR as skip gate |
| Med | 5b over-fires misaligned → repair latency returns | Monitor `response_repair_ran` rate; fix 5b, don’t re-open aligned polish |
| Med | Consumers assume `finalize_ran` ⇒ voice rewrite | Doc + new fields; update Hub/trace copy |
| Low | Dead references to old verb in docs/scripts | Grep purge in same PR |

## Recommended next patch

1. Write implementation plan under `docs/superpowers/plans/2026-09-12-kill-voice-finalize.md`.
2. Implement in worktree `Orion-Sapienform-kill-voice-finalize` / branch `docs/kill-voice-finalize` → rename branch to `fix/kill-voice-finalize` at first code commit if preferred.
3. TDD the passthrough gate and surprise_resolved first; then motor policy; then verb swap; then purge.

## Approval record

- Juniper: option C full kill; corrections 1–6 (2026-09-12) incorporated above.
- Proceed to implementation plan after human review of this file.
