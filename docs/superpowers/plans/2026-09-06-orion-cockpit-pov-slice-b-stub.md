# Orion Cockpit POV — Slice B/C follow-on stub

**Date:** 2026-09-06  
**Spec:** `docs/superpowers/specs/2026-09-06-orion-cockpit-pov-design.md`  
**Slice A plan:** `docs/superpowers/plans/2026-09-06-orion-cockpit-pov-slice-a.md` (complete)

This is a checklist pointer only — invoke **writing-plans** before implementation.

---

## Slice B — Thick inputs (replace gap hops)

| Stage | Deferred work | Likely capture site |
|-------|---------------|---------------------|
| **association** | Attention / open loops / repair / trajectory reads fed into the turn | `orion/hub/turn_orchestrator.py` — association handoff |
| **stance_inputs** | Full bundle fed into stance (not just decision output) | Stance / thought path (where stance brief is assembled) |
| **motor_boot** | Exact system/prefix context given to FCC motor at assembly time | `orion/harness/fcc_motor.py` / `services/orion-harness-governor/` — prefix assembly site |

**Files likely to touch:** `orion/hub/cockpit_emit.py`, `orion/hub/turn_orchestrator.py`, stance/thought producers, `orion/harness/fcc_motor.py`, harness governor runner.

**Acceptance:** gap beads for these stages become real hops with copyable `raw` payloads; scrubber shows actual prefix text at `motor_boot`.

---

## Slice C — Offboarding completeness

| Stage | Deferred work | Likely capture site |
|-------|---------------|---------------------|
| **closure** (and any missing post-motor side-effects) | Remaining offboarding beyond outcome/closure already emitted in Slice A | `orion/harness/finalize.py`, offboarding side-effect paths |

**Acceptance:** every canonical stage has real hops or an owned gap with no silent omission.
