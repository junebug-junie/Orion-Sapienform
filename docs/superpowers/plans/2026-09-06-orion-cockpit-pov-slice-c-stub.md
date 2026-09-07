# Orion Cockpit POV — Slice C follow-on stub

**Spec:** `docs/superpowers/specs/2026-09-06-orion-cockpit-pov-design.md`
**Slice B plan:** `docs/superpowers/plans/2026-09-06-orion-cockpit-pov-slice-b.md`

Invoke **writing-plans** before implementation.

## Remaining thickness

| Stage | Deferred work | Likely capture site |
|-------|---------------|---------------------|
| **ingress** | User message, attachments, observation molecule as a real hop (today: gap) | `orion/hub/turn_orchestrator.py` ingress / `emit_observation` |
| **closure** extras | Any post-motor side-effects beyond outcome/closure already emitted in Slice A | `orion/harness/finalize.py`, offboarding paths |
| **stance_inputs** enrichment | Optional: mind_coloring / Thought-assembled `build_stance_react_context` echo if Hub-sent dict proves too thin live | `services/orion-thought/app/bus_listener.py` |

**Acceptance:** every canonical stage has a real hop or an owned gap with no silent omission; live + rewind still work.
