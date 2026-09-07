# PR report: recent-attention cue in the Hub Cockpit HUD

**Date:** 2026-09-07
**Branch:** `feat/cockpit-recent-attention-hop`
**Program:** Sentience Striving Program (self-modeling / continuity)

## ⚠️ Merge-order dependency: merge PR #2141 first

This patch reads `ctx["recent_attention"]`, a key only PR #2141
(`feat/recent-attention-chat-cue`) ever sets. That PR is open, not merged, as
of this report. **Until #2141 merges, this patch is a fully-tested,
intentional no-op** -- `ctx.get("recent_attention")` returns `None`, the new
`hop_from_stance_inputs` summary fields never populate, nothing breaks and
nothing shows in the cockpit. Confirmed live by code review: this branch was
cut from `main` before #2141, and `git merge-base --is-ancestor` between the
two branches returns false. Merge #2141 first (or after, order doesn't
matter for correctness -- both patches are additive and independent); this
one only becomes *visibly* live once #2141's producer exists.

## Summary

- The Hub's existing Cockpit HUD (live, per-turn visualization of a chat
  turn's processing stages) already has a `stance_inputs` hop whose
  `summary` panel renders generically -- any key present shows up, no
  frontend change needed for a new field.
- Wires the recent-attention cue (PR #2141) into that existing hop: a new
  `_inject_recent_attention_to_inputs()` in `chat_stance.py` copies
  `ctx["recent_attention"]` into the `stance_inputs` dict the same way
  `_inject_prior_stance_to_inputs()` already does for `prior_stance`.
- `orion/cockpit/builders.py`'s `hop_from_stance_inputs()` now surfaces
  `recent_attention_items` (count) and `recent_attention_stale` (bool) in
  the hop's `summary`, so Juniper can see at a glance, per turn, whether
  Oríon's ambient attention sense had anything fresh without opening the raw
  JSON.
- No new table, bus channel, schema, or frontend JS/CSS change.

## Outcome moved

Once #2141 is also live: every chat turn's Cockpit HUD entry will show
whether Oríon's stance synthesis had a fresh or stale sense of its own
recent attention, inspectable per-turn -- not just as an aggregate on the
Hub Surface dashboard, but tied to the specific turn it shaped.

## Current architecture

- Cockpit HUD: `orion/schemas/cockpit_sighting.py` (`CockpitHopV1`, `extra="forbid"`),
  `orion/cockpit/builders.py` (pure hop builders), `orion/hub/cockpit_emit.py`
  (`emit_stance_hops` -- the only real caller of `hop_from_stance_inputs`),
  `orion/hub/turn_orchestrator.py` (builds the outer wrapper dict passed as
  `stance_inputs`, with the real per-turn dict nested one level deeper under
  its own `"stance_inputs"` key -- confirmed by reading `turn_orchestrator.py`
  lines 936-940 and the existing `test_hop_from_stance_inputs_ok` fixture
  shape). Frontend: `services/orion-hub/static/js/cockpit-hud.js`'s inspector
  panel already renders `Object.keys(hop.summary)` generically -- no
  hardcoded field list, confirmed by reading the render function directly.
- `chat_stance.py`'s `build_chat_stance_inputs()` already has one precedent
  for this exact pattern: `_inject_prior_stance_to_inputs()`, which populates
  both a `ctx` top-level key (for the Jinja template render) and the
  `inputs` dict (for the Cockpit hop) from a single source. This patch adds
  a second, analogous injector for `recent_attention`.

## Architecture touched

```
ctx["recent_attention"]  (set by PR #2141's executor.py wiring -- NOT this patch)
  --> chat_stance.py::_inject_recent_attention_to_inputs()
  --> inputs["recent_attention"]  (build_chat_stance_inputs()'s returned dict)
  --> turn_orchestrator.py: stance_req.stance_inputs
  --> cockpit_emit.py::emit_stance_hops(stance_inputs={..., "stance_inputs": dict(stance_req.stance_inputs)})
  --> orion.cockpit.builders.hop_from_stance_inputs()
  --> CockpitHopV1.summary["recent_attention_items"/"recent_attention_stale"]
  --> cockpit-hud.js inspector panel (generic summary-row render, no JS change needed)
```

## Files changed

- `services/orion-cortex-exec/app/chat_stance.py`: new
  `_inject_recent_attention_to_inputs()`, called right after
  `_inject_prior_stance_to_inputs()` in `build_chat_stance_inputs()`.
  Docstring explicit about the #2141 dependency (fixed at review -- see
  below).
- `orion/cockpit/builders.py`: `hop_from_stance_inputs()` now reads the
  nested `stance_inputs["stance_inputs"]["recent_attention"]` shape and adds
  `recent_attention_items`/`recent_attention_stale` to the hop's `summary`
  when present.
- `orion/cockpit/tests/test_builders.py`: two new tests -- summary populated
  when present (nested shape), summary keys absent when not.
- `services/orion-cortex-exec/tests/test_chat_relational_stance.py`: three
  new tests mirroring the existing `_inject_prior_stance_to_inputs` test
  triplet (present / absent / empty-dict no-op).

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed: `CockpitHopV1` for `stage="stance_inputs"` gains two
  optional `summary` keys when the (currently not-yet-live) recent-attention
  cue is present. `CockpitHopV1.summary` is a plain `dict[str, Any]`, not
  `extra="forbid"`, so this is additive with no schema migration needed.
- Compatibility notes: fully additive, fails safe to today's exact behavior
  when `recent_attention` is absent (which is every turn until #2141 merges).

## Env/config changes

None. No new keys.

## Tests run

```text
python -m pytest orion/cockpit/tests/test_builders.py -q
=> 18 passed

python -m pytest services/orion-cortex-exec/tests/test_chat_relational_stance.py -q
=> 40 passed

python scripts/check_env_template_parity.py
=> PASS (85 services compared)
```

Note on one unrelated pre-existing failure investigated during this patch:
`services/orion-cortex-exec/tests/test_chat_stance_brief.py::
test_build_chat_stance_inputs_falls_back_when_identity_missing` fails when
run together with `test_chat_stance_shared_spine.py` **in this worktree
specifically** (likely a missing `.env`/identity-kernel default only present
in the primary checkout). Confirmed via `git stash` bisection: the failure
reproduces identically with 100% of this patch's changes stashed away
(pristine HEAD content), so it is a worktree-environment artifact, not
caused by this patch. Not touched.

## Evals run

No eval harness exists for Cockpit hop builders (pure functions, unit-tested
only, matching every other `hop_from_*` builder in this file).

## Docker/build/smoke checks

Not run -- no ports, health checks, or compose config touched. This is a
plain Python change to two already-imported modules; a live smoke of the
actual Cockpit HUD requires #2141 to be live first (see merge-order note
above), so a deterministic test/review pass was run instead per CLAUDE.md
section 8.

## Review findings fixed

- Finding: `_inject_recent_attention_to_inputs`'s docstring asserted, as
  present-tense fact, that `ctx["recent_attention"]` "already reaches
  chat_stance_brief.j2" -- true only once PR #2141 merges; false in this
  branch's own tree today (verified: `recent_attention_reader.py` and
  `orion/substrate/recent_attention_cue.py` do not exist here, `executor.py`
  never sets the key, `chat_stance_brief.j2` has no `recent_attention`
  reference). As merged today (pre-#2141), this patch's new code path is
  fully dead -- not broken, but inert, and the docstring overclaimed it.
  - Fix: rewrote the docstring to state the #2141 dependency explicitly and
    plainly, and added the merge-order warning at the top of this report.
  - Evidence: `services/orion-cortex-exec/app/chat_stance.py`'s
    `_inject_recent_attention_to_inputs` docstring; this report's opening
    section.

## Restart required

```bash
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
```

Same restart PR #2141 needs -- if both merge, one redeploy of
`orion-cortex-exec` picks up both. This patch alone changes nothing
observable without #2141 also live.

## Risks / concerns

- Severity: low
- Concern: two open PRs (#2141, this one) both touch `chat_stance.py` /
  cortex-exec internals; merging in either order is safe (verified
  independent/additive), but neither shows real cockpit output alone.
- Mitigation: this report states the dependency explicitly; nothing to do
  beyond merging both before expecting to see it live.

## PR link

(filled in after `gh pr create`)
