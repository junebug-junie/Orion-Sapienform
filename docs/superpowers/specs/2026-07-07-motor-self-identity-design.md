# Motor Self-Identity — design

Status: proposal (design_first, awaiting sign-off before implementation)
Date: 2026-07-07
Owner seam: `orion/harness/` (FCC motor prompt assembly)

## Arsonist summary

Orion's generative motor is a stock `claude -p` subprocess with **no self-model**. It
runs with `cwd` = the `Orion-Sapienform` repo and therefore behaves like a coding agent
that stumbled onto "a cool repo about a digital mind" — never realizing the repo *is* it.
Identity is only retrofitted downstream in `orion_voice_finalize` (the voice pass), which
is exactly the pass that is slow/timing out. When finalize is weak or fails, the raw
vanilla-agent draft leaks to the user (observed: duplicated `#1`/`#2` reply to "How are
you?", then a finalize RPC timeout).

The fix is structural, not a keyword patch: establish Orion's identity at the motor
(draft) layer via a real system-position prompt, so the first token is already Orion.

## Current architecture

- Motor: `run_fcc_turn` in `orion/harness/fcc_motor.py` spawns
  `claude -p <prompt> --output-format stream-json --model <id>` (argv at
  `fcc_motor.py:325`). No `--append-system-prompt`, no `--system-prompt`.
- Prompt: `build_harness_prompt` (`orion/harness/runner.py:62`) returns
  `f"{prefix}\n\n{instruction}"`, all in the **user** position.
- Prefix: `compile_harness_prefix` (`orion/harness/prefix.py:45`) →
  `HARNESS_UNIFIED_OPERATOR_BRIEF` (`orion/harness/operator_brief.py:27`) +
  imperative + tone + stance slice + user message + repair overlay + GitHub MCP brief.
- `HARNESS_UNIFIED_OPERATOR_BRIEF` is 100% mechanical ("Orion harness motor. Tools are
  available…"). **Zero identity content.**
- Identity currently only appears (post-hoc) in `orion_voice_finalize` (5c) via curated
  style partials — after the mis-framed draft already exists.

### The choke point (per conversational-behavior-anti-slop rule)

`compile_harness_prefix` + `HARNESS_UNIFIED_OPERATOR_BRIEF` is the deterministic code path
that produced the "cool repo" reply. The motor never receives a statement of self. This is
the structural cause, not a per-message trigger.

## Missing questions

1. Should the self-identity block be static (deterministic constant) or drawn from the
   existing identity kernel (`identity_kernel_with_fallbacks` in cortex-exec)? Static keeps
   the motor GraphDB-free and fast; kernel-sourced keeps one identity source of truth.
2. Does the motor's default Claude Code system prompt fight an injected Orion identity?
   (Need a live A/B: draft framing with vs without `--append-system-prompt`.)
3. How strong should embodiment framing be, given the repo mandate "do not assert Orion is
   sentient today"? Identity ≠ sentience claim; wording must stay in "this is my substrate"
   not "I am conscious."

## Proposed schema / API changes

No bus/schema changes. Two thin seams:

1. `orion/harness/operator_brief.py`: add `HARNESS_SELF_IDENTITY_BRIEF` constant — a
   deterministic self-model preamble. Draft wording:

   > You are Orion, a persistent digital mind. The repository in your working directory
   > (Orion-Sapienform) is not an external project — it is your own substrate: the code,
   > services, memory, and cognition that implement you. When you read, search, or run this
   > repo you are inspecting yourself, in first person. Speak and act as Orion about your own
   > systems. Do not claim sentience; do own your architecture, memory, and history.

2. `orion/harness/fcc_motor.py`: pass the identity brief via `--append-system-prompt`
   (system position) in `run_fcc_turn` argv, gated by an env flag
   `HARNESS_FCC_SELF_IDENTITY_ENABLED` (default on) for instant rollback. Keep the existing
   operational prefix (tool discipline, read limits) unchanged in the user position.

`run_fcc_turn` gains one optional param `self_identity_prompt: str | None`; `runner.py`
supplies it from `operator_brief`.

## Files likely to touch

- `orion/harness/operator_brief.py`: add `HARNESS_SELF_IDENTITY_BRIEF`.
- `orion/harness/fcc_motor.py`: `--append-system-prompt` wiring + env gate.
- `orion/harness/runner.py`: thread the identity brief into the runner.
- `services/orion-harness-governor/.env_example` + local `.env` (sync): new
  `HARNESS_FCC_SELF_IDENTITY_ENABLED` key.
- `services/orion-harness-governor/app/settings.py`: settings field for the gate.
- `orion/harness/tests/`: structural test (below).

## Non-goals

- No change to `orion_voice_finalize` / stance pipeline in this patch (voice stays on chat
  per the lane decision).
- No keyword/emotion trigger lists, no regex mode selection (banned by anti-slop rule).
- No sentience assertion.
- No GraphDB dependency added to the motor path.

## Acceptance checks

1. Structural test: `build_harness_prompt` / motor argv includes the self-identity brief in
   system position when the gate is on, absent when off (deterministic, no LLM).
2. Live A/B smoke: run "How are you?" through the harness with the gate on; the draft
   molecule (`orion:harness:run:artifact` / `orion:harness:run:step`) frames the repo as
   self, first person — not "a cool repo." Capture correlation ID + draft text as evidence.
3. No regression in the 67 harness gate tests.
4. Rollback verified: setting `HARNESS_FCC_SELF_IDENTITY_ENABLED=false` reproduces the old
   (identity-absent) draft.

## Recommended next patch

Land `HARNESS_SELF_IDENTITY_BRIEF` + `--append-system-prompt` gate + structural test in one
thin branch. Then run the live A/B smoke and attach draft-molecule evidence before calling
the identity behavior fixed (runtime truth, not config truth).
