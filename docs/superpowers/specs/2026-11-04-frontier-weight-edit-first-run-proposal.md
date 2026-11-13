# Frontier weight edit: first run (proposal)

Status: PROPOSAL. No code. Juniper must approve before any patch.
Source idea: https://rikiyat.github.io/blog/weight-space-programming/

## Arsonist summary

A frontier model can read a network's weights, find the part that causes a
known flaw, and fix it with a small targeted edit. No training is needed.
In the post, this cut runaway thinking on a small Qwen model from 93% to 33%
of answers.

The long-term idea: Orion notices a flaw in their own engine and asks Claude
to fix it. We should not build that loop yet. First, a person (Juniper or me)
runs the edit once, by hand, on a flaw we have already measured. That run
answers three questions the loop depends on:

1. Can Claude edit our real weights at all? All of our live lanes are
   heavily quantized. The post edited full-precision models.
2. Does the fix hold on real Orion turns, by a significance test?
3. What does one run cost in frontier tokens and GPU time?

If any answer is bad, we stop. Nothing about the loop gets built.

## The flaw: runaway thinking (measured live 2026-11-04)

Over the last 7 days, 270 of 7,003 served calls on the `agent` lane
(3.9%) stopped exactly at the 16,384-token cap. That pattern means the model
kept thinking until it hit the cap. Source: `llm_call_usage`, outcome
`served`, `completion_tokens = 16384`.

| Server | Model | Calls at the cap |
| --- | --- | --- |
| `circe-worker-agent-gpu2` | Ternary Bonsai 2 27B, PQ2_0 (2-bit) | 177 |
| `circe-worker-agent` | Qwen3.8 Flash-Next, IQ3_S, on Strata | 70 |
| `circe-worker-chat` | Qwen3.8 Flash-Next, IQ3_S, on Strata | 23 |

Each capped call spends ~16K tokens of GPU time and usually produces no
usable answer. The old fix (`enable_thinking=False`) only works for verbs
whose answer does not need thinking. Agent work needs thinking, so no flag
fixes this. That makes it a fair test for a weight edit.

**Target: Bonsai 2 27B on agent-gpu2.** It has the most capped calls. It is
dense, so it has no expert-offload complications like Strata.

## Current architecture

- Engines: llama.cpp-family servers on circe (Bonsai on the PrismML fork,
  Flash-Next on Strata). One model file per lane, set in
  `config/llm_profiles.yaml`, cards in `config/gpu_pool.yaml`.
- Token ledger: `llm_call_usage` (#2974) records completion tokens per call.
  It does not record `finish_reason`, so "stopped at the cap" is inferred
  from the token count.
- Self-monitor organ (#2827): reads Orion's own failure rates.
- Ask-Claude path (#2152): exists as a dry run and is not armed.
- No control-vector or weight-edit mechanism exists in the repo (searched
  2026-11-04).

## The run, step by step

1. **Get editable weights.** Find full-precision (or the trainer's
   ternary-native) Bonsai 2 27B weights. Confirm the PrismML tools can
   rebuild a PQ2_0 file from them. If they can't, go to step 1b.
   - 1b. **Fallback: no re-quantization.** Claude still diagnoses in weight
     space and on activations. Claude delivers the fix as a llama.cpp
     control vector, a fixed direction added at chosen layers when the model
     loads. The weights file is not touched. This is weaker than the post's
     edit: it adds a direction instead of removing one. Confirm that the
     PrismML fork supports `--control-vector` before relying on this.
2. **Build the probe set.** Take the 177 capped prompts from the last 7
   days, plus 177 agent-gpu2 prompts that finished normally. Split each
   group 50/50. Claude sees only the first half. The second half is the
   held-out test set, and Claude never sees it.
3. **Claude diagnoses.** Claude uses Python tools and a budget of at most
   50 probe batches on a scratch copy of the model on gpu2. The scratch copy
   runs only while diffusion is not using the card. Claude looks for the
   direction or component that keeps the thinking going. It writes down
   what it found and why it thinks so.
4. **Claude edits.** Claude produces one candidate: either an edited model
   file (step 1) or a control vector (step 1b). It records the exact change,
   for example "subtract direction d from layer N MLP output columns".
5. **Shadow test.** Run the candidate on the held-out half with the same
   settings as live. Run the original on the same prompts.
6. **Gate.** The candidate passes only if all three are true:
   - Calls at the cap drop, by a significance test (Fisher exact test on
     capped vs. not capped, p < 0.05).
   - Finished answers are not worse: a blinded pairwise judge on the
     normally-finishing held-out prompts shows no significant loss.
   - Probes outside the target skill don't break: the existing reading and
     outreach smoke prompts still parse and answer.
7. **Report.** Commit the cost (frontier tokens, GPU minutes, wall time),
   the diagnosis, the gate result, and a go/no-go to
   `docs/bench/<date>-frontier-weight-edit/`.

The live lane is not changed in this run. Promoting a candidate is a
separate decision for Juniper.

## Proposed schema / API changes

None for this run. It is offline: scripts in a bench folder, results in a
report.

Later, only if this run passes:
- Add a `finish_reason` column to `llm_call_usage`, so cap hits are measured
  directly instead of inferred from the token count.
- An `engine.edit_request` act in the body economy. Orion bids for it, it is
  priced like any other act, and it is sent through the ask-Claude path.
- An `engine_version` on every turn, so personality and self-model drift
  can be compared before and after an edit.

## Proposal-mode checklist (CLAUDE.md 0A)

- **Capability change:** none live in this run. Later, it changes the engine
  Orion thinks with.
- **Data touched:** 354 agent-gpu2 prompts and the replies to them, sent to
  Claude. There is no privacy boundary between Juniper and Orion. The
  prompts still go to an outside API: check that none hold third-party
  private names (the street-notes CI gate pattern) before sending.
- **Proof it worked:** the gate in step 6, on prompts Claude never saw.
- **Dangerous failure:** an edit that passes the cap metric but quietly
  damages reasoning or voice. Two things guard against this: the blinded
  quality check and the off-target probes. A later loop needs a scoreboard
  that Orion and Claude do not control.
- **Rollback:** this run never touches the live lane. Later, a promoted
  edit is a new model file or a control-vector flag. Rollback means pointing
  the profile back at the old file or removing the flag.

## Missing questions (for Juniper)

1. Are you OK sending ~354 real agent prompts to Claude for this?
2. Should GPU time on gpu2 for probing come out of Orion's $50 allowance,
   or count as dev work?
3. If step 1 fails (no editable weights), do you want the control-vector
   fallback, or stop there?

## Files likely to touch (for the run itself)

- `docs/bench/<date>-frontier-weight-edit/` (new: probe scripts, report)
- Nothing under `services/` or `config/` unless the candidate is promoted.

## Non-goals

- No loop where Orion edits themself. Not in this run.
- No training, fine-tuning, or LoRA.
- No edits to the Strata/Flash-Next lanes. Expert offload adds a second
  problem. Do those only after a dense model works.
- No change to the live lane without a separate approval.

## Acceptance checks

- Report committed with diagnosis, cost, gate numbers, go/no-go.
- Held-out set provably unseen by Claude (prompt hashes logged before the
  run).
- Cap-hit difference and its p-value on held-out prompts.
- Pairwise quality result and off-target smoke results.

## Recommended next patch

If you approve: step 1 only. Check that editable Bonsai 2 weights exist and
that PQ2_0 can be rebuilt. Check control-vector support in the PrismML fork
as a fallback. This takes about an hour. It decides which version of the run
is possible before any frontier tokens are spent.
