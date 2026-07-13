# Endogenous action motor-nerve series — status (2026-07-13)

**Mode:** Status record, not a new design. Written because the parent spec doc itself (`docs/superpowers/specs/2026-07-13-endogenous-action-motor-nerve-spec.md`, the full P0–P7 patch breakdown) lives only on the still-unmerged `docs/endogenous-action-spec` branch — main has the child patches but not the plan they came from. This doc is self-contained and doesn't depend on that branch merging first.

## Where the series stands

| Patch | What it does | Status |
| --- | --- | --- |
| P0 | Un-lie `dispatch_status` — `prepared_for_dispatch` vs. an evidence-requiring `dispatched` | **Merged** (PR #1010) |
| P1 | The motor nerve itself — Layer 9 (`orion-execution-dispatch-runtime`) actually sends prepared candidates to `orion-cortex-exec` over the bus | **Merged** (PR #1017) |
| P2 | Experience loop (episode journal flag + felt-state lane) | Not started — depends on P1 burn-in |
| P3 | Drive satisfaction + Hub operator inform | Not started — depends on P1 burn-in |
| P4 | Two new capabilities (`recall.query.readonly`, `self_experiment.create`) | Not started — depends on P1 burn-in |
| P5 | Attention-bound proposal template | Not started — depends on P1 burn-in |
| P6 | Gate-instrument fix (`UNMEASURABLE` verdict, `--window-hours`, retention caveat) | **Merged** (PR #1020) |
| P7 | Endogenous origination enable decision | Not started — gated on 2 weeks of clean P1–P3 burn-in |

## The load-bearing fact

**`EXECUTION_DISPATCH_MODE` still defaults to `dry_run`.** Nothing in the live environment has changed as a result of P0, P1, or P6 landing — no real dispatch has been sent, no real cortex-exec call has fired from this pipeline. All three merged patches are infrastructure and honesty fixes sitting inert behind that one flag, exactly as designed.

P6's re-run confirmed verdict (a) (endogenous drift) is **GO** on live data (1h: `median_abs_trajectory` 0.0442, 7d: 0.0408, both ≥ the 0.03 threshold) — this answers the question "would the origination signal have anything to fire on," but it does **not** change the P7 sequencing. P7 is gated on 2 weeks of clean **P1–P3** burn-in, and P2/P3 haven't started, let alone run for two weeks. A GO on 0(a) makes the eventual answer honest; it doesn't shorten the path to it.

Verdict (b) (internal economy, gating P4 of the *separate* internal-economy spec, not this series) is now correctly **UNMEASURABLE** rather than a false NO-GO — P6's live re-run also root-caused *why*: the Fuseki `DriveAudit` graph has had zero writes since 2026-06-19 (RDF materialization was disabled that day for load reasons, commit `e9b233e9`). That's an open, separate infra question, not something this series resolves.

## What actually needs to happen before P2 starts

Per P1's own PR report risk note: a closely-watched manual burn-in of the live send path — flip `EXECUTION_DISPATCH_MODE=dispatch_read_only` in a real environment, watch `GET :8121/latest`'s `theater_tripwire_active` and `dispatch_count` fields, check `substrate_dispatch_results` rows directly for real (non-empty) observations — before treating P1 as "on" in any meaningful sense. This is an operational step, not a code patch; it isn't captured by any of P2–P7's file lists.

## Known gap, not fixed by this doc

The full P0–P7 patch-series spec (`docs/superpowers/specs/2026-07-13-endogenous-action-motor-nerve-spec.md`) itself never merged — it exists only on `docs/endogenous-action-spec`, still open. Anyone reading only `main` can see the three shipped patches' own design docs and PR reports, but not the original single document that laid out the full P0–P7 sequence and its risk register. That branch needs its own PR, separate from this status record.

## Non-goals

- Not a new patch, not a design proposal, not a decision about P2–P7's implementation order.
- Not merging or superseding `docs/endogenous-action-spec` — that branch's own PR is still the right home for the original full spec.
- Not flipping any flag, not starting any burn-in — this is a status snapshot only.
