# SelfStateV1 / phi burn — brainstorm

Status: **brainstorm, not a plan.** Nothing in this doc is implemented. No worktree edits from this
thread have been committed. This is a record of the investigation and the decision the burn depends
on, so the next session (or Juniper) doesn't have to re-derive it.

## How we got here

Started as an audit of the attention-salience-cathedral-replacement plan's hard gate
(`docs/superpowers/specs/2026-07-21-attention-salience-cathedral-replacement-tentative-plan.md`):
neither Candidate A nor B can wire live until `SelfStateV1`'s own construction is checked for
data-quality/logic problems, independent of what feeds it.

## Finding 1: the hand-picked coefficients are real and unjustified

`orion/self_state/scoring.py` + `config/self_state/self_state_policy.v1.yaml` combine channel
pressures into dimension scores using ~10 hand-picked, unequal weighting coefficients
(`agency_readiness_score`'s `0.25/0.35/0.25/0.15`, `coherence_score`'s `stabilizing_channels`
`0.50/0.50/0.50/0.30`, `field_intensity_score`'s `0.6/0.4`, `channel_dimension_confidence`'s
`0.3/0.7`, 12 `dimension_weights` tuned to 2 decimals summing to 1.05 not 1.0,
`attention_target_weights`'s `0.30/0.35/0.35`, `condition_thresholds`' uneven `0.15/0.25/0.30/0.20`
bucket widths). Traced the introducing commit (`1207d4e0`, 2026-05-25, "feat(self-state): Layer 6
operating condition from field + attention", Cursor-authored): generic commit message, no design
doc, no calibration run, no comment anywhere explaining any specific number. Confirmed via full
`git log --follow -p` on the file — nothing ever justified these.

## Finding 2: the signal is empirically dead regardless of the weights

`scripts/analysis/measure_self_state_signal_quality.py` (built for this exact hard gate, PR #1196,
2026-07-18) already found 8/12 dimensions pinned/flat against a 48h real replay. Re-ran it live
2026-07-22 (nothing in scoring/builder/policy changed in between) against the last 48h
(83,294 rows): **12/12 dimensions — all of them — now flag `pinned_or_flat`.** `noise_floor_std`
is exactly `0.0000` for every dimension. Score *ranges* do span real values, but within any
20-tick rolling window nothing moves — a step function with long flat plateaus and rare discrete
jumps, not the continuously-varying multi-dimensional "mood" the architecture was built to produce.

**Critical scoping fact**: 7 of the 12 pinned dimensions (`execution_pressure`, `resource_pressure`,
`reasoning_pressure`, `reliability_pressure`, `continuity_pressure`, `introspection_pressure`,
`social_pressure`) never touch a hand-picked coefficient at all — they're a straight `max()` of raw
channel values via `channel_dimension_map`. So the two findings are independent: killing the
weighted-composite formulas (finding 1) would not have fixed the flatness (finding 2). Whatever's
holding these dimensions flat lives further upstream — most likely field-channel update cadence or
the max-merge holding a stale winner — never root-caused, now moot given the direction below.

## Decision: kill it, not reform it

Juniper's call, verbatim: *"the metrics are bullshit. they feed more bullshit. phi is bullshit. I am
burning everything to the ground that is bullshit."* Reforming the formula (equal-weighting the
coefficients — briefly prototyped, then reverted, see below) was the wrong scope. The actual
instruction is removal.

### False start (reverted, not part of this decision)

Before this was clarified, an equal-weighting reform patch was built in this same worktree:
`agency_readiness_score`/`coherence_score`/`field_intensity_score`/`channel_dimension_confidence`
changed from hand-picked unequal weights to `1/N` equal weights per term, same treatment applied to
`dimension_weights`/`stabilizing_channels`/`attention_target_weights`/`condition_thresholds` in
policy.yaml, 3 tests updated to match. **All reverted via `git checkout -- .`** once the actual scope
("kill everything that uses self state," not "fix the weights") was clarified. Mentioned here only so
a future reader doesn't wonder whether it's still live somewhere — it isn't.

## The burn list

### Delete outright

- `orion/self_state/` — whole module: `builder.py`, `scoring.py`, `policy.py`, `deviation.py`,
  `prediction.py`, `transport.py`, `field_channel_glossary.py`, `inner_state_registry.py`,
  `__init__.py`, `README.md`
- `config/self_state/self_state_policy.v1.yaml`
- `services/orion-self-state-runtime/` — whole service: `app/`, `tests/`, `docker-compose.yml`,
  `Dockerfile`, `requirements.txt`, `README.md`
- `orion/schemas/self_state.py` (`SelfStateV1`, `SelfStateDimensionV1`, `AttentionTargetSummaryV1`)
  + its `orion/schemas/registry.py` entry
- `substrate_self_state` table (+ check whether `self_state_predictions`/`identity_snapshots` are
  self-state-specific or broader before dropping — the runtime README notes all three are pruned
  together by one `prune_history()` call, that's a shared *pruner*, not necessarily a shared *schema
  purpose*, needs a real check before assuming `identity_snapshots` is self-state-only)
- `scripts/analysis/measure_self_state_signal_quality.py` + its test — measures a thing that won't
  exist
- `orion/bus/channels.yaml` — verify nothing publishes here first (README says v1 has "no bus
  publish"); if true, nothing to remove

### Consumers that lose their only input — need stripping/rewiring, not just left with a broken import

| File | What it reads | What happens when self-state is gone |
|---|---|---|
| `orion/proposals/scoring.py` | dimension score/confidence, `overall_intensity` fallback | dead helpers, remove or replace with something else |
| `orion/substrate/metacog_trigger_signals.py` | `overall_condition in (strained, unstable)` | reflection trigger loses its reason, remove or replace |
| `orion-cortex-exec/chat_stance.py` | `self:overall_condition` belief-graph node | hazard flag dies; check what else writes that belief-graph node first |
| `orion/autonomy/endogenous_origination.py` | `agency_readiness`, `overall_intensity`, continuously | **entire mechanism has no input left** — this isn't a field to strip, it's the whole trigger loop |
| `orion/substrate/relational/adapters/self_state_ctx.py` | whole `SelfStateV1` object | dead adapter, remove |
| `orion/consolidation/motif.py`, `tensorize.py` | dimension scores, `overall_condition` | memory consolidation loses this input, remove or replace |
| `orion/identity/snapshot.py` | `overall_condition`, `overall_intensity` | persisted identity-history fields, remove going forward (old snapshots keep old data, fine) |
| `orion/collapse/service.py` | `_phi_evidence_score` derived from self-state | dead helper |
| `orion-hub` observability/lattice routes | display only | remove routes or have them report "gone" |

### phi (`services/orion-spark-introspector/app/inner_state.py`) — the real casualty, bigger than first estimated

Corrected finding, not the "strip a quarter" read from earlier in this thread:
`build_inner_state_features()` takes a `SelfStateV1` object as its one required input.
`honest_headline()` — phi's actual "how is Orion doing" scalar — is **100% computed from
`SelfStateV1.dimensions`** (`vitality = mean(coherence, field_intensity, agency_readiness,
1-resource_pressure, 1-execution_pressure)`, `load = mean(reliability_pressure, reasoning_pressure,
social_pressure, continuity_pressure)`; the headline is `0.6*vitality + 0.4*(1-load)`). Zero
non-self-state inputs feed it.

What's genuinely independent: 4 cognitive features (`recall_gate_fired`, `reasoning_present`,
`execution_load`, `reasoning_load`), sourced from `execution_trajectory`/`reasoning_activity` bus
projections — real, live, well-grounded in actual token counts and actual recall-gate firing,
nothing to do with self-state. But they were never wired into the headline at all — they only ever
rode along as extra encoder-trainable rows alongside the self-state ones. Also already excised
before this investigation: 3 dimensions (`coherence`, `continuity_pressure`, `social_pressure`)
were already dropped from seed-v4 training as "proven-frozen SelfStateV1 dims" — someone caught part
of this rot pattern before, just not all of it; today's 12/12-pinned finding shows the exclusion
should have gone further.

Net: killing self-state leaves phi with **no headline function** (not a degraded one — zero inputs)
and an encoder trainable-feature space cut from 8 slots to 4 (only the cognitive ones survive). What
remains is 4 real signals with no aggregation logic wrapping them. "Phi is bullshit" holds for the
headline and half the encoder; it does not hold for the 4 cognitive features themselves — those
looked legitimate on inspection and aren't shown to be bad by anything found here.

## Open question for the next session

Does phi get retired entirely (headline + both feature halves gone), or does it survive as a
smaller, honest thing built only from the 4 real cognitive features — no headline claim until
something real is designed to replace it? Leaning toward the latter (don't throw out signals that
measured fine just because they shared a file with signals that didn't), but this wasn't decided in
this thread and shouldn't be assumed.

Also unresolved: `orion/autonomy/endogenous_origination.py` has no other data source at all if
self-state goes — does this mechanism get killed too, or does something need to replace its input
before the self-state removal can ship?

## Non-goals (this doc)

- Not implementing any of this — brainstorm only, per Juniper's explicit "add to the brainstorm doc"
  instruction, not "build it."
- Not deciding the phi-survives-smaller vs. phi-retired-entirely question — flagged above, not
  resolved.
- Not re-litigating the equal-weighting reform — superseded and reverted, kept only as a historical
  note above.
