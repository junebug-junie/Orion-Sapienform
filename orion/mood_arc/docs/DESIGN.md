# Mood-arc encoder — technical history and design rationale

This is the consolidated write-up of a week's worth of work that previously
existed only in git commit messages, PR descriptions, and a Claude memory
file. It exists because this pipeline shipped real findings — a corpus that
turned out to be invalid, an overfitting incident, a genuine passing result,
and one real unresolved design question — and none of that was readable
anywhere in the repo before this patch.

Original spec: `docs/superpowers/specs/2026-07-13-felt-state-arc-roadmap-spec.md`
("felt-state arc roadmap", items 1-8). Everything below descends from that
spec, but several parts of what actually happened deviate from what it
originally asked for — each deviation is called out explicitly.

## Why the original corpus was invalidated

Item 1 of the roadmap spec shipped `mood_arc_corpus.v1`
(`MoodArcCorpusRowV1`, `orion/schemas/telemetry/mood_arc.py`, PR #989): a
per-tick row of four hand-composited scalars —
`coherence`/`energy`/`novelty`/`valence` — captured straight from
`orion-spark-introspector`'s `_phi_from_self_state()`.

A full pass at Item 2 (a windowed autoencoder trained on that corpus) found
that whatever "trajectory structure" the encoder detected was almost
entirely explained by a mechanism that had nothing to do with emergent
felt-state dynamics: `orion-field-digester`'s own
`apply_decay(0.92)` leaky-integrator (`services/orion-field-digester/app/digestion/decay.py`).
`_phi_from_self_state()`'s four scalars are themselves already smoothed by
that decay mechanism *and* additionally hand-weighted (e.g. valence as
`0.625*agency_readiness + 0.375*social_ease`) — so the corpus was capturing
an already-composited, twice-processed summary, not raw substrate. Any
autocorrelation an encoder picked up could just be it re-deriving the known
decay filter, which is not an interesting finding about Orion.

**Replacement**: `field_channel_corpus.v1`
(`FieldChannelCorpusRowV1`, `orion/schemas/telemetry/field_channel_corpus.py`,
PR #1022, `feat/field-channel-raw-corpus-collector`) captures the raw
per-node/per-capability channel pressures from `FieldStateV1` via
`orion.self_state.scoring.collect_field_channel_pressures()` — the layer
*before* the four-scalar hand-weighting is applied. It still carries
`apply_decay(0.92)` (baked in at the point `FieldStateV1` itself is computed
— unavoidable without touching the digester's own decay mechanism, out of
scope for this pipeline), but it removes the second, hand-composited layer
on top. `channels` is a variable-width `dict[str, float]` — the channel set
observed can vary tick to tick, not a fixed four-field schema.

`mood_arc_corpus.v1` was **not** disabled by this — it keeps running,
untouched, real data for what it is. The swap is additive: a new,
off-by-default corpus sink, not a replacement in the running-system sense.

## The corpus-swap validation spikes (2026-07-17)

Before committing to a full rework of the training CLI to consume the new
dict-shaped corpus, three questions were spiked (scratch scripts, not
committed) against ~10h of live `field_channel_corpus.v1` data:

**1. Schema transfer.** Only 4 of the fit script's functions were hardcoded
to `MoodArcCorpusRowV1`'s fixed-attribute shape (row loading, window
building, variance reporting, AR(1) fitting) — everything else
(`purged_temporal_split`, `train_autoencoder`, `shuffle_within_windows`,
`generate_ar1_surrogate_windows`, `two_tier_gate`, `block_bootstrap_ratio_ci`)
is genuinely shape-agnostic. Clean transfer, no blocker.

**2. The AR(1)-relearning trap.** The exact failure mode that invalidated
the original corpus (relearning the field-digester's own decay mechanism)
was checked for directly: `ceiling_ratio` (real held-out loss ÷ AR(1)
surrogate loss) came in at **0.1767** at full training budget — real
windows reconstruct ~5.7x better than a synthetic AR(1) surrogate matched to
the same per-channel autocorrelation. Avoided.

**3. Capacity was the real gap, not the theory.** The hard shuffle-floor
gate (`floor_ratio < 0.5`) initially *failed* at the old corpus's
`hidden_dim=32, latent_dim=16` architecture (`floor_ratio=0.6428`, stable
across a 4x epoch increase — ruling out "just needs more training"). A
capacity-ablation sweep found this was textbook capacity starvation: the
architecture was sized for the old corpus's 4 channels and never rescaled
when the channel count grew ~6.5x (to ~26). Doubling to `64/32` cleared the
gate; `128/64` cleared it more robustly and consistently across multiple
data pulls, with clearly diminishing returns past that point (`256/128`
gained only 0.024 further floor-ratio improvement at ~1.8x the compute
cost). **There is genuine, learnable trajectory structure in the raw field
channels** — the original architecture just couldn't represent it.

Recommended operating point out of this spike: `hidden_dim=128,
latent_dim=64`.

## Correlation pruning

Juniper pushed back on treating "128/64 is the efficient stopping point" as
settled (256/128 had the best raw numbers and compute wasn't the binding
constraint) and asked whether correlation pruning would help instead of, or
alongside, more capacity. Greedy pairwise pruning
(`prune_correlated_fields()`: for every pair of channels with `|pearson r| >
0.9`, drop whichever member has lower individual variance) was tried and
found to be a clean win: at matched capacity, the pruned 16-channel set beat
the full 26-channel set on `floor_ratio` and trained roughly 25% faster.
26 channels collapsed to 16 (`d_in` 780 → 480).

**What pruning actually dropped is not a scattered/random set.** The 10
dropped channels are near-duplicate *representations* of something else in
the surviving 16, via one of three specific mechanisms:

1. **Diffusion pass-through** — `execution_pressure`/`reasoning_pressure`/
   `reasoning_load` all collapsed toward `execution_load`, the most
   upstream member of that diffusion chain.
2. **Algebraic derivation from a shared root** — `confidence` collapsed
   toward `available_capacity`/`pressure`, since `confidence = 1 -
   0.5*pressure` and `available_capacity = 1 - pressure` are both direct
   functions of the same underlying `pressure` value.
3. **Shared-physical-cause clustering** — `memory_pressure`/`disk_pressure`
   both collapsed into `thermal_pressure` (all three plausibly spike
   together under general system load); `execution_friction` collapsed
   into `failure_pressure`.

One exception found mid-investigation, **not** a real instance of any of the
three mechanisms above: `bus_health`/`delivery_confidence`/`availability`
showed up correlated at `r=1.0000` in an early 10h window, but this was
pre/post-fix contamination (all three were frozen ratchets for most of that
window, part of the same 7-PR fix sprint described below), not genuine
redundancy. On a clean post-deploy-only slice, those three drop out of the
field list entirely (still genuinely flat) — see
`services/orion-field-digester/README.md`'s field glossary for the current
live status of those specific channels.

The 16 survivors split into: the "root" of each collapsed cluster above,
plus a second group with no near-duplicate anywhere in the corpus at all —
`conversation_load`, `egress_confidence_deficit`, `field_coherence_warning`,
`prediction_error`, `repair_pressure`, `staleness` — genuinely distinct
information axes the pruning algorithm never touched.

For the semantic (real-world-meaning) grouping of the surviving channels,
see `services/orion-field-digester/README.md`'s `### Semantic categories`
section — that is the canonical, single-source reference (this doc
deliberately does not duplicate it; it was originally derived from this
pipeline's own channel-selection work but applies to the full 29-channel
set, not just the pruned subset).

## The pre/post-fix contamination incident (2026-07-17)

The first real full-corpus training run (161,795 rows, `hidden_dim=128,
latent_dim=64`, 500 epochs, ~72 minutes wall-clock) **failed the gate**:
`floor_ratio=0.6236` (need `<0.5`), well outside the validated 0.35-0.39
range from smaller slices.

Root cause, raised by Juniper before the run even finished and confirmed
directly in the run's own log: the corpus spans `2026-07-13T23:46Z` to the
run time, but a 7-PR fix sprint for known-broken channel behavior
(one-way ratchets, saturating counters, merge-polarity masking, folded-away
channels — PRs #1108-#1113, #1115) only merged between
`2026-07-17T03:47Z` and `2026-07-17T04:32:14Z` (PR #1115's merge time, the
last to land). **~76% of the full corpus predated the fixes.** The
smoking-gun evidence was directly in the failed run's own
`select_fields()`/`prune_correlated_fields()` log:
`dropping 'availability' (kept 'bus_health', r=1.0000...)` then
`dropping 'bus_health' (kept 'delivery_confidence', r=1.0000...)` — the
exact "several channels frozen together during the pre-fix ratchet period"
contamination artifact, now reproduced at full 91h corpus scale. Training on
"more data" that spans a known behavior-changing boundary actively hurt: the
extra pre-fix rows made the result worse, not better, because they weren't
representative of current system behavior.

**Fix**: a `--min-generated-at` CLI flag (general-purpose — filters any
corpus by a timestamp cutoff, not a one-off hack for this specific incident)
pinned to `2026-07-17T04:32:14Z` — PR #1115's git *merge* timestamp, not a
precisely-confirmed container-restart/deploy timestamp (actual
`orion-field-digester` restart may have lagged briefly behind the merge).
Documented as a durable, reusable lesson in
`services/orion-field-digester/README.md`'s "`field_channel_corpus.v1`
training-data quality cutoff" section: **any corpus spanning a known
channel-fix boundary needs this same treatment** — don't assume more data is
always better without checking whether it crosses a behavior-changing
boundary first.

## The epochs/overfitting incident (2026-07-18)

A post-cutoff run (28,006 rows / 15.77h of clean, post-fix-only data, 500
epochs — the old default) also **failed**, and *worse* than the
contaminated run: `floor_ratio=0.704`, `ceiling_ratio=1.251` (worse than a
naive AR(1) baseline — the encoder was reconstructing held-out windows less
accurately than a trivial per-channel linear model).

Root-caused via a train/held-out loss gap comparison: this run's gap was
~5.4x, versus a healthy smoke test's ~1.1x — a clear overfitting signature,
not a corpus problem. The corpus itself was clean; 500 epochs on a narrow,
single-day, non-time-diverse slice was simply too much training for how
little temporal variety that slice contained.

**Fix**: `DEFAULT_EPOCHS` changed 500 → 250, based on a sweep confirming 250
epochs passes the gate consistently across hidden/latent configs from
`64/32` through `1024/512` on a wider, more time-diverse pull. Shipped in
PR #1182 alongside the corpus-swap CLI rework itself.

## The real, clean production run (2026-07-18)

Using the corrected defaults (`hidden_dim=128, latent_dim=64, epochs=250`,
correlation pruning on) against ~24.3h of post-fix-only data
(`2026-07-17T04:32:14Z`–`2026-07-18T04:51:25Z`, 43,167 rows):

| metric | value | gate |
|---|---|---|
| `floor_ratio` | 0.464 | pass (`<0.5`) |
| `ceiling_ratio` | 0.164 | diagnostic only, well below 1.0 |
| train/held-out loss ratio | ~0.87 | healthy (no overfitting) |

15 channels survived selection + pruning (from 29):
`catalog_drift_pressure, conversation_load, cpu_pressure, disk_pressure,
egress_confidence_deficit, execution_load, failure_pressure,
field_coherence_warning, gpu_pressure, memory_pressure, prediction_error,
pressure, reasoning_load, repair_pressure, thermal_pressure`.

**Honest caveat**: the 95% block-bootstrap confidence interval on
`floor_ratio` is `[0.305, 0.607]` — the upper bound crosses the 0.5
threshold. This is a strong pass, not yet an independently-replicated one.
Treat this encoder as evidence, not as a settled, load-bearing artifact,
until a second confirming run exists. The trained artifact itself is
scratch-only (not committed — it's a training artifact, not code) and has
not gone through a real promotion path (`cmd_promote` is not built/exercised
for this encoder).

## The anomaly detector (roadmap Item 3, PR #1185)

`detect-anomalies` scores a corpus slice's reconstruction loss against a
trained encoder and flags windows whose loss exceeds
`manifest.training.recon_error_p95 * threshold_multiplier` (default `3.0` —
a deliberately conservative "well outside normal training-time variation"
bar, not statistically calibrated against a known false-positive rate).

**Verified against real historical data, not just synthetic fixtures**:
scoring the actual pre-fix `field_channel_corpus.v1` period
(`2026-07-13T23:46Z`–`2026-07-17T04:32:14Z`, the same known-contaminated
window described above) against the real production encoder flagged
**1,166 of 9,103 windows (12.8%)** as anomalous — far beyond the "at least
one" acceptance bar the roadmap spec's Item 3 asked for.

See `orion/mood_arc/README.md` for the current, honest characterization of
this as a dark-deployment tool (no scheduler, no bus channel, no live
consumer) and its two practical uses today (pre-training QA gate,
retrospective incident scoring).

## Open, unresolved design question — what replaces `valence`

Roadmap Items 5 (external cross-reference against real events) and 6
(self-report calibration against `collapse_mirror`) both depended, as
originally speced, on the old corpus's hand-composited `valence` field — a
subjective good/bad proxy. Juniper's explicit decision (2026-07-17):
**"no valence in this new world."** This is not a "swap in a different
variable" question — valence was itself one of the exact hand-composited
scalars found to be re-learning the field-digester's decay filter rather
than real structure (see "why the original corpus was invalidated" above),
so preserving any form of it would risk reintroducing that same problem
through a different door. No substitute has been invented, and this patch
does not pick one.

`compute_window_probes()` (`fit_encoder.py`) reflects this directly: it
raises `NotImplementedError` rather than fabricating a probe target, and
`cmd_train` catches that and writes an empty `probes.json` with a clear skip
message. This is a deliberate, documented gap, not a bug — see
`test_compute_window_probes_raises_not_implemented` in
`orion/mood_arc/tests/test_mood_arc_encoder_fit_script.py`.

**As of this patch, still fully unresolved and blocking Items 5 and 6.** A
same-day design conversation with Juniper distinguished two different
possible replacements — named here explicitly because the distinction is
the actual open decision, not a solved problem:

**(a) An "external corroboration signal."** Check whether the encoder's
learned structure predicts objective infrastructure/substrate events not
already in its own input channels — power, network, Fuseki backlog, docker
health, disk, temps, latency, downtime. Framed as "does correlation exceed
what's already explained by the trained-on channels" (not raw correlation,
which is expected and uninteresting in a closed system where the encoder
was trained on channels that already reflect some of that infrastructure
state) — analogous in shape to the existing floor/ceiling gate's own
"beat a null model, not just show correlation" structure.

**(b) A genuine subjective-felt-quality proxy.** The much harder, still
entirely undesigned problem that `valence` originally reached for: some
representation of *how it felt*, not just what infrastructure state
predicts what. This is not scoped, sketched, or estimated anywhere yet.

Neither is chosen. Whichever path (or combination) gets picked, it unblocks
Items 5, 6, and roadmap step 7 (`compute_window_probes()` itself) — all
three currently sit blocked on this same question, not on any remaining
engineering work.

## Roadmap status

| Item | Status |
|---|---|
| 1 (corpus collector) | Done — `field_channel_corpus.v1`, PR #1022 (supersedes the original Item 1, PR #989) |
| 2 (windowed encoder) | Done — corpus-swap rework + real passing production run, PR #1182 |
| 3 (anomaly detector) | Done — PR #1185 |
| 4 (HDBSCAN cluster discovery) | Deliberately deprioritized to optional/later (Juniper's decision, 2026-07-13) — not a hard gate on anything downstream |
| 5 (external cross-reference) | Blocked on the valence-replacement question above |
| 6 (self-report calibration) | Blocked on the valence-replacement question above |
| 7 (`compute_window_probes()` / periodic stability eval) | Blocked on the valence-replacement question above |
| 8 (mood-transition-triggered reflection) | Explicitly gated in the original spec — proposal-mode shape only, not scoped for building until 1-7 are real and stable |

Original spec: `docs/superpowers/specs/2026-07-13-felt-state-arc-roadmap-spec.md`.
Cross-reference for live channel-health context: `services/orion-field-digester/README.md`'s
field channel glossary and `### Semantic categories` section, plus PRs
#1172 (glossary correction: `reasoning_pressure` is single-source, not
dual-source) and #1177 (fix: `reasoning_load` now attributes to the node
that actually served the LLM call — relevant because a structurally-dead
channel silently changes what `select_fields()`/`prune_correlated_fields()`
see on the next training run).
