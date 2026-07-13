"""Field-channel corpus row schema -- Item 1 v2 of docs/superpowers/specs/
2026-07-13-felt-state-arc-roadmap-spec.md.

**Correction, 2026-07-13 (this schema's own session).** The original Item 1
(`MoodArcCorpusRowV1`, orion/schemas/telemetry/mood_arc.py, PR #989, merged)
captured `_phi_from_self_state()`'s OUTPUT: `coherence`/`energy`/`novelty`/
`valence`, four hand-tuned heuristic scalars, each already smoothed by
`orion-field-digester`'s `apply_decay(0.92)` leaky integrator AND
additionally hand-weighted (e.g. `0.625*agency_readiness +
0.375*social_ease` for valence). A full session of downstream work (a
windowed autoencoder, Item 2, `scripts/fit_mood_arc_encoder.py`) found that
any "trajectory structure" the encoder detected in that corpus was almost
entirely explained by the known decay mechanism, not anything emergent --
because the corpus was capturing an already-composited, already-smoothed
4-scalar summary, not raw substrate. See the roadmap spec doc's own
"Correction, 2026-07-13" note under Item 1 for the full empirical trace
(two-tier shuffle/AR(1)-surrogate gate work, `feat/mood-arc-encoder-cli`).

This schema is the corrected replacement target: it captures the RAW per-
node/per-capability channel pressures from `FieldStateV1`, via
`orion.self_state.scoring.collect_field_channel_pressures(field) ->
tuple[dict[str, float], dict[str, str]]` -- the function that merges
`node_vectors` + `capability_vectors` into one flat channel-name-keyed dict
(e.g. `cpu_pressure`, `gpu_pressure`, `memory_pressure`,
`thermal_pressure`, `execution_load`, `execution_friction`,
`reliability_pressure`, typically 10-20 channels), already used by
`coherence_score()`/`uncertainty_score()`. This is BEFORE any of the
coherence/novelty/valence hand-weighting is applied -- the right layer to
test for genuine emergent structure.

It still has `apply_decay(0.92)` baked in: that smoothing happens at the
point `FieldStateV1` itself is computed, in `orion-field-digester`'s own
digestion-tick mechanics (`app/tensor/update_rules.py::run_digestion_tick`,
`app/digestion/decay.py`), and is unavoidable without touching the
digester's own mechanism -- explicitly out of scope for this patch. What
this schema removes, relative to `MoodArcCorpusRowV1`, is the SECOND layer:
the 4-scalar hand-composited summary on top of the already-decayed raw
channels.

`mood_arc_corpus.v1` (the old sink) is NOT superseded operationally -- it
keeps running, untouched, real data for what it is. This is a new,
separate, off-by-default, additive corpus sink; see
`orion/self_state/inner_state_registry.py`'s `field_channel_corpus.v1` and
`mood_arc_corpus.v1` entries for the composition-status bookkeeping.

**Row width is NOT fixed.** `channels` is a flat channel_name -> value
dict; the channel set can vary tick to tick depending on which nodes/
capabilities are active that tick (a node with no vector contribution this
tick simply contributes no keys). Downstream consumers (a future Item 2
rework, not this task) must handle a variable key set -- e.g. by union-ing
observed channels across the corpus and filling absence with 0.0, not
assuming a fixed schema shaped like `MoodArcCorpusRowV1`'s four named
float fields.

This patch does NOT rework `scripts/fit_mood_arc_encoder.py` to consume
this new dict-shaped corpus. That is separate, future work -- the existing
script still trains against `mood_arc_corpus.v1` rows unchanged.
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class FieldChannelCorpusRowV1(BaseModel):
    """One per-tick training-data row of raw `FieldStateV1` channel
    pressures, straight from `collect_field_channel_pressures()`'s first
    return value (the merged channel dict, NOT the provenance dict).
    """

    model_config = ConfigDict(extra="forbid")

    generated_at: datetime
    tick_id: str  # FieldStateV1.tick_id -- the per-digestion-tick identifier
    channels: dict[str, float]
