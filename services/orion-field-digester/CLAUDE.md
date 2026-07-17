# services/orion-field-digester — read before touching decay/perturbation math

This service (`app/digestion/decay.py`, `app/digestion/perturbation.py`) has had the **same
class of bug found and fixed independently multiple times** because the write-up wasn't checked
first. Before changing `apply_decay()` or `apply_perturbations()`:

**Read this service's own `README.md`, section "Decay vs. injection-interval mismatch", and
`orion/autonomy/drives_and_autonomy_retrospective.md` §5b/§5c.** They have the exact mechanism,
the live evidence, and what's already fixed. Don't re-derive it from scratch.

Known, load-bearing facts as of 2026-07-17 (verify against the README before assuming these are
still current):

- `apply_decay()` no longer decays a `NODE_DECAY_CHANNELS` channel unconditionally every
  `RECEIPT_POLL_INTERVAL_SEC=2s` tick — it holds the value flat while within
  `FIELD_DECAY_STALENESS_THRESHOLD_SEC` (default 90s) of its last real write
  (`FieldStateV1.node_vector_updated_at`, fixed PR #1144). If a channel is written directly to
  `node_vectors` **outside** `apply_perturbations()` (as `worker.py`'s `field_coherence_warning`
  and `suppression.py`'s `staleness` reset both do), it must also record a
  `node_vector_updated_at` stamp or it silently falls back to the old unconditional-decay
  behavior for that one channel — this has already been missed once by review.
- This is the mirror-image bug to `orion/spark/concept_induction`'s `DriveEngine.update()`: same
  root cause (decay/injection-cadence mismatch), opposite symptom (this one produces a
  sawtooth from decay being *too fast* relative to injection; the other produces a
  clamp-collapse from decay being *too slow*). If you're debugging a saturation, oscillation, or
  collapse symptom anywhere in the drive/field pipeline, check both `CLAUDE.md`s, not just one.
