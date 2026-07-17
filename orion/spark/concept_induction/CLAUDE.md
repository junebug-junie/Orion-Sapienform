# orion/spark/concept_induction — read before touching drive/tension math

This directory (`drives.py`, `tensions.py`, `bus_worker.py`) has had the **same class of bug
found and fixed independently multiple times** because the write-up wasn't checked first. Before
changing `DriveEngine.update()`, `extract_tensions_from_self_state()`, or `_update_drive_pressures`:

**Read `orion/autonomy/drives_and_autonomy_retrospective.md` first — §5b through §5e
specifically.** It has the exact mechanisms, the live evidence, and what's already fixed vs.
still open. Don't re-derive it from scratch; that's exactly what already happened three times.

Known, load-bearing facts as of 2026-07-17 (verify against the retrospective before assuming
these are still current):

- `DriveEngine.update()`'s live path (`leaky_math_enabled=True`) applies tension impacts
  **sequentially per event**, not summed-then-clamped — a sum-then-clamp design let a fold batch
  collapse multiple drives to an identical value (fixed PR #1148, §5d). If you're changing this
  aggregation again, know why the sequential design exists before reverting to a sum.
- `_update_drive_pressures` (`bus_worker.py`) only writes the persisted integrator once per
  `_DRIVE_FOLD_INTERVAL_SEC=900s` (O2, PR #1126) — every bus event still gets a live decay-only
  read. If two drives ever end up sharing an identical persisted pressure, nothing here detects
  or breaks that tie automatically; it self-heals only when a differentiating tension happens to
  land (§5e — this took ~10.5h once, not instant).
- The six-drive taxonomy (`coherence, continuity, capability, relational, predictive, autonomy`)
  has no independently-derived rationale — it came from an external design chat. See
  `docs/superpowers/specs/2026-07-11-drive-taxonomy-conceptual-audit-design.md` before assuming
  it's settled.

This directory shares the exact same bug *class* — a decay mechanism whose injection cadence
isn't reconciled against its own decay rate — with `services/orion-field-digester`. If you're
debugging a saturation/oscillation/collapse symptom here, check that service's `CLAUDE.md` too.
