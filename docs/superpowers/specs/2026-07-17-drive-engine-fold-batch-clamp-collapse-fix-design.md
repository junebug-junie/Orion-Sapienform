# DriveEngine fold-batch clamp collapse: sequential per-tension update

Status: quick spec, implement same-session. User-approved fix direction (idea #3 from a
brainstorming session on `orion/autonomy/drives_and_autonomy_retrospective.md`, confirmed live
via post-deploy verification of the field-digester decay-hold fix — PR #1144 — before
implementation: `coherence`, `capability`, `predictive` observed pinned to a byte-identical
value, `0.45036942460343243`, across multiple consecutive `drive_audits` rows in production).

## Current architecture

`DriveEngine.update()` (`orion/spark/concept_induction/drives.py:53-110`) is called once per
fold by `_update_drive_pressures()` (`orion/spark/concept_induction/bus_worker.py:805-873`, O2)
with a batch of every `TensionEventV1` buffered since the last fold (at most once per
`_DRIVE_FOLD_INTERVAL_SEC=900s`, pending list capped at `_MAX_PENDING_DRIVE_TENSIONS=500`).

Inside `update()`, for the live (`leaky_math_enabled=True`, default) path:

```python
impact_sum = {k: 0.0 for k in DRIVE_KEYS}
for event in tensions:
    mag = self._clamp01(event.magnitude)
    for drive, weight in sorted(event.drive_impacts.items()):
        if drive not in impact_sum:
            continue
        impact_sum[drive] += mag * self._clamp_signed(weight)   # UNBOUNDED across the batch

for drive in DRIVE_KEYS:
    base = prev_p[drive] * decay
    impulse = self._clamp_signed(impact_sum[drive])              # clamped ONCE, after summing
    if impulse >= 0.0:
        pressures[drive] = self._clamp01(base + impulse * (1.0 - base))
    else:
        pressures[drive] = self._clamp01(base + impulse * base)
```

`impact_sum[drive]` accumulates unbounded across every tension in the batch; only the *final*
sum is clamped to `[-1, 1]`. Once that clamp saturates to exactly `1.0` (or `-1.0`), the
leaky-integrator update collapses to `pressures[drive] = clamp01(base + 1.0*(1-base)) = 1.0`
**regardless of `base`** — any drive whose batch-summed impact exceeds the clamp bound lands on
the exact same final value as every other drive that also exceeded it, independent of how much
it actually exceeded by or what its starting pressure was.

## Confirmed live (2026-07-17, post-deploy of the field-digester fix)

```text
{"autonomy": 0.5249, "coherence": 0.45036942460343243, "capability": 0.45036942460343243,
 "continuity": 0.5324, "predictive": 0.45036942460343243, "relational": 0.5695}
```

`coherence`, `capability`, `predictive` are byte-identical across multiple consecutive audits
while `autonomy`/`continuity`/`relational` stay differentiated — the exact collapse signature,
caught live with the field-digester's sawtooth already fixed (ruling that mechanism out as the
cause of this specific artifact; this is a distinct bug in `DriveEngine.update()`'s own
aggregation math, not a symptom of upstream signal noise).

## Proposed fix

Apply the leaky-integrator recurrence **sequentially, once per tension event**, instead of
summing every event's impact first and clamping once at the end. Restructure `update()`'s live
path to update a per-drive running pressure value inside the same single pass over `tensions`
the current code already does for accumulation — no new loop, just move the clamp-and-update
step inside the existing per-event loop instead of deferring it:

```python
p = {drive: prev_p[drive] * decay for drive in DRIVE_KEYS}
if self.cfg.leaky_math_enabled:
    for event in tensions:
        mag = self._clamp01(event.magnitude)
        for drive, weight in sorted(event.drive_impacts.items()):
            if drive not in p:
                continue
            impulse = self._clamp_signed(mag * self._clamp_signed(weight))
            if impulse >= 0.0:
                p[drive] = self._clamp01(p[drive] + impulse * (1.0 - p[drive]))
            else:
                p[drive] = self._clamp01(p[drive] + impulse * p[drive])
    pressures = p
else:
    # legacy soft_saturate path: UNCHANGED, still sum-then-saturate. Not the
    # live path (default leaky_math_enabled=True), kept only as a documented
    # rollback -- out of scope for this fix.
    ...
```

Each event's own `impulse` (`mag * clamp_signed(weight)`, both already individually bounded to
`[0,1]`/`[-1,1]`) is inherently within `[-1, 1]` before the outer `clamp_signed` even applies —
the outer clamp becomes a no-op safety net rather than the mechanism that erases information.
After `N` same-sign events, pressure asymptotically approaches its bound via the same
diminishing-headroom recurrence already used for a single event, but **different drives with
different exact sequences of contributing tensions retain distinguishable trajectories** instead
of being forced to an identical value by a single hard clamp over an unbounded sum. This
directly reproduces the live symptom's fix: three drives sitting at a **mid-range** value
(`0.450...`, nowhere near the `[0,1]` ceiling) collapsing to byte-identical is only possible
under the old sum-then-clamp design (many tensions summing past `±1` and getting clamped to
exactly the same boundary value); under sequential application there is no shared clamp event to
produce that collapse for a mid-range value at all.

**Order dependency, intentional and safe.** Unlike the old sum (commutative), sequential
application means *order* of `tensions` now affects the exact final pressure for drives touched
by multiple events. This is acceptable and does not need to be flagged as a new risk: the caller
(`_update_drive_pressures`, `bus_worker.py:833-834`) already buffers `pending` as a plain
`list`, appended via `pending.extend(tensions)` in arrival order — stable, deterministic, and
the same list gets passed to `update()` as `tensions_to_apply = pending` on fold. Same input
list always produces the same output.

**Backward compatibility, verified by hand for every existing single-event test case in
`orion/spark/concept_induction/tests/test_drives_leaky.py`:** for a batch of exactly one event,
`impact_sum[drive]` under the old code equals `mag * clamp_signed(weight)` — already within
`[-1, 1]` before the outer clamp, so the outer clamp was already a no-op in the single-event
case. The new sequential code computes the identical `impulse = clamp_signed(mag *
clamp_signed(weight))` for that one event and applies the identical branch
(`impulse >= 0.0` vs not) with the identical formula. Every single-tension test's assertions are
expected to hold unchanged, byte-for-byte. This must be verified by running the existing suite,
not assumed.

**Two existing tests specifically constrain multi-event batches and must keep passing
unmodified:** `test_signal_only_competition_tension_contributes_zero_pressure` and
`test_competition_tension_alone_is_pure_decay` — both feed a `tension.drive_competition.v1`
event with `drive_impacts={}` (empty) alongside or instead of a normal tension, and assert it
contributes exactly zero pressure change regardless of position in the batch. Under sequential
application this holds automatically: an event with no entry for a given drive in
`drive_impacts` simply never touches that drive's running `p` value in the inner loop — no
special-casing required.

## Non-goals

- Not touching the legacy `soft_saturate` path (`leaky_math_enabled=False`) — it's the
  rollback-only path, not live, and untouched by this fix.
- Not implementing the log-odds/logit-space redesign sketched in
  `orion/autonomy/drives_and_autonomy_retrospective.md` §5b — a bigger lift needing threshold
  recalibration, explicitly deferred there as "not started, not decided."
- Not wiring `orion.autonomy.tension_ratelimit.TensionRateLimiter` into the tension-minting path
  upstream (a separate, still-open idea from the same brainstorm — rate-limiting minting volume
  is a different lever than fixing the aggregation math, and this fix does not depend on it).
- Not changing `_DRIVE_FOLD_INTERVAL_SEC`, `_MAX_PENDING_DRIVE_TENSIONS`, or any O2 fold-cadence
  behavior — this fix is entirely inside `DriveEngine.update()`'s own math.

## Acceptance checks

1. Every existing test in `orion/spark/concept_induction/tests/test_drives_leaky.py` passes
   unmodified (byte-for-byte backward compatibility for single-event and empty-impact-event
   cases, per the hand-verified math above).
2. New regression test reproducing the live collapse: a batch of many same-sign, same-drive
   tensions applied to a drive starting from a **mid-range** pressure (not near 0 or 1) no
   longer collapses to an identical value shared with a *different* drive that received a
   different number of tensions at the same starting pressure — the two drives' final pressures
   must differ, proving the aggregation preserves differentiation instead of collapsing at the
   clamp boundary.
3. A drive that legitimately receives enough tensions to approach its bound still moves toward
   `1.0` (or `0.0` for relief) — saturation itself is not disabled, only the shared-identical-
   value collapse artifact is fixed. `docs/superpowers/specs/2026-07-17-field-digester-decay-
   hold-fix-design.md`'s own text already anticipated this: sequential application "does not
   solve saturation on its own," it only removes the hard collapse-to-identical-ceiling symptom.
4. Order-sensitivity is deterministic: running `update()` twice with the exact same `tensions`
   list (same order) produces byte-identical output both times.
5. Full `orion/spark/concept_induction/tests` suite passes (not just `test_drives_leaky.py` —
   check for other callers/consumers of `DriveEngine.update()`'s exact numeric output that this
   change could affect for multi-event batches).
