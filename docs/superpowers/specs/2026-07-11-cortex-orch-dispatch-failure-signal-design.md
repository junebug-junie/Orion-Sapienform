# Design: cortex-orch dispatch-failure signal → Organ Signals tab

**Date:** 2026-07-11
**Status:** Design decided (plumbing question resolved) — NOT approved for build, NOT built
**Scope:** `orion/signals/adapters/cortex_orch.py`, `orion/signals/registry.py`, `orion/substrate/signal_bridge.py`, `services/orion-cortex-orch/app/main.py`, `services/orion-cortex-orch/app/orchestrator.py`, `services/orion-hub/static/js/organ-signals-graph-ui.js` (+ focused tests)
**Continues:** brainstorm ideas #2–#6 from the 2026-07-11 endogenous-error-catching session, re-grounded against the correct pipeline (see Arsonist summary)
**Revision note (2026-07-11, same day):** first draft left the bus-plumbing question open and it was going to break — publishing on `orion:cortex:request` would have made cortex-orch's own failure report re-enter `handle()` as a fake new inbound request (that channel is cortex-orch's own RPC listener, per `main.py:570-575`). Replaced with a decided architecture below: reuse the existing per-reply result channel, gate emission with a metadata flag instead of scoping by code location.

---

## Arsonist summary

`cortex_orch` is already a registered organ in the `OrionSignalV1` system (`orion/signals/registry.py:359`), but its one adapter (`CortexOrchAdapter`) only ever speaks at intake: it fires on `cortex.orch.request`/`cortex.orch.plan` and emits exactly one `signal_kind`, `"plan_resolution"` — a guess about the incoming request (verb, pack count), never a fact about what happened to it. Every place cortex-orch's own runtime genuinely fails today — RPC timeout waiting on cortex-exec (`main.py:524`), envelope decode / unhandled exception (`main.py:550`), workflow crash (`orchestrator.py:367` via `execute_chat_workflow`) — is caught, logged with `logger.exception`, and turned into a clean `CortexClientResult(ok=False, ...)` for the RPC caller. None of it becomes an `OrionSignalV1`. None of it reaches the Organ Signals tab. cortex-exec, by contrast, already emits a completion-time signal (`cognition_run`/`cognition_step`, `orion/signals/adapters/cognition_trace.py`) carrying a real `success`/`error_present` dimension. cortex-orch has no analog — Orion can see cortex-exec sweat, but not the dispatcher that called it.

This spec covers ideas #2–#6 from the brainstorm, re-grounded: the first pass wrongly assumed the `orion/substrate/execution_loop` grammar pipeline (`GrammarEventV1` → `execution_friction`) was what feeds the Hub's Organ Signals tab. It isn't. The tab is fed by a separate, parallel system: `OrionSignalV1` → `orion-signal-gateway` → Hub's `signals_inspect_cache.py` / `correlation_chain_fallback.py` → `organ-signals-graph-ui.js`. This document targets the correct pipeline.

---

## Direct answer: does #2–#6 solve double counting?

**Not automatically — but there's now a structural mechanism that solves the common case by construction, decided below, not just scoped by hope.**

Rejected approach: reuse `causal_parents`/`prior_signals` for dedup. Every current adapter links `causal_parents` by pulling from `prior_signals`, which the gateway populates from `SignalWindow.get_all()` (`services/orion-signal-gateway/app/signal_window.py`) — **the single most recent signal per `organ_id`, evicted after 30s, with no correlation_id scoping at all.** Under any concurrent load, `prior_signals["cortex_exec"]` at the moment cortex-orch's adapter runs could belong to a completely unrelated turn. Reusing that pattern for dedup wouldn't just double-count — it would silently mislink signals across turns, a worse failure than the one it's meant to prevent. It must not be used for this.

Rejected approach #2: emit a cortex-orch outcome signal on every reply, mirroring `cognition_run` (`success` dimension on every turn). This looked clean until traced through: cortex-orch's reply already carries `ok=False` whenever cortex-exec reports a normal internal step failure (`main.py:402-408`, not an exception at all — the success code path, computed from `result_payload.get("status")`). Emitting on every reply would mean cortex-orch and cortex-exec both report the *same* step failure as two separate organ signals, every single time — not a rare edge case, the common case. Rejected.

**Decided mechanism: a metadata flag set only at cortex-orch's own point of failure, checked by the adapter before it emits anything.**

`CortexClientResult.metadata: Dict[str, Any]` (`orion/schemas/cortex/contracts.py:168`) already carries an orch-origin marker in one of the two target except-blocks — `main.py`'s `TimeoutError` handler already sets `metadata={"timeout": True, "orch_timeout_terminalized": True}` (line 540). This spec:

1. Normalizes that to one flag both orch-native failure sites set: `metadata["orch_dispatch_failure"] = True`, `metadata["orch_dispatch_error_kind"] = type(e).__name__` — added to `main.py:524` (`TimeoutError`), `main.py:550` (generic `Exception`), and `orchestrator.py:367`'s workflow-crash `CortexClientResult` (all three already build a `CortexClientResult`/`CortexOrchResult` reply; this is a one-line addition to each `metadata={...}` literal).
2. `CortexOrchAdapter.adapt()` only emits `dispatch_failure` when `payload.get("metadata", {}).get("orch_dispatch_failure") is True`. A normal exec-propagated `ok=false` reply never sets this flag, so the adapter never fires for it — **no timing dependency, no window lookup, no causal-parent bookkeeping required to avoid the common-case duplicate.** It's decided at the exact line that knows whether cortex-orch itself is the one that failed.

This is strictly better than the original draft's "scope by which except-block fired" framing: it's the same effective boundary, but expressed as data on the envelope that already flows through the channel the gateway already subscribes to (see Proposed schema/API changes), instead of requiring new plumbing to get the fact there at all.

**The residual edge case that survives, unrelated to which mechanism is used:** nothing cancels cortex-exec's work when orch's `asyncio.wait_for` times out (`orchestrator.py:737-742`). Exec keeps running and may *later* publish a `cognition_run` (success or failure) for the same `correlation_id`, after orch already emitted a `dispatch_failure` signal for giving up. These are two independently-true facts about one bad turn (the turn missed its deadline **and** — maybe — the underlying computation also failed), not a clean duplicate — and no amount of flag-gating at emission time can know the future. Two options, not mutually exclusive:

1. **Accept it as a known limitation.** Treat cortex-orch's `dispatch_failure` as an *event* count (something failed to complete in time), not an *incident* count. Document this explicitly in the signal's `notes`. This is the cheapest option and matches how the rest of this repo treats friction signals (rare, individually meaningful, not pre-aggregated).
2. **Reconciliation via the correlation-scoped endpoint, not the gateway window.** The Hub already has a real correlation-scoped view (`/api/signals/correlation/<id>`, backed by `signals_inspect_cache.py`/`correlation_chain_fallback.py` — unlike the gateway's window, this *is* keyed correctly). Because `signal_id` is deterministic (`make_signal_id(organ_id, source_event_id)` = `sha256(f"{organ_id}:{source_event_id}")`), a late-arriving `cortex_exec` `cognition_run` for a `correlation_id` that already has an orch `dispatch_failure` on record could, at ingest time in the gateway or as a small Hub-side reconciliation pass, set `causal_parents` to point at each other retroactively — a UI/graph-level "these two nodes are one incident" link, not a hard count merge. This is a real second patch, not part of the smallest buildable version below.

**Recommendation: ship option 1 now, name option 2 as a deferred follow-up.** It's honest, it's cheap, and it doesn't require touching the gateway's windowing model to get value.

---

## Current architecture (verified in tree)

```text
services/orion-cortex-orch/app/main.py:handle()
  ├─ ValidationError (359)         → CortexClientResult(ok=False), no orch-origin flag  (caller's fault, excluded — see Non-goals)
  ├─ workflow Exception (289)      → CortexClientResult(ok=False), needs orch_dispatch_failure flag added  ← target
  ├─ TimeoutError (524)            → CortexClientResult(ok=False), ALREADY sets orch_timeout_terminalized  ← target (normalize flag name)
  ├─ generic Exception (550)       → CortexClientResult(ok=False), no orch-origin flag  ← target (add flag)
  └─ normal step failure (402-408) → CortexClientResult(ok=False), propagated from cortex-exec's own result
       (every reply — success, orch failure, or propagated exec failure — is sent as the RPC
        reply on channel f"{channel_cortex_result_prefix}:{correlation_id}" = orion:cortex:result:<id>,
        settings.py:29-31; this ALREADY matches the "orion:cortex:*" wildcard the gateway subscribes to)

orion/signals/registry.py:359  cortex_orch entry
  signal_kinds=["plan_resolution"]        # intake-only, unchanged
  causal_parent_organs=["cortex_gateway"]
  bus_channels=["orion:cortex:request"]   # NOTE: registry lists only the intake channel;
                                           # gateway's actual subscription is the broader
                                           # ORGAN_CHANNELS wildcard "orion:cortex:*"
                                           # (signal-gateway/app/settings.py:83), which already
                                           # covers orion:cortex:result:* — no new channel needed.

orion/signals/adapters/cortex_orch.py    CortexOrchAdapter
  can_handle(): cortex.orch.request / cortex.orch.plan / verb+packs shape   (unchanged, intake)
  adapt(): always signal_kind="plan_resolution", dims={confidence, level}  (unchanged, intake)
  + NEW can_handle() branch: CortexClientResult-shaped payload (has "ok", "verb", "metadata")
  + NEW adapt() branch: only if metadata.orch_dispatch_failure is True → signal_kind="dispatch_failure"

orion-signal-gateway/app/processor.py    ingests adapters, calls .adapt(),
  passes prior_signals = SignalWindow.get_all()  (latest-per-organ, 30s, NOT correlation-scoped —
                                                    not used for the dedup decision in this design)
  publishes OrionSignalV1 onto orion:signals:*

services/orion-hub
  signals_inspect_cache.py     live cache, subscribes orion:signals:*
  correlation_chain_fallback.py  synthesizes a chain when live cache misses a turn
  organ-signals-graph-ui.js    renders nodes (organ_id/signal_kind/dimensions/causal_parents),
                                 currently styles only is_stub — no success/failure styling

orion/substrate/signal_bridge.py
  SUPPORTED_SIGNAL_KINDS = {(cortex_exec, cognition_run), (cortex_exec, cognition_step),
                             (memory_consolidation, turn_change)}
  # cortex_orch is NOT in this set — even a well-formed cortex_orch signal today
  # would be silently ignored by signal_to_molecule()
```

**UNVERIFIED, flagged not asserted:** whether `SubstrateMoleculeV1` (`molecule_kind="organ_signal"`) produced by `signal_bridge.py` is actually consumed anywhere downstream (`orion-substrate-runtime`'s appraisal/repair-pressure worker only matched on `molecule_kind="observation"` in a quick trace, not `"organ_signal"`). This looks like the same "wired but behaviorally inert" pattern already logged elsewhere in this repo (voluntary-attention, DriveEngine per-drive nuance). Do not claim idea #3 ("feeds metacog") lands a live behavioral effect until this is traced live — see Acceptance checks.

---

## Proposed schema / API changes

0. **Metadata flag, not a new channel** (resolves the prior draft's open question). Add to the `metadata` dict of the `CortexClientResult` built at three sites:
   - `main.py:524` (`TimeoutError`) — rename/normalize existing `orch_timeout_terminalized` usage to also set `orch_dispatch_failure=True`, `orch_dispatch_error_kind="TimeoutError"`
   - `main.py:550` (generic `Exception`) — add `metadata={"orch_dispatch_failure": True, "orch_dispatch_error_kind": type(e).__name__}`
   - `orchestrator.py:367` → the `workflow_result` built at `main.py:293-316` — add the same two keys
   - Every other reply path (success, or `ok=false` propagated from cortex-exec at `main.py:402-408`) leaves these keys absent. This absence, not code-path scoping, is what the adapter checks.

1. **New `signal_kind` for cortex_orch: `"dispatch_failure"`** (idea #2 — separate from `plan_resolution`, keeps intake vs. outcome semantically distinct for this organ).
   - `orion/signals/registry.py`: extend `cortex_orch` entry — `signal_kinds=["plan_resolution", "dispatch_failure"]`, `canonical_dimensions` gains `success`, `error_present`. `bus_channels` gains the result-channel wildcard the adapter now also matches (document it even though the gateway's actual subscription list in `orion-signal-gateway` is what governs delivery — this registry field is descriptive/contract documentation per `orion/bus/channels.yaml` conventions, not itself a subscription mechanism).
   - `CortexOrchAdapter.can_handle()` gains a branch: `"ok" in payload and "verb" in payload and "metadata" in payload` (i.e., `CortexClientResult` shape) → route to failure check.
   - `CortexOrchAdapter.adapt()` gains a branch: if shape matches AND `payload.get("metadata", {}).get("orch_dispatch_failure") is True`, build the `dispatch_failure` signal; otherwise return `None` (no signal — this is the dedup gate, see Direct answer above).
   - Dimensions (all floats, per `OrionSignalV1.dimensions: Dict[str, float]` — no room for verb/lane as native fields):
     - `success: 0.0` (always, by definition of the signal existing)
     - `error_present: 1.0`
     - `level`: `1.0` for `TimeoutError` (it consumed the full budget), `0.5` flat for generic exceptions/workflow crashes (mirrors `CortexOrchAdapter`'s existing `level` semantics — a magnitude, not a category)
   - `source_event_id = correlation_id` (from `payload["correlation_id"]`, already present on `CortexClientResult`) — same id cortex-exec's `cognition_run` would use for the same turn, which is what makes future reconciliation (residual-edge-case option 2 above) possible later without a schema change.

2. **Idea #4/#5 (verb/error-kind tagging, timeout vs. exception distinction):** since `dimensions` is float-only, structured context goes where `CortexOrchAdapter`'s existing `plan_resolution` signal already puts it — `summary` (human-readable) and `notes` (list, capped at 5, per `OrionSignalV1._cap_notes`). Convention:
   - `summary = f"orch dispatch failed verb={payload.get('verb')} error_kind={metadata.get('orch_dispatch_error_kind')}"`
   - `notes = [f"error_kind:{metadata.get('orch_dispatch_error_kind')}"]`
   - **Lane is not reliably available at these three sites** — checked live in `main.py`'s except-blocks; `lane_decision` is computed inside `call_verb_runtime`/`orchestrator.py` and isn't in scope by the time the outer `except` fires. Threading it through would mean adding a field to `CortexClientResult.metadata` from *inside* orchestrator.py before the exception propagates, which is a real but separate small addition — not included in the smallest buildable version below; name it as a fast-follow, don't silently drop it.
   - This is a **one signal_kind, tagged by notes/summary** design, not multiple signal_kinds per error type — matches the existing repo convention (avoids a keyword-cathedral-shaped enum).

3. **`orion/substrate/signal_bridge.py`:** add `("cortex_orch", "dispatch_failure")` to `SUPPORTED_SIGNAL_KINDS`, so `dimensions_to_gradients()` picks up `error_present` into the `contradiction` gradient the same way cortex-exec's step failures already do. This is required for idea #3 to have any chance of reaching anything past the Organ Signals tab — but per the UNVERIFIED note above, confirm live whether that gradient is consumed by anything before describing this as "feeding metacog."

4. **No new HTTP endpoint required for idea #6 (Hub surfacing).** The Organ Signals tab already renders any `organ_id`/`signal_kind` the gateway publishes, and the correlation-scoped view (`/api/signals/correlation/<id>`) already exists. The actual gap is **visual**: `organ-signals-graph-ui.js` only has a `node[is_stub = true]` style rule (lines 419, 489) — no rule keys off `dimensions.success`/`error_present`. Smallest buildable version: one additional Cytoscape selector, e.g. `node[error_present = 1]`, styled distinctly (red border, matching whatever convention the stub styling uses), so a `dispatch_failure` node is visually distinguishable from a healthy `plan_resolution` node without reading raw JSON.

---

## Files likely to touch

- `services/orion-cortex-orch/app/main.py` — add/normalize `metadata["orch_dispatch_failure"]`/`metadata["orch_dispatch_error_kind"]` at lines 524 (`TimeoutError`) and 550 (generic `Exception`)
- `services/orion-cortex-orch/app/orchestrator.py` — same two metadata keys on the `workflow_result` built in the `except Exception` block around line 293-316 (the block whose `try` starts at 279, catching `execute_chat_workflow` failures)
- `orion/signals/registry.py` — extend `cortex_orch` entry (`signal_kinds`, `canonical_dimensions`)
- `orion/signals/adapters/cortex_orch.py` — new `can_handle()` branch for `CortexClientResult`-shaped payloads, new `adapt()` branch gated on the metadata flag
- `orion/substrate/signal_bridge.py` — add `("cortex_orch", "dispatch_failure")` to `SUPPORTED_SIGNAL_KINDS`
- `services/orion-hub/static/js/organ-signals-graph-ui.js` — add failure-state Cytoscape styling
- Tests: `orion/signals/adapters/tests/test_cortex_orch_adapter.py` (new — no existing test file for this adapter found in tree), covering (a) the flagged case emits, (b) an `ok=false` reply *without* the flag (simulating a propagated exec failure) does not, (c) a normal success reply does not; `services/orion-hub/tests/test_organ_signals_correlation_mode.py` (extend); a pinning test in `services/orion-cortex-orch/tests/` proving today's blindness before the fix lands (folds in brainstorm idea #7 as the acceptance mechanism rather than a separate idea)

No new bus channel, no new HTTP endpoint, no change to the signal-gateway's subscription list (`orion:cortex:*` in `ORGAN_CHANNELS` already covers the result channel cortex-orch already publishes to). The plumbing question from the prior draft is closed.

---

## Non-goals

- Reconciliation/retroactive causal-linking for the late-arriving-exec-signal edge case (residual-edge-case option 2 above) — name it, don't build it here.
- Touching the `orion/substrate/execution_loop` grammar pipeline (`execution_friction`/`failure_pressure`/`metacog_trigger_signals.py`) — that's a different, already-functioning system for cortex-exec/harness step-level friction; this spec does not merge cortex-orch into it (idea #1 from the original brainstorm, explicitly out of scope here per the user's steer to Organ Signals only).
- Fixing `SignalWindow`'s lack of correlation-id scoping in general — flagged as a structural gap, not fixed by this patch.
- `lane` on the failure signal's `summary`/`notes` — not reliably in scope at the three flag-setting sites; deferred as a fast-follow (see Proposed schema/API changes item 2), not silently dropped.
- Ingress `ValidationError` (`main.py:359`) is deliberately excluded from `orch_dispatch_failure` — that's a malformed-request problem (the caller's fault), not cortex-orch's own operational friction. Endogenous error catching means Orion noticing *its own* struggle, not logging every bad input.
- Any new drive/pressure/enum taxonomy beyond `success`/`error_present` — no new keyword cathedral.

---

## Acceptance checks

- A test reproduces today's blindness: raise a `TimeoutError` inside a mocked `call_verb_runtime`, assert **zero** `OrionSignalV1` / bus events result (pins current behavior before the fix).
- Same test, post-fix: asserts exactly one `dispatch_failure` signal is published, with `success=0.0`, `error_present=1.0`, `source_event_id=correlation_id`.
- **Dedup proof (the actual point of this spec):** a test simulates cortex-exec returning a normal `ok=false` result (a propagated step failure, no `orch_dispatch_failure` flag) through `main.py`'s success path (402-408) — assert **zero** `dispatch_failure` signals are emitted for it. This is the test that proves the mechanism, more important than the timeout/exception tests above.
- A concurrency test proves no cross-correlation mislinking: two concurrent orch dispatches (different correlation_ids), one fails — assert the failure signal's `source_event_id` matches only its own correlation_id.
- Live trace check (per CLAUDE.md §0A "runtime truth beats config truth"): after deploy, confirm via `docker logs` + the Hub's `/api/signals/correlation/<id>` that a real orch timeout produces a visible `dispatch_failure` node in the Organ Signals tab for that correlation_id — not just that unit tests pass. Also confirm live that the gateway's existing `orion:cortex:*` subscription actually delivers the reply-channel message to the adapter (the wildcard match is verified by reading `settings.py`, not yet by a live log line).
- Live trace check, separately: confirm or refute whether `signal_bridge.py`'s molecule output for `(cortex_orch, dispatch_failure)` is actually consumed by anything in `orion-substrate-runtime` before writing any doc claiming idea #3 "feeds metacog" — if unconsumed, say so plainly (matches the UNVERIFIED note above) rather than leave it implied.
- UI: a `dispatch_failure` node renders visually distinct from a `plan_resolution` node in `organ-signals-graph-ui.js` (manual screenshot or existing JS test pattern extended).

---

## Recommended next patch

Smallest slice: the three metadata-flag additions (`main.py` ×2, `orchestrator.py` ×1) + registry + adapter (both branches) + the pinning test + the dedup-proof test + the UI styling rule. This is now a complete, self-consistent slice with no open architecture questions — unlike the prior draft, `signal_bridge.py`/`SUPPORTED_SIGNAL_KINDS` is the only piece still deliberately deferred (land the Organ-Signals-visible fact first, prove it's real with the live trace check, *then* decide whether extending it into the substrate-molecule path is worth it once idea #3's UNVERIFIED question is actually answered).
