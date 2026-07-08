# φ Truthful Inner State — Plan 1 (Honest Feature Vector + Kill the Lying φ) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the geometric-mean φ that pins the Hub "PHI (Order)" number at the `0.01` floor with an honest, decontaminated, per-tick `InnerStateFeaturesV1` vector and an arithmetic cold-start headline that no single saturated infra signal can floor — and start persisting the corpus the encoder (Plan 2) will train on.

**Architecture:** A new shared schema `InnerStateFeaturesV1` (registered + channelized) is produced every substrate self-state tick by `orion-spark-introspector` from the `SelfStateV1` it already receives. A new introspector module `app/inner_state.py` assembles felt+cognitive features (dropping the proven-dead `policy_pressure`/`uncertainty`), keeps infra signals in a separate sub-vector that φ never reads, robust-scales each feature, computes an arithmetic honest headline, and freezes φ (`phi_health`) when the vector goes degenerate. The felt `reliability_pressure` is decontaminated at source (config) so catalog/contract drift can no longer masquerade as felt unreliability. The Hub EKG WebSocket `stats.phi` is repointed from the old coherence-geometric-mean to the honest headline, and the UI label is retired.

**Tech Stack:** Python 3, Pydantic v2 (`BaseModel`/`ConfigDict`), pytest (+ `pytest-asyncio`), FastAPI/WebSocket (introspector), Redis bus (`OrionBusAsync`), YAML self-state/field policies, vanilla JS static frontend.

---

## Scope, split, and deferrals (read first)

This is **Plan 1** of a two-plan split of the spec `docs/superpowers/specs/2026-07-07-phi-inner-state-truthful-design.md`. The spec is a fused build whose encoder cannot be trained until a non-degenerate corpus accrues from the lanes being lit — a hard dependency, so it is split:

- **Plan 1 (this doc):** light the lanes, emit the honest decontaminated feature vector, kill the geometric-mean headline, ship the cold-start Hub readout, persist the corpus, fix the UUID crash. Ships working software and a non-lying φ on its own.
- **Plan 2 (later, once corpus exists):** the self-supervised encoder (`φ = f(latent)`, reconstruction-error-as-coherence), per-latent probes, the discrete cognitive-event features read directly from `ExecutionRunStateV1` (recall-gate fired, `reasoning_present`, step-fail rate), the cross-service grammar-truth liveness gate, and the continuous `Δφ` reward handoff to `phi-intrinsic-reward-value-learning`.

### Build-now vs deferred — READ THIS BEFORE WRITING ANY CODE

This split has a failure mode: an implementer sees the word "deferred" and quietly ships a stub, an empty vector, a hardcoded value, or a `pass` where Plan 1 requires a **fully working** implementation. That is a plan violation, not a shortcut. The line below is exact. When in doubt, **build it fully** — nothing in the "BUILD NOW" column may be stubbed, mocked in production code, hardcoded, or left `raise NotImplementedError`.

Three signals below are **partial-scope**: Plan 1 builds a *specific, fully-working subset* and Plan 2 extends it. For those, the boundary is drawn per-field so there is no ambiguity about what "done" means in Plan 1.

| Capability | Plan 1 — BUILD NOW (fully, no stubs) | Plan 2 — DO NOT BUILD (leave out) |
|---|---|---|
| **Feature vector** | Emit a **fully populated** `InnerStateFeaturesV1` every tick with **all 10 felt dimensions + `overall_intensity`** carrying real `raw_value`, real `scaled_value`, real `source`. An empty or single-element `features=[]` is a FAILURE. | — |
| **Cognitive substance** | Source from `SelfStateV1` dimensions that the lit cortex-exec lane already feeds: `execution_pressure`, `reasoning_pressure`, and the decontaminated `reliability_pressure` (whose `execution_friction`/`failure_pressure` inputs survive Task 2). These MUST be real dimension scores read live from the tick — not zeros, not constants. | Discrete cortex-exec grammar events read from a **direct `ExecutionRunStateV1` cross-service read**: recall-gate fired (bool), `reasoning_present` (bool), `exec_step_failed` rate. Do **not** add an `ExecutionRunStateV1` subscription/HTTP read to the introspector in Plan 1. |
| **GIGO guard** | The **degeneracy-streak freeze** — FULLY implemented and tested: identical felt tuple across `PHI_DEGENERATE_STREAK` ticks ⇒ `phi_health="frozen"`, headline held at last-good. This is the guard that would have fired on the pinned-`0.01` state. It must actually freeze, verified by `test_degeneracy_freeze_after_streak`. | The **cross-service grammar-truth liveness gate** calling `build_substrate_grammar_truth` (lives in `orion-substrate-runtime`). In Plan 1, `build_inner_state_features` still **accepts** `grammar_degraded` and honors it (proven by `test_grammar_degraded_forces_frozen`), but the caller in `handle_self_state` passes the literal `False` — do not wire the cross-service source yet. |
| **`grammar_truth_degraded` field** | Present on the schema, defaults `False`, threaded through `build_inner_state_features`, and **honored** (a `True` argument must freeze — tested). | Populating it from the real substrate-runtime grammar-truth signal. |
| **Headline** | Honest arithmetic aggregate over felt inputs, FULLY implemented (`honest_headline`), wired to the Hub EKG `stats.phi`. Cold-start is the *only* headline in Plan 1 and it must be real, not `0.5` placeholder. | Encoder headline (`headline_source="encoder"`, `φ = f(latent)`, recon-error). |
| **Corpus sink** | FULLY working append-only JSONL writer, wired into `handle_self_state`, gated only by `INNER_FEATURES_CORPUS_PATH`. Must actually write lines (tested). | Training on the corpus; any encoder read of it. |
| **Encoder** | Nothing. Only the two default-OFF env placeholders (`ORION_PHI_ENCODER_ENABLED=false`, `ORION_PHI_ENCODER_WEIGHTS=`) for env parity. Do **not** create an encoder module, weights loader, or `φ = f(latent)` path. | The entire encoder: module, fit/inference, probes, attributions, recon-error, `Δφ` reward payload + handoff to `phi-intrinsic-reward-value-learning`. |
| **Decontamination** | FULLY land the config edits (Task 2) so catalog/contract drift cannot reach felt `reliability_pressure`; regression-tested both directions. | — |
| **Lane lighting** | Flip the 4 code defaults to `True` (Task 10) so the cognitive signal flows by default. | — |

**Anti-laziness rules for the implementer:**
- No `TODO`/`FIXME`/`pass`/`NotImplementedError`/`# Plan 2` stubs in shipped Plan-1 code paths. Deferred = *absent*, not *stubbed*.
- Every "BUILD NOW" row has at least one test in this plan that fails if you stub it. Do not weaken or skip those tests to make a stub pass.
- If you think a "BUILD NOW" item is impossible or out of scope, STOP and surface it — do not silently downgrade it to a placeholder.
- The `features` list must be built from the live tick every call. If you find yourself returning a constant vector, you have mis-implemented the extractor.

**Repo rules that bind this plan (from `AGENTS.md`):**
- Work in a dedicated branch/worktree: `git worktree add ../Orion-Sapienform-phi-inner-state -b feat/phi-inner-state` (or `git switch -c feat/phi-inner-state`).
- Never stage/commit `.env`. After any `.env_example` change run `python scripts/sync_local_env_from_example.py` from repo root.
- `ORION_BUS_URL=redis://100.92.216.81:6379/0` (Tailscale node) — never `bus-core`/`redis`.
- Contract changes update `orion/schemas/registry.py` + `orion/bus/channels.yaml` in the same changeset.
- Config changes to `config/self_state/*` and `config/field/*` require a self-state-runtime + field-digester restart (commands in "Restart required" at the end).

---

## File structure (created / modified)

**Created:**
- `orion/schemas/telemetry/inner_state.py` — `InnerFeatureV1` + `InnerStateFeaturesV1` shared schema.
- `services/orion-spark-introspector/app/inner_state.py` — scaler, feature specs, extractor, honest headline, degeneracy freeze.
- `services/orion-spark-introspector/app/inner_state_sink.py` — append-only JSONL corpus sink.
- `tests/test_inner_state_schema.py` — schema round-trip + dead-signal exclusion (repo-root test lane).
- `tests/test_inner_state_features.py` — scaler, extractor, headline, degeneracy freeze, pinned-`0.01` regression.
- `tests/test_self_state_reliability_decontamination.py` — builder mapping fix regression.
- `services/orion-spark-introspector/tests/test_inner_state_emit.py` — `handle_self_state` UUID fix + emit + WS payload + sink.
- `services/orion-spark-introspector/evals/eval_inner_state_readout.py` — replay eval (headline responsive & non-degenerate).

**Modified:**
- `orion/schemas/registry.py` — import + `_REGISTRY` + `SCHEMA_REGISTRY` entries.
- `orion/bus/channels.yaml` — `orion:self:inner_features` channel.
- `config/self_state/self_state_policy.v1.yaml` — remove `contract_pressure: reliability_pressure` map.
- `config/field/orion_field_topology.v1.yaml` — remove the two `contract_pressure → reliability_pressure` edges.
- `services/orion-spark-introspector/app/settings.py` — new inner-features settings.
- `services/orion-spark-introspector/app/worker.py` — UUID fix, feature build/emit in `handle_self_state`, WS `stats.phi` repoint.
- `services/orion-spark-introspector/app/static/index.html` + `tissue_viz.js` — retire "PHI (Order)" label.
- `services/orion-spark-introspector/.env_example` — new keys.
- `services/orion-cortex-exec/app/settings.py` + `.env_example` — lane default flip.
- `services/orion-substrate-runtime/app/settings.py` + `.env_example` — lane default flip.
- `services/orion-hub/app/settings.py` + `.env_example` — lane default flip.

---

## Task 1: Fix the introspector `handle_self_state` UUID crash

The substrate publishes self-state every ~2s, but `handle_self_state` crashes constructing `SparkStateSnapshotEnvelope` because `Envelope.correlation_id` requires a `UUID` and it is handed the non-UUID string `ss.self_state_id`. This half-kills the substrate snapshot rail. Coerce to a deterministic UUIDv5, keeping the human-readable id in the payload (`Optional[str]`) and metadata — mirroring the existing `_resolve_cortex_correlation_uuid` pattern.

**Files:**
- Test: `services/orion-spark-introspector/tests/test_inner_state_emit.py`
- Modify: `services/orion-spark-introspector/app/worker.py:2290-2295`

- [ ] **Step 1: Write the failing test**

Create `services/orion-spark-introspector/tests/test_inner_state_emit.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

import app.worker as worker
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.self_state import SelfStateDimensionV1, SelfStateV1

# NOTE: handle_self_state does `SelfStateV1.model_validate(payload)` inside a
# try/except that swallows ValidationError and returns early. A malformed
# payload would therefore NOT exercise the UUID crash at all — the test would
# be testing nothing. So build a REAL, valid SelfStateV1 and dump it. The
# self_state_id is intentionally a non-UUID string (that is the bug under test);
# it lives in the payload, not in a UUID field, so it is valid here.
_NOW = datetime(2026, 7, 7, 12, 0, tzinfo=timezone.utc)
_SCORES = {
    "coherence": 1.0, "field_intensity": 1.0, "agency_readiness": 0.41,
    "execution_pressure": 0.0, "reasoning_pressure": 0.05,
    "resource_pressure": 1.0, "reliability_pressure": 1.0,
    "continuity_pressure": 0.0, "introspection_pressure": 0.0,
    "social_pressure": 0.0, "uncertainty": 0.0, "policy_pressure": 0.0,
}


def _self_state() -> SelfStateV1:
    return SelfStateV1(
        self_state_id="self.state:tick_abc:policy.v1",  # non-UUID on purpose
        generated_at=_NOW,
        source_field_tick_id="tick_abc",
        source_field_generated_at=_NOW,
        source_attention_frame_id="frame_abc",
        source_attention_generated_at=_NOW,
        overall_intensity=0.4,
        overall_confidence=0.6,
        overall_condition="steady",
        trajectory_condition="stable",
        dimensions={
            k: SelfStateDimensionV1(dimension_id=k, score=v, confidence=0.6)
            for k, v in _SCORES.items()
        },
        dominant_field_channels={
            "contract_pressure": 1.0, "catalog_drift_pressure": 1.0, "bus_health": 1.0,
        },
        dimension_trajectory={},
    )


def _self_state_payload() -> dict:
    return _self_state().model_dump(mode="json")


@pytest.mark.asyncio
async def test_handle_self_state_uuid_crash_fixed(monkeypatch) -> None:
    captured = {}

    class _Bus:
        enabled = True

        async def publish(self, channel, env):
            captured["channel"] = channel
            captured["env"] = env

    monkeypatch.setattr(worker, "_pub_bus", _Bus(), raising=False)
    monkeypatch.setattr(worker.manager, "broadcast", AsyncMock(), raising=False)

    env = BaseEnvelope(
        kind="substrate.self_state.v1",
        source=ServiceRef(name="substrate-runtime", node="athena"),
        payload=_self_state_payload(),
    )

    # Must NOT raise pydantic ValidationError on correlation_id.
    await worker.handle_self_state(env)

    snap_env = captured["env"]
    # Envelope correlation_id is coerced to a real UUID...
    assert isinstance(snap_env.correlation_id, UUID)
    # ...while the human-readable id is preserved in the payload.
    assert snap_env.payload.correlation_id == "self.state:tick_abc:policy.v1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest services/orion-spark-introspector/tests/test_inner_state_emit.py::test_handle_self_state_uuid_crash_fixed -v`
Expected: FAIL — the payload validates fine (the `model_validate` try/except at worker.py:2230 is NOT the crash site), then a pydantic `ValidationError` (`uuid_parsing`) **propagates out of** `handle_self_state` from the `SparkStateSnapshotEnvelope(correlation_id=ss.self_state_id, ...)` construction at line 2290, which is outside any try/except. The test fails with that ValidationError.

- [ ] **Step 3: Apply the minimal fix**

In `services/orion-spark-introspector/app/worker.py`, `handle_self_state` currently (lines 2290-2295):

```python
    snap_env = SparkStateSnapshotEnvelope(
        source=_svc_ref(),
        correlation_id=ss.self_state_id,
        causality_chain=env.causality_chain,
        payload=snap,
    )
```

Replace with (coerce to a stable UUIDv5; `NAMESPACE_URL`, `uuid5` are already imported at line 11):

```python
    snap_corr = uuid5(NAMESPACE_URL, f"orion:spark:self_state:{ss.self_state_id}")
    snap_env = SparkStateSnapshotEnvelope(
        source=_svc_ref(),
        correlation_id=snap_corr,
        causality_chain=env.causality_chain,
        payload=snap,
    )
```

(The payload `snap` already sets `correlation_id=ss.self_state_id`, whose field is `Optional[str]` — see `orion/schemas/telemetry/spark.py:35` — so the human-readable id is preserved.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest services/orion-spark-introspector/tests/test_inner_state_emit.py::test_handle_self_state_uuid_crash_fixed -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add services/orion-spark-introspector/app/worker.py services/orion-spark-introspector/tests/test_inner_state_emit.py
git commit -m "fix(spark-introspector): coerce self_state_id to UUID for snapshot envelope"
```

---

## Task 2: Decontaminate felt `reliability_pressure` from catalog/contract drift

`catalog_drift_pressure` (a `channels.yaml` completeness check) flows into `contract_pressure` and then into the felt `reliability_pressure` dimension via `max()`, flooring φ. Break only the `contract_pressure → reliability_pressure` links; keep the real reliability inputs (`observer_failure_pressure`, `execution_friction`, `failure_pressure`).

**Files:**
- Test: `tests/test_self_state_reliability_decontamination.py`
- Modify: `config/self_state/self_state_policy.v1.yaml:31`
- Modify: `config/field/orion_field_topology.v1.yaml:120` (and verify `:67`)

- [ ] **Step 1: Write the failing test**

Create `tests/test_self_state_reliability_decontamination.py`:

```python
from datetime import datetime, timezone
from pathlib import Path

from orion.attention.field_attention.builder import build_attention_frame
from orion.attention.field_attention.policy import load_attention_policy
from orion.self_state.builder import build_self_state
from orion.self_state.policy import load_self_state_policy
from orion.schemas.field_state import FieldStateV1

REPO = Path(__file__).resolve().parents[1]
ATTENTION_POLICY = load_attention_policy(REPO / "config" / "attention" / "field_attention_policy.v1.yaml")
SELF_POLICY = load_self_state_policy(REPO / "config" / "self_state" / "self_state_policy.v1.yaml")
NOW = datetime(2026, 7, 7, 12, 0, tzinfo=timezone.utc)


def test_contract_pressure_does_not_floor_reliability() -> None:
    # contract_pressure maxed (catalog drift), but NO real reliability failure.
    field = FieldStateV1(
        generated_at=NOW,
        tick_id="tick_contract_only",
        capability_vectors={
            "capability:transport": {
                "contract_pressure": 1.0,
                "catalog_drift_pressure": 1.0,
                "failure_pressure": 0.0,
                "execution_friction": 0.0,
            }
        },
    )
    attention = build_attention_frame(field=field, policy=ATTENTION_POLICY, now=NOW)
    state = build_self_state(field=field, attention=attention, policy=SELF_POLICY, now=NOW)
    rel = state.dimensions["reliability_pressure"].score
    assert rel < 0.5, f"catalog/contract drift must not floor felt reliability (got {rel})"


def test_real_failure_still_raises_reliability() -> None:
    field = FieldStateV1(
        generated_at=NOW,
        tick_id="tick_real_failure",
        capability_vectors={
            "capability:transport": {
                "contract_pressure": 0.0,
                "failure_pressure": 1.0,
            }
        },
    )
    attention = build_attention_frame(field=field, policy=ATTENTION_POLICY, now=NOW)
    state = build_self_state(field=field, attention=attention, policy=SELF_POLICY, now=NOW)
    assert state.dimensions["reliability_pressure"].score >= 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_self_state_reliability_decontamination.py -v`
Expected: `test_contract_pressure_does_not_floor_reliability` FAILS (rel == 1.0); the second test passes.

- [ ] **Step 3: Remove the contract→reliability map in self-state policy**

In `config/self_state/self_state_policy.v1.yaml`, delete line 31:

```yaml
  contract_pressure: reliability_pressure
```

(Leave `execution_friction: reliability_pressure` (35) and `failure_pressure: reliability_pressure` (36) intact — those are real reliability signals. `contract_pressure` now maps to nothing, so catalog completeness stops feeding any felt dimension.)

- [ ] **Step 4: Remove the two contract→reliability edges in field topology**

In `config/field/orion_field_topology.v1.yaml`, delete the mapping at line 120:

```yaml
      contract_pressure: reliability_pressure
```

Also delete the catalog→contract propagation at line 67 (so catalog drift no longer becomes contract pressure that could reach any felt path):

```yaml
      catalog_drift_pressure: contract_pressure
```

Leave `observer_failure_pressure: reliability_pressure` (68), `execution_friction: reliability_pressure` (79), and `failure_pressure: reliability_pressure` (80) intact.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_self_state_reliability_decontamination.py -v`
Expected: both tests PASS.

- [ ] **Step 6: Run the existing self-state suite to confirm no regression**

Run: `pytest tests/test_self_state_builder.py tests/test_self_state_scoring.py -q`
Expected: PASS (no test asserted the contaminated behavior).

- [ ] **Step 7: Commit**

```bash
git add config/self_state/self_state_policy.v1.yaml config/field/orion_field_topology.v1.yaml tests/test_self_state_reliability_decontamination.py
git commit -m "fix(self-state): stop catalog/contract drift flooring felt reliability_pressure"
```

---

## Task 3: Robust rolling scaler (pure module)

Per-feature median/IQR standardization so a saturated infra channel can't dominate and sparse channels aren't drowned. Pure and deterministic for tests.

**Files:**
- Create: `services/orion-spark-introspector/app/inner_state.py`
- Test: `tests/test_inner_state_features.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_inner_state_features.py`:

```python
import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "spark_inner_state",
    Path(__file__).resolve().parents[1]
    / "services" / "orion-spark-introspector" / "app" / "inner_state.py",
)
inner_state = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(inner_state)


def test_robust_scale_zero_when_iqr_degenerate() -> None:
    assert inner_state.robust_scale(0.5, median=0.5, iqr=0.0) == 0.0


def test_robust_scale_standardizes() -> None:
    # value one IQR above median -> ~1.0
    assert inner_state.robust_scale(2.0, median=1.0, iqr=1.0) == 1.0
    assert inner_state.robust_scale(0.0, median=1.0, iqr=1.0) == -1.0


def test_rolling_scaler_centers_after_history() -> None:
    s = inner_state.RollingRobustScaler(maxlen=16)
    for v in [0.0, 0.25, 0.5, 0.75, 1.0]:
        s.observe("cpu", v)
    scaled = s.scale("cpu", 0.5)  # 0.5 is the median of the window
    assert abs(scaled) < 1e-6


def test_rolling_scaler_saturated_input_does_not_explode() -> None:
    s = inner_state.RollingRobustScaler(maxlen=16)
    for v in [0.0, 0.1, 0.2, 0.3, 0.4]:
        s.observe("pressure", v)
    # a single saturated 1.0 is scaled to a bounded value, not a raw 1.0 dominating
    scaled = s.scale("pressure", 1.0)
    assert scaled <= inner_state.SCALE_CLIP
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_inner_state_features.py -k "robust or rolling" -v`
Expected: FAIL — `services/orion-spark-introspector/app/inner_state.py` does not exist.

- [ ] **Step 3: Create the scaler module**

Create `services/orion-spark-introspector/app/inner_state.py`:

```python
"""Honest inner-state feature assembly for spark-introspector.

Replaces the geometric-mean φ (which a single saturated infra signal could
floor) with a decontaminated, robust-scaled feature vector plus an arithmetic
cold-start headline and a degeneracy freeze (the GIGO guard).
"""
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from statistics import median as _median
from typing import Deque, Dict, List, Optional, Tuple

# NOTE: do NOT import orion.schemas.telemetry.inner_state here. That schema is
# created in Task 4; the scaler in this task does not need it. The extractor in
# Task 5 adds the import at that point (Task 4 has run by then). Importing it now
# would make the Task 3 test's importlib module-load raise ModuleNotFoundError.

SCALE_CLIP = 4.0


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    idx = pct * (len(s) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def robust_scale(value: float, *, median: float, iqr: float) -> float:
    if iqr <= 1e-9:
        return 0.0
    scaled = (float(value) - median) / iqr
    return max(-SCALE_CLIP, min(SCALE_CLIP, scaled))


class RollingRobustScaler:
    """Bounded rolling median/IQR standardizer, one window per feature name."""

    def __init__(self, maxlen: int = 256) -> None:
        self._maxlen = max(2, int(maxlen))
        self._windows: Dict[str, Deque[float]] = {}

    def observe(self, name: str, value: float) -> None:
        w = self._windows.get(name)
        if w is None:
            w = deque(maxlen=self._maxlen)
            self._windows[name] = w
        w.append(float(value))

    def scale(self, name: str, value: float) -> float:
        w = self._windows.get(name)
        if not w or len(w) < 2:
            return 0.0
        vals = list(w)
        med = _median(vals)
        iqr = _percentile(vals, 0.75) - _percentile(vals, 0.25)
        return robust_scale(value, median=med, iqr=iqr)

    def observe_and_scale(self, name: str, value: float) -> float:
        self.observe(name, value)
        return self.scale(name, value)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_inner_state_features.py -k "robust or rolling" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add services/orion-spark-introspector/app/inner_state.py tests/test_inner_state_features.py
git commit -m "feat(spark-introspector): robust rolling scaler for inner-state features"
```

---

## Task 4: `InnerStateFeaturesV1` schema + registry + channel

**Files:**
- Create: `orion/schemas/telemetry/inner_state.py`
- Modify: `orion/schemas/registry.py` (import ~441 area; `_REGISTRY` ~712; `SCHEMA_REGISTRY` ~1231)
- Modify: `orion/bus/channels.yaml` (after the spark entries ~657)
- Test: `tests/test_inner_state_schema.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_inner_state_schema.py`:

```python
from datetime import datetime, timezone

from orion.schemas.registry import resolve
from orion.schemas.telemetry.inner_state import InnerFeatureV1, InnerStateFeaturesV1


def _sample() -> InnerStateFeaturesV1:
    return InnerStateFeaturesV1(
        generated_at=datetime(2026, 7, 7, 12, 0, tzinfo=timezone.utc),
        self_state_id="self.state:tick_abc:policy.v1",
        features=[
            InnerFeatureV1(name="coherence", raw_value=1.0, scaled_value=0.0,
                           source="self_state.dimensions.coherence"),
        ],
        infra=[
            InnerFeatureV1(name="contract_pressure", raw_value=1.0, scaled_value=0.0,
                           source="self_state.dominant_field_channels.contract_pressure"),
        ],
        headline=0.7,
        headline_source="cold_start_aggregate",
        phi_health="ok",
    )


def test_registry_resolves_inner_state_features() -> None:
    model = resolve("InnerStateFeaturesV1")
    assert model is InnerStateFeaturesV1


def test_round_trip_every_feature_has_raw_scaled_source() -> None:
    payload = _sample()
    data = payload.model_dump()
    restored = InnerStateFeaturesV1.model_validate(data)
    assert restored.headline == 0.7
    for f in restored.features + restored.infra:
        assert f.name and f.source is not None
        assert isinstance(f.raw_value, float)
        assert isinstance(f.scaled_value, float)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_inner_state_schema.py -v`
Expected: FAIL — `orion.schemas.telemetry.inner_state` does not exist / `resolve("InnerStateFeaturesV1")` raises.

- [ ] **Step 3: Create the schema**

Create `orion/schemas/telemetry/inner_state.py`:

```python
"""InnerStateFeaturesV1 — Orion's honest, decontaminated inner-state vector.

Felt+cognitive `features` are the only signals φ reads. `infra` signals are
retained for provenance/equilibrium and are NEVER read by φ. Proven-dead
signals (`policy_pressure`, `uncertainty`) are excluded entirely.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class InnerFeatureV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    raw_value: float
    scaled_value: float
    source: str


class InnerStateFeaturesV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    features_version: str = "seed-v1"
    generated_at: datetime
    self_state_id: Optional[str] = None
    source_service: str = "spark-introspector"

    # felt + cognitive signals — the ONLY inputs φ reads
    features: List[InnerFeatureV1] = Field(default_factory=list)
    # infra provenance only — NEVER read by φ
    infra: List[InnerFeatureV1] = Field(default_factory=list)

    # cold-start honest headline (arithmetic; no geometric floor)
    headline: float = 0.5
    headline_source: str = "cold_start_aggregate"  # or "encoder" (Plan 2)

    # GIGO guard
    phi_health: str = "ok"  # ok | degenerate | frozen
    phi_degenerate_streak: int = 0
    grammar_truth_degraded: bool = False

    liveness: Dict[str, bool] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

- [ ] **Step 4: Register the schema**

In `orion/schemas/registry.py`, add the import next to the existing spark telemetry import at line 441:

```python
from orion.schemas.telemetry.inner_state import InnerFeatureV1, InnerStateFeaturesV1
```

Add to `_REGISTRY` (dict starting line 656), next to the spark entries at line 713:

```python
    "InnerStateFeaturesV1": InnerStateFeaturesV1,
    "InnerFeatureV1": InnerFeatureV1,
```

Add to `SCHEMA_REGISTRY` (dict starting line 1231), following the `SchemaRegistration` pattern:

```python
    "InnerStateFeaturesV1": SchemaRegistration(
        model=InnerStateFeaturesV1,
        kind="self.inner_features.v1",
    ),
```

- [ ] **Step 5: Add the bus channel**

In `orion/bus/channels.yaml`, after the `orion:spark:telemetry` block (ends line 657), add:

```yaml
  - name: "orion:self:inner_features"
    kind: "telemetry"
    schema_id: "InnerStateFeaturesV1"
    message_kind: "self.inner_features.v1"
    producer_services: ["orion-spark-introspector"]
    consumer_services: ["orion-hub", "orion-sql-writer"]
    stability: "experimental"
    since: "2026-07-07"
```

- [ ] **Step 6: Run tests + contract gates**

(NOTE: `scripts/check_schema_registry.py` / `scripts/check_bus_channels.py` do **not** exist in this repo — contract enforcement is pytest-based. Use the real gates.)

Run: `pytest tests/test_inner_state_schema.py tests/test_channel_prefix_guardrail.py tests/test_schema_registry_import_light.py -v`
Expected: all PASS — the schema round-trips + resolves, the new `orion:self:inner_features` channel satisfies the `orion:` prefix guardrail, and the light registry-import test still passes with the new entries.

- [ ] **Step 7: Commit**

```bash
git add orion/schemas/telemetry/inner_state.py orion/schemas/registry.py orion/bus/channels.yaml tests/test_inner_state_schema.py
git commit -m "feat(schemas): add InnerStateFeaturesV1 contract + channel"
```

---

## Task 5: Feature extractor (SelfStateV1 → decontaminated vector)

Assemble the felt+cognitive vector from `SelfStateV1`, drop the proven-dead signals, and keep infra in the separate sub-vector. Extends `services/orion-spark-introspector/app/inner_state.py`.

**Files:**
- Modify: `services/orion-spark-introspector/app/inner_state.py`
- Test: `tests/test_inner_state_features.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_inner_state_features.py`:

```python
class _Dim:
    def __init__(self, score: float) -> None:
        self.score = score


class _FakeSelfState:
    def __init__(self, dims: dict, **kw) -> None:
        self.dimensions = {k: _Dim(v) for k, v in dims.items()}
        self.dimension_trajectory = kw.get("traj", {})
        self.self_state_id = kw.get("sid", "self.state:tick_x:policy.v1")
        self.generated_at = kw.get("gen")
        self.overall_intensity = kw.get("intensity", 0.4)
        self.overall_condition = kw.get("condition", "steady")
        self.trajectory_condition = kw.get("trajectory", "stable")
        self.dominant_field_channels = kw.get("infra", {})


def _pinned_state():
    from datetime import datetime, timezone
    return _FakeSelfState(
        {
            "coherence": 1.0, "field_intensity": 1.0, "agency_readiness": 0.41,
            "execution_pressure": 0.0, "reasoning_pressure": 0.05,
            "resource_pressure": 1.0, "reliability_pressure": 1.0,
            "continuity_pressure": 0.0, "introspection_pressure": 0.0,
            "social_pressure": 0.0, "uncertainty": 0.0, "policy_pressure": 0.0,
        },
        gen=datetime(2026, 7, 7, 12, 0, tzinfo=timezone.utc),
        infra={"contract_pressure": 1.0, "catalog_drift_pressure": 1.0, "bus_health": 1.0},
    )


def test_dead_signals_excluded_from_features() -> None:
    scaler = inner_state.RollingRobustScaler(maxlen=8)
    payload, _felt, _streak = inner_state.build_inner_state_features(
        _pinned_state(), scaler, features_version="seed-v1",
        grammar_degraded=False, prev_felt=None, prev_headline=None,
        degenerate_streak=0, degenerate_limit=20,
    )
    names = {f.name for f in payload.features}
    assert "policy_pressure" not in names
    assert "uncertainty" not in names
    # infra signals present ONLY in the infra sub-vector
    infra_names = {f.name for f in payload.infra}
    assert "contract_pressure" in infra_names
    assert "contract_pressure" not in names


def test_features_carry_raw_scaled_source() -> None:
    scaler = inner_state.RollingRobustScaler(maxlen=8)
    payload, _f, _s = inner_state.build_inner_state_features(
        _pinned_state(), scaler, features_version="seed-v1",
        grammar_degraded=False, prev_felt=None, prev_headline=None,
        degenerate_streak=0, degenerate_limit=20,
    )
    assert payload.features, "expected felt features"
    for f in payload.features:
        assert f.source.startswith("self_state.")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_inner_state_features.py -k "dead_signals or raw_scaled_source" -v`
Expected: FAIL — `build_inner_state_features` not defined.

- [ ] **Step 3: Add feature specs + extractor**

First add the schema import to the top of `services/orion-spark-introspector/app/inner_state.py` (Task 4 created it; the import will now resolve):

```python
from orion.schemas.telemetry.inner_state import InnerFeatureV1, InnerStateFeaturesV1
```

Then append to `services/orion-spark-introspector/app/inner_state.py`:

```python
# Felt + cognitive dimensions φ reads. (execution_pressure / reasoning_pressure
# are the cortex-exec-lane-fed cognitive signal; reliability_pressure is now
# decontaminated at source.) policy_pressure + uncertainty are proven dead and
# are NOT listed here.
FELT_DIMENSIONS: Tuple[str, ...] = (
    "coherence",
    "field_intensity",
    "agency_readiness",
    "execution_pressure",
    "reasoning_pressure",
    "resource_pressure",
    "reliability_pressure",
    "continuity_pressure",
    "social_pressure",
    "introspection_pressure",
)

DROPPED_DIMENSIONS: frozenset[str] = frozenset({"policy_pressure", "uncertainty"})

# Retained for provenance only; φ never reads these.
INFRA_CHANNELS: Tuple[str, ...] = (
    "bus_health",
    "delivery_confidence",
    "transport_integrity",
    "contract_pressure",
    "catalog_drift_pressure",
)


def _dim_score(ss, key: str, default: float = 0.0) -> float:
    dim = getattr(ss, "dimensions", {}).get(key)
    return float(dim.score) if dim is not None else default


def _felt_tuple(ss) -> Tuple[float, ...]:
    return tuple(round(_dim_score(ss, k), 4) for k in FELT_DIMENSIONS)


def build_inner_state_features(
    ss,
    scaler: RollingRobustScaler,
    *,
    features_version: str,
    grammar_degraded: bool,
    prev_felt: Optional[Tuple[float, ...]],
    prev_headline: Optional[float],
    degenerate_streak: int,
    degenerate_limit: int,
) -> Tuple[InnerStateFeaturesV1, Tuple[float, ...], int]:
    """Assemble one InnerStateFeaturesV1 from a SelfStateV1.

    Returns (payload, current_felt_tuple, new_degenerate_streak).
    """
    felt_tuple = _felt_tuple(ss)

    features: List[InnerFeatureV1] = []
    raw_map: Dict[str, float] = {}
    for key in FELT_DIMENSIONS:
        raw = _dim_score(ss, key)
        raw_map[key] = raw
        features.append(
            InnerFeatureV1(
                name=key,
                raw_value=round(raw, 4),
                scaled_value=round(scaler.observe_and_scale(key, raw), 4),
                source=f"self_state.dimensions.{key}",
            )
        )

    # overall_intensity as an extra felt feature
    intensity = float(getattr(ss, "overall_intensity", 0.0) or 0.0)
    raw_map["overall_intensity"] = intensity
    features.append(
        InnerFeatureV1(
            name="overall_intensity",
            raw_value=round(intensity, 4),
            scaled_value=round(scaler.observe_and_scale("overall_intensity", intensity), 4),
            source="self_state.overall_intensity",
        )
    )

    infra: List[InnerFeatureV1] = []
    dom = getattr(ss, "dominant_field_channels", {}) or {}
    for ch in INFRA_CHANNELS:
        if ch in dom:
            infra.append(
                InnerFeatureV1(
                    name=ch,
                    raw_value=round(float(dom[ch]), 4),
                    scaled_value=0.0,  # infra never scaled/read by φ
                    source=f"self_state.dominant_field_channels.{ch}",
                )
            )

    # --- degeneracy freeze (GIGO guard) ---
    if prev_felt is not None and felt_tuple == prev_felt:
        new_streak = degenerate_streak + 1
    else:
        new_streak = 0

    degenerate = new_streak >= degenerate_limit
    headline = honest_headline(raw_map)
    phi_health = "ok"
    if grammar_degraded or degenerate:
        phi_health = "frozen"
        if prev_headline is not None:
            headline = prev_headline

    gen = getattr(ss, "generated_at", None) or datetime.now(timezone.utc)
    payload = InnerStateFeaturesV1(
        features_version=features_version,
        generated_at=gen,
        self_state_id=getattr(ss, "self_state_id", None),
        features=features,
        infra=infra,
        headline=round(_clamp01(headline), 4),
        headline_source="cold_start_aggregate",
        phi_health=phi_health,
        phi_degenerate_streak=new_streak,
        grammar_truth_degraded=bool(grammar_degraded),
        liveness={"self_state": True, "grammar_truth": not grammar_degraded},
        metadata={
            "overall_condition": getattr(ss, "overall_condition", None),
            "trajectory_condition": getattr(ss, "trajectory_condition", None),
        },
    )
    return payload, felt_tuple, new_streak
```

**Ordering note (no stubs):** `build_inner_state_features` calls `honest_headline`, whose real body is added in Task 6 Step 3. Add Task 6's `honest_headline` implementation to the module **in this same edit** (paste the Task 6 Step 3 body now) so the module imports cleanly; run Task 5's and Task 6's test steps in their own order. Do NOT ship a stubbed `honest_headline` — that violates the no-stub rule in the scope contract.

**Infra provenance caveat (honest):** `dominant_field_channels` holds only the top-N field channels above the builder's dominance threshold (`builder.py:231-236`), so at runtime an `INFRA_CHANNELS` name may be absent and `infra` may be `[]` for a given tick. That is acceptable: φ never reads `infra`, and acceptance check #2 requires infra channels appear *only* in the infra sub-vector when present — not that they are always present. Do not fabricate infra values to fill it.

- [ ] **Step 4: (defer run to Task 6)** — `build_inner_state_features` references `honest_headline`; run the combined suite at Task 6 Step 4.

- [ ] **Step 5: Commit (with Task 6 body)** — see Task 6 Step 5.

---

## Task 6: Honest arithmetic headline + degeneracy freeze regression

The headline must NOT collapse when a single infra/pressure signal saturates. Arithmetic aggregate, and a regression test on the recorded pinned-`0.01` inputs.

**Files:**
- Modify: `services/orion-spark-introspector/app/inner_state.py`
- Test: `tests/test_inner_state_features.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_inner_state_features.py`:

```python
def test_honest_headline_not_floored_on_pinned_state() -> None:
    # The live pinned-0.01 state: reliability & resource maxed, but coherence,
    # field_intensity, agency healthy. Old geometric mean -> 0.01. Honest -> >0.5.
    raw = {
        "coherence": 1.0, "field_intensity": 1.0, "agency_readiness": 0.41,
        "execution_pressure": 0.0, "reasoning_pressure": 0.05,
        "resource_pressure": 1.0, "reliability_pressure": 1.0,
        "continuity_pressure": 0.0, "social_pressure": 0.0,
    }
    h = inner_state.honest_headline(raw)
    assert h > 0.5, f"honest headline must not floor on pinned infra (got {h})"


def test_single_saturated_pressure_does_not_collapse_headline() -> None:
    healthy = {
        "coherence": 0.9, "field_intensity": 0.8, "agency_readiness": 0.8,
        "execution_pressure": 0.1, "reasoning_pressure": 0.1,
        "resource_pressure": 0.1, "reliability_pressure": 0.1,
        "continuity_pressure": 0.1, "social_pressure": 0.1,
    }
    base = inner_state.honest_headline(healthy)
    poisoned = dict(healthy, reliability_pressure=1.0)
    after = inner_state.honest_headline(poisoned)
    assert after > 0.4, "one maxed pressure must not floor the headline"
    assert after < base, "but it should still lower it somewhat"


def test_degeneracy_freeze_after_streak() -> None:
    from datetime import datetime, timezone
    scaler = inner_state.RollingRobustScaler(maxlen=64)
    st = _pinned_state()
    prev_felt = None
    prev_headline = None
    streak = 0
    payload = None
    for _ in range(21):  # limit=20 -> frozen on the 21st identical tick
        payload, prev_felt, streak = inner_state.build_inner_state_features(
            st, scaler, features_version="seed-v1", grammar_degraded=False,
            prev_felt=prev_felt, prev_headline=prev_headline,
            degenerate_streak=streak, degenerate_limit=20,
        )
        prev_headline = payload.headline
    assert payload.phi_health == "frozen"
    assert payload.phi_degenerate_streak >= 20


def test_grammar_degraded_forces_frozen() -> None:
    scaler = inner_state.RollingRobustScaler(maxlen=8)
    payload, _f, _s = inner_state.build_inner_state_features(
        _pinned_state(), scaler, features_version="seed-v1", grammar_degraded=True,
        prev_felt=None, prev_headline=0.62, degenerate_streak=0, degenerate_limit=20,
    )
    assert payload.phi_health == "frozen"
    assert payload.headline == 0.62
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_inner_state_features.py -k "headline or degeneracy or grammar_degraded" -v`
Expected: FAIL — `honest_headline` not defined.

- [ ] **Step 3: Add `honest_headline`**

Append to `services/orion-spark-introspector/app/inner_state.py`:

```python
def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def honest_headline(raw: Dict[str, float]) -> float:
    """Arithmetic vitality/load aggregate. No geometric collapse: a single
    saturated pressure lowers but cannot floor the headline."""
    def g(k: str, default: float = 0.0) -> float:
        return float(raw.get(k, default))

    vitality = _mean([
        g("coherence"),
        g("field_intensity"),
        g("agency_readiness"),
        1.0 - g("resource_pressure"),
        1.0 - g("execution_pressure"),
    ])
    load = _mean([
        g("reliability_pressure"),
        g("reasoning_pressure"),
        g("social_pressure"),
        g("continuity_pressure"),
    ])
    return _clamp01(0.6 * vitality + 0.4 * (1.0 - load))
```

- [ ] **Step 4: Run the full inner-state suite**

Run: `pytest tests/test_inner_state_features.py -v`
Expected: all PASS (scaler, extractor, dead-signal exclusion, headline regression, degeneracy + grammar freeze).

- [ ] **Step 5: Commit**

```bash
git add services/orion-spark-introspector/app/inner_state.py tests/test_inner_state_features.py
git commit -m "feat(spark-introspector): honest arithmetic headline + degeneracy freeze"
```

---

## Task 7: Corpus JSONL sink

Persist each `InnerStateFeaturesV1` to an append-only JSONL so Plan 2's encoder has a real, non-degenerate history to train on.

**Files:**
- Create: `services/orion-spark-introspector/app/inner_state_sink.py`
- Test: `tests/test_inner_state_features.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_inner_state_features.py`:

```python
def test_corpus_sink_appends_jsonl(tmp_path) -> None:
    import importlib.util as _u
    from pathlib import Path as _P
    spec = _u.spec_from_file_location(
        "spark_inner_sink",
        _P(__file__).resolve().parents[1]
        / "services" / "orion-spark-introspector" / "app" / "inner_state_sink.py",
    )
    sink_mod = _u.module_from_spec(spec)
    spec.loader.exec_module(sink_mod)

    from datetime import datetime, timezone
    from orion.schemas.telemetry.inner_state import InnerStateFeaturesV1

    path = tmp_path / "corpus.jsonl"
    sink = sink_mod.InnerStateCorpusSink(str(path))
    p = InnerStateFeaturesV1(generated_at=datetime(2026, 7, 7, tzinfo=timezone.utc), headline=0.5)
    sink.append(p)
    sink.append(p)
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    import json
    assert json.loads(lines[0])["headline"] == 0.5


def test_corpus_sink_disabled_when_no_path() -> None:
    import importlib.util as _u
    from pathlib import Path as _P
    spec = _u.spec_from_file_location(
        "spark_inner_sink2",
        _P(__file__).resolve().parents[1]
        / "services" / "orion-spark-introspector" / "app" / "inner_state_sink.py",
    )
    sink_mod = _u.module_from_spec(spec)
    spec.loader.exec_module(sink_mod)
    sink = sink_mod.InnerStateCorpusSink("")
    assert sink.enabled is False
    from datetime import datetime, timezone
    from orion.schemas.telemetry.inner_state import InnerStateFeaturesV1
    sink.append(InnerStateFeaturesV1(generated_at=datetime(2026, 7, 7, tzinfo=timezone.utc)))  # no-op, no raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_inner_state_features.py -k "corpus_sink" -v`
Expected: FAIL — `inner_state_sink.py` does not exist.

- [ ] **Step 3: Create the sink**

Create `services/orion-spark-introspector/app/inner_state_sink.py`:

```python
"""Append-only JSONL corpus sink for InnerStateFeaturesV1 (Plan 2 training data)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from orion.schemas.telemetry.inner_state import InnerStateFeaturesV1


class InnerStateCorpusSink:
    def __init__(self, path: str) -> None:
        self._path: Optional[Path] = Path(path) if path else None
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self._path is not None

    def append(self, payload: InnerStateFeaturesV1) -> None:
        if self._path is None:
            return
        line = json.dumps(payload.model_dump(mode="json"), separators=(",", ":"))
        # flush (not fsync) per tick: this is re-derivable training data, and the
        # self-state tick runs ~every 2s — a per-line fsync is needless I/O.
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_inner_state_features.py -k "corpus_sink" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add services/orion-spark-introspector/app/inner_state_sink.py tests/test_inner_state_features.py
git commit -m "feat(spark-introspector): append-only JSONL corpus sink for inner-state"
```

---

## Task 8: Introspector settings for inner features

**Files:**
- Modify: `services/orion-spark-introspector/app/settings.py`
- Modify: `services/orion-spark-introspector/.env_example`
- Test: `services/orion-spark-introspector/tests/test_inner_state_emit.py`

- [ ] **Step 1: Write the failing test**

Append to `services/orion-spark-introspector/tests/test_inner_state_emit.py`:

```python
def test_inner_features_settings_defaults() -> None:
    from app.settings import Settings
    s = Settings()
    assert s.inner_features_enabled is True
    assert s.inner_features_version == "seed-v1"
    assert s.channel_inner_features == "orion:self:inner_features"
    assert s.phi_degenerate_streak == 20
    assert s.orion_phi_encoder_enabled is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest services/orion-spark-introspector/tests/test_inner_state_emit.py::test_inner_features_settings_defaults -v`
Expected: FAIL — attributes not defined.

- [ ] **Step 3: Add settings fields**

In `services/orion-spark-introspector/app/settings.py`, add (near the other `channel_*`/feature fields, e.g. after `spark_state_valid_for_ms` at line 111):

```python
    # --- Honest inner-state features (phi Plan 1) ---
    inner_features_enabled: bool = Field(True, alias="INNER_FEATURES_ENABLED")
    inner_features_version: str = Field("seed-v1", alias="INNER_FEATURES_VERSION")
    inner_features_scaler_window_sec: int = Field(900, alias="INNER_FEATURES_SCALER_WINDOW_SEC")
    inner_features_scaler_maxlen: int = Field(256, alias="INNER_FEATURES_SCALER_MAXLEN")
    channel_inner_features: str = Field("orion:self:inner_features", alias="CHANNEL_INNER_FEATURES")
    inner_features_corpus_path: str = Field("", alias="INNER_FEATURES_CORPUS_PATH")
    phi_degenerate_streak: int = Field(20, alias="PHI_DEGENERATE_STREAK")
    # Encoder (Plan 2) — default-off placeholders for env parity only.
    orion_phi_encoder_enabled: bool = Field(False, alias="ORION_PHI_ENCODER_ENABLED")
    orion_phi_encoder_weights: str = Field("", alias="ORION_PHI_ENCODER_WEIGHTS")
```

- [ ] **Step 4: Add env keys**

In `services/orion-spark-introspector/.env_example`, append:

```
# --- Honest inner-state features (phi Plan 1) ---
INNER_FEATURES_ENABLED=true
INNER_FEATURES_VERSION=seed-v1
INNER_FEATURES_SCALER_WINDOW_SEC=900
INNER_FEATURES_SCALER_MAXLEN=256
CHANNEL_INNER_FEATURES=orion:self:inner_features
INNER_FEATURES_CORPUS_PATH=/data/orion/phi/inner_state_corpus.jsonl
PHI_DEGENERATE_STREAK=20
# Encoder (Plan 2) — default-off:
ORION_PHI_ENCODER_ENABLED=false
ORION_PHI_ENCODER_WEIGHTS=
```

- [ ] **Step 5: Sync local .env**

Run: `python scripts/sync_local_env_from_example.py`
Expected: reports new keys synced into local `services/orion-spark-introspector/.env` (report any skip-listed keys).

- [ ] **Step 6: Run test + env parity**

(NOTE: `scripts/check_env_template_parity.py` does **not** exist; `scripts/sync_local_env_from_example.py` is the parity mechanism and was run in Step 5.)

Run: `pytest services/orion-spark-introspector/tests/test_inner_state_emit.py::test_inner_features_settings_defaults -v`
Expected: test PASS. Confirm Step 5's sync reported the new keys and left no unexpected diff in the local `.env` (report any skip-listed keys).

- [ ] **Step 7: Verify `.env` is not staged**

Run: `git check-ignore services/orion-spark-introspector/.env; git status --short`
Expected: the path prints (ignored); no `.env` appears as tracked.

- [ ] **Step 8: Commit**

```bash
git add services/orion-spark-introspector/app/settings.py services/orion-spark-introspector/.env_example services/orion-spark-introspector/tests/test_inner_state_emit.py
git commit -m "feat(spark-introspector): inner-features settings + env keys"
```

---

## Task 9: Emit `InnerStateFeaturesV1` per tick + repoint Hub `stats.phi`

Wire the extractor into `handle_self_state`: build the vector, publish it, append to the corpus, and broadcast the honest headline to the Hub EKG (replacing the geometric-mean coherence at the WS `stats.phi` sites).

**Files:**
- Modify: `services/orion-spark-introspector/app/worker.py` (module state near line 81; `handle_self_state` 2226-2316; WS sites 1074, 1266, 1549)
- Test: `services/orion-spark-introspector/tests/test_inner_state_emit.py`

- [ ] **Step 1: Write the failing test**

Append to `services/orion-spark-introspector/tests/test_inner_state_emit.py`:

```python
@pytest.mark.asyncio
async def test_handle_self_state_emits_inner_features_and_honest_phi(monkeypatch, tmp_path) -> None:
    published = []
    broadcasts = []

    class _Bus:
        enabled = True

        async def publish(self, channel, env):
            published.append((channel, env))

    async def _capture_broadcast(payload):
        broadcasts.append(payload)

    monkeypatch.setattr(worker, "_pub_bus", _Bus(), raising=False)
    monkeypatch.setattr(worker.manager, "broadcast", _capture_broadcast, raising=False)
    monkeypatch.setattr(worker.settings, "inner_features_corpus_path", str(tmp_path / "c.jsonl"), raising=False)
    # fresh module state
    monkeypatch.setattr(worker, "_INNER_SCALER", worker._new_inner_scaler(), raising=False)
    monkeypatch.setattr(worker, "_INNER_PREV_FELT", None, raising=False)
    monkeypatch.setattr(worker, "_INNER_PREV_HEADLINE", None, raising=False)
    monkeypatch.setattr(worker, "_INNER_DEGENERATE_STREAK", 0, raising=False)

    env = BaseEnvelope(
        kind="substrate.self_state.v1",
        source=ServiceRef(name="substrate-runtime", node="athena"),
        payload=_self_state_payload(),
    )
    await worker.handle_self_state(env)

    channels = [c for c, _ in published]
    assert worker.settings.channel_inner_features in channels

    # the tissue.update carries the HONEST headline (>0.5), not the 0.01 floor
    tissue = [b for b in broadcasts if b.get("type") == "tissue.update"]
    assert tissue, "expected a tissue.update broadcast"
    assert tissue[-1]["stats"]["phi"] > 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest services/orion-spark-introspector/tests/test_inner_state_emit.py::test_handle_self_state_emits_inner_features_and_honest_phi -v`
Expected: FAIL — `_new_inner_scaler`/`_INNER_*` not defined; inner-features channel not published; `stats.phi` still the geometric-mean coherence (≈0.01).

- [ ] **Step 3: Add module state + helpers**

In `services/orion-spark-introspector/app/worker.py`, after `_PREV_PHI` (line 82), add:

```python
from app.inner_state import (
    RollingRobustScaler,
    build_inner_state_features,
)
from app.inner_state_sink import InnerStateCorpusSink


def _new_inner_scaler() -> RollingRobustScaler:
    return RollingRobustScaler(maxlen=int(getattr(settings, "inner_features_scaler_maxlen", 256)))


_INNER_SCALER: RollingRobustScaler = _new_inner_scaler()
_INNER_PREV_FELT = None
_INNER_PREV_HEADLINE = None
_INNER_DEGENERATE_STREAK = 0
_INNER_LAST_HEADLINE: float | None = None  # honest headline for WS reads
_INNER_SINK = InnerStateCorpusSink(getattr(settings, "inner_features_corpus_path", "") or "")


class InnerStateFeaturesEnvelope(Envelope[InnerStateFeaturesV1]):
    kind: str = Field("self.inner_features.v1", frozen=True)
```

Add the schema import at the top of the file with the other `orion.schemas` imports:

```python
from orion.schemas.telemetry.inner_state import InnerStateFeaturesV1
```

- [ ] **Step 4: Build + emit inside `handle_self_state`**

In `handle_self_state`, after `set_latest_self_state(ss)` (line 2238), add:

```python
    global _INNER_PREV_FELT, _INNER_PREV_HEADLINE, _INNER_DEGENERATE_STREAK, _INNER_LAST_HEADLINE
    if settings.inner_features_enabled:
        inner, _INNER_PREV_FELT, _INNER_DEGENERATE_STREAK = build_inner_state_features(
            ss,
            _INNER_SCALER,
            features_version=settings.inner_features_version,
            grammar_degraded=False,  # Plan 2: wire cross-service grammar-truth gate
            prev_felt=_INNER_PREV_FELT,
            prev_headline=_INNER_PREV_HEADLINE,
            degenerate_streak=_INNER_DEGENERATE_STREAK,
            degenerate_limit=int(settings.phi_degenerate_streak),
        )
        _INNER_PREV_HEADLINE = inner.headline
        _INNER_LAST_HEADLINE = inner.headline
        _INNER_SINK.append(inner)
        if _pub_bus and _pub_bus.enabled:
            try:
                await _pub_bus.publish(
                    settings.channel_inner_features,
                    InnerStateFeaturesEnvelope(source=_svc_ref(), correlation_id=uuid4(), payload=inner),
                )
            except Exception as exc:
                logger.warning("Failed to publish inner_features: %s", exc)
```

- [ ] **Step 5: Repoint the substrate-tick WS `stats.phi`**

In `handle_self_state`, change the tissue WS payload `stats.phi` (line 2305) from:

```python
                "phi": float(phi_now.get("coherence", 0.5)),
```

to the honest headline:

```python
                "phi": float(_INNER_LAST_HEADLINE if _INNER_LAST_HEADLINE is not None else phi_now.get("coherence", 0.5)),
```

- [ ] **Step 6: Repoint the candidate/trace WS `stats.phi` sites**

Add a helper next to `_get_phi_stats` (line 149):

```python
def _headline_stat(phi_stats: Dict[str, float]) -> float:
    """Honest inner-state headline for WS EKG; falls back to legacy coherence."""
    if _INNER_LAST_HEADLINE is not None:
        return float(_INNER_LAST_HEADLINE)
    return float(phi_stats.get("coherence", 0.0))
```

Replace the WS-facing coherence reads at lines 1074, 1266, and 1549:
- Line 1074 `"phi": float(phi_stats.get("coherence", 0.0)),` → `"phi": _headline_stat(phi_stats),`
- Line 1266 `"phi": float(phi_stats.get("coherence", 0.0)),` → `"phi": _headline_stat(phi_stats),`
- Line 1549 `"phi": coherence_stat,` → `"phi": _headline_stat(phi_stats),`

(Do NOT touch the durable `SparkTelemetryPayload.phi` at line 1375 or the `SparkStateSnapshotV1.phi` dict — those keep the component breakdown; only the Hub EKG `stats.phi` headline changes in Plan 1.)

- [ ] **Step 7: Run test to verify it passes**

Run: `pytest services/orion-spark-introspector/tests/test_inner_state_emit.py -v`
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add services/orion-spark-introspector/app/worker.py services/orion-spark-introspector/tests/test_inner_state_emit.py
git commit -m "feat(spark-introspector): emit InnerStateFeaturesV1 + honest headline to Hub EKG"
```

---

## Task 10: Light up the cognitive grammar lanes as deployed defaults

The cortex-exec/chat grammar lanes (which feed `execution_pressure`/`reasoning_pressure`) are code-default-off while `.env_example` ships them `true` — a parity gap. Flip the four code defaults to match, so the cognitive signal flows in deployment, not only where the operator set env.

**Files:**
- Modify: `services/orion-cortex-exec/app/settings.py:133`
- Modify: `services/orion-substrate-runtime/app/settings.py:20-28`
- Modify: `services/orion-hub/app/settings.py:520`
- Test: `tests/test_grammar_lane_defaults.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_grammar_lane_defaults.py`:

```python
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load_settings(rel: str, mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cortex_exec_grammar_default_on() -> None:
    src = (REPO / "services/orion-cortex-exec/app/settings.py").read_text()
    assert 'publish_cortex_exec_grammar: bool = Field(True' in src


def test_substrate_reducers_default_on() -> None:
    import re
    src = (REPO / "services/orion-substrate-runtime/app/settings.py").read_text()
    # enable_chat_grammar_reducer is single-line; execution_trajectory is multi-line
    # (Field(\n    False, ...)) so match across whitespace/newlines with a regex.
    assert 'enable_chat_grammar_reducer: bool = Field(True' in src
    assert re.search(
        r"enable_execution_trajectory_reducer[^=]*=\s*Field\(\s*True", src
    ), "ENABLE_EXECUTION_TRAJECTORY_REDUCER default must be flipped to True"


def test_hub_chat_grammar_default_on() -> None:
    src = (REPO / "services/orion-hub/app/settings.py").read_text()
    assert 'PUBLISH_HUB_CHAT_GRAMMAR: bool = Field(default=True' in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_grammar_lane_defaults.py -v`
Expected: FAIL — defaults are currently `False`.

- [ ] **Step 3: Flip cortex-exec default**

In `services/orion-cortex-exec/app/settings.py:133`, change:

```python
    publish_cortex_exec_grammar: bool = Field(False, alias="PUBLISH_CORTEX_EXEC_GRAMMAR")
```
to:
```python
    publish_cortex_exec_grammar: bool = Field(True, alias="PUBLISH_CORTEX_EXEC_GRAMMAR")
```

- [ ] **Step 4: Flip substrate-runtime defaults**

In `services/orion-substrate-runtime/app/settings.py`, change the `enable_execution_trajectory_reducer` Field default (lines 20-22) to `True`, and line 28:

```python
    enable_chat_grammar_reducer: bool = Field(False, alias="ENABLE_CHAT_GRAMMAR_REDUCER")
```
to:
```python
    enable_chat_grammar_reducer: bool = Field(True, alias="ENABLE_CHAT_GRAMMAR_REDUCER")
```

(For `enable_execution_trajectory_reducer`, set the first positional `Field(...)` argument to `True`; keep the `alias="ENABLE_EXECUTION_TRAJECTORY_REDUCER"`.)

- [ ] **Step 5: Flip hub default**

In `services/orion-hub/app/settings.py:520`, change:

```python
    PUBLISH_HUB_CHAT_GRAMMAR: bool = Field(default=False, alias="PUBLISH_HUB_CHAT_GRAMMAR")
```
to:
```python
    PUBLISH_HUB_CHAT_GRAMMAR: bool = Field(default=True, alias="PUBLISH_HUB_CHAT_GRAMMAR")
```

- [ ] **Step 6: Confirm `.env_example` parity + sync**

(NOTE: `scripts/check_env_template_parity.py` does **not** exist; sync is the mechanism.)

Run: `python scripts/sync_local_env_from_example.py`
Expected: sync reports no changes for these keys (env_example already ships them `true`). `PUBLISH_CORTEX_EXEC_GRAMMAR` is on the sync skip-list — report it explicitly if skipped.

- [ ] **Step 7: Run test to verify it passes**

Run: `pytest tests/test_grammar_lane_defaults.py -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add services/orion-cortex-exec/app/settings.py services/orion-substrate-runtime/app/settings.py services/orion-hub/app/settings.py tests/test_grammar_lane_defaults.py
git commit -m "feat(grammar): default-on cortex-exec/chat grammar lanes (code parity with env)"
```

---

## Task 11: Retire the "PHI (Order)" Hub label

The Hub EKG iframe (`services/orion-hub/templates/index.html:143` → `/spark/ui`) renders `services/orion-spark-introspector/app/static/index.html`. The big stat `val-phi` is fed by the WS `stats.phi` (now the honest headline). Relabel so the UI names what it now shows, with provenance.

**Files:**
- Modify: `services/orion-spark-introspector/app/static/index.html:80,101,111`
- Modify: `services/orion-spark-introspector/app/static/tissue_viz.js:109`

- [ ] **Step 1: Relabel the stat box**

In `services/orion-spark-introspector/app/static/index.html:80`, change:

```html
                <div class="stat-box"><span class="stat-label">PHI (Order)</span><span id="val-phi" class="stat-val" style="color: #0f0;">0.00</span></div>
```
to:
```html
                <div class="stat-box"><span class="stat-label" title="Honest inner-state headline (cold-start feature aggregate; encoder φ pending)">INNER STATE (φ)</span><span id="val-phi" class="stat-val" style="color: #0f0;">0.00</span></div>
```

- [ ] **Step 2: Relabel the diagnostics row + legend**

In `services/orion-spark-introspector/app/static/index.html:101`, change the row label `<td>PHI</td>` to `<td>Inner State (φ)</td>`.

In `services/orion-spark-introspector/app/static/index.html:111`, change the expander title/description to reflect the honest aggregate:

```html
                        <div class="exp-col"><div class="exp-title">φ Inner State (headline)</div>Honest aggregate of Orion's felt+cognitive signals (arithmetic, not a single-factor floor).<br><strong>High:</strong> Integrated, resourced, low load.<br><strong>Low:</strong> Fragmented or heavily loaded.</div>
```

- [ ] **Step 3: Update the JS diagnostics label**

In `services/orion-spark-introspector/app/static/tissue_viz.js:109`, the `updateRow('phi', ...)` call renders the diagnostics state text. Change the human-readable state strings to reflect inner state (keep the `'phi'` row id so `getElementById('diag-phi-*')` still matches):

```javascript
        updateRow('phi', p, p > 0.5 ? "HIGH" : "LOW", p > 0.5 ? "Integrated / Resourced" : "Loaded / Fragmented");
```

- [ ] **Step 4: Interaction smoke (manual, documented)**

Because this is a static UI fed by a live WS, verify the rendered label + live value together (server-start alone is not proof):

Run:
```bash
docker compose --env-file .env --env-file services/orion-spark-introspector/.env \
  -f services/orion-spark-introspector/docker-compose.yml up -d --build
curl -fsS http://localhost:8000/spark/ui | grep -i "INNER STATE"
```
Expected: the served HTML contains `INNER STATE (φ)` (confirms the template is the one served and the relabel is live). Then load `/spark/ui` in the Hub and confirm the big number tracks a value `> 0.01` (the honest headline), not the old floor. (Port per the service compose; adjust if different.)

- [ ] **Step 5: Commit**

```bash
git add services/orion-spark-introspector/app/static/index.html services/orion-spark-introspector/app/static/tissue_viz.js
git commit -m "feat(hub-ekg): retire PHI (Order) label; show honest inner-state headline"
```

---

## Task 12: Replay eval — headline responsive & non-degenerate

Prove on the recorded pinned-`0.01` window that φ is no longer floored, infra can't move the felt vector, and the headline responds to real cognitive load changes.

**Files:**
- Create: `services/orion-spark-introspector/evals/eval_inner_state_readout.py`

- [ ] **Step 1: Write the eval**

Create `services/orion-spark-introspector/evals/eval_inner_state_readout.py`:

```python
"""Eval: honest inner-state headline is responsive and never floored.

Run: python services/orion-spark-introspector/evals/eval_inner_state_readout.py
Exits non-zero on failure so it can gate CI.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
# Running a script puts the script's dir on sys.path[0], NOT repo root, and
# `orion` is not pip-installed. inner_state.py imports orion.schemas.* at import
# time, so repo root must be on the path before we exec the module.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
_spec = importlib.util.spec_from_file_location(
    "spark_inner_state_eval",
    REPO / "services" / "orion-spark-introspector" / "app" / "inner_state.py",
)
inner_state = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(inner_state)


def _raw(**over):
    base = {
        "coherence": 1.0, "field_intensity": 1.0, "agency_readiness": 0.41,
        "execution_pressure": 0.0, "reasoning_pressure": 0.05,
        "resource_pressure": 1.0, "reliability_pressure": 1.0,
        "continuity_pressure": 0.0, "social_pressure": 0.0,
    }
    base.update(over)
    return base


def main() -> int:
    failures = []

    pinned = inner_state.honest_headline(_raw())
    if pinned <= 0.01:
        failures.append(f"pinned window still floored: {pinned}")

    # infra can't move the felt vector: reliability comes ONLY from felt inputs;
    # changing an infra-only concept (not in the raw felt map) must not change headline.
    h1 = inner_state.honest_headline(_raw())
    h2 = inner_state.honest_headline(_raw())  # identical felt inputs
    if h1 != h2:
        failures.append("headline non-deterministic on identical felt inputs")

    # responsive: rising execution load lowers headline
    calm = inner_state.honest_headline(_raw(execution_pressure=0.0, reasoning_pressure=0.0,
                                            resource_pressure=0.1, reliability_pressure=0.1))
    busy = inner_state.honest_headline(_raw(execution_pressure=0.9, reasoning_pressure=0.8,
                                            resource_pressure=0.1, reliability_pressure=0.1))
    if not (busy < calm):
        failures.append(f"headline not responsive to cognitive load: calm={calm} busy={busy}")

    if failures:
        print("EVAL FAIL:")
        for f in failures:
            print(" -", f)
        return 1
    print(f"EVAL PASS: pinned={pinned:.3f} calm={calm:.3f} busy={busy:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run the eval**

Run: `python services/orion-spark-introspector/evals/eval_inner_state_readout.py`
Expected: `EVAL PASS: pinned=0.70 ...` (exit 0).

- [ ] **Step 3: Commit**

```bash
git add services/orion-spark-introspector/evals/eval_inner_state_readout.py
git commit -m "test(spark-introspector): replay eval for honest inner-state headline"
```

---

## Task 13: Full gate + review

- [ ] **Step 1: Run the touched test suites**

Run:
```bash
pytest tests/test_inner_state_schema.py tests/test_inner_state_features.py \
       tests/test_self_state_reliability_decontamination.py tests/test_grammar_lane_defaults.py \
       services/orion-spark-introspector/tests/test_inner_state_emit.py \
       services/orion-spark-introspector/tests/test_spark_meta_patch_tissue_refresh.py -q
```
Expected: all PASS.

- [ ] **Step 2: Contract + parity gates**

(NOTE: this repo has no `scripts/check_*.py` gates — contract enforcement is pytest-based; only `scripts/sync_local_env_from_example.py` exists.)

Run:
```bash
git diff --check
python scripts/sync_local_env_from_example.py
pytest tests/test_channel_prefix_guardrail.py tests/test_schema_registry_import_light.py -q
```
Expected: no whitespace errors; sync reports the introspector keys present (note any skip-listed); both contract tests PASS.

- [ ] **Step 3: Run the eval**

Run: `python services/orion-spark-introspector/evals/eval_inner_state_readout.py`
Expected: `EVAL PASS`.

- [ ] **Step 4: Code review**

Dispatch the code-reviewer subagent over the branch diff. Fix all material findings and re-run Steps 1-3. Record findings + fixes for the PR report.

- [ ] **Step 5: Push + PR**

```bash
git push -u origin HEAD
```
Then open the PR with the report (template in `AGENTS.md §18`), including the restart commands below.

---

## Restart required (report to Juniper; do not sudo yourself)

Config + code changes take effect only after restart of the affected services:

```bash
# self-state policy + field topology decontamination:
docker compose --env-file .env --env-file services/orion-substrate-runtime/.env \
  -f services/orion-substrate-runtime/docker-compose.yml up -d --build
docker compose --env-file .env --env-file services/orion-field-digester/.env \
  -f services/orion-field-digester/docker-compose.yml up -d --build

# lane default flips:
docker compose --env-file .env --env-file services/orion-cortex-exec/.env \
  -f services/orion-cortex-exec/docker-compose.yml up -d --build
docker compose --env-file .env --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml up -d --build

# feature emission + UI:
docker compose --env-file .env --env-file services/orion-spark-introspector/.env \
  -f services/orion-spark-introspector/docker-compose.yml up -d --build
```

Runtime verification after restart:
```bash
# inner-features flowing on the bus:
redis-cli -u redis://100.92.216.81:6379/0 PSUBSCRIBE "orion:self:inner_features"
# corpus accumulating:
tail -f /data/orion/phi/inner_state_corpus.jsonl
# Hub headline no longer floored:
curl -fsS http://localhost:8000/spark/ui | grep -i "INNER STATE"
```

---

## Self-review (against the spec)

**Spec coverage:**
- Light up lanes → Task 10. ✅ (defaults flipped; verification in Restart section)
- Feature vector `InnerStateFeaturesV1`, raw+scaled+source per feature → Tasks 4, 5. ✅
- Dead-signal exclusion (`policy_pressure`, `uncertainty`); infra sub-vector only → Tasks 4, 5 (`test_dead_signals_excluded_from_features`). ✅
- Robust scaling (median/IQR) → Task 3. ✅
- Decontaminate builder (`contract_pressure`/`catalog_drift` out of felt reliability) → Task 2. ✅
- `phi_health` freeze on degenerate + would-have-fired on pinned-`0.01` → Task 6. ✅
- Cold-start honest headline, no geometric floor, felt+cognitive only → Task 6 + Task 9. ✅
- Corpus persistence → Task 7. ✅
- Hub retire "PHI (Order)" + provenance → Task 11. ✅
- Registry + channel `orion:self:inner_features` → Task 4. ✅
- Env keys incl. encoder placeholders → Task 8. ✅
- Replay eval (pinned window not floored; responsive) → Task 12. ✅
- UUID crash (blocks substrate snapshot rail) → Task 1. ✅

**Spec items deferred to Plan 2 (documented in "Scope, split, and deferrals"):** the self-supervised encoder + `φ = f(latent)` + recon-error-as-coherence; per-latent probes/attributions; discrete cognitive-event features from a direct `ExecutionRunStateV1` read (recall-gate fired, `reasoning_present`, step-fail rate); cross-service grammar-truth liveness gate; continuous `Δφ` reward handoff. Plan-1 `grammar_truth_degraded` defaults `False` and the degeneracy-streak freeze stands in as the GIGO guard.

**Type consistency:** `build_inner_state_features(...) -> (InnerStateFeaturesV1, felt_tuple, streak)` used identically in Tasks 5, 6, 9, and tests. `honest_headline(raw: dict) -> float`, `RollingRobustScaler(maxlen)`, `robust_scale(value, *, median, iqr)`, `InnerStateCorpusSink(path).append(payload)` consistent across tasks. Schema field names (`features`, `infra`, `headline`, `headline_source`, `phi_health`, `phi_degenerate_streak`, `grammar_truth_degraded`) match between schema (Task 4), extractor (Tasks 5-6), and emit test (Task 9).

**Placeholder scan:** no `TODO`/`fill in`/"add validation" placeholders; every code step shows complete code. The one cross-task coupling (`honest_headline` referenced in Task 5, defined in Task 6) is called out explicitly with the instruction to commit the module body together.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-07-phi-inner-state-truthful.md`. Two execution options:

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration (REQUIRED SUB-SKILL: superpowers:subagent-driven-development).

**2. Inline Execution** — execute tasks in this session with checkpoints (REQUIRED SUB-SKILL: superpowers:executing-plans).

Which approach?
