# Substrate Signal Bridge V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Materialize existing `OrionSignalV1` records produced by `CognitionTraceAdapter` (organ `cortex_exec`, signal kinds `cognition_run` and `cognition_step`) into the shared substrate grammar as `SubstrateMoleculeV1` of a new `organ_signal` kind, so the substrate can consume real organ signals without inventing a parallel signal schema.

**Architecture:** Add one pure projection module `orion/substrate/signal_bridge.py` that takes an `OrionSignalV1` and returns one valid `SubstrateMoleculeV1`. Register exactly one new molecule kind (`organ_signal`) in `default_registry()`; do not touch atoms, predicates, gradient keys, or any existing signal/substrate code. Optionally add a sidecar bus worker that appends bridged molecules to `MoleculeJsonlStore` and records them on `SubstrateExperimentHarness`, so the existing daily rollup picks them up under `organ_coverage["cortex_exec"]`.

**Tech Stack:** Python 3.12, Pydantic v2, pytest. Reuses `orion.schema_kernel`, `orion.substrate.molecules`, `orion.substrate.molecule_store`, `orion.substrate.experiment.harness`, `orion.signals.models`. Does not modify them.

---

## Spec deviations from the supplied design doc

These are corrections made while reading the actual repo; they are intentional and consistent with the spec's own non-negotiable constraint "Do not add new atom kinds."

1. **Atom key `process` does not exist.** The supplied spec asserts "the repo already defines `signal`, `process`, `context`, `gradient`, and `evidence`." It does not. `orion/schema_kernel/atom.py:14-27` declares 12 atom kinds: `signal, constraint, attention, state, change, relation, context, agency, evidence, gradient, persistence, boundary`. Since the spec also says "Do not add new atom kinds," this plan substitutes `agency` for the `source_process` role. `ConceptAtomV1(key="agency", ..., description="The locus from which a transformation originates.", axes=("source", "intent"))` is the closest semantic fit: the cognition run is the agent/locus that produced the signal.

2. **`registry.py:default_registry()` is the legitimate one-line schema expansion.** Add `"organ_signal"` to the `molecule_kinds=(…)` tuple at `orion/schema_kernel/registry.py:98-103`. Nothing else in the schema kernel is modified.

3. **No live bus worker by default.** The supplied spec marks `SubstrateSignalBusWorker` as optional. This plan keeps it optional and gated behind Tasks 7–8; the substrate molecule round-trip and the end-to-end harness/rollup proof do not require a real `OrionBusAsync` instance.

---

## File structure

Each file owns one responsibility. Bridge stays pure (no I/O); worker stays thin (I/O only).

| File | Status | Responsibility |
| --- | --- | --- |
| `orion/schema_kernel/registry.py` | Modify (one-line) | Register `organ_signal` molecule kind in `default_registry()`. |
| `orion/substrate/signal_bridge.py` | Create | Pure projection: `OrionSignalV1 → SubstrateMoleculeV1`. Gradient mapping, supports-check, batch helper. |
| `orion/substrate/signal_bus_worker.py` | Create (Tasks 7–8 only) | Subscribe to `orion:signals:cortex_exec`, bridge, persist to `MoleculeJsonlStore`, optionally record on harness. |
| `tests/test_substrate_signal_bridge.py` | Create | Bridge unit tests: happy path, failed run, step error, unsupported skip, registry validation. |
| `tests/test_substrate_signal_bus_worker.py` | Create (Tasks 7–8 only) | Worker handle_envelope tests using fake envelope. |
| `tests/test_substrate_signal_bridge_e2e.py` | Create | End-to-end: bridge → store → harness → daily rollup includes `cortex_exec` under `organ_coverage`. |

Tests live in repo-root `tests/` per existing convention (`tests/test_substrate_*.py`).

**Do NOT modify** (locked by spec):
- `orion/schema_kernel/atom.py`
- `orion/schema_kernel/relation.py`
- `orion/schema_kernel/gradient.py`
- `orion/substrate/molecules.py`
- `orion/substrate/molecule_store.py`
- `orion/substrate/operators.py`
- `orion/signals/models.py`
- `orion/signals/adapters/cognition_trace.py`
- `services/orion-signal-gateway/app/processor.py`

---

## Shared test fixtures

Several tests construct an `OrionSignalV1`. Put this helper at module top of `tests/test_substrate_signal_bridge.py` and re-import it (or duplicate it verbatim) in `tests/test_substrate_signal_bridge_e2e.py` and `tests/test_substrate_signal_bus_worker.py`. It mirrors the real shape `CognitionTraceAdapter` emits at `orion/signals/adapters/cognition_trace.py:98-120`.

```python
from datetime import datetime, timezone
from orion.signals.models import OrganClass, OrionSignalV1


def make_cognition_run_signal(
    *,
    signal_kind: str = "cognition_run",
    dimensions: dict[str, float] | None = None,
    source_event_id: str = "corr-test-1",
) -> OrionSignalV1:
    now = datetime.now(timezone.utc)
    return OrionSignalV1(
        signal_id="sig-test-1",
        organ_id="cortex_exec",
        organ_class=OrganClass.endogenous,
        signal_kind=signal_kind,
        dimensions=dimensions or {},
        causal_parents=[],
        source_event_id=source_event_id,
        otel_trace_id="00000000000000000000000000000001",
        otel_span_id="0000000000000001",
        observed_at=now,
        emitted_at=now,
        summary="test signal",
        notes=[],
    )
```

---

## Task 1: Register `organ_signal` molecule kind in `default_registry()`

**Files:**
- Modify: `orion/schema_kernel/registry.py:98-103`
- Test: `tests/test_substrate_signal_bridge.py` (new file — add a single targeted test here first)

- [ ] **Step 1: Write the failing registry test**

Create `tests/test_substrate_signal_bridge.py` with just this test for now:

```python
"""Substrate signal bridge: OrionSignalV1 → SubstrateMoleculeV1."""

from __future__ import annotations

from orion.schema_kernel import default_registry


def test_default_registry_includes_organ_signal_molecule_kind():
    registry = default_registry()
    assert registry.has_molecule_kind("organ_signal"), (
        "default_registry() must register 'organ_signal' so the signal bridge "
        "can emit molecules that validate."
    )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_substrate_signal_bridge.py::test_default_registry_includes_organ_signal_molecule_kind -v`
Expected: `FAILED` — assertion error, `'organ_signal'` not in registered molecule kinds.

- [ ] **Step 3: Add `organ_signal` to default molecule kinds**

In `orion/schema_kernel/registry.py`, change the tuple in `default_registry()`:

```python
def default_registry() -> SchemaKernelRegistry:
    """Build a registry seeded with the canonical defaults.

    Molecule kinds expected by the MVP organs are pre-registered so emit calls
    do not blow up before the first run.
    """

    return SchemaKernelRegistry(
        atoms=DEFAULT_ATOMS,
        predicates=DEFAULT_PREDICATES,
        molecule_kinds=(
            "observation",
            "claim",
            "pressure",
            "contradiction",
            "organ_signal",
        ),
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_substrate_signal_bridge.py::test_default_registry_includes_organ_signal_molecule_kind -v`
Expected: `PASSED`.

- [ ] **Step 5: Run the full existing substrate test suite to confirm no regression**

Run: `pytest tests/test_substrate_kernel_molecules.py tests/test_substrate_kernel_atoms.py tests/test_substrate_kernel_operators.py tests/test_substrate_organ_emit.py tests/test_substrate_experiment_harness.py -v`
Expected: all pre-existing tests still PASS (no test currently asserts the molecule-kinds tuple length, so the new kind cannot break them).

- [ ] **Step 6: Commit**

```bash
git add orion/schema_kernel/registry.py tests/test_substrate_signal_bridge.py
git commit -m "feat(schema_kernel): register organ_signal molecule kind for substrate bridge"
```

---

## Task 2: Bridge skeleton + happy-path `cognition_run` conversion

**Files:**
- Create: `orion/substrate/signal_bridge.py`
- Modify: `tests/test_substrate_signal_bridge.py`

- [ ] **Step 1: Add the helper + happy-path test**

Append to `tests/test_substrate_signal_bridge.py`:

```python
from datetime import datetime, timezone

import pytest

from orion.schema_kernel import default_registry
from orion.signals.models import OrganClass, OrionSignalV1
from orion.substrate.molecules import validate_molecule


def make_cognition_run_signal(
    *,
    signal_kind: str = "cognition_run",
    dimensions: dict[str, float] | None = None,
    source_event_id: str = "corr-test-1",
) -> OrionSignalV1:
    now = datetime.now(timezone.utc)
    return OrionSignalV1(
        signal_id="sig-test-1",
        organ_id="cortex_exec",
        organ_class=OrganClass.endogenous,
        signal_kind=signal_kind,
        dimensions=dimensions or {},
        causal_parents=[],
        source_event_id=source_event_id,
        otel_trace_id="00000000000000000000000000000001",
        otel_span_id="0000000000000001",
        observed_at=now,
        emitted_at=now,
        summary="test signal",
        notes=[],
    )


def test_cognition_run_success_converts_to_organ_signal_molecule():
    from orion.substrate.signal_bridge import signal_to_molecule

    signal = make_cognition_run_signal(
        dimensions={
            "success": 1.0,
            "step_count": 0.15,
            "latency_level": 0.30,
            "recall_used": 1.0,
            "reasoning_present": 1.0,
            "final_text_present": 1.0,
        },
    )

    molecule = signal_to_molecule(signal)

    assert molecule.molecule_kind == "organ_signal"
    assert molecule.provenance["organ"] == "cortex_exec"
    assert molecule.provenance["signal_kind"] == "cognition_run"
    assert molecule.provenance["signal_id"] == "sig-test-1"
    assert molecule.provenance["source_event_id"] == "corr-test-1"
    assert molecule.atoms["primary"] == "signal"
    assert molecule.atoms["source_process"] == "agency"
    assert molecule.atoms["source_context"] == "context"
    assert molecule.atoms["field"] == "gradient"
    assert molecule.atoms["witness"] == "evidence"

    assert molecule.gradients["salience"] == pytest.approx(0.30)
    assert molecule.gradients["contradiction"] == pytest.approx(0.0)
    assert molecule.gradients["novelty"] == pytest.approx(0.0)
    assert molecule.gradients["coherence"] == pytest.approx(1.0)

    validate_molecule(molecule, default_registry())
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_substrate_signal_bridge.py::test_cognition_run_success_converts_to_organ_signal_molecule -v`
Expected: `FAILED` — `ImportError: cannot import name 'signal_to_molecule' from 'orion.substrate.signal_bridge'` (module does not exist yet).

- [ ] **Step 3: Create the bridge module with full implementation**

Note: the bridge file is small enough that there is no value in splitting `dimensions_to_gradients` across tasks. Write the full module now; subsequent tasks add tests that exercise the already-implemented branches.

Create `orion/substrate/signal_bridge.py`:

```python
"""Substrate signal bridge — project OrionSignalV1 into SubstrateMoleculeV1.

This module is a pure projection. It does not subscribe to a bus, does not
persist anything, and does not introduce a new signal schema. It exists so the
substrate can consume the existing organ signal stream through one well-defined
seam.

Supported inputs (MVP):
    (organ_id="cortex_exec", signal_kind="cognition_run")
    (organ_id="cortex_exec", signal_kind="cognition_step")

These are the signals produced by orion.signals.adapters.cognition_trace.
"""

from __future__ import annotations

from typing import Iterable

from orion.schema_kernel import (
    ConceptRelationV1,
    clamp_gradient,
    default_registry,
)
from orion.signals.models import OrionSignalV1
from orion.substrate.molecules import SubstrateMoleculeV1, validate_molecule


SUPPORTED_SIGNAL_KINDS: frozenset[tuple[str, str]] = frozenset(
    {
        ("cortex_exec", "cognition_run"),
        ("cortex_exec", "cognition_step"),
    }
)


def supports_signal(signal: OrionSignalV1) -> bool:
    return (signal.organ_id, signal.signal_kind) in SUPPORTED_SIGNAL_KINDS


def _dim(signal: OrionSignalV1, key: str, default: float = 0.0) -> float:
    value = (signal.dimensions or {}).get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def signal_intensity(signal: OrionSignalV1) -> float:
    return clamp_gradient(
        max(
            _dim(signal, "salience"),
            _dim(signal, "level"),
            _dim(signal, "latency_level"),
            _dim(signal, "step_count"),
            _dim(signal, "service_count"),
        )
    )


def signal_confidence(signal: OrionSignalV1) -> float:
    return clamp_gradient(_dim(signal, "confidence", 1.0))


def dimensions_to_gradients(signal: OrionSignalV1) -> dict[str, float]:
    dims = signal.dimensions or {}
    success_present = "success" in dims
    success = _dim(signal, "success", 0.0)

    salience = signal_intensity(signal)

    contradiction = max(
        _dim(signal, "contradiction"),
        _dim(signal, "error_present"),
        1.0 - success if success_present else 0.0,
    )

    novelty = max(
        _dim(signal, "novelty"),
        _dim(signal, "surprise"),
    )

    coherence = max(
        _dim(signal, "coherence"),
        success if success_present else 0.0,
    )

    confidence = signal_confidence(signal)

    return {
        "salience": clamp_gradient(salience),
        "contradiction": clamp_gradient(contradiction),
        "novelty": clamp_gradient(novelty),
        "coherence": clamp_gradient(coherence * confidence),
    }


def signal_to_molecule(signal: OrionSignalV1) -> SubstrateMoleculeV1:
    if not supports_signal(signal):
        raise ValueError(
            f"unsupported substrate signal bridge: "
            f"{signal.organ_id}.{signal.signal_kind}"
        )

    intensity = signal_intensity(signal)
    confidence = signal_confidence(signal)

    molecule = SubstrateMoleculeV1(
        molecule_kind="organ_signal",
        atoms={
            "primary": "signal",
            "source_process": "agency",
            "source_context": "context",
            "field": "gradient",
            "witness": "evidence",
        },
        relations=[
            ConceptRelationV1(
                source="primary",
                predicate="references",
                target="source_process",
                weight=1.0,
            ),
            ConceptRelationV1(
                source="primary",
                predicate="elicits",
                target="field",
                weight=intensity,
            ),
            ConceptRelationV1(
                source="witness",
                predicate="supports",
                target="primary",
                weight=confidence,
                polarity=1.0,
            ),
        ],
        gradients=dimensions_to_gradients(signal),
        provenance={
            "organ": signal.organ_id,
            "signal_id": signal.signal_id,
            "signal_kind": signal.signal_kind,
            "source_event_id": signal.source_event_id,
            "otel_trace_id": signal.otel_trace_id,
            "otel_span_id": signal.otel_span_id,
        },
        payload={
            "dimensions": dict(signal.dimensions or {}),
            "summary": signal.summary,
            "notes": list(signal.notes or []),
            "causal_parents": list(signal.causal_parents or []),
            "observed_at": signal.observed_at.isoformat(),
            "emitted_at": signal.emitted_at.isoformat(),
        },
    )

    validate_molecule(molecule, default_registry())
    return molecule


def signals_to_molecules(
    signals: Iterable[OrionSignalV1],
) -> list[SubstrateMoleculeV1]:
    return [signal_to_molecule(s) for s in signals if supports_signal(s)]
```

- [ ] **Step 4: Run the happy-path test to verify it passes**

Run: `pytest tests/test_substrate_signal_bridge.py::test_cognition_run_success_converts_to_organ_signal_molecule -v`
Expected: `PASSED`.

Cross-check on math (so a regression jumps out):
- `signal_intensity = max(0, 0, 0.30, 0.15, 0) = 0.30` → `salience == 0.30` ✓
- `success_present=True, success=1.0` → `contradiction = max(0, 0, 1.0 - 1.0) = 0.0` ✓
- `coherence = max(0, 1.0) * confidence(default 1.0) = 1.0` ✓
- `novelty = max(0, 0) = 0.0` ✓

- [ ] **Step 5: Commit**

```bash
git add orion/substrate/signal_bridge.py tests/test_substrate_signal_bridge.py
git commit -m "feat(substrate): add OrionSignalV1 → SubstrateMoleculeV1 bridge"
```

---

## Task 3: Failed run maps to contradiction

The implementation already covers this branch; this task locks it with a test.

**Files:**
- Modify: `tests/test_substrate_signal_bridge.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_substrate_signal_bridge.py`:

```python
def test_cognition_run_failure_maps_to_contradiction_gradient():
    from orion.substrate.signal_bridge import signal_to_molecule

    signal = make_cognition_run_signal(
        dimensions={
            "success": 0.0,
            "step_count": 0.30,
            "latency_level": 0.70,
        },
    )

    molecule = signal_to_molecule(signal)

    assert molecule.gradients["contradiction"] == pytest.approx(1.0)
    assert molecule.gradients["salience"] == pytest.approx(0.70)
    assert molecule.gradients["coherence"] == pytest.approx(0.0)
    assert molecule.gradients["novelty"] == pytest.approx(0.0)
```

- [ ] **Step 2: Run the test to verify it passes**

Run: `pytest tests/test_substrate_signal_bridge.py::test_cognition_run_failure_maps_to_contradiction_gradient -v`
Expected: `PASSED` (the implementation already handles this; if it fails, the math above is wrong — fix `dimensions_to_gradients`).

Cross-check on math:
- `success_present=True, success=0.0` → `contradiction = max(0, 0, 1.0 - 0.0) = 1.0` ✓
- `salience = max(0, 0, 0.70, 0.30, 0) = 0.70` ✓
- `coherence = max(0, 0.0) * 1.0 = 0.0` ✓

- [ ] **Step 3: Commit**

```bash
git add tests/test_substrate_signal_bridge.py
git commit -m "test(substrate): cover cognition_run failure → contradiction in signal bridge"
```

---

## Task 4: `cognition_step` with `error_present` maps to contradiction

**Files:**
- Modify: `tests/test_substrate_signal_bridge.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_substrate_signal_bridge.py`:

```python
def test_cognition_step_error_present_maps_to_contradiction():
    from orion.substrate.signal_bridge import signal_to_molecule

    signal = make_cognition_run_signal(
        signal_kind="cognition_step",
        dimensions={
            "success": 0.0,
            "latency_level": 0.20,
            "error_present": 1.0,
            "service_count": 0.20,
        },
    )

    molecule = signal_to_molecule(signal)

    assert molecule.provenance["signal_kind"] == "cognition_step"
    assert molecule.gradients["contradiction"] == pytest.approx(1.0)
    assert molecule.gradients["salience"] == pytest.approx(0.20)
    assert molecule.gradients["coherence"] == pytest.approx(0.0)
```

- [ ] **Step 2: Run the test to verify it passes**

Run: `pytest tests/test_substrate_signal_bridge.py::test_cognition_step_error_present_maps_to_contradiction -v`
Expected: `PASSED`.

Cross-check:
- `contradiction = max(0, 1.0 (error_present), 1.0 - 0.0) = 1.0` ✓
- `salience = max(0, 0, 0.20, 0, 0.20) = 0.20` ✓

- [ ] **Step 3: Commit**

```bash
git add tests/test_substrate_signal_bridge.py
git commit -m "test(substrate): cover cognition_step error_present → contradiction"
```

---

## Task 5: Unsupported signal is skipped by batch helper, rejected by single

**Files:**
- Modify: `tests/test_substrate_signal_bridge.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_substrate_signal_bridge.py`:

```python
def test_signals_to_molecules_skips_unsupported_signals():
    from orion.substrate.signal_bridge import signals_to_molecules

    unsupported = make_cognition_run_signal(
        signal_kind="cognition_run",
    )
    # Override organ_id to something the bridge does not handle.
    unsupported_other_organ = unsupported.model_copy(
        update={"organ_id": "biometrics", "signal_kind": "gpu_load"}
    )
    supported = make_cognition_run_signal(
        dimensions={"success": 1.0, "step_count": 0.10, "latency_level": 0.10},
    )

    result = signals_to_molecules([unsupported_other_organ, supported])

    assert len(result) == 1
    assert result[0].provenance["organ"] == "cortex_exec"
    assert result[0].provenance["signal_kind"] == "cognition_run"


def test_signal_to_molecule_raises_on_unsupported_signal():
    from orion.substrate.signal_bridge import signal_to_molecule

    rogue = make_cognition_run_signal().model_copy(
        update={"organ_id": "biometrics", "signal_kind": "gpu_load"}
    )

    with pytest.raises(ValueError, match="unsupported substrate signal bridge"):
        signal_to_molecule(rogue)
```

- [ ] **Step 2: Run the tests to verify they pass**

Run: `pytest tests/test_substrate_signal_bridge.py -k "unsupported" -v`
Expected: both `PASSED`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_substrate_signal_bridge.py
git commit -m "test(substrate): batch helper skips unsupported signals; single raises"
```

---

## Task 6: Lock that every bridged molecule validates against `default_registry()`

The bridge already calls `validate_molecule()` internally, but this test pins it as a contract so any future refactor that drops the call gets caught.

**Files:**
- Modify: `tests/test_substrate_signal_bridge.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_substrate_signal_bridge.py`:

```python
def test_bridged_molecule_validates_against_default_registry_for_both_kinds():
    from orion.substrate.signal_bridge import signal_to_molecule

    run = make_cognition_run_signal(
        dimensions={"success": 1.0, "step_count": 0.10, "latency_level": 0.10},
    )
    step = make_cognition_run_signal(
        signal_kind="cognition_step",
        dimensions={
            "success": 1.0,
            "latency_level": 0.10,
            "error_present": 0.0,
            "service_count": 0.20,
        },
    )

    for molecule in (signal_to_molecule(run), signal_to_molecule(step)):
        # Re-validation outside the bridge proves the grammar didn't drift:
        # all atom keys, all predicates, all gradient keys, and the molecule
        # kind are part of the canonical kernel registry.
        validate_molecule(molecule, default_registry())

        # Spot-check that every atom role maps to a real registered atom key
        # — guards against a silent regression to a missing atom like `process`.
        registry = default_registry()
        for role, atom_key in molecule.atoms.items():
            assert registry.has_atom(atom_key), (
                f"role {role!r} maps to unregistered atom {atom_key!r}"
            )
        for relation in molecule.relations:
            assert registry.has_predicate(relation.predicate)
```

- [ ] **Step 2: Run the test to verify it passes**

Run: `pytest tests/test_substrate_signal_bridge.py::test_bridged_molecule_validates_against_default_registry_for_both_kinds -v`
Expected: `PASSED`.

- [ ] **Step 3: Run the full bridge test file**

Run: `pytest tests/test_substrate_signal_bridge.py -v`
Expected: 6 tests, all `PASSED`.

- [ ] **Step 4: Commit**

```bash
git add tests/test_substrate_signal_bridge.py
git commit -m "test(substrate): pin signal bridge molecules to default_registry grammar"
```

---

## Task 7 (OPTIONAL — only if wiring live bus now): SubstrateSignalBusWorker

Skip Tasks 7 and 8 if the slice is "bridge only." The end-to-end acceptance in Task 9 does not depend on this worker — it constructs `OrionSignalV1`, calls the bridge, and writes to the store directly. The worker only adds the live-bus seam.

**Files:**
- Create: `orion/substrate/signal_bus_worker.py`

- [ ] **Step 1: Write the failing worker test**

Create `tests/test_substrate_signal_bus_worker.py`:

```python
"""SubstrateSignalBusWorker — bridges live OrionSignalV1 envelopes to the store."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.signals.models import OrganClass, OrionSignalV1
from orion.substrate.experiment.harness import SubstrateExperimentHarness
from orion.substrate.molecule_store import MoleculeJsonlStore


def _signal_envelope(
    *, signal_kind: str = "cognition_run", dimensions: dict[str, float] | None = None
) -> BaseEnvelope:
    now = datetime.now(timezone.utc)
    signal = OrionSignalV1(
        signal_id="sig-bus-1",
        organ_id="cortex_exec",
        organ_class=OrganClass.endogenous,
        signal_kind=signal_kind,
        dimensions=dimensions or {"success": 1.0, "step_count": 0.10, "latency_level": 0.10},
        causal_parents=[],
        source_event_id="corr-bus-1",
        observed_at=now,
        emitted_at=now,
        summary="bus test",
        notes=[],
    )
    return BaseEnvelope(
        kind=f"signal.{signal.organ_id}.{signal.signal_kind}",
        source=ServiceRef(name="orion-signal-gateway"),
        payload=signal.model_dump(mode="json"),
    )


@pytest.mark.asyncio
async def test_worker_bridges_supported_signal_into_store(tmp_path):
    from orion.substrate.signal_bus_worker import SubstrateSignalBusWorker

    store = MoleculeJsonlStore(tmp_path / "bridged.jsonl")
    harness = SubstrateExperimentHarness()
    worker = SubstrateSignalBusWorker(store=store, harness=harness)

    await worker.handle_envelope(_signal_envelope())

    bridged = store.filter(organ="cortex_exec")
    assert len(bridged) == 1
    assert bridged[0].molecule_kind == "organ_signal"
    emit_records = harness.all_emit_records()
    assert len(emit_records) == 1
    assert emit_records[0].organ == "cortex_exec"


@pytest.mark.asyncio
async def test_worker_skips_unsupported_organ(tmp_path):
    from orion.substrate.signal_bus_worker import SubstrateSignalBusWorker

    store = MoleculeJsonlStore(tmp_path / "bridged.jsonl")
    worker = SubstrateSignalBusWorker(store=store, harness=None)

    env = _signal_envelope()
    # Mutate the payload to a non-supported (organ, kind) pair.
    bad_payload = {**env.payload, "organ_id": "biometrics", "signal_kind": "gpu_load"}
    bad_env = BaseEnvelope(kind=env.kind, source=env.source, payload=bad_payload)

    await worker.handle_envelope(bad_env)

    assert store.filter(organ="biometrics") == []
    assert store.filter(organ="cortex_exec") == []


@pytest.mark.asyncio
async def test_worker_tolerates_non_signal_payload(tmp_path):
    from orion.substrate.signal_bus_worker import SubstrateSignalBusWorker

    store = MoleculeJsonlStore(tmp_path / "bridged.jsonl")
    worker = SubstrateSignalBusWorker(store=store, harness=None)

    junk = BaseEnvelope(
        kind="signal.cortex_exec.cognition_run",
        source=ServiceRef(name="orion-signal-gateway"),
        payload={"not": "a signal"},
    )

    # Must not raise; malformed payloads are silently dropped.
    await worker.handle_envelope(junk)
    assert len(store) == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_substrate_signal_bus_worker.py -v`
Expected: `FAILED` — `ImportError: cannot import name 'SubstrateSignalBusWorker' from 'orion.substrate.signal_bus_worker'`.

- [ ] **Step 3: Implement the worker**

Create `orion/substrate/signal_bus_worker.py`:

```python
"""Substrate signal bus worker.

Subscribes (via an external bus driver) to signal envelopes emitted by the
orion-signal-gateway and bridges the supported subset into the substrate
molecule store. The worker does not mutate gateway behavior; it runs alongside.

The handle_envelope() seam is bus-driver agnostic so tests can drive it without
spinning up a real OrionBusAsync instance.
"""

from __future__ import annotations

import logging
from typing import Optional

from orion.core.bus.bus_schemas import BaseEnvelope
from orion.signals.models import OrionSignalV1
from orion.substrate.experiment.harness import SubstrateExperimentHarness
from orion.substrate.molecule_store import MoleculeJsonlStore
from orion.substrate.signal_bridge import signal_to_molecule, supports_signal


logger = logging.getLogger(__name__)


class SubstrateSignalBusWorker:
    """Bridges OrionSignalV1 envelopes into substrate molecules."""

    def __init__(
        self,
        *,
        store: MoleculeJsonlStore,
        harness: Optional[SubstrateExperimentHarness] = None,
    ) -> None:
        self._store = store
        self._harness = harness

    async def handle_envelope(self, env: BaseEnvelope) -> None:
        payload = env.payload
        if not isinstance(payload, dict):
            return

        try:
            signal = OrionSignalV1.model_validate(payload)
        except Exception as exc:  # noqa: BLE001 — defensive against gateway drift
            logger.debug(
                "substrate signal bus worker: payload not OrionSignalV1: %s", exc
            )
            return

        if not supports_signal(signal):
            return

        molecule = signal_to_molecule(signal)
        self._store.add(molecule)
        if self._harness is not None:
            self._harness.record_emit(molecule, organ=signal.organ_id)
```

- [ ] **Step 4: Confirm `pytest-asyncio` is wired**

Run: `pytest tests/test_substrate_signal_bus_worker.py -v`
Expected: `PASSED`. If the asyncio marker is unrecognized, check the project's `pytest.ini` for `asyncio_mode = auto`. If not present, change the test decorators to use `@pytest.mark.asyncio` together with whichever plugin the rest of the suite uses (run `grep -rn "pytest.mark.asyncio" tests/ | head -3` to confirm convention; almost all existing tests use this same decorator).

- [ ] **Step 5: Commit**

```bash
git add orion/substrate/signal_bus_worker.py tests/test_substrate_signal_bus_worker.py
git commit -m "feat(substrate): add optional bus worker bridging organ signals to store"
```

---

## Task 8 (OPTIONAL): Worker doc string + log-only on bridge failure

This is folded into Task 7; skip if Task 7 was skipped.

---

## Task 9: End-to-end acceptance — bridged molecule shows up in daily rollup

This is the proof the spec asks for. It does **not** require the bus worker; it drives the bridge directly so the test passes whether Tasks 7–8 ran or not.

**Files:**
- Create: `tests/test_substrate_signal_bridge_e2e.py`

- [ ] **Step 1: Write the failing end-to-end test**

Create `tests/test_substrate_signal_bridge_e2e.py`:

```python
"""End-to-end: bridge → MoleculeJsonlStore → harness → daily rollup."""

from __future__ import annotations

from datetime import datetime, timezone

from orion.signals.models import OrganClass, OrionSignalV1
from orion.substrate.experiment.daily_rollup import compute_daily_rollup
from orion.substrate.experiment.harness import SubstrateExperimentHarness
from orion.substrate.molecule_store import MoleculeJsonlStore
from orion.substrate.signal_bridge import signal_to_molecule


def _signal(
    *, signal_id: str, signal_kind: str, dimensions: dict[str, float]
) -> OrionSignalV1:
    now = datetime.now(timezone.utc)
    return OrionSignalV1(
        signal_id=signal_id,
        organ_id="cortex_exec",
        organ_class=OrganClass.endogenous,
        signal_kind=signal_kind,
        dimensions=dimensions,
        causal_parents=[],
        source_event_id=f"corr-{signal_id}",
        observed_at=now,
        emitted_at=now,
        summary=f"{signal_kind} {signal_id}",
        notes=[],
    )


def test_bridged_signals_land_in_daily_rollup_under_cortex_exec(tmp_path):
    store = MoleculeJsonlStore(tmp_path / "molecules.jsonl")
    harness = SubstrateExperimentHarness()

    raw_signals = [
        _signal(
            signal_id="s-ok",
            signal_kind="cognition_run",
            dimensions={
                "success": 1.0,
                "step_count": 0.10,
                "latency_level": 0.20,
            },
        ),
        _signal(
            signal_id="s-fail",
            signal_kind="cognition_run",
            dimensions={
                "success": 0.0,
                "step_count": 0.40,
                "latency_level": 0.80,
            },
        ),
        _signal(
            signal_id="s-step-err",
            signal_kind="cognition_step",
            dimensions={
                "success": 0.0,
                "latency_level": 0.30,
                "error_present": 1.0,
                "service_count": 0.40,
            },
        ),
    ]

    for raw in raw_signals:
        molecule = signal_to_molecule(raw)
        store.add(molecule)
        harness.record_emit(molecule, organ=raw.organ_id)

    today = datetime.now(timezone.utc).date()
    metrics = compute_daily_rollup(day=today, harness=harness, store=store)

    # organ_coverage proves the substrate now "knows about" cortex_exec.
    assert metrics.organ_coverage.by_organ.get("cortex_exec") == 3

    # gradient_distribution should reflect real signal dimensions — at least
    # one molecule has contradiction == 1.0, so the max is 1.0.
    contradiction_stat = next(
        g for g in metrics.gradient_distribution if g.key == "contradiction"
    )
    assert contradiction_stat.max == 1.0
    assert contradiction_stat.mean > 0.0

    salience_stat = next(
        g for g in metrics.gradient_distribution if g.key == "salience"
    )
    assert salience_stat.max == 0.80  # latency_level on s-fail dominates

    # contradiction_clusters should have at least one entry, all members
    # share the organ_signal atom signature.
    assert len(metrics.contradiction_clusters) >= 1
    cluster = metrics.contradiction_clusters[0]
    assert "signal" in cluster.shared_atoms
    assert "gradient" in cluster.shared_atoms
    assert cluster.contradiction_sum >= 1.0
```

- [ ] **Step 2: Run the test to verify it passes**

Run: `pytest tests/test_substrate_signal_bridge_e2e.py -v`
Expected: `PASSED`.

If the salience assertion fails because of intensity-clamping or because of an unexpected dimension fallback, recompute by hand from `dimensions_to_gradients` rather than tweaking the assertion. The expected values from the test data:
- `s-ok`: salience max(0, 0, 0.20, 0.10, 0) = 0.20; contradiction 0; coherence 1.0
- `s-fail`: salience max(0, 0, 0.80, 0.40, 0) = 0.80; contradiction 1.0; coherence 0.0
- `s-step-err`: salience max(0, 0, 0.30, 0, 0.40) = 0.40; contradiction 1.0; coherence 0.0

So `salience.max == 0.80`, `contradiction.max == 1.0`, `contradiction.mean = 2/3 ≈ 0.667`, `coherence.max == 1.0`. The test only asserts on bounds and presence, so day-of-week / time-zone drift can't break it.

- [ ] **Step 3: Run the full substrate test suite to confirm no regression**

Run: `pytest tests/ -k substrate -v`
Expected: all substrate tests `PASSED` (pre-existing + new).

- [ ] **Step 4: Commit**

```bash
git add tests/test_substrate_signal_bridge_e2e.py
git commit -m "test(substrate): end-to-end bridge → store → harness → daily rollup"
```

---

## Acceptance criteria checklist

Run through this list after Task 9 completes. Every box must be checked.

- [ ] A real-shaped `OrionSignalV1` (cortex_exec.cognition_run) converts into one valid `SubstrateMoleculeV1` of kind `organ_signal` (Task 2).
- [ ] Bridge uses only existing atom keys — `signal`, `agency`, `context`, `gradient`, `evidence` — verified by `validate_molecule(..., default_registry())` (Task 6).
- [ ] Bridge uses only existing predicates: `references`, `elicits`, `supports` (Task 6).
- [ ] Bridge uses only canonical gradient keys: `salience`, `contradiction`, `novelty`, `coherence` (Task 6).
- [ ] Molecule validates with `default_registry()` (Task 6).
- [ ] Molecule persists through `MoleculeJsonlStore.add()` and round-trips (Task 9 — the rollup reads from the store).
- [ ] Harness records the molecule emit; emit shows in `harness.all_emit_records()` (Task 7 if worker, Task 9 unconditionally).
- [ ] A daily rollup shows the signal-derived molecule under `organ_coverage.by_organ["cortex_exec"]` (Task 9).
- [ ] No new `OrionSignalV1` subclass introduced; no edits to `orion/signals/**` (verify: `git diff main -- orion/signals/` returns empty for this branch's bridge work).
- [ ] Existing signal gateway behavior unchanged unless the optional Task 7 worker is wired into a real bus driver (the worker module itself does not subscribe to anything — that wiring is out of scope for this plan).
- [ ] Only one schema-kernel expansion: `organ_signal` added to `default_registry()` (Task 1). No new atoms, predicates, gradient keys, or molecule shape.

---

## Non-goals (do not expand scope)

- Do not add `CognitiveLoadSignalV1`, `RouteInstabilitySignalV1`, or any new signal model.
- Do not add new atom kinds (even though `process` would be a tempting fit — use `agency`).
- Do not add new predicates or gradient keys.
- Do not wire the bus worker into a live `OrionBusAsync` subscriber loop in this plan. That belongs in a follow-up once the substrate side of the round-trip is observed in the daily rollup.
- Do not create `orion/analytics/`.
- Do not refactor the signal gateway processor (`services/orion-signal-gateway/app/processor.py`).
- Do not modify `orion/signals/adapters/cognition_trace.py`.
