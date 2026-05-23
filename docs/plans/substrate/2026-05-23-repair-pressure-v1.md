# repair_pressure_v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one vertical proving loop where a raw chat turn becomes a substrate `observation` molecule, a deterministic detector extracts `RepairEvidenceV1` items, an appraiser reduces them to `RepairPressureAppraisalV1`, the appraisal is bridged to an `OrionSignalV1(organ_id="graph_cognition", signal_kind="repair_pressure")`, and a small contract consumer flips the next chat response into `repair_concrete` mode.

**Architecture:** New library-only package `orion/substrate/appraisal/` containing: `models.py` (two pydantic models), `evidence.py` (deterministic phrase-match detectors), `repair_pressure.py` (explicit-formula reducer), `signal_bridge.py` (appraisal → `OrionSignalV1`), `windowing.py` (recent-molecule selector), `contract.py` (behavior consumer). One additive edit to `orion/signals/registry.py` registers `repair_pressure` as a canonical signal kind on the existing `graph_cognition` organ entry and extends its `canonical_dimensions`. No new substrate gradients, no new schema-kernel atoms, no new molecule kinds, no chat pipeline rewrite.

**Tech Stack:** Python 3.12, Pydantic v2, pytest. Reuses `orion.schema_kernel`, `orion.substrate.molecules`, `orion.mind.substrate_emit`, `orion.signals.models`, `orion.signals.signal_ids`, `orion.signals.registry`. Does not modify any of those except for the additive registry edit in Task 1.

---

## Arsonist constraints respected

These are non-negotiable. Every task below preserves them.

1. **No new canonical substrate gradients.** `salience, contradiction, novelty, coherence` only (`orion/schema_kernel/gradient.py:9-13`). Repair pressure dimensions live on `OrionSignalV1.dimensions` and `RepairPressureAppraisalV1.dimensions`, **never** on `SubstrateMoleculeV1.gradients`.
2. **No new schema-kernel atoms.** `ATOM_KINDS` at `orion/schema_kernel/atom.py:14-27` is closed at 12. Evidence kinds are not atoms.
3. **No new substrate molecule kinds.** `default_registry()` already registers `observation, claim, pressure, contradiction` (`orion/schema_kernel/registry.py:97-104`). The chat turn becomes an `observation` molecule via the existing `orion.mind.substrate_emit.emit_observation` helper. `RepairEvidenceV1` is a separate pydantic model, **not** a `SubstrateMoleculeV1`.
4. **Causal layering is strict.** Raw chat → observation molecule → evidence → appraisal → signal → contract. No layer silently does another's job.
5. **Payload is not the machine contract.** `RepairEvidenceV1.features` and `.span` are audit-only. The machine contract is `dimensions[level]`, `dimensions[confidence]`, and the typed evidence-kind enum.
6. **Fail closed.** Empty / weak evidence forces `level=0.0` and `confidence ≤ 0.25` with `"no_repair_evidence"` in notes.

---

## File structure

| File | Status | Responsibility |
| --- | --- | --- |
| `orion/signals/registry.py` | Modify (lines 189-203) | Append `repair_pressure` to `graph_cognition.signal_kinds`; extend `canonical_dimensions` with the seven repair dimensions + `level`. |
| `orion/substrate/appraisal/__init__.py` | Create | Re-export the public API (`RepairEvidenceV1`, `RepairPressureAppraisalV1`, `extract_repair_evidence`, `appraise_repair_pressure`, `repair_appraisal_to_signal`, `select_recent_chat_molecules`, `apply_repair_pressure_contract`). |
| `orion/substrate/appraisal/models.py` | Create | `RepairEvidenceV1`, `RepairPressureAppraisalV1`. |
| `orion/substrate/appraisal/evidence.py` | Create | Deterministic phrase-match detector returning `list[RepairEvidenceV1]` per molecule. |
| `orion/substrate/appraisal/repair_pressure.py` | Create | `appraise_repair_pressure(molecules, *, window_id) -> RepairPressureAppraisalV1`. Explicit formula v1; no LLM. |
| `orion/substrate/appraisal/signal_bridge.py` | Create | `repair_appraisal_to_signal(appraisal, *, observed_at=None) -> OrionSignalV1`. |
| `orion/substrate/appraisal/windowing.py` | Create | `select_recent_chat_molecules(molecules, *, source_id=None, max_age_seconds=300, max_count=20)`. |
| `orion/substrate/appraisal/contract.py` | Create | `apply_repair_pressure_contract(base_contract, signal) -> dict[str, Any]`. |
| `tests/test_repair_pressure_models.py` | Create | Pydantic model validation. |
| `tests/test_repair_pressure_evidence.py` | Create | Detector phrase-match thresholds (spec §14.1). |
| `tests/test_repair_pressure_appraisal.py` | Create | High / low / fail-closed reducer cases (spec §14.2). |
| `tests/test_repair_pressure_signal_bridge.py` | Create | Bridge organ_id, signal_kind, dimensions, causal_parents (spec §14.3). |
| `tests/test_repair_pressure_behavior_contract.py` | Create | Contract mode switching (spec §14.4). |
| `tests/test_repair_pressure_windowing.py` | Create | Window selection rules. |
| `tests/test_repair_pressure_e2e.py` | Create | Definition-of-Done causal chain assertion (spec §17). |

**Do NOT modify** (architecture-locked):
- `orion/schema_kernel/atom.py` — atom set is closed.
- `orion/schema_kernel/gradient.py` — gradient keys are closed.
- `orion/schema_kernel/registry.py` — molecule kinds stay at four.
- `orion/substrate/molecules.py` — molecule shape is shared.
- `orion/signals/models.py` — signal envelope is shared.
- `orion/mind/substrate_emit.py` — emit helper already correct.

---

## Shared test helpers

Several test files build a chat observation molecule. Put this helper at the top of each test file that needs it (or in a shared `tests/_repair_helpers.py` module imported by them). It mirrors the real shape `orion.mind.substrate_emit.emit_observation` produces.

```python
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.mind.substrate_emit import emit_observation
from orion.substrate.molecules import SubstrateMoleculeV1


def make_chat_observation(
    text: str,
    *,
    source_id: str = "msg-test",
    created_at: datetime | None = None,
) -> SubstrateMoleculeV1:
    mol = emit_observation(surface_text=text, source_id=source_id)
    if created_at is not None:
        mol.created_at = created_at
        mol.last_touched_at = created_at
    return mol
```

---

## Task 1: Register `repair_pressure` as canonical signal kind on `graph_cognition`

**Files:**
- Modify: `orion/signals/registry.py:189-203`
- Test: `tests/test_repair_pressure_signal_bridge.py` (new file — add the targeted registry test here first)

- [ ] **Step 1: Write the failing registry test**

Create `tests/test_repair_pressure_signal_bridge.py` with just this test for now:

```python
"""Repair pressure signal bridge tests."""

from __future__ import annotations

from orion.signals.registry import ORGAN_REGISTRY


def test_graph_cognition_registers_repair_pressure_signal_kind():
    entry = ORGAN_REGISTRY["graph_cognition"]
    assert "repair_pressure" in entry.signal_kinds, (
        "graph_cognition organ must register 'repair_pressure' so the appraisal "
        "signal bridge emits a canonical signal kind, not an ad-hoc one."
    )


def test_graph_cognition_canonical_dimensions_cover_repair_pressure():
    entry = ORGAN_REGISTRY["graph_cognition"]
    required = {
        "level",
        "specificity_demand",
        "trust_rupture",
        "coherence_gap",
        "repetition_failure",
        "operational_block",
        "explicit_repair_command",
        "confidence",
    }
    missing = required - set(entry.canonical_dimensions)
    assert not missing, f"graph_cognition canonical_dimensions missing: {missing}"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_repair_pressure_signal_bridge.py -v`
Expected: both FAIL — `repair_pressure` not in signal_kinds, dimensions missing.

- [ ] **Step 3: Modify the `graph_cognition` registry entry**

Edit `orion/signals/registry.py`, replace the existing `graph_cognition` block (lines 189-203) with:

```python
    "graph_cognition": OrionOrganRegistryEntry(
        organ_id="graph_cognition",
        organ_class=OrganClass.endogenous,
        service="orion-cortex-exec",
        signal_kinds=[
            "metacog_perception",
            "coherence_state",
            "goal_pressure",
            "repair_pressure",
        ],
        canonical_dimensions=[
            "coherence",
            "tension",
            "goal_pressure",
            "confidence",
            "level",
            "specificity_demand",
            "trust_rupture",
            "coherence_gap",
            "repetition_failure",
            "operational_block",
            "explicit_repair_command",
            "assistant_accountability_demand",
        ],
        # recall omitted here to keep static causal_parent_organs acyclic (recall→autonomy→graph_cognition).
        causal_parent_organs=["social_memory"],
        bus_channels=["orion:cognition:trace", "orion:metacog:trace"],
        notes=[
            "Recall-linked metacog is carried via autonomy/recall in the mesh; not duplicated as a "
            "registry parent edge to avoid a recall↔graph_cognition cycle in static DAG validation.",
            "repair_pressure is a substrate-derived appraisal emitted by orion.substrate.appraisal; "
            "see docs/plans/substrate/2026-05-23-repair-pressure-v1.md.",
        ],
    ),
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_repair_pressure_signal_bridge.py -v`
Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add orion/signals/registry.py tests/test_repair_pressure_signal_bridge.py
git commit -m "feat(signals): register repair_pressure on graph_cognition organ"
```

---

## Task 2: Create `RepairEvidenceV1` and `RepairPressureAppraisalV1` models

**Files:**
- Create: `orion/substrate/appraisal/__init__.py`
- Create: `orion/substrate/appraisal/models.py`
- Test: `tests/test_repair_pressure_models.py`

- [ ] **Step 1: Write the failing model tests**

Create `tests/test_repair_pressure_models.py`:

```python
"""RepairEvidenceV1 / RepairPressureAppraisalV1 model tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from orion.substrate.appraisal.models import (
    RepairEvidenceV1,
    RepairPressureAppraisalV1,
)


def _evidence(**overrides) -> RepairEvidenceV1:
    base = dict(
        evidence_id="ev_1",
        source_molecule_id="mol_abc",
        evidence_kind="specificity_demand",
        detector="phrase_match_v1",
        score=0.8,
        confidence=0.7,
        span="give me nuts and bolts",
        features={"phrase_strength": 0.9},
    )
    base.update(overrides)
    return RepairEvidenceV1(**base)


def test_evidence_accepts_known_kind():
    ev = _evidence()
    assert ev.evidence_kind == "specificity_demand"
    assert 0.0 <= ev.score <= 1.0
    assert 0.0 <= ev.confidence <= 1.0


def test_evidence_rejects_unknown_kind():
    with pytest.raises(ValidationError):
        _evidence(evidence_kind="anger")


def test_evidence_rejects_extra_fields():
    with pytest.raises(ValidationError):
        RepairEvidenceV1(
            evidence_id="ev_2",
            source_molecule_id="mol_a",
            evidence_kind="trust_rupture",
            detector="phrase_match_v1",
            score=0.5,
            confidence=0.5,
            mystery_field="nope",
        )


def test_evidence_clamps_via_validation():
    with pytest.raises(ValidationError):
        _evidence(score=1.5)
    with pytest.raises(ValidationError):
        _evidence(confidence=-0.1)


def test_appraisal_roundtrip():
    appraisal = RepairPressureAppraisalV1(
        appraisal_id="app_1",
        window_id="win_1",
        dimensions={
            "level": 0.6,
            "specificity_demand": 0.8,
            "trust_rupture": 0.0,
            "coherence_gap": 0.0,
            "repetition_failure": 0.0,
            "operational_block": 0.0,
            "explicit_repair_command": 0.0,
            "confidence": 0.5,
        },
        evidence=[_evidence()],
        causal_molecule_ids=["mol_abc"],
        confidence=0.5,
        summary="test",
    )
    assert appraisal.appraisal_kind == "repair_pressure"
    assert appraisal.dimensions["level"] == 0.6


def test_appraisal_rejects_extra_fields():
    with pytest.raises(ValidationError):
        RepairPressureAppraisalV1(
            appraisal_id="app_2",
            window_id="win_2",
            dimensions={"level": 0.1, "confidence": 0.1},
            evidence=[],
            causal_molecule_ids=[],
            confidence=0.1,
            mystery="x",
        )
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_repair_pressure_models.py -v`
Expected: `ModuleNotFoundError: No module named 'orion.substrate.appraisal'`.

- [ ] **Step 3: Create the package**

Create `orion/substrate/appraisal/__init__.py`:

```python
"""Substrate-derived appraisers. See docs/plans/substrate/2026-05-23-repair-pressure-v1.md."""

from .models import RepairEvidenceV1, RepairPressureAppraisalV1

__all__ = [
    "RepairEvidenceV1",
    "RepairPressureAppraisalV1",
]
```

- [ ] **Step 4: Create the models**

Create `orion/substrate/appraisal/models.py`:

```python
"""Pydantic models for the repair_pressure appraisal pipeline.

These models are NOT SubstrateMoleculeV1 — they live outside the substrate
grammar by design. Repair evidence and appraisal dimensions are domain
concepts that must not pollute the canonical substrate atoms/gradients.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


EvidenceKind = Literal[
    "specificity_demand",
    "trust_rupture",
    "coherence_gap",
    "repetition_failure",
    "operational_block",
    "explicit_repair_command",
    "assistant_accountability_demand",
]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class RepairEvidenceV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    source_molecule_id: str

    evidence_kind: EvidenceKind

    detector: str
    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)

    # Audit only. Do not treat as machine contract.
    span: str | None = None
    features: dict[str, float] = Field(default_factory=dict)

    created_at: datetime = Field(default_factory=_utcnow)


class RepairPressureAppraisalV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    appraisal_id: str
    window_id: str
    appraisal_kind: Literal["repair_pressure"] = "repair_pressure"

    dimensions: dict[str, float]
    evidence: list[RepairEvidenceV1]
    causal_molecule_ids: list[str]

    confidence: float = Field(ge=0.0, le=1.0)
    summary: str | None = None
    notes: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_repair_pressure_models.py -v`
Expected: all 6 PASS.

- [ ] **Step 6: Commit**

```bash
git add orion/substrate/appraisal/__init__.py orion/substrate/appraisal/models.py tests/test_repair_pressure_models.py
git commit -m "feat(substrate): add RepairEvidenceV1 + RepairPressureAppraisalV1 models"
```

---

## Task 3: Evidence detector — deterministic phrase matching

**Files:**
- Create: `orion/substrate/appraisal/evidence.py`
- Modify: `orion/substrate/appraisal/__init__.py` (export `extract_repair_evidence`)
- Test: `tests/test_repair_pressure_evidence.py`

- [ ] **Step 1: Write the failing evidence tests**

Create `tests/test_repair_pressure_evidence.py`:

```python
"""Repair evidence detector tests — spec §14.1."""

from __future__ import annotations

from orion.mind.substrate_emit import emit_observation
from orion.substrate.appraisal.evidence import extract_repair_evidence
from orion.substrate.molecules import SubstrateMoleculeV1


def _obs(text: str) -> SubstrateMoleculeV1:
    return emit_observation(surface_text=text, source_id="msg-test")


def _max_score(evidence, kind):
    items = [e for e in evidence if e.evidence_kind == kind]
    return max((e.score for e in items), default=0.0)


def test_specificity_demand_nuts_and_bolts():
    ev = extract_repair_evidence([_obs("not hand wavy, give me nuts and bolts")])
    assert _max_score(ev, "specificity_demand") >= 0.75


def test_trust_rupture_garbage_directions():
    ev = extract_repair_evidence([_obs("you gave me garbage directions")])
    assert _max_score(ev, "trust_rupture") >= 0.75


def test_coherence_gap_swamp():
    ev = extract_repair_evidence([_obs("this is becoming a swamp")])
    assert _max_score(ev, "coherence_gap") >= 0.70


def test_repetition_failure_again_you_keep():
    ev = extract_repair_evidence([_obs("again, you keep doing this")])
    assert _max_score(ev, "repetition_failure") >= 0.65


def test_operational_block_design_spec():
    ev = extract_repair_evidence([_obs("build me a design spec for Claude")])
    assert _max_score(ev, "operational_block") >= 0.70


def test_explicit_repair_command_arsonist_pov():
    ev = extract_repair_evidence([_obs("arsonist POV only here")])
    assert _max_score(ev, "explicit_repair_command") >= 0.65


def test_evidence_emits_span_and_source_molecule_id():
    mol = _obs("you gave me garbage directions")
    ev = extract_repair_evidence([mol])
    matches = [e for e in ev if e.evidence_kind == "trust_rupture"]
    assert matches
    e = matches[0]
    assert e.source_molecule_id == mol.molecule_id
    assert e.span is not None and "garbage" in e.span.lower()
    assert 0.0 <= e.confidence <= 1.0


def test_evidence_returns_empty_for_neutral_text():
    ev = extract_repair_evidence([_obs("what is the weather like?")])
    assert ev == []


def test_evidence_skips_molecules_without_surface_text():
    mol = SubstrateMoleculeV1(molecule_kind="observation")
    assert extract_repair_evidence([mol]) == []
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_repair_pressure_evidence.py -v`
Expected: import error (`extract_repair_evidence` not defined).

- [ ] **Step 3: Implement the detector**

Create `orion/substrate/appraisal/evidence.py`:

```python
"""Deterministic phrase-match detector for repair evidence.

No LLM. No embeddings. Every signal is explainable as a phrase hit with a
clear feature vector.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Iterable

from orion.substrate.molecules import SubstrateMoleculeV1

from .models import EvidenceKind, RepairEvidenceV1


DETECTOR_NAME = "phrase_match_v1"


@dataclass(frozen=True)
class _Phrase:
    pattern: str  # case-insensitive plain substring OR regex if regex=True
    weight: float  # phrase_match_strength contribution
    regex: bool = False


# Phrase tables — one per evidence kind. Source: spec §7.2.
_PHRASES: dict[EvidenceKind, tuple[_Phrase, ...]] = {
    "specificity_demand": (
        _Phrase("nuts and bolts", 0.95),
        _Phrase("operationalize", 0.85),
        _Phrase("concrete", 0.70),
        _Phrase("exactly how", 0.80),
        _Phrase("from a to z", 0.85),
        _Phrase("not hand wavy", 0.90),
        _Phrase("design spec", 0.85),
        _Phrase("implementation", 0.65),
        _Phrase("what files", 0.80),
        _Phrase("how the hell", 0.75),
        _Phrase("stop being vague", 0.90),
    ),
    "trust_rupture": (
        _Phrase("you gave me garbage", 1.0),
        _Phrase("garbage directions", 0.97),
        _Phrase("checked out", 0.85),
        _Phrase("bullshit", 0.85),
        _Phrase("fuckery", 0.85),
        _Phrase("bad directions", 0.85),
        _Phrase("not useful", 0.70),
        _Phrase("stop fucking me around", 0.95),
    ),
    "coherence_gap": (
        _Phrase("swamp", 0.92),
        _Phrase("hand wavy", 0.80),
        _Phrase("doesn't converge", 0.85),
        _Phrase("bolting on", 0.80),
        _Phrase(r"\bugly\b", 0.65, regex=True),
        _Phrase("drift", 0.70),
        _Phrase("making shit up", 0.85),
        _Phrase("not the paradigm", 0.85),
    ),
    "repetition_failure": (
        _Phrase(r"\bagain\b", 0.70, regex=True),
        _Phrase(r"\bstill\b", 0.55, regex=True),
        _Phrase("you keep", 0.85),
        _Phrase("y'all been", 0.75),
        _Phrase("previous", 0.55),
        _Phrase("same thing", 0.75),
        _Phrase("more garbage", 0.85),
    ),
    "operational_block": (
        _Phrase("i need to figure out", 0.80),
        _Phrase("i need this to plug in", 0.85),
        _Phrase("i need a design", 0.85),
        _Phrase("so i can build", 0.80),
        _Phrase("pass off to claude", 0.85),
        _Phrase("what is needed", 0.70),
        _Phrase("design spec for claude", 0.90),
        _Phrase("build me a design spec", 0.90),
    ),
    "explicit_repair_command": (
        _Phrase(r"\bstop\b", 0.75, regex=True),
        _Phrase(r"\bfocus\b", 0.70, regex=True),
        _Phrase("try again", 0.75),
        _Phrase("back up", 0.70),
        _Phrase(r"\bdo not\b", 0.70, regex=True),
        _Phrase("arsonist pov only", 0.95),
        _Phrase("arsonist pov", 0.85),
        _Phrase("no hand waving", 0.85),
    ),
    "assistant_accountability_demand": (
        _Phrase("you are", 0.55),
        _Phrase("you gave", 0.75),
        _Phrase("your work", 0.70),
        _Phrase("your directions", 0.80),
        _Phrase("you keep", 0.80),
        _Phrase("checked out", 0.80),
    ),
}


# Imperative cues raise confidence + score independently of phrase weight.
_IMPERATIVE_PAT = re.compile(
    r"\b(stop|focus|do not|don't|give me|build me|need|must)\b", re.IGNORECASE
)


def _new_evidence_id() -> str:
    return f"ev_{uuid.uuid4().hex[:16]}"


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _phrase_hit(text_lower: str, phrase: _Phrase) -> tuple[bool, str | None]:
    """Return (matched, span_text_or_None)."""
    if phrase.regex:
        m = re.search(phrase.pattern, text_lower, re.IGNORECASE)
        if m:
            return True, m.group(0)
        return False, None
    idx = text_lower.find(phrase.pattern.lower())
    if idx >= 0:
        # Return the matched span as-is.
        return True, text_lower[idx : idx + len(phrase.pattern)]
    return False, None


def _extract_text(molecule: SubstrateMoleculeV1) -> str | None:
    payload = molecule.payload or {}
    for key in ("surface_text", "claim_text"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _score_for_kind(
    text: str,
    kind: EvidenceKind,
    salience: float,
) -> tuple[float, float, str | None, dict[str, float]] | None:
    """Return (score, confidence, span, features) for the strongest hit, or None."""
    text_lower = text.lower()
    best_weight = 0.0
    best_span: str | None = None
    for phrase in _PHRASES[kind]:
        matched, span = _phrase_hit(text_lower, phrase)
        if matched and phrase.weight > best_weight:
            best_weight = phrase.weight
            best_span = span
    if best_weight == 0.0:
        return None

    imperative_strength = 1.0 if _IMPERATIVE_PAT.search(text) else 0.0
    # "Repetition marker" is a coarse signal that the user is re-asserting.
    repetition_marker = 1.0 if re.search(r"\b(again|still|keep)\b", text_lower) else 0.0

    score = _clamp01(
        0.78 * best_weight
        + 0.12 * imperative_strength
        + 0.06 * repetition_marker
        + 0.04 * salience
    )

    # Confidence: high for direct phrase, lower for regex word-boundary cues.
    confidence = _clamp01(0.50 + 0.35 * best_weight + 0.10 * imperative_strength)

    features = {
        "phrase_weight": best_weight,
        "imperative_strength": imperative_strength,
        "repetition_marker": repetition_marker,
        "substrate_salience": salience,
    }
    return score, confidence, best_span, features


def extract_repair_evidence(
    molecules: Iterable[SubstrateMoleculeV1],
) -> list[RepairEvidenceV1]:
    """Return all repair evidence found across a sequence of molecules.

    Each (molecule, evidence_kind) pair yields at most one RepairEvidenceV1 —
    the strongest phrase match for that kind in that molecule.
    """

    out: list[RepairEvidenceV1] = []
    for molecule in molecules:
        text = _extract_text(molecule)
        if not text:
            continue
        salience = float(molecule.gradients.get("salience", 0.0))
        for kind in _PHRASES.keys():
            scored = _score_for_kind(text, kind, salience)
            if scored is None:
                continue
            score, confidence, span, features = scored
            out.append(
                RepairEvidenceV1(
                    evidence_id=_new_evidence_id(),
                    source_molecule_id=molecule.molecule_id,
                    evidence_kind=kind,
                    detector=DETECTOR_NAME,
                    score=score,
                    confidence=confidence,
                    span=span,
                    features=features,
                )
            )
    return out
```

- [ ] **Step 4: Export from package init**

Edit `orion/substrate/appraisal/__init__.py` — replace its contents with:

```python
"""Substrate-derived appraisers. See docs/plans/substrate/2026-05-23-repair-pressure-v1.md."""

from .evidence import DETECTOR_NAME, extract_repair_evidence
from .models import RepairEvidenceV1, RepairPressureAppraisalV1

__all__ = [
    "DETECTOR_NAME",
    "RepairEvidenceV1",
    "RepairPressureAppraisalV1",
    "extract_repair_evidence",
]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_repair_pressure_evidence.py -v`
Expected: all 9 PASS. If a threshold is borderline, adjust phrase weights — not test thresholds (spec §14.1 is the contract).

- [ ] **Step 6: Commit**

```bash
git add orion/substrate/appraisal/evidence.py orion/substrate/appraisal/__init__.py tests/test_repair_pressure_evidence.py
git commit -m "feat(substrate): deterministic phrase-match detector for repair evidence"
```

---

## Task 4: Appraisal reducer — explicit formula v1

**Files:**
- Create: `orion/substrate/appraisal/repair_pressure.py`
- Modify: `orion/substrate/appraisal/__init__.py` (export `appraise_repair_pressure`)
- Test: `tests/test_repair_pressure_appraisal.py`

- [ ] **Step 1: Write the failing appraisal tests**

Create `tests/test_repair_pressure_appraisal.py`:

```python
"""Repair pressure reducer tests — spec §14.2."""

from __future__ import annotations

from orion.mind.substrate_emit import emit_observation
from orion.substrate.appraisal.repair_pressure import appraise_repair_pressure
from orion.substrate.molecules import SubstrateMoleculeV1


def _obs(text: str, source_id: str = "msg-test") -> SubstrateMoleculeV1:
    return emit_observation(surface_text=text, source_id=source_id)


def test_high_repair_pressure_triggers_concrete_mode():
    # Each line below contributes at least one of the seven evidence kinds.
    # Across the window every kind fires at least once; the level formula
    # (spec §9.3) needs broad coverage to clear 0.75.
    molecules = [
        _obs("you gave me garbage directions"),  # trust_rupture + assistant_accountability_demand
        _obs("you keep making shit up — again"),  # repetition_failure + coherence_gap + assistant_accountability_demand
        _obs("this is becoming a swamp, doesn't converge"),  # coherence_gap
        _obs(
            "okay, arsonist POV only here: build me a design spec for Claude, "
            "not hand wavy, give me nuts and bolts"
        ),  # specificity_demand + operational_block + explicit_repair_command
    ]
    appraisal = appraise_repair_pressure(molecules, window_id="win-high")
    assert appraisal.dimensions["level"] >= 0.75, appraisal.dimensions
    assert appraisal.confidence >= 0.60, appraisal.confidence
    assert set(appraisal.causal_molecule_ids) >= {m.molecule_id for m in molecules}


def test_low_repair_pressure_neutral_chat():
    molecules = [_obs("what is the weather like?")]
    appraisal = appraise_repair_pressure(molecules, window_id="win-low")
    assert appraisal.dimensions["level"] <= 0.25, appraisal.dimensions
    assert appraisal.confidence <= 0.45, appraisal.confidence


def test_fail_closed_no_molecules():
    appraisal = appraise_repair_pressure([], window_id="win-empty")
    assert appraisal.dimensions["level"] == 0.0
    assert appraisal.confidence <= 0.25
    assert "no_repair_evidence" in appraisal.notes
    assert appraisal.evidence == []
    assert appraisal.causal_molecule_ids == []


def test_single_weak_evidence_caps_confidence():
    molecules = [_obs("again")]
    appraisal = appraise_repair_pressure(molecules, window_id="win-weak")
    assert appraisal.confidence <= 0.45, appraisal.confidence


def test_appraisal_records_all_required_dimensions():
    molecules = [_obs("not hand wavy, build me a design spec for Claude")]
    appraisal = appraise_repair_pressure(molecules, window_id="win-req")
    required = {
        "level",
        "specificity_demand",
        "trust_rupture",
        "coherence_gap",
        "repetition_failure",
        "operational_block",
        "explicit_repair_command",
        "assistant_accountability_demand",
        "confidence",
    }
    missing = required - appraisal.dimensions.keys()
    assert not missing, f"missing dimensions: {missing}"


def test_appraisal_kind_is_repair_pressure():
    appraisal = appraise_repair_pressure(
        [_obs("you gave me garbage directions")], window_id="win-kind"
    )
    assert appraisal.appraisal_kind == "repair_pressure"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_repair_pressure_appraisal.py -v`
Expected: import error — `appraise_repair_pressure` not defined.

- [ ] **Step 3: Implement the reducer**

Create `orion/substrate/appraisal/repair_pressure.py`:

```python
"""Repair pressure reducer — explicit formula v1.

No mystery models, no LLM. Inputs: a window of SubstrateMoleculeV1.
Output: RepairPressureAppraisalV1 with deterministic dimensions.
"""

from __future__ import annotations

import uuid
from typing import Iterable, get_args

from orion.substrate.molecules import SubstrateMoleculeV1

from .evidence import extract_repair_evidence
from .models import EvidenceKind, RepairEvidenceV1, RepairPressureAppraisalV1


_EVIDENCE_KINDS: tuple[EvidenceKind, ...] = tuple(get_args(EvidenceKind))


def _new_appraisal_id() -> str:
    return f"app_{uuid.uuid4().hex[:16]}"


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _kind_score(evidence: list[RepairEvidenceV1], kind: EvidenceKind) -> float:
    """Max score across evidence items of this kind. Spec §9.2."""
    return max((e.score for e in evidence if e.evidence_kind == kind), default=0.0)


def _gradient_mean(molecules: list[SubstrateMoleculeV1], key: str) -> float:
    if not molecules:
        return 0.0
    values = [float(m.gradients.get(key, 0.0)) for m in molecules]
    return sum(values) / len(values)


def _molecule_quality(molecules: list[SubstrateMoleculeV1]) -> float:
    if not molecules:
        return 0.0
    valid = 0
    for m in molecules:
        if m.provenance and m.gradients:
            valid += 1
    return valid / len(molecules)


def appraise_repair_pressure(
    molecules: Iterable[SubstrateMoleculeV1],
    *,
    window_id: str,
) -> RepairPressureAppraisalV1:
    """Reduce a window of molecules to a RepairPressureAppraisalV1.

    Layering: this function does NOT touch any chat code, does NOT emit a
    signal, does NOT change any contract. It only computes.
    """

    mol_list = list(molecules)
    evidence = extract_repair_evidence(mol_list)

    # Per-kind aggregation (spec §9.2 — max across detector hits).
    kind_scores: dict[EvidenceKind, float] = {
        kind: _kind_score(evidence, kind) for kind in _EVIDENCE_KINDS
    }

    mean_salience = _gradient_mean(mol_list, "salience")
    mean_contradiction = _gradient_mean(mol_list, "contradiction")
    mean_coherence = _gradient_mean(mol_list, "coherence")
    mean_novelty = _gradient_mean(mol_list, "novelty")

    # Level formula (spec §9.3).
    raw_level = (
        0.22 * kind_scores["specificity_demand"]
        + 0.20 * kind_scores["trust_rupture"]
        + 0.16 * kind_scores["coherence_gap"]
        + 0.12 * kind_scores["repetition_failure"]
        + 0.12 * kind_scores["operational_block"]
        + 0.10 * kind_scores["explicit_repair_command"]
        + 0.08 * kind_scores["assistant_accountability_demand"]
        + 0.10 * mean_salience
        + 0.08 * mean_contradiction
        - 0.10 * mean_coherence
    )
    level = _clamp01(raw_level)

    # Confidence formula (spec §9.4).
    notes: list[str] = []
    if not evidence:
        level = 0.0
        confidence = min(0.25, 0.10 * _molecule_quality(mol_list))
        notes.append("no_repair_evidence")
    else:
        mean_evidence_confidence = sum(e.confidence for e in evidence) / len(evidence)
        kinds_present = {e.evidence_kind for e in evidence}
        evidence_coverage = len(kinds_present) / len(_EVIDENCE_KINDS)
        quality = _molecule_quality(mol_list)
        confidence = _clamp01(
            0.50 * mean_evidence_confidence
            + 0.25 * evidence_coverage
            + 0.15 * min(1.0, len(evidence) / 5.0)
            + 0.10 * quality
        )
        # Single-weak-evidence cap (spec §9.4 fail-closed clause).
        if len(evidence) == 1 and evidence[0].score < 0.65:
            confidence = min(confidence, 0.45)
            notes.append("single_weak_evidence")

    dimensions: dict[str, float] = {
        "level": level,
        "specificity_demand": kind_scores["specificity_demand"],
        "trust_rupture": kind_scores["trust_rupture"],
        "coherence_gap": kind_scores["coherence_gap"],
        "repetition_failure": kind_scores["repetition_failure"],
        "operational_block": kind_scores["operational_block"],
        "explicit_repair_command": kind_scores["explicit_repair_command"],
        "assistant_accountability_demand": kind_scores[
            "assistant_accountability_demand"
        ],
        "confidence": confidence,
        # Optional substrate-derived dimensions (spec §6.2 "Optional dimensions").
        "salience": mean_salience,
        "contradiction": mean_contradiction,
        "coherence": mean_coherence,
        "novelty": mean_novelty,
    }

    causal_ids = [m.molecule_id for m in mol_list]

    return RepairPressureAppraisalV1(
        appraisal_id=_new_appraisal_id(),
        window_id=window_id,
        dimensions=dimensions,
        evidence=evidence,
        causal_molecule_ids=causal_ids,
        confidence=confidence,
        summary=None if not evidence else _summarize(level, kind_scores),
        notes=notes,
    )


def _summarize(level: float, kind_scores: dict[EvidenceKind, float]) -> str:
    top_kinds = sorted(
        ((k, v) for k, v in kind_scores.items() if v > 0.0),
        key=lambda kv: kv[1],
        reverse=True,
    )[:3]
    parts = [f"{k}={v:.2f}" for k, v in top_kinds]
    return f"repair_pressure level={level:.2f}; top: " + ", ".join(parts)
```

- [ ] **Step 4: Export from package init**

Edit `orion/substrate/appraisal/__init__.py` — replace its contents with:

```python
"""Substrate-derived appraisers. See docs/plans/substrate/2026-05-23-repair-pressure-v1.md."""

from .evidence import DETECTOR_NAME, extract_repair_evidence
from .models import RepairEvidenceV1, RepairPressureAppraisalV1
from .repair_pressure import appraise_repair_pressure

__all__ = [
    "DETECTOR_NAME",
    "RepairEvidenceV1",
    "RepairPressureAppraisalV1",
    "appraise_repair_pressure",
    "extract_repair_evidence",
]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_repair_pressure_appraisal.py -v`
Expected: all 6 PASS. If `test_high_repair_pressure_triggers_concrete_mode` is borderline-low, the phrase weights in Task 3 are too conservative — raise the strongest hits to 0.95.

- [ ] **Step 6: Commit**

```bash
git add orion/substrate/appraisal/repair_pressure.py orion/substrate/appraisal/__init__.py tests/test_repair_pressure_appraisal.py
git commit -m "feat(substrate): explicit-formula repair pressure reducer"
```

---

## Task 5: Signal bridge — appraisal → `OrionSignalV1`

**Files:**
- Create: `orion/substrate/appraisal/signal_bridge.py`
- Modify: `orion/substrate/appraisal/__init__.py` (export `repair_appraisal_to_signal`)
- Test: extend `tests/test_repair_pressure_signal_bridge.py`

- [ ] **Step 1: Add the failing bridge tests**

Append to `tests/test_repair_pressure_signal_bridge.py`:

```python
from datetime import datetime, timezone

from orion.mind.substrate_emit import emit_observation
from orion.signals.models import OrganClass, OrionSignalV1
from orion.substrate.appraisal.repair_pressure import appraise_repair_pressure
from orion.substrate.appraisal.signal_bridge import repair_appraisal_to_signal


def _high_appraisal():
    molecules = [
        emit_observation(surface_text="you gave me garbage directions", source_id="m1"),
        emit_observation(surface_text="you keep making shit up — again", source_id="m1"),
        emit_observation(surface_text="this is becoming a swamp, doesn't converge", source_id="m1"),
        emit_observation(
            surface_text=(
                "okay, arsonist POV only here: build me a design spec for Claude, "
                "not hand wavy, give me nuts and bolts"
            ),
            source_id="m1",
        ),
    ]
    return appraise_repair_pressure(molecules, window_id="win-bridge")


def test_bridge_emits_graph_cognition_repair_pressure_signal():
    appraisal = _high_appraisal()
    signal = repair_appraisal_to_signal(appraisal)
    assert isinstance(signal, OrionSignalV1)
    assert signal.organ_id == "graph_cognition"
    assert signal.signal_kind == "repair_pressure"
    assert signal.organ_class == OrganClass.endogenous


def test_bridge_dimensions_match_appraisal():
    appraisal = _high_appraisal()
    signal = repair_appraisal_to_signal(appraisal)
    for key in (
        "level",
        "specificity_demand",
        "trust_rupture",
        "coherence_gap",
        "repetition_failure",
        "operational_block",
        "explicit_repair_command",
        "confidence",
    ):
        assert signal.dimensions[key] == appraisal.dimensions[key], key


def test_bridge_carries_causal_parents():
    appraisal = _high_appraisal()
    signal = repair_appraisal_to_signal(appraisal)
    assert signal.causal_parents == appraisal.causal_molecule_ids
    assert signal.source_event_id == appraisal.appraisal_id


def test_bridge_uses_deterministic_signal_id():
    appraisal = _high_appraisal()
    s1 = repair_appraisal_to_signal(appraisal)
    s2 = repair_appraisal_to_signal(appraisal)
    assert s1.signal_id == s2.signal_id


def test_bridge_observed_at_override():
    appraisal = _high_appraisal()
    when = datetime(2026, 5, 23, 12, 0, tzinfo=timezone.utc)
    signal = repair_appraisal_to_signal(appraisal, observed_at=when)
    assert signal.observed_at == when


def test_bridge_caps_notes():
    appraisal = _high_appraisal()
    appraisal.notes = [f"note_{i}" for i in range(10)]
    signal = repair_appraisal_to_signal(appraisal)
    assert len(signal.notes) <= 5
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_repair_pressure_signal_bridge.py -v`
Expected: import error — `signal_bridge` module not present.

- [ ] **Step 3: Implement the bridge**

Create `orion/substrate/appraisal/signal_bridge.py`:

```python
"""Bridge a RepairPressureAppraisalV1 into an OrionSignalV1.

The bridge owns the mapping from appraisal dimensions to canonical signal
dimensions registered on the graph_cognition organ. It is pure: no I/O,
no clock skew beyond `emitted_at`.
"""

from __future__ import annotations

from datetime import datetime, timezone

from orion.signals.models import OrganClass, OrionSignalV1
from orion.signals.signal_ids import make_signal_id

from .models import RepairPressureAppraisalV1


ORGAN_ID = "graph_cognition"
SIGNAL_KIND = "repair_pressure"
TTL_MS = 15_000


_DIMENSION_KEYS = (
    "level",
    "specificity_demand",
    "trust_rupture",
    "coherence_gap",
    "repetition_failure",
    "operational_block",
    "explicit_repair_command",
    "assistant_accountability_demand",
    "confidence",
)


def repair_appraisal_to_signal(
    appraisal: RepairPressureAppraisalV1,
    *,
    observed_at: datetime | None = None,
) -> OrionSignalV1:
    """Project the appraisal onto the canonical graph_cognition/repair_pressure signal."""

    dimensions: dict[str, float] = {
        key: appraisal.dimensions[key] for key in _DIMENSION_KEYS
    }

    return OrionSignalV1(
        signal_id=make_signal_id(ORGAN_ID, appraisal.appraisal_id),
        organ_id=ORGAN_ID,
        organ_class=OrganClass.endogenous,
        signal_kind=SIGNAL_KIND,
        dimensions=dimensions,
        causal_parents=list(appraisal.causal_molecule_ids),
        source_event_id=appraisal.appraisal_id,
        observed_at=observed_at or appraisal.created_at,
        emitted_at=datetime.now(timezone.utc),
        ttl_ms=TTL_MS,
        summary=appraisal.summary,
        notes=list(appraisal.notes[:5]),
    )
```

- [ ] **Step 4: Export from package init**

Edit `orion/substrate/appraisal/__init__.py` — replace its contents with:

```python
"""Substrate-derived appraisers. See docs/plans/substrate/2026-05-23-repair-pressure-v1.md."""

from .evidence import DETECTOR_NAME, extract_repair_evidence
from .models import RepairEvidenceV1, RepairPressureAppraisalV1
from .repair_pressure import appraise_repair_pressure
from .signal_bridge import ORGAN_ID, SIGNAL_KIND, repair_appraisal_to_signal

__all__ = [
    "DETECTOR_NAME",
    "ORGAN_ID",
    "RepairEvidenceV1",
    "RepairPressureAppraisalV1",
    "SIGNAL_KIND",
    "appraise_repair_pressure",
    "extract_repair_evidence",
    "repair_appraisal_to_signal",
]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_repair_pressure_signal_bridge.py -v`
Expected: all PASS (2 registry + 6 bridge = 8 tests).

- [ ] **Step 6: Commit**

```bash
git add orion/substrate/appraisal/signal_bridge.py orion/substrate/appraisal/__init__.py tests/test_repair_pressure_signal_bridge.py
git commit -m "feat(substrate): repair appraisal -> OrionSignalV1 bridge"
```

---

## Task 6: Windowing — recent chat molecule selection

**Files:**
- Create: `orion/substrate/appraisal/windowing.py`
- Modify: `orion/substrate/appraisal/__init__.py` (export `select_recent_chat_molecules`)
- Test: `tests/test_repair_pressure_windowing.py`

- [ ] **Step 1: Write the failing windowing tests**

Create `tests/test_repair_pressure_windowing.py`:

```python
"""Windowing tests for repair pressure appraisal."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.mind.substrate_emit import emit_observation
from orion.substrate.appraisal.windowing import select_recent_chat_molecules
from orion.substrate.molecules import SubstrateMoleculeV1


def _obs(text: str, source_id: str, age_seconds: int) -> SubstrateMoleculeV1:
    mol = emit_observation(surface_text=text, source_id=source_id)
    when = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
    mol.created_at = when
    mol.last_touched_at = when
    return mol


def test_windowing_prefers_source_id_match():
    a = _obs("topic A", "conv-1", age_seconds=10)
    b = _obs("topic B", "conv-2", age_seconds=10)
    out = select_recent_chat_molecules([a, b], source_id="conv-1")
    assert [m.molecule_id for m in out] == [a.molecule_id]


def test_windowing_drops_too_old():
    fresh = _obs("fresh", "conv-1", age_seconds=10)
    stale = _obs("stale", "conv-1", age_seconds=10_000)
    out = select_recent_chat_molecules([fresh, stale], max_age_seconds=300)
    assert [m.molecule_id for m in out] == [fresh.molecule_id]


def test_windowing_caps_count_and_sorts_desc():
    molecules = [_obs(f"t{i}", "conv-1", age_seconds=i) for i in range(50)]
    out = select_recent_chat_molecules(molecules, max_count=5)
    assert len(out) == 5
    ages = [m.created_at for m in out]
    assert ages == sorted(ages, reverse=True)


def test_windowing_empty_input():
    assert select_recent_chat_molecules([]) == []


def test_windowing_without_source_id_returns_all_fresh():
    a = _obs("a", "conv-1", age_seconds=10)
    b = _obs("b", "conv-2", age_seconds=20)
    out = select_recent_chat_molecules([a, b])
    assert {m.molecule_id for m in out} == {a.molecule_id, b.molecule_id}
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_repair_pressure_windowing.py -v`
Expected: import error — module not present.

- [ ] **Step 3: Implement windowing**

Create `orion/substrate/appraisal/windowing.py`:

```python
"""Select the recent chat-window molecules to feed the repair appraiser."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable

from orion.substrate.molecules import SubstrateMoleculeV1


def select_recent_chat_molecules(
    molecules: Iterable[SubstrateMoleculeV1],
    *,
    source_id: str | None = None,
    max_age_seconds: int = 300,
    max_count: int = 20,
) -> list[SubstrateMoleculeV1]:
    """Return at most ``max_count`` molecules in newest-first order.

    Rules (spec §12):
    - If ``source_id`` is provided, keep only molecules whose
      ``provenance['source_id']`` matches.
    - Drop molecules older than ``max_age_seconds``.
    - Sort by ``created_at`` descending.
    - Cap at ``max_count``.
    """

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(seconds=max_age_seconds)
    keep: list[SubstrateMoleculeV1] = []
    for m in molecules:
        if source_id is not None and m.provenance.get("source_id") != source_id:
            continue
        if m.created_at < cutoff:
            continue
        keep.append(m)

    keep.sort(key=lambda m: m.created_at, reverse=True)
    return keep[:max_count]
```

- [ ] **Step 4: Export from package init**

Edit `orion/substrate/appraisal/__init__.py` — replace its contents with:

```python
"""Substrate-derived appraisers. See docs/plans/substrate/2026-05-23-repair-pressure-v1.md."""

from .evidence import DETECTOR_NAME, extract_repair_evidence
from .models import RepairEvidenceV1, RepairPressureAppraisalV1
from .repair_pressure import appraise_repair_pressure
from .signal_bridge import ORGAN_ID, SIGNAL_KIND, repair_appraisal_to_signal
from .windowing import select_recent_chat_molecules

__all__ = [
    "DETECTOR_NAME",
    "ORGAN_ID",
    "RepairEvidenceV1",
    "RepairPressureAppraisalV1",
    "SIGNAL_KIND",
    "appraise_repair_pressure",
    "extract_repair_evidence",
    "repair_appraisal_to_signal",
    "select_recent_chat_molecules",
]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_repair_pressure_windowing.py -v`
Expected: all 5 PASS.

- [ ] **Step 6: Commit**

```bash
git add orion/substrate/appraisal/windowing.py orion/substrate/appraisal/__init__.py tests/test_repair_pressure_windowing.py
git commit -m "feat(substrate): windowing helper for recent chat molecules"
```

---

## Task 7: Behavior consumer — `apply_repair_pressure_contract`

**Files:**
- Create: `orion/substrate/appraisal/contract.py`
- Modify: `orion/substrate/appraisal/__init__.py` (export `apply_repair_pressure_contract`)
- Test: `tests/test_repair_pressure_behavior_contract.py`

- [ ] **Step 1: Write the failing contract tests**

Create `tests/test_repair_pressure_behavior_contract.py`:

```python
"""Behavior contract tests for repair pressure — spec §14.4 and §11.1."""

from __future__ import annotations

from datetime import datetime, timezone

from orion.signals.models import OrganClass, OrionSignalV1
from orion.substrate.appraisal.contract import (
    REPAIR_PRESSURE_DEBUG_KEY,
    apply_repair_pressure_contract,
)


def _signal(level: float, confidence: float) -> OrionSignalV1:
    now = datetime.now(timezone.utc)
    return OrionSignalV1(
        signal_id="sig-test",
        organ_id="graph_cognition",
        organ_class=OrganClass.endogenous,
        signal_kind="repair_pressure",
        dimensions={
            "level": level,
            "specificity_demand": 0.9,
            "trust_rupture": 0.8,
            "coherence_gap": 0.7,
            "repetition_failure": 0.0,
            "operational_block": 0.6,
            "explicit_repair_command": 0.8,
            "assistant_accountability_demand": 0.0,
            "confidence": confidence,
        },
        causal_parents=["mol_a", "mol_b"],
        source_event_id="app-test",
        observed_at=now,
        emitted_at=now,
    )


def test_high_repair_pressure_forces_repair_concrete_mode():
    contract = apply_repair_pressure_contract({}, _signal(level=0.86, confidence=0.82))
    assert contract["mode"] == "repair_concrete"
    rules = contract["rules"]
    assert any("one concrete operational path" in r for r in rules)
    assert any("do not build" in r.lower() for r in rules)
    assert any("tests/acceptance checks" in r.lower() for r in rules)


def test_mid_repair_pressure_forces_concrete_bias():
    contract = apply_repair_pressure_contract({}, _signal(level=0.55, confidence=0.70))
    assert contract["mode"] == "concrete_bias"
    rules = contract["rules"]
    assert any("more specific" in r.lower() for r in rules)


def test_weak_repair_pressure_does_not_change_mode():
    contract = apply_repair_pressure_contract(
        {"mode": "default"}, _signal(level=0.30, confidence=0.90)
    )
    assert contract["mode"] == "default"
    assert "rules" not in contract or not contract["rules"]


def test_low_confidence_high_level_does_not_change_mode():
    contract = apply_repair_pressure_contract(
        {"mode": "default"}, _signal(level=0.86, confidence=0.40)
    )
    # level alone is not enough — confidence must also clear the bar.
    assert contract["mode"] == "default"


def test_none_signal_is_noop():
    contract = apply_repair_pressure_contract({"mode": "default"}, None)
    assert contract == {"mode": "default"}


def test_unrelated_signal_kind_is_ignored():
    sig = _signal(level=0.86, confidence=0.82)
    sig = sig.model_copy(update={"signal_kind": "goal_pressure"})
    contract = apply_repair_pressure_contract({"mode": "default"}, sig)
    assert contract["mode"] == "default"


def test_inspectable_debug_metadata_present_when_high():
    signal = _signal(level=0.86, confidence=0.82)
    contract = apply_repair_pressure_contract({}, signal)
    debug = contract[REPAIR_PRESSURE_DEBUG_KEY]
    assert debug["level"] == 0.86
    assert debug["confidence"] == 0.82
    assert debug["mode_applied"] == "repair_concrete"
    assert "specificity_demand" in debug["evidence_kinds"]
    assert debug["causal_molecule_ids"] == ["mol_a", "mol_b"]


def test_base_contract_is_not_mutated():
    base = {"mode": "default", "rules": ["keep"]}
    out = apply_repair_pressure_contract(base, _signal(level=0.86, confidence=0.82))
    assert base == {"mode": "default", "rules": ["keep"]}, base
    assert out is not base
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_repair_pressure_behavior_contract.py -v`
Expected: import error — `contract` module not present.

- [ ] **Step 3: Implement the contract consumer**

Create `orion/substrate/appraisal/contract.py`:

```python
"""Behavior consumer: repair_pressure signal → response contract mode.

This is intentionally a pure function over a contract dict. It does not
import or modify any existing chat pipeline. Callers can opt in by passing
their assembled contract through `apply_repair_pressure_contract`.
"""

from __future__ import annotations

import copy
from typing import Any

from orion.signals.models import OrionSignalV1

from .signal_bridge import SIGNAL_KIND


REPAIR_PRESSURE_DEBUG_KEY = "repair_pressure"

_LEVEL_HIGH = 0.75
_LEVEL_MID = 0.45
_CONFIDENCE_MIN = 0.60

_RULES_REPAIR_CONCRETE: tuple[str, ...] = (
    "no broad architecture wandering",
    "no spiritual abstraction",
    "no unsolicited future roadmap unless asked",
    "answer with one concrete operational path",
    "include file/module boundaries",
    "include 'do not build' section",
    "include tests/acceptance checks",
    "acknowledge correction briefly",
    "prefer deterministic logic over LLM vibe scoring",
)

_RULES_CONCRETE_BIAS: tuple[str, ...] = (
    "be more specific",
    "show assumptions",
    "include next concrete action",
)


def _evidence_kinds_from_dimensions(dims: dict[str, float]) -> list[str]:
    candidate_kinds = (
        "specificity_demand",
        "trust_rupture",
        "coherence_gap",
        "repetition_failure",
        "operational_block",
        "explicit_repair_command",
        "assistant_accountability_demand",
    )
    return [k for k in candidate_kinds if dims.get(k, 0.0) > 0.0]


def apply_repair_pressure_contract(
    base_contract: dict[str, Any],
    signal: OrionSignalV1 | None,
) -> dict[str, Any]:
    """Return a new contract dict adjusted by the repair_pressure signal.

    Spec §11.1:
        level >= 0.75 and confidence >= 0.60  → mode = "repair_concrete"
        0.45 <= level < 0.75                  → mode = "concrete_bias"
        otherwise                              → unchanged
    """

    contract = copy.deepcopy(base_contract)

    if signal is None or signal.signal_kind != SIGNAL_KIND:
        return contract

    level = float(signal.dimensions.get("level", 0.0))
    confidence = float(signal.dimensions.get("confidence", 0.0))

    if level >= _LEVEL_HIGH and confidence >= _CONFIDENCE_MIN:
        mode_applied = "repair_concrete"
        contract["mode"] = mode_applied
        contract["rules"] = list(_RULES_REPAIR_CONCRETE)
    elif _LEVEL_MID <= level < _LEVEL_HIGH:
        mode_applied = "concrete_bias"
        contract["mode"] = mode_applied
        contract["rules"] = list(_RULES_CONCRETE_BIAS)
    else:
        return contract  # no change, no debug payload

    contract[REPAIR_PRESSURE_DEBUG_KEY] = {
        "level": level,
        "confidence": confidence,
        "mode_applied": mode_applied,
        "evidence_kinds": _evidence_kinds_from_dimensions(signal.dimensions),
        "causal_molecule_ids": list(signal.causal_parents),
        "source_event_id": signal.source_event_id,
    }
    return contract
```

- [ ] **Step 4: Export from package init**

Edit `orion/substrate/appraisal/__init__.py` — replace its contents with:

```python
"""Substrate-derived appraisers. See docs/plans/substrate/2026-05-23-repair-pressure-v1.md."""

from .contract import REPAIR_PRESSURE_DEBUG_KEY, apply_repair_pressure_contract
from .evidence import DETECTOR_NAME, extract_repair_evidence
from .models import RepairEvidenceV1, RepairPressureAppraisalV1
from .repair_pressure import appraise_repair_pressure
from .signal_bridge import ORGAN_ID, SIGNAL_KIND, repair_appraisal_to_signal
from .windowing import select_recent_chat_molecules

__all__ = [
    "DETECTOR_NAME",
    "ORGAN_ID",
    "REPAIR_PRESSURE_DEBUG_KEY",
    "RepairEvidenceV1",
    "RepairPressureAppraisalV1",
    "SIGNAL_KIND",
    "apply_repair_pressure_contract",
    "appraise_repair_pressure",
    "extract_repair_evidence",
    "repair_appraisal_to_signal",
    "select_recent_chat_molecules",
]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_repair_pressure_behavior_contract.py -v`
Expected: all 8 PASS.

- [ ] **Step 6: Commit**

```bash
git add orion/substrate/appraisal/contract.py orion/substrate/appraisal/__init__.py tests/test_repair_pressure_behavior_contract.py
git commit -m "feat(substrate): behavior consumer apply_repair_pressure_contract"
```

---

## Task 8: End-to-end Definition-of-Done test

**Files:**
- Test: `tests/test_repair_pressure_e2e.py`

This task adds no new production code — it locks the full causal chain (spec §17).

- [ ] **Step 1: Write the end-to-end test**

Create `tests/test_repair_pressure_e2e.py`:

```python
"""Definition-of-Done: full causal chain inspectable end-to-end. Spec §17."""

from __future__ import annotations

from orion.mind.substrate_emit import emit_observation
from orion.substrate.appraisal import (
    REPAIR_PRESSURE_DEBUG_KEY,
    appraise_repair_pressure,
    apply_repair_pressure_contract,
    extract_repair_evidence,
    repair_appraisal_to_signal,
    select_recent_chat_molecules,
)


def test_full_causal_chain_changes_next_response_contract():
    chat_lines = [
        "you gave me garbage directions",
        "you keep making shit up — again",
        "this is becoming a swamp, doesn't converge",
        "okay, arsonist POV only here: build me a design spec for Claude, not hand wavy, give me nuts and bolts",
    ]
    molecules = [
        emit_observation(surface_text=line, source_id="conv-DoD") for line in chat_lines
    ]

    window = select_recent_chat_molecules(molecules, source_id="conv-DoD")
    assert window, "windowing must keep at least the current turn"

    evidence = extract_repair_evidence(window)
    kinds = {e.evidence_kind for e in evidence}
    assert {"specificity_demand", "trust_rupture", "coherence_gap"} <= kinds

    appraisal = appraise_repair_pressure(window, window_id="win-DoD")
    assert appraisal.dimensions["level"] >= 0.75
    assert appraisal.confidence >= 0.60
    assert appraisal.causal_molecule_ids  # non-empty

    signal = repair_appraisal_to_signal(appraisal)
    assert signal.organ_id == "graph_cognition"
    assert signal.signal_kind == "repair_pressure"
    assert signal.causal_parents == appraisal.causal_molecule_ids

    contract = apply_repair_pressure_contract({"mode": "default"}, signal)
    assert contract["mode"] == "repair_concrete"
    debug = contract[REPAIR_PRESSURE_DEBUG_KEY]
    assert debug["mode_applied"] == "repair_concrete"
    assert set(debug["evidence_kinds"]) >= {
        "specificity_demand",
        "trust_rupture",
        "coherence_gap",
    }
    assert debug["causal_molecule_ids"] == appraisal.causal_molecule_ids


def test_neutral_turn_does_not_change_contract():
    molecules = [emit_observation(surface_text="what is the weather like?", source_id="conv-low")]
    window = select_recent_chat_molecules(molecules, source_id="conv-low")
    appraisal = appraise_repair_pressure(window, window_id="win-low")
    signal = repair_appraisal_to_signal(appraisal)
    contract = apply_repair_pressure_contract({"mode": "default"}, signal)
    assert contract["mode"] == "default"
    assert REPAIR_PRESSURE_DEBUG_KEY not in contract
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_repair_pressure_e2e.py -v`
Expected: both PASS.

- [ ] **Step 3: Run the whole repair-pressure suite together**

Run:
```bash
pytest tests/test_repair_pressure_models.py \
       tests/test_repair_pressure_evidence.py \
       tests/test_repair_pressure_appraisal.py \
       tests/test_repair_pressure_signal_bridge.py \
       tests/test_repair_pressure_windowing.py \
       tests/test_repair_pressure_behavior_contract.py \
       tests/test_repair_pressure_e2e.py -v
```
Expected: every test PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_repair_pressure_e2e.py
git commit -m "test(substrate): end-to-end repair_pressure causal chain"
```

---

## Task 9: Acceptance audit and PR

**Files:** none modified. This task locks the spec §15 acceptance gates.

- [ ] **Step 1: Re-check spec §15 acceptance criteria by hand**

For each of the 12 acceptance items, confirm a test or code path enforces it:

| # | Criterion | Where enforced |
|---|---|---|
| 1 | Raw chat text does not directly produce an organ signal. | `tests/test_repair_pressure_e2e.py` — must go through molecules → evidence → appraisal → signal. |
| 2 | Raw chat first becomes one or more substrate molecules. | `emit_observation` produces `SubstrateMoleculeV1`; validated by `validate_molecule` in the substrate kernel tests. |
| 3 | Repair evidence is extracted with explicit kinds. | `tests/test_repair_pressure_evidence.py` covers all 7 kinds (incl. `assistant_accountability_demand` via the `you keep` phrase). |
| 4 | Repair pressure is computed by a deterministic reducer. | `repair_pressure.appraise_repair_pressure` — pure function, no I/O, no LLM. |
| 5 | Derived repair pressure is emitted as `OrionSignalV1`. | `tests/test_repair_pressure_signal_bridge.py`. |
| 6 | Signal includes causal molecule IDs. | `test_bridge_carries_causal_parents`. |
| 7 | High repair pressure changes the next response contract. | `test_high_repair_pressure_forces_repair_concrete_mode`, `test_full_causal_chain_changes_next_response_contract`. |
| 8 | Debug metadata exposes evidence, scores, confidence, causal chain. | `test_inspectable_debug_metadata_present_when_high`. |
| 9 | Weak evidence fails closed. | `test_low_repair_pressure_neutral_chat`, `test_fail_closed_no_molecules`, `test_single_weak_evidence_caps_confidence`, `test_low_confidence_high_level_does_not_change_mode`. |
| 10 | No new substrate gradients introduced. | Grep: `git diff main -- orion/schema_kernel/gradient.py` must be empty. |
| 11 | No new psychological atoms introduced. | Grep: `git diff main -- orion/schema_kernel/atom.py` must be empty. |
| 12 | Payload does not become the machine contract. | `RepairEvidenceV1.features` / `.span` are only read by the detector itself for audit; consumers read `dimensions[*]` and `evidence_kinds[*]`. |

Run the verification greps:

```bash
git diff main -- orion/schema_kernel/atom.py orion/schema_kernel/gradient.py orion/schema_kernel/registry.py orion/substrate/molecules.py
```

Expected: empty (no changes to those files).

- [ ] **Step 2: Run the full repo test suite to confirm nothing else broke**

Run: `pytest tests/ -x -q 2>&1 | tail -40`
Expected: zero new failures attributable to this branch. (Pre-existing failures unrelated to substrate/appraisal can be noted and skipped.)

- [ ] **Step 3: Push the branch**

```bash
git push -u origin feat/repair-pressure-v1
```

- [ ] **Step 4: Open the PR**

```bash
gh pr create --title "feat(substrate): repair_pressure_v1 — first substrate-derived organ signal" --body "$(cat <<'EOF'
# repair_pressure_v1 — first real substrate-derived organ signal

## Summary

Build one vertical proving loop:

```
raw chat turn
  → substrate observation molecule
  → repair evidence (deterministic phrase match)
  → repair pressure appraisal (explicit formula)
  → OrionSignalV1 graph_cognition/repair_pressure
  → response contract mode = repair_concrete
  → inspectable causal chain
```

If the chain does not change the next response, it is not cognition. It is journaling. This PR makes the chain change the next response, end to end, with tests.

## Arsonist constraints respected

- **No new substrate gradients.** `salience, contradiction, novelty, coherence` only.
- **No new schema-kernel atoms.** All 12 stay.
- **No new substrate molecule kinds.** Chat turn uses the existing `observation` kind via `orion.mind.substrate_emit.emit_observation`.
- **Strict causal layering.** Evidence is a separate pydantic model, not a molecule; appraisal is a separate pydantic model, not a signal; signal is `OrionSignalV1`; contract is a dict.
- **Payload is not the machine contract.** Consumers read `OrionSignalV1.dimensions` and the `evidence_kinds[*]` debug array — not phrase spans or feature dicts.
- **Fail closed.** Empty / single-weak evidence forces `level=0.0` (empty) or caps `confidence ≤ 0.45` (single weak), with explicit notes.

## What's inside

### `orion/substrate/appraisal/` (new package)

| File | Responsibility |
|---|---|
| `models.py` | `RepairEvidenceV1`, `RepairPressureAppraisalV1` (pydantic, `extra="forbid"`). |
| `evidence.py` | Deterministic phrase-match detector covering 7 evidence kinds. |
| `repair_pressure.py` | Explicit-formula reducer (spec §9.3 / §9.4). |
| `signal_bridge.py` | `repair_appraisal_to_signal` → `OrionSignalV1(graph_cognition/repair_pressure)`. |
| `windowing.py` | `select_recent_chat_molecules` — source_id filter, age cap, count cap. |
| `contract.py` | `apply_repair_pressure_contract` — pure function flipping response mode. |

### `orion/signals/registry.py` (additive edit)

`graph_cognition` organ entry gains:

- `signal_kinds += ["repair_pressure"]`
- `canonical_dimensions += ["level", "specificity_demand", "trust_rupture", "coherence_gap", "repetition_failure", "operational_block", "explicit_repair_command", "assistant_accountability_demand"]`

No causal-parent edges change. No other organ touched.

### Tests

| Test file | Coverage |
|---|---|
| `tests/test_repair_pressure_models.py` | Pydantic shape, `extra="forbid"`, bounds. |
| `tests/test_repair_pressure_evidence.py` | All 6 spec §14.1 phrase thresholds, span auditing, neutral-text empty-result. |
| `tests/test_repair_pressure_appraisal.py` | High / low / fail-closed / single-weak / dimensions present. |
| `tests/test_repair_pressure_signal_bridge.py` | Registry shape + bridge organ/kind/dimensions/causal_parents/deterministic id. |
| `tests/test_repair_pressure_windowing.py` | source_id filter, age cap, count cap, sort order. |
| `tests/test_repair_pressure_behavior_contract.py` | repair_concrete / concrete_bias / no-op / low-confidence-guard / debug metadata / base immutability. |
| `tests/test_repair_pressure_e2e.py` | Full causal chain (spec §17 Definition of Done). |

## Acceptance audit

All 12 spec §15 criteria are enforced by tests or by structural absence-of-diff against the architecture-locked files (`atom.py`, `gradient.py`, `registry.py`, `molecules.py`). See the audit table in `docs/plans/substrate/2026-05-23-repair-pressure-v1.md`, Task 9.

## What this PR deliberately avoids

- No LLM scoring. No embeddings. No GraphDB. No new substrate fields.
- No chat pipeline rewrite. The contract consumer is a pure function callers opt into.
- No service extraction. Phase 5 (`services/orion-substrate-appraiser`) stays unbuilt until the library proves useful.
- No emotion classifier. This is not sentiment.

## Files

```
orion/signals/registry.py                              (additive edit)
orion/substrate/appraisal/__init__.py                  (new)
orion/substrate/appraisal/models.py                    (new)
orion/substrate/appraisal/evidence.py                  (new)
orion/substrate/appraisal/repair_pressure.py           (new)
orion/substrate/appraisal/signal_bridge.py             (new)
orion/substrate/appraisal/windowing.py                 (new)
orion/substrate/appraisal/contract.py                  (new)
tests/test_repair_pressure_models.py                   (new)
tests/test_repair_pressure_evidence.py                 (new)
tests/test_repair_pressure_appraisal.py                (new)
tests/test_repair_pressure_signal_bridge.py            (new)
tests/test_repair_pressure_windowing.py                (new)
tests/test_repair_pressure_behavior_contract.py        (new)
tests/test_repair_pressure_e2e.py                      (new)
```

## Test plan

- [ ] `pytest tests/test_repair_pressure_models.py -v` passes
- [ ] `pytest tests/test_repair_pressure_evidence.py -v` passes
- [ ] `pytest tests/test_repair_pressure_appraisal.py -v` passes
- [ ] `pytest tests/test_repair_pressure_signal_bridge.py -v` passes
- [ ] `pytest tests/test_repair_pressure_windowing.py -v` passes
- [ ] `pytest tests/test_repair_pressure_behavior_contract.py -v` passes
- [ ] `pytest tests/test_repair_pressure_e2e.py -v` passes
- [ ] `git diff main -- orion/schema_kernel/atom.py orion/schema_kernel/gradient.py orion/schema_kernel/registry.py orion/substrate/molecules.py` is empty
- [ ] Full repo `pytest tests/ -x -q` introduces no new failures

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Phase mapping (spec §16)

| Spec phase | Task(s) in this plan |
|---|---|
| Phase 1 — library vertical slice | Tasks 2–6 (models, evidence, reducer, bridge, windowing). |
| Phase 2 — chat integration shim | Already in place: `orion.mind.substrate_emit.emit_observation` is the integration point. The plan uses it directly in Task 8 e2e. No further pipeline changes in this PR. |
| Phase 3 — behavior contract | Task 7 (`apply_repair_pressure_contract`). |
| Phase 4 — inspect/debug | Task 7's `REPAIR_PRESSURE_DEBUG_KEY` payload + Task 8's e2e assertion that the debug payload contains molecules used / evidence kinds / level / confidence / mode applied. |
| Phase 5 — optional service extraction | Out of scope for this PR. |

---

## Risks / open questions

- **Phrase-table calibration.** If a spec §14.1 threshold misses by ε after implementation, raise the strongest phrase weight in `_PHRASES` to 0.95; do **not** lower the test threshold (the test is the spec).
- **`assistant_accountability_demand` overlap.** Phrases like "you keep" hit both `repetition_failure` and `assistant_accountability_demand`. This is intentional — both kinds are emitted from one molecule, max-aggregated downstream.
- **Windowing source_id key.** `select_recent_chat_molecules` reads `provenance['source_id']`. `emit_observation` sets that key when `source_id` is passed. Tests must pass a `source_id` to assert source-bound windowing.
- **Existing graph_cognition consumers.** Downstream consumers of `graph_cognition.canonical_dimensions` (if any read it for validation) will now see additional dimension keys. The registry is metadata only; no runtime validator currently rejects extras. If a strict-validator is added later, this PR's dimensions are already declared canonical.
