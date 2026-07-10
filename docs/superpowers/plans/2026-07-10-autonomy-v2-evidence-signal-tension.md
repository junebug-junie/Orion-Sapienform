# AutonomyStateV2 Evidence Redesign via `signal_tension` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace AutonomyStateV2 keyword/theater evidence with a typed omit-when-empty compiler and map-driven `chat_evidence_to_tension` pressure math, proven by a movement fixture, while keeping the reducer flag default-off and hard-isolating phi / SelfState / DriveEngine.

**Architecture:** Chat locals (`social` / `social_bridge` / reasoning upstream / infra / user message) feed `AutonomyEvidenceCompiler`, which emits `AutonomyEvidenceRefV1` only when upstream is proven non-empty (with optional `signal_kind` / `dimension` / `value` / `observed_at`). Pressure-eligible refs mint `TensionEventV1` via `chat_evidence_to_tension` (same family as `failure_to_tension`: map lookup, no DeviationGate/EWMA). The reducer folds `magnitude * drive_impacts` into `drive_pressures` and derives tension kinds from pressure thresholds only. Stance wires locals **before** writing `ctx["chat_social_bridge_summary"]`.

**Tech Stack:** Python 3.12, Pydantic v2, pytest, YAML (`config/autonomy/signal_drive_map.yaml`), existing `orion.autonomy.signal_tension` / `SignalDriveMap`.

**Spec:** `docs/superpowers/specs/2026-07-10-autonomy-v2-evidence-signal-tension-design.md`

**Worktree (create before implementing):**
```bash
cd /mnt/scripts/Orion-Sapienform
git fetch origin
git worktree add .worktrees/autonomy-v2-evidence-signal-tension -b feat/autonomy-v2-evidence-signal-tension origin/main
cd .worktrees/autonomy-v2-evidence-signal-tension
```

---

## File map

| File | Action | Responsibility |
|------|--------|----------------|
| `orion/autonomy/models.py` | Modify | Optional `signal_kind` / `dimension` / `value` on `AutonomyEvidenceRefV1` |
| `orion/autonomy/evidence_compiler.py` | Create | Proven-non-empty gates → evidence refs + omit debug |
| `orion/autonomy/signal_tension.py` | Modify | Add `chat_evidence_to_tension` (direct map adapter) |
| `orion/autonomy/reducer.py` | Modify | Tension fold; delete keyword pressure / blob keyword OR paths |
| `config/autonomy/signal_drive_map.yaml` | Modify | Add `chat_social_hazard` + `chat_reasoning_quality` rows |
| `services/orion-cortex-exec/app/chat_stance.py` | Modify | Locals → compiler → reducer; stop reading ctx bridge for this turn |
| `docs/autonomy_state_v2_reducer.md` | Modify | Evidence contract + enable bar; remove polarity-blind limitation claim |
| `orion/autonomy/tests/test_evidence_compiler.py` | Create | Omit/emit gates |
| `orion/autonomy/tests/test_signal_tension.py` | Modify | Chat adapter cases |
| `orion/autonomy/tests/test_signal_drive_map.py` | Modify | Assert new kinds present |
| `orion/autonomy/tests/test_autonomy_reducer.py` | Modify | Rewrite keyword-based tests; add map-driven movement |
| `orion/autonomy/tests/test_autonomy_isolation.py` | Create | No AutonomyStateV2 → phi / SelfState wire |
| `orion/autonomy/evals/run_autonomy_v2_movement_eval.py` | Create | Trace-proven pressure / dominant_drive movement |
| `services/orion-cortex-exec/tests/test_chat_stance_autonomy_v2.py` | Modify | Locals ordering + omit empty-repo + debug ctx |

**Out of scope (do not touch):** `DriveEngine`, `build_self_state`, spark introspector phi corpus, DeviationGate warm-up for chat, Hub UI, live reasoning-repository producer.

---

### Task 1: Extend `AutonomyEvidenceRefV1` schema

**Files:**
- Modify: `orion/autonomy/models.py` (`AutonomyEvidenceRefV1`)
- Test: `orion/autonomy/tests/test_evidence_ref_schema.py` (create)

- [ ] **Step 1: Write the failing schema test**

```python
# orion/autonomy/tests/test_evidence_ref_schema.py
from __future__ import annotations

from datetime import datetime

from orion.autonomy.models import AutonomyEvidenceRefV1


def test_evidence_ref_accepts_optional_signal_fields() -> None:
    fixed = datetime(2026, 7, 10, 12, 0, 0)
    ev = AutonomyEvidenceRefV1(
        evidence_id="social_bridge:abc",
        source="social_bridge",
        kind="relational_signal",
        summary="cooldown_active",
        confidence=0.6,
        observed_at=fixed,
        signal_kind="chat_social_hazard",
        dimension="cooldown_active",
        value=1.0,
    )
    assert ev.signal_kind == "chat_social_hazard"
    assert ev.dimension == "cooldown_active"
    assert ev.value == 1.0


def test_evidence_ref_defaults_signal_fields_to_none() -> None:
    ev = AutonomyEvidenceRefV1(
        evidence_id="user_turn:x",
        source="user_message",
        kind="user_turn",
        summary="hi",
    )
    assert ev.signal_kind is None
    assert ev.dimension is None
    assert ev.value is None
    assert ev.observed_at is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest orion/autonomy/tests/test_evidence_ref_schema.py -v`  
Expected: FAIL with unexpected keyword argument `signal_kind` (or similar Pydantic validation error).

- [ ] **Step 3: Minimal schema change**

In `orion/autonomy/models.py`, update `AutonomyEvidenceRefV1` to:

```python
class AutonomyEvidenceRefV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    evidence_id: str
    source: str
    kind: str
    summary: str | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    observed_at: datetime | None = None
    # Optional typed pressure fields. Audit-only refs may omit these.
    # Confidence values remain kind-literal constants in v1 (uncalibrated).
    signal_kind: str | None = None
    dimension: str | None = None
    value: float | None = Field(default=None, ge=0.0, le=1.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest orion/autonomy/tests/test_evidence_ref_schema.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add orion/autonomy/models.py orion/autonomy/tests/test_evidence_ref_schema.py
git commit -m "$(cat <<'EOF'
feat(autonomy): optional signal fields on AutonomyEvidenceRefV1

EOF
)"
```

---

### Task 2: Grow `signal_drive_map.yaml` for chat kinds

**Files:**
- Modify: `config/autonomy/signal_drive_map.yaml`
- Modify: `orion/autonomy/tests/test_signal_drive_map.py`

- [ ] **Step 1: Write failing map assertions**

Append to `orion/autonomy/tests/test_signal_drive_map.py`:

```python
def test_chat_social_hazard_exact_dims_present() -> None:
    m = load_signal_drive_map()
    assert "chat_social_hazard" in m.signal_kinds()
    for dim in ("cooldown_active", "duplicate_message", "self_message_loop"):
        rule = m.match("chat_social_hazard", dim)
        assert rule is not None, dim
        assert rule.worse == "up"
        assert "relational" in rule.drives


def test_chat_reasoning_quality_fallback_present() -> None:
    m = load_signal_drive_map()
    rule = m.match("chat_reasoning_quality", "fallback")
    assert rule is not None
    assert rule.worse == "up"
    assert "coherence" in rule.drives
    assert "predictive" in rule.drives


def test_unmapped_chat_hazard_dimension_is_none() -> None:
    m = load_signal_drive_map()
    assert m.match("chat_social_hazard", "context_excluded:foo") is None
    assert m.match("chat_social_hazard", "peer_targeted_elsewhere") is None
```

- [ ] **Step 2: Run to verify fail**

Run: `pytest orion/autonomy/tests/test_signal_drive_map.py::test_chat_social_hazard_exact_dims_present orion/autonomy/tests/test_signal_drive_map.py::test_chat_reasoning_quality_fallback_present -v`  
Expected: FAIL (`chat_social_hazard` not in signal_kinds).

- [ ] **Step 3: Add YAML rows**

Append under `signal_kinds:` in `config/autonomy/signal_drive_map.yaml`:

```yaml
  # Chat AutonomyStateV2 evidence (direct adapter; no DeviationGate).
  # Exact hazard keys only — prefix hazards like context_excluded:* are audit-only in v1.
  chat_social_hazard:
    cooldown_active:
      worse: up
      drives: {relational: 0.5, capability: 0.2}
    duplicate_message:
      worse: up
      drives: {relational: 0.4}
    self_message_loop:
      worse: up
      drives: {relational: 0.5, capability: 0.3}

  chat_reasoning_quality:
    fallback:
      worse: up
      drives: {coherence: 0.5, predictive: 0.3}
```

- [ ] **Step 4: Run map tests**

Run: `pytest orion/autonomy/tests/test_signal_drive_map.py -q`  
Expected: PASS (including existing biometrics/failure cases).

- [ ] **Step 5: Commit**

```bash
git add config/autonomy/signal_drive_map.yaml orion/autonomy/tests/test_signal_drive_map.py
git commit -m "$(cat <<'EOF'
feat(autonomy): map chat_social_hazard and chat_reasoning_quality

EOF
)"
```

---

### Task 3: `chat_evidence_to_tension` adapter

**Files:**
- Modify: `orion/autonomy/signal_tension.py`
- Modify: `orion/autonomy/tests/test_signal_tension.py`

- [ ] **Step 1: Write failing adapter tests**

Append to `orion/autonomy/tests/test_signal_tension.py`:

```python
from datetime import datetime

from orion.autonomy.models import AutonomyEvidenceRefV1
from orion.autonomy.signal_tension import chat_evidence_to_tension


def test_chat_evidence_mapped_hazard_mints_tension() -> None:
    ev = AutonomyEvidenceRefV1(
        evidence_id="h1",
        source="social_bridge",
        kind="relational_signal",
        summary="cooldown_active",
        confidence=0.6,
        observed_at=datetime(2026, 7, 10, 12, 0, 0),
        signal_kind="chat_social_hazard",
        dimension="cooldown_active",
        value=1.0,
    )
    t = chat_evidence_to_tension(ev, SDM)
    assert t is not None
    assert t.kind == "tension.chat_evidence.v1"
    assert t.magnitude > 0.0
    assert t.drive_impacts.get("relational", 0.0) > 0.0


def test_chat_evidence_unmapped_or_missing_fields_returns_none() -> None:
    bare = AutonomyEvidenceRefV1(
        evidence_id="h2",
        source="social_bridge",
        kind="relational_signal",
        summary="context_excluded:memory",
        confidence=0.6,
    )
    assert chat_evidence_to_tension(bare, SDM) is None

    unmapped = AutonomyEvidenceRefV1(
        evidence_id="h3",
        source="social_bridge",
        kind="relational_signal",
        summary="peer_targeted_elsewhere",
        signal_kind="chat_social_hazard",
        dimension="peer_targeted_elsewhere",
        value=1.0,
    )
    assert chat_evidence_to_tension(unmapped, SDM) is None


def test_chat_evidence_zero_value_returns_none() -> None:
    ev = AutonomyEvidenceRefV1(
        evidence_id="h4",
        source="reasoning",
        kind="reasoning_quality",
        summary="fallback",
        signal_kind="chat_reasoning_quality",
        dimension="fallback",
        value=0.0,
    )
    assert chat_evidence_to_tension(ev, SDM) is None


def test_chat_evidence_never_raises_on_garbage() -> None:
    class Boom:
        signal_kind = "chat_social_hazard"
        dimension = "cooldown_active"
        value = 1.0
        evidence_id = "x"
        summary = "x"

        @property
        def kind(self):
            raise RuntimeError("boom")

    assert chat_evidence_to_tension(Boom(), SDM) is None  # type: ignore[arg-type]
```

- [ ] **Step 2: Run to verify fail**

Run: `pytest orion/autonomy/tests/test_signal_tension.py::test_chat_evidence_mapped_hazard_mints_tension -v`  
Expected: FAIL (`ImportError` / `chat_evidence_to_tension` not defined).

- [ ] **Step 3: Implement adapter**

Add to `orion/autonomy/signal_tension.py` (near the other adapters; update module docstring to list the fourth entry point):

```python
CHAT_EVIDENCE_TENSION_KIND = "tension.chat_evidence.v1"


def chat_evidence_to_tension(
    ev: "AutonomyEvidenceRefV1",
    sdm: SignalDriveMap,
    *,
    channel: str = "orion:cortex_exec:chat_stance",
) -> Optional[TensionEventV1]:
    """Map a pressure-eligible AutonomyEvidenceRefV1 directly (no EWMA).

    Requires signal_kind + dimension + value. Unmapped / missing → None.
    Never raises.
    """
    try:
        signal_kind = getattr(ev, "signal_kind", None)
        dimension = getattr(ev, "dimension", None)
        value = getattr(ev, "value", None)
        if not signal_kind or not dimension or value is None:
            return None
        rule = sdm.match(str(signal_kind), str(dimension))
        if rule is None:
            return None
        v = _clamp01(float(value))
        if v <= 0.0:
            return None
        raw_by_drive = {d: v * w for d, w in rule.drives.items()}
        summary = (getattr(ev, "summary", None) or f"{signal_kind}.{dimension}")[:240]
        return _build_tension(
            kind=CHAT_EVIDENCE_TENSION_KIND,
            raw_by_drive=raw_by_drive,
            channel=channel,
            correlation_id=getattr(ev, "evidence_id", None),
            summary=str(summary),
        )
    except Exception:
        return None
```

Add a TYPE_CHECKING import for `AutonomyEvidenceRefV1` if needed to avoid cycles, or import it at runtime from `orion.autonomy.models` (models does not import signal_tension — runtime import is fine).

- [ ] **Step 4: Run adapter + existing tension tests**

Run: `pytest orion/autonomy/tests/test_signal_tension.py -q`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add orion/autonomy/signal_tension.py orion/autonomy/tests/test_signal_tension.py
git commit -m "$(cat <<'EOF'
feat(autonomy): add chat_evidence_to_tension direct adapter

EOF
)"
```

---

### Task 4: `AutonomyEvidenceCompiler` (proven-non-empty gates)

**Files:**
- Create: `orion/autonomy/evidence_compiler.py`
- Create: `orion/autonomy/tests/test_evidence_compiler.py`

- [ ] **Step 1: Write failing compiler tests**

```python
# orion/autonomy/tests/test_evidence_compiler.py
from __future__ import annotations

from datetime import datetime

from orion.autonomy.evidence_compiler import compile_autonomy_evidence


FIXED = datetime(2026, 7, 10, 15, 0, 0)


def test_empty_reasoning_repo_omits_reasoning_quality() -> None:
    result = compile_autonomy_evidence(
        user_message="hi",
        social={"hazards": []},
        social_bridge={"hazards": []},
        reasoning_summary={"fallback_recommended": True},
        reasoning_upstream_nonempty=False,
        autonomy_debug={"orion": {"availability": "available"}},
        now=FIXED,
    )
    kinds = [e.kind for e in result.evidence]
    assert "reasoning_quality" not in kinds
    assert any(o.get("kind") == "reasoning_quality" for o in result.omitted)
    omit = next(o for o in result.omitted if o["kind"] == "reasoning_quality")
    assert omit["reason"] == "empty_upstream"


def test_reasoning_quality_emits_only_with_upstream_and_fallback() -> None:
    result = compile_autonomy_evidence(
        user_message=None,
        social={},
        social_bridge={},
        reasoning_summary={"fallback_recommended": True},
        reasoning_upstream_nonempty=True,
        autonomy_debug={},
        now=FIXED,
    )
    rq = [e for e in result.evidence if e.kind == "reasoning_quality"]
    assert len(rq) == 1
    assert rq[0].signal_kind == "chat_reasoning_quality"
    assert rq[0].dimension == "fallback"
    assert rq[0].value == 1.0
    assert rq[0].observed_at == FIXED


def test_hazards_from_social_locals_not_ctx() -> None:
    result = compile_autonomy_evidence(
        user_message="x",
        social={"hazards": ["cooldown_active", "context_excluded:memory"]},
        social_bridge={"hazards": ["duplicate_message"]},
        reasoning_summary={"fallback_recommended": False},
        reasoning_upstream_nonempty=False,
        autonomy_debug={"orion": {"availability": "degraded"}},
        now=FIXED,
    )
    rel = [e for e in result.evidence if e.kind == "relational_signal"]
    summaries = {e.summary for e in rel}
    assert "cooldown_active" in summaries
    assert "duplicate_message" in summaries
    assert "context_excluded:memory" in summaries

    mapped = {e.summary: e for e in rel}
    assert mapped["cooldown_active"].signal_kind == "chat_social_hazard"
    assert mapped["cooldown_active"].dimension == "cooldown_active"
    assert mapped["cooldown_active"].value == 1.0
    # Unmapped prefix hazard is audit-only (no pressure fields).
    assert mapped["context_excluded:memory"].signal_kind is None
    assert mapped["context_excluded:memory"].dimension is None

    infra = [e for e in result.evidence if e.kind == "infra_health"]
    assert len(infra) == 1
    assert infra[0].observed_at == FIXED
    assert all(e.observed_at == FIXED for e in result.evidence)


def test_user_turn_and_infra_emitted_without_pressure_fields() -> None:
    result = compile_autonomy_evidence(
        user_message="hello there",
        social={},
        social_bridge={},
        reasoning_summary={},
        reasoning_upstream_nonempty=False,
        autonomy_debug={"orion": {"availability": "available"}},
        now=FIXED,
    )
    user = next(e for e in result.evidence if e.kind == "user_turn")
    assert user.signal_kind is None
    assert user.source == "user_message"
    infra = next(e for e in result.evidence if e.kind == "infra_health")
    assert infra.signal_kind is None


def test_infra_omitted_when_availability_unknown() -> None:
    result = compile_autonomy_evidence(
        user_message=None,
        social={},
        social_bridge={},
        reasoning_summary={},
        reasoning_upstream_nonempty=False,
        autonomy_debug={"orion": {"availability": "weird"}},
        now=FIXED,
    )
    assert not any(e.kind == "infra_health" for e in result.evidence)
    assert any(o.get("reason") == "availability_not_recognized" for o in result.omitted)


def test_compiler_never_raises() -> None:
    result = compile_autonomy_evidence(
        user_message=object(),  # type: ignore[arg-type]
        social="bad",  # type: ignore[arg-type]
        social_bridge=None,
        reasoning_summary=None,
        reasoning_upstream_nonempty=True,
        autonomy_debug=None,
        now=FIXED,
    )
    assert isinstance(result.evidence, list)
```

- [ ] **Step 2: Run to verify fail**

Run: `pytest orion/autonomy/tests/test_evidence_compiler.py -v`  
Expected: FAIL (`ModuleNotFoundError: orion.autonomy.evidence_compiler`).

- [ ] **Step 3: Implement compiler**

Create `orion/autonomy/evidence_compiler.py`:

```python
"""Compile turn-local AutonomyStateV2 evidence with proven-non-empty gates.

Must be called with explicit locals (social / social_bridge / reasoning flags).
Never reads ctx["chat_social_bridge_summary"] — that key is written after the
reducer runs on the live chat path.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from orion.autonomy.models import AutonomyEvidenceRefV1

_INFRA_AVAIL = frozenset({"available", "degraded", "empty", "unavailable"})
_MAPPED_SOCIAL_HAZARDS = frozenset(
    {"cooldown_active", "duplicate_message", "self_message_loop"}
)

# Kind-literal confidences (uncalibrated v1 constants).
_CONF_USER = 0.9
_CONF_INFRA = 1.0
_CONF_REASONING = 0.6
_CONF_RELATIONAL = 0.6


@dataclass
class AutonomyEvidenceCompileResult:
    evidence: list[AutonomyEvidenceRefV1] = field(default_factory=list)
    omitted: list[dict[str, str]] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)


def _unique_hazards(*groups: Any) -> list[str]:
    out: list[str] = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        raw = group.get("hazards") or []
        if not isinstance(raw, (list, tuple)):
            continue
        for h in raw:
            s = str(h).strip()
            if s and s not in out:
                out.append(s)
    return out


def compile_autonomy_evidence(
    *,
    user_message: Any,
    social: Any,
    social_bridge: Any,
    reasoning_summary: Any,
    reasoning_upstream_nonempty: bool,
    autonomy_debug: Any,
    now: datetime,
) -> AutonomyEvidenceCompileResult:
    """Emit evidence only when upstream is proven non-empty. Never raises."""
    result = AutonomyEvidenceCompileResult()
    try:
        msg = ""
        try:
            msg = str(user_message or "").strip()
        except Exception:
            msg = ""
        if msg:
            digest = hashlib.sha256(msg[:200].encode()).hexdigest()[:16]
            result.evidence.append(
                AutonomyEvidenceRefV1(
                    evidence_id=f"user_turn:{digest}",
                    source="user_message",
                    kind="user_turn",
                    summary=msg[:200],
                    confidence=_CONF_USER,
                    observed_at=now,
                )
            )
        else:
            result.omitted.append({"kind": "user_turn", "reason": "empty_message"})

        debug = autonomy_debug if isinstance(autonomy_debug, dict) else {}
        orion_dbg = debug.get("orion") if isinstance(debug.get("orion"), dict) else {}
        avail = str(orion_dbg.get("availability") or "").strip()
        if avail in _INFRA_AVAIL:
            result.evidence.append(
                AutonomyEvidenceRefV1(
                    evidence_id=f"infra_health:autonomy_graph:{avail}",
                    source="infra",
                    kind="infra_health",
                    summary=f"autonomy graph availability={avail}",
                    confidence=_CONF_INFRA,
                    observed_at=now,
                )
            )
        elif avail:
            result.omitted.append(
                {"kind": "infra_health", "reason": "availability_not_recognized"}
            )
        else:
            result.omitted.append({"kind": "infra_health", "reason": "missing_availability"})

        rs = reasoning_summary if isinstance(reasoning_summary, dict) else {}
        fallback = bool(rs.get("fallback_recommended"))
        if not reasoning_upstream_nonempty:
            result.omitted.append({"kind": "reasoning_quality", "reason": "empty_upstream"})
        elif not fallback:
            result.omitted.append(
                {"kind": "reasoning_quality", "reason": "no_quality_signal"}
            )
        else:
            result.evidence.append(
                AutonomyEvidenceRefV1(
                    evidence_id="reasoning:fallback_recommended",
                    source="reasoning",
                    kind="reasoning_quality",
                    summary="reasoning fallback recommended",
                    confidence=_CONF_REASONING,
                    observed_at=now,
                    signal_kind="chat_reasoning_quality",
                    dimension="fallback",
                    value=1.0,
                )
            )

        hazards = _unique_hazards(social, social_bridge)
        if not hazards:
            result.omitted.append({"kind": "relational_signal", "reason": "no_hazards"})
        for hazard in hazards:
            hid = hashlib.sha256(hazard[:80].encode()).hexdigest()[:12]
            mapped = hazard in _MAPPED_SOCIAL_HAZARDS
            result.evidence.append(
                AutonomyEvidenceRefV1(
                    evidence_id=f"social_bridge:{hid}",
                    source="social_bridge",
                    kind="relational_signal",
                    summary=hazard[:200],
                    confidence=_CONF_RELATIONAL,
                    observed_at=now,
                    signal_kind="chat_social_hazard" if mapped else None,
                    dimension=hazard if mapped else None,
                    value=1.0 if mapped else None,
                )
            )

        result.debug = {
            "emitted_kinds": [e.kind for e in result.evidence],
            "omitted": list(result.omitted),
            "hazard_count": len(hazards),
            "reasoning_upstream_nonempty": bool(reasoning_upstream_nonempty),
        }
    except Exception as exc:
        result.omitted.append({"kind": "_compiler", "reason": f"degraded:{type(exc).__name__}"})
        result.debug = {"error": type(exc).__name__}
    return result
```

- [ ] **Step 4: Run compiler tests**

Run: `pytest orion/autonomy/tests/test_evidence_compiler.py -q`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add orion/autonomy/evidence_compiler.py orion/autonomy/tests/test_evidence_compiler.py
git commit -m "$(cat <<'EOF'
feat(autonomy): add AutonomyEvidenceCompiler with omit-when-empty gates

EOF
)"
```

---

### Task 5: Reducer — tension fold; delete keyword cathedral

**Files:**
- Modify: `orion/autonomy/reducer.py`
- Modify: `orion/autonomy/tests/test_autonomy_reducer.py`

- [ ] **Step 1: Rewrite / add failing reducer tests**

Replace keyword-era tests in `orion/autonomy/tests/test_autonomy_reducer.py` as follows.

**Delete or rewrite** these existing tests (they encode the banned keyword path):

- `test_reducer_capability_timeout_unavailable`
- `test_reducer_polarity_blind_no_contradiction_still_raises_coherence`
- `test_reducer_determinism_fixed_now` (uses `"regression detected"` prose)

Add / replace with:

```python
from orion.autonomy.signal_drive_map import load_signal_drive_map


def test_reducer_mapped_hazard_moves_relational_pressure() -> None:
    prior = _base_v2(drive_pressures={"relational": 0.05, "coherence": 0.1})
    fixed = datetime(2026, 5, 2, 12, 0, 0)
    r = reduce_autonomy_state(
        AutonomyReducerInputV1(
            subject="orion",
            previous_state=prior,
            evidence=[
                AutonomyEvidenceRefV1(
                    evidence_id="h1",
                    source="social_bridge",
                    kind="relational_signal",
                    summary="cooldown_active",
                    confidence=0.6,
                    observed_at=fixed,
                    signal_kind="chat_social_hazard",
                    dimension="cooldown_active",
                    value=1.0,
                )
            ],
            action_outcomes=[],
            now=fixed,
        )
    )
    assert r.state.drive_pressures["relational"] > 0.05
    assert abs(r.delta.drive_deltas.get("relational", 0.0)) > 0.0
    assert "latest_direct_evidence_at" in r.state.freshness


def test_reducer_unmapped_hazard_does_not_move_pressures() -> None:
    prior = _base_v2(drive_pressures={"relational": 0.2, "coherence": 0.2})
    fixed = datetime(2026, 5, 2, 12, 0, 0)
    before = dict(prior.drive_pressures)
    r = reduce_autonomy_state(
        AutonomyReducerInputV1(
            subject="orion",
            previous_state=prior,
            evidence=[
                AutonomyEvidenceRefV1(
                    evidence_id="h1",
                    source="social_bridge",
                    kind="relational_signal",
                    summary="context_excluded:memory",
                    confidence=0.6,
                    observed_at=fixed,
                )
            ],
            action_outcomes=[],
            now=fixed,
        )
    )
    for k in before:
        assert r.state.drive_pressures.get(k, 0.0) == before.get(k, 0.0)


def test_reducer_prose_keywords_no_longer_move_pressures() -> None:
    """Acceptance: keyword tokens must not move pressures."""
    prior = _base_v2(drive_pressures={"coherence": 0.1, "capability": 0.1, "relational": 0.1})
    fixed = datetime(2026, 5, 2, 12, 0, 0)
    r = reduce_autonomy_state(
        AutonomyReducerInputV1(
            subject="orion",
            previous_state=prior,
            evidence=[
                AutonomyEvidenceRefV1(
                    evidence_id="e1",
                    source="graph",
                    kind="note",
                    summary="frustration repair contradiction timeout unavailable stale",
                    confidence=0.8,
                    observed_at=fixed,
                )
            ],
            action_outcomes=[],
            now=fixed,
        )
    )
    assert r.state.drive_pressures["coherence"] == 0.1
    assert r.state.drive_pressures["capability"] == 0.1
    assert r.state.drive_pressures["relational"] == 0.1
    # Tension kinds must not OR in from the prose blob either.
    assert "tension.coherence_break.v1" not in r.state.tension_kinds
    assert "tension.capability_gap.v1" not in r.state.tension_kinds
    assert "tension.relational_repair.v1" not in r.state.tension_kinds


def test_reducer_reasoning_fallback_moves_coherence() -> None:
    prior = _base_v2(drive_pressures={"coherence": 0.05, "predictive": 0.05})
    fixed = datetime(2026, 5, 2, 12, 0, 0)
    r = reduce_autonomy_state(
        AutonomyReducerInputV1(
            subject="orion",
            previous_state=prior,
            evidence=[
                AutonomyEvidenceRefV1(
                    evidence_id="reasoning:fallback_recommended",
                    source="reasoning",
                    kind="reasoning_quality",
                    summary="reasoning fallback recommended",
                    confidence=0.6,
                    observed_at=fixed,
                    signal_kind="chat_reasoning_quality",
                    dimension="fallback",
                    value=1.0,
                )
            ],
            action_outcomes=[],
            now=fixed,
        )
    )
    assert r.state.drive_pressures["coherence"] > 0.05
    assert r.state.drive_pressures["predictive"] > 0.05


def test_reducer_tension_kinds_from_pressure_thresholds_only() -> None:
    prior = _base_v2(drive_pressures={"coherence": 0.30})
    fixed = datetime(2026, 5, 2, 12, 0, 0)
    # No incoming evidence; prior pressure already at threshold.
    r = reduce_autonomy_state(
        AutonomyReducerInputV1(
            subject="orion",
            previous_state=prior,
            evidence=[],
            action_outcomes=[],
            now=fixed,
        )
    )
    assert "tension.coherence_break.v1" in r.state.tension_kinds


def test_reducer_no_keyword_helpers_in_module() -> None:
    import inspect
    from orion.autonomy import reducer as reducer_mod

    src = inspect.getsource(reducer_mod)
    for banned in (
        '_apply_single_evidence_pressures',
        '"contradiction"',
        '"frustration"',
        '"stale" in blob',
        '"timeout" in blob',
    ):
        assert banned not in src, banned


def test_reducer_determinism_fixed_now_typed() -> None:
    fixed = datetime(2026, 5, 2, 12, 0, 0)
    prior = _base_v2(drive_pressures={"relational": 0.0})
    inp = AutonomyReducerInputV1(
        subject="orion",
        previous_state=prior,
        evidence=[
            AutonomyEvidenceRefV1(
                evidence_id="z1",
                source="social_bridge",
                kind="relational_signal",
                summary="self_message_loop",
                confidence=0.6,
                observed_at=fixed,
                signal_kind="chat_social_hazard",
                dimension="self_message_loop",
                value=1.0,
            )
        ],
        action_outcomes=[],
        now=fixed,
    )
    a = reduce_autonomy_state(inp).state.model_dump(mode="json")
    b = reduce_autonomy_state(inp).state.model_dump(mode="json")
    assert a == b
```

Keep existing tests that still apply: cold start, user/infra no pressure, proxy inhibition, surprise confidence, evidence trim, no_fresh_evidence.

- [ ] **Step 2: Run to verify new tests fail / old keyword tests fail against new intent**

Run: `pytest orion/autonomy/tests/test_autonomy_reducer.py::test_reducer_mapped_hazard_moves_relational_pressure orion/autonomy/tests/test_autonomy_reducer.py::test_reducer_prose_keywords_no_longer_move_pressures -v`  
Expected: FAIL (mapped hazard does not move yet; prose still moves until keyword path deleted).

- [ ] **Step 3: Implement reducer changes**

In `orion/autonomy/reducer.py`:

1. Add imports:

```python
from orion.autonomy.signal_drive_map import SignalDriveMap, load_signal_drive_map
from orion.autonomy.signal_tension import chat_evidence_to_tension
from orion.core.schemas.drives import TensionEventV1
```

2. **Delete** `_evidence_text`, `_apply_single_evidence_pressures`, `_combined_evidence_text`.

3. Replace pressure application + tension derivation + confidence blob penalties:

```python
_MAX_PRESSURE_STEP = 0.15


def _fold_tension_into_pressures(
    pressures: dict[str, float], tension: TensionEventV1
) -> dict[str, float]:
    out = dict(pressures)
    mag = float(tension.magnitude or 0.0)
    for drive, impact in (tension.drive_impacts or {}).items():
        if drive not in _DRIVE_KEYS:
            continue
        added = min(_MAX_PRESSURE_STEP, mag * float(impact or 0.0))
        if added > 0.0:
            out[drive] = min(1.0, out[drive] + added)
    return out


def _derive_tension_kinds(pressures: dict[str, float]) -> list[str]:
    kinds: list[str] = []

    def add(name: str) -> None:
        if name not in kinds:
            kinds.append(name)

    if pressures.get("coherence", 0.0) >= 0.25:
        add("tension.coherence_break.v1")
    if pressures.get("continuity", 0.0) >= 0.25:
        add("tension.continuity_gap.v1")
    if pressures.get("capability", 0.0) >= 0.25:
        add("tension.capability_gap.v1")
    if pressures.get("relational", 0.0) >= 0.25:
        add("tension.relational_repair.v1")

    ranked = sorted(_DRIVE_KEYS, key=lambda k: pressures.get(k, 0.0), reverse=True)
    if len(ranked) >= 2:
        p1 = pressures.get(ranked[0], 0.0)
        p2 = pressures.get(ranked[1], 0.0)
        if p1 >= 0.25 and p2 >= 0.25 and abs(p1 - p2) < 0.08:
            add("tension.drive_competition.v1")
    return kinds


def _infra_unavailable(evidence: list[AutonomyEvidenceRefV1]) -> bool:
    for ev in evidence:
        if ev.kind != "infra_health":
            continue
        summary = (ev.summary or "").lower()
        if "availability=unavailable" in summary or "availability=degraded" in summary:
            return True
    return False
```

4. In `reduce_autonomy_state`, replace the pressure loop and blob usage:

```python
    pressures = _normalize_pressures(dict(working.drive_pressures))
    prev_dom = working.dominant_drive

    sdm: SignalDriveMap = load_signal_drive_map()
    minted: list[dict[str, Any]] = []
    for ev in inp.evidence:
        # user_turn / infra_health are never pressure-eligible (no signal fields).
        tension = chat_evidence_to_tension(ev, sdm)
        if tension is None:
            continue
        pressures = _fold_tension_into_pressures(pressures, tension)
        minted.append(
            {
                "kind": tension.kind,
                "signal_kind": ev.signal_kind,
                "dimension": ev.dimension,
                "drives": sorted((tension.drive_impacts or {}).keys()),
            }
        )

    working.drive_pressures = {k: pressures[k] for k in _DRIVE_KEYS}
    dominant, active_drives = _dominant_and_active(pressures, prev_dom)
    working.dominant_drive = dominant
    working.active_drives = active_drives
    working.tension_kinds = _derive_tension_kinds(pressures)
```

5. Confidence: **remove** these lines:

```python
    if "stale" in blob or "missing context" in blob:
        conf -= 0.05
    if "timeout" in blob or "unavailable" in blob:
        conf -= 0.05
```

Optional typed replacement (allowed by spec):

```python
    if _infra_unavailable(working.evidence_refs):
        conf -= 0.05
```

6. Inhibition `dependency_unavailable`: replace blob check with:

```python
            elif c.kind == "triage_capability_gap" and pressures.get("capability", 0.0) >= 0.35:
                if _infra_unavailable(working.evidence_refs):
                    reason = "dependency_unavailable"
```

7. Attach mint debug onto delta notes or leave for stance debug (stance Task 6 records compiler + mint debug in ctx). Optionally append a short note:

```python
        notes=[
            "delta compares against upgraded V1 or copied V2 baseline at turn start",
            "changed_fields are surface diffs vs that baseline, not a persisted prior-turn snapshot",
            f"tensions_minted={len(minted)}",
        ],
```

Do **not** call `SignalTensionSource`, DeviationGate, or `DriveEngine`.

- [ ] **Step 4: Run reducer tests**

Run: `pytest orion/autonomy/tests/test_autonomy_reducer.py -q`  
Expected: PASS

Also confirm keyword helpers are gone:

Run: `rg -n "_apply_single_evidence_pressures|frustration|contradiction" orion/autonomy/reducer.py`  
Expected: no pressure-path keyword hits (comments about deletion are fine; prefer zero hits).

- [ ] **Step 5: Commit**

```bash
git add orion/autonomy/reducer.py orion/autonomy/tests/test_autonomy_reducer.py
git commit -m "$(cat <<'EOF'
feat(autonomy): fold chat tensions into drive_pressures; drop keyword path

EOF
)"
```

---

### Task 6: Stance wiring — locals before ctx write

**Files:**
- Modify: `services/orion-cortex-exec/app/chat_stance.py`
- Modify: `services/orion-cortex-exec/tests/test_chat_stance_autonomy_v2.py`

- [ ] **Step 1: Write failing stance tests**

Append to `services/orion-cortex-exec/tests/test_chat_stance_autonomy_v2.py`:

```python
from datetime import datetime, timezone

from orion.core.schemas.reasoning import ClaimV1
from orion.core.schemas.reasoning_io import ReasoningWriteContextV1, ReasoningWriteRequestV1
from orion.reasoning import InMemoryReasoningRepository


def _claim() -> ClaimV1:
    return ClaimV1(
        anchor_scope="orion",
        subject_ref="project:orion_sapienform",
        status="canonical",
        authority="local_inferred",
        confidence=0.9,
        salience=0.8,
        novelty=0.4,
        risk_tier="low",
        observed_at=datetime.now(timezone.utc),
        provenance={
            "evidence_refs": ["ev:1"],
            "source_channel": "orion:test",
            "source_kind": "unit",
            "producer": "pytest",
        },
        claim_text="Reasoning continuity is strong.",
        claim_kind="identity_signal",
    )


def test_chat_stance_empty_repo_omits_reasoning_quality_evidence(monkeypatch) -> None:
    state = AutonomyStateV1(
        subject="orion",
        model_layer="self-model",
        entity_id="self:orion",
        dominant_drive="coherence",
        drive_pressures={"coherence": 0.05},
        active_drives=["coherence"],
        tension_kinds=[],
        goal_headlines=[],
        source="graph",
    )
    monkeypatch.setattr(chat_stance, "_load_autonomy_state", lambda _ctx: _fake_autonomy_bundle(state))
    monkeypatch.setenv("AUTONOMY_STATE_V2_REDUCER_ENABLED", "true")
    ctx: dict = {"user_message": "hello", "correlation_id": "c-omit"}
    chat_stance.build_chat_stance_inputs(ctx)
    debug = ctx.get("chat_autonomy_evidence_debug") or {}
    omitted = debug.get("omitted") or []
    assert any(o.get("kind") == "reasoning_quality" and o.get("reason") == "empty_upstream" for o in omitted)
    v2 = ctx.get("chat_autonomy_state_v2") or {}
    kinds = [e.get("kind") for e in (v2.get("evidence_refs") or [])]
    assert "reasoning_quality" not in kinds


def test_chat_stance_social_locals_reach_reducer(monkeypatch) -> None:
    state = AutonomyStateV1(
        subject="orion",
        model_layer="self-model",
        entity_id="self:orion",
        dominant_drive=None,
        drive_pressures={"relational": 0.0, "coherence": 0.0},
        active_drives=[],
        tension_kinds=[],
        goal_headlines=[],
        source="graph",
    )
    monkeypatch.setattr(chat_stance, "_load_autonomy_state", lambda _ctx: _fake_autonomy_bundle(state))

    def _fake_social(_beliefs, _ctx):
        return (
            {"social_posture": [], "hazards": ["cooldown_active"], "relationship_facets": []},
            {"posture": [], "hazards": ["cooldown_active"], "framing": [], "summary": []},
        )

    monkeypatch.setattr(chat_stance, "_project_social_from_beliefs", _fake_social)
    monkeypatch.setenv("AUTONOMY_STATE_V2_REDUCER_ENABLED", "true")
    ctx: dict = {"user_message": "ping", "correlation_id": "c-haz"}
    # Intentionally do NOT pre-seed chat_social_bridge_summary — ordering bug regression.
    assert "chat_social_bridge_summary" not in ctx
    chat_stance.build_chat_stance_inputs(ctx)
    v2 = ctx["chat_autonomy_state_v2"]
    summaries = [e.get("summary") for e in (v2.get("evidence_refs") or []) if e.get("kind") == "relational_signal"]
    assert "cooldown_active" in summaries
    assert v2["drive_pressures"]["relational"] > 0.0
    assert isinstance(ctx.get("chat_autonomy_evidence_debug"), dict)
    assert isinstance(ctx.get("chat_autonomy_tension_debug"), dict)


def test_chat_stance_reasoning_upstream_emits_quality(monkeypatch) -> None:
    state = AutonomyStateV1(
        subject="orion",
        model_layer="self-model",
        entity_id="self:orion",
        dominant_drive="coherence",
        drive_pressures={"coherence": 0.05, "predictive": 0.05},
        active_drives=["coherence"],
        tension_kinds=[],
        goal_headlines=[],
        source="graph",
    )
    monkeypatch.setattr(chat_stance, "_load_autonomy_state", lambda _ctx: _fake_autonomy_bundle(state))
    monkeypatch.setenv("AUTONOMY_STATE_V2_REDUCER_ENABLED", "true")

    repo = InMemoryReasoningRepository()
    # Empty-ish path that still has an artifact but compiler may still recommend fallback
    # depending on subject_refs — force fallback via monkeypatch after compile if needed.
    repo.write_artifacts(
        ReasoningWriteRequestV1(
            context=ReasoningWriteContextV1(
                source_family="manual",
                source_kind="unit",
                source_channel="orion:test",
                producer="pytest",
            ),
            artifacts=[_claim()],
        )
    )

    original_compile = chat_stance._compile_reasoning_summary

    def _compile_force_fallback(ctx):
        out = original_compile(ctx)
        summary = dict(out.get("summary") or {})
        summary["fallback_recommended"] = True
        out = dict(out)
        out["summary"] = summary
        return out

    monkeypatch.setattr(chat_stance, "_compile_reasoning_summary", _compile_force_fallback)
    ctx: dict = {"user_message": "who?", "reasoning_repository": repo}
    chat_stance.build_chat_stance_inputs(ctx)
    v2 = ctx["chat_autonomy_state_v2"]
    kinds = [e.get("kind") for e in (v2.get("evidence_refs") or [])]
    assert "reasoning_quality" in kinds
    assert v2["drive_pressures"]["coherence"] > 0.05
```

- [ ] **Step 2: Run to verify fail**

Run: `PYTHONPATH=services/orion-cortex-exec:services/orion-cortex-exec/app:. pytest services/orion-cortex-exec/tests/test_chat_stance_autonomy_v2.py::test_chat_stance_social_locals_reach_reducer -v`  
(Use the service’s normal pytest invocation if a conftest already sets `PYTHONPATH`.)  
Expected: FAIL (`chat_autonomy_evidence_debug` missing and/or relational pressure still 0 because evidence builder still reads empty ctx).

- [ ] **Step 3: Wire stance**

In `services/orion-cortex-exec/app/chat_stance.py`:

1. Replace import of only `AutonomyEvidenceRefV1` usage for the old builder — add:

```python
from orion.autonomy.evidence_compiler import compile_autonomy_evidence
from orion.autonomy.signal_drive_map import load_signal_drive_map
from orion.autonomy.signal_tension import chat_evidence_to_tension
```

2. Add helper:

```python
def _reasoning_upstream_nonempty(ctx: Dict[str, Any]) -> bool:
    raw = ctx.get("reasoning_artifacts")
    if isinstance(raw, list) and raw:
        return True
    repo = ctx.get("reasoning_repository")
    if repo is None:
        return False
    try:
        latest = repo.list_latest(limit=1)
        return bool(latest)
    except Exception:
        return False
```

3. **Delete** `_build_autonomy_reducer_evidence`.

4. Replace `_run_autonomy_reducer` with a locals-aware version:

```python
def _run_autonomy_reducer(
    ctx: Dict[str, Any],
    autonomy: Dict[str, Any],
    *,
    social: Dict[str, Any],
    social_bridge: Dict[str, Any],
    reasoning: Dict[str, Any],
):
    now = datetime.utcnow()
    compile_result = compile_autonomy_evidence(
        user_message=ctx.get("user_message") or ctx.get("message") or "",
        social=social,
        social_bridge=social_bridge,
        reasoning_summary=(reasoning.get("summary") if isinstance(reasoning, dict) else {}) or {},
        reasoning_upstream_nonempty=_reasoning_upstream_nonempty(ctx),
        autonomy_debug=autonomy.get("debug") if isinstance(autonomy.get("debug"), dict) else {},
        now=now,
    )
    ctx["chat_autonomy_evidence_debug"] = {
        "emitted_kinds": [e.kind for e in compile_result.evidence],
        "omitted": list(compile_result.omitted),
        **(compile_result.debug or {}),
    }
    # Trace tensions that will be minted (same map the reducer uses).
    sdm = load_signal_drive_map()
    tension_debug = []
    for ev in compile_result.evidence:
        t = chat_evidence_to_tension(ev, sdm)
        if t is None:
            continue
        tension_debug.append(
            {
                "kind": t.kind,
                "signal_kind": ev.signal_kind,
                "dimension": ev.dimension,
                "drives": sorted((t.drive_impacts or {}).keys()),
                "magnitude": t.magnitude,
            }
        )
    ctx["chat_autonomy_tension_debug"] = {"minted": tension_debug}

    state_obj = autonomy.get("state")
    subj = getattr(state_obj, "subject", None) if state_obj is not None else None
    subject = str(subj or "orion")
    return reduce_autonomy_state(
        AutonomyReducerInputV1(
            subject=subject,
            previous_state=state_obj,
            evidence=compile_result.evidence,
            action_outcomes=load_action_outcomes(subject=subject),
            now=now,
        )
    )
```

Add `from datetime import datetime` near the top of `chat_stance.py` if missing (current file does not import it).

5. In `build_chat_stance_inputs`, change the gated call to pass locals **that already exist** at that point (`social`, `social_bridge`, `reasoning`):

```python
    if os.getenv("AUTONOMY_STATE_V2_REDUCER_ENABLED", "").strip().lower() == "true":
        try:
            before_pressures = None
            if autonomy.get("state") is not None:
                before_pressures = dict(getattr(autonomy["state"], "drive_pressures", None) or {})
            v2_result = _run_autonomy_reducer(
                ctx,
                autonomy,
                social=social,
                social_bridge=social_bridge,
                reasoning=reasoning,
            )
            ctx["chat_autonomy_state_v2"] = v2_result.state.model_dump(mode="json")
            ctx["chat_autonomy_state_delta"] = v2_result.delta.model_dump(mode="json")
            ctx["chat_autonomy_movement_debug"] = {
                "dominant_drive_before": getattr(autonomy.get("state"), "dominant_drive", None),
                "dominant_drive_after": v2_result.state.dominant_drive,
                "pressures_before": before_pressures,
                "pressures_after": dict(v2_result.state.drive_pressures or {}),
                "new_tensions": list(v2_result.delta.new_tensions or []),
                "resolved_tensions": list(v2_result.delta.resolved_tensions or []),
            }
            inputs["autonomy"]["state_v2"] = ctx["chat_autonomy_state_v2"]
            inputs["autonomy"]["delta"] = ctx["chat_autonomy_state_delta"]
        except Exception as exc:
            logger.warning("autonomy_reducer_v2_failed error=%s", exc)
```

Confirm this block still runs **before** `ctx["chat_social_bridge_summary"] = social_bridge` (today the write is ~line 2347; reducer is ~2308 — keep that order).

6. Env gate stays default-off (`AUTONOMY_STATE_V2_REDUCER_ENABLED=` empty in `.env_example`). **Do not** flip the default.

- [ ] **Step 4: Run stance autonomy tests**

Run: `pytest services/orion-cortex-exec/tests/test_chat_stance_autonomy_v2.py -q`  
Expected: PASS (including prior enable/disable/exception tests).

- [ ] **Step 5: Commit**

```bash
git add services/orion-cortex-exec/app/chat_stance.py services/orion-cortex-exec/tests/test_chat_stance_autonomy_v2.py
git commit -m "$(cat <<'EOF'
feat(cortex): wire AutonomyEvidenceCompiler locals into V2 reducer

EOF
)"
```

---

### Task 7: Movement eval (enable bar)

**Files:**
- Create: `orion/autonomy/evals/run_autonomy_v2_movement_eval.py`

- [ ] **Step 1: Write the eval (fails until Task 5–6 green; run as gate)**

```python
#!/usr/bin/env python3
"""Trace-proven AutonomyStateV2 pressure movement from typed chat evidence.

Enable bar for AUTONOMY_STATE_V2_REDUCER_ENABLED: do not flip production until
this eval exits 0.
"""
from __future__ import annotations

import sys
from datetime import datetime

from orion.autonomy.evidence_compiler import compile_autonomy_evidence
from orion.autonomy.models import AutonomyStateV2
from orion.autonomy.reducer import AutonomyReducerInputV1, reduce_autonomy_state


def _cold(**overrides: object) -> AutonomyStateV2:
    base = AutonomyStateV2(
        subject="orion",
        model_layer="self-model",
        entity_id="self:orion",
        source="eval",
        generated_at=datetime(2026, 7, 10, 12, 0, 0),
        schema_version="autonomy.state.v2",
        confidence=0.5,
        unknowns=[],
        evidence_refs=[],
        freshness={},
        attention_items=[],
        candidate_impulses=[],
        inhibited_impulses=[],
        last_action_outcomes=[],
        drive_pressures={
            "coherence": 0.05,
            "continuity": 0.05,
            "relational": 0.05,
            "autonomy": 0.05,
            "capability": 0.05,
            "predictive": 0.05,
        },
        dominant_drive=None,
        active_drives=[],
        tension_kinds=[],
    )
    return base.model_copy(update=overrides)


def main() -> int:
    fixed = datetime(2026, 7, 10, 16, 0, 0)
    compiled = compile_autonomy_evidence(
        user_message="are you looping?",
        social={"hazards": ["self_message_loop", "context_excluded:x"]},
        social_bridge={"hazards": ["cooldown_active"]},
        reasoning_summary={"fallback_recommended": True},
        reasoning_upstream_nonempty=True,
        autonomy_debug={"orion": {"availability": "available"}},
        now=fixed,
    )
    assert any(e.kind == "reasoning_quality" for e in compiled.evidence)
    assert any(e.dimension == "self_message_loop" for e in compiled.evidence)
    assert any(e.summary == "context_excluded:x" and e.signal_kind is None for e in compiled.evidence)

    baseline = _cold()
    before = dict(baseline.drive_pressures)
    before_dom = baseline.dominant_drive

    result = reduce_autonomy_state(
        AutonomyReducerInputV1(
            subject="orion",
            previous_state=baseline,
            evidence=compiled.evidence,
            action_outcomes=[],
            now=fixed,
        )
    )
    after = result.state.drive_pressures
    moved = {
        k: round(after[k] - before[k], 6)
        for k in before
        if abs(after[k] - before[k]) > 1e-9
    }
    print("omitted=", compiled.omitted)
    print("moved=", moved)
    print("dominant_before=", before_dom, "dominant_after=", result.state.dominant_drive)
    print("new_tensions=", result.delta.new_tensions)

    if not moved:
        print("FAIL: no drive_pressures movement")
        return 1
    if after.get("relational", 0.0) <= before.get("relational", 0.0):
        print("FAIL: relational did not increase from mapped hazards")
        return 1
    if after.get("coherence", 0.0) <= before.get("coherence", 0.0):
        print("FAIL: coherence did not increase from reasoning fallback")
        return 1
    # Dominant may change once pressures clear the 0.15 threshold.
    if result.state.dominant_drive is None and max(after.values()) >= 0.15:
        print("FAIL: expected dominant_drive once pressure >= 0.15")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run eval**

Run: `python orion/autonomy/evals/run_autonomy_v2_movement_eval.py`  
Expected: `PASS` / exit 0.

- [ ] **Step 3: Commit**

```bash
git add orion/autonomy/evals/run_autonomy_v2_movement_eval.py
git commit -m "$(cat <<'EOF'
test(autonomy): add AutonomyStateV2 movement eval enable bar

EOF
)"
```

---

### Task 8: Isolation check (hard ban)

**Files:**
- Create: `orion/autonomy/tests/test_autonomy_isolation.py`

- [ ] **Step 1: Write isolation test**

```python
# orion/autonomy/tests/test_autonomy_isolation.py
from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]  # orion/autonomy/tests → repo root

# Modules that must not import AutonomyStateV2 / reduce_autonomy_state.
_BANNED_ROOTS = [
    REPO / "orion" / "self_state",
    REPO / "services" / "orion-spark-introspector",
]


def _python_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [p for p in root.rglob("*.py") if p.is_file()]


def _imports_autonomy_v2(path: Path) -> list[str]:
    hits: list[str] = []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return hits
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            names = {a.name for a in node.names}
            if "AutonomyStateV2" in names or "reduce_autonomy_state" in names:
                hits.append(f"{path}: from {mod} import {sorted(names)}")
            if mod in {"orion.autonomy.models", "orion.autonomy.reducer", "orion.autonomy"}:
                if names & {"AutonomyStateV2", "reduce_autonomy_state", "AutonomyEvidenceRefV1"}:
                    hits.append(f"{path}: from {mod} import {sorted(names)}")
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name in {"orion.autonomy.reducer", "orion.autonomy.models"}:
                    hits.append(f"{path}: import {a.name}")
    return hits


def test_autonomy_state_v2_not_wired_into_phi_or_self_state() -> None:
    hits: list[str] = []
    for root in _BANNED_ROOTS:
        for path in _python_files(root):
            hits.extend(_imports_autonomy_v2(path))
    assert hits == [], "AutonomyStateV2 isolation violated:\n" + "\n".join(hits)
```

- [ ] **Step 2: Run isolation test**

Run: `pytest orion/autonomy/tests/test_autonomy_isolation.py -v`  
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add orion/autonomy/tests/test_autonomy_isolation.py
git commit -m "$(cat <<'EOF'
test(autonomy): ban AutonomyStateV2 imports into self_state/phi paths

EOF
)"
```

---

### Task 9: Operator docs

**Files:**
- Modify: `docs/autonomy_state_v2_reducer.md`

- [ ] **Step 1: Update operator notes**

Replace the “Known limitations” polarity-blind item and document the new evidence contract. Final doc body should include:

```markdown
# AutonomyStateV2 reducer (operator notes)

## What this is

An optional, **env-gated** deterministic reducer that combines graph-loaded autonomy (`AutonomyStateV1`), turn-level **typed** evidence (user message, infra availability, reasoning quality when upstream artifacts exist, social hazards from stance locals), and optional action outcomes into **`AutonomyStateV2`** plus a **`AutonomyStateDeltaV1`** for one chat turn.

Pressure math uses the shared `signal_drive_map` via `chat_evidence_to_tension` (same family as endogenous `failure_to_tension`). Keyword substring matching is **removed**.

## What this is **not**

- **Not** sentience, consciousness, or moral status.
- **Not** durable persistence: V2 exists in **request context only**.
- **Not** an input to phi features, `build_self_state`, or homeostatic `DriveEngine`.
- Empty reasoning repositories do **not** emit `reasoning_quality` theater.

## Evidence contract (omit-when-empty)

| Kind | Emit when | Moves pressures? |
|------|-----------|------------------|
| `user_turn` | non-empty user message | No |
| `infra_health` | availability ∈ {available, degraded, empty, unavailable} | No |
| `reasoning_quality` | upstream repo/artifacts non-empty **and** `fallback_recommended` | Yes (`chat_reasoning_quality`/`fallback`) |
| `relational_signal` | hazards on social/social_bridge locals | Yes only for mapped exact keys |

Mapped social dimensions (v1): `cooldown_active`, `duplicate_message`, `self_message_loop`.  
Prefix hazards (`context_excluded:*`, etc.) are audit-only unless YAML + test grow.

Confidence values on evidence are **kind-literal constants (uncalibrated)** in v1.

## Environment flag

Default **off**. Enable only after the movement eval is green:

```bash
python orion/autonomy/evals/run_autonomy_v2_movement_eval.py
# exit 0 required before considering:
AUTONOMY_STATE_V2_REDUCER_ENABLED=true
```

When unset or not `true`, cortex skips the reducer entirely.

## Debug keys (when flag on)

- `ctx["chat_autonomy_evidence_debug"]` — emitted/omitted kinds + reasons
- `ctx["chat_autonomy_tension_debug"]` — minted tensions
- `ctx["chat_autonomy_movement_debug"]` — pressures / dominant_drive before vs after

## Known limitations

1. **No durable state** — each turn rebuilds prior state from graph V1; delta is relative to the upgrade baseline at turn start.
2. **Sparse map** — unmapped hazards record evidence but do not move pressures (preferred over fake motion).
3. **Dual pipelines** — chat reducer and endogenous tick both use `signal_drive_map` helpers but chat does not feed `DriveEngine`.
```

- [ ] **Step 2: Commit**

```bash
git add docs/autonomy_state_v2_reducer.md
git commit -m "$(cat <<'EOF'
docs(autonomy): document typed evidence contract and enable bar

EOF
)"
```

---

### Task 10: Full gate + self-check

- [ ] **Step 1: Run focused suite**

```bash
pytest orion/autonomy/tests/test_evidence_ref_schema.py \
  orion/autonomy/tests/test_evidence_compiler.py \
  orion/autonomy/tests/test_signal_drive_map.py \
  orion/autonomy/tests/test_signal_tension.py \
  orion/autonomy/tests/test_autonomy_reducer.py \
  orion/autonomy/tests/test_autonomy_isolation.py \
  services/orion-cortex-exec/tests/test_chat_stance_autonomy_v2.py -q

python orion/autonomy/evals/run_autonomy_v2_movement_eval.py
```

Expected: all green; eval prints `PASS`.

- [ ] **Step 2: Grep guards**

```bash
rg -n "_build_autonomy_reducer_evidence|_apply_single_evidence_pressures" \
  orion/autonomy services/orion-cortex-exec/app/chat_stance.py

rg -n 'hit\(\(|"frustration"|"contradiction".*coherence' orion/autonomy/reducer.py

rg -n "AUTONOMY_STATE_V2_REDUCER_ENABLED" services/orion-cortex-exec/.env_example
```

Expected: no old builder/keyword pressure helpers; `.env_example` still has empty / default-off value.

- [ ] **Step 3: Confirm no env template key changes** (this patch should not add keys). If any `.env_example` comment-only edit happened, run:

```bash
python scripts/sync_local_env_from_example.py
```

- [ ] **Step 4: Final commit if any leftover doc/test fixes**

```bash
git status --short
git diff --check
```

---

## Self-review (plan vs spec)

| Spec acceptance | Task |
|-----------------|------|
| 1. Empty reasoning repo does not emit `reasoning_quality` | Task 4 + Task 6 |
| 2. Hazards from social/social_bridge locals | Task 4 + Task 6 |
| 3. Mapped hazard bumps drives; unmapped does not | Task 2 + Task 5 + Task 7 |
| 4. Keyword tokens gone from pressure / tension_kinds / confidence blob | Task 5 |
| 5. `observed_at` → freshness | Task 4 + Task 5 |
| 6. Movement fixture/eval with flag-on proof path | Task 7 (+ Task 6 tests) |
| 7. Isolation AutonomyStateV2 ↛ phi/self-state | Task 8 |
| 8. Default env remains disabled; operator doc | Task 6 + Task 9 |

**Placeholder scan:** none intentional — all steps include concrete code/commands.  
**Type consistency:** `compile_autonomy_evidence(...)`, `chat_evidence_to_tension(ev, sdm)`, optional fields `signal_kind` / `dimension` / `value` / `observed_at` used uniformly.  
**Hard ban:** Tasks 5–6 explicitly forbid DriveEngine / DeviationGate / SelfState wiring; Task 8 enforces.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-10-autonomy-v2-evidence-signal-tension.md`. Two execution options:

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
