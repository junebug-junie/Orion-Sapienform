# Repair Pressure v2 + Pre-Turn Appraisal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace phrase-match repair pressure with a same-turn pre-turn appraisal RPC (Hub → cortex-exec) that scores seven evidence kinds via logprob probes and attaches `repair_pressure_contract` metadata before chat.

**Architecture:** Hub builds a paired `turn_window` from `messages[]` and calls `orion:cortex:pre_turn_appraisal:request`; cortex-exec runs `logprob_probe_v2` via LLM gateway, assembles `TurnAppraisalBundleV1`, and returns `metadata_attachments`; Hub copies attachments onto `CortexChatRequest.metadata` (speech wiring unchanged). Legacy `phrase_match_v1` pipeline stays behind `ENABLE_REPAIR_PRESSURE_V2=false` for rollback.

**Tech Stack:** Python 3.12, Pydantic v2, pytest, Redis bus RPC (`OrionBusAsync.rpc_request`), existing LLM gateway logprob rails (`return_logprobs`, `logprob_probe_mode=native_completion`).

**Spec:** `docs/superpowers/specs/2026-07-03-repair-pressure-v2-pre-turn-appraisal-design.md`

---

## File map

| File | Role |
|------|------|
| `orion/schemas/pre_turn_appraisal.py` | **Create** — `PreTurnAppraisalRequestV1`, `TurnAppraisalBundleV1`, paradigm slice models |
| `orion/bus/channels.yaml` | Add pre_turn_appraisal request/result channels |
| `orion/schemas/registry.py` | Register new schema_ids |
| `orion/substrate/appraisal/turn_window.py` | **Create** — `messages[]` → bounded role-tagged window |
| `orion/substrate/appraisal/probe/logprob_runner.py` | **Create** — YES/NO parse + sigmoid score + margin confidence |
| `orion/substrate/appraisal/paradigms/base.py` | **Create** — `AppraisalParadigm` protocol |
| `orion/substrate/appraisal/paradigms/registry.py` | **Create** — explicit `PARADIGM_REGISTRY` dict |
| `orion/substrate/appraisal/paradigms/repair_pressure_v2.py` | **Create** — probe template, appraise, contract_delta |
| `orion/substrate/appraisal/contract.py` | Add kind-aware v2 rule assembly + `assemble_repair_contract_delta` |
| `config/substrate/repair_pressure_weights.v2.yaml` | **Create** — eval-calibrated kind weights |
| `services/orion-cortex-exec/app/pre_turn_appraisal.py` | **Create** — RPC handler + paradigm dispatch |
| `services/orion-cortex-exec/app/main.py` | Second Rabbit listener for pre_turn_appraisal channel |
| `services/orion-cortex-exec/app/settings.py` | v2 flags + probe route + weights path |
| `services/orion-cortex-exec/.env_example` | Document new keys |
| `services/orion-hub/scripts/pre_turn_appraisal_client.py` | **Create** — bus RPC client |
| `services/orion-hub/scripts/pre_turn_appraisal_wiring.py` | **Create** — attach bundle → chat req + chip summary |
| `services/orion-hub/scripts/api_routes.py` | Call v2 wiring before cortex when enabled |
| `services/orion-hub/scripts/websocket_handler.py` | Same |
| `services/orion-hub/scripts/grammar_emit.py` | Read `grammar_scalars` from bundle |
| `services/orion-hub/app/settings.py` | Hub pre-turn appraisal flags |
| `services/orion-hub/.env_example` | Document new keys |
| `orion/substrate/evals/repair_pressure_v2_eval.py` | **Create** — transcript fixture harness |
| `orion/substrate/evals/fixtures/repair_pressure/*.json` | **Create** — positive/negative/neutral threads |
| `tests/test_logprob_probe_runner.py` | **Create** — pure logprob scoring tests |
| `tests/test_repair_pressure_v2_paradigm.py` | **Create** — paradigm unit tests (mocked LLM) |
| `services/orion-cortex-exec/tests/test_pre_turn_appraisal_rpc.py` | **Create** — handler tests |
| `services/orion-hub/tests/test_pre_turn_appraisal_wiring.py` | **Create** — Hub attach tests |

---

## Task 1 — Schemas, bus channels, registry

**Files:**
- Create: `orion/schemas/pre_turn_appraisal.py`
- Modify: `orion/bus/channels.yaml`
- Modify: `orion/schemas/registry.py`
- Test: `tests/test_pre_turn_appraisal_schemas.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_pre_turn_appraisal_schemas.py`:

```python
from __future__ import annotations

from orion.schemas.pre_turn_appraisal import (
    PreTurnAppraisalRequestV1,
    TurnAppraisalBundleV1,
    TurnAppraisalParadigmSliceV1,
    TurnWindowMessageV1,
)


def test_request_round_trip_minimal() -> None:
    req = PreTurnAppraisalRequestV1(
        correlation_id="corr-1",
        session_id="sess-1",
        turn_window=[
            TurnWindowMessageV1(role="user", content="give me nuts and bolts"),
            TurnWindowMessageV1(role="assistant", content="here is a high level plan"),
        ],
        paradigms_requested=["repair_pressure"],
        contract_before={"mode": "default"},
    )
    data = req.model_dump(mode="json")
    assert PreTurnAppraisalRequestV1.model_validate(data).correlation_id == "corr-1"


def test_bundle_carries_metadata_attachments() -> None:
    bundle = TurnAppraisalBundleV1(
        correlation_id="corr-1",
        paradigms={
            "repair_pressure": TurnAppraisalParadigmSliceV1(
                appraisal_kind="repair_pressure",
                level=0.82,
                confidence=0.71,
                dimensions={"specificity_demand": 0.91},
                evidence=[],
                contract_delta={"mode": "repair_concrete", "rules": ["include file/module boundaries"]},
            )
        },
        metadata_attachments={
            "repair_pressure_contract": {"mode": "repair_concrete", "rules": ["include file/module boundaries"]}
        },
        grammar_scalars={"repair_pressure": {"level": 0.82, "confidence": 0.71}},
    )
    assert bundle.metadata_attachments["repair_pressure_contract"]["mode"] == "repair_concrete"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /mnt/scripts/Orion-Sapienform
PYTHONPATH=. ./orion_dev/bin/python -m pytest tests/test_pre_turn_appraisal_schemas.py -v
```

Expected: FAIL — `ModuleNotFoundError: orion.schemas.pre_turn_appraisal`

- [ ] **Step 3: Implement schemas**

Create `orion/schemas/pre_turn_appraisal.py`:

```python
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from orion.substrate.appraisal.models import EvidenceKind, RepairEvidenceV1


class TurnWindowMessageV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant", "system"]
    content: str = Field(min_length=1)


class PreTurnAppraisalOptionsV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fail_closed: bool = True
    timeout_ms: int = Field(default=800, ge=100, le=5000)
    max_turns: int = Field(default=8, ge=1, le=32)


class PreTurnAppraisalRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    correlation_id: str
    session_id: str
    turn_window: list[TurnWindowMessageV1]
    paradigms_requested: list[str] = Field(default_factory=lambda: ["repair_pressure"])
    contract_before: dict[str, Any] = Field(default_factory=lambda: {"mode": "default"})
    options: PreTurnAppraisalOptionsV1 = Field(default_factory=PreTurnAppraisalOptionsV1)


class TurnAppraisalParadigmSliceV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    appraisal_kind: Literal["repair_pressure"]
    level: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    dimensions: dict[str, float] = Field(default_factory=dict)
    evidence: list[RepairEvidenceV1] = Field(default_factory=list)
    contract_delta: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class TurnAppraisalBundleV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    correlation_id: str
    paradigms: dict[str, TurnAppraisalParadigmSliceV1] = Field(default_factory=dict)
    metadata_attachments: dict[str, Any] = Field(default_factory=dict)
    grammar_scalars: dict[str, dict[str, float]] = Field(default_factory=dict)
    failed_paradigms: list[str] = Field(default_factory=list)
```

Add to `orion/bus/channels.yaml` (near other cortex channels):

```yaml
  - name: "orion:cortex:pre_turn_appraisal:request"
    kind: "request"
    schema_id: "PreTurnAppraisalRequestV1"
    producer_services: ["orion-hub"]
    consumer_services: ["orion-cortex-exec"]
    stability: "experimental"
    since: "2026-07-03"

  - name: "orion:cortex:pre_turn_appraisal:result:*"
    kind: "result"
    schema_id: "TurnAppraisalBundleV1"
    producer_services: ["orion-cortex-exec"]
    consumer_services: ["orion-hub"]
    stability: "experimental"
    since: "2026-07-03"
```

In `orion/schemas/registry.py`, import and register:

```python
from orion.schemas.pre_turn_appraisal import (
    PreTurnAppraisalRequestV1,
    TurnAppraisalBundleV1,
    TurnAppraisalParadigmSliceV1,
    TurnWindowMessageV1,
)
```

Add to `_REGISTRY`:

```python
    "PreTurnAppraisalRequestV1": PreTurnAppraisalRequestV1,
    "TurnAppraisalBundleV1": TurnAppraisalBundleV1,
    "TurnAppraisalParadigmSliceV1": TurnAppraisalParadigmSliceV1,
    "TurnWindowMessageV1": TurnWindowMessageV1,
```

- [ ] **Step 4: Run tests and registry checks**

Run:

```bash
PYTHONPATH=. ./orion_dev/bin/python -m pytest tests/test_pre_turn_appraisal_schemas.py -v
python scripts/check_schema_registry.py
python scripts/check_bus_channels.py
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add orion/schemas/pre_turn_appraisal.py orion/bus/channels.yaml orion/schemas/registry.py tests/test_pre_turn_appraisal_schemas.py
git commit -m "feat(substrate): add pre-turn appraisal bus schemas and channels"
```

---

## Task 2 — Turn window builder

**Files:**
- Create: `orion/substrate/appraisal/turn_window.py`
- Test: `tests/test_turn_window.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_turn_window.py`:

```python
from __future__ import annotations

from orion.substrate.appraisal.turn_window import build_turn_window


def test_build_turn_window_includes_assistant_messages() -> None:
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
        {"role": "user", "content": "be specific"},
    ]
    window = build_turn_window(messages, max_turns=8)
    roles = [m.role for m in window]
    assert roles == ["user", "assistant", "user"]


def test_build_turn_window_caps_at_max_turns() -> None:
    messages = [{"role": "user", "content": f"msg-{i}"} for i in range(20)]
    window = build_turn_window(messages, max_turns=4)
    assert len(window) == 4
    assert window[-1].content == "msg-19"


def test_build_turn_window_skips_empty_and_unknown_roles() -> None:
    messages = [
        {"role": "user", "content": "ok"},
        {"role": "tool", "content": "ignored"},
        {"role": "assistant", "content": "   "},
        {"role": "assistant", "content": "visible"},
    ]
    window = build_turn_window(messages, max_turns=8)
    assert len(window) == 2
    assert window[0].role == "user"
    assert window[1].content == "visible"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=. ./orion_dev/bin/python -m pytest tests/test_turn_window.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement turn window**

Create `orion/substrate/appraisal/turn_window.py`:

```python
from __future__ import annotations

from typing import Any, Iterable

from orion.schemas.pre_turn_appraisal import TurnWindowMessageV1


_ALLOWED_ROLES = frozenset({"user", "assistant", "system"})


def build_turn_window(
    messages: Iterable[dict[str, Any] | TurnWindowMessageV1],
    *,
    max_turns: int = 8,
) -> list[TurnWindowMessageV1]:
    """Normalize chat messages[] into a bounded paired turn window."""
    cap = max(1, int(max_turns))
    out: list[TurnWindowMessageV1] = []
    for item in messages:
        if isinstance(item, TurnWindowMessageV1):
            if item.content.strip():
                out.append(item)
            continue
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        if role not in _ALLOWED_ROLES:
            continue
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        out.append(TurnWindowMessageV1(role=role, content=content))  # type: ignore[arg-type]
    if len(out) > cap:
        out = out[-cap:]
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
PYTHONPATH=. ./orion_dev/bin/python -m pytest tests/test_turn_window.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add orion/substrate/appraisal/turn_window.py tests/test_turn_window.py
git commit -m "feat(substrate): add paired turn_window builder for pre-turn appraisal"
```

---

## Task 3 — Logprob probe scorer (pure functions)

**Files:**
- Create: `orion/substrate/appraisal/probe/logprob_runner.py`
- Create: `orion/substrate/appraisal/probe/__init__.py`
- Test: `tests/test_logprob_probe_runner.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_logprob_probe_runner.py`:

```python
from __future__ import annotations

from orion.substrate.appraisal.probe.logprob_runner import (
    parse_yes_no_lines,
    score_binary_logprob,
    score_kind_from_answer_token,
)


def test_parse_yes_no_lines() -> None:
    text = """
specificity_demand: YES
trust_rupture: NO
coherence_gap: yes
"""
    parsed = parse_yes_no_lines(text)
    assert parsed["specificity_demand"] == "YES"
    assert parsed["trust_rupture"] == "NO"
    assert parsed["coherence_gap"] == "YES"


def test_score_binary_logprob_sigmoid() -> None:
    score = score_binary_logprob(logprob_yes=-0.12, logprob_no=-2.4)
    assert 0.85 < score < 0.99


def test_score_kind_from_answer_token_uses_margin_as_confidence() -> None:
    entry = {
        "token": "YES",
        "logprob": -0.12,
        "top_logprobs": [
            {"token": "YES", "logprob": -0.12},
            {"token": "NO", "logprob": -2.4},
        ],
    }
    scored = score_kind_from_answer_token("specificity_demand", entry)
    assert scored is not None
    assert scored.evidence_kind == "specificity_demand"
    assert scored.detector == "logprob_probe_v2"
    assert scored.score > 0.8
    assert scored.confidence == 2.28
    assert scored.features["logprob_yes"] == -0.12
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=. ./orion_dev/bin/python -m pytest tests/test_logprob_probe_runner.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement logprob runner**

Create `orion/substrate/appraisal/probe/__init__.py` (empty) and `orion/substrate/appraisal/probe/logprob_runner.py`:

```python
from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass
from typing import Any

from orion.substrate.appraisal.models import EvidenceKind, RepairEvidenceV1


DETECTOR_NAME = "logprob_probe_v2"
_LINE_RE = re.compile(r"^([a-z_]+):\s*(YES|NO)\s*$", re.IGNORECASE | re.MULTILINE)


def _new_evidence_id() -> str:
    return f"ev_{uuid.uuid4().hex[:16]}"


def parse_yes_no_lines(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for m in _LINE_RE.finditer(text or ""):
        out[m.group(1).lower()] = m.group(2).upper()
    return out


def score_binary_logprob(*, logprob_yes: float, logprob_no: float) -> float:
    """Spec: score = sigmoid(logprob_YES - logprob_NO)."""
    delta = float(logprob_yes) - float(logprob_no)
    return 1.0 / (1.0 + math.exp(-delta))


def _top1_margin(entry: dict[str, Any]) -> float | None:
    tops = entry.get("top_logprobs")
    if not isinstance(tops, list) or len(tops) < 2:
        return None
    lps = [float(t["logprob"]) for t in tops if isinstance(t, dict) and isinstance(t.get("logprob"), (int, float))]
    if len(lps) < 2:
        return None
    lps.sort(reverse=True)
    return lps[0] - lps[1]


def _yes_no_logprobs(entry: dict[str, Any]) -> tuple[float | None, float | None]:
    yes_lp = no_lp = None
    tops = entry.get("top_logprobs")
    if isinstance(tops, list):
        for t in tops:
            if not isinstance(t, dict):
                continue
            tok = str(t.get("token") or "").strip().upper()
            lp = t.get("logprob")
            if not isinstance(lp, (int, float)):
                continue
            if tok == "YES":
                yes_lp = float(lp)
            elif tok == "NO":
                no_lp = float(lp)
    return yes_lp, no_lp


@dataclass(frozen=True)
class KindProbeScore:
    evidence_kind: EvidenceKind
    score: float
    confidence: float
    features: dict[str, float]


def score_kind_from_answer_token(kind: EvidenceKind, entry: dict[str, Any]) -> KindProbeScore | None:
    yes_lp, no_lp = _yes_no_logprobs(entry)
    if yes_lp is None or no_lp is None:
        return None
    margin = _top1_margin(entry)
    if margin is None:
        return None
    score = score_binary_logprob(logprob_yes=yes_lp, logprob_no=no_lp)
    return KindProbeScore(
        evidence_kind=kind,
        score=max(0.0, min(1.0, score)),
        confidence=max(0.0, min(1.0, margin)),
        features={"logprob_yes": yes_lp, "logprob_no": no_lp, "margin": margin},
    )


def kind_probe_to_evidence(scored: KindProbeScore, *, source_molecule_id: str = "turn_window") -> RepairEvidenceV1:
    return RepairEvidenceV1(
        evidence_id=_new_evidence_id(),
        source_molecule_id=source_molecule_id,
        evidence_kind=scored.evidence_kind,
        detector=DETECTOR_NAME,
        score=scored.score,
        confidence=scored.confidence,
        features=scored.features,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
PYTHONPATH=. ./orion_dev/bin/python -m pytest tests/test_logprob_probe_runner.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add orion/substrate/appraisal/probe/ tests/test_logprob_probe_runner.py
git commit -m "feat(substrate): add logprob_probe_v2 scorer for repair evidence"
```

---

## Task 4 — Kind-aware contract assembly (v2)

**Files:**
- Modify: `orion/substrate/appraisal/contract.py`
- Test: `tests/test_repair_pressure_contract_v2.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_repair_pressure_contract_v2.py`:

```python
from __future__ import annotations

from orion.substrate.appraisal.contract import assemble_repair_contract_delta


def test_kind_active_rules_union_when_score_high() -> None:
    kind_scores = {
        "specificity_demand": 0.91,
        "trust_rupture": 0.80,
        "coherence_gap": 0.30,
        "repetition_failure": 0.0,
        "operational_block": 0.70,
        "explicit_repair_command": 0.0,
        "assistant_accountability_demand": 0.0,
    }
    delta = assemble_repair_contract_delta(
        contract_before={"mode": "default"},
        level=0.82,
        confidence=0.71,
        kind_scores=kind_scores,
    )
    assert delta["mode"] == "repair_concrete"
    rules = delta["rules"]
    assert any("file/module boundaries" in r for r in rules)
    assert any("acknowledge correction briefly" in r for r in rules)
    assert not any("one concrete operational path" in r for r in rules)  # coherence_gap below 0.65


def test_mid_level_yields_concrete_bias() -> None:
    delta = assemble_repair_contract_delta(
        contract_before={"mode": "default"},
        level=0.55,
        confidence=0.70,
        kind_scores={"specificity_demand": 0.80},
    )
    assert delta["mode"] == "concrete_bias"


def test_low_level_returns_unchanged_contract() -> None:
    before = {"mode": "default", "rules": ["keep"]}
    delta = assemble_repair_contract_delta(
        contract_before=before,
        level=0.20,
        confidence=0.90,
        kind_scores={},
    )
    assert delta["mode"] == "default"
    assert delta.get("rules") == ["keep"]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=. ./orion_dev/bin/python -m pytest tests/test_repair_pressure_contract_v2.py -v
```

Expected: FAIL — `ImportError: assemble_repair_contract_delta`

- [ ] **Step 3: Implement kind-aware assembly**

Append to `orion/substrate/appraisal/contract.py`:

```python
_KIND_RULE_THRESHOLD = 0.65

_KIND_RULES: dict[str, str] = {
    "specificity_demand": "include file/module boundaries",
    "trust_rupture": "acknowledge correction briefly",
    "coherence_gap": "answer with one concrete operational path",
    "repetition_failure": "address the repeated ask directly",
    "operational_block": "include tests/acceptance checks; do not build section",
    "explicit_repair_command": "obey constraint (span in evidence audit)",
    "assistant_accountability_demand": "show assumptions",
}


def assemble_repair_contract_delta(
    *,
    contract_before: dict[str, Any],
    level: float,
    confidence: float,
    kind_scores: dict[str, float],
) -> dict[str, Any]:
    """v2: level gates mode; active kind scores union rules."""
    out = copy.deepcopy(contract_before)
    if level >= _LEVEL_HIGH and confidence >= _CONFIDENCE_MIN:
        out["mode"] = "repair_concrete"
        base_rules = list(_RULES_REPAIR_CONCRETE)
    elif _LEVEL_MID <= level < _LEVEL_HIGH:
        out["mode"] = "concrete_bias"
        base_rules = list(_RULES_CONCRETE_BIAS)
    else:
        return out

    kind_rules = [
        rule
        for kind, rule in _KIND_RULES.items()
        if float(kind_scores.get(kind, 0.0)) >= _KIND_RULE_THRESHOLD
    ]
    # Union while preserving order: base first, then kind-specific additions.
    seen: set[str] = set()
    merged: list[str] = []
    for r in base_rules + kind_rules:
        if r not in seen:
            seen.add(r)
            merged.append(r)
    out["rules"] = merged
    return out
```

Export from `orion/substrate/appraisal/__init__.py`:

```python
from .contract import assemble_repair_contract_delta
```

- [ ] **Step 4: Run tests**

Run:

```bash
PYTHONPATH=. ./orion_dev/bin/python -m pytest tests/test_repair_pressure_contract_v2.py tests/test_repair_pressure_behavior_contract.py -v
```

Expected: PASS (v1 behavior tests still pass via `apply_repair_pressure_contract`)

- [ ] **Step 5: Commit**

```bash
git add orion/substrate/appraisal/contract.py orion/substrate/appraisal/__init__.py tests/test_repair_pressure_contract_v2.py
git commit -m "feat(substrate): add kind-aware repair contract delta assembly for v2"
```

---

## Task 5 — Paradigm plugin rail + repair_pressure_v2

**Files:**
- Create: `orion/substrate/appraisal/paradigms/base.py`
- Create: `orion/substrate/appraisal/paradigms/registry.py`
- Create: `orion/substrate/appraisal/paradigms/repair_pressure_v2.py`
- Create: `config/substrate/repair_pressure_weights.v2.yaml`
- Test: `tests/test_repair_pressure_v2_paradigm.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_repair_pressure_v2_paradigm.py`:

```python
from __future__ import annotations

import pytest

from orion.schemas.pre_turn_appraisal import PreTurnAppraisalRequestV1, TurnWindowMessageV1
from orion.substrate.appraisal.paradigms.repair_pressure_v2 import (
    RepairPressureV2Paradigm,
    build_repair_probe_prompt,
    reduce_repair_level,
)


def test_build_repair_probe_prompt_lists_seven_kinds() -> None:
    window = [
        TurnWindowMessageV1(role="user", content="you gave garbage directions"),
        TurnWindowMessageV1(role="assistant", content="try this plan"),
    ]
    prompt = build_repair_probe_prompt(window)
    for kind in (
        "specificity_demand",
        "trust_rupture",
        "coherence_gap",
        "repetition_failure",
        "operational_block",
        "explicit_repair_command",
        "assistant_accountability_demand",
    ):
        assert kind in prompt


def test_reduce_repair_level_uses_yaml_weights() -> None:
    kind_scores = {k: 0.9 for k in (
        "specificity_demand", "trust_rupture", "coherence_gap",
        "repetition_failure", "operational_block", "explicit_repair_command",
        "assistant_accountability_demand",
    )}
    weights = {k: 0.14 for k in kind_scores}  # sums ~0.98
    level, confidence = reduce_repair_level(kind_scores, confidences={k: 0.8 for k in kind_scores}, weights=weights)
    assert level >= 0.75
    assert confidence >= 0.60


@pytest.mark.asyncio
async def test_paradigm_fail_closed_on_empty_llm(monkeypatch) -> None:
    paradigm = RepairPressureV2Paradigm(llm_caller=lambda _prompt: {"text": "", "llm_uncertainty": {"available": False}})
    req = PreTurnAppraisalRequestV1(
        correlation_id="c1",
        session_id="s1",
        turn_window=[TurnWindowMessageV1(role="user", content="be specific")],
    )
    slice_ = await paradigm.run(req)
    assert slice_.level == 0.0
    assert "no_repair_evidence" in slice_.notes
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=. ./orion_dev/bin/python -m pytest tests/test_repair_pressure_v2_paradigm.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement paradigm rail**

Create `orion/substrate/appraisal/paradigms/base.py`:

```python
from __future__ import annotations

from typing import Protocol

from orion.schemas.pre_turn_appraisal import PreTurnAppraisalRequestV1, TurnAppraisalParadigmSliceV1


class AppraisalParadigm(Protocol):
    name: str

    async def run(self, req: PreTurnAppraisalRequestV1) -> TurnAppraisalParadigmSliceV1: ...
```

Create `orion/substrate/appraisal/paradigms/registry.py`:

```python
from __future__ import annotations

from typing import Callable

from .base import AppraisalParadigm

PARADIGM_REGISTRY: dict[str, Callable[[], AppraisalParadigm]] = {}
```

Create `config/substrate/repair_pressure_weights.v2.yaml`:

```yaml
# Calibrated starting weights — tune via orion/substrate/evals/repair_pressure_v2_eval.py
specificity_demand: 0.18
trust_rupture: 0.17
coherence_gap: 0.14
repetition_failure: 0.13
operational_block: 0.13
explicit_repair_command: 0.13
assistant_accountability_demand: 0.12
```

Create `orion/substrate/appraisal/paradigms/repair_pressure_v2.py` with:

- `build_repair_probe_prompt(window)` — formats paired thread + seven YES/NO lines instruction
- `reduce_repair_level(kind_scores, confidences, weights)` — weighted sum; confidence = min confidence for kinds with score > 0.5
- `RepairPressureV2Paradigm` class with injectable `llm_caller` for tests
- `run()` parses LLM text + per-line logprobs from `llm_uncertainty.per_token` (fail-closed if unavailable)
- Calls `assemble_repair_contract_delta` for `contract_delta`
- Returns `TurnAppraisalParadigmSliceV1`

Register in `registry.py`:

```python
from .repair_pressure_v2 import RepairPressureV2Paradigm

PARADIGM_REGISTRY["repair_pressure"] = RepairPressureV2Paradigm
```

- [ ] **Step 4: Run tests**

Run:

```bash
PYTHONPATH=. ./orion_dev/bin/python -m pytest tests/test_repair_pressure_v2_paradigm.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add orion/substrate/appraisal/paradigms/ config/substrate/repair_pressure_weights.v2.yaml tests/test_repair_pressure_v2_paradigm.py
git commit -m "feat(substrate): add repair_pressure_v2 logprob paradigm plugin"
```

---

## Task 6 — Cortex-exec RPC handler

**Files:**
- Create: `services/orion-cortex-exec/app/pre_turn_appraisal.py`
- Modify: `services/orion-cortex-exec/app/main.py`
- Modify: `services/orion-cortex-exec/app/settings.py`
- Modify: `services/orion-cortex-exec/.env_example`
- Test: `services/orion-cortex-exec/tests/test_pre_turn_appraisal_rpc.py`

- [ ] **Step 1: Write the failing test**

Create `services/orion-cortex-exec/tests/test_pre_turn_appraisal_rpc.py`:

```python
from __future__ import annotations

import pytest

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.pre_turn_appraisal import PreTurnAppraisalRequestV1, TurnWindowMessageV1
from app.pre_turn_appraisal import handle_pre_turn_appraisal_request


@pytest.mark.asyncio
async def test_handler_returns_bundle_with_correlation_id(monkeypatch) -> None:
    async def _fake_run(_req):
        from orion.schemas.pre_turn_appraisal import TurnAppraisalParadigmSliceV1

        return TurnAppraisalParadigmSliceV1(
            appraisal_kind="repair_pressure",
            level=0.80,
            confidence=0.70,
            dimensions={"level": 0.80},
            contract_delta={"mode": "repair_concrete", "rules": ["be specific"]},
        )

    monkeypatch.setattr("app.pre_turn_appraisal._run_repair_pressure_paradigm", _fake_run)

    payload = PreTurnAppraisalRequestV1(
        correlation_id="corr-x",
        session_id="sess",
        turn_window=[TurnWindowMessageV1(role="user", content="nuts and bolts")],
    ).model_dump(mode="json")

    env = BaseEnvelope(
        kind="pre_turn_appraisal.request.v1",
        source=ServiceRef(name="orion-hub", version="test"),
        correlation_id="corr-x",
        payload=payload,
    )
    reply = await handle_pre_turn_appraisal_request(env)
    bundle = reply.payload
    assert bundle["correlation_id"] == "corr-x"
    assert "repair_pressure" in bundle["paradigms"]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /mnt/scripts/Orion-Sapienform/services/orion-cortex-exec
PYTHONPATH=/mnt/scripts/Orion-Sapienform:/mnt/scripts/Orion-Sapienform/services/orion-cortex-exec \
  ../../orion_dev/bin/python -m pytest tests/test_pre_turn_appraisal_rpc.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement handler + settings**

Add to `services/orion-cortex-exec/app/settings.py`:

```python
    enable_repair_pressure_v2: bool = Field(True, alias="ENABLE_REPAIR_PRESSURE_V2")
    repair_pressure_weights_v2_path: str = Field(
        "config/substrate/repair_pressure_weights.v2.yaml",
        alias="REPAIR_PRESSURE_WEIGHTS_V2_PATH",
    )
    repair_pressure_probe_route: str = Field("quick", alias="REPAIR_PRESSURE_PROBE_ROUTE")
    channel_pre_turn_appraisal_request: str = Field(
        "orion:cortex:pre_turn_appraisal:request",
        alias="CHANNEL_PRE_TURN_APPRAISAL_REQUEST",
    )
    channel_pre_turn_appraisal_result_prefix: str = Field(
        "orion:cortex:pre_turn_appraisal:result",
        alias="CHANNEL_PRE_TURN_APPRAISAL_RESULT_PREFIX",
    )
```

Create `services/orion-cortex-exec/app/pre_turn_appraisal.py`:

```python
from __future__ import annotations

import asyncio
import logging
from typing import Any
from uuid import uuid4

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ChatRequestPayload, LLMMessage, ServiceRef
from orion.schemas.pre_turn_appraisal import PreTurnAppraisalRequestV1, TurnAppraisalBundleV1
from orion.substrate.appraisal import REPAIR_PRESSURE_CONTRACT_METADATA_KEY
from orion.substrate.appraisal.paradigms.registry import PARADIGM_REGISTRY
from orion.substrate.appraisal.paradigms.repair_pressure_v2 import RepairPressureV2Paradigm

from .settings import settings

logger = logging.getLogger("orion.cortex.pre_turn_appraisal")


def _source() -> ServiceRef:
    return ServiceRef(name=settings.service_name, version=settings.service_version, node=settings.node_name)


async def _llm_probe_call(bus: OrionBusAsync, *, prompt: str, route: str, timeout_sec: float) -> dict[str, Any]:
    rpc_corr = str(uuid4())
    reply_channel = f"orion:exec:result:LLMGatewayService:{rpc_corr}"
    payload = ChatRequestPayload(
        messages=[LLMMessage(role="user", content=prompt)],
        route=route,
        options={
            "return_logprobs": True,
            "logprobs_top_k": 8,
            "logprob_summary_only": False,
            "logprob_probe_mode": "native_completion",
            "max_tokens": 128,
            "purpose": "repair_pressure_probe",
            "skip_spark_candidate_publish": True,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    env = BaseEnvelope(
        kind="llm.chat.request",
        source=_source(),
        correlation_id=rpc_corr,
        reply_to=reply_channel,
        payload=payload.model_dump(mode="json"),
    )
    msg = await bus.rpc_request(settings.channel_llm_intake, env, reply_channel=reply_channel, timeout_sec=timeout_sec)
    decoded = bus.codec.decode(msg.get("data"))
    if not decoded.ok or not isinstance(decoded.envelope.payload, dict):
        return {"text": "", "llm_uncertainty": {"available": False}}
    return decoded.envelope.payload


async def _run_repair_pressure_paradigm(req: PreTurnAppraisalRequestV1, *, bus: OrionBusAsync):
    timeout_sec = max(0.1, req.options.timeout_ms / 1000.0)

    async def caller(prompt: str) -> dict[str, Any]:
        return await _llm_probe_call(bus, prompt=prompt, route=settings.repair_pressure_probe_route, timeout_sec=timeout_sec)

    paradigm = RepairPressureV2Paradigm(
        llm_caller=caller,
        weights_path=settings.repair_pressure_weights_v2_path,
    )
    return await paradigm.run(req)


async def handle_pre_turn_appraisal_request(env: BaseEnvelope) -> BaseEnvelope:
    payload_obj = env.payload.model_dump(mode="json") if hasattr(env.payload, "model_dump") else env.payload
    req = PreTurnAppraisalRequestV1.model_validate(payload_obj or {})
    failed: list[str] = []
    paradigms: dict[str, Any] = {}
    metadata_attachments: dict[str, Any] = {}
    grammar_scalars: dict[str, dict[str, float]] = {}

    if not settings.enable_repair_pressure_v2:
        failed.append("repair_pressure")
    elif "repair_pressure" in req.paradigms_requested:
        try:
            slice_ = await asyncio.wait_for(
                _run_repair_pressure_paradigm(req, bus=_get_bus()),
                timeout=max(0.1, req.options.timeout_ms / 1000.0),
            )
            paradigms["repair_pressure"] = slice_.model_dump(mode="json")
            grammar_scalars["repair_pressure"] = {"level": slice_.level, "confidence": slice_.confidence}
            before_mode = str((req.contract_before or {}).get("mode") or "")
            after_mode = str((slice_.contract_delta or {}).get("mode") or before_mode)
            if before_mode != after_mode:
                metadata_attachments[REPAIR_PRESSURE_CONTRACT_METADATA_KEY] = dict(slice_.contract_delta)
        except Exception:
            logger.warning("pre_turn_appraisal_repair_pressure_failed corr=%s", req.correlation_id, exc_info=True)
            failed.append("repair_pressure")

    bundle = TurnAppraisalBundleV1(
        correlation_id=req.correlation_id,
        paradigms=paradigms,
        metadata_attachments=metadata_attachments,
        grammar_scalars=grammar_scalars,
        failed_paradigms=failed,
    )
    return BaseEnvelope(
        kind="pre_turn_appraisal.result.v1",
        source=_source(),
        correlation_id=req.correlation_id,
        causality_chain=env.causality_chain,
        payload=bundle.model_dump(mode="json"),
    )


_BUS: OrionBusAsync | None = None


def _get_bus() -> OrionBusAsync:
    assert _BUS is not None, "pre_turn_appraisal bus not initialized"
    return _BUS


def bind_pre_turn_appraisal_bus(bus: OrionBusAsync) -> None:
    global _BUS
    _BUS = bus
```

Wire in `main.py`:

```python
from .pre_turn_appraisal import bind_pre_turn_appraisal_bus, handle_pre_turn_appraisal_request

pre_turn_appraisal_svc = Rabbit(
    _cfg(),
    request_channel=settings.channel_pre_turn_appraisal_request,
    handler=handle_pre_turn_appraisal_request,
)
```

In `main()` after `await svc.bus.connect()`:

```python
bind_pre_turn_appraisal_bus(_rpc_bus or svc.bus)
```

In the gather block, run both listeners (pattern from orion-state-service):

```python
await asyncio.gather(svc.start(), pre_turn_appraisal_svc.start(), health_task)
```

Add keys to `services/orion-cortex-exec/.env_example`.

- [ ] **Step 4: Run tests**

Run:

```bash
cd /mnt/scripts/Orion-Sapienform/services/orion-cortex-exec
PYTHONPATH=/mnt/scripts/Orion-Sapienform:/mnt/scripts/Orion-Sapienform/services/orion-cortex-exec \
  ../../orion_dev/bin/python -m pytest tests/test_pre_turn_appraisal_rpc.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add services/orion-cortex-exec/app/pre_turn_appraisal.py services/orion-cortex-exec/app/main.py \
        services/orion-cortex-exec/app/settings.py services/orion-cortex-exec/.env_example \
        services/orion-cortex-exec/tests/test_pre_turn_appraisal_rpc.py
git commit -m "feat(cortex-exec): add pre_turn_appraisal RPC handler"
```

---

## Task 7 — Hub RPC client + wiring

**Files:**
- Create: `services/orion-hub/scripts/pre_turn_appraisal_client.py`
- Create: `services/orion-hub/scripts/pre_turn_appraisal_wiring.py`
- Modify: `services/orion-hub/app/settings.py`
- Modify: `services/orion-hub/.env_example`
- Test: `services/orion-hub/tests/test_pre_turn_appraisal_wiring.py`

- [ ] **Step 1: Write the failing test**

Create `services/orion-hub/tests/test_pre_turn_appraisal_wiring.py`:

```python
from __future__ import annotations

from orion.schemas.cortex.contracts import CortexChatRequest
from orion.schemas.pre_turn_appraisal import TurnAppraisalBundleV1, TurnAppraisalParadigmSliceV1
from orion.substrate.appraisal import REPAIR_PRESSURE_CONTRACT_METADATA_KEY
from scripts.pre_turn_appraisal_wiring import apply_pre_turn_appraisal_bundle


def test_apply_bundle_attaches_metadata_when_mode_changes() -> None:
    req = CortexChatRequest(prompt="hi", mode="brain")
    bundle = TurnAppraisalBundleV1(
        correlation_id="c1",
        paradigms={
            "repair_pressure": TurnAppraisalParadigmSliceV1(
                appraisal_kind="repair_pressure",
                level=0.82,
                confidence=0.71,
                contract_delta={"mode": "repair_concrete", "rules": ["include file/module boundaries"]},
            )
        },
        metadata_attachments={
            "repair_pressure_contract": {"mode": "repair_concrete", "rules": ["include file/module boundaries"]}
        },
        grammar_scalars={"repair_pressure": {"level": 0.82, "confidence": 0.71}},
    )
    summary = apply_pre_turn_appraisal_bundle(req, bundle, enabled=True)
    assert req.metadata[REPAIR_PRESSURE_CONTRACT_METADATA_KEY]["mode"] == "repair_concrete"
    assert summary is not None
    assert summary["level"] == 0.82
    assert summary["changed_behavior"] == "repair_concrete"


def test_apply_bundle_skips_when_disabled() -> None:
    req = CortexChatRequest(prompt="hi", mode="brain")
    bundle = TurnAppraisalBundleV1(correlation_id="c1")
    summary = apply_pre_turn_appraisal_bundle(req, bundle, enabled=False)
    assert summary is None
    assert req.metadata is None
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /mnt/scripts/Orion-Sapienform/services/orion-hub
PYTHONPATH=/mnt/scripts/Orion-Sapienform:/mnt/scripts/Orion-Sapienform/services/orion-hub \
  ../../orion_dev/bin/python -m pytest tests/test_pre_turn_appraisal_wiring.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement client + wiring**

Add to `services/orion-hub/app/settings.py`:

```python
    ENABLE_PRE_TURN_APPRAISAL: bool = Field(default=False, alias="ENABLE_PRE_TURN_APPRAISAL")
    PRE_TURN_APPRAISAL_PARADIGMS: str = Field(default="repair_pressure", alias="PRE_TURN_APPRAISAL_PARADIGMS")
    PRE_TURN_APPRAISAL_TIMEOUT_MS: int = Field(default=800, alias="PRE_TURN_APPRAISAL_TIMEOUT_MS")
    CHANNEL_PRE_TURN_APPRAISAL_REQUEST: str = Field(
        default="orion:cortex:pre_turn_appraisal:request",
        alias="CHANNEL_PRE_TURN_APPRAISAL_REQUEST",
    )
    CHANNEL_PRE_TURN_APPRAISAL_RESULT_PREFIX: str = Field(
        default="orion:cortex:pre_turn_appraisal:result",
        alias="CHANNEL_PRE_TURN_APPRAISAL_RESULT_PREFIX",
    )
```

Create `services/orion-hub/scripts/pre_turn_appraisal_client.py` (mirror `bus_clients/cortex_client.py` RPC pattern):

```python
import logging
import uuid
from typing import Optional

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.pre_turn_appraisal import PreTurnAppraisalRequestV1, TurnAppraisalBundleV1
from scripts.settings import settings

logger = logging.getLogger("hub.bus.pre_turn_appraisal")


class PreTurnAppraisalClient:
    def __init__(self, bus: OrionBusAsync):
        self.bus = bus
        self._source = ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION)

    async def appraise(
        self,
        request: PreTurnAppraisalRequestV1,
        *,
        correlation_id: Optional[str] = None,
    ) -> TurnAppraisalBundleV1 | None:
        correlation_id = correlation_id or request.correlation_id or str(uuid.uuid4())
        reply_to = f"{settings.CHANNEL_PRE_TURN_APPRAISAL_RESULT_PREFIX}:{correlation_id}"
        timeout_sec = max(0.1, request.options.timeout_ms / 1000.0)
        envelope = BaseEnvelope(
            kind="pre_turn_appraisal.request.v1",
            source=self._source,
            correlation_id=correlation_id,
            reply_to=reply_to,
            payload=request.model_dump(mode="json"),
        )
        try:
            msg = await self.bus.rpc_request(
                settings.CHANNEL_PRE_TURN_APPRAISAL_REQUEST,
                envelope,
                reply_channel=reply_to,
                timeout_sec=timeout_sec,
            )
        except TimeoutError:
            logger.warning("[%s] pre_turn_appraisal RPC timeout", correlation_id)
            return None
        decoded = self.bus.codec.decode(msg.get("data"))
        if not decoded.ok:
            return None
        payload = decoded.envelope.payload
        if isinstance(payload, dict):
            return TurnAppraisalBundleV1.model_validate(payload)
        return None
```

Create `services/orion-hub/scripts/pre_turn_appraisal_wiring.py`:

```python
from __future__ import annotations

from typing import Any

from orion.schemas.cortex.contracts import CortexChatRequest
from orion.schemas.pre_turn_appraisal import TurnAppraisalBundleV1
from orion.substrate.appraisal.view_model import pressure_label


def apply_pre_turn_appraisal_bundle(
    req: CortexChatRequest,
    bundle: TurnAppraisalBundleV1 | None,
    *,
    enabled: bool,
) -> dict[str, Any] | None:
    if not enabled or bundle is None:
        return None
    if bundle.metadata_attachments:
        meta = dict(req.metadata or {})
        meta.update(bundle.metadata_attachments)
        req.metadata = meta
    rp = bundle.paradigms.get("repair_pressure")
    if rp is None:
        return None
    level = float(rp.level)
    confidence = float(rp.confidence)
    before_attached = bool(bundle.metadata_attachments)
    behavior = None
    if before_attached:
        contract = bundle.metadata_attachments.get("repair_pressure_contract") or rp.contract_delta
        behavior = str((contract or {}).get("mode") or "")
    return {
        "turn_id": bundle.correlation_id,
        "appraisal_kind": "repair_pressure",
        "level": level,
        "level_label": pressure_label(level),
        "confidence": confidence,
        "behavior_applied": behavior,
        "evidence_count": len(rp.evidence),
        "changed_behavior": behavior,
        "chip_label": f"{behavior or 'no behavior change'} · {pressure_label(level)} repair pressure · {len(rp.evidence)} evidence drivers",
    }
```

Document keys in `services/orion-hub/.env_example`.

- [ ] **Step 4: Run tests**

Run:

```bash
cd /mnt/scripts/Orion-Sapienform/services/orion-hub
PYTHONPATH=/mnt/scripts/Orion-Sapienform:/mnt/scripts/Orion-Sapienform/services/orion-hub \
  ../../orion_dev/bin/python -m pytest tests/test_pre_turn_appraisal_wiring.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add services/orion-hub/scripts/pre_turn_appraisal_client.py services/orion-hub/scripts/pre_turn_appraisal_wiring.py \
        services/orion-hub/app/settings.py services/orion-hub/.env_example \
        services/orion-hub/tests/test_pre_turn_appraisal_wiring.py
git commit -m "feat(hub): add pre_turn_appraisal bus client and chat wiring helper"
```

---

## Task 8 — Hub chat handler integration

**Files:**
- Modify: `services/orion-hub/scripts/api_routes.py`
- Modify: `services/orion-hub/scripts/websocket_handler.py`
- Modify: `services/orion-hub/scripts/grammar_emit.py` (read bundle scalars)
- Test: extend `services/orion-hub/tests/test_handle_chat_request_substrate_effect.py`

- [ ] **Step 1: Write the failing integration test**

Append to `services/orion-hub/tests/test_handle_chat_request_substrate_effect.py`:

```python
@pytest.mark.asyncio
async def test_v2_pre_turn_appraisal_wiring_attaches_metadata(monkeypatch):
    from orion.schemas.pre_turn_appraisal import TurnAppraisalBundleV1, TurnAppraisalParadigmSliceV1
    from scripts import api_routes

    bundle = TurnAppraisalBundleV1(
        correlation_id="corr-v2",
        paradigms={
            "repair_pressure": TurnAppraisalParadigmSliceV1(
                appraisal_kind="repair_pressure",
                level=0.86,
                confidence=0.82,
                contract_delta={"mode": "repair_concrete", "rules": ["be specific"]},
            )
        },
        metadata_attachments={"repair_pressure_contract": {"mode": "repair_concrete", "rules": ["be specific"]}},
    )

    class _FakeClient:
        async def appraise(self, *_a, **_k):
            return bundle

    monkeypatch.setattr(api_routes.settings, "ENABLE_PRE_TURN_APPRAISAL", True)
    monkeypatch.setattr(api_routes, "PreTurnAppraisalClient", lambda _bus: _FakeClient())
    # ... invoke handle path with high-frustration prompt; assert metadata attached
```

(Flesh out the test using the existing `test_high_repair_pressure_attaches_contract_metadata` pattern in that file — same cortex mock, but stub pre_turn client instead of phrase pipeline.)

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /mnt/scripts/Orion-Sapienform/services/orion-hub
PYTHONPATH=/mnt/scripts/Orion-Sapienform:/mnt/scripts/Orion-Sapienform/services/orion-hub \
  ../../orion_dev/bin/python -m pytest tests/test_handle_chat_request_substrate_effect.py::test_v2_pre_turn_appraisal_wiring_attaches_metadata -v
```

Expected: FAIL

- [ ] **Step 3: Wire into chat handlers**

In `api_routes.py` and `websocket_handler.py`, replace the substrate block with:

```python
substrate_summary = None
substrate_snapshot = None
pre_turn_bundle = None

if settings.ENABLE_PRE_TURN_APPRAISAL and bus is not None:
    from scripts.pre_turn_appraisal_client import PreTurnAppraisalClient
    from orion.schemas.pre_turn_appraisal import PreTurnAppraisalRequestV1
    from orion.substrate.appraisal.turn_window import build_turn_window

    turn_window = build_turn_window(continuity_messages or [{"role": "user", "content": user_prompt}])
    pre_turn_bundle = await PreTurnAppraisalClient(bus).appraise(
        PreTurnAppraisalRequestV1(
            correlation_id=corr_id,
            session_id=session_id,
            turn_window=turn_window,
            paradigms_requested=[p.strip() for p in settings.PRE_TURN_APPRAISAL_PARADIGMS.split(",") if p.strip()],
            contract_before={"mode": "default"},
            options={"timeout_ms": settings.PRE_TURN_APPRAISAL_TIMEOUT_MS},
        )
    )
    from scripts.pre_turn_appraisal_wiring import apply_pre_turn_appraisal_bundle
    substrate_summary = apply_pre_turn_appraisal_bundle(req, pre_turn_bundle, enabled=True)
elif not settings.ENABLE_PRE_TURN_APPRAISAL:
    substrate_summary, substrate_snapshot = run_substrate_effect_pipeline(...)
    attach_repair_pressure_contract(req, substrate_snapshot, enabled=settings.ENABLE_REPAIR_PRESSURE_SPEECH_WIRING)
else:
    apply_pre_turn_appraisal_bundle(req, pre_turn_bundle, enabled=settings.ENABLE_REPAIR_PRESSURE_SPEECH_WIRING)
```

For grammar emit, when `pre_turn_bundle` is present:

```python
scalars = (pre_turn_bundle.grammar_scalars or {}).get("repair_pressure") or {}
repair_pressure_level=float(scalars.get("level", 0.0)),
repair_pressure_confidence=float(scalars.get("confidence", 0.0)),
```

- [ ] **Step 4: Run tests**

Run:

```bash
cd /mnt/scripts/Orion-Sapienform/services/orion-hub
PYTHONPATH=/mnt/scripts/Orion-Sapienform:/mnt/scripts/Orion-Sapienform/services/orion-hub \
  ../../orion_dev/bin/python -m pytest tests/test_handle_chat_request_substrate_effect.py tests/test_repair_pressure_wiring.py -v
python /mnt/scripts/Orion-Sapienform/scripts/sync_local_env_from_example.py
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add services/orion-hub/scripts/api_routes.py services/orion-hub/scripts/websocket_handler.py \
        services/orion-hub/scripts/grammar_emit.py services/orion-hub/tests/test_handle_chat_request_substrate_effect.py
git commit -m "feat(hub): wire pre_turn_appraisal RPC into chat handlers"
```

---

## Task 9 — Eval harness + transcript fixtures

**Files:**
- Create: `orion/substrate/evals/repair_pressure_v2_eval.py`
- Create: `orion/substrate/evals/fixtures/repair_pressure/ops_frustration_positive.json`
- Create: `orion/substrate/evals/fixtures/repair_pressure/grounding_negative.json`
- Create: `orion/substrate/evals/fixtures/repair_pressure/neutral.json`
- Test: run eval module

- [ ] **Step 1: Write fixture files**

`ops_frustration_positive.json` (paired thread excerpt):

```json
{
  "id": "ops_frustration_positive",
  "class": "positive",
  "turn_window": [
    {"role": "user", "content": "you gave me garbage directions — again"},
    {"role": "assistant", "content": "Here is another high-level architecture overview."},
    {"role": "user", "content": "Stop hand waving. Build me a design spec for Claude with nuts and bolts, file boundaries, tests."}
  ],
  "mock_probe_lines": {
    "specificity_demand": {"yes_lp": -0.10, "no_lp": -2.5, "margin": 2.4},
    "trust_rupture": {"yes_lp": -0.15, "no_lp": -2.2, "margin": 2.0},
    "coherence_gap": {"yes_lp": -0.30, "no_lp": -1.8, "margin": 1.5},
    "repetition_failure": {"yes_lp": -0.20, "no_lp": -2.0, "margin": 1.8},
    "operational_block": {"yes_lp": -0.12, "no_lp": -2.6, "margin": 2.5},
    "explicit_repair_command": {"yes_lp": -0.18, "no_lp": -2.1, "margin": 1.9},
    "assistant_accountability_demand": {"yes_lp": -0.25, "no_lp": -1.9, "margin": 1.6}
  },
  "expect": {"level_min": 0.75, "confidence_min": 0.60, "mode": "repair_concrete", "active_kinds_min": 3}
}
```

`grounding_negative.json`:

```json
{
  "id": "grounding_negative",
  "class": "negative",
  "turn_window": [
    {"role": "user", "content": "You lied about the API existing. That is a hallucination."},
    {"role": "assistant", "content": "You're right, I should not have claimed that endpoint exists."}
  ],
  "mock_probe_lines": {
    "specificity_demand": {"yes_lp": -2.0, "no_lp": -0.15, "margin": 1.8},
    "trust_rupture": {"yes_lp": -1.5, "no_lp": -0.20, "margin": 1.3},
    "coherence_gap": {"yes_lp": -1.2, "no_lp": -0.18, "margin": 1.0},
    "repetition_failure": {"yes_lp": -2.5, "no_lp": -0.10, "margin": 2.4},
    "operational_block": {"yes_lp": -2.0, "no_lp": -0.12, "margin": 1.9},
    "explicit_repair_command": {"yes_lp": -1.8, "no_lp": -0.14, "margin": 1.6},
    "assistant_accountability_demand": {"yes_lp": -1.0, "no_lp": -0.30, "margin": 0.7}
  },
  "expect": {"level_max": 0.45, "mode_unchanged": true, "trust_rupture_max": 0.55, "coherence_gap_max": 0.55}
}
```

`neutral.json`:

```json
{
  "id": "neutral",
  "class": "neutral",
  "turn_window": [{"role": "user", "content": "what is the weather like?"}],
  "mock_probe_lines": {},
  "expect": {"level_max": 0.25}
}
```

- [ ] **Step 2: Write eval harness**

Create `orion/substrate/evals/repair_pressure_v2_eval.py` that:

1. Loads fixtures from `fixtures/repair_pressure/*.json`
2. Builds synthetic LLM responses from `mock_probe_lines`
3. Runs `RepairPressureV2Paradigm` with injected `llm_caller`
4. Asserts per-class expectations from spec acceptance table
5. Exposes `pytest` tests AND `python -m orion.substrate.evals.repair_pressure_v2_eval` CLI

- [ ] **Step 3: Run eval**

Run:

```bash
cd /mnt/scripts/Orion-Sapienform
PYTHONPATH=. ./orion_dev/bin/python -m pytest orion/substrate/evals/repair_pressure_v2_eval.py -v
PYTHONPATH=. ./orion_dev/bin/python -m orion.substrate.evals.repair_pressure_v2_eval
```

Expected: all fixtures PASS

- [ ] **Step 4: Commit**

```bash
git add orion/substrate/evals/
git commit -m "test(substrate): add repair_pressure_v2 transcript eval fixtures and harness"
```

---

## Task 10 — Legacy deprecation + env sync

**Files:**
- Modify: `services/orion-hub/scripts/substrate_effect_pipeline.py` (guard with flag)
- Modify: `services/orion-cortex-exec/app/settings.py` (document legacy fallback)
- Run: env sync + agent checks

- [ ] **Step 1: Guard legacy pipeline**

At top of `run_substrate_effect_pipeline`, early-return when Hub has v2 enabled (caller should not invoke, but belt-and-suspenders):

```python
from scripts.settings import settings as hub_settings

if getattr(hub_settings, "ENABLE_PRE_TURN_APPRAISAL", False):
    logger.debug("substrate_effect_pipeline_skipped_v2_enabled turn_id=%s", turn_id)
    return None, None
```

- [ ] **Step 2: Sync env and run gates**

Run:

```bash
cd /mnt/scripts/Orion-Sapienform
python scripts/sync_local_env_from_example.py
python scripts/check_env_template_parity.py
python scripts/check_schema_registry.py
python scripts/check_bus_channels.py
PYTHONPATH=. ./orion_dev/bin/python -m pytest \
  tests/test_pre_turn_appraisal_schemas.py \
  tests/test_turn_window.py \
  tests/test_logprob_probe_runner.py \
  tests/test_repair_pressure_contract_v2.py \
  tests/test_repair_pressure_v2_paradigm.py \
  tests/test_repair_pressure_behavior_contract.py \
  services/orion-cortex-exec/tests/test_pre_turn_appraisal_rpc.py \
  services/orion-hub/tests/test_pre_turn_appraisal_wiring.py \
  services/orion-hub/tests/test_repair_pressure_wiring.py \
  orion/substrate/evals/repair_pressure_v2_eval.py \
  -q
```

Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add services/orion-hub/scripts/substrate_effect_pipeline.py
git commit -m "chore(hub): skip legacy phrase_match pipeline when pre_turn_appraisal enabled"
```

---

## Self-review (spec coverage)

| Spec requirement | Task |
|------------------|------|
| PreTurnAppraisalRequestV1 / TurnAppraisalBundleV1 bus contract | Task 1 |
| Paired turn window (user + assistant) | Task 2 |
| logprob_probe_v2 scoring (no phrase tables) | Task 3 |
| Seven kinds → contract rules union | Task 4 |
| Paradigm plugin rail + repair_pressure_v2 | Task 5 |
| Cortex-exec RPC handler | Task 6 |
| Hub client + chat wiring | Tasks 7–8 |
| Eval positive/negative/neutral fixtures | Task 9 |
| Deprecate phrase_match hot path | Task 10 |
| Speech wiring unchanged (`repair_pressure_contract` metadata) | Tasks 7–8 reuse existing key |
| Fail-closed on timeout / missing logprobs | Tasks 5–6 |
| Env parity + registry checks | Task 10 |

No placeholder steps remain. Type names consistent across tasks.

---

## Restart required (after full implementation)

```bash
# Rebuild and restart cortex-exec (new RPC listener)
docker compose \
  --env-file .env \
  --env-file services/orion-cortex-exec/.env \
  -f services/orion-cortex-exec/docker-compose.yml \
  up -d --build

# Restart hub (new RPC client + wiring)
docker compose \
  --env-file .env \
  --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml \
  up -d --build
```

Enable v2 on Hub after smoke:

```bash
# services/orion-hub/.env
ENABLE_PRE_TURN_APPRAISAL=true
PRE_TURN_APPRAISAL_PARADIGMS=repair_pressure
PRE_TURN_APPRAISAL_TIMEOUT_MS=800
```
