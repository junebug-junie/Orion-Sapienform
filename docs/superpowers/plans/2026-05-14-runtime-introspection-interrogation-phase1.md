# Runtime Introspection Interrogation (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship Phase 1 of runtime introspection per `docs/superpowers/specs/2026-05-09-runtime-introspection-interrogation-design.md`: Hub routes interrogation-shaped user text onto the agent lane with verb `interrogate_runtime_state`, a pinned capability skill executes deterministic assembly of the strict evidence contract plus human-readable report derived only from that contract, and a fail-closed claim guardrail blocks unsupported hard claims.

**Architecture:** Hub performs cheap intent detection (feature-flagged) and emits `CortexChatRequest` with `mode=agent`, `verb=interrogate_runtime_state`, recall disabled, and introspection options. Agent-chain capability resolution uses a **semantic one-to-one map** (same pattern as `answer_current_datetime` → `skills.system.time_now.v1`) so `interrogate_runtime_state` always selects `skills.runtime.interrogate_runtime_state.v1`. Cortex-exec implements that skill as a `@verb` adapter (like `skills.docker.ps_status.v1`) that calls a new pure library under `orion/runtime_interrogation/` to build `RuntimeInterrogationContractV1`, run redaction + guardrail, render markdown, and return JSON in the existing `SkillVerbOutput` shape. Phase 1 deliberately marks most hops `unknown` until later tasks add real bus/state probes; the **shape** and **guardrails** are the deliverable.

**Tech Stack:** Python 3, Pydantic v2, YAML cognition verbs, existing Redis Orion bus, `pytest`, services `orion-hub`, `orion-agent-chain`, `orion-cortex-exec`, shared package `orion/`.

---

## File structure (create / modify)

| Path | Responsibility |
|------|----------------|
| `orion/schemas/runtime_interrogation/v1.py` | Strict-layer Pydantic models matching the spec YAML (query scope, correlation resolution, hop status, trace context, evidence, claims, fallback, root cause, next steps, summary, redaction envelope). |
| `orion/runtime_interrogation/__init__.py` | Package marker. |
| `orion/runtime_interrogation/correlation_resolution.py` | Deterministic resolution order + caps + authorization for cross-scope. |
| `orion/runtime_interrogation/redaction.py` | Secret/token/header redaction for evidence blobs and nested dicts. |
| `orion/runtime_interrogation/claim_guardrail.py` | Validator emitting counts + `guardrail_outcome`. |
| `orion/runtime_interrogation/human_report.py` | Markdown sections strictly derived from the contract. |
| `orion/runtime_interrogation/phase1_assemble.py` | Builds initial contract from `PlanExecutionRequest` / `VerbContext` meta (correlation_id, trace_id, session_id, options). |
| `orion/cognition/verbs/interrogate_runtime_state.yaml` | Top-level capability-backed verb (mirror `assess_runtime_state.yaml`). |
| `orion/cognition/verbs/skills.runtime.interrogate_runtime_state.v1.yaml` | Skill manifest consumed by `ActionsSkillRegistry`. |
| `services/orion-agent-chain/app/capability_bridge.py` | Pin `interrogate_runtime_state` → `skills.runtime.interrogate_runtime_state.v1`. |
| `services/orion-cortex-exec/app/verb_adapters.py` | Register `InterrogateRuntimeStateSkillVerb` for `skills.runtime.interrogate_runtime_state.v1`. |
| `services/orion-hub/scripts/runtime_introspection_routing.py` | Pure `looks_like_runtime_introspection(prompt) -> bool` + optional `apply_runtime_introspection_routing(...)`. |
| `services/orion-hub/scripts/chat_request_builder.py` | Call routing helper when settings enabled. |
| `services/orion-hub/app/settings.py` | `HUB_RUNTIME_INTROSPECTION_ROUTING_ENABLED`, `HUB_RUNTIME_INTROSPECTION_SHADOW_MODE`. |
| `services/orion-hub/.env_example` | Document new env keys (empty/safe defaults). |
| `orion/runtime_interrogation/tests/test_correlation_resolution.py` | Unit tests for resolution + gating. |
| `orion/runtime_interrogation/tests/test_claim_guardrail.py` | Unit tests for fail-closed guardrail. |
| `orion/runtime_interrogation/tests/test_redaction.py` | Unit tests for redaction rules. |
| `orion/runtime_interrogation/tests/test_human_report.py` | Snapshot-style string tests for section order + fallback banner. |
| `services/orion-hub/tests/test_runtime_introspection_routing.py` | Tests for `build_cortex_chat_request` when flag on/off. |
| `services/orion-cortex-exec/tests/test_interrogate_runtime_state_skill.py` | Async test invoking the verb adapter with fake payload. |

---

### Task 1: Strict contract Pydantic models

**Files:**
- Create: `orion/schemas/runtime_interrogation/__init__.py`
- Create: `orion/schemas/runtime_interrogation/v1.py`

- [ ] **Step 1: Write the failing import test**

Create `orion/schemas/runtime_interrogation/tests/test_contract_import.py`:

```python
from orion.schemas.runtime_interrogation.v1 import RuntimeInterrogationContractV1


def test_contract_minimal_construct():
    c = RuntimeInterrogationContractV1.model_validate(
        {
            "query_scope": {
                "mode": "current_turn",
                "session_scope_required": True,
                "as_of_utc": "2026-05-14T12:00:00+00:00",
            },
            "correlation_resolution": {
                "target_correlation_id": None,
                "requested_prior_turns": 0,
                "requested_correlation_ids": [],
                "resolved_correlation_ids": ["abc"],
                "resolution_strategy": "current_only",
                "authorization_scope": "same_session",
                "authorization_decision": "allow",
                "authorization_reason": "default_same_session",
            },
            "hop_status": {
                "hub": "unknown",
                "cortex_gateway": "unknown",
                "cortex_orch": "unknown",
                "cortex_exec": "unknown",
                "llm_gateway": "unknown",
                "state_service": "unknown",
                "mind_artifact_path": "unknown",
            },
            "trace_context": {"trace_ids": [], "channel_refs": [], "hop_event_refs": []},
            "evidence": [],
            "claims": [],
            "fallback": {
                "fallback_level": "none",
                "fallback_used": False,
                "fallback_reason": None,
                "missing_primitives": [],
                "degraded_hops": [],
                "budget_clipped": False,
            },
            "root_cause_summary": {
                "first_failing_boundary": "unknown",
                "rankings": [],
            },
            "recommended_next_steps": [],
            "summary": {
                "headline": "Phase 1 contract skeleton",
                "certainty": "low",
            },
            "redaction_applied": False,
            "redaction_rules_triggered": [],
        }
    )
    assert c.summary.certainty == "low"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /mnt/scripts/Orion-Sapienform
PYTHONPATH=. ./venv/bin/python -m pytest orion/schemas/runtime_interrogation/tests/test_contract_import.py::test_contract_minimal_construct -v
```

Expected: `ModuleNotFoundError` or collection error until `v1.py` exists.

- [ ] **Step 3: Implement `v1.py`**

Create `orion/schemas/runtime_interrogation/__init__.py`:

```python
from orion.schemas.runtime_interrogation.v1 import RuntimeInterrogationContractV1

__all__ = ["RuntimeInterrogationContractV1"]
```

Create `orion/schemas/runtime_interrogation/v1.py`:

```python
from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

QueryScopeMode = Literal["current_turn", "target_correlation", "prior_window", "explicit_set"]
ResolutionStrategy = Literal["explicit_set", "target_plus_prior", "current_only"]
AuthScope = Literal["same_session", "cross_scope"]
AuthDecision = Literal["allow", "deny"]
HopState = Literal["ok", "degraded", "fail", "unknown"]
EvidenceType = Literal["artifact_row", "execution_step", "hop_event", "trace_span", "derived_inference"]
StalenessClass = Literal["fresh", "stale", "unknown"]
ClaimSupport = Literal["true", "false", "partial"]
FallbackLevel = Literal["none", "partial", "best_effort", "insufficient_evidence"]
NextStepAction = Literal[
    "config_change",
    "routing_policy_change",
    "primitive_missing",
    "service_health",
    "data_lag",
    "prompt_contract_update",
]
Priority = Literal["high", "medium", "low"]
Certainty = Literal["high", "medium", "low"]
GuardrailOutcome = Literal["pass", "degraded", "blocked"]


class QueryScopeV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: QueryScopeMode
    session_scope_required: bool = True
    as_of_utc: str


class CorrelationResolutionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_correlation_id: Optional[str] = None
    requested_prior_turns: int = 0
    requested_correlation_ids: List[str] = Field(default_factory=list)
    resolved_correlation_ids: List[str] = Field(default_factory=list)
    resolution_strategy: ResolutionStrategy
    authorization_scope: AuthScope
    authorization_decision: AuthDecision
    authorization_reason: str


class HopStatusV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hub: HopState = "unknown"
    cortex_gateway: HopState = "unknown"
    cortex_orch: HopState = "unknown"
    cortex_exec: HopState = "unknown"
    llm_gateway: HopState = "unknown"
    state_service: HopState = "unknown"
    mind_artifact_path: HopState = "unknown"


class TraceContextV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_ids: List[str] = Field(default_factory=list)
    channel_refs: List[str] = Field(default_factory=list)
    hop_event_refs: List[str] = Field(default_factory=list)


class EvidenceRecordV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    evidence_type: EvidenceType
    source: str
    source_id: str
    correlation_id: str
    observed_at_utc: str
    freshness_ms: Optional[int] = None
    staleness_class: StalenessClass = "unknown"
    confidence_base: float = Field(ge=0.0, le=1.0)
    redacted: bool = False
    payload: Dict[str, Any] = Field(default_factory=dict, description="Redacted probe payload.")


class ClaimV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    statement: str
    supported: ClaimSupport
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_refs: List[str] = Field(default_factory=list)
    conflict_set: List[str] = Field(default_factory=list)
    selected_authority: Optional[EvidenceType] = None
    selection_reason: str = ""


class FallbackV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fallback_level: FallbackLevel
    fallback_used: bool
    fallback_reason: Optional[str] = None
    missing_primitives: List[str] = Field(default_factory=list)
    degraded_hops: List[str] = Field(default_factory=list)
    budget_clipped: bool = False


class RootCauseRankingV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cause_id: str
    description: str
    score: float
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_refs: List[str] = Field(default_factory=list)
    contradiction_count: int = 0
    freshness_penalty: float = 0.0


class RootCauseSummaryV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    first_failing_boundary: str
    rankings: List[RootCauseRankingV1] = Field(default_factory=list)


class RecommendedNextStepV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str
    action_type: NextStepAction
    description: str
    expected_outcome: str
    evidence_refs: List[str] = Field(default_factory=list)
    priority: Priority = "medium"


class SummaryV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    headline: str
    certainty: Certainty


class GuardrailReportV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    unsupported_claim_count: int = 0
    blocked_claim_count: int = 0
    downgraded_claim_count: int = 0
    guardrail_outcome: GuardrailOutcome = "pass"


class RuntimeInterrogationContractV1(BaseModel):
    """Strict authoritative layer (spec: Runtime Introspection Interrogation)."""

    model_config = ConfigDict(extra="forbid")

    query_scope: QueryScopeV1
    correlation_resolution: CorrelationResolutionV1
    hop_status: HopStatusV1
    trace_context: TraceContextV1
    evidence: List[EvidenceRecordV1]
    claims: List[ClaimV1]
    fallback: FallbackV1
    root_cause_summary: RootCauseSummaryV1
    recommended_next_steps: List[RecommendedNextStepV1]
    summary: SummaryV1
    redaction_applied: bool = False
    redaction_rules_triggered: List[str] = Field(default_factory=list)
    guardrail: GuardrailReportV1 = Field(default_factory=GuardrailReportV1)
    human_report_markdown: str = ""
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
PYTHONPATH=. ./venv/bin/python -m pytest orion/schemas/runtime_interrogation/tests/test_contract_import.py::test_contract_minimal_construct -v
```

Expected: `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add orion/schemas/runtime_interrogation/
git commit -m "feat(runtime-interrogation): add strict contract Pydantic models"
```

---

### Task 2: Correlation resolution and authorization

**Files:**
- Create: `orion/runtime_interrogation/correlation_resolution.py`
- Create: `orion/runtime_interrogation/tests/test_correlation_resolution.py`

- [ ] **Step 1: Write failing tests**

Create `orion/runtime_interrogation/tests/test_correlation_resolution.py`:

```python
import pytest

from orion.runtime_interrogation.correlation_resolution import (
    CorrelationResolutionInput,
    resolve_correlation_resolution,
)


def test_explicit_set_requires_cross_scope_cap():
    out = resolve_correlation_resolution(
        CorrelationResolutionInput(
            session_correlation_id="sess-turn",
            explicit_correlation_ids=["other-turn"],
            target_correlation_id=None,
            prior_turns=0,
            operator_capabilities=[],
        )
    )
    assert out.authorization_decision == "deny"
    assert out.resolved_correlation_ids == []


def test_explicit_set_allowed_with_cap():
    out = resolve_correlation_resolution(
        CorrelationResolutionInput(
            session_correlation_id="sess-turn",
            explicit_correlation_ids=["other-turn"],
            target_correlation_id=None,
            prior_turns=0,
            operator_capabilities=["introspection.cross_scope"],
        )
    )
    assert out.authorization_decision == "allow"
    assert out.resolved_correlation_ids == ["other-turn"]


def test_default_current_only():
    out = resolve_correlation_resolution(
        CorrelationResolutionInput(
            session_correlation_id="only-one",
            explicit_correlation_ids=[],
            target_correlation_id=None,
            prior_turns=0,
            operator_capabilities=[],
        )
    )
    assert out.resolution_strategy == "current_only"
    assert out.resolved_correlation_ids == ["only-one"]
```

- [ ] **Step 2: Run pytest expecting failure**

```bash
PYTHONPATH=. ./venv/bin/python -m pytest orion/runtime_interrogation/tests/test_correlation_resolution.py -v
```

Expected: import or attribute errors.

- [ ] **Step 3: Implement module**

Create `orion/runtime_interrogation/__init__.py` (empty).

Create `orion/runtime_interrogation/correlation_resolution.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

from orion.schemas.runtime_interrogation.v1 import CorrelationResolutionV1

ResolutionStrategy = Literal["explicit_set", "target_plus_prior", "current_only"]
AuthScope = Literal["same_session", "cross_scope"]
AuthDecision = Literal["allow", "deny"]

MAX_CORRELATIONS = 16
MAX_PRIOR_TURNS = 8


@dataclass(frozen=True)
class CorrelationResolutionInput:
    session_correlation_id: str
    explicit_correlation_ids: List[str]
    target_correlation_id: Optional[str]
    prior_turns: int
    operator_capabilities: List[str]


def resolve_correlation_resolution(inp: CorrelationResolutionInput) -> CorrelationResolutionV1:
    caps = {str(c).strip() for c in inp.operator_capabilities if str(c).strip()}
    cross_allowed = "introspection.cross_scope" in caps

    explicit = [str(x).strip() for x in inp.explicit_correlation_ids if str(x).strip()][:MAX_CORRELATIONS]
    if explicit:
        same_session_only = all(cid == inp.session_correlation_id for cid in explicit)
        if not same_session_only and not cross_allowed:
            return CorrelationResolutionV1(
                target_correlation_id=inp.target_correlation_id,
                requested_prior_turns=max(0, min(inp.prior_turns, MAX_PRIOR_TURNS)),
                requested_correlation_ids=explicit,
                resolved_correlation_ids=[],
                resolution_strategy="explicit_set",
                authorization_scope="cross_scope",
                authorization_decision="deny",
                authorization_reason="cross_scope_capability_required",
            )
        return CorrelationResolutionV1(
            target_correlation_id=inp.target_correlation_id,
            requested_prior_turns=max(0, min(inp.prior_turns, MAX_PRIOR_TURNS)),
            requested_correlation_ids=explicit,
            resolved_correlation_ids=explicit,
            resolution_strategy="explicit_set",
            authorization_scope="cross_scope" if not same_session_only else "same_session",
            authorization_decision="allow",
            authorization_reason="explicit_set_authorized",
        )

    if inp.target_correlation_id and str(inp.target_correlation_id).strip():
        tid = str(inp.target_correlation_id).strip()
        if tid != inp.session_correlation_id and not cross_allowed:
            return CorrelationResolutionV1(
                target_correlation_id=tid,
                requested_prior_turns=max(0, min(inp.prior_turns, MAX_PRIOR_TURNS)),
                requested_correlation_ids=[],
                resolved_correlation_ids=[],
                resolution_strategy="target_plus_prior",
                authorization_scope="cross_scope",
                authorization_decision="deny",
                authorization_reason="target_correlation_outside_session_requires_capability",
            )
        resolved = [tid]
        return CorrelationResolutionV1(
            target_correlation_id=tid,
            requested_prior_turns=max(0, min(inp.prior_turns, MAX_PRIOR_TURNS)),
            requested_correlation_ids=[],
            resolved_correlation_ids=resolved[:MAX_CORRELATIONS],
            resolution_strategy="target_plus_prior",
            authorization_scope="same_session",
            authorization_decision="allow",
            authorization_reason="target_in_session",
        )

    sid = str(inp.session_correlation_id or "").strip() or "unknown"
    return CorrelationResolutionV1(
        target_correlation_id=None,
        requested_prior_turns=max(0, min(inp.prior_turns, MAX_PRIOR_TURNS)),
        requested_correlation_ids=[],
        resolved_correlation_ids=[sid],
        resolution_strategy="current_only",
        authorization_scope="same_session",
        authorization_decision="allow",
        authorization_reason="default_current_turn",
    )
```

- [ ] **Step 4: Run pytest expecting pass**

```bash
PYTHONPATH=. ./venv/bin/python -m pytest orion/runtime_interrogation/tests/test_correlation_resolution.py -v
```

- [ ] **Step 5: Commit**

```bash
git add orion/runtime_interrogation/
git commit -m "feat(runtime-interrogation): correlation resolution with cross-scope gate"
```

---

### Task 3: Redaction helpers

**Files:**
- Create: `orion/runtime_interrogation/redaction.py`
- Create: `orion/runtime_interrogation/tests/test_redaction.py`

- [ ] **Step 1: Write failing tests**

```python
from orion.runtime_interrogation.redaction import redact_mapping


def test_redacts_authorization_header_key():
    src = {"headers": {"Authorization": "Bearer secret", "X-Debug": "1"}}
    out, rules = redact_mapping(src)
    assert "authorization_header" in rules
    assert out["headers"]["Authorization"] == "[REDACTED]"


def test_redacts_nested_token_string():
    src = {"payload": 'token="abc123"'}
    out, rules = redact_mapping(src)
    assert "token_like" in rules
    assert "abc123" not in str(out)
```

- [ ] **Step 2: Run pytest expecting failure**

```bash
PYTHONPATH=. ./venv/bin/python -m pytest orion/runtime_interrogation/tests/test_redaction.py -v
```

- [ ] **Step 3: Implement `redaction.py`**

```python
from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Tuple

_SENSITIVE_KEYS = frozenset(
    k.lower()
    for k in (
        "Authorization",
        "authorization",
        "X-Api-Key",
        "api_key",
        "apikey",
        "password",
        "secret",
        "token",
        "access_token",
        "refresh_token",
    )
)

_TOKENISH = re.compile(r"(?i)(bearer\s+)(\S+)")
_INLINE_TOKEN = re.compile(r'(?i)(token|apikey|api_key|secret)\s*=\s*"([^"]+)"')


def redact_mapping(data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    rules: List[str] = []
    out = copy.deepcopy(data)

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in list(obj.items()):
                lk = str(k).lower()
                if lk in _SENSITIVE_KEYS or lk.endswith("_token") or lk.endswith("_secret"):
                    obj[k] = "[REDACTED]"
                    rules.append("sensitive_key")
                else:
                    walk(v)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)
        elif isinstance(obj, str):
            pass

    walk(out)
    serialized = str(out)

    def redact_str(s: str) -> str:
        nonlocal serialized
        s2, n = _TOKENISH.subn(r"\1[REDACTED]", s)
        if n:
            rules.append("bearer_token")
        s3, n2 = _INLINE_TOKEN.subn(lambda m: f'{m.group(1)}="[REDACTED]"', s2)
        if n2:
            rules.append("token_like")
        return s3

    def walk_str(obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, str):
                    obj[k] = redact_str(v)
                else:
                    walk_str(v)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if isinstance(item, str):
                    obj[i] = redact_str(item)
                else:
                    walk_str(item)

    walk_str(out)
    return out, sorted(set(rules))
```

- [ ] **Step 4: Run pytest**

```bash
PYTHONPATH=. ./venv/bin/python -m pytest orion/runtime_interrogation/tests/test_redaction.py -v
```

- [ ] **Step 5: Commit**

```bash
git add orion/runtime_interrogation/redaction.py orion/runtime_interrogation/tests/test_redaction.py
git commit -m "feat(runtime-interrogation): redact secrets in evidence payloads"
```

---

### Task 4: Claim guardrail (fail-closed)

**Files:**
- Create: `orion/runtime_interrogation/claim_guardrail.py`
- Create: `orion/runtime_interrogation/tests/test_claim_guardrail.py`

- [ ] **Step 1: Write failing tests**

```python
from orion.schemas.runtime_interrogation.v1 import (
    ClaimV1,
    EvidenceRecordV1,
    FallbackV1,
    GuardrailReportV1,
    HopStatusV1,
    QueryScopeV1,
    RootCauseSummaryV1,
    RuntimeInterrogationContractV1,
    SummaryV1,
    TraceContextV1,
    CorrelationResolutionV1,
)

from orion.runtime_interrogation.claim_guardrail import apply_claim_guardrail


def _base_contract(**kwargs):
    base = dict(
        query_scope=QueryScopeV1(mode="current_turn", session_scope_required=True, as_of_utc="2026-05-14T00:00:00Z"),
        correlation_resolution=CorrelationResolutionV1(
            requested_prior_turns=0,
            requested_correlation_ids=[],
            resolved_correlation_ids=["c1"],
            resolution_strategy="current_only",
            authorization_scope="same_session",
            authorization_decision="allow",
            authorization_reason="ok",
        ),
        hop_status=HopStatusV1(),
        trace_context=TraceContextV1(),
        evidence=[
            EvidenceRecordV1(
                evidence_id="e1",
                evidence_type="hop_event",
                source="hub",
                source_id="h1",
                correlation_id="c1",
                observed_at_utc="2026-05-14T00:00:00Z",
                confidence_base=1.0,
            )
        ],
        claims=[],
        fallback=FallbackV1(fallback_level="none", fallback_used=False),
        root_cause_summary=RootCauseSummaryV1(first_failing_boundary="n/a"),
        recommended_next_steps=[],
        summary=SummaryV1(headline="h", certainty="low"),
        guardrail=GuardrailReportV1(),
    )
    base.update(kwargs)
    return RuntimeInterrogationContractV1.model_validate(base)


def test_blocks_claim_with_missing_evidence_ref():
    c = _base_contract(
        claims=[
            ClaimV1(
                claim_id="k1",
                statement="Hub returned 500",
                supported="true",
                confidence=1.0,
                evidence_refs=["missing"],
            )
        ]
    )
    out = apply_claim_guardrail(c)
    assert out.guardrail.blocked_claim_count >= 1
    assert out.guardrail.guardrail_outcome == "blocked"
    assert out.fallback.fallback_level == "insufficient_evidence"


def test_downgrades_soft_inference_without_refs():
    c = _base_contract(
        claims=[
            ClaimV1(
                claim_id="k2",
                statement="Maybe network blip",
                supported="true",
                confidence=0.4,
                evidence_refs=[],
            )
        ]
    )
    out = apply_claim_guardrail(c)
    assert out.guardrail.downgraded_claim_count >= 1
    assert out.claims[0].supported == "false"
```

- [ ] **Step 2: Run pytest expecting failure**

```bash
PYTHONPATH=. ./venv/bin/python -m pytest orion/runtime_interrogation/tests/test_claim_guardrail.py -v
```

- [ ] **Step 3: Implement `claim_guardrail.py`**

```python
from __future__ import annotations

from typing import Set

from orion.schemas.runtime_interrogation.v1 import ClaimV1, RuntimeInterrogationContractV1


def apply_claim_guardrail(contract: RuntimeInterrogationContractV1) -> RuntimeInterrogationContractV1:
    evidence_ids: Set[str] = {e.evidence_id for e in contract.evidence}
    unsupported = 0
    blocked = 0
    downgraded = 0
    new_claims: list[ClaimV1] = []

    for claim in contract.claims:
        c = claim.model_copy(deep=True)
        if not c.evidence_refs:
            if c.supported == "true" and c.confidence >= 0.75:
                blocked += 1
                c.supported = "false"
                c.confidence = 0.0
                c.selection_reason = "guardrail:block_no_evidence_refs"
            else:
                downgraded += 1
                c.supported = "false"
                c.confidence = min(c.confidence, 0.35)
                c.selection_reason = "guardrail:downgrade_no_evidence_refs"
            new_claims.append(c)
            continue

        missing = [ref for ref in c.evidence_refs if ref not in evidence_ids]
        if missing and c.supported == "true":
            blocked += 1
            c.supported = "false"
            c.confidence = 0.0
            c.selection_reason = "guardrail:block_missing_evidence"
            new_claims.append(c)
            continue

        if c.supported == "false" and not missing:
            unsupported += 1

        new_claims.append(c)

    outcome = "pass"
    if blocked:
        outcome = "blocked"
    elif downgraded:
        outcome = "degraded"

    fb = contract.fallback.model_copy(deep=True)
    if outcome == "blocked":
        fb.fallback_level = "insufficient_evidence"
        fb.fallback_used = True
        fb.fallback_reason = "claim_guardrail_blocked_publish"
        fb.missing_primitives = sorted(set(fb.missing_primitives + ["strict_claim_evidence_binding"]))

    return contract.model_copy(
        update={
            "claims": new_claims,
            "guardrail": contract.guardrail.model_copy(
                update={
                    "unsupported_claim_count": unsupported,
                    "blocked_claim_count": blocked,
                    "downgraded_claim_count": downgraded,
                    "guardrail_outcome": outcome,
                }
            ),
            "fallback": fb,
        }
    )
```

- [ ] **Step 4: Run pytest**

```bash
PYTHONPATH=. ./venv/bin/python -m pytest orion/runtime_interrogation/tests/test_claim_guardrail.py -v
```

- [ ] **Step 5: Commit**

```bash
git add orion/runtime_interrogation/claim_guardrail.py orion/runtime_interrogation/tests/test_claim_guardrail.py
git commit -m "feat(runtime-interrogation): fail-closed claim-evidence guardrail"
```

---

### Task 5: Human report renderer

**Files:**
- Create: `orion/runtime_interrogation/human_report.py`
- Create: `orion/runtime_interrogation/tests/test_human_report.py`

- [ ] **Step 1: Write failing test**

```python
from orion.runtime_interrogation.human_report import render_human_report_markdown


def test_fallback_banner_prepended():
    from orion.schemas.runtime_interrogation.v1 import (
        ClaimV1,
        CorrelationResolutionV1,
        EvidenceRecordV1,
        FallbackV1,
        GuardrailReportV1,
        HopStatusV1,
        QueryScopeV1,
        RootCauseSummaryV1,
        RootCauseRankingV1,
        RecommendedNextStepV1,
        RuntimeInterrogationContractV1,
        SummaryV1,
        TraceContextV1,
    )

    c = RuntimeInterrogationContractV1(
        query_scope=QueryScopeV1(mode="current_turn", session_scope_required=True, as_of_utc="2026-05-14T00:00:00Z"),
        correlation_resolution=CorrelationResolutionV1(
            requested_prior_turns=0,
            requested_correlation_ids=[],
            resolved_correlation_ids=["x"],
            resolution_strategy="current_only",
            authorization_scope="same_session",
            authorization_decision="allow",
            authorization_reason="ok",
        ),
        hop_status=HopStatusV1(),
        trace_context=TraceContextV1(),
        evidence=[
            EvidenceRecordV1(
                evidence_id="e1",
                evidence_type="trace_span",
                source="orch",
                source_id="s1",
                correlation_id="x",
                observed_at_utc="2026-05-14T00:00:00Z",
                confidence_base=1.0,
            )
        ],
        claims=[
            ClaimV1(
                claim_id="c1",
                statement="Observed correlation x",
                supported="true",
                confidence=0.9,
                evidence_refs=["e1"],
            )
        ],
        fallback=FallbackV1(fallback_level="partial", fallback_used=True, fallback_reason="probe_budget"),
        root_cause_summary=RootCauseSummaryV1(
            first_failing_boundary="orch->exec",
            rankings=[
                RootCauseRankingV1(
                    cause_id="r1",
                    description="Unknown stall",
                    score=0.5,
                    confidence=0.4,
                    evidence_refs=["e1"],
                )
            ],
        ),
        recommended_next_steps=[
            RecommendedNextStepV1(
                step_id="n1",
                action_type="service_health",
                description="Check exec logs",
                expected_outcome="Confirm handler latency",
                evidence_refs=["e1"],
                priority="high",
            )
        ],
        summary=SummaryV1(headline="Partial visibility", certainty="medium"),
        guardrail=GuardrailReportV1(),
    )
    md = render_human_report_markdown(c)
    assert "Fallback mode active" in md
    assert "## Root Cause Summary" in md
    assert "## Recommended Next Steps" in md
```

- [ ] **Step 2: Run pytest expecting failure**

```bash
PYTHONPATH=. ./venv/bin/python -m pytest orion/runtime_interrogation/tests/test_human_report.py -v
```

- [ ] **Step 3: Implement `human_report.py`**

```python
from __future__ import annotations

from orion.schemas.runtime_interrogation.v1 import RuntimeInterrogationContractV1


def render_human_report_markdown(contract: RuntimeInterrogationContractV1) -> str:
    lines: list[str] = []
    if contract.fallback.fallback_used:
        lines.append("Fallback mode active: best-effort analysis with reduced certainty.")
        lines.append("")

    lines.append("## What happened")
    lines.append(
        f"- Scope: `{contract.query_scope.mode}` as of `{contract.query_scope.as_of_utc}` "
        f"(session_scope_required={contract.query_scope.session_scope_required})."
    )
    lines.append(
        f"- Correlations resolved ({contract.correlation_resolution.resolution_strategy}): "
        f"{', '.join(contract.correlation_resolution.resolved_correlation_ids) or '—'}."
    )
    lines.append(f"- Authorization: `{contract.correlation_resolution.authorization_decision}` — {contract.correlation_resolution.authorization_reason}.")
    lines.append("")
    lines.append("## Root Cause Summary")
    lines.append(f"- First failing boundary: `{contract.root_cause_summary.first_failing_boundary}`.")
    for r in contract.root_cause_summary.rankings:
        lines.append(
            f"  - **{r.cause_id}** (score={r.score:.2f}, confidence={r.confidence:.2f}): {r.description} "
            f"[evidence: {', '.join(r.evidence_refs) or '—'}]"
        )
    if not contract.root_cause_summary.rankings:
        lines.append("  - No ranked root causes (insufficient evidence in Phase 1 probes).")
    lines.append("")
    lines.append("## Recommended Next Steps")
    for s in contract.recommended_next_steps:
        lines.append(
            f"- **{s.step_id}** [{s.priority}] ({s.action_type}): {s.description} — _{s.expected_outcome}_ "
            f"[evidence: {', '.join(s.evidence_refs) or '—'}]"
        )
    if not contract.recommended_next_steps:
        lines.append("- No automated recommendations; collect additional hop-level traces.")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- **{contract.summary.headline}** (certainty: **{contract.summary.certainty}**).")
    lines.append("")
    lines.append("## Known Unknowns / Degraded Signals")
    lines.append(f"- Guardrail outcome: `{contract.guardrail.guardrail_outcome}`.")
    lines.append(
        f"- Counts: unsupported={contract.guardrail.unsupported_claim_count}, "
        f"blocked={contract.guardrail.blocked_claim_count}, downgraded={contract.guardrail.downgraded_claim_count}."
    )
    if contract.fallback.missing_primitives:
        lines.append(f"- Missing primitives: {', '.join(contract.fallback.missing_primitives)}.")
    lines.append(f"- Redaction applied: {contract.redaction_applied} ({', '.join(contract.redaction_rules_triggered) or 'none'}).")
    return "\n".join(lines).strip() + "\n"
```

- [ ] **Step 4: Wire `human_report_markdown` in assembler (Task 6)**  
  (If you strictly commit after each task, call `render_human_report_markdown` from Task 6 and re-run this test after Task 6 sets `human_report_markdown` on the contract.)

- [ ] **Step 5: Commit**

```bash
git add orion/runtime_interrogation/human_report.py orion/runtime_interrogation/tests/test_human_report.py
git commit -m "feat(runtime-interrogation): human report from strict contract only"
```

---

### Task 6: Phase 1 assembler + skill verb adapter

**Files:**
- Create: `orion/runtime_interrogation/phase1_assemble.py`
- Modify: `services/orion-cortex-exec/app/verb_adapters.py` (append new `@verb` class near other `skills.*` verbs)
- Create: `services/orion-cortex-exec/tests/test_interrogate_runtime_state_skill.py`

- [ ] **Step 1: Write failing exec test**

Create `services/orion-cortex-exec/tests/test_interrogate_runtime_state_skill.py`:

```python
import asyncio
import sys
from pathlib import Path
from uuid import uuid4

import pytest

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR.parent))

from orion.core.verbs.base import VerbContext
from orion.schemas.cortex.schemas import ExecutionPlan, PlanExecutionArgs, PlanExecutionRequest, ExecutionStep


@pytest.mark.asyncio
async def test_interrogate_runtime_state_skill_smoke():
    from app import verb_adapters  # noqa: WPS433 — side-effect registration

    plan = ExecutionPlan(
        verb_name="skills.runtime.interrogate_runtime_state.v1",
        label="t",
        description="t",
        category="ExecutiveControl",
        priority="low",
        interruptible=True,
        can_interrupt_others=False,
        timeout_ms=5000,
        max_recursion_depth=0,
        steps=[
            ExecutionStep(
                verb_name="skills.runtime.interrogate_runtime_state.v1",
                step_name="skills.runtime.interrogate_runtime_state.v1",
                description="probe",
                order=0,
                services=[],
                prompt_template=None,
                requires_gpu=False,
                requires_memory=False,
                timeout_ms=5000,
            )
        ],
        metadata={},
    )
    corr = str(uuid4())
    payload = PlanExecutionRequest(
        plan=plan,
        args=PlanExecutionArgs(
            request_id=corr,
            extra={
                "options": {
                    "introspection_operator_capabilities": [],
                    "runtime_interrogation": {
                        "explicit_correlation_ids": [],
                        "target_correlation_id": None,
                        "prior_turns": 0,
                    },
                }
            },
        ),
        context={"metadata": {"session_id": "sess1", "trace_id": "tid1"}},
    )
    ctx = VerbContext(request_id=corr, caller="test", meta={"correlation_id": corr})
    cls = verb_adapters.InterrogateRuntimeStateSkillVerb
    out, effects = await cls().execute(ctx, payload)
    assert out.ok is True
    meta = out.metadata or {}
    sr = meta.get("skill_result") or {}
    assert "strict_contract" in sr
    assert "human_report_markdown" in sr
    assert isinstance(effects, list)
```

Adjust import path if your test harness uses `services/orion-cortex-exec` as cwd with `app` package; if the test fails on import, use the same `importlib` pattern as `services/orion-cortex-exec/tests/test_skill_verbs.py`.

- [ ] **Step 2: Run pytest expecting failure**

```bash
PYTHONPATH=. ./venv/bin/python -m pytest services/orion-cortex-exec/tests/test_interrogate_runtime_state_skill.py -v
```

Expected: `InterrogateRuntimeStateSkillVerb` missing.

- [ ] **Step 3: Implement `phase1_assemble.py`**

```python
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from orion.schemas.runtime_interrogation.v1 import (
    ClaimV1,
    CorrelationResolutionV1,
    EvidenceRecordV1,
    FallbackV1,
    GuardrailReportV1,
    HopStatusV1,
    QueryScopeV1,
    RootCauseRankingV1,
    RootCauseSummaryV1,
    RuntimeInterrogationContractV1,
    SummaryV1,
    TraceContextV1,
    RecommendedNextStepV1,
)
from orion.runtime_interrogation.claim_guardrail import apply_claim_guardrail
from orion.runtime_interrogation.correlation_resolution import CorrelationResolutionInput, resolve_correlation_resolution
from orion.runtime_interrogation.human_report import render_human_report_markdown
from orion.runtime_interrogation.redaction import redact_mapping


def _opts(extra: Dict[str, Any]) -> Dict[str, Any]:
    return extra.get("options") if isinstance(extra.get("options"), dict) else {}


def build_phase1_contract(*, correlation_id: str, metadata: Dict[str, Any], extra: Dict[str, Any]) -> RuntimeInterrogationContractV1:
    opts = _opts(extra)
    ri = opts.get("runtime_interrogation") if isinstance(opts.get("runtime_interrogation"), dict) else {}
    caps = opts.get("introspection_operator_capabilities")
    if not isinstance(caps, list):
        caps = []
    explicit = ri.get("explicit_correlation_ids") if isinstance(ri.get("explicit_correlation_ids"), list) else []
    explicit_ids = [str(x) for x in explicit if str(x).strip()]
    target = ri.get("target_correlation_id")
    prior = int(ri.get("prior_turns") or 0)

    corr_res = resolve_correlation_resolution(
        CorrelationResolutionInput(
            session_correlation_id=correlation_id,
            explicit_correlation_ids=explicit_ids,
            target_correlation_id=str(target).strip() if target else None,
            prior_turns=prior,
            operator_capabilities=[str(x) for x in caps],
        )
    )

    now = datetime.now(timezone.utc).isoformat()
    trace_id = str(metadata.get("trace_id") or "").strip()
    trace = TraceContextV1(trace_ids=[trace_id] if trace_id else [], channel_refs=[], hop_event_refs=[])

    ev_id = "evidence:phase1:intake"
    evidence: List[EvidenceRecordV1] = [
        EvidenceRecordV1(
            evidence_id=ev_id,
            evidence_type="hop_event",
            source="cortex-exec",
            source_id="runtime_interrogation.phase1",
            correlation_id=correlation_id,
            observed_at_utc=now,
            confidence_base=1.0,
            payload={"session_id": str(metadata.get("session_id") or ""), "note": "phase1_stub_probe"},
        )
    ]
    redacted_payload, rules = redact_mapping(evidence[0].payload)
    evidence[0] = evidence[0].model_copy(update={"payload": redacted_payload, "redacted": bool(rules)})

    claims: List[ClaimV1] = [
        ClaimV1(
            claim_id="claim:intake_context_present",
            statement="Executor received a Phase 1 runtime interrogation request with correlation id and optional trace id.",
            supported="true",
            confidence=0.95,
            evidence_refs=[ev_id],
            selection_reason="intake_metadata_only",
        )
    ]

    contract = RuntimeInterrogationContractV1(
        query_scope=QueryScopeV1(mode="current_turn", session_scope_required=True, as_of_utc=now),
        correlation_resolution=corr_res,
        hop_status=HopStatusV1(),
        trace_context=trace,
        evidence=evidence,
        claims=claims,
        fallback=FallbackV1(fallback_level="partial", fallback_used=True, fallback_reason="phase1_probe_stub", missing_primitives=["hub_gateway_orch_exec_llm_state_mind_probes"]),
        root_cause_summary=RootCauseSummaryV1(
            first_failing_boundary="unknown",
            rankings=[
                RootCauseRankingV1(
                    cause_id="rc:insufficient_hop_probes",
                    description="Phase 1 ships contract+guardrails; hop-level probes mostly not wired.",
                    score=0.2,
                    confidence=0.55,
                    evidence_refs=[ev_id],
                )
            ],
        ),
        recommended_next_steps=[
            RecommendedNextStepV1(
                step_id="next:widen_probes",
                action_type="primitive_missing",
                description="Implement state-service and verb-rail readers for hop_status fields.",
                expected_outcome="Replace unknown hop states with ok/degraded/fail based on evidence.",
                evidence_refs=[ev_id],
                priority="high",
            )
        ],
        summary=SummaryV1(headline="Phase 1 runtime interrogation contract assembled (stub probes).", certainty="low"),
        redaction_applied=bool(rules),
        redaction_rules_triggered=rules,
        guardrail=GuardrailReportV1(),
    )
    if corr_res.authorization_decision == "deny":
        contract = contract.model_copy(
            update={
                "claims": [],
                "evidence": [],
                "fallback": FallbackV1(
                    fallback_level="insufficient_evidence",
                    fallback_used=True,
                    fallback_reason="authorization_denied",
                    missing_primitives=["correlation_scope"],
                    degraded_hops=[],
                    budget_clipped=False,
                ),
                "summary": SummaryV1(headline="Introspection denied by policy (cross-scope).", certainty="high"),
            }
        )

    contract = apply_claim_guardrail(contract)
    md = render_human_report_markdown(contract)
    return contract.model_copy(update={"human_report_markdown": md})
```

- [ ] **Step 4: Add verb class to `verb_adapters.py`**

Append (imports at top of file: add `InterrogateRuntimeStateSkillVerb` dependencies — reuse existing imports for `BaseVerb`, `PlanExecutionRequest`, `SkillVerbOutput`, `_skill_result_output`, `_metadata_from_payload`):

```python
from orion.runtime_interrogation.phase1_assemble import build_phase1_contract


@verb("skills.runtime.interrogate_runtime_state.v1")
class InterrogateRuntimeStateSkillVerb(BaseVerb[PlanExecutionRequest, SkillVerbOutput]):
    input_model = PlanExecutionRequest
    output_model = SkillVerbOutput

    async def execute(self, ctx: VerbContext, payload: PlanExecutionRequest) -> Tuple[SkillVerbOutput, List[VerbEffectV1]]:
        correlation_id = str(ctx.meta.get("correlation_id") or payload.args.request_id or "")
        metadata = _metadata_from_payload(payload)
        extra = payload.args.extra if isinstance(payload.args.extra, dict) else {}
        strict = build_phase1_contract(correlation_id=correlation_id, metadata=metadata, extra=extra)
        result = {
            "strict_contract": strict.model_dump(mode="json"),
            "human_report_markdown": strict.human_report_markdown,
        }
        return _skill_result_output(skill_name="skills.runtime.interrogate_runtime_state.v1", result=result), []
```

- [ ] **Step 5: Run pytest**

```bash
PYTHONPATH=. ./venv/bin/python -m pytest services/orion-cortex-exec/tests/test_interrogate_runtime_state_skill.py -v
```

- [ ] **Step 6: Commit**

```bash
git add orion/runtime_interrogation/phase1_assemble.py services/orion-cortex-exec/app/verb_adapters.py services/orion-cortex-exec/tests/test_interrogate_runtime_state_skill.py
git commit -m "feat(runtime-interrogation): phase1 assembler and cortex-exec skill verb"
```

---

### Task 7: Cognition YAML verbs + capability pin

**Files:**
- Create: `orion/cognition/verbs/interrogate_runtime_state.yaml`
- Create: `orion/cognition/verbs/skills.runtime.interrogate_runtime_state.v1.yaml`
- Modify: `services/orion-agent-chain/app/capability_bridge.py`

- [ ] **Step 1: Add YAML files**

`orion/cognition/verbs/interrogate_runtime_state.yaml` (copy structure from `assess_runtime_state.yaml`, change name and descriptions):

```yaml
name: interrogate_runtime_state
label: Interrogate Runtime State
description: >
  Operator-grade runtime introspection: deterministic evidence assembly, strict
  contract with claim guardrails, and human-readable report derived only from that contract.
category: ExecutiveControl
priority: high
interruptible: true
can_interrupt_others: false
requires_gpu: false
requires_memory: false
timeout_ms: 120000
max_recursion_depth: 0
services: []

execution_mode: capability_backed
requires_capability_selector: true
preferred_skill_families:
  - system_inspection
side_effect_level: none

input_schema:
  type: object
  properties:
    text:
      type: string
  required: [text]

output_schema:
  type: object
  properties:
    strict_contract:
      type: object
    human_report_markdown:
      type: string
```

`orion/cognition/verbs/skills.runtime.interrogate_runtime_state.v1.yaml`:

```yaml
name: skills.runtime.interrogate_runtime_state.v1
label: Skills — Runtime interrogation (Phase 1)
description: Deterministic Phase 1 runtime interrogation skill executed in cortex-exec verb adapters.
category: ExecutiveControl
priority: high
interruptible: true
can_interrupt_others: false
requires_gpu: false
requires_memory: false
timeout_ms: 120000
max_recursion_depth: 0
services: []
plan:
  - name: skills.runtime.interrogate_runtime_state.v1
    description: Assemble strict runtime interrogation contract (Phase 1).
    order: 0
    services: []
```

- [ ] **Step 2: Pin semantic verb in capability bridge**

In `services/orion-agent-chain/app/capability_bridge.py`, extend `_SEMANTIC_VERB_TO_SKILL`:

```python
_SEMANTIC_VERB_TO_SKILL: dict[str, str] = {
    "answer_current_datetime": "skills.system.time_now.v1",
    "inspect_gpu_status": "skills.gpu.nvidia_smi_snapshot.v1",
    "show_biometrics_snapshot": "skills.biometrics.snapshot.v1",
    "list_biometrics_recent_readings": "skills.biometrics.raw_recent.v1",
    "inspect_docker_container_status": "skills.docker.ps_status.v1",
    "send_operator_notification": "skills.system.notify_chat_message.v1",
    "show_landing_pad_metrics": "skills.landing_pad.metrics_snapshot.v1",
    "interrogate_runtime_state": "skills.runtime.interrogate_runtime_state.v1",
}
```

- [ ] **Step 3: Extend semantic pin tests**

Modify `services/orion-agent-chain/tests/test_capability_semantic_pins.py`:

1. In `test_semantic_verb_pins_cover_prompt_skills`, add:

```python
    assert _SEMANTIC_VERB_TO_SKILL["interrogate_runtime_state"] == "skills.runtime.interrogate_runtime_state.v1"
```

2. The existing `test_resolve_capability_decision_uses_pins` loop already iterates `_SEMANTIC_VERB_TO_SKILL`; after adding the map entry it will automatically assert the new skill resolves with confidence `1.0`.

Run:

```bash
cd /mnt/scripts/Orion-Sapienform/services/orion-agent-chain
PYTHONPATH=../../:.. pytest tests/test_capability_semantic_pins.py -v
```

(If your environment uses the repo root instead: `PYTHONPATH=/mnt/scripts/Orion-Sapienform:/mnt/scripts/Orion-Sapienform/services/orion-agent-chain`.)

- [ ] **Step 4: Commit**

```bash
git add orion/cognition/verbs/interrogate_runtime_state.yaml orion/cognition/verbs/skills.runtime.interrogate_runtime_state.v1.yaml services/orion-agent-chain/app/capability_bridge.py services/orion-agent-chain/tests/test_capability_semantic_pins.py
git commit -m "feat(runtime-interrogation): verb YAML and capability bridge pin"
```

---

### Task 8: Hub routing (feature flag + shadow mode)

**Files:**
- Create: `services/orion-hub/scripts/runtime_introspection_routing.py`
- Modify: `services/orion-hub/scripts/chat_request_builder.py`
- Modify: `services/orion-hub/app/settings.py`
- Modify: `services/orion-hub/.env_example`
- Create: `services/orion-hub/tests/test_runtime_introspection_routing.py`

- [ ] **Step 1: Write failing hub test**

Create `services/orion-hub/tests/test_runtime_introspection_routing.py` using the same **importlib file-location** pattern as `services/orion-hub/tests/test_cortex_request_builder.py` (the Hub test suite does not always have `scripts` on `sys.path` as a package). Load `chat_request_builder.py` into a module named e.g. `hub_chat_request_builder`, then patch attributes on that loaded module object:

```python
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from orion.schemas.cortex.contracts import CortexChatRequest

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "services" / "orion-hub" / "scripts" / "chat_request_builder.py"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SPEC = importlib.util.spec_from_file_location("hub_chat_request_builder", MODULE_PATH)
assert SPEC and SPEC.loader
hub_chat_request_builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hub_chat_request_builder)


def test_routes_when_flag_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hub_chat_request_builder.settings, "HUB_RUNTIME_INTROSPECTION_ROUTING_ENABLED", True, raising=False)
    monkeypatch.setattr(hub_chat_request_builder.settings, "HUB_RUNTIME_INTROSPECTION_SHADOW_MODE", False, raising=False)

    req, _dbg, _ = hub_chat_request_builder.build_cortex_chat_request(
        prompt="Why did you answer that way last turn?",
        payload={"mode": "brain"},
        session_id="s1",
        user_id="u1",
        trace_id="t1",
        source_label="test",
    )
    assert isinstance(req, CortexChatRequest)
    assert req.mode == "agent"
    assert req.verb == "interrogate_runtime_state"
    ro = req.recall or {}
    assert ro.get("enabled") is False


def test_no_route_when_flag_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hub_chat_request_builder.settings, "HUB_RUNTIME_INTROSPECTION_ROUTING_ENABLED", False, raising=False)

    req, _dbg, _ = hub_chat_request_builder.build_cortex_chat_request(
        prompt="Why did you answer that way last turn?",
        payload={"mode": "brain"},
        session_id="s1",
        user_id="u1",
        trace_id="t1",
        source_label="test",
    )
    assert req.mode == "brain"
```

- [ ] **Step 2: Run pytest expecting failure**

```bash
PYTHONPATH=. ./venv/bin/python -m pytest services/orion-hub/tests/test_runtime_introspection_routing.py -v
```

- [ ] **Step 3: Implement routing helper**

Create `services/orion-hub/scripts/runtime_introspection_routing.py`:

```python
from __future__ import annotations

import re
from typing import Any, Dict


_PATTERNS = (
    re.compile(r"\bwhy did (you|orion|the model)\b", re.I),
    re.compile(r"\bwhy was (my|the) (answer|reply|response)\b", re.I),
    re.compile(r"\bwhat route\b.*\b(take|did)\b", re.I),
    re.compile(r"\bruntime introspection\b", re.I),
    re.compile(r"\b(show|explain) (the )?(trace|routing)\b", re.I),
)


def looks_like_runtime_introspection(prompt: str) -> bool:
    t = str(prompt or "").strip()
    if len(t) < 12:
        return False
    return any(p.search(t) for p in _PATTERNS)


def apply_introspection_options(options: Dict[str, Any], *, shadow: bool) -> None:
    opts = options
    opts["runtime_introspection_routed"] = True
    if shadow:
        opts["runtime_interrogation_shadow_only"] = True
```

Add to `services/orion-hub/app/settings.py` inside `Settings`:

```python
    HUB_RUNTIME_INTROSPECTION_ROUTING_ENABLED: bool = Field(default=False, alias="HUB_RUNTIME_INTROSPECTION_ROUTING_ENABLED")
    HUB_RUNTIME_INTROSPECTION_SHADOW_MODE: bool = Field(default=False, alias="HUB_RUNTIME_INTROSPECTION_SHADOW_MODE")
```

In `services/orion-hub/scripts/chat_request_builder.py`, the smallest safe integration is to apply overrides **immediately before** `req = CortexChatRequest(...)` (after `debug` fields are known), so you do not fight ordering between `selected_verbs`, `recall_payload`, and `options`:

```python
from . import runtime_introspection_routing as rir

# immediately before `req = CortexChatRequest(...)`
if settings.HUB_RUNTIME_INTROSPECTION_ROUTING_ENABLED and rir.looks_like_runtime_introspection(prompt):
    mode = "agent"
    verb_override = "interrogate_runtime_state"
    recall_payload = {"enabled": False, "required": False, "mode": "hybrid"}
    use_recall = False
    rir.apply_introspection_options(options, shadow=bool(settings.HUB_RUNTIME_INTROSPECTION_SHADOW_MODE))
```

If you prefer an earlier hook, you may instead introduce `verb_override = None` right after `mode = _normalize_mode(...)` and wrap the `selected_verbs` block with `if verb_override is None:` — but the pre-`CortexChatRequest` override keeps the diff minimal and avoids partial state.

Append to `services/orion-hub/.env_example`:

```bash
# Runtime introspection interrogation (Phase 1 Hub routing)
HUB_RUNTIME_INTROSPECTION_ROUTING_ENABLED=false
HUB_RUNTIME_INTROSPECTION_SHADOW_MODE=false
```

- [ ] **Step 4: Run pytest**

```bash
PYTHONPATH=. ./venv/bin/python -m pytest services/orion-hub/tests/test_runtime_introspection_routing.py -v
```

- [ ] **Step 5: Commit**

```bash
git add services/orion-hub/scripts/runtime_introspection_routing.py services/orion-hub/scripts/chat_request_builder.py services/orion-hub/app/settings.py services/orion-hub/.env_example services/orion-hub/tests/test_runtime_introspection_routing.py
git commit -m "feat(hub): optional routing for runtime introspection queries"
```

---

### Task 9: Integration follow-ups (post Phase 1 contract)

These are **explicitly after** the contract path is green; keep separate commits.

- [ ] **Wire real hop probes** starting with `StateServiceClient` usage patterns in `services/orion-cortex-orch/app/orchestrator.py` and bus RPC conventions documented in `docs/platform_routing_wiring_map.md`: populate `hop_status` and `evidence[]` from `orion:state:request/reply` and executor step metadata where available.

- [ ] **Extend `CortexClientResult.metadata`** in `services/orion-cortex-orch/app/main.py` result assembly to pass through `strict_contract` when `options.runtime_interrogation_shadow_only` is true (Hub shadow mode): user-visible `final_text` stays unchanged while operators read metadata from Hub debug panels.

- [ ] **Planner prompt tuning** (optional): add a short tool description hint in `services/orion-planner-react` tool manifests so `interrogate_runtime_state` is chosen without user typing the verb explicitly when `allowed_verbs` constrains the planner — only if product requires it.

---

## Self-review (spec coverage)

1. **Spec coverage mapping**
   - Dedicated path / verb id `interrogate_runtime_state`: Tasks 6–7.
   - Strict machine-readable contract: Tasks 1, 6.
   - Human report derived from strict only: Tasks 5–6.
   - Anti-fabrication guardrail: Task 4.
   - Correlation traversal + gating + caps: Task 2 (`MAX_*` constants).
   - Redaction policy fields: Tasks 3, 6 (`redaction_applied`, `redaction_rules_triggered`).
   - Fallback semantics + banner: Tasks 5–6 (`fallback` + renderer).
   - Hub routing away from `chat_general` for introspection-shaped prompts: Task 8.
   - Rollout feature flag + shadow: Task 8 (`HUB_*` settings).
   - Multi-hop evidence (full inventory): Task 9 (deferred from Phase 1 minimum).
   - Phase 2 cognition surfaces / Phase 3 multi-correlation compare: not in this plan — separate plans per spec phased delivery.

2. **Placeholder scan:** None intentionally; all steps include concrete code or file paths.

3. **Type consistency:** `ClaimV1.supported` uses string literals `"true"|"false"|"partial"` to match spec YAML wording; guardrail and tests must use the same literals.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-14-runtime-introspection-interrogation-phase1.md`. Two execution options:

**1. Subagent-Driven (recommended)** — Dispatch a fresh subagent per task, review between tasks, fast iteration. **REQUIRED SUB-SKILL:** superpowers:subagent-driven-development.

**2. Inline Execution** — Execute tasks in this session using superpowers:executing-plans with checkpoints between tasks.

Which approach?
