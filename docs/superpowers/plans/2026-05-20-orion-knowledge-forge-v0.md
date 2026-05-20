# Orion Knowledge Forge v0 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a markdown-and-git **research-to-execution compiler** (`orion-knowledge/` corpus + `orion/knowledge_forge/` Python tools) that ingests raw sources into atomic claims, gates mutations to execution-facing artifacts through a review queue, and emits task-specific **context packs** for Cursor/Codex — without building a wiki cathedral or silent LLM spec rewrites.

**Architecture:** Immutable `raw/` sources are authority. Claims in `orion-knowledge/claims/{accepted,disputed,stale,superseded}/` are the atomic layer with typed relations (`supports`, `contradicts`, `supersedes`, `depends_on`, `implements`, `tested_by`, `blocked_by`, `motivated_by`). Decisions (ADRs), specs, and context packs are separate layers; the LLM may only **propose** patches to execution-facing files via `reviews/pending/`. A small Python CLI (`python -m orion.knowledge_forge`) loads the corpus, lints schemas and references, applies approved reviews, and compiles context packs from **accepted claims + reviewed/execution_ready specs only**. Wiki pages under `wiki/` are disposable compiled views; they are never fed wholesale to dev agents.

**Tech Stack:** Python 3.12, Pydantic v2, PyYAML (already in repo), pathlib, Jinja2 (optional for context-pack templates; inline string templates acceptable in v0), pytest at repo root (`PYTHONPATH=.` + `./venv/bin/python`).

**Design source:** Operator brief “Orion Knowledge Forge” (2026-05-20) — claim-first LLMWiki-shaped compiler, not a concept wiki.

**Explicit v0 non-goals:** LlamaIndex ingestion, LangGraph HITL workflows, MCP server, automated LLM ingest loop (prompt templates + manual/agent-assisted ingest only), GraphDB materialization (see `docs/superpowers/specs/2026-05-04-orion-meta-services-architecture-graph-design.md` for future bridge).

---

## File structure

| Path | Responsibility |
|------|----------------|
| `orion-knowledge/` | Git-tracked knowledge corpus (raw, claims, decisions, specs, context_packs, reviews, wiki, evals). |
| `orion-knowledge/AGENTS.md` | LLMWiki-style schema + ingest/compile rules for agents operating on the corpus. |
| `orion-knowledge/README.md` | Human orientation: authority layers, v0 workflow, CLI commands. |
| `orion/knowledge_forge/__init__.py` | Package marker. |
| `orion/knowledge_forge/models.py` | Pydantic v2 models: `SourceV1`, `ClaimV1`, `DecisionV1`, `SpecV1`, `ContextPackV1`, typed relation enums. |
| `orion/knowledge_forge/paths.py` | Resolve corpus root (`ORION_KNOWLEDGE_ROOT` env or repo-relative default). |
| `orion/knowledge_forge/yaml_doc.py` | Load/save single YAML documents with stable key ordering for diffs. |
| `orion/knowledge_forge/store.py` | Index all primitives by `id`, list by status/component, resolve cross-refs. |
| `orion/knowledge_forge/lint.py` | Schema validation, dangling refs, banned untyped links, execution-facing mutation guard. |
| `orion/knowledge_forge/review.py` | Parse pending patch files, apply accepted patches to target paths, move to `reviews/accepted/`. |
| `orion/knowledge_forge/compile.py` | Build markdown context packs from spec + accepted claims + file hints. |
| `orion/knowledge_forge/probes.py` | Source→claim coverage probe (v0 eval gate). |
| `orion/knowledge_forge/cli.py` | `lint`, `review list`, `review apply`, `compile context-pack`, `probe source`. |
| `orion/knowledge_forge/__main__.py` | `python -m orion.knowledge_forge` entry. |
| `tests/test_knowledge_forge_models.py` | Model validation unit tests. |
| `tests/test_knowledge_forge_store.py` | Store indexing + lookup tests. |
| `tests/test_knowledge_forge_lint.py` | Lint rules with fixture corpus. |
| `tests/test_knowledge_forge_review.py` | Review apply/reject tests. |
| `tests/test_knowledge_forge_compile.py` | Context pack compiler tests. |
| `tests/test_knowledge_forge_golden_path.py` | End-to-end: fixture source → claims → spec → context pack. |
| `tests/fixtures/knowledge_forge/` | Minimal in-repo fixture corpus for tests (not the live `orion-knowledge/` tree). |

**Relationship to existing docs:**

- `docs/superpowers/specs/` and `docs/superpowers/plans/` remain the historical archive during v0.
- New execution-facing design intent should land in `orion-knowledge/specs/` once the forge is live.
- Add a single bridge claim in Task 13 pointing at `docs/superpowers/specs/2026-05-14-substrate-tier-telemetry-persistence-design.md` as a migrated example.

**Authority rule (normative):**

```text
raw sources → claims → decisions/specs → context packs → code/tests
```

Wiki pages are compiled maps, not authority.

---

### Task 1: Corpus directory scaffold

**Files:**

- Create: `orion-knowledge/README.md`
- Create: `orion-knowledge/wiki/index.md`
- Create: `orion-knowledge/wiki/log.md`
- Create: `orion-knowledge/raw/sources/.gitkeep`
- Create: `orion-knowledge/raw/conversations/.gitkeep`
- Create: `orion-knowledge/claims/accepted/.gitkeep`
- Create: `orion-knowledge/claims/disputed/.gitkeep`
- Create: `orion-knowledge/claims/stale/.gitkeep`
- Create: `orion-knowledge/claims/superseded/.gitkeep`
- Create: `orion-knowledge/decisions/.gitkeep`
- Create: `orion-knowledge/specs/design/.gitkeep`
- Create: `orion-knowledge/specs/plan/.gitkeep`
- Create: `orion-knowledge/specs/execution_ready/.gitkeep`
- Create: `orion-knowledge/context_packs/cursor/.gitkeep`
- Create: `orion-knowledge/context_packs/codex/.gitkeep`
- Create: `orion-knowledge/context_packs/orion/.gitkeep`
- Create: `orion-knowledge/reviews/pending/.gitkeep`
- Create: `orion-knowledge/reviews/accepted/.gitkeep`
- Create: `orion-knowledge/reviews/rejected/.gitkeep`
- Create: `orion-knowledge/evals/probes/.gitkeep`
- Create: `orion-knowledge/evals/lint_reports/.gitkeep`

- [ ] **Step 1: Create README with authority layers**

Create `orion-knowledge/README.md`:

```markdown
# Orion Knowledge Forge

Research-to-execution compiler for Orion design and implementation context.

## Authority layers (strongest → weakest)

1. **Raw sources** (`raw/`) — immutable inputs; never edited after ingest
2. **Claims** (`claims/`) — atomic, source-backed statements
3. **Decisions** (`decisions/`) — ADRs with rationale
4. **Specs** (`specs/`) — reviewed design/plan intent
5. **Context packs** (`context_packs/`) — task bundles for agents
6. **Wiki** (`wiki/`) — human-readable compiled views (disposable)

Code and tests are ground truth for runtime behavior.

## v0 workflow

1. Drop source: `raw/sources/YYYY-MM-DD-topic.md`
2. Extract claims (agent-assisted using `AGENTS.md` ingest rules)
3. Propose spec/decision patches → `reviews/pending/` (never direct overwrite of `execution_ready/`)
4. Human approves: `python -m orion.knowledge_forge review apply <patch_id>`
5. Compile pack: `python -m orion.knowledge_forge compile context-pack --spec spec:... --out context_packs/cursor/...`
6. Hand pack to Cursor/Codex — not the whole wiki

## CLI

```bash
cd /path/to/Orion-Sapienform
PYTHONPATH=. ./venv/bin/python -m orion.knowledge_forge lint
PYTHONPATH=. ./venv/bin/python -m orion.knowledge_forge review list
PYTHONPATH=. ./venv/bin/python -m orion.knowledge_forge compile context-pack --spec spec:substrate-tier-telemetry-v1
```
```

- [ ] **Step 2: Create empty wiki index and log**

Create `orion-knowledge/wiki/index.md`:

```markdown
# Orion Knowledge Forge — Index

| Area | Path |
|------|------|
| Accepted claims | `claims/accepted/` |
| Disputed claims | `claims/disputed/` |
| Decisions (ADRs) | `decisions/` |
| Design specs | `specs/design/` |
| Execution-ready specs | `specs/execution_ready/` |
| Cursor context packs | `context_packs/cursor/` |
| Pending reviews | `reviews/pending/` |
```

Create `orion-knowledge/wiki/log.md`:

```markdown
# Forge log

| Date | Event | Ref |
|------|-------|-----|
| 2026-05-20 | Initialized Orion Knowledge Forge v0 scaffold | — |
```

- [ ] **Step 3: Create `.gitkeep` files**

Run:

```bash
cd /mnt/scripts/Orion-Sapienform
mkdir -p orion-knowledge/{raw/{sources,conversations,papers,screenshots,transcripts},wiki/{concepts,systems,components,research_threads},claims/{accepted,disputed,stale,superseded},decisions,specs/{design,plan,execution_ready},context_packs/{cursor,codex,orion},reviews/{pending,accepted,rejected},evals/{probes,lint_reports}}
touch orion-knowledge/raw/sources/.gitkeep orion-knowledge/claims/accepted/.gitkeep orion-knowledge/reviews/pending/.gitkeep
```

- [ ] **Step 4: Commit scaffold**

```bash
git add orion-knowledge/
git commit -m "feat: scaffold orion-knowledge corpus for Knowledge Forge v0"
```

---

### Task 2: Pydantic primitives (five document types)

**Files:**

- Create: `orion/knowledge_forge/__init__.py`
- Create: `orion/knowledge_forge/models.py`
- Test: `tests/test_knowledge_forge_models.py`

- [ ] **Step 1: Write failing model tests**

Create `tests/test_knowledge_forge_models.py`:

```python
from __future__ import annotations

import pytest
from pydantic import ValidationError

from orion.knowledge_forge.models import (
    ClaimStatusV1,
    ClaimV1,
    ContextPackV1,
    DecisionV1,
    SourceV1,
    SpecV1,
    SpecStatusV1,
)


def test_claim_requires_typed_id_and_statement() -> None:
    claim = ClaimV1.model_validate(
        {
            "type": "claim",
            "id": "claim:orion:recall-routing:0007",
            "statement": "chat_general should be the canonical speech step for thought capture.",
            "status": "accepted",
            "source_refs": ["source:2026-05-20-recall-chat"],
            "confidence": "high",
        }
    )
    assert claim.status == ClaimStatusV1.accepted


def test_claim_rejects_bare_related_links() -> None:
    with pytest.raises(ValidationError):
        ClaimV1.model_validate(
            {
                "type": "claim",
                "id": "claim:orion:bad:0001",
                "statement": "bad",
                "status": "speculative",
                "source_refs": [],
                "related": ["claim:other"],
            }
        )


def test_spec_execution_ready_requires_acceptance_tests() -> None:
    with pytest.raises(ValidationError):
        SpecV1.model_validate(
            {
                "type": "spec",
                "id": "spec:substrate-tier-telemetry-v1",
                "status": "execution_ready",
                "component": "orion-substrate-telemetry",
                "requirements": ["persist tier outcomes"],
                "non_goals": [],
                "acceptance_tests": [],
                "source_claims": [],
            }
        )


def test_context_pack_requires_target_and_task() -> None:
    pack = ContextPackV1.model_validate(
        {
            "type": "context_pack",
            "id": "ctx:substrate-tier-telemetry-v1",
            "target": "cursor",
            "task": "Implement substrate tier telemetry persistence v0",
            "included_specs": ["spec:substrate-tier-telemetry-v1"],
            "allowed_sources": [],
            "excluded_context": ["orion-meta-services v2 graph automation"],
        }
    )
    assert pack.target.value == "cursor"
```

- [ ] **Step 2: Run test — expect import failure**

```bash
cd /mnt/scripts/Orion-Sapienform
PYTHONPATH=. ./venv/bin/python -m pytest tests/test_knowledge_forge_models.py -v
```

Expected: `ModuleNotFoundError: orion.knowledge_forge`

- [ ] **Step 3: Implement models**

Create `orion/knowledge_forge/__init__.py`:

```python
"""Orion Knowledge Forge — claim-first research-to-execution compiler."""
```

Create `orion/knowledge_forge/models.py`:

```python
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ClaimStatusV1(str, Enum):
    accepted = "accepted"
    disputed = "disputed"
    stale = "stale"
    superseded = "superseded"
    speculative = "speculative"


class SpecStatusV1(str, Enum):
    draft = "draft"
    reviewed = "reviewed"
    execution_ready = "execution_ready"
    implemented = "implemented"
    stale = "stale"


class DecisionStatusV1(str, Enum):
    proposed = "proposed"
    accepted = "accepted"
    superseded = "superseded"


class ContextPackTargetV1(str, Enum):
    cursor = "cursor"
    codex = "codex"
    orion_agent = "orion-agent"
    human = "human"


class SourceKindV1(str, Enum):
    conversation = "conversation"
    paper = "paper"
    code = "code"
    issue = "issue"
    design_doc = "design_doc"


class TrustLevelV1(str, Enum):
    primary = "primary"
    secondary = "secondary"
    speculative = "speculative"


class TypedRelationsV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    supports: list[str] = Field(default_factory=list)
    contradicts: list[str] = Field(default_factory=list)
    supersedes: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    implements: list[str] = Field(default_factory=list)
    tested_by: list[str] = Field(default_factory=list)
    blocked_by: list[str] = Field(default_factory=list)
    motivated_by: list[str] = Field(default_factory=list)


class SourceV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["source"]
    id: str
    kind: SourceKindV1
    path: str
    trust_level: TrustLevelV1 = TrustLevelV1.primary


class ClaimV1(TypedRelationsV1):
    model_config = ConfigDict(extra="forbid")

    type: Literal["claim"]
    id: str
    statement: str
    status: ClaimStatusV1
    source_refs: list[str]
    confidence: Literal["high", "medium", "low"] = "medium"
    used_by: list[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def claim_id_prefix(cls, value: str) -> str:
        if not value.startswith("claim:"):
            raise ValueError("claim id must start with claim:")
        return value


class DecisionV1(TypedRelationsV1):
    model_config = ConfigDict(extra="forbid")

    type: Literal["decision"]
    id: str
    status: DecisionStatusV1
    decision: str
    rationale: str
    consequences: list[str] = Field(default_factory=list)
    source_claims: list[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def decision_id_prefix(cls, value: str) -> str:
        if not value.startswith("adr:"):
            raise ValueError("decision id must start with adr:")
        return value


class SpecV1(TypedRelationsV1):
    model_config = ConfigDict(extra="forbid")

    type: Literal["spec"]
    id: str
    status: SpecStatusV1
    component: str
    requirements: list[str]
    non_goals: list[str] = Field(default_factory=list)
    acceptance_tests: list[str] = Field(default_factory=list)
    source_claims: list[str] = Field(default_factory=list)
    likely_files: list[str] = Field(default_factory=list)
    known_traps: list[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def spec_id_prefix(cls, value: str) -> str:
        if not value.startswith("spec:"):
            raise ValueError("spec id must start with spec:")
        return value

    @model_validator(mode="after")
    def execution_ready_needs_tests(self) -> SpecV1:
        if self.status == SpecStatusV1.execution_ready and not self.acceptance_tests:
            raise ValueError("execution_ready specs require acceptance_tests")
        return self


class ContextPackV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["context_pack"]
    id: str
    target: ContextPackTargetV1
    task: str
    allowed_sources: list[str] = Field(default_factory=list)
    included_specs: list[str] = Field(default_factory=list)
    excluded_context: list[str] = Field(default_factory=list)
    included_claim_ids: list[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def context_pack_id_prefix(cls, value: str) -> str:
        if not value.startswith("ctx:"):
            raise ValueError("context pack id must start with ctx:")
        return value
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
PYTHONPATH=. ./venv/bin/python -m pytest tests/test_knowledge_forge_models.py -v
```

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add orion/knowledge_forge/ tests/test_knowledge_forge_models.py
git commit -m "feat: add Knowledge Forge pydantic document models"
```

---

### Task 3: Corpus path resolution + YAML loader

**Files:**

- Create: `orion/knowledge_forge/paths.py`
- Create: `orion/knowledge_forge/yaml_doc.py`
- Test: `tests/test_knowledge_forge_store.py` (first half — loader only)

- [ ] **Step 1: Write failing loader test**

Append to `tests/test_knowledge_forge_store.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from orion.knowledge_forge.paths import resolve_corpus_root
from orion.knowledge_forge.yaml_doc import load_yaml_doc, save_yaml_doc


def test_resolve_corpus_root_defaults_to_repo_orion_knowledge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_repo = tmp_path / "repo"
    corpus = fake_repo / "orion-knowledge"
    corpus.mkdir(parents=True)
    monkeypatch.chdir(fake_repo)
    monkeypatch.delenv("ORION_KNOWLEDGE_ROOT", raising=False)
    assert resolve_corpus_root() == corpus.resolve()


def test_yaml_roundtrip_preserves_keys(tmp_path: Path) -> None:
    path = tmp_path / "claim.yaml"
    doc = {"type": "claim", "id": "claim:test:0001", "statement": "x", "status": "accepted", "source_refs": []}
    save_yaml_doc(path, doc)
    loaded = load_yaml_doc(path)
    assert loaded == doc
```

- [ ] **Step 2: Run test — expect failure**

```bash
PYTHONPATH=. ./venv/bin/python -m pytest tests/test_knowledge_forge_store.py::test_yaml_roundtrip_preserves_keys -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement paths + yaml_doc**

Create `orion/knowledge_forge/paths.py`:

```python
from __future__ import annotations

import os
from pathlib import Path


def resolve_corpus_root() -> Path:
    env = os.environ.get("ORION_KNOWLEDGE_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    # Walk up from cwd looking for orion-knowledge/
    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents]:
        root = candidate / "orion-knowledge"
        if root.is_dir():
            return root.resolve()
    return (cwd / "orion-knowledge").resolve()
```

Create `orion/knowledge_forge/yaml_doc.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_doc(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError(f"YAML document must be a mapping: {path}")
    return data


def save_yaml_doc(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)
    path.write_text(text, encoding="utf-8")
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
PYTHONPATH=. ./venv/bin/python -m pytest tests/test_knowledge_forge_store.py -v
```

- [ ] **Step 5: Commit**

```bash
git add orion/knowledge_forge/paths.py orion/knowledge_forge/yaml_doc.py tests/test_knowledge_forge_store.py
git commit -m "feat: add Knowledge Forge corpus path and YAML helpers"
```

---

### Task 4: Document store index

**Files:**

- Create: `orion/knowledge_forge/store.py`
- Modify: `tests/test_knowledge_forge_store.py`
- Create: `tests/fixtures/knowledge_forge/claims/accepted/claim-test-0001.yaml`

- [ ] **Step 1: Add fixture claim**

Create `tests/fixtures/knowledge_forge/claims/accepted/claim-test-0001.yaml`:

```yaml
type: claim
id: claim:test:0001
statement: Fixture claim for store indexing.
status: accepted
source_refs:
  - source:test:fixture
confidence: high
supports: []
contradicts: []
supersedes: []
depends_on: []
implements: []
tested_by: []
blocked_by: []
motivated_by: []
used_by: []
```

- [ ] **Step 2: Write failing store test**

Append to `tests/test_knowledge_forge_store.py`:

```python
from orion.knowledge_forge.store import KnowledgeStore


def test_store_loads_claims_from_fixture_corpus() -> None:
    fixture_root = Path(__file__).resolve().parent / "fixtures" / "knowledge_forge"
    store = KnowledgeStore(fixture_root)
    store.load()
    claim = store.get("claim:test:0001")
    assert claim is not None
    assert claim.statement.startswith("Fixture claim")
```

- [ ] **Step 3: Run test — expect failure**

```bash
PYTHONPATH=. ./venv/bin/python -m pytest tests/test_knowledge_forge_store.py::test_store_loads_claims_from_fixture_corpus -v
```

- [ ] **Step 4: Implement store**

Create `orion/knowledge_forge/store.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from orion.knowledge_forge.models import (
    ClaimV1,
    ContextPackV1,
    DecisionV1,
    SourceV1,
    SpecV1,
)
from orion.knowledge_forge.yaml_doc import load_yaml_doc

DocModel = SourceV1 | ClaimV1 | DecisionV1 | SpecV1 | ContextPackV1

_MODEL_BY_TYPE: dict[str, type[DocModel]] = {
    "source": SourceV1,
    "claim": ClaimV1,
    "decision": DecisionV1,
    "spec": SpecV1,
    "context_pack": ContextPackV1,
}


@dataclass
class KnowledgeStore:
    root: Path
    by_id: dict[str, DocModel] = field(default_factory=dict)
    paths_by_id: dict[str, Path] = field(default_factory=dict)

    def load(self) -> None:
        self.by_id.clear()
        self.paths_by_id.clear()
        for path in sorted(self.root.rglob("*.yaml")):
            if "reviews/" in path.as_posix():
                continue
            doc = load_yaml_doc(path)
            typed = self._coerce(doc)
            if typed.id in self.by_id:
                raise ValueError(f"duplicate id {typed.id} in {path}")
            self.by_id[typed.id] = typed
            self.paths_by_id[typed.id] = path

    def get(self, doc_id: str) -> DocModel | None:
        return self.by_id.get(doc_id)

    def claims(self) -> list[ClaimV1]:
        return [d for d in self.by_id.values() if isinstance(d, ClaimV1)]

    def specs(self) -> list[SpecV1]:
        return [d for d in self.by_id.values() if isinstance(d, SpecV1)]

    def _coerce(self, doc: dict[str, Any]) -> DocModel:
        doc_type = doc.get("type")
        model = _MODEL_BY_TYPE.get(str(doc_type))
        if model is None:
            raise ValueError(f"unknown document type: {doc_type}")
        return model.model_validate(doc)  # type: ignore[return-value]
```

- [ ] **Step 5: Run tests — expect PASS**

```bash
PYTHONPATH=. ./venv/bin/python -m pytest tests/test_knowledge_forge_store.py -v
```

- [ ] **Step 6: Commit**

```bash
git add orion/knowledge_forge/store.py tests/fixtures/knowledge_forge/ tests/test_knowledge_forge_store.py
git commit -m "feat: add KnowledgeStore index for YAML corpus documents"
```

---

### Task 5: Lint — schema, dangling refs, banned keys

**Files:**

- Create: `orion/knowledge_forge/lint.py`
- Test: `tests/test_knowledge_forge_lint.py`
- Create: `tests/fixtures/knowledge_forge/claims/disputed/claim-test-bad-ref.yaml`

- [ ] **Step 1: Add fixture with dangling ref**

Create `tests/fixtures/knowledge_forge/claims/disputed/claim-test-bad-ref.yaml`:

```yaml
type: claim
id: claim:test:bad-ref
statement: References missing claim.
status: disputed
source_refs:
  - source:missing:0001
confidence: low
depends_on:
  - claim:does:not:exist
supports: []
contradicts: []
supersedes: []
implements: []
tested_by: []
blocked_by: []
motivated_by: []
used_by: []
```

- [ ] **Step 2: Write failing lint tests**

Create `tests/test_knowledge_forge_lint.py`:

```python
from __future__ import annotations

from pathlib import Path

from orion.knowledge_forge.lint import lint_corpus
from orion.knowledge_forge.store import KnowledgeStore


def test_lint_reports_dangling_claim_ref() -> None:
    root = Path(__file__).resolve().parent / "fixtures" / "knowledge_forge"
    store = KnowledgeStore(root)
    store.load()
    report = lint_corpus(store)
    codes = {issue.code for issue in report.issues}
    assert "dangling_ref" in codes


def test_lint_passes_minimal_valid_fixture() -> None:
    root = Path(__file__).resolve().parent / "fixtures" / "knowledge_forge"
    # Remove bad fixture temporarily by linting only accepted claim file subset
    store = KnowledgeStore(root)
    store.load()
    store.by_id.pop("claim:test:bad-ref", None)
    report = lint_corpus(store)
    assert report.ok is True
```

- [ ] **Step 3: Run test — expect failure**

```bash
PYTHONPATH=. ./venv/bin/python -m pytest tests/test_knowledge_forge_lint.py -v
```

- [ ] **Step 4: Implement lint**

Create `orion/knowledge_forge/lint.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field

from orion.knowledge_forge.models import ClaimV1, DecisionV1, SpecV1, TypedRelationsV1
from orion.knowledge_forge.store import KnowledgeStore, DocModel

RELATION_FIELDS = tuple(TypedRelationsV1.model_fields.keys())


@dataclass
class LintIssue:
    code: str
    doc_id: str
    message: str


@dataclass
class LintReport:
    issues: list[LintIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues


def lint_corpus(store: KnowledgeStore) -> LintReport:
    report = LintReport()
    for doc_id, doc in store.by_id.items():
        _lint_relations(store, doc_id, doc, report)
        if isinstance(doc, ClaimV1) and not doc.source_refs:
            report.issues.append(
                LintIssue("missing_source_ref", doc_id, "claims must cite at least one source_ref")
            )
        if isinstance(doc, SpecV1):
            for claim_id in doc.source_claims:
                if claim_id not in store.by_id:
                    report.issues.append(
                        LintIssue("dangling_ref", doc_id, f"source_claims missing {claim_id}")
                    )
    return report


def _lint_relations(store: KnowledgeStore, doc_id: str, doc: DocModel, report: LintReport) -> None:
    if not isinstance(doc, (ClaimV1, DecisionV1, SpecV1)):
        return
    for field_name in RELATION_FIELDS:
        targets = getattr(doc, field_name)
        for target_id in targets:
            if target_id not in store.by_id:
                report.issues.append(
                    LintIssue(
                        "dangling_ref",
                        doc_id,
                        f"{field_name} references missing id {target_id}",
                    )
                )
```

- [ ] **Step 5: Run tests — expect PASS**

```bash
PYTHONPATH=. ./venv/bin/python -m pytest tests/test_knowledge_forge_lint.py -v
```

- [ ] **Step 6: Commit**

```bash
git add orion/knowledge_forge/lint.py tests/test_knowledge_forge_lint.py tests/fixtures/knowledge_forge/
git commit -m "feat: add Knowledge Forge corpus linter with dangling ref detection"
```

---

### Task 6: Review queue — pending patches cannot touch execution_ready directly

**Files:**

- Create: `orion/knowledge_forge/review.py`
- Test: `tests/test_knowledge_forge_review.py`

**Patch file format (normative):**

```markdown
---
patch_id: review:spec-substrate-tier-telemetry-v1-001
target: specs/execution_ready/substrate-tier-telemetry-v1.yaml
action: create
status: pending
author: agent
---

```yaml
type: spec
id: spec:substrate-tier-telemetry-v1
...
```
```

- [ ] **Step 1: Write failing review tests**

Create `tests/test_knowledge_forge_review.py`:

```python
from __future__ import annotations

from pathlib import Path

from orion.knowledge_forge.review import apply_pending_patch, list_pending_patches


def test_list_pending_patches_finds_fixture(tmp_path: Path) -> None:
    pending = tmp_path / "reviews" / "pending"
    pending.mkdir(parents=True)
    patch = pending / "review-test-001.patch.md"
    patch.write_text(
        """---
patch_id: review:test:001
target: specs/design/test.yaml
action: create
status: pending
---

```yaml
type: spec
id: spec:test:001
status: draft
component: test
requirements: []
```
""",
        encoding="utf-8",
    )
    found = list_pending_patches(tmp_path)
    assert len(found) == 1
    assert found[0].patch_id == "review:test:001"


def test_apply_pending_patch_writes_target(tmp_path: Path) -> None:
    pending = tmp_path / "reviews" / "pending"
    accepted = tmp_path / "reviews" / "accepted"
    pending.mkdir(parents=True)
    accepted.mkdir(parents=True)
    patch_path = pending / "review-test-002.patch.md"
    patch_path.write_text(
        """---
patch_id: review:test:002
target: specs/design/test.yaml
action: create
status: pending
---

```yaml
type: spec
id: spec:test:002
status: draft
component: test
requirements: ["one"]
non_goals: []
acceptance_tests: []
source_claims: []
```
""",
        encoding="utf-8",
    )
    apply_pending_patch(tmp_path, "review:test:002")
    target = tmp_path / "specs" / "design" / "test.yaml"
    assert target.is_file()
    assert not patch_path.exists()
    assert (accepted / "review-test-002.patch.md").is_file()
```

- [ ] **Step 2: Run test — expect failure**

```bash
PYTHONPATH=. ./venv/bin/python -m pytest tests/test_knowledge_forge_review.py -v
```

- [ ] **Step 3: Implement review module**

Create `orion/knowledge_forge/review.py`:

```python
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from orion.knowledge_forge.yaml_doc import save_yaml_doc

_FRONTMATTER_RE = re.compile(r"^---\n(?P<meta>.*?)\n---\n", re.DOTALL)
_FENCE_RE = re.compile(r"```yaml\n(?P<body>.*?)\n```", re.DOTALL)

EXECUTION_READY_PREFIX = "specs/execution_ready/"


@dataclass
class PendingPatch:
    patch_id: str
    target: str
    action: str
    path: Path


def list_pending_patches(corpus_root: Path) -> list[PendingPatch]:
    pending_dir = corpus_root / "reviews" / "pending"
    if not pending_dir.is_dir():
        return []
    out: list[PendingPatch] = []
    for path in sorted(pending_dir.glob("*.patch.md")):
        meta, _ = _parse_patch_file(path)
        out.append(
            PendingPatch(
                patch_id=str(meta["patch_id"]),
                target=str(meta["target"]),
                action=str(meta["action"]),
                path=path,
            )
        )
    return out


def apply_pending_patch(corpus_root: Path, patch_id: str) -> Path:
    pending_dir = corpus_root / "reviews" / "pending"
    for path in pending_dir.glob("*.patch.md"):
        meta, body = _parse_patch_file(path)
        if meta.get("patch_id") != patch_id:
            continue
        target_rel = str(meta["target"])
        if target_rel.startswith(EXECUTION_READY_PREFIX):
            # v0: execution_ready writes must come from reviewed specs via compile promotion,
            # not direct agent patch — force path through specs/design or specs/plan first.
            raise ValueError(
                "direct patches to specs/execution_ready/ are forbidden; "
                "promote via reviewed spec + human approval"
            )
        doc = yaml.safe_load(body)
        if not isinstance(doc, dict):
            raise ValueError(f"patch body must be YAML mapping: {path}")
        target_path = corpus_root / target_rel
        if meta.get("action") == "create" and target_path.exists():
            raise ValueError(f"refusing to overwrite existing target: {target_path}")
        save_yaml_doc(target_path, doc)
        accepted_dir = corpus_root / "reviews" / "accepted"
        accepted_dir.mkdir(parents=True, exist_ok=True)
        path.replace(accepted_dir / path.name)
        return target_path
    raise FileNotFoundError(f"pending patch not found: {patch_id}")


def _parse_patch_file(path: Path) -> tuple[dict[str, str], str]:
    text = path.read_text(encoding="utf-8")
    m = _FRONTMATTER_RE.match(text)
    if not m:
        raise ValueError(f"missing frontmatter: {path}")
    meta = yaml.safe_load(m.group("meta"))
    if not isinstance(meta, dict):
        raise ValueError(f"invalid frontmatter: {path}")
    rest = text[m.end() :]
    fm = _FENCE_RE.search(rest)
    if not fm:
        raise ValueError(f"missing yaml fenced block: {path}")
    return {str(k): str(v) for k, v in meta.items()}, fm.group("body")
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
PYTHONPATH=. ./venv/bin/python -m pytest tests/test_knowledge_forge_review.py -v
```

- [ ] **Step 5: Commit**

```bash
git add orion/knowledge_forge/review.py tests/test_knowledge_forge_review.py
git commit -m "feat: add Knowledge Forge review queue with execution_ready guard"
```

---

### Task 7: Context pack compiler

**Files:**

- Create: `orion/knowledge_forge/compile.py`
- Test: `tests/test_knowledge_forge_compile.py`
- Create: `tests/fixtures/knowledge_forge/specs/execution_ready/spec-test-compile.yaml`

- [ ] **Step 1: Add fixture spec linked to claim**

Create `tests/fixtures/knowledge_forge/specs/execution_ready/spec-test-compile.yaml`:

```yaml
type: spec
id: spec:test:compile
status: execution_ready
component: orion-test
requirements:
  - Expose GET /health
non_goals:
  - No UI
acceptance_tests:
  - pytest services/orion-test/tests/test_health.py passes
source_claims:
  - claim:test:0001
likely_files:
  - services/orion-test/app/main.py
known_traps:
  - Do not import hub redis client in test service
supports: []
contradicts: []
supersedes: []
depends_on: []
implements: []
tested_by: []
blocked_by: []
motivated_by: []
```

- [ ] **Step 2: Write failing compile test**

Create `tests/test_knowledge_forge_compile.py`:

```python
from __future__ import annotations

from pathlib import Path

from orion.knowledge_forge.compile import compile_context_pack_markdown
from orion.knowledge_forge.store import KnowledgeStore


def test_compile_context_pack_includes_only_accepted_claims() -> None:
    root = Path(__file__).resolve().parent / "fixtures" / "knowledge_forge"
    store = KnowledgeStore(root)
    store.load()
    md = compile_context_pack_markdown(store, spec_id="spec:test:compile", task="Ship health endpoint")
    assert "## Goal" in md
    assert "Expose GET /health" in md
    assert "Fixture claim for store indexing" in md
    assert "## Known traps" in md
    assert "Do not import hub redis" in md
```

- [ ] **Step 3: Run test — expect failure**

```bash
PYTHONPATH=. ./venv/bin/python -m pytest tests/test_knowledge_forge_compile.py -v
```

- [ ] **Step 4: Implement compiler**

Create `orion/knowledge_forge/compile.py`:

```python
from __future__ import annotations

from orion.knowledge_forge.models import ClaimStatusV1, ClaimV1, SpecV1, SpecStatusV1
from orion.knowledge_forge.store import KnowledgeStore


def compile_context_pack_markdown(store: KnowledgeStore, *, spec_id: str, task: str) -> str:
    spec = store.get(spec_id)
    if not isinstance(spec, SpecV1):
        raise ValueError(f"spec not found: {spec_id}")
    if spec.status not in (SpecStatusV1.reviewed, SpecStatusV1.execution_ready):
        raise ValueError(f"spec {spec_id} must be reviewed or execution_ready")

    claims = _accepted_claims_for_spec(store, spec)
    lines = [
        f"# Context Pack: {spec.component}",
        "",
        "## Goal",
        task,
        "",
        "## Current accepted facts",
    ]
    for claim in claims:
        lines.append(f"- {claim.statement} (`{claim.id}`)")
    lines.extend(["", "## Required behavior"])
    for req in spec.requirements:
        lines.append(f"- {req}")
    lines.extend(["", "## Non-goals"])
    for ng in spec.non_goals:
        lines.append(f"- {ng}")
    lines.extend(["", "## Relevant files"])
    for path in spec.likely_files:
        lines.append(f"- `{path}`")
    lines.extend(["", "## Acceptance tests"])
    for test in spec.acceptance_tests:
        lines.append(f"- {test}")
    lines.extend(["", "## Known traps"])
    for trap in spec.known_traps:
        lines.append(f"- {trap}")
    lines.append("")
    return "\n".join(lines)


def _accepted_claims_for_spec(store: KnowledgeStore, spec: SpecV1) -> list[ClaimV1]:
    out: list[ClaimV1] = []
    for claim_id in spec.source_claims:
        doc = store.get(claim_id)
        if not isinstance(doc, ClaimV1):
            continue
        if doc.status != ClaimStatusV1.accepted:
            continue
        out.append(doc)
    return out
```

- [ ] **Step 5: Run test — expect PASS**

```bash
PYTHONPATH=. ./venv/bin/python -m pytest tests/test_knowledge_forge_compile.py -v
```

- [ ] **Step 6: Commit**

```bash
git add orion/knowledge_forge/compile.py tests/test_knowledge_forge_compile.py tests/fixtures/knowledge_forge/specs/
git commit -m "feat: compile Cursor context packs from reviewed specs and accepted claims"
```

---

### Task 8: Source coverage probe (v0 eval gate)

**Files:**

- Create: `orion/knowledge_forge/probes.py`
- Test: extend `tests/test_knowledge_forge_golden_path.py` (create in Task 9) or add `tests/test_knowledge_forge_probes.py`

- [ ] **Step 1: Write failing probe test**

Create `tests/test_knowledge_forge_probes.py`:

```python
from __future__ import annotations

from pathlib import Path

from orion.knowledge_forge.probes import probe_source_coverage
from orion.knowledge_forge.store import KnowledgeStore


def test_probe_flags_uncited_sentences_in_source() -> None:
    root = Path(__file__).resolve().parent / "fixtures" / "knowledge_forge"
    source = root / "raw" / "sources" / "test-source.md"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        "Alpha requirement must persist telemetry.\nBeta requirement is unrelated.\n",
        encoding="utf-8",
    )
    store = KnowledgeStore(root)
    store.load()
    report = probe_source_coverage(
        store,
        source_path=source,
        source_id="source:test:fixture",
        min_keyword="telemetry",
    )
    assert report.ok is False
    assert "uncited_keyword" in {i.code for i in report.issues}
```

- [ ] **Step 2: Run test — expect failure**

```bash
PYTHONPATH=. ./venv/bin/python -m pytest tests/test_knowledge_forge_probes.py -v
```

- [ ] **Step 3: Implement probe**

Create `orion/knowledge_forge/probes.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from orion.knowledge_forge.models import ClaimV1
from orion.knowledge_forge.store import KnowledgeStore


@dataclass
class ProbeIssue:
    code: str
    message: str


@dataclass
class ProbeReport:
    issues: list[ProbeIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues


def probe_source_coverage(
    store: KnowledgeStore,
    *,
    source_path: Path,
    source_id: str,
    min_keyword: str,
) -> ProbeReport:
    text = source_path.read_text(encoding="utf-8").lower()
    report = ProbeReport()
    if min_keyword.lower() not in text:
        return report

    claim_text = " ".join(
        c.statement.lower()
        for c in store.claims()
        if isinstance(c, ClaimV1) and source_id in c.source_refs
    )
    if min_keyword.lower() not in claim_text:
        report.issues.append(
            ProbeIssue(
                "uncited_keyword",
                f"source mentions {min_keyword!r} but no claim citing {source_id} covers it",
            )
        )
    return report
```

- [ ] **Step 4: Run test — expect PASS**

```bash
PYTHONPATH=. ./venv/bin/python -m pytest tests/test_knowledge_forge_probes.py -v
```

- [ ] **Step 5: Commit**

```bash
git add orion/knowledge_forge/probes.py tests/test_knowledge_forge_probes.py
git commit -m "feat: add source coverage probe for Knowledge Forge eval gate"
```

---

### Task 9: CLI entrypoint

**Files:**

- Create: `orion/knowledge_forge/cli.py`
- Create: `orion/knowledge_forge/__main__.py`

- [ ] **Step 1: Implement CLI (no separate test — covered by golden path)**

Create `orion/knowledge_forge/cli.py`:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from orion.knowledge_forge.compile import compile_context_pack_markdown
from orion.knowledge_forge.lint import lint_corpus
from orion.knowledge_forge.paths import resolve_corpus_root
from orion.knowledge_forge.probes import probe_source_coverage
from orion.knowledge_forge.review import apply_pending_patch, list_pending_patches
from orion.knowledge_forge.store import KnowledgeStore
from orion.knowledge_forge.yaml_doc import save_yaml_doc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="orion.knowledge_forge")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("lint")

    review = sub.add_parser("review")
    review_sub = review.add_subparsers(dest="review_cmd", required=True)
    review_sub.add_parser("list")
    apply_p = review_sub.add_parser("apply")
    apply_p.add_argument("patch_id")

    compile_p = sub.add_parser("compile")
    compile_sub = compile_p.add_subparsers(dest="compile_cmd", required=True)
    ctx = compile_sub.add_parser("context-pack")
    ctx.add_argument("--spec", required=True)
    ctx.add_argument("--task", required=True)
    ctx.add_argument("--out", required=True)

    probe_p = sub.add_parser("probe")
    probe_sub = probe_p.add_subparsers(dest="probe_cmd", required=True)
    src = probe_sub.add_parser("source")
    src.add_argument("--source-id", required=True)
    src.add_argument("--path", required=True)
    src.add_argument("--keyword", required=True)

    args = parser.parse_args(argv)
    root = resolve_corpus_root()

    if args.cmd == "lint":
        store = KnowledgeStore(root)
        store.load()
        report = lint_corpus(store)
        for issue in report.issues:
            print(f"{issue.code}\t{issue.doc_id}\t{issue.message}")
        return 0 if report.ok else 1

    if args.cmd == "review" and args.review_cmd == "list":
        for patch in list_pending_patches(root):
            print(f"{patch.patch_id}\t{patch.target}\t{patch.action}")
        return 0

    if args.cmd == "review" and args.review_cmd == "apply":
        target = apply_pending_patch(root, args.patch_id)
        print(f"applied\t{target}")
        return 0

    if args.cmd == "compile" and args.compile_cmd == "context-pack":
        store = KnowledgeStore(root)
        store.load()
        md = compile_context_pack_markdown(store, spec_id=args.spec, task=args.task)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        pack_id = args.spec.replace("spec:", "ctx:", 1)
        meta_path = out.with_suffix(".yaml")
        save_yaml_doc(
            meta_path,
            {
                "type": "context_pack",
                "id": pack_id,
                "target": "cursor",
                "task": args.task,
                "included_specs": [args.spec],
                "allowed_sources": [],
                "excluded_context": [],
            },
        )
        print(f"wrote\t{out}")
        return 0

    if args.cmd == "probe" and args.probe_cmd == "source":
        store = KnowledgeStore(root)
        store.load()
        report = probe_source_coverage(
            store,
            source_path=Path(args.path),
            source_id=args.source_id,
            min_keyword=args.keyword,
        )
        for issue in report.issues:
            print(f"{issue.code}\t{issue.message}")
        return 0 if report.ok else 1

    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
```

Create `orion/knowledge_forge/__main__.py`:

```python
from orion.knowledge_forge.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke CLI lint on fixture corpus**

```bash
cd /mnt/scripts/Orion-Sapienform
ORION_KNOWLEDGE_ROOT=tests/fixtures/knowledge_forge PYTHONPATH=. ./venv/bin/python -m orion.knowledge_forge lint
```

Expected: exit code 1 (dangling ref fixture present) with `dangling_ref` lines

- [ ] **Step 3: Commit**

```bash
git add orion/knowledge_forge/cli.py orion/knowledge_forge/__main__.py
git commit -m "feat: add Knowledge Forge CLI for lint, review, compile, probe"
```

---

### Task 10: AGENTS.md — ingest and compile rules for agents

**Files:**

- Create: `orion-knowledge/AGENTS.md`

- [ ] **Step 1: Write AGENTS.md**

Create `orion-knowledge/AGENTS.md`:

```markdown
# Orion Knowledge Forge — agent contract

You are maintaining a **research-to-execution compiler**, not a wiki.

## Authority

- `raw/` is immutable after ingest. Never edit existing raw files.
- Claims in `claims/` are the atomic truth layer.
- Specs in `specs/execution_ready/` are execution-facing. You may not edit them directly.
- Propose changes only via `reviews/pending/*.patch.md`.

## Ingest prompt (use when adding a raw source)

When given `raw/sources/<file>`:

1. Read the full source.
2. Extract atomic claims as YAML files under `claims/disputed/` first (set `status: speculative` or `status: disputed` in the YAML).
3. Each claim MUST include `source_refs` pointing at a `source:*` id.
4. Use typed relations only: `supports`, `contradicts`, `supersedes`, `depends_on`, `implements`, `tested_by`, `blocked_by`, `motivated_by`.
5. Update `wiki/concepts/<topic>.md` as a compiled view citing claim ids inline.
6. If a spec or ADR must change, write a patch in `reviews/pending/` — do not mutate `specs/execution_ready/`.
7. Append one line to `wiki/log.md`.

## Compile prompt (use when preparing Cursor work)

Given a spec id and task description:

1. Run: `python -m orion.knowledge_forge compile context-pack --spec <id> --task "<task>" --out context_packs/cursor/<slug>.md`
2. Hand the generated markdown to Cursor — not the whole wiki.
3. Include only accepted claims. Flag disputed claims in `reviews/pending/` instead.

## Forbidden

- Untyped `related` links
- Silent overwrite of `specs/execution_ready/`
- Feeding `wiki/` wholesale into implementation agents
- Treating wiki prose as authority over claims
```

- [ ] **Step 2: Commit**

```bash
git add orion-knowledge/AGENTS.md
git commit -m "docs: add Knowledge Forge AGENTS.md ingest and compile rules"
```

---

### Task 11: Golden migration — substrate tier telemetry example

**Files:**

- Create: `orion-knowledge/raw/sources/2026-05-14-substrate-tier-telemetry-design-ref.md`
- Create: `orion-knowledge/claims/accepted/claim-substrate-telemetry-0001.yaml`
- Create: `orion-knowledge/claims/accepted/claim-substrate-telemetry-0002.yaml`
- Create: `orion-knowledge/specs/execution_ready/substrate-tier-telemetry-v1.yaml`
- Create: `orion-knowledge/wiki/concepts/substrate-tier-telemetry.md`
- Test: `tests/test_knowledge_forge_golden_path.py`

- [ ] **Step 1: Add source pointer (immutable summary, not full spec paste)**

Create `orion-knowledge/raw/sources/2026-05-14-substrate-tier-telemetry-design-ref.md`:

```markdown
# Source reference: substrate tier telemetry design

Imported from repo archive:
`docs/superpowers/specs/2026-05-14-substrate-tier-telemetry-persistence-design.md`

Key facts captured as claims:

- New service `orion-substrate-telemetry` subscribes to `orion:substrate:tier_outcomes`.
- Append-only Postgres persistence; orch optionally merges facet into MindRunRequest.
- Hub reads via HTTP proxy, not Redis bus subscription.
```

- [ ] **Step 2: Add accepted claims**

Create `orion-knowledge/claims/accepted/claim-substrate-telemetry-0001.yaml`:

```yaml
type: claim
id: claim:orion:substrate-telemetry:0001
statement: orion-substrate-telemetry subscribes to orion:substrate:tier_outcomes and append-only persists tier outcome rows to Postgres.
status: accepted
source_refs:
  - source:2026-05-14-substrate-tier-telemetry-design-ref
confidence: high
supports: []
contradicts: []
supersedes: []
depends_on: []
implements: []
tested_by: []
blocked_by: []
motivated_by: []
used_by:
  - spec:substrate-tier-telemetry-v1
```

Create `orion-knowledge/claims/accepted/claim-substrate-telemetry-0002.yaml`:

```yaml
type: claim
id: claim:orion:substrate-telemetry:0002
statement: orion-cortex-orch optionally HTTP-fetches persisted substrate telemetry and merges into MindRunRequestV1.snapshot_inputs.facets.substrate_telemetry before calling Mind.
status: accepted
source_refs:
  - source:2026-05-14-substrate-tier-telemetry-design-ref
confidence: high
supports: []
contradicts: []
supersedes: []
depends_on:
  - claim:orion:substrate-telemetry:0001
implements: []
tested_by: []
blocked_by: []
motivated_by: []
used_by:
  - spec:substrate-tier-telemetry-v1
```

- [ ] **Step 3: Add execution-ready spec**

Create `orion-knowledge/specs/execution_ready/substrate-tier-telemetry-v1.yaml`:

```yaml
type: spec
id: spec:substrate-tier-telemetry-v1
status: execution_ready
component: orion-substrate-telemetry
requirements:
  - Subscribe to orion:substrate:tier_outcomes via BaseChassis
  - Validate kind substrate.tier_outcomes.v1 before insert
  - Expose GET /v1/substrate/tier-outcomes/latest and /history by correlation_id
  - Orch merges facet when ORION_SUBSTRATE_TELEMETRY_BASE_URL is set
non_goals:
  - Hub Redis bus subscription for tier outcomes
  - Idempotent dedupe of duplicate bus deliveries in v0
acceptance_tests:
  - PYTHONPATH=. ./venv/bin/python -m pytest services/orion-substrate-telemetry/tests/ -q
  - PYTHONPATH=. ./venv/bin/python -m pytest services/orion-cortex-orch/tests/test_substrate_telemetry_orch.py -q
source_claims:
  - claim:orion:substrate-telemetry:0001
  - claim:orion:substrate-telemetry:0002
likely_files:
  - services/orion-substrate-telemetry/app/main.py
  - services/orion-cortex-orch/app/mind_runtime.py
  - orion/bus/channels.yaml
known_traps:
  - Duplicate bus deliveries create multiple rows; dedupe is deferred
  - Do not block BaseChassis loop on bad frames
supports: []
contradicts: []
supersedes: []
depends_on: []
implements: []
tested_by: []
blocked_by: []
motivated_by: []
```

- [ ] **Step 4: Add source registry YAML**

Create `orion-knowledge/raw/sources/source-2026-05-14-substrate-tier-telemetry-design-ref.yaml`:

```yaml
type: source
id: source:2026-05-14-substrate-tier-telemetry-design-ref
kind: design_doc
path: raw/sources/2026-05-14-substrate-tier-telemetry-design-ref.md
trust_level: primary
```

- [ ] **Step 5: Write golden path test**

Create `tests/test_knowledge_forge_golden_path.py`:

```python
from __future__ import annotations

from pathlib import Path

from orion.knowledge_forge.compile import compile_context_pack_markdown
from orion.knowledge_forge.lint import lint_corpus
from orion.knowledge_forge.store import KnowledgeStore


def test_live_corpus_lints_clean() -> None:
    root = Path(__file__).resolve().parents[1] / "orion-knowledge"
    store = KnowledgeStore(root)
    store.load()
    report = lint_corpus(store)
    assert report.ok, [f"{i.code}: {i.message}" for i in report.issues]


def test_compile_substrate_telemetry_context_pack() -> None:
    root = Path(__file__).resolve().parents[1] / "orion-knowledge"
    store = KnowledgeStore(root)
    store.load()
    md = compile_context_pack_markdown(
        store,
        spec_id="spec:substrate-tier-telemetry-v1",
        task="Verify substrate tier telemetry persistence matches spec",
    )
    assert "orion-substrate-telemetry" in md
    assert "MindRunRequestV1" in md
```

- [ ] **Step 6: Run golden path tests**

```bash
cd /mnt/scripts/Orion-Sapienform
PYTHONPATH=. ./venv/bin/python -m pytest tests/test_knowledge_forge_golden_path.py -v
```

Expected: 2 passed

- [ ] **Step 7: Compile example context pack**

```bash
PYTHONPATH=. ./venv/bin/python -m orion.knowledge_forge compile context-pack \
  --spec spec:substrate-tier-telemetry-v1 \
  --task "Verify substrate tier telemetry persistence matches spec" \
  --out orion-knowledge/context_packs/cursor/substrate-tier-telemetry-v1.md
PYTHONPATH=. ./venv/bin/python -m orion.knowledge_forge lint
```

Expected: lint exit 0; context pack file exists

- [ ] **Step 8: Commit**

```bash
git add orion-knowledge/ tests/test_knowledge_forge_golden_path.py
git commit -m "feat: seed Knowledge Forge with substrate telemetry golden corpus"
```

---

### Task 12: Lint report artifact + wiki log update

**Files:**

- Modify: `orion/knowledge_forge/cli.py` (optional `--report-out`)
- Modify: `orion-knowledge/wiki/log.md`

- [ ] **Step 1: Add `--report-out` to lint subcommand**

In `orion/knowledge_forge/cli.py`, extend lint branch:

```python
    lint_p = sub.add_parser("lint")
    lint_p.add_argument("--report-out", default="")
```

After building report:

```python
        if args.report_out:
            out = Path(args.report_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            lines = [f"{i.code}\t{i.doc_id}\t{i.message}" for i in report.issues]
            out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
```

- [ ] **Step 2: Generate lint report and update log**

```bash
PYTHONPATH=. ./venv/bin/python -m orion.knowledge_forge lint \
  --report-out orion-knowledge/evals/lint_reports/2026-05-20.txt
```

Append to `orion-knowledge/wiki/log.md`:

```markdown
| 2026-05-20 | Golden corpus lint clean; substrate telemetry context pack compiled | spec:substrate-tier-telemetry-v1 |
```

- [ ] **Step 3: Run full knowledge_forge test suite**

```bash
PYTHONPATH=. ./venv/bin/python -m pytest tests/test_knowledge_forge_*.py -q
```

Expected: all passed

- [ ] **Step 4: Commit**

```bash
git add orion/knowledge_forge/cli.py orion-knowledge/
git commit -m "chore: wire lint report output and log golden path completion"
```

---

## v1 roadmap (out of scope for this plan — do not implement yet)

| Phase | Capability |
|-------|------------|
| v1 | LLM-assisted ingest script calling `orion-llm-gateway` structured output; disputed-claim dashboard markdown; spec promotion workflow `design → reviewed → execution_ready` |
| v2 | LangGraph durable review workflow; result ingestor (git diff + pytest output → claims) |
| v3 | MCP server exposing `get_context_pack`, `search_claims`, `list_disputed_claims` |
| Bridge | Export accepted architecture claims to `orion-meta-services` ExtractionScope snapshots |

---

## Self-review checklist

**Spec coverage:**

| Requirement | Task |
|-------------|------|
| Markdown repo + git corpus | Task 1 |
| Five primitives (source, claim, decision, spec, context pack) | Task 2 |
| Claim-first atomic layer | Tasks 2, 4, 11 |
| Typed relations, ban untyped links | Task 2 (`extra=forbid` on models) |
| Review queue — no silent spec rewrite | Task 6 |
| Context pack compiler | Task 7 |
| Ingest/compile agent rules (LLMWiki schema file) | Task 10 |
| Probes/evals gate | Task 8 |
| v0 workflow (ingest → review → compile → execute) | Tasks 10, 11, README |
| Orion-specific golden example | Task 11 |
| CLI lint loop | Tasks 5, 9, 12 |

**Placeholder scan:** No TBD steps. All code blocks are complete for v0.

**Type consistency:** IDs use prefixes `source:`, `claim:`, `adr:`, `spec:`, `ctx:`, `review:` throughout.

**Gap note:** Automated LLM ingest is deliberately manual/agent-prompt-driven in v0; v1 row covers structured ingest.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-20-orion-knowledge-forge-v0.md`.

**Two execution options:**

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
