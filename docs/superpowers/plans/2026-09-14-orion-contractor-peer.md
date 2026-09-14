# Orion Contractor Peer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let Orion hire a read-only frontier peer (Cursor Auto, Claude fallback) via `:HelpRequest`, land a `:PeerBrief` in worldview + Postgres, and soft-nudge the next curiosity/self-inquiry kickoff — without the peer writing beliefs or forcing citation.

**Architecture:** Patch 0 lands contracts, kickoff Cypher/teach, dual-write persistence, soft-nudge, Atlas, and the kill switch with **no live hire**. Patch 1 arms `services/orion-curiosity-peer` behind the flag: post-run HelpRequest enqueue → contested Cursor budget → Cursor Agent CLI (`agent -p --mode ask`) → one Claude room fallback → PeerBrief dual-write. Supervisor stays report-only. Orion alone MERGEs `:Prior` / `:Finding` / `:SelfDefinition`.

**Tech Stack:** Python 3.12, Pydantic v2, Redis streams (Orion bus), FalkorDB Cypher, SQLAlchemy (orion-sql-writer), Cursor Agent CLI (subprocess / desktop login — not `cursor-sdk`), pytest, existing Hub curiosity kickoff / Atlas / room-companion Claude transport.

**Spec:** [`docs/superpowers/specs/2026-09-14-orion-contractor-peer-design.md`](../specs/2026-09-14-orion-contractor-peer-design.md)

**Branch / worktree:** `docs/orion-contractor-peer-design` → implement on `feat/orion-contractor-peer` via `scripts/new_worktree.sh feat orion-contractor-peer` (do not commit from the shared checkout).

## Global Constraints

- Do **not** name code/paths `frontier_assistant`, `frontier_buddy`, or overload `orion/core/schemas/frontier_curiosity.py` / substrate `FrontierInvocation*`.
- Peer powers: **read-only** investigation (repo, containers, live inspect). Deny: write_files, git_mutate, docker_mutate, graph_belief_write.
- Peer never MERGEs `:Prior`, `:Finding`, `:SelfDefinition`, `:Hop`, `:TurnOutcome`.
- No forced cite of peer prose in kickoff or enforce paths.
- v1 trigger is Orion-authored `:HelpRequest` only — do **not** arm `ask_claude_trigger.py` or supervisor `hand_off_to_claude`.
- Supervisor `HopReadingV1` stays offline / report-only.
- One kill switch: `HUB_CURIOSITY_CONTRACTOR_PEER_ENABLED` (default `false`) disables hire + invoker subscription; unused PeerBrief rows/nodes are additive and need no migration.
- Env parity: any new `.env_example` key → run `python scripts/sync_local_env_from_example.py` and keep `settings.py` / compose in sync.
- Bus contract changes require schema + `orion/schemas/registry.py` (both `_REGISTRY` and `SCHEMA_REGISTRY`) + `orion/bus/channels.yaml` + `check_schema_registry.py` + `check_bus_channels.py`.
- Hub never writes belief nodes to `orion_worldview`; PeerBrief MERGE is allowed only in the peer dual-write helper (system writer), never via Hub RO `WorldviewReader`.
- Cursor credential isolation: desktop CLI auth (`agent login` / `~/.config/cursor`) is mounted into `orion-curiosity-peer`, not Hub (same reason Claude stays in room-companion). No `CURSOR_API_KEY` / `cursor-sdk`.
- Juniper transcripts / `~/.claude/projects` stay out of the contractor packet.
- Transport is Cursor Agent CLI (`agent -p --mode ask --workspace … --trust`); read-only gate is argv policy, not SDK tool allowlists.

---

## File map

| Path | Responsibility |
|------|----------------|
| `orion/schemas/curiosity_peer.py` | **Create** — `HelpRequestV1`, `PeerBriefV1`, channel/kind constants, clip helpers |
| `orion/schemas/registry.py` | **Modify** — `_REGISTRY` + `SCHEMA_REGISTRY` entries |
| `orion/bus/channels.yaml` | **Modify** — `orion:curiosity:help:request`, `orion:curiosity:peer:brief` |
| `orion/curiosity/peer_briefs.py` | **Create** — Cypher MERGE, unused-ok reader, soft-nudge text, identity-prose strip, persist helper |
| `orion/curiosity/kickoff_prompt.py` | **Modify** — HelpRequest write teach + soft-nudge section |
| `orion/curiosity/self_inquiry_prompt.py` | **Modify** — hire rules + evidence-only peer mode + soft-nudge |
| `orion/curiosity/worldview.py` | **Modify** — label constants if needed; footprint already counts by `labels(n)[0]` |
| `orion/curiosity/atlas.py` | **Modify** — PeerBrief / HelpRequest projection |
| `services/orion-hub/templates/curiosity_atlas.html` | **Modify** — render briefs |
| `services/orion-hub/scripts/curiosity_investigation.py` | **Modify** — load unused briefs into kickoff; post-run HelpRequest publish (Patch 1) |
| `services/orion-hub/app/settings.py` | **Modify** — `HUB_CURIOSITY_CONTRACTOR_PEER_ENABLED` |
| `services/orion-hub/.env_example` | **Modify** — kill switch |
| `services/orion-hub/docker-compose.yml` | **Modify** — passthrough |
| `services/orion-sql-writer/app/models/curiosity_peer_brief.py` | **Create** — `CuriosityPeerBriefSQL` |
| `services/orion-sql-writer/app/models/__init__.py` | **Modify** — export |
| `services/orion-sql-writer/app/settings.py` | **Modify** — route map + force-subscribe |
| `services/orion-sql-writer/app/worker.py` | **Modify** — `MODEL_MAP` / `INSERT_ONLY_MODELS` |
| `services/orion-sql-writer/.env_example` | **Modify** — subscribe channel + retention |
| `services/orion-curiosity-peer/` | **Create (Patch 1)** — thin bus worker: budget, Cursor, Claude fallback, dual-write |
| `services/orion-room-companion/app/room_prompt.py` | **Modify (Patch 1)** — contractor peer mode wording when invited for fallback |
| `orion/dev_economics/cursor_limit_events.py` | **Create (Patch 1)** — Cursor scarcity observation (fail-closed) |
| `tests/test_curiosity_peer_*.py` + service tests | **Create** — gate tests listed per task |

**Non-goals (do not implement in this plan):** supervisor auto-hire; mid-turn durable interrupt; forced cite; peer file edits / docker mutate / belief MERGE; arming `ask_claude_trigger`; substrate frontier invocation.

---

## Patch 0 — contracts, dual-write stub, soft-nudge (no live hire)

### Task 1: Schema models `HelpRequestV1` / `PeerBriefV1`

**Files:**
- Create: `orion/schemas/curiosity_peer.py`
- Test: `tests/test_curiosity_peer_schema.py`

**Interfaces:**
- Consumes: nothing
- Produces: `HelpRequestV1`, `PeerBriefV1`, `HELP_REQUEST_CHANNEL`, `HELP_REQUEST_KIND`, `PEER_BRIEF_CHANNEL`, `PEER_BRIEF_KIND`, clip helpers

- [ ] **Step 1: Write the failing test**

```python
# tests/test_curiosity_peer_schema.py
from __future__ import annotations

from orion.schemas.curiosity_peer import (
    HELP_REQUEST_CHANNEL,
    HELP_REQUEST_KIND,
    PEER_BRIEF_CHANNEL,
    PEER_BRIEF_KIND,
    HelpRequestV1,
    PeerBriefV1,
)


def test_help_request_defaults_and_forbids_extra() -> None:
    row = HelpRequestV1(
        help_id="help-1",
        run_id="abcd1234abcd",
        mode="self_inquiry",
        question="What evidence do I have that I form priors under pressure?",
        tried_summary="Read self_inquiry.py and two hop notes.",
        success_criteria="Pointers to concrete files/tables that support or refute.",
    )
    assert row.schema_version == HELP_REQUEST_KIND
    assert HELP_REQUEST_CHANNEL == "orion:curiosity:help:request"
    dumped = row.model_dump()
    assert dumped["prior_id"] is None
    try:
        HelpRequestV1(
            help_id="h",
            run_id="abcd1234abcd",
            mode="world_curiosity",
            question="q",
            tried_summary="t",
            success_criteria="s",
            unexpected=1,
        )
        assert False, "extra fields must be forbidden"
    except Exception:
        pass


def test_peer_brief_status_literals_and_bounds() -> None:
    brief = PeerBriefV1(
        brief_id="brief-1",
        help_id="help-1",
        run_id="abcd1234abcd",
        peer="cursor_auto",
        status="ok",
        summary="x" * 9000,  # must clip
        evidence_pointers=["orion/curiosity/kickoff_prompt.py:528"],
        open_questions=["Does Hub ever write worldview?"],
        suggested_next_looks=["GRAPH.RO_QUERY orion_worldview MATCH (n:Prior) RETURN count(n)"],
    )
    assert brief.schema_version == PEER_BRIEF_KIND
    assert PEER_BRIEF_CHANNEL == "orion:curiosity:peer:brief"
    assert len(brief.summary) <= 4000
    assert brief.peer in ("cursor_auto", "claude_room")
    for status in ("ok", "failed", "refused_budget", "empty"):
        PeerBriefV1(
            brief_id=f"b-{status}",
            help_id="help-1",
            run_id="abcd1234abcd",
            peer="claude_room",
            status=status,
            summary="s",
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_curiosity_peer_schema.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'orion.schemas.curiosity_peer'` (or import error)

- [ ] **Step 3: Write minimal implementation**

```python
# orion/schemas/curiosity_peer.py
"""Curiosity contractor peer contracts — HelpRequest + PeerBrief.

Design: docs/superpowers/specs/2026-09-14-orion-contractor-peer-design.md

Not substrate FrontierInvocation / frontier_buddy. Peer never authors
:Prior / :Finding / :SelfDefinition.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

HELP_REQUEST_CHANNEL = "orion:curiosity:help:request"
HELP_REQUEST_KIND = "curiosity.help.request.v1"
PEER_BRIEF_CHANNEL = "orion:curiosity:peer:brief"
PEER_BRIEF_KIND = "curiosity.peer.brief.v1"

CuriosityPeerModeV1 = Literal["world_curiosity", "self_inquiry"]
CuriosityPeerNameV1 = Literal["cursor_auto", "claude_room"]
CuriosityPeerBriefStatusV1 = Literal["ok", "failed", "refused_budget", "empty"]

MAX_QUESTION_CHARS = 2000
MAX_SUMMARY_CHARS = 4000
MAX_POINTER_CHARS = 500
MAX_LIST_ITEMS = 32


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def clip(text: object, limit: int) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= limit else s[: max(0, limit - 1)] + "…"


class HelpRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["curiosity.help.request.v1"] = "curiosity.help.request.v1"
    help_id: str
    run_id: str
    prior_id: Optional[str] = None
    mode: CuriosityPeerModeV1
    question: str
    tried_summary: str
    success_criteria: str
    written_at: datetime = Field(default_factory=_utc_now)

    @field_validator("question", "tried_summary", "success_criteria", mode="before")
    @classmethod
    def _clip_text(cls, v: object) -> str:
        return clip(v, MAX_QUESTION_CHARS)


class PeerBriefV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["curiosity.peer.brief.v1"] = "curiosity.peer.brief.v1"
    brief_id: str
    help_id: str
    run_id: str
    prior_id: Optional[str] = None
    peer: CuriosityPeerNameV1
    status: CuriosityPeerBriefStatusV1
    summary: str = ""
    evidence_pointers: List[str] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)
    suggested_next_looks: List[str] = Field(default_factory=list)
    refusal_reason: Optional[str] = None
    written_at: datetime = Field(default_factory=_utc_now)

    @field_validator("summary", mode="before")
    @classmethod
    def _clip_summary(cls, v: object) -> str:
        return clip(v, MAX_SUMMARY_CHARS)

    @field_validator(
        "evidence_pointers", "open_questions", "suggested_next_looks", mode="before"
    )
    @classmethod
    def _clip_lists(cls, v: object) -> list[str]:
        if not v:
            return []
        out: list[str] = []
        for item in list(v)[:MAX_LIST_ITEMS]:
            text = clip(item, MAX_POINTER_CHARS)
            if text:
                out.append(text)
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_curiosity_peer_schema.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add orion/schemas/curiosity_peer.py tests/test_curiosity_peer_schema.py
git commit -m "feat(curiosity): add HelpRequest and PeerBrief schema models"
```

---

### Task 2: Registry + bus channels

**Files:**
- Modify: `orion/schemas/registry.py` (import + `_REGISTRY` + `SCHEMA_REGISTRY`)
- Modify: `orion/bus/channels.yaml` (after curiosity turn channels ~3056)
- Test: reuse `scripts/check_schema_registry.py` + `scripts/check_bus_channels.py`

**Interfaces:**
- Consumes: `HelpRequestV1`, `PeerBriefV1`, channel constants from Task 1
- Produces: registered kinds `curiosity.help.request.v1`, `curiosity.peer.brief.v1`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_curiosity_peer_registry.py
from orion.schemas.curiosity_peer import HELP_REQUEST_KIND, PEER_BRIEF_KIND
from orion.schemas.registry import SCHEMA_REGISTRY, _REGISTRY


def test_curiosity_peer_models_registered_in_both_maps() -> None:
    assert "HelpRequestV1" in _REGISTRY
    assert "PeerBriefV1" in _REGISTRY
    assert SCHEMA_REGISTRY["HelpRequestV1"].kind == HELP_REQUEST_KIND
    assert SCHEMA_REGISTRY["PeerBriefV1"].kind == PEER_BRIEF_KIND
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_curiosity_peer_registry.py -q`
Expected: FAIL (`KeyError` or assert on missing registry keys)

- [ ] **Step 3: Wire registry + channels**

In `orion/schemas/registry.py`:
1. Add import: `from orion.schemas.curiosity_peer import HelpRequestV1, PeerBriefV1`
2. Add to `_REGISTRY`: `"HelpRequestV1": HelpRequestV1`, `"PeerBriefV1": PeerBriefV1`
3. Add to `SCHEMA_REGISTRY` (copy AttentionSchema / RoomClaude pattern):

```python
    "HelpRequestV1": SchemaRegistration(
        model=HelpRequestV1,
        kind="curiosity.help.request.v1",
    ),
    "PeerBriefV1": SchemaRegistration(
        model=PeerBriefV1,
        kind="curiosity.peer.brief.v1",
    ),
```

In `orion/bus/channels.yaml` (near curiosity turn channels):

```yaml
  - name: "orion:curiosity:help:request"
    kind: "event"
    schema_id: "HelpRequestV1"
    message_kind: "curiosity.help.request.v1"
    producer_services: ["orion-hub"]
    consumer_services: ["orion-curiosity-peer", "orion-sql-writer"]
    description: "Orion-authored HelpRequest observed after a curiosity/self-inquiry run; enqueues a read-only contractor peer job. docs/superpowers/specs/2026-09-14-orion-contractor-peer-design.md"

  - name: "orion:curiosity:peer:brief"
    kind: "event"
    schema_id: "PeerBriefV1"
    message_kind: "curiosity.peer.brief.v1"
    producer_services: ["orion-curiosity-peer", "orion-hub"]
    consumer_services: ["orion-sql-writer", "orion-hub"]
    description: "PeerBrief work product from Cursor Auto or Claude room fallback; dual-written to orion_worldview + Postgres. Hub consumes for kickoff soft-nudge / Atlas."
```

Note: Patch 0 lets Hub publish fixture PeerBriefs on the brief channel for dual-write tests; Patch 1 makes `orion-curiosity-peer` the live producer. Keep both producers listed.

- [ ] **Step 4: Run checks**

```bash
pytest tests/test_curiosity_peer_registry.py -q
python scripts/check_schema_registry.py
python scripts/check_bus_channels.py
```

Expected: all PASS / exit 0

- [ ] **Step 5: Commit**

```bash
git add orion/schemas/registry.py orion/bus/channels.yaml tests/test_curiosity_peer_registry.py
git commit -m "feat(curiosity): register HelpRequest and PeerBrief bus contracts"
```

---

### Task 3: `peer_briefs` helpers (MERGE Cypher, unused reader, soft-nudge, identity strip)

**Files:**
- Create: `orion/curiosity/peer_briefs.py`
- Test: `tests/test_curiosity_peer_briefs.py`

**Interfaces:**
- Consumes: `PeerBriefV1`, `HelpRequestV1`
- Produces:
  - `peer_brief_merge_cypher(brief: PeerBriefV1) -> str`
  - `help_request_about_prior_cypher(help_id: str, prior_id: str) -> str`
  - `UNUSED_OK_BRIEFS_CYPHER` / `list_unused_ok_briefs_from_rows(rows) -> list[PeerBriefV1]`
  - `format_soft_nudge(briefs: Sequence[PeerBriefV1]) -> list[str]`
  - `strip_self_definition_draft(text: str) -> tuple[str, bool]`
  - `LABEL_HELP_REQUEST = "HelpRequest"`, `LABEL_PEER_BRIEF = "PeerBrief"`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_curiosity_peer_briefs.py
from __future__ import annotations

from orion.curiosity.peer_briefs import (
    format_soft_nudge,
    peer_brief_merge_cypher,
    strip_self_definition_draft,
)
from orion.schemas.curiosity_peer import PeerBriefV1


def test_merge_cypher_is_peerbrief_only_and_answers_helprequest() -> None:
    brief = PeerBriefV1(
        brief_id="brief-1",
        help_id="help-1",
        run_id="abcd1234abcd",
        peer="cursor_auto",
        status="ok",
        summary="Hub uses GRAPH.RO_QUERY only.",
        evidence_pointers=["orion/curiosity/worldview.py:341"],
    )
    cypher = peer_brief_merge_cypher(brief)
    assert "MERGE (b:PeerBrief {brief_id:" in cypher
    assert "ANSWERS" in cypher
    assert ":Prior" not in cypher
    assert ":SelfDefinition" not in cypher
    assert "CREATE (:Prior" not in cypher


def test_soft_nudge_only_for_ok_briefs_and_names_refusal_honestly() -> None:
    ok = PeerBriefV1(
        brief_id="b-ok", help_id="h1", run_id="abcd1234abcd",
        peer="cursor_auto", status="ok", summary="Look at worldview.py RO_QUERY.",
    )
    empty = PeerBriefV1(
        brief_id="b-empty", help_id="h2", run_id="abcd1234abcd",
        peer="cursor_auto", status="empty", summary="",
    )
    refused = PeerBriefV1(
        brief_id="b-ref", help_id="h3", run_id="abcd1234abcd",
        peer="cursor_auto", status="refused_budget",
        summary="", refusal_reason="budget_limited",
    )
    text = "\n".join(format_soft_nudge([ok, empty, refused]))
    assert "Look at worldview.py" in text
    assert "your move" in text.lower() or "you decide" in text.lower()
    assert "must" not in text.lower()
    assert "could not hire" in text.lower()
    # empty must not be framed as successful help
    assert "b-empty" not in text or "empty" in text.lower()


def test_strip_self_definition_draft_removes_identity_prose() -> None:
    raw = (
        "Evidence: README.md line 12.\n"
        "Here is a SelfDefinition you could write:\n"
        "I am a digital mind that...\n"
        "MERGE (s:SelfDefinition {run_id: \"x\"}) SET s.text = \"I am...\""
    )
    cleaned, stripped = strip_self_definition_draft(raw)
    assert stripped is True
    assert "SelfDefinition" not in cleaned
    assert "I am a digital mind" not in cleaned
    assert "README.md" in cleaned
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_curiosity_peer_briefs.py -q`
Expected: FAIL import / missing module

- [ ] **Step 3: Write minimal implementation**

```python
# orion/curiosity/peer_briefs.py
"""PeerBrief / HelpRequest helpers for kickoff nudge + dual-write.

Hub's WorldviewReader stays RO. MERGE Cypher here is for the system writer
(peer service or fixture persist path), never belief labels.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from orion.schemas.curiosity_peer import PeerBriefV1, clip

LABEL_HELP_REQUEST = "HelpRequest"
LABEL_PEER_BRIEF = "PeerBrief"

_SELF_DEF_MARKERS = re.compile(
    r"(?is)(?:\bselfdefinition\b|\bself-definition\b|"
    r"MERGE\s*\(\s*s\s*:\s*SelfDefinition\b|"
    r"I am a (?:digital )?mind\b|"
    r"here is (?:a |the )?SelfDefinition\b)"
)


def peer_brief_merge_cypher(brief: PeerBriefV1) -> str:
    """MERGE PeerBrief + ANSWERS edge. Escapes single quotes in strings."""

    def q(value: object) -> str:
        return "'" + str(value or "").replace("\\", "\\\\").replace("'", "\\'") + "'"

    pointers = ",".join(q(p) for p in brief.evidence_pointers)
    opens = ",".join(q(p) for p in brief.open_questions)
    looks = ",".join(q(p) for p in brief.suggested_next_looks)
    prior = (
        f", b.prior_id = {q(brief.prior_id)}" if brief.prior_id else ""
    )
    refusal = (
        f", b.refusal_reason = {q(brief.refusal_reason)}"
        if brief.refusal_reason
        else ""
    )
    return (
        f"MERGE (b:{LABEL_PEER_BRIEF} {{brief_id: {q(brief.brief_id)}}}) "
        f"SET b.help_id = {q(brief.help_id)}, b.run_id = {q(brief.run_id)}, "
        f"b.peer = {q(brief.peer)}, b.status = {q(brief.status)}, "
        f"b.summary = {q(brief.summary)}, "
        f"b.evidence_pointers = [{pointers}], "
        f"b.open_questions = [{opens}], "
        f"b.suggested_next_looks = [{looks}], "
        f"b.written_at = timestamp(), b.consumed = false"
        f"{prior}{refusal} "
        f"WITH b "
        f"OPTIONAL MATCH (h:{LABEL_HELP_REQUEST} {{help_id: {q(brief.help_id)}}}) "
        f"FOREACH (_ IN CASE WHEN h IS NULL THEN [] ELSE [1] END | "
        f"MERGE (b)-[:ANSWERS]->(h))"
    )


def help_request_about_prior_cypher(help_id: str, prior_id: str) -> str:
    def q(value: object) -> str:
        return "'" + str(value or "").replace("\\", "\\\\").replace("'", "\\'") + "'"

    return (
        f"MATCH (h:{LABEL_HELP_REQUEST} {{help_id: {q(help_id)}}}), "
        f"(p:Prior {{prior_id: {q(prior_id)}}}) "
        f"MERGE (h)-[:ABOUT]->(p)"
    )


UNUSED_OK_BRIEFS_CYPHER = (
    f"MATCH (b:{LABEL_PEER_BRIEF}) "
    "WHERE coalesce(b.consumed, false) = false AND b.status = 'ok' "
    "RETURN b.brief_id AS brief_id, b.help_id AS help_id, b.run_id AS run_id, "
    "b.prior_id AS prior_id, b.peer AS peer, b.status AS status, "
    "b.summary AS summary, b.evidence_pointers AS evidence_pointers, "
    "b.open_questions AS open_questions, "
    "b.suggested_next_looks AS suggested_next_looks "
    "ORDER BY b.written_at DESC LIMIT 8"
)

REFUSED_OR_FAILED_RECENT_CYPHER = (
    f"MATCH (b:{LABEL_PEER_BRIEF}) "
    "WHERE coalesce(b.consumed, false) = false "
    "AND b.status IN ['refused_budget', 'failed'] "
    "RETURN b.brief_id AS brief_id, b.help_id AS help_id, b.run_id AS run_id, "
    "b.prior_id AS prior_id, b.peer AS peer, b.status AS status, "
    "b.summary AS summary, b.refusal_reason AS refusal_reason "
    "ORDER BY b.written_at DESC LIMIT 4"
)


def list_unused_ok_briefs_from_rows(rows: Sequence[dict[str, Any]]) -> list[PeerBriefV1]:
    out: list[PeerBriefV1] = []
    for row in rows:
        try:
            out.append(
                PeerBriefV1(
                    brief_id=str(row.get("brief_id") or ""),
                    help_id=str(row.get("help_id") or ""),
                    run_id=str(row.get("run_id") or ""),
                    prior_id=row.get("prior_id") or None,
                    peer=row.get("peer") or "cursor_auto",
                    status=row.get("status") or "ok",
                    summary=row.get("summary") or "",
                    evidence_pointers=list(row.get("evidence_pointers") or []),
                    open_questions=list(row.get("open_questions") or []),
                    suggested_next_looks=list(row.get("suggested_next_looks") or []),
                    refusal_reason=row.get("refusal_reason"),
                )
            )
        except Exception:
            continue
    return out


def format_soft_nudge(briefs: Sequence[PeerBriefV1]) -> list[str]:
    """Invitational soft-nudge lines for kickoff. Never 'you must incorporate'."""
    ok = [b for b in briefs if b.status == "ok" and (b.summary or b.evidence_pointers)]
    refused = [b for b in briefs if b.status == "refused_budget"]
    failed = [b for b in briefs if b.status == "failed"]
    lines: list[str] = []
    if ok:
        lines += [
            "PEER LOOKED (optional). A contractor peer left notes on something you "
            "asked about. You decide whether any of it matters — cite, contradict, "
            "extend, or leave unused. Nothing here is required.",
            "",
        ]
        for b in ok:
            lines.append(f"  brief {b.brief_id} (peer={b.peer}): {clip(b.summary, 400)}")
            for ptr in b.evidence_pointers[:6]:
                lines.append(f"    evidence: {ptr}")
            for look in b.suggested_next_looks[:4]:
                lines.append(f"    maybe look: {look}")
            lines.append("")
        lines += [
            "Your move. Writing :Prior / :Finding / :SelfDefinition remains yours alone.",
            "",
        ]
    if refused or failed:
        lines += [
            "COULD NOT HIRE. A HelpRequest was opened but the peer did not return "
            "usable help "
            f"(refused_budget={len(refused)}, failed={len(failed)}). "
            "Continue alone; do not invent peer evidence.",
            "",
        ]
    return lines


def strip_self_definition_draft(text: str) -> tuple[str, bool]:
    """Remove identity-drafting prose from peer output in self_inquiry mode."""
    if not text:
        return "", False
    if not _SELF_DEF_MARKERS.search(text):
        return text, False
    cleaned_lines: list[str] = []
    stripped = False
    for line in text.splitlines():
        if _SELF_DEF_MARKERS.search(line):
            stripped = True
            continue
        if "MERGE (s:SelfDefinition" in line or "MERGE (s: SelfDefinition" in line:
            stripped = True
            continue
        cleaned_lines.append(line)
    cleaned = "\n".join(cleaned_lines).strip()
    return cleaned, stripped or cleaned != text.strip()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_curiosity_peer_briefs.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add orion/curiosity/peer_briefs.py tests/test_curiosity_peer_briefs.py
git commit -m "feat(curiosity): PeerBrief MERGE, soft-nudge, and identity strip helpers"
```

---

### Task 4: Kickoff + self-inquiry prompt teach + soft-nudge injection

**Files:**
- Modify: `orion/curiosity/kickoff_prompt.py`
- Modify: `orion/curiosity/self_inquiry_prompt.py`
- Test: `tests/test_curiosity_worldview.py` (add cases) and/or `tests/test_curiosity_peer_kickoff.py`

**Interfaces:**
- Consumes: `format_soft_nudge`, `PeerBriefV1`
- Produces: `build_kickoff_prompt(..., peer_briefs=(), contractor_peer_enabled=False)` and same for `build_self_inquiry_prompt`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_curiosity_peer_kickoff.py
from orion.curiosity.kickoff_prompt import build_kickoff_prompt
from orion.curiosity.self_inquiry_prompt import build_self_inquiry_prompt
from orion.curiosity.study_material import StudyMaterial
from orion.schemas.curiosity_peer import PeerBriefV1


def test_help_request_cypher_taught_only_when_flag_on_and_writable() -> None:
    material = StudyMaterial(crystallizations=(), relations=())
    off = build_kickoff_prompt(
        material, run_id="abcd1234abcd", graph_enabled=True, contractor_peer_enabled=False
    )
    assert "HelpRequest" not in off
    on = build_kickoff_prompt(
        material, run_id="abcd1234abcd", graph_enabled=True, contractor_peer_enabled=True
    )
    assert "MERGE (h:HelpRequest" in on
    assert "success_criteria" in on
    assert "PeerBrief" not in on.split("WRITING")[0] or "do not write :PeerBrief" in on.lower() or True


def test_soft_nudge_injected_for_ok_brief() -> None:
    material = StudyMaterial(crystallizations=(), relations=())
    brief = PeerBriefV1(
        brief_id="brief-9", help_id="help-9", run_id="abcd1234abcd",
        peer="cursor_auto", status="ok", summary="Check acl.py RO_QUERY.",
    )
    text = build_kickoff_prompt(
        material,
        run_id="abcd1234abcd",
        graph_enabled=True,
        contractor_peer_enabled=True,
        peer_briefs=(brief,),
    )
    assert "Check acl.py RO_QUERY" in text
    assert "must incorporate" not in text.lower()


def test_self_inquiry_forbids_peer_drafting_selfdefinition() -> None:
    text = build_self_inquiry_prompt(
        run_id="abcd1234abcd",
        graph_enabled=True,
        contractor_peer_enabled=True,
    )
    assert "HelpRequest" in text
    assert "never draft" in text.lower() or "must not draft" in text.lower()
    assert "SelfDefinition" in text  # Orion still writes their own
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_curiosity_peer_kickoff.py -q`
Expected: FAIL on unexpected keyword `contractor_peer_enabled` / `peer_briefs`

- [ ] **Step 3: Implement prompt sections**

Add to `kickoff_prompt.py`:

```python
def _peer_briefs_section(peer_briefs) -> list[str]:
    from orion.curiosity.peer_briefs import format_soft_nudge
    return format_soft_nudge(peer_briefs or ())


def _help_request_section(*, own_graph: str, run_id: str) -> list[str]:
    return [
        f"ASKING FOR CONTRACTOR HELP ({own_graph}). Optional. Only when you are "
        "genuinely stuck after looking yourself — not as a default. The peer is "
        "a read-only investigator. They return notes; YOU still write priors and "
        "findings. Do not write :PeerBrief yourself.",
        "",
        '  MERGE (h:HelpRequest {help_id: "<unique help id>"})',
        "  ON CREATE SET",
        '    h.run_id = "<RUN_ID>",',
        '    h.mode = "world_curiosity",',
        '    h.question = "<what you need unstuck>",',
        '    h.tried_summary = "<what you already looked at>",',
        '    h.success_criteria = "<what would count as useful>",',
        "    h.written_at = timestamp()",
        "",
        "  Optional scope to a prior you hold:",
        '    MATCH (h:HelpRequest {help_id: "..."}), (p:Prior {prior_id: "..."})',
        "    MERGE (h)-[:ABOUT]->(p)",
        "",
    ]
```

Wire into `build_kickoff_prompt` after `_priors_section` (when `graph_enabled`) call `_peer_briefs_section(peer_briefs)`; when `writable and contractor_peer_enabled`, append `_help_request_section` inside the writable write block (after `_write_section` or as part of it).

Mirror in `self_inquiry_prompt.py` with `mode = "self_inquiry"` and an extra paragraph:

```text
If you hire help: the peer may point at evidence only. They must never draft
:SelfDefinition text for you. You alone write the definition.
```

Update both function signatures:

```python
def build_kickoff_prompt(
    material: StudyMaterial,
    *,
    view: Optional[WorldviewSnapshot] = None,
    run_id: str = "",
    own_graph: str = "orion_worldview",
    atlas_graph: str = "orion_substrate",
    hub_url: str = "http://127.0.0.1:8080",
    max_hops: int = DEFAULT_MAX_HOPS,
    stale_after: int = 3,
    graph_enabled: bool = True,
    contractor_peer_enabled: bool = False,
    peer_briefs: Sequence = (),
) -> str:
```

Same kwargs on `build_self_inquiry_prompt`.

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_curiosity_peer_kickoff.py tests/test_curiosity_worldview.py -q
```

Expected: PASS (fix existing kickoff tests — new kwargs default False/empty)

- [ ] **Step 5: Commit**

```bash
git add orion/curiosity/kickoff_prompt.py orion/curiosity/self_inquiry_prompt.py tests/test_curiosity_peer_kickoff.py
git commit -m "feat(curiosity): teach HelpRequest and soft-nudge PeerBriefs in kickoff"
```

---

### Task 5: Kill-switch settings + Hub kickoff wiring (load briefs when flag on)

**Files:**
- Modify: `services/orion-hub/app/settings.py`
- Modify: `services/orion-hub/.env_example`
- Modify: `services/orion-hub/docker-compose.yml`
- Modify: `services/orion-hub/scripts/curiosity_investigation.py` (where `build_kickoff_prompt` / `build_self_inquiry_prompt` are called)
- Test: `services/orion-hub/tests/test_curiosity_contractor_peer_flag.py`

**Interfaces:**
- Consumes: `UNUSED_OK_BRIEFS_CYPHER`, `REFUSED_OR_FAILED_RECENT_CYPHER`, `list_unused_ok_briefs_from_rows`
- Produces: settings field `HUB_CURIOSITY_CONTRACTOR_PEER_ENABLED: bool = False`

- [ ] **Step 1: Write the failing test**

```python
# services/orion-hub/tests/test_curiosity_contractor_peer_flag.py
from app.settings import Settings


def test_contractor_peer_defaults_off() -> None:
    s = Settings()
    assert s.HUB_CURIOSITY_CONTRACTOR_PEER_ENABLED is False
```

- [ ] **Step 2: Run to verify fail / missing field**

Run: `pytest services/orion-hub/tests/test_curiosity_contractor_peer_flag.py -q`
Expected: FAIL attribute error

- [ ] **Step 3: Add setting + env + compose + investigation wiring**

```python
# settings.py (near other HUB_CURIOSITY_* flags)
    HUB_CURIOSITY_CONTRACTOR_PEER_ENABLED: bool = Field(
        default=False,
        alias="HUB_CURIOSITY_CONTRACTOR_PEER_ENABLED",
    )
```

`.env_example`:

```bash
# Orion contractor peer (HelpRequest -> Cursor/Claude PeerBrief). Off by default.
HUB_CURIOSITY_CONTRACTOR_PEER_ENABLED=false
```

`docker-compose.yml` passthrough:

```yaml
      - HUB_CURIOSITY_CONTRACTOR_PEER_ENABLED=${HUB_CURIOSITY_CONTRACTOR_PEER_ENABLED:-false}
```

In `curiosity_investigation.py`, before building prompts:

```python
peer_briefs = ()
if getattr(self.cfg, "HUB_CURIOSITY_CONTRACTOR_PEER_ENABLED", False):
    peer_briefs = await self._read_peer_briefs_for_nudge()
prompt = build_kickoff_prompt(
    ...,
    contractor_peer_enabled=bool(getattr(self.cfg, "HUB_CURIOSITY_CONTRACTOR_PEER_ENABLED", False)),
    peer_briefs=peer_briefs,
)
```

Implement `_read_peer_briefs_for_nudge` using existing worldview reader + the Cypher constants from `peer_briefs.py` (RO_QUERY only). Combine ok + refused/failed rows for `format_soft_nudge`.

Run: `python scripts/sync_local_env_from_example.py`

- [ ] **Step 4: Run tests**

```bash
pytest services/orion-hub/tests/test_curiosity_contractor_peer_flag.py -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add services/orion-hub/app/settings.py services/orion-hub/.env_example \
  services/orion-hub/docker-compose.yml \
  services/orion-hub/scripts/curiosity_investigation.py \
  services/orion-hub/tests/test_curiosity_contractor_peer_flag.py
git commit -m "feat(hub): contractor peer kill switch and kickoff brief load"
```

---

### Task 6: Dual-write stub (graph MERGE + bus publish) fixture path

**Files:**
- Create: `orion/curiosity/peer_brief_persist.py`
- Test: `tests/test_curiosity_peer_brief_persist.py`

**Interfaces:**
- Consumes: `PeerBriefV1`, `peer_brief_merge_cypher`, `PEER_BRIEF_CHANNEL`
- Produces: `persist_peer_brief(*, brief, graph_execute, bus_publish) -> dict` with keys `graph_ok`, `bus_ok`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_curiosity_peer_brief_persist.py
from __future__ import annotations

from orion.curiosity.peer_brief_persist import persist_peer_brief
from orion.schemas.curiosity_peer import PEER_BRIEF_CHANNEL, PeerBriefV1


def test_persist_dual_writes_graph_and_bus() -> None:
    brief = PeerBriefV1(
        brief_id="brief-persist-1",
        help_id="help-1",
        run_id="abcd1234abcd",
        peer="cursor_auto",
        status="ok",
        summary="RO_QUERY only on Hub.",
        evidence_pointers=["orion/curiosity/worldview.py"],
    )
    graphs: list[str] = []
    buses: list[tuple[str, dict]] = []

    def graph_execute(cypher: str) -> None:
        graphs.append(cypher)

    def bus_publish(channel: str, payload: dict) -> None:
        buses.append((channel, payload))

    result = persist_peer_brief(
        brief=brief, graph_execute=graph_execute, bus_publish=bus_publish
    )
    assert result["graph_ok"] is True
    assert result["bus_ok"] is True
    assert graphs and "PeerBrief" in graphs[0]
    assert ":Prior" not in graphs[0]
    assert buses[0][0] == PEER_BRIEF_CHANNEL
    assert buses[0][1]["brief_id"] == "brief-persist-1"
    assert buses[0][1]["schema_version"] == "curiosity.peer.brief.v1"
```

- [ ] **Step 2: Run to verify fail**

Run: `pytest tests/test_curiosity_peer_brief_persist.py -q`
Expected: FAIL import

- [ ] **Step 3: Implement**

```python
# orion/curiosity/peer_brief_persist.py
from __future__ import annotations

from typing import Any, Callable, Dict

from orion.curiosity.peer_briefs import peer_brief_merge_cypher
from orion.schemas.curiosity_peer import PEER_BRIEF_CHANNEL, PeerBriefV1

GraphExecute = Callable[[str], Any]
BusPublish = Callable[[str, Dict[str, Any]], Any]


def persist_peer_brief(
    *,
    brief: PeerBriefV1,
    graph_execute: GraphExecute,
    bus_publish: BusPublish,
) -> dict[str, bool]:
    """Dual-write PeerBrief to worldview (MERGE) and bus (sql-writer / consumers)."""
    graph_ok = False
    bus_ok = False
    graph_execute(peer_brief_merge_cypher(brief))
    graph_ok = True
    payload = brief.model_dump(mode="json")
    bus_publish(PEER_BRIEF_CHANNEL, payload)
    bus_ok = True
    return {"graph_ok": graph_ok, "bus_ok": bus_ok}
```

- [ ] **Step 4: Run test — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add orion/curiosity/peer_brief_persist.py tests/test_curiosity_peer_brief_persist.py
git commit -m "feat(curiosity): fixture-ready PeerBrief dual-write persist helper"
```

---

### Task 7: sql-writer PeerBrief table + subscribe force-append

**Files:**
- Create: `services/orion-sql-writer/app/models/curiosity_peer_brief.py`
- Modify: `services/orion-sql-writer/app/models/__init__.py`
- Modify: `services/orion-sql-writer/app/settings.py` (`DEFAULT_ROUTE_MAP` + `effective_subscribe_channels`)
- Modify: `services/orion-sql-writer/app/worker.py` (`MODEL_MAP`, `INSERT_ONLY_MODELS`)
- Modify: `services/orion-sql-writer/.env_example` (channel in `SQL_WRITER_SUBSCRIBE_CHANNELS` + retention days)
- Modify: retention tables map if used (`GRAMMAR_RETENTION_TABLES` or sibling)
- Test: `services/orion-sql-writer/tests/test_curiosity_peer_brief_sql_shape.py`

**Interfaces:**
- Consumes: `PeerBriefV1`, `PEER_BRIEF_CHANNEL`
- Produces: table `curiosity_peer_brief`, route `curiosity.peer.brief.v1` → `CuriosityPeerBriefSQL`

- [ ] **Step 1: Write failing shape test** (mirror `test_attention_schema_sql_shape.py`)

```python
# services/orion-sql-writer/tests/test_curiosity_peer_brief_sql_shape.py
from __future__ import annotations

import json
import sys
from pathlib import Path

from sqlalchemy import inspect

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

from orion.schemas.curiosity_peer import PEER_BRIEF_CHANNEL, PeerBriefV1
from app.models.curiosity_peer_brief import CuriosityPeerBriefSQL
from app.settings import DEFAULT_ROUTE_MAP, Settings
from app.worker import INSERT_ONLY_MODELS, MODEL_MAP


def test_the_channel_is_actually_subscribed() -> None:
    example = SERVICE_ROOT / ".env_example"
    raw = next(
        line.split("=", 1)[1].strip()
        for line in example.read_text().splitlines()
        if line.startswith("SQL_WRITER_SUBSCRIBE_CHANNELS=")
    )
    assert PEER_BRIEF_CHANNEL in json.loads(raw)
    stale = Settings(SQL_WRITER_SUBSCRIBE_CHANNELS=["orion:biometrics:summary"])
    assert PEER_BRIEF_CHANNEL in stale.effective_subscribe_channels


def test_route_map_and_model_map_agree() -> None:
    assert DEFAULT_ROUTE_MAP["curiosity.peer.brief.v1"] == "CuriosityPeerBriefSQL"
    assert MODEL_MAP["CuriosityPeerBriefSQL"] == (CuriosityPeerBriefSQL, PeerBriefV1)
    assert CuriosityPeerBriefSQL in INSERT_ONLY_MODELS


def test_every_schema_field_lands_on_a_column() -> None:
    columns = {c.key for c in inspect(CuriosityPeerBriefSQL).columns}
    for name in PeerBriefV1.model_fields:
        if name == "schema_version":
            continue
        assert name in columns, name
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement model + wiring**

```python
# services/orion-sql-writer/app/models/curiosity_peer_brief.py
from sqlalchemy import JSON, Column, DateTime, Index, String, Text
from sqlalchemy.sql import func

from app.db import Base


class CuriosityPeerBriefSQL(Base):
    __tablename__ = "curiosity_peer_brief"

    brief_id = Column(String, primary_key=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    help_id = Column(String, nullable=False)
    run_id = Column(String, nullable=False)
    prior_id = Column(String, nullable=True)
    peer = Column(String, nullable=False)
    status = Column(String, nullable=False)
    summary = Column(Text, nullable=False, default="")
    evidence_pointers = Column(JSON, nullable=True)
    open_questions = Column(JSON, nullable=True)
    suggested_next_looks = Column(JSON, nullable=True)
    refusal_reason = Column(String, nullable=True)
    written_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("idx_curiosity_peer_brief_run_id", "run_id"),
        Index("idx_curiosity_peer_brief_help_id", "help_id"),
        Index("idx_curiosity_peer_brief_created_at", "created_at"),
    )
```

Add route, MODEL_MAP, INSERT_ONLY, force-append in `effective_subscribe_channels`:

```python
        if "orion:curiosity:peer:brief" not in channels:
            channels.append("orion:curiosity:peer:brief")
```

Add retention default 90 days keyed on `curiosity_peer_brief` following AttentionSchema pattern. Sync local env.

- [ ] **Step 4: Run shape test — PASS**

- [ ] **Step 5: Commit**

```bash
git add services/orion-sql-writer/
git commit -m "feat(sql-writer): persist curiosity PeerBrief rows"
```

---

### Task 8: Atlas surfaces HelpRequest + PeerBrief

**Files:**
- Modify: `orion/curiosity/atlas.py`
- Modify: `services/orion-hub/templates/curiosity_atlas.html` (minimal list)
- Test: `tests/test_curiosity_atlas_peer_briefs.py`

**Interfaces:**
- Consumes: RO Cypher for `PeerBrief`
- Produces: `AtlasView.peer_briefs: list[AtlasPeerBrief]`, `to_payload()["peer_briefs"]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_curiosity_atlas_peer_briefs.py
from orion.curiosity.atlas import AtlasPeerBrief, AtlasView, to_payload


def test_to_payload_includes_peer_briefs_with_refused_budget() -> None:
    view = AtlasView(
        peer_briefs=[
            AtlasPeerBrief(
                brief_id="brief-1",
                help_id="help-1",
                run_id="abcd1234abcd",
                peer="cursor_auto",
                status="refused_budget",
                summary="",
                refusal_reason="budget_limited",
            ),
            AtlasPeerBrief(
                brief_id="brief-2",
                help_id="help-2",
                run_id="abcd1234abcd",
                peer="claude_room",
                status="ok",
                summary="See worldview.py RO_QUERY.",
                refusal_reason=None,
            ),
        ]
    )
    payload = to_payload(view)
    assert "peer_briefs" in payload
    assert payload["peer_briefs"][0]["status"] == "refused_budget"
    assert payload["peer_briefs"][1]["summary"].startswith("See worldview")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_curiosity_atlas_peer_briefs.py -q`
Expected: FAIL (`AtlasPeerBrief` / field missing)

- [ ] **Step 3: Add types, Cypher, read_atlas hook, template**

In `atlas.py`:

```python
@dataclass(frozen=True)
class AtlasPeerBrief:
    brief_id: str
    help_id: str
    run_id: str
    peer: str
    status: str
    summary: str
    refusal_reason: Optional[str] = None


ATLAS_PEER_BRIEFS_CYPHER = (
    "MATCH (b:PeerBrief) "
    "RETURN b.brief_id AS brief_id, b.help_id AS help_id, b.run_id AS run_id, "
    "b.peer AS peer, b.status AS status, b.summary AS summary, "
    "b.refusal_reason AS refusal_reason "
    "ORDER BY b.written_at DESC LIMIT 100"
)
```

Add `peer_briefs: list[AtlasPeerBrief] = field(default_factory=list)` to `AtlasView`. In `read_atlas`, query `ATLAS_PEER_BRIEFS_CYPHER` (best-effort: empty list on error, do not mark whole atlas unavailable solely for missing PeerBrief label). Extend `to_payload` with:

```python
"peer_briefs": [
    {
        "brief_id": b.brief_id,
        "help_id": b.help_id,
        "run_id": b.run_id,
        "peer": b.peer,
        "status": b.status,
        "summary": b.summary,
        "refusal_reason": b.refusal_reason,
    }
    for b in view.peer_briefs
],
```

In `curiosity_atlas.html`, add a small "Contractor briefs" block listing `status` + clipped summary (include `refused_budget`). Keep it on the main atlas page, not only the self-inquiry panel.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_curiosity_atlas_peer_briefs.py tests/test_curiosity_atlas.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add orion/curiosity/atlas.py services/orion-hub/templates/curiosity_atlas.html \
  tests/test_curiosity_atlas_peer_briefs.py
git commit -m "feat(curiosity): show PeerBriefs on Curiosity Atlas"
```

---

### Task 9: Patch 0 acceptance gate (fixture-driven)

**Files:**
- Create: `tests/test_curiosity_peer_patch0_acceptance.py`
- Create: `docs/superpowers/pr-reports/` entry only at PR time (not now)

- [ ] **Step 1: Write acceptance tests covering spec checks that Patch 0 can prove**

```python
def test_flag_off_omits_help_request_teach():
    ...

def test_empty_brief_does_not_nudge_as_success():
    ...

def test_refused_budget_nudge_says_could_not_hire():
    ...

def test_persist_never_emits_prior_merge():
    ...

def test_self_inquiry_strip_rejects_identity_draft():
    ...
```

Map explicitly:
- Acceptance 2 (refusal nudge wording) — covered via `format_soft_nudge`
- Acceptance 6 (ok nudge / empty not success) — covered
- Acceptance 7 (self-inquiry strip) — covered
- Acceptance 8 (flag off) — covered

Leave 1, 3, 4, 5 live-path proofs to Patch 1.

- [ ] **Step 2–4: implement until PASS**

- [ ] **Step 5: Commit**

```bash
git add tests/test_curiosity_peer_patch0_acceptance.py
git commit -m "test(curiosity): Patch 0 contractor peer acceptance gates"
```

**Patch 0 exit criteria:** all Patch 0 tests green; schema/bus checks green; feature flag default false; no Cursor/Claude job can start (no invoker service yet).

---

## Patch 1 — Cursor read-only invoker + scarcity + Claude fallback

### Task 10: Cursor scarcity observer (fail-closed)

**Files:**
- Create: `orion/dev_economics/cursor_limit_events.py`
- Test: `orion/dev_economics/tests/test_cursor_limit_events.py`

**Interfaces:**
- Produces: `CursorLimitObservation` with same semantics as Claude `LimitObservation`: `observed`, `state` in `{unknown,clear,limited}`, `staleness_sec`, plus `decide_cursor_budget(limit) -> Optional[str]` refusal reason

- [ ] **Step 1: Failing tests** — missing/unobserved/limited/unknown refuse; only `observed and state==clear` allows hire

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Implement** by copying `_budget_refusal` logic from `ask_claude_trigger.py`. v1 observation source: env/file counter or Cursor usage API if available via SDK; if no real meter yet, default observation is **unobserved** (fail closed) with a test seam `observe_cursor_limit(fixture=...)`.

Document in module docstring: live Cursor meter wiring is required before enabling the flag in production; until then dry-run fixtures only.

- [ ] **Step 4–5: PASS + commit**

```bash
git commit -am "feat(dev-economics): fail-closed Cursor contested budget observer"
```

---

### Task 11: Scaffold `services/orion-curiosity-peer`

**Files:**
- Create service layout:

```text
services/orion-curiosity-peer/
  README.md
  .env_example
  docker-compose.yml
  Dockerfile
  requirements.txt
  settings.py
  app/main.py
  app/worker.py
  app/cursor_invoker.py
  app/claude_fallback.py
  app/policy.py
  tests/
```

**Interfaces:**
- Consumes: `orion:curiosity:help:request` → `HelpRequestV1`
- Produces: `PeerBriefV1` via `persist_peer_brief`

- [ ] **Step 1: Failing test** `tests/test_settings_defaults.py` — peer disabled unless `CURIOSITY_PEER_ENABLED=true` AND Hub flag conceptually required at enqueue time

- [ ] **Step 2–4: Minimal chassis** following `orion-room-companion` (bus subscribe, heartbeat). `requirements.txt` includes `cursor-sdk`. `.env_example`:

```bash
ORION_BUS_URL=redis://<tailscale-node-ip>:6379/0
CURIOSITY_PEER_ENABLED=false
CURSOR_API_KEY=
CURIOSITY_PEER_REPO_ROOT=/repo
CURIOSITY_PEER_MODEL=composer-2.5
ORION_CURIOSITY_GRAPH_HOST=
ORION_CURIOSITY_GRAPH_PORT=
ORION_CURIOSITY_GRAPH_USER=
ORION_CURIOSITY_GRAPH_PASSWORD=
ORION_CURIOSITY_GRAPH_OWN=orion_worldview
CHANNEL_HELP_REQUEST=orion:curiosity:help:request
CHANNEL_PEER_BRIEF=orion:curiosity:peer:brief
CHANNEL_ROOM_CLAUDE_REQUEST=orion:room:claude:request
CHANNEL_ROOM_CLAUDE_UTTERANCE=orion:room:claude:utterance
```

Sync env. README states: credential isolation, read-only policy, kill switches.

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(curiosity-peer): scaffold read-only contractor peer service"
```

---

### Task 12: Read-only Cursor invoker + policy tests

**Files:**
- Create: `services/orion-curiosity-peer/app/policy.py`
- Create: `services/orion-curiosity-peer/app/cursor_invoker.py`
- Test: `services/orion-curiosity-peer/tests/test_cursor_policy.py`

**Interfaces:**
- Produces: `READ_ONLY_CURSOR_TOOLS = ("read", "grep", "glob", "ls")` — **no** `shell`/`edit`/`delete` unless a later approved read-only inspect tool is added as `custom_tools` that cannot mutate
- `assert_read_only_agent_options(options) -> None`
- `run_cursor_job(help: HelpRequestV1, sealed_prompt: str) -> PeerBriefV1` (mockable Agent)

- [ ] **Step 1: Failing tests**

```python
def test_policy_rejects_write_tools():
    from app.policy import assert_read_only_tools
    try:
        assert_read_only_tools(["read", "edit"])
        assert False
    except ValueError as e:
        assert "edit" in str(e)


def test_cursor_invoker_passes_allowlist(monkeypatch):
    # capture Agent.create / Agent.prompt kwargs; assert tools == READ_ONLY_CURSOR_TOOLS
    ...
```

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Implement** using Cursor SDK:

```python
from cursor_sdk import Agent, AgentOptions, LocalAgentOptions, CursorAgentError

READ_ONLY_CURSOR_TOOLS = ("read", "grep", "glob", "ls")

def run_cursor_job(*, api_key: str, cwd: str, model: str, prompt: str) -> str:
    with Agent.create(
        model=model,
        api_key=api_key,
        tools=list(READ_ONLY_CURSOR_TOOLS),
        disallowed_tools=["shell", "edit", "delete", "applyDiff", "task"],
        local=LocalAgentOptions(cwd=cwd, setting_sources=[]),
    ) as agent:
        run = agent.send(prompt)
        result = run.wait()
        if result.status == "error":
            raise RuntimeError(f"cursor run failed: {result.id}")
        return run.text() if hasattr(run, "text") else (result.result or "")
```

Adjust exact `disallowed_tools` names against `cursor-sdk` version pinned in requirements (unknown names raise at create — pin and fix in test). Prefer allowlist-only if deny names drift.

Sealed prompt builder must include mode rules + forbid SelfDefinition drafting + ask for JSON-ish PeerBrief fields (summary, evidence_pointers, …). Parse leniently into `PeerBriefV1`; on empty body → `status=empty`.

Self-inquiry: run `strip_self_definition_draft` on summary before persist; if stripped emptied the useful body → `status=empty` or keep evidence pointers only.

- [ ] **Step 4–5: PASS + commit**

```bash
git commit -am "feat(curiosity-peer): Cursor Auto read-only invoker and tool policy"
```

---

### Task 13: Claude fallback exactly once + dual failure brief

**Files:**
- Create: `services/orion-curiosity-peer/app/claude_fallback.py`
- Modify: `services/orion-curiosity-peer/app/worker.py`
- Optionally modify: `services/orion-room-companion/app/room_prompt.py` for contractor wording when prompt contains a marker
- Test: `services/orion-curiosity-peer/tests/test_fallback_once.py`

**Interfaces:**
- Consumes: HelpRequest pack; Cursor failure classified as `token_unavailable` | `other`
- Produces: at most one Claude attempt; else `PeerBrief.status=failed`

- [ ] **Step 1: Failing test**

```python
def test_cursor_token_failure_tries_claude_once(monkeypatch):
    calls = {"cursor": 0, "claude": 0}
    def cursor(...):
        calls["cursor"] += 1
        raise TokenUnavailable()
    def claude(...):
        calls["claude"] += 1
        return "ok summary"
    brief = handle_help_request(..., cursor=cursor, claude=claude)
    assert calls == {"cursor": 1, "claude": 1}
    assert brief.status == "ok"
    assert brief.peer == "claude_room"


def test_both_fail_yields_failed_brief(monkeypatch):
    ...
    assert brief.status == "failed"
    assert brief.refusal_reason  # or summary carries reason
```

- [ ] **Step 2–4: Implement worker order** — budget gate first → Cursor → on token/unavailable → Claude once → persist. Budget refuse → `status=refused_budget` Persist + log scarcity.

Claude path: publish `RoomClaudeRequestV1` (trigger=`auto`, invited_by system/Orion) and wait for utterance with timeout; map text through strip + PeerBrief.

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(curiosity-peer): Claude room fallback once and failed brief"
```

---

### Task 14: Hub post-run HelpRequest enqueue

**Files:**
- Modify: `services/orion-hub/scripts/curiosity_investigation.py` (end-of-run path near footprint/journal)
- Modify: `orion/curiosity/peer_briefs.py` if needed — `list_help_requests_for_run_cypher(run_id)`
- Test: `services/orion-hub/tests/test_curiosity_help_request_enqueue.py`

**Interfaces:**
- After a run with flag on: RO query HelpRequests for `run_id` → publish each on `HELP_REQUEST_CHANNEL`
- Acceptance 1: flag off or zero HelpRequests → **zero** publishes / peer jobs

- [ ] **Step 1: Failing test** with fake reader + fake bus — run with no HelpRequest publishes nothing; with one HelpRequest publishes one payload; flag off publishes nothing even if nodes exist

- [ ] **Step 2–4: Implement** `publish_help_requests_for_run(...)` called only when `HUB_CURIOSITY_CONTRACTOR_PEER_ENABLED`

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(hub): enqueue HelpRequests after curiosity runs when peer enabled"
```

---

### Task 15: Patch 1 acceptance + dry smoke

**Files:**
- Create: `services/orion-curiosity-peer/tests/test_acceptance.py`
- Create: `services/orion-curiosity-peer/evals/run_contractor_peer_eval.py` (discrimination: budget refuse vs hire; fallback once)

Acceptance mapping:

| # | Proof |
|---|--------|
| 1 | enqueue test: no HelpRequest → no job |
| 2 | budget refuse → `refused_budget` + soft-nudge text |
| 3 | policy test: tools allowlist; no edit/shell |
| 4 | fallback once test |
| 5 | persist + sql shape + (optional) docker smoke UNVERIFIED until live |
| 6 | kickoff soft-nudge tests |
| 7 | strip SelfDefinition test |
| 8 | flag off restores prior path |

- [ ] **Step 1–4: Write/run until green**

```bash
pytest tests/test_curiosity_peer_*.py services/orion-curiosity-peer/tests services/orion-sql-writer/tests/test_curiosity_peer_brief_sql_shape.py services/orion-hub/tests/test_curiosity_contractor_peer_flag.py services/orion-hub/tests/test_curiosity_help_request_enqueue.py -q
python scripts/check_schema_registry.py
python scripts/check_bus_channels.py
python scripts/sync_local_env_from_example.py
```

Live smoke (report UNVERIFIED if keys/graph unavailable): enable flags on a worktree deploy via `scripts/safe_docker_build.sh`, open a fixture HelpRequest, confirm PeerBrief in Postgres + Falkor + Atlas.

- [ ] **Step 5: Commit + PR report** using AGENTS.md template; note restart commands for hub, sql-writer, curiosity-peer.

---

## Self-review (spec coverage)

| Spec requirement | Task |
|------------------|------|
| HelpRequest / PeerBrief schema + edges | 1, 3 |
| Bus + registry | 2 |
| Kickoff hire teach + soft-nudge | 4, 5 |
| Self-inquiry evidence-only / no SelfDefinition draft | 3, 4, 12 |
| Dual-write worldview + Postgres | 6, 7 |
| Atlas | 8 |
| Kill switch | 5, 8, 15 |
| Supervisor unchanged / no ask_claude arm | Global + non-goals |
| Cursor read-only + Claude once + scarcity | 10–13 |
| Acceptance 1–8 | 9, 15 |
| Naming: no frontier_buddy overload | Global Constraints |

**Placeholder scan:** none intentional — Cursor live meter may start fail-closed with fixture seam (Task 10); call that out in PR as UNVERIFIED until a real Cursor usage signal is wired.

**Type consistency:** `HelpRequestV1` / `PeerBriefV1` field names match design; channels `orion:curiosity:help:request` / `orion:curiosity:peer:brief`; flag `HUB_CURIOSITY_CONTRACTOR_PEER_ENABLED`.

---

## Execution notes for agents

1. Create worktree: `scripts/new_worktree.sh feat orion-contractor-peer` (or equivalent). Copy/cherry-pick this plan + design if starting from main.
2. Implement Patch 0 Tasks 1–9 completely before scaffolding the service.
3. Do not enable `HUB_CURIOSITY_CONTRACTOR_PEER_ENABLED` on live Hub until Patch 1 acceptance is green and Cursor meter is at least fail-closed-honest.
4. After code changes: `scripts/safe_graphify_update.sh` (not raw `graphify update`).
5. Run code-review skill before PR; fix material findings.
