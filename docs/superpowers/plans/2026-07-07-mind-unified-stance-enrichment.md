# Mind → Unified Turn Stance Enrichment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run `orion-mind` inside `orion-thought` before the `stance_react` verb and inject a strict allow-listed, mode-agnostic self/attention "coloring" block as advisory prompt context, so unified-turn stance feels more alive without contradicting grounding or mis-framing tooling/coding turns.

**Architecture:** New `services/orion-thought/app/mind_enrichment.py` holds four pure/async seams: a light Mind snapshot builder, an HTTP client (fail-open), an allow-list coloring selector, and an artifact publisher. `run_stance_react` calls them behind `ORION_THOUGHT_MIND_ENRICHMENT_ENABLED` and threads the coloring into `build_stance_react_context`. The `stance_react` LLM stays the sole author of `ThoughtEventV1` and reconciles Mind's advisory input. Everything is flag-gated (default-off) and fails open to byte-identical current behavior.

**Tech Stack:** Python 3.11, Pydantic v2, `httpx` (new dep), FastAPI service, Redis bus (`OrionBusAsync`), Jinja2 prompt (`stance_react.j2`), pytest.

---

## Background the engineer needs (read before starting)

You are working in the Orion monorepo. Two cognition pipelines exist:

- **Brain / legacy path:** Hub → cortex-gateway → `orion-cortex-orch`. cortex-orch already runs Mind (`services/orion-cortex-orch/app/mind_runtime.py`) and can let Mind drive the stance. **We are NOT touching this path.** It is only a *reference* for how to call Mind over HTTP and publish its artifact.
- **Unified turn path:** Hub `turn_orchestrator` → `orion-thought` (RPC on `orion:thought:request`) → `orion-harness-governor`. `orion-thought` computes stance cold via the `stance_react` verb. **This is the path we enrich.**

Key facts (verified against the code, do not re-derive):

- `StanceReactRequestV1` (`orion/schemas/thought.py:87`) fields: `correlation_id`, `session_id`, `user_message`, `association: HubAssociationBundleV1`, `repair_bundle`, `stance_inputs: dict`, `llm_profile`. There is **no** grounding capsule at request time; the capsule is attached *after* the exec (`bus_listener.py:146`). So the "must not contradict grounding" guarantee is enforced *downstream* by `stance_react`, not at Mind-input time.
- `MindRunResultV1` (`orion/mind/v1.py:134`) has `.ok`, `.mind_run_id: UUID`, `.snapshot_hash: str`, `.brief: MindHandoffBriefV1`, `.decision: MindControlDecisionV1`.
- `MindHandoffBriefV1` (`orion/mind/v1.py:120`) has `.mind_quality: MindQualityV1`, `.active_frontier: ActiveCognitiveFrontierV1 | None`, `.stance_payload: dict` (a raw `ChatStanceBrief` dump), `.shadow_synthesis: MindShadowSynthesisV1 | None`.
- `ActiveCognitiveFrontierV1.selected` is a `list[SelectedFrontierMatterV1]` (`orion/mind/synthesis_v1.py:167`); each item has `.label: str`, `.summary: str`, `.score: float`.
- `ChatStanceBrief` (`orion/schemas/chat_stance.py`) fields we care about: `reflective_themes: list[str]`, `self_relevance: str`, `identity_salience: Literal["low","medium","high"]`, `juniper_relevance: str`. Task-control fields to NEVER select: `task_mode`, `answer_strategy`, `conversation_frame`, `response_priorities`, `response_hazards`.
- `MindRunRequestV1` (`orion/mind/v1.py:38`): `correlation_id`, `session_id`, `trigger`, `snapshot_inputs: dict`, `policy: MindRunPolicyV1`. `MindRunPolicyV1` has `n_loops_max` (ge=1), `wall_time_ms_max` (ge=1), `router_profile_id`.
- Mind's evidence pack (`services/orion-mind/app/evidence.py:120`) reads `snapshot["user_text"]` (→ single `current_turn` item), `snapshot["messages_tail"]`, and `snapshot["facets"]` keyed by `recall_bundle`, `cognitive_projection`/`cognitive_projection_degraded`, `autonomy_compact`, `social_compact`, `situation_compact`, `identity_background`. **Do not invent facet keys.** For `situation_compact` it simply `json.dumps` the whole dict (`evidence.py:229-236`), so any real-text dict works.
- **Hard precondition C1:** `meaningful_synthesis` only comes from `orion-mind`'s LLM path, gated by `MIND_LLM_SYNTHESIS_ENABLED=true` on the *orion-mind* service (a different service — will NOT appear in orion-thought env parity checks). If off, this feature is a silent no-op. This is an operational/rollout check, not a code change here.

### Design decision locked for v1 (autonomous call, flagged)

The design text loosely suggested folding association/broadcast text into an `identity_background` facet. **We do NOT do that** — open loops are *situation/attention* context, not identity, and there is no real identity text pre-grounding. Instead the light snapshot folds the broadcast's `selected_description` + top open-loop `description`s into the accepted **`situation_compact`** facet (real text, bounded, only when non-empty). This is honest (accepted facet, real text, not ornamental) and materially thickens Mind's evidence beyond a bare user turn. `identity_background` is left unpopulated in v1.

## File Structure

- `services/orion-thought/app/mind_enrichment.py` — **new.** All enrichment logic: `build_light_mind_request`, `run_mind_for_thought`, `select_mind_coloring`, `publish_mind_run_artifact_for_thought`. One file, four seams, each independently testable.
- `services/orion-thought/app/settings.py` — **modify.** Add Mind flags/URLs/timeouts.
- `services/orion-thought/app/bus_listener.py` — **modify.** Thread `mind_coloring` param into `build_stance_react_context` / `build_stance_react_plan_request` / `run_stance_react`; call enrichment behind the flag; optional artifact publish.
- `orion/cognition/prompts/stance_react.j2` — **modify.** Add advisory `mind_coloring` block.
- `services/orion-thought/.env_example`, `docker-compose.yml`, `requirements.txt`, `README.md` — **modify.** Config/deps/docs.
- `services/orion-thought/tests/test_*.py` — **new.** Selector, snapshot, context, prompt-render, fail-open, artifact-mode.
- `services/orion-thought/evals/test_mind_enrichment_eval.py` — **new.** Anti-contradiction + aliveness eval.

No changes to `orion/bus/channels.yaml` or `orion/schemas/registry.py` (no new event shape; `MindRunArtifactV1` / `orion:mind:artifact` already registered — verify in Task 8).

---

## Task 0: Branch setup

**Files:** none (git only)

- [ ] **Step 1: Start clean and branch**

Run:

```bash
git status --short
git switch main
git pull --ff-only
git switch -c feat/mind-unified-stance-enrichment
```

Expected: clean tree, new branch checked out. If the tree is dirty with unrelated work, stop and classify per AGENTS.md §2 before continuing.

- [ ] **Step 2: Note the test invocation**

All orion-thought tests run with this prefix (from `services/orion-thought/README.md`). Use it verbatim throughout:

```bash
PYTHONPATH=services/orion-thought:. python -m pytest services/orion-thought/tests/ -q
```

If `python` is not the project interpreter, use `./orion_dev/bin/python` instead (README uses it). Pick whichever resolves `pydantic` + `orion` imports; use it consistently.

---

## Task 1: Coloring selector (pure, allow-list)

**Files:**
- Create: `services/orion-thought/app/mind_enrichment.py`
- Test: `services/orion-thought/tests/test_mind_coloring_selector.py`

This is the load-bearing USE/DROP split. It is a strict allow-list (copy only named keys), never a deny-list.

- [ ] **Step 1: Write the failing test**

Create `services/orion-thought/tests/test_mind_coloring_selector.py`:

```python
from __future__ import annotations

from uuid import uuid4

from app.mind_enrichment import MIND_COLORING_ALLOWED_KEYS, select_mind_coloring
from orion.mind.synthesis_v1 import ActiveCognitiveFrontierV1, SelectedFrontierMatterV1
from orion.mind.v1 import MindControlDecisionV1, MindHandoffBriefV1, MindRunResultV1


def _selected(label: str, summary: str, score: float) -> SelectedFrontierMatterV1:
    return SelectedFrontierMatterV1(
        matter_id=f"m-{label}",
        source_claim_id=f"c-{label}",
        label=label,
        summary=summary,
        matter_kind="curiosity_affordance",
        score=score,
    )


def _stance_payload() -> dict:
    # A full ChatStanceBrief dump — carries BOTH self-relational and task-control fields.
    return {
        "conversation_frame": "reflective",
        "task_mode": "reflective_dialogue",
        "identity_salience": "high",
        "user_intent": "connect",
        "self_relevance": "This touches my continuity with Juniper.",
        "juniper_relevance": "Juniper is checking in on me.",
        "reflective_themes": ["continuity", "trust", "the shape of our work"],
        "response_priorities": ["companion_presence"],
        "response_hazards": ["avoid_task_tracking"],
        "answer_strategy": "companion",
        "stance_summary": "warm, present",
    }


def _result(*, ok: bool, quality: str, with_frontier: bool = True) -> MindRunResultV1:
    frontier = None
    if with_frontier:
        frontier = ActiveCognitiveFrontierV1(
            selected=[
                _selected("continuity", "the unresolved thread about our last session", 0.91),
                _selected("trust", "whether Juniper felt heard last time", 0.77),
                _selected("curiosity", "what changed since we last spoke", 0.62),
                _selected("overflow", "a fourth item that must be truncated away", 0.40),
            ]
        )
    brief = MindHandoffBriefV1(
        mind_quality=quality,  # type: ignore[arg-type]
        active_frontier=frontier,
        stance_payload=_stance_payload(),
        shadow_synthesis=None,  # proves no projection dependency
    )
    return MindRunResultV1(
        mind_run_id=uuid4(),
        ok=ok,
        snapshot_hash="deadbeef",
        decision=MindControlDecisionV1(route_kind="chat", mode_binding="advisory", allowed_verbs=["speak"]),
        brief=brief,
        mind_quality=quality,  # type: ignore[arg-type]
    )


def test_meaningful_synthesis_key_set_equals_allow_list() -> None:
    coloring = select_mind_coloring(_result(ok=True, quality="meaningful_synthesis"), max_items=3)
    assert coloring is not None
    assert set(coloring.keys()) == MIND_COLORING_ALLOWED_KEYS


def test_task_control_fields_never_cross() -> None:
    coloring = select_mind_coloring(_result(ok=True, quality="meaningful_synthesis"), max_items=3)
    assert coloring is not None
    forbidden = {
        "task_mode", "answer_strategy", "conversation_frame", "response_priorities",
        "response_hazards", "route_kind", "mode_binding", "allowed_verbs", "mode_suggestion",
    }
    assert forbidden.isdisjoint(coloring.keys())
    # And none leak into nested values as dict keys.
    import json
    blob = json.dumps(coloring)
    for token in ("task_mode", "answer_strategy", "response_hazards", "mode_binding"):
        assert token not in blob


def test_themes_and_curiosity_survive_without_shadow_synthesis() -> None:
    coloring = select_mind_coloring(_result(ok=True, quality="meaningful_synthesis"), max_items=3)
    assert coloring is not None
    assert coloring["reflective_themes"] == ["continuity", "trust", "the shape of our work"]
    # curiosity_threads derive from active_frontier.selected[].summary
    assert coloring["curiosity_threads"] == [
        "the unresolved thread about our last session",
        "whether Juniper felt heard last time",
        "what changed since we last spoke",
    ]


def test_attention_frontier_shape_and_truncation() -> None:
    coloring = select_mind_coloring(_result(ok=True, quality="meaningful_synthesis"), max_items=3)
    assert coloring is not None
    af = coloring["attention_frontier"]
    assert len(af) == 3  # 4th item truncated
    assert af[0] == {"label": "continuity", "summary": "the unresolved thread about our last session", "score": 0.91}


def test_provenance_present() -> None:
    result = _result(ok=True, quality="meaningful_synthesis")
    coloring = select_mind_coloring(result, max_items=3)
    assert coloring is not None
    assert coloring["mind_quality"] == "meaningful_synthesis"
    assert coloring["mind_run_id"] == str(result.mind_run_id)
    assert coloring["snapshot_hash"] == "deadbeef"


def test_non_meaningful_returns_none() -> None:
    assert select_mind_coloring(_result(ok=True, quality="shadow_synthesis"), max_items=3) is None
    assert select_mind_coloring(_result(ok=True, quality="fallback_contract_only"), max_items=3) is None


def test_not_ok_returns_none() -> None:
    assert select_mind_coloring(_result(ok=False, quality="meaningful_synthesis"), max_items=3) is None


def test_empty_substance_returns_none() -> None:
    # meaningful_synthesis but no frontier and empty stance_payload -> no substance -> skip.
    empty = MindRunResultV1(
        mind_run_id=uuid4(),
        ok=True,
        brief=MindHandoffBriefV1(mind_quality="meaningful_synthesis", active_frontier=None, stance_payload={}),
        mind_quality="meaningful_synthesis",
    )
    assert select_mind_coloring(empty, max_items=3) is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
PYTHONPATH=services/orion-thought:. python -m pytest services/orion-thought/tests/test_mind_coloring_selector.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'app.mind_enrichment'` (or ImportError for `select_mind_coloring`).

- [ ] **Step 3: Write the minimal implementation**

Create `services/orion-thought/app/mind_enrichment.py` with the selector (other functions added in later tasks):

```python
"""orion-thought → orion-mind advisory enrichment (unified turn coloring).

The unified turn computes stance cold via the ``stance_react`` verb. This module
optionally runs Mind first and selects a strict, mode-agnostic self/attention
subset as an *advisory* prompt prior. ``stance_react`` remains the sole author of
ThoughtEventV1 and reconciles this coloring. Everything fails open.
"""
from __future__ import annotations

import logging
from typing import Any

from orion.mind.v1 import MindRunResultV1

logger = logging.getLogger("orion-thought.mind_enrichment")

# Strict allow-list of coloring keys. Any un-listed ChatStanceBrief / decision
# field is absent by construction (no deny-list, no leakage of future fields).
MIND_COLORING_ALLOWED_KEYS: frozenset[str] = frozenset(
    {
        "attention_frontier",
        "reflective_themes",
        "curiosity_threads",
        "self_relevance",
        "identity_salience",
        "juniper_relevance",
        "mind_quality",
        "mind_run_id",
        "snapshot_hash",
    }
)

_MAX_STR_CHARS = 240


def _clip(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()[:_MAX_STR_CHARS]
    return value


def _str_list(value: Any, *, max_items: int) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    out: list[str] = []
    for item in value:
        text = str(item).strip()[:_MAX_STR_CHARS]
        if text:
            out.append(text)
        if len(out) >= max_items:
            break
    return out


def select_mind_coloring(result: MindRunResultV1, *, max_items: int = 3) -> dict[str, Any] | None:
    """Project the mode-agnostic self/attention subset of a Mind run.

    Returns None (skip enrichment) unless the run is ok AND produced
    meaningful_synthesis AND carries at least one substantive signal. Never
    injects an empty shell. Selection is a strict allow-list.
    """
    if not result.ok:
        return None
    brief = result.brief
    if brief.mind_quality != "meaningful_synthesis":
        return None

    frontier = brief.active_frontier
    selected = list(frontier.selected) if frontier is not None else []
    selected = selected[:max_items]
    attention_frontier = [
        {
            "label": _clip(m.label),
            "summary": _clip(m.summary),
            "score": round(float(m.score), 4),
        }
        for m in selected
    ]
    curiosity_threads = [_clip(m.summary) for m in selected if str(m.summary).strip()][:max_items]

    stance_payload = brief.stance_payload if isinstance(brief.stance_payload, dict) else {}
    reflective_themes = _str_list(stance_payload.get("reflective_themes"), max_items=max_items)
    self_relevance = _clip(stance_payload.get("self_relevance")) if stance_payload.get("self_relevance") else None
    identity_salience = stance_payload.get("identity_salience") or None
    juniper_relevance = _clip(stance_payload.get("juniper_relevance")) if stance_payload.get("juniper_relevance") else None

    # No empty-shell cognition: require at least one substantive signal.
    has_substance = bool(
        attention_frontier or reflective_themes or curiosity_threads
        or self_relevance or juniper_relevance
    )
    if not has_substance:
        return None

    return {
        "attention_frontier": attention_frontier,
        "reflective_themes": reflective_themes,
        "curiosity_threads": curiosity_threads,
        "self_relevance": self_relevance,
        "identity_salience": identity_salience,
        "juniper_relevance": juniper_relevance,
        "mind_quality": brief.mind_quality,
        "mind_run_id": str(result.mind_run_id),
        "snapshot_hash": result.snapshot_hash,
    }
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
PYTHONPATH=services/orion-thought:. python -m pytest services/orion-thought/tests/test_mind_coloring_selector.py -q
```

Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
git add services/orion-thought/app/mind_enrichment.py services/orion-thought/tests/test_mind_coloring_selector.py
git commit -m "feat(orion-thought): add allow-list Mind coloring selector"
```

---

## Task 2: Light Mind snapshot builder (pure)

**Files:**
- Modify: `services/orion-thought/app/mind_enrichment.py`
- Test: `services/orion-thought/tests/test_mind_light_snapshot.py`

Builds a `MindRunRequestV1` from a `StanceReactRequestV1` with NO cognitive-projection cold rebuild. Snapshot = user_text (current_turn) + optional `recall_bundle` facet (only if already on `stance_inputs`) + optional `situation_compact` facet from the broadcast (real open-loop text, bounded).

- [ ] **Step 1: Write the failing test**

Create `services/orion-thought/tests/test_mind_light_snapshot.py`:

```python
from __future__ import annotations

from app.mind_enrichment import build_light_mind_request
from orion.schemas.attention_frame import (
    AttentionBroadcastProjectionV1,
    AttentionFrameV1,
    OpenLoopV1,
)
from orion.schemas.thought import HubAssociationBundleV1, StanceReactRequestV1


def _request(*, broadcast=None, stance_inputs=None) -> StanceReactRequestV1:
    return StanceReactRequestV1(
        correlation_id="corr-1",
        session_id="sess-1",
        user_message="I've been thinking about where our work is heading.",
        association=HubAssociationBundleV1(
            correlation_id="corr-1",
            broadcast=broadcast,
            broadcast_stale=False,
            read_source="felt_state_reader",
        ),
        repair_bundle=None,
        stance_inputs=stance_inputs or {"user_message": "..."},
    )


def _broadcast() -> AttentionBroadcastProjectionV1:
    frame = AttentionFrameV1(
        correlation_id="corr-1",
        open_loops=[
            OpenLoopV1(id="ol-1", description="the unresolved deploy decision", why_it_matters="blocks progress"),
            OpenLoopV1(id="ol-2", description="whether the refactor is worth it"),
        ],
    )
    return AttentionBroadcastProjectionV1(
        frame=frame,
        selected_description="the deploy decision is the live thread",
        selected_open_loop_id="ol-1",
        attended_node_ids=["n-1", "n-2"],
    )


def test_snapshot_carries_user_text_and_policy() -> None:
    req = build_light_mind_request(_request(), wall_time_ms=12000, router_profile="default")
    assert req.correlation_id == "corr-1"
    assert req.session_id == "sess-1"
    assert req.trigger == "user_turn"
    assert req.snapshot_inputs["user_text"].startswith("I've been thinking")
    assert req.snapshot_inputs["messages_tail"] == []
    assert req.policy.n_loops_max == 1
    assert req.policy.wall_time_ms_max == 12000
    assert req.policy.router_profile_id == "default"


def test_no_projection_facet_ever() -> None:
    req = build_light_mind_request(_request(broadcast=_broadcast()), wall_time_ms=12000, router_profile="default")
    facets = req.snapshot_inputs.get("facets", {})
    assert "cognitive_projection" not in facets
    assert "cognitive_projection_degraded" not in facets


def test_recall_bundle_folded_only_when_present() -> None:
    without = build_light_mind_request(_request(), wall_time_ms=12000, router_profile="default")
    assert "recall_bundle" not in without.snapshot_inputs.get("facets", {})

    recall = {"fragments": [{"snippet": "we discussed continuity"}], "citations": []}
    with_recall = build_light_mind_request(
        _request(stance_inputs={"user_message": "...", "recall_bundle": recall}),
        wall_time_ms=12000,
        router_profile="default",
    )
    assert with_recall.snapshot_inputs["facets"]["recall_bundle"] == recall


def test_situation_compact_from_broadcast_open_loops() -> None:
    req = build_light_mind_request(_request(broadcast=_broadcast()), wall_time_ms=12000, router_profile="default")
    situation = req.snapshot_inputs["facets"]["situation_compact"]
    text = str(situation)
    assert "deploy decision" in text
    assert "refactor is worth it" in text


def test_no_situation_facet_without_broadcast() -> None:
    req = build_light_mind_request(_request(broadcast=None), wall_time_ms=12000, router_profile="default")
    assert "situation_compact" not in req.snapshot_inputs.get("facets", {})
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
PYTHONPATH=services/orion-thought:. python -m pytest services/orion-thought/tests/test_mind_light_snapshot.py -q
```

Expected: FAIL with `ImportError: cannot import name 'build_light_mind_request'`.

- [ ] **Step 3: Write the minimal implementation**

Add to `services/orion-thought/app/mind_enrichment.py` (imports at top, function below the selector):

```python
from orion.mind.v1 import MindRunPolicyV1, MindRunRequestV1
from orion.schemas.thought import StanceReactRequestV1

_MAX_USER_TEXT_CHARS = 20_000
_MAX_OPEN_LOOPS = 6
```

```python
def _situation_compact_from_broadcast(request: StanceReactRequestV1) -> dict[str, Any] | None:
    """Fold real open-loop / selected-description text into the accepted
    situation_compact facet. Returns None when there is no usable text.
    """
    broadcast = request.association.broadcast
    if broadcast is None:
        return None
    loops: list[dict[str, str]] = []
    for loop in (broadcast.frame.open_loops or [])[:_MAX_OPEN_LOOPS]:
        description = (loop.description or "").strip()
        if not description:
            continue
        entry: dict[str, str] = {"description": description[:_MAX_STR_CHARS]}
        why = (loop.why_it_matters or "").strip()
        if why:
            entry["why_it_matters"] = why[:_MAX_STR_CHARS]
        loops.append(entry)
    selected = (broadcast.selected_description or "").strip()
    if not loops and not selected:
        return None
    compact: dict[str, Any] = {"attention_situation": True}
    if selected:
        compact["selected_focus"] = selected[:_MAX_STR_CHARS]
    if loops:
        compact["open_loops"] = loops
    return compact


def build_light_mind_request(
    request: StanceReactRequestV1,
    *,
    wall_time_ms: int,
    router_profile: str,
) -> MindRunRequestV1:
    """Build a bounded Mind request with NO cognitive-projection cold rebuild.

    Evidence in v1 is the current user turn (as a single current_turn item),
    plus recall_bundle (only if already threaded on stance_inputs) and a
    situation_compact facet derived from the attention broadcast.
    """
    user_text = (request.user_message or "").strip()[:_MAX_USER_TEXT_CHARS]
    snapshot: dict[str, Any] = {"user_text": user_text, "messages_tail": []}

    facets: dict[str, Any] = {}
    stance_inputs = request.stance_inputs if isinstance(request.stance_inputs, dict) else {}
    recall_bundle = stance_inputs.get("recall_bundle")
    if isinstance(recall_bundle, dict) and recall_bundle:
        facets["recall_bundle"] = recall_bundle
    situation = _situation_compact_from_broadcast(request)
    if situation:
        facets["situation_compact"] = situation
    if facets:
        snapshot["facets"] = facets

    return MindRunRequestV1(
        correlation_id=request.correlation_id,
        session_id=request.session_id,
        trigger="user_turn",
        snapshot_inputs=snapshot,
        policy=MindRunPolicyV1(
            n_loops_max=1,
            wall_time_ms_max=max(1, int(wall_time_ms)),
            router_profile_id=router_profile or "default",
        ),
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
PYTHONPATH=services/orion-thought:. python -m pytest services/orion-thought/tests/test_mind_light_snapshot.py -q
```

Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add services/orion-thought/app/mind_enrichment.py services/orion-thought/tests/test_mind_light_snapshot.py
git commit -m "feat(orion-thought): add light Mind snapshot builder (no projection rebuild)"
```

---

## Task 3: Settings, env, deps, docker

**Files:**
- Modify: `services/orion-thought/app/settings.py`
- Modify: `services/orion-thought/.env_example`
- Modify: `services/orion-thought/docker-compose.yml`
- Modify: `services/orion-thought/requirements.txt`
- Test: `services/orion-thought/tests/test_settings_mind_enrichment.py`

- [ ] **Step 1: Write the failing test**

Create `services/orion-thought/tests/test_settings_mind_enrichment.py`:

```python
import importlib


def test_mind_enrichment_defaults_off(monkeypatch):
    for key in (
        "ORION_THOUGHT_MIND_ENRICHMENT_ENABLED",
        "ORION_THOUGHT_MIND_ARTIFACT_PUBLISH_ENABLED",
        "ORION_THOUGHT_MIND_TIMEOUT_SEC",
        "ORION_THOUGHT_MIND_WALL_MS",
        "ORION_THOUGHT_MIND_ROUTER_PROFILE",
        "ORION_THOUGHT_MIND_MAX_RESPONSE_BYTES",
        "ORION_THOUGHT_MIND_COLORING_MAX_ITEMS",
        "ORION_MIND_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)
    import app.settings as s
    importlib.reload(s)
    assert s.settings.mind_enrichment_enabled is False
    assert s.settings.mind_artifact_publish_enabled is False
    assert s.settings.mind_timeout_sec == 15.0
    assert s.settings.mind_wall_ms == 12000
    assert s.settings.mind_router_profile == "default"
    assert s.settings.mind_max_response_bytes == 2_000_000
    assert s.settings.mind_coloring_max_items == 3
    assert s.settings.mind_base_url == "http://orion-mind:6611"
    assert s.settings.channel_mind_artifact == "orion:mind:artifact"


def test_mind_enrichment_reads_env(monkeypatch):
    monkeypatch.setenv("ORION_THOUGHT_MIND_ENRICHMENT_ENABLED", "true")
    monkeypatch.setenv("ORION_THOUGHT_MIND_WALL_MS", "9000")
    import app.settings as s
    importlib.reload(s)
    assert s.settings.mind_enrichment_enabled is True
    assert s.settings.mind_wall_ms == 9000
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
PYTHONPATH=services/orion-thought:. python -m pytest services/orion-thought/tests/test_settings_mind_enrichment.py -q
```

Expected: FAIL with `AttributeError: 'ThoughtSettings' object has no attribute 'mind_enrichment_enabled'`.

- [ ] **Step 3: Add settings fields**

In `services/orion-thought/app/settings.py`, insert after the salience block (after line 111, before `settings = ThoughtSettings()`):

```python
    # --- Mind stance enrichment (unified turn; default-off) ---
    # Runs orion-mind before stance_react and injects an advisory self/attention
    # coloring. Silent no-op unless orion-mind has MIND_LLM_SYNTHESIS_ENABLED=true
    # (a separate service — not visible to this service's env-parity check).
    mind_enrichment_enabled: bool = Field(False, alias="ORION_THOUGHT_MIND_ENRICHMENT_ENABLED")
    mind_base_url: str = Field("http://orion-mind:6611", alias="ORION_MIND_BASE_URL")
    mind_timeout_sec: float = Field(15.0, alias="ORION_THOUGHT_MIND_TIMEOUT_SEC")
    mind_wall_ms: int = Field(12_000, alias="ORION_THOUGHT_MIND_WALL_MS")
    mind_router_profile: str = Field("default", alias="ORION_THOUGHT_MIND_ROUTER_PROFILE")
    mind_max_response_bytes: int = Field(2_000_000, alias="ORION_THOUGHT_MIND_MAX_RESPONSE_BYTES")
    mind_artifact_publish_enabled: bool = Field(
        False, alias="ORION_THOUGHT_MIND_ARTIFACT_PUBLISH_ENABLED"
    )
    mind_coloring_max_items: int = Field(3, alias="ORION_THOUGHT_MIND_COLORING_MAX_ITEMS")
    channel_mind_artifact: str = Field("orion:mind:artifact", alias="CHANNEL_MIND_ARTIFACT")
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
PYTHONPATH=services/orion-thought:. python -m pytest services/orion-thought/tests/test_settings_mind_enrichment.py -q
```

Expected: PASS (2 passed).

- [ ] **Step 5: Update `.env_example`**

Append to `services/orion-thought/.env_example`:

```bash

# --- Mind stance enrichment (unified turn; default-off) ---
# Run orion-mind before stance_react and inject an advisory self/attention
# coloring into the verb context. Fail-open, flag-gated.
# NOTE: This is a silent no-op unless orion-mind has MIND_LLM_SYNTHESIS_ENABLED=true
# (that key lives on the orion-mind service, not here) AND orion-thought can reach
# ORION_MIND_BASE_URL. Verify both operationally at rollout.
ORION_THOUGHT_MIND_ENRICHMENT_ENABLED=false
ORION_MIND_BASE_URL=http://orion-mind:6611
ORION_THOUGHT_MIND_TIMEOUT_SEC=15
ORION_THOUGHT_MIND_WALL_MS=12000
ORION_THOUGHT_MIND_ROUTER_PROFILE=default
ORION_THOUGHT_MIND_MAX_RESPONSE_BYTES=2000000
ORION_THOUGHT_MIND_ARTIFACT_PUBLISH_ENABLED=false
ORION_THOUGHT_MIND_COLORING_MAX_ITEMS=3
CHANNEL_MIND_ARTIFACT=orion:mind:artifact
```

- [ ] **Step 6: Update `docker-compose.yml`**

In `services/orion-thought/docker-compose.yml`, add to the `environment:` list (after the `STANCE_REACT_TIMEOUT_SEC` line, ~line 23):

```yaml
      - ORION_THOUGHT_MIND_ENRICHMENT_ENABLED=${ORION_THOUGHT_MIND_ENRICHMENT_ENABLED:-false}
      - ORION_MIND_BASE_URL=${ORION_MIND_BASE_URL:-http://orion-mind:6611}
      - ORION_THOUGHT_MIND_TIMEOUT_SEC=${ORION_THOUGHT_MIND_TIMEOUT_SEC:-15}
      - ORION_THOUGHT_MIND_WALL_MS=${ORION_THOUGHT_MIND_WALL_MS:-12000}
      - ORION_THOUGHT_MIND_ROUTER_PROFILE=${ORION_THOUGHT_MIND_ROUTER_PROFILE:-default}
      - ORION_THOUGHT_MIND_MAX_RESPONSE_BYTES=${ORION_THOUGHT_MIND_MAX_RESPONSE_BYTES:-2000000}
      - ORION_THOUGHT_MIND_ARTIFACT_PUBLISH_ENABLED=${ORION_THOUGHT_MIND_ARTIFACT_PUBLISH_ENABLED:-false}
      - ORION_THOUGHT_MIND_COLORING_MAX_ITEMS=${ORION_THOUGHT_MIND_COLORING_MAX_ITEMS:-3}
      - CHANNEL_MIND_ARTIFACT=${CHANNEL_MIND_ARTIFACT:-orion:mind:artifact}
```

- [ ] **Step 7: Add the `httpx` dependency**

In `services/orion-thought/requirements.txt`, add after the `redis>=5.0.0` line:

```text
httpx>=0.27
```

- [ ] **Step 8: Sync local `.env` from the example (MANDATORY — AGENTS.md)**

Run from repo root:

```bash
python scripts/sync_local_env_from_example.py
```

Expected: reports the 9 new `ORION_THOUGHT_MIND_*` / `CHANNEL_MIND_ARTIFACT` / `ORION_MIND_BASE_URL` keys added to `services/orion-thought/.env`. If any key is skipped, report it explicitly in the PR (none expected here — skip-list is `ORION_KNOWLEDGE_ROOT`, `PUBLISH_CORTEX_EXEC_GRAMMAR`).

- [ ] **Step 9: Run env parity + confirm `.env` not staged**

Run:

```bash
python scripts/check_env_template_parity.py
git check-ignore services/orion-thought/.env
git status --short
```

Expected: parity check passes; `git check-ignore` prints `services/orion-thought/.env`; `git status` does NOT list `.env` as tracked.

- [ ] **Step 10: Commit**

```bash
git add services/orion-thought/app/settings.py services/orion-thought/.env_example \
  services/orion-thought/docker-compose.yml services/orion-thought/requirements.txt \
  services/orion-thought/tests/test_settings_mind_enrichment.py
git commit -m "feat(orion-thought): add Mind enrichment settings, env, deps, compose wiring"
```

---

## Task 4: Mind HTTP client (fail-open)

**Files:**
- Modify: `services/orion-thought/app/mind_enrichment.py`
- Test: `services/orion-thought/tests/test_mind_http_client.py`

Async POST to `{base}/v1/mind/run`, tight timeout, bounded body, returns `MindRunResultV1 | None`. ANY error → log `mind_enrichment_failed` and return `None`.

- [ ] **Step 1: Write the failing test**

Create `services/orion-thought/tests/test_mind_http_client.py`:

```python
from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest

from app import mind_enrichment
from orion.mind.v1 import MindHandoffBriefV1, MindRunRequestV1, MindRunResultV1
from orion.mind.v1 import MindRunPolicyV1


def _settings(**over):
    base = dict(
        mind_base_url="http://orion-mind:6611",
        mind_timeout_sec=5.0,
        mind_max_response_bytes=2_000_000,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _req() -> MindRunRequestV1:
    return MindRunRequestV1(
        correlation_id="corr-1",
        snapshot_inputs={"user_text": "hi", "messages_tail": []},
        policy=MindRunPolicyV1(n_loops_max=1, wall_time_ms_max=12000),
    )


def _ok_result_json() -> dict:
    return MindRunResultV1(
        mind_run_id=uuid4(),
        ok=True,
        brief=MindHandoffBriefV1(mind_quality="meaningful_synthesis"),
        mind_quality="meaningful_synthesis",
    ).model_dump(mode="json")


@pytest.mark.asyncio
async def test_ok_returns_result(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/mind/run"
        return httpx.Response(200, json=_ok_result_json())

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(mind_enrichment, "_mind_transport", lambda: transport, raising=False)
    result = await mind_enrichment.run_mind_for_thought(_req(), settings=_settings(), correlation_id="corr-1")
    assert result is not None
    assert result.ok is True
    assert result.brief.mind_quality == "meaningful_synthesis"


@pytest.mark.asyncio
async def test_timeout_fails_open(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("boom", request=request)

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(mind_enrichment, "_mind_transport", lambda: transport, raising=False)
    result = await mind_enrichment.run_mind_for_thought(_req(), settings=_settings(), correlation_id="corr-1")
    assert result is None


@pytest.mark.asyncio
async def test_http_500_fails_open(monkeypatch):
    transport = httpx.MockTransport(lambda req: httpx.Response(500, text="nope"))
    monkeypatch.setattr(mind_enrichment, "_mind_transport", lambda: transport, raising=False)
    result = await mind_enrichment.run_mind_for_thought(_req(), settings=_settings(), correlation_id="corr-1")
    assert result is None


@pytest.mark.asyncio
async def test_oversized_body_fails_open(monkeypatch):
    big = _ok_result_json()
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json=big))
    monkeypatch.setattr(mind_enrichment, "_mind_transport", lambda: transport, raising=False)
    result = await mind_enrichment.run_mind_for_thought(
        _req(), settings=_settings(mind_max_response_bytes=1), correlation_id="corr-1"
    )
    assert result is None


@pytest.mark.asyncio
async def test_empty_base_url_fails_open(monkeypatch):
    result = await mind_enrichment.run_mind_for_thought(
        _req(), settings=_settings(mind_base_url=""), correlation_id="corr-1"
    )
    assert result is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
PYTHONPATH=services/orion-thought:. python -m pytest services/orion-thought/tests/test_mind_http_client.py -q
```

Expected: FAIL with `AttributeError: module 'app.mind_enrichment' has no attribute 'run_mind_for_thought'`.

- [ ] **Step 3: Write the minimal implementation**

Add to `services/orion-thought/app/mind_enrichment.py` (add `import httpx` and `from typing import Any` if not present):

```python
import httpx


def _mind_transport() -> httpx.BaseTransport | None:
    """Seam for tests to inject an httpx.MockTransport. Returns None in prod
    so AsyncClient uses its default transport.
    """
    return None


async def run_mind_for_thought(
    req: "MindRunRequestV1",
    *,
    settings: Any,
    correlation_id: str,
) -> "MindRunResultV1 | None":
    """POST the Mind request; return the parsed result or None (fail-open)."""
    base = (getattr(settings, "mind_base_url", "") or "").rstrip("/")
    if not base:
        logger.warning("mind_enrichment_failed corr=%s reason=unconfigured_base_url", correlation_id)
        return None
    url = f"{base}/v1/mind/run"
    timeout_sec = float(getattr(settings, "mind_timeout_sec", 15.0))
    timeout = httpx.Timeout(
        connect=min(10.0, timeout_sec),
        read=timeout_sec,
        write=min(30.0, timeout_sec),
        pool=5.0,
    )
    max_body = int(getattr(settings, "mind_max_response_bytes", 2_000_000))
    transport = _mind_transport()
    client_kwargs: dict[str, Any] = {"timeout": timeout}
    if transport is not None:
        client_kwargs["transport"] = transport
    try:
        async with httpx.AsyncClient(**client_kwargs) as client:
            resp = await client.post(url, json=req.model_dump(mode="json"))
            resp.raise_for_status()
            raw = resp.content
            if len(raw) > max_body:
                raise RuntimeError(f"mind_response_too_large:{len(raw)}")
            return MindRunResultV1.model_validate(resp.json())
    except Exception as exc:  # noqa: BLE001 — fail-open by contract
        logger.warning(
            "mind_enrichment_failed corr=%s reason=%s err=%s",
            correlation_id,
            type(exc).__name__,
            exc,
        )
        return None
```

Note: also add `MindRunRequestV1` to the existing `from orion.mind.v1 import ...` line (it is already imported for Task 2's builder — confirm the import list reads `from orion.mind.v1 import MindRunPolicyV1, MindRunRequestV1, MindRunResultV1`).

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
PYTHONPATH=services/orion-thought:. python -m pytest services/orion-thought/tests/test_mind_http_client.py -q
```

Expected: PASS (5 passed). If `pytest.mark.asyncio` errors with "async def functions are not natively supported", confirm `anyio`/`pytest-asyncio` is available (the grounding-capsule test already uses `@pytest.mark.asyncio`, so the plugin is present).

- [ ] **Step 5: Commit**

```bash
git add services/orion-thought/app/mind_enrichment.py services/orion-thought/tests/test_mind_http_client.py
git commit -m "feat(orion-thought): add fail-open Mind HTTP client"
```

---

## Task 5: Prompt block + context wiring

**Files:**
- Modify: `orion/cognition/prompts/stance_react.j2`
- Modify: `services/orion-thought/app/bus_listener.py`
- Test: `services/orion-thought/tests/test_stance_context_mind_coloring.py`
- Test: `services/orion-thought/tests/test_stance_prompt_renders_coloring.py`

### 5a: `build_stance_react_context` accepts optional `mind_coloring`

- [ ] **Step 1: Write the failing context test**

Create `services/orion-thought/tests/test_stance_context_mind_coloring.py`:

```python
from __future__ import annotations

from app.bus_listener import build_stance_react_context
from orion.schemas.thought import HubAssociationBundleV1, StanceReactRequestV1


def _request() -> StanceReactRequestV1:
    return StanceReactRequestV1(
        correlation_id="corr-1",
        session_id="sess-1",
        user_message="where is our work heading?",
        association=HubAssociationBundleV1(
            correlation_id="corr-1",
            broadcast=None,
            broadcast_stale=True,
            read_source="hub_sql_fallback",
        ),
        repair_bundle=None,
        stance_inputs={"user_message": "where is our work heading?"},
    )


def test_context_without_coloring_has_no_key() -> None:
    ctx = build_stance_react_context(_request())
    assert "mind_coloring" not in ctx


def test_context_without_coloring_is_baseline() -> None:
    # Passing mind_coloring=None must be byte-identical to omitting it.
    assert build_stance_react_context(_request()) == build_stance_react_context(_request(), mind_coloring=None)


def test_context_with_coloring_adds_key() -> None:
    coloring = {"attention_frontier": [{"label": "x", "summary": "y", "score": 0.5}], "reflective_themes": ["z"]}
    ctx = build_stance_react_context(_request(), mind_coloring=coloring)
    assert ctx["mind_coloring"] == coloring
```

- [ ] **Step 2: Run to verify it fails**

Run:

```bash
PYTHONPATH=services/orion-thought:. python -m pytest services/orion-thought/tests/test_stance_context_mind_coloring.py -q
```

Expected: FAIL — `build_stance_react_context() got an unexpected keyword argument 'mind_coloring'`.

- [ ] **Step 3: Modify `build_stance_react_context` and `build_stance_react_plan_request`**

In `services/orion-thought/app/bus_listener.py`, replace the `build_stance_react_context` function (lines 56-69) with:

```python
def build_stance_react_context(
    request: StanceReactRequestV1,
    *,
    mind_coloring: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context: dict[str, Any] = {
        "user_message": request.user_message,
        "stance_inputs": {"user_message": request.user_message},
        "association": slim_association_for_prompt(request.association),
        "repair_bundle": slim_repair_bundle_for_prompt(request.repair_bundle),
        "coalition_projection": _coalition_projection(request),
        "metadata": {
            "correlation_id": request.correlation_id,
            "session_id": request.session_id,
            "llm_profile": request.llm_profile,
            "mode": "brain",
        },
    }
    if mind_coloring is not None:
        context["mind_coloring"] = mind_coloring
    return context
```

Then update `build_stance_react_plan_request` (line 72) to accept and forward the coloring:

```python
def build_stance_react_plan_request(
    request: StanceReactRequestV1,
    *,
    mind_coloring: dict[str, Any] | None = None,
) -> PlanExecutionRequest:
    plan = build_plan_for_verb("stance_react", mode="brain")
    return PlanExecutionRequest(
        plan=plan,
        args=PlanExecutionArgs(
            request_id=request.correlation_id,
            trigger_source=settings.service_name,
            extra={
                "llm_profile": request.llm_profile,
                "mode": "brain",
            },
        ),
        context=build_stance_react_context(request, mind_coloring=mind_coloring),
    )
```

- [ ] **Step 4: Run to verify it passes**

Run:

```bash
PYTHONPATH=services/orion-thought:. python -m pytest services/orion-thought/tests/test_stance_context_mind_coloring.py -q
```

Expected: PASS (3 passed).

### 5b: Prompt renders the advisory block

- [ ] **Step 5: Write the failing prompt-render test**

Create `services/orion-thought/tests/test_stance_prompt_renders_coloring.py`:

```python
from __future__ import annotations

from pathlib import Path

from jinja2 import Environment

_PROMPT = Path(__file__).resolve().parents[3] / "orion" / "cognition" / "prompts" / "stance_react.j2"


def _render(**ctx) -> str:
    template = Environment().from_string(_PROMPT.read_text())
    base = {
        "user_message": "hi",
        "stance_inputs": {"user_message": "hi"},
        "association": {},
        "repair_bundle": None,
        "coalition_projection": None,
    }
    base.update(ctx)
    return template.render(**base)


def test_block_absent_without_coloring() -> None:
    out = _render()
    assert "PRIOR SELF-SIGNAL" not in out


def test_block_present_with_coloring() -> None:
    coloring = {
        "attention_frontier": [{"label": "continuity", "summary": "our last thread", "score": 0.9}],
        "reflective_themes": ["continuity"],
        "curiosity_threads": ["what changed"],
        "self_relevance": "touches my continuity",
        "identity_salience": "high",
    }
    out = _render(mind_coloring=coloring)
    assert "PRIOR SELF-SIGNAL" in out
    assert "reconcile, do not obey" in out
    assert "those WIN" in out
    assert "continuity" in out


def test_block_does_not_introduce_output_keys() -> None:
    # The advisory block must not tell the model to emit new top-level JSON keys.
    out = _render(mind_coloring={"attention_frontier": [], "reflective_themes": ["x"]})
    assert "do not invent extra top-level keys" in out or "do not invent" in out.lower()
```

- [ ] **Step 6: Run to verify it fails**

Run:

```bash
PYTHONPATH=services/orion-thought:. python -m pytest services/orion-thought/tests/test_stance_prompt_renders_coloring.py -q
```

Expected: FAIL — `test_block_present_with_coloring` fails (no "PRIOR SELF-SIGNAL"); `test_block_does_not_introduce_output_keys` may pass already (existing prompt has "do not invent extra top-level keys" at line 27).

- [ ] **Step 7: Add the advisory block to `stance_react.j2`**

In `orion/cognition/prompts/stance_react.j2`, insert AFTER the `coalition_projection` block (after line 21, before the `POSTURE ASSESSMENT` header on line 23):

```jinja
{% if mind_coloring %}
PRIOR SELF-SIGNAL (advisory — from Oríon's Mind; reconcile, do not obey)
- This is Oríon's own background self/attention read, computed before this turn's routing.
- Use it to color imperative/tone and the felt layer so Oríon sounds like an ongoing presence.
- It is NOT task instruction. If it conflicts with user_message, association, grounding,
  or the actual task (technical/agent/coding), those WIN. Never let it force chat framing.
- It does not add output keys — still emit only valid ThoughtEventV1; do not invent extra top-level keys.
- attention_frontier: {{ mind_coloring.attention_frontier }}
- reflective_themes: {{ mind_coloring.reflective_themes }}
- curiosity_threads: {{ mind_coloring.curiosity_threads }}
- self_relevance: {{ mind_coloring.self_relevance }}
- identity_salience: {{ mind_coloring.identity_salience }}
{% endif %}
```

- [ ] **Step 8: Run to verify it passes**

Run:

```bash
PYTHONPATH=services/orion-thought:. python -m pytest services/orion-thought/tests/test_stance_prompt_renders_coloring.py -q
```

Expected: PASS (3 passed).

- [ ] **Step 9: Commit**

```bash
git add orion/cognition/prompts/stance_react.j2 services/orion-thought/app/bus_listener.py \
  services/orion-thought/tests/test_stance_context_mind_coloring.py \
  services/orion-thought/tests/test_stance_prompt_renders_coloring.py
git commit -m "feat: wire advisory mind_coloring into stance_react context + prompt"
```

---

## Task 6: `run_stance_react` integration + fail-open

**Files:**
- Modify: `services/orion-thought/app/bus_listener.py`
- Test: `services/orion-thought/tests/test_mind_enrichment_fail_open.py`

Wire enrichment into `run_stance_react` behind the flag. When the flag is off, or Mind returns `None`, or the selector returns `None`, the plan request must be byte-identical to today.

- [ ] **Step 1: Write the failing integration test**

Create `services/orion-thought/tests/test_mind_enrichment_fail_open.py`:

```python
from __future__ import annotations

import importlib

import pytest

from orion.schemas.thought import HubAssociationBundleV1, StanceReactRequestV1


class _FakeCortexClient:
    def __init__(self, exec_result: dict) -> None:
        self._exec_result = exec_result
        self.captured_context = None

    async def execute_plan(self, *, req, **_kwargs) -> dict:
        self.captured_context = req.context
        return self._exec_result


def _request() -> StanceReactRequestV1:
    return StanceReactRequestV1(
        correlation_id="corr-1",
        session_id="sess-1",
        user_message="how are you today?",
        association=HubAssociationBundleV1(
            correlation_id="corr-1",
            broadcast=None,
            broadcast_stale=True,
            read_source="hub_sql_fallback",
        ),
        repair_bundle=None,
        stance_inputs={"user_message": "how are you today?"},
    )


def _stance_json() -> str:
    return (
        '{"imperative":"Stay present with Juniper.","tone":"warm",'
        '"strain_refs":["hub:turn:corr-1"],"evidence_refs":["hub:turn:corr-1"],'
        '"stance_harness_slice":{"task_mode":"reflective_dialogue",'
        '"conversation_frame":"reflective","answer_strategy":"companion"}}'
    )


@pytest.mark.asyncio
async def test_enrichment_disabled_is_baseline(monkeypatch):
    monkeypatch.setenv("ORION_THOUGHT_MIND_ENRICHMENT_ENABLED", "false")
    import app.settings as s
    importlib.reload(s)
    import app.bus_listener as bl
    importlib.reload(bl)

    client = _FakeCortexClient({"final_text": _stance_json(), "metadata": {}})
    thought = await bl.run_stance_react(_request(), bus=None, cortex_client=client)
    assert thought.imperative == "Stay present with Juniper."
    assert "mind_coloring" not in client.captured_context


@pytest.mark.asyncio
async def test_enrichment_enabled_but_mind_fails_open(monkeypatch):
    monkeypatch.setenv("ORION_THOUGHT_MIND_ENRICHMENT_ENABLED", "true")
    import app.settings as s
    importlib.reload(s)
    import app.mind_enrichment as me
    importlib.reload(me)
    import app.bus_listener as bl
    importlib.reload(bl)

    async def _boom(*_a, **_k):
        return None  # simulate Mind timeout/error → fail-open

    monkeypatch.setattr(bl, "run_mind_for_thought", _boom)

    client = _FakeCortexClient({"final_text": _stance_json(), "metadata": {}})
    thought = await bl.run_stance_react(_request(), bus=None, cortex_client=client)
    assert thought.imperative == "Stay present with Juniper."
    assert "mind_coloring" not in client.captured_context


@pytest.mark.asyncio
async def test_enrichment_enabled_meaningful_injects_coloring(monkeypatch):
    monkeypatch.setenv("ORION_THOUGHT_MIND_ENRICHMENT_ENABLED", "true")
    import app.settings as s
    importlib.reload(s)
    import app.mind_enrichment as me
    importlib.reload(me)
    import app.bus_listener as bl
    importlib.reload(bl)

    async def _mind(*_a, **_k):
        return object()  # sentinel; selector is monkeypatched below

    monkeypatch.setattr(bl, "run_mind_for_thought", _mind)
    monkeypatch.setattr(
        bl,
        "select_mind_coloring",
        lambda *_a, **_k: {"attention_frontier": [], "reflective_themes": ["continuity"]},
    )

    client = _FakeCortexClient({"final_text": _stance_json(), "metadata": {}})
    thought = await bl.run_stance_react(_request(), bus=None, cortex_client=client)
    assert client.captured_context["mind_coloring"] == {
        "attention_frontier": [],
        "reflective_themes": ["continuity"],
    }
    assert thought.imperative == "Stay present with Juniper."
```

- [ ] **Step 2: Run to verify it fails**

Run:

```bash
PYTHONPATH=services/orion-thought:. python -m pytest services/orion-thought/tests/test_mind_enrichment_fail_open.py -q
```

Expected: FAIL — `test_enrichment_enabled_meaningful_injects_coloring` fails (`mind_coloring` not in context; `run_stance_react` ignores enrichment).

- [ ] **Step 3: Wire enrichment into `run_stance_react`**

In `services/orion-thought/app/bus_listener.py`:

Add imports near the top (after the `.settings import settings` line):

```python
from .mind_enrichment import build_light_mind_request, run_mind_for_thought, select_mind_coloring
```

Then replace `run_stance_react` (lines 125-149) with:

```python
async def _maybe_build_mind_coloring(
    request: StanceReactRequestV1,
    *,
    bus: OrionBusAsync | None,
) -> dict[str, Any] | None:
    """Run Mind and select advisory coloring. Fail-open: any None short-circuits."""
    if not settings.mind_enrichment_enabled:
        return None
    mind_req = build_light_mind_request(
        request,
        wall_time_ms=settings.mind_wall_ms,
        router_profile=settings.mind_router_profile,
    )
    result = await run_mind_for_thought(
        mind_req,
        settings=settings,
        correlation_id=request.correlation_id,
    )
    if result is None:
        return None
    coloring = select_mind_coloring(result, max_items=settings.mind_coloring_max_items)
    if settings.mind_artifact_publish_enabled and bus is not None:
        await publish_mind_run_artifact_for_thought(
            bus,
            source=_source(),
            request=request,
            mind_req=mind_req,
            mind_res=result,
            channel=settings.channel_mind_artifact,
        )
    logger.info(
        "mind_enrichment corr=%s mind_run_id=%s quality=%s coloring=%s",
        request.correlation_id,
        result.mind_run_id,
        result.brief.mind_quality,
        "fired" if coloring else "skipped",
    )
    return coloring


async def run_stance_react(
    request: StanceReactRequestV1,
    *,
    bus: OrionBusAsync,
    cortex_client: CortexExecClient | None = None,
) -> ThoughtEventV1:
    client = cortex_client or CortexExecClient(bus)
    mind_coloring = await _maybe_build_mind_coloring(request, bus=bus)
    plan_request = build_stance_react_plan_request(request, mind_coloring=mind_coloring)
    exec_result = await client.execute_plan(
        source=_source(),
        req=plan_request,
        correlation_id=request.correlation_id,
        timeout_sec=settings.stance_react_timeout_sec,
    )
    raw_payload = extract_stance_react_payload(exec_result)
    thought = parse_stance_react_payload(
        raw_payload,
        correlation_id=request.correlation_id,
        session_id=request.session_id,
    )
    enriched = apply_stance_react_pipeline(thought, request)
    capsule = _extract_grounding_capsule(exec_result)
    if capsule is not None:
        enriched = enriched.model_copy(update={"grounding_capsule": capsule})
    return enriched
```

Note: `publish_mind_run_artifact_for_thought` is added in Task 7. For this task, temporarily guard the call so the module imports: add a stub in `mind_enrichment.py` now (it will be fleshed out in Task 7):

```python
async def publish_mind_run_artifact_for_thought(*_args: Any, **_kwargs: Any) -> None:
    """Placeholder — implemented in Task 7."""
    return None
```

And import it in `bus_listener.py`:

```python
from .mind_enrichment import (
    build_light_mind_request,
    publish_mind_run_artifact_for_thought,
    run_mind_for_thought,
    select_mind_coloring,
)
```

- [ ] **Step 4: Run to verify it passes**

Run:

```bash
PYTHONPATH=services/orion-thought:. python -m pytest services/orion-thought/tests/test_mind_enrichment_fail_open.py services/orion-thought/tests/test_stance_react_grounding_capsule.py -q
```

Expected: PASS (both the new fail-open tests and the pre-existing grounding-capsule regression tests). The grounding-capsule tests must still pass because they run with `mind_enrichment_enabled=false` by default → `_maybe_build_mind_coloring` returns `None` immediately.

- [ ] **Step 5: Commit**

```bash
git add services/orion-thought/app/bus_listener.py services/orion-thought/app/mind_enrichment.py \
  services/orion-thought/tests/test_mind_enrichment_fail_open.py
git commit -m "feat(orion-thought): run Mind enrichment in run_stance_react (fail-open, flag-gated)"
```

---

## Task 7: Artifact publish + `mode="orion"` tag

**Files:**
- Modify: `services/orion-thought/app/mind_enrichment.py`
- Test: `services/orion-thought/tests/test_mind_artifact_mode_tag.py`

Publish `MindRunArtifactV1` to `orion:mind:artifact` so the existing state-journaler → `mind_runs` → hub `/api/mind/runs*` EKG lights up for unified turns. Tag `request_summary_jsonb.mode = "orion"`.

- [ ] **Step 1: Write the failing test**

Create `services/orion-thought/tests/test_mind_artifact_mode_tag.py`:

```python
from __future__ import annotations

from uuid import uuid4

import pytest

from app.mind_enrichment import publish_mind_run_artifact_for_thought
from orion.core.bus.bus_schemas import ServiceRef
from orion.mind.v1 import MindHandoffBriefV1, MindRunPolicyV1, MindRunRequestV1, MindRunResultV1
from orion.schemas.mind.artifact import MindRunArtifactV1
from orion.schemas.thought import HubAssociationBundleV1, StanceReactRequestV1


class _FakeBus:
    def __init__(self) -> None:
        self.published: list[tuple[str, object]] = []

    async def publish(self, channel: str, envelope) -> None:
        self.published.append((channel, envelope))


class _RaisingBus:
    async def publish(self, *_a, **_k) -> None:
        raise RuntimeError("bus down")


def _request() -> StanceReactRequestV1:
    return StanceReactRequestV1(
        correlation_id="corr-1",
        session_id="sess-1",
        user_message="hi",
        association=HubAssociationBundleV1(
            correlation_id="corr-1", broadcast=None, broadcast_stale=True,
            read_source="hub_sql_fallback",
        ),
        repair_bundle=None,
        stance_inputs={"user_message": "hi"},
    )


def _mind_req() -> MindRunRequestV1:
    return MindRunRequestV1(
        correlation_id="corr-1", session_id="sess-1", trigger="user_turn",
        snapshot_inputs={"user_text": "hi", "messages_tail": []},
        policy=MindRunPolicyV1(n_loops_max=1, wall_time_ms_max=12000, router_profile_id="default"),
    )


def _mind_res() -> MindRunResultV1:
    return MindRunResultV1(
        mind_run_id=uuid4(), ok=True, snapshot_hash="hash-1",
        brief=MindHandoffBriefV1(mind_quality="meaningful_synthesis"),
        mind_quality="meaningful_synthesis",
    )


@pytest.mark.asyncio
async def test_artifact_published_with_orion_mode() -> None:
    bus = _FakeBus()
    await publish_mind_run_artifact_for_thought(
        bus,
        source=ServiceRef(name="orion-thought", node="athena", version="0.1.0"),
        request=_request(), mind_req=_mind_req(), mind_res=_mind_res(),
        channel="orion:mind:artifact",
    )
    assert len(bus.published) == 1
    channel, envelope = bus.published[0]
    assert channel == "orion:mind:artifact"
    artifact = MindRunArtifactV1.model_validate(envelope.payload)
    assert artifact.request_summary_jsonb["mode"] == "orion"
    assert artifact.request_summary_jsonb["correlation_id"] == "corr-1"
    assert artifact.ok is True
    assert artifact.router_profile_id == "default"


@pytest.mark.asyncio
async def test_artifact_publish_failure_is_swallowed() -> None:
    # Publish failure must never propagate out of the stance stage.
    await publish_mind_run_artifact_for_thought(
        _RaisingBus(),
        source=ServiceRef(name="orion-thought", node="athena", version="0.1.0"),
        request=_request(), mind_req=_mind_req(), mind_res=_mind_res(),
        channel="orion:mind:artifact",
    )  # must not raise
```

- [ ] **Step 2: Run to verify it fails**

Run:

```bash
PYTHONPATH=services/orion-thought:. python -m pytest services/orion-thought/tests/test_mind_artifact_mode_tag.py -q
```

Expected: FAIL — `test_artifact_published_with_orion_mode` asserts a publish happened but the placeholder is a no-op (`len(bus.published) == 0`).

- [ ] **Step 3: Replace the placeholder with the real publisher**

In `services/orion-thought/app/mind_enrichment.py`, add imports:

```python
from datetime import datetime, timezone

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.mind.artifact import MindRunArtifactV1
```

Replace the `publish_mind_run_artifact_for_thought` placeholder from Task 6 with:

```python
_MIND_ARTIFACT_KIND = "mind.run.artifact.v1"


async def publish_mind_run_artifact_for_thought(
    bus: Any,
    *,
    source: "ServiceRef",
    request: "StanceReactRequestV1",
    mind_req: "MindRunRequestV1",
    mind_res: MindRunResultV1,
    channel: str,
) -> None:
    """Publish MindRunArtifactV1 for a unified-turn Mind run (mode='orion').

    Log-and-continue: an artifact publish failure must never fail the stance stage.
    """
    try:
        summary = {
            "correlation_id": request.correlation_id,
            "verb": "stance_react",
            "mode": "orion",
            "session_id": request.session_id,
        }
        artifact = MindRunArtifactV1(
            mind_run_id=mind_res.mind_run_id,
            correlation_id=request.correlation_id,
            session_id=request.session_id,
            trigger=mind_req.trigger,
            ok=mind_res.ok,
            error_code=mind_res.error_code,
            snapshot_hash=mind_res.snapshot_hash,
            router_profile_id=mind_req.policy.router_profile_id,
            result_jsonb=mind_res.model_dump(mode="json"),
            request_summary_jsonb=summary,
            created_at_utc=datetime.now(timezone.utc),
        )
        env = BaseEnvelope(
            kind=_MIND_ARTIFACT_KIND,
            source=source,
            correlation_id=request.correlation_id,
            payload=artifact.model_dump(mode="json"),
        )
        await bus.publish(channel, env)
        logger.info(
            "mind_run_artifact_publish corr=%s mind_run_id=%s mode=orion ok=%s",
            request.correlation_id,
            artifact.mind_run_id,
            artifact.ok,
        )
    except Exception as exc:  # noqa: BLE001 — observability must never fail the turn
        logger.warning(
            "mind_artifact_publish_failed corr=%s err=%s",
            request.correlation_id,
            exc,
        )
```

Also remove the placeholder stub added in Task 6 (the one that just `return None`). Ensure `StanceReactRequestV1` and `MindRunRequestV1` are imported (already added in Tasks 2 & 4).

- [ ] **Step 4: Run to verify it passes**

Run:

```bash
PYTHONPATH=services/orion-thought:. python -m pytest services/orion-thought/tests/test_mind_artifact_mode_tag.py -q
```

Expected: PASS (2 passed).

- [ ] **Step 5: Verify `BaseEnvelope.correlation_id` accepts a plain string**

The reference publisher in cortex-orch passes `correlation_id=correlation_id` (a str) to `BaseEnvelope` (`mind_runtime.py:879`), so a string is accepted. If validation fails at runtime with a UUID coercion error, wrap with the existing `_envelope_correlation_id` helper pattern from `bus_listener.py:36`. Run the artifact test again to confirm no coercion error.

- [ ] **Step 6: Commit**

```bash
git add services/orion-thought/app/mind_enrichment.py services/orion-thought/tests/test_mind_artifact_mode_tag.py
git commit -m "feat(orion-thought): publish MindRunArtifactV1 with mode=orion for unified turns"
```

---

## Task 8: Contract sanity + full gate suite

**Files:** none (verification only)

- [ ] **Step 1: Confirm no new bus/schema contract is needed**

`MindRunArtifactV1` and `orion:mind:artifact` are already registered (published today by cortex-orch). Confirm:

```bash
rg "mind.run.artifact.v1|orion:mind:artifact" orion/bus/channels.yaml orion/schemas/registry.py
```

Expected: both the channel and schema id already appear. If EITHER is missing, STOP — the design assumed they were registered; add the registration in the same changeset and note it in the PR. (Design §"Files likely to touch" claims no registry change is needed; this step verifies that claim.)

- [ ] **Step 2: Run the deterministic contract gates**

Run:

```bash
python scripts/check_env_template_parity.py
python scripts/check_schema_registry.py
python scripts/check_bus_channels.py
```

Expected: all three pass. If `check_env_template_parity.py` fails on orion-thought, re-run `python scripts/sync_local_env_from_example.py` (Task 3 Step 8) and re-check.

- [ ] **Step 3: Run the full orion-thought gate suite**

Run:

```bash
PYTHONPATH=services/orion-thought:. python -m pytest services/orion-thought/tests/ -q
```

Expected: PASS. New files: `test_mind_coloring_selector.py`, `test_mind_light_snapshot.py`, `test_settings_mind_enrichment.py`, `test_mind_http_client.py`, `test_stance_context_mind_coloring.py`, `test_stance_prompt_renders_coloring.py`, `test_mind_enrichment_fail_open.py`, `test_mind_artifact_mode_tag.py`. Pre-existing tests (reverie, grounding-capsule, artifact) must remain green.

- [ ] **Step 4: Lint check**

Use the ReadLints tool on `services/orion-thought/app/mind_enrichment.py` and `services/orion-thought/app/bus_listener.py`. Fix any introduced lints.

- [ ] **Step 5: Commit (only if lint fixes were needed)**

```bash
git add -A
git commit -m "chore(orion-thought): fix lints in Mind enrichment wiring"
```

---

## Task 9: Enrichment eval (anti-contradiction + aliveness)

**Files:**
- Create: `services/orion-thought/evals/test_mind_enrichment_eval.py`

This is a **required deliverable** (design §Testing/Eval). The load-bearing guarantee — technical/coding/agent turns keep their `task_mode`/`conversation_frame` (no chat-mode leakage) — is latent LLM behavior that unit tests cannot catch. The eval measures the *structural* guarantee that the enrichment pipeline cannot mechanically overwrite stance fields, plus that the coloring block carries self/curiosity signal on relational turns.

Because the eval must run without a live LLM (evals run in CI-like conditions), it exercises the **deterministic** seams end-to-end: it proves (a) the selector never emits task-control fields across a mix of relational/technical/agent Mind results, and (b) that `build_stance_react_context` injects coloring without touching the authoritative stance fields (which only the LLM writes). It documents the live-LLM leakage check as the rollout step 4 gate.

- [ ] **Step 1: Write the eval**

Create `services/orion-thought/evals/test_mind_enrichment_eval.py`:

```python
"""Eval: Mind coloring is mode-agnostic and never leaks task-control into the
unified stance. Distinct from unit tests — this scores a labeled mix of
relational / technical / agent Mind results for the anti-contradiction and
aliveness guarantees the design calls load-bearing.

Run: pytest services/orion-thought/evals -q
"""
from __future__ import annotations

from uuid import uuid4

from app.bus_listener import build_stance_react_context
from app.mind_enrichment import select_mind_coloring
from orion.mind.synthesis_v1 import ActiveCognitiveFrontierV1, SelectedFrontierMatterV1
from orion.mind.v1 import MindControlDecisionV1, MindHandoffBriefV1, MindRunResultV1
from orion.schemas.thought import HubAssociationBundleV1, StanceReactRequestV1

_TASK_CONTROL_TOKENS = (
    "task_mode", "answer_strategy", "conversation_frame", "response_priorities",
    "response_hazards", "route_kind", "mode_binding", "allowed_verbs", "mode_suggestion",
)


def _frontier(summaries: list[str]) -> ActiveCognitiveFrontierV1:
    return ActiveCognitiveFrontierV1(
        selected=[
            SelectedFrontierMatterV1(
                matter_id=f"m{i}", source_claim_id=f"c{i}", label=f"l{i}",
                summary=s, matter_kind="curiosity_affordance", score=0.8 - i * 0.1,
            )
            for i, s in enumerate(summaries)
        ]
    )


def _result(*, frame: str, task_mode: str, themes: list[str], summaries: list[str]) -> MindRunResultV1:
    payload = {
        "conversation_frame": frame,
        "task_mode": task_mode,
        "identity_salience": "medium",
        "user_intent": "x",
        "self_relevance": "matters to me",
        "juniper_relevance": "matters to Juniper",
        "reflective_themes": themes,
        "response_priorities": ["p1"],
        "response_hazards": ["h1"],
        "answer_strategy": "strat",
        "stance_summary": "s",
    }
    return MindRunResultV1(
        mind_run_id=uuid4(), ok=True, snapshot_hash="h",
        decision=MindControlDecisionV1(route_kind="chat", allowed_verbs=["speak"], mode_suggestion="brain"),
        brief=MindHandoffBriefV1(
            mind_quality="meaningful_synthesis",
            active_frontier=_frontier(summaries),
            stance_payload=payload,
        ),
        mind_quality="meaningful_synthesis",
    )


# (label, mind_result)
CASES = [
    ("relational", _result(frame="reflective", task_mode="reflective_dialogue",
                            themes=["continuity", "trust"], summaries=["what changed since we spoke"])),
    ("technical", _result(frame="technical", task_mode="technical_collaboration",
                          themes=["the deploy risk"], summaries=["the failing migration"])),
    ("agent", _result(frame="planning", task_mode="triage",
                      themes=["the blocked task"], summaries=["which tool to call next"])),
]


def _request() -> StanceReactRequestV1:
    return StanceReactRequestV1(
        correlation_id="corr-1", session_id="sess-1", user_message="…",
        association=HubAssociationBundleV1(
            correlation_id="corr-1", broadcast=None, broadcast_stale=True,
            read_source="hub_sql_fallback",
        ),
        repair_bundle=None, stance_inputs={"user_message": "…"},
    )


def test_no_task_control_leaks_across_all_modes() -> None:
    import json
    for label, result in CASES:
        coloring = select_mind_coloring(result, max_items=3)
        assert coloring is not None, f"{label}: expected non-empty coloring"
        blob = json.dumps(coloring)
        for token in _TASK_CONTROL_TOKENS:
            assert token not in blob, f"{label}: task-control token {token!r} leaked"


def test_context_injection_does_not_touch_stance_authoring() -> None:
    # The coloring lands only under the mind_coloring key; the authoritative
    # stance fields are authored by the LLM downstream, never by this pipeline.
    _, result = CASES[0]
    coloring = select_mind_coloring(result, max_items=3)
    ctx = build_stance_react_context(_request(), mind_coloring=coloring)
    assert set(ctx["mind_coloring"].keys()) == set(coloring.keys())
    # No stance-authoring key is fabricated by context assembly.
    assert "stance_harness_slice" not in ctx
    assert "imperative" not in ctx


def test_relational_turn_carries_self_and_curiosity_signal() -> None:
    _, result = CASES[0]
    coloring = select_mind_coloring(result, max_items=3)
    assert coloring is not None
    assert coloring["self_relevance"]
    assert coloring["curiosity_threads"]
    assert coloring["reflective_themes"]
```

- [ ] **Step 2: Run the eval**

Run:

```bash
PYTHONPATH=services/orion-thought:. python -m pytest services/orion-thought/evals/ -q
```

Expected: PASS (the new eval + the pre-existing reverie hollow-guard eval).

- [ ] **Step 3: Commit**

```bash
git add services/orion-thought/evals/test_mind_enrichment_eval.py
git commit -m "test(orion-thought): add Mind enrichment anti-contradiction + aliveness eval"
```

---

## Task 10: README + docker config smoke

**Files:**
- Modify: `services/orion-thought/README.md`

- [ ] **Step 1: Document the feature**

Append a section to `services/orion-thought/README.md`:

```markdown
## Mind stance enrichment (unified turn)

Set `ORION_THOUGHT_MIND_ENRICHMENT_ENABLED=true` (default `false`) to run
`orion-mind` before `stance_react` and inject an advisory self/attention
`mind_coloring` block into the verb context. `stance_react` stays the sole
author of `ThoughtEventV1` and reconciles the coloring (existing inputs win —
it never forces chat framing on technical/agent turns).

Module: `app/mind_enrichment.py` (snapshot builder, fail-open HTTP client,
allow-list coloring selector, artifact publisher).

Flags:

| Env key | Default | Role |
|---------|---------|------|
| `ORION_THOUGHT_MIND_ENRICHMENT_ENABLED` | `false` | Master switch |
| `ORION_MIND_BASE_URL` | `http://orion-mind:6611` | Mind endpoint |
| `ORION_THOUGHT_MIND_TIMEOUT_SEC` | `15` | HTTP read timeout |
| `ORION_THOUGHT_MIND_WALL_MS` | `12000` | Mind policy wall time |
| `ORION_THOUGHT_MIND_ROUTER_PROFILE` | `default` | Mind router profile |
| `ORION_THOUGHT_MIND_MAX_RESPONSE_BYTES` | `2000000` | Response body cap |
| `ORION_THOUGHT_MIND_ARTIFACT_PUBLISH_ENABLED` | `false` | Publish `mind_runs` artifact (`mode=orion`) |
| `ORION_THOUGHT_MIND_COLORING_MAX_ITEMS` | `3` | Coloring list cap |

**Preconditions (silent no-op if unmet):**
1. `orion-mind` must have `MIND_LLM_SYNTHESIS_ENABLED=true` — `meaningful_synthesis`
   (the only quality that fires coloring) is produced only by Mind's LLM path.
2. `orion-thought` must be able to reach `ORION_MIND_BASE_URL`.

Everything fails open: Mind unconfigured / unreachable / slow / low-quality →
byte-identical to today's stance behavior.
```

- [ ] **Step 2: Validate docker compose config**

Run:

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-thought/.env \
  -f services/orion-thought/docker-compose.yml config
```

Expected: prints the resolved config with all nine new `ORION_THOUGHT_MIND_*` / `CHANNEL_MIND_ARTIFACT` env values present. If Docker is unavailable in this environment, say so plainly and skip (the deterministic parity check in Task 8 already covers env wiring).

- [ ] **Step 3: Commit**

```bash
git add services/orion-thought/README.md
git commit -m "docs(orion-thought): document Mind stance enrichment flags + preconditions"
```

---

## Task 11: Review gate + PR

**Files:** none (process)

- [ ] **Step 1: Re-run all gates**

Run:

```bash
python scripts/check_env_template_parity.py
python scripts/check_schema_registry.py
python scripts/check_bus_channels.py
PYTHONPATH=services/orion-thought:. python -m pytest services/orion-thought/tests/ services/orion-thought/evals/ -q
git diff --check
```

Expected: all pass; `git diff --check` reports no whitespace errors.

- [ ] **Step 2: Run code review in a subagent**

Dispatch the `code-reviewer` subagent against the branch diff (base `main`). Focus: allow-list completeness (no task-control leakage), fail-open coverage, no empty-shell coloring, env parity, that `mind_enrichment_enabled=false` is byte-identical to prior behavior. Fix all material findings and re-run the affected checks.

- [ ] **Step 3: Confirm `.env` is not staged**

Run:

```bash
git check-ignore services/orion-thought/.env
git status --short
```

Expected: `.env` ignored; not in the staged/unstaged tracked list.

- [ ] **Step 4: Push and open PR**

```bash
git push -u origin HEAD
```

Then create the PR with the AGENTS.md §18 Markdown template. Required sections: Summary, Outcome moved, Architecture touched, Files changed, Schema/bus/API changes (none — reuse `MindRunArtifactV1`), Env/config changes (9 new keys + `.env` synced), Tests run, Evals run, Docker/build/smoke checks, Review findings fixed, Restart required, Risks/concerns, PR link.

- [ ] **Step 5: List the operator restart command in the PR**

The operator (not the agent) must rebuild for `httpx` + new env:

```bash
docker compose --env-file .env --env-file services/orion-thought/.env \
  -f services/orion-thought/docker-compose.yml up -d --build
```

And run the rollout pre-flight from the design §Rollout (confirm `orion-mind` `MIND_LLM_SYNTHESIS_ENABLED=true`, confirm reachability, then flip `ORION_THOUGHT_MIND_ENRICHMENT_ENABLED=true`).

---

## Acceptance checks (map to design §Acceptance checks)

- [ ] Flag off → `run_stance_react` context has no `mind_coloring`; grounding-capsule regression tests unchanged (Task 6 `test_enrichment_disabled_is_baseline`).
- [ ] Flag on, meaningful Mind → `mind_coloring` in verb context + rendered in prompt; `ThoughtEventV1` still valid (Task 6 `test_enrichment_enabled_meaningful_injects_coloring` + Task 5b render tests).
- [ ] Coloring key set equals the allow-list exactly; task-control never crosses (Task 1 `test_meaningful_synthesis_key_set_equals_allow_list`, `test_task_control_fields_never_cross`; Task 9 eval).
- [ ] Mind down/slow/unreachable → turn still completes (Task 4 fail-open tests + Task 6 `test_enrichment_enabled_but_mind_fails_open`).
- [ ] Unified-turn Mind runs publish with `mode="orion"` (Task 7 `test_artifact_published_with_orion_mode`).
- [ ] "At least one real unified turn produces a non-empty `meaningful_synthesis` coloring end-to-end" — **operational, at rollout** (design §Rollout step 4). Not automatable here; guards C1/I5. Note in PR as the manual acceptance gate.
- [ ] Technical/coding/agent turns not chat-framed — structural guarantee proven by Task 9 eval; residual latent-LLM check is the rollout step-4 transcript comparison (design flags this as measured-at-rollout).

## Notes on scope boundaries (do not build)

- Mind driving/replacing the stance (legacy `mind_skip_stance_synthesis`) — out of scope ("Phase B").
- Task-control output (`task_mode`, `answer_strategy`, `allowed_verbs`, `route_kind`, `mode_binding`, `mode_suggestion`) — never selected.
- Cold projection rebuild on the turn-critical path — never; `shadow_synthesis` stays `None`, coloring must not depend on it.
- No changes to `StanceReactRequestV1`, bus channel semantics, Hub, or UI/EKG code.
- No keyword/phrase/regex detector for user state (anti-slop rule).
