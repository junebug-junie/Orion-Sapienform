# Hire Determination Grounding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ground Orion’s Cursor-vs-local-crawl choice in origin-aware Mind framing, soft work-shape labels, and an Orion-authored `:InvestigationRole` — without Python auto-hire and without breaking Juniper chat.

**Architecture:** Thread `utterance_origin` (`juniper` | `orion`) into Mind requests (flag + situation prose). Guard Mind→thought coloring so `user_intent` / uncertainty / soft depth labels pass for Orion-origin turns without leaking `conversation_frame` / `task_mode`. On curiosity, appraise a short investigation subject while the full kickoff stays the harness prompt. Teach + read `:InvestigationRole`; keep HelpRequest as the hire ticket (after a short local look). Optional empty-`tried_summary` enqueue skip when hops exist.

**Tech Stack:** Python 3, Pydantic, pytest, FalkorDB Cypher via existing `WorldviewReader`, Hub `execute_unified_turn`, `orion-thought` mind enrichment, `orion-mind` stance handoff.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-09-15-orion-hire-determination-grounding-design.md` (revised; PR #2233)
- Parent peer contract unchanged: `docs/superpowers/specs/2026-09-14-orion-contractor-peer-design.md`
- **Orion only** authors `:InvestigationRole` and `:HelpRequest` — Python never stamps hire/role choices
- **Do not** key hire on AST/HOT attention winners, thermals, or hop-count auto-hire
- Soft labels are **advisory** (`shallow`/`deep`/`unknown`, etc.) — not calibrated scorers
- Never pass `conversation_frame` / `task_mode` / `answer_strategy` / response priority-hazard control fields through Mind coloring
- Neighbor file: `curiosity_investigation.py` already has outreach hop wiring (PR #2237) — rebase onto current `main` before coding; do not revert outreach
- Work in a **new** worktree from current `main`: e.g. `../Orion-Sapienform-hire-determination-impl` on `feat/hire-determination-grounding` (do not implement on the docs-only branch alone; cherry-pick or include the design/plan docs)
- Do not commit `.env`
- After code changes, run `scripts/safe_graphify_update.sh` from the worktree (not bare `graphify update .`)
- `HUB_CURIOSITY_CONTRACTOR_PEER_ENABLED=false` must still omit HelpRequest teach and enqueue zero jobs

## File map

| File | Responsibility |
|---|---|
| `orion/mind/v1.py` | Optional `utterance_origin` on `MindRunRequestV1` |
| `orion/schemas/chat_stance.py` | Optional soft work-shape fields on `ChatStanceBrief` |
| `services/orion-mind/app/stance_handoff.py` | Prompt: when origin=orion, ask for soft labels |
| `services/orion-thought/app/mind_enrichment.py` | Origin → situation prose; guarded `select_mind_coloring` |
| `orion/hub/turn_orchestrator.py` | Kwargs `utterance_origin`, `mind_appraisal_text` → stance/Mind |
| `services/orion-hub/scripts/api_routes.py` (or WS chat entry) | Set `utterance_origin="juniper"` on Hub chat unified turns |
| `services/orion-hub/scripts/curiosity_investigation.py` | `orion` origin + subject-sized appraisal text |
| `orion/curiosity/investigation_subject.py` | **New** — pure subject builder |
| `orion/curiosity/kickoff_prompt.py` / `self_inquiry_prompt.py` | Role + hire teach rewrite |
| `orion/curiosity/worldview.py` | `:InvestigationRole` label + RO readers |
| `orion/curiosity/peer_briefs.py` | Empty `tried_summary` gate when hops exist |
| Tests under `services/orion-thought/tests/`, `orion/curiosity/tests/` or `tests/`, `services/orion-hub/tests/`, `services/orion-mind/tests/` | TDD per task |
| Spec/plan docs | Already exist — carry onto feat branch |

---

### Task 1: `utterance_origin` on Mind request + situation prose

**Files:**
- Modify: `orion/mind/v1.py`
- Modify: `services/orion-thought/app/mind_enrichment.py`
- Modify: `orion/hub/turn_orchestrator.py` (signature + `stance_req` construction ~918–932)
- Modify: Hub chat caller(s) that invoke `execute_unified_turn` with `reading_context="unified_chat"` (at least `orion/hub/turn_orchestrator.py` internal chat helper and/or `services/orion-hub/scripts/api_routes.py`)
- Modify: `services/orion-hub/scripts/curiosity_investigation.py` (`execute_unified_turn` call ~1945)
- Test: `services/orion-thought/tests/test_mind_light_snapshot.py` (extend) and/or new `services/orion-thought/tests/test_mind_utterance_origin.py`
- Test: `services/orion-hub/tests/test_turn_orchestrator_ws_frames.py` or focused new test that stance_inputs carry origin

**Interfaces:**
- Consumes: existing `StanceReactRequestV1.stance_inputs`, `build_light_mind_request`
- Produces:
  - `MindRunRequestV1.utterance_origin: Literal["juniper", "orion"] | None = None`
  - `execute_unified_turn(..., utterance_origin: str | None = None, mind_appraisal_text: str | None = None)`
  - `stance_inputs["utterance_origin"]` and situation prose in Mind snapshot facets

- [ ] **Step 1: Write the failing tests**

Create `services/orion-thought/tests/test_mind_utterance_origin.py`:

```python
from __future__ import annotations

from orion.schemas.attention_frame import AttentionBroadcastProjectionV1, AttentionFrameV1
from orion.schemas.hub_association import HubAssociationBundleV1
from orion.schemas.thought import StanceReactRequestV1
from app.mind_enrichment import build_light_mind_request


def _stance(user_message: str, *, origin: str | None) -> StanceReactRequestV1:
    inputs: dict = {"user_message": user_message}
    if origin is not None:
        inputs["utterance_origin"] = origin
    # Mirror services/orion-thought/tests/test_mind_light_snapshot.py::_request
    return StanceReactRequestV1(
        correlation_id="corr-origin-1",
        session_id="sess-1",
        user_message=user_message,
        association=HubAssociationBundleV1(
            correlation_id="corr-origin-1",
            broadcast=None,
            broadcast_stale=False,
            read_source="felt_state_reader",
        ),
        repair_bundle=None,
        stance_inputs=inputs,
    )


def test_build_light_mind_request_sets_utterance_origin_field() -> None:
    req = build_light_mind_request(
        _stance("hello from juniper", origin="juniper"),
        wall_time_ms=1000,
        router_profile="default",
    )
    assert req.utterance_origin == "juniper"


def test_situation_compact_includes_origin_prose_for_orion() -> None:
    req = build_light_mind_request(
        _stance("investigate claim X", origin="orion"),
        wall_time_ms=1000,
        router_profile="default",
    )
    facets = (req.snapshot_inputs or {}).get("facets") or {}
    situation = facets.get("situation_compact") or {}
    note = str(situation.get("utterance_origin_note") or "")
    assert "orion" in note.lower()
    assert req.utterance_origin == "orion"


def test_missing_origin_stays_none_and_omits_origin_note() -> None:
    req = build_light_mind_request(
        _stance("legacy caller", origin=None),
        wall_time_ms=1000,
        router_profile="default",
    )
    assert req.utterance_origin is None
    facets = (req.snapshot_inputs or {}).get("facets") or {}
    situation = facets.get("situation_compact") or {}
    assert "utterance_origin_note" not in situation
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest services/orion-thought/tests/test_mind_utterance_origin.py -v`
Expected: FAIL (no `utterance_origin` on `MindRunRequestV1` and/or builder ignores stance_inputs)

- [ ] **Step 3: Minimal implementation**

1. In `orion/mind/v1.py`, on `MindRunRequestV1` add:

```python
utterance_origin: Literal["juniper", "orion"] | None = None
```

2. In `build_light_mind_request`, read origin from `stance_inputs.get("utterance_origin")`, validate to `juniper`/`orion` or None, set on the request, and merge into `situation_compact`:

```python
ORIGIN_NOTES = {
    "juniper": "This utterance is from Juniper (human collaborator).",
    "orion": "This utterance is from Orion (self-authored investigation subject).",
}
```

If there is no broadcast-derived situation, still create a compact dict with only `utterance_origin_note` when origin is set.

3. In `execute_unified_turn`, add kwargs `utterance_origin: str | None = None` and `mind_appraisal_text: str | None = None` (appraisal used in Task 3; for Task 1 only thread origin into `stance_inputs`).

```python
stance_inputs = {"user_message": user_message}
if utterance_origin in ("juniper", "orion"):
    stance_inputs["utterance_origin"] = utterance_origin
```

4. Hub chat path: pass `utterance_origin="juniper"`.
5. Curiosity `execute_unified_turn` call: pass `utterance_origin="orion"`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest services/orion-thought/tests/test_mind_utterance_origin.py services/orion-thought/tests/test_mind_light_snapshot.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add orion/mind/v1.py services/orion-thought/app/mind_enrichment.py \
  orion/hub/turn_orchestrator.py services/orion-hub/scripts/curiosity_investigation.py \
  services/orion-hub/scripts/api_routes.py \
  services/orion-thought/tests/test_mind_utterance_origin.py
# include any other chat entrypoints you actually edited
git commit -m "$(cat <<'EOF'
feat(mind): utterance_origin flag + prose on Mind requests

EOF
)"
```

---

### Task 2: Soft work-shape labels + guarded Mind coloring allow-list

**Files:**
- Modify: `orion/schemas/chat_stance.py`
- Modify: `services/orion-mind/app/stance_handoff.py` (`_STANCE_SYSTEM` / handoff when origin known)
- Modify: `services/orion-thought/app/mind_enrichment.py` (`MIND_COLORING_ALLOWED_KEYS`, `select_mind_coloring`)
- Modify: `services/orion-thought/tests/test_mind_coloring_selector.py`
- Test: `services/orion-mind/tests/` — extend or add handoff test that soft keys survive coerce when present
- Check: `scripts/check_schema_registry.py` if ChatStanceBrief registration cares (usually model import only)

**Interfaces:**
- Consumes: `MindRunResultV1.brief.stance_payload`, `ActiveCognitiveFrontierV1.selected[].features`
- Produces: coloring keys (when policy allows):
  - always (both origins, if present): `user_intent`, `uncertainty_summary`
  - only `utterance_origin=="orion"`: `expected_depth`, `cross_cutting`, `foresight_note`
  - never: `conversation_frame`, `task_mode`, `answer_strategy`, `response_priorities`, `response_hazards`

- [ ] **Step 1: Write the failing tests**

Extend `services/orion-thought/tests/test_mind_coloring_selector.py`:

```python
def test_user_intent_passes_for_juniper_origin() -> None:
    coloring = select_mind_coloring(
        _result(ok=True, quality="meaningful_synthesis"),
        max_items=3,
        utterance_origin="juniper",
    )
    assert coloring is not None
    assert coloring.get("user_intent") == "connect"
    assert "conversation_frame" not in coloring
    assert "task_mode" not in coloring
    assert "expected_depth" not in coloring  # orion-only soft label


def test_orion_origin_passes_soft_work_shape_labels() -> None:
    payload = _stance_payload()
    payload["expected_depth"] = "deep"
    payload["cross_cutting"] = "yes"
    payload["foresight_note"] = "Likely multi-service archaeology."
    coloring = select_mind_coloring(
        _result_with_payload(payload),
        max_items=3,
        utterance_origin="orion",
    )
    assert coloring is not None
    assert coloring["expected_depth"] == "deep"
    assert coloring["cross_cutting"] == "yes"
    assert "multi-service" in coloring["foresight_note"]


def test_soft_labels_blocked_for_juniper_even_if_payload_has_them() -> None:
    payload = _stance_payload()
    payload["expected_depth"] = "deep"
    coloring = select_mind_coloring(
        _result_with_payload(payload),
        max_items=3,
        utterance_origin="juniper",
    )
    assert coloring is not None
    assert "expected_depth" not in coloring
```

Update `test_meaningful_synthesis_key_set_equals_allow_list` — allow-list is now **base keys ∪ conditional keys**. Prefer asserting `set(coloring.keys()) <= MIND_COLORING_ALLOWED_KEYS` and that base keys still appear, **or** split into `MIND_COLORING_BASE_KEYS` + `MIND_COLORING_ORION_WORK_SHAPE_KEYS`. Do not leave the old equality assertion broken without replacing it.

Also add a ChatStanceBrief round-trip test that optional soft fields survive `model_validate` / `model_dump`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest services/orion-thought/tests/test_mind_coloring_selector.py -v`
Expected: FAIL (`user_intent` absent; no `utterance_origin` param)

- [ ] **Step 3: Minimal implementation**

1. Add optional fields to `ChatStanceBrief`:

```python
expected_depth: Literal["shallow", "deep", "unknown"] | None = None
cross_cutting: Literal["yes", "no", "unknown"] | None = None
foresight_note: str | None = Field(default=None, max_length=240)
```

2. Update stance handoff system prompt: when building the user pack for handoff, if `utterance_origin == "orion"` (thread from Mind request snapshot / pack), instruct the model to also fill those three fields; otherwise omit the instruction. Defaults/`unknown` are fine.

3. Change `select_mind_coloring(result, *, max_items=3, utterance_origin: str | None = None)`:
   - Always allow-list-add `user_intent` via `_clip_str_or_none(stance_payload.get("user_intent"))` when non-empty
   - Build `uncertainty_summary` from top selected frontier matters (e.g. clip `f"{label}:{confidence:.2f}"` join, max 240 chars) when frontier exists
   - If `utterance_origin == "orion"`, also copy soft label fields when present and valid
   - Callers in thought bus_listener must pass origin from the Mind request / stance_inputs

4. Keep forbidden-field regression test green.

- [ ] **Step 4: Run tests**

Run:

```bash
pytest services/orion-thought/tests/test_mind_coloring_selector.py \
  services/orion-mind/tests/test_mind_llm_pipeline.py -q
python scripts/check_schema_registry.py
```

Expected: PASS (or only pre-existing unrelated failures — fix any you introduced)

- [ ] **Step 5: Commit**

```bash
git add orion/schemas/chat_stance.py services/orion-mind/app/stance_handoff.py \
  services/orion-thought/app/mind_enrichment.py \
  services/orion-thought/tests/test_mind_coloring_selector.py \
  services/orion-mind/tests/
git commit -m "$(cat <<'EOF'
feat(thought): guard Mind coloring — intent + orion work-shape labels

EOF
)"
```

---

### Task 3: Curiosity investigation subject for Mind appraisal

**Files:**
- Create: `orion/curiosity/investigation_subject.py`
- Create: `orion/curiosity/tests/test_investigation_subject.py` (or `tests/test_investigation_subject.py` if that package test layout is the repo norm — prefer next to `orion/curiosity/tests/` if it exists, else `tests/`)
- Modify: `orion/hub/turn_orchestrator.py` — when `mind_appraisal_text` set, Mind/stance use it; harness/`user_message` stays full prompt
- Modify: `services/orion-hub/scripts/curiosity_investigation.py` — build subject; pass `mind_appraisal_text=...`
- Test: hub unit test that the durable/kickoff path passes subject ≠ full kickoff

**Interfaces:**
- Consumes: worldview continuation note + optional chosen prior claim string available at kickoff build time
- Produces: `build_investigation_subject(*, claim: str | None, continue_note: str | None, max_chars: int = 1200) -> str`

- [ ] **Step 1: Write the failing tests**

```python
from orion.curiosity.investigation_subject import build_investigation_subject


def test_subject_prefers_claim_and_continue_note() -> None:
    text = build_investigation_subject(
        claim="Concept decay never reduces activation",
        continue_note="Still do not know who sets half-life",
    )
    assert "Concept decay never reduces activation" in text
    assert "half-life" in text
    assert "MERGE (h:HelpRequest" not in text
    assert "ASKING FOR CONTRACTOR" not in text


def test_subject_when_claim_missing() -> None:
    text = build_investigation_subject(claim=None, continue_note="keep pulling on ACL")
    assert "not yet chosen" in text.lower() or "no claim" in text.lower()
    assert "ACL" in text


def test_subject_clips_long_inputs() -> None:
    huge = "x" * 5000
    text = build_investigation_subject(claim=huge, continue_note=huge, max_chars=400)
    assert len(text) <= 400
```

Add a hub test (monkeypatch `execute_unified_turn`) asserting kwargs include `mind_appraisal_text` that does not contain the HelpRequest teach block while `user_message` still does (kickoff prompt).

- [ ] **Step 2: Run tests — expect FAIL**

Run: `pytest orion/curiosity/tests/test_investigation_subject.py -v` (adjust path)

- [ ] **Step 3: Implement subject builder + wire**

```python
# orion/curiosity/investigation_subject.py
def build_investigation_subject(*, claim: str | None, continue_note: str | None, max_chars: int = 1200) -> str:
    claim_s = (claim or "").strip()
    note_s = (continue_note or "").strip()
    claim_line = claim_s if claim_s else "Investigation claim: not yet chosen."
    note_line = f"Continue note: {note_s}" if note_s else "Continue note: (none)."
    text = (
        "Orion investigation subject (self-authored).\n"
        f"{claim_line}\n"
        f"{note_line}\n"
    )
    return text.strip()[: max(64, int(max_chars))]
```

In `execute_unified_turn`:

- Keep harness / motor `user_message` as the full prompt.
- For `StanceReactRequestV1.user_message` and Mind snapshot `user_text`, use `mind_appraisal_text` when provided; else `user_message`.
- Put both in stance_inputs: `"user_message": <appraisal or full>`, `"harness_user_message": user_message` only if needed for debugging — do not break existing consumers that expect `stance_inputs["user_message"]` to match stance `user_message`.

In curiosity kickoff path, derive claim from the worldview snapshot / selected prior if already available; otherwise `None`. Pass continue note from `view.continuation`.

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest orion/curiosity/tests/test_investigation_subject.py \
  services/orion-hub/tests/test_curiosity_investigation.py -q --tb=short
```

- [ ] **Step 5: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat(curiosity): Mind appraises investigation subject, not full kickoff

EOF
)"
```

---

### Task 4: `:InvestigationRole` teach + RO reader

**Files:**
- Modify: `orion/curiosity/kickoff_prompt.py` — replace stuck-only hire framing; add role Cypher teach; keep HelpRequest template with “after a short look / role ≠ enqueue”
- Modify: `orion/curiosity/self_inquiry_prompt.py` — same role/hire teach pattern (self-inquiry still forbids peer drafting SelfDefinition)
- Modify: `orion/curiosity/worldview.py` — `LABEL_INVESTIGATION_ROLE = "InvestigationRole"`, list/read helpers
- Modify: `tests/test_curiosity_peer_kickoff.py` (and self-inquiry peer tests)
- Create: `tests/test_investigation_role_worldview.py` or under `orion/curiosity/tests/`
- Optional: Hub post-run log line counting roles vs helps (no auto-write)

**Interfaces:**
- Consumes: none from Python writers (Orion MERGEs)
- Produces:
  - Teach strings containing `InvestigationRole` and `local_crawl` / `hire_cursor`
  - `list_investigation_roles_for_run_cypher(run_id: str) -> str`
  - `InvestigationRoleRecord` dataclass: `run_id`, `choice`, `why`, `written_at`
  - `read_investigation_roles(reader, run_id) -> list[InvestigationRoleRecord]` (latest-wins helper OK)

- [ ] **Step 1: Write the failing tests**

Update `tests/test_curiosity_peer_kickoff.py`:

```python
def test_role_split_taught_when_peer_enabled() -> None:
    on = build_kickoff_prompt(
        _empty_material(),
        run_id="abcd1234abcd",
        graph_enabled=True,
        contractor_peer_enabled=True,
    )
    assert "InvestigationRole" in on
    assert "local_crawl" in on
    assert "hire_cursor" in on
    assert "genuinely stuck" not in on.lower()
    assert "not as a default" not in on.lower()
    assert "MERGE (h:HelpRequest" in on
    assert "does not wake" in on.lower() or "not enqueue" in on.lower() or "after" in on.lower()


def test_role_teach_omitted_when_peer_flag_off() -> None:
    off = build_kickoff_prompt(
        _empty_material(),
        run_id="abcd1234abcd",
        graph_enabled=True,
        contractor_peer_enabled=False,
    )
    assert "InvestigationRole" not in off
    assert "HelpRequest" not in off
```

Reader test with fake rows:

```python
def test_latest_investigation_role_wins() -> None:
    # feed two fake rows; helper returns hire_cursor as latest
    ...
```

- [ ] **Step 2: Run — expect FAIL**

Run: `pytest tests/test_curiosity_peer_kickoff.py -v`

- [ ] **Step 3: Implement teach + reader**

Rewrite `_help_request_section` into a clearer `_role_and_help_section` (or keep name, change body):

- Early: MERGE `:InvestigationRole {run_id, choice, why, written_at}` with choices `local_crawl` | `hire_cursor`; mid-run revise by writing again (latest `written_at` wins).
- Soft disclosure hooks via existing `extra_lines` if Mind labels are available later (optional; empty OK).
- HelpRequest: still taught; clarify role choice does not enqueue Cursor; write HelpRequest after a short local look with real `tried_summary`.

Worldview RO cypher (parameterize like HelpRequest — refuse non-hex `run_id`):

```cypher
MATCH (r:InvestigationRole) WHERE r.run_id = '<rid>'
RETURN r.run_id AS run_id, r.choice AS choice, r.why AS why, r.written_at AS written_at
ORDER BY r.written_at ASC
```

- [ ] **Step 4: Run tests + peer acceptance**

```bash
pytest tests/test_curiosity_peer_kickoff.py tests/test_curiosity_peer_patch0_acceptance.py -q
pytest services/orion-hub/tests/test_curiosity_help_request_enqueue.py -q
```

Expected: PASS; flag-off still omits HelpRequest.

- [ ] **Step 5: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat(curiosity): teach InvestigationRole role split; drop stuck-only hire frame

EOF
)"
```

---

### Task 5: Empty `tried_summary` theater gate

**Files:**
- Modify: `orion/curiosity/peer_briefs.py` (`publish_help_requests_for_run`)
- Modify: `services/orion-hub/tests/test_curiosity_help_request_enqueue.py`

**Interfaces:**
- Consumes: `HelpRequestV1`, optional hop count via `read_hop_notes(reader, run_id)` or inline hop Cypher count
- Produces: skip publish + warning log when `not tried_summary.strip()` and hop_count > 0; still publish when hops == 0 (cannot demand look that did not happen)

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_skips_empty_tried_summary_when_hops_exist() -> None:
    reader = FakeReader(
        help_rows=[{
            "help_id": "h1",
            "run_id": "abcd1234abcd",
            "mode": "world_curiosity",
            "question": "Why?",
            "tried_summary": "   ",
            "success_criteria": "A pointer",
        }],
        hop_count=2,
    )
    bus = FakeBus()
    n = await publish_help_requests_for_run(
        enabled=True, run_id="abcd1234abcd", reader=reader, bus=bus
    )
    assert n == 0
    assert bus.published == []
```

Adapt `FakeReader` in the existing test file to support hop queries, or monkeypatch `read_hop_notes`.

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement gate inside `publish_help_requests_for_run`**

Before publish loop body:

```python
hops = await asyncio.to_thread(read_hop_notes, reader, run_id)  # or count-only helper
hop_count = len(hops)
...
if not (help_req.tried_summary or "").strip() and hop_count > 0:
    logger.warning(
        "curiosity_help_request_skipped_empty_tried_summary help_id=%s run=%s hops=%s",
        help_req.help_id, run_id, hop_count,
    )
    continue
```

No keyword classifiers on `question`.

- [ ] **Step 4: Run tests**

```bash
pytest services/orion-hub/tests/test_curiosity_help_request_enqueue.py -q
```

- [ ] **Step 5: Commit**

```bash
git commit -m "$(cat <<'EOF'
fix(curiosity): skip HelpRequest enqueue when tried_summary empty but hops exist

EOF
)"
```

---

### Task 6: Docs pointer, agent-check, live acceptance note

**Files:**
- Modify: `docs/superpowers/specs/2026-09-15-orion-hire-determination-grounding-design.md` — status line → implementing / shipped sections as accurate
- Create: `docs/superpowers/pr-reports/2026-09-18-hire-determination-grounding-pr.md` (Orion PR template) when opening the impl PR
- Optional one-liner in `orion/curiosity/README.md` only if it already documents HelpRequest teach

- [ ] **Step 1: Run focused gate suite**

```bash
pytest services/orion-thought/tests/test_mind_utterance_origin.py \
  services/orion-thought/tests/test_mind_coloring_selector.py \
  services/orion-thought/tests/test_mind_light_snapshot.py \
  tests/test_curiosity_peer_kickoff.py \
  tests/test_curiosity_peer_patch0_acceptance.py \
  services/orion-hub/tests/test_curiosity_help_request_enqueue.py \
  orion/curiosity/tests/test_investigation_subject.py -q
python scripts/check_schema_registry.py
python scripts/check_env_template_parity.py
git diff --check
```

- [ ] **Step 2: Live smoke (operator / this session) — mark UNVERIFIED if skipped**

After deploy of thought + hub:

1. One Hub Juniper chat turn: confirm origin path did not force curiosity framing (inspect thought/prefix — no `expected_depth` steering).
2. One curiosity durable kickoff: inspect Mind/thought for claim-shaped `user_intent` / subject, not “write Cypher like this.”
3. If Orion writes `InvestigationRole` + optional HelpRequest, confirm graph reads distinguish `local_crawl` / missing / HelpRequest.

Record correlation ids in the PR report. If live check not run: explicitly `UNVERIFIED` in the PR report (do not claim runtime proof).

- [ ] **Step 3: Graphify + commit docs**

```bash
scripts/safe_graphify_update.sh
git add docs/superpowers/
git commit -m "$(cat <<'EOF'
docs(curiosity): hire-determination implementation notes + PR report

EOF
)"
```

- [ ] **Step 4: Push + open PR**

```bash
git push -u origin HEAD
gh pr create --title "feat(curiosity): hire determination grounding" --body "..."
```

Use AGENTS.md PR template in the body.

---

## Self-review (plan vs spec)

| Spec requirement | Task |
|---|---|
| Originator flag + prose | Task 1 |
| Soft labels + guarded allow-list; no frame leak | Task 2 |
| Curiosity Mind subject | Task 3 |
| `:InvestigationRole` + teach rewrite; HelpRequest after short look | Task 4 |
| Empty tried_summary gate | Task 5 |
| Peer flag-off acceptance preserved | Tasks 4–5 tests |
| No attention-winner / Python auto-hire | Global constraints |
| Live thought event proof | Task 6 (or UNVERIFIED) |

No TBD/TODO placeholders remain in task steps. Soft label enum names are fixed here (`expected_depth`, `cross_cutting`, `foresight_note`) so Tasks 2 and 4 stay consistent.

## Rollback

- Revert feat commits independently (origin / coloring / subject / teach / gate).
- `HUB_CURIOSITY_CONTRACTOR_PEER_ENABLED=false` kills enqueue + teach.
