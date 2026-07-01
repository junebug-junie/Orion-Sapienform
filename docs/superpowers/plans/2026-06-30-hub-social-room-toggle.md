# Hub Social-Room Toggle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let Juniper flip a per-session toggle in the Hub UI that routes direct chat with Orion onto the existing `chat_social_room` pipeline (today only reachable via the external CallSyne bridge), with Hub itself pulling continuity from `orion-social-memory` and a tunable redaction posture knob.

**Architecture:** `orion-social-room-bridge` already proves the pattern for external rooms: set `chat_profile="social_room"`, attach `external_room`/`external_participant`, pre-fetch continuity from `orion-social-memory`'s `GET /summary`, and let the existing write-back path (`publish_social_room_turn` → `orion:chat:social:v1` → `orion-sql-writer` → `orion-social-memory`) persist the turn. This plan reuses that exact contract for Hub's own websocket chat path instead of building new adapters: a UI toggle marks the payload `social_room_mode="hub_direct"`, `websocket_handler.py` resolves a **stable** (not session-scoped) Hub↔Juniper identity, calls the already-wired `orion-social-memory` client (`_fetch_social_memory` in `api_routes.py`, which already has a base URL configured in Hub's settings/env/compose), and merges the result into the payload before `build_chat_request()` runs. A new `social_redaction_posture` knob ("strict" default / "relaxed") threads through `social_room.py`'s keyword-based redaction and artifact-dialogue-scope defaults — PII regex detection (email/phone/long-digit) stays on regardless of posture, since that is hygiene, not personality caution. No new services, no new HTTP endpoints, no new env plumbing beyond one setting.

**Tech Stack:** Python 3.12, FastAPI/websockets (`orion-hub`), Pydantic v2, `aiohttp` (existing `_fetch_social_memory` helper), pytest, vanilla JS/Tailwind classes (`orion-hub` static UI).

**Non-goals (v1):** No changes to `orion-social-room-bridge`, `orion-social-memory`, or the HTTP `/api/chat` REST path (Hub's own chat UI uses the websocket path exclusively — see `services/orion-hub/static/js/app.js:10467` / `10840`). No new persistence layer; continuity persistence is entirely handled by the pre-existing `orion-social-memory` bus consumer. **No CallSyne fanout:** Hub-direct social room must never invoke `orion-social-room-bridge`, never call CallSyne post APIs, and never emit `external.room.*` bus events. The bridge is inbound-only (CallSyne → Hub → CallSyne); Hub UI chat stays on the websocket → cortex → optional internal bus write-back path only.

**CallSyne isolation (read this before implementing):**

| Path | Trigger | Outbound to CallSyne? |
|---|---|---|
| CallSyne bridge | Inbound CallSyne webhook/poll | **Yes** — bridge `post_message()` after Hub reply |
| Hub-direct toggle | Hub UI websocket + `social_room_mode=hub_direct` | **No** — reply stays in Hub UI |

Hub reuses the `chat_social_room` verb and `orion-social-memory` continuity machinery, not the external delivery rail. Continuity is namespaced under `platform=hub`, `room_id=hub-direct` (not `callsyne:*`). The `external_room` / `external_participant` fields on Hub-direct payloads are **schema labels for social-memory indexing only** — they do not route through the bridge.

Every Hub-direct turn must carry an audit tag `social_room_mode: "hub_direct"` in payload and `client_meta` so stored turns and routing debug are unambiguously Hub-local (Task 3 + Task 7).

---

## File map

| File | Role |
|---|---|
| `services/orion-hub/scripts/social_room.py` | Redaction-posture-aware scoring/dialogue logic; new Hub-direct identity + summary-merge helpers |
| `services/orion-hub/scripts/cortex_request_builder.py` | Resolves the effective redaction posture (payload override > env default) and threads it into the social-room metadata block |
| `services/orion-hub/app/settings.py` | New `HUB_SOCIAL_ROOM_REDACTION_POSTURE` setting |
| `services/orion-hub/.env_example` | New `HUB_SOCIAL_ROOM_REDACTION_POSTURE` example value |
| `services/orion-hub/scripts/websocket_handler.py` | Detects the toggle, resolves identity, fetches `orion-social-memory` summary, merges into the outgoing payload |
| `services/orion-hub/templates/index.html` | New "Social room" toggle + "Strict/Relaxed" posture selector in the chat composer toolbar |
| `services/orion-hub/static/js/app.js` | Wires the toggle/selector into the outgoing websocket chat payload |
| `services/orion-hub/tests/test_social_room.py` | New/extended unit tests for posture-aware logic and the identity/merge helpers |
| `services/orion-hub/tests/test_cortex_request_builder.py` | New unit tests for the posture getter |
| `services/orion-hub/tests/test_hub_direct_social_room_isolation.py` | Regression tests: no bridge/CallSyne wiring; `social_room_mode` audit tag present |

---

### Task 1 — Posture-aware redaction scoring

**Files:**
- Modify: `services/orion-hub/scripts/social_room.py:5` (imports), `:68` (constant), `:93-139` (`_redaction_score`, `build_social_redaction`)
- Test: `services/orion-hub/tests/test_social_room.py`

- [ ] **Step 1.1: Write the failing tests**

Append to `services/orion-hub/tests/test_social_room.py`:

```python
def test_redaction_score_relaxed_ignores_keyword_hits_but_keeps_pii_regex() -> None:
    strict_score, strict_reasons = hub_social_room._redaction_score(
        "this is private, keep it secret", posture="strict"
    )
    relaxed_score, relaxed_reasons = hub_social_room._redaction_score(
        "this is private, keep it secret", posture="relaxed"
    )
    assert strict_score > 0.0
    assert "mentions_private" in strict_reasons
    assert relaxed_score == 0.0
    assert relaxed_reasons == []

    email_strict, _ = hub_social_room._redaction_score("reach me at a@b.com", posture="strict")
    email_relaxed, _ = hub_social_room._redaction_score("reach me at a@b.com", posture="relaxed")
    assert email_strict == email_relaxed
    assert email_relaxed > 0.0


def test_build_social_redaction_relaxed_scores_lower_than_strict_for_same_text() -> None:
    strict = hub_social_room.build_social_redaction(
        prompt="this is a private secret", response="okay", posture="strict"
    )
    relaxed = hub_social_room.build_social_redaction(
        prompt="this is a private secret", response="okay", posture="relaxed"
    )
    assert strict.overall_score > relaxed.overall_score
    assert relaxed.overall_score == 0.0
```

- [ ] **Step 1.2: Run the tests to verify they fail**

Run: `cd /mnt/scripts/Orion-Sapienform && PYTHONPATH=. ./orion_dev/bin/python -m pytest services/orion-hub/tests/test_social_room.py -k "redaction_score_relaxed or build_social_redaction_relaxed" -q --tb=short`
Expected: FAIL — `TypeError: _redaction_score() got an unexpected keyword argument 'posture'`

- [ ] **Step 1.3: Add the posture type and thread it through the two functions**

In `services/orion-hub/scripts/social_room.py`, change the typing import on line 5:

```python
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple
```

Add a posture type and default constant right after the existing `_BLOCKED_MEMORY_RE` definition (line 68):

```python
_BLOCKED_MEMORY_RE = re.compile(r"\b(sealed|private|password|secret|ssn|mirror|journal)\b", re.IGNORECASE)
SocialRedactionPosture = Literal["strict", "relaxed"]
DEFAULT_SOCIAL_REDACTION_POSTURE: SocialRedactionPosture = "strict"


def _normalize_redaction_posture(value: Any) -> SocialRedactionPosture:
    candidate = str(value or "").strip().lower()
    return candidate if candidate in ("strict", "relaxed") else DEFAULT_SOCIAL_REDACTION_POSTURE
```

Replace `_redaction_score` (lines 93-117):

```python
def _redaction_score(text: str | None, *, posture: str = DEFAULT_SOCIAL_REDACTION_POSTURE) -> tuple[float, list[str]]:
    raw = str(text or "")
    score = 0.0
    reasons: list[str] = []
    if _EMAIL_RE.search(raw):
        score += 0.45
        reasons.append("contains_email")
    if _PHONE_RE.search(raw):
        score += 0.35
        reasons.append("contains_phone")
    if _LONG_DIGIT_RE.search(raw):
        score += 0.30
        reasons.append("contains_long_numeric_token")
    if _normalize_redaction_posture(posture) != "relaxed":
        lowered = raw.lower()
        for needle, weight in (
            ("address", 0.20),
            ("password", 0.50),
            ("ssn", 0.60),
            ("secret", 0.20),
            ("private", 0.15),
        ):
            if needle in lowered:
                score += weight
                reasons.append(f"mentions_{needle}")
    return min(score, 1.0), reasons
```

Replace `build_social_redaction` (lines 120-139):

```python
def build_social_redaction(
    *,
    prompt: str,
    response: str,
    memory_digest: str | None = None,
    posture: str = DEFAULT_SOCIAL_REDACTION_POSTURE,
) -> SocialRedactionScoreV1:
    posture = _normalize_redaction_posture(posture)
    prompt_score, prompt_reasons = _redaction_score(prompt, posture=posture)
    response_score, response_reasons = _redaction_score(response, posture=posture)
    memory_score, memory_reasons = _redaction_score(memory_digest, posture=posture)
    overall = max(prompt_score, response_score, memory_score)
    if overall >= 0.7:
        level = "high"
    elif overall >= 0.35:
        level = "medium"
    else:
        level = "low"
    return SocialRedactionScoreV1(
        prompt_score=prompt_score,
        response_score=response_score,
        memory_score=memory_score,
        overall_score=overall,
        recall_safe=overall < 0.7,
        redaction_level=level,
        reasons=list(dict.fromkeys(prompt_reasons + response_reasons + memory_reasons)),
    )
```

- [ ] **Step 1.4: Run the tests to verify they pass**

Run: `cd /mnt/scripts/Orion-Sapienform && PYTHONPATH=. ./orion_dev/bin/python -m pytest services/orion-hub/tests/test_social_room.py -k "redaction_score_relaxed or build_social_redaction_relaxed" -q --tb=short`
Expected: PASS

- [ ] **Step 1.5: Commit**

```bash
cd /mnt/scripts/Orion-Sapienform
git add services/orion-hub/scripts/social_room.py services/orion-hub/tests/test_social_room.py
git commit -m "feat(hub-social-room): add posture-aware redaction scoring"
```

---

### Task 2 — Thread posture into blocked-memory checks, scope defaults, and skill selection

**Files:**
- Modify: `services/orion-hub/scripts/social_room.py:210-222` (`_blocked_memory_text`, `_dialogue_scope_hint`), `:269-...` (`build_social_artifact_dialogue`), `:590-618` (`_skill_from_heuristics`), `:621-677` (`select_social_room_skill`)
- Test: `services/orion-hub/tests/test_social_room.py`

- [ ] **Step 2.1: Write the failing tests**

Append to `services/orion-hub/tests/test_social_room.py`:

```python
def test_dialogue_scope_hint_defaults_narrower_under_strict_than_relaxed() -> None:
    assert hub_social_room._dialogue_scope_hint("remember this for later", posture="strict") == "session_only"
    assert hub_social_room._dialogue_scope_hint("remember this for later", posture="relaxed") == "peer_local"


def test_build_social_artifact_dialogue_strict_blocks_private_wording() -> None:
    _, _, confirmation, result, _ = hub_social_room.build_social_artifact_dialogue(
        payload={},
        prompt="keep this journal entry between us",
        posture="strict",
    )
    assert confirmation is not None
    assert confirmation.decision_state == "declined"
    assert result is not None


def test_build_social_artifact_dialogue_relaxed_allows_private_wording() -> None:
    _, _, confirmation, result, _ = hub_social_room.build_social_artifact_dialogue(
        payload={},
        prompt="keep this journal entry between us",
        posture="relaxed",
    )
    assert confirmation is None
    assert result is not None
    assert "declined" not in (result.summary or "").lower()


def test_select_social_room_skill_forwards_posture_to_artifact_dialogue() -> None:
    allowlist = hub_social_room.resolve_social_skill_allowlist()
    strict_selection, strict_result, _ = hub_social_room.select_social_room_skill(
        payload={},
        prompt="keep this journal entry between us",
        skills_enabled=True,
        allowlist=allowlist,
        posture="strict",
    )
    relaxed_selection, relaxed_result, _ = hub_social_room.select_social_room_skill(
        payload={},
        prompt="keep this journal entry between us",
        skills_enabled=True,
        allowlist=allowlist,
        posture="relaxed",
    )
    assert strict_selection.selected_skill == "social_artifact_dialogue"
    assert relaxed_selection.selected_skill == "social_artifact_dialogue"
    assert strict_result is not None
    assert relaxed_result is not None
    assert "wouldn" in (strict_result.summary or "").lower()
    assert "wouldn" not in (relaxed_result.summary or "").lower()
```

- [ ] **Step 2.2: Run the tests to verify they fail**

Run: `cd /mnt/scripts/Orion-Sapienform && PYTHONPATH=. ./orion_dev/bin/python -m pytest services/orion-hub/tests/test_social_room.py -k "dialogue_scope_hint_defaults or build_social_artifact_dialogue_strict or build_social_artifact_dialogue_relaxed or forwards_posture" -q --tb=short`
Expected: FAIL — `TypeError: _dialogue_scope_hint() got an unexpected keyword argument 'posture'`

- [ ] **Step 2.3: Update `_blocked_memory_text` and `_dialogue_scope_hint`**

Replace lines 210-222 of `services/orion-hub/scripts/social_room.py`:

```python
def _blocked_memory_text(text: str | None, *, posture: str = DEFAULT_SOCIAL_REDACTION_POSTURE) -> bool:
    if _normalize_redaction_posture(posture) == "relaxed":
        return False
    return bool(_BLOCKED_MEMORY_RE.search(str(text or "")))


def _dialogue_scope_hint(prompt: str, *, posture: str = DEFAULT_SOCIAL_REDACTION_POSTURE) -> str:
    lowered = prompt.lower()
    if any(needle in lowered for needle in ("room-local", "room local", "in this room", "for the room", "here in the room")):
        return "room_local"
    if any(needle in lowered for needle in ("peer-local", "peer local", "between us", "with me", "about me")):
        return "peer_local"
    if "session-only" in lowered or "session only" in lowered:
        return "session_only"
    return "peer_local" if _normalize_redaction_posture(posture) == "relaxed" else "session_only"
```

- [ ] **Step 2.4: Thread `posture` through `build_social_artifact_dialogue`**

In `services/orion-hub/scripts/social_room.py`, update the signature (was lines 269-273):

```python
def build_social_artifact_dialogue(
    *,
    payload: Dict[str, Any],
    prompt: str,
    posture: str = DEFAULT_SOCIAL_REDACTION_POSTURE,
) -> Tuple[Optional[SocialArtifactProposalV1], Optional[SocialArtifactRevisionV1], Optional[SocialArtifactConfirmationV1], Optional[SocialSkillResultV1], str]:
    posture = _normalize_redaction_posture(posture)
    lowered = prompt.lower()
    if not _MEMORY_DIALOGUE_RE.search(prompt):
        pending = _proposal_source(payload)
        if pending is None:
            return None, None, None, None, "no shared-artifact dialogue cue matched"
    if _blocked_memory_text(prompt, posture=posture):
```

The rest of the function body (from the `confirmation = SocialArtifactConfirmationV1(...)` block onward) is unchanged, **except** every remaining call to `_dialogue_scope_hint(prompt)` inside this function must become `_dialogue_scope_hint(prompt, posture=posture)`. Use Grep on the function body for `_dialogue_scope_hint(prompt)` and update each occurrence to pass `posture=posture`.

- [ ] **Step 2.5: Thread `posture` through the skill-selection layer**

Replace `_skill_from_heuristics` (lines 590-618):

```python
def _skill_from_heuristics(
    payload: Dict[str, Any],
    prompt: str,
    allowlist: List[SocialSkillName],
    *,
    posture: str = DEFAULT_SOCIAL_REDACTION_POSTURE,
) -> tuple[SocialSkillName | None, str]:
    lowered = prompt.lower()
    policy = payload.get("social_turn_policy") or {}
    open_threads = payload.get("social_room_continuity", {}).get("open_threads") or []

    if "social_artifact_dialogue" in allowlist:
        proposal, revision, confirmation, result, reason = build_social_artifact_dialogue(
            payload=payload, prompt=prompt, posture=posture
        )
        if any(item is not None for item in (proposal, revision, confirmation)) and result is not None:
            return "social_artifact_dialogue", reason
    if "social_self_ground" in allowlist and any(needle in lowered for needle in ("who are you", "what are you", "remind me who you are")):
        return "social_self_ground", "explicit self/identity request"
    if "social_summarize_thread" in allowlist and any(needle in lowered for needle in ("summarize", "summary", "recap", "what were we just talking about", "catch me up")):
        return "social_summarize_thread", "explicit request to summarize the room/thread"
    if "social_safe_recall" in allowlist and any(needle in lowered for needle in ("remember", "recall", "what do you know about me", "what do you remember")):
        return "social_safe_recall", "explicit safe-recall request"
    if "social_room_reflection" in allowlist and any(needle in lowered for needle in ("what do you notice about this room", "what's happening in this room", "reflect on this room")):
        return "social_room_reflection", "explicit room-dynamic reflection request"
    if "social_exit_or_pause" in allowlist and any(
        needle in lowered
        for needle in ("let's pause", "let’s pause", "pause here", "let's stop", "step back", "no need to answer", "give us space")
    ):
        return "social_exit_or_pause", "explicit pause / disengagement cue"
    if "social_followup_question" in allowlist and (
        any(needle in lowered for needle in ("what should we ask", "where should we go next", "what's the next question"))
        or (str(policy.get("decision") or "") == "ask_follow_up" and open_threads)
        or (float(policy.get("novelty_score") or 1.0) <= 0.2 and open_threads)
    ):
        return "social_followup_question", "open-thread continuation would benefit from one grounded follow-up question"
    return None, "no narrow social-skill trigger matched"
```

Replace `select_social_room_skill` (lines 621-677):

```python
def select_social_room_skill(
    *,
    payload: Dict[str, Any],
    prompt: str,
    skills_enabled: bool,
    allowlist: List[SocialSkillName],
    posture: str = DEFAULT_SOCIAL_REDACTION_POSTURE,
) -> tuple[SocialSkillSelectionV1, SocialSkillResultV1 | None, SocialSkillRequestV1]:
    posture = _normalize_redaction_posture(posture)
    request = _make_skill_request(payload, prompt=prompt, allowlist=allowlist)
    if not skills_enabled:
        selection = SocialSkillSelectionV1(
            considered_skills=allowlist,
            selection_reason="social skill surfacing disabled",
            suppressed_reason="skills_disabled",
            request_id=request.request_id,
        )
        logger.debug("social_skill_suppressed reason=%s", selection.suppressed_reason)
        return selection, None, request
    if not allowlist:
        selection = SocialSkillSelectionV1(
            considered_skills=[],
            selection_reason="social skill surfacing has an empty allowlist",
            suppressed_reason="empty_allowlist",
            request_id=request.request_id,
        )
        logger.debug("social_skill_suppressed reason=%s", selection.suppressed_reason)
        return selection, None, request

    skill_name, reason = _skill_from_heuristics(payload, prompt, allowlist, posture=posture)
    if skill_name is None:
        selection = SocialSkillSelectionV1(
            considered_skills=allowlist,
            selection_reason=reason,
            suppressed_reason="no_skill_needed",
            request_id=request.request_id,
        )
        logger.debug("social_skill_none reason=%s", reason)
        return selection, None, request

    builder_map = {
        "social_summarize_thread": _summarize_thread,
        "social_safe_recall": _safe_recall,
        "social_self_ground": _self_ground,
        "social_followup_question": _followup_question,
        "social_room_reflection": _room_reflection,
        "social_exit_or_pause": _exit_or_pause,
        "social_artifact_dialogue": lambda payload, request: build_social_artifact_dialogue(
            payload=payload, prompt=request.prompt, posture=posture
        )[3],  # type: ignore[return-value]
    }
    result = builder_map[skill_name](payload, request)
    selection = SocialSkillSelectionV1(
        considered_skills=allowlist,
        selected_skill=skill_name,
        used=True,
        selection_reason=reason,
        request_id=request.request_id,
    )
    logger.info("social_skill_selected skill=%s reason=%s", skill_name, reason)
    return selection, result, request
```

- [ ] **Step 2.6: Run the tests to verify they pass**

Run: `cd /mnt/scripts/Orion-Sapienform && PYTHONPATH=. ./orion_dev/bin/python -m pytest services/orion-hub/tests/test_social_room.py -k "dialogue_scope_hint_defaults or build_social_artifact_dialogue_strict or build_social_artifact_dialogue_relaxed or forwards_posture" -q --tb=short`
Expected: PASS

- [ ] **Step 2.7: Run the full existing social_room test file to check for regressions**

Run: `cd /mnt/scripts/Orion-Sapienform && PYTHONPATH=. ./orion_dev/bin/python -m pytest services/orion-hub/tests/test_social_room.py -q --tb=short`
Expected: PASS (all prior tests plus new ones)

- [ ] **Step 2.8: Commit**

```bash
cd /mnt/scripts/Orion-Sapienform
git add services/orion-hub/scripts/social_room.py services/orion-hub/tests/test_social_room.py
git commit -m "feat(hub-social-room): thread redaction posture through artifact dialogue and skill selection"
```

---

### Task 3 — Flow posture through `social_room_client_meta` and `build_social_room_turn` (write-back path)

**Files:**
- Modify: `services/orion-hub/scripts/social_room.py:680-745` (`social_room_client_meta`), `:815-853` (`build_social_room_turn`)
- Test: `services/orion-hub/tests/test_social_room.py`

- [ ] **Step 3.1: Write the failing tests**

Append to `services/orion-hub/tests/test_social_room.py`:

```python
def test_social_room_client_meta_includes_redaction_posture_default() -> None:
    meta = hub_social_room.social_room_client_meta(
        payload={"chat_profile": "social_room"},
        route_debug={},
        trace_verb=None,
        memory_digest=None,
    )
    assert meta["social_redaction_posture"] == "strict"


def test_social_room_client_meta_prefers_route_debug_posture_over_payload() -> None:
    meta = hub_social_room.social_room_client_meta(
        payload={"chat_profile": "social_room", "social_redaction_posture": "strict"},
        route_debug={"social_redaction_posture": "relaxed"},
        trace_verb=None,
        memory_digest=None,
    )
    assert meta["social_redaction_posture"] == "relaxed"


def test_build_social_room_turn_uses_posture_from_client_meta() -> None:
    relaxed_turn = hub_social_room.build_social_room_turn(
        prompt="this is private, keep it secret",
        response="okay",
        session_id="s1",
        correlation_id="c1",
        user_id="juniper",
        source="hub_ws",
        recall_profile=None,
        trace_verb=None,
        client_meta={"social_redaction_posture": "relaxed"},
        memory_digest=None,
    )
    strict_turn = hub_social_room.build_social_room_turn(
        prompt="this is private, keep it secret",
        response="okay",
        session_id="s1",
        correlation_id="c1",
        user_id="juniper",
        source="hub_ws",
        recall_profile=None,
        trace_verb=None,
        client_meta={"social_redaction_posture": "strict"},
        memory_digest=None,
    )
    assert relaxed_turn.redaction.overall_score == 0.0
    assert strict_turn.redaction.overall_score > 0.0


def test_social_room_client_meta_includes_hub_direct_mode_tag() -> None:
    meta = hub_social_room.social_room_client_meta(
        payload={"chat_profile": "social_room", "social_room_mode": "hub_direct"},
        route_debug={},
        trace_verb=None,
        memory_digest=None,
    )
    assert meta["social_room_mode"] == "hub_direct"
```

- [ ] **Step 3.2: Run the tests to verify they fail**

Run: `cd /mnt/scripts/Orion-Sapienform && PYTHONPATH=. ./orion_dev/bin/python -m pytest services/orion-hub/tests/test_social_room.py -k "social_room_client_meta_includes_redaction or social_room_client_meta_prefers_route_debug or build_social_room_turn_uses_posture" -q --tb=short`
Expected: FAIL — `KeyError: 'social_redaction_posture'`

- [ ] **Step 3.3: Update `social_room_client_meta`**

In `services/orion-hub/scripts/social_room.py`, inside `social_room_client_meta` (after the existing `style_adaptation = ...` line, i.e. after line 694), add:

```python
    redaction_posture = _normalize_redaction_posture(
        route_debug.get("social_redaction_posture") or payload.get("social_redaction_posture")
    )
    social_room_mode = str(
        route_debug.get("social_room_mode") or payload.get("social_room_mode") or ""
    ).strip().lower() or None
```

Then add `"social_redaction_posture": redaction_posture,` and `"social_room_mode": social_room_mode,` as entries of the returned dict, right after `"chat_profile": SOCIAL_ROOM_PROFILE,` (line 714):

```python
    return {
        "chat_profile": SOCIAL_ROOM_PROFILE,
        "social_redaction_posture": redaction_posture,
        "social_room_mode": social_room_mode,
        "social_grounding_state": grounding.model_dump(mode="json"),
```

- [ ] **Step 3.4: Update `build_social_room_turn`**

Replace lines 828-833 of `services/orion-hub/scripts/social_room.py`:

```python
    social_meta = dict(client_meta or {})
    grounding_state = SocialGroundingStateV1.model_validate(
        social_meta.get("social_grounding_state") or {}
    )
    concept_evidence = build_social_concept_evidence(social_meta.get("social_concept_evidence"))
    posture = _normalize_redaction_posture(social_meta.get("social_redaction_posture"))
    redaction = build_social_redaction(prompt=prompt, response=response, memory_digest=memory_digest, posture=posture)
```

- [ ] **Step 3.5: Run the tests to verify they pass**

Run: `cd /mnt/scripts/Orion-Sapienform && PYTHONPATH=. ./orion_dev/bin/python -m pytest services/orion-hub/tests/test_social_room.py -q --tb=short`
Expected: PASS (all tests, including the ones from Tasks 1-2)

- [ ] **Step 3.6: Commit**

```bash
cd /mnt/scripts/Orion-Sapienform
git add services/orion-hub/scripts/social_room.py services/orion-hub/tests/test_social_room.py
git commit -m "feat(hub-social-room): flow redaction posture through client_meta into the write-back turn"
```

---

### Task 4 — Hub setting + env var for the default posture

**Files:**
- Modify: `services/orion-hub/app/settings.py:325` (near `HUB_SOCIAL_STYLE_CONFIDENCE_FLOOR`)
- Modify: `services/orion-hub/.env_example:273` (near `HUB_SOCIAL_STYLE_CONFIDENCE_FLOOR`)

- [ ] **Step 4.1: Add the setting field**

In `services/orion-hub/app/settings.py`, change:

```python
    HUB_SOCIAL_STYLE_CONFIDENCE_FLOOR: float = Field(default=0.35, alias="HUB_SOCIAL_STYLE_CONFIDENCE_FLOOR")
```

to:

```python
    HUB_SOCIAL_STYLE_CONFIDENCE_FLOOR: float = Field(default=0.35, alias="HUB_SOCIAL_STYLE_CONFIDENCE_FLOOR")
    HUB_SOCIAL_ROOM_REDACTION_POSTURE: str = Field(default="strict", alias="HUB_SOCIAL_ROOM_REDACTION_POSTURE")
```

- [ ] **Step 4.2: Add the `.env_example` entry**

In `services/orion-hub/.env_example`, change:

```
HUB_SOCIAL_STYLE_CONFIDENCE_FLOOR=0.35
```

to:

```
HUB_SOCIAL_STYLE_CONFIDENCE_FLOOR=0.35
# "strict" keeps the private/secret/password/ssn keyword blocklist active for shared-artifact
# dialogue; "relaxed" is intended for the trusted Hub-direct 1:1 toggle and skips those keyword
# hits (PII regexes for email/phone/long-digit numbers stay on regardless of posture).
HUB_SOCIAL_ROOM_REDACTION_POSTURE=strict
```

- [ ] **Step 4.3: Sync local `.env`**

Run: `cd /mnt/scripts/Orion-Sapienform && python scripts/sync_local_env_from_example.py`
Expected: exits 0; `services/orion-hub/.env` now contains `HUB_SOCIAL_ROOM_REDACTION_POSTURE=strict` if it didn't already.

- [ ] **Step 4.4: Commit**

```bash
cd /mnt/scripts/Orion-Sapienform
git add services/orion-hub/app/settings.py services/orion-hub/.env_example
git commit -m "feat(hub): add HUB_SOCIAL_ROOM_REDACTION_POSTURE setting"
```

---

### Task 5 — Resolve and wire the posture in `cortex_request_builder.py`

**Files:**
- Modify: `services/orion-hub/scripts/cortex_request_builder.py:102-107` (add getter), `:594-606` (call sites), `:626-670` (metadata block), `:731-739` (debug block)
- Test: `services/orion-hub/tests/test_cortex_request_builder.py`

- [ ] **Step 5.1: Write the failing tests**

Append to `services/orion-hub/tests/test_cortex_request_builder.py`:

```python
def test_hub_social_redaction_posture_defaults_to_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HUB_SOCIAL_ROOM_REDACTION_POSTURE", raising=False)
    assert hub_builder._hub_social_redaction_posture({}) == "strict"


def test_hub_social_redaction_posture_reads_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HUB_SOCIAL_ROOM_REDACTION_POSTURE", "relaxed")
    assert hub_builder._hub_social_redaction_posture({}) == "relaxed"


def test_hub_social_redaction_posture_payload_override_wins_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HUB_SOCIAL_ROOM_REDACTION_POSTURE", "strict")
    assert hub_builder._hub_social_redaction_posture({"social_redaction_posture": "relaxed"}) == "relaxed"


def test_hub_social_redaction_posture_rejects_invalid_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HUB_SOCIAL_ROOM_REDACTION_POSTURE", "yolo")
    assert hub_builder._hub_social_redaction_posture({}) == "strict"


def test_build_chat_request_social_room_metadata_includes_redaction_posture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HUB_SOCIAL_ROOM_REDACTION_POSTURE", raising=False)
    req, debug, _ = hub_builder.build_chat_request(
        payload={
            "chat_profile": "social_room",
            "social_redaction_posture": "relaxed",
            "messages": [{"role": "user", "content": "hello"}],
        },
        session_id="sid-social",
        user_id="juniper",
        trace_id="trace-social",
        default_mode="brain",
        auto_default_enabled=False,
        source_label="hub_ws",
        prompt="hello",
    )
    assert req.metadata["social_redaction_posture"] == "relaxed"
    assert debug["social_redaction_posture"] == "relaxed"
```

- [ ] **Step 5.2: Run the tests to verify they fail**

Run: `cd /mnt/scripts/Orion-Sapienform && PYTHONPATH=. ./orion_dev/bin/python -m pytest services/orion-hub/tests/test_cortex_request_builder.py -k "hub_social_redaction_posture or social_room_metadata_includes_redaction" -q --tb=short`
Expected: FAIL — `AttributeError: module 'hub_cortex_request_builder' has no attribute '_hub_social_redaction_posture'`

- [ ] **Step 5.3: Add the getter**

In `services/orion-hub/scripts/cortex_request_builder.py`, after `_hub_social_style_confidence_floor` (lines 102-107), add:

```python
def _hub_social_redaction_posture(payload: Dict[str, Any]) -> str:
    override = str((payload or {}).get("social_redaction_posture") or "").strip().lower()
    if override in ("strict", "relaxed"):
        return override
    env_value = os.getenv("HUB_SOCIAL_ROOM_REDACTION_POSTURE", "strict").strip().lower()
    return env_value if env_value in ("strict", "relaxed") else "strict"
```

- [ ] **Step 5.4: Wire the resolved posture into the call sites and metadata**

In `services/orion-hub/scripts/cortex_request_builder.py`, change lines 594-606 from:

```python
        artifact_proposal, artifact_revision, artifact_confirmation, _, _ = build_social_artifact_dialogue(
            payload=payload,
            prompt=prompt,
        )
        skills_enabled = _hub_social_skills_enabled()
        if isinstance(skill_cfg, dict) and skill_cfg.get("enabled") is not None:
            skills_enabled = _normalize_flag(skill_cfg.get("enabled"), default=_hub_social_skills_enabled())
        selection, skill_result, skill_request = select_social_room_skill(
            payload=payload,
            prompt=prompt,
            skills_enabled=skills_enabled,
            allowlist=skill_allowlist,
        )
```

to:

```python
        redaction_posture = _hub_social_redaction_posture(payload)
        artifact_proposal, artifact_revision, artifact_confirmation, _, _ = build_social_artifact_dialogue(
            payload=payload,
            prompt=prompt,
            posture=redaction_posture,
        )
        skills_enabled = _hub_social_skills_enabled()
        if isinstance(skill_cfg, dict) and skill_cfg.get("enabled") is not None:
            skills_enabled = _normalize_flag(skill_cfg.get("enabled"), default=_hub_social_skills_enabled())
        selection, skill_result, skill_request = select_social_room_skill(
            payload=payload,
            prompt=prompt,
            skills_enabled=skills_enabled,
            allowlist=skill_allowlist,
            posture=redaction_posture,
        )
```

Then add `"social_redaction_posture": redaction_posture,` to the `metadata.update({...})` block, right after `"chat_profile": SOCIAL_ROOM_PROFILE,` (line 628):

```python
        metadata.update(
            {
                "chat_profile": SOCIAL_ROOM_PROFILE,
                "social_redaction_posture": redaction_posture,
                "social_grounding_state": build_social_grounding_state(payload=payload).model_dump(mode="json"),
```

- [ ] **Step 5.5: Surface it in the `debug` dict**

Still in `services/orion-hub/scripts/cortex_request_builder.py`, inside the `if social_room:` block that populates extra `debug` keys (the block starting at line 731 with `debug["social_skill_allowlist"] = ...`), add:

```python
    if social_room:
        debug["social_redaction_posture"] = metadata.get("social_redaction_posture")
        debug["social_room_mode"] = payload.get("social_room_mode")
        debug["social_skill_allowlist"] = metadata.get("social_skill_request", {}).get("allowlist") or []
```

- [ ] **Step 5.6: Run the tests to verify they pass**

Run: `cd /mnt/scripts/Orion-Sapienform && PYTHONPATH=. ./orion_dev/bin/python -m pytest services/orion-hub/tests/test_cortex_request_builder.py -q --tb=short`
Expected: PASS (all tests, including prior coverage)

- [ ] **Step 5.7: Commit**

```bash
cd /mnt/scripts/Orion-Sapienform
git add services/orion-hub/scripts/cortex_request_builder.py services/orion-hub/tests/test_cortex_request_builder.py
git commit -m "feat(hub): resolve and surface social_redaction_posture in the cortex request builder"
```

---

### Task 6 — Hub-direct identity + `orion-social-memory` summary-merge helpers

**Files:**
- Modify: `services/orion-hub/scripts/social_room.py` (add near the top-level constants and after `is_social_room_payload`)
- Test: `services/orion-hub/tests/test_social_room.py`

These are pure functions with no I/O, so they can be fully unit tested without mocking HTTP calls. The actual `orion-social-memory` fetch happens in Task 7.

- [ ] **Step 6.1: Write the failing tests**

Append to `services/orion-hub/tests/test_social_room.py`:

```python
def test_hub_direct_room_identity_uses_a_stable_room_id_not_session_scoped() -> None:
    identity_a = hub_social_room.hub_direct_room_identity("juniper")
    identity_b = hub_social_room.hub_direct_room_identity("juniper")
    assert identity_a == identity_b
    assert identity_a["external_room"] == {"platform": "hub", "room_id": "hub-direct"}
    assert identity_a["external_participant"]["participant_id"] == "juniper"


def test_hub_direct_room_identity_defaults_participant_id_when_missing() -> None:
    identity = hub_social_room.hub_direct_room_identity(None)
    assert identity["external_participant"]["participant_id"] == "juniper"
    assert identity["external_participant"]["participant_kind"] == "human"


def test_apply_social_memory_summary_to_payload_merges_known_fields_only() -> None:
    payload = {"chat_profile": "social_room"}
    summary = {
        "participant": {"participant_id": "juniper"},
        "room": {"room_id": "hub-direct"},
        "stance": {"warmth": 0.8},
        "peer_style": None,
        "room_ritual": None,
        "context_window": {"thread_key": "t1"},
        "context_selection_decision": None,
        "context_candidates": [],
        "episode_snapshot": {"summary": "not part of the mapped fields"},
    }
    merged = hub_social_room.apply_social_memory_summary_to_payload(payload, summary)
    assert merged["social_peer_continuity"] == {"participant_id": "juniper"}
    assert merged["social_room_continuity"] == {"room_id": "hub-direct"}
    assert merged["social_stance_snapshot"] == {"warmth": 0.8}
    assert merged["social_context_window"] == {"thread_key": "t1"}
    assert "social_peer_style_hint" not in merged
    assert "social_episode_snapshot" not in merged
    assert merged["chat_profile"] == "social_room"


def test_apply_social_memory_summary_to_payload_noop_for_missing_summary() -> None:
    payload = {"chat_profile": "social_room"}
    assert hub_social_room.apply_social_memory_summary_to_payload(payload, None) == payload
```

- [ ] **Step 6.2: Run the tests to verify they fail**

Run: `cd /mnt/scripts/Orion-Sapienform && PYTHONPATH=. ./orion_dev/bin/python -m pytest services/orion-hub/tests/test_social_room.py -k "hub_direct_room_identity or apply_social_memory_summary_to_payload" -q --tb=short`
Expected: FAIL — `AttributeError: module 'hub_social_room' has no attribute 'hub_direct_room_identity'`

- [ ] **Step 6.3: Add the helpers**

In `services/orion-hub/scripts/social_room.py`, directly below the `is_social_room_payload` function (after line 81), add:

```python
HUB_DIRECT_ROOM_PLATFORM = "hub"
HUB_DIRECT_ROOM_ID = "hub-direct"


def hub_direct_room_identity(user_id: str | None) -> Dict[str, Any]:
    """Stable platform/room/participant identity for Hub's own social_room toggle.

    The room id is a fixed constant (not session-scoped) so peer/room continuity in
    orion-social-memory persists across toggle-on sessions instead of resetting every time.
    """
    participant_id = str(user_id or "juniper").strip() or "juniper"
    return {
        "external_room": {
            "platform": HUB_DIRECT_ROOM_PLATFORM,
            "room_id": HUB_DIRECT_ROOM_ID,
        },
        "external_participant": {
            "participant_id": participant_id,
            "participant_name": "Juniper",
            "participant_kind": "human",
        },
    }


_SOCIAL_MEMORY_SUMMARY_KEY_MAP = {
    "social_peer_continuity": "participant",
    "social_room_continuity": "room",
    "social_stance_snapshot": "stance",
    "social_peer_style_hint": "peer_style",
    "social_room_ritual_summary": "room_ritual",
    "social_context_window": "context_window",
    "social_context_selection_decision": "context_selection_decision",
    "social_context_candidates": "context_candidates",
}


def apply_social_memory_summary_to_payload(payload: Dict[str, Any], summary: Dict[str, Any] | None) -> Dict[str, Any]:
    """Merge an orion-social-memory GET /summary response into an outgoing chat payload.

    Mirrors what orion-social-room-bridge already does in its hub_payload.update() block
    (services/orion-social-room-bridge/app/service.py), so Hub-direct toggle turns get the
    same continuity/style/ritual context as bridge-routed external-room turns.
    """
    merged = dict(payload)
    if not isinstance(summary, dict):
        return merged
    for payload_key, summary_key in _SOCIAL_MEMORY_SUMMARY_KEY_MAP.items():
        value = summary.get(summary_key)
        if value:
            merged[payload_key] = value
    return merged
```

- [ ] **Step 6.4: Run the tests to verify they pass**

Run: `cd /mnt/scripts/Orion-Sapienform && PYTHONPATH=. ./orion_dev/bin/python -m pytest services/orion-hub/tests/test_social_room.py -q --tb=short`
Expected: PASS (all tests)

- [ ] **Step 6.5: Commit**

```bash
cd /mnt/scripts/Orion-Sapienform
git add services/orion-hub/scripts/social_room.py services/orion-hub/tests/test_social_room.py
git commit -m "feat(hub-social-room): add Hub-direct identity and social-memory summary merge helpers"
```

---

### Task 7 — Wire the toggle into `websocket_handler.py`

**Files:**
- Modify: `services/orion-hub/scripts/websocket_handler.py:31` (import), add new helper after `_normalize_bool` (line 124), wire into `websocket_endpoint` (after line 727)

`orion-hub` has no existing websocket-level test harness (`services/orion-hub/tests/test_websocket*.py` does not exist), so this task is verified via the pure helpers already covered by Task 6's tests plus the manual live check in Task 9. Keep this function thin: resolve identity, fetch, merge — no new branching logic beyond what Task 6 already tests.

- [ ] **Step 7.1: Extend the import**

In `services/orion-hub/scripts/websocket_handler.py`, change line 31 from:

```python
from scripts.social_room import is_social_room_payload, social_room_client_meta
```

to:

```python
from scripts.social_room import (
    apply_social_memory_summary_to_payload,
    hub_direct_room_identity,
    is_social_room_payload,
    social_room_client_meta,
)
```

- [ ] **Step 7.2: Add the enrichment helper**

In `services/orion-hub/scripts/websocket_handler.py`, directly after `_normalize_bool` (after line 124, before `_log_hub_route_decision`), add:

```python
async def _apply_hub_direct_social_room_mode(data: Dict[str, Any]) -> Dict[str, Any]:
    """Populate chat_profile/continuity fields for the Hub UI's own social_room toggle.

    Hub-local only: does NOT call orion-social-room-bridge or CallSyne. Mirrors the
    bridge's social-memory prefetch pattern (identity + GET /summary) but stops at
    build_chat_request() — reply delivery stays in the Hub websocket UI.

    The write-back half (publish_social_room_turn -> orion-sql-writer ->
    orion-social-memory) fires automatically off chat_profile=="social_room".
    """
    if str(data.get("social_room_mode") or "").strip().lower() != "hub_direct":
        return data
    identity = hub_direct_room_identity(data.get("user_id"))
    enriched = dict(data)
    enriched["chat_profile"] = "social_room"
    enriched["social_room_mode"] = "hub_direct"  # audit tag stamped on every Hub-direct turn
    enriched.setdefault("external_room", identity["external_room"])
    enriched.setdefault("external_participant", identity["external_participant"])
    posture = str(data.get("social_redaction_posture") or "").strip().lower()
    if posture in ("strict", "relaxed"):
        enriched["social_redaction_posture"] = posture
    try:
        from scripts.api_routes import _fetch_social_memory

        summary = await _fetch_social_memory(
            "/summary",
            {
                "platform": identity["external_room"]["platform"],
                "room_id": identity["external_room"]["room_id"],
                "participant_id": identity["external_participant"]["participant_id"],
            },
        )
    except Exception as exc:
        logger.warning("hub_direct_social_memory_fetch_failed error=%s", exc)
        return enriched
    return apply_social_memory_summary_to_payload(enriched, summary)
```

- [ ] **Step 7.3: Call it before `build_chat_request`**

In `services/orion-hub/scripts/websocket_handler.py`, change (was line 727-729):

```python
            data = dict(data)
            data = inject_session_presence(data, str(session_id or "anonymous"), presence_context_store)
            data["mutation_cognition_context"] = build_mutation_cognition_context()
```

to:

```python
            data = dict(data)
            data = await _apply_hub_direct_social_room_mode(data)
            data = inject_session_presence(data, str(session_id or "anonymous"), presence_context_store)
            data["mutation_cognition_context"] = build_mutation_cognition_context()
```

- [ ] **Step 7.4: Compile-check the changed file**

Run: `cd /mnt/scripts/Orion-Sapienform && ./orion_dev/bin/python -m compileall -q services/orion-hub/scripts/websocket_handler.py services/orion-hub/scripts/social_room.py services/orion-hub/scripts/cortex_request_builder.py`
Expected: exits 0, no output (compileall is silent on success)

- [ ] **Step 7.5: Run the full `orion-hub` social-room-adjacent test files to check for regressions**

Run: `cd /mnt/scripts/Orion-Sapienform && PYTHONPATH=. ./orion_dev/bin/python -m pytest services/orion-hub/tests/test_social_room.py services/orion-hub/tests/test_social_room_turn_publish.py services/orion-hub/tests/test_cortex_request_builder.py services/orion-hub/tests/test_social_inspection_api.py services/orion-hub/tests/test_hub_direct_social_room_isolation.py -q --tb=short`
Expected: PASS

- [ ] **Step 7.6: Commit**

```bash
cd /mnt/scripts/Orion-Sapienform
git add services/orion-hub/scripts/websocket_handler.py
git commit -m "feat(hub): wire the Hub-direct social_room toggle into the websocket chat path"
```

---

### Task 7b — CallSyne isolation guardrails

**Files:**
- Create: `services/orion-hub/tests/test_hub_direct_social_room_isolation.py`
- Modify: `services/orion-hub/scripts/cortex_request_builder.py` (debug block — already covered in Task 5 Step 5.5 if not yet applied)

Static regression tests only — no live CallSyne stack required. These guard against accidentally wiring Hub-direct chat through the bridge in future edits.

- [ ] **Step 7b.1: Write the isolation test file**

Create `services/orion-hub/tests/test_hub_direct_social_room_isolation.py`:

```python
from __future__ import annotations

import asyncio
import importlib
import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
HUB_ROOT = Path(__file__).resolve().parents[1]
for candidate in (str(REPO_ROOT), str(HUB_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

for key, value in {
    "CHANNEL_VOICE_TRANSCRIPT": "orion:voice:transcript",
    "CHANNEL_VOICE_LLM": "orion:voice:llm",
    "CHANNEL_VOICE_TTS": "orion:voice:tts",
    "CHANNEL_COLLAPSE_INTAKE": "orion:collapse:intake",
    "CHANNEL_COLLAPSE_TRIAGE": "orion:collapse:triage",
}.items():
    os.environ.setdefault(key, value)

SOCIAL_ROOM_PATH = HUB_ROOT / "scripts" / "social_room.py"
WS_PATH = HUB_ROOT / "scripts" / "websocket_handler.py"
SPEC = importlib.util.spec_from_file_location("hub_social_room", SOCIAL_ROOM_PATH)
assert SPEC and SPEC.loader
hub_social_room = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hub_social_room)


def test_hub_direct_identity_uses_hub_platform_not_callsyne() -> None:
    identity = hub_social_room.hub_direct_room_identity("juniper")
    assert identity["external_room"]["platform"] == "hub"
    assert identity["external_room"]["room_id"] == "hub-direct"
    assert "callsyne" not in identity["external_room"]["platform"]


def test_social_room_client_meta_stamps_hub_direct_mode() -> None:
    meta = hub_social_room.social_room_client_meta(
        payload={"chat_profile": "social_room", "social_room_mode": "hub_direct"},
        route_debug={},
        trace_verb=None,
        memory_digest=None,
    )
    assert meta["social_room_mode"] == "hub_direct"


def test_websocket_handler_source_has_no_bridge_or_callsyne_wiring() -> None:
    source = WS_PATH.read_text(encoding="utf-8").lower()
    forbidden = (
        "orion-social-room-bridge",
        "callsyne_client",
        "post_message",
        "webhooks/callsyne",
        "external.room.post",
    )
    for needle in forbidden:
        assert needle not in source, f"websocket_handler must not reference {needle!r}"


@pytest.mark.asyncio
async def test_apply_hub_direct_social_room_mode_only_calls_social_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ws_spec = importlib.util.spec_from_file_location("hub_ws_handler", WS_PATH)
    assert ws_spec and ws_spec.loader
    ws_module = importlib.util.module_from_spec(ws_spec)
    ws_spec.loader.exec_module(ws_module)

    calls: list[tuple[str, dict[str, object]]] = []

    async def fake_fetch(path: str, params: dict[str, object]) -> dict[str, object]:
        calls.append((path, params))
        return {"participant": {"participant_id": "juniper"}, "room": {"room_id": "hub-direct"}}

    api_routes = importlib.import_module("scripts.api_routes")
    monkeypatch.setattr(api_routes, "_fetch_social_memory", fake_fetch)

    result = await ws_module._apply_hub_direct_social_room_mode({"social_room_mode": "hub_direct"})
    assert result["social_room_mode"] == "hub_direct"
    assert result["chat_profile"] == "social_room"
    assert result["external_room"]["platform"] == "hub"
    assert calls == [
        (
            "/summary",
            {"platform": "hub", "room_id": "hub-direct", "participant_id": "juniper"},
        )
    ]


@pytest.mark.asyncio
async def test_apply_hub_direct_social_room_mode_noop_when_toggle_off() -> None:
    ws_spec = importlib.util.spec_from_file_location("hub_ws_handler", WS_PATH)
    assert ws_spec and ws_spec.loader
    ws_module = importlib.util.module_from_spec(ws_spec)
    ws_spec.loader.exec_module(ws_module)

    payload = {"mode": "brain", "text_input": "hello"}
    assert await ws_module._apply_hub_direct_social_room_mode(payload) == payload
```

- [ ] **Step 7b.2: Run the isolation tests (expect fail until Tasks 3, 6, 7 land)**

Run: `cd /mnt/scripts/Orion-Sapienform && PYTHONPATH=. ./orion_dev/bin/python -m pytest services/orion-hub/tests/test_hub_direct_social_room_isolation.py -q --tb=short`
Expected before Tasks 3/6/7: FAIL on missing helpers or missing `_apply_hub_direct_social_room_mode`
Expected after Tasks 3/6/7: PASS

- [ ] **Step 7b.3: Commit**

```bash
cd /mnt/scripts/Orion-Sapienform
git add services/orion-hub/tests/test_hub_direct_social_room_isolation.py
git commit -m "test(hub): add CallSyne isolation guardrails for Hub-direct social room"
```

---

### Task 8 — Front-end toggle + posture selector

**Files:**
- Modify: `services/orion-hub/templates/index.html` (chat composer toolbar, near lines 261-268)
- Modify: `services/orion-hub/static/js/app.js` (DOM ref block near lines 150-153; payload build near lines 10792-10812)

- [ ] **Step 8.1: Add the toggle + posture selector markup**

In `services/orion-hub/templates/index.html`, change:

```html
            <label class="flex items-center gap-1.5 text-xs text-gray-400" for="textToSpeechToggle">
              <input type="checkbox" id="textToSpeechToggle" class="h-3 w-3 text-red-600 rounded bg-gray-700 border-gray-600" />
              <span>TTS</span>
            </label>
            <button id="presenceOpenButton" type="button" class="rounded border border-indigo-500/40 bg-indigo-500/10 px-2 py-1 text-[10px] font-semibold text-indigo-200 hover:bg-indigo-500/20">Presence</button>
```

to:

```html
            <label class="flex items-center gap-1.5 text-xs text-gray-400" for="textToSpeechToggle">
              <input type="checkbox" id="textToSpeechToggle" class="h-3 w-3 text-red-600 rounded bg-gray-700 border-gray-600" />
              <span>TTS</span>
            </label>
            <label class="flex items-center gap-1.5 text-xs text-gray-400" for="socialRoomToggle" title="Route this session's chat through the chat_social_room verb (peer-mode Orion) instead of chat_general">
              <input type="checkbox" id="socialRoomToggle" class="h-3 w-3 text-pink-500 rounded bg-gray-700 border-gray-600" />
              <span>Social room</span>
            </label>
            <select
              id="socialRedactionPostureSelect"
              class="bg-gray-800 text-gray-200 rounded border border-gray-700 px-1.5 py-1 text-[10px]"
              aria-label="Social room redaction posture"
              title="Strict keeps the private/secret/password/ssn keyword blocklist active; Relaxed skips it for this trusted 1:1 toggle (PII regexes for email/phone/long numbers stay on either way)"
            >
              <option value="strict" selected>Strict</option>
              <option value="relaxed">Relaxed</option>
            </select>
            <button id="presenceOpenButton" type="button" class="rounded border border-indigo-500/40 bg-indigo-500/10 px-2 py-1 text-[10px] font-semibold text-indigo-200 hover:bg-indigo-500/20">Presence</button>
```

- [ ] **Step 8.2: Add the DOM refs**

In `services/orion-hub/static/js/app.js`, change:

```js
  const textToSpeechToggle = document.getElementById('textToSpeechToggle');
  const recallToggle = document.getElementById('recallToggle');
  const recallRequiredToggle = document.getElementById('recallRequiredToggle');
  const noWriteToggle = document.getElementById('noWriteToggle');
```

to:

```js
  const textToSpeechToggle = document.getElementById('textToSpeechToggle');
  const recallToggle = document.getElementById('recallToggle');
  const recallRequiredToggle = document.getElementById('recallRequiredToggle');
  const noWriteToggle = document.getElementById('noWriteToggle');
  const socialRoomToggle = document.getElementById('socialRoomToggle');
  const socialRedactionPostureSelect = document.getElementById('socialRedactionPostureSelect');
```

- [ ] **Step 8.3: Wire the toggle into the outgoing chat payload**

In `services/orion-hub/static/js/app.js`, change:

```js
       presence_context: presenceContext,
       surface_context: { surface: 'hub_desktop', input_modality: 'typed' },
       llm_route: effectiveRoute,
    };
```

to:

```js
       presence_context: presenceContext,
       surface_context: { surface: 'hub_desktop', input_modality: 'typed' },
       llm_route: effectiveRoute,
       social_room_mode: (socialRoomToggle && socialRoomToggle.checked) ? 'hub_direct' : null,
       social_redaction_posture: (socialRoomToggle && socialRoomToggle.checked && socialRedactionPostureSelect)
         ? socialRedactionPostureSelect.value
         : null,
    };
```

- [ ] **Step 8.4: Manual verification**

The toggle is intentionally per-session, in-memory JS state only (no `localStorage` persistence) — reloading the page or opening a new session resets it to off, matching decision #1 (per-session granularity). Verify by:

1. Restart `orion-hub` (or reload the page against a running dev instance).
2. Open the chat composer toolbar and confirm the "Social room" checkbox and "Strict/Relaxed" selector render next to the existing "TTS" toggle.
3. Reload the page — confirm the checkbox is unchecked again (per-session, not persisted).

- [ ] **Step 8.5: Commit**

```bash
cd /mnt/scripts/Orion-Sapienform
git add services/orion-hub/templates/index.html services/orion-hub/static/js/app.js
git commit -m "feat(hub-ui): add per-session social room toggle and redaction posture selector"
```

---

### Task 9 — Final verification

- [ ] **Step 9.1: Run the full targeted test set**

```bash
cd /mnt/scripts/Orion-Sapienform
PYTHONPATH=. ./orion_dev/bin/python -m pytest \
  services/orion-hub/tests/test_social_room.py \
  services/orion-hub/tests/test_social_room_turn_publish.py \
  services/orion-hub/tests/test_cortex_request_builder.py \
  services/orion-hub/tests/test_social_inspection_api.py \
  services/orion-hub/tests/test_hub_direct_social_room_isolation.py \
  -q --tb=short
```
Expected: all PASS, no regressions from prior tasks.

- [ ] **Step 9.2: Compile-check every touched Python file**

```bash
cd /mnt/scripts/Orion-Sapienform
./orion_dev/bin/python -m compileall -q \
  services/orion-hub/scripts/social_room.py \
  services/orion-hub/scripts/cortex_request_builder.py \
  services/orion-hub/scripts/websocket_handler.py \
  services/orion-hub/app/settings.py
```
Expected: exits 0.

- [ ] **Step 9.3: Live acceptance check (requires a running `orion-hub` + `orion-social-memory` stack)**

```bash
cd /mnt/scripts/Orion-Sapienform
docker compose --env-file .env --env-file services/orion-hub/.env -f services/orion-hub/docker-compose.yml up -d --build
```

Then in the Hub UI:

1. Leave "Social room" unchecked, send a normal message — confirm the reply and routing debug panel still show `chat_general` (unchanged baseline behavior).
2. Check "Social room" (posture left at "Strict"), send a message — confirm the routing debug panel now shows verb `chat_social_room` and `chat_profile: social_room`.
3. Check the Hub logs for `hub_direct_social_memory_fetch_failed` — it should **not** appear (confirms the `GET /summary` call to `orion-social-memory` succeeded).
4. Send a second message in the same toggled-on session mentioning something like "let's keep that as a peer-local note between us" — confirm no `hub_direct_social_memory_fetch_failed` warning and that the reply reflects the `chat_social_room` persona.
5. Reload the page (new session) with the toggle off, then re-enable it and ask "what do you remember about our room" — because `room_id` is the stable `hub-direct` constant (not session-scoped), any continuity persisted from step 4 via the existing bus write-back path should be retrievable here.
6. Switch the posture selector to "Relaxed", send a message containing "this is private, keep it a secret" — confirm in logs/inspection that the redaction score for that turn is lower than the same phrase would score under "Strict" (spot-check via the existing Social Inspection panel or `GET /api/social-room/inspection` proxy for the `hub`/`hub-direct` room).
7. **CallSyne isolation check:** With social room toggled on, confirm Hub logs show **no** lines containing `room_outbound_post_succeeded`, `callsyne`, or `orion-social-room-bridge`. Confirm routing debug includes `social_room_mode: "hub_direct"`. If `orion-social-room-bridge` is running, its logs should show **no** new `room_orion_invocation_sent` entries triggered by Hub UI sends (bridge only fires on inbound CallSyne webhooks/polls, not Hub websocket traffic).

- [ ] **Step 9.4: Update the plan checkboxes**

Mark every completed task's checkboxes in this file as done once verified.

---

## Self-review notes

- **Spec coverage:** toggle granularity (per-session, Task 8.4) ✅; Hub calling `orion-social-memory` directly (Tasks 6-7) ✅; redaction posture as a Hub-tunable knob (Tasks 1-5, 8) ✅; **CallSyne isolation** (Non-goals + Task 7b) ✅.
- **CallSyne isolation:** Hub-direct path is websocket → cortex → optional internal bus only. Bridge/CallSyne post is inbound-triggered only. `platform=hub` / `room_id=hub-direct` namespace keeps continuity separate from `callsyne:*` rooms. `social_room_mode: "hub_direct"` audit tag in payload + `client_meta` (Task 3 + Task 7).
- **Write-back:** deliberately *not* touched — `publish_social_room_turn` in `services/orion-hub/scripts/chat_history.py` already gates purely on `client_meta.chat_profile == "social_room"`, so it fires automatically for Hub-direct toggle turns once Task 7 sets `chat_profile`. Confirmed via `services/orion-hub/scripts/chat_history.py:550-599` and `orion/bus/channels.yaml:873-881`. Write-back publishes `social.turn.v1` to internal bus — **not** `external.room.post.*`.
- **No placeholders:** every step above has complete, copy-pasteable code; no "add error handling"-style stand-ins.
- **Type consistency:** `posture` parameter name and `SocialRedactionPosture` literal (`"strict" | "relaxed"`) are identical across `social_room.py`, `cortex_request_builder.py`, and the settings/env var; `social_room_mode` / `social_redaction_posture` payload keys match between `app.js` and `websocket_handler.py`.
