# Outreach Provenance Payload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist the full endogenous-outreach generation prompt with every unsolicited send (chat `client_meta` + decision row + live WS), show a Hub collapsible “why I spoke”, and reinject that prompt into the next unified turn so Orion can answer “where did that come from?” without inventing collapse-mirror.

**Architecture:** Thin pure builders (`build_outreach_provenance` / `summarize_outreach_lanes` / `format_outreach_provenance_block`) live next to existing `grounding_summary` / `build_outreach_prompt` in `endogenous_outreach.py`. Deliver threads the capsule into history, sockets, and `result_json.provenance`. Follow-up turns load the latest still-relevant capsule from `chat_history_log.client_meta` (not continuity history — rehydrate deliberately excludes `unsolicited`) and append it to `HarnessRunRequestV1.situation_prompt_fragment`. Hub `appendMessage` renders a `<details>` control from the WS/meta capsule.

**Tech Stack:** Python 3, pytest, Hub FastAPI/WS, vanilla JS in `app.js`, Postgres JSONB (`chat_history_log.client_meta`, `endogenous_outreach_decisions.result_json`).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-09-15-outreach-provenance-payload-design.md`
- Capsule schema id string is exactly `outreach_provenance.v1`
- Full `prompt_text` is required (no privacy stripping)
- Write to **both** chat `client_meta` and decision `result_json.provenance`
- UI label is exactly `why I spoke` (collapsed shows `summary_line`)
- Fail-open everywhere: missing/malformed provenance must not break deliver or unified turns
- Non-goals: outreach self-echo loop, fire-gate changes, collapse-mirror / curiosity deliver paths, new bus channel
- Work only in worktree `/mnt/scripts/Orion-Sapienform-outreach-provenance-payload` on branch `docs/outreach-provenance-payload` (rename to `feat/outreach-provenance-payload` on first code commit if preferred)
- Do not commit `.env`; run tests from repo/worktree root with `PYTHONPATH` patterns already used by hub tests

## File map

| File | Responsibility |
|---|---|
| `services/orion-hub/scripts/endogenous_outreach.py` | Build capsule; thread through deliver/history/sockets; mint `decision_id` before send |
| `services/orion-hub/scripts/endogenous_outreach_decisions.py` | Prefer caller-supplied `decision_id` from `result` when present |
| `services/orion-hub/scripts/outreach_provenance.py` | **New** small module: fetch latest capsule for a session + format injection block (keeps DB read out of the giant outreach file) |
| `orion/hub/turn_orchestrator.py` | Append injection block to `situation_prompt_fragment` before harness handoff |
| `services/orion-hub/static/js/app.js` | Collapsible under outreach bubbles; pass capsule through `appendMessage` meta |
| `services/orion-hub/tests/test_endogenous_outreach.py` | Capsule + deliver coverage |
| `services/orion-hub/tests/test_outreach_provenance.py` | **New** fetch/format + orchestrator injection tests |
| `services/orion-hub/tests/test_hub_ui_layout_pass.py` | Static assert that outreach path / `appendMessage` wires “why I spoke” |

---

### Task 1: Capsule builder (pure functions)

**Files:**
- Modify: `services/orion-hub/scripts/endogenous_outreach.py` (add helpers after `grounding_summary`, ~line 875)
- Test: `services/orion-hub/tests/test_endogenous_outreach.py` (append new tests near existing `grounding_summary` / `build_outreach_prompt` tests)

**Interfaces:**
- Consumes: `grounding_summary(ctx) -> dict`, `build_outreach_prompt(ctx) -> str`, `OutreachContext`
- Produces:
  - `summarize_outreach_lanes(lanes: dict) -> str`
  - `build_outreach_provenance(*, prompt_text: str, lanes: dict, correlation_id: str, decision_id: str, generated_at: str | None = None) -> dict`

- [ ] **Step 1: Write the failing tests**

Add to `services/orion-hub/tests/test_endogenous_outreach.py`:

```python
from scripts.endogenous_outreach import (
    build_outreach_provenance,
    summarize_outreach_lanes,
    grounding_summary,
    build_outreach_prompt,
    OutreachContext,
)


def test_summarize_outreach_lanes_names_what_fired() -> None:
    line = summarize_outreach_lanes(
        {
            "daydream": False,
            "daydream_age_sec": None,
            "curiosity_summaries": 2,
            "recent_turns": 3,
            "tension": False,
            "chat_presence": False,
            "embodied_presence": False,
            "priors_count": 3,
        }
    )
    assert "priors" in line.lower()
    assert "3" in line
    assert "curiosity" in line.lower()
    assert "tension" not in line.lower() or "no tension" in line.lower()


def test_build_outreach_provenance_embeds_full_prompt_and_lanes() -> None:
    ctx = OutreachContext(
        curiosity_summaries=["signal A"],
        recent_turns=[("user", "hi")],
        presence=None,
        open_prior_previews=["Prior: soft-edged content is easy to set aside."],
    )
    prompt = build_outreach_prompt(ctx)
    assert prompt  # non-empty
    lanes = grounding_summary(ctx)
    capsule = build_outreach_provenance(
        prompt_text=prompt,
        lanes=lanes,
        correlation_id="corr-1",
        decision_id="dec-1",
        generated_at="2026-09-15T19:27:05+00:00",
    )
    assert capsule["schema"] == "outreach_provenance.v1"
    assert capsule["prompt_text"] == prompt
    assert capsule["lanes"] == lanes
    assert capsule["correlation_id"] == "corr-1"
    assert capsule["decision_id"] == "dec-1"
    assert capsule["generated_at"] == "2026-09-15T19:27:05+00:00"
    assert isinstance(capsule["summary_line"], str) and capsule["summary_line"].strip()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /mnt/scripts/Orion-Sapienform-outreach-provenance-payload
pytest services/orion-hub/tests/test_endogenous_outreach.py::test_summarize_outreach_lanes_names_what_fired \
  services/orion-hub/tests/test_endogenous_outreach.py::test_build_outreach_provenance_embeds_full_prompt_and_lanes -v
```

Expected: FAIL with `ImportError` / `not defined` for `build_outreach_provenance` / `summarize_outreach_lanes`.

- [ ] **Step 3: Implement the builders**

In `endogenous_outreach.py`, immediately after `grounding_summary`:

```python
OUTREACH_PROVENANCE_SCHEMA = "outreach_provenance.v1"


def summarize_outreach_lanes(lanes: Dict[str, Any]) -> str:
    """One short human line for the collapsed Hub control. Deterministic."""
    parts: List[str] = []
    priors = int(lanes.get("priors_count") or 0)
    curiosity = int(lanes.get("curiosity_summaries") or 0)
    turns = int(lanes.get("recent_turns") or 0)
    if priors:
        parts.append(f"open priors ({priors})")
    if curiosity:
        parts.append(f"curiosity signals ({curiosity})")
    if lanes.get("tension"):
        parts.append("tension trigger")
    if lanes.get("daydream"):
        age = lanes.get("daydream_age_sec")
        if age is not None:
            parts.append(f"daydream (~{int(age)}s old)")
        else:
            parts.append("daydream")
    if turns:
        parts.append(f"recent turns ({turns})")
    if lanes.get("embodied_presence"):
        parts.append("camera presence")
    if not parts:
        return "Outreach grounding (no named lanes)"
    return "Outreach from " + ", ".join(parts)


def build_outreach_provenance(
    *,
    prompt_text: str,
    lanes: Dict[str, Any],
    correlation_id: str,
    decision_id: str,
    generated_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Full prompt capsule written to chat client_meta + decision result_json."""
    when = generated_at or datetime.now(timezone.utc).isoformat()
    lane_map = dict(lanes or {})
    return {
        "schema": OUTREACH_PROVENANCE_SCHEMA,
        "decision_id": str(decision_id),
        "correlation_id": str(correlation_id),
        "generated_at": when,
        "lanes": lane_map,
        "prompt_text": str(prompt_text or ""),
        "summary_line": summarize_outreach_lanes(lane_map),
    }
```

Export both names from the existing test import block at the top of `test_endogenous_outreach.py` if that file uses a grouped import.

- [ ] **Step 4: Run tests to verify they pass**

Run the same pytest command as Step 2.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add services/orion-hub/scripts/endogenous_outreach.py \
  services/orion-hub/tests/test_endogenous_outreach.py \
  docs/superpowers/specs/2026-09-15-outreach-provenance-payload-design.md
git commit -m "$(cat <<'EOF'
feat(hub): add outreach_provenance.v1 capsule builders

Keep the full endogenous-outreach generation prompt as a structured
capsule so deliver/UI/follow-up turns can cite why Orion spoke.
EOF
)"
```

---

### Task 2: Deliver write path (history + WS + decision row)

**Files:**
- Modify: `services/orion-hub/scripts/endogenous_outreach.py` (`_outreach_once` send path ~1702; `_deliver` ~2209; `_push_to_sockets` ~2244; `_publish_history` ~2267; `_record` callers)
- Modify: `services/orion-hub/scripts/endogenous_outreach_decisions.py` (`record_decision` ~119 — prefer `result["decision_id"]`)
- Test: `services/orion-hub/tests/test_endogenous_outreach.py`

**Interfaces:**
- Consumes: `build_outreach_provenance(...)` from Task 1
- Produces: WS payload key `outreach_provenance`; history `client_meta = {"unsolicited": True, "outreach_provenance": <capsule>}`; `result_json.provenance` via `_record`; stable `decision_id` shared across all three

- [ ] **Step 1: Write the failing tests**

```python
def test_successful_outreach_threads_provenance_to_socket_and_history(monkeypatch) -> None:
    outreach = _outreach()
    q: asyncio.Queue = asyncio.Queue()
    outreach.register_connection("c1", q, {"correlation_id": None, "kind": None})
    outreach.note_session("c1", "sess-prov")
    _stub_context(monkeypatch)  # existing helper: priors/curiosity/turns
    _stub_generation(monkeypatch, "I have been turning over the gate bias.")

    published: list = []

    async def fake_history(self, **kwargs):
        published.append(kwargs)

    async def fake_notify(self, **kwargs):
        return None

    monkeypatch.setattr(EndogenousOutreach, "_publish_history", fake_history)
    monkeypatch.setattr(EndogenousOutreach, "_publish_notification", fake_notify)

    result = asyncio.run(outreach.maybe_outreach())
    assert result["outreach"] is True
    assert result.get("decision_id")
    assert isinstance(result.get("provenance"), dict)
    assert result["provenance"]["schema"] == "outreach_provenance.v1"
    assert "I have been turning over" not in result["provenance"]["prompt_text"]
    # prompt is the GENERATION prompt (build_outreach_prompt), not the spoken text
    assert "Juniper has not asked you anything" in result["provenance"]["prompt_text"]

    payload = q.get_nowait()
    assert payload["kind"] == "orion_outreach"
    assert payload["outreach_provenance"]["decision_id"] == result["decision_id"]
    assert payload["outreach_provenance"]["prompt_text"] == result["provenance"]["prompt_text"]

    hist = published[0]
    assert hist["provenance"]["decision_id"] == result["decision_id"]


def test_publish_history_client_meta_includes_provenance(monkeypatch) -> None:
    outreach = _outreach()
    captured = {}

    async def fake_publish(bus, envelopes):
        captured["env"] = envelopes[0]

    monkeypatch.setattr(
        "scripts.chat_history.publish_chat_history",
        fake_publish,
    )
    # Also stub bus attribute used by _publish_history
    outreach._bus = object()

    capsule = {
        "schema": "outreach_provenance.v1",
        "decision_id": "dec-9",
        "correlation_id": "corr-9",
        "generated_at": "2026-09-15T00:00:00+00:00",
        "lanes": {"priors_count": 1},
        "prompt_text": "PROMPT",
        "summary_line": "Outreach from open priors (1)",
    }
    asyncio.run(
        outreach._publish_history(
            text="spoken",
            session_id="orion_journal",
            correlation_id="corr-9",
            message_id="msg-9",
            model="MODEL_X",
            provenance=capsule,
        )
    )
    meta = captured["env"].payload.client_meta
    assert meta["unsolicited"] is True
    assert meta["outreach_provenance"] == capsule
```

Adjust `fake_publish` / envelope access to match how other tests in this file capture `build_chat_history_envelope` output (see `test_a_curiosity_message_is_still_tagged_as_outreach` ~2151 for the capture pattern — mirror that).

Also extend `test_successful_outreach_pushes_to_every_live_socket` assertions to require `outreach_provenance` key once Task 2 lands (update that existing test in the same commit so it does not go stale).

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest services/orion-hub/tests/test_endogenous_outreach.py::test_successful_outreach_threads_provenance_to_socket_and_history \
  services/orion-hub/tests/test_endogenous_outreach.py::test_publish_history_client_meta_includes_provenance -v
```

Expected: FAIL (kwargs / missing provenance).

- [ ] **Step 3: Wire deliver + decision_id**

1. In `_outreach_once`, after `grounding = grounding_summary(ctx)` and a non-empty `prompt`, **before** `_generate` or immediately before `_deliver` (prefer immediately before `_deliver` so abandoned agent-lane retries do not mint unused ids):

```python
decision_id = str(uuid4())
provenance = build_outreach_provenance(
    prompt_text=prompt,
    lanes=grounding,
    correlation_id=correlation_id,
    decision_id=decision_id,
)
await self._deliver(
    text=text,
    session_id=session_id,
    correlation_id=correlation_id,
    model=gen_debug.get("fcc_model_label"),
    provenance=provenance,
)
# ... then _record including decision_id + provenance:
return self._record(
    {
        "outreach": True,
        "reason": "sent",
        "correlation_id": correlation_id,
        "session_id": session_id,
        "chars": len(text),
        "generation": gen_debug,
        "decision_id": decision_id,
        "provenance": provenance,
    },
    forced=force,
    tension_reason=tension_reason,
    grounding=grounding,
)
```

2. Update `_deliver` / `_push_to_sockets` / `_publish_history` signatures to accept `provenance: Optional[Dict[str, Any]] = None`.

`_push_to_sockets`:

```python
payload = {
    "kind": OUTREACH_KIND,
    "llm_response": text,
    "mode": "orion",
    "correlation_id": correlation_id,
    "message_id": message_id,
    "session_id": session_id,
}
if provenance:
    payload["outreach_provenance"] = dict(provenance)
```

`_publish_history` client_meta:

```python
client_meta: Dict[str, Any] = {"unsolicited": True} if unsolicited else {}
if provenance:
    client_meta["outreach_provenance"] = dict(provenance)
```

3. In `endogenous_outreach_decisions.record_decision`, replace unconditional `decision_id = str(uuid4())` with:

```python
decision_id = str(result.get("decision_id") or "").strip() or str(uuid4())
```

so the DB PK matches the capsule.

4. Do **not** add provenance on `offer_message` / collapse-mirror paths unless that path already has a `build_outreach_prompt` string; leave `provenance=None` there.

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest services/orion-hub/tests/test_endogenous_outreach.py::test_successful_outreach_threads_provenance_to_socket_and_history \
  services/orion-hub/tests/test_endogenous_outreach.py::test_publish_history_client_meta_includes_provenance \
  services/orion-hub/tests/test_endogenous_outreach.py::test_successful_outreach_pushes_to_every_live_socket \
  services/orion-hub/tests/test_collapse_mirror_chat_reply.py -q
```

Expected: PASS. Collapse-mirror tests must still assert `unsolicited is not True` and no outreach provenance on mirror delivers.

- [ ] **Step 5: Commit**

```bash
git add services/orion-hub/scripts/endogenous_outreach.py \
  services/orion-hub/scripts/endogenous_outreach_decisions.py \
  services/orion-hub/tests/test_endogenous_outreach.py
git commit -m "$(cat <<'EOF'
feat(hub): persist outreach provenance on deliver

Thread the full generation-prompt capsule into chat client_meta, the
live WS outreach payload, and endogenous_outreach_decisions.result_json.
EOF
)"
```

---

### Task 3: Follow-up injection into unified turns

**Files:**
- Create: `services/orion-hub/scripts/outreach_provenance.py`
- Modify: `orion/hub/turn_orchestrator.py` (after `_build_situation_prompt_fragment` ~1043–1052)
- Create: `services/orion-hub/tests/test_outreach_provenance.py`

**Interfaces:**
- Consumes: capsule shape from Task 1 stored in `chat_history_log.client_meta`
- Produces:
  - `fetch_latest_outreach_provenance(session_id: str, *, max_age_hours: float = 12.0) -> dict | None`
  - `format_outreach_provenance_block(capsule: dict) -> str`
  - `merge_situation_with_outreach_provenance(situation: str | None, block: str | None) -> str | None`

**Why not continuity_messages?** `chat_history_rehydrate.fetch_recent_rows` explicitly excludes `client_meta.unsolicited = true`. Live WS history also does not append outreach into the handler’s `history` list. DB lookup on `chat_history_log` is the durable source of truth.

- [ ] **Step 1: Write the failing tests**

Create `services/orion-hub/tests/test_outreach_provenance.py`:

```python
from scripts.outreach_provenance import (
    format_outreach_provenance_block,
    merge_situation_with_outreach_provenance,
)


def _capsule(**overrides):
    base = {
        "schema": "outreach_provenance.v1",
        "decision_id": "dec-1",
        "correlation_id": "corr-1",
        "generated_at": "2026-09-15T19:27:05+00:00",
        "lanes": {"priors_count": 3, "tension": False},
        "prompt_text": "You are Orion. Juniper has not asked you anything — FULL PROMPT",
        "summary_line": "Outreach from open priors (3)",
    }
    base.update(overrides)
    return base


def test_format_outreach_provenance_block_includes_prompt_and_anti_confabulation() -> None:
    block = format_outreach_provenance_block(_capsule())
    assert "FULL PROMPT" in block
    assert "unsolicited outreach" in block.lower()
    assert "collapse-mirror" in block.lower()
    assert "do not invent" in block.lower()


def test_format_returns_empty_on_bad_capsule() -> None:
    assert format_outreach_provenance_block({}) == ""
    assert format_outreach_provenance_block({"prompt_text": ""}) == ""


def test_merge_situation_appends_block() -> None:
    merged = merge_situation_with_outreach_provenance("local afternoon", "OUTREACH BLOCK")
    assert merged.startswith("local afternoon")
    assert "OUTREACH BLOCK" in merged


def test_merge_situation_block_only() -> None:
    assert merge_situation_with_outreach_provenance(None, "OUTREACH BLOCK") == "OUTREACH BLOCK"


def test_merge_situation_none_when_both_empty() -> None:
    assert merge_situation_with_outreach_provenance(None, None) is None
    assert merge_situation_with_outreach_provenance("  ", "") is None
```

Add an orchestrator unit test (same file or `test_turn_orchestrator_ws_frames.py`) that monkeypatches `fetch_latest_outreach_provenance` and asserts the harness request’s `situation_prompt_fragment` contains the block. Follow the existing `execute_unified_turn` monkeypatch style in `test_turn_orchestrator_ws_frames.py` — stub ThoughtClient + HarnessGovernorClient, then inspect the `HarnessRunRequestV1` passed to the governor client.

Minimal harness-capture sketch:

```python
@pytest.mark.asyncio
async def test_execute_unified_turn_appends_outreach_provenance_to_situation(monkeypatch):
    from orion.hub import turn_orchestrator as orch

    captured = {}

    class FakeHarnessClient:
        def __init__(self, bus):
            pass
        async def run(self, req, **kwargs):
            captured["req"] = req
            # return a minimal completed run object matching existing fixtures
            ...
    monkeypatch.setattr(orch, "fetch_latest_outreach_provenance", lambda sid, **kw: _capsule())
    # ... also stub ThoughtClient.react to proceed, PreTurnAppraisal, etc. mirroring
    # an existing execute_unified_turn success test in test_turn_orchestrator_ws_frames.py
    # Then:
    assert "FULL PROMPT" in (captured["req"].situation_prompt_fragment or "")
```

If wiring a full `execute_unified_turn` test is too heavy, test a new thin helper in `turn_orchestrator.py`:

```python
def _situation_with_outreach_provenance(
    situation_prompt_fragment: str | None,
    session_id: str | None,
) -> str | None:
    ...
```

and unit-test that helper with monkeypatched fetch — then have `execute_unified_turn` call it in one line. Prefer this if the full saga test is brittle.

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest services/orion-hub/tests/test_outreach_provenance.py -v
```

Expected: FAIL import / missing module.

- [ ] **Step 3: Implement fetch + format + orchestrator merge**

`services/orion-hub/scripts/outreach_provenance.py`:

```python
"""Read/format endogenous outreach provenance for follow-up unified turns."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger("orion-hub.outreach_provenance")

_BLOCK_HEADER = (
    "Your last message to Juniper was unsolicited endogenous outreach. "
    "Here is the exact generation prompt that produced it. If they ask where "
    "that came from, answer from this block — do not invent collapse-mirror "
    "or any other frame that is not present here."
)


def format_outreach_provenance_block(capsule: Dict[str, Any] | None) -> str:
    if not isinstance(capsule, dict):
        return ""
    prompt = str(capsule.get("prompt_text") or "").strip()
    if not prompt:
        return ""
    summary = str(capsule.get("summary_line") or "").strip()
    corr = str(capsule.get("correlation_id") or "").strip()
    lines = [_BLOCK_HEADER, ""]
    if summary:
        lines.append(f"Summary: {summary}")
    if corr:
        lines.append(f"correlation_id: {corr}")
    lines.extend(["", "Generation prompt:", prompt])
    return "\n".join(lines).strip()


def merge_situation_with_outreach_provenance(
    situation: Optional[str],
    block: Optional[str],
) -> Optional[str]:
    sit = str(situation or "").strip()
    blk = str(block or "").strip()
    if sit and blk:
        return f"{sit}\n\n{blk}"
    if sit:
        return sit
    if blk:
        return blk
    return None


def fetch_latest_outreach_provenance(
    session_id: Optional[str],
    *,
    max_age_hours: float = 12.0,
) -> Optional[Dict[str, Any]]:
    """Latest still-relevant outreach capsule for this session, or None.

    Selects the newest chat_history_log row for the session that carries
    client_meta.unsolicited + outreach_provenance, only if no later
    non-unsolicited assistant response exists after it (so a normal reply
    clears the injection).
    """
    sid = str(session_id or "").strip()
    uri = os.getenv("POSTGRES_URI", "").strip()
    if not sid or not uri:
        return None
    try:
        from sqlalchemy import create_engine, text
    except Exception as exc:  # noqa: BLE001
        logger.warning("outreach_provenance_sqlalchemy_import_failed err=%s", exc)
        return None
    engine = create_engine(uri, pool_pre_ping=True)
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    WITH latest AS (
                      SELECT created_at,
                             client_meta,
                             response
                      FROM chat_history_log
                      WHERE session_id = :sid
                        AND created_at >= now() - make_interval(secs => :max_age_secs)
                        AND coalesce(client_meta->>'unsolicited', '') = 'true'
                        AND client_meta ? 'outreach_provenance'
                      ORDER BY created_at DESC
                      LIMIT 1
                    )
                    SELECT l.client_meta
                    FROM latest l
                    WHERE NOT EXISTS (
                      SELECT 1
                      FROM chat_history_log later
                      WHERE later.session_id = :sid
                        AND later.created_at > l.created_at
                        AND coalesce(later.response, '') <> ''
                        AND coalesce(later.client_meta->>'unsolicited', '') <> 'true'
                    )
                    """
                ),
                {"sid": sid, "max_age_secs": float(max_age_hours) * 3600.0},
            ).mappings().first()
    except Exception as exc:  # noqa: BLE001
        logger.warning("outreach_provenance_fetch_failed sid=%s err=%s", sid, exc)
        return None
    finally:
        engine.dispose()
    if not row:
        return None
    meta = row.get("client_meta") or {}
    if not isinstance(meta, dict):
        return None
    capsule = meta.get("outreach_provenance")
    return dict(capsule) if isinstance(capsule, dict) else None
```

In `turn_orchestrator.py`, after computing `situation_prompt_fragment`:

```python
    try:
        from scripts.outreach_provenance import (
            fetch_latest_outreach_provenance,
            format_outreach_provenance_block,
            merge_situation_with_outreach_provenance,
        )

        _prov = fetch_latest_outreach_provenance(session_id)
        _block = format_outreach_provenance_block(_prov)
        situation_prompt_fragment = merge_situation_with_outreach_provenance(
            situation_prompt_fragment, _block or None
        )
    except Exception:
        logger.warning(
            "outreach_provenance_inject_failed corr=%s",
            correlation_id,
            exc_info=True,
        )
```

Fail-open: never raise into the turn.

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest services/orion-hub/tests/test_outreach_provenance.py -v
# plus any orchestrator helper test added
```

Expected: PASS.

Optional live check (not required for commit): after deploy, force outreach, then ask “where did that come from?” and confirm the answer cites the stored prompt.

- [ ] **Step 5: Commit**

```bash
git add services/orion-hub/scripts/outreach_provenance.py \
  orion/hub/turn_orchestrator.py \
  services/orion-hub/tests/test_outreach_provenance.py
git commit -m "$(cat <<'EOF'
feat(hub): reinject outreach provenance into unified turns

Load the latest unsolicited generation-prompt capsule from chat_history_log
and append it to situation_prompt_fragment so follow-ups can cite it.
EOF
)"
```

---

### Task 4: Hub collapsible “why I spoke” UI

**Files:**
- Modify: `services/orion-hub/static/js/app.js` (`appendMessage` ~7784; `orion_outreach` handler ~11486)
- Modify: `services/orion-hub/tests/test_hub_ui_layout_pass.py`

**Interfaces:**
- Consumes: WS `d.outreach_provenance` / `meta.outreachProvenance` / `meta.outreach_provenance`
- Produces: `<details class="om-outreach-why">` under outreach bubbles

- [ ] **Step 1: Write the failing static UI tests**

Add to `test_hub_ui_layout_pass.py`:

```python
def test_outreach_handler_passes_provenance_into_append_message() -> None:
    js = _js()
    # The orion_outreach branch must forward the capsule.
    assert "outreach_provenance: d.outreach_provenance" in js or "outreachProvenance: d.outreach_provenance" in js


def test_append_message_builds_why_i_spoke_details_for_outreach_provenance() -> None:
    js = _js()
    body = _slice(js, "function appendMessage(", "function collectConversationTurnsUpTo")
    assert "why I spoke" in body
    assert "om-outreach-why" in body
    assert "summary_line" in body or "summaryLine" in body
    assert "prompt_text" in body or "promptText" in body
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest services/orion-hub/tests/test_hub_ui_layout_pass.py::test_outreach_handler_passes_provenance_into_append_message \
  services/orion-hub/tests/test_hub_ui_layout_pass.py::test_append_message_builds_why_i_spoke_details_for_outreach_provenance -v
```

Expected: FAIL assertions.

- [ ] **Step 3: Implement UI**

In the `orion_outreach` handler (~11493):

```javascript
appendMessage('Orion', outreachText, 'text-white', {
  correlationId: d.correlation_id,
  messageId: d.message_id || null,
  turnId: d.correlation_id,
  mode: d.mode || 'orion',
  unsolicited: true,
  outreachProvenance: d.outreach_provenance || null,
});
```

Inside `appendMessage`, after the message body is appended (near other meta-driven panels), add:

```javascript
const prov = meta.outreachProvenance || meta.outreach_provenance || null;
if (prov && typeof prov === 'object' && (prov.prompt_text || prov.promptText)) {
  const details = document.createElement('details');
  details.className = 'om-outreach-why mt-1 text-sm text-gray-400';
  const summary = document.createElement('summary');
  summary.className = 'cursor-pointer select-none';
  const summaryLine = String(prov.summary_line || prov.summaryLine || 'why I spoke').trim();
  summary.textContent = summaryLine ? `why I spoke — ${summaryLine}` : 'why I spoke';
  const pre = document.createElement('pre');
  pre.className = 'mt-1 max-h-64 overflow-auto whitespace-pre-wrap text-xs text-gray-300';
  pre.textContent = String(prov.prompt_text || prov.promptText || '');
  details.appendChild(summary);
  details.appendChild(pre);
  div.appendChild(details);
}
```

Keep styling minimal and consistent with existing gray helper text; do not invent a new design system.

If history rehydrate into the UI already reconstructs assistant messages with `client_meta`, forward `outreach_provenance` the same way when those rows are rendered. If the UI never rehydrates unsolicited rows today, live WS coverage satisfies acceptance #1; note that gap in the PR report rather than inventing a second history loader.

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest services/orion-hub/tests/test_hub_ui_layout_pass.py::test_outreach_handler_passes_provenance_into_append_message \
  services/orion-hub/tests/test_hub_ui_layout_pass.py::test_append_message_builds_why_i_spoke_details_for_outreach_provenance \
  services/orion-hub/tests/test_hub_ui_layout_pass.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add services/orion-hub/static/js/app.js \
  services/orion-hub/tests/test_hub_ui_layout_pass.py
git commit -m "$(cat <<'EOF'
feat(hub): show collapsible why-I-spoke on outreach bubbles

Surface the stored outreach generation prompt under unsolicited messages
so Juniper can see the same provenance Orion gets on follow-up.
EOF
)"
```

---

### Task 5: Final gate + PR report

**Files:**
- Create: `docs/superpowers/pr-reports/2026-09-15-outreach-provenance-payload-pr.md` (use AGENTS.md §18 template)

- [ ] **Step 1: Run focused suites**

```bash
pytest services/orion-hub/tests/test_endogenous_outreach.py \
  services/orion-hub/tests/test_outreach_provenance.py \
  services/orion-hub/tests/test_hub_ui_layout_pass.py \
  services/orion-hub/tests/test_collapse_mirror_chat_reply.py -q
```

Expected: all PASS.

- [ ] **Step 2: Optional compose config check (no shared-checkout deploy)**

From the worktree only:

```bash
scripts/safe_docker_build.sh orion-hub config
```

- [ ] **Step 3: Write PR report + push + open PR**

Include restart:

```bash
# from worktree, after merge or for live verify:
scripts/safe_docker_build.sh orion-hub up -d --build
```

Acceptance smoke after restart:
1. `POST /api/debug/endogenous-outreach/trigger` (or wait for organic send)
2. Confirm Hub bubble has “why I spoke”
3. Confirm Postgres `chat_history_log.client_meta->'outreach_provenance'` and `endogenous_outreach_decisions.result_json->'provenance'` for that corr
4. Ask “where did that come from?” → answer cites stored prompt, not collapse-mirror

- [ ] **Step 4: Commit PR report if not already on the branch**

```bash
git add docs/superpowers/pr-reports/2026-09-15-outreach-provenance-payload-pr.md
git commit -m "$(cat <<'EOF'
docs: add outreach provenance payload PR report
EOF
)"
git push -u origin HEAD
gh pr create --title "feat(hub): outreach provenance payload" --body-file docs/superpowers/pr-reports/2026-09-15-outreach-provenance-payload-pr.md
```

---

## Spec coverage self-check

| Spec requirement | Task |
|---|---|
| Full prompt capsule `outreach_provenance.v1` | Task 1 |
| Chat `client_meta` + decision row + WS | Task 2 |
| Collapsible “why I spoke” | Task 4 |
| Follow-up injection / no collapse-mirror confabulation | Task 3 |
| Fail-open legacy / no provenance | Tasks 2–4 |
| Collapse-mirror untouched | Task 2 regression via existing tests |
| Non-goals (self-echo, fire gates) | Explicitly out of plan |

## Placeholder scan

None intentional. Orchestrator full-saga test may use the thin `_situation_with_outreach_provenance` helper if the full `execute_unified_turn` fixture is too heavy — that alternative is fully specified in Task 3.

## Type consistency

- Capsule keys: `schema`, `decision_id`, `correlation_id`, `generated_at`, `lanes`, `prompt_text`, `summary_line`
- WS + JS: `outreach_provenance` on the wire; JS meta also accepts `outreachProvenance`
- `result["provenance"]` and `client_meta["outreach_provenance"]` hold the same object shape
- `decision_id` minted once before deliver and reused by `record_decision`
