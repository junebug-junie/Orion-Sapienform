# AI Town Pair-Turn Read Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** NPC start/first-continue inject last pair utterances plus that speaker's other-thread utterances from `social_room_turns`, never `/summary` topic bags, never other people's chats.

**Architecture:** Pure filter in social-memory over `social_room_turns` rows. HTTP `GET /town-continuity`. Convex `fetchTownContinuity(speaker, other)` formats two labeled blocks. Identity sheet stays. Embodiment `/summary` stays.

**Tech Stack:** Pydantic schemas, FastAPI, SQLAlchemy `text()` SELECT, AI Town convex patch, pytest.

**Spec:** `docs/superpowers/specs/2026-08-29-aitown-pair-turn-read-design.md`

## Global Constraints

- Approach B only: raw turns from `social_room_turns`, not `/summary` for NPC speech.
- Town facts: speaker's other `{a}--{b}` threads only. Sofia↔Cam never enters Nico→Juniper.
- Slug match is exact on the two `--`-split parts. No substring `LIKE`.
- Caps: 8 pair utterances, 4 town utterances, oldest-first.
- One stored row → two utterances when both prompt and response are non-empty (other, then row speaker).
- Text clamp 160 characters.
- `GET /town-continuity` requires `platform`, `room_id`, `thread_id`, `speaker_id`. 400 if `thread_id` or `speaker_id` missing/blank.
- Tags must contain `aitown`. Skip `redaction.recall_safe is false`. Skip missing JSON paths.
- Fetch once at start and first continue (`priorMessages.length <= 1`). Fail-open 3s. Empty lists → omit both prompt blocks.
- Prompt labels exactly: `Earlier with them:` and `From your other conversations:`
- Keep identity / `agentPrompts`. No don’t-lists. No `LLM_MODEL` change. No synthesizer rewrite. No sql-writer index. Do not import sql-writer models.
- Embodiment `GET /summary` unchanged.
- Work in this worktree (`feat/aitown-pair-turn-read`). Do not commit `.env`. Follow TDD. Commit per task.

---

### Task 1: Schema + registry

**Files:**
- Modify: `orion/schemas/social_chat.py`
- Modify: `orion/schemas/registry.py`
- Create: `tests/test_town_continuity_schema.py`

**Interfaces:**
- Consumes: existing `social_chat.py` pydantic style (`extra="forbid"`)
- Produces: `TownContinuityTurnV1`, `TownContinuityReadV1` registered in `SCHEMA_REGISTRY`

- [ ] **Step 1: Write the failing test**

```python
from pydantic import ValidationError
import pytest

from orion.schemas.registry import SCHEMA_REGISTRY
from orion.schemas.social_chat import TownContinuityReadV1, TownContinuityTurnV1


def test_town_continuity_schemas_are_registered() -> None:
    assert SCHEMA_REGISTRY["TownContinuityTurnV1"] is TownContinuityTurnV1
    assert SCHEMA_REGISTRY["TownContinuityReadV1"] is TownContinuityReadV1


def test_empty_lists_are_valid() -> None:
    body = TownContinuityReadV1(
        thread_id="juniper-feld--nico-sable",
        speaker_id="nico-sable",
        pair_turns=[],
        town_turns=[],
    )
    assert body.pair_turns == []
    assert body.town_turns == []


def test_turn_roundtrip() -> None:
    turn = TownContinuityTurnV1(
        speaker="Nico Sable",
        other="Juniper Feld",
        text="the pie sat out",
        thread_id="juniper-feld--nico-sable",
        created_at="2026-08-29T22:00:00+00:00",
    )
    again = TownContinuityTurnV1.model_validate(turn.model_dump())
    assert again.text == "the pie sat out"


def test_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        TownContinuityTurnV1(
            speaker="Nico Sable",
            other="Juniper Feld",
            text="hi",
            thread_id="juniper-feld--nico-sable",
            created_at="2026-08-29T22:00:00+00:00",
            client_meta={"nope": True},
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_town_continuity_schema.py -q`
Expected: FAIL (import error — types not defined)

- [ ] **Step 3: Write minimal implementation**

Append to `orion/schemas/social_chat.py`:

```python
class TownContinuityTurnV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    speaker: str
    other: str
    text: str
    thread_id: str
    created_at: str


class TownContinuityReadV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    thread_id: str
    speaker_id: str
    pair_turns: List[TownContinuityTurnV1] = Field(default_factory=list)
    town_turns: List[TownContinuityTurnV1] = Field(default_factory=list)
```

Import both in `orion/schemas/registry.py` from `orion.schemas.social_chat` and add `"TownContinuityTurnV1"` and `"TownContinuityReadV1"` to `SCHEMA_REGISTRY` next to `SocialRoomTurnStoredV1`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_town_continuity_schema.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add orion/schemas/social_chat.py orion/schemas/registry.py tests/test_town_continuity_schema.py
git commit -m "$(cat <<'EOF'
Add TownContinuity read schemas for speaker-grounded pair history.

EOF
)"
```

---

### Task 2: Pure filter (divination gate)

**Files:**
- Create: `services/orion-social-memory/app/town_continuity.py`
- Create: `services/orion-social-memory/tests/test_town_continuity.py`

**Interfaces:**
- Consumes: `TownContinuityTurnV1`, `TownContinuityReadV1`, `orion.town_cast.TOWN_PARTICIPANT_SLUGS`
- Produces:
  - `speaker_on_thread(thread_id: str, speaker_id: str) -> bool`
  - `utterances_from_row(row: dict) -> list[TownContinuityTurnV1]`
  - `select_town_continuity(*, platform: str, room_id: str, thread_id: str, speaker_id: str, rows: list[dict]) -> TownContinuityReadV1`

Row dict keys: `prompt`, `response`, `tags` (list or JSON string), `redaction` (dict or JSON), `client_meta` (dict or JSON), `created_at` (str or datetime).

- [ ] **Step 1: Write the failing tests**

Cover at least:

1. Nico + Juniper pair rows become pair utterances including Juniper's `prompt` line.
2. Nico↔Sofia rows land in `town_turns`.
3. Sofia↔Cam rows do **not** appear for `speaker_id=nico-sable`.
4. `recall_safe=false` skipped.
5. Missing `aitown` tag skipped.
6. Caps: 9 pair rows flatten then first 8 oldest utterances; town cap 4.
7. `thread_id` with extra `--` segments: `speaker_on_thread` is False (must be exactly two parts).
8. Empty rows → empty lists.

Helper to build a row:

```python
def _row(*, thread_id, speaker_id, speaker_name, prompt, response, created_at, tags=None, recall_safe=True):
    return {
        "prompt": prompt,
        "response": response,
        "tags": tags if tags is not None else ["aitown"],
        "redaction": {"recall_safe": recall_safe},
        "client_meta": {
            "external_room": {
                "platform": "aitown",
                "room_id": "aitown-town",
                "thread_id": thread_id,
            },
            "external_participant": {
                "participant_id": speaker_id,
                "participant_name": speaker_name,
            },
        },
        "created_at": created_at,
    }
```

Assert Nico→Juniper select does not contain `"jailbreak"` if that text only exists on `cam-lin--sofia-bell`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest services/orion-social-memory/tests/test_town_continuity.py -q`
Expected: FAIL (module missing)

- [ ] **Step 3: Write minimal implementation**

`town_continuity.py`:

- `_parse_json(value)` if str then `json.loads`, if dict/list return as-is, else `{}` / `[]`.
- `_thread_parts(thread_id)` split on `--`; return two slugs or None.
- `speaker_on_thread`: parts is not None and speaker_id in parts.
- Inverse slug map from `TOWN_PARTICIPANT_SLUGS`.
- `utterances_from_row`: skip if platform/room mismatch, no thread_id, no aitown tag, recall_safe is False. Emit prompt utterance (other slug's display name) then response utterance (row speaker). Clamp text to 160. Skip empty text.
- `select_town_continuity`: collect pair vs town via `speaker_on_thread` and thread_id equality; sort each by `(created_at, speaker)` oldest-first; slice `[:8]` and `[:4]`; return `TownContinuityReadV1`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest services/orion-social-memory/tests/test_town_continuity.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add services/orion-social-memory/app/town_continuity.py services/orion-social-memory/tests/test_town_continuity.py
git commit -m "$(cat <<'EOF'
Filter town continuity from raw turns, speaker-grounded.

EOF
)"
```

---

### Task 3: GET /town-continuity

**Files:**
- Modify: `services/orion-social-memory/app/service.py`
- Modify: `services/orion-social-memory/app/main.py`
- Modify: `services/orion-social-memory/tests/test_social_memory_service.py` (add tests at end; do not rewrite existing)
- Modify: `services/orion-social-memory/README.md`

**Interfaces:**
- Consumes: `select_town_continuity`, `get_session`
- Produces: `SocialMemoryService.get_town_continuity(*, platform, room_id, thread_id, speaker_id) -> TownContinuityReadV1`
- HTTP: `GET /town-continuity` → model_dump JSON. 400 `thread_id_and_speaker_id_required` if either blank. 400 `platform_and_room_id_required` if platform/room blank.

- [ ] **Step 1: Write the failing tests**

In `test_social_memory_service.py`:

1. `test_get_town_continuity_endpoint_requires_thread_and_speaker` — TestClient GET without thread_id → 400.
2. `test_get_town_continuity_reads_social_room_turns` — create `social_room_turns` on the sqlite engine used by `_service_and_session` (or a dedicated helper), insert one Nico↔Juniper row and one Sofia↔Cam row, call `svc.get_town_continuity(...)`, assert pair has the Juniper prompt line and town does not have the Cam line.

Create the extra table with raw SQL in the test helper:

```sql
CREATE TABLE social_room_turns (
  turn_id TEXT PRIMARY KEY,
  prompt TEXT NOT NULL,
  response TEXT NOT NULL,
  text TEXT NOT NULL DEFAULT '',
  tags JSON,
  redaction JSON,
  client_meta JSON,
  created_at TEXT
);
```

Insert JSON columns as JSON strings if SQLite needs it. Service must `_parse_json` either way.

3. `test_get_town_continuity_missing_table_is_empty` — service against Base-only sqlite (no social_room_turns) returns empty lists, does not raise.

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `pytest services/orion-social-memory/tests/test_social_memory_service.py::test_get_town_continuity_endpoint_requires_thread_and_speaker services/orion-social-memory/tests/test_social_memory_service.py::test_get_town_continuity_reads_social_room_turns services/orion-social-memory/tests/test_social_memory_service.py::test_get_town_continuity_missing_table_is_empty -q`
Expected: FAIL (attribute / 404)

- [ ] **Step 3: Write minimal implementation**

`get_town_continuity`: open session, `SELECT prompt, response, tags, redaction, client_meta, created_at FROM social_room_turns`. On any exception (missing table), return empty `TownContinuityReadV1`. Map rows to dicts. Call `select_town_continuity`. Close session in `finally` like `get_summary`.

`main.py` route mirrors `/summary` query style.

README: document `GET /town-continuity` with the four query params and one curl example. Do not remove `/summary`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest services/orion-social-memory/tests/test_town_continuity.py services/orion-social-memory/tests/test_social_memory_service.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add services/orion-social-memory/app/service.py services/orion-social-memory/app/main.py services/orion-social-memory/tests/test_social_memory_service.py services/orion-social-memory/README.md
git commit -m "$(cat <<'EOF'
Add GET /town-continuity from social_room_turns.

EOF
)"
```

---

### Task 4: Convex fetch swap

**Files:**
- Modify: `services/orion-ai-town/patches/orion-town-continuity-ingest.patch`
- Modify: `services/orion-ai-town/tests/test_town_continuity_prompt_patch.py`
- Modify: `services/orion-ai-town/README.md`

**Interfaces:**
- Consumes: `GET /town-continuity` JSON
- Produces: `fetchTownContinuity(speakerName: string, otherName: string): Promise<string>`
- Start + first continue call `fetchTownContinuity(player.name, otherPlayer.name)`
- Prompt inject: if returned string non-empty, `prompt.push(continuity)` (the formatter includes the headers)

- [ ] **Step 1: Write the failing tests (update existing assertions)**

Change `test_town_continuity_prompt_patch.py`:

- `test_continuity_patch_fetches_summary_once_at_start`: assert `/town-continuity` present, `/summary` **absent** from this patch (ingest patch must not call `/summary`). Ingest `/ingest-turn` stays.
- `test_continuity_patch_fetches_on_first_continue`: assert `priorMessages.length <= 1`, `Earlier with them:`, `From your other conversations:`, **not** `What you remember:`.
- Add `test_continuity_patch_fetch_uses_speaker_and_other`: assert `fetchTownContinuity(player.name, otherPlayer.name)` (or equivalent two-arg call) and query params `thread_id` + `speaker_id`.
- Add `test_continuity_patch_omits_topic_bag_labels`: assert patch does not contain `What you remember:` or `safe_continuity_summary`.
- Update README test if the README sentence still says GET `/summary`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest services/orion-ai-town/tests/test_town_continuity_prompt_patch.py -q`
Expected: FAIL on `/town-continuity` / labels

- [ ] **Step 3: Write minimal implementation**

Replace `fetchTownContinuity` in the patch:

```typescript
async function fetchTownContinuity(speakerName: string, otherName: string): Promise<string> {
  try {
    const speakerId = townSlug(speakerName);
    const threadId = townThreadId(speakerName, otherName);
    const base = (process.env.SOCIAL_MEMORY_URL || '').trim().replace(/\/+$/, '');
    if (!speakerId || !threadId || !base) {
      return '';
    }
    const params = new URLSearchParams({
      platform: 'aitown',
      room_id: 'aitown-town',
      thread_id: threadId,
      speaker_id: speakerId,
    });
    const res = await fetch(`${base}/town-continuity?${params.toString()}`, {
      signal: AbortSignal.timeout(3000),
    });
    if (!res.ok) {
      return '';
    }
    const data = await res.json();
    return formatTownContinuity(data);
  } catch {
    return '';
  }
}

function formatTownContinuity(data: unknown): string {
  const body = data && typeof data === 'object' ? (data as Record<string, unknown>) : {};
  const pair = Array.isArray(body.pair_turns) ? body.pair_turns : [];
  const town = Array.isArray(body.town_turns) ? body.town_turns : [];
  const lines: string[] = [];
  const render = (rows: unknown[]) => {
    const out: string[] = [];
    for (const row of rows) {
      if (!row || typeof row !== 'object') continue;
      const speaker = String((row as { speaker?: unknown }).speaker || '').trim();
      const text = String((row as { text?: unknown }).text || '').trim();
      if (speaker && text) out.push(`${speaker}: ${text}`);
    }
    return out;
  };
  const pairLines = render(pair);
  const townLines = render(town);
  if (pairLines.length) {
    lines.push('Earlier with them:', ...pairLines);
  }
  if (townLines.length) {
    lines.push('From your other conversations:', ...townLines);
  }
  return lines.join('\n');
}
```

Update both call sites to `fetchTownContinuity(player.name, otherPlayer.name)`.
Replace `What you remember: ${continuity}` with just `prompt.push(continuity)` when non-empty.

Do not add quest/secret/code/key lines. Do not touch `orion-npc-answer-first.patch` (not on this branch).

README: NPC start/first-continue fetch `GET /town-continuity` with `thread_id` + `speaker_id`. Topic bags are not injected. Embodiment still uses `/summary`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest services/orion-ai-town/tests -q`
Expected: PASS (skip world.ts test if that skip already exists)

- [ ] **Step 5: Commit**

```bash
git add services/orion-ai-town/patches/orion-town-continuity-ingest.patch services/orion-ai-town/tests/test_town_continuity_prompt_patch.py services/orion-ai-town/README.md
git commit -m "$(cat <<'EOF'
Swap NPC continuity fetch to speaker-grounded pair turns.

EOF
)"
```
