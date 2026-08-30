# AI Town Cast Cull and Town Continuity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cull four AI Town NPCs, make the remaining four speak from their jobs, and wire NPC speech into the existing `aitown-town` social-memory room.

**Architecture:** Cards and `compose_identity` change what the NPC model sees. `orion-social-memory` gains `POST /ingest-turn` that publishes `SocialRoomTurnV1` on `orion:chat:social:turn`. Convex conversation.ts fetches `/summary` once at conversation start and POSTs NPC lines when the other speaker is not Orion. Embodiment switches `participant_id` from Convex `p:*` to stable slugs from `orion/town_cast.py`. One room, pair threads, no second store.

**Tech Stack:** Python 3, FastAPI, pytest, AI Town Convex patches (unified diffs), Orion bus `SocialRoomTurnV1`.

**Spec:** `docs/superpowers/specs/2026-08-29-aitown-cast-cull-and-town-continuity-design.md`

**Worktree:** `/mnt/scripts/Orion-Sapienform-aitown-npc-continuity` on branch `docs/aitown-npc-continuity`. Commit here. Do not work in the shared checkout. Do not commit `.env`. If you add an `.env_example` key, run `python scripts/sync_local_env_from_example.py` from the worktree root.

## Global Constraints

- Live NPCs only: Mara Vale, Nico Sable, Sofia Bell, Cam Lin. Dead (archived, not deleted): Juno Park, Tessa Quinn, Vale Moreno, Dr. Elian Cross.
- Room key is `platform=aitown`, `room_id=aitown-town`. Pair thread is `thread_id="{slug_a}--{slug_b}"` with slugs sorted ascending.
- Participant slugs are the explicit map in `orion/town_cast.py`, never `name.lower().replace(" ", "-")`.
- Slug values: `mara-vale`, `nico-sable`, `sofia-bell`, `cam-lin`, `juniper-feld`, `orion`.
- `compose_identity` emits role, conversation_style, `Today: {daily_loop[0]}`, public_description, one-sentence private_pressure, one-sentence orion_dynamic. It does not emit deeper_bio or signature_line.
- Live identity strings must not contain `lighting`, `glow`, `shadows`, or `echoes`.
- Nico’s lighting signature is gone from the live card.
- Concrete-grounding patch must not name `light, shadows, glow, echoes, silence`. Positive contract only: `Answer as your job. Name a person, object, or task from your role or from what they just said. If you have nothing new to add, end the conversation.`
- `plans[cid]` is second-person from `daily_loop[0]`.
- `POST /ingest-turn` publishes `SocialRoomTurnV1` on `orion:chat:social:turn` / `social.turn.v1`. It does not call `process_social_turn` directly.
- Ingest without bearer token is 401. Empty `SOCIAL_MEMORY_INGEST_TOKEN` is fail-closed (401).
- Convex publishes NPC lines only when the other player is not Orion. Orion↔anyone stays embodiment-only.
- Ingest and summary fetch fail-open: never block NPC speech.
- Summary GET is once per conversation start, not every line. Injected continuity cap is 400 characters.
- `orion-social-memory` is added to `orion:chat:social:turn` `producer_services`.
- No new Convex memory tables. No cortex for NPC dialogue. No migration of old `p:*` rows.
- TDD: failing test first, then implementation. Commits from this worktree only.

---

### Task 1: Cast cull, archive, job cards, compose_identity

**Files:**
- Create: `orion/town_cast.py`
- Create: `orion/tests/test_town_cast.py`
- Create: `services/orion-ai-town/cards/archived/2026-08-29-retired-cast.yaml`
- Modify: `services/orion-ai-town/cards/town_cards.yaml`
- Modify: `services/orion-ai-town/scripts/generate_descriptions.py`
- Modify: `services/orion-ai-town/tests/test_generate_descriptions.py`
- Modify: `services/orion-ai-town/README.md` (cast list: four NPCs, archive path)
- Modify: `patches/orion-character.patch` only if `upstream/data/characters.ts` exists in this worktree; otherwise leave a note in the commit body that the operator must regenerate after clone. Do not invent a patch against missing upstream.

**Interfaces:**
- Consumes: existing `town_cards.yaml` schema (`id`, `name`, `role`, `public_description`, `daily_loop`, `conversation_style`, `orion_dynamic`, `private_pressure`)
- Produces: `orion.town_cast.TOWN_PARTICIPANT_SLUGS: dict[str, str]` mapping display name → slug; `orion.town_cast.slug_for_name(name: str) -> str | None`; `orion.town_cast.thread_id_for(a: str, b: str) -> str | None`; `orion.town_cast.ORION_DISPLAY_NAME = "Orion"`; `NPC_ORDER = ["mara_vale", "nico_sable", "sofia_bell", "cam_lin"]`; `compose_identity` new contract; `plans` derived from `daily_loop[0]`

- [ ] **Step 1: Write failing tests**

Add `orion/tests/test_town_cast.py`:

```python
from orion.town_cast import ORION_DISPLAY_NAME, TOWN_PARTICIPANT_SLUGS, slug_for_name, thread_id_for

def test_slug_map_is_explicit_six_rows():
    assert TOWN_PARTICIPANT_SLUGS == {
        "Mara Vale": "mara-vale",
        "Nico Sable": "nico-sable",
        "Sofia Bell": "sofia-bell",
        "Cam Lin": "cam-lin",
        "Juniper Feld": "juniper-feld",
        "Orion": "orion",
    }
    assert ORION_DISPLAY_NAME == "Orion"

def test_slug_for_name_unknown_is_none():
    assert slug_for_name("Dr. Elian Cross") is None
    assert slug_for_name("") is None

def test_thread_id_is_sorted_slugs():
    assert thread_id_for("Sofia Bell", "Cam Lin") == "cam-lin--sofia-bell"
    assert thread_id_for("Cam Lin", "Sofia Bell") == "cam-lin--sofia-bell"

def test_thread_id_unknown_is_none():
    assert thread_id_for("Sofia Bell", "Juno Park") is None
```

Replace `test_compose_identity_is_rich_and_collapsed` and `test_render_descriptions_emits_eight_valid_sprites` in `services/orion-ai-town/tests/test_generate_descriptions.py` with:

```python
_BANNED_BAIT = ("lighting", "glow", "shadows", "echoes")

def test_npc_order_is_the_live_four():
    assert gen.NPC_ORDER == ["mara_vale", "nico_sable", "sofia_bell", "cam_lin"]

def test_compose_identity_uses_job_fields_not_signature():
    by = {c["id"]: c for c in _cards()["characters"]}
    ident = gen.compose_identity(by["mara_vale"])
    assert "\n" not in ident
    assert "systems cartographer" in ident.lower()
    assert "Today:" in ident
    assert "diagrams" in ident.lower() or "maps" in ident.lower()
    assert "description of your logs" not in ident.lower()
    for bait in _BANNED_BAIT:
        assert bait not in ident.lower()

def test_live_identities_have_no_light_bait():
    by = {c["id"]: c for c in _cards()["characters"]}
    for cid in gen.NPC_ORDER:
        ident = gen.compose_identity(by[cid]).lower()
        for bait in _BANNED_BAIT:
            assert bait not in ident, f"{cid} identity contains {bait}"

def test_plans_come_from_daily_loop():
    cards = _cards()
    by = {c["id"]: c for c in cards["characters"]}
    for cid in gen.NPC_ORDER:
        loop0 = " ".join(by[cid]["daily_loop"][0].split()).lower()
        # plan is second person; must share a concrete noun/verb from daily_loop[0]
        plan = cards["plans"][cid].lower()
        assert plan.startswith("you ")
        assert any(token in plan for token in loop0.split() if len(token) > 4)

def test_render_descriptions_emits_four_valid_sprites():
    ts = gen.render_descriptions(_cards())
    assert ts.count("    character: '") == 4
    for cid in gen.NPC_ORDER:
        assert f"character: '{_cards()['sprites'][cid]}'" in ts
    for dead in ("Juno Park", "Tessa Quinn", "Vale Moreno", "Dr. Elian Cross", "Elian Cross"):
        assert dead not in ts

def test_orion_blurb_does_not_name_retired_cast():
    by = {c["id"]: c for c in _cards()["characters"]}
    blurb = gen.compose_presence_blurb(by["orion"])
    for dead in ("Elian", "Juno", "Tessa", "Vale"):
        assert dead not in blurb

def test_archived_retired_cast_exists():
    archived = _SERVICE / "cards" / "archived" / "2026-08-29-retired-cast.yaml"
    text = archived.read_text(encoding="utf-8")
    for name in ("Juno Park", "Tessa Quinn", "Vale Moreno", "Dr. Elian Cross"):
        assert name in text
```

Keep `test_cards_have_all_expected_ids` working: live `town_cards.yaml` still has `juniper_feld` and `orion`; retired ids live only in the archive file.

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /mnt/scripts/Orion-Sapienform-aitown-npc-continuity
PYTHONPATH=. pytest orion/tests/test_town_cast.py services/orion-ai-town/tests/test_generate_descriptions.py -q
```

Expected: FAIL (module missing and/or old identity/eight-NPC assertions).

- [ ] **Step 3: Implement**

`orion/town_cast.py`:

```python
from __future__ import annotations

ORION_DISPLAY_NAME = "Orion"

TOWN_PARTICIPANT_SLUGS: dict[str, str] = {
    "Mara Vale": "mara-vale",
    "Nico Sable": "nico-sable",
    "Sofia Bell": "sofia-bell",
    "Cam Lin": "cam-lin",
    "Juniper Feld": "juniper-feld",
    "Orion": "orion",
}

def slug_for_name(name: str) -> str | None:
    key = str(name or "").strip()
    if not key:
        return None
    return TOWN_PARTICIPANT_SLUGS.get(key)

def thread_id_for(name_a: str, name_b: str) -> str | None:
    left = slug_for_name(name_a)
    right = slug_for_name(name_b)
    if left is None or right is None:
        return None
    return "--".join(sorted((left, right)))
```

`generate_descriptions.py` changes:

```python
NPC_ORDER = ["mara_vale", "nico_sable", "sofia_bell", "cam_lin"]

def compose_identity(card: dict) -> str:
    name = str(card.get("name") or "They").strip()
    role = _collapse(card.get("role"))
    style = _collapse(card.get("conversation_style"))
    loop = card.get("daily_loop") or []
    today = _collapse(loop[0]) if loop else ""
    parts = [
        f"{name} is the town's {role}." if role else "",
        style,
        f"Today: {today}." if today else "",
        _collapse(card.get("public_description")),
    ]
    pressure = _collapse(card.get("private_pressure"))
    if pressure:
        first = pressure.split(".")[0].strip()
        parts.append(first + "." if first and not first.endswith(".") else first)
    dynamic = _collapse(card.get("orion_dynamic"))
    if dynamic:
        first = dynamic.split(".")[0].strip()
        parts.append(first + "." if first and not first.endswith(".") else first)
    return " ".join(p for p in parts if p)
```

`render_descriptions` already uses `plans[cid]`. Update YAML `plans:` for the four live ids to second-person `daily_loop[0]`:

```yaml
plans:
  mara_vale: "You update the town maps: who talks to whom, which paths and dependencies actually work."
  nico_sable: "You collect diner gossip and turn it into tonight's event."
  sofia_bell: "You run the diner: coffee, pie, who sat where, who is lying."
  cam_lin: "You poke a device or a town system to see what is actually enforced."
```

Card edits (live `town_cards.yaml` only):

- Remove the four retired character blocks. Move them verbatim plus their `sprites` and old `plans` entries into `cards/archived/2026-08-29-retired-cast.yaml`.
- Live `sprites:` keep only `mara_vale`, `nico_sable`, `sofia_bell`, `cam_lin`, `orion`.
- Nico `signature_line` must not mention lighting. Use: `"Buy a ticket. The lineup is better than the rumor."`
- Each live NPC `orion_dynamic` is one sentence (already true for Mara/Nico/Sofia/Cam — keep them one sentence).
- Mara `daily_loop[0]` stays map work. Nico `[0]` diner gossip. Sofia `[0]` runs the diner. Cam `[0]` builds mods / probes systems — align `plans` tokens with that first item (`collects gossip` / `runs the diner` / `builds mods` / `updates` maps). If a plan token check is tight, set `daily_loop[0]` to match the plan nouns.
- Orion `deeper_bio` / `town_presence`: delete sentences that name Elian, Juno, Tessa, Vale. Remaining pressure sources: Mara, Nico, Sofia, Cam, Juniper.

README cast paragraph: four NPCs + archive path + wipe still required.

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=. pytest orion/tests/test_town_cast.py services/orion-ai-town/tests/test_generate_descriptions.py -q
```

Expected: PASS. Skip `test_juniper_blurb_present_in_world_ts` if `upstream/convex/world.ts` is missing in this worktree (guard that test with `pytest.importorskip` / `skipif not WORLD_TS.exists()` so a worktree without upstream stays green).

- [ ] **Step 5: Commit**

```bash
git add orion/town_cast.py orion/tests/test_town_cast.py \
  services/orion-ai-town/cards/town_cards.yaml \
  services/orion-ai-town/cards/archived/2026-08-29-retired-cast.yaml \
  services/orion-ai-town/scripts/generate_descriptions.py \
  services/orion-ai-town/tests/test_generate_descriptions.py \
  services/orion-ai-town/README.md
# plus orion-character.patch only if regenerated from real upstream
git commit -m "Cull four AI Town NPCs and inject job fields into identities."
```

---

### Task 2: Remove the light-word magnet from the grounding patch

**Files:**
- Modify: `services/orion-ai-town/patches/orion-concrete-grounding-prompt.patch`
- Modify: `services/orion-ai-town/tests/test_concrete_grounding_prompt_patch.py`

**Interfaces:**
- Consumes: existing patch that inserts one prompt line in `continueConversationMessage`
- Produces: same hunk location, new line text exactly: `` `Answer as your job. Name a person, object, or task from your role or from what they just said. If you have nothing new to add, end the conversation.`, ``

- [ ] **Step 1: Write the failing test**

Add to `test_concrete_grounding_prompt_patch.py`:

```python
def test_patch_does_not_name_light_bait_words():
    patch = _PATCH.read_text(encoding="utf-8")
    lowered = patch.lower()
    for bait in ("light,", "shadows", "glow", "echoes", "silence"):
        assert bait not in lowered

def test_patch_uses_positive_job_contract():
    patch = _PATCH.read_text(encoding="utf-8")
    assert "Answer as your job" in patch
    assert "Name a person, object, or task" in patch
```

Keep `test_patch_adds_concrete_grounding_instruction_to_continue_conversation` but change the asserted prefix from `Ground your reply in something specific and concrete` to `Answer as your job`.

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest services/orion-ai-town/tests/test_concrete_grounding_prompt_patch.py -q
```

Expected: FAIL on bait words still present.

- [ ] **Step 3: Edit the patch**

In `orion-concrete-grounding-prompt.patch`, replace the added line with:

```text
+    `Answer as your job. Name a person, object, or task from your role or from what they just said. If you have nothing new to add, end the conversation.`,
```

Do not add a banned-word list. Keep the file a single-hunk patch on `convex/agent/conversation.ts`. If you have `upstream/` and the old patch is applied, regenerate the hunk so it still applies after `orion-anti-repetition-prompt.patch`.

- [ ] **Step 4: Run tests**

```bash
pytest services/orion-ai-town/tests/test_concrete_grounding_prompt_patch.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git commit -m "Stop naming light metaphors in the AI Town grounding prompt."
```

---

### Task 3: social-memory ingest-turn publisher

**Files:**
- Modify: `services/orion-social-memory/app/settings.py`
- Modify: `services/orion-social-memory/app/main.py`
- Modify: `services/orion-social-memory/app/service.py`
- Modify: `services/orion-social-memory/.env_example`
- Modify: `services/orion-social-memory/docker-compose.yml`
- Modify: `services/orion-social-memory/README.md`
- Modify: `orion/bus/channels.yaml` (`orion:chat:social:turn` producer_services)
- Create: `services/orion-social-memory/tests/test_ingest_turn.py`

**Interfaces:**
- Consumes: `SocialRoomTurnV1`, existing `SocialMemoryService._publish`
- Produces: `SocialMemoryService.ingest_turn(turn: SocialRoomTurnV1) -> None` publishes to channel `orion:chat:social:turn` kind `social.turn.v1`; FastAPI `POST /ingest-turn`; settings field `social_memory_ingest_token: str` alias `SOCIAL_MEMORY_INGEST_TOKEN` default `""`

- [ ] **Step 1: Write failing tests**

`services/orion-social-memory/tests/test_ingest_turn.py` — follow `test_social_memory_service.py` fake-bus style. Construct a `SocialRoomTurnV1` with `prompt="the urn is dying"`, `response="I'll pull a spare from the back"`, `client_meta` `external_room={platform:aitown,room_id:aitown-town,thread_id:cam-lin--sofia-bell}`, `external_participant={participant_id:sofia-bell,participant_name:Sofia Bell,participant_kind:npc}`.

```python
async def test_ingest_turn_publishes_social_turn_v1():
    # svc.ingest_turn(turn)
    # assert one publish on orion:chat:social:turn with kind social.turn.v1
    # payload prompt/response/thread_id match

async def test_ingest_turn_does_not_call_process_social_turn(monkeypatch):
    # if process_social_turn is invoked, fail

def test_ingest_http_401_without_token():
    # TestClient POST /ingest-turn with no Authorization → 401

def test_ingest_http_401_when_server_token_empty():
    # settings token "" even with Bearer foo → 401

def test_ingest_http_401_wrong_token():
    # server token "secret", Bearer other → 401
```

Also add a test (or extend an existing channel test) that `orion/bus/channels.yaml` `orion:chat:social:turn` `producer_services` contains `orion-social-memory`. Prefer `python scripts/check_bus_channels.py` after the yaml edit.

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest services/orion-social-memory/tests/test_ingest_turn.py -q
```

- [ ] **Step 3: Implement**

Settings: `social_memory_ingest_token: str = Field("", alias="SOCIAL_MEMORY_INGEST_TOKEN")`

`.env_example` and `docker-compose.yml` environment: `SOCIAL_MEMORY_INGEST_TOKEN=${SOCIAL_MEMORY_INGEST_TOKEN:-}` with comment “empty = ingest disabled (401)”.

`service.ingest_turn`:

```python
async def ingest_turn(self, turn: SocialRoomTurnV1) -> None:
    await self._publish("orion:chat:social:turn", "social.turn.v1", turn)
```

`main.py` `POST /ingest-turn`: read `Authorization: Bearer <token>`. Compare to `settings.social_memory_ingest_token` with `hmac.compare_digest` only when both sides are non-empty. If configured token is empty or header missing/wrong → 401. On success, `SocialRoomTurnV1.model_validate(body)` then `await service.ingest_turn(turn)` then `{"ok": True}`.

channels.yaml: add `"orion-social-memory"` to `orion:chat:social:turn` producer_services.

README: document POST /ingest-turn and the token.

Then: `python scripts/sync_local_env_from_example.py` from worktree root. Do not commit `.env`.

- [ ] **Step 4: Run tests**

```bash
pytest services/orion-social-memory/tests/test_ingest_turn.py -q
python scripts/check_bus_channels.py
python scripts/check_env_template_parity.py
```

Expected: PASS. Report any sync-script skipped keys.

- [ ] **Step 5: Commit**

```bash
git commit -m "Add social-memory ingest-turn so town NPCs can publish room turns."
```

---

### Task 4: Convex conversation.ts continuity patch

**Files:**
- Create: `services/orion-ai-town/patches/orion-town-continuity-ingest.patch`
- Create: `services/orion-ai-town/tests/test_town_continuity_prompt_patch.py`
- Modify: `services/orion-ai-town/scripts/apply_upstream_patches.sh` (append the new patch after `orion-concrete-grounding-prompt.patch`)
- Modify: `services/orion-ai-town/README.md` (Convex env: `SOCIAL_MEMORY_URL`, `SOCIAL_MEMORY_INGEST_TOKEN`, `AITOWN_ORION_NAME=Orion`)

**Interfaces:**
- Consumes: Task 3 `POST /ingest-turn` and existing `GET /summary`; Task 1 slug/thread rules (hardcode the same six-name map in TypeScript — do not infer slugs)
- Produces: patch that adds helpers and calls them from `startConversationMessage` / `continueConversationMessage` / `leaveConversationMessage`

TypeScript helpers the patch must introduce (exact names):

```typescript
const TOWN_PARTICIPANT_SLUGS: Record<string, string> = {
  'Mara Vale': 'mara-vale',
  'Nico Sable': 'nico-sable',
  'Sofia Bell': 'sofia-bell',
  'Cam Lin': 'cam-lin',
  'Juniper Feld': 'juniper-feld',
  'Orion': 'orion',
};

function townSlug(name: string): string | null {
  return TOWN_PARTICIPANT_SLUGS[name] ?? null;
}

function townThreadId(a: string, b: string): string | null {
  const left = townSlug(a);
  const right = townSlug(b);
  if (!left || !right) return null;
  return [left, right].sort().join('--');
}

function isOrionName(name: string): boolean {
  const configured = (process.env.AITOWN_ORION_NAME || 'Orion').trim();
  return name.trim() === configured;
}
```

`fetchTownContinuity(otherName: string): Promise<string>` — GET `${SOCIAL_MEMORY_URL}/summary?platform=aitown&room_id=aitown-town&participant_id={slug}`. Build a string from `room.recent_thread_summary` or `room.current_thread_summary`, then other’s `safe_continuity_summary`. Slice to 400 chars. On any error or missing URL/slug, return `""`.

`ingestTownTurn({speaker, other, prompt, response, conversationId})` — if `isOrionName(other)` return. If either slug missing return. POST `/ingest-turn` with Bearer token, `SocialRoomTurnV1`-shaped JSON: `source: "orion-ai-town"`, `prompt`, `response`, `text: `${other}: ${prompt}\n${speaker}: ${response}``, `session_id: \`aitown:${conversationId}\``, `tags: ["aitown"]`, `client_meta.external_room = {platform:"aitown", room_id:"aitown-town", thread_id}`, `client_meta.external_participant = {participant_id: speakerSlug, participant_name: speaker, participant_kind: "npc"}`. Swallow errors.

Call `fetchTownContinuity` from `startConversationMessage` and push a prompt line `What you remember: ${continuity}` only when non-empty.

Call `ingestTownTurn` after a successful completion in `continueConversationMessage` and `leaveConversationMessage` (speaker = `player.name`, other = `otherPlayer.name`).

Fail-open: wrap fetch/ingest in try/catch.

- [ ] **Step 1: Write failing patch tests** (`test_town_continuity_prompt_patch.py`), same style as `test_concrete_grounding_prompt_patch.py`:

```python
def test_continuity_patch_registered_after_grounding():
    # apply script lists orion-town-continuity-ingest.patch after orion-concrete-grounding-prompt.patch

def test_continuity_patch_hardcodes_slug_map():
    patch = _PATCH.read_text(encoding="utf-8")
    assert "'Sofia Bell': 'sofia-bell'" in patch
    assert "name.lower()" not in patch
    assert "replace(\" \", \"-\")" not in patch

def test_continuity_patch_skips_orion_as_other():
    assert "isOrionName" in patch
    assert "AITOWN_ORION_NAME" in patch

def test_continuity_patch_fetches_summary_once_at_start():
    assert "startConversationMessage" in patch
    assert "/summary" in patch
    assert "aitown-town" in patch

def test_continuity_patch_ingests_on_continue_and_leave():
    assert "ingestTownTurn" in patch
    assert "/ingest-turn" in patch
    assert "continueConversationMessage" in patch
    assert "leaveConversationMessage" in patch
```

- [ ] **Step 2: Run — expect FAIL** (patch missing).

- [ ] **Step 3: Write the patch and register it.** If `upstream/` exists, apply prior patches then implement in `conversation.ts` and `git diff` out the new patch. If `upstream/` is missing, write a valid unified diff that the tests can still read (tests assert patch text). Do not add GET /summary inside the per-line continue loop.

- [ ] **Step 4: Run**

```bash
pytest services/orion-ai-town/tests/test_town_continuity_prompt_patch.py services/orion-ai-town/tests/test_concrete_grounding_prompt_patch.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git commit -m "Patch AI Town NPC speech to read and write aitown-town continuity."
```

---

### Task 5: Embodiment stable slugs

**Files:**
- Modify: `services/orion-embodiment/app/worker.py`
- Modify: `services/orion-embodiment/tests/test_worker_conversation_memory.py`
- Modify: `services/orion-embodiment/README.md`

**Interfaces:**
- Consumes: `orion.town_cast.slug_for_name`, `thread_id_for`
- Produces: `_publish_conversation_memory` and `_fetch_participant_continuity` use slug as `participant_id` when `slug_for_name(participant_name)` is not None; otherwise skip social publish/fetch (fail-open). `client_meta.external_room` for social turns also sets `thread_id` via `thread_id_for("Orion", participant_name)` when available.

- [ ] **Step 1: Write failing tests** in `test_worker_conversation_memory.py`:

A publish test that the partner is named `Sofia Bell` must assert `external_participant.participant_id == "sofia-bell"` on the **social** turn (chat_history may keep player_id if already tested — social-memory-facing id is the slug). Assert `external_room.thread_id == "orion--sofia-bell"`.

A fetch test must assert the `/summary` query `participant_id=sofia-bell` not `p:12`.

A test that an unknown name skips social publish (no social channel call) or fetch returns None without hitting HTTP.

Update any existing assertion that expected `participant_id == "p9"` or similar on the social turn.

- [ ] **Step 2: Run — expect FAIL**

```bash
pytest services/orion-embodiment/tests/test_worker_conversation_memory.py -q
```

- [ ] **Step 3: Implement** in `_publish_conversation_memory` / `_fetch_participant_continuity`: import `slug_for_name`, `thread_id_for`, `ORION_DISPLAY_NAME` from `orion.town_cast`. Resolve slug from `participant_name`. If no slug, skip the social publish (still allow chat.history with player_id). Fetch uses slug only.

- [ ] **Step 4: Run the embodiment conversation-memory tests. Expected: PASS.**

- [ ] **Step 5: Commit**

```bash
git commit -m "Key AI Town social-memory participants by stable slugs."
```

README: note slug keying and that wipe orphans old `p:*` rows.

---

## Spec coverage (self-review)

| Spec requirement | Task |
|---|---|
| Cull + archive four | 1 |
| Rewrite remaining as jobs / Nico lighting gone | 1 |
| compose_identity job fields, no signature | 1 |
| daily_loop as plan | 1 |
| Orion blurb without dead names | 1 |
| Kill light-word list | 2 |
| POST /ingest-turn + channel producer | 3 |
| Env token + sync | 3 |
| Convex fetch once + ingest skip Orion | 4 |
| Embodiment slug + thread_id | 5 |
| Operator wipe runbook | already in spec; README mentions in 1 and 4 |
| No Convex dyad store / no cortex for NPCs | non-goals, no task |

No TBD. Slug map name is `TOWN_PARTICIPANT_SLUGS` in Python and TypeScript.
