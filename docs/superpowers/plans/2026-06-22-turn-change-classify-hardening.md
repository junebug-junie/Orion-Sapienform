# Turn-change Classify Hardening — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route turn-change classify RPC to the metacog instruct lane (no thinking), with env override and audit fields on the spark_meta patch.

**Architecture:** `orion-memory-consolidation` already classifies off the chat hot path (`memory:turn:persisted` → `classify_turn` → `chat.history.spark_meta.patch`). Phase 1 adds `TURN_CHANGE_CLASSIFY_ROUTE` (default `metacog`) to drive the LLM gateway RPC payload, forces `chat_template_kwargs.enable_thinking=false` on every classify call (including session-window reappraisal), and echoes `turn_change_classify_route` on the patch for audit. BPE-safe logprob parsing (`boundary.py`) and `reconcile_novelty_with_shift` are pre-existing on `main` — verify only, do not reimplement.

**Tech Stack:** Python 3.12, pydantic-settings, Orion bus (`OrionBusAsync`, `ChatRequestPayload`), `orion-memory-consolidation`, pytest

**Design spec:** `docs/superpowers/specs/2026-06-22-turn-change-classify-hardening-design.md`

**Worktree:** Implement in an isolated worktree (`using-superpowers:using-git-worktrees`) before touching main.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `services/orion-memory-consolidation/app/settings.py` | Modify | `TURN_CHANGE_CLASSIFY_ROUTE` env field (default `metacog`) |
| `services/orion-memory-consolidation/.env_example` | Modify | Document route + existing classify env keys |
| `services/orion-memory-consolidation/README.md` | Modify | Env table entry for classify route |
| `services/orion-memory-consolidation/app/classify.py` | Modify | Route resolver, gateway payload, patch audit field |
| `services/orion-memory-consolidation/tests/test_classify_turn_change.py` | Modify | RPC payload + route fallback + quick override tests |
| `services/orion-memory-consolidation/app/boundary.py` | Verify only | BPE-safe logprob parser (already on main) |
| `orion/memory/turn_change_classify.py` | Verify only | `reconcile_novelty_with_shift` (already on main) |
| `scripts/sync_local_env_from_example.py` | Run | Sync local `.env` after `.env_example` change |

**Out of scope (Phase 1):** Spark telemetry re-emit, `novelty_source`, golden turn CI suite, thinking-model escalation.

---

### Task 1: Settings env contract

**Files:**
- Modify: `services/orion-memory-consolidation/app/settings.py`
- Modify: `services/orion-memory-consolidation/.env_example`
- Modify: `services/orion-memory-consolidation/README.md`

- [ ] **Step 1: Write the failing test**

Add to `services/orion-memory-consolidation/tests/test_classify_turn_change.py`:

```python
def test_settings_default_classify_route_is_metacog():
    from app.settings import Settings

    s = Settings()
    assert s.TURN_CHANGE_CLASSIFY_ROUTE == "metacog"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. ./orion_dev/bin/python -m pytest services/orion-memory-consolidation/tests/test_classify_turn_change.py::test_settings_default_classify_route_is_metacog -v`

Expected: FAIL with `AttributeError: 'Settings' object has no attribute 'TURN_CHANGE_CLASSIFY_ROUTE'` (or similar)

- [ ] **Step 3: Add settings field**

In `services/orion-memory-consolidation/app/settings.py`, after `MEMORY_CLASSIFY_TIMEOUT_SEC`:

```python
    # Gateway route for turn-change classify RPC (metacog = instruct-only; avoid thinking lanes).
    TURN_CHANGE_CLASSIFY_ROUTE: str = Field(default="metacog", alias="TURN_CHANGE_CLASSIFY_ROUTE")
```

In `services/orion-memory-consolidation/.env_example`, after `MEMORY_CLASSIFY_TIMEOUT_SEC=8.0`:

```bash
# Gateway route for turn-change classify (metacog = llama-3-8b-instruct; avoid thinking lanes)
TURN_CHANGE_CLASSIFY_ROUTE=metacog
```

In `services/orion-memory-consolidation/README.md` env table, add row:

```markdown
| `TURN_CHANGE_CLASSIFY_ROUTE` | `metacog` | Gateway route for classify RPC (`metacog` or `quick`) |
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. ./orion_dev/bin/python -m pytest services/orion-memory-consolidation/tests/test_classify_turn_change.py::test_settings_default_classify_route_is_metacog -v`

Expected: PASS

- [ ] **Step 5: Sync local env**

Run: `python scripts/sync_local_env_from_example.py`

Expected: exit 0; `services/orion-memory-consolidation/.env` contains `TURN_CHANGE_CLASSIFY_ROUTE`

- [ ] **Step 6: Commit**

```bash
git add services/orion-memory-consolidation/app/settings.py \
  services/orion-memory-consolidation/.env_example \
  services/orion-memory-consolidation/README.md \
  services/orion-memory-consolidation/tests/test_classify_turn_change.py
git commit -m "feat(memory-consolidation): add TURN_CHANGE_CLASSIFY_ROUTE setting"
```

---

### Task 2: Route resolver helper

**Files:**
- Modify: `services/orion-memory-consolidation/app/classify.py`
- Test: `services/orion-memory-consolidation/tests/test_classify_turn_change.py`

- [ ] **Step 1: Write the failing tests**

Add to `services/orion-memory-consolidation/tests/test_classify_turn_change.py`:

```python
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("metacog", "metacog"),
        ("METACOG", "metacog"),
        ("quick", "quick"),
        (" chat ", "metacog"),  # invalid → fallback
        ("chat-thinking", "metacog"),
        ("", "metacog"),
    ],
)
def test_resolve_classify_route_allowlist(raw, expected):
    from app.settings import Settings

    settings = Settings(TURN_CHANGE_CLASSIFY_ROUTE=raw)
    assert classify_mod._resolve_classify_route(settings) == expected
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. ./orion_dev/bin/python -m pytest services/orion-memory-consolidation/tests/test_classify_turn_change.py::test_resolve_classify_route_allowlist -v`

Expected: FAIL with `AttributeError: module ... has no attribute '_resolve_classify_route'`

- [ ] **Step 3: Implement resolver**

Add near top of `services/orion-memory-consolidation/app/classify.py` (after logger):

```python
_CLASSIFY_ROUTES = frozenset({"metacog", "quick"})


def _resolve_classify_route(settings) -> str:
    route = str(getattr(settings, "TURN_CHANGE_CLASSIFY_ROUTE", "metacog") or "metacog").strip().lower()
    if route not in _CLASSIFY_ROUTES:
        logger.warning("invalid TURN_CHANGE_CLASSIFY_ROUTE=%r; falling back to metacog", route)
        return "metacog"
    return route
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. ./orion_dev/bin/python -m pytest services/orion-memory-consolidation/tests/test_classify_turn_change.py::test_resolve_classify_route_allowlist -v`

Expected: PASS (6 cases)

- [ ] **Step 5: Commit**

```bash
git add services/orion-memory-consolidation/app/classify.py \
  services/orion-memory-consolidation/tests/test_classify_turn_change.py
git commit -m "feat(memory-consolidation): add classify route resolver with metacog fallback"
```

---

### Task 3: Gateway RPC payload — route + thinking disabled

**Files:**
- Modify: `services/orion-memory-consolidation/app/classify.py:95-134`
- Test: `services/orion-memory-consolidation/tests/test_classify_turn_change.py`

- [ ] **Step 1: Write the failing test**

Add to `services/orion-memory-consolidation/tests/test_classify_turn_change.py`:

```python
@pytest.mark.asyncio
async def test_classify_turn_rpc_uses_metacog_route_and_disables_thinking():
    bus = AsyncMock()
    content = "NOVEL: NO\nSHIFT: NONE\nMEMORY: NO\nBOUNDARY: NO\n"
    captured: dict = {}

    async def _rpc_request(channel, env, **kwargs):
        captured["channel"] = channel
        captured["env"] = env
        captured["kwargs"] = kwargs
        return {"data": b"x"}

    bus.rpc_request = _rpc_request

    def _decode_side_effect(_):
        class _R:
            ok = True
            envelope = type("E", (), {"payload": _llm_raw(content, novel_lp=-2.0, shift_token="NONE")})()

        return _R()

    bus.codec.decode = Mock(side_effect=_decode_side_effect)
    prior = [{"correlation_id": "prev", "prompt": "cats", "response": "cute"}]
    turn = MemoryTurnPersistedV1(
        correlation_id=str(uuid4()), prompt="more cats", response="still cute", spark_meta={}
    )
    from app.settings import settings as app_settings

    patch = await classify_mod.classify_turn(bus, turn=turn, prior_turns=prior, settings=app_settings)
    payload = captured["env"].payload
    assert payload["route"] == "metacog"
    assert payload["options"]["llm_route"] == "metacog"
    assert payload["options"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["options"]["return_logprobs"] is True
    assert patch["turn_change_classify_route"] == "metacog"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. ./orion_dev/bin/python -m pytest services/orion-memory-consolidation/tests/test_classify_turn_change.py::test_classify_turn_rpc_uses_metacog_route_and_disables_thinking -v`

Expected: FAIL — `route` not `metacog`, or missing `chat_template_kwargs`, or missing `turn_change_classify_route` on patch

- [ ] **Step 3: Wire `_llm_classify` payload**

Replace `_llm_classify` body in `services/orion-memory-consolidation/app/classify.py`:

```python
async def _llm_classify(bus: OrionBusAsync, *, prompt: str, settings) -> dict:
    llm_route = _resolve_classify_route(settings)
    rpc_corr = str(uuid4())
    reply_channel = f"orion:exec:result:LLMGatewayService:{rpc_corr}"
    payload = ChatRequestPayload(
        messages=[LLMMessage(role="user", content=prompt)],
        route=llm_route,
        options={
            "return_logprobs": True,
            "logprobs_top_k": 8,
            "logprob_summary_only": False,
            "max_tokens": 24,
            "llm_route": llm_route,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    env = BaseEnvelope(
        kind="llm.chat.request",
        source=ServiceRef(
            name=settings.SERVICE_NAME,
            version=settings.SERVICE_VERSION,
            node=settings.NODE_NAME,
        ),
        correlation_id=rpc_corr,
        reply_to=reply_channel,
        payload=payload.model_dump(mode="json"),
    )
    msg = await bus.rpc_request(
        settings.CHANNEL_LLM_INTAKE,
        env,
        reply_channel=reply_channel,
        timeout_sec=float(settings.MEMORY_CLASSIFY_TIMEOUT_SEC),
    )
    decoded = bus.codec.decode(msg.get("data"))
    if not decoded.ok:
        raise RuntimeError(decoded.error)
    result_payload = decoded.envelope.payload
    content = str(result_payload.get("content") or result_payload.get("text") or "")
    raw = result_payload.get("raw") if isinstance(result_payload.get("raw"), dict) else {}
    return scores_from_llm_result(content, raw)
```

In `classify_turn`, after successful `_llm_classify`, resolve route for logging and patch:

```python
    classify_route = _resolve_classify_route(settings)
    logger.info(
        "turn_change_classify corr=%s route=%s novelty=%s shift=%s confidence=%s source=%s mem=%s bnd=%s",
        turn.correlation_id,
        classify_route,
        scores.get("novelty_score"),
        scores.get("shift_kind"),
        scores.get("confidence"),
        scores.get("scoring_source"),
        scores.get("memory_significance_score"),
        scores.get("conversation_boundary_score"),
    )
```

And include on the success return dict:

```python
    return {
        "turn_change_appraisal": appraisal,
        "turn_change_classify_route": classify_route,
        "memory_significance_score": scores.get("memory_significance_score"),
        "conversation_boundary_score": scores.get("conversation_boundary_score"),
        "memory_classify_status": "ok" if status == "ok" else "degraded",
        "memory_classify_ts": datetime.now(timezone.utc).isoformat(),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. ./orion_dev/bin/python -m pytest services/orion-memory-consolidation/tests/test_classify_turn_change.py::test_classify_turn_rpc_uses_metacog_route_and_disables_thinking -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add services/orion-memory-consolidation/app/classify.py \
  services/orion-memory-consolidation/tests/test_classify_turn_change.py
git commit -m "feat(memory-consolidation): route classify RPC to metacog with thinking disabled"
```

---

### Task 4: Invalid route fallback + quick bake-off override

**Files:**
- Modify: `services/orion-memory-consolidation/tests/test_classify_turn_change.py`

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.asyncio
async def test_classify_turn_invalid_route_falls_back_to_metacog(monkeypatch):
    bus = AsyncMock()
    content = "NOVEL: NO\nSHIFT: NONE\nMEMORY: NO\nBOUNDARY: NO\n"
    captured: dict = {}

    async def _rpc_request(channel, env, **kwargs):
        captured["env"] = env
        return {"data": b"x"}

    bus.rpc_request = _rpc_request

    def _decode_side_effect(_):
        class _R:
            ok = True
            envelope = type("E", (), {"payload": _llm_raw(content, novel_lp=-2.0, shift_token="NONE")})()

        return _R()

    bus.codec.decode = Mock(side_effect=_decode_side_effect)
    prior = [{"correlation_id": "prev", "prompt": "cats", "response": "cute"}]
    turn = MemoryTurnPersistedV1(
        correlation_id=str(uuid4()), prompt="more cats", response="still cute", spark_meta={}
    )
    from app.settings import Settings

    bad_settings = Settings(TURN_CHANGE_CLASSIFY_ROUTE="chat-thinking")
    patch = await classify_mod.classify_turn(bus, turn=turn, prior_turns=prior, settings=bad_settings)
    payload = captured["env"].payload
    assert payload["route"] == "metacog"
    assert payload["options"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert patch["turn_change_classify_route"] == "metacog"


@pytest.mark.asyncio
async def test_classify_turn_quick_route_override():
    bus = AsyncMock()
    content = "NOVEL: NO\nSHIFT: NONE\nMEMORY: NO\nBOUNDARY: NO\n"
    captured: dict = {}

    async def _rpc_request(channel, env, **kwargs):
        captured["env"] = env
        return {"data": b"x"}

    bus.rpc_request = _rpc_request

    def _decode_side_effect(_):
        class _R:
            ok = True
            envelope = type("E", (), {"payload": _llm_raw(content, novel_lp=-2.0, shift_token="NONE")})()

        return _R()

    bus.codec.decode = Mock(side_effect=_decode_side_effect)
    prior = [{"correlation_id": "prev", "prompt": "cats", "response": "cute"}]
    turn = MemoryTurnPersistedV1(
        correlation_id=str(uuid4()), prompt="more cats", response="still cute", spark_meta={}
    )
    from app.settings import Settings

    quick_settings = Settings(TURN_CHANGE_CLASSIFY_ROUTE="quick")
    patch = await classify_mod.classify_turn(bus, turn=turn, prior_turns=prior, settings=quick_settings)
    payload = captured["env"].payload
    assert payload["route"] == "quick"
    assert payload["options"]["llm_route"] == "quick"
    assert payload["options"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert patch["turn_change_classify_route"] == "quick"
```

- [ ] **Step 2: Run tests to verify they fail (if quick test is new)**

Run: `PYTHONPATH=. ./orion_dev/bin/python -m pytest services/orion-memory-consolidation/tests/test_classify_turn_change.py::test_classify_turn_invalid_route_falls_back_to_metacog services/orion-memory-consolidation/tests/test_classify_turn_change.py::test_classify_turn_quick_route_override -v`

Expected: `test_classify_turn_quick_route_override` FAIL if not yet written; invalid-route test may already pass

- [ ] **Step 3: No code change needed if Task 2–3 landed**

Resolver already allowlists `quick` and falls back invalid values to `metacog`. Reappraisal path (`reappraise_with_session_window`) calls `_llm_classify` and inherits the same route + thinking settings.

- [ ] **Step 4: Run tests to verify they pass**

Run: same command as Step 2

Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
git add services/orion-memory-consolidation/tests/test_classify_turn_change.py
git commit -m "test(memory-consolidation): classify route fallback and quick override"
```

---

### Task 5: Reappraisal inherits route + thinking disabled

**Files:**
- Test: `services/orion-memory-consolidation/tests/test_classify_turn_change.py`

- [ ] **Step 1: Extend existing reappraisal test**

In `test_classify_turn_low_margin_triggers_session_window_reappraisal`, capture both RPC envelopes and assert both use metacog + thinking disabled:

```python
    captured_envs: list = []

    async def _rpc_request(channel, env, **kwargs):
        captured_envs.append(env)
        return {"data": b"x"}

    bus.rpc_request = _rpc_request
    # ... existing decode mock ...
    patch = await classify_mod.classify_turn(bus, turn=turn, prior_turns=prior, settings=app_settings)
    assert len(captured_envs) == 2
    for env in captured_envs:
        assert env.payload["route"] == "metacog"
        assert env.payload["options"]["chat_template_kwargs"] == {"enable_thinking": False}
```

- [ ] **Step 2: Run test to verify it passes**

Run: `PYTHONPATH=. ./orion_dev/bin/python -m pytest services/orion-memory-consolidation/tests/test_classify_turn_change.py::test_classify_turn_low_margin_triggers_session_window_reappraisal -v`

Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add services/orion-memory-consolidation/tests/test_classify_turn_change.py
git commit -m "test(memory-consolidation): reappraisal RPC also disables thinking"
```

---

### Task 6: Worker publishes audit field on spark_meta patch

**Files:**
- Test: `services/orion-memory-consolidation/tests/test_classify_turn_change.py`

Acceptance criterion: patch on bus includes `turn_change_classify_route` inside `spark_meta`.

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_worker_spark_meta_patch_includes_classify_route(monkeypatch):
    worker = _load("app/worker.py", "memory_consolidation_worker")
    published = []

    bus = AsyncMock()

    async def _publish(channel, env):
        published.append((channel, env))

    bus.publish = _publish

    async def _fake_classify(bus, *, turn, prior_turns, settings):
        return {
            "turn_change_appraisal": {"turn_change_status": "ok", "novelty_score": 0.2},
            "turn_change_classify_route": "metacog",
            "memory_classify_status": "ok",
        }

    monkeypatch.setattr(worker, "classify_turn", _fake_classify)

    window_store = AsyncMock()
    window_store._get_open_window = AsyncMock(return_value=None)
    window_store.get_window_turns = AsyncMock(return_value=[])
    window_store.append_turn = AsyncMock()
    suggest_runner = AsyncMock()

    corr = str(uuid4())
    turn = MemoryTurnPersistedV1(correlation_id=corr, prompt="hi", response="hello", spark_meta={})
    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

    env = BaseEnvelope(
        kind="memory.turn.persisted.v1",
        correlation_id=corr,
        source=ServiceRef(name="sql-writer", version="0.1", node="local"),
        payload=turn.model_dump(mode="json"),
    )

    await worker.handle_memory_turn_persisted(
        env, bus=bus, window_store=window_store, suggest_runner=suggest_runner
    )

    patch_envs = [e for ch, e in published if "spark_meta:patch" in ch]
    assert len(patch_envs) == 1
    assert patch_envs[0].payload["spark_meta"]["turn_change_classify_route"] == "metacog"
```

- [ ] **Step 2: Run test**

Run: `PYTHONPATH=. ./orion_dev/bin/python -m pytest services/orion-memory-consolidation/tests/test_classify_turn_change.py::test_worker_spark_meta_patch_includes_classify_route -v`

Expected: PASS (worker already passes full `patch_fields` dict into `ChatHistorySparkMetaPatchV1.spark_meta`)

- [ ] **Step 3: Commit**

```bash
git add services/orion-memory-consolidation/tests/test_classify_turn_change.py
git commit -m "test(memory-consolidation): spark_meta patch carries classify route audit"
```

---

### Task 7: Verify pre-existing parser + reconcile (no code changes)

**Files:**
- Verify: `services/orion-memory-consolidation/app/boundary.py`
- Verify: `orion/memory/turn_change_classify.py`
- Test: `services/orion-memory-consolidation/tests/test_boundary.py`
- Test: `tests/test_turn_change_classify.py` (if present)

Spec marks these as already on `main`. Do not reimplement; run tests to confirm regression coverage.

- [ ] **Step 1: Run boundary parser tests**

Run: `PYTHONPATH=. ./orion_dev/bin/python -m pytest services/orion-memory-consolidation/tests/test_boundary.py -q`

Expected: all PASS

- [ ] **Step 2: Run turn_change_classify unit tests**

Run: `PYTHONPATH=. ./orion_dev/bin/python -m pytest tests/test_turn_change_classify.py -q`

Expected: all PASS (skip gracefully if file absent — check with `ls tests/test_turn_change_classify.py`)

- [ ] **Step 3: Confirm existing classify integration tests still pass**

Run: `PYTHONPATH=. ./orion_dev/bin/python -m pytest services/orion-memory-consolidation/tests/test_classify_turn_change.py -q --ignore-glob='*worker*'`

Or run classify-only tests by name prefix:

Run: `PYTHONPATH=. ./orion_dev/bin/python -m pytest services/orion-memory-consolidation/tests/test_classify_turn_change.py -k 'classify_turn' -q`

Expected: all classify-path tests PASS (worker tests may fail on unrelated import issues — not Phase 1 scope)

---

### Task 8: Full service verification

- [ ] **Step 1: Compile service**

Run: `python -m compileall services/orion-memory-consolidation`

Expected: exit 0

- [ ] **Step 2: Run classify test file**

Run: `./scripts/test_service.sh orion-memory-consolidation services/orion-memory-consolidation/tests/test_classify_turn_change.py -q`

Expected: classify-path tests PASS; note any worker import failures separately

- [ ] **Step 3: Final commit if any doc/test polish remains**

```bash
git status
# commit only if dirty
```

---

## Phase 1B — Deferred (separate PR)

Do not implement in this plan.

| Item | Notes |
|------|-------|
| Spark telemetry re-emit | When `spark_meta:patch` lands, refresh trace `novelty` in spark-introspector |
| `novelty_source` | Distinguish tissue vs appraisal on telemetry |
| Golden turn suite | CI regression from live chat fixtures |

---

## Self-Review (spec coverage)

| Spec requirement | Task |
|------------------|------|
| `TURN_CHANGE_CLASSIFY_ROUTE` default `metacog` | Task 1 |
| Thinking models out of scope for classify | Task 3 (`enable_thinking: false`) |
| BPE-safe logprob parser on main | Task 7 (verify) |
| `reconcile_novelty_with_shift` on main | Task 7 (verify) |
| Audit `turn_change_classify_route` on patch | Tasks 3, 6 |
| Unit tests: route + thinking disabled | Tasks 3, 4, 5 |
| Existing classify tests pass (mocked bus) | Task 8 |
| Golden turn suite | Deferred |
| Spark telemetry / `novelty_source` | Phase 1B deferred |

**Placeholder scan:** No TBD/TODO/implement-later steps in tasks above.

**Type consistency:** `turn_change_classify_route` is a `str` on patch; `_resolve_classify_route` returns `str` in `{"metacog","quick"}`; gateway payload uses same string for `route` and `options.llm_route`.
